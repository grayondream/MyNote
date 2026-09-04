# RHI主流实现调研

&emsp;&emsp;RHI（Render Hardware Interface，渲染硬件接口）是渲染引擎与底层图形 API（如 Direct3D、Vulkan、Metal、OpenGL 等）之间的抽象层。它负责向上层提供统一的资源管理、着色器绑定、命令提交、状态管理等接口，同时将调用翻译到不同的图形后端。不同引擎或中间件对 RHI 的设计目标不同：有的追求简单易用和广泛的平台覆盖，有的追求显式控制与高性能，有的则作为大型引擎的内部抽象层。本文介绍 bgfx、The Forge、Unreal Engine RHI、Diligent Engine、NVRHI、Filament、O3DE Atom RHI 的 RHI 实现，并在最后进行对比。

## 1 BGFX

&emsp;&emsp;bgfx 是一个跨平台、开源的图形渲染库，其核心目标是为游戏和图形应用提供一个轻量级、易于集成且性能良好的渲染抽象层。bgfx 的 RHI 设计强调 **简单易用** 与 **广泛平台覆盖**，它并非追求极致显式控制（如 Vulkan 原生接口），而是通过合理抽象在易用性和性能之间取得平衡。bgfx 支持 Direct3D 11/12、OpenGL 4.3+、OpenGL ES 3.0+、Vulkan、Metal、WebGPU、GNM（PS4）、NVN（Switch）、AGC（PS5）等多种后端，其内部 RHI 实现是这一跨平台能力的基石。

### 1.1 整体架构概览

&emsp;&emsp;bgfx 采用经典的三层架构设计：

```
┌─────────────────────────────────────────────────┐
│              应用层 (Application)                │
│         bgfx::submit / bgfx::setState ...       │
├─────────────────────────────────────────────────┤
│          公共 API + Context 层                    │
│   bgfx.h (公共API) → Context (内部调度核心)       │
│   Frame (帧数据) / CommandBuffer (命令缓冲)       │
├─────────────────────────────────────────────────┤
│            后端抽象层 (Renderer)                  │
│   renderer_d3d11.cpp / renderer_vk.cpp          │
│   renderer_mtl.cpp / renderer_gl.cpp ...         │
├─────────────────────────────────────────────────┤
│         底层图形 API + 平台                      │
│   D3D11 / D3D12 / Vulkan / Metal / GL / WebGPU  │
└─────────────────────────────────────────────────┘
```

&emsp;&emsp;核心源文件组织如下：

| 文件 | 职责 |
|------|------|
| `include/bgfx/bgfx.h` | 公共 API 头文件（自动生成），定义所有用户可见的枚举、Handle、函数签名 |
| `include/bgfx/defines.h` | 宏定义与常量（状态位、标志位等） |
| `src/bgfx_p.h` | 内部头文件，定义 Context、Frame、SortKey、Handle 内部表示等核心数据结构 |
| `src/bgfx.cpp` | Context 的 API 实现，帧提交、资源创建/销毁的命令序列化 |
| `src/renderer.h` | 后端抽象接口，定义 `RendererContextI` 虚接口，各后端需实现此接口 |
| `src/renderer_*.h/cpp` | 各平台后端的具体实现 |
| `src/config.h` | 编译时配置宏（最大资源数量、线程模式等） |
| `src/shader.cpp` / `src/vertexlayout.cpp` | 着色器解析与顶点布局处理 |

### 1.2 公共 API 设计

#### 1.2.1 Handle 系统

&emsp;&emsp;bgfx 采用 **轻量级 Handle** 体系管理所有 GPU 资源。每种资源类型定义一个 `Handle` 结构体，内部仅包含一个 `uint16_t idx`：

```cpp
#define BGFX_HANDLE(_name) \
    struct _name { uint16_t idx; };
```

&emsp;&emsp;支持的 Handle 类型包括：
- `VertexBufferHandle`、`IndexBufferHandle` — 静态顶点/索引缓冲
- `DynamicVertexBufferHandle`、`DynamicIndexBufferHandle` — 动态缓冲
- `ShaderHandle`、`ProgramHandle` — 着色器与着色器程序
- `TextureHandle`、`FrameBufferHandle` — 纹理与帧缓冲
- `UniformHandle` — Uniform 变量
- `VertexLayoutHandle` — 顶点布局声明
- `OcclusionQueryHandle` — 遮挡查询

&emsp;&emsp;Handle 的合法性通过 `bgfx::isValid()` 检查（判断 `idx != kInvalidHandle`），内部还维护了 `HandleAlloc` 分配器来验证 Handle 是否已被分配。

#### 1.2.2 命令提交模型

&emsp;&emsp;bgfx 采用 **帧缓冲命令提交** 模型。应用层的 API 调用（如 `bgfx::submit`、`bgfx::setState`）并不直接调用底层图形 API，而是将命令序列化到 `Frame` 的内部缓冲区中。每一帧的流程为：

```
帧开始 (bgfx::frame)
  → 应用层提交命令（submit/setState/touch...）
  → 命令被序列化到 Frame 的 CommandBuffer
  → 帧结束：Frame 排序 → 后端消费命令 → 执行 GPU 调用
```

&emsp;&emsp;在 `Context::submit` 中，每次 `bgfx::submit` 会生成一个 `RenderItem`，包含排序键（SortKey）、绑定信息（RenderBind）以及绘制参数。这些数据存储在 `Frame::m_renderItem` 和 `Frame::m_renderBind` 中，帧结束时统一排序处理。

#### 1.2.3 Encoder 机制

&emsp;&emsp;bgfx 支持多线程编码器（Encoder）。`Encoder` 允许多个线程并行提交绘制命令，每个 Encoder 拥有独立的 UniformBuffer。Encoder 通过 `bgfx::begin` 创建，通过 `bgfx::end` 提交。多线程模式下（`BGFX_CONFIG_MULTITHREADED=1`），API 线程和渲染线程通过信号量（`m_apiSem`、`m_renderSem`）同步：

- **API 线程**：接收应用层调用，序列化命令到 `m_submit`（当前提交帧）
- **渲染线程**：从 `m_render`（上一帧）读取命令并执行 GPU 调用

&emsp;&emsp;单帧延迟（frame latency）确保 API 线程和渲染线程可以并行工作。`m_frame[2]` 数组中，`m_submit` 指向当前写入帧，`m_render` 指向当前渲染帧，通过 `swap()` 交换。

### 1.3 内部核心架构

#### 1.3.1 Context 结构

&emsp;&emsp;`Context`（定义在 `bgfx_p.h` 约第 7300 行起）是 bgfx 的核心调度枢纽，主要成员包括：

| 成员 | 说明 |
|------|------|
| `m_frame[2]` | 双缓冲 Frame（多线程时为 3 个，含一个中间帧） |
| `m_submit` / `m_render` | 指向当前提交帧和当前渲染帧的指针 |
| `m_encoder[]` | Encoder 实例数组，支持多线程并行编码 |
| `m_indexBufferHandle` 等 | 各种资源的 HandleAlloc 分配器 |
| `m_indexBuffers[]` 等 | 资源引用表（存储资源元数据） |
| `m_shaderRef[]`、`m_programRef[]` 等 | 引用计数的资源信息表 |
| `m_vertexLayoutRef` | 顶点布局的引用计数与去重 |
| `m_uniformHashMap` | Uniform 名称到 Handle 的哈希映射 |
| `m_dynIndexBufferAllocator` 等 | 动态缓冲的子分配器（NonLocalAllocator） |
| `m_clearColor[]` | 调色板颜色缓存 |
| `m_seq[]` | 每个 View 的序列号计数器 |

#### 1.3.2 Frame 结构

&emsp;&emsp;`Frame`（定义在 `bgfx_p.h:3157`）存储一帧的所有渲染数据：

| 成员 | 说明 |
|------|------|
| `m_view[BGFX_CONFIG_MAX_VIEWS]` | View 数组（最大 256 个 View） |
| `m_viewRemap[]` / `m_viewOrder[]` | View 重映射与排序 |
| `m_renderItem` | 绘制调用数据（`FrameArenaT<RenderItem>`，支持动态增长） |
| `m_renderBind` | 绑定状态数据（`FrameArenaT<RenderBind>`） |
| `m_blitItem` / `m_blitKeys` | Blit 操作数据 |
| `m_sortKeys[]` / `m_sortValues[]` | 排序键值对数组 |
| `m_frameCache` | 帧缓存（包含 `MatrixCache` 和 `RectCache`） |
| `m_cmdPre` / `m_cmdPost` | 资源命令缓冲区（创建/销毁资源的 CommandBuffer） |
| `m_uniformBuffer[]` | 每个 Encoder 的 Uniform 缓冲 |
| `m_textVideoMem` | 调试文本内存（`BGFX_CONFIG_DEBUG_TEXT` 时启用） |
| `m_perfStats` | 帧性能统计 |
| `m_occlusion[]` | 遮挡查询结果 |

&emsp;&emsp;`Frame` 使用 `FrameArenaT<T, BlockT>` 模板实现动态块分配。当 `BGFX_CONFIG_DYNAMIC_FRAME_STORAGE=1`（默认）时，渲染项、绑定项等数据按需分配内存块（每块 `kDrawCallBlock=1024` 项），未使用的块不占用内存。帧间通过 `adjustCapacity()` 观察峰值使用量，逐步收缩至实际需要的大小。

#### 1.3.3 SortKey 排序键

&emsp;&emsp;bgfx 通过 **64 位排序键** 实现绘制调用的自动排序。排序键的位域布局为：

```
┌──────────┬──────────┬──────────┬──────────┬─────────┐
│  View    │ Depth    │ Seq/Prog │ Blend    │ HasAlpha│
│ (8 bit)  │(32 bit)  │(20/9 bit)│ (1 bit)  │ (1 bit) │
└──────────┴──────────┴──────────┴──────────┴─────────┘
```

&emsp;&emsp;排序策略（`SortKey::Sort` 枚举）：
- **SortDrawState**：按 View → Program → Blend → Depth 排序，尽量减少状态切换
- **SortDrawSequence**：按 View → 提交顺序排序，保持调用顺序
- **SortComputeState**：计算着色器按 View → Program 排序

&emsp;&emsp;帧结束时调用 `Frame::sort()`，使用基数排序（Radix Sort）对 `m_sortKeys` 排序，同时交换 `m_sortValues` 保持索引对应关系。排序后渲染线程按序遍历即可。

### 1.4 后端抽象层

#### 1.4.1 RendererContextI 接口

&emsp;&emsp;每个后端需要实现 `RendererContextI` 虚接口（定义在 `renderer.h`），核心方法包括：

| 方法 | 说明 |
|------|------|
| `init()` / `shutdown()` | 初始化/销毁后端 |
| `createIndexBuffer()` / `destroyIndexBuffer()` | 创建/销毁索引缓冲 |
| `createVertexBuffer()` / `destroyVertexBuffer()` | 创建/销毁顶点缓冲 |
| `createShader()` / `destroyShader()` | 创建/销毁着色器 |
| `createProgram()` / `destroyProgram()` | 创建/销毁着色器程序 |
| `createTexture()` / `destroyTexture()` | 创建/销毁纹理 |
| `createFrameBuffer()` / `destroyFrameBuffer()` | 创建/销毁帧缓冲 |
| `createUniform()` / `destroyUniform()` | 创建/销毁 Uniform |
| `submit()` | 提交一帧的渲染命令 |
| `flip()` | 交换前后缓冲 |
| `RendererContextType` | 返回后端类型枚举 |

&emsp;&emsp;各后端实现对应的命名空间：
- `bgfx::d3d11` — `renderer_d3d11.cpp`（~7700 行）
- `bgfx::d3d12` — `renderer_d3d12.cpp`
- `bgfx::vk` — `renderer_vk.cpp`
- `bgfx::mtl` — `renderer_mtl.cpp`
- `bgfx::gl` — `renderer_gl.cpp`
- `bgfx::webgpu` — `renderer_webgpu.cpp`
- `bgfx::noop` — `renderer_noop.cpp`（空实现，用于测试）
- `bgfx::gnm` / `bgfx::nvn` / `bgfx::agc` — 主机平台后端

#### 1.4.2 后端资源类型映射

&emsp;&emsp;每个后端定义对应的 GPU 资源类型，例如 D3D11 后端：

```cpp
struct BufferD3D11 {
    ID3D11Buffer*           m_ptr;
    ID3D11ShaderResourceView*  m_srv;
    ID3D11UnorderedAccessView* m_uav;
    uint32_t m_size;
    uint16_t m_flags;
    bool     m_dynamic;
};

struct ShaderD3D11 {
    ID3D11VertexShader*    m_ptr;
    ID3DBlob*              m_code;
    UniformBuffer*         m_constantBuffer;
    uint32_t               m_hash;
    // ... uniform 相关元数据
};

struct TextureD3D11 {
    ID3D11Texture2D*          m_texture2D;
    ID3D11Texture3D*          m_texture3D;
    ID3D11TextureCube*        m_textureCube;
    ID3D11ShaderResourceView* m_srv;
    // ... 逐 mip 层级的 RTV/DSV/UAV
};
```

&emsp;&emsp;Vulkan 后端（`renderer_vk.h`，~979 行）更为复杂，需要管理 VkInstance、VkDevice、VkQueue、VkCommandPool、VkPipeline、VkDescriptorSet 等 Vulkan 对象，并通过 `VK_IMPORT` 宏动态加载 Vulkan 函数指针。

#### 1.4.3 状态映射

&emsp;&emsp;bgfx 定义了一套与后端无关的状态位（`defines.h` 中的 `BGFX_STATE_*` 宏），各后端在 `submit()` 时将这些状态位映射到对应的 API 调用。例如 D3D11 后端：

| bgfx 状态 | D3D11 映射 |
|-----------|-----------|
| `BGFX_STATE_WRITE_RGB` / `BGFX_STATE_WRITE_A` | `D3D11_COLOR_WRITE_ENABLE` |
| `BGFX_STATE_DEPTH_TEST_*` | `D3D11_COMPARISON_FUNC` |
| `BGFX_STATE_BLEND_*` | `D3D11_BLEND` + `D3D11_BLEND_OP` |
| `BGFX_STATE_CULL_*` | `D3D11_CULL_MODE` |
| `BGFX_STATE_MSAA` | `D3D11_SAMPLE_DESC` |

### 1.5 资源管理

#### 1.5.1 资源创建流程

&emsp;&emsp;资源创建遵循统一模式：

1. 用户调用 `bgfx::createXxx()`（如 `bgfx::createIndexBuffer`）
2. Context 层分配 Handle，记录资源元数据
3. 将创建命令序列化到 `CommandBuffer`（`m_cmdPre` 或 `m_cmdPost`）
4. 渲染线程在 `rendererExecCommands()` 中消费命令，调用后端 `RendererContextI::createXxx()`

&emsp;&emsp;命令缓冲区使用紧凑的二进制格式序列化，命令 ID 为 `CommandBuffer::Enum`（如 `CreateIndexBuffer`、`CreateTexture` 等），后续紧跟参数数据。

#### 1.5.2 动态缓冲子分配

&emsp;&emsp;动态索引/顶点缓冲使用 `NonLocalAllocator` 进行子分配，避免频繁创建/销毁小缓冲。初始分配 `BGFX_CONFIG_DYNAMIC_INDEX_BUFFER_SIZE`（默认 1MB）和 `BGFX_CONFIG_DYNAMIC_VERTEX_BUFFER_SIZE`（默认 3MB）的内存块，后续通过子分配器切分。当块耗尽时自动扩展。

#### 1.5.3 Transient 缓冲

&emsp;&emsp;`TransientVertexBuffer` 和 `TransientIndexBuffer` 是帧内临时缓冲，通过 `bgfx::allocTransientVertexBuffer` 从 `Frame` 的连续内存中分配。这些缓冲仅在当前帧有效，无需显式销毁，适合粒子系统、骨骼动画等每帧变化的数据。

#### 1.5.4 引用计数与延迟销毁

&emsp;&emsp;资源使用引用计数管理生命周期。例如 `ShaderRef` 包含 `m_refCount`，`ProgramRef` 也包含 `m_refCount`。当引用计数归零时，资源被加入 `Frame::m_freeXxx` 队列，在帧结束时通过 `freeAllHandles()` 统一释放。

&emsp;&emsp;顶点布局（`VertexLayoutRef`）使用额外的去重机制：通过 `m_vertexLayoutRef` 的 HashMap 按哈希值去重，多个顶点缓冲共享同一布局时仅创建一次。

### 1.6 着色器与材质

#### 1.6.1 着色器编译

&emsp;&emsp;bgfx 使用 `shaderc` 工具将 GLSL/HLSL 着色器编译为各后端的中间格式（如 D3D11 的 DXBC、Vulkan 的 SPIR-V、Metal 的 metallib）。着色器二进制格式为：

```
Magic (4B) → 常量/Uniform 信息 → 采样器绑定 → 顶点输出/片段输入哈希 → 着色器代码
```

&emsp;&emsp;`ShaderRef` 记录了关键元数据：
- `m_hashIn` / `m_hashOut` — 着色器输入/输出签名哈希（用于验证 VS/FS 匹配）
- `m_samplerUniform[]` — 采样器对应的 Uniform 索引
- `m_samplerDimension[]` — 每个采样器的纹理维度（1D/2D/3D/Cube）
- `m_rawSrvMask` / `m_rawUavMask` — 原始 SRV/UAV 绑定掩码

#### 1.6.2 着色器程序链接

&emsp;&emsp;`bgfx::createProgram(vsh, fsh)` 创建着色器程序时：
1. 验证 VS 输出哈希与 FS 输入哈希匹配
2. 验证 VS/FS 中相同采样器的纹理维度一致
3. 通过 HashMap 去重（相同 vsh+fsh 组合复用已有 Program）
4. 将创建命令发送到渲染线程

#### 1.6.3 预定义 Uniform

&emsp;&emsp;bgfx 自动为着色器注入以下预定义 Uniform（由 `PredefinedUniform::Enum` 定义）：

| Uniform | 说明 |
|---------|------|
| `u_viewRect` | View 视口矩形 (x, y, w, h) |
| `u_viewTexel` | View 像素大小 (1/w, 1/h) |
| `u_view` | View 矩阵 |
| `u_invView` | View 逆矩阵 |
| `u_proj` | 投影矩阵 |
| `u_invProj` | 投影逆矩阵 |
| `u_viewProj` | View×Projection |
| `u_invViewProj` | (View×Proj)⁻¹ |
| `u_model[i]` | 模型矩阵（支持实例化数组） |
| `u_modelView` / `u_modelViewProj` | 组合矩阵 |
| `u_alphaRef` | Alpha 测试参考值 |
| `u_indirectArgBase` | 间接绘制参数基址 |

&emsp;&emsp;这些 Uniform 在 `ViewState::setPredefined()` 中由渲染线程自动设置，应用层无需手动传递。

### 1.7 多线程模型

&emsp;&emsp;bgfx 支持两种线程模式（由 `BGFX_CONFIG_MULTITHREADED` 控制）：

#### 单线程模式
&emsp;&emsp;所有 API 调用和 GPU 提交在同一线程完成。`m_frame[1]`，无信号量同步。

#### 多线程模式
&emsp;&emsp;分为 **API 线程** 和 **渲染线程**：

```
API 线程                          渲染线程
┌────────────────┐               ┌────────────────┐
│ bgfx::submit() │               │ renderFrame()  │
│ 序列化到       │ ── swap ──→  │ 从 m_render 读 │
│ m_submit       │               │ 取并执行命令    │
│                │ ←─ wait ──   │                │
└────────────────┘               └────────────────┘
```

&emsp;&emsp;同步机制：
- `m_apiSem`：API 线程等待渲染线程完成上一帧
- `m_renderSem`：渲染线程等待 API 线程完成当前帧的序列化
- `m_encoderEndSem`：等待所有 Encoder 结束编码

&emsp;&emsp;Encoder 之间通过 `m_encoderApiLock` 互斥锁保护共享状态。多 Encoder 并行编码时，每个 Encoder 写入独立的 UniformBuffer，帧结束时通过 `encoderApiWait()` 汇总所有 Encoder 的提交统计。

### 1.8 View 与渲染排序

#### 1.8.1 View 概念

&emsp;&emsp;bgfx 使用 **View** 作为渲染排序的基本单位。每个 View 包含：
- 视口（Rect）、裁剪矩形
- View/Proj 矩阵
- 清除操作（颜色/深度/模板）
- 绑定的 FrameBuffer
- 排序模式（`ViewMode::Sequential` 或 `SortDrawState`）
- 最大绘制调用数限制

&emsp;&emsp;View 之间通过 `setViewOrder()` 定义执行顺序。默认按 ViewId 顺序执行。支持视图重映射（`m_viewRemap`）。

#### 1.8.2 渲染流程

```
for each View in viewOrder:
    if View绑定了FrameBuffer:
        设置 RenderTarget
    执行清除操作
    for each DrawCall in this View (按SortKey排序):
        绑定 Program/State/Uniforms
        设置预定义 Uniform
        执行 Draw/DrawIndexed/DrawIndirect
    执行 Blit 操作
```

### 1.9 关键配置参数

&emsp;&emsp;bgfx 通过 `config.h` 提供大量编译时配置：

| 宏 | 默认值 | 说明 |
|----|--------|------|
| `BGFX_CONFIG_MAX_DRAW_CALLS` | 65535 | 每帧最大绘制调用数 |
| `BGFX_CONFIG_MAX_VIEWS` | 256 | 最大 View 数量 |
| `BGFX_CONFIG_MAX_TEXTURES` | 4096 | 最大纹理 Handle 数 |
| `BGFX_CONFIG_MAX_SHADERS` | 512 | 最大着色器数 |
| `BGFX_CONFIG_MAX_VERTEX_LAYOUTS` | 64 | 最大顶点布局数 |
| `BGFX_CONFIG_MULTITHREADED` | 1（非 Emscripten） | 是否启用多线程 |
| `BGFX_CONFIG_DYNAMIC_FRAME_STORAGE` | 1 | 动态帧存储（按需分配） |
| `BGFX_CONFIG_DRAW_CALL_BLOCK` | 1024 | 动态分配块大小 |
| `BGFX_CONFIG_MAX_TRANSIENT_VERTEX_BUFFER_SIZE` | 6MB | Transient VB 大小 |
| `BGFX_CONFIG_MAX_TRANSIENT_INDEX_BUFFER_SIZE` | 2MB | Transient IB 大小 |
| `BGFX_CONFIG_DEFAULT_MAX_ENCODERS` | 8（多线程） | 最大 Encoder 数 |

### 1.10 设计特点与优劣分析

**优点：**
- **极简 API**：用户只需了解 Handle + submit 模型，学习成本低
- **广泛平台覆盖**：单一代码库支持 10+ 图形后端
- **自动排序优化**：基于 SortKey 的排序自动减少状态切换
- **多线程编码**：支持多 Encoder 并行提交，充分利用多核
- **零资源泄漏**：引用计数 + 帧延迟销毁确保资源安全
- **动态帧存储**：按需分配内存，避免过度预留
- **预定义 Uniform**：自动处理矩阵变换，简化着色器编写

**局限：**
- **低级控制有限**：不暴露底层 API 的全部能力（如 Vulkan 的子通道、Metal 的资源状态管理）
- **帧延迟**：多线程模式下存在 1 帧延迟
- **排序键位宽限制**：深度排序 32 位、程序索引 9 位（最多 512 个 Program），大型场景可能不够
- **不支持计算着色器的高级特性**：计算着色器绑定相对简单
- **调试能力依赖回调**：调试工具集成通过 `CallbackI` 接口，不如原生 API 直接

### 1.11 代码规模

&emsp;&emsp;bgfx 核心代码量（不含示例和工具）：

| 文件 | 约行数 |
|------|--------|
| `bgfx_p.h` | ~7400 |
| `bgfx.cpp` | ~7400 |
| `renderer_d3d11.cpp` | ~7700 |
| `renderer_vk.cpp` | ~9000+ |
| `renderer_gl.cpp` | ~8000+ |
| `renderer_mtl.cpp` | ~6000+ |
| `renderer_d3d12.cpp` | ~7000+ |
| `renderer.h` | ~900 |
| 各 `renderer_*.h` | 500-1000 |

&emsp;&emsp;总体而言，bgfx 是一个成熟、实用的 RHI 实现，适合需要快速跨平台部署且不要求极致底层控制的项目。它的设计哲学是"隐藏复杂性，暴露简单性"，通过内部排序、状态缓存和命令批处理在抽象层自动完成大量优化工作。

## 2 The-Forge

&emsp;&emsp;The-Forge 是一个跨平台的图形渲染框架，由 AMD GPUOpen 团队开发，专注于提供高性能、低开销的图形抽象层。其核心设计哲学是"小团队友好"，采用纯 C99 实现后端代码，强调编译期多态和显式控制。The-Forge 支持 Direct3D 12、Vulkan、Metal 等现代图形 API，并已在 Windows、macOS、iOS、Android、Linux（Steam Deck）、Quest 以及所有主要游戏主机上验证。

### 2.1 整体架构

&emsp;&emsp;The-Forge 采用分层架构设计：

```
┌─────────────────────────────────────────────┐
│              Game / Application Layer        │
│   (IApp, ICameraController, IUI, IFont)     │
├─────────────────────────────────────────────┤
│          Renderer / Resource System          │
│   (IResourceLoader, VisibilityBuffer, ...)   │
├──────────────────────┬──────────────────────┤
│   IGraphics.h 公共API │   FSL 着色器系统     │
│  (跨平台抽象接口)      │  (统一Shader Resource│
│                       │   Table 绑定模型)    │
├──────────┬───────────┼──────────┬───────────┤
│ D3D12.c  │ Vulkan.c  │Metal.mm  │ Orbis/    │
│ (C99)    │ (C99)     │(ObjC++)  │ Prospero  │
├──────────┴───────────┴──────────┴───────────┤
│     VMA (Vulkan) / D3D12MA (D3D12)          │
│     OS抽象 / 线程系统 / 内存管理              │
└─────────────────────────────────────────────┘
```

&emsp;&emsp;核心源文件组织如下：

| 目录/文件 | 职责 |
|-----------|------|
| `Common_3/Graphics/Interfaces/IGraphics.h` | 公共 API 头文件，定义所有用户可见的枚举、结构体、函数签名 |
| `Common_3/Graphics/Direct3D12/Direct3D12.c` | D3D12 后端实现（~6,650 行 C99） |
| `Common_3/Graphics/Vulkan/Vulkan.c` | Vulkan 后端实现（~9,627 行 C99） |
| `Common_3/Graphics/Metal/MetalRenderer.mm` | Metal 后端实现（ObjC++） |
| `Common_3/Graphics/ThirdParty/VMA` | Vulkan 内存分配器 |
| `Common_3/Graphics/ThirdParty/D3D12MA` | D3D12 内存分配器 |
| `Common_3/Utilities/` | 数学库、线程系统、内存管理、RingBuffer |

### 2.2 公共 API 设计

&emsp;&emsp;The-Forge 的 API 设计以纯 C99 为核心，所有资源创建遵循统一的描述符模式。关键 API 函数包括：

| 函数 | 用途 |
|------|------|
| `initRenderer` / `exitRenderer` | 创建/销毁渲染器 |
| `initQueue` / `exitQueue` | 创建/销毁命令队列 |
| `initCmd` / `exitCmd` | 分配/释放命令缓冲区 |
| `addBuffer` / `removeBuffer` | 创建/销毁 Buffer |
| `addTexture` / `removeTexture` | 创建/销毁 Texture |
| `addRenderTarget` / `removeRenderTarget` | 创建/销毁渲染目标 |
| `addShaderBinary` / `removeShader` | 创建/销毁着色器 |
| `addPipeline` / `removePipeline` | 创建/销毁管线状态 |
| `addDescriptorSet` / `removeDescriptorSet` | 创建/销毁描述符集 |
| `addResourceHeap` / `removeResourceHeap` | 创建/销毁资源堆 |

&emsp;&emsp;命令提交采用录制-提交模型，通过 `GpuCmdRing` 高层封装简化命令管理。每帧流程为：获取命令缓冲区 → 录制命令 → 提交到队列 → 呈现交换链。

### 2.3 后端抽象层

&emsp;&emsp;The-Forge 使用编译期多态而非运行时多态。在 `GraphicsConfig.h` 中根据平台预定义选择后端：

```c
// 平台-后端映射
#if defined(_WINDOWS)
    #include "Direct3D12/Direct3D12Config.h"    // Windows默认D3D12
#elif defined(__APPLE__)
    #include "Metal/MetalConfig.h"              // Apple用Metal
#elif defined(__ANDROID__) || defined(__linux__)
    #include "Vulkan/VulkanConfig.h"            // Android/Linux用Vulkan
#endif
```

&emsp;&emsp;核心数据结构（Buffer、Texture、Pipeline 等）使用联合体+条件编译包含每个后端的原生句柄，确保零运行时开销。

### 2.4 资源管理

&emsp;&emsp;The-Forge 在不同后端使用统一的内存分配策略：

| 后端 | 内存分配器 | 说明 |
|------|-----------|------|
| **Vulkan** | VMA | GPUOpen 的 Vulkan 内存分配器 |
| **D3D12** | D3D12MA | GPUOpen 的 D3D12 内存分配器 |
| **Metal** | VMA (跨平台适配) | 通过 VMA 适配器统一管理 |

&emsp;&emsp;内存使用类型分为：`GPU_ONLY`、`CPU_ONLY`、`CPU_TO_GPU`（上传）、`GPU_TO_CPU`（回读）。支持资源堆（Resource Heap）管理，处理 D3D12 的堆层级限制。

### 2.5 多线程模型

&emsp;&emsp;The-Forge 支持多线程命令录制。关键设计：

- 每个线程从 `GpuCmdRing` 获取独立的 `CmdPool` + `Cmd`
- 命令缓冲区使用 `VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT` 标记
- 同一个 Queue 的提交通过互斥锁串行化

&emsp;&emsp;`GpuCmdRing` 是一个轻量级的命令缓冲区环形分配器，典型配置为双缓冲，一个在 GPU 执行，一个在 CPU 录制。

### 2.6 着色器处理

&emsp;&emsp;The-Forge 使用 FSL（Forge Shading Language），一种 HLSL 超集。核心特点：

- **Shader Resource Table (SRT)**：C++ 和着色器之间共享的统一资源绑定表
- 着色器跨平台编译为各后端字节码（DXIL/SPIR-V/MSL）
- 通过 `fsl_srt.h` 在 C++ 和着色器之间共享数据结构定义

&emsp;&emsp;Root Signature 定义了统一的资源绑定布局，支持四个描述符集（持久资源、每帧更新、每批次更新、每Draw更新）和静态采样器。

### 2.7 设计特点

&emsp;&emsp;The-Forge 的独特设计决策包括：

1. **纯 C99 后端实现**：代码量更小，编译更快，更适合小团队维护
2. **编译期多态**：通过 `#ifdef` 在编译期选择渲染后端，零运行时开销
3. **VMA 统一内存管理**：三个主流后端都使用 VMA 变体，简化内存管理
4. **Triangle Visibility Buffer**：框架内置了先进的 Visibility Buffer 渲染技术
5. **Dynamic Rendering 优先**：拥抱 `VK_KHR_dynamic_rendering` 扩展，避免 Vulkan Render Pass 的复杂性
6. **异步资源加载**：通过 `IResourceLoader` 提供统一的异步资源加载

&emsp;&emsp;总体而言，The-Forge 是一个面向专业游戏开发的跨平台图形框架，适合需要高性能、显式控制且团队规模较小的项目。它的设计哲学是"显式控制，最小抽象"，通过 C99 实现和编译期多态在性能和可维护性之间取得平衡。

## 3 UE-RHI（Unreal Engine RHI）

&emsp;&emsp;Unreal Engine RHI（以下简称 UE-RHI）是 Unreal Engine 的渲染硬件接口层，负责将引擎渲染器的调用翻译到不同的图形 API 后端。与 bgfx、The-Forge 等独立图形库不同，UE-RHI 是一个 **深度嵌入大型游戏引擎** 的 RHI 实现，其设计紧密围绕 Unreal Engine 的渲染器架构，提供了业界最完整的跨平台图形抽象之一。UE-RHI 支持 D3D11、D3D12、Vulkan、Metal、OpenGL、NullDrv 等多个后端，并在 Windows、Linux、macOS、iOS、Android、Console 等平台上运行。

### 3.1 整体架构

&emsp;&emsp;UE-RHI 采用分层架构，核心代码位于 `Engine/Source/Runtime/RHI/` 目录下，各后端实现位于独立的模块中：

```
┌──────────────────────────────────────────────────────────┐
│                   渲染器层 (Renderer)                      │
│          FDeferredShadingRenderer / FScene                │
│          SetGraphicsPipelineState / DrawPrimitive         │
├──────────────────────────────────────────────────────────┤
│            RHI 命令列表层 (FRHICommandList)                │
│   FRHICommandListImmediate / FRHIComputeCommandList       │
│   命令录制 → 并行翻译 → RHI线程执行                         │
├──────────────────────────────────────────────────────────┤
│              RHI 公共抽象层 (RHI Module)                    │
│   FDynamicRHI (虚基类) / FRHIResource / IRHICommandContext │
│   PipelineStateCache / FRHITransition                     │
├──────────────────────────────────────────────────────────┤
│              后端实现层 (Platform RHI)                      │
│  D3D12RHI/  VulkanRHI/  OpenGLDrv/  NullDrv/              │
│  FD3D12DynamicRHI  FVulkanDynamicRHI  FOpenGLDynamicRHI   │
├──────────────────────────────────────────────────────────┤
│            底层图形 API + 操作系统                          │
│  D3D11 / D3D12 / Vulkan / Metal / OpenGL / 平台窗口系统    │
└──────────────────────────────────────────────────────────┘
```

&emsp;&emsp;源码组织如下：

| 目录/文件 | 职责 |
|-----------|------|
| `RHI/Public/RHI.h` | RHI 总入口头文件，定义像素格式、投影矩阵调整、状态初始化器等 |
| `RHI/Public/DynamicRHI.h` | **核心抽象类 `FDynamicRHI`**，定义所有后端必须实现的纯虚接口 |
| `RHI/Public/RHIResources.h` | RHI 资源类层次结构（FRHIResource、FRHITexture、FRHIBuffer 等） |
| `RHI/Public/RHICommandList.h` | RHI 命令列表框架，命令录制、并行翻译、执行模型 |
| `RHI/Public/RHIContext.h` | 渲染上下文接口（IRHICommandContext / IRHIComputeContext） |
| `RHI/Public/RHIShaderParameters.h` | 着色器参数绑定系统 |
| `RHI/Public/RHITransition.h` | 资源状态转换（barrier）系统 |
| `RHI/Public/RHIAccess.h` | 资源访问状态枚举（ERHIAccess） |
| `RHI/Public/RHIPipeline.h` | 管线类型定义（Graphics / AsyncCompute） |
| `RHI/Public/RHITransientResourceAllocator.h` | 瞬态资源分配器接口 |
| `RHI/Public/PipelineStateCache.h` | PSO（Pipeline State Object）缓存系统 |
| `RHICore/` | 跨后端共享的核心工具（描述符分配器、池分配器等） |
| `D3D12RHI/` | D3D12 后端实现（72个私有源文件） |
| `VulkanRHI/` | Vulkan 后端实现 |
| `OpenGLDrv/` | OpenGL 后端实现 |
| `NullDrv/` | 空设备驱动（用于无头模式、测试等） |

&emsp;&emsp;后端标识通过 `ERHIInterfaceType` 枚举区分：

```cpp
enum class ERHIInterfaceType
{
    Hidden, Null, D3D11, D3D12, Vulkan, Metal, Agx, OpenGL,
};
```

### 3.2 公共 API 设计

#### 3.2.1 后端抽象类 FDynamicRHI

&emsp;&emsp;`FDynamicRHI` 是 UE-RHI 的核心抽象类，所有后端都必须继承并实现其纯虚方法。它定义了约 80+ 个纯虚函数，覆盖资源创建、状态管理、命令提交等所有 GPU 操作：

```cpp
class FDynamicRHI
{
public:
    virtual ~FDynamicRHI() {}
    virtual void Init() = 0;
    virtual void Shutdown() = 0;
    virtual const TCHAR* GetName() = 0;

    // 状态对象创建
    virtual FSamplerStateRHIRef RHICreateSamplerState(const FSamplerStateInitializerRHI& Initializer) = 0;
    virtual FRasterizerStateRHIRef RHICreateRasterizerState(const FRasterizerStateInitializerRHI& Initializer) = 0;
    virtual FDepthStencilStateRHIRef RHICreateDepthStencilState(const FDepthStencilStateInitializerRHI& Initializer) = 0;
    virtual FBlendStateRHIRef RHICreateBlendState(const FBlendStateInitializerRHI& Initializer) = 0;

    // 着色器创建
    virtual FPixelShaderRHIRef RHICreatePixelShader(TArrayView<const uint8> Code, const FSHAHash& Hash) = 0;
    virtual FVertexShaderRHIRef RHICreateVertexShader(TArrayView<const uint8> Code, const FSHAHash& Hash) = 0;
    virtual FComputeShaderRHIRef RHICreateComputeShader(TArrayView<const uint8> Code, const FSHAHash& Hash) = 0;

    // 管线状态对象
    virtual FGraphicsPipelineStateRHIRef RHICreateGraphicsPipelineState(const FGraphicsPipelineStateInitializer& Initializer) = 0;
    virtual FComputePipelineStateRHIRef RHICreateComputePipelineState(FRHIComputeShader* ComputeShader) = 0;

    // 资源创建
    virtual FTextureRHIRef RHICreateTexture(const FRHITextureCreateDesc& CreateDesc) = 0;
    virtual FBufferRHIRef RHICreateBuffer(FRHICommandListBase& RHICmdList, FRHIBufferDesc const& Desc,
        ERHIAccess ResourceState, FRHIResourceCreateInfo& CreateInfo) = 0;

    // 视口与呈现
    virtual FViewportRHIRef RHICreateViewport(void* WindowHandle, uint32 SizeX, uint32 SizeY,
        bool bIsFullscreen, EPixelFormat PreferredPixelFormat) = 0;

    // 命令上下文管理
    virtual IRHIComputeContext* RHIGetDefaultContext() = 0;
    virtual IRHIComputeContext* RHIGetCommandContext(ERHIPipeline Pipeline, FRHIGPUMask GPUMask) = 0;
    virtual IRHIPlatformCommandList* RHIFinalizeContext(IRHIComputeContext* Context) = 0;
    virtual void RHISubmitCommandLists(TArrayView<IRHIPlatformCommandList*> CommandLists, bool bFlushResources) = 0;
};
```

&emsp;&emsp;全局指针 `GDynamicRHI` 指向当前活跃的后端实例，通过 `GetDynamicRHI<T>()` 模板可获取具体后端类型：

```cpp
extern FDynamicRHI* GDynamicRHI;

template<typename TRHI>
FORCEINLINE TRHI* GetDynamicRHI()
{
    return CastDynamicRHI<TRHI>(GDynamicRHI);
}
```

#### 3.2.2 状态初始化器模式

&emsp;&emsp;UE-RHI 采用 **状态初始化器（Initializer）** 模式创建不可变状态对象。每种状态（采样器、光栅化、深度模板、混合）都有对应的 Initializer 结构体，包含所有配置参数。状态对象创建后不可修改，通过哈希和相等比较可被 PSO 缓存系统复用：

```cpp
struct FSamplerStateInitializerRHI
{
    ESamplerFilter Filter = SF_Point;
    ESamplerAddressMode AddressU = AM_Wrap;
    ESamplerAddressMode AddressV = AM_Wrap;
    ESamplerAddressMode AddressW = AM_Wrap;
    float MipBias = 0.0f;
    float MinMipLevel = 0.0f;
    float MaxMipLevel = FLT_MAX;
    int32 MaxAnisotropy = 0;
    uint32 BorderColor = 0;
    ESamplerCompareFunction SamplerComparisonFunction = SCF_Never;
};

struct FRasterizerStateInitializerRHI
{
    ERasterizerFillMode FillMode = FM_Point;
    ERasterizerCullMode CullMode = CM_None;
    float DepthBias = 0.0f;
    float SlopeScaleDepthBias = 0.0f;
    ERasterizerDepthClipMode DepthClipMode = ERasterizerDepthClipMode::DepthClip;
    bool bAllowMSAA = false;
    bool bEnableLineAA = false;
};

struct FDepthStencilStateInitializerRHI
{
    bool bEnableDepthWrite;
    ECompareFunction DepthTest;
    bool bEnableFrontFaceStencil;
    ECompareFunction FrontFaceStencilTest;
    EStencilOp FrontFaceStencilFailStencilOp;
    // ... 完整的模板测试配置
};

class FBlendStateInitializerRHI
{
public:
    struct FRenderTarget
    {
        EBlendOperation ColorBlendOp;
        EBlendFactor ColorSrcBlend;
        EBlendFactor ColorDestBlend;
        EBlendOperation AlphaBlendOp;
        EBlendFactor AlphaSrcBlend;
        EBlendFactor AlphaDestBlend;
        EColorWriteMask ColorWriteMask;
    };
    TStaticArray<FRenderTarget, MaxSimultaneousRenderTargets> RenderTargets;
    bool bUseIndependentRenderTargetBlendStates;
    bool bUseAlphaToCoverage;
};
```

#### 3.2.3 资源创建方式

&emsp;&emsp;资源通过 `FRHIResourceCreateInfo` 提供初始数据，支持直接数据、BulkData、ResourceArray 三种方式。纹理创建使用 `FRHITextureCreateDesc` 描述符：

```cpp
struct FRHIResourceCreateInfo
{
    FResourceBulkDataInterface* BulkData;       // 用于纹理
    FResourceArrayInterface* ResourceArray;     // 用于缓冲区
    FClearValueBinding ClearValueBinding;
    FRHIGPUMask GPUMask;                        // 多GPU支持
    bool bWithoutNativeResource;
    const TCHAR* DebugName;
    uint32 ExtData;
};

// 缓冲区创建
FBufferRHIRef RHICreateBuffer(FRHICommandListBase& RHICmdList, FRHIBufferDesc const& Desc,
    ERHIAccess ResourceState, FRHIResourceCreateInfo& CreateInfo);

// 纹理创建
FTextureRHIRef RHICreateTexture(const FRHITextureCreateDesc& CreateDesc);
```

### 3.3 后端抽象层

&emsp;&emsp;UE-RHI 的后端抽象通过 `FDynamicRHI` 虚基类实现，每个后端模块编译为独立的动态库。后端实现的核心文件组织如下（以 D3D12 为例）：

| 文件 | 职责 |
|------|------|
| `D3D12RHI.h` | 公共定义（SRV/UAV/CB 限制、平台宏等） |
| `D3D12RHIPrivate.h` | 私有头文件，包含所有内部依赖 |
| `D3D12Adapter.h/cpp` | DXGI 适配器封装，枚举设备 |
| `D3D12Device.h/cpp` | ID3D12Device 封装，设备创建与能力查询 |
| `D3D12RHI.cpp` | FD3D12DynamicRHI 实现，RHI 接口入口 |
| `D3D12CommandList.h/cpp` | D3D12 命令列表管理 |
| `D3D12CommandContext.h/cpp` | 命令上下文，翻译 RHI 调用到 D3D12 |
| `D3D12Resources.h/cpp` | D3D12 资源封装 |
| `D3D12Texture.h/cpp` | D3D12 纹理实现 |
| `D3D12Buffer.cpp` | D3D12 缓冲区实现 |
| `D3D12PipelineState.h/cpp` | D3D12 PSO 管理 |
| `D3D12RootSignature.h/cpp` | D3D12 根签名管理 |
| `D3D12Shader.h` | D3D12 着色器封装 |
| `D3D12Descriptors.h/cpp` | D3D12 描述符堆管理 |
| `D3D12BindlessDescriptors.h/cpp` | Bindless 描述符支持 |
| `D3D12Allocation.h/cpp` | GPU 内存分配（D3D12MA） |
| `D3D12TransientResourceAllocator.h/cpp` | 瞬态资源分配 |
| `D3D12Submission.h/cpp` | 命令队列提交 |
| `D3D12Viewport.h/cpp` | 交换链/视口管理 |
| `D3D12RayTracing.h/cpp` | 光线追踪支持 |
| `D3D12StateCachePrivate.h` | PSO 磁盘缓存 |

&emsp;&emsp;Vulkan 后端结构类似，核心类为 `FVulkanDynamicRHI`，继承自 `IVulkanDynamicRHI`（平台扩展接口）再继承自 `FDynamicRHI`。Vulkan 后端通过 `IVulkanDynamicRHI` 接口暴露原生 Vulkan 对象（VkInstance、VkDevice、VkPhysicalDevice 等），供需要原生访问的插件使用：

```cpp
class FVulkanDynamicRHI : public IVulkanDynamicRHI
{
    // IVulkanDynamicRHI 原生访问接口
    virtual VkInstance RHIGetVkInstance() const override;
    virtual VkDevice RHIGetVkDevice() const override;
    virtual VkPhysicalDevice RHIGetVkPhysicalDevice() const override;
    virtual VkQueue RHIGetGraphicsVkQueue() const override;
    virtual VkImage RHIGetVkImage(FRHITexture* InTexture) const override;
    // ... 完整的 FDynamicRHI 接口实现
};
```

### 3.4 资源管理

#### 3.4.1 引用计数与生命周期

&emsp;&emsp;所有 RHI 资源继承自 `FRHIResource`，使用 **原子引用计数** 管理生命周期。引用计数器 `FAtomicFlags` 将引用计数、删除标记、删除中标记打包到单个 `std::atomic_uint` 中，通过原子操作实现无锁引用计数：

```cpp
class FRHIResource
{
    class FAtomicFlags
    {
        static constexpr uint32 MarkedForDeleteBit = 1 << 30;
        static constexpr uint32 DeletingBit        = 1 << 31;
        static constexpr uint32 NumRefsMask        = ~(MarkedForDeleteBit | DeletingBit);
        std::atomic_uint Packed = { 0 };

    public:
        int32 AddRef(std::memory_order MemoryOrder) {
            uint32 OldPacked = Packed.fetch_add(1, MemoryOrder);
            return (OldPacked & NumRefsMask) + 1;
        }
        int32 Release(std::memory_order MemoryOrder) {
            uint32 OldPacked = Packed.fetch_sub(1, MemoryOrder);
            return (OldPacked & NumRefsMask) - 1;
        }
        bool MarkForDelete(std::memory_order MemoryOrder) {
            uint32 OldPacked = Packed.fetch_or(MarkedForDeleteBit, MemoryOrder);
            return (OldPacked & MarkedForDeleteBit) != 0;
        }
    };

    uint32 AddRef() const;   // acquire 语义
    uint32 Release() const;  // release 语义，归零时调用 Destroy()
    void Delete();           // 标记删除
};
```

&emsp;&emsp;资源销毁通过 **延迟删除队列** 实现。当引用计数归零时，资源被放入 `PendingDeletes` 或 `PendingDeletesWithLifetimeExtension` 队列（MPMC 无锁队列），在渲染线程的 `FlushPendingDeletes` 中统一销毁，避免在非渲染线程直接释放 GPU 资源：

```cpp
void FRHIResource::Destroy() const
{
    if (!AtomicFlags.MarkForDelete(std::memory_order_release))
    {
        if (bAllowExtendLifetime)
            PendingDeletesWithLifetimeExtension.ProduceItem(const_cast<FRHIResource*>(this));
        else
            PendingDeletes.ProduceItem(const_cast<FRHIResource*>(this));
    }
}
```

#### 3.4.2 瞬态资源分配

&emsp;&emsp;UE-RHI 提供了完整的 **瞬态资源分配系统**，用于帧内临时纹理和缓冲区的高效分配与复用。`IRHITransientResourceAllocator` 是后端必须实现的接口：

```cpp
struct FRHITransientHeapAllocation
{
    FRHITransientHeap* Heap;
    uint64 Size;        // 对齐后的分配大小
    uint64 Offset;      // 堆内偏移
    uint32 AlignmentPad;
};

struct FRHITransientPageAllocation
{
    TArray<FRHITransientPagePoolAllocation> PoolAllocations;
    TArray<FRHITransientPageSpan> Spans;
};

class FRHITransientResource
{
    FRHIResource* Resource;
    uint64 GpuVirtualAddress;
    uint64 Hash;
    uint64 Size;
    ERHITransientAllocationType AllocationType;  // Heap 或 Page
    ERHITransientResourceType ResourceType;       // Texture 或 Buffer

    void Acquire(const TCHAR* InName, uint32 InAcquirePassIndex, uint64 InAllocatorCycle);
    void Discard(uint32 InDiscardPassIndex);
    void AddAliasingOverlap(FRHITransientResource* InResource);
};
```

&emsp;&emsp;瞬态资源支持两种分配策略：**堆分配**（`Heap`）和 **页分配**（`Page`）。堆分配器适用于大块连续内存，页分配器适用于小块离散分配。通过 `AliasingOverlaps` 机制，生命周期不重叠的瞬态资源可以共享同一块物理内存（**资源别名/Aliasing**），大幅减少 GPU 内存占用。

#### 3.4.3 GPU 内存分配

&emsp;&emsp;D3D12 后端使用 `FD3D12Allocation` 体系管理 GPU 内存，通过 `FD3D12Allocator`（基于 D3D12MA 或自定义分配器）提供子分配能力。Vulkan 后端类似地使用 VMA（Vulkan Memory Allocator）。核心内存分配结构：

```cpp
// D3D12 分配结构（简化）
struct FD3D12ResourceLocation
{
    FD3D12Resource* Resource;
    uint64 Offset;
    // 支持上传堆、读回堆、默认堆等不同堆类型
};

// 资源驻留管理
class FD3D12ResidencyManager
{
    // 管理 GPU 资源的驻留/淘汰（Eviction）
    // 当 GPU 内存不足时，将不常用的资源淘汰到系统内存
};
```

#### 3.4.4 资源状态转换

&emsp;&emsp;UE-RHI 实现了完整的 **资源状态跟踪与转换（Barriers）** 系统。`ERHIAccess` 枚举定义了所有可能的资源访问状态：

```cpp
enum class ERHIAccess : uint32
{
    Unknown          = 0,       // 未知状态，需全量屏障
    CPURead          = 1 << 0,  // CPU 可读
    Present          = 1 << 1,  // 呈现
    IndirectArgs     = 1 << 2,  // 间接参数
    VertexOrIndexBuffer = 1 << 3,
    SRVCompute       = 1 << 4,  // 计算着色器 SRV
    SRVGraphics      = 1 << 5,  // 图形着色器 SRV
    CopySrc          = 1 << 6,
    DSVRead          = 1 << 8,  // 深度模板只读
    UAVCompute       = 1 << 9,  // 计算 UAV
    UAVGraphics      = 1 << 10, // 图形 UAV
    RTV              = 1 << 11, // 渲染目标
    CopyDest         = 1 << 12,
    DSVWrite         = 1 << 14, // 深度模板读写
    BVHRead          = 1 << 15, // 光线追踪加速结构
    BVHWrite         = 1 << 16,
    Discard          = 1 << 17, // 瞬态资源释放状态
    ShadingRateSource = 1 << 18,
};
```

&emsp;&emsp;资源转换通过 `FRHITransitionInfo` 描述，支持子资源级别的精细控制：

```cpp
struct FRHITransitionInfo : public FRHISubresourceRange
{
    union {
        FRHIResource* Resource;
        FRHITexture* Texture;
        FRHIBuffer* Buffer;
        FRHIUnorderedAccessView* UAV;
        FRHIRayTracingAccelerationStructure* BVH;
    };
    ERHIAccess AccessBefore;
    ERHIAccess AccessAfter;
    EResourceTransitionFlags Flags;
    // 支持 MipIndex、ArraySlice、PlaneSlice 子资源粒度
};
```

&emsp;&emsp;状态转换由 `FRHITransition` 对象封装，后端通过 `RHICreateTransition` / `RHIReleaseTransition` 实现平台特定的 barrier 逻辑。UE-RHI 还支持 **状态合并**（State Merging），通过 `GRHIMergeableAccessMask` 控制可合并的状态范围，减少不必要的 barrier。

### 3.5 多线程模型

#### 3.5.1 命令列表架构

&emsp;&emsp;UE-RHI 采用 **命令列表 + 并行翻译 + RHI 线程执行** 的多线程模型，这是其最核心的架构特征之一：

```
┌─────────────────────────────────────────────────┐
│   渲染线程 (Render Thread)                        │
│   FRHICommandListImmediate (立即命令列表)          │
│   → 录制 RHI 命令                                │
├─────────────────────────────────────────────────┤
│   并行工作线程 (Parallel Workers)                 │
│   FRHICommandList (并行命令列表)                  │
│   → 录制命令 → FinishRecording()                 │
│   → 触发并行翻译到平台命令列表                     │
├─────────────────────────────────────────────────┤
│   RHI 线程 (RHI Thread)                          │
│   → 按顺序执行翻译后的平台命令列表                  │
│   → 提交到 GPU 命令队列                          │
├─────────────────────────────────────────────────┤
│   异步计算队列 (Async Compute)                    │
│   → 并行执行计算着色器                            │
└─────────────────────────────────────────────────┘
```

#### 3.5.2 命令录制与延迟执行

&emsp;&emsp;RHI 命令采用 **延迟录制** 模式。命令被序列化为类型擦除的 `FRHICommandBase` 对象链表，存储在线性分配器中。每种 RHI 调用对应一个 `FRHICommand<TCmd>` 派生类：

```cpp
struct FRHICommandBase
{
    FRHICommandBase* Next = nullptr;
    virtual void ExecuteAndDestruct(FRHICommandListBase& CmdList,
        FRHICommandListDebugContext& DebugContext) = 0;
};

template<typename TCmd, typename NameType>
struct FRHICommand : public FRHICommandBase
{
    void ExecuteAndDestruct(FRHICommandListBase& CmdList,
        FRHICommandListDebugContext& Context) override final
    {
        TCmd* ThisCmd = static_cast<TCmd*>(this);
        ThisCmd->Execute(CmdList);  // 类型安全的执行
        ThisCmd->~TCmd();
    }
};
```

&emsp;&emsp;命令通过 `ALLOC_COMMAND` 宏在命令列表的线性内存分配器上分配，零堆分配开销：

```cpp
#define ALLOC_COMMAND(...) new ( AllocCommand(sizeof(__VA_ARGS__), alignof(__VA_ARGS__)) ) __VA_ARGS__
```

#### 3.5.3 并行命令列表

&emsp;&emsp;渲染器可以创建多个并行命令列表 `FRHICommandList`，在工作线程上同时录制命令。录制完成后通过 `FinishRecording()` 标记为可执行，触发并行翻译（Parallel Translation）：

```cpp
class FRHICommandListBase : public FNoncopyable
{
    void* AllocCommand(int32 AllocSize, int32 Alignment);
    void FinishRecording();  // 标记完成，可被调度执行

    template<typename LAMBDA>
    void EnqueueLambda(LAMBDA&& Lambda);  // 入队 Lambda 命令

    ERHIPipeline SwitchPipeline(ERHIPipeline Pipeline);  // 切换图形/计算管线
};
```

#### 3.5.4 Bypass 模式与即时执行

&emsp;&emsp;UE-RHI 支持 **Bypass 模式**：当命令列表处于 Bypass 状态时，命令不被录制而是立即执行（通过 `IRHICommandContext` 直接调用后端）。这用于简化调试和特殊场景：

```cpp
bool IsBottomOfPipe() const { return Bypass() || IsExecuting(); }

template<typename LAMBDA>
void EnqueueLambda(LAMBDA&& Lambda)
{
    if (IsBottomOfPipe())
        Lambda(*this);  // 立即执行
    else
        ALLOC_COMMAND(TRHILambdaCommand<...>)(Forward<LAMBDA>(Lambda));  // 录制
}
```

#### 3.5.5 线程安全设计

&emsp;&emsp;UE-RHI 的线程安全设计通过多个机制保证：

| 机制 | 说明 |
|------|------|
| **原子引用计数** | `FRHIResource::AddRef/Release` 使用 `std::atomic` + `memory_order_acquire/release` |
| **命令列表隔离** | 每个命令列表独立录制，通过线性分配器避免竞争 |
| **RHI 线程串行化** | 所有命令最终在 RHI 线程上串行执行 |
| **FlushType 注释** | 每个 API 函数都标注了线程安全级别（Thread safe / Wait RHI Thread / Flush Immediate 等） |
| **FScopedRHIThreadStaller** | 在需要时暂停 RHI 线程以安全访问共享状态 |

### 3.6 着色器处理

#### 3.6.1 着色器类型与频率

&emsp;&emsp;UE-RHI 定义了 10 种着色器频率（`EShaderFrequency`），覆盖图形、计算和光线追踪管线：

```cpp
enum EShaderFrequency : uint8
{
    SF_Vertex        = 0,   SF_Mesh         = 1,
    SF_Amplification = 2,   SF_Pixel        = 3,
    SF_Geometry      = 4,   SF_Compute      = 5,
    SF_RayGen        = 6,   SF_RayMiss      = 7,
    SF_RayHitGroup   = 8,   SF_RayCallable  = 9,
    SF_NumFrequencies = 10,
};
```

&emsp;&emsp;着色器类层次结构：

```
FRHIResource
├── FRHIShader (基类, 含 Hash + Frequency)
│   ├── FRHIGraphicsShader
│   │   ├── FRHIVertexShader
│   │   ├── FRHIMeshShader
│   │   ├── FRHIAmplificationShader
│   │   ├── FRHIPixelShader
│   │   └── FRHIGeometryShader
│   ├── FRHIComputeShader (含 Stats)
│   └── FRHIRayTracingShader (含 PayloadType/PayloadSize)
│       ├── FRHIRayGenShader
│       ├── FRHIRayMissShader
│       ├── FRHIRayCallableShader
│       └── FRHIRayHitGroupShader
```

#### 3.6.2 着色器参数绑定模型

&emsp;&emsp;UE-RHI 使用 **批量着色器参数**（Batched Shader Parameters）模型。参数通过 `FRHIBatchedShaderParameters` 收集，在 `SetBatchedShaderParameters` 调用时绑定到着色器：

```cpp
struct FRHIShaderParameter
{
    uint16 BufferIndex;   // 绑定槽位
    uint16 BaseIndex;     // 数据起始索引
    uint16 ByteOffset;    // 参数数据偏移
    uint16 ByteSize;      // 参数数据大小
};

struct FRHIShaderParameterResource
{
    enum class EType : uint8 {
        Texture, ResourceView, UnorderedAccessView, Sampler, UniformBuffer
    };
    FRHIResource* Resource;
    uint16 Index;
    EType Type;
};

struct FRHIBatchedShaderParameters
{
    TArray<uint8> ParametersData;                    // 原始参数数据块
    TArray<FRHIShaderParameter> Parameters;          // 常量参数
    TArray<FRHIShaderParameterResource> ResourceParameters;  // 资源绑定
    TArray<FRHIShaderParameterResource> BindlessParameters;  // Bindless 绑定
};
```

&emsp;&emsp;UE-RHI 支持 **Bindless 资源绑定**，通过 `ERHIBindlessConfiguration` 控制 Bindless 纹理和采样器的使用级别。Bindless 参数存储在 `BindlessParameters` 数组中，后端实现将这些参数映射到描述符堆中的 GPU 可见描述符。

#### 3.6.3 Uniform Buffer

&emsp;&emsp;UE-RHI 使用 `FRHIUniformBuffer` 作为着色器常量的载体，采用 **不可变创建** 模式——创建时传入内容，后续只能通过 `RHIUpdateUniformBuffer` 更新整个缓冲区内容：

```cpp
virtual FUniformBufferRHIRef RHICreateUniformBuffer(
    const void* Contents, const FRHIUniformBufferLayout* Layout,
    EUniformBufferUsage Usage, EUniformBufferValidation Validation) = 0;

virtual void RHIUpdateUniformBuffer(FRHICommandListBase& RHICmdList,
    FRHIUniformBuffer* UniformBufferRHI, const void* Contents) = 0;
```

&emsp;&emsp;Uniform Buffer 布局通过 `FRHIUniformBufferLayout` 描述，包含静态绑定槽位（`FUniformBufferStaticSlot`）用于全局绑定（如 View、Scene、Primitive 等）。

#### 3.6.4 PSO 缓存与预编译

&emsp;&emsp;UE-RHI 实现了完整的 **Pipeline State Object 缓存系统**（`PipelineStateCache.h`），支持运行时缓存、磁盘序列化和异步预编译：

```cpp
namespace PipelineStateCache
{
    FGraphicsPipelineState* GetAndOrCreateGraphicsPipelineState(
        FRHICommandList& RHICmdList, const FGraphicsPipelineStateInitializer& Initializer,
        EApplyRendertargetOption ApplyFlags, EPSOPrecacheResult PSOPrecacheResult);

    FPSOPrecacheRequestResult PrecacheGraphicsPipelineState(
        const FGraphicsPipelineStateInitializer& Initializer, ...);

    FPSOPrecacheRequestResult PrecacheComputePipelineState(
        FRHIComputeShader* ComputeShader, bool bForcePrecache = false);
}
```

&emsp;&emsp;预编译系统支持在加载屏幕期间提前编译 PSO，减少运行时卡顿。`EPSOPrecacheResult` 跟踪预编译状态：Active（编译中）、Complete（完成）、Missed（未命中需运行时编译）、TooLate（编译太晚）等。

### 3.7 设计特点

&emsp;&emsp;UE-RHI 相对于其他 RHI 实现的独特设计决策包括：

1. **大型引擎深度集成**：不同于 bgfx/The-Forge 等独立库，UE-RHI 深度嵌入 Unreal Engine 渲染器，与 FRHICommandList、FGPUScene、FRDGBuilder 等模块紧密协作，提供最完整的 AAA 级渲染支持。

2. **三线程命令执行模型**：渲染线程录制 → 并行翻译 → RHI 线程执行，这是目前最复杂的多线程 RHI 架构之一。通过命令列表的 Bypass 模式支持直接执行路径。

3. **完整的资源状态跟踪**：`ERHIAccess` 枚举覆盖 18+ 种资源状态，支持子资源级别的 barrier 控制（Mip、ArraySlice、PlaneSlice），并通过状态合并减少冗余 barrier。

4. **瞬态资源别名系统**：通过 `AliasingOverlaps` 机制，生命周期不重叠的纹理/缓冲区共享物理内存，是大规模场景渲染中内存优化的关键。

5. **Bindless 渐进支持**：通过 `ERHIBindlessConfiguration` 提供分级的 Bindless 支持（None → Partial → Full），兼容传统绑定模型和现代 Bindless 架构。

6. **PSO 完整生命周期管理**：从运行时缓存、磁盘序列化、异步预编译到驱逐策略，形成完整的 PSO 管理体系，是解决现代图形 API（D3D12/Vulkan）PSO 编译卡顿问题的重要方案。

7. **多 GPU 支持**：原生支持 `FRHIGPUMask` 多 GPU 掩码和 `FRHITransferResources` 跨 GPU 资源传输，支持 NVIDIA SLI / AMD CrossFire 等多 GPU 配置。

8. **光线追踪一等公民**：RHI 层原生支持 `RHI_RAYTRACING`，提供完整的加速结构（BLAS/TLAS）、光线追踪管线、Shader Binding Table 管理。

9. **验证层集成**：内置 `RHIValidation` 模块，在开发构建中自动验证 API 使用正确性（资源状态、线程安全、参数有效性等），帮助开发者及早发现错误。

&emsp;&emsp;总体而言，UE-RHI 是目前功能最完整、架构最复杂的 RHI 实现之一。它的设计哲学是"为大型引擎提供完整的跨平台图形抽象"，通过复杂的命令列表系统、资源状态管理和 PSO 缓存机制，在保持跨平台兼容性的同时最大化 GPU 利用率。虽然学习曲线陡峭，但它代表了工业级 RHI 实现的最高水平。

## 4 Diligent Engine

&emsp;&emsp;Diligent Engine 是一个现代、跨平台的底层图形 API 抽象层和渲染框架，由 Diligent Graphics LLC 开发，采用 Apache 2.0 许可证开源。它支持 Direct3D 11、Direct3D 12、OpenGL/OpenGL ES、Vulkan、Metal、WebGPU 等多种图形后端，覆盖 Windows、Linux、macOS、iOS、Android、Web 等平台。Diligent Engine 的设计哲学是提供**一致的前端 API**，使用 HLSL 作为统一着色语言，同时支持平台特定的着色器格式（GLSL、MSL、SPIR-V、DX 字节码），并在性能上充分利用现代图形 API 的特性。

### 4.1 整体架构

#### 4.1.1 模块化仓库结构

&emsp;&emsp;Diligent Engine 采用模块化的 Git 子仓库（submodule）组织方式，各模块物理和逻辑上清晰分离：

```
DiligentEngine/                    (主仓库)
├── DiligentCore/                  (核心图形抽象层)
│   ├── Graphics/
│   │   ├── GraphicsEngine/        (公共接口定义)
│   │   │   └── interface/         (所有 IRenderDevice, IDeviceContext 等接口)
│   │   ├── GraphicsEngineD3D11/   (Direct3D 11 后端)
│   │   ├── GraphicsEngineD3D12/   (Direct3D 12 后端)
│   │   ├── GraphicsEngineD3DBase/ (D3D11/D3D12 共享基础设施)
│   │   ├── GraphicsEngineOpenGL/  (OpenGL/GLES 后端)
│   │   ├── GraphicsEngineVulkan/  (Vulkan 后端)
│   │   ├── GraphicsEngineMetal/   (Metal 后端，商业许可)
│   │   ├── GraphicsEngineWebGPU/  (WebGPU 后端)
│   │   ├── GraphicsEngineNextGenBase/ (下一代 API 共享基础)
│   │   ├── GraphicsAccessories/   (纹理加载、格式转换等工具)
│   │   ├── ShaderTools/           (着色器编译与反射)
│   │   ├── HLSL2GLSLConverterLib/ (HLSL→GLSL 转换)
│   │   └── SuperResolution/       (超分辨率支持)
│   ├── Common/                    (通用工具类)
│   ├── Platforms/                 (平台抽象层)
│   ├── Primitives/                (基础类型、对象模型)
│   └── ThirdParty/               (第三方依赖)
├── DiligentTools/                 (工具模块：纹理加载、ImGui、资源包等)
├── DiligentFX/                    (高级渲染效果：PBR、Bloom、SSR 等)
├── DiligentSamples/               (示例与教程)
└── Tests/                         (单元测试)
```

&emsp;&emsp;依赖关系如下：

```
     Core
      │
      ├──────►Tools───────────┐
      │       │               │
      │       ▼               │
      ├──────►FX──────────┐   │
      │                  │   │
      │                  ▼   ▼
      └───────────────►Samples
```

&emsp;&emsp;各模块可独立使用，用户只需链接所需的后端库。CMake 构建系统支持通过 `DILIGENT_NO_DIRECT3D11`、`DILIGENT_NO_VULKAN` 等选项选择性禁用特定后端。

#### 4.1.2 后端组织架构

&emsp;&emsp;Diligent Core 的 `Graphics/` 目录是整个 RHI 的核心，其组织遵循"接口 + 实现"分离模式：

```
Graphics/
├── GraphicsEngine/
│   └── interface/          ← 所有公共接口与数据结构
│       ├── RenderDevice.h        IRenderDevice 接口
│       ├── DeviceContext.h       IDeviceContext 接口
│       ├── PipelineState.h       IPipelineState 接口与 PSO 描述符
│       ├── Buffer.h / Texture.h  资源描述与接口
│       ├── Shader.h              着色器接口
│       ├── ShaderResourceBinding.h  SRB 接口
│       ├── EngineFactory.h       引擎工厂接口
│       ├── GraphicsTypes.h       通用图形类型定义
│       ├── CommandQueue.h        命令队列接口
│       ├── DeviceMemory.h        设备内存接口
│       ├── Fence.h / Query.h     同步与查询
│       ├── RenderPass.h          渲染通道
│       ├── Framebuffer.h         帧缓冲
│       ├── BottomLevelAS.h       底层加速结构（光线追踪）
│       ├── TopLevelAS.h          顶层加速结构
│       ├── ShaderBindingTable.h  着色器绑定表
│       └── PipelineResourceSignature.h  管线资源签名
├── GraphicsEngineD3D11/    ← D3D11 实现
├── GraphicsEngineD3D12/    ← D3D12 实现
├── GraphicsEngineD3DBase/  ← D3D11/D3D12 共享代码
├── GraphicsEngineOpenGL/   ← OpenGL/GLES 实现
├── GraphicsEngineVulkan/   ← Vulkan 实现
├── GraphicsEngineMetal/    ← Metal 实现
└── GraphicsEngineWebGPU/   ← WebGPU 实现
```

&emsp;&emsp;这种设计的核心思想是：**公共接口（interface）定义所有后端必须实现的契约，各后端目录提供具体实现**。用户代码只依赖 `interface/` 中的头文件，不需要包含任何后端特定的代码。

### 4.2 公共 API 设计

#### 4.2.1 核心对象模型

&emsp;&emsp;Diligent Engine 采用**基于 COM 风格的引用计数对象模型**。所有 GPU 对象都继承自 `IObject`，通过 `RefCntAutoPtr<T>` 智能指针管理生命周期：

```cpp
// 所有对象的基类
class IObject {
    virtual Uint32 METHOD(AddRef)(THIS) PURE;
    virtual Uint32 METHOD(Release)(THIS) PURE;
};

// 资源对象继承自 IDeviceObject → IObject
class IDeviceObject : public IObject {
    virtual const DeviceObjectAttribs& METHOD(GetDesc)(THIS) CONST PURE;
};

// 具体资源接口
class IBuffer      : public IDeviceObject { ... };
class ITexture     : public IDeviceObject { ... };
class IShader      : public IDeviceObject { ... };
class IPipelineState : public IDeviceObject { ... };
class ISampler     : public IDeviceObject { ... };
```

&emsp;&emsp;对象创建通过 `IRenderDevice` 接口统一完成：

```cpp
class IRenderDevice : public IObject {
    void CreateBuffer(const BufferDesc& BuffDesc, const BufferData* pBuffData, IBuffer** ppBuffer);
    void CreateTexture(const TextureDesc& TexDesc, const TextureData* pData, ITexture** ppTexture);
    void CreateShader(const ShaderCreateInfo& ShaderCI, IShader** ppShader, IDataBlob** ppCompilerOutput = nullptr);
    void CreateSampler(const SamplerDesc& SamDesc, ISampler** ppSampler);
    void CreateGraphicsPipelineState(const GraphicsPipelineStateCreateInfo& CI, IPipelineState** ppPSO);
    void CreateComputePipelineState(const ComputePipelineStateCreateInfo& CI, IPipelineState** ppPSO);
    void CreateRayTracingPipelineState(const RayTracingPipelineStateCreateInfo& CI, IPipelineState** ppPSO);
    void CreateFence(const FenceDesc& Desc, IFence** ppFence);
    void CreateQuery(const QueryDesc& Desc, IQuery** ppQuery);
    void CreateRenderPass(const RenderPassDesc& Desc, IRenderPass** ppRenderPass);
    void CreateFramebuffer(const FramebufferDesc& Desc, IFramebuffer** ppFramebuffer);
    void CreateBLAS(const BottomLevelASDesc& Desc, IBottomLevelAS** ppBLAS);
    void CreateTLAS(const TopLevelASDesc& Desc, ITopLevelAS** ppTLAS);
    void CreateSBT(const ShaderBindingTableDesc& Desc, IShaderBindingTable** ppSBT);
    void CreatePipelineResourceSignature(const PipelineResourceSignatureDesc& Desc, IPipelineResourceSignature** ppSignature);
    void CreateDeviceMemory(const DeviceMemoryCreateInfo& CreateInfo, IDeviceMemory** ppMemory);
    void CreatePipelineStateCache(const PipelineStateCacheCreateInfo& CreateInfo, IPipelineStateCache** ppPSOCache);
    void CreateDeferredContext(IDeviceContext** ppContext);
};
```

#### 4.2.2 引擎初始化模型

&emsp;&emsp;Diligent Engine 的初始化采用**工厂模式**，每个后端有独立的工厂接口。用户通过加载动态库或静态链接获取工厂，然后创建设备和上下文：

```cpp
// 以 D3D12 为例
auto* pFactoryD3D12 = GetEngineFactoryD3D12();  // 静态链接
// 或 auto* pFactoryD3D12 = LoadGraphicsEngineD3D12();  // 动态加载 DLL

EngineD3D12CreateInfo EngineCI;
pFactoryD3D12->CreateDeviceAndContextsD3D12(EngineCI, &m_pDevice, &m_pImmediateContext);

Win32NativeWindow Window{hWnd};
pFactoryD3D12->CreateSwapChainD3D12(m_pDevice, m_pImmediateContext, SCDesc,
                                     FullScreenModeDesc{}, Window, &m_pSwapChain);
```

&emsp;&emsp;各后端的工厂接口：

| 后端 | 工厂函数 | 创建信息结构 |
|------|---------|-------------|
| D3D11 | `GetEngineFactoryD3D11()` | `EngineD3D11CreateInfo` |
| D3D12 | `GetEngineFactoryD3D12()` | `EngineD3D12CreateInfo` |
| OpenGL | `GetEngineFactoryOpenGL()` | `EngineGLCreateInfo` |
| Vulkan | `GetEngineFactoryVk()` | `EngineVkCreateInfo` |
| Metal | `GetEngineFactoryMtl()` | `EngineMtlCreateInfo` |
| WebGPU | `GetEngineFactoryWebGPU()` | `EngineWebGPUCreateInfo` |

&emsp;&emsp;引擎还支持**附着到已有的原生设备/上下文**，实现与现有渲染环境的互操作。例如 OpenGL 后端可以附着到应用程序已经创建的 GL context。

#### 4.2.3 命令提交模型

&emsp;&emsp;Diligent Engine 的命令提交通过 `IDeviceContext` 接口完成，支持**立即上下文（Immediate Context）**和**延迟上下文（Deferred Context）**两种模式：

```cpp
class IDeviceContext : public IObject {
    // 资源状态转换
    void TransitionResourceStates(Uint32 BarrierCount, StateTransitionDesc* pBarriers);

    // 渲染目标设置
    void SetRenderTargets(Uint32 NumRenderTargets, ITextureView** ppRTVs,
                          ITextureView* pDSV, RESOURCE_STATE_TRANSITION_MODE StateTransitionMode);

    // 管线与资源绑定
    void SetPipelineState(IPipelineState* pPipelineState);
    void CommitShaderResources(IShaderResourceBinding* pSRB,
                               COMMIT_SHADER_RESOURCES_FLAGS Flags = COMMIT_SHADER_RESOURCES_FLAG_NONE);

    // 绘制命令
    void Draw(const DrawAttribs& DrawAttrs);
    void DrawIndexed(const DrawIndexedAttribs& DrawAttrs);
    void DrawIndirect(const DrawIndirectAttribs& DrawAttrs);
    void DrawIndexedIndirect(const DrawIndexedIndirectAttribs& DrawAttrs);
    void DrawMesh(const DrawMeshAttribs& DrawAttrs);
    void MultiDraw(const MultiDrawAttribs& DrawAttrs);
    void MultiDrawIndexed(const MultiDrawIndexedIndexedAttribs& DrawAttrs);

    // 计算命令
    void DispatchCompute(const DispatchComputeAttribs& DispatchAttrs);
    void DispatchComputeIndirect(const DispatchComputeIndirectAttribs& DispatchAttrs);

    // 光线追踪命令
    void TraceRays(const TraceRaysAttribs& TraceRaysAttrs);
    void TraceRaysIndirect(const TraceRaysIndirectAttribs& TraceRaysAttrs);

    // 命令列表（用于延迟上下文多线程录制）
    IDeviceContext* BeginCommandList(const COMMAND_LIST_TYPE& CmdListType);
    void ExecuteCommandList(IDeviceContext* pCmdList);

    // 队列操作
    void EnqueueSignal(IFence* pFence, Uint64 Value);
    void WaitIdle();

    // 纹理操作
    void CopyTexture(const CopyTextureAttribs& CopyAttrs);
    void ResolveTextureSubresource(const ResolveTextureSubresourceAttribs& ResolveAttrs);

    // 缓冲区映射
    void* MapBuffer(IBuffer* pBuffer, MAP_TYPE MapType, MAP_FLAGS MapFlags);
    void UnmapBuffer(IBuffer* pBuffer, MAP_TYPE MapType);
};
```

&emsp;&emsp;资源状态转换有三种模式：

| 模式 | 说明 |
|------|------|
| `RESOURCE_STATE_TRANSITION_MODE_NONE` | 不执行状态转换 |
| `RESOURCE_STATE_TRANSITION_MODE_TRANSITION` | 自动将资源转换到命令所需的状态 |
| `RESOURCE_STATE_TRANSITION_MODE_VERIFY` | 不转换，但验证状态是否正确（仅 Debug 构建） |

#### 4.2.4 资源描述符

&emsp;&emsp;所有资源通过描述符结构体创建，描述符具有默认值构造函数，用户只需设置需要修改的字段：

```cpp
BufferDesc BuffDesc;
BuffDesc.Name           = "Uniform buffer";
BuffDesc.BindFlags      = BIND_UNIFORM_BUFFER;
BuffDesc.Usage          = USAGE_DYNAMIC;
BuffDesc.uiSizeInBytes  = sizeof(ShaderConstants);
BuffDesc.CPUAccessFlags = CPU_ACCESS_WRITE;
m_pDevice->CreateBuffer(BuffDesc, nullptr, &m_pConstantBuffer);

TextureDesc TexDesc;
TexDesc.Name      = "My texture 2D";
TexDesc.Type      = TEXTURE_TYPE_2D;
TexDesc.Width     = 1024;
TexDesc.Height    = 1024;
TexDesc.Format    = TEX_FORMAT_RGBA8_UNORM;
TexDesc.Usage     = USAGE_DEFAULT;
TexDesc.BindFlags = BIND_SHADER_RESOURCE | BIND_RENDER_TARGET | BIND_UNORDERED_ACCESS;
m_pDevice->CreateTexture(TexDesc, nullptr, &m_pTestTex);
```

&emsp;&emsp;关键枚举类型：

| 枚举 | 值 | 说明 |
|------|-----|------|
| `USAGE_DEFAULT` | 0 | GPU 读写，不可 CPU 映射 |
| `USAGE_IMMUTABLE` | 1 | 创建时初始化，之后不可修改 |
| `USAGE_DYNAMIC` | 2 | CPU 可写，GPU 只读 |
| `USAGE_STAGING` | 3 | CPU 可读，用于回读 |
| `BIND_UNIFORM_BUFFER` | 0x01 | 常量缓冲区 |
| `BIND_SHADER_RESOURCE` | 0x02 | 着色器资源视图 |
| `BIND_RENDER_TARGET` | 0x04 | 渲染目标视图 |
| `BIND_DEPTH_STENCIL` | 0x08 | 深度模板视图 |
| `BIND_UNORDERED_ACCESS` | 0x10 | 无序访问视图 |

### 4.3 后端抽象层

#### 4.3.1 抽象策略

&emsp;&emsp;Diligent Engine 的后端抽象采用**接口-实现分离**的 COM 风格，核心策略为：

1. **统一接口层**：`GraphicsEngine/interface/` 定义所有公共接口（`IRenderDevice`、`IDeviceContext`、`IPipelineState` 等），使用 `DILIGENT_BEGIN_INTERFACE` 宏定义纯虚函数表
2. **后端实现层**：每个后端目录实现这些接口的具体类（如 `RenderDeviceD3D12Impl`、`DeviceContextVkImpl`）
3. **共享基础设施**：`GraphicsEngineD3DBase` 和 `GraphicsEngineNextGenBase` 提供 D3D11/D3D12 以及下一代 API 的共享代码

&emsp;&emsp;后端实现的类继承关系（以 D3D12 为例）：

```
IRenderDevice (接口)
  └── IDeviceObject (基础实现)
        └── RenderDeviceBase (通用实现)
              └── RenderDeviceNextGenBase (D3D12/Vulkan 共享)
                    └── RenderDeviceD3D12Impl (D3D12 特定实现)
```

#### 4.3.2 后端特性矩阵

| 后端 | 最低版本 | 特殊能力 | 平台 |
|------|---------|---------|------|
| D3D11 | 11.1 | NVAPI 扩展、原生多线程 | Windows |
| D3D12 | SDK 10.0.19041 | 光线追踪、Mesh Shader、Tile Shader | Windows/UWP |
| OpenGL | 4.1 | 传统兼容 | Windows/Linux/macOS/Android/Web |
| Vulkan | 1.0 | 完整显式控制、光线追踪、Mesh Shader | 全平台（iOS/macOS 通过 MoltenVK） |
| Metal | 1.0 | Apple 平台原生、Tile Shading | macOS/iOS/tvOS/visionOS（商业许可） |
| WebGPU | - | 浏览器原生 | Web/Chrome |

#### 4.3.3 低级 API 互操作

&emsp;&emsp;Diligent Engine 提供丰富的低级 API 互操作能力，允许应用程序在需要时直接访问原生 API 对象：

```cpp
// D3D12: 获取原生设备和命令队列
IDXGIAdapter* pDxgiAdapter = pDeviceD3D12->GetDxgiAdapter();
ID3D12Device* pd3d12Device = pDeviceD3D12->GetD3D12Device();

// Vulkan: 获取原生 VkDevice 和 VkInstance
VkDevice vkDevice = pDeviceVk->GetVkDevice();
VkInstance vkInstance = pDeviceVk->GetVkInstance();

// 附着到已有的原生上下文
EngineD3D12CreateInfo EngineCI;
EngineCI.pNativeDevice = existingD3D12Device;
EngineCI.pCommandQueue = existingCommandQueue;
pFactoryD3D12->CreateDeviceAndContextsD3D12(EngineCI, &pDevice, &pContext);
```

### 4.4 管线状态对象（PSO）

#### 4.4.1 单体 PSO 设计

&emsp;&emsp;Diligent Engine 采用 **D3D12/Vulkan 风格的单体 PSO** 设计，一个 PSO 包含完整的管线状态：所有着色器阶段、输入布局、混合状态、光栅化状态、深度模板状态等。

&emsp;&emsp;支持的管线类型：

| 管线类型 | 用途 | 创建方法 |
|---------|------|---------|
| `PIPELINE_TYPE_GRAPHICS` | 传统图形管线 | `CreateGraphicsPipelineState()` |
| `PIPELINE_TYPE_COMPUTE` | 计算管线 | `CreateComputePipelineState()` |
| `PIPELINE_TYPE_MESH` | Mesh Shader 管线 | 通过 `GraphicsPipelineStateCreateInfo`（设置 AS/MS 着色器） |
| `PIPELINE_TYPE_RAY_TRACING` | 光线追踪管线 | `CreateRayTracingPipelineState()` |
| `PIPELINE_TYPE_TILE` | Tile Shading 管线 | `CreateTilePipelineState()` |

&emsp;&emsp;图形 PSO 创建示例：

```cpp
GraphicsPipelineStateCreateInfo PSOCreateInfo;
PipelineStateDesc& PSODesc = PSOCreateInfo.PSODesc;

PSODesc.Name = "My pipeline state";
PSODesc.PipelineType = PIPELINE_TYPE_GRAPHICS;

// 渲染目标格式
PSOCreateInfo.GraphicsPipeline.NumRenderTargets = 1;
PSOCreateInfo.GraphicsPipeline.RTVFormats[0] = TEX_FORMAT_RGBA8_UNORM_SRGB;
PSOCreateInfo.GraphicsPipeline.DSVFormat = TEX_FORMAT_D32_FLOAT;

// 深度模板状态
PSOCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable = true;
PSOCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = true;

// 混合状态
auto& RT0 = PSOCreateInfo.GraphicsPipeline.BlendDesc.RenderTargets[0];
RT0.BlendEnable = True;
RT0.SrcBlend = BLEND_FACTOR_SRC_ALPHA;
RT0.DestBlend = BLEND_FACTOR_INV_SRC_ALPHA;
RT0.BlendOp = BLEND_OPERATION_ADD;

// 光栅化状态
PSOCreateInfo.GraphicsPipeline.RasterizerDesc.FillMode = FILL_MODE_SOLID;
PSOCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode = CULL_MODE_NONE;

// 输入布局
LayoutElement LayoutElems[] = {
    LayoutElement(0, 0, 3, VT_FLOAT32, False),
    LayoutElement(1, 0, 4, VT_UINT8,   True),
    LayoutElement(2, 0, 2, VT_FLOAT32, False),
};
PSOCreateInfo.GraphicsPipeline.InputLayout.LayoutElements = LayoutElems;
PSOCreateInfo.GraphicsPipeline.InputLayout.NumElements = _countof(LayoutElems);

// 着色器绑定
PSOCreateInfo.pVS = m_pVS;
PSOCreateInfo.pPS = m_pPS;

m_pDevice->CreateGraphicsPipelineState(PSOCreateInfo, &m_pPSO);
```

#### 4.4.2 管线资源布局

&emsp;&emsp;Diligent Engine 引入了**三级着色器变量分类**来优化资源绑定：

| 变量类型 | 说明 | 绑定方式 | 性能 |
|---------|------|---------|------|
| `SHADER_RESOURCE_VARIABLE_TYPE_STATIC` | 全局常量（相机、光照等），绑定后不改变 | 直接绑定到 PSO | 最高 |
| `SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE` | 按材质频率变化的资源（漫反射纹理、法线贴图等） | 通过 SRB，每个 SRB 实例绑定一次 | 高 |
| `SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC` | 频繁随机变化的资源 | 通过 SRB，可多次重新绑定 | 较低 |

```cpp
// 定义资源变量类型
ShaderResourceVariableDesc ShaderVars[] = {
    {SHADER_TYPE_PIXEL, "g_StaticTexture",  SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
    {SHADER_TYPE_PIXEL, "g_MutableTexture", SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {SHADER_TYPE_PIXEL, "g_DynamicTexture", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC}
};
PSODesc.ResourceLayout.Variables = ShaderVars;
PSODesc.ResourceLayout.NumVariables = _countof(ShaderVars);
PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

// 不可变采样器
ImmutableSamplerDesc ImtblSampler;
ImtblSampler.ShaderStages = SHADER_TYPE_PIXEL;
ImtblSampler.Desc.MinFilter = FILTER_TYPE_LINEAR;
ImtblSampler.TextureName = "g_MutableTexture";
PSODesc.ResourceLayout.NumImmutableSamplers = 1;
PSODesc.ResourceLayout.ImmutableSamplers = &ImtblSampler;
```

#### 4.4.3 Pipeline Resource Signature（PRS）

&emsp;&emsp;Diligent Engine 支持**显式管线资源签名**，允许将资源布局从 PSO 中独立出来，在多个 PSO 之间共享：

```cpp
// 创建显式资源签名
PipelineResourceSignatureDesc SignatureDesc;
SignatureDesc.Name = "MyPRS";
// ... 定义变量和采样器 ...

RefCntAutoPtr<IPipelineResourceSignature> pSignature;
m_pDevice->CreatePipelineResourceSignature(SignatureDesc, &pSignature);

// 在 PSO 创建时使用显式签名
PSOCreateInfo.ppResourceSignatures = &pSignature;
PSOCreateInfo.ResourceSignaturesCount = 1;
```

### 4.5 着色器资源绑定

#### 4.5.1 三级绑定模型

&emsp;&emsp;Diligent Engine 的资源绑定基于三级变量分类，核心流程为：

```
1. 静态资源 → 直接绑定到 PSO
   m_pPSO->GetStaticVariableByName(SHADER_TYPE_PIXEL, "g_tex2DShadowMap")->Set(pShadowMapSRV);

2. 创建 SRB → 绑定 Mutable/Dynamic 资源
   m_pPSO->CreateShaderResourceBinding(&m_pSRB, true);
   m_pSRB->GetVariable(SHADER_TYPE_PIXEL, "tex2DDiffuse")->Set(pDiffuseTexSRV);

3. 提交资源到上下文
   m_pContext->CommitShaderResources(m_pSRB, COMMIT_SHADER_RESOURCES_FLAG_TRANSITION_RESOURCES);
```

#### 4.5.2 Shader Resource Binding（SRB）

&emsp;&emsp;SRB 是 `IPipelineState` 或 `IPipelineResourceSignature` 创建的对象，封装了 mutable 和 dynamic 资源的绑定状态：

```cpp
class IShaderResourceBinding : public IObject {
    IShaderResourceVariable* GetVariable(SHADER_TYPE ShaderType, const Char* Name);
    IShaderResourceVariable* GetVariableByIndex(SHADER_TYPE ShaderType, Uint32 Index);
    Uint32 GetVariableCount(SHADER_TYPE ShaderType);

    void BindResources(SHADER_TYPE ShaderStages, IResourceMapping* pResourceMapping,
                       BIND_SHADER_RESOURCES_FLAGS Flags);
};
```

&emsp;&emsp;SRB 的分配粒度可通过 `PSODesc.SRBAllocationGranularity` 配置，控制内部资源的预分配策略。

#### 4.5.3 资源映射（Resource Mapping）

&emsp;&emsp;除了逐变量绑定，还支持批量资源映射：

```cpp
ResourceMappingEntry Entries[] = {
    {"g_Texture", pTexture->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE)}
};
ResourceMappingCreateInfo ResMappingCI;
ResMappingCI.pEntries = Entries;
ResMappingCI.NumEntries = _countof(Entries);

RefCntAutoPtr<IResourceMapping> pResMapping;
pRenderDevice->CreateResourceMapping(ResMappingCI, &pResMapping);

// 批量绑定静态资源
m_pPSO->BindStaticResources(SHADER_TYPE_VERTEX | SHADER_TYPE_PIXEL,
                             pResMapping, BIND_SHADER_RESOURCES_VERIFY_ALL_RESOLVED);

// 批量绑定 SRB 资源
m_pSRB->BindResources(SHADER_TYPE_VERTEX | SHADER_TYPE_PIXEL,
                       pResMapping, BIND_SHADER_RESOURCES_VERIFY_ALL_RESOLVED);
```

### 4.6 资源管理

#### 4.6.1 引用计数与自动释放

&emsp;&emsp;Diligent Engine 所有 GPU 对象使用**侵入式引用计数**，通过 `RefCntAutoPtr<T>` 智能指针自动管理生命周期：

```cpp
RefCntAutoPtr<IBuffer> pBuffer;
m_pDevice->CreateBuffer(BuffDesc, nullptr, &pBuffer);
// pBuffer 离开作用域时自动 Release()
```

&emsp;&emsp;引擎维护**延迟释放队列**，GPU 正在使用的资源不会被立即释放，而是在确认 GPU 不再使用后释放：

```cpp
// 手动触发延迟资源释放
m_pDevice->ReleaseStaleResources();

// 等待 GPU 完成所有操作
m_pDevice->IdleGPU();
```

#### 4.6.2 设备内存管理

&emsp;&emsp;Diligent Engine 提供底层设备内存管理接口，允许应用程序直接控制内存分配：

```cpp
class IDeviceMemory : public IDeviceObject {
    void* GetNativeHandle();
    Uint64 GetSize();
};
```

&emsp;&emsp;这在需要显式内存管理的高级场景（如稀疏资源、光线追踪加速结构）中特别有用。

#### 4.6.3 资源视图

&emsp;&emsp;所有资源都可以创建视图（View），视图定义了资源的特定使用方式：

```cpp
// 纹理默认视图
ITextureView* pRTV = pTexture->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
ITextureView* pSRV = pTexture->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
ITextureView* pUAV = pTexture->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS);

// 创建额外视图
TextureViewDesc ViewDesc;
ViewDesc.ViewType = TEXTURE_VIEW_SHADER_RESOURCE;
ViewDesc.Format = TEX_FORMAT_RGBA8_UNORM;
pTexture->CreateView(ViewDesc, &pAdditionalSRV);

// 缓冲区视图
BufferViewDesc BuffViewDesc;
BuffViewDesc.ViewType = BUFFER_VIEW_SHADER_RESOURCE;
BuffViewDesc.Format = TEX_FORMAT_R32_FLOAT;
BuffViewDesc.ByteOffset = 0;
BuffViewDesc.ByteWidth = sizeof(float) * 1024;
pBuffer->CreateView(BuffViewDesc, &pBuffSRV);
```

### 4.7 多线程模型

#### 4.7.1 延迟上下文与命令列表

&emsp;&emsp;Diligent Engine 支持**多线程命令录制**，通过延迟上下文（Deferred Context）和命令列表（Command List）实现：

```cpp
// 创建延迟上下文（创建引擎时指定数量，或运行时创建）
RefCntAutoPtr<IDeviceContext> pDeferredCtx;
m_pDevice->CreateDeferredContext(&pDeferredCtx);

// 开始录制命令列表，指定目标立即上下文 ID
IDeviceContext* pCmdList = pDeferredCtx->BeginCommandList(
    COMMAND_LIST_TYPE_COMPUTE,  // 命令类型
    m_ImmediateContextId        // 目标立即上下文的 ID
);

// 在延迟上下文中录制命令（与立即上下文 API 完全相同）
pDeferredCtx->SetRenderTargets(1, &pRTV, pDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
pDeferredCtx->DrawIndexed(DrawAttrs);

// 完成录制
pDeferredCtx->FinishCommandList();

// 在立即上下文中执行命令列表
m_pImmediateContext->ExecuteCommandList(pCmdList);
```

#### 4.7.2 上下文与队列

&emsp;&emsp;引擎支持**多命令队列**（Graphics、Compute、Transfer），每个立即上下文绑定到特定队列：

```cpp
DeviceContextDesc CtxDesc;
CtxDesc.Name = "Graphics Context";
CtxDesc.QueueType = COMMAND_QUEUE_TYPE_GRAPHICS;
CtxDesc.ContextId = 0;
CtxDesc.QueueId = 0;  // 对应 adapter 的队列索引

DeviceContextDesc ComputeCtxDesc;
ComputeCtxDesc.Name = "Compute Context";
ComputeCtxDesc.QueueType = COMMAND_QUEUE_TYPE_COMPUTE;
ComputeCtxDesc.ContextId = 1;
ComputeCtxDesc.QueueId = 1;
```

&emsp;&emsp;`ImmediateContextMask` 标志位控制 PSO 可以在哪些上下文中使用：

```cpp
PSODesc.ImmediateContextMask = (1ULL << 0) | (1ULL << 1);  // 可用于上下文 0 和 1
```

#### 4.7.3 同步原语

&emsp;&emsp;提供 Fence 用于 CPU-GPU 和 GPU-GPU 同步：

```cpp
// 创建 Fence
FenceDesc FenceCI;
FenceCI.Name = "FrameFence";
RefCntAutoPtr<IFence> pFence;
m_pDevice->CreateFence(FenceCI, &pFence);

// GPU 完成时发信号
m_pImmediateContext->EnqueueSignal(pFence, CurrentFenceValue);

// CPU 等待 GPU 完成
pFence->Wait(FenceValue);

// 命令队列等待
m_pImmediateContext->EnqueueWait(pFence, FenceValue);
```

### 4.8 着色器处理

#### 4.8.1 统一着色语言

&emsp;&emsp;Diligent Engine 使用 **HLSL 作为统一着色语言**，所有后端都支持 HLSL 输入：

| 源语言 | D3D11/D3D12 | OpenGL/GLES | Vulkan | Metal |
|--------|------------|-------------|--------|-------|
| HLSL | 原生支持 | 转换为 GLSL | 编译为 SPIR-V | 编译为 MSL |
| GLSL | 不支持 | 原生支持 | 原生支持 | 通过 SPIR-V |
| MSL | 不支持 | 不支持 | 不支持 | 原生支持 |

#### 4.8.2 着色器创建

```cpp
ShaderCreateInfo ShaderCI;
ShaderCI.Desc.Name = "My Pixel Shader";
ShaderCI.FilePath = "MyShader.fx";
ShaderCI.EntryPoint = "PSMain";
ShaderCI.Desc.ShaderType = SHADER_TYPE_PIXEL;
ShaderCI.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;

// 创建着色器源流工厂
RefCntAutoPtr<IShaderSourceInputStreamFactory> pShaderSourceFactory;
m_pEngineFactory->CreateDefaultShaderSourceStreamFactory("shaders", &pShaderSourceFactory);
ShaderCI.pShaderSourceStreamFactory = pShaderSourceFactory;

// 宏定义
ShaderMacroHelper Macros;
Macros.AddShaderMacro("USE_SHADOWS", 1);
Macros.AddShaderMacro("NUM_SHADOW_SAMPLES", 4);
Macros.Finalize();
ShaderCI.Macros = Macros;

RefCntAutoPtr<IShader> pShader;
m_pDevice->CreateShader(ShaderCI, &pShader);
```

#### 4.8.3 着色器反射

&emsp;&emsp;创建着色器时，引擎自动进行着色器反射，提取资源绑定信息。反射数据用于：
- 验证资源绑定的正确性
- 构建 SRB 内部结构
- 提供调试信息

```cpp
// 获取编译输出
RefCntAutoPtr<IDataBlob> pCompilerOutput;
m_pDevice->CreateShader(ShaderCI, &pShader, &pCompilerOutput);
if (pCompilerOutput) {
    const char* pMessages = reinterpret_cast<const char*>(pCompilerOutput->GetDataPtr());
    // pMessages 包含编译器输出和完整着色器源码
}
```

#### 4.8.4 异步着色器编译

&emsp;&emsp;支持异步管线状态创建，避免卡顿：

```cpp
PSOCreateInfo.Flags = PSO_CREATE_FLAG_ASYNCHRONOUS;
m_pDevice->CreateGraphicsPipelineState(PSOCreateInfo, &m_pPSO);

// 检查编译状态
if (m_pPSO->GetStatus() == PIPELINE_STATE_STATUS_READY) {
    // 管线可用
} else if (m_pPSO->GetStatus() == PIPELINE_STATE_STATUS_COMPILING) {
    // 仍在编译中
}
```

#### 4.8.5 着色器编译管线

&emsp;&emsp;Diligent Engine 内置着色器编译管线，支持运行时编译：

- **HLSL→SPIR-V**：集成 glslang（可通过 `DILIGENT_NO_GLSLANG` 禁用）
- **HLSL→GLSL**：内置 HLSL2GLSLConverterLib
- **HLSL→MSL**：通过 SPIRV-Cross 进行跨编译
- **离线编译**：提供 Render State Packager 工具，预编译着色器为目标平台格式

### 4.9 渲染状态描述语言

#### 4.9.1 JSON 渲染状态表示

&emsp;&emsp;Diligent Engine 支持基于 JSON 的渲染状态描述语言（Diligent Render State Notation），可以声明式地定义着色器、PSO、资源签名等：

```json
{
    "Shaders": [
        {
            "Desc": { "Name": "My Vertex Shader", "ShaderType": "VERTEX" },
            "SourceLanguage": "HLSL",
            "FilePath": "cube.vsh"
        },
        {
            "Desc": { "Name": "My Pixel Shader", "ShaderType": "PIXEL" },
            "SourceLanguage": "HLSL",
            "FilePath": "cube.psh"
        }
    ],
    "Pipelines": [
        {
            "PSODesc": { "Name": "My Pipeline", "PipelineType": "GRAPHICS" },
            "GraphicsPipeline": {
                "DepthStencilDesc": { "DepthEnable": true },
                "RTVFormats": { "0": "RGBA8_UNORM_SRGB" },
                "RasterizerDesc": { "CullMode": "FRONT" }
            },
            "pVS": "My Vertex Shader",
            "pPS": "My Pixel Shader"
        }
    ]
}
```

&emsp;&emsp;支持两种使用方式：
1. **运行时解析**：通过 DiligentTools 中的 RenderStateNotation 解析器动态加载
2. **离线打包**：通过 RenderStatePackager 工具预编译为优化的二进制格式

### 4.10 高级特性

| 特性 | 支持情况 |
|------|---------|
| 光线追踪 | ✅ BLAS/TLAS/SBT/TraceRays |
| Mesh Shader | ✅ Amplification + Mesh Shader |
| Tile Shading | ✅ Metal 后端原生支持 |
| 可变速率着色 (VRS) | ✅ Per-primitive / Texture-based |
| 稀疏资源 | ✅ Sparse Texture 支持 |
| Wave 操作 | ✅ Wave Intrinsics |
| 特化常量 | ✅ Vulkan/WebGPU 后端 |
| 异步计算 | ✅ 多命令队列 |
| PSO 缓存 | ✅ `IPipelineStateCache` |
| 设备内存管理 | ✅ `IDeviceMemory` |
| 超分辨率 | ✅ SuperResolution 模块 |

### 4.11 与其他 RHI 实现的对比

&emsp;&emsp;Diligent Engine 的独特之处在于：

| 对比维度 | Diligent Engine | bgfx | UE-RHI |
|---------|----------------|------|--------|
| **设计哲学** | 现代 API 风格 + 广泛后端覆盖 | 简单易用 + 极致抽象 | 大型引擎内部集成 |
| **着色语言** | 统一 HLSL | 自定义着色语言 | 平台原生着色器 |
| **资源绑定** | 三级变量分类 + SRB | Handle 绑定 + Draw Call 排序 | 着色器参数映射 |
| **PSO 设计** | 单体 PSO (D3D12/Vulkan 风格) | 无显式 PSO 概念 | 单体 PSO + 状态初始化器 |
| **内存管理** | 引用计数 + 延迟释放 | Handle + 内部池 | 引用计数 + RHI 堆 |
| **多线程** | 延迟上下文 + 命令列表 | 帧缓冲 + 后端线程 | 命令列表 + 密封体 |
| **API 风格** | COM 接口 | C API | C++ 虚函数 |
| **开源程度** | 完全开源 (Apache 2.0) | 完全开源 (BSD) | 引擎内开源 |
| **互操作性** | 丰富的原生 API 访问 | 有限 | UE 内部专用 |

&emsp;&emsp;总体而言，Diligent Engine 是目前最全面的开源图形 RHI 实现之一。它的独特优势在于：(1) 同时支持传统 API（D3D11、OpenGL）和现代 API（D3D12、Vulkan、Metal、WebGPU），(2) 使用 HLSL 作为统一着色语言并通过编译管线自动转换到各后端格式，(3) COM 风格的接口设计使其既可以 C++ 也可以 C# 使用，(4) 丰富的互操作能力允许与原生 API 无缝协作。对于需要跨平台图形抽象但又不想被特定引擎绑定的项目，Diligent Engine 是一个极佳的选择。

## 5 NVRHI
## 6 Filament
## 7 O3DE-RHI
