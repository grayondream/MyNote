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
## 3 UE-RHI
## 4 Diligent Engine
## 5 NVRHI
## 6 Filament
## 7 O3DE-RHI
