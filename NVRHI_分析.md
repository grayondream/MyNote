# NVRHI（NVIDIA Rendering Hardware Interface）深度分析

## 1. 整体架构

### 1.1 目录结构

```
NVRHI/
├── include/nvrhi/           # 公共头文件
│   ├── nvrhi.h              # 核心接口定义（~4000行）
│   ├── nvrhiHLSL.h          # HLSL相关类型定义
│   ├── vulkan.h              # Vulkan后端公共接口
│   ├── d3d11.h               # D3D11后端公共接口
│   ├── d3d12.h               # D3D12后端公共接口
│   ├── validation.h          # 验证层接口
│   ├── utils.h               # 工具函数
│   └── common/               # 公共基础类型
│       ├── resource.h         # IResource/RefCountPtr 基础设施
│       ├── containers.h       # static_vector 等容器
│       ├── misc.h             # 杂项工具
│       ├── aftermath.h        # NSight Aftermath 集成
│       └── resourcebindingmap.h # 资源绑定映射
├── src/
│   ├── common/               # 跨后端共享实现
│   │   ├── state-tracking.cpp/h  # 资源状态追踪
│   │   ├── format-info.cpp    # 格式信息查询
│   │   ├── dxgi-format.cpp/h  # DXGI格式转换
│   │   ├── misc.cpp           # 杂项实现
│   │   ├── utils.cpp          # 工具实现
│   │   ├── aftermath.cpp      # Aftermath集成
│   │   └── versioning.h       # 版本兼容检查
│   ├── vulkan/               # Vulkan 1.3 后端（18个文件）
│   ├── d3d12/                # Direct3D 12 后端（16个文件）
│   ├── d3d11/                # Direct3D 11 后端（11个文件）
│   └── validation/           # 验证层（3个文件）
├── doc/                      # 文档
├── tools/                    # 工具
└── cmake/                    # CMake配置
```

### 1.2 模块依赖关系

```
┌─────────────────────────────────────────────┐
│              用户应用程序                       │
├─────────────────────────────────────────────┤
│              nvrhi::validation               │
├──────────┬──────────┬──────────┬────────────┤
│ nvrhi_vk │nvrhi_d3d12│nvrhi_d3d11│  nvrhi    │
├──────────┴──────────┴──────────┴────────────┤
│              src/common                      │
│  (state-tracking, format-info, utils)        │
├─────────────────────────────────────────────┤
│           Graphics APIs                     │
│  (Vulkan 1.3 / D3D12 / D3D11)               │
└─────────────────────────────────────────────┘
```

NVRHI 支持两种构建模式：
- **静态库模式**：各后端编译为独立静态库（`nvrhi`, `nvrhi_d3d11`, `nvrhi_d3d12`, `nvrhi_vk`）
- **动态库模式**：通过 `NVRHI_BUILD_SHARED=ON` 构建为单一 DLL/SO

---

## 2. 公共API设计

### 2.1 核心接口层次

NVRHI 采用经典的 COM 风格接口设计，所有资源接口继承自 `IResource`：

```cpp
// include/nvrhi/common/resource.h
class IResource {
public:
    virtual unsigned long AddRef() = 0;
    virtual unsigned long Release() = 0;
    virtual unsigned long GetRefCount() = 0;
    virtual Object getNativeObject(ObjectType objectType) { return nullptr; }
};
```

**引用计数**通过 `RefCountPtr<T>` 智能指针管理（类似 `Microsoft::WRL::ComPtr`），并使用 `RefCounter<T>` 模板作为实现基类：

```cpp
template<class T>
class RefCounter : public T {
    std::atomic<unsigned long> m_refCount = 1;
public:
    virtual unsigned long AddRef() override { return ++m_refCount; }
    virtual unsigned long Release() override {
        unsigned long result = --m_refCount;
        if (result == 0) delete this;
        return result;
    }
};
```

**核心接口列表**：

| 接口 | 说明 | Handle类型 |
|------|------|-----------|
| `IDevice` | 设备抽象，资源创建入口 | `DeviceHandle` |
| `ICommandList` | GPU命令录制 | `CommandListHandle` |
| `ITexture` | 纹理资源 | `TextureHandle` |
| `IBuffer` | 缓冲区资源 | `BufferHandle` |
| `IShader` | 着色器 | `ShaderHandle` |
| `IShaderLibrary` | 着色器库（RT） | `ShaderLibraryHandle` |
| `ISampler` | 采样器 | `SamplerHandle` |
| `IGraphicsPipeline` | 图形管线 | `GraphicsPipelineHandle` |
| `IComputePipeline` | 计算管线 | `ComputePipelineHandle` |
| `IMeshletPipeline` | Meshlet管线 | `MeshletPipelineHandle` |
| `rt::IPipeline` | 光线追踪管线 | `rt::PipelineHandle` |
| `IBindingLayout` | 绑定布局 | `BindingLayoutHandle` |
| `IBindingSet` | 绑定集合 | `BindingSetHandle` |
| `IFramebuffer` | 帧缓冲 | `FramebufferHandle` |
| `IHeap` | 显存堆 | `HeapHandle` |
| `IStagingTexture` | 暂存纹理 | `StagingTextureHandle` |

### 2.2 Builder模式

NVRHI 大量使用 Builder 模式（链式调用）进行描述符初始化：

```cpp
auto texDesc = TextureDesc()
    .setWidth(1920).setHeight(1080)
    .setFormat(Format::RGBA8_UNORM)
    .setIsRenderTarget(true)
    .setDebugName("GBuffer_Albedo")
    .enableAutomaticStateTracking(ResourceStates::Common);
```

所有 `set*` 方法返回 `constexpr` 引用，支持编译期求值。

### 2.3 命令提交模型

NVRHI 采用**命令列表（Command List）模型**：

```cpp
// 典型使用流程
auto cmdList = device->createCommandList(CommandListParameters().setQueueType(CommandQueue::Graphics));
cmdList->open();
cmdList->setGraphicsState(state);
cmdList->draw(DrawArguments().setVertexCount(3).setInstanceCount(1));
cmdList->close();
device->executeCommandList(cmdList, CommandQueue::Graphics);
```

**命令队列类型**：
```cpp
enum class CommandQueue : uint8_t {
    Graphics = 0,  // 图形队列
    Compute,       // 计算队列
    Copy,          // 拷贝队列
    Count
};
```

DX11 后端仅支持单一即时上下文（`enableImmediateExecution = true`），而 DX12/Vulkan 支持真正的多命令列表并行录制。

---

## 3. 后端抽象层

### 3.1 后端实现策略

NVRHI 采用**接口继承 + 后端特化**的方式实现多API支持：

```
nvrhi::IDevice (接口)
├── nvrhi::d3d11::Device (继承并实现)
├── nvrhi::d3d12::Device (继承并实现)
└── nvrhi::vulkan::Device (继承并实现，额外扩展 vulkan::IDevice)
```

每个后端有独立的 `*-backend.h` 定义内部实现类，对外通过工厂函数创建：

```cpp
// Vulkan
namespace nvrhi::vulkan {
    NVRHI_API DeviceHandle createDevice(const DeviceDesc& desc);
}

// D3D11
namespace nvrhi::d3d11 {
    NVRHI_API DeviceHandle createDevice(const DeviceDesc& desc);
}

// D3D12
namespace nvrhi::d3d12 {
    NVRHI_API DeviceHandle createDevice(const DeviceDesc& desc);
}
```

### 3.2 后端文件组织对照

| 功能模块 | Vulkan | D3D12 | D3D11 |
|----------|--------|-------|-------|
| 设备初始化 | vulkan-device.cpp | d3d12-device.cpp | d3d11-device.cpp |
| 命令列表 | vulkan-commandlist.cpp | d3d12-commandlist.cpp | d3d11-commandlist.cpp |
| 纹理 | vulkan-texture.cpp | d3d12-texture.cpp | d3d11-texture.cpp |
| 缓冲区 | vulkan-buffer.cpp | d3d12-buffer.cpp | d3d11-buffer.cpp |
| 着色器 | vulkan-shader.cpp | d3d12-shader.cpp | d3d11-shader.cpp |
| 图形管线 | vulkan-graphics.cpp | d3d12-graphics.cpp | d3d11-graphics.cpp |
| 计算管线 | vulkan-compute.cpp | d3d12-compute.cpp | d3d11-compute.cpp |
| 资源绑定 | vulkan-resource-bindings.cpp | d3d12-resource-bindings.cpp | d3d11-resource-bindings.cpp |
| 状态追踪 | vulkan-state-tracking.cpp | d3d12-state-tracking.cpp | - |
| 光线追踪 | vulkan-raytracing.cpp | d3d12-raytracing.cpp | - |
| Meshlet | vulkan-meshlets.cpp | d3d12-meshlets.cpp | - |
| 上传管理 | vulkan-upload.cpp | d3d12-upload.cpp | - |
| 查询 | vulkan-queries.cpp | d3d12-queries.cpp | d3d11-queries.cpp |

### 3.3 验证层

验证层通过**装饰器模式**实现，包装真实后端设备：

```cpp
namespace nvrhi::validation {
    NVRHI_API DeviceHandle createValidationLayer(IDevice* underlyingDevice);
}
```

验证层检查：
- 资源状态合法性
- 绑定集与布局一致性
- 帧缓冲与PSO兼容性
- 资源生命周期
- 纹理/缓冲区状态转换正确性

---

## 4. 资源管理

### 4.1 资源创建方式

**方式一：直接创建（自动分配内存）**
```cpp
auto texture = device->createTexture(TextureDesc()
    .setWidth(1024).setHeight(1024)
    .setFormat(Format::RGBA16_FLOAT)
    .setIsUAV(true));
```

**方式二：虚拟资源 + 手动绑定（适用于DX12稀疏资源）**
```cpp
auto texture = device->createTexture(TextureDesc()
    .setIsVirtual(true).setIsTiled(true)...);
auto heap = device->createHeap(HeapDesc().setCapacity(size).setType(HeapType::DeviceLocal));
device->bindTextureMemory(texture, heap, 0);
```

### 4.2 堆管理

```cpp
enum class HeapType : uint8_t {
    DeviceLocal,  // GPU专用内存
    Upload,       // CPU→GPU上传堆
    Readback      // GPU→CPU回读堆
};
```

Vulkan 后端使用 `VulkanAllocator` 封装 `vkAllocateMemory`/`vkFreeMemory`，支持：
- Buffer device address（`VK_KHR_buffer_device_address`）
- 导出内存（跨API/跨设备共享）
- 专用分配（dedicated allocation）

### 4.3 内存子分配

**UploadManager** 实现命令列表级别的上传缓冲区子分配：

```cpp
class UploadManager {
    std::list<std::shared_ptr<BufferChunk>> m_ChunkPool;
    std::shared_ptr<BufferChunk> m_CurrentChunk;
    
    bool suballocateBuffer(uint64_t size, Buffer** pBuffer, uint64_t* pOffset,
                           void** pCpuVA, uint64_t currentVersion, uint32_t alignment = 256);
    void submitChunks(uint64_t currentVersion, uint64_t submittedVersion);
};
```

每个 `BufferChunk` 最小 4096 字节（GPU页大小），按需增长。

### 4.4 资源生命周期追踪

**自动生命周期追踪**（`BindingSetDesc::trackLiveness`）：
- 命令列表记录对 `IBindingSet` 的引用
- 命令列表执行期间保持 `BindingSet` 存活
- GPU 完成执行后通过 `ICommandListLifetimeTracker::runGarbageCollection()` 释放

**Volatile Buffer 版本控制**（Vulkan 特有）：
- 每个 volatile buffer 有 `maxVersions` 个版本
- 使用原子操作的 `BufferVersionItem` 追踪每个版本的状态
- 支持多命令列表并发写入不同版本

```cpp
// VulkanBackend.h 中的版本追踪设计
struct VolatileBufferState {
    int latestVersion = 0;
    int minVersion = 0;
    int maxVersion = 0;
    bool initialized = false;
};
```

### 4.5 RTXMU 集成

可选的 [RTXMU](https://github.com/NVIDIA-RTX/RTXMU) 集成用于管理 BLAS 内存：
- 自动 BLAS 压缩（`compactBottomLevelAccelStructs()`）
- 跨帧内存复用
- 通过 `NVRHI_WITH_RTXMU` CMake 变量启用

---

## 5. 多线程模型

### 5.1 命令列表并行录制

**DX12/Vulkan**：支持真正的并行命令列表录制

```cpp
// 每个线程创建独立的命令列表
auto cmdList1 = device->createCommandList(params);
auto cmdList2 = device->createCommandList(params);

// 线程1
cmdList1->open();
// ... 录制命令 ...
cmdList1->close();

// 线程2（并行）
cmdList2->open();
// ... 录制命令 ...
cmdList2->close();

// 主线程提交
device->executeCommandLists({cmdList1, cmdList2});
```

**DX11**：不支持多线程命令录制，所有命令列表映射到单一即时上下文。

### 5.2 线程安全机制

| 机制 | 用途 |
|------|------|
| `std::mutex m_Mutex` | Device 级别，保护资源创建 |
| `std::recursive_mutex m_Mutex` | Queue 级别，保护提交操作 |
| `std::atomic<uint64_t>` | Volatile buffer 版本追踪 |
| `std::atomic<unsigned long>` | 引用计数 |
| `std::mutex m_Mutex` | Texture 子资源视图创建 |

### 5.3 ICommandListLifetimeTracker

每个提交线程应拥有独立的生命周期追踪器，避免竞争：

```cpp
class ICommandListLifetimeTracker : public IResource {
public:
    virtual void runGarbageCollection() = 0; // 轮询GPU完成状态，释放资源
};
```

Vulkan 实现中：
```cpp
class CommandListLifetimeTracker final : public RefCounter<ICommandListLifetimeTracker> {
    std::list<TrackedCommandBufferPtr> m_CommandBuffersInFlight;
    Queue* m_Queue;
};
```

### 5.4 队列同步

Vulkan 后端使用 Timeline Semaphore 进行队列间同步：

```cpp
class Queue {
    vk::Semaphore trackingSemaphore;  // Timeline semaphore
    uint64_t m_LastSubmittedID = 0;
    uint64_t m_LastFinishedID = 0;
    
    uint64_t submit(ICommandList* const* ppCmd, size_t numCmd);
    uint64_t updateLastFinishedID();
};
```

---

## 6. 着色器处理

### 6.1 着色器类型

```cpp
enum class ShaderType : uint16_t {
    None            = 0x0000,
    Compute         = 0x0020,
    Vertex          = 0x0001,
    Hull            = 0x0002,
    Domain          = 0x0004,
    Geometry        = 0x0008,
    Pixel           = 0x0010,
    Amplification   = 0x0040,  // Task Shader
    Mesh            = 0x0080,
    RayGeneration   = 0x0100,
    AnyHit          = 0x0200,
    ClosestHit      = 0x0400,
    Miss            = 0x0800,
    Intersection    = 0x1000,
    Callable        = 0x2000,
};
```

### 6.2 着色器创建

NVRHI 接受预编译的字节码：
- **DX11/12**：DXBC 或 DXIL（HLSL 编译产物）
- **Vulkan**：SPIR-V（HLSL 通过 DXC 编译，或 GLSL 通过 glslangValidator）

```cpp
// 创建着色器
auto shader = device->createShader(
    ShaderDesc().setShaderType(ShaderType::Vertex).setEntryName("vs_main"),
    spirvBinary.data(), spirvBinary.size());
```

### 6.3 着色器特化（Specialization Constants）

Vulkan 后端支持着色器特化常量：

```cpp
ShaderSpecialization specs[] = {
    ShaderSpecialization::UInt32(0, 1024),
    ShaderSpecialization::Float(1, 3.14f)
};
auto specializedShader = device->createShaderSpecialization(baseShader, specs, 2);
```

实现上复用同一个 `VkShaderModule`，仅附加不同的 specialization info。

### 6.4 着色器库（Shader Library）

用于光线追踪管线，单个 SPIR-V/DXIL 可包含多个入口点：

```cpp
auto shaderLib = device->createShaderLibrary(spirvData, spirvSize);
auto rayGenShader = shaderLib->getShader("raygen", ShaderType::RayGeneration);
auto missShader = shaderLib->getShader("miss", ShaderType::Miss);
```

### 6.5 绑定模型

NVRHI 使用**三级绑定模型**：

**1. BindingLayout** - 描述着色器期望的资源布局
```cpp
auto layout = device->createBindingLayout(BindingLayoutDesc()
    .setVisibility(ShaderType::Pixel)
    .addItem(BindingLayoutItem::Texture_SRV(0))
    .addItem(BindingLayoutItem::Sampler(0))
    .addItem(BindingLayoutItem::ConstantBuffer(0)));
```

**2. BindingSet** - 填充具体的资源绑定
```cpp
auto bindingSet = device->createBindingSet(BindingSetDesc()
    .addItem(BindingSetItem::Texture_SRV(0, albedoTexture))
    .addItem(BindingSetItem::Sampler(0, linearSampler))
    .addItem(BindingSetItem::ConstantBuffer(0, cb)), layout);
```

**3. Bindless Layout** - 无限制描述符数组
```cpp
auto bindlessLayout = device->createBindlessLayout(BindlessLayoutDesc()
    .setVisibility(ShaderType::All)
    .setFirstSlot(0).setMaxCapacity(1024)
    .addRegisterSpace(BindingLayoutItem::Texture_SRV(0)));
```

**Vulkan 描述符集映射**：
- `registerSpace` 映射到 Vulkan descriptor set index
- `registerSpaceIsDescriptorSet = true` 启用此行为
- `VulkanBindingOffsets` 控制 HLSL→SPIR-V 寄存器偏移

---

## 7. 状态追踪与屏障

### 7.1 资源状态模型

```cpp
enum class ResourceStates : uint32_t {
    Common                  = 0x00000001,
    ConstantBuffer          = 0x00000002,
    VertexBuffer            = 0x00000004,
    IndexBuffer             = 0x00000008,
    IndirectArgument        = 0x00000010,
    PixelShaderResource     = 0x00000020,
    NonPixelShaderResource  = 0x00000040,
    UnorderedAccess         = 0x00000080,
    RenderTarget            = 0x00000100,
    DepthWrite              = 0x00000200,
    CopyDest                = 0x00001000,
    CopySource              = 0x00002000,
    Present                 = 0x00010000,
    AccelStructRead         = 0x00020000,
    AccelStructWrite        = 0x00040000,
    // ... 更多状态
};
```

### 7.2 自动屏障放置

NVRHI 的核心特性之一是**自动资源状态追踪与屏障插入**：

```cpp
// CommandListResourceStateTracker（src/common/state-tracking.h）
class CommandListResourceStateTracker {
    std::unordered_map<TextureStateExtension*, std::unique_ptr<TextureState>> m_TextureStates;
    std::unordered_map<BufferStateExtension*, std::unique_ptr<BufferState>> m_BufferStates;
    std::vector<TextureBarrier> m_TextureBarriers;
    std::vector<BufferBarrier> m_BufferBarriers;

    void requireTextureState(TextureStateExtension* texture, TextureSubresourceSet subresources, ResourceStates state);
    void requireBufferState(BufferStateExtension* buffer, ResourceStates state);
};
```

**工作流程**：
1. `setGraphicsState()` / `draw()` 等调用时，自动调用 `requireTextureState()`
2. 比较当前追踪状态与目标状态，生成 barrier
3. Barrier 放入 pending 列表
4. `commitBarriers()` 或下一次 `setGraphicsState()` 时提交

**永久状态优化**：
```cpp
// 标记资源为永久状态，后续命令列表无需追踪
cmdList->setPermanentTextureState(texture, ResourceStates::ShaderResource);
```

### 7.3 UAV 屏障控制

```cpp
cmdList->setEnableUavBarriersForTexture(uavTexture, false); // 禁用连续UAV间的屏障
```

---

## 8. 光线追踪支持

### 8.1 加速结构

```cpp
// BLAS
auto blas = device->createAccelStruct(AccelStructDesc()
    .addBottomLevelGeometry(GeometryDesc()
        .setTriangles(GeometryTriangles()
            .setVertexBuffer(vb).setIndexBuffer(ib)
            .setVertexCount(3).setIndexCount(3)))
    .setBuildFlags(AccelStructBuildFlags::PreferFastTrace));

// TLAS
auto tlas = device->createAccelStruct(AccelStructDesc()
    .setTopLevelMaxInstances(1));
```

### 8.2 光线追踪管线

```cpp
auto rtPipeline = device->createRayTracingPipeline(rt::PipelineDesc()
    .addShader(rt::PipelineShaderDesc()
        .setExportName("raygen").setShader(rayGenShader))
    .addHitGroup(rt::PipelineHitGroupDesc()
        .setExportName("hitgroup").setClosestHitShader(chsShader))
    .addShader(rt::PipelineShaderDesc()
        .setExportName("miss").setShader(missShader))
    .addBindingLayout(globalLayout)
    .setMaxPayloadSize(sizeof(HitPayload))
    .setMaxRecursionDepth(2));
```

### 8.3 Shader Table

```cpp
auto shaderTable = rtPipeline->createShaderTable();
shaderTable->setRayGenerationShader("raygen", rayGenBindings);
shaderTable->addMissShader("miss", missBindings);
shaderTable->addHitGroup("hitgroup", hitGroupBindings);
```

### 8.4 高级特性

- **Opacity Micromap (OMM)**：半透明微图加速
- **Cluster Acceleration Structure (CLAS)**：集群级加速结构
- **Linear Swept Spheres (LSS)**：线性扫掠球体几何
- **Shader Execution Reordering (SER)**：着色器执行重排序

---

## 9. 设计特点与对比

### 9.1 独特设计决策

| 特性 | NVRHI | bgfx | Diligent | UE-RHI |
|------|-------|------|----------|--------|
| 接口风格 | COM-style接口 | C API封装 | COM-style | 虚函数抽象 |
| 资源绑定 | Layout+Set | 预定义uniform | SRB+PSO | ShaderParameter |
| 状态管理 | 自动追踪+可选手动 | 手动barrier | 手动barrier | 手动barrier |
| 着色器语言 | HLSL→SPIRIV/DXBC | 自有+跨编译 | HLSL统一 | HLSL |
| 光线追踪 | 完整支持(OMM/CLAS/LSS) | 基础支持 | 完整支持 | 引擎集成 |
| Meshlet | 完整支持 | 无 | 无 | UE5集成 |
| NVAPI扩展 | 深度集成 | 无 | 无 | 部分集成 |
| Vulkan版本 | 1.3 | 1.0+ | 1.0+ | 1.1+ |
| CoopVec | 支持 | 无 | 无 | 无 |

### 9.2 核心设计优势

1. **自动状态追踪**：最显著的差异化特性，减少用户错误
2. **Vulkan volatile buffer 版本管理**：精巧的原子操作设计，支持多队列并发
3. **RTXMU 集成**：自动 BLAS 压缩管理
4. **NVIDIA 硬件特性支持**：NVAPI 扩展、Aftermath 崩溃诊断
5. **轻量级验证层**：可选的调试支持
6. **C++17 constexpr Builder**：零开销的描述符初始化
7. **模块化后端**：可按需选择启用的图形API

### 9.3 适用场景

NVRHI 主要面向 **NVIDIA SDK 和示例项目**，而非通用引擎：
- RTX 系列 SDK（RTXGI、RTXDI、RTXPT 等）
- Donut 渲染框架
- 神经网络着色（RTXNS）
- 纹理压缩（RTXNTC）

其设计理念是**最小化抽象开销**，提供接近原生API的性能，同时隐藏跨API的繁琐差异。

---

## 10. 关键数据结构总结

| 结构 | 大小 | 用途 |
|------|------|------|
| `BindingLayoutItem` | 8 bytes | 绑定布局项 |
| `BindingSetItem` | 40 bytes | 绑定项（含资源引用+子资源范围） |
| `InstanceDesc` | 64 bytes | RT 实例描述（与硬件对齐） |
| `TextureSubresourceSet` | 16 bytes | 纹理子资源范围 |
| `BufferRange` | 16 bytes | 缓冲区范围 |

---

*分析基于 NVRHI 源码，commit 对应 2024-2025 年版本。*
