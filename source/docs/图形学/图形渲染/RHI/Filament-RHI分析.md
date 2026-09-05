# Filament RHI 实现深度分析

> 分析日期：2026-09-05
> 源码路径：`/home/ares/workspace/sources/rhi/filament`
> 版本：基于最新main分支

---

## 1. 整体架构

### 1.1 目录结构

```
filament/
├── filament/                    # 核心引擎
│   ├── backend/                 # RHI 抽象层（核心分析对象）
│   │   ├── include/backend/     # 公共 API 头文件
│   │   ├── include/private/     # 私有头文件（Driver, HandleAllocator, CommandStream）
│   │   └── src/                 # 后端实现
│   │       ├── vulkan/          # Vulkan 后端 (58 files)
│   │       ├── metal/           # Metal 后端 (30 files)
│   │       ├── opengl/          # OpenGL/ES 后端 (31 files)
│   │       ├── webgpu/          # WebGPU 后端 (53 files)
│   │       ├── noop/            # 空实现后端 (4 files)
│   │       └── *.cpp            # 公共后端代码
│   ├── include/filament/        # 公共引擎 API
│   └── src/                     # 引擎实现
├── libs/
│   ├── filamat/                 # 着色器材质编译器
│   ├── filabridge/              # 材质系统桥接
│   ├── filaflat/                # 材质文件格式解析
│   ├── bluevk/                  # Vulkan 函数加载器
│   ├── bluegl/                  # OpenGL 函数加载器
│   ├── math/                    # 数学库
│   └── utils/                   # 工具库
├── shaders/                     # 着色器源码
└── tools/                       # 开发工具
```

### 1.2 架构分层

Filament 采用**三层架构**：

| 层级 | 职责 | 关键类 |
|------|------|--------|
| **公共 API 层** | 用户接口，资源创建/销毁，渲染状态设置 | `Engine`, `Renderer`, `View`, `Scene`, `Material` |
| **Backend 抽象层** | 命令录制/分发，资源管理，跨后端统一接口 | `Driver`, `CommandStream`, `HandleAllocator`, `DriverBase` |
| **图形 API 实现层** | 具体图形 API 调用 | `VulkanDriver`, `MetalDriver`, `OpenGLDriver`, `WebGPUDriver` |

### 1.3 后端支持矩阵

| 后端 | 平台 | 默认平台 | Shader 语言 |
|------|------|----------|-------------|
| OpenGL/ES | Android, iOS, macOS, Linux, Windows, WASM | Android, WASM | ESSL 1.0 / ESSL 3.0 |
| Vulkan | Android, Linux, Windows, macOS | Linux, Windows | SPIR-V |
| Metal | iOS, macOS | iOS, macOS | MSL / Metal Library |
| WebGPU | 所有平台（通过浏览器） | - | WGSL |
| Noop | 测试用 | - | - |

---

## 2. 公共 API 设计

### 2.1 Engine 初始化与后端选择

```cpp
// PlatformFactory::create() 中的后端选择逻辑
if (*backend == Backend::DEFAULT) {
    // Android → OpenGL, iOS/macOS → Metal, Linux/Windows → Vulkan, WASM → OpenGL
}
```

Engine 创建流程：
1. `Engine::create()` → `PlatformFactory::create()` 创建平台特定 `Platform`
2. `Platform::createDriver()` 创建具体 `Driver` 实现
3. `Driver` 通过 `CommandStream` 与引擎线程通信

### 2.2 资源管理模型

Filament 使用**类型化句柄（Typed Handle）**系统：

```cpp
template<typename T> class Handle : public HandleBase {
    HandleId mId;  // 27-bit index + 4-bit age + 1-bit heap flag
};
```

**句柄分配器分层设计**（`HandleAllocator`）：

| 池化类型 | GL 大小 | VK 大小 | MTL 大小 | 用途 |
|----------|---------|---------|----------|------|
| Pool P0 | 32B | 64B | 32B | 小型资源（HwFence, HwSync 等） |
| Pool P1 | 96B | 160B | 64B | 中型资源（HwBufferObject, HwIndexBuffer 等） |
| Pool P2 | 184B | 312B | 552B | 大型资源（HwTexture, HwProgram 等） |
| Heap | malloc | malloc | malloc | 溢出资源 |

**Use-After-Free 检测**：每个池化句柄带 4-bit age 标签，释放后 age 递增，使用时校验。

### 2.3 硬件资源抽象

`DriverBase.h` 定义了所有硬件资源的基础结构：

```cpp
struct HwTexture : public HwBase {
    uint32_t width, height, depth;
    SamplerType target;
    uint8_t levels : 4;
    uint8_t samples : 4;
    TextureFormat format;
    TextureUsage usage;
    HwStream* hwStream;
};

struct HwProgram : public HwBase {
    utils::CString name;
};

struct HwRenderTarget : public HwBase {
    uint32_t width, height;
};
```

### 2.4 命令提交模型

Filament 采用**单线程命令流（Command Stream）**模型：

```
用户线程 ─── CommandStream::method() ───→ CircularBuffer ───→ Driver 线程执行
                    │
                    └─ 同步命令直接调用 Driver virtual 方法
```

**命令分类**（通过 `DriverAPI.inc` 宏定义）：

| 类型 | 宏 | 行为 |
|------|-----|------|
| 异步命令 | `DECL_DRIVER_API` | 序列化到命令缓冲区，延迟执行 |
| 同步命令 | `DECL_DRIVER_API_SYNCHRONOUS` | 直接通过虚函数调用 |
| 返回值命令 | `DECL_DRIVER_API_RETURN` | 异步执行但返回句柄（句柄立即可用） |

**CommandStreamDispatcher** 使用模板生成命令分发表，避免虚函数开销：

```cpp
template<typename ConcreteDriver>
class ConcreteDispatcher {
    static void methodName(Driver& driver, CommandBase* base, intptr_t* next) {
        auto& cmd = *static_cast<COMMAND_TYPE(methodName)*>(base);
        // 直接调用具体 Driver 方法，无虚函数
        static_cast<ConcreteDriver&>(driver).methodName(std::get<I>(cmd.mArgs)...);
    }
};
```

---

## 3. 后端抽象层

### 3.1 Driver 接口层次

```
Driver (纯虚接口)
  └── DriverBase (公共实现)
        ├── OpenGLDriverBase → OpenGLDriver
        ├── VulkanDriver
        ├── MetalDriver
        ├── WebGPUDriver
        └── NoopDriver
```

### 3.2 Platform 抽象

`Platform` 是后端工厂接口，负责：
1. 创建图形 API 上下文（Vulkan Instance/Device, GL Context）
2. 管理原生窗口（SwapChain）
3. 提供 Blob 缓存（着色器缓存）
4. 处理帧时间戳查询

```cpp
class Platform {
    virtual Driver* createDriver(void* sharedContext, const DriverConfig& driverConfig) = 0;
    virtual int getOSVersion() const = 0;
    virtual utils::CString getDeviceInfo(DeviceInfoType infoType, Driver* driver) const = 0;
    // ...
};
```

**平台实现矩阵**：

| 后端 | Android | iOS | macOS | Linux | Windows | WASM |
|------|---------|-----|-------|-------|---------|------|
| OpenGL | PlatformEGLAndroid | PlatformCocoaTouchGL | PlatformCocoaGL/OSMesa | PlatformGLX/EGLHeadless | PlatformWGL | PlatformWebGL |
| Vulkan | VulkanPlatformAndroid | VulkanPlatformApple | VulkanPlatformApple | VulkanPlatformLinux | VulkanPlatformWindows | - |
| Metal | - | PlatformMetal | PlatformMetal | - | - | - |
| WebGPU | WebGPUPlatformAndroid | WebGPUPlatformApple | WebGPUPlatformApple | WebGPUPlatformLinux | WebGPUPlatformWindows | WebGPUPlatformWasm |

### 3.3 Vulkan 后端架构

```
VulkanDriver
├── VulkanContext          # Vulkan 实例/设备/物理设备信息
├── VulkanCommands         # 命令缓冲区管理（CommandBufferPool）
├── VulkanPipelineCache    # 图形管线缓存
├── VulkanPipelineLayoutCache  # 管线布局缓存
├── VulkanDescriptorSetLayoutCache  # 描述符集布局缓存
├── VulkanDescriptorSetCache    # 描述符集缓存与绑定
├── VulkanStagePool        # CPU→GPU 暂存缓冲区池
├── VulkanBufferCache      # GPU 缓冲区池
├── VulkanSamplerCache     # 采样器缓存
├── VulkanFboCache         # 帧缓冲对象缓存
├── VulkanSemaphoreManager # 信号量管理
├── VulkanFencePool        # 栅栏池
├── VulkanBlitter          # Blit 操作
├── VulkanReadPixels       # 像素回读
├── VulkanQueryManager     # GPU 查询
├── VulkanYcbcrConversionCache  # YCbCr 转换缓存
├── VulkanExternalImageManager  # 外部图像管理
├── VulkanStreamedImageManager  # 流式图像管理
└── ResourceManager        # Vulkan 资源引用计数管理
```

**关键设计特点**：

1. **VMA（Vulkan Memory Allocator）集成**：
   ```cpp
   // 创建 VMA 分配器时禁用内部同步（后端保证单线程访问）
   VmaAllocatorCreateInfo allocatorInfo {
       .flags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT,
       // ...
   };
   ```

2. **命令缓冲区依赖链**：通过 `VkSemaphore` 链式连接，保证顺序执行。

3. **资源引用计数**：`resource_ptr<T>` 类似 `shared_ptr`，支持线程安全和非线程安全两种模式。

---

## 4. 资源管理

### 4.1 内存分配策略

**Vulkan 后端使用三级内存管理**：

| 级别 | 机制 | 用途 |
|------|------|------|
| **VMA** | `VmaAllocator` | GPU 内存分配（设备本地/主机可见） |
| **Stage Pool** | `VulkanStagePool` | CPU→GPU 暂存缓冲区，分段复用 |
| **Buffer Cache** | `VulkanBufferCache` | GPU 缓冲区池化复用 |

**VulkanStagePool 设计**：

```cpp
class VulkanStage {
    VmaAllocation mMemory;
    VkBuffer mBuffer;
    uint32_t mCapacity;
    void* mMapping;  // 持久映射

    class Segment : public fvkmemory::Resource {
        // 分段使用，引用计数管理
        // 释放时回调 offset 归还给父 Stage
    };
};
```

**Buffer Cache 延迟回收**：

```cpp
class VulkanBufferCache {
    using BufferPool = std::multimap<uint32_t, UnusedGpuBuffer>;
    // 按大小排序的多集合，记录最后访问帧号
    // gc() 时回收超过一定帧数未使用的缓冲区
};
```

### 4.2 Vulkan 资源引用计数系统

Filament 为 Vulkan 后端实现了独立的资源管理系统（`fvkmemory` 命名空间）：

```cpp
// 两种资源基类
struct Resource {           // 非线程安全资源
    uint32_t mCount;        // 引用计数
    ResourceType restype;   // 资源类型枚举
    HandleId id;            // 句柄 ID
    bool mHandleConsideredDestroyed;
};

struct ThreadSafeResource { // 线程安全资源（原子引用计数）
    std::atomic<uint32_t> mCount;
    // ...
};
```

**线程安全资源类型**（可从任意线程访问）：
- `PROGRAM` - 着色器程序
- `FENCE` - 栅栏
- `TIMER_QUERY` - 定时器查询
- `SYNC` - 同步对象

**resource_ptr 智能指针**：

```cpp
template<typename D>
struct resource_ptr {
    D* mRef;
    
    // 创建并关联到 ResourceManager
    static resource_ptr make(ResourceManager* rm, Handle<B> const& handle, ARGS&&... args);
    
    // 引用计数归零时自动析构
    ~resource_ptr() { if (mRef) mRef->dec(); }
};
```

### 4.3 资源类型系统

```cpp
enum class ResourceType : uint8_t {
    BUFFER_OBJECT = 0, INDEX_BUFFER = 1, PROGRAM = 2,
    RENDER_TARGET = 3, SWAP_CHAIN = 4, RENDER_PRIMITIVE = 5,
    TEXTURE = 6, TEXTURE_STATE = 7, TIMER_QUERY = 8,
    VERTEX_BUFFER = 9, VERTEX_BUFFER_INFO = 10,
    DESCRIPTOR_SET_LAYOUT = 11, DESCRIPTOR_SET = 12,
    FENCE = 13, VULKAN_BUFFER = 14, STAGE_SEGMENT = 15,
    STAGE_IMAGE = 16, SYNC = 17, MEMORY_MAPPED_BUFFER = 18,
    SEMAPHORE = 19, STREAM = 20, FRAMEBUFFER = 21,
    RENDER_PASS = 22,
};
```

---

## 5. 多线程模型

### 5.1 线程架构

```
┌─────────────────┐     ┌─────────────────┐
│   用户线程       │     │   Driver 线程    │
│  (Application)  │     │  (Render Thread) │
├─────────────────┤     ├─────────────────┤
│ Engine::render() │     │ CommandBuffer   │
│ Scene::addEntity│     │ Queue::waitFor  │
│ Material::create│     │ Commands()      │
│     │           │     │     │           │
│     ▼           │     │     ▼           │
│ CommandStream   │────▶│ Driver::execute │
│   .method()     │     │   .method()     │
└─────────────────┘     └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Service Thread │
                    │ (用户回调调度)    │
                    └─────────────────┘
```

### 5.2 CommandBufferQueue

```cpp
class CommandBufferQueue {
    CircularBuffer mCircularBuffer;           // 环形缓冲区
    std::vector<Range> mCommandBuffersToExecute;  // 待执行命令列表
    std::mutex mLock;
    std::condition_variable mCondition;
    size_t mFreeSpace;                        // 剩余空间
    
    // 用户线程：录制命令，空间不足时阻塞等待
    void flush();
    
    // Driver 线程：等待并执行命令
    std::vector<Range> waitForCommands() const;
    
    // Driver 线程：释放已执行的命令缓冲区
    void releaseBuffer(Range const& buffer);
};
```

**关键特性**：
- 环形缓冲区实现无锁命令传递
- 空间不足时用户线程自动阻塞（背压机制）
- 命令录制完成后添加 `NoopCommand` 终止符
- 支持暂停/恢复（用于帧同步）

### 5.3 Vulkan 命令缓冲区管理

```cpp
class VulkanCommands {
    std::unique_ptr<CommandBufferPool> mPool;          // 普通命令池
    std::unique_ptr<CommandBufferPool> mProtectedPool; // 受保护内容命令池
    
    // 获取当前录制命令缓冲区
    VulkanCommandBuffer& get();
    
    // 提交当前命令缓冲区
    bool flush();
    
    // 依赖注入（用于 SwapChain 获取）
    void injectDependency(VkSemaphore next, VkPipelineStageFlags waitStage);
    
    // 获取完成信号量（用于 Present）
    fvkmemory::resource_ptr<VulkanSemaphore> acquireFinishedSignal();
};
```

**命令缓冲区生命周期**：
1. `get()` → 获取/创建录制状态的命令缓冲区
2. 录制 Vulkan 命令
3. `flush()` → 提交到队列，状态变为"执行中"
4. `gc()` → 回收已完成的命令缓冲区

### 5.4 同步机制

| 机制 | 用途 | 实现 |
|------|------|------|
| **VkSemaphore** | 命令缓冲区间依赖 | `VulkanSemaphoreManager` 管理信号量池 |
| **VkFence** | CPU-GPU 同步 | `VulkanFencePool` 管理栅栏池 |
| **Condition Variable** | 用户线程-Driver线程同步 | `CommandBufferQueue` 中的条件变量 |
| **Atomic** | 状态查询 | `VulkanCmdFence::mStatus` 原子变量 |

**VulkanCmdFence 状态管理**：
```cpp
struct VulkanCmdFence {
    std::atomic<VkResult> mStatus{VK_NOT_READY};
    VkFence mFence;
    
    // 从任意线程安全查询状态
    void refreshStatus(VkDevice device) {
        mStatus.store(vkGetFenceStatus(device, mFence));
    }
};
```

---

## 6. 着色器处理

### 6.1 着色器编译管线

```
Material 文件 (.mat)
    │
    ▼
filamat (MaterialBuilder)
    │  - 解析材质定义
    │  - 生成 GLSL (ESSL310/GLSL450)
    │  - 应用变体（Variant）系统
    ▼
SPIRV-Cross
    │  - GLSL → SPIR-V
    │  - SPIR-V → MSL (Metal)
    │  - SPIR-V → WGSL (WebGPU)
    │  - SPIR-V → 优化后的 GLSL
    ▼
Backend 特定格式
    ├── ESSL 3.0 (OpenGL ES)
    ├── SPIR-V (Vulkan)
    ├── MSL (Metal)
    ├── Metal Library (预编译)
    └── WGSL (WebGPU)
```

### 6.2 MaterialBuilder 配置

```cpp
class MaterialBuilderBase {
    enum class Platform { DESKTOP, MOBILE, ALL };
    enum class TargetApi : uint8_t {
        OPENGL = 0x01, VULKAN = 0x02, METAL = 0x04, WEBGPU = 0x08
    };
    enum class TargetLanguage { GLSL, SPIRV };
};
```

### 6.3 着色器语言支持

```cpp
enum class ShaderLanguage {
    ESSL1 = 0,          // OpenGL ES 2.0
    ESSL3 = 1,          // OpenGL ES 3.0 / OpenGL 4.x
    SPIRV = 2,          // Vulkan
    MSL = 3,            // Metal Shading Language
    METAL_LIBRARY = 4,  // 预编译 Metal Library
    WGSL = 5,           // WebGPU
};
```

### 6.4 着色器编译服务

**OpenGL ShaderCompilerService**：

```cpp
class ShaderCompilerService {
    // 异步编译支持
    program_token_t createProgram(CString const& name, Program&& program);
    GLuint getProgram(program_token_t& token);  // 阻塞获取结果
    
    // 优先级队列
    enum class CompilerPriorityQueue { CRITICAL, HIGH, LOW };
};
```

**CompilerThreadPool**：
```cpp
class CompilerThreadPool {
    std::vector<std::thread> mCompilerThreads;  // 编译线程池
    std::array<Queue, 3> mQueues;               // 三级优先级队列
    
    void init(uint32_t threadCount, ThreadSetup&& setup, ThreadCleanup&& cleanup);
    void queue(CompilerPriorityQueue priority, program_token_t const& token, Job&& job);
    Job dequeue(program_token_t const& token);
};
```

### 6.5 描述符集绑定模型

Filament 使用 **Vulkan 风格的描述符集**（Descriptor Set）模型：

```cpp
// 描述符集布局
struct DescriptorSetLayout {
    std::variant<StaticString, CString, monostate> label;
    FixedCapacityVector<DescriptorSetLayoutDescriptor> descriptors;
};

// 描述符类型
enum class DescriptorType : uint8_t {
    SAMPLER_2D_FLOAT, SAMPLER_2D_INT, SAMPLER_2D_UINT, SAMPLER_2D_DEPTH,
    SAMPLER_2D_ARRAY_FLOAT, /* ... */
    SAMPLER_CUBEMAP_FLOAT, /* ... */
    SAMPLER_3D_FLOAT, /* ... */
    SAMPLER_EXTERNAL,
    UNIFORM_BUFFER,
    SHADER_STORAGE_BUFFER,
    INPUT_ATTACHMENT,
};
```

**Vulkan 描述符集管理**：
- `VulkanDescriptorSetLayoutCache` - 布局缓存（按位掩码索引）
- `VulkanDescriptorSetCache` - 描述符集分配与更新
- 支持 4 个描述符集（`MAX_DESCRIPTOR_SET_COUNT = 4`）
- 支持动态偏移（Dynamic Offset）

---

## 7. 设计特点与独特之处

### 7.1 与其他 RHI 实现的对比

| 特性 | Filament | bgfx | UE-RHI | Diligent | NVRHI |
|------|----------|------|--------|----------|-------|
| **命令模型** | 单线程命令流 + 环形缓冲区 | 跨线程命令缓冲区 | 命令列表 + RHI线程 | 延迟上下文 | 命令列表 |
| **资源管理** | 类型化句柄 + 池化 | 索引句柄 | 引用计数 | COM 接口 | COM 接口 |
| **PSO 管理** | 运行时缓存 | 编译时绑定 | PSO 缓存 | 单体 PSO | PSO 缓存 |
| **着色器语言** | GLSL → SPIRV/MSL/WGSL | GLSL/HLSL/MSL | HLSL | HLSL (统一) | HLSL |
| **多后端支持** | 5 (GL/VK/MTL/WGPU/Noop) | 8+ | 3 (D3D11/D3D12/VK) | 3 (D3D11/D3D12/VK) | 3 (D3D11/D3D12/VK) |
| **线程模型** | 用户线程→Driver线程→Service线程 | 无锁多线程 | 渲染线程→RHI线程 | 延迟上下文多线程 | 依赖应用 |

### 7.2 Filament 的独特设计

#### 7.2.1 宏驱动的 Driver API

使用 `DriverAPI.inc` 文件和宏系统自动生成：
- Driver 虚函数接口
- CommandStream 命令方法
- Dispatcher 分发表
- 命令类型定义

```cpp
// DriverAPI.inc 中的声明
DECL_DRIVER_API_N(createTexture, SamplerType, target, uint8_t, levels, ...)
DECL_DRIVER_API_N(beginRenderPass, RenderTargetHandle, rth, const RenderPassParams&, params)
DECL_DRIVER_API_0(endRenderPass)
```

#### 7.2.2 环形缓冲区命令队列

用户线程和 Driver 线程通过共享环形缓冲区通信：
- 无锁设计，性能优异
- 自动背压（空间不足时用户线程阻塞）
- 支持暂停/恢复

#### 7.2.3 VMA 外部同步模式

Vulkan 后端禁用 VMA 内部同步，因为 Filament 保证所有 VMA 调用在单线程执行，提升性能。

#### 7.2.4 资源引用计数的双模式

```cpp
// 非线程安全（默认，性能更好）
struct Resource { uint32_t mCount; };

// 线程安全（特定资源类型）
struct ThreadSafeResource { std::atomic<uint32_t> mCount; };
```

仅 `PROGRAM`, `FENCE`, `TIMER_QUERY`, `SYNC` 使用线程安全模式。

#### 7.2.5 Descriptor Set 位掩码优化

```cpp
struct Bitmask {
    UniformBufferBitmask ubo;
    UniformBufferBitmask dynamicUbo;
    SamplerBitmask sampler;
    InputAttachmentBitmask inputAttachment;
    SamplerBitmask externalSampler;  // 外部采样器特殊处理
};
```

使用位掩码快速检查布局兼容性，避免字符串比较。

#### 7.2.6 Feature Level 系统

```cpp
enum class FeatureLevel : uint8_t {
    FEATURE_LEVEL_0 = 0,  // OpenGL ES 2.0
    FEATURE_LEVEL_1,      // OpenGL ES 3.0 (默认)
    FEATURE_LEVEL_2,      // OpenGL ES 3.1 + 16 纹理单元
    FEATURE_LEVEL_3,      // OpenGL ES 3.1 + 31 纹理单元 (Metal)
};
```

### 7.3 异步资源创建

Filament 支持异步资源创建（`createTextureAsync`, `createBufferObjectAsync`）：

```cpp
DECL_DRIVER_API_TAGGED_R_N(backend::TextureHandle, createTextureAsync,
    SamplerType, target, uint8_t, levels, TextureFormat, format,
    uint8_t, samples, uint32_t, w, uint32_t, h, uint32_t, depth,
    TextureUsage, usage,
    CallbackHandler*, handler, AsyncCallback, callback, void*, user)
```

异步创建流程：
1. 立即返回句柄（句柄可用）
2. 后端在 Driver 线程执行实际创建
3. 完成后调用用户回调通知

### 7.4 Blob 缓存支持

```cpp
class Platform {
    using InsertBlobFunc = Invocable<void(const void* key, size_t keySize,
                                          const void* value, size_t valueSize)>;
    using RetrieveBlobFunc = Invocable<size_t(const void* key, size_t keySize,
                                               void* value, size_t valueSize)>;
    
    void setBlobFunc(InsertBlobFunc&& insert, RetrieveBlobFunc&& retrieve);
};
```

允许应用层提供着色器缓存实现，加速后续启动。

---

## 8. 总结

Filament 的 RHI 实现展现了**工程精度极高的跨平台图形抽象**：

1. **宏驱动的 API 生成**：`DriverAPI.inc` 系统消除了接口、分发、类型定义之间的不一致风险
2. **高效的线程模型**：环形缓冲区 + 条件变量实现低延迟命令传递
3. **精细的资源管理**：三级句柄池 + 引用计数 + 延迟回收 + Use-After-Free 检测
4. **Vulkan 深度优化**：VMA 外部同步、描述符集位掩码、命令缓冲区依赖链
5. **灵活的着色器管线**：GLSL 中间表示 + 多目标转译 + 并行编译 + Blob 缓存
6. **渐进式功能支持**：Feature Level 系统适配不同硬件能力

Filament 的设计哲学是**简单高效**，避免过度抽象，直接暴露图形 API 的能力，同时通过精心设计的抽象层确保跨平台兼容性。这种设计使其特别适合**移动端 PBR 渲染**场景。

---

*分析基于 Filament 最新 main 分支源码*
