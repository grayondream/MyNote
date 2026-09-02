# RHI 设计考量：不同 API 差异分析

**摘要**：现代图形 API 形成了两条技术路线——以 OpenGL/GLES/DX11 为代表的"驱动托管"路线和以 DX12/Metal/Vulkan 为代表的"显式控制"路线。本文从 Render 与 Compute 两个维度系统对比六大主流 API（OpenGL、OpenGL ES、Direct3D 11、Direct3D 12、Metal、Vulkan）在线程模型、资源管理、绑定方式、管线状态、同步机制、着色器工具链等方面的差异，分析各 API 设计背后的硬件与历史考量，最后落到工程实践：设计一个跨平台 RHI（Render Hardware Interface）时需要考虑的关键问题。

**关键字**：图形 API；OpenGL；OpenGL ES；Direct3D；Metal；Vulkan；异步计算；RHI；渲染硬件接口；跨平台渲染；描述符；管线状态对象

## 1 简介

&emsp;&emsp;图形 API 是应用程序与 GPU 驱动之间的接口层。过去十几年里，这个领域发生了一次范式转移：早期 API 把资源分配、状态跟踪、同步、内存驻留等复杂性全部藏在驱动里，应用写起来简单，但 CPU 驱动开销高、行为不可预测；新一代 API 则把这些责任全部交还给应用——代码更冗长，但性能更可控、多线程能力更强。

&emsp;&emsp;理解这些差异不只是为了"选哪个 API"，更是为了理解 GPU 的工作方式本身。当你需要写一个跨引擎、跨平台的渲染抽象层（RHI）时，这些差异会直接决定抽象层的形状。RHI 的本质问题可以用一句话概括：**用什么形状的接口，才能让所有后端都不扭曲地映射到各自的本地 API？**

![](https://learnopengl.com/img/getting-started/pipeline.png)

&emsp;&emsp;本文先给出六大 API 的总览与定位，再从同步机制、资源、线程模型等多个维度逐项对比它们的差异，接着分析各 API 如此设计的背后考量，最后系统梳理 RHI 设计中的关键问题与工程取舍。

## 2 六大 API 总览

&emsp;&emsp;现代 GPU 虽然采用统一着色器架构，图形与计算共享同一组可编程核心，但二者侧重截然不同：图形任务依赖光栅化器、纹理单元、ROP 等固定功能硬件，沿顶点→光栅化→像素着色的刚性流水线执行，对每帧实时性要求苛刻，内存访问具有较强空间局部性，线程间通信少、同步隐式；而计算任务则完全脱离固定管线，通过计算着色器或 CUDA 等自由定义线程块与共享内存，显式控制寄存器、各级缓存和全局内存的分配与访问，支持组内屏障、原子操作等丰富同步机制，精度范围更广（FP64/FP32/FP16/BF16/INT8 乃至张量核心加速的矩阵运算），且对单次延迟容忍度更高。在现代 API（Vulkan、Direct3D 12、Metal）中，二者可通过异步队列并行执行，使物理模拟、AI 推理等计算负载与图形渲染重叠，从而充分利用硬件资源、降低整体帧时间。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/render_api_dx_metal_gl_vulkan.png)

| 维度 | OpenGL | OpenGL ES | Direct3D 11 | Direct3D 12 | Metal | Vulkan |
| ---- | ------ | --------- | ----------- | ----------- | ----- | ------ |
| 制定者 | Khronos | Khronos | Microsoft | Microsoft | Apple | Khronos |
| 平台覆盖 | 桌面全平台（macOS 停在 4.1 且已弃用） | 移动/嵌入式/Web（WebGL） | Windows/Xbox | Windows/Xbox | macOS/iOS/tvOS/watchOS | Windows/Linux/Android/Switch 等 |
| 抽象层级 | 高（驱动托管） | 高（驱动托管） | 中高（驱动托管） | 低（显式） | 低（显式但精简） | 低（显式且繁琐） |
| 着色器语言 | GLSL | ESSL | HLSL（DXBC） | HLSL（DXIL） | MSL | SPIR-V |
| 多线程命令录制 | 几乎不行（单上下文） | 不行 | 有限（Deferred Context） | 原生支持 | 原生支持 | 原生支持 |
| 内存管理 | 驱动托管 | 驱动托管 | 驱动托管 | 显式（堆/放置资源） | 半显式（Heap/Storage Mode） | 显式（Memory Type/Heap） |
| 同步机制 | 隐式为主 | 隐式为主 | 隐式为主 | 显式 Barrier/Fence | 自动跟踪 + 手动选项 | 完全显式 Barrier/Semaphore |
| 入门难度 | 容易 | 容易 | 中等 | 难 | 中等 | 难 |

&emsp;&emsp;可以把这六个 API 排成一条"控制力光谱"：**GL ≈ GLES < DX11 < Metal ≈ DX12 < Vulkan**。越往右，应用承担的责任越多（校验、同步、内存、生命周期），驱动的"魔法"越少，性能越可预测。这条光谱不是线性的——Metal 位于 DX12 和 Vulkan 之间，因为它在提供显式控制的同时保留了部分自动化（如 Hazard Tracking），体现了 Apple 对"开发效率与性能可控"的独特平衡。

## 3 不同API特性
### 3.1 OpenGL/OpenGL ES

&emsp;&emsp;OpenGL（Open Graphics Library）是历史悠久的跨平台图形 API，自1992年发布以来一直是图形渲染的事实标准。OpenGL ES（Embedded Systems）是其针对移动和嵌入式设备的精简版本，广泛用于 Android、iOS 和 WebGL。

&emsp;&emsp;OpenGL 采用状态机模型，通过全局状态控制渲染行为，早期以立即模式指定顶点数据，现代核心模式则改用顶点缓冲区对象（VBO）和顶点数组对象（VAO）来管理几何数据。它使用 GLSL 编写顶点、片段及几何着色器，并通过扩展机制支持新硬件特性；内存管理、资源分配和同步等完全由驱动托管，应用无需显式控制底层细节。

&emsp;&emsp;OpenGL 的主要优势在于 API 直观、学习曲线平缓，跨平台支持 Windows、Linux 和 macOS（尽管 macOS 已停更在 4.1），且拥有 RenderDoc 等成熟调试工具和丰富教程。但其劣势也很明显：上下文模型限制多线程命令录制，驱动内部状态跟踪与验证带来较高 CPU 开销，内存管理不透明可能导致碎片和性能波动，同时不同平台版本碎片化严重，功能支持不一致。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/Linux_kernel_and_OpenGL_video_games.png)

&emsp;&emsp;在设计跨平台 RHI 时，OpenGL 后端通常作为“基线”实现，用于兼容旧硬件或快速原型。RHI 需要将 OpenGL 的状态机模型映射到显式 API 的管线状态对象（PSO）模型，这可能需要维护内部状态缓存和延迟提交。由于 OpenGL 的多线程限制，RHI 可能需要将命令录制限制在主线程，或使用辅助上下文进行纹理上传等操作。

### 3.2 Vulkan

&emsp;&emsp;Vulkan 是 Khronos Group 于 2016 年发布的现代图形 API，旨在提供低开销、跨平台的硬件抽象。它取代了 OpenGL 的地位，成为 Android、Linux 和 Windows 上高性能图形和计算的首选。Vulkan 采用完全显式控制模型，应用负责内存管理、资源状态同步、管线状态构建等所有细节，驱动仅执行最小验证。它支持多线程原生命令录制，命令缓冲区可在多线程上并行录制，显著提升 CPU 利用率。着色器使用 SPIR-V 二进制格式，支持离线编译和跨后端共享。资源绑定通过描述符集（Descriptor Set）实现，支持动态偏移和更新。渲染通道（Render Pass）显式定义，可优化 tile-based GPU 的加载/存储操作。同步原语包括信号量（Semaphore）、栅栏（Fence）、事件（Event）等，提供精细的同步控制。

&emsp;&emsp;Vulkan 的主要优势在于高性能和低驱动开销，适合 CPU 受限的应用（如游戏引擎、实时渲染）。它具有良好的跨平台支持，覆盖 Windows、Linux、Android、Switch 等，避免平台锁定。计算与图形队列可并行执行，支持异步计算。通过扩展机制，Vulkan 支持光线追踪、网格着色器等前沿特性。然而，Vulkan 的复杂度较高，代码量远大于 OpenGL，需要管理大量样板代码（如实例、设备、队列、命令池等）。学习曲线陡峭，概念繁多（如内存类型、队列族、描述符布局），初学者难以掌握。开发时依赖验证层捕获错误，但验证层可能影响性能。不同厂商的驱动质量不一，可能导致跨平台兼容性问题。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/render_api_dx_metal_gl_vulkan%E2%80%94%E2%80%94vulkan_driver.png)

&emsp;&emsp;在设计跨平台 RHI 时，Vulkan 是主要目标后端之一。RHI 需要将高层抽象（如资源绑定、同步）映射到 Vulkan 的显式模型。关键挑战包括：管理内存分配（推荐使用 VMA 库）、处理资源状态转换（通过管线屏障）、构建管线状态对象（VkPipeline）、调度异步计算。RHI 通常隐藏 Vulkan 的样板代码，提供简化的资源创建和绑定接口。

### 3.3 Direct3D 11

&emsp;&emsp;Direct3D 11（D3D11）是微软于 2009 年发布的图形 API，作为 DirectX 家族的一员，长期主导 Windows 和 Xbox 平台的游戏开发。它在易用性和性能之间取得了良好平衡，至今仍被广泛使用。Direct3D 11 采用驱动托管模型，内存管理、资源分配、同步等由驱动自动处理，应用只需调用 API 创建和销毁对象。它支持特性级别（Feature Level）从 9.3 到 11.1，兼容不同代 GPU。着色器使用 HLSL（High-Level Shading Language）编写，编译为 DXBC（DirectX Bytecode）。提供有限的多线程支持，通过延迟上下文（Deferred Context）可将命令录制到延迟上下文，但最终仍需主上下文执行。支持通用计算着色器，但异步计算能力有限。

&emsp;&emsp;Direct3D 11 的主要优势在于易用性，API 设计直观，学习曲线平缓，适合快速开发。经过多年验证，驱动成熟稳定，兼容性好。工具链完善，Visual Studio 图形调试器、PIX 等工具支持良好。广泛支持 Windows 7 及以上系统，Xbox 360/One 向后兼容。然而，其多线程限制明显，Deferred Context 性能提升有限，且不能与 Immediate Context 并行。驱动内部状态跟踪和验证导致 CPU 开销，性能可预测性较低。内存管理不透明，应用无法控制显存分配，可能导致内存碎片和性能波动。相比 Vulkan 和 DX12，缺少光线追踪、网格着色器等新特性。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/render_api_dx_metal_gl_vulkan%E2%80%94%E2%80%94dx11_driver.png)

&emsp;&emsp;在设计跨平台 RHI 时，Direct3D 11 是重要后端，尤其针对 Windows 平台和旧硬件。RHI 需要将高层绑定模型映射到 D3D11 的状态机，例如将描述符集转换为常量缓冲区、纹理和采样器的单独绑定。由于 D3D11 的多线程限制，RHI 通常将命令录制限制在主线程，或使用延迟上下文进行并行录制，但需注意性能开销。

### 3.4 Direct3D 12

&emsp;&emsp;Direct3D 12（D3D12）是微软于 2015 年发布的低开销图形 API，旨在提供对 GPU 硬件的完全控制，与 Vulkan 和 Metal 竞争。它要求应用显式管理资源、内存和同步，但回报是更高的性能和可预测性。Direct3D 12 采用完全显式控制模型，应用负责内存分配、资源状态管理、同步和管线状态构建。支持多线程原生命令录制，命令列表（Command List）可在多线程上并行录制，显著提升 CPU 利用率。根签名（Root Signature）定义管线使用的资源布局，支持描述符表、根描述符和根常量。管线状态对象（PSO）将所有管线状态（着色器、混合、深度模板等）打包为一个不可变对象，减少状态切换开销。显式内存管理通过堆（Heap）和放置资源（Placed Resource）管理显存，支持内存别名。计算队列与图形队列并行执行，支持异步计算和复制。

&emsp;&emsp;Direct3D 12 的主要优势在于高性能，低驱动开销，适合 CPU 受限的应用（如 AAA 游戏、实时渲染）。支持 Windows 10 及以上和 Xbox One 系列，具有良好的跨平台能力。计算与图形队列可并行执行，支持异步计算。通过 DirectX Raytracing（DXR）和网格着色器等前沿特性，提供先进的图形功能。然而，Direct3D 12 的复杂度较高，代码量远大于 D3D11，需要管理大量样板代码（如命令队列、命令分配器、围栏等）。学习曲线陡峭，概念繁多（如堆类型、根签名、描述符堆），初学者难以掌握。开发时依赖调试层捕获错误，但调试层可能影响性能。平台限制明显，仅支持 Windows 10 及以上和 Xbox，跨平台能力有限。

&emsp;&emsp;在设计跨平台 RHI 时，Direct3D 12 是主要目标后端之一，尤其针对 Windows 和 Xbox 平台。RHI 需要将高层抽象映射到 D3D12 的显式模型，关键挑战包括：管理内存分配（推荐使用 D3D12MA 库）、处理资源状态转换（通过资源屏障）、构建管线状态对象（PSO）、调度异步计算。RHI 通常隐藏 D3D12 的样板代码，提供简化的资源创建和绑定接口。

### 3.5 Metal

&emsp;&emsp;Metal 是 Apple 于 2014 年发布的低开销图形和计算 API，专为 macOS、iOS、tvOS 和 watchOS 设计。它提供了对 Apple 硬件的深度优化，同时保持了相对简洁的 API 设计。Metal 采用半显式控制模型，应用负责资源创建和存储模式选择，但内存管理和同步部分由驱动自动处理（如 Hazard Tracking）。支持多线程原生命令录制，命令缓冲区（Command Buffer）可在多线程上并行录制，支持并行渲染和计算。着色器使用 Metal Shading Language（MSL），基于 C++ 语法，支持内核函数（Kernel Function）用于计算。资源堆（Heap）允许应用创建堆并在其中分配资源，支持内存别名和子分配。存储模式（Storage Mode）控制资源内存位置和 CPU 可见性（共享、私有、托管）。渲染通道（Render Pass）显式定义附件操作，优化 tile-based GPU 的加载/存储。

&emsp;&emsp;Metal 的主要优势在于 Apple 生态深度集成，与 Core Animation、Core Image、ARKit 等框架无缝协作。针对 Apple 硬件（如 A 系列、M 系列芯片）深度优化，充分利用 tile-based 架构，提供卓越的性能。相比 Vulkan 和 DX12，API 设计更简洁，学习曲线较平缓。计算队列与图形队列共享内存，避免数据拷贝，实现统一计算与图形。然而，Metal 平台锁定明显，仅支持 Apple 平台，无法跨平台。生态限制较大，依赖 Apple 的工具链（Xcode）和更新周期。文档和社区规模较小，相比 Vulkan 和 DirectX 资源有限。macOS 上的 Metal 功能集略落后于 iOS（如光线追踪支持较晚）。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/render_api_dx_metal_gl_vulkan%E2%80%94%E2%80%94metal_driver.jpg)

&emsp;&emsp;在设计跨平台 RHI 时，Metal 是关键后端，尤其针对 Apple 平台。RHI 需要将高层抽象映射到 Metal 的半显式模型，关键挑战包括：管理存储模式（共享、私有、托管）、处理资源状态（依赖 Hazard Tracking 或手动屏障）、构建渲染通道描述符、调度异步计算。RHI 通常隐藏 Metal 的样板代码，提供简化的资源创建和绑定接口，并利用 Apple 的框架集成（如 CAMetalLayer）。

## 4 不同维度对比不同API
### 4.1 资源管理与显存控制

&emsp;&emsp;资源管理与显存控制是图形 API 差异最显著的领域之一。旧一代 API 将显存分配、驻留、生命周期完全交给驱动，开发者只需调用 `glGenTextures`、`glBufferData` 等接口，驱动在后台完成所有簿记工作；新一代 API 则将内存管理的责任完全下放给应用，开发者必须显式地选择内存类型、创建堆、管理资源状态，甚至处理 GPU 前后端内存的可见性。这种转变带来了更高的性能上限和可预测性，但也大幅增加了代码复杂度。

#### 4.1.1 内存管理模型对比

| API | 内存管理模型 | 分配单元 | 驱动职责 | 应用职责 |
| --- | ---------- | -------- | -------- | -------- |
| **OpenGL** | 完全驱动托管 | 无显式概念 | 所有分配、驻留、回收 | 仅调用 API 创建/销毁对象 |
| **OpenGL ES** | 完全驱动托管 | 无显式概念 | 同 OpenGL | 同 OpenGL |
| **Direct3D 11** | 驱动托管 + 分类提示 | 无显式堆 | 所有分配、驻留、回收；根据 `D3D11_USAGE` 提示优化 | 指定用途（默认/上传/读回） |
| **Direct3D 12** | 完全显式 | 堆（Heap） | 分配物理显存页 | 创建堆、选择内存类型、管理驻留 |
| **Metal** | 半显式 | 堆（Heap） | 管理物理内存 | 选择存储模式（共享/私有/托管） |
| **Vulkan** | 完全显式 | 内存堆（Memory Heap） | 暴露可用内存类型 | 选择内存类型、分配内存块、绑定资源 |

&emsp;&emsp;在显式 API 中，内存管理的核心流程是：**查询可用内存类型 → 创建堆/分配内存 → 创建资源并绑定内存 → 管理资源状态与驻留**。这一流程在 DX12、Metal、Vulkan 中大体一致，但细节差异显著。

#### 4.1.2 显存分配与堆管理

##### 4.1.2.1 Direct3D 12 的堆与放置资源

&emsp;&emsp;DX12 引入了 `ID3D12Heap` 接口，应用可以创建一个大堆，然后在堆内放置多个资源（Placed Resources）。堆本身有 `D3D12_HEAP_TYPE` 分类：

| 堆类型 | 显存位置 | CPU 访问 | 典型用途 |
| ------ | -------- | -------- | -------- |
| `D3D12_HEAP_TYPE_DEFAULT` | 显存（VRAM） | 不可直接访问 | 纹理、缓冲区、渲染目标 |
| `D3D12_HEAP_TYPE_UPLOAD` | 系统内存 | 可写 | 上传缓冲区（纹理/缓冲区数据） |
| `D3D12_HEAP_TYPE_READBACK` | 系统内存 | 可读 | GPU 回读（查询、截图） |
| `D3D12_HEAP_TYPE_CUSTOM` | 自定义 | 自定义 | 高级用法（多适配器等） |

```cpp
// DX12: 创建堆并放置资源
D3D12_HEAP_PROPERTIES heapProps = {};
heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

D3D12_HEAP_DESC heapDesc = {};
heapDesc.SizeInBytes = 256 * 1024 * 1024; // 256MB
heapDesc.Properties = heapProps;
heapDesc.Flags = D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES;

ComPtr<ID3D12Heap> heap;
device->CreateHeap(&heapDesc, IID_PPV_ARGS(&heap));

// 在堆内place new纹理
D3D12_RESOURCE_DESC texDesc = CD3DX12_RESOURCE_DESC::Texture2D(
    DXGI_FORMAT_R8G8B8A8_UNORM, 1024, 1024);

D3D12_RESOURCE_ALLOCATION_INFO allocInfo =
    device->GetResourceAllocationInfo(0, 1, &texDesc);

ComPtr<ID3D12Resource> texture;
device->CreatePlacedResource(
    heap.Get(),
    D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT, 
    &texDesc,
    D3D12_RESOURCE_STATE_COMMON,
    nullptr,
    IID_PPV_ARGS(&texture));
```

&emsp;&emsp;放置资源的关键优势是**减少内存碎片和分配开销**。应用可以预分配一个大堆，然后将生命周期相近的资源放在同一堆中，避免频繁的驱动分配调用。

##### 4.1.2.2 Vulkan 的内存类型与堆

&emsp;&emsp;Vulkan 通过 `vkGetPhysicalDeviceMemoryProperties` 查询设备支持的内存类型和内存堆。每个内存类型关联一个内存堆（物理显存区域），并具有特定的属性标志：

| 属性标志 | 含义 | 典型用途 |
| -------- | ---- | -------- |
| `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` | 仅 GPU 可访问 | 纹理、渲染目标、GPU 专用缓冲区 |
| `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` | CPU 可映射 | 上传/回读缓冲区 |
| `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` | CPU 写入自动对 GPU 可见 | 无需显式刷新映射 |
| `VK_MEMORY_PROPERTY_HOST_CACHED_BIT` | CPU 缓存加速读取 | 频繁回读的数据 |
| `VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT` | 按需分配 | 临时渲染目标（移动端 tile 内存） |
| `VK_MEMORY_PROPERTY_PROTECTED_BIT` | 受保护内存 | DRM 内容 |

```cpp
// Vulkan: 查询内存类型
VkPhysicalDeviceMemoryProperties memProps;
vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

// 寻找设备本地且主机可见的内存类型（用于统一内存架构）
uint32_t unifiedMemType = UINT32_MAX;
for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
    if ((memProps.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
        (memProps.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
        unifiedMemType = i;
        break;
    }
}

// 分配缓冲区内存
VkMemoryAllocateInfo allocInfo = {};
allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
allocInfo.allocationSize = memReqs.size;
allocInfo.memoryTypeIndex = unifiedMemType;

VkDeviceMemory bufferMemory;
vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
vkBindBufferMemory(device, buffer, bufferMemory, 0);
```

&emsp;&emsp;Vulkan 的内存模型允许应用实现**子分配（Sub-allocation）**：分配一大块内存，然后通过偏移量将不同资源绑定到同一内存块。这是性能优化的关键手段，但需要应用自行管理对齐和屏障。

##### 4.1.2.3 Metal 的堆与存储模式

&emsp;&emsp;Metal 提供了三种存储模式，控制纹理和缓冲区的内存位置与 CPU 可见性：

| 存储模式 | 内存位置 | CPU 访问 | 典型用途 |
| -------- | -------- | -------- | -------- |
| `MTLResourceStorageModeShared` | 统一内存架构：系统内存；离散架构：驱动选择 | CPU 和 GPU 均可访问 | 频繁 CPU 更新的数据 |
| `MTLResourceStorageModePrivate` | 仅 GPU 可访问 | 不可直接访问 | GPU 专用资源（纹理、渲染目标） |
| `MTLResourceStorageModeManaged` | 系统内存（CPU）+ GPU 缓存 | CPU 可映射，需手动同步 | macOS 上的高效 CPU/GPU 共享 |

```swift
// Metal: 创建堆并分配资源
let device = MTLCreateSystemDefaultDevice()!

// 创建私有堆（GPU 专用）
let heapDescriptor = MTLHeapDescriptor()
heapDescriptor.size = 256 * 1024 * 1024  // 256MB
heapDescriptor.storageMode = .private
heapDescriptor.hazardTrackingMode = .tracked

let heap = device.makeHeap(descriptor: heapDescriptor)!

// 在堆中创建纹理
let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: .rgba8Unorm,
    width: 1024, height: 1024,
    mipmapped: false)
textureDescriptor.storageMode = .private
textureDescriptor.usage = [.shaderRead, .renderTarget]

let texture = heap.makeTexture(descriptor: textureDescriptor)!
```

&emsp;&emsp;Metal 的堆管理比 DX12 和 Vulkan 更简洁，应用只需关注存储模式，驱动会处理内存对齐和页管理。

#### 4.1.3 资源状态管理

&emsp;&emsp;在显式 API 中，资源状态（Resource State）管理是同步的基础。GPU 在不同阶段以不同方式访问资源，应用必须在访问前将资源转换到正确的状态。

| API | 状态管理方式 | 状态粒度 | 状态转换 |
| --- | ---------- | -------- | -------- |
| **Direct3D 11** | 驱动自动处理 | 无显式状态 | 自动（隐式同步） |
| **Direct3D 12** | 显式屏障 | 资源整体或子资源 | `ResourceBarrier` 显式转换 |
| **Metal** | 自动跟踪 + 手动选项 | 资源整体 | Hazard Tracking 自动检测冲突 |
| **Vulkan** | 显式管线屏障 | 资源整体或子资源 | `vkCmdPipelineBarrier` 显式转换 |

```cpp
// DX12: 资源状态转换
D3D12_RESOURCE_BARRIER barrier = {};
barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
barrier.Transition.pResource = texture.Get();
barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

commandList->ResourceBarrier(1, &barrier);
```

```cpp
// Vulkan: 管线屏障（图像布局转换）
VkImageMemoryBarrier barrier = {};
barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
barrier.image = image;
barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
barrier.subresourceRange.levelCount = 1;
barrier.subresourceRange.layerCount = 1;
barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

vkCmdPipelineBarrier(
    commandBuffer,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    0, 0, nullptr, 0, nullptr, 1, &barrier);
```

&emsp;&emsp;Metal 的 Hazard Tracking 机制是独特的折中：驱动在后台跟踪资源访问冲突，当检测到 Hazard 时自动插入屏障。应用可以选择完全依赖自动跟踪（简化代码），或手动插入屏障以获得更高性能。

#### 4.1.4 资源别名与内存复用

&emsp;&emsp;在资源生命周期不重叠的情况下，多个资源可以共享同一块内存，这就是**资源别名（Resource Aliasing）**。这对减少显存占用至关重要。

| API | 别名支持 | 实现方式 |
| --- | -------- | -------- |
| **Direct3D 12** | 原生支持 | 创建放置资源时使用同一堆，通过屏障管理访问 |
| **Metal** | 原生支持 | 使用 `makeAliasable()` 标记资源可被别名 |
| **Vulkan** | 原生支持 | 稀疏绑定（Sparse Binding）或资源别名(Memory Aliasing) |
| **OpenGL/DX11** | 不直接支持 | 驱动可能内部优化，但应用无法控制 |

```swift
// Metal: 资源别名
let texture1 = heap.makeTexture(descriptor: desc1)!
texture1.makeAliasable()  // 标记为可别名

let texture2 = heap.makeTexture(descriptor: desc2)!  // 复用同一内存
// texture1 和 texture2 不会同时使用
```

#### 4.1.5 跨平台 RHI 的资源管理策略

&emsp;&emsp;设计跨平台 RHI 时，资源管理需要在不同 API 之间找到最大公约数。常见的策略包括：

##### 4.1.5.1 统一内存模型抽象

&emsp;&emsp;RHI 可以定义一套内存类型枚举，映射到各后端的具体内存类型：

```cpp
enum class RHIMemoryType {
    DeviceLocal,      // GPU 专用显存
    HostVisible,      // CPU 可映射内存（上传/回读）
    HostCoherent,     // CPU 写入自动对 GPU 可见
    Unified,          // 统一内存（设备本地 + 主机可见）
    LazilyAllocated,  // 按需分配（移动端 tile 内存）
};
```

##### 4.1.5.2 资源分配器（Resource Allocator）

&emsp;&emsp;工程实践中通常使用**子分配器**减少显存分配次数。流行的库包括：

| 库名 | 支持 API | 特点 |
| ---- | -------- | ---- |
| **VMA (Vulkan Memory Allocator)** | Vulkan | 轻量、子分配、别名管理 |
| **D3D12MA (D3D12 Memory Allocator)** | DX12 | 与 VMA 接口一致 |
| **MetalKit** | Metal | Apple 官方，简单易用 |
| **AMD GPUOpen Memory Allocator** | DX12/Vulkan | 跨 API |

&emsp;&emsp;这些分配器提供统一的接口，隐藏各后端的内存管理细节，是 RHI 设计的重要基础设施。

##### 4.1.5.3 资源生命周期管理

&emsp;&emsp;资源销毁需要确保 GPU 不再使用该资源。不同 API 的策略不同：

| API | 销毁策略 |
| --- | -------- |
| **OpenGL** | 调用 `glDelete*`，驱动延迟回收 |
| **Direct3D 11** | 调用 `Release()`，驱动延迟回收 |
| **Direct3D 12** | 必须确保 GPU 完成使用后才能销毁，通常通过围栏（Fence） |
| **Metal** | 引用计数，GPU 使用完自动释放 |
| **Vulkan** | 必须显式销毁，需等待设备空闲或使用删除队列 |

&emsp;&emsp;RHI 通常实现**延迟删除队列**：将待销毁资源放入队列，每帧检查 GPU 完成状态，安全释放已不再使用的资源。

#### 4.1.6 移动端优化考量

&emsp;&emsp;移动 GPU（如 ARM Mali、Qualcomm Adreno、Apple A/M 系列）通常采用 **Tile-Based Deferred Rendering (TBDR)** 架构，显存管理有特殊考量：

| 特性 | 影响 | RHI 应对 |
| ---- | ---- | -------- |
| **统一内存架构** | CPU 和 GPU 共享物理内存 | 使用 `HostVisible + DeviceLocal` 内存类型 |
| **Tile 内存** | 片上小容量高速内存 | 使用 `LazilyAllocated` 内存，优化 tile 利用率 |
| **带宽限制** | 系统内存带宽有限 | 减少纹理格式大小，使用压缩纹理（ASTC/ETC） |
| **功耗敏感** | 内存访问影响功耗 | 优化内存局部性，减少随机访问 |

&emsp;&emsp;在 Vulkan 中，`VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT` 允许驱动使用 tile 内存而非系统内存存储渲染目标，这对移动端性能至关重要。Metal 的 `MTLResourceStorageModePrivate` 在 iOS 上默认使用 tile 内存。

#### 4.1.7 小结

&emsp;&emsp;资源管理与显存控制是显式 API 复杂性的主要来源。从 OpenGL 的"完全托管"到 Vulkan 的"完全显式"，开发者承担的责任逐步增加，但性能上限和可预测性也同步提升。RHI 设计的关键是在这些差异之上建立统一抽象，同时保留各后端的优化路径。子分配器、延迟删除、资源别名等模式是工程实践中的标准解决方案。

## 5 多线程命令录制与提交

&emsp;&emsp;多线程命令录制是现代图形 API 提升 CPU 利用率的关键特性。旧一代 API（OpenGL、GLES、DX11）的多线程支持有限，而新一代 API（DX12、Metal、Vulkan）则提供了原生的多线程命令录制能力。这种差异直接影响 RHI 的设计：RHI 必须提供统一的多线程抽象，同时处理各后端的实现细节。

### 5.1 多线程模型对比

| API | 多线程支持 | 命令录制模型 | 命令提交模型 | 资源绑定模型 |
| --- | ---------- | ------------ | ------------ | ------------ |
| **OpenGL** | 几乎不行 | 单上下文，单线程 | 单上下文提交 | 全局状态，单线程 |
| **OpenGL ES** | 几乎不行 | 单上下文，单线程 | 单上下文提交 | 全局状态，单线程 |
| **Direct3D 11** | 有限（Deferred Context） | 延迟上下文可多线程录制 | 主上下文提交 | 全局状态，单线程 |
| **Direct3D 12** | 原生支持 | 命令列表可多线程录制 | 命令队列提交 | 描述符堆，多线程 |
| **Metal** | 原生支持 | 命令缓冲区可多线程录制 | 命令队列提交 | 参数缓冲区，多线程 |
| **Vulkan** | 原生支持 | 命令缓冲区可多线程录制 | 队列提交 | 描述符集，多线程 |

### 5.2 命令录制模型

#### 5.2.1 OpenGL 的单线程限制

&emsp;&emsp;OpenGL 使用上下文（Context）模型，一个上下文绑定到一个线程。虽然可以通过共享上下文在多线程间共享纹理和缓冲区，但命令录制必须在主线程进行。这导致 CPU 利用率低下，尤其是在复杂场景中。

```cpp
// OpenGL: 单线程命令录制
glBindTexture(GL_TEXTURE_2D, texture);
glDrawArrays(GL_TRIANGLES, 0, 3);
// 所有 OpenGL 调用必须在同一线程
```

#### 5.2.2 Direct3D 11 的延迟上下文

&emsp;&emsp;DX11 引入了延迟上下文（Deferred Context），允许在多线程上录制命令，但最终必须通过主上下文（Immediate Context）执行。延迟上下文录制的命令列表可以提交给立即上下文执行，但存在性能开销。

```cpp
// DX11: 延迟上下文多线程录制
ID3D11DeviceContext* deferredContext;
device->CreateDeferredContext(0, &deferredContext);

// 在子线程录制命令
deferredContext->Draw(3, 0);

// 获取命令列表
ID3D11CommandList* commandList;
deferredContext->FinishCommandList(FALSE, &commandList);

// 主线程执行命令列表
immediateContext->ExecuteCommandList(commandList, TRUE);
```

#### 5.2.3 Direct3D 12 的命令列表

&emsp;&emsp;DX12 使用命令列表（Command List）模型，每个命令列表可以独立录制，无需同步。多个线程可以同时录制不同的命令列表，最后提交到命令队列。命令列表是轻量级对象，可以快速创建和重置。

```cpp
// DX12: 多线程命令录制
// 每个线程有独立的命令分配器和命令列表
struct ThreadContext {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commandList;
};

// 子线程录制命令
void RenderThread(ThreadContext& ctx) {
    ctx.allocator->Reset();
    ctx.commandList->Reset(ctx.allocator.Get(), nullptr);
    
    // 录制渲染命令
    ctx.commandList->DrawInstanced(3, 1, 0, 0);
    
    ctx.commandList->Close();
}

// 主线程提交所有命令列表
std::vector<ID3D12CommandList*> commandLists;
for (auto& ctx : threadContexts) {
    commandLists.push_back(ctx.commandList.Get());
}
commandQueue->ExecuteCommandLists(commandLists.size(), commandLists.data());
```

#### 5.2.4 Metal 的命令缓冲区

&emsp;&emsp;Metal 使用命令缓冲区（Command Buffer）模型，每个命令缓冲区可以从命令队列创建，并在单一线程上录制；多个线程可各自创建并录制不同的命令缓冲区以实现并行。

```swift
// Metal: 多线程命令录制
let commandBuffer = commandQueue.makeCommandBuffer()!

// 在任意线程录制命令
let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
encoder.endEncoding()

// 提交命令缓冲区
commandBuffer.commit()
```

#### 5.2.5 Vulkan 的命令缓冲区

&emsp;&emsp;Vulkan 使用命令缓冲区（Command Buffer）模型，每个命令缓冲区可以从命令池（Command Pool）创建。命令池与队列族关联，不同线程可以使用不同的命令池并行录制。命令缓冲区录制完成后提交到队列。

```cpp
// Vulkan: 多线程命令录制
// 每个线程有自己的命令池
VkCommandPoolCreateInfo poolInfo = {};
poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; 
poolInfo.queueFamilyIndex = graphicsQueueFamily;
vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);

// 分配命令缓冲区
VkCommandBuffer commandBuffer;
VkCommandBufferAllocateInfo allocInfo = {};
allocInfo.commandPool = commandPool;
allocInfo.commandBufferCount = 1;
vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

// 录制命令
vkBeginCommandBuffer(commandBuffer, &beginInfo);
vkCmdDraw(commandBuffer, 3, 1, 0, 0);
vkEndCommandBuffer(commandBuffer);

// 提交到队列
VkSubmitInfo submitInfo = {};
submitInfo.commandBufferCount = 1;
submitInfo.pCommandBuffers = &commandBuffer;
vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
```

### 5.3 命令提交与队列

&emsp;&emsp;命令提交模型决定了命令如何被 GPU 执行。不同 API 提供了不同粒度的队列和提交机制。

| API | 队列模型 | 提交粒度 | 异步执行 |
| --- | -------- | -------- | -------- |
| **OpenGL** | 单一上下文队列 | 隐式批量提交 | 驱动控制 |
| **Direct3D 11** | 单一命令队列 | 隐式批量提交 | 驱动控制 |
| **Direct3D 12** | 多队列（图形/计算/复制） | 显式命令列表提交 | 显式控制 |
| **Metal** | 多队列（图形/计算） | 显式命令缓冲区提交 | 显式控制 |
| **Vulkan** | 多队列（图形/计算/传输） | 显式命令缓冲区提交 | 显式控制 |

### 5.4 资源绑定与多线程

&emsp;&emsp;在多线程环境下，资源绑定需要线程安全。不同 API 提供了不同的绑定模型。

| API | 绑定模型 | 线程安全 | 多线程绑定策略 |
| --- | -------- | -------- | -------------- |
| **OpenGL** | 全局状态 | 否 | 单线程绑定 |
| **Direct3D 11** | 全局状态 | 否 | 单线程绑定 |
| **Direct3D 12** | 描述符堆 + 根签名 | 是 | 描述符堆本身非线程安全，需每线程独立 |
| **Metal** | 参数缓冲区 | 是 | 每个线程独立参数缓冲区 |
| **Vulkan** | 描述符集 | 是 | 每个线程独立描述符集 |

### 5.5 同步与屏障

&emsp;&emsp;多线程命令录制引入了新的同步需求：线程间同步、命令列表间同步、GPU-CPU 同步。不同 API 提供了不同的同步机制。

| API | 线程间同步 | 命令列表间同步 | GPU-CPU 同步 |
| --- | ---------- | -------------- | ------------ |
| **OpenGL** | 不适用 | 驱动内部同步 | 隐式同步 |
| **Direct3D 11** | 不适用 | 驱动内部同步 | 隐式同步 |
| **Direct3D 12** | 应用显式同步 | 围栏（Fence） | 围栏（Fence） |
| **Metal** | 驱动自动跟踪 | 事件（Event） | 事件（Event） |
| **Vulkan** | 应用显式同步 | 信号量（Semaphore） | 栅栏（Fence） |

### 5.6 跨平台 RHI 的多线程策略

&emsp;&emsp;设计跨平台 RHI 时，多线程抽象需要平衡各后端的差异。常见的策略包括：

#### 5.6.1 统一命令录制模型

&emsp;&emsp;RHI 可以定义统一的命令列表（Command List）接口，映射到各后端的具体实现：

```cpp
class RHICommandList {
public:
    virtual void Begin() = 0;
    virtual void End() = 0;
    virtual void Draw(uint32_t vertexCount, uint32_t instanceCount) = 0;
    virtual void Dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) = 0;
};
```

#### 5.6.2 线程本地存储

&emsp;&emsp;RHI 可以使用线程本地存储（TLS）为每个线程维护独立的命令录制上下文：

```cpp
class RHIContext {
    static thread_local RHICommandList* currentCommandList;
public:
    static void SetCurrentCommandList(RHICommandList* list) {
        currentCommandList = list;
    }
    static RHICommandList& GetCurrentCommandList() {
        return *currentCommandList;
    }
};
```

#### 5.6.3 命令队列抽象

&emsp;&emsp;RHI 可以定义统一的命令队列接口，支持图形、计算和传输队列：

```cpp
enum class RHIQueueType {
    Graphics,
    Compute,
    Transfer,
};

class RHICommandQueue {
public:
    virtual void Submit(RHICommandList& commandList) = 0;
    virtual void WaitForIdle() = 0;
};
```

### 5.7 小结

&emsp;&emsp;多线程命令录制是现代图形 API 的核心特性，直接决定了 CPU 利用率和渲染性能。OpenGL 和 DX11 的多线程支持有限，而 DX12、Metal 和 Vulkan 提供了原生的多线程能力。RHI 设计的关键是在这些差异之上建立统一的多线程抽象，同时保留各后端的优化路径。线程本地存储、统一命令列表和队列抽象是工程实践中的标准解决方案。

## 6 同步机制与屏障

&emsp;&emsp;同步机制是图形 API 中确保数据一致性和执行顺序的关键部分。在 GPU 并行执行的环境下，应用必须显式或隐式地管理 CPU-GPU 同步、GPU-GPU 同步、内存可见性和资源状态转换。不同 API 提供了不同粒度的同步原语，直接影响 RHI 的设计。

### 6.1 同步机制对比

| API | CPU-GPU 同步 | GPU-GPU 同步 | 内存可见性 | 资源状态管理 |
| --- | ------------ | ------------ | ---------- | ------------ |
| **OpenGL** | 隐式（`glFinish`/`glFenceSync`） | 隐式（驱动内部） | 隐式 | 驱动自动 |
| **Direct3D 11** | 隐式（`Map`/`Unmap`） | 隐式（驱动内部） | 隐式 | 驱动自动 |
| **Direct3D 12** | 显式围栏（Fence） | 显式围栏（Fence） | 显式屏障 | 显式资源屏障 |
| **Metal** | 显式（等待命令缓冲完成） | 显式事件（Event） | 自动跟踪 + 手动选项 | Hazard Tracking |
| **Vulkan** | 显式栅栏（Fence） | 显式信号量（Semaphore） | 显式内存屏障 | 显式管线屏障 |

### 6.2 栅栏与围栏（Fence）

&emsp;&emsp;栅栏（Fence）是最基本的同步原语，用于 CPU-GPU 同步和 GPU-GPU 同步。应用在命令队列中插入栅栏信号，然后等待或轮询栅栏状态。

#### 6.2.1 Direct3D 12 的围栏

&emsp;&emsp;DX12 使用 `ID3D12Fence` 接口，支持 CPU-GPU 同步和多队列同步。围栏有一个单调递增的计数器，GPU 完成指定操作后递增计数器，CPU 可以等待计数器达到特定值。

```cpp
ComPtr<ID3D12Fence> fence;
device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
uint64_t fenceValue = 0;

// 在命令队列中插入围栏信号（假设命令已提交）
commandQueue->Signal(fence.Get(), ++fenceValue);

// CPU 等待 GPU 完成
HANDLE eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
fence->SetEventOnCompletion(fenceValue, eventHandle);
WaitForSingleObject(eventHandle, INFINITE);
CloseHandle(eventHandle);
```

#### 6.2.2 Metal 的事件

&emsp;&emsp;Metal 使用 `MTLEvent` 接口，支持 CPU-GPU 同步和多队列同步。事件可以与命令缓冲区关联，当命令缓冲区完成时触发事件。

```swift
// CPU-GPU 同步
let commandBuffer = commandQueue.makeCommandBuffer()!
// 录制命令...
commandBuffer.commit()
commandBuffer.waitUntilCompleted() // CPU 等待 GPU 完成

// GPU-GPU 同步（跨命令缓冲区）
let event = device.makeEvent()!
let commandBuffer1 = commandQueue.makeCommandBuffer()!
let encoder1 = commandBuffer1.makeBlitCommandEncoder()!
encoder1.encodeSignalEvent(event, value: 1)
encoder1.endEncoding()
commandBuffer1.commit()

let commandBuffer2 = commandQueue.makeCommandBuffer()!
let encoder2 = commandBuffer2.makeBlitCommandEncoder()!
encoder2.encodeWaitForEvent(event, value: 1)
encoder2.endEncoding()
commandBuffer2.commit()
```

#### 6.2.3 Vulkan 的栅栏

&emsp;&emsp;Vulkan 使用 `VkFence` 接口，支持 CPU-GPU 同步。栅栏与队列提交关联，当队列中的所有命令执行完成后，驱动递增栅栏的信号值。

```cpp
// Vulkan: 栅栏同步
VkFence fence;
VkFenceCreateInfo fenceInfo = {};
vkCreateFence(device, &fenceInfo, nullptr, &fence);

// 提交命令并关联栅栏
vkQueueSubmit(queue, 1, &submitInfo, fence);

// CPU 等待 GPU 完成
vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
vkResetFences(device, 1, &fence);
```

### 6.3 信号量与事件（Semaphore & Event）

&emsp;&emsp;信号量（Semaphore）用于 GPU-GPU 同步，确保不同队列或不同命令缓冲区之间的执行顺序。事件（Event）用于更细粒度的同步，可以在命令流中插入同步点。

#### 6.3.1 Vulkan 的信号量

&emsp;&emsp;Vulkan 使用 `VkSemaphore` 接口，支持图形队列与计算队列、呈现队列之间的同步。信号量在队列提交时信号，在另一个队列等待。

```cpp
// Vulkan: 信号量同步
VkSemaphore imageAvailableSemaphore;
VkSemaphore renderFinishedSemaphore;

// 图形队列等待图像可用信号，然后信号渲染完成
VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
submitInfo.waitSemaphoreCount = 1;
submitInfo.pWaitSemaphores = &imageAvailableSemaphore;
submitInfo.pWaitDstStageMask = waitStages;
submitInfo.signalSemaphoreCount = 1;
submitInfo.pSignalSemaphores = &renderFinishedSemaphore;

vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);

// 呈现队列等待渲染完成信号
presentInfo.waitSemaphoreCount = 1;
presentInfo.pWaitSemaphores = &renderFinishedSemaphore;
vkQueuePresentKHR(presentQueue, &presentInfo);
```

#### 6.3.2 Vulkan 的事件

&emsp;&emsp;Vulkan 使用 `VkEvent` 接口，支持命令流内的细粒度同步。事件可以在命令缓冲区中设置和等待，用于依赖不同命令流的结果。

```cpp
// Vulkan: 事件同步
VkEvent event;
VkEventCreateInfo eventInfo = {};
vkCreateEvent(device, &eventInfo, nullptr, &event);

// 在命令缓冲区中设置事件
vkCmdSetEvent(commandBuffer, event, VK_PIPELINE_STAGE_TRANSFER_BIT);

// 在另一个命令缓冲区中等待事件
vkCmdWaitEvents(commandBuffer, 1, &event,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
    0, nullptr, 0, nullptr, 0, nullptr);
```

### 6.4 内存屏障与管线屏障

&emsp;&emsp;内存屏障（Memory Barrier）确保内存操作的顺序性和可见性。管线屏障（Pipeline Barrier）确保命令执行的顺序性，并可以插入内存屏障。

#### 6.4.1 Direct3D 12 的资源屏障

&emsp;&emsp;DX12 使用资源屏障（Resource Barrier）管理资源状态转换和内存依赖。屏障类型包括状态转换屏障、别名屏障和 UAV 屏障。

```cpp
// DX12: 资源屏障
D3D12_RESOURCE_BARRIER barrier = {};
barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
barrier.Transition.pResource = texture.Get();
barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

commandList->ResourceBarrier(1, &barrier);
```

#### 6.4.2 Vulkan 的管线屏障

&emsp;&emsp;Vulkan 使用管线屏障（Pipeline Barrier）管理命令执行顺序和内存依赖。屏障可以指定源阶段和目标阶段，以及内存依赖的访问类型。

```cpp
// Vulkan: 管线屏障
VkMemoryBarrier memoryBarrier = {};
memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

vkCmdPipelineBarrier(
    commandBuffer,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
```

#### 6.4.3 Metal 的 Hazard Tracking

&emsp;&emsp;Metal 使用 Hazard Tracking 机制自动检测资源访问冲突。驱动在后台跟踪资源使用状态，当检测到 Hazard 时自动插入屏障。应用也可以手动插入屏障以优化性能。

```swift
// Metal: 使用 MTLFence 手动同步
let fence = device.makeFence()!

let encoder1 = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
// 渲染到纹理...
encoder1.endEncoding()

// 在第二个编码器前等待第一个编码器的写入完成
let encoder2 = commandBuffer.makeBlitCommandEncoder()!
encoder2.waitForFence(fence, before: .fragment) // 假设后续使用片段着色器读取
encoder2.endEncoding()
```

### 6.5 资源状态转换

&emsp;&emsp;资源状态转换是同步的重要组成部分，确保资源在不同使用阶段处于正确的状态。不同 API 提供了不同粒度的状态管理。

| API | 状态粒度 | 状态转换方式 | 状态跟踪 |
| --- | -------- | ------------ | -------- |
| **OpenGL** | 无显式状态 | 驱动自动 | 驱动自动 |
| **Direct3D 11** | 无显式状态 | 驱动自动 | 驱动自动 |
| **Direct3D 12** | 资源整体或子资源 | 显式资源屏障 | 应用手动 |
| **Metal** | 资源整体 | Hazard Tracking 自动 + 手动选项 | 驱动自动 + 应用手动 |
| **Vulkan** | 资源整体或子资源 | 显式管线屏障 | 应用手动 |

### 6.6 跨平台 RHI 的同步策略

&emsp;&emsp;设计跨平台 RHI 时，同步抽象需要平衡各后端的差异。常见的策略包括：

#### 6.6.1 统一同步原语

&emsp;&emsp;RHI 可以定义统一的栅栏和信号量接口，映射到各后端的具体实现：

```cpp
class RHIFence {
public:
    virtual void Signal(uint64_t value) = 0;
    virtual void Wait(uint64_t value) = 0;
    virtual bool IsComplete(uint64_t value) = 0;
};

class RHISemaphore {
public:
    // 由驱动管理内部状态
};
```

#### 6.6.2 自动状态转换

&emsp;&emsp;RHI 可以维护资源状态表，自动插入必要的状态转换屏障：

```cpp
class RHIResource {
    RHIResourceState currentState;
public:
    void TransitionTo(RHICommandList& cmdList, RHIResourceState newState) {
        if (currentState != newState) {
            cmdList.InsertBarrier(this, currentState, newState);
            currentState = newState;
        }
    }
};
```

#### 6.6.3 延迟同步

&emsp;&emsp;RHI 可以延迟同步操作，批量处理多个屏障，减少 API 调用开销：

```cpp
class RHICommandList {
    std::vector<RHIBarrier> pendingBarriers;
public:
    void InsertBarrier(RHIResource* resource, RHIResourceState before, RHIResourceState after) {
        pendingBarriers.push_back({resource, before, after});
    }
    void FlushBarriers() {
        // 批量提交所有挂起的屏障
        pendingBarriers.clear();
    }
};
```

### 6.7 小结

&emsp;&emsp;同步机制是图形 API 中确保数据一致性和执行顺序的关键部分。OpenGL 和 DX11 使用隐式同步，而 DX12、Metal 和 Vulkan 提供显式同步原语。RHI 设计的关键是在这些差异之上建立统一的同步抽象，同时保留各后端的优化路径。自动状态转换、延迟同步和统一同步原语是工程实践中的标准解决方案。

## 7 管线状态与着色器架构

&emsp;&emsp;管线状态与着色器架构是图形 API 的核心部分，决定了渲染管线的配置和着色器的执行方式。旧一代 API 使用全局状态机模型，而新一代 API 则采用不可变的管线状态对象（PSO）模型。这种转变提高了性能，但也增加了复杂度。

### 7.1 管线状态模型对比

| API | 管线状态模型 | 状态切换开销 | 着色器绑定模型 |
| --- | ------------ | ------------ | -------------- |
| **OpenGL** | 全局状态机 | 高（驱动验证） | 随时绑定着色器程序 |
| **Direct3D 11** | 全局状态机 | 中（驱动验证） | 随时绑定着色器 |
| **Direct3D 12** | 管线状态对象（PSO） | 低（预编译） | 根签名 + 描述符堆 |
| **Metal** | 管线状态对象（PSO） | 低（预编译） | 参数缓冲区 |
| **Vulkan** | 管线状态对象（PSO） | 低（预编译） | 描述符集 + 管线布局 |

### 7.2 管线状态对象（PSO）

&emsp;&emsp;管线状态对象（Pipeline State Object, PSO）是现代图形 API 的核心概念。PSO 将所有管线状态（着色器、混合、深度模板、光栅化等）打包为一个不可变对象，驱动可以提前编译和优化。

#### 7.2.1 Direct3D 12 的 PSO

&emsp;&emsp;DX12 使用 `ID3D12PipelineState` 接口，包含图形 PSO 和计算 PSO。PSO 在创建时绑定所有着色器和状态，之后可以快速切换。

```cpp
// DX12: 创建图形 PSO
D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
psoDesc.InputLayout = { inputLayout, numInputLayoutElements };
psoDesc.pRootSignature = rootSignature.Get();
psoDesc.VS = { vertexShaderBlob->GetBufferPointer(), vertexShaderBlob->GetBufferSize() };
psoDesc.PS = { pixelShaderBlob->GetBufferPointer(), pixelShaderBlob->GetBufferSize() };
psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
psoDesc.SampleMask = UINT_MAX;
psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
psoDesc.NumRenderTargets = 1;
psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
psoDesc.SampleDesc.Count = 1;

ComPtr<ID3D12PipelineState> pso;
device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso));

// 绑定 PSO
commandList->SetPipelineState(pso.Get());
```

#### 7.2.2 Metal 的 PSO

&emsp;&emsp;Metal 使用 `MTLRenderPipelineState` 和 `MTLComputePipelineState` 接口。PSO 在创建时编译着色器，之后可以快速切换。

```swift
// Metal: 创建图形 PSO
let pipelineDescriptor = MTLRenderPipelineDescriptor()
pipelineDescriptor.vertexFunction = vertexFunction
pipelineDescriptor.fragmentFunction = fragmentFunction
pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

let pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)

// 绑定 PSO
renderEncoder.setRenderPipelineState(pipelineState)
```

#### 7.2.3 Vulkan 的 PSO

&emsp;&emsp;Vulkan 使用 `VkPipeline` 接口，包含图形管线和计算管线。PSO 在创建时指定所有状态和着色器，之后可以快速切换。

```cpp
// Vulkan: 创建图形 PSO
VkGraphicsPipelineCreateInfo pipelineInfo = {};
pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
pipelineInfo.stageCount = 2;
pipelineInfo.pStages = shaderStages;
pipelineInfo.pVertexInputState = &vertexInputInfo;
pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
pipelineInfo.pViewportState = &viewportInfo;
pipelineInfo.pRasterizationState = &rasterizerInfo;
pipelineInfo.pMultisampleState = &multisamplingInfo;
pipelineInfo.pDepthStencilState = &depthStencilInfo;
pipelineInfo.pColorBlendState = &colorBlendingInfo;
pipelineInfo.layout = pipelineLayout;
pipelineInfo.renderPass = renderPass;

VkPipeline graphicsPipeline;
vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline);

// 绑定 PSO
vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
```

### 7.3 着色器语言与编译

&emsp;&emsp;着色器语言和编译模型是 API 差异的重要方面。不同 API 使用不同的着色器语言和编译流程。

| API | 着色器语言 | 编译格式 | 编译时机 | 跨平台支持 |
| --- | ---------- | -------- | -------- |------------|
| **OpenGL** | GLSL | GLSL 源码 | 运行时编译 | 有限（需驱动支持） |
| **OpenGL ES** | ESSL | ESSL 源码 | 运行时编译 | 有限（需驱动支持） |
| **Direct3D 11** | HLSL | DXBC（字节码） | 离线编译 | 仅 Windows/Xbox |
| **Direct3D 12** | HLSL | DXIL（LLVM IR） | 离线编译 | 仅 Windows/Xbox |
| **Metal** | MSL | Metal Library（LLVM IR） | 离线编译 | 仅 Apple 平台 |
| **Vulkan** | GLSL/HLSL | SPIR-V（二进制） | 离线编译 | 跨平台 |

#### 7.3.1 SPIR-V 的跨平台优势

&emsp;&emsp;SPIR-V 是 Vulkan 的标准着色器格式，也是 OpenGL 的可选格式。它是一种紧凑的二进制格式，可以离线编译，避免运行时编译的开销和兼容性问题。

```cpp
// Vulkan: 加载 SPIR-V 着色器
std::vector<char> vertexShaderCode = ReadFile("vertex.spv");
VkShaderModuleCreateInfo createInfo = {};
createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
createInfo.codeSize = vertexShaderCode.size();
createInfo.pCode = reinterpret_cast<const uint32_t*>(vertexShaderCode.data());

VkShaderModule vertexShaderModule;
vkCreateShaderModule(device, &createInfo, nullptr, &vertexShaderModule);
```

#### 7.3.2 着色器交叉编译

&emsp;&emsp;跨平台 RHI 通常需要支持多种着色器语言。工具链如 SPIRV-Cross 可以将 SPIR-V 转换为 HLSL、MSL 或 GLSL，实现一次编写多平台编译。

```cpp
// SPIRV-Cross: 着色器交叉编译
spirv_cross::CompilerGLSL glsl(std::move(spirvBinary));
spirv_cross::CompilerGLSL::Options options;
options.version = 330;
options.es = false;
glsl.set_common_options(options);

std::string glslSource = glsl.compile();
```

### 7.4 绑定模型

&emsp;&emsp;绑定模型决定了着色器如何访问资源（缓冲区、纹理、采样器）。不同 API 提供了不同粒度的绑定模型。

#### 7.4.1 OpenGL 的全局绑定

&emsp;&emsp;OpenGL 使用全局状态机，资源绑定通过全局函数调用。每次绘制前需要重新绑定所有资源，导致开销。

```cpp
// OpenGL: 全局绑定
glUseProgram(shaderProgram);
glBindVertexArray(vao);
glBindTexture(GL_TEXTURE_2D, texture);
glDrawArrays(GL_TRIANGLES, 0, 3);
```

#### 7.4.2 Direct3D 11 的常量缓冲区

&emsp;&emsp;DX11 使用常量缓冲区（Constant Buffer）绑定资源，支持按槽位绑定顶点着色器、像素着色器等。

```cpp
// DX11: 常量缓冲区绑定
ID3D11Buffer* constantBuffers[] = { constantBuffer };
deviceContext->VSSetConstantBuffers(0, 1, constantBuffers);
deviceContext->PSSetShaderResources(0, 1, &shaderResourceView);
deviceContext->PSSetSamplers(0, 1, &samplerState);
```

#### 7.4.3 Direct3D 12 的根签名

&emsp;&emsp;DX12 使用根签名（Root Signature）定义管线使用的资源布局。根签名包含根参数（根描述符、根常量、描述符表），描述符堆包含实际的描述符。

```cpp
// DX12: 根签名绑定
// 创建根签名
CD3DX12_ROOT_PARAMETER1 rootParameters[1];
rootParameters[0].InitAsDescriptorTable(1, &descriptorRange, D3D12_SHADER_VISIBILITY_ALL);

ComPtr<ID3D12RootSignature> rootSignature;
// ... 创建根签名

// 绑定描述符表
commandList->SetGraphicsRootDescriptorTable(0, descriptorHeap->GetGPUDescriptorHandleForHeapStart());
```

#### 7.4.4 Metal 的参数缓冲区

&emsp;&emsp;Metal 使用参数缓冲区（Argument Buffer）绑定资源，支持在单个缓冲区中打包多个资源引用。

```swift
// Metal: 参数缓冲区绑定
let argumentEncoder = device.makeArgumentEncoder(descriptor: argumentDescriptor)
argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
argumentEncoder.setTexture(texture, at: 0)
argumentEncoder.setSamplerState(samplerState, at: 1)

renderEncoder.setVertexBuffer(argumentBuffer, offset: 0, at: 0)
renderEncoder.setFragmentBuffer(argumentBuffer, offset: 0, at: 0)
```

#### 7.4.5 Vulkan 的描述符集

&emsp;&emsp;Vulkan 使用描述符集（Descriptor Set）绑定资源，描述符集布局定义资源类型和数量，描述符池分配描述符集。

```cpp
// Vulkan: 描述符集绑定
// 创建描述符集布局
VkDescriptorSetLayoutBinding binding = {};
binding.binding = 0;
binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
binding.descriptorCount = 1;
binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

VkDescriptorSetLayoutCreateInfo layoutInfo = {};
layoutInfo.bindingCount = 1;
layoutInfo.pBindings = &binding;

VkDescriptorSetLayout descriptorSetLayout;
vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);

// 分配描述符集
VkDescriptorSetAllocateInfo allocInfo = {};
allocInfo.descriptorPool = descriptorPool;
allocInfo.descriptorSetCount = 1;
allocInfo.pSetLayouts = &descriptorSetLayout;

VkDescriptorSet descriptorSet;
vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);

// 更新描述符集
VkDescriptorImageInfo imageInfo = {};
imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
imageInfo.imageView = textureView;
imageInfo.sampler = sampler;

VkWriteDescriptorSet descriptorWrite = {};
descriptorWrite.dstSet = descriptorSet;
descriptorWrite.dstBinding = 0;
descriptorWrite.descriptorCount = 1;
descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
descriptorWrite.pImageInfo = &imageInfo;

vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);

// 绑定描述符集
vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
    pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
```

### 7.5 跨平台 RHI 的管线状态策略

&emsp;&emsp;设计跨平台 RHI 时，管线状态抽象需要平衡各后端的差异。常见的策略包括：

#### 7.5.1 统一 PSO 接口

&emsp;&emsp;RHI 可以定义统一的 PSO 接口，映射到各后端的具体实现：

```cpp
class RHIPipelineState {
public:
    virtual void Bind(RHICommandList& cmdList) = 0;
};

class RHIGraphicsPipelineState : public RHIPipelineState {
public:
    // 图形管线状态
};

class RHIComputePipelineState : public RHIPipelineState {
public:
    // 计算管线状态
};
```

#### 7.5.2 统一绑定接口

&emsp;&emsp;RHI 可以定义统一的绑定接口，隐藏各后端的绑定模型差异：

```cpp
class RHIDescriptorSet {
public:
    virtual void SetBuffer(RHIBuffer* buffer, uint32_t binding) = 0;
    virtual void SetTexture(RHITexture* texture, uint32_t binding) = 0;
    virtual void SetSampler(RHISampler* sampler, uint32_t binding) = 0;
    virtual void Bind(RHICommandList& cmdList, RHIPipelineStage stage) = 0;
};
```

#### 7.5.3 着色器交叉编译

&emsp;&emsp;RHI 可以集成 SPIRV-Cross 等工具，实现 SPIR-V 到各后端着色器语言的自动转换：

```cpp
class RHIShaderCompiler {
public:
    static RHIShader Compile(const std::string& source, RHIShaderStage stage);
    static RHIShader CrossCompile(const std::string& spirvSource, RHIPlatform platform);
};
```

### 7.6 小结

&emsp;&emsp;管线状态与着色器架构是图形 API 的核心部分，决定了渲染管线的配置和着色器的执行方式。OpenGL 和 DX11 使用全局状态机，而 DX12、Metal 和 Vulkan 采用不可变的 PSO 模型。RHI 设计的关键是在这些差异之上建立统一的管线状态抽象，同时保留各后端的优化路径。统一 PSO 接口、统一绑定接口和着色器交叉编译是工程实践中的标准解决方案。

## 8 渲染目标与暂存区管理

&emsp;&emsp;渲染目标与暂存区管理是图形 API 中处理帧缓冲、渲染通道和交换链的关键部分。旧一代 API 使用隐式的帧缓冲管理，而新一代 API 则提供了显式的渲染通道和附件管理。这种转变提高了性能，但也增加了复杂度。

### 8.1 渲染目标模型对比

| API | 渲染目标模型 | 渲染通道支持 | 交换链管理 | 加载/存储操作 |
| --- | ------------ | ------------ |------------| -------------- |
| **OpenGL** | 帧缓冲对象（FBO） | 隐式（通过 FBO） | 驱动管理 | 隐式 |
| **Direct3D 11** | 渲染目标视图（RTV） | 隐式（通过 OM 阶段） | 驱动管理 | 隐式 |
| **Direct3D 12** | 渲染目标描述符 | 显式渲染通道 | 应用管理 | 显式 |
| **Metal** | 渲染通道描述符 | 显式渲染通道 | 应用管理 | 显式 |
| **Vulkan** | 帧缓冲附件 | 显式渲染通道 | 应用管理 | 显式 |

### 8.2 帧缓冲与渲染通道

#### 8.2.1 OpenGL 的帧缓冲对象

&emsp;&emsp;OpenGL 使用帧缓冲对象（Frame Buffer Object, FBO）管理渲染目标。FBO 可以附加颜色、深度和模板附件，支持渲染到纹理。

```cpp
// OpenGL: 帧缓冲对象
GLuint fbo;
glGenFramebuffers(1, &fbo);
glBindFramebuffer(GL_FRAMEBUFFER, fbo);

// 附加颜色附件
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

// 附加深度附件
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);

// 检查完整性
if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    // 处理错误
}
```

#### 8.2.2 Direct3D 12 的渲染通道

&emsp;&emsp;DX12 使用显式渲染通道（Render Pass）管理渲染目标。渲染通道定义附件的加载/存储操作，以及子通道之间的依赖关系。

```cpp
// DX12: 渲染通道（通过根签名和管线状态隐式定义）
// DX12 通过 BeginRenderPass/EndRenderPass 提供显式渲染通道，用于声明附件的加载/存储操作和优化 tile 内存使用。
D3D12_RENDER_PASS_RENDER_TARGET_DESC renderTarget = {};
renderTarget.cpuDescriptor = rtvHandle;
renderTarget.BeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
renderTarget.BeginningAccess.Clear.ClearValue.Color[0] = 0.0f;
renderTarget.EndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;

commandList->BeginRenderPass(1, &renderTarget, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
// ... 渲染命令
commandList->EndRenderPass();
```

#### 8.2.3 Metal 的渲染通道

&emsp;&emsp;Metal 使用显式渲染通道描述符（Render Pass Descriptor）管理渲染目标。渲染通道描述符定义附件的加载/存储操作和存储模式。

```swift
// Metal: 渲染通道描述符
let renderPassDescriptor = MTLRenderPassDescriptor()
renderPassDescriptor.colorAttachments[0].texture = renderTargetTexture
renderPassDescriptor.colorAttachments[0].loadAction = .clear
renderPassDescriptor.colorAttachments[0].storeAction = .store
renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
renderPassDescriptor.depthAttachment.texture = depthTexture
renderPassDescriptor.depthAttachment.loadAction = .clear
renderPassDescriptor.depthAttachment.storeAction = .dontCare

let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
// ... 渲染命令
renderEncoder.endEncoding()
```

#### 8.2.4 Vulkan 的帧缓冲与渲染通道

&emsp;&emsp;Vulkan 使用显式帧缓冲（Framebuffer）和渲染通道（Render Pass）管理渲染目标。渲染通道定义附件的加载/存储操作和子通道依赖，帧缓冲绑定实际的附件图像。

```cpp
// Vulkan: 渲染通道与帧缓冲
// 创建渲染通道
VkAttachmentDescription colorAttachment = {};
colorAttachment.sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION;
colorAttachment.format = VK_FORMAT_B8G8R8A8_UNORM;
colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

VkAttachmentReference colorAttachmentRef = {};
colorAttachmentRef.attachment = 0;
colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

VkSubpassDescription subpass = {};
subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
subpass.colorAttachmentCount = 1;
subpass.pColorAttachments = &colorAttachmentRef;

VkRenderPassCreateInfo renderPassInfo = {};
renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
renderPassInfo.attachmentCount = 1;
renderPassInfo.pAttachments = &colorAttachment;
renderPassInfo.subpassCount = 1;
renderPassInfo.pSubpasses = &subpass;

VkRenderPass renderPass;
vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass);

// 创建帧缓冲
VkFramebufferCreateInfo framebufferInfo = {};
framebufferInfo.renderPass = renderPass;
framebufferInfo.attachmentCount = 1;
framebufferInfo.pAttachments = &swapchainImageView;
framebufferInfo.width = swapchainExtent.width;
framebufferInfo.height = swapchainExtent.height;
framebufferInfo.layers = 1;

VkFramebuffer framebuffer;
vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffer);

// 渲染通道内渲染
VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
VkRenderPassBeginInfo renderPassBeginInfo = {};
renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
renderPassBeginInfo.renderPass = renderPass;
renderPassBeginInfo.framebuffer = framebuffer;
renderPassBeginInfo.renderArea.extent = swapchainExtent;
renderPassBeginInfo.clearValueCount = 1;
renderPassBeginInfo.pClearValues = &clearColor;

vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
// ... 渲染命令
vkCmdEndRenderPass(commandBuffer);
```

### 8.3 交换链管理

&emsp;&emsp;交换链（Swap Chain）管理屏幕呈现，协调 CPU 和 GPU 的帧同步。不同 API 提供了不同粒度的交换链控制。

| API | 交换链管理 | 呈现模式 | 垂直同步控制 |
| --- |------------|----------|--------------|
| **OpenGL** | 驱动管理（`glfwSwapBuffers`） | 隐式双缓冲 | 驱动控制 |
| **Direct3D 11** | 驱动管理（`IDXGISwapChain`） | 显式双缓冲 | 驱动控制 |
| **Direct3D 12** | 应用管理（`IDXGISwapChain3`） | 显式多缓冲 | 应用控制 |
| **Metal** | 应用管理（`CAMetalLayer`） | 显式三缓冲 | 应用控制 |
| **Vulkan** | 应用管理（`VkSwapchainKHR`） | 显式多缓冲 | 应用控制 |

#### 8.3.1 Vulkan 的交换链

&emsp;&emsp;Vulkan 的交换链管理最为复杂，需要应用显式创建交换链、获取图像、管理呈现同步。

```cpp
// Vulkan: 交换链管理
// 创建交换链
VkSwapchainCreateInfoKHR swapchainInfo = {};
swapchainInfo.surface = surface;
swapchainInfo.minImageCount = 3;
swapchainInfo.imageFormat = surfaceFormat.format;
swapchainInfo.imageExtent = extent;
swapchainInfo.imageArrayLayers = 1;
swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
swapchainInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
swapchainInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;

VkSwapchainKHR swapchain;
vkCreateSwapchainKHR(device, &swapchainInfo, nullptr, &swapchain);

// 获取交换链图像
std::vector<VkImage> swapchainImages;
vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());

// 呈现图像
uint32_t imageIndex;
vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

VkPresentInfoKHR presentInfo = {};
presentInfo.swapchainCount = 1;
presentInfo.pSwapchains = &swapchain;
presentInfo.pImageIndices = &imageIndex;
presentInfo.pWaitSemaphores = &renderFinishedSemaphore;

vkQueuePresentKHR(presentQueue, &presentInfo);
```

### 8.4 暂存区与加载/存储操作

&emsp;&emsp;暂存区（Staging）管理渲染目标的加载（Load）和存储（Store）操作，决定附件内容在渲染通道前后的处理方式。

| API | 加载操作 | 存储操作 | 暂存策略 |
| --- |----------|----------|----------|
| **OpenGL** | 隐式（`glClear`） | 隐式（自动保存） | 驱动管理 |
| **Direct3D 11** | 隐式（`ClearRenderTargetView`） | 隐式（自动保存） | 驱动管理 |
| **Direct3D 12** | 显式（`D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_*`） | 显式（`D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_*`） | 应用控制 |
| **Metal** | 显式（`MTLLoadAction*`） | 显式（`MTLStoreAction*`） | 应用控制 |
| **Vulkan** | 显式（`VK_ATTACHMENT_LOAD_OP_*`） | 显式（`VK_ATTACHMENT_STORE_OP_*`） | 应用控制 |

#### 8.4.1 优化加载/存储操作

&emsp;&emsp;显式加载/存储操作允许应用优化渲染目标的内存访问。常见的优化策略包括：

- **清除（Clear）**：用固定值填充附件，丢弃原有内容。
- **保持（Preserve）**：保留附件原有内容，用于后续渲染通道。
- **放弃（Don't Care）**：不关心附件原有内容，通常用于临时渲染目标。
- **存储（Store）**：保存渲染结果，用于后续采样或呈现。

### 8.5 跨平台 RHI 的渲染目标策略

&emsp;&emsp;设计跨平台 RHI 时，渲染目标抽象需要平衡各后端的差异。常见的策略包括：

#### 8.5.1 统一渲染通道接口

&emsp;&emsp;RHI 可以定义统一的渲染通道接口，映射到各后端的具体实现：

```cpp
class RHIRenderPass {
public:
    virtual void Begin(RHICommandList& cmdList) = 0;
    virtual void End(RHICommandList& cmdList) = 0;
    virtual void SetClearColor(float r, float g, float b, float a) = 0;
};
```

#### 8.5.2 统一帧缓冲接口

&emsp;&emsp;RHI 可以定义统一的帧缓冲接口，隐藏各后端的附件管理差异：

```cpp
class RHIFramebuffer {
public:
    virtual void AttachColor(uint32_t index, RHITexture* texture) = 0;
    virtual void AttachDepth(RHITexture* texture) = 0;
    virtual void Validate() = 0;
};
```

#### 8.5.3 统一交换链接口

&emsp;&emsp;RHI 可以定义统一的交换链接口，管理屏幕呈现：

```cpp
class RHISwapChain {
public:
    virtual RHITexture* GetCurrentImage() = 0;
    virtual void Present() = 0;
    virtual void Resize(uint32_t width, uint32_t height) = 0;
};
```

### 8.6 小结

&emsp;&emsp;渲染目标与暂存区管理是图形 API 中处理帧缓冲、渲染通道和交换链的关键部分。OpenGL 和 DX11 使用隐式管理，而 DX12、Metal 和 Vulkan 提供显式控制。RHI 设计的关键是在这些差异之上建立统一的渲染目标抽象，同时保留各后端的优化路径。统一渲染通道接口、统一帧缓冲接口和统一交换链接口是工程实践中的标准解决方案。

## 9 小结

&emsp;&emsp;本文系统对比了六大主流图形 API（OpenGL、OpenGL ES、Direct3D 11、Direct3D 12、Metal、Vulkan）在资源管理、多线程、同步机制、管线状态、着色器架构、渲染目标等多个维度的差异，并分析了这些差异背后的硬件与历史考量。

### 9.1 核心差异总结

&emsp;&emsp;六大 API 的差异可以压缩成一句话：**责任在谁手里**。GL/GLES/DX11 把资源、同步、内存交给驱动，换来简单；DX12/Metal/Vulkan 把这些责任交还给应用，换来性能与可控。

| 维度 | 驱动托管 API（OpenGL/DX11） | 显式控制 API（DX12/Metal/Vulkan） |
|------|---------------------------|----------------------------------|
| **资源管理** | 驱动分配、驻留、回收 | 应用创建堆、管理内存、控制驻留 |
| **多线程** | 有限或不支持 | 原生多线程命令录制 |
| **同步** | 隐式同步 | 显式栅栏、信号量、屏障 |
| **管线状态** | 全局状态机，高切换开销 | 不可变 PSO，低切换开销 |
| **着色器** | 运行时编译，跨平台有限 | 离线编译，SPIR-V 跨平台 |
| **渲染目标** | 隐式帧缓冲管理 | 显式渲染通道和附件管理 |

### 9.2 RHI 设计的关键原则

![](https://docs.godotengine.org/en/stable/_images/renderers_rendering_layers.webp)

&emsp;&emsp;设计跨平台 RHI 时，需要遵循以下关键原则：

1. **接口形状像最显式的 API**：以 Vulkan/DX12 为基准设计接口，然后向旧 API 降级仿真。
2. **能力分级**：根据后端能力提供不同级别的功能，确保旧硬件也能运行。
3. **统一绑定模型**：抽象描述符集、根签名、参数缓冲区等差异，提供统一的资源绑定接口。
4. **显式同步抽象**：隐藏栅栏、信号量、屏障的差异，提供统一的同步原语。
5. **着色器交叉编译**：集成 SPIRV-Cross 等工具，实现 SPIR-V 到各后端着色器语言的自动转换。
6. **零开销热路径**：确保关键渲染路径没有额外抽象开销。
7. **跨后端测试**：建立完整的测试套件，确保各后端行为一致。

### 9.3 工程实践建议

&emsp;&emsp;在实际工程中，需要根据自己的场景考虑如何实现RHI，比如对于高性能场景可能薄薄的封装一层比较合适，而对于需要多平台支持的场景略微牺牲性能所有render API都对上层透明可能更好：

- **内存管理**：使用子分配器（如 VMA、D3D12MA）减少显存分配次数，支持延迟删除和资源别名。
- **多线程架构**：采用线程本地存储和任务队列，平衡各后端的线程模型差异。
- **同步策略**：实现自动状态转换和延迟同步，减少显式同步的复杂度。
- **管线状态缓存**：缓存编译后的 PSO，避免重复创建和编译开销。
- **渲染通道抽象**：统一渲染通道和帧缓冲接口，支持移动端的 TBDR 优化。


### 9.4 结语

&emsp;&emsp;图形 API 的差异不是随意演化的，而是历史包袱、目标硬件（IMR vs TBDR、统一内存 vs 离散显存）与成本模型共同作用的产物。也正因为如此，设计 RHI 的第一原则才显得顺理成章：**接口的形状应该像最显式的那个 API，然后向老 API 降级仿真**——再加上能力分级、统一绑定、显式同步、着色器交叉编译这几根支柱，以及零开销热路径和跨后端测试这两条纪律。做到这些，一套代码跑遍六家后端就不再是玄学，而是工程。

&emsp;&emsp;理解这些差异不仅是为了写一个跨平台渲染器，更是为了理解 GPU 的工作方式本身。当你深入理解了这些 API 的设计哲学，你就能更好地利用硬件能力，创造出更高效、更精美的图形应用。
