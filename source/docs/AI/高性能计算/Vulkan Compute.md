# Vulkan Compute 

## 1 Vulkan
### 1.1 Vulkan简介
&emsp;&emsp;Vulkan 是由 Khronos 主导开发的跨平台图形与计算API，于2016年2月正式发布1.0版本，旨在解决传统图形API（如 OpenGL）在高性能、低开销、多线程支持等方面的局限性。Vulkan 的核心定位是“跨平台、显式控制、低开销的统一图形与计算API”，打破了传统图形API与计算API分离的格局，让图形渲染与并行计算能够共用一套底层资源模型与调度体系，大幅提升协同效率。
&emsp;&emsp;Vulkan 的设计围绕“显式控制、低开销、可预测性与可扩展性”展开，其核心思想是将传统图形 API 中由驱动隐式处理的行为全部上移至应用层，从而实现对 GPU 的精细化控制。
- **完全显式化设计**：资源管理、内存分配、同步机制与状态转换均由开发者显式控制，驱动不再进行隐式状态推导与同步；
- **接近零抽象开销（Zero-overhead abstraction）**：通过管线状态对象（PSO）预编译、命令缓冲区复用及多线程命令记录机制，显著降低 CPU 端开销，使性能瓶颈从 API 层转移至应用设计；
- **统一资源与执行模型**：图形与计算共享统一的资源抽象（Buffer/Image/Memory）与队列体系，为高效协同执行提供基础；
- **规范驱动的一致性**：通过严格定义的 API 行为模型减少驱动差异，但仍需针对不同硬件特性进行适配；
- **可扩展架构**：基于扩展机制与特性查询体系，使新硬件能力能够在保持核心 API 稳定的前提下持续演进。

### 1.2 Vulkan vs OpenCL vs OpenGL
&emsp;&emsp;在深入理解Vulkan之间，这里先简单对比下Vlukan和OpenGL/OpenCL之间的区别，以方便本身了解这两个API的读者更加容易理解Vulkan。

&emsp;&emsp;OpenGL 是一个典型的**全局状态机模型**：所有渲染操作都依赖上下文中隐式维护的状态（如绑定的缓冲、着色器、纹理等），绘制指令本身不携带完整信息，这使其易于上手但也带来状态耦合强、行为不透明和性能不可控等问题；相比之下，Vulkan 采用**显式状态与命令缓冲模型**，所有资源绑定、同步与管线状态都需要开发者明确指定并预先记录到 Command Buffer 中，从而消除了隐式状态带来的不确定性，实现更高的性能、可预测性以及多线程扩展能力。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/1cdaa6f0f3c60944d0348f268922c5b2.png)


![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/a6816000336d73e6e665f307d9e13273.png)

&emsp;&emsp;OpenCL核心模型围绕**设备抽象（Device）、上下文（Context）、命令队列（Command Queue）与内核（Kernel）执行**展开，开发者通过显式地管理内存对象（Buffer/Image）、数据传输以及 Kernel 调度来驱动计算任务，强调数据并行与硬件抽象；相比之下，Vulkan 虽然同样提供计算能力（Compute Pipeline），但其设计更偏向**统一图形与计算的低开销显式 API**，在资源绑定（Descriptor）、同步（Pipeline Barrier）和命令录制（Command Buffer）层面提供更细粒度控制，能够将计算与图形任务高效融合，并在多线程与性能可预测性方面优于 OpenCL，但代价是开发复杂度更高、抽象层更底层。

| 维度        | Vulkan                   | OpenGL             | OpenCL                 |
| --------- | ------------------------ | ------------------ | ---------------------- |
| **核心定位**  | 图形 + 计算统一 API            | 图形渲染 API           | 通用并行计算 API             |
| **执行模型**  | Command Buffer（预录制 + 提交） | 即时调用（全局状态驱动）       | Kernel + Command Queue |
| **状态管理**  | 完全显式（无隐式状态）              | 全局状态机（强隐式）         | 显式（以内核和内存为中心）          |
| **控制粒度**  | 极细（同步 / 内存 / 调度全可控）      | 粗（依赖驱动）            | 中（计算调度可控）              |
| **多线程能力** | 强（原生支持并行录制）              | 弱（Context 限制）      | 中（多队列执行）               |
| **图形能力**  | 完整支持                     | 完整支持               | 不支持                    |
| **计算能力**  | 强（Compute Pipeline）      | 较弱（Compute Shader） | 核心能力                   |
| **典型特点**  | 高性能、可预测、复杂度高             | 易用、状态隐式、性能不稳定      | 专注数据并行、跨设备抽象           |


## 2 Vulkan Compute Model
&emsp;&emsp;Vulkan通过一套统一的执行与资源模型（Pipeline + Descriptor + CommandBuffer + Queue），将图形管线和计算管线抽象为并列的两种执行形式：图形管线是包含固定功能阶段的多阶段流水线，而计算管线则是更通用的单阶段并行执行模型；二者共享相同的资源绑定、调度与同步机制，从而实现数据在 GPU 内的无缝流动与协同执行。
![](https://vulkan-tutorial.com/images/vulkan_pipeline_block_diagram.png)

&emsp;&emsp;Vulkan Compute 的管线是相对图形管线独立的，开发人员可以根据具体应用场景灵活构建纯计算流程或与图形流程协同的混合 Pipeline，而无需依赖固定功能阶段或渲染流程；这种独立性体现在 Compute Pipeline 仅由 Shader 与资源绑定（Descriptor）驱动，通过 Command Buffer 显式调度执行，并可自由选择队列、同步策略与内存访问方式，从而实现从单一数据并行计算到复杂多阶段 GPU 数据流水线的高效构建，同时保持对执行顺序、资源可见性以及性能行为的完全可控。

&emsp;&emsp;下面将围绕对象模型、资源模型、执行模型和同步模型四个模型理解Vulkan Compute的底层架构设计。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/336afe31a56bd3b7f71e7a9ffbb416cd.png)

### 2.1 对象模型
&emsp;&emsp;Vulkan将所有功能模型封装成强类型对象，比如VkInstance,VkDevice等。通过这些模型来显示描述和执行GPU的能力，资源，资源绑定和执行，并且所有的对象都要显示的配置和管理。相对于OPenGL通过索引id管理对象更加直观明了，与OpenCL相比更加精细，对象之间分工明确。

&emsp;&emsp;Vulkan的对象按照功能分类可分为：
* **顶层对象：`Instance`。**

  * `Instance` 是 Vulkan 的全局入口对象，主要负责建立应用程序与底层驱动之间的连接，并完成 Vulkan 运行时的初始化工作，包括扩展（Extensions）加载、验证层（Validation Layers）启用以及物理设备（PhysicalDevice）枚举。
  * 需要注意的是，`Instance` 并不承担资源管理或计算执行职责，其功能更接近于一个**全局运行时入口（runtime entry point）**；相比之下，OpenCL 中的 context 更接近 Vulkan 中的 `Device`，因为它直接关联设备资源与执行环境。
  * 此外，与 OpenGL context 不同，Vulkan 的 `Instance` 不包含任何隐式状态，也不与线程或渲染目标绑定，因此通常一个应用仅需一个 `Instance` 即可满足需求。

---

* **设备对象：`PhysicalDevice`、`Device`、`Queue`。**

  * `PhysicalDevice` 表示实际的 GPU 硬件，用于查询设备能力与属性，例如支持的队列类型、内存类型以及硬件限制等，本身不参与实际执行。
  * `Device`（逻辑设备）是基于 `PhysicalDevice` 创建的核心对象，代表应用与 GPU 之间的“执行契约”，负责资源创建（如 Buffer、Image）、内存管理以及管线构建等，是所有计算操作的基础。
  * `Queue` 是 `Device` 中的执行单元，用于提交并执行命令缓冲区（CommandBuffer）。不同类型的队列（如 Compute、Graphics、Transfer）对应 GPU 上不同的执行能力，并支持多队列并发执行（如异步计算）。
  * 三者构成从“硬件能力描述 → 使用方式定义 → 实际执行”的完整链路。

---

* **资源对象：`Buffer`、`Image`、`DeviceMemory`、`BufferView` / `ImageView`、`DescriptorSet`、`DescriptorPool`、`DescriptorSetLayout`。**

  * 资源对象负责数据的存储、组织与访问，是计算任务的数据基础。
  * `Buffer` 与 `Image` 分别表示线性数据与结构化数据，而 `DeviceMemory` 则是实际的物理内存分配单元；二者相互解耦，资源必须显式绑定内存后才能使用。
  * `BufferView` / `ImageView` 用于定义资源的访问视图，使同一资源可以以不同格式或子区域被访问。
  * `DescriptorSet` 及其相关对象（Layout / Pool）用于将资源绑定到 Shader，使 GPU 在执行时能够访问对应的数据。
  * 整体上，资源对象定义了“数据是什么以及如何被访问”。
  
---

* **管线与命令对象：`ShaderModule`、`PipelineLayout`、`ComputePipeline`、`CommandPool`、`CommandBuffer`。**
  * `ShaderModule` 表示以 SPIR-V 格式编译后的计算程序，是 GPU 执行逻辑的核心。
  * `PipelineLayout` 定义 Shader 所需的资源接口（如 DescriptorSet 布局和 Push Constant），相当于 Shader 与外部资源之间的“接口契约”。
  * `ComputePipeline` 封装计算执行所需的完整状态（Shader + 资源布局），通常为不可变对象，创建成本较高但执行效率高。
  * `CommandPool` 用于管理命令缓冲区的内存分配，而 `CommandBuffer` 则负责记录具体的 GPU 指令（如绑定管线、分发计算任务等）。

&emsp;&emsp;对象的创建和管理都需要显式地进行，特别是在使用完成后必须调用对应的销毁函数进行释放，否则会导致显存或系统资源泄漏。此外，不同对象之间存在严格的层级与依赖关系，这种关系构成了 Vulkan 对象模型的基础约束：所有对象均由其上级对象创建（如 Device 依赖 Instance，资源对象与管线对象依赖 Device，CommandBuffer 依赖 CommandPool），并且在生命周期管理上必须遵循“先创建父对象、再创建子对象；销毁时先销毁子对象、再销毁父对象”的原则。

&emsp;&emsp;对象创建的核心流程：
  1.  创建 Instance（顶层对象）→ 
  2.  枚举 PhysicalDevice（获取 GPU 硬件）→ 
  3.  创建 Device（逻辑设备）和 Compute Queue → 
  4.  创建 CommandPool（命令池）→ 
  5.  分配 CommandBuffer（命令缓冲区）→ 
  6.  创建资源对象和管线对象 → 
  7.  执行计算 → 
  8.  按依赖顺序销毁所有对象。

&emsp;&emsp;这种显式管理机制不仅体现在对象的创建与销毁上，还体现在资源绑定、内存分配与同步控制等多个方面——例如 Buffer 与 Image 需要手动绑定 DeviceMemory，DescriptorSet 需要从 DescriptorPool 分配，命令必须通过 CommandBuffer 录制并提交到 Queue 执行。Vulkan 不提供任何隐式资源管理或自动生命周期控制，所有行为均由开发者明确指定。

&emsp;&emsp;这种设计虽然增加了开发复杂度，但带来的好处是完全可预测的性能行为与资源控制能力：开发者可以精确掌控内存分配策略、对象复用方式以及命令调度顺序，从而避免传统 API 中由于隐式管理带来的性能波动与不可控开销。

### 2.2 资源模型
&emsp;&emsp;资源模型是 Vulkan 中用于描述**数据如何存储、组织以及被 Shader 访问**的一整套机制。与 OpenGL 或早期 CUDA 等提供高度抽象与隐式内存管理的 API 不同，Vulkan 取消了底层的自动数据迁移与依赖推导。在 Vulkan 资源模型中，物理内存分配、逻辑视图映射以及缓存同步被彻底解耦，这为实现 GPU 硬件利用率最大化提供了必要的控制接口。

&emsp;&emsp;Vulkan 的资源管理体系主要依托于三个底层抽象：物理内存 (Device Memory)、逻辑资源 (Buffers/Images) 以及描述符映射 (Descriptors)。


### 2.3 执行模型
&emsp;&emsp;

### 2.4 同步模型
&emsp;&emsp;