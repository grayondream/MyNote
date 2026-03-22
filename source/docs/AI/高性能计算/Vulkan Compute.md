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

&emsp;&emsp;下面将围绕对象模型、资源模型、执行模型三个模型理解Vulkan Compute的底层架构设计。

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

---

**物理内存（Device Memory）**

&emsp;&emsp;物理内存是GPU可直接寻址的真实显存或主机可见内存，是Vulkan资源真正占用的存储载体。Vulkan不提供隐式内存分配，所有内存必须由应用显式申请、绑定、释放。驱动会向应用暴露多个内存堆（Memory Heap）与内存类型（Memory Type），分别对应显存容量、CPU 可访问性、缓存策略等属性。应用必须通过查询设备内存属性，选择满足使用场景的内存类型：
- **DEVICE_LOCAL**：仅GPU 可高速访问，适合常驻 GPU 的渲染数据；
- **HOST_VISIBLE**：CPU可映射读写，常用于上传 / 下载数据；
- **HOST_COHERENT**：无需显式刷新缓存，保证 CPU/GPU 视图一致；
- **HOST_CACHED**：CPU侧启用缓存，提升读效率但可能需要显式同步。
>Vulkan 允许将多个 Buffer 和 Image 放置在同一块物理内存中（通过 Offset），从而极大减少了内存碎片的产生；而 OpenCL 的资源通常是独立的内存块。
---

**逻辑资源（Buffer / Image）**
&emsp;&emsp;Buffer（缓冲区）与 Image（图像）是 Vulkan 对外暴露的逻辑资源对象，本身不持有内存，只描述数据的用途、结构与访问规则，需要后续绑定到具体物理内存上才能使用。
* `Buffer`：线性数据结构（数组、结构体、SSBO），适用于通用计算；
* `Image`：多维结构数据（2D/3D/纹理），适用于空间局部性强的访问模式；

>OpenCL 的 cl_mem 对象在创建时就已经锁定了其背后的存储空间。虽然 OpenCL 2.0 引入了 SVM（共享虚拟内存），但在资源与存储的解耦灵活性上仍不及 Vulkan。

---

**描述符与描述符集（Descriptors & Descriptor Sets）**
&emsp;&emsp;Shader 并不直接连接到 Buffer 或 Image。描述符是 Shader 访问外部资源的绑定接口，负责将 Buffer/Image/Sampler 等资源映射到着色器的绑定槽，实现 CPU 侧资源与 GPU 着色器的连接。
- **描述符 (Descriptor)**： 指向资源的“句柄”或“指针”，包含资源的类型、状态（如 Image Layout）以及它所在的内存范围。
- **描述符集 (Descriptor Sets)**： 将一组描述符打包在一起。Shader 通过绑定这些集合来访问资源。
- **描述符集布局 (Descriptor Set Layout)**： 定义了 Shader 期望的接口模板，类似于函数签名的参数列表。

>描述符集从描述符池（Descriptor Pool）中分配，池预先指定各类描述符数量上限，避免运行时超额分配。这种设计使驱动可提前规划内存布局，提升绑定效率。

>在 OpenCL 中，如果要在两个不同的 Kernel 之间共享大量资源，必须重复设置参数；而在 Vulkan 中，只需要切换一个 Descriptor Set 即可，这极大地降低了渲染/计算管线切换时的驱动负载（Driver Overhead）。
&emsp;&emsp;支持的资源类型包括：
- **UNIFORM_BUFFER**：uniform 缓冲，只读、尺寸较小，适合常量参数；
- **STORAGE_BUFFER**：存储缓冲，支持读写，适合大规模并行计算；
- **COMBINED_IMAGE_SAMPLER**：纹理采样器，Shader 中采样贴图；
- **STORAGE_IMAGE**：存储图像，支持像素级随机读写。

---

**资源访问与视图机制**

&emsp;&emsp;为了提升资源使用的灵活性，Vulkan 引入了“视图（View）”机制：

* `BufferView` / `ImageView` 用于定义：
  * 数据格式（如 float4 / rgba8）
  * 访问范围（子区域 / 子资源）
* 同一资源可以创建多个视图，实现：
  * 数据重解释（reinterpret）
  * 多用途访问（如计算 + 采样）

&emsp;&emsp;Vulkan的资源控制完全是显示的，相对于OpenGL的资源控制依赖驱动层的全局状态机隐式控制，OpenCL可以控制一部分资源属性，但不是全部。
| 特性    | Vulkan        | OpenGL | OpenCL    |
| ----- | ------------- | ------ | --------- |
| 内存管理  | 完全显式          | 隐式     | 半显式       |
| 资源绑定  | DescriptorSet | 全局状态机  | Kernel 参数 |
| 数据迁移  | 手动控制          | 自动     | 半自动       |
| 同步机制  | 显式 Barrier    | 隐式     | 事件驱动      |
| 性能可控性 | 极高            | 低      | 中         |

>资源在不同队列、不同阶段间传递时，必须使用内存屏障、管线屏障保证可见性与执行顺序。
---

&emsp;&emsp;Vulkan的资源管理典型流程：
- 创建资源（Buffer/Image）→ 
- 分配内存（DeviceMemory）→ 
- 绑定资源与内存 →  
- 创建 DescriptorSet 并写入资源 →  
- 提交 GPU 使用 →  
- 同步与回收 →  
- 销毁资源与内存 →  

### 2.3 执行模型

&emsp;&emsp;Vulkan 执行模型，是定义命令从 CPU 端生成、录制、提交，到 GPU 端调度、并行执行、完成反馈的全生命周期规则，同时涵盖 GPU 硬件管线阶段、任务并行机制、内存可见性与同步约束的一整套核心体系。与 OpenGL/Direct3D 11 等传统 API 采用的 “立即模式 + 隐式驱动调度” 截然不同，Vulkan 彻底消除了驱动层的黑盒自动同步、隐式状态管理与命令重排，将GPU执行流程的全维度控制权完全交予应用。这一设计既为实现极致的硬件利用率、多线程并行能力与低延迟渲染提供了基础，也要求应用严格遵循执行模型的规则，否则将产生未定义行为、渲染错误与性能损耗。

&emsp;&emsp;Vulkan 执行模型的核心体系，依托五大核心抽象构建：**队列（Queue）、命令缓冲区（Command Buffer）、管线（Pipeline）、同步原语与渲染通道（Render Pass）** 以及配套的着色器执行模型。

---

**队列与队列族**

&emsp;&emsp;队列是GPU硬件执行任务的唯一入口，也是GPU硬件层面独立的执行流。每个队列对应GPU硬件的一个执行流，同硬件的多个队列之间可完全并行执行任务，无需CPU干预。Vulkan 并不假设 GPU 只有一条路可走。硬件能力被划分为不同的**队列族 (Queue Families)**：

- **图形队列**：支持所有图形渲染命令、计算命令与传输命令，是功能最完整的队列家族，所有 Vulkan 实现都必须支持至少一个图形队列家族；
- **计算队列**：仅支持计算调度命令与传输命令，不依赖图形管线，可与图形队列完全并行执行，常用于异步计算、后处理、物理模拟等场景；
- **传输队列**：仅支持内存拷贝、数据传输类命令，专门用于异步数据上传 / 下载，不占用图形 / 计算队列的硬件资源，可实现纹理、顶点数据的后台无感传输；
- **稀疏绑定队列**：专门用于稀疏资源的内存绑定更新，支持对大纹理、缓冲区的部分内存进行动态映射与解绑。

---

**命令缓冲区（CommandBuffer）**
&emsp;&emsp;&emsp;&emsp;命令缓冲区是 CPU 向 GPU 传递执行指令的载体。与 OpenGL/OpenCL的立即执行模式（每调用一个绘制函数就直接向 GPU 提交命令）不同，Vulkan 采用 “先录制、后提交” 的模式：CPU 先将所有要执行的指令录制到命令缓冲区中，录制完成后，再一次性批量提交到 GPU 队列执行。可以同时向一个“异步计算队列”提交物理模拟任务，向“图形队列”提交渲染任务，实现真正的**异构并发执行**。

>命令缓冲区不能直接创建，必须从命令池中分配。

* **录制 (Recording)：** 开发者在 CPU 端通过 `vkBeginCommandBuffer` 开始录制一系列指令（如绑定管线、设置描述符、分发计算任务）。这个过程是**线程安全**的，意味着可以在多个 CPU 核心上并行录制不同的命令缓冲区。
* **提交 (Submission)：** 录制完成后，通过 `vkQueueSubmit` 将缓冲区一次性推送到 GPU 队列。
* **与OpenCL对比：**
    * **OpenCL：** 每次 `clEnqueue` 都会产生一定的驱动开销，频繁调用会造成 CPU 瓶颈。
    * **Vulkan：** 录制一次命令缓冲区可以多次提交（重用），且多线程录制消除了单核提交的瓶颈。

| 特性 | Vulkan | OpenCL |
| :--- | :--- | :--- |
| **任务生成** | 离线并行录制 (Command Buffer) | 在线顺序入队 (clEnqueue) |
| **多线程支持** | 原生支持多线程录制，极低 CPU 负载 | 驱动层级的线程限制较多 |
| **硬件通道** | 显式区分计算、图形、传输队列 | 抽象为统一的 Command Queue |
| **同步开销** | 极低（由开发者精确控制） | 较高（驱动需要维护复杂的事件状态机） |
| **内核切换** | 切换 Pipeline 状态开销极小 | 切换 Kernel 涉及较重的 Context 切换 |

&emsp;&emsp;Vulkan 的执行模型更贴近现代 GPU 的硬件结构：它不再是一个简单的“命令分发器”，而是一个由**多线程录制器**、**多功能队列**和**精确同步网格**组成的复杂系统。对于高性能计算而言，这意味着可以榨干 GPU 的每一颗流处理器，而不会让 CPU 在驱动层“空转”。

---

**管线（Pipeline）**
&emsp;&emsp;管线是 GPU 执行任务的核心程序容器，定义了 GPU 处理数据的完整执行流程、着色器代码与固定功能硬件的状态配置。Vulkan 的管线采用 “预编译、预固化” 设计，绝大多数状态在管线创建时就已固定，驱动可在创建阶段完成全链路的编译优化，彻底消除传统 API 运行时的管线重编译开销。

&emsp;&emsp;Vulkan 提供两类核心管线，对应渲染与计算两大核心场景，二者均通过 ** 管线布局（Pipeline Layout）** 与资源模型关联，定义着色器可访问的描述符集与推送常量（Push Constant）布局。
- 图形管线是 Vulkan 渲染的核心，对应 GPU 硬件的图形渲染流水线，分为可编程着色器阶段与固定功能阶段，主要配置渲染管线中各个stage的参数或者着色器等。
- 计算管线是 GPU 通用计算的核心，相比图形管线结构极简，仅包含单个可编程计算着色器阶段，无固定功能阶段依赖，无需绑定渲染通道，可独立提交到计算队列执行。调度类似OpenCL的工作组和工作项。

---

**渲染通道（Render Pass）**
&emsp;&emsp;渲染通道（VkRenderPass）是 Vulkan 图形渲染的核心抽象，定义了帧缓冲区附件（颜色、深度、模板附件）的生命周期、加载 / 存储操作，以及渲染流程的子通道划分。子通道依赖（Subpass Dependency）是渲染通道内的专用同步机制，用于定义不同子通道之间的执行与内存依赖，相比通用管线屏障，可针对 Tile-Based 架构 GPU 做深度优化，减少附件数据的内存读写，大幅降低移动平台的带宽开销。

&emsp;&emsp;计算管线是 GPU 通用计算的核心，相比图形管线结构极简，仅包含单个可编程计算着色器阶段，无固定功能阶段依赖，无需绑定渲染通道，可独立提交到计算队列执行。

---

**同步原语**
&emsp;&emsp;在 OpenGL 或 OpenCL 的早期版本中，驱动程序通常会在后台隐式地帮开发者处理许多同步问题（例如阻塞 CPU 等待 GPU 完成，或者自动刷新缓存）。而在 Vulkan 中，所有的命令提交都是完全异步的。如果不显式地告诉 GPU 任务之间的依赖关系，它会以最高效（但也最不可预测）的乱序方式执行，从而导致严重的数据竞争和画面撕裂。
&emsp;&emsp;为了建立秩序，Vulkan 提供了四种核心的同步原语，它们的粒度和作用域各不相同：
- 栅栏 (Fence)
  * **作用域：** **GPU -> CPU** 的单向通知。
  * **核心机制：** 当 CPU 向 GPU 提交了一批命令（`vkQueueSubmit`）后，可以附带一个 Fence。CPU 随后可以调用 `vkWaitForFences` 进入休眠状态，直到 GPU 执行完这批命令并发出信号。
  * **典型场景：** **帧同步 (Frame Pacing)**。CPU 必须等待 GPU 彻底渲染完第 N 帧，才能开始复用第 N 帧对应的命令缓冲区 (Command Buffer) 和统一变量 (Uniform Buffers)，否则 CPU 会覆盖 GPU 正在读取的数据。
- 信号量 (Semaphore)
  * **作用域：** **GPU 队列 -> GPU 队列**（或同一队列的不同提交批次之间）。完全在 GPU 时间线上发生，**不需要 CPU 介入**。
  * **核心机制：** 一个操作（如渲染完毕）发出信号 (Signal)，另一个操作（如屏幕呈现）等待信号 (Wait)。
  * **典型场景：** **渲染流水线接力**。例如：
    1. 交换链 (Swapchain) 准备好了一张图像 -> *触发 Semaphore A*。
    2. 图形队列等待 *Semaphore A*，开始绘制。绘制完成后 -> *触发 Semaphore B*。
    3. 呈现引擎 (Presentation Engine) 等待 *Semaphore B*，然后将图像推送到屏幕。
- 管线屏障 (Pipeline Barrier)
  * **作用域：** **命令缓冲区内部 (Intra-Command Buffer)**。它是 Vulkan 中最复杂、但也最常用的微操工具。
  * **核心机制：** 屏障不仅控制**执行流**（Execution Dependency，例如：顶点着色器必须先跑完，片元着色器才能跑），还控制**内存可见性**（Memory Dependency，例如：写入 L2 缓存的数据必须刷新到显存，以便下一个阶段读取）。
  * **典型场景：** * **Image Layout Transition (图像布局转换)：** 将一张图片从 `Compute Shader` 的“通用写入布局”转换为 `Graphics Pipeline` 的“只读采样布局”。
    * **解决 Read-After-Write (RAW) 冲突：** 确保计算着色器算出的粒子坐标，能被后续的顶点着色器正确读取，而不是读到缓存里的脏数据。
- 事件 (Event)
  * **作用域：** 可以由 CPU 设置，GPU 等待；或者 GPU 设置，GPU 等待。
  * **核心机制：** 类似于把一个 Pipeline Barrier 拆成了两半：先在前面 `vkCmdSetEvent`，然后在后面 `vkCmdWaitEvents`。它允许 GPU 在设置事件和等待事件之间，去执行其他不相关的指令，从而提升硬件利用率。
  * **典型场景：** 极其追求极致优化的细粒度调度（但在实际开发中，开发者为了代码可维护性，往往更倾向于直接使用 Pipeline Barrier）。

| 原语名称 | 谁发出信号 (Signal) | 谁等待 (Wait) | 解决的核心问题 | 性能开销 |
| :--- | :--- | :--- | :--- | :--- |
| **Fence** | GPU | CPU | 防止 CPU 跑得比 GPU 快，覆盖使用中的资源 | 较高 (涉及 CPU 阻塞) |
| **Semaphore** | GPU 队列 | GPU 队列 | 保证大块任务（如计算与渲染）的宏观顺序 | 中等 |
| **Barrier** | GPU 管线阶段 | GPU 管线阶段 | 保证微观管线阶段的顺序、缓存刷新、图像布局转换 | 极低 (纯硬件流水线控制) |


| 特性         | Vulkan                                            | OpenCL                         | 说明                                                                   |
| ---------- | ------------------------------------------------- | ------------------------------ | -------------------------------------------------------------------- |
| **同步机制类型** | 显式同步（Pipeline / Memory Barriers）                  | 事件驱动（Event / clWaitForEvents）  | Vulkan 需要显式在命令流中声明资源访问顺序；OpenCL 可以通过事件串联命令完成同步                       |
| **控制粒度**   | 精确到阶段（Pipeline Stage）和访问类型（Access Mask）           | 基于命令队列粒度，无法精细控制单个内存访问          | Vulkan 能精确控制读/写冲突；OpenCL 同步粗粒度，通常按 kernel 或 buffer 级别同步              |
| **同步对象**   | 无全局锁，使用 Barrier、Semaphore、Fence                   | Event、clFinish、clWaitForEvents | Vulkan 通过命令缓冲区显式插入 Barrier；OpenCL 通过事件或队列顺序控制执行                      |
| **跨队列同步**  | 支持，通过 Semaphore / Queue Submit + Pipeline Barrier | 支持，通过事件和命令队列关联                 | Vulkan 可以在不同 Compute / Graphics / Transfer 队列之间同步资源；OpenCL 也可以，但粒度较粗 |
| **性能开销**   | 可控，低开销，最小化隐式等待                                    | 相对不可控，驱动决定具体行为                 | Vulkan 精确控制资源访问顺序，减少不必要的等待；OpenCL 隐式等待可能增加延迟                         |
| **易用性**    | 复杂，需要手动管理                                         | 简单，事件自动管理依赖                    | Vulkan 对新手门槛高，但性能可预测；OpenCL 易上手，但可控性有限                               |
| **使用场景优势** | 高性能计算、复杂管线、多队列异步计算                                | 通用 GPU 计算、简化同步需求               | Vulkan 适合需要最大性能控制和精确资源管理的场景；OpenCL 适合快速开发和跨平台计算                      |

**着色器执行模型**
&emsp;&emsp;着色器执行模型是 Vulkan 执行模型在 GPU 可编程阶段的延伸，定义了着色器代码的执行方式、并行调用规则、内存访问与同步规范。它将高级语言（如 GLSL/HLSL）编写的逻辑映射到 GPU 的大规模并行硬件架构上。
- [Vulkan High Level Shader Language Comparison](https://docs.vulkan.org/guide/latest/high_level_shader_language_comparison.html)

这一节将深入探讨 **着色器执行模型 (Shader Execution Model)**。如果说 2.3 节的执行模型是宏观的“调度指挥部”，那么着色器执行模型就是微观的“单兵作战手册”，它定义了数以千计的线程如何协同工作。

---

**着色器执行模型**

&emsp;&emsp;着色器执行模型是 Vulkan 执行模型在 GPU 可编程阶段的延伸，定义了着色器代码的执行方式、并行调用规则、内存访问与同步规范。它将高级语言（如 GLSL/HLSL）编写的逻辑映射到 GPU 的大规模并行硬件架构上。

&emsp;&emsp;Vulkan 的并行层次结构与 OpenCL 高度相似，但术语略有不同，理解这种对应关系是迁移算法的关键：

* **着色器调用 (Shader Invocation)：** 执行着色器程序的最小单元。对应 OpenCL 的 **Work-item**。
* **本地工作组 (Local Workgroup)：** 一组同时执行并可以共享内存（Shared Memory）的调用集合。对应 OpenCL 的 **Work-group**。
* **派发网格 (Dispatch Grid)：** 由多个工作组构成的三维空间。通过 `vkCmdDispatch` 定义，对应 OpenCL 的 **NDRange**。

&emsp;&emsp;Vulkan 着色器定义了严格的内存层级与可见性规则，不同内存层级的访问性能与同步约束完全不同。开发者必须显式管理数据在硬件各级缓存（L1/L2/显存）之间的流动。
- **寄存器与私有内存 (Private/Function)**
  - 访问性能： 极高（单时钟周期）。
  - 可见性： 仅对当前调用 (Invocation) 可见。
  - 同步约束： 无需同步。这是着色器内部定义的局部变量。
- **工作组本地存储 (Workgroup/Shared Memory)**
  - 访问性能： 高（对应 GPU 的片上 SRAM/LDS）。
  - 可见性： 对同一个 Workgroup 内的所有调用可见。
  - 同步约束： 必须使用 controlBarrier（确保执行同步）和 memoryBarrierWorkgroup（确保写入操作对组内其他线程可见）。
  - OpenCL 对比： 等同于 OpenCL 的 __local 内存。
- **存储缓冲区与图像 (Storage Buffer / Image / Global)**
  - 访问性能： 中到低（涉及 L2 缓存甚至 VRAM 高延迟访问）。
  - 可见性： 全局可见。
  - 同步约束： 极其严格。即使在同一个 Workgroup 内，一个线程写入 Global 内存，另一个线程也不保证能立刻读到最新值。必须使用 memoryBarrierBuffer 或在变量声明时添加 coherent 修饰符。
  - OpenCL 对比： 等同于 OpenCL 的 __global 内存。

&emsp;&emsp;在着色器内部，Vulkan 通过 **SPIR-V 存储类** 显式定义了数据的“可见范围”和“生存周期”。这比 OpenCL 的内存模型更加严苛：

* **Input/Output：** 用于管线阶段间传递数据（如 Vertex 传给 Fragment）。
* **Uniform：** 只读的常量数据，通常映射到 GPU 的常量缓存。
* **StorageBuffer：** 可读写的通用数据缓冲区（对应 OpenCL 的 `__global`）。
* **Workgroup (Shared)：** 仅在当前工作组内可见的快速内存（对应 OpenCL 的 `__local`）。

> **关键差异：** Vulkan 引入了 `NonWritable`、`NonReadable` 以及 `Coherent` 等修饰符。如果在 Shader 中写入了一个 Storage Buffer 并希望立即读取，必须显式使用 `memoryBarrierBuffer()`，否则 GPU 可能会因为 L1/L2 缓存未刷新而读到旧值。


&emsp;&emsp;**子组 (Subgroup)** 对应硬件底层的执行单元（如 NVIDIA 的 **Warp** 或 AMD 的 **Wavefront**，通常为 32 或 64 个线程）。

* **硬件级通信：** 子组内的线程可以通过硬件指令直接交换数据（如 `subgroupShuffle`），而**无需访问内存或使用 Barrier**。
* **与 OpenCL 对比：** OpenCL 原生规范长期缺乏对 Warp/Wave 级别的标准化支持（通常依赖厂商扩展），而 Vulkan 将 **Subgroup Operations** 纳入了核心规范。这使得开发者可以编写出极高性能的规约（Reduction）和前缀和（Scan）算法。

&emsp;&emsp;在图形管线中，着色器执行受限于光栅化顺序；但在 **计算着色器 (Compute Shader)** 中，执行模型拥有最高的自由度：

1.  **原子操作 (Atomic Operations)：** 支持在全局内存和共享内存上进行跨线程的原子加减、比较交换 (CAS)，这是构建并发数据结构（如无锁队列）的基础。
2.  **控制流同步 (Control Flow Sync)：** 使用 `barrier()` 确保工作组内所有线程到达同一执行点，防止 Read-After-Write 冲突。

---


| 特性 | Vulkan 着色器执行模型 | OpenCL 内核执行模型 |
| :--- | :--- | :--- |
| **中间格式** | 强制使用 **SPIR-V** 二进制 | 支持源码字符串或 SPIR-V |
| **屏障语义** | 区分执行屏障与内存屏障，极其精确 | 相对模糊的 `barrier()` 语义 |
| **硬件映射** | 原生支持 Subgroup (Warp/Wave) 级别优化 | 依赖厂商扩展，跨平台一致性较弱 |
| **资源绑定** | 基于 Descriptor Sets 的静态布局 | 基于 `clSetKernelArg` 的动态绑定 |

## 3 从一个Compute例子看具体的组件

   
# 参考文献
- [Vulkan High Level Shader Language Comparison](https://docs.vulkan.org/guide/latest/high_level_shader_language_comparison.html)