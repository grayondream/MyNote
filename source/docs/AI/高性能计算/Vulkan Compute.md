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


## 2 Vulkan Compute 组建
&emsp;&emsp;Vulkan通过一套统一的执行与资源模型（Pipeline + Descriptor + CommandBuffer + Queue），将图形管线和计算管线抽象为并列的两种执行形式：图形管线是包含固定功能阶段的多阶段流水线，而计算管线则是更通用的单阶段并行执行模型；二者共享相同的资源绑定、调度与同步机制，从而实现数据在 GPU 内的无缝流动与协同执行。因此许多
![](https://vulkan-tutorial.com/images/vulkan_pipeline_block_diagram.png)