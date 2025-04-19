# Nvidia显卡架构演进
## 1 简介
&emsp;&emsp;显示卡（英语：Display Card）简称显卡，也称图形卡（Graphics Card），是个人电脑上以图形处理器（GPU）为核心的扩展卡，用途是提供中央处理器以外的微处理器帮助计算图像信息，并将计算机系统所需要的显示信息进行转换并提供逐行或隔行扫描信号给显示设备，是连接显示器和个人电脑主板的重要组件，是“人机交互”的重要设备之一。显卡有时被称为独立显卡或专用显卡，以强调它们与主板上的集成图形处理器（集成显卡）或中央处理器 (CPU) 的区别。
&emsp;&emsp;早期显卡主要用来进行图像显示，其主要应用场景为游戏渲染等领域。而自从深度学习开启21世纪的人工智能热潮之后，显卡也被用来进行计算加速。从此之后，显卡厂商也将显卡的并行计算能力作为衡量显卡性能的标准之一。众多显卡厂商中，其中Nvidia是其中的佼佼者。
&emsp;&emsp;为了跟上AI热潮，且我本人对于并行计算也比较感兴趣，因此这里总结下Nvidia显卡架构演进来学习底层硬件结构指导自己。

&emsp;&emsp;Nvidia成立于1993年4月，截止2025年，这30年里其发布了众多的显卡型号。每一代显卡都有各自的新特性和新的侧重点，但是总的来说主要分为两种类型架构：早期架构和统一架构。前者因为时间的流逝逐渐被新的显卡型号替代，且不具备太大的参考意义。因此本文主要聚焦统一架构，对于早期架构不会详细描述。

## 2 早期架构
&emsp;&emsp;早期架构并不是一个正式的架构名称，而是为了和后续的统一架构区分。
&emsp;&emsp;Nvidia早期架构主要聚焦在提升图形性能，比如提升纹理的处理能力，引入更多硬件加速来提升渲染性能。从刚开始的NV1到典型其市场地位的GeForce256，Nvidia不断提升使得3D Video Game成为了现实。这些早期架构虽然比较简陋，但是正是有这些早期的尝试才有了现如今的辉煌。

### 2.1 NV1（发布于1995）
&emsp;&emsp;NV1 是由 NVIDIA 用了两年研发，于1995年5月发布的显示芯片[1]。它亦是 NVIDIA 自创立起的首款产品。NVIDIA 亦授权 SGS Thomson Microelectronics 生产，芯片型号为 STG2000X B。当时还没有像 Direct3D 的多边形 3D 标准，所以 nVIDIA 使用二次方程纹理贴图作为立体图形的实现方式。它不但拥有完整的 2D/3D 核心，还内置声音处理核心。随后微软在 Windows 95 制订 Direct3D 多边形立体标准，纵使 NV1 的二次方程纹理贴图是出色的技术，但始终不兼容 Direct3D，亦不支持当时还很流行的 Glide，导致该显卡市场上响应不佳。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/Yuan_3DS-100.jpg)

### 2.2 RIVA128（发布于1997）
&emsp;&emsp;NVIDIA RIVA 128 (1997) 是 NVIDIA 走向成功的关键一步。作为其首款消费级显卡，RIVA 128 并非完美，但它在当时以合理的价格提供了显著的性能提升，特别是对 Direct3D 5.0 的支持使其在游戏中表现出色。它采用 128 位显存接口，在纹理填充率方面表现良好，但缺乏硬件加速的三角形设置引擎是其弱点，在复杂场景中性能会明显下降。尽管如此，RIVA 128 凭借其性价比和相对出色的性能，成功打入市场，为 NVIDIA 赢得了声誉，并为后续 RIVA TNT 等更强大的产品奠定了基础。它标志着 NVIDIA 从一家默默无闻的小公司成长为图形卡市场的重要参与者。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/250px-NVidia_Riva_128.jpg)

### 2.3 RIVA TNT（发布于1998）
&emsp;&emsp;NVIDIA RIVA TNT (1998) 是 RIVA 128 的继任者，也是 NVIDIA 在图形卡市场取得更大成功的关键。TNT 的核心改进在于其双纹理引擎 (Twin Texel engine)，使其能够在每个时钟周期处理两个纹理单元，从而有效地将纹理填充率翻倍，显著提升了游戏性能。这使得 RIVA TNT 在当时成为极具竞争力的产品，赢得了众多游戏玩家的青睐。尽管 RIVA TNT 仍然缺乏硬件加速的三角形设置引擎，这在一定程度上限制了其在复杂 3D 场景中的表现，但凭借其卓越的纹理处理能力和相对较低的价格，它迅速成为市场上的热门选择，进一步巩固了 NVIDIA 在图形卡领域的地位，并为后续的 GeForce 系列铺平了道路。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/Creative_Labs_Graphics_Blaster_Riva_TNT.jpg)

### 2.4 GeForce256（发布于1999）
&emsp;&emsp;NVIDIA GeForce 256 (1999) 是图形卡发展史上的一个里程碑，通常被认为是“第一款 GPU”。它首次集成了硬件 T&L (Transform and Lighting) 引擎，将顶点转换和光照计算从 CPU 转移到 GPU 处理，极大地提升了 3D 图形的性能。NVIDIA 也正是用这款产品定义了“GPU”一词，强调其作为独立图形处理器的作用。GeForce 256 支持 DirectX 7，拥有出色的单纹理填充率，并引入了立方体环境贴图等先进技术。虽然在多边形处理能力上仍有不足，但 GeForce 256 凭借其革命性的硬件 T&L 设计和卓越的整体性能，迅速成为市场领导者，为现代 GPU 架构奠定了基础，并开启了 NVIDIA 在图形处理器领域的霸主地位。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/VisionTek_GeForce_256.jpg)
## 3 统一架构
&emsp;&emsp;NVIDIA 统一架构（Unified Architecture）是 NVIDIA 在 2006 年发布的 GeForce 8 系列显卡中引入的革命性设计。它打破了传统 GPU 中顶点着色器和像素着色器分离的架构，采用统一的着色器单元，可以动态地分配计算资源给任何类型的着色任务。这意味着 GPU 能够更有效地利用其计算能力，不再受限于特定着色器的性能瓶颈。

&emsp;&emsp;统一架构还引入了 CUDA (Compute Unified Device Architecture)，使 GPU 不仅可以用于图形渲染，还可以用于通用计算。这为 GPU 在科学计算、人工智能等领域开辟了广阔的应用前景。

&emsp;&emsp;总而言之，NVIDIA 统一架构通过灵活的资源分配和 CUDA 的引入，极大地提高了 GPU 的效率和通用性，是 GPU 发展史上的一个重要转折点，奠定了现代 GPU 的基础。

### 3.1 Tesla（2006-2010）
&emsp;&emsp;Tesla 架构发布于 2006 年。Tesla 架构全新的 CUDA 架构，支持使用 C 语言进行 GPU 编程，可以用于通用数据并行计算。Tesla 架构具有 128 个流处理器，带宽高达 86GB/s，标志着 GPU 开始从专用图形处理器转变为通用数据并行处理器。

&emsp;&emsp;Tesla架构的第一款产品为Nvidia G80。G80 作为 NVIDIA 首款 Tesla 架构的基础，具有以下里程碑式的创新：
- C 语言支持： 首次允许开发者使用熟悉的 C 语言进行 GPU 编程，降低了 GPU 编程的门槛（CUDA）。
- 统一架构： 采用单一、统一的处理器取代了独立的顶点和像素管线，能够灵活执行各种类型的程序（顶点、几何、像素和计算程序）。
- 标量线程处理器： 使用标量线程处理器，简化了编程模型，程序员无需手动管理向量寄存器。
- SIMT 执行模型： 引入单指令多线程 (SIMT) 执行模型，允许多个线程并发执行同一指令，提高了并行效率。
- 线程间通信机制： 提供了共享内存和屏障同步机制，方便线程之间进行数据共享和同步，增强了程序的灵活性。

>&emsp;&emsp;CUDA 是一种硬件和软件架构，它使 NVIDIA GPU 能够执行使用 C、C++、Fortran、OpenCL、DirectCompute 和其他语言编写的程序。CUDA 程序调用并行内核。内核在一组并行线程上并行执行。程序员或编译器将这些线程组织成线程块和线程块网格。GPU 在并行线程块网格上实例化一个内核程序。线程块中的每个线程执行内核的一个实例，并且在其线程块中具有线程 ID、程序计数器、寄存器、每个线程的私有内存、输入和输出结果。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gpu%20g80.jpg)

&emsp;&emsp;这些创新使得 G80 不仅在图形处理方面表现出色，也为 GPU 在通用计算领域的应用奠定了坚实的基础，开启了 GPGPU (General-Purpose computing on Graphics Processing Units) 的时代。同时，其全面支持Direct3D 10和DirectX 10 Shader Model 4.0，凭借其内部128位浮点精度、无限长度着色器以及对多重纹理和渲染目标的支持，实现了卓越的图形处理能力。同时，它还集成了NVIDIA Lumenex技术和PureVideo HD技术，分别在图像增强和高清视频处理方面表现出色，并通过SLI技术支持多GPU并行，为用户带来前所未有的视觉体验。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/nvidia%20g80.png)

### 3.2 Fermi（2010-2012）
&emsp;&emsp;NVIDIA Fermi 架构是 2010 年发布的一款具有里程碑意义的 GPU 微架构，它标志着 NVIDIA 在 GPU 计算领域的重大突破。作为 Tesla 架构的继任者，Fermi 架构主要应用于 GeForce 400 和 500 系列显卡，以及 Quadro 和 Tesla 系列专业卡，旨在提供卓越的图形性能和强大的并行计算能力。

&emsp;&emsp;Fermi 架构的核心在于其对计算的优化。它是 NVIDIA 首个完整的 GPU 计算架构，显著提升了双精度浮点性能，满足了科学计算和工程模拟等领域的需求。它全面兼容 IEEE 754-2008 浮点标准，支持融合乘加运算 (FMA)，保证了计算的精度和可靠性。同时，Fermi 架构还引入了 ECC 保护，覆盖从寄存器到 DRAM 的各个环节，提高了数据完整性。

&emsp;&emsp;在架构设计上，Fermi 采用了统一寻址模型，简化了内存管理，并实现了所有级别的缓存，提高了数据访问效率。每个流式多处理器 (SM) 最多包含 32 个 CUDA 核心，这些核心能够并行执行大量的线程。此外，Fermi 还采用了可配置的 L1 缓存，允许根据不同的应用场景灵活分配共享内存和 L1 缓存的大小。双 Warp 调度器和双指令分派单元的设计，使得每个 SM 能够并发执行两个 Warp，进一步提升了并行执行效率。

&emsp;&emsp;Fermi 架构还特别针对图形处理进行了优化。它包含一个专为曲面细分和位移贴图优化的 PolyMorph 引擎，能够提供更逼真的游戏画面。此外，Fermi 也是 NVIDIA 最早支持 Microsoft Direct3D 12 feature_level 11 渲染 API 的微架构，为新一代游戏和图形应用提供了硬件支持。

&emsp;&emsp;尽管 Fermi 架构采用的是 40nm 工艺制造，拥有高达 30 亿个晶体管，但其创新的架构设计和强大的计算能力，为 NVIDIA 在 GPU 领域奠定了坚实的基础。Fermi 架构不仅是一款成功的游戏显卡架构，更是一款重要的 GPU 计算平台，推动了 GPU 在科学研究、人工智能等领域的应用。它以意大利物理学家恩里科·费米的名字命名，也象征着 NVIDIA 在 GPU 技术上的不断创新和突破

&emsp;&emsp;总结下来，Fermi架构的关键特性有：
- 计算 GPU： Fermi 是 NVIDIA 首个完整的 GPU 计算架构。
- 双精度浮点性能： 在单芯片上提供高水平的双精度浮点性能。
- IEEE 754-2008 标准： 兼容 IEEE 754-2008 浮点标准，包括融合乘加运算 (FMA)。
- ECC 保护： 提供从寄存器到 DRAM 的 ECC 保护。
- 统一寻址： 具有直接的线性寻址模型，并在所有级别进行缓存。
- CUDA 核心： 每个流式多处理器 (SM) 最多包含 32 个 CUDA 核心。
- 可配置的缓存： 每个 SM 的 L1 缓存可配置为支持共享内存以及本地和全局内存操作的缓存。64 KB 内存可以配置为 48 KB 共享内存 + 16 KB L1 缓存，或 16 KB 共享内存 + 48 KB L1 缓存。
- 双 Warp 调度器： 每个 SM 具有两个 Warp 调度器和两个指令分派单元，允许并发执行两个 Warp。
- 多线程： 多个线程被分组到最多包含 1,536 个线程的线程块中。
- PolyMorph 引擎： Fermi 架构包含一个 PolyMorph 引擎，专为曲面细分和位移贴图优化。

#### 3.2.1 Fermi Architecture Overview
&emsp;&emsp;首款基于 Fermi 的 GPU 拥有高达 512 个 CUDA 核心，由 30 亿个晶体管实现。每个 CUDA 核心在一个时钟周期内为一个线程执行一个浮点或整数指令。这 512 个 CUDA 核心被组织成 16 个 SM（流式多处理器），每个 SM 包含 32 个核心。该 GPU 具有六个 64 位内存分区，构成一个 384 位内存接口，最多支持总计 6 GB 的 GDDR5 DRAM 内存。主机接口通过 PCI-Express 将 GPU 连接到 CPU。GigaThread 全局调度器将线程块分发到 SM 线程调度器。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/Fermi_overview_2.png)

#### 3.2.2 Stream Multiprocessor(SM)

&emsp;&emsp;每个 SM 具有 32 个 CUDA 处理器，比之前的 SM 设计增加了四倍。每个 CUDA 处理器都有一个完全流水线的整数算术逻辑单元 (ALU) 和浮点单元 (FPU)。之前的 GPU 使用 IEEE 754-1985 浮点运算。Fermi 架构实现了新的 IEEE 754-2008 浮点标准，为单精度和双精度算术运算提供了融合乘加 (FMA) 指令。FMA 通过在单个最终舍入步骤中完成乘法和加法，而不在加法中损失精度，从而改进了乘加 (MAD) 指令。FMA 比单独执行操作更准确。GT200 实现了双精度 FMA。

&emsp;&emsp;在 GT200 中，整数 ALU 的乘法运算精度限制为 24 位；因此，整数算术运算需要多指令仿真序列。在 Fermi 中，新设计的整数 ALU 支持所有指令的完整 32 位精度，与标准编程语言要求一致。整数 ALU 也经过优化，可以有效地支持 64 位和扩展精度运算。支持各种指令，包括布尔运算。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/StreamProcessor.png)

&emsp;&emsp;SM中除了CUDA核心，还有其他负责数据传输和特殊运算的单元：
- Warp 调度器（Warp Schedulers）： 每个 SM 有两个 Warp 调度器。每个调度器从其准备就绪的 Warps 中选择一个 Warp 执行。 双 Warp 调度器允许每个时钟周期从两个不同的 Warps 发出两条指令。 这种双重调度能力提高了 SM 的吞吐量，并有助于隐藏延迟。
- 分派单元（Dispatch Units）： 与 Warp 调度器配对，每个 SM 有两个分派单元。 每个分派单元负责将 Warp 调度器选择的指令分派给 CUDA 核心或其他执行单元。 两个分派单元允许每个时钟周期分派两条指令，从而提高指令吞吐量。
- 寄存器文件（Register File）： 用于存储线程的局部变量和临时数据。 Fermi 架构具有较大的寄存器文件，为每个 SM 提供更多的寄存器。 增加的寄存器数量减少了对全局内存的访问，提高了性能。
- 加载/存储单元（LD/ST Units）： 负责从内存加载数据和将数据存储到内存。 Fermi 架构具有专用的加载/存储单元，以高效地处理内存访问操作。 这些单元支持各种内存访问模式，包括对齐和非对齐的访问。
- 特殊功能单元（SFU）： 用于执行特殊功能指令，如三角函数、指数函数和对数函数。 Fermi 架构具有专用的 SFU，以加速这些计算密集型操作。 SFU 允许 GPU 高效地执行复杂的数学计算，这对于图形渲染和科学计算至关重要。

### 3.3 Kepler（2012-2014）
#### 3.3.1 Kepler Overview
&emsp;&emsp;NVIDIA Kepler 架构是继 Fermi 架构之后，于 2012 年推出的 GPU 微架构。它主要应用于 GeForce 600、700 系列显卡，以及 Quadro 和 Tesla 系列专业卡。Kepler 架构在能效比和计算能力上都取得了显著的进步，旨在提供更出色的图形性能和更强大的并行计算能力。

&emsp;&emsp;Kepler 架构的核心改进在于其对能效的优化。相较于 Fermi，Kepler 采用了更先进的 28nm 工艺制造，降低了功耗和发热量。它引入了动态超频技术 GPU Boost，能够根据负载自动调整 GPU 的频率，在保证性能的同时降低功耗。此外，Kepler 还采用了更高效的 SMX (Streaming Multiprocessor eXtreme) 设计，取代了 Fermi 架构的 SM。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/Kepler.png)

&emsp;&emsp;Kepler 架构还增强了图形处理能力。它支持 TXAA 抗锯齿技术，能够提供更平滑的游戏画面。此外，Kepler 还支持 NVIDIA 的 NVENC 视频编码器，能够加速视频编码过程，提高视频编辑和流媒体应用的效率。

&emsp;&emsp;在计算方面，Kepler 架构提升了单精度浮点性能，但降低了双精度浮点性能。这是因为 NVIDIA 将 Kepler 架构定位为主要面向游戏和图形应用，而这些应用对单精度浮点性能的需求更高。对于需要高双精度浮点性能的应用，NVIDIA 提供了 Tesla 系列专业卡，这些卡采用了 Kepler 架构的特殊版本，保留了较高的双精度浮点性能。

&emsp;&emsp;总的来说，Kepler 架构是一款成功的 GPU 微架构，它在能效比、图形性能和计算能力上都取得了显著的进步。它为 NVIDIA 在 GPU 领域保持领先地位奠定了坚实的基础，并推动了 GPU 在游戏、图形、人工智能等领域的应用。

#### 3.3.2 Streaming Multiprocessor (SMX)

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/kepler_smx.png)
&emsp;&emsp;SMX 单元拥有 192 个单精度 CUDA 核心，每个核心都具备完整的浮点和整数运算单元，并保留了 Fermi 架构引入的 IEEE 754-2008 标准的单精度和双精度运算，包括 FMA 操作。Kepler 的设计目标之一是显著提升双精度性能，这对于高性能计算至关重要。此外，它还保留了用于快速近似超越运算的特殊功能单元 (SFU)，数量是 Fermi GF110 SM 的 8 倍。与 GK104 SMX 单元类似，GK110/210 SMX 单元中的核心使用主 GPU 时钟，而非之前的 2 倍 Shader 时钟。Kepler 的重点是每瓦性能，因此选择使用更多的核心以较低的 GPU 时钟运行，从而优化功耗，即使这意味着增加了一些面积成本。

#### 3.3.3 Dynamic Parallelism

&emsp;&emsp;Kepler 架构引入的动态并行（Dynamic Parallelism）是一项重要的创新，它极大地提升了 GPU 的灵活性和效率。在之前的 GPU 架构中，GPU 只能由 CPU 发起 kernel 函数，并且 kernel 函数的执行是静态的，无法在 GPU 内部动态地启动新的 kernel 函数。这意味着所有的任务调度和同步都必须由 CPU 来完成，限制了 GPU 的并行能力和效率。

&emsp;&emsp;动态并行的核心思想是允许 GPU 在执行 kernel 函数的过程中，动态地启动新的 kernel 函数，而无需 CPU 的干预。这意味着 GPU 可以根据实际的计算需求，自主地进行任务调度和同步，从而更好地利用 GPU 的并行计算资源。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/kepler_dp.png)

### 3.4 Maxwell（2014-2016）

### 3.5 Pascal (2016-2017）
### 3.6 Volta (2017-2018)
### 3.7 Turing (2018-2020)
### 3.8 Ampere (2020-2022)
### 3.9 Ada Lovelace (2022-至今)
### 3.10 Blackwell (2024-)

## 4 参考文献
- [Nvidia-NV1](https://zh.wikipedia.org/wiki/NV1)
- [Nvidai-GeForce256](https://zh.wikipedia.org/zh-cn/GeForce_256)
- [Nvidia-架构](https://www.nvidia.cn/technologies/)
- [Nvidia GPU架构梳理](https://zhuanlan.zhihu.com/p/394352476)
- [NVIDIA GPU 核心与架构演进史](https://www.chenshaowen.com/blog/nvidia-gpu-cores-and-architecture-evolution-history.html)
- [Impact analysis of conditional and loop statements for the NVIDIA G80 architecture](http://www.scielo.org.co/pdf/inde/n27/n27a08.pdf)
- [GPU Architecture: the Fermi’s example](https://agenda.infn.it/event/7260/contributions/66624/attachments/48346/57204/GPU_NMR_notes_2.pdf)
- [Whitepaper NVIDIA’s Next Generation CUDATM Compute Architecture: Fermi](https://www.nvidia.com/content/pdf/fermi_white_papers/nvidiafermicomputearchitecturewhitepaper.pdf)
- [List of Fermi series GeForce GPUs](https://nvidia.custhelp.com/app/answers/detail/a_id/4656/~/list-of-fermi-series-geforce-gpus)
- [Whitepaper NVIDIA’s Next Generation CUDATM Compute Architecture: Kepler TM GK110/210](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf)
