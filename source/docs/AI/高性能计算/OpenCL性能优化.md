# OpenCL性能优化
&emsp;&emsp;GPU 优化的核心逻辑确实高度一致：尽量让计算单元（ALU）停不下来，同时别让数据传输（Memory Wall）成为瓶颈。而对应的套路基本是固定的比如：
- 利用共享内存（Shared Memory）打内存压缩；
- 保证访存合并（Memory Coalescing）；
- 消除分支分歧（Warp Divergence）；
- 流水线并行（Streams & Graphs）；
- 寄存器压力控制（Register Pressure）；
- 使用特定的硬件单元（Tensor Cores / RT Cores）。

&emsp;&emsp;下面根据一些典型的场景，来逐步拆解OpenCL GPU优化的基本方式。

## 1 直方图
