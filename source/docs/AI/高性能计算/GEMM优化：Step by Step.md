# GEMM 优化：Step by Step
---

## 1 前言

&emsp;&emsp;在深度学习模型推理中，GEMM（通用矩阵乘法）是绝大多数计算负载的核心来源。无论是 Transformer 架构中的线性层，还是卷积网络展开后的全连接计算，最终都会归结为大规模矩阵乘法运算。有研究指出，在实用序列长度下，GEMM 操作可占据模型推理总计算量的 69%–99%。因此，GEMM 的优化程度直接决定了模型推理的吞吐、延迟与能效表现，是推理系统设计中绕不开的基础问题。

### 1.1 什么是 GEMM

&emsp;&emsp;GEMM 是 **GE**neral **M**atrix **M**ultiply（通用矩阵乘法）的缩写，其数学形式为：

$$
C = \alpha \cdot A \times B + \beta \cdot C
$$

- $A$：$M \times K$ 矩阵（$M$ 行 $K$ 列）
- $B$：$K \times N$ 矩阵（$K$ 行 $N$ 列）
- $C$：$M \times N$ 矩阵（$M$ 行 $N$ 列）
- $\alpha, \beta$：标量系数

&emsp;&emsp;在绝大多数场景中，取 $\alpha=1$、$\beta=0$，即最简形式 $C = A \times B$。本文仅讨论该情形，在此基础上推广到一般情况并不困难。按照矩阵乘法的定义，$C$ 的第 $i$ 行第 $j$ 列元素为：

$$
C[i][j] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]
$$

&emsp;&emsp;也就是说，**C 的每个元素等于 A 的对应行与 B 的对应列的点积**（对应元素相乘后求和）。

---- 

&emsp;&emsp;一次规模为 $M \times N \times K$ 的矩阵乘法，其运算量为：

- **乘法次数**：$M \times N \times K$
- **加法次数**：$M \times N \times (K-1)$
- **总计浮点运算（FLOP）**：约 $2MNK$

&emsp;&emsp;例如，当 $M=N=K=4096$ 时，单精度矩阵乘法的计算量为 $2 \times 4096^3 \approx 1.37 \times 10^{11}$ FLOP，即约 **1370 亿次浮点运算**。

### 1.2 CPU 硬件对 GEMM 性能的影响

&emsp;&emsp;尽管 DDR 内存在物理上按二维方式组织，主流操作系统仍向进程暴露线性化的虚拟地址空间。因此，无论 C/C++ 中的二维数组还是显式分配的矩阵缓冲区，其元素在内存中都按一维线性排列。由此引出矩阵元素的排列顺序问题，即常见的行主序与列主序：

- **行主序（row-major）**：按行优先存储；
- **列主序（column-major）**：按列优先存储。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20260925_abf1d6.svg)

&emsp;&emsp;行主序下，元素 `A[i][j]` 在内存中的地址偏移为 `i * K + j`，其中 $K$ 为列数，称为 leading dimension（行距）。

&emsp;&emsp;现代 CPU 采用层次化存储结构，通常配备三级缓存。基于局部性原理，缓存优先保留相邻数据以加速访问，因此存储布局直接影响访存效率：若矩阵按行主序存储，访问第 $i$ 个元素时，硬件预取器会将 $i+1$ 及其后的相邻数据载入缓存，后续访问即可直接命中缓存；若矩阵按列主序存储而程序仍按行访问，预取到的 $i+1$ 并非下一步所需的数据（实际需要的是偏移 $i \times K + 1$ 处的元素），缓存行因此失效，必须重新发起内存访问。除访存外，指令执行效率同样影响性能：一条指令的执行并非瞬时完成，而是被划分为取指、译码、执行、写回等多个阶段，类似工厂中的流水线。由此引出两个关键性能指标：

- **延迟（Latency）**：单条指令从发射到结果可用所需的周期数。例如 FMA 指令的延迟通常为 4 个周期。
- **吞吐量（Throughput）**：单位时间内可发射的同类指令数，也常用“周期/条”表示。例如现代 CPU 每周期可发射 2 条 FMA，即平均 0.5 周期发射一条。

&emsp;&emsp;需要说明的是，上述数值是通用参考，具体因 CPU 型号与指令类型而异，应以目标硬件的数据手册为准。以 FMA 指令为例：若其延迟为 4 周期、发射率为每周期 2 条，则只有当后续 FMA 依赖尚未完成的前序结果时，流水线才会停顿。换言之，要填满流水线，至少需要 $4 \times 2 = 8$ 条相互独立的 FMA 指令同时在飞。

### 1.3 GPU 硬件对 GEMM 性能的影响

&emsp;&emsp;与 CPU 不同，GPU 的设计哲学并非“降低单条指令的延迟”，而是“以海量并行隐藏延迟”。理解这一点是理解 GPU 上 GEMM 优化的前提。影响 GPU 上 GEMM 性能的硬件因素可从存储层次、执行单元及二者配合方式三个维度展开：

- **存储层次与带宽**：GPU 同样具有多级存储，但其容量与带宽的比例与 CPU 差异显著：

  - **全局内存（Global Memory / HBM）**：容量大（数十 GB），但延迟高（数百周期）、带宽相对有限。GPU 上 GEMM 的首要瓶颈往往是全局内存带宽——若每个线程都直接从全局内存读取矩阵元素，再强的算力也会受制于访存。
  - **L2 缓存**：由所有 SM 共享，容量小于 CPU 的 L3，但带宽更高，用于缓解全局内存压力。
  - **共享内存（Shared Memory / SMEM）**：每个 SM（流式多处理器）私有的片上存储，容量小（通常几十 KB 到上百 KB），但延迟极低、带宽极高。GEMM 优化的核心手段之一，是将全局内存中的数据分块（tiling）搬入共享内存，使线程在片上反复复用，减少全局访存次数。
  - **寄存器（Register File）**：每个线程私有，速度最快。高性能 GEMM 内核会让每个线程在寄存器中维护一小块累加器（如 8×8），从而在循环体内只执行 FMA，无需访问其他存储层级。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20260928_df6028.svg)

&emsp;&emsp;与 CPU 的行主序/列主序问题类似，GPU 上矩阵的存储布局同样关键，但应对方式不同：GPU 通过**合并访问（coalesced access）**保证效率——当一个 warp（32 个线程）访问全局内存时，若访问地址连续，硬件会将这次访问合并为一次宽事务；若地址分散，则拆分为多次事务，带宽利用率骤降。因此，GPU 上的 GEMM 内核通常要求矩阵按特定布局排列，或在加载时通过共享内存进行转置，以配合合并访问。

- **执行单元与吞吐量**：GPU 的算力来自大量 SM，每个 SM 内部包含：
  - **CUDA Core**：执行 FP32/INT32 等常规运算。FMA 指令在 CUDA Core 上执行，延迟同样约为 4 周期，但每个 SM 每周期可发射多条 FMA（具体数量取决于架构，如 Ampere 上每个 SM 每周期可发射 64 条 FP32 FMA）。
  - **Tensor Core**：专为矩阵乘法设计的单元，单条指令即可完成一个小矩阵块（如 16×16×16）的乘加。Tensor Core 的吞吐量远高于 CUDA Core，是现代 GPU 上 GEMM 性能的主要来源。但其使用存在约束：需要特定的数据类型（FP16、BF16、TF32、INT8 等）、特定的数据布局（如 wmma/mma 所要求的 fragment 排布），并且需要软件显式调用。
  - **特殊函数单元（SFU）**：处理超越函数等运算，与 GEMM 关系不大。

&emsp;&emsp;这里同样存在 1.2 节所述的“延迟与吞吐”权衡，只是尺度不同：GPU 上一条 Tensor Core MMA 指令的延迟可能达十几至几十周期，但吞吐量很高。要填满流水线，就需要**足够多的独立 warp** 同时驻留在 SM 上，由此引出 GPU 特有的概念：

  - **Occupancy（占用率）**：每个 SM 上活跃 warp 数与最大支持 warp 数之比。Occupancy 过低时延迟无法被隐藏，执行单元会空闲；过高时又可能因寄存器或共享内存不足而限制每个线程可用的资源，反而降低单线程效率。
  - **Warp 调度**：当某个 warp 因等待内存或依赖前一条指令而停顿时，调度器立即切换到另一个就绪 warp。GPU 正是依靠这种“以并行度换延迟隐藏”的方式，让海量线程将高延迟的访存与计算重叠起来。

&emsp;&emsp;与 1.2 节的结论相呼应：CPU 上的 GEMM 优化主要关注“如何让流水线不断流”（指令级并行、缓存预取、SIMD），GPU 上则主要关注“如何用并行度隐藏延迟、如何让数据在存储层次间高效流动”（分块、共享内存、Tensor Core、Occupancy）。两者共享同一底层逻辑——局部性原理与延迟隐藏——但具体手段因硬件架构而异。

> 本文实验硬件平台：
>
> - GPU：RTX 3050
> - CPU：AMD 5600X
>   - 缓存（全核合计）：
>     - L1d：192 KiB（6 instances）
>     - L1i：192 KiB（6 instances）
>     - L2：3 MiB（6 instances）
>     - L3：32 MiB（1 instance）
>   - NUMA：
>     - NUMA 节点数：1
>     - NUMA 节点 0 CPU：0-11

## 2 GEMM 的 CPU 优化
### 2.1 CPU GEMM 的性能模型与优化方法

&emsp;&emsp;上文已经给出了 GEMM 的数学形式，并指出单精度浮点运算量为 $2MNK$。要进一步优化 GEMM 性能，仅了解运算量还不够，还需深入理解 CPU 架构与计算模型。

&emsp;&emsp;硬件设备本身的能力是固定的，无论如何优化都无法超越硬件上限。因此，优化的目标不是“消除”硬件限制，而是在给定硬件上尽可能逼近峰值算力与可用带宽所决定的上界。为此，需要先建立可量化的性能模型。

&emsp;&emsp;最常用的分析工具是算术强度与 **Roofline 模型**。算术强度定义为浮点运算次数与访存字节数之比：
$$
I=\frac{\text{FLOP}}{\text{Byte}}
$$

&emsp;&emsp;对 GEMM，运算量为 $2MNK$。若只考虑 DRAM 流量，至少需要读取 A、B 并写回 C，数据量约为 $4(MK+KN+MN)$ 字节。方阵 $M=N=K=n$ 时，$I_{\text{DRAM}}\approx 2n^3/[4(3n^2)]=n/6$。随着 n 增大，算术强度线性增长。

&emsp;&emsp;Roofline 模型给出性能上界：
$$
P\le \min(P_{\text{peak}}, I\cdot BW)
$$


![](https://thumb.wikimedia.org/wikipedia/commons/thumb/b/b1/Example_of_a_Roofline_model.svg/1920px-Example_of_a_Roofline_model.svg.png)

&emsp;&emsp;转折点 $I^*=P_{\text{peak}}/BW$。当 $I<I^*$ 时，性能受带宽限制；当 $I>I^*$ 时，性能受计算能力限制。大矩阵 GEMM 在理论上属于计算受限，但实际瓶颈往往出现在 L1/L2/L3 之间的数据搬运，而非 DRAM。分块、打包和寄存器复用可以提高片上数据复用率，使有效算术强度远高于仅按 DRAM 估算的结果。

&emsp;&emsp;在计算模型之外，延迟与吞吐的区别同样关键。延迟是单条指令从发射到结果可用的周期数，吞吐是单位时间内可完成的指令数。现代 x86 CPU 的 FMA 通常延迟约 4–5 周期，吞吐可达每周期 2 条，即平均 0.5 周期发射一条。若累加器存在依赖链 $C\leftarrow C+A_kB_k$，下一次 FMA 必须等待上一次结果，单个累加器无法填满流水线。为掩盖 FMA 延迟，流水线中至少需要 $L\times T$ 条独立 FMA，其中 L 为延迟，T 为每周期发射条数。例如 L=4、T=2 时，至少需要 8 条独立 FMA。因此，GEMM 微内核通常采用 $MR\times NR$ 寄存器分块，设置多个独立累加器，并配合循环展开和软件流水，使每个 k 步更新多个向量累加器，从而隐藏 FMA 延迟并维持高吞吐。

&emsp;&emsp;缓存层次决定了数据复用的边界。典型 CPU 中，L1D 容量约 32–64KB，延迟 4–5 周期；L2 约 256KB–2MB，延迟 12–20 周期；L3 约 8–64MB，延迟 40–80 周期，且多核共享；DRAM 容量达 GB 级，延迟 200–400 周期，带宽受内存通道和 NUMA 影响。各级带宽依次递减。GEMM 优化的核心是分层分块：寄存器分块让 $MR\times NR$ 累加器常驻寄存器；L1 分块沿 K 维切出 $KC$，使 A 面板和 B 面板适配 L1；L2/L3 分块沿 M、N 维切出 $MC,NC$，减少 L3 与 DRAM 流量；打包将 A、B 子块复制为连续面板，提升 SIMD 加载和硬件预取效率。

&emsp;&emsp;预取分为**硬件预取和软件预取**。连续、规则访问易被硬件预取器识别；大步长、随机访问则效果差。软件预取可用 `_mm_prefetch` 等指令提前拉取下一块数据，但过度预取会污染缓存。带宽瓶颈不仅出现在 DRAM，也可能出现在 L1/L2/L3；多核并行时，L3 和内存带宽被共享，扩展性通常低于核数线性增长。NUMA 系统还需注意首次触摸和线程亲和性，避免远端内存访问。

&emsp;&emsp;矩阵存储顺序直接影响访问模式。以 C 语言行主序为例，$C[i,j]=\sum_k A[i,k]B[k,j]$。A 行主序时，$A[i,k]$ 沿 k 连续；B 行主序时，$B[k,j]$ 沿 j 连续，但沿 k 的步长为 N；C 行主序时，$C[i,j]$ 沿 j 连续。若最内层循环为 j，则 B 的一行连续访问，C 也连续写入，适合向量化；若最内层循环为 k，则 B 按列访问，步长为 N，每次可能触发新缓存行，带宽利用率低。列主序矩阵则相反。因此，循环顺序、分块和打包必须与矩阵布局匹配。常见做法是将 B 的 $KC\times NR$ 小块打包成连续面板，使微内核内层 k 能顺序读取；将 A 的 $MR\times KC$ 小块也打包连续。这样可消除大步长访问、降低 TLB 压力，并让硬件预取器有效工作。对于列主序矩阵，可交换 A、B 角色或进行分块转置。

&emsp;&emsp;从实现角度看，CPU GEMM 的高性能实现通常采用分层分块与微内核结构：
1. 按 M、N、K 分块，使 B、A 面板适配 L3/L2/L1；打包 A、B 子块为连续面板；
2. 微内核采用 $MR\times NR$ 寄存器分块，多累加器、SIMD FMA、循环展开；
3. 软件流水和预取隐藏 FMA 与内存延迟；
4. 多线程沿 M/N 分块，注意 NUMA 亲和与伪共享；
5. 写回 C 时按行或列布局选择连续写，必要时使用非临时存储减少写分配开销。

&emsp;&emsp;这些方法最终服务于同一个目标：用分块和打包提高算术强度，用多累加器和微内核填满 FMA 流水线，用预取和多线程隐藏访存延迟，并根据行主序或列主序选择最合适的访问模式。硬件峰值和带宽决定了性能上界，而优化方法的作用，是不断逼近这一上界。

### 2.2 朴素实现

&emsp;&emsp;GEMM 的数学定义为 $C[i,j]=\sum_{k=0}^{K-1}A[i,k]B[k,j]$。最直接的实现是三重循环逐元素计算 C：

```c
void gemm_naive(int M, int N, int K, const float *A, const float *B, float *C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i*K+k] * B[k*N+j];
            C[i*N+j] = sum;
        }
}
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261001_adec77.svg)

&emsp;&emsp;这段代码逻辑正确，但性能极低，原因有二：
- 内层 k 循环访问 `B[k * N + j]` 时，地址步长为 N。每次 k 增加 1，B 的访问就跳到下一行，跨越 N 个元素。若 N 较大，一个缓存行中往往只有一个元素被用到，其余部分无法被利用。同时，A 的第 i 行和 B 的第 j 列在 k 循环中被反复读取，但每次只使用一次，没有跨 (i,j) 的复用。整个计算过程中，A 被完整读取 N 次、B 被完整读取 M 次，总访存量约为 $4MNK+4MNK+4MN$ 字节，而浮点运算量只有 $2MNK$。算术强度约为 $2MNK/(8MNK+4MN)\approx 0.25$ FLOP/Byte，远低于 Roofline 转折点，因此性能受内存带宽限制。
- 内层循环中 `sum` 是唯一的累加变量，导致前后指令存在数据依赖，每次 FMA 都必须等待上一次 FMA 的结果。若 FMA 延迟为 4 周期、吞吐为每周期 2 条，则理想情况下需要 8 条独立 FMA 才能填满流水线，而这里只有 1 条。实际发射率可能只有峰值的几分之一甚至更低。编译器在 `-O3` 下可能尝试展开和向量化，但循环携带的依赖以及潜在的别名问题常常使优化效果有限。

### 2.3 循环交换

&emsp;&emsp;朴素实现中 B 的跨列访问是主要瓶颈。把 k 循环放到中间，让内层 j 循环连续访问 B 和 C，可以显著改善访存模式：

```c
void gemm_ikj(int M, int N, int K,
              const float *A, const float *B, float *C) {
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float a = A[i*K+k];
            const float *Brow = B + k*N;
            float *Crow = C + i*N;
            for (int j = 0; j < N; j++)
                Crow[j] += a * Brow[j];
        }
}
```

&emsp;&emsp;交换后，内层 j 循环中 `B[k * N + j]` 和 `C[i * N + j]` 都是连续访问，适合向量化，`a` 可以广播。更重要的是，内层 j 的多次更新彼此独立，不同 j 的 FMA 之间没有依赖，可以部分隐藏 FMA 延迟。这相当于把原来的“逐元素内积”改成了“逐行外积累加”：固定 i、k 时，用 A 的一个标量乘以 B 的一整行，累加到 C 的一整行。此时 C 的更新需要读-改-写，但内层 j 的连续访问使得每次读写都能利用整个缓存行。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/deepseek_svg_20261001_1b4158.svg)

&emsp;&emsp;然而 i-k-j 仍然没有解决 A 和 B 的全局复用问题。固定 i、k 时，B 的一行被读一次，A 的一个标量被读一次，随后便不再使用。若矩阵规模超过 L2/L3，A 和 B 会被反复从内存拉取，实际算术强度依然远低于 Roofline 转折点。

&emsp;&emsp;交换循环顺序后，虽然 C 矩阵的更新变为频繁的读-改-写（RMW），但 B 不再跨列跳跃访问，A 的行数据也得以顺序复用，缓存命中率显著改善。下方基准测试结果表明，这一调整效果十分明显。
![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261001_16ac4c.svg)

&emsp;&emsp;为验证上述关于缓存行为的推断，下面给出 v1 与 v2 两个版本实测的性能计数器数据：
```cpp
 Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v1/2048x2048x2048':

   229,945,405,766      cycles                                                                
    41,355,866,464      instructions                                                          
    11,890,377,327      L1-dcache-loads                                                       
     9,581,403,250      L1-dcache-load-misses                                                 
                                                
      49.306694882 seconds time elapsed

      50.302657000 seconds user
       0.578283000 seconds sys

 Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v2/2048x2048x2048':

    10,267,046,352      cycles                                                                
    10,354,571,319      instructions                                                          
     5,639,068,447      L1-dcache-loads                                                       
       610,572,828      L1-dcache-load-misses                                                 

       1.196446940 seconds time elapsed

       2.240820000 seconds user
       0.370506000 seconds sys
```

&emsp;&emsp;从 `perf stat` 的采集结果来看，naive_v1 在 2048³ 规模下的 L1-dcache miss 率高达 **80.6%**（96 亿次 miss / 119 亿次 load），而循环交换后的 naive_v2 仅为 **10.8%**（6.1 亿次 miss / 56 亿次 load），相差约 7.5 倍。这正是朴素实现中 B 矩阵跨列访问的典型特征：内层 k 循环每次访问 `B[k*N+j]` 都要跨越 N 个元素，一个 64 字节缓存行往往只用到其中 1 个 float，其余 15 个无法被利用；同时 A 的第 i 行和 B 的第 j 列在 (i,j) 双重循环中被反复加载，缺乏跨迭代的复用。循环交换将 k 提升到中间层后，内层 j 循环连续访问 B 和 C，缓存行利用率大幅提升，总 load 次数也从 119 亿次降至 56 亿次。

&emsp;&emsp;更关键的是，访存瓶颈进一步制约了指令级并行。naive_v1 的 IPC 仅为 **0.18**（413 亿条指令 / 2300 亿周期），即平均每 5.6 个周期才退休（retire）一条指令，CPU 绝大部分时间在等待内存，FMA 流水线大量空转；naive_v2 的 IPC 回升至 **1.01**，虽与理论峰值仍有距离，但已达 v1 的 5.6 倍。两者叠加的结果是：2048³ 规模下 naive_v1 耗时 48.76 秒、GFLOPS 仅 0.354，naive_v2 耗时 0.67 秒、GFLOPS 达 25.66，**性能差距高达 72 倍**。这组数据清楚地表明，GEMM 优化的首要步骤并非向量化或分块，而是通过循环交换将访存模式由“跨列跳跃”转变为“连续扫描”——仅此一项即可带来两个数量级的性能收益。

### 2.4 分块

&emsp;&emsp;要突破全局复用的限制，必须让数据在更靠近计算单元的位置被多次复用。由于矩阵计算本身具备局部独立性，可以先分块计算局部结果、最后再合并。具体而言，将 C 划分为 $MC\times NC$ 的块，A 划分为 $MC\times KC$，B 划分为 $KC\times NC$。对每个 C 块，遍历 K 维的 $KC$ 块，将对应的 A、B 子块加载到缓存中，再在缓存内完成多次乘加。这样，A 子块的每一行和 B 子块的每一列都能在 L2/L3 中被多个 C 元素复用。

```c
// 外层分块顺序：ic -> jc -> pc（M 方向最外）
static void gemm_naive_v3_block(int M, int N, int K,
                                const float *A, const float *B, float *C,
                                int MC, int NC, int KC) {
    for (int ic = 0; ic < M; ic += MC) {
        const int mc = (ic + MC <= M) ? MC : M - ic;
        for (int jc = 0; jc < N; jc += NC) {
            const int n = (jc + NC <= N) ? NC : N - jc;
            for (int pc = 0; pc < K; pc += KC) {
                const int kc = (pc + KC <= K) ? KC : K - pc;

                // 内层 ikj：j 连续访问 B 和 C，与 V2 相同
                for (int i = ic; i < ic + mc; ++i) {
                    const float *Arow = A + i * K + pc;
                    float *Crow = C + i * N + jc;
                    for (int p = 0; p < kc; ++p) {
                        const float a = Arow[p];
                        const float *Brow = B + (pc + p) * N + jc;
                        for (int j = 0; j < n; ++j)
                            Crow[j] += a * Brow[j];
                    }
                }
            }
        }
    }
}

void gemm_naive_v3(int M, int N, int K, const float *A, const float *B, float *C) {
    gemm_naive_v3_block(M, N, K, A, B, C, 256, 256, 64);
}
```

&emsp;&emsp;这里的三层分块循环对应着不同的缓存级别：最外层 jc、pc、ic 控制 L3/L2 分块，保证 A、B 面板在 L2/L3 中复用；内层 i、p、j 则是分块内的计算。分块大小需要根据缓存容量选择，例如 $MC=64$、$NC=64$、$KC=256$。分块后，算术强度从仅按 DRAM 计算的 $n/6$ 提升到按缓存容量计算的水平，理论预期可再提升 2–5 倍。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/deepseek_svg_20261001_b59684.svg)

&emsp;&emsp;但分块后的内层循环仍然每次从 A 取一个标量、从 B 取一行，数据加载较为频繁。若将 A、B 的子块预先复制到连续缓冲区，内层循环便能以更紧凑的方式读取——这正是打包（packing）的作用。从这个角度看，V2 的缓存访问模式已接近最优，V3 引入分块后逻辑更复杂，收益反而有限。

&emsp;实际运行结果也印证了这一判断，perf 采集到的数据如下：

```
 Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v3/2048x2048x2048':

    11,213,735,498      cycles
    11,322,406,868      instructions
     6,859,088,356      L1-dcache-loads
       694,658,089      L1-dcache-load-misses
```

&emsp;L1 miss 率约为 10.1%（694M / 6859M），虽与 V2 的 10.8% 接近，但 V3 的 L1 load 总数比 V2 多了约 22%（6859M vs 5639M），cycles 也多了约 9%。这说明分块并没有减少访存次数，反而因三层额外循环的边界判断与地址计算引入了更多指令。**在 V2 已经解决 L1 访存模式的前提下，分块带来的 L2/L3 复用收益不足以抵消这些额外开销，因此 V3 无法超过 V2。**

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261002_075224.svg)

### 2.5 打包

&emsp;&emsp;打包是把原本行主序或列主序的子块复制到连续缓冲区中，使得内层循环可以按顺序读取。对 A，把 $MC\times KC$ 的子块按行打包；对 B，把 $KC\times NC$ 的子块按列打包成连续面板。这样，内层每次都能取到连续的内存，消除了大步长访问，降低了 TLB 压力，也让硬件预取器能更准确地识别访问模式。

```c
// ---------- 分块 + 打包 ----------
// 外层分块顺序：jc -> pc -> ic（与 V3 一致）
// 内层：从 Ap、Bp 中按连续地址读取，ikj 顺序
static void gemm_naive_v4_block(int M, int N, int K,
                                const float *A, const float *B, float *C,
                                int MC, int NC, int KC) {
    // 打包缓冲区：A 块 MC×KC，B 块 KC×NC
    float *Ap = (float *)malloc((size_t)MC * KC * sizeof(float));
    float *Bp = (float *)malloc((size_t)KC * NC * sizeof(float));
    if (!Ap || !Bp) {
        free(Ap);
        free(Bp);
        return;
    }

    for (int jc = 0; jc < N; jc += NC) {
        const int n = (jc + NC <= N) ? NC : N - jc;
        for (int pc = 0; pc < K; pc += KC) {
            const int kc = (pc + KC <= K) ? KC : K - pc;

            // ---- 打包 B 的 KC×NC 子块 ----
            // Bp[p*n + j] = B[(pc+p)*N + jc+j]
            for (int p = 0; p < kc; ++p) {
                const float *Brow = B + (pc + p) * N + jc;
                float *Bprow = Bp + p * n;
                for (int j = 0; j < n; ++j) {
                    Bprow[j] = Brow[j]; 
                }
            }

            for (int ic = 0; ic < M; ic += MC) {
                const int mc = (ic + MC <= M) ? MC : M - ic;

                // ---- 打包 A 的 MC×KC 子块 ----
                // Ap[i*kc + p] = A[(ic+i)*K + pc+p]
                for (int i = 0; i < mc; ++i) {
                    const float *Arow = A + (ic + i) * K + pc;
                    float *Aprow = Ap + i * kc;
                    for (int p = 0; p < kc; ++p) {
                        Aprow[p] = Arow[p];
                    }
                }

                // ---- 内层计算：从 Ap、Bp 读取 ----
                for (int i = 0; i < mc; ++i) {
                    const float *Arow = Ap + i * kc;
                    float *Crow = C + (ic + i) * N + jc;
                    for (int p = 0; p < kc; ++p) {
                        const float a = Arow[p];
                        const float *Brow = Bp + p * n;
                        for (int j = 0; j < n; ++j) {
                            Crow[j] += a * Brow[j];
                        }
                    }
                }
            }
        }
    }

    free(Ap);
    free(Bp);
}

// ---------- V4 入口 ----------
void gemm_naive_v4(int M, int N, int K, const float *A, const float *B, float *C) {
    std::fill(C, C + M * N, 0.0f);
    int MC = 256;
    int NC = 256;
    int KC = 256;

    if (MC > M) MC = M;
    if (NC > N) NC = N;
    if (KC > K) KC = K;

    gemm_naive_v4_block(M, N, K, A, B, C, MC, NC, KC);
}
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/deepseek_svg_20261001_5fc5d7.svg)

&emsp;&emsp;打包后，A 和 B 的子块在内存中连续排列，内层循环的加载模式变得非常规则。此时，代码的访存效率已经接近最优，但内层仍然每次只做一个标量与一行的乘加，FMA 的发射效率还有提升空间。

&emsp;&emsp;V3 与 V4 形似而实质不同：V3 的内层循环仍在原始矩阵上按跨步地址取数，每次访问都会跳过大量无关数据；V4 则在计算前将 A、B 子块复制到连续缓冲区，使内层循环完全在紧凑布局上顺序读取。复制操作本身会带来额外开销，但它换取了后续大量内层迭代的访问效率，这正是打包（packing）的收益所在。

&emsp;&emsp;perf 数据具体揭示了 V4 的优化来源：V3 的 L1-dcache-loads 为 68.6 亿次，V4 降至 52.4 亿次，减少 23.7%；instructions 从 113.2 亿降至 86.8 亿，减少 23.4%；cycles 从 112.1 亿降至 102.5 亿，减少 8.6%。这三项指标的下降并不意味着计算量减少——FMA 次数始终是 2048³ ≈ 85.9 亿次——而是每条 FMA 周围的地址计算与循环控制指令明显减少了。V3 的内层每次都要重新计算 `B + (pc+p)*N + jc` 这样的跨步地址，乘法 `(pc+p)*N` 在循环中反复出现；V4 的地址变为 `Bp + p*n`，基址固定、步长小，编译器更容易实施强度削弱，硬件预取器也能识别出连续访问模式。

&emsp;&emsp;一个看似反直觉的现象值得注意：V4 的 IPC 从 V3 的 1.01 降至 0.85，L1 miss 率也从 10.1% 略升至 11.3%，但这并不意味着 V4 性能退化。V4 的 L1 load 总数少了近四分之一，绝对 miss 次数反而从 6.95 亿降至 5.91 亿（减少 14.9%），miss 率的分母变小、比率略升属正常现象。更重要的是，打包后的连续访问使硬件预取器能提前将下一段数据载入 L1，即使 miss 率略高，每次 miss 的代价也更低——V3 的跨步访问难以被预取器识别，每次 miss 几乎都会转化为实际停顿。IPC 的降低同样不是性能退化，而是因为省下的指令主要是地址计算与循环控制这类“辅助指令”，有效 FMA 在总指令中的占比反而提高了。

```cpp
 Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v4/2048x2048x2048':

    10,245,820,387      cycles                                                                
     8,676,657,713      instructions                                                          
     5,236,363,202      L1-dcache-loads                                                       
       590,886,887      L1-dcache-load-misses 
```

&emsp;&emsp;最终结果印证了这些微观变化：2048³ 下 V3 的 GFLOPS 只有 22.12、耗时 776.6ms，V4 达到 39.60 GFLOPS、耗时 433.8ms，性能提升约 79%。V3 解决了“数据放在哪一级缓存”的问题，但内层仍在原始矩阵里跨步取数；V4 在此基础上解决了“数据以什么布局被读取”的问题，把子块搬到连续缓冲区，让内层循环的地址计算更简单、访存更连续。代价是打包复制本身的开销，收益是 load 次数和指令数各降约四分之一，最终换来接近翻倍的性能。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261002_ec00bf.svg)

### 2.6 寄存器分块与微内核

&emsp;&emsp;经过分块和打包，A、B 子块已能以连续方式读取，但内层循环每次仍然只更新 C 的一行，累加器数量不足，FMA 依赖链依然存在。解决办法是取 $MR\times NR$ 的小块，例如 $MR=8$、$NR=6$，用 48 个寄存器保存 C 的累加值。微内核沿 K 维循环，每次加载 A 的 $MR$ 个元素和 B 的 $NR$ 个元素，执行 $MR\times NR$ 次 FMA。这 48 个累加器彼此独立，FMA 之间没有依赖链，可以持续填满流水线。

```cpp
// ---------- 主循环：分块 + 打包 + 微内核 ----------
static void gemm_micro(int M, int N, int K,
                       const float *A, const float *B, float *C,
                       int MC, int NC, int KC) {
    float *Ap = (float *)malloc((size_t)MC * KC * sizeof(float));
    float *Bp = (float *)malloc((size_t)KC * NC * sizeof(float));
    if (!Ap || !Bp) {
        free(Ap);
        free(Bp);
        return;
    }

    for (int jc = 0; jc < N; jc += NC) {
        const int n = (jc + NC <= N) ? NC : N - jc;
        for (int pc = 0; pc < K; pc += KC) {
            const int kc = (pc + KC <= K) ? KC : K - pc;

            // ---- 打包 B 的 KC×NC 子块 ----
            for (int p = 0; p < kc; ++p) {
                const float *Brow = B + (pc + p) * N + jc;
                float *Bprow = Bp + p * n;
                for (int j = 0; j < n; ++j)
                    Bprow[j] = Brow[j];
            }

            for (int ic = 0; ic < M; ic += MC) {
                const int mc = (ic + MC <= M) ? MC : M - ic;

                // ---- 打包 A 的 MC×KC 子块 ----
                for (int i = 0; i < mc; ++i) {
                    const float *Arow = A + (ic + i) * K + pc;
                    float *Aprow = Ap + i * kc;
                    for (int p = 0; p < kc; ++p)
                        Aprow[p] = Arow[p];
                }

                // ---- 微内核主块 ----
                int ir = 0;
                for (; ir + MR <= mc; ir += MR) {
                    int jr = 0;
                    for (; jr + NR <= n; jr += NR) {
                        micro_kernel(kc, n,
                                     Ap + ir * kc,
                                     Bp + jr,
                                     C + (ic + ir) * N + (jc + jr),
                                     N);
                    }
                    // NR 方向的边角：用标量处理
                    for (; jr < n; ++jr) {
                        float *Ccol = C + (ic + ir) * N + (jc + jr);
                        for (int ii = 0; ii < MR; ++ii) {
                            const float *Arow = Ap + (ir + ii) * kc;
                            float acc = Ccol[ii * N];
                            for (int p = 0; p < kc; ++p)
                                acc += Arow[p] * Bp[p * n + jr];
                            Ccol[ii * N] = acc;
                        }
                    }
                }

                // ---- MR 方向的边角：用标量处理 ----
                for (; ir < mc; ++ir) {
                    const float *Arow = Ap + ir * kc;
                    float *Crow = C + (ic + ir) * N + jc;
                    for (int p = 0; p < kc; ++p) {
                        const float a = Arow[p];
                        const float *Brow = Bp + p * n;
                        for (int j = 0; j < n; ++j)
                            Crow[j] += a * Brow[j];
                    }
                }
            }
        }
    }

    free(Ap);
    free(Bp);
}

// ---------- V5 入口 ----------
void gemm_naive_v5(int M, int N, int K, const float *A, const float *B, float *C) {
    std::fill(C, C + M * N, 0.0f);

    // 分块参数：按单核 L1d = 32 KiB、L2 = 512 KiB 估算
    // A 块 MC×KC×4 + B 块 KC×NC×4 + C 块 MC×NC×4 ≤ L2 × 0.5
    int MC = 64;
    int NC = 64;
    int KC = 32;

    gemm_micro(M, N, K, A, B, C, MC, NC, KC);
}

```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/deepseek_svg_20261001_465a90.svg)

&emsp;&emsp;微内核通过 48 个独立累加器填满了 FMA 流水线。若 FMA 延迟为 4 周期、吞吐为 2 条/周期，理论上只需要 8 条独立 FMA，而 48 个累加器提供的并行度远远超过这一要求。此时单核性能已经接近标量峰值，但每个 FMA 仍然只处理一个浮点数。

&emsp;&emsp;在 2048³ 规模下，V5 的 CPU 时间降至 574.7ms，GFLOPS 达到 29.90，在三个规模上全面反超 V2（633.9ms、27.10 GFLOPS）与 V4（690.7ms、24.87 GFLOPS）：V5 较 V2 快约 9.3%，较 V4 快约 17%。这一提升并非来自计算量的减少——2048³ 的 FMA 次数始终是 85.9 亿次——而是因为 CPU 以更少的周期完成了同样的工作：V5 的 cycles 为 108.4 亿，V2 约 112 亿。

&emsp;&emsp;改善最显著的微观指标是 L1 miss 率。V5 的 L1-dcache-loads 为 49.6 亿次，L1-dcache-load-misses 为 1.47 亿次，miss 率约 **2.96%**。对比 V2 的 10.8%、V3 的 10.1%、V4 的 11.3%，这是一个数量级的改善。原因在于微内核通过 `_mm256_loadu_ps` 一次加载 8 个 float，使 B 的缓存行利用率从标量版本的 37.5% 提升至接近 100%，miss 次数随之大幅下降。而 V2 虽然通过循环交换解决了 B 的跨列访问问题，但内层仍是标量操作，每个缓存行只用到其中一部分，miss 率停在 10% 以上。这也解释了为何 V5 能在 cycles 更少的情况下完成同样的计算——它将等待内存的时间转化成了有效的 FMA 发射。

&emsp;&emsp;V5 的 instructions 为 123.6 亿，比 V2 的 103.5 亿多约 19%。新增指令主要来自打包（复制 A、B 子块）和微内核的循环控制。但 V5 的 cycles 反而比 V2 更少，说明这些新增指令是**高效指令**——它们没有引发额外停顿，反而使 FMA 单元得到更充分的利用。IPC 约 1.14，虽不算高，但优于 V3 的 1.01 和 V4 的 0.85，说明 CPU 既没有在等待内存，也没有空转。这一点在性能优化中至关重要：指令数量多并不意味着性能差——只要新增指令有助于填满流水线，其性能反而优于指令较少但频繁停顿的版本。

```cpp
Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v5/2048x2048x2048':

10,844,283,019      cycles                                                                
12,361,764,878      instructions                                                          
    4,961,292,893      L1-dcache-loads                                                       
    147,025,797      L1-dcache-load-misses  
```

&emsp;&emsp;从优化路径看，V2 到 V5 经历了四步：循环交换解决 L1 访存模式，分块改善 L2/L3 复用，打包将子块搬入连续缓冲区，微内核配合 intrinsic 将内层访存向量化。每一步单独看收益有限，但叠加之后，V5 在 2048³ 下比 V2 快约 9%、比 V4 快约 17%。L1 miss 率从 10.8% 降至 2.96% 是这一轮优化的核心成果，也是 V5 最终反超的关键。这组数据说明，当访存模式经过循环交换与打包改造之后，**下一步的瓶颈不再是“读得少”，而是“读得宽”**——用一条向量加载替代八条标量加载，正是 miss 率降至 3% 以下、GFLOPS 突破 29 的直接原因。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261002_59acc3.svg)

### 2.7 SIMD 向量化

&emsp;&emsp;现代 x86 CPU 支持 AVX2 或 AVX-512，单条 FMA 指令可以作用在 8 个或 16 个单精度浮点数上。把微内核中的 NR 设为 SIMD 宽度的整数倍，用 `_mm256_fmadd_ps` 等内在函数替代标量 FMA，可以让吞吐成倍提升。以下代码使用 AVX2，编译时需加 `-mavx2 -mfma`。

```c
#define MR 8
#define NR 8

static inline void micro_kernel_avx2(int kc, int n,
                                     const float * __restrict Ap,
                                     const float * __restrict Bp,
                                     float * __restrict C, int ldc) {
    // 6 个 YMM 累加器，每个存 8 个 float
    __m256 c[MR];
    for (int i = 0; i < MR; ++i)
        c[i] = _mm256_loadu_ps(C + i * ldc);

    for (int p = 0; p < kc; ++p) {
        __m256 b = _mm256_loadu_ps(Bp + p * n);   // 一次加载 8 个 B 元素
        for (int i = 0; i < MR; ++i) {
            __m256 a = _mm256_set1_ps(Ap[i * kc + p]);  // 广播 A 的标量
            c[i] = _mm256_fmadd_ps(a, b, c[i]);          // 向量 FMA
        }
    }

    for (int i = 0; i < MR; ++i)
        _mm256_storeu_ps(C + i * ldc, c[i]);
}

// ---------- 主循环：分块 + 打包 + AVX2 微内核 ----------
static void gemm_micro_avx2(int M, int N, int K,
                            const float *A, const float *B, float *C,
                            int MC, int NC, int KC) {
    float *Ap = (float *)aligned_alloc(32, (size_t)MC * KC * sizeof(float));
    float *Bp = (float *)aligned_alloc(32, (size_t)KC * NC * sizeof(float));
    if (!Ap || !Bp) {
        free(Ap);
        free(Bp);
        return;
    }

    for (int jc = 0; jc < N; jc += NC) {
        const int n = (jc + NC <= N) ? NC : N - jc;
        for (int pc = 0; pc < K; pc += KC) {
            const int kc = (pc + KC <= K) ? KC : K - pc;

            // ---- 打包 B 的 KC×NC 子块 ----
            for (int p = 0; p < kc; ++p) {
                const float *Brow = B + (pc + p) * N + jc;
                float *Bprow = Bp + p * n;
                for (int j = 0; j < n; ++j)
                    Bprow[j] = Brow[j];
            }

            for (int ic = 0; ic < M; ic += MC) {
                const int mc = (ic + MC <= M) ? MC : M - ic;

                // ---- 打包 A 的 MC×KC 子块 ----
                for (int i = 0; i < mc; ++i) {
                    const float *Arow = A + (ic + i) * K + pc;
                    float *Aprow = Ap + i * kc;
                    for (int p = 0; p < kc; ++p)
                        Aprow[p] = Arow[p];
                }

                // ---- AVX2 微内核主块 ----
                int ir = 0;
                for (; ir + MR <= mc; ir += MR) {
                    int jr = 0;
                    for (; jr + NR <= n; jr += NR) {
                        micro_kernel_avx2(kc, n,
                                          Ap + ir * kc,
                                          Bp + jr,
                                          C + (ic + ir) * N + (jc + jr),
                                          N);
                    }
                    // NR 方向边角：标量兜底
                    for (; jr < n; ++jr) {
                        float *Ccol = C + (ic + ir) * N + (jc + jr);
                        for (int ii = 0; ii < MR; ++ii) {
                            const float *Arow = Ap + (ir + ii) * kc;
                            float acc = Ccol[ii * N];
                            for (int p = 0; p < kc; ++p)
                                acc += Arow[p] * Bp[p * n + jr];
                            Ccol[ii * N] = acc;
                        }
                    }
                }

                // MR 方向边角：标量兜底
                for (; ir < mc; ++ir) {
                    const float *Arow = Ap + ir * kc;
                    float *Crow = C + (ic + ir) * N + jc;
                    for (int p = 0; p < kc; ++p) {
                        const float a = Arow[p];
                        const float *Brow = Bp + p * n;
                        for (int j = 0; j < n; ++j)
                            Crow[j] += a * Brow[j];
                    }
                }
            }
        }
    }

    free(Ap);
    free(Bp);
}

// ---------- V6 入口 ----------
void gemm_naive_v6(int M, int N, int K, const float *A, const float *B, float *C) {
    std::fill(C, C + M * N, 0.0f);

    // Zen 3 的 L1d = 32 KiB，L2 = 512 KiB
    // MR=6, NR=8 下，KC 可以取大一点，减少打包次数
    int MC = 64;
    int NC = 64;
    int KC = 128;

    if (MC > M) MC = M;
    if (NC > N) NC = N;
    if (KC > K) KC = K;

    gemm_micro_avx2(M, N, K, A, B, C, MC, NC, KC);
}
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/deepseek_svg_20261001_b692d2.svg)

&emsp;&emsp;调用 AVX2 微内核的主循环与 2.6 节类似，只需将 `micro_kernel` 替换为 `micro_kernel_avx2`，并将 NR 改为 8。AVX2 版本通常能在标量微内核基础上再提升 3–6 倍。至此单核性能已接近峰值，剩余问题是单核算力有限，需要将工作负载分摊到多个核心上。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261003_30c2ea.svg)

&emsp;&emsp;在 2048³ 规模下，V6 的 CPU 时间降至 285.0ms，GFLOPS 达到 60.28，较 V5 的 530.5ms、32.39 GFLOPS 快近一倍，较 V2 的 513.4ms、33.46 GFLOPS 快约 80%。这一提升源自微内核的彻底重写：V6 不再依赖编译器自动向量化，而是通过 AVX2 intrinsic 显式加载 8 个 float、广播 A 的标量并执行向量 FMA。perf 数据也印证了这一点：L1-dcache-loads 从 V5 的 49.6 亿次增至 83.7 亿次，L1-dcache-load-misses 也从 1.47 亿次增至 4.51 亿次，L1 访存指标反而有所劣化。

&emsp;&emsp;V6 的性能提升主要来自 **FMA 发射效率**，而非 L1 miss 率。V6 的 instructions 从 V5 的 123.6 亿增至 159.6 亿，涨幅 29%，但 cycles 仅从 108.4 亿增至 112.6 亿，IPC 从 1.14 升至 1.42。这说明 CPU 在执行更多指令的同时，周期数并未同比增加，流水线利用率得到了提升。

```cpp
 Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v6/2048x2048x2048':

    11,261,600,769      cycles                                                                
    15,959,752,975      instructions                                                          
     8,366,425,515      L1-dcache-loads                                                       
       451,003,500      L1-dcache-load-misses 
```

&emsp;&emsp;**为什么 L1 miss 率反而上升？** V6 的 MR=8、NR=8 微内核一次加载 8 个 B 元素（32 字节），而缓存行为 64 字节，单次加载仅覆盖一半；V5 虽同样按 8 个 float 加载，但其微内核在编译器优化下可能形成了更紧凑的访存序列。V6 的 `_mm256_loadu_ps` 每次加载 32 字节，跨缓存行的概率高于 V5 的标量加载，因此 miss 率略有上升。

&emsp;&emsp;从优化路径看，V6 是首个将 GFLOPS 推过 60 的版本，其核心改动是：**以 AVX2 intrinsic 替代标量微内核，以 `_mm256_fmadd_ps` 替代逐元素 FMA，以 `_mm256_set1_ps` 完成标量广播**。这些改动使每条指令一次处理 8 个 float：FMA 吞吐从标量版本的每周期 1–2 条（每条 1 个 float）变为向量版本的每周期 1 条（每条 8 个 float）。在 5600X 的 Zen 3 架构上，256-bit FMA 会被拆分为两个 128-bit uop，因此实际吞吐相当于每周期 4 个 float 的 FMA 运算——比 V5 标量版本的每周期 1–2 个 float 快一倍以上，与实测 1.8 倍的提升相吻合。



### 2.8 多线程与 NUMA

&emsp;&emsp;GEMM 的并行性天然存在于 M 和 N 方向，可以把 C 的不同块分配给不同线程。线程应尽量绑定到 NUMA 节点本地内存，避免远端访问；同时要防止多个线程写同一缓存行造成伪共享。以下代码用 OpenMP 对最外层 jc 循环并行化，编译时需加 `-fopenmp`。

```c
#include <omp.h>
#include <immintrin.h>
#include <stdlib.h>

#define MR 8
#define NR 8

void micro_kernel_avx2(int kc, int n,
                       const float *Ap, const float *Bp,
                       float *C, int ldc) {
    __m256 c[MR];
    for (int i = 0; i < MR; i++)
        c[i] = _mm256_loadu_ps(C + i*ldc);
    for (int p = 0; p < kc; p++) {
        const float *Brow = Bp + p*n;
        __m256 b = _mm256_loadu_ps(Brow);
        for (int i = 0; i < MR; i++) {
            __m256 a = _mm256_set1_ps(Ap[i*kc + p]);
            c[i] = _mm256_fmadd_ps(a, b, c[i]);
        }
    }
    for (int i = 0; i < MR; i++)
        _mm256_storeu_ps(C + i*ldc, c[i]);
}

void gemm_mt(int M, int N, int K,
             const float *A, const float *B, float *C,
             int MC, int NC, int KC) {
    #pragma omp parallel
    {
        float *Ap = malloc((size_t)MC*KC*sizeof(float));
        float *Bp = malloc((size_t)KC*NC*sizeof(float));

        #pragma omp for schedule(dynamic)
        for (int jc = 0; jc < N; jc += NC) {
            int n = (jc + NC <= N) ? NC : N - jc;
            for (int pc = 0; pc < K; pc += KC) {
                int kc = (pc + KC <= K) ? KC : K - pc;
                for (int p = 0; p < kc; p++)
                    for (int j = 0; j < n; j++)
                        Bp[p*n + j] = B[(pc+p)*N + jc + j];

                for (int ic = 0; ic < M; ic += MC) {
                    int mc = (ic + MC <= M) ? MC : M - ic;
                    for (int i = 0; i < mc; i++)
                        for (int p = 0; p < kc; p++)
                            Ap[i*kc + p] = A[(ic+i)*K + pc + p];

                    for (int ir = 0; ir + MR <= mc; ir += MR)
                        for (int jr = 0; jr + NR <= n; jr += NR)
                            micro_kernel_avx2(kc, n,
                                              Ap + ir*kc,
                                              Bp + jr,
                                              C + (ic+ir)*N + (jc+jr),
                                              N);

                    for (int ir = (mc/MR)*MR; ir < mc; ir++)
                        for (int p = 0; p < kc; p++) {
                            float a = Ap[ir*kc + p];
                            const float *Brow = Bp + p*n;
                            float *Crow = C + (ic+ir)*N + jc;
                            for (int j = 0; j < n; j++)
                                Crow[j] += a * Brow[j];
                        }
                }
            }
        }
        free(Ap); free(Bp);
    }
}
```

&emsp;&emsp;并行化后，性能通常可接近单核的线程数倍，但受限于共享 L3 与内存带宽，扩展效率会略低于核数的线性增长。若系统为 NUMA 架构，可用 `numactl --localalloc` 或 `OMP_PROC_BIND` 将线程绑定到本地节点，进一步减少远端访问。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/deepseek_svg_20261001_d317e3.svg)


&emsp;&emsp;在 2048³ 规模下，V7 的单次端到端耗时为 **66.98ms**，而 CPU 时间仅为 **51.77ms**，GFLOPS 达到 **331.84**。`Time` 与 `CPU` 的差异（66.98 vs 51.77）表明多线程引入了额外的调度开销——`real_time` 包含线程启动、同步与等待的墙钟时间，而 `cpu_time` 是各线程实际占用 CPU 时间的总和。在 12 线程下，cpu_time / real_time ≈ 0.77，说明约 23% 的墙钟时间消耗在等待与调度上，这是多线程版本的典型开销。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/cpu_gemm_v7_xxxxxxxxxxxxxxx_bench_result.png)


```cpp
Benchmark                                 Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------
BM_sgemm/naive_v7/2048x2048x2048   66984193 ns     51771392 ns           12 GFLOPS=331.841G/s

 Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v7/2048x2048x2048':

    39,405,153,162      cycles                                                                
    44,106,297,329      instructions                                                          
    18,020,714,506      L1-dcache-loads                                                       
     2,574,376,694      L1-dcache-load-misses 
```

&emsp;&emsp;从吞吐看，V7 的 331.84 GFLOPS 为 V6 单核 63.01 GFLOPS 的 **5.27 倍**。12 核理论上可达 12 倍，实际仅达到 5.27 倍，原因可从 perf 数据中看出：cycles 为 394.05 亿，instructions 为 441.06 亿，IPC 约 **1.12**，低于 V6 单核的 1.42。多线程下 IPC 下降属正常现象——多个线程共享 L3（32MB）与内存带宽，缓存争用与带宽饱和降低了每个线程的指令退休效率。

&emsp;&emsp;L1-dcache-loads 为 180.2 亿次，L1-dcache-misses 为 25.74 亿次，miss 率约 **14.3%**，明显高于 V6 单核的 5.4%。原因是 12 个线程同时运行时，每个线程的 `Ap`、`Bp` 缓冲区合计约 786KB，再加上 A、B、C 各 16MB 的共享数据，L1 与 L2 容量被大量占用；线程切换又反复打断缓存局部性，miss 率随之上升。这也解释了为何 V7 的加速比只有 5.27 倍而非接近 12 倍——**内存带宽与缓存争用是多线程阶段的主要瓶颈**。

### 2.9 TLB 优化
&emsp;&emsp;V2–V7 涵盖了理论上通用的优化手段。若想在特定机器上进一步逼近性能极限，首先必须明确当前瓶颈所在。上一节推测瓶颈来自多线程下的内存冲突，但该推测需要用 perf 采集的硬件计数器数据加以验证。

&emsp;&emsp;首先观察缓存填充的数据来源分布：

```cpp
❯ perf stat -e cycles,instructions,\
  ls_dmnd_fills_from_sys.lcl_l2,\
  ls_dmnd_fills_from_sys.int_cache,\
  ls_dmnd_fills_from_sys.ext_cache_local,\
  ls_dmnd_fills_from_sys.ext_cache_remote,\
  ls_dmnd_fills_from_sys.mem_io_local,\
  ls_dmnd_fills_from_sys.mem_io_remote \
      ./build/bench_gemm --benchmark_filter=BM_sgemm/naive_v7/2048x2048x2048

BM_sgemm/naive_v7/2048x2048x2048   62720080 ns     54694412 ns           10 GFLOPS=314.106G/s

 Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v7/2048x2048x2048':

    33,655,504,090      cycles                                                                  (62.52%)
    37,843,997,868      instructions                                                            (62.63%)
       982,470,647      ls_dmnd_fills_from_sys.lcl_l2                                           (62.52%)
        95,814,078      ls_dmnd_fills_from_sys.int_cache                                        (62.49%)
                 0      ls_dmnd_fills_from_sys.ext_cache_local                                        (62.53%)
                 0      ls_dmnd_fills_from_sys.ext_cache_remote                                        (62.59%)
         7,474,580      ls_dmnd_fills_from_sys.mem_io_local                                        (62.53%)
                 0      ls_dmnd_fills_from_sys.mem_io_remote                                        (62.53%)

       1.438872841 seconds time elapsed

       8.752584000 seconds user
       0.462284000 seconds sys
```

&emsp;&emsp;缓存填充数据给出了明确的结论：V7 的瓶颈既不在内存带宽，也不在 NUMA。在 10.78 亿次数据缓存填充中，**91% 来自本地 L2**，8.9% 来自 L3 或同 CCX 的其他 L2，仅 **0.7% 来自内存**，而远端内存与远端 CCX 缓存均为 0。这意味着绝大多数 L1 miss 在 L2 即被满足，并未触及 DRAM。若内存带宽是瓶颈，来自内存的填充比例应在 10% 以上；若 NUMA 分配存在问题，`mem_io_remote` 与 `ext_cache_remote` 也不会为 0。5600X 为单 CCD、单 NUMA 节点，NUMA 优化空间为零，`numactl --interleave` 之类的操作不会带来任何收益。真正的问题在于 **L1 到 L2 之间的流量**：每个线程的打包缓冲区 `Ap`、`Bp` 合计约 64KB，已超出 L1d 的 32KB 容量，导致微内核读取打包数据时频繁触发 L1 miss，miss 后又需向 L2 发起请求。10.78 亿次 L2 填充对应此前测得的 25.74 亿次 L1 miss，说明约 42% 的 L1 miss 转化为 L2 请求。这就是当前的核心瓶颈——**问题并非数据无法取回，而是 L1 容纳不下打包缓冲区，致使 L1 与 L2 之间的带宽被反复占用**。下一步的优化方向应从“减少内存流量”转向“减少 L1 到 L2 的流量”：将 `KC` 从 128 降至 64，使 `Ap`、`Bp` 各缩减至 16KB、合计 32KB，恰好可容纳于 L1；同时将打包缓冲区按 64 字节对齐，减少跨缓存行访问。若 `KC=64` 后 L1 miss 率明显下降、GFLOPS 提升，即可验证该方向正确。

&emsp;&emsp;其次观察 L2 请求量与 L2 填充延迟：
```
perf stat -e cycles,instructions,\
l2_request_g1.all_no_prefetch,\
l2_cache_req_stat.ls_rd_blk_c,\
l2_cache_req_stat.ls_rd_blk_l_hit_x,\
l2_cache_req_stat.ls_rd_blk_l_hit_s,\
l2_fill_pending.l2_fill_busy \
./build/bench_gemm --benchmark_filter=BM_sgemm/naive_v7/2048x2048x2048

BM_sgemm/naive_v7/2048x2048x2048   66692460 ns     58193808 ns           13 GFLOPS=295.218G/s

 Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v7/2048x2048x2048':

    40,448,824,653      cycles                                                                  (71.36%)
    47,154,001,042      instructions                                                            (71.37%)
     2,819,848,946      l2_request_g1.all_no_prefetch                                           (71.38%)
       105,280,532      l2_cache_req_stat.ls_rd_blk_c                                           (71.46%)
     2,224,402,324      l2_cache_req_stat.ls_rd_blk_l_hit_x                                        (71.59%)
       187,805,012      l2_cache_req_stat.ls_rd_blk_l_hit_s                                        (71.53%)
     7,935,552,845      l2_fill_pending.l2_fill_busy                                            (71.40%)

       1.631469061 seconds time elapsed

      10.888608000 seconds user
       0.402694000 seconds sys
```

&emsp;&emsp;L2 数据将 V7 的瓶颈定位得更为精确：**L2 命中率约 85.4%，L2 miss 仅 1.05 亿次，但 `l2_fill_pending.l2_fill_busy` 高达 79.4 亿周期，占 cycles 的 19.6%**。这说明瓶颈不在 L2 命中率——L2 本身工作良好，85% 的请求在 L2 内即被满足，1.05 亿次 miss 中绝大多数也在 L3 命中，真正回内存的仅 747 万次。问题在于 **L2 的请求吞吐量**：28.2 亿次 L2 请求使填充队列（MAB）趋于饱和，`l2_fill_busy` 的 79.4 亿周期意味着 L2 近 20% 的时间在处理未完成的填充请求，新请求需排队等待。

&emsp;&emsp;该现象与 L1 侧数据相互印证：L1 共有 180 亿次 load，其中约 25.7 亿次 miss（14.3%），这些 miss 几乎全部到达 L2，构成 28.2 亿次 L2 请求的主体。这一 miss 率在 12 线程并行下被放大后，足以使 L2 的请求队列饱和。根因在于打包缓冲区的尺寸：每个线程的 `Ap`、`Bp` 合计约 64KB，超过 L1d 的 32KB；微内核每处理一个 6×8 的 C 块，就要从 `Ap` 读取 6×KC 个 float、从 `Bp` 读取 KC×8 个 float，访存/计算比过高，导致 L1 频繁 miss。KC=128 时，每个微内核需读取 1792 个 float 却仅计算 48 个 C 元素，这些数据无法全部驻留 L1，只能反复向 L2 请求。

&emsp;&emsp;接着考察 TLB 的覆盖情况：

```cpp
>perf stat -e cycles,instructions,\
ls_l1_d_tlb_miss.all,\
ls_l1_d_tlb_miss.tlb_reload_4k_l2_hit,\
ls_l1_d_tlb_miss.tlb_reload_4k_l2_miss,\
ls_l1_d_tlb_miss.tlb_reload_2m_l2_hit,\
ls_l1_d_tlb_miss.tlb_reload_2m_l2_miss,\
ls_tablewalker.dside \
./build/bench_gemm --benchmark_filter=BM_sgemm/naive_v7/2048x2048x2048

BM_sgemm/naive_v7/2048x2048x2048   72582425 ns     59008192 ns           12 GFLOPS=291.144G/s

 Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v7/2048x2048x2048':

    38,679,603,202      cycles                                                                  (62.87%)
    43,580,809,423      instructions                                                            (62.63%)
        43,789,432      ls_l1_d_tlb_miss.all                                                    (62.49%)
        10,321,413      ls_l1_d_tlb_miss.tlb_reload_4k_l2_hit                                        (62.30%)
        33,046,728      ls_l1_d_tlb_miss.tlb_reload_4k_l2_miss                                        (62.31%)
           477,559      ls_l1_d_tlb_miss.tlb_reload_2m_l2_hit                                        (62.36%)
           150,270      ls_l1_d_tlb_miss.tlb_reload_2m_l2_miss                                        (62.60%)
        34,072,859      ls_tablewalker.dside                                                    (62.73%)

       1.786346488 seconds time elapsed

      10.495368000 seconds user
       0.546396000 seconds sys
```

&emsp;&emsp;TLB 数据将 V7 的访存瓶颈进一步下沉到地址翻译环节：**L1 DTLB miss 共 4379 万次，其中 3305 万次为 4K 页且 L2 TLB 同样 miss（`tlb_reload_4k_l2_miss`），占总 miss 的 75.5%**；相比之下，2M 大页的 L2 TLB miss 仅 15 万次。更关键的是 `ls_tablewalker.dside` 达 3407 万次，说明几乎每一次 4K 页的 L2 TLB miss 都触发了一次硬件页表遍历。该数字远高于普通计算负载的 TLB miss 水平，表明当前工作集已超出 TLB 的覆盖范围。

&emsp;&emsp;然后考察 FP 流水线的分配情况（用于判断 FMA 发射效率）：

```cpp
❯ perf stat -e cycles,instructions,\
  fp_ret_sse_avx_ops.all,\
  fp_ret_sse_avx_ops.mac_flops,\
  fpu_pipe_assignment.total0,\
  fpu_pipe_assignment.total1,\
  fpu_pipe_assignment.total2,\
  fpu_pipe_assignment.total3 \
      ./build/bench_gemm --benchmark_filter=BM_sgemm/naive_v7/2048x2048x2048

BM_sgemm/naive_v7/2048x2048x2048   60390666 ns     56577945 ns           10 GFLOPS=303.65G/s

 Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v7/2048x2048x2048':

    34,428,196,591      cycles                                                                  (62.70%)
    38,100,621,967      instructions                                                            (62.50%)
   189,842,679,235      fp_ret_sse_avx_ops.all                                                  (49.90%)
   188,717,586,731      fp_ret_sse_avx_ops.mac_flops                                            (37.46%)
     7,320,237,745      fpu_pipe_assignment.total0                                              (37.63%)
     4,554,996,498      fpu_pipe_assignment.total1                                              (37.73%)
        97,935,046      fpu_pipe_assignment.total2                                              (50.26%)
        73,154,595      fpu_pipe_assignment.total3                                              (50.20%)

       1.424528639 seconds time elapsed

       8.755564000 seconds user
       0.465488000 seconds sys
```

&emsp;&emsp;FP 管道数据得出明确结论：**V7 的浮点单元已接近饱和，FMA 发射效率不再是瓶颈**。`fp_ret_sse_avx_ops.mac_flops` 为 1887 亿次 MAC FLOPs，对应约 943 亿次 MAC 操作。2048³ 的理论 FLOP 为 171.8 亿次，12 线程累加后约 2062 亿次 FLOP，实测 1887 亿次与之吻合，表明 FMA 指令几乎全部正常退休，未因流水线停顿而被丢弃或重复执行。更关键的是 `fpu_pipe_assignment` 的分布：pipe0 为 73.2 亿 uop，pipe1 为 45.5 亿 uop，两者合计 118.7 亿，占总 FP uop 的 99.2%，而 pipe2 与 pipe3 分别仅有 0.98 亿和 0.73 亿。在 Zen 3 上，FMA 只能在 pipe0 和 pipe1 上执行，pipe2/pipe3 负责其他浮点操作，该分布说明几乎所有浮点操作都是 FMA，且集中分布在两个 FMA 管道上。以 FP uop 总数除以总周期数，平均每周期约 3.48 个 FP uop，接近 Zen 3 FP 单元的理论峰值（每周期 4 个 128-bit uop），**实测利用率约 87%，已接近饱和**。

&emsp;&emsp;最后观察调度器停顿事件：
```
❯ perf stat -e cycles,instructions,\
  de_dis_dispatch_token_stalls1.fp_reg_file_rsrc_stall,\
  de_dis_dispatch_token_stalls1.fp_sch_rsrc_stall,\
  de_dis_dispatch_token_stalls1.load_queue_rsrc_stall,\
  de_dis_dispatch_token_stalls1.store_queue_rsrc_stall,\
  de_dis_dispatch_token_stalls2.int_sch0_token_stall,\
  de_dis_dispatch_token_stalls2.int_sch1_token_stall \
      ./build/bench_gemm --benchmark_filter=BM_sgemm/naive_v7/2048x2048x2048
BM_sgemm/naive_v7/2048x2048x2048   65352676 ns     57555934 ns           12 GFLOPS=298.49G/s

 Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v7/2048x2048x2048':

    39,149,344,869      cycles                                                                  (62.56%)
    43,840,229,460      instructions                                                            (62.57%)
     4,891,622,698      de_dis_dispatch_token_stalls1.fp_reg_file_rsrc_stall                                        (62.50%)
         2,394,160      de_dis_dispatch_token_stalls1.fp_sch_rsrc_stall                                        (62.46%)
       767,763,817      de_dis_dispatch_token_stalls1.load_queue_rsrc_stall                                        (62.42%)
       102,056,763      de_dis_dispatch_token_stalls1.store_queue_rsrc_stall                                        (62.53%)
        23,247,838      de_dis_dispatch_token_stalls2.int_sch0_token_stall                                        (62.64%)
        14,914,287      de_dis_dispatch_token_stalls2.int_sch1_token_stall  
```

&emsp;&emsp;调度器停顿数据将 V7 的瓶颈定位到**浮点寄存器文件**上。在六项停顿事件中，`de_dis_dispatch_token_stalls1.fp_reg_file_rsrc_stall` 达 **48.9 亿周期**，占 cycles（391 亿）的 **12.5%**，是其中最大的一项。相比之下，`load_queue_rsrc_stall` 仅 7.68 亿（2.0%），`store_queue_rsrc_stall` 为 1.02 亿（0.3%），`fp_sch_rsrc_stall` 与两个整数调度器停顿均为千万级，可以忽略。这一分布说明，制约 V7 前端的并非加载队列或存储队列，而是**浮点寄存器文件容量不足**——微内核所需的 YMM 寄存器数量超过物理寄存器文件的可用条目，调度器只能让指令排队等待。

---

&emsp;&emsp;综合上述 perf 数据，V7 的瓶颈已不在数据供给，而是转移到指令发射与地址翻译两个环节。缓存填充数据排除了内存带宽与 NUMA 因素：91% 的填充来自本地 L2，仅 0.7% 来自内存，远端节点为 0。L2 数据进一步表明，L2 命中率为 85.4%、真正回内存的仅 747 万次，但 `l2_fill_pending.l2_fill_busy` 高达 79.4 亿周期（占 cycles 的 19.6%），说明 L2 的填充队列已被 28.2 亿次请求压至饱和。TLB 数据解释了填充队列拥塞的原因：L1 DTLB miss 4379 万次，其中 3305 万次为 4K 页且 L2 TLB 同样 miss，触发 3407 万次页表遍历——TLB 无法覆盖 16MB 矩阵的工作集。FP 管道数据表明浮点单元利用率已达 87%、接近饱和；而调度器停顿数据显示 `fp_reg_file_rsrc_stall` 占 cycles 的 12.5%，说明 FP 寄存器文件分配紧张，微内核的 YMM 寄存器需求超过了硬件可用条目。

&emsp;&emsp;据此，下一步优化可从三个方向推进。**第一，减少 L1 到 L2 的流量**：将 `KC` 从 128 降至 64，使每个线程的 `Ap`、`Bp` 从各 32KB 缩减至各 16KB、合计 32KB 恰好容纳于 L1d，L1 miss 率有望从 14.3% 明显下降，`l2_request_g1.all_no_prefetch` 与 `l2_fill_busy` 也将同步回落。**第二，以 2MB 大页替代 4K 页**：通过 `madvise(MADV_HUGEPAGE)` 或 `mmap` + `MAP_HUGETLB` 将 A、B、C 与打包缓冲区置于大页之上，16MB 矩阵仅需 8 个 TLB 条目，L2 TLB miss 可从 3305 万次降至接近零，页表遍历次数大幅减少，L2 填充队列的拥塞亦将得到缓解。**第三，降低 FP 寄存器压力**：将 NR 从 8 降至 4、改用 128-bit XMM，或将 MR 从 6 降至 4，减少微内核同时占用的 YMM 累加器数量。Zen 3 的 256-bit FMA 本就会拆分为两个 128-bit uop，改用 128-bit 不会损失吞吐，反而能缓解寄存器重命名压力，`fp_reg_file_rsrc_stall` 有望从 48.9 亿降至 20 亿以下。三个方向分别对应 L1/L2 流量、TLB 覆盖、寄存器容量三个瓶颈，既可独立尝试，也可分别以 `l2_fill_busy`、`tlb_reload_4k_l2_miss`、`fp_reg_file_rsrc_stall` 作为指标验证效果。若三条措施均落实，GFLOPS 有望从当前的 291~331 提升至 350 以上；若需进一步提高，则需借助 AVX-512 或多路 CPU——仅靠调优已接近 Zen 3 单 CCD 的上限。

---

```cpp
// ---------- 大页分配：统一走 mmap + munmap ----------
static float *alloc_hugepage(size_t bytes) {
    void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p != MAP_FAILED) {
        return (float *)p;
    }

    p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    madvise(p, bytes, MADV_HUGEPAGE);
    return (float *)p;
}

static void free_hugepage(float *p, size_t bytes) {
    if (!p) return;
    munmap(p, bytes);
}

constexpr int kV8MR = 4;
constexpr int kV8NR = 16;

// 微内核：C[4x16] = sum_p A[4xp] * B[p x16]
// accum=false 时累加器清零起步（K 方向首轮，beta=0 不读 C）；
// accum=true 时从 C 加载继续累加（多轮 pc 切分时保持部分和）。
// Ap 行主序 4 x kc；Bp 行主序 kc x n（n >= 16）
static inline void micro_kernel_v8(int kc, int n,
                                   const float* __restrict Ap,
                                   const float* __restrict Bp,
                                   float* __restrict C, int ldc, bool accum) {
    __m256 c00, c01, c10, c11, c20, c21, c30, c31;
    if (accum) {
        c00 = _mm256_loadu_ps(C);
        c01 = _mm256_loadu_ps(C + 8);
        c10 = _mm256_loadu_ps(C + ldc);
        c11 = _mm256_loadu_ps(C + ldc + 8);
        c20 = _mm256_loadu_ps(C + 2 * ldc);
        c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
        c30 = _mm256_loadu_ps(C + 3 * ldc);
        c31 = _mm256_loadu_ps(C + 3 * ldc + 8);
    } else {
        c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = _mm256_setzero_ps();
    }

    for (int p = 0; p < kc; ++p) {
        _mm_prefetch((const char*)(Bp + (size_t)(p + 8) * n), _MM_HINT_T0);
        const __m256 b0 = _mm256_loadu_ps(Bp + (size_t)p * n);
        const __m256 b1 = _mm256_loadu_ps(Bp + (size_t)p * n + 8);
        const __m256 a0 = _mm256_set1_ps(Ap[p]);
        const __m256 a1 = _mm256_set1_ps(Ap[kc + p]);
        const __m256 a2 = _mm256_set1_ps(Ap[2 * kc + p]);
        const __m256 a3 = _mm256_set1_ps(Ap[3 * kc + p]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
    }

    _mm256_storeu_ps(C, c00);
    _mm256_storeu_ps(C + 8, c01);
    _mm256_storeu_ps(C + ldc, c10);
    _mm256_storeu_ps(C + ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc, c20);
    _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc, c30);
    _mm256_storeu_ps(C + 3 * ldc + 8, c31);
}

// 物理核数（SMT 检测）：FMA 受限的内核开 SMT 只会争抢 FPU 口与缓存，
// 实测 6 物理核 > 12 逻辑核（2048 下 576 vs 534 GFLOPS）。
// 读不到拓扑时退回逻辑核数。
static int physical_core_count() {
    static const int cached = [] {
        std::ifstream f("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list");
        const unsigned hc = std::thread::hardware_concurrency();
        if (f.good() && hc > 0) return (int)(hc / 2);
        return hc > 0 ? (int)hc : omp_get_max_threads();
    }();
    return cached;
}

void gemm_naive_v8(int M, int N, int K, const float* A, const float* B, float* C) {
    // 分块参数（Zen3 实测决胜：2048 下 ~576 GFLOPS，超 OpenBLAS 537）：
    // - Ap 面板 128x256x4=128KB、Bp 面板 256x96x4=96KB，合计 ~224KB，L2(512KB) 内
    // - C tile 128x96x4=48KB，pc 多轮间重读基本命中 L1
    // 调参辅助：GEMM_V8_MC/NC/KC/THREADS 环境变量可覆盖（仅调参用）
    int MC = 128, NC = 96, KC = 256;
    if (const char* e = std::getenv("GEMM_V8_MC")) MC = atoi(e);
    if (const char* e = std::getenv("GEMM_V8_NC")) NC = atoi(e);
    if (const char* e = std::getenv("GEMM_V8_KC")) KC = atoi(e);
    if (MC > M) MC = M;
    if (NC > N) NC = N;
    if (KC > K) KC = K;

    int nthreads = omp_get_max_threads();
    if (const char* e = std::getenv("GEMM_V8_THREADS")) {
        nthreads = atoi(e);
    } else if (nthreads > physical_core_count()) {
        nthreads = physical_core_count();
    }

    // tile 数不足 2x 线程数时对半缩块（下限 64），保证并行负载均衡
    const long nth = nthreads;
    while ((long)((M + MC - 1) / MC) * ((N + NC - 1) / NC) < 2 * nth &&
           (MC > 64 || NC > 64)) {
        if (MC >= NC && MC > 64)
            MC /= 2;
        else if (NC > 64)
            NC /= 2;
        else
            MC /= 2;
    }

    const size_t ap_size = (size_t)MC * KC;
    const size_t bp_size = (size_t)KC * NC;

    #pragma omp parallel for collapse(2) schedule(static) num_threads(nthreads)
    for (int ic = 0; ic < M; ic += MC) {
        for (int jc = 0; jc < N; jc += NC) {
            // 打包面板放 2MB 大页：面板以 4K 页流式扫过 64 项 L1 dTLB 时 miss 高达
            // 5 亿+/进程；大页后该项趋近于零（每线程一块 2MB，静态复用不释放）。
            static thread_local float* tls_area = nullptr;
            static thread_local size_t tls_bytes = 0;
            static thread_local bool tls_from_malloc = false;
            const size_t need_bytes =
                ((ap_size + bp_size) * sizeof(float) + (1u << 21) - 1) & ~((1u << 21) - 1);
            if (tls_bytes < need_bytes) {
                if (tls_area && !tls_from_malloc) free_hugepage(tls_area, tls_bytes);
                tls_area = alloc_hugepage(need_bytes);
                tls_from_malloc = false;
                if (tls_area) {
                    tls_bytes = need_bytes;
                } else {  // 兜底：普通 malloc（几乎不会走到），不复用不释放
                    tls_area = (float*)std::malloc(need_bytes);
                    tls_from_malloc = true;
                }
            }
            float* Ap = tls_area;
            float* Bp = tls_area + ap_size;
            const int mc = (ic + MC <= M) ? MC : M - ic;
            const int nc = (jc + NC <= N) ? NC : N - jc;

            for (int pc = 0; pc < K; pc += KC) {
                const int kc = (pc + KC <= K) ? KC : K - pc;

                // 打包 A：mc x kc（源行距 K）
                for (int i = 0; i < mc; ++i) {
                    const float* src = A + (size_t)(ic + i) * K + pc;
                    float* dst = Ap + (size_t)i * kc;
                    for (int p = 0; p < kc; ++p) dst[p] = src[p];
                }
                // 打包 B：kc x nc（源行距 N）
                for (int p = 0; p < kc; ++p) {
                    const float* src = B + (size_t)(pc + p) * N + jc;
                    float* dst = Bp + (size_t)p * nc;
                    for (int j = 0; j < nc; ++j) dst[j] = src[j];
                }

                // 微内核主循环（pc 首轮写，后续轮从 C 累加）
                const bool acc = (pc > 0);
                int ir = 0;
                for (; ir + kV8MR <= mc; ir += kV8MR) {
                    int jr = 0;
                    for (; jr + kV8NR <= nc; jr += kV8NR) {
                        micro_kernel_v8(kc, nc, Ap + (size_t)ir * kc,
                                        Bp + jr,
                                        C + (size_t)(ic + ir) * N + (jc + jr), N,
                                        acc);
                    }
                    for (; jr < nc; ++jr) {  // NR 尾巴
                        for (int ii = 0; ii < kV8MR; ++ii) {
                            float* cp = C + (size_t)(ic + ir + ii) * N + (jc + jr);
                            const float* arow = Ap + (size_t)(ir + ii) * kc;
                            float accv = acc ? *cp : 0.0f;
                            for (int p = 0; p < kc; ++p)
                                accv += arow[p] * Bp[(size_t)p * nc + jr];
                            *cp = accv;
                        }
                    }
                }
                for (; ir < mc; ++ir) {  // MR 尾巴
                    const float* arow = Ap + (size_t)ir * kc;
                    float* crow = C + (size_t)(ic + ir) * N + jc;
                    if (!acc)
                        for (int j = 0; j < nc; ++j) crow[j] = 0.0f;
                    for (int p = 0; p < kc; ++p) {
                        const float a = arow[p];
                        const float* brow = Bp + (size_t)p * nc;
                        for (int j = 0; j < nc; ++j) crow[j] += a * brow[j];
                    }
                }
            }
        }
    }
}
```

&emsp;&emsp;V8 的核心突破在于微内核从 4×8 调整为 **4×16**，将独立的 FMA 依赖链从 4 条增加到 8 条。旧版微内核每个 `p` 迭代仅有 4 条独立 FMA，而 Zen 3 的 FMA 延迟为 4 周期、双发射，填满流水线至少需要 8 条在途 FMA，旧版的性能上限因此只有峰值的一半。新版以 8 个 YMM 累加器（`c00` 至 `c31`）支撑 8 条独立链，微内核无寄存器溢出，14 µop/迭代恰好使 FMA 端口达到饱和。仅此一步即将 2048³ 的 GFLOPS 从 341 提升至 434。同时，代码摒弃了旧版“将 A、B、C 整块拷贝到大页”的做法，改为直接写回 C：K 方向首轮将累加器清零（beta=0 语义），后续轮次从 C 加载继续累加，节省了 48MB 的拷贝开销；打包面板通过 `thread_local` 复用，每个线程仅分配一次。

&emsp;&emsp;第二层优化针对 **TLB**。打包面板最初使用普通 `malloc`，在 4K 页下每个面板需扫过 64 个 L1 dTLB 条目，2048³ 规模下 dTLB miss 高达 5 亿次以上。改用 `MAP_HUGETLB` 的 2MB 大页后，面板的 TLB 覆盖需求从 64 个条目降至 1 个，`ls_l1_d_tlb_miss.all` 从数亿次降至 6032 万次，`ls_tablewalker.dside` 降至 3047 万次。这一步使 GFLOPS 从 434 提升至 505。分块参数也在该阶段定型：`MC=128, NC=96, KC=256`，Ap 面板 128×256×4 = 128KB，Bp 面板 256×96×4 = 96KB，合计 224KB，贴近 L2 的 512KB；C tile 128×96×4 = 48KB，pc 多轮之间的重读基本命中 L1。最后将线程数从 12 逻辑核限制为 6 物理核（通过 SMT 检测），因为 FMA 受限时 SMT 兄弟线程只会争抢 FPU 端口与缓存资源；实测 6 物理核下 2048³ 的性能反而优于 12 逻辑核，从 534 GFLOPS 提升至 576 GFLOPS。

```cpp
BM_sgemm/naive_v8/2048x2048x2048   29604181 ns     29601578 ns           23 GFLOPS=580.37G/s

 Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v8/2048x2048x2048':

    32,705,072,258      cycles                                                                  (20.30%)
    95,152,719,693      instructions                                                            (20.28%)
    30,288,891,127      L1-dcache-loads                                                         (20.23%)
     6,112,268,676      L1-dcache-load-misses                                                   (20.26%)
     6,150,887,253      l2_request_g1.all_no_prefetch                                           (20.28%)
       379,578,303      l2_cache_req_stat.ls_rd_blk_c                                           (20.37%)
     5,165,316,163      l2_cache_req_stat.ls_rd_blk_l_hit_x                                        (20.41%)
        32,568,411      l2_cache_req_stat.ls_rd_blk_l_hit_s                                        (20.45%)
     8,729,636,405      l2_fill_pending.l2_fill_busy                                            (20.56%)
       707,272,436      ls_dmnd_fills_from_sys.lcl_l2                                           (20.57%)
       356,024,257      ls_dmnd_fills_from_sys.int_cache                                        (20.50%)
        11,432,513      ls_dmnd_fills_from_sys.mem_io_local                                        (20.45%)
                 0      ls_dmnd_fills_from_sys.mem_io_remote                                        (20.46%)
        60,317,299      ls_l1_d_tlb_miss.all                                                    (20.46%)
        29,354,170      ls_l1_d_tlb_miss.tlb_reload_4k_l2_miss                                        (20.41%)
           378,417      ls_l1_d_tlb_miss.tlb_reload_2m_l2_hit                                        (20.45%)
        30,469,290      ls_tablewalker.dside                                                    (20.46%)
   493,406,794,090      fp_ret_sse_avx_ops.all                                                  (16.31%)
   490,958,750,371      fp_ret_sse_avx_ops.mac_flops                                            (12.20%)
    18,098,239,969      fpu_pipe_assignment.total0                                              (12.21%)
    18,284,670,962      fpu_pipe_assignment.total1                                              (12.21%)
     4,969,425,664      de_dis_dispatch_token_stalls1.fp_reg_file_rsrc_stall                                        (16.49%)
       717,518,728      de_dis_dispatch_token_stalls1.load_queue_rsrc_stall                                        (16.42%)
     6,026,550,713      branches                                                                (20.46%)
        39,793,513      branch-misses                                                           (20.39%)
```

&emsp;&emsp;最终结果：2048³ 下为 **29604181 ns、580.37 GFLOPS**，256³ 和 1024³ 分别达到 526 和 576 GFLOPS；前两项超过 OpenBLAS，2048³ 达到 OpenBLAS 的 91%。perf 数据同样印证了微内核的良好状态：`fpu_pipe_assignment.total0` 与 `total1` 分别为 181.0 亿和 182.8 亿，几乎完美地 50/50 分布，FP 管道每周期约 3.5 个 uop，利用率接近 Zen 3 的 4 uop/周期上限；`fp_reg_file_rsrc_stall` 为 49.7 亿，占 cycles 的 15.2%，处于可接受水平；L1 miss 率 20.2% 看似偏高，但 `l2_fill_pending.l2_fill_busy` 仅 87.3 亿，说明 L1 miss 后大部分在 L2 快速命中，未形成填充队列瓶颈；`mem_io_local` 仅 1143 万次，内存带宽完全不构成约束。这组数据表明 V8 已将单 CCD 的 FMA 吞吐逼近理论极限。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/cpu_gemm_v8_xxxxxxxxxxxxxxx_bench_result.png)

&emsp;&emsp;至此，从朴素实现到多线程 SIMD 微内核的完整优化路径已全部走完。每一步都在解决前一步遗留的瓶颈：循环交换解决 B 的跨步访问，分块解决全局复用，打包提升微内核的加载效率，寄存器分块隐藏 FMA 延迟，SIMD 提升单指令吞吐，多线程突破单核算力上限。硬件峰值与带宽决定了性能上界，而这些优化方法的作用，就是让实际性能不断逼近这一上界。


## 3 GEMM 的 GPU 优化

### 3.1 GPU GEMM 的性能模型与优化方法

&emsp;&emsp;上文已经给出了 GEMM 的数学形式，并指出单精度浮点运算量为 $2MNK$,完成了CPU优化GEMM。要进一步优化性能，则需要在 GPU 上执行 GEMM，为了进一步优化GPU性能，仅了解运算量还不够，还需深入理解 GPU 架构与计算模型。与 CPU 不同，GPU 的设计哲学并非“降低单条指令的延迟”，而是“以海量并行隐藏延迟”。因此，CPU 上关于缓存层次、指令级并行的直觉，不能直接照搬到 GPU；需要从存储层次、执行单元及二者配合方式三个维度重新建立性能模型。

&emsp;&emsp;根据**Roofline 模型**，GPU 的全局内存带宽通常远低于其峰值算力，转折点对应的算术强度很高，因此**GPU 上 GEMM 的首要瓶颈往往是全局内存带宽**——若每个线程都直接从全局内存读取矩阵元素，再强的算力也会受制于访存。

&emsp;&emsp;与 CPU 的层次化缓存类似，GPU 同样具有多级存储，但容量与带宽的比例与 CPU 差异显著：

- **全局内存（Global Memory / HBM）**：容量大（数十 GB），但延迟高（数百周期）、带宽相对有限。GPU 上 GEMM 优化的第一要务，是让数据尽可能在片上复用，减少全局访存。
- **L2 缓存**：由所有 SM 共享，容量小于 CPU 的 L3，但带宽更高，用于缓解全局内存压力。
- **共享内存（Shared Memory / SMEM）**：每个 SM（流式多处理器）私有的片上存储，容量小（通常几十 KB 到上百 KB），但延迟极低、带宽极高。GEMM 优化的核心手段之一，是将全局内存中的数据分块（tiling）搬入共享内存，使线程在片上反复复用，减少全局访存次数。
- **寄存器（Register File）**：每个线程私有，速度最快。高性能 GEMM 内核会让每个线程在寄存器中维护一小块累加器（如 8×8），从而在循环体内只执行 FMA，无需访问其他存储层级。

&emsp;&emsp;与 CPU 的行主序/列主序问题类似，GPU 上矩阵的存储布局同样关键，但应对方式不同：GPU 通过**合并访问（coalesced access）**保证效率——当一个 warp（32 个线程）访问全局内存时，若访问地址连续，硬件会将这次访问合并为一次宽事务；若地址分散，则拆分为多次事务，带宽利用率骤降。因此，GPU 上的 GEMM 内核通常要求矩阵按特定布局排列，或在加载时通过共享内存进行转置，以配合合并访问。

&emsp;&emsp;执行单元方面，GPU 的算力来自大量 SM，每个 SM 内部包含：

- **CUDA Core**：执行 FP32/INT32 等常规运算。FMA 指令在 CUDA Core 上执行，延迟同样约为 4 周期，但每个 SM 每周期可发射多条 FMA（具体数量取决于架构，如 Ampere 上每个 SM 每周期可发射 64 条 FP32 FMA）。
- **Tensor Core**：专为矩阵乘法设计的单元，单条指令即可完成一个小矩阵块（如 16×16×16）的乘加。Tensor Core 的吞吐量远高于 CUDA Core，是现代 GPU 上 GEMM 性能的主要来源。但其使用存在约束：需要特定的数据类型（FP16、BF16、TF32、INT8 等）、特定的数据布局（如 wmma/mma 所要求的 fragment 排布），并且需要软件显式调用。
- **特殊函数单元（SFU）**：处理超越函数等运算，与 GEMM 关系不大。

&emsp;&emsp;这里同样存在延迟与吞吐的权衡，只是尺度不同：GPU 上一条 Tensor Core MMA 指令的延迟可能达十几至几十周期，但吞吐量很高。要填满流水线，就需要**足够多的独立 warp** 同时驻留在 SM 上，由此引出 GPU 特有的概念：

- **Occupancy（占用率）**：每个 SM 上活跃 warp 数与最大支持 warp 数之比。Occupancy 过低时延迟无法被隐藏，执行单元会空闲；过高时又可能因寄存器或共享内存不足而限制每个线程可用的资源，反而降低单线程效率。
- **Warp 调度**：当某个 warp 因等待内存或依赖前一条指令而停顿时，调度器立即切换到另一个就绪 warp。GPU 正是依靠这种“以并行度换延迟隐藏”的方式，让海量线程将高延迟的访存与计算重叠起来。

&emsp;&emsp;预取方面，GPU 上的“预取”更多体现为**分块加载**：在计算当前 tile 的同时，将下一个 tile 从全局内存搬入共享内存或寄存器，使访存与计算重叠。过度分块会增加共享内存占用、降低 Occupancy，因此分块大小需要在“复用率”和“Occupancy”之间取平衡。

&emsp;&emsp;矩阵存储顺序在 GPU 上同样直接影响访存模式。以行主序为例，$C[i,j]=\sum_k A[i,k]B[k,j]$。若一个 warp 内的线程按 j 方向连续排列，则对 B 的访问沿 j 连续，可合并；对 A 的访问是广播（所有线程读同一个 $A[i,k]$），也能高效处理。若线程按 k 方向连续排列，则对 B 的访问步长为 N，无法合并，带宽利用率骤降。因此，GPU 上的 GEMM 内核通常让线程沿 N 方向排列，或通过共享内存转置来配合合并访问。

&emsp;&emsp;从实现角度看，GPU GEMM 的高性能实现通常采用分层分块与微内核结构（具体思路和CPU优化类似）：
1. 将 C 划分为若干 tile，每个线程块（block）负责一个 tile；
2. 将 A、B 的子块从全局内存搬入共享内存，线程在共享内存上反复复用；
3. 每个线程在寄存器中维护 $MR\times NR$ 的小累加器，沿 K 维循环执行 FMA；
4. 使用 `__syncthreads()` 同步共享内存的加载与计算，避免数据竞争；
5. 通过双缓冲（double buffering）将下一块数据的加载与当前块的计算重叠；
6. 在支持 Tensor Core 的架构上，用 `wmma` 或 `mma` 指令替代 CUDA Core 的 FMA，将吞吐提升一个数量级；
7. 调整 block 大小、tile 大小、寄存器用量，使 Occupancy 与复用率达到最佳平衡。

&emsp;&emsp;这些方法最终服务于同一个目标：用分块和共享内存提高片上数据复用率，用多 warp 和 Occupancy 隐藏访存延迟，用 Tensor Core 提升单指令吞吐，并根据矩阵布局选择最合适的访问模式。GPU 的峰值算力和全局内存带宽决定了性能上界，而优化方法的作用，是不断逼近这一上界。

---

### 3.2 朴素实现

&emsp;&emsp;GPU 上 GEMM 的数学形式与 CPU 完全一致：$C[i,j]=\sum_{k=0}^{K-1}A[i,k]B[k,j]$。最直接的 CUDA 实现，是让每个线程负责 C 的一个元素，在全局内存上完成一次内积：

```cpp
__global__ void sgemm_naive_kernel(int M, int N, int K,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C) {
  // row-major：C[M×N] = A[M×K] · B[K×N]，alpha=1, beta=0
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = acc;
}

void sgemm_cuda_naive_v1(int M, int N, int K, const float* A, const float* B, float* C) {
  const size_t a_bytes = sizeof(float) * static_cast<size_t>(M) * K;
  const size_t b_bytes = sizeof(float) * static_cast<size_t>(K) * N;
  const size_t c_bytes = sizeof(float) * static_cast<size_t>(M) * N;

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
  CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
  CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
  CUDA_CHECK(cudaMemcpy(d_a, A, a_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, B, b_bytes, cudaMemcpyHostToDevice));

  dim3 block(32, 32);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  sgemm_naive_kernel<<<grid, block>>>(M, N, K, d_a, d_b, d_c);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(C, d_c, c_bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
}
```

&emsp;&emsp;主机端负责分配设备内存、把 A、B 拷贝到显存、启动内核、把 C 拷回主机。block 取 `(32, 32)`，grid 按输出矩阵的维度除以 block 维度向上取整。2048³ 下 `grid = (64, 64)`、`block = (32, 32)`，共 4096 个 block、419 万个线程，每个线程计算 C 的一个元素。

&emsp;&emsp;从合并访问的角度看，这个实现没有明显缺陷。内层 k 循环访问 `B[k * N + col]` 时，同一 warp 内 32 个线程的 `col` 连续，对 B 的访问沿 N 方向连续，硬件会合并为一次宽事务；访问 `A[row * K + k]` 时，同一 warp 内所有线程的 `row` 和 `k` 相同，读的是同一个地址，硬件按广播处理，效率同样很高。所以 naive 实现的访存模式在“合并访问”这一层是合格的。

&emsp;&emsp;真正的问题在于**数据复用为零**。每个线程独立完成一次长度为 K 的内积：A 的第 row 行被同一行的 N 个线程各读一遍，B 的第 col 列被同一列的 M 个线程各读一遍。整个计算过程中，A 被读取 N 次、B 被读取 M 次，总访存量约为 $4MNK + 4MNK + 4MN$ 字节，而浮点运算量只有 $2MNK$。算术强度约为 $2MNK/(8MNK+4MN) \approx 0.25$ FLOP/Byte，远低于 Roofline 转折点。

&emsp;&emsp;以 2048³ 为例，$2MNK \approx 1.72 \times 10^{10}$ FLOP，而访存量约 $8MNK \approx 6.87 \times 10^{10}$ 字节。若按 DRAM 带宽 224 GB/s 估算，理论耗时约 0.31 秒；但实测耗时只有 37.58 毫秒，比这个估算快了一个数量级。这说明**实际访存并未全部落到 DRAM**——L1 和 L2 缓存吸收了大量重复读取。要判断真正的瓶颈，需要看 Nsight Compute 的硬件计数器。

&emsp;&emsp;用 `ncu` 采集 2048³ 内核的 SpeedOfLight 数据，结果如下：

```cpp
❯ sudo ncu --set roofline -k sgemm_naive_kernel --launch-skip 6 --launch-count 1 \
        -f -o roofline_naive \
        ./build/bench_gemm --benchmark_filter=BM_sgemm/cuda_naive_v1/2048x2048x2048 && sudo ncu --import roofline_naive.ncu-rep --section SpeedOfLight
==PROF== Report: /media/nvme0n1/workspace/gemm_test/roofline_naive.ncu-rep
[9841] bench_gemm@127.0.0.1
  gemm::sgemm_naive_kernel(int, int, int, const float *, const float *, float *) (64, 64, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- -------------
    Metric Name               Metric Unit  Metric Value
    ----------------------- ------------- -------------
    DRAM Frequency          cycle/nsecond          6.79
    SM Frequency            cycle/nsecond          1.55
    Elapsed Cycles                  cycle    58,341,477
    Memory Throughput                   %         92.09
    DRAM Throughput                     %         15.37
    Duration                      msecond         37.58
    L1/TEX Cache Throughput             %         92.35
    L2 Cache Throughput             %         10.60
    Compute (SM) Throughput             %         92.09
    SM Active Cycles                cycle 58,179,103.35
    ----------------------- ------------- -------------
```

&emsp;&emsp;这组数据揭示了一个与直觉不同的瓶颈分布。`DRAM Throughput` 只有 **15.37%**，说明显存带宽远未饱和；`L2 Cache Throughput` 只有 **10.60%**，L2 也没有成为瓶颈。真正的热点在 **L1/TEX 缓存**：`L1/TEX Cache Throughput` 达 **92.35%**，几乎被打满。`Memory Throughput` 和 `Compute (SM) Throughput` 都是 **92.09%**，看似计算和访存同时受限，但结合 naive 内核没有做任何计算优化的背景，这个“Compute”并非来自 FMA 单元——FMA 单元此时几乎空闲，92% 的忙碌来自 **LSU（加载/存储单元）**，它被大量 `A[row*K+k]` 和 `B[k*N+col]` 的加载指令塞满。

&emsp;&emsp;换句话说，naive 内核的瓶颈是 **L1/TEX 带宽和 LSU 吞吐**，而不是显存带宽。这与 CPU 上的 naive_v1 情况如出一辙：不是数据取不回来，而是数据以低效的方式被反复取用。区别在于 GPU 的 L1 带宽和并行度远高于 CPU，因此绝对性能高出三个数量级，但相对峰值仍然很低。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gpu_gemm_v1_xxxxxxxxxxxxxxxxxxx.png)

&emsp;&emsp;优化方向很明确：**把数据从 L1 搬到更靠近计算单元的地方，在 block 内反复复用**。具体来说，让一个 block 负责 C 的一个 tile，把对应的 A、B 子块搬进**共享内存**，block 内所有线程从共享内存读数据，而不是每次都从 L1 读。共享内存的带宽比 L1 更高，且 bank 可并行访问；同时，每个线程在寄存器中维护多个累加器（如 4×4），减少对 C 的读写次数。这样 A、B 的每个元素只从 L1/全局内存读一次，却在 block 内被复用多次，L1 压力和 LSU 压力都能大幅下降。这正是 3.3 节要讨论的共享内存分块。

### 3.3 共享内存分块

&emsp;&emsp;3.2 节的数据给出了一个与直觉不同的结论：naive 实现的瓶颈不在显存带宽，而在 **L1/TEX 缓存**。`L1/TEX Cache Throughput` 达 92.35%，而 `DRAM Throughput` 只有 15.37%。这说明 A、B 的每个元素虽然被反复读取，但大部分请求被 L1 吸收，真正打到显存的数据量并不大。问题在于 **L1 的访问次数太多**：每个线程独立完成一次长度为 K 的内积，A 的第 row 行被同一行的 N 个线程各读一遍，B 的第 col 列被同一列的 M 个线程各读一遍，数据复用为零。

&emsp;&emsp;要减少 L1 访问，就需要让数据在更靠近计算单元的地方被复用。GPU 上最直接的复用载体是**共享内存（Shared Memory / SMEM）**：每个 SM 私有的片上存储，延迟低、带宽高，且同一 block 内的所有线程都能访问。共享内存分块的思路是：把输出矩阵 C 划分为若干 `TILE × TILE` 的小块，每个线程块负责一个 C 的 tile；沿 K 方向按 `TILE` 步长切分，每次把一个 `TILE × TILE` 的 A 子块和一个 `TILE × TILE` 的 B 子块从全局内存搬进共享内存，block 内所有线程在共享内存上完成这一段的乘加，再推进到下一个 K 切片。这样，A、B 的每个元素在 block 内被复用 `TILE` 次，L1 的访问次数大幅下降。

&emsp;&emsp;实现的关键有三点：共享内存的声明、加载时的边界处理、同步与计算。完整代码如下：

```cpp
// ---------- 共享内存分块 GEMM：v2 ----------
#define TILE 32

__global__ void sgemm_shared_kernel_v2(int M, int N, int K,
                                       const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C) {
    __shared__ float A_s[TILE][TILE];
    __shared__ float B_s[TILE][TILE];

    const int row_base = blockIdx.y * TILE;
    const int col_base = blockIdx.x * TILE;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row = row_base + ty;
    const int col = col_base + tx;

    float acc = 0.0f;

    for (int k_base = 0; k_base < K; k_base += TILE) {
        // ---- 加载 A 子块到共享内存 ----
        const int a_row = row_base + ty;
        const int a_col = k_base + tx;
        if (a_row < M && a_col < K)
            A_s[ty][tx] = A[a_row * K + a_col];
        else
            A_s[ty][tx] = 0.0f;

        // ---- 加载 B 子块到共享内存 ----
        const int b_row = k_base + ty;
        const int b_col = col_base + tx;
        if (b_row < K && b_col < N)
            B_s[ty][tx] = B[b_row * N + b_col];
        else
            B_s[ty][tx] = 0.0f;

        __syncthreads();

        // ---- 在共享内存上完成 TILE 长度的内积 ----
        for (int k = 0; k < TILE; ++k)
            acc += A_s[ty][k] * B_s[k][tx];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

void sgemm_cuda_shared_v2(int M, int N, int K,
                          const float* A, const float* B, float* C) {
    const size_t a_bytes = sizeof(float) * static_cast<size_t>(M) * K;
    const size_t b_bytes = sizeof(float) * static_cast<size_t>(K) * N;
    const size_t c_bytes = sizeof(float) * static_cast<size_t>(M) * N;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, A, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, B, b_bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    sgemm_shared_kernel_v2<<<grid, block>>>(M, N, K, d_a, d_b, d_c);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C, d_c, c_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}
```

&emsp;&emsp;相比 naive 版本，这段代码有三处关键改动。
- **第一，共享内存声明**：`A_s` 和 `B_s` 各是一个 `TILE × TILE` 的 `float` 数组，共 `2 × 32 × 32 × 4 = 8 KB`，RTX 3050 每个 SM 有 100 KB 共享内存，8 KB 的占用不会限制 Occupancy。**
- **第二，加载时的边界处理**：当 `M`、`N`、`K` 不是 `TILE` 的整数倍时，越界的元素填 0。填 0 而不是跳过，是为了让后面的乘加仍然执行，不引入额外的分支；由于 `0 * x = 0`，填 0 不会影响最终结果。
- **第三，两次 `__syncthreads()`**：第一次在加载完成后，确保共享内存里的数据对 block 内所有线程可见；第二次在计算完成后，确保没有线程在下一轮加载时覆盖别人还在读的数据。缺少任何一次同步，都会导致读写竞争或读到旧数据。

&emsp;&emsp;共享内存分块的性能提升来自**数据复用**。naive 版本中，A 的每个元素被同一行的 N 个线程各读一遍；分块后，A 的每个元素被 block 内 `TILE` 个线程复用，L1 访问次数降到原来的 `1/TILE`。2048³ 下 `TILE = 32`，L1 的 A 访问量理论上降到 1/32。

&emsp;&emsp;实测结果如下：

```cpp
BM_sgemm/cuda_naive_v2/2048x2048x2048  100602078 ns    GFLOPS=182.12

Performance counter stats (ncu --set roofline):
    Elapsed Cycles                  cycle    46,313,231
    Duration                      msecond         29.84
    Memory Throughput                   %         79.80
    DRAM Throughput                     %         19.08
    L1/TEX Cache Throughput             %         79.94
    L2 Cache Throughput                 %         13.32
    Compute (SM) Throughput             %         79.80
    SM Active Cycles                cycle    46,218,668
```

&emsp;&emsp;这组数据与 3.2 节的 naive 实现对比，可以清楚看到共享内存分块的作用：

| 指标 | `_v1`（naive） | `_v2`（共享内存分块） | 变化 |
|------|---------------|---------------------|------|
| GFLOPS | 160.22 | **182.12** | 升 13.7% |
| Duration | 37.58 ms | **29.84 ms** | 降 20.6% |
| Elapsed Cycles | 58,341,477 | **46,313,231** | 降 20.6% |
| L1/TEX Cache Throughput | 92.35% | **79.94%** | 降 12.4 个百分点 |
| DRAM Throughput | 15.37% | 19.08% | 略升 |
| L2 Cache Throughput | 10.60% | 13.32% | 略升 |
| Compute (SM) Throughput | 92.09% | **79.80%** | 降 12.3 个百分点 |

&emsp;&emsp;三个关键指标的下降说明共享内存分块确实缓解了 L1 的压力：`L1/TEX Cache Throughput` 从 92.35% 降到 79.94%，`Compute (SM) Throughput` 从 92.09% 降到 79.80%，`Elapsed Cycles` 降了 20.6%。`DRAM Throughput` 从 15.37% 升到 19.08%，说明共享内存分块后，从全局内存加载 A、B 子块的行为更集中，但显存带宽仍然远未饱和。

&emsp;&emsp;不过，性能提升只有 13.7%，远低于理论预期的 `1/TILE` 级别。原因在于两个尚未解决的瓶颈。

&emsp;&emsp;**第一，共享内存的 bank conflict。** 当前写法里 `A_s[ty][k]` 的访问模式中，同一个 warp 内 32 个线程的 `ty` 从 0 到 31、`k` 相同，访问的是 `A_s[0..31][k]` 这一列，步长为 `TILE = 32` 个 float。32 是 32 的倍数，所有线程访问的地址都落在同一个 bank 上，产生 **32 路 bank conflict**。这使共享内存的实际带宽降到理论值的 1/32。修法是把共享内存的列数从 `TILE` 增加到 `TILE + 1`，即 `A_s[TILE][TILE + 1]`。这样列的步长变成 33，和 32 互质，不同 `ty` 的线程访问的 bank 分散开。这是 CUDA 编程中常见的 **padding 技巧**，改动只有一行，但能显著减少 bank conflict。

&emsp;&emsp;**第二，每个线程仍然只计算一个 C 元素。** 共享内存分块减少了 A、B 的 L1 访问，但每个线程对 C 的写回仍然是 1 次，对 A_s、B_s 的读取仍然是一次完整的 `TILE` 长度内积。要进一步提高 FMA 效率，需要让每个线程在寄存器中维护多个累加器，一次计算 `MR × NR` 个 C 元素。这不仅能减少对共享内存的读取次数，还能提高 FMA 的发射效率，同时让 A_s、B_s 的元素被更多次复用。这正是 3.4 节要讨论的**寄存器分块**。

&emsp;&emsp;从性能模型看，共享内存分块把算术强度从 naive 版本的 0.25 FLOP/Byte 提升到约 `TILE/2` 的水平。`TILE = 32` 时算术强度约 16 FLOP/Byte，已经越过 Roofline 转折点，性能从“带宽受限”转向“计算受限”。这正是共享内存分块的意义：**不是让数据取回来，而是让数据取回来之后被用得更充分**。接下来的寄存器分块会把“用得更充分”推到极致——让每个从共享内存读出的数据，在寄存器中被复用 `MR` 或 `NR` 次，把 L1 和共享内存的访问都降到最低。

### 3.4 寄存器分块与线程级并行

&emsp;&emsp;3.3 节的共享内存分块把 A、B 的 L1 访问降到原来的 `1/TILE`，但每个线程仍然只计算 C 的一个元素。`A_s[ty][k]` 和 `B_s[k][tx]` 的每次读取只服务一次 FMA，共享内存的读取次数与 FMA 次数之比仍然是 2:1。要进一步提高计算访存比，需要让每个线程一次计算 `MR × NR` 个 C 元素，在寄存器中维护多个累加器，使每个从共享内存读出的数据在寄存器中被复用多次。

&emsp;&emsp;具体做法是：把 block 的 `TILE × TILE` 输出块进一步划分为 `(TILE/MR) × (TILE/NR)` 个线程，每个线程负责一个 `MR × NR` 的小块。沿 K 方向循环时，每个线程从 `A_s` 读取 `MR` 个元素、从 `B_s` 读取 `NR` 个元素，执行 `MR × NR` 次 FMA。这样，`A_s` 的每个元素被复用 `NR` 次，`B_s` 的每个元素被复用 `MR` 次，共享内存的读取次数降到 `1/MR + 1/NR`。同时，`MR × NR` 个累加器彼此独立，FMA 之间没有依赖链，可以填满流水线。

&emsp;&emsp;下面给出 `MR = 4`、`NR = 4` 的版本。block 取 `TILE × TILE = 32 × 32`，每个 block 有 `(32/4) × (32/4) = 8 × 8 = 64` 个线程，每个线程计算 `4 × 4` 个 C 元素。宏名统一加 `SG_` 前缀，避免和前面版本的 `TILE`、`MR`、`NR` 冲突。

```cpp
#define SG_TILE 32
#define SG_MR 4
#define SG_NR 4

__global__ void sgemm_register_kernel_v3(int M, int N, int K,
                                         const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C) {
    __shared__ float A_s[SG_TILE][SG_TILE];
    __shared__ float B_s[SG_TILE][SG_TILE];

    const int row_base = blockIdx.y * SG_TILE;
    const int col_base = blockIdx.x * SG_TILE;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    // 当前线程负责的 MR x NR 小块左上角
    const int row = row_base + ty * SG_MR;
    const int col = col_base + tx * SG_NR;

    // 每个线程维护 MR x NR 个累加器
    float acc[SG_MR][SG_NR];
    for (int i = 0; i < SG_MR; ++i)
        for (int j = 0; j < SG_NR; ++j)
            acc[i][j] = 0.0f;

    // 加载时，32x32 的 tile 共 1024 个元素，8x8=64 个线程各负责
    // 4x4=16 个（行按 blockDim.y 步进、列按 blockDim.x 步进），恰好铺满。
    for (int k_base = 0; k_base < K; k_base += SG_TILE) {
        // ---- 加载 A 子块：A_s[i][j]，i 覆盖 32 行、j 覆盖 32 列 ----
        for (int i = ty; i < SG_TILE; i += blockDim.y) {
            const int a_row = row_base + i;
            for (int j = tx; j < SG_TILE; j += blockDim.x) {
                const int a_col = k_base + j;
                if (a_row < M && a_col < K)
                    A_s[i][j] = A[a_row * K + a_col];
                else
                    A_s[i][j] = 0.0f;
            }
        }

        // ---- 加载 B 子块：B_s[i][j]，i 覆盖 32 行（k 维）、j 覆盖 32 列 ----
        for (int j = tx; j < SG_TILE; j += blockDim.x) {
            const int b_col = col_base + j;
            for (int i = ty; i < SG_TILE; i += blockDim.y) {
                const int b_row = k_base + i;
                if (b_row < K && b_col < N)
                    B_s[i][j] = B[b_row * N + b_col];
                else
                    B_s[i][j] = 0.0f;
            }
        }

        __syncthreads();

        // ---- 在共享内存上完成 TILE 长度的乘加 ----
        for (int k = 0; k < SG_TILE; ++k) {
            float a_frag[SG_MR];
            float b_frag[SG_NR];
            for (int i = 0; i < SG_MR; ++i)
                a_frag[i] = A_s[ty * SG_MR + i][k];
            for (int j = 0; j < SG_NR; ++j)
                b_frag[j] = B_s[k][tx * SG_NR + j];

            for (int i = 0; i < SG_MR; ++i)
                for (int j = 0; j < SG_NR; ++j)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }

        __syncthreads();
    }

    // ---- 写回 C ----
    for (int i = 0; i < SG_MR; ++i) {
        for (int j = 0; j < SG_NR; ++j) {
            const int r = row + i;
            const int c = col + j;
            if (r < M && c < N)
                C[r * N + c] = acc[i][j];
        }
    }
}

void sgemm_cuda_register_v3(int M, int N, int K,
                            const float* A, const float* B, float* C) {
    const size_t a_bytes = sizeof(float) * static_cast<size_t>(M) * K;
    const size_t b_bytes = sizeof(float) * static_cast<size_t>(K) * N;
    const size_t c_bytes = sizeof(float) * static_cast<size_t>(M) * N;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, A, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, B, b_bytes, cudaMemcpyHostToDevice));

    // block 取 (TILE/MR) x (TILE/NR)，每个线程负责 MR x NR 个 C 元素
    dim3 block(SG_TILE / SG_NR, SG_TILE / SG_MR);
    dim3 grid((N + SG_TILE - 1) / SG_TILE, (M + SG_TILE - 1) / SG_TILE);
    sgemm_register_kernel_v3<<<grid, block>>>(M, N, K, d_a, d_b, d_c);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C, d_c, c_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

static Registrar reg_cuda_register_v3("cuda_naive_v3", &sgemm_cuda_register_v3);

```

&emsp;&emsp;相比 3.3 节的共享内存版本，这段代码有四处关键改动。
- **第一，block 从 `(32, 32)` 变成 `(8, 8)`**：`TILE = 32`、`MR = NR = 4` 时，每个线程负责 `4 × 4` 的 C 小块，block 内需要的线程数是 `(32/4) × (32/4) = 8 × 8 = 64`。线程数从 1024 降到 64，但每个线程的计算量从 1 个 FMA 链变成 16 条 FMA 链，指令级并行度大幅提升。
- **第二，加载循环用双重跨步**：因为 block 只有 64 个线程，而 `A_s`、`B_s` 各有 1024 个元素，每个线程需要加载 16 个元素。用 `for (i = ty; i < 32; i += 8)` 外层跨行、`for (j = tx; j < 32; j += 8)` 内层跨列，正好覆盖 32×32 全部位置。
- **第三，内层循环引入 `a_frag` 和 `b_frag`**：每个 k 步，先从 `A_s` 读 `MR` 个元素到 `a_frag`，从 `B_s` 读 `NR` 个元素到 `b_frag`，再做 `MR × NR` 次 FMA。这两个局部数组会被编译器放入寄存器，使每个从共享内存读出的数据在寄存器中被复用 `MR` 或 `NR` 次。
- **第四，累加器用二维数组 `acc[MR][NR]`**：16 个累加器彼此独立，FMA 之间没有依赖链，可以填满流水线。

&emsp;&emsp;实测结果如下：

```cpp
BM_sgemm/cuda_naive_v3/2048x2048x2048   75339098 ns    GFLOPS=248.51

Performance counter stats (ncu --set roofline):
    Elapsed Cycles                  cycle    26,120,020
    Duration                      msecond         16.89
    Memory Throughput                   %         52.24
    DRAM Throughput                     %         10.84
    L1/TEX Cache Throughput             %         52.27
    L2 Cache Throughput                 %         15.65
    Compute (SM) Throughput             %         28.49
    SM Active Cycles                cycle    26,106,796
```

&emsp;&emsp;这组数据与 3.3 节的共享内存版本对比，可以看到寄存器分块的作用：

| 指标 | `_v2`（共享内存分块） | `_v3`（寄存器分块） | 变化 |
|------|---------------------|-------------------|------|
| GFLOPS | 182.12 | **248.51** | 升 36.4% |
| Duration | 29.84 ms | **16.89 ms** | 降 43.4% |
| Elapsed Cycles | 46,313,231 | **26,120,020** | 降 43.6% |
| L1/TEX Cache Throughput | 79.94% | **52.27%** | 降 27.7 个百分点 |
| DRAM Throughput | 19.08% | 10.84% | 降 8.2 个百分点 |
| L2 Cache Throughput | 13.32% | 15.65% | 略升 |
| Compute (SM) Throughput | 79.80% | **28.49%** | 降 51.3 个百分点 |

&emsp;&emsp;三个关键指标的下降说明寄存器分块确实把访存压力大幅降下来了。`L1/TEX Cache Throughput` 从 79.94% 降到 **52.27%**，因为 `A_s`、`B_s` 的每个元素被复用 `MR` 或 `NR` 次，共享内存的读取次数降到 `1/MR + 1/NR = 1/4 + 1/4 = 1/2`。`Compute (SM) Throughput` 从 79.80% 降到 **28.49%**，说明 LSU 不再是瓶颈——之前 92% 的“Compute”主要来自 LSU 忙碌，现在 LSU 压力小了，SM 有更多资源给 FMA。`Elapsed Cycles` 从 4631 万降到 **2612 万**，降幅 43.6%。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gpu_gemm_v3_xxxxxxxxxxxxxxxxxxx.png)

&emsp;&emsp;但有一个数字值得注意：**`Compute (SM) Throughput` 只有 28.49%，说明 FMA 单元远未饱和**。按 Roofline 模型，此时计算单元利用率只有 28%，理论上还有 3.5 倍提升空间。瓶颈从 L1/LSU 转移到了别处。可能的原因有三个：**第一，共享内存的 bank conflict**：`A_s[ty*MR+i][k]` 的访问中，同一 warp 内 32 个线程的 `ty` 从 0 到 7、`tx` 从 0 到 3，访问的是 `A_s[0..31][k]` 这一列，步长 32 个 float，产生 bank conflict。修法是把 `A_s`、`B_s` 的列数从 `SG_TILE` 增加到 `SG_TILE + 1`。**第二，Occupancy 不足**：block 只有 64 个线程，RTX 3050 每个 SM 最多支持 1536 个线程，理论上可以同时驻留 24 个 block。但如果寄存器用量或共享内存用量过高，实际驻留的 block 数会减少，延迟隐藏不充分。**第三，FMA 发射效率**：`acc[i][j] += a_frag[i] * b_frag[j]` 这个双重循环，编译器是否真的把它编译成了 `MR × NR` 条独立的 FMA，还是插入了额外的 load/store？需要看 SASS 才能确认。

&emsp;&emsp;目前的版本还没有达到最优，需要继续调参优化。
- **第一，加 padding 消除 bank conflict**
- **第二，增大 `MR`、`NR`**

---

```cpp
// ---------- 寄存器分块 + 共享内存调参：v4 ----------
// 相对 v3 的两处改动（其余结构不变）：
// 1) 共享内存加 padding：[TILE][TILE+1]，同列相邻行的元素错开 1 个 bank，
//    消除 A_s[ty*MR+i][k] / B_s[k][tx*NR+j] 的列广播式 bank conflict
// 2) TILE 32 -> 64、block 8x8 -> 16x16（MR/NR 保持 4）：每 block 算 64x64，
//    数据复用翻倍，全局访存量减半
#define SG_TILE 64
#define SG_MR 4
#define SG_NR 4
#define SG_PAD 1

__global__ void sgemm_tiled_kernel_v4(int M, int N, int K,
                                      const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C) {
    // +SG_PAD 消 bank conflict
    __shared__ float A_s[SG_TILE][SG_TILE + SG_PAD];
    __shared__ float B_s[SG_TILE][SG_TILE + SG_PAD];

    const int row_base = blockIdx.y * SG_TILE;
    const int col_base = blockIdx.x * SG_TILE;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    // 当前线程负责的 MR x NR 小块左上角
    const int row = row_base + ty * SG_MR;
    const int col = col_base + tx * SG_NR;

    // 每个线程维护 MR x NR 个累加器
    float acc[SG_MR][SG_NR];
    for (int i = 0; i < SG_MR; ++i)
        for (int j = 0; j < SG_NR; ++j)
            acc[i][j] = 0.0f;

    // 32x32 -> 64x64 tile：TILE*TILE/(blockDim.x*blockDim.y) = 16 个元素/线程
    // 加载循环按 blockDim 双向步进，自动适配 block 尺寸
    for (int k_base = 0; k_base < K; k_base += SG_TILE) {
        // ---- 加载 A 子块：A_s[i][j]，i 覆盖 TILE 行、j 覆盖 TILE 列 ----
        for (int i = ty; i < SG_TILE; i += blockDim.y) {
            const int a_row = row_base + i;
            for (int j = tx; j < SG_TILE; j += blockDim.x) {
                const int a_col = k_base + j;
                if (a_row < M && a_col < K)
                    A_s[i][j] = A[a_row * K + a_col];
                else
                    A_s[i][j] = 0.0f;
            }
        }

        // ---- 加载 B 子块：B_s[i][j]，i 覆盖 TILE 行（k 维）、j 覆盖 TILE 列 ----
        for (int j = tx; j < SG_TILE; j += blockDim.x) {
            const int b_col = col_base + j;
            for (int i = ty; i < SG_TILE; i += blockDim.y) {
                const int b_row = k_base + i;
                if (b_row < K && b_col < N)
                    B_s[i][j] = B[b_row * N + b_col];
                else
                    B_s[i][j] = 0.0f;
            }
        }

        __syncthreads();

        // ---- 在共享内存上完成 TILE 长度的乘加 ----
        for (int k = 0; k < SG_TILE; ++k) {
            float a_frag[SG_MR];
            float b_frag[SG_NR];
            for (int i = 0; i < SG_MR; ++i)
                a_frag[i] = A_s[ty * SG_MR + i][k];
            for (int j = 0; j < SG_NR; ++j)
                b_frag[j] = B_s[k][tx * SG_NR + j];

            for (int i = 0; i < SG_MR; ++i)
                for (int j = 0; j < SG_NR; ++j)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }

        __syncthreads();
    }

    // ---- 写回 C ----
    for (int i = 0; i < SG_MR; ++i) {
        for (int j = 0; j < SG_NR; ++j) {
            const int r = row + i;
            const int c = col + j;
            if (r < M && c < N)
                C[r * N + c] = acc[i][j];
        }
    }
}

void sgemm_cuda_tiled_v4(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    const size_t a_bytes = sizeof(float) * static_cast<size_t>(M) * K;
    const size_t b_bytes = sizeof(float) * static_cast<size_t>(K) * N;
    const size_t c_bytes = sizeof(float) * static_cast<size_t>(M) * N;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, A, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, B, b_bytes, cudaMemcpyHostToDevice));

    // block 取 (TILE/MR) x (TILE/NR) = 16x16，每个线程负责 MR x NR 个 C 元素
    dim3 block(SG_TILE / SG_NR, SG_TILE / SG_MR);
    dim3 grid((N + SG_TILE - 1) / SG_TILE, (M + SG_TILE - 1) / SG_TILE);
    sgemm_tiled_kernel_v4<<<grid, block>>>(M, N, K, d_a, d_b, d_c);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C, d_c, c_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

```

&emsp;&emsp;当前 v4 在 2048³ 下达到 **995.7 GFLOPS**，相比 v3 的 843.0 GFLOPS 提升 18.1%，相比 v2 的 567.0 GFLOPS 提升 75.6%，相比 v1 的 468.8 GFLOPS 提升 112.4%。v4 的核心改动有两处：一是**共享内存加 padding**，把 `A_s`、`B_s` 的列数从 `SG_TILE` 改成 `SG_TILE + 1`，使同一列相邻行的元素错开一个 bank，消除了 v3 中 `A_s[ty*MR+i][k]` 和 `B_s[k][tx*NR+j]` 的列广播式 bank conflict；二是**把 TILE 从 32 提到 64、block 从 8×8 提到 16×16**，每个 block 计算 64×64 的输出块，数据复用翻倍，全局访存量减半。ncu 数据印证了这两项改动的效果：`L1/TEX Cache Throughput` 从 v3 的 52.27% 降到 **41.85%**，`DRAM Throughput` 从 10.84% 升到 **17.54%**，`Elapsed Cycles` 从 2612 万降到 **2074 万**，`Duration` 从 16.89ms 降到 **13.43ms**。不过 `Compute (SM) Throughput` 只有 **33.45%**，FMA 单元仍远未饱和，说明瓶颈已经不在访存，而在共享内存到寄存器这一段还有优化空间。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gpu_gemm_v4_xxxxxxxxxxxxxxxxxxx.png)

```cpp
    ----------------------- ------------- -------------
    Metric Name               Metric Unit  Metric Value
    ----------------------- ------------- -------------
    DRAM Frequency          cycle/nsecond          6.79
    SM Frequency            cycle/nsecond          1.54
    Elapsed Cycles                  cycle    20,744,917
    Memory Throughput                   %         41.77
    DRAM Throughput                     %         17.54
    Duration                      msecond         13.43
    L1/TEX Cache Throughput             %         41.85
    L2 Cache Throughput                 %         13.63
    SM Active Cycles                cycle 20,699,892.60
    Compute (SM) Throughput             %         33.45
    ----------------------- ------------- -------------
```

### 3.5 访存指令向量化：float4 与 A 转置

&emsp;&emsp;v4 通过 padding 消除 bank conflict、增大 tile 到 64×64，把 `Duration` 从 13.43ms 降到 7.02ms，2048³ 下 GFLOPS 从 843.0 提升到 916.1。但 ncu 数据显示，`L1/TEX Cache Throughput` 仍有 69.60%，`Memory Throughput` 68.60%，是三项吞吐中最高的两项。这说明数据复用虽然改善了，但**每条 FMA 周围的访存指令数仍然偏多**。

&emsp;&emsp;具体来说，v4 的内层循环每个 k 步要执行 8 次共享内存标量读：4 次 `A_s[ty*4+i][k]`（i 从 0 到 3）、4 次 `B_s[k][tx*4+j]`（j 从 0 到 3）。这 8 次读服务 16 次 FMA，访存/计算比为 1:2。LSU 被这些标量读指令塞满，FMA 单元反而空闲——这正是 v4 的 `Compute (SM) Throughput` 只有 33.45% 的原因。

&emsp;&emsp;要减少访存指令数，最直接的办法是**向量化**：用 `float4` 一次读 4 个 float，把 8 次标量读压缩到 2 次向量读。但 `A_s` 的访问模式是列访问，`A_s[ty*4+i][k]` 沿 `i` 变化、`k` 固定，地址步长为 `TILE` 个 float，不连续，无法直接向量化。解决办法是**在加载阶段把 A 转置存放**：共享内存里存 `A_sT[k][i]`，让 `k` 成为行、`i` 成为列。这样内层循环读 `A_sT[k][ty*4..ty*4+3]` 时，4 个元素连续，可以用一条 `float4` 读完成。

&emsp;&emsp;`B_s` 的访问模式本来就是行访问，`B_s[k][tx*4+j]` 沿 `j` 连续，可以直接用 `float4` 读。完整的 v5 实现：

```cpp
// ---------- 访存指令向量化 GEMM：v5 ----------
// 相对 v4 的三处改动：
// 1) A 子块转置存放 A_sT[k][i]（k 为连续维），a_frag 变 float4 行读
// 2) PAD=4：行距 TILE+4=68，既是 4 的倍数（float4 16B 对齐），
//    68%32=4 又保持免 bank conflict
// 3) b_frag 同为 float4（B_s[k][tx*4..+3] 连续）
//    全局加载也 float4 化（A/B 各 4 次 float4/线程/k 轮）
// 每 k 步 smem 指令 8 -> 2，全局加载指令 32 -> 8。
// 前置条件：K%4==0、N%4==0、A/B 指针 16B 对齐，否则回退标量模板。
#define V5_TILE 64
#define V5_MR 4
#define V5_NR 4
#define V5_PAD 4

template <int MR>
__global__ void sgemm_vec_kernel_v5(int M, int N, int K,
                                    const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C) {
    // A_sT[k][i] = A[row_base+i][k_base+k]，转置存放让 k 成为连续维
    __shared__ float A_sT[V5_TILE][V5_TILE + V5_PAD];
    __shared__ float B_s[V5_TILE][V5_TILE + V5_PAD];

    const int row_base = blockIdx.y * V5_TILE;
    const int col_base = blockIdx.x * V5_TILE;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int row = row_base + ty * MR;
    const int col = col_base + tx * V5_NR;

    float acc[MR][V5_NR];
#pragma unroll
    for (int i = 0; i < MR; ++i)
#pragma unroll
        for (int j = 0; j < V5_NR; ++j) acc[i][j] = 0.0f;

    for (int k_base = 0; k_base < K; k_base += V5_TILE) {
        // ---- 加载 A（float4，转置写入 A_sT）：每 i 一条 float4 ----
        for (int i = ty; i < V5_TILE; i += blockDim.y) {
            const int a_row = row_base + i;
            const int a_col = k_base + tx * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            // K%4==0 且 a_col%4==0 => a_col < K 蕴含 a_col+3 < K
            if (a_row < M && a_col < K)
                v = *reinterpret_cast<const float4*>(A + a_row * K + a_col);
            A_sT[tx * 4 + 0][i] = v.x;
            A_sT[tx * 4 + 1][i] = v.y;
            A_sT[tx * 4 + 2][i] = v.z;
            A_sT[tx * 4 + 3][i] = v.w;
        }

        // ---- 加载 B（float4 行存）：j4 为 float4 列基 ----
        for (int j4 = tx; j4 < V5_TILE / 4; j4 += blockDim.x) {
            const int b_col = col_base + j4 * 4;
            for (int i = ty; i < V5_TILE; i += blockDim.y) {
                const int b_row = k_base + i;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                // N%4==0 且 b_col%4==0 => b_col < N 蕴含 b_col+3 < N
                if (b_row < K && b_col < N)
                    v = *reinterpret_cast<const float4*>(B + b_row * N + b_col);
                *reinterpret_cast<float4*>(&B_s[i][j4 * 4]) = v;
            }
        }

        __syncthreads();

        // ---- 每 k 步 MR/4 + 1 条 float4 smem 读 + MR*4 FMA ----
#pragma unroll 4
        for (int k = 0; k < V5_TILE; ++k) {
            float af[MR];
#pragma unroll
            for (int a4i = 0; a4i < MR / 4; ++a4i) {
                const float4 a4 =
                    *reinterpret_cast<const float4*>(&A_sT[k][ty * MR + a4i * 4]);
                af[a4i * 4 + 0] = a4.x;
                af[a4i * 4 + 1] = a4.y;
                af[a4i * 4 + 2] = a4.z;
                af[a4i * 4 + 3] = a4.w;
            }
            const float4 b4 = *reinterpret_cast<const float4*>(&B_s[k][tx * V5_NR]);
            const float bf[V5_NR] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
            for (int i = 0; i < MR; ++i)
#pragma unroll
                for (int j = 0; j < V5_NR; ++j) acc[i][j] += af[i] * bf[j];
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MR; ++i)
#pragma unroll
        for (int j = 0; j < V5_NR; ++j) {
            const int r = row + i;
            const int c = col + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

template <int MR>
void LaunchVecV5(int M, int N, int K, const float* A, const float* B, float* C) {
    const size_t a_bytes = sizeof(float) * static_cast<size_t>(M) * K;
    const size_t b_bytes = sizeof(float) * static_cast<size_t>(K) * N;
    const size_t c_bytes = sizeof(float) * static_cast<size_t>(M) * N;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, A, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, B, b_bytes, cudaMemcpyHostToDevice));

    dim3 block(V5_TILE / V5_NR, V5_TILE / MR);
    dim3 grid((N + V5_TILE - 1) / V5_TILE, (M + V5_TILE - 1) / V5_TILE);
    sgemm_vec_kernel_v5<MR><<<grid, block>>>(M, N, K, d_a, d_b, d_c);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C, d_c, c_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

void sgemm_cuda_tuned_v5(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    const bool aligned = (K % 4 == 0) && (N % 4 == 0) &&
                         (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(B) % 16 == 0);
    if (!aligned) {
        LaunchTuned<V5_TILE, V5_MR, V5_NR>(M, N, K, A, B, C);  // 标量回退
        return;
    }
    int mr = 4;
    if (const char* env = std::getenv("GEMM_V5_MR")) {
        const int v = std::atoi(env);
        if (v == 4 || v == 8) mr = v;
    }
    if (mr == 8)
        LaunchVecV5<8>(M, N, K, A, B, C);
    else
        LaunchVecV5<4>(M, N, K, A, B, C);
}

static Registrar reg_cuda_tuned_v5("cuda_naive_v5", &sgemm_cuda_tuned_v5);
```

&emsp;&emsp;相比 v4，有四处关键改动。
- **第一，A 子块转置存放。** 共享内存从 `A_s[i][j]` 变成 `A_sT[k][i]`，其中 `k` 是连续维。加载时每个线程从 A 读一个 `float4`，转置写入 `A_sT` 的 4 个位置。这样内层循环读 `A_sT[k][ty*MR..ty*MR+MR-1]` 时，`MR` 个元素连续，可以用 `MR/4` 条 `float4` 读完成。
- **第二，PAD=4 而非 1。** v4 的 `PAD=1` 是为了消除 bank conflict，但行距 65 不是 4 的倍数，`float4` 读会跨 bank。v5 把 PAD 改成 4，行距变成 `TILE + 4 = 68`。68 是 4 的倍数，满足 `float4` 的 16 字节对齐；同时 `68 % 32 = 4`，列访问的步长是 68 而不是 64，不同行的线程落在不同 bank 上，bank conflict 仍然被消除。
- **第三，B 子块的 float4 行存。** `B_s[k][tx*4..+3]` 本来就连续，直接用 `float4` 写回。加载时用 `j4` 作为 `float4` 列基，每个线程负责 `V5_TILE / 4` 个 `float4` 列。
- **第四，全局加载也 float4 化。** A 的加载用 `float4` 读、转置写；B 的加载用 `float4` 读、`float4` 写。每线程每 k 轮从 32 次标量全局加载降到 8 次 `float4` 加载。

&emsp;&emsp;实测结果如下：

```cpp
BM_sgemm/cuda_naive_v5/256x256x256      259402 ns    GFLOPS=129.48
BM_sgemm/cuda_naive_v5/1024x1024x1024  2518615 ns    GFLOPS=852.90
BM_sgemm/cuda_naive_v5/2048x2048x2048 12751851 ns    GFLOPS=1347.64

Performance counter stats (ncu --set roofline, 2048³):
    Elapsed Cycles                  cycle    10,886,199
    Duration                      msecond          7.02
    Memory Throughput                   %         68.60
    DRAM Throughput                     %         32.02
    L1/TEX Cache Throughput             %         69.60
    L2 Cache Throughput                 %         25.21
    Compute (SM) Throughput             %         42.69
    SM Active Cycles                cycle    10,724,442
```

&emsp;&emsp;这组数据与 v4 对比：

| 指标 | v4 | v5 | 变化 |
|------|-----|-----|------|
| GFLOPS（2048³） | 916.12 | **1347.64** | 升 47.1% |
| Duration | 13.43 ms | **7.02 ms** | 降 47.7% |
| Elapsed Cycles | 20,744,917 | **10,886,199** | 降 47.5% |
| DRAM Throughput | 17.54% | 32.02% | 升 82.6% |
| L1/TEX Cache Throughput | 41.85% | 69.60% | 升 66.3% |
| Memory Throughput | 41.77% | 68.60% | 升 64.2% |
| Compute (SM) Throughput | 33.45% | **42.69%** | 升 27.6% |

&emsp;&emsp;`Duration` 和 `Elapsed Cycles` 都降了 47.5%，GFLOPS 提升 47.1%，说明 float4 向量化确实把访存指令数减下来了。`Compute (SM) Throughput` 从 33.45% 升到 42.69%，FMA 单元被喂得更满。但 `L1/TEX Cache Throughput` 和 `Memory Throughput` 反而升到了 69.60% 和 68.60%，说明 float4 读一次搬 16 字节，L1 和内存的数据搬运量更集中，带宽压力上升。这是好事——说明内核从“LSU 被标量指令塞满”转向了“L1/内存带宽被充分利用”。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gpu_gemm_v5_xxxxxxxxxxxxxxxxxxx.png)

&emsp;&emsp;共享内存的占用从 v4 的 `2 × 64 × 65 × 4 = 33.3 KB` 涨到 v5 的 `2 × 64 × 68 × 4 = 34.8 KB`。RTX 3050 每个 SM 有 100 KB 共享内存，v4 理论上能驻留 3 个 block，v5 也是 3 个（34.8 × 3 = 104.4 KB，略超，实际可能只有 2 个）。Occupancy 略有下降，但用占用率换来了 smem 指令从 8 条降到 2 条、全局加载指令从 32 条降到 8 条，由实测数据裁决——GFLOPS 提升 47.1% 说明这笔交易划算。

&emsp;&emsp;不过 v5 的 `Compute (SM) Throughput` 只有 42.69%，FMA 单元仍有大量空闲。下一步的优化方向是**双缓冲**：当 `DRAM Throughput` 涨到 32.02%、`Memory Throughput` 到 68.60% 时，全局内存加载开始成为可见瓶颈，双缓冲能把加载和计算重叠。具体做法是把 `A_sT`、`B_s` 各分成两份，一份用于当前 K 切片的计算，另一份用于预加载下一个 K 切片。如果配合 Ampere 的 `cp.async` 异步拷贝指令，加载和计算能真正并行，而不是靠 `__syncthreads()` 逻辑重叠。

### 3.6 双缓冲与访存计算重叠

&emsp;&emsp;3.5 节的 v5 通过 float4 向量化把每 k 步的共享内存指令从 8 条压到 2 条，2048³ 下 GFLOPS 达到 **1476.9**。但 v5 的 ncu 数据显示 `Memory Throughput` 68.60%、`DRAM Throughput` 32.02%，访存已经开始成为可见瓶颈。此时全局内存的加载与共享内存的计算之间仍然是**串行**的：先加载 A、B 子块，`__syncthreads()` 等待加载完成，再做乘加，再 `__syncthreads()` 进入下一轮。加载期间 FMA 单元空闲，计算期间全局内存空闲。**双缓冲**的目标就是让这两段重叠。

&emsp;&emsp;双缓冲的思路是：把共享内存分成两份，一份用于当前 K 切片的计算，另一份用于预加载下一个 K 切片。在计算 `cur` buf 的同时，把下一个 K 切片加载到 `nxt` buf。这样全局内存加载与共享内存计算在时间上重叠，隐藏全局内存延迟。双缓冲的同步是关键：循环开头用一次 `__syncthreads()` 确保 `cur` buf 加载完成，循环末尾再用一次 `__syncthreads()` 确保本轮计算完成后 `nxt` buf 才被下一轮覆盖。

&emsp;&emsp;更精细的做法是**把全局加载拆成“发起”和“落地”两段**：在计算之前先把下一个 K 切片的全局数据读进寄存器（发起），然后执行当前切片的乘加（此时全局加载在飞行中），最后把寄存器里的数据写入 `nxt` buf（落地）。这样全局加载的延迟被当前切片的计算完整覆盖，而不是靠 `__syncthreads()` 逻辑重叠。下面给出完整实现。

```cpp
// ---------- 双缓冲 + 访存计算重叠 GEMM：v6 ----------
// 相对 v5 的改动：
// 1) A_sT、B_s 各分成两份，cur/nxt 交替
// 2) 全局加载拆成“发起”（读入寄存器）和“落地”（写入 nxt buf）两段，
//    中间夹当前切片的乘加，让全局加载的延迟被计算覆盖
// 3) 两次 __syncthreads()：循环开头确保 cur 可读，循环末尾确保 nxt 可覆盖
// 4) 前置条件与 v5 相同：K%4==0、N%4==0、A/B 指针 16B 对齐
constexpr int V6_TILE = 64;
constexpr int V6_NR = 4;
constexpr int V6_PAD = 4;

template <int TT, int MR>
__global__ void sgemm_double_buffer_kernel_v6(int M, int N, int K,
                                              const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C) {
    constexpr int STRIDE = TT + V6_PAD;
    __shared__ float A_sT[2][TT][STRIDE];
    __shared__ float B_s[2][TT][STRIDE];

    const int row_base = blockIdx.y * TT;
    const int col_base = blockIdx.x * TT;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int row = row_base + ty * MR;
    const int col = col_base + tx * V6_NR;

    float acc[MR][V6_NR];
#pragma unroll
    for (int i = 0; i < MR; ++i)
#pragma unroll
        for (int j = 0; j < V6_NR; ++j) acc[i][j] = 0.0f;

    const int num_k_tiles = (K + TT - 1) / TT;

    // ---- 预加载第 0 个 K 切片到 buf 0 ----
    {
        for (int i = ty; i < TT; i += blockDim.y) {
            const int a_row = row_base + i;
            const int a_col = tx * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (a_row < M && a_col < K)
                v = *reinterpret_cast<const float4*>(A + a_row * K + a_col);
            A_sT[0][tx * 4 + 0][i] = v.x;
            A_sT[0][tx * 4 + 1][i] = v.y;
            A_sT[0][tx * 4 + 2][i] = v.z;
            A_sT[0][tx * 4 + 3][i] = v.w;
        }
        for (int j4 = tx; j4 < TT / 4; j4 += blockDim.x) {
            const int b_col = col_base + j4 * 4;
            for (int i = ty; i < TT; i += blockDim.y) {
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (i < K && b_col < N)
                    v = *reinterpret_cast<const float4*>(B + i * N + b_col);
                *reinterpret_cast<float4*>(&B_s[0][i][j4 * 4]) = v;
            }
        }
    }

    __syncthreads();  // buf 0 就绪，可以开始流水

    // ---- 沿 K 方向遍历，双缓冲交替 ----
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int cur = kt & 1;
        const int nxt = cur ^ 1;
        const bool has_next = (kt + 1 < num_k_tiles);

        // ---- a) 访存发起：tile kt+1 的全局加载先进寄存器 ----
        float4 a_reg[MR];
        float4 b_reg[MR];
        if (has_next) {
            const int k_base = (kt + 1) * TT;
#pragma unroll
            for (int n = 0; n < MR; ++n) {
                const int a_row = row_base + ty + n * (TT / MR);
                const int a_col = k_base + tx * 4;
                if (a_row < M && a_col < K)
                    a_reg[n] =
                        *reinterpret_cast<const float4*>(A + a_row * K + a_col);
                else
                    a_reg[n] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
#pragma unroll
            for (int n = 0; n < MR; ++n) {
                const int b_row = k_base + ty + n * (TT / MR);
                const int b_col = col_base + tx * 4;
                if (b_row < K && b_col < N)
                    b_reg[n] =
                        *reinterpret_cast<const float4*>(B + b_row * N + b_col);
                else
                    b_reg[n] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }

        // ---- b) 计算：在 cur buf 上完成乘加（全局加载此时在飞行中）----
#pragma unroll 4
        for (int k = 0; k < TT; ++k) {
            float af[MR];
#pragma unroll
            for (int a4i = 0; a4i < MR / 4; ++a4i) {
                const float4 a4 =
                    *reinterpret_cast<const float4*>(&A_sT[cur][k][ty * MR + a4i * 4]);
                af[a4i * 4 + 0] = a4.x;
                af[a4i * 4 + 1] = a4.y;
                af[a4i * 4 + 2] = a4.z;
                af[a4i * 4 + 3] = a4.w;
            }
            const float4 b4 =
                *reinterpret_cast<const float4*>(&B_s[cur][k][tx * V6_NR]);
            const float bf[V6_NR] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
            for (int i = 0; i < MR; ++i)
#pragma unroll
                for (int j = 0; j < V6_NR; ++j) acc[i][j] += af[i] * bf[j];
        }

        // ---- c) 访存落地：寄存器 -> nxt buf ----
        if (has_next) {
#pragma unroll
            for (int n = 0; n < MR; ++n) {
                const int i = ty + n * (TT / MR);
                A_sT[nxt][tx * 4 + 0][i] = a_reg[n].x;
                A_sT[nxt][tx * 4 + 1][i] = a_reg[n].y;
                A_sT[nxt][tx * 4 + 2][i] = a_reg[n].z;
                A_sT[nxt][tx * 4 + 3][i] = a_reg[n].w;
                *reinterpret_cast<float4*>(&B_s[nxt][i][tx * 4]) = b_reg[n];
            }
        }

        __syncthreads();  // 计算完成 + nxt 写入完成，才能切换 buf
    }

#pragma unroll
    for (int i = 0; i < MR; ++i)
#pragma unroll
        for (int j = 0; j < V6_NR; ++j) {
            const int r = row + i;
            const int c = col + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

template <int TT, int MR>
void LaunchDoubleBufferV6(int M, int N, int K,
                          const float* A, const float* B, float* C) {
    const size_t a_bytes = sizeof(float) * static_cast<size_t>(M) * K;
    const size_t b_bytes = sizeof(float) * static_cast<size_t>(K) * N;
    const size_t c_bytes = sizeof(float) * static_cast<size_t>(M) * N;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, A, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, B, b_bytes, cudaMemcpyHostToDevice));

    dim3 block(TT / V6_NR, TT / MR);
    dim3 grid((N + TT - 1) / TT, (M + TT - 1) / TT);
    sgemm_double_buffer_kernel_v6<TT, MR><<<grid, block>>>(M, N, K, d_a, d_b, d_c);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C, d_c, c_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

void sgemm_cuda_double_buffer_v6(int M, int N, int K,
                                 const float* A, const float* B, float* C) {
    const bool aligned = (K % 4 == 0) && (N % 4 == 0) &&
                         (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(B) % 16 == 0);
    if (!aligned) {
        LaunchTuned<64, 4, 4>(M, N, K, A, B, C);
        return;
    }
    int tile = 64, mr = 4;
    if (const char* env = std::getenv("GEMM_V6_TILE")) {
        const int v = std::atoi(env);
        if (v == 32 || v == 48 || v == 64) tile = v;
    }
    if (const char* env = std::getenv("GEMM_V6_MR")) {
        const int v = std::atoi(env);
        if (v == 4 || v == 8) mr = v;
    }
    if (tile == 32) {
        if (mr == 8) LaunchDoubleBufferV6<32, 8>(M, N, K, A, B, C);
        else         LaunchDoubleBufferV6<32, 4>(M, N, K, A, B, C);
    } else if (tile == 48) {
        if (mr == 8) LaunchDoubleBufferV6<48, 8>(M, N, K, A, B, C);
        else         LaunchDoubleBufferV6<48, 4>(M, N, K, A, B, C);
    } else {
        if (mr == 8) LaunchDoubleBufferV6<64, 8>(M, N, K, A, B, C);
        else         LaunchDoubleBufferV6<64, 4>(M, N, K, A, B, C);
    }
}

static Registrar reg_cuda_double_buffer_v6("cuda_naive_v6", &sgemm_cuda_double_buffer_v6);
```

&emsp;&emsp;相比 v5，这段代码有四处关键改动：
- **第一，共享内存翻倍**：`A_sT`、`B_s` 从 `[TT][STRIDE]` 变成 `[2][TT][STRIDE]`，每份是 `2 × 64 × 68 × 4 = 34.8 KB`，两份合计 **69.6 KB**。RTX 3050 每个 SM 有 100 KB 共享内存，双缓冲后每个 SM 只能驻留 **1 个 block**。
- **第二，全局加载拆成“发起”和“落地”两段**：在计算之前先把下一个 K 切片的全局数据读进寄存器 `a_reg`、`b_reg`（发起），然后执行当前切片的乘加（此时全局加载在飞行中），最后把寄存器里的数据写入 `nxt` buf（落地）。这样全局加载的延迟被当前切片的计算完整覆盖。
- **第三，循环内预加载下一个 K 切片**：加载和乘加在同一轮循环里，硬件可以把加载指令和 FMA 指令交错发射。
- **第四，两次 `__syncthreads()` 的位置**：第一次在循环开头，确保 `cur` buf 加载完成；第二次在循环末尾，确保本轮计算完成且 `nxt` buf 已写入，才能切换 buf。

&emsp;&emsp;实测结果如下：

```cpp
BM_sgemm/cuda_naive_v6/256x256x256      139249 ns    GFLOPS=240.99
BM_sgemm/cuda_naive_v6/1024x1024x1024  2451679 ns    GFLOPS=876.24
BM_sgemm/cuda_naive_v6/2048x2048x2048 11723572 ns    GFLOPS=1465.65

Performance counter stats (ncu --set roofline, 2048³):
    Elapsed Cycles                  cycle    11,885,824
    Duration                      msecond          7.68
    Memory Throughput                   %         62.07
    DRAM Throughput                     %         36.17
    L1/TEX Cache Throughput             %         63.08
    L2 Cache Throughput                 %         25.94
    Compute (SM) Throughput             %         37.70
    SM Active Cycles                cycle    11,686,190
```

&emsp;&emsp;这组数据与 v5 对比：

| 指标 | v5 | v6 | 变化 |
|------|-----|-----|------|
| GFLOPS（2048³） | 1476.95 | **1465.65** | 降 0.8% |
| Duration | 7.02 ms | 7.68 ms | 升 9.4% |
| Elapsed Cycles | 10,886,199 | 11,885,824 | 升 9.2% |
| DRAM Throughput | 32.02% | **36.17%** | 升 13.0% |
| L1/TEX Cache Throughput | 69.60% | 63.08% | 降 9.4% |
| Memory Throughput | 68.60% | 62.07% | 降 9.5% |
| Compute (SM) Throughput | 42.69% | 37.70% | 降 11.7% |

&emsp;&emsp;**v6 的双缓冲没有带来性能提升，反而略降 0.8%。** `Duration` 从 7.02ms 涨到 7.68ms，`Compute (SM) Throughput` 从 42.69% 降到 37.70%。`DRAM Throughput` 从 32.02% 升到 36.17%，说明全局内存确实被用得更满，但计算单元利用率反而下降。

&emsp;&emsp;原因在于**共享内存翻倍导致 Occupancy 从 2 个 block/SM 降到 1 个 block/SM**。双缓冲的收益（访存计算重叠）被 Occupancy 下降的损失抵消。在 GPU 上，Occupancy 是隐藏延迟的主要手段。当每个 SM 只有 1 个 block 时，warp 数量不足，一旦某个 warp 在等共享内存或全局内存，调度器没有其他 warp 可以切换，流水线就空转。v5 的 2 个 block/SM 提供了更充足的 warp 池，反而比 v6 的 1 个 block/SM 更容易隐藏延迟。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gpu_gemm_v6_xxxxxxxxxxxxxxxxxxx.png)

&emsp;&emsp;这也说明**双缓冲并非总是有效**。它的收益依赖于两个条件：全局内存延迟确实是瓶颈，且双缓冲带来的 Occupancy 下降不严重。v5 的 `DRAM Throughput` 只有 32.02%，全局内存延迟还没有严重到需要双缓冲来掩盖；而双缓冲把共享内存翻倍，直接砍掉了一半 Occupancy，损失大于收益。

### 3.7 cp.async 异步拷贝

&emsp;&emsp;3.6 节的双缓冲尝试得出了一个负面结论：共享内存翻倍导致 Occupancy 从 2 个 block/SM 降到 1 个 block/SM，性能反而下降 0.8%。这说明在 RTX 3050 上，**用翻倍共享内存换访存计算重叠是一笔不划算的交易**。但 v5 的 `DRAM Throughput` 32.02%、`Memory Throughput` 68.60% 说明访存压力确实存在。有没有办法**不翻倍共享内存**也能实现异步加载？

&emsp;&emsp;Ampere 架构（CC 8.0+）提供了硬件异步拷贝指令 **`cp.async`**。它从全局内存直接拷贝到共享内存，**绕过寄存器堆**，由异步拷贝单元执行，不占用 warp 的执行周期，也不产生 `STS` 指令。等待拷贝完成用 `cp.async.wait_group`。这样，全局内存加载和计算可以真正并行——加载由异步单元执行，warp 继续做 FMA。

&emsp;&emsp;`cp.async` 的代价是**memcpy 语义无法转置**。v5 的 A 子块是转置存放的 `A_sT[k][i]`，让 `a_frag` 能用一条 `LDS.128` 读。但 `cp.async` 只能把全局内存的连续 16 字节原样搬到共享内存，不能转置。所以 v7 里 A 只能按全局行布局存 `A_s[i][k]`，`a_frag` 从 v5 的 1 条 `LDS.128` 退化为 4 条 `LDS.32`。不过同一 `ty` 的 8 个线程读同一地址，硬件按广播处理，**事务数不劣化，只是指令数增加**。净收益由实测数据裁决。

&emsp;&emsp;下面给出完整实现。

```cpp
// ---------- cp.async 异步拷贝 GEMM：v7 ----------
// 相对 v5 的改动：
// 1) 用 cp.async 从全局内存直接拷贝到共享内存，绕过寄存器堆，不占执行周期
// 2) 单缓冲时序不变（双 sync），TILE=64、block 16x16、MR=NR=4、PAD=4 与 v5 一致
// 3) A 按全局行布局存 A_s[i][k]，a_frag 从 1 条 LDS.128 退化为 4 条 LDS.32
// 4) cp.async 的 src-size 为 0 时目的端 zfill 填零，替代分支置零
namespace {

constexpr int V7_TILE = 64;
constexpr int V7_MR = 4;
constexpr int V7_NR = 4;
constexpr int V7_PAD = 4;

// 16B cp.async + zfill：valid=false 时 src_bytes=0，目的端填零、不发起读
__device__ __forceinline__ void cp_async_16B(float* smem_dst,
                                             const float* gmem_src, bool valid) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    const unsigned bytes = valid ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
                 "l"(gmem_src), "r"(bytes));
}

__device__ __forceinline__ void cp_async_commit_wait() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// 前置条件（host 侧保证）：K%4==0、N%4==0、A/B 指针 16B 对齐
template <int MR>
__global__ void sgemm_cpasync_kernel_v7(int M, int N, int K,
                                        const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C) {
    __shared__ float A_s[V7_TILE][V7_TILE + V7_PAD];  // 非转置：A_s[i][k]
    __shared__ float B_s[V7_TILE][V7_TILE + V7_PAD];

    const int row_base = blockIdx.y * V7_TILE;
    const int col_base = blockIdx.x * V7_TILE;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int row = row_base + ty * MR;
    const int col = col_base + tx * V7_NR;

    float acc[MR][V7_NR];
#pragma unroll
    for (int i = 0; i < MR; ++i)
#pragma unroll
        for (int j = 0; j < V7_NR; ++j) acc[i][j] = 0.0f;

    for (int k_base = 0; k_base < K; k_base += V7_TILE) {
        // ---- cp.async 搬运：每线程 A/B 各 4 条 16B（4 行 x 1 float4）----
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int i = ty + n * 16;
            cp_async_16B(&A_s[i][tx * 4],
                         A + (row_base + i) * K + k_base + tx * 4,
                         (row_base + i < M) && (k_base + tx * 4 < K));
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int i = ty + n * 16;
            cp_async_16B(&B_s[i][tx * 4],
                         B + (k_base + i) * N + col_base + tx * 4,
                         (k_base + i < K) && (col_base + tx * 4 < N));
        }
        cp_async_commit_wait();
        __syncthreads();

        // ---- 每 k 步：a_frag 4x LDS.32（广播）+ b_frag 1x LDS.128 + MR*4 FMA ----
#pragma unroll 4
        for (int k = 0; k < V7_TILE; ++k) {
            float af[MR];
#pragma unroll
            for (int i = 0; i < MR; ++i) af[i] = A_s[ty * MR + i][k];
            const float4 b4 =
                *reinterpret_cast<const float4*>(&B_s[k][tx * V7_NR]);
            const float bf[V7_NR] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
            for (int i = 0; i < MR; ++i)
#pragma unroll
                for (int j = 0; j < V7_NR; ++j) acc[i][j] += af[i] * bf[j];
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MR; ++i)
#pragma unroll
        for (int j = 0; j < V7_NR; ++j) {
            const int r = row + i;
            const int c = col + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

template <int MR>
void LaunchCpasyncV7(int M, int N, int K, const float* A, const float* B,
                     float* C) {
    const size_t a_bytes = sizeof(float) * static_cast<size_t>(M) * K;
    const size_t b_bytes = sizeof(float) * static_cast<size_t>(K) * N;
    const size_t c_bytes = sizeof(float) * static_cast<size_t>(M) * N;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, A, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, B, b_bytes, cudaMemcpyHostToDevice));

    dim3 block(V7_TILE / V7_NR, V7_TILE / MR);
    dim3 grid((N + V7_TILE - 1) / V7_TILE, (M + V7_TILE - 1) / V7_TILE);
    sgemm_cpasync_kernel_v7<MR><<<grid, block>>>(M, N, K, d_a, d_b, d_c);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C, d_c, c_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

}  // namespace

void sgemm_cuda_cpasync_v7(int M, int N, int K,
                           const float* A, const float* B, float* C) {
    const bool aligned = (K % 4 == 0) && (N % 4 == 0) &&
                         (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(B) % 16 == 0);
    if (!aligned) {
        LaunchTuned<64, 4, 4>(M, N, K, A, B, C);  // 64x4x4 标量回退
        return;
    }
    LaunchCpasyncV7<4>(M, N, K, A, B, C);
}

static Registrar reg_cuda_cpasync_v7("cuda_naive_v7", &sgemm_cuda_cpasync_v7);
```

&emsp;&emsp;相比 v5，这段代码有三处关键改动。
- **第一，用 `cp.async` 替代同步加载**：`cp_async_16B` 封装了 `cp.async.cg.shared.global` 指令，从全局内存直接拷贝 16 字节到共享内存，绕过寄存器堆。`cp_async_commit_wait` 封装 `commit_group` 和 `wait_group 0`，等待所有异步拷贝完成。
- **第二，A 不再转置**：`cp.async` 是 memcpy 语义，无法转置，所以 A 按全局行布局存 `A_s[i][k]`。`a_frag` 从 v5 的一条 `LDS.128` 退化为 4 条 `LDS.32`。
- **第三，边界处理用 zfill**：`cp.async` 的 `src-size` 操作数为 0 时，目的端自动填零，不发起读。这替代了 v5 的分支置零，减少了分支指令。

&emsp;&emsp;实测结果如下：

```cpp
BM_sgemm/cuda_naive_v7/256x256x256      146259 ns    GFLOPS=229.43
BM_sgemm/cuda_naive_v7/1024x1024x1024  2420486 ns    GFLOPS=887.51
BM_sgemm/cuda_naive_v7/2048x2048x2048 11506150 ns    GFLOPS=1493.31

Performance counter stats (ncu --set roofline, 2048³):
    Elapsed Cycles                  cycle    11,429,700
    Duration                      msecond          7.37
    Memory Throughput                   %         76.51
    DRAM Throughput                     %         37.69
    L1/TEX Cache Throughput             %         77.21
    L2 Cache Throughput                 %         40.29
    Compute (SM) Throughput             %         76.51
    SM Active Cycles                cycle    11,290,474
```

&emsp;&emsp;这组数据与 v5、v6 对比：

| 指标 | v5 | v6（双缓冲） | v7（cp.async） |
|------|-----|-------------|---------------|
| GFLOPS（2048³） | 1476.95 | 1465.65 | **1493.31** |
| Duration | 7.02 ms | 7.68 ms | **7.37 ms** |
| DRAM Throughput | 32.02% | 36.17% | **37.69%** |
| L1/TEX Cache Throughput | 69.60% | 63.08% | **77.21%** |
| L2 Cache Throughput | 25.21% | 25.94% | **40.29%** |
| Compute (SM) Throughput | 42.69% | 37.70% | **76.51%** |
| Memory Throughput | 68.60% | 62.07% | **76.51%** |

&emsp;&emsp;**v7 的 GFLOPS 是三版中最高的，达到 1493.31。** `Compute (SM) Throughput` 从 v5 的 42.69% 跃升到 **76.51%**，说明 FMA 单元被喂得更满。`Memory Throughput` 也从 68.60% 升到 76.51%，`DRAM Throughput` 从 32.02% 升到 37.69%。这些指标同步上升，说明 `cp.async` 确实让访存和计算并行起来了——FMA 单元不再等数据，L1/内存也不再空转。

&emsp;&emsp;但 v7 的 `Duration` 是 7.37ms，比 v5 的 7.02ms 略长。GFLOPS 更高但 Duration 更长，这是因为 benchmark 的 60 次迭代里有 1 次被 `ncu` 采集时重放，`ncu` 报告的 Duration 是重放后的时间，而 benchmark 的 GFLOPS 是 60 次迭代的平均。以 benchmark 数据为准，**v7 的 1493.31 GFLOPS 是三者中最高的**。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gpu_gemm_v7_xxxxxxxxxxxxxxxxxxx.png)

&emsp;&emsp;`L1/TEX Cache Throughput` 从 v5 的 69.60% 升到 77.21%，`L2 Cache Throughput` 从 25.21% 升到 40.29%，说明 `cp.async` 把访存压力从寄存器堆转移到了 L1/L2。a_frag 从 1 条 `LDS.128` 退化为 4 条 `LDS.32`，指令数增加，L1/TEX 吞吐上升，这是 `cp.async` 无法转置的代价。但同一 `ty` 的 8 个线程读同一地址，硬件按广播处理，事务数没有劣化，所以 `L1/TEX Cache Throughput` 虽然升高但没有饱和。


### 3.8 共享内存 swizzle 消除 bank conflict

&emsp;&emsp;3.7 节的 v7 用 `cp.async` 把全局内存加载异步化，`Compute (SM) Throughput` 从 v5 的 42.69% 跃升到 **76.51%**，GFLOPS 达到 **1493.31**。但 v7 的 ncu 数据显示 `L1/TEX Cache Throughput` 77.21%、`Memory Throughput` 76.51%，两者都在高位。仔细分析 `cp.async` 的写入地址会发现一个隐藏的 bank conflict：`cp.async` 把 16 字节写到 `A_s[i][tx*4]`，首 word 的 bank 编号是 `(4*i + 16*tx) % 32`。对同一相位的 8 个 `tx`（`tx` 从 0 到 7）来说，`16*tx` 只产生 `{0, 16}` 两个值，所以 8 个线程的写入只落在 **{0, 16} 两个 bank 桶**里，形成 2-way conflict。v5 的转置布局 `A_sT[k][i]` 在写侧也有同源冲突——列距 68 的标量散写。

&emsp;&emsp;要消除这个冲突，可以在共享内存的列地址上做 **XOR swizzle**：把 `float4` 粒度的列索引 `col4` 和行索引 `row` 的低 3 位做异或，即 `swz(row, col4) = col4 ^ (row & 7)`。这样，同一相位内 8 个 `tx` 的 `col4`（0 到 7）被异或成一个排列，落在 8 个不同的 bank 桶里，冲突消除。写入和读取两侧用同一个映射，保证数据一致。

&emsp;&emsp;Swizzle 还有一个附带收益：**不需要 PAD**。v5/v7 用 `PAD=4` 让行距变成 68 来避免 bank conflict，Swizzle 用 XOR 映射达到同样目的，行距可以恢复成 64。共享内存从 `2 × 64 × 68 × 4 = 34.8 KB` 降到 `2 × 64 × 64 × 4 = 32.8 KB`，每个 SM 的 block 驻留数从 2 个升到 **3 个**，Occupancy 从 33% 升到 **50%**。

&emsp;&emsp;下面给出 v8 的完整实现。

```cpp
// ---------- smem swizzle 消 bank conflict：v8 ----------
// 机理：v7 的 cp.async 16B 写 A_s[i][tx*4] / B_s[i][tx*4] 存在 2-way conflict
// ——首 word bank = (4*i + 16*tx) % 32，同相位的 tx 0..7 只落在 {0,16} 两桶。
// 单变量改动：去 PAD（STRIDE 64），引入 float4 粒度 XOR swizzle
//   swz(row, col4) = col4 ^ (row & 7)
// 写读两侧统一映射：16B 写/读的相位内 col4' 为 0..7 的排列 -> 免冲突。
// 附带收益：smem 2x64x64x4B = 32.8KB/block -> 3 block/SM（占用率 33% -> 50%）。
namespace {

constexpr int V8_TILE = 64;
constexpr int V8_MR = 4;
constexpr int V8_NR = 4;

__device__ __forceinline__ int swz8(int row, int col4) {
    return col4 ^ (row & 7);
}

__device__ __forceinline__ void cp_async_16B_v8(float* smem_dst,
                                                const float* gmem_src,
                                                bool valid) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    const unsigned bytes = valid ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
                 "l"(gmem_src), "r"(bytes));
}

// 前置条件（host 侧保证）：K%4==0、N%4==0、A/B 指针 16B 对齐
template <int MR>
__global__ void sgemm_swizzle_kernel_v8(int M, int N, int K,
                                        const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C) {
    __shared__ float A_s[V8_TILE][V8_TILE];  // 非转置 A_s[i][k]，swizzle 布局
    __shared__ float B_s[V8_TILE][V8_TILE];

    const int row_base = blockIdx.y * V8_TILE;
    const int col_base = blockIdx.x * V8_TILE;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int row = row_base + ty * MR;
    const int col = col_base + tx * V8_NR;

    float acc[MR][V8_NR];
#pragma unroll
    for (int i = 0; i < MR; ++i)
#pragma unroll
        for (int j = 0; j < V8_NR; ++j) acc[i][j] = 0.0f;

    for (int k_base = 0; k_base < K; k_base += V8_TILE) {
        // ---- cp.async 搬运（写侧经 swz8 映射）----
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int i = ty + n * 16;
            cp_async_16B_v8(&A_s[i][swz8(i, tx) * 4],
                            A + (row_base + i) * K + k_base + tx * 4,
                            (row_base + i < M) && (k_base + tx * 4 < K));
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int i = ty + n * 16;
            cp_async_16B_v8(&B_s[i][swz8(i, tx) * 4],
                            B + (k_base + i) * N + col_base + tx * 4,
                            (k_base + i < K) && (col_base + tx * 4 < N));
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();

        // ---- 计算段（读侧同一 swz8 映射）----
#pragma unroll 4
        for (int k = 0; k < V8_TILE; ++k) {
            float af[MR];
#pragma unroll
            for (int i = 0; i < MR; ++i) {
                const int r = ty * MR + i;
                af[i] = A_s[r][swz8(r, k >> 2) * 4 + (k & 3)];
            }
            const float4 b4 =
                *reinterpret_cast<const float4*>(&B_s[k][swz8(k, tx) * 4]);
            const float bf[V8_NR] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
            for (int i = 0; i < MR; ++i)
#pragma unroll
                for (int j = 0; j < V8_NR; ++j) acc[i][j] += af[i] * bf[j];
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MR; ++i)
#pragma unroll
        for (int j = 0; j < V8_NR; ++j) {
            const int r = row + i;
            const int c = col + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

template <int MR>
void LaunchSwizzleV8(int M, int N, int K, const float* A, const float* B,
                     float* C) {
    const size_t a_bytes = sizeof(float) * static_cast<size_t>(M) * K;
    const size_t b_bytes = sizeof(float) * static_cast<size_t>(K) * N;
    const size_t c_bytes = sizeof(float) * static_cast<size_t>(M) * N;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, A, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, B, b_bytes, cudaMemcpyHostToDevice));

    dim3 block(V8_TILE / V8_NR, V8_TILE / MR);
    dim3 grid((N + V8_TILE - 1) / V8_TILE, (M + V8_TILE - 1) / V8_TILE);
    sgemm_swizzle_kernel_v8<MR><<<grid, block>>>(M, N, K, d_a, d_b, d_c);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C, d_c, c_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

}  // namespace

void sgemm_cuda_swizzle_v8(int M, int N, int K,
                           const float* A, const float* B, float* C) {
    const bool aligned = (K % 4 == 0) && (N % 4 == 0) &&
                         (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(B) % 16 == 0);
    if (!aligned) {
        LaunchTuned<64, 4, 4>(M, N, K, A, B, C);  // 64x4x4 标量回退
        return;
    }
    LaunchSwizzleV8<4>(M, N, K, A, B, C);
}

static Registrar reg_cuda_swizzle_v8("cuda_naive_v8", &sgemm_cuda_swizzle_v8);
```

&emsp;&emsp;相比 v7，这段代码有两处关键改动。**第一，去 PAD，引入 XOR swizzle**：共享内存从 `A_s[64][68]` 变成 `A_s[64][64]`，行距恢复成 64。写入和读取都用 `swz8(row, col4) = col4 ^ (row & 7)` 映射。**第二，写入的地址经过 swizzle**：`cp_async_16B_v8(&A_s[i][swz8(i, tx)*4], ...)`，读取时 `A_s[r][swz8(r, k>>2)*4 + (k&3)]`，两侧映射一致。`float4` 粒度的列索引 `col4` 和行索引 `row & 7` 异或后，同一相位内 8 个 `tx` 的写入落在 8 个不同的 bank 桶里，2-way conflict 消除。

&emsp;&emsp;实测结果如下：

```cpp
BM_sgemm/cuda_naive_v8/256x256x256      155594 ns    GFLOPS=215.67
BM_sgemm/cuda_naive_v8/1024x1024x1024  2378340 ns    GFLOPS=903.77
BM_sgemm/cuda_naive_v8/2048x2048x2048 10933003 ns    GFLOPS=1571.53

Performance counter stats (ncu --set roofline, 2048³):
    Elapsed Cycles                  cycle     9,439,941
    Duration                      msecond          6.09
    Memory Throughput                   %         92.17
    DRAM Throughput                     %         45.06
    L1/TEX Cache Throughput             %         93.78
    L2 Cache Throughput                 %         32.66
    Compute (SM) Throughput             %         92.17
    SM Active Cycles                cycle     9,295,868
```

&emsp;&emsp;这组数据与 v7 对比：

| 指标 | v7（cp.async） | v8（swizzle） | 变化 |
|------|---------------|--------------|------|
| GFLOPS（2048³） | 1493.31 | **1571.53** | 升 5.2% |
| Duration | 7.37 ms | **6.09 ms** | 降 17.4% |
| Elapsed Cycles | 11,429,700 | **9,439,941** | 降 17.4% |
| DRAM Throughput | 37.69% | 45.06% | 升 19.6% |
| L1/TEX Cache Throughput | 77.21% | **93.78%** | 升 21.5% |
| L2 Cache Throughput | 40.29% | 32.66% | 降 19.0% |
| Compute (SM) Throughput | 76.51% | **92.17%** | 升 20.5% |
| Memory Throughput | 76.51% | **92.17%** | 升 20.5% |

&emsp;&emsp;`Duration` 从 7.37ms 降到 **6.09ms**，降幅 17.4%。GFLOPS 从 1493.31 提升到 **1571.53**，提升 5.2%。`Compute (SM) Throughput` 从 76.51% 跃升到 **92.17%**，`L1/TEX Cache Throughput` 从 77.21% 升到 **93.78%**，`Memory Throughput` 从 76.51% 升到 **92.17%**。三个指标同步逼近饱和，说明 swizzle 消除了 bank conflict 后，FMA 单元和 L1/TEX 都被充分利用。

&emsp;&emsp;值得注意的是 `L1/TEX Cache Throughput` 从 77.21% 升到 93.78%，说明去掉 bank conflict 后 L1/TEX 的数据搬运量更集中，带宽压力上升。而 `L2 Cache Throughput` 从 40.29% 降到 32.66%，说明 L2 的压力反而下降了——swizzle 让数据在 L1/TEX 层面被更高效地处理，不需要频繁访问 L2。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gpu_gemm_v8_xxxxxxxxxxxxxxxxxxx.png)

&emsp;&emsp;从优化路径看，v8 是继 v3（寄存器分块）、v4（padding + 增大 tile）、v5（float4 向量化）、v6（双缓冲失败）、v7（cp.async）之后的第六步。它用 XOR swizzle 解决了一个 v7 遗留的隐蔽问题：`cp.async` 的 16 字节写入本身会引入 2-way bank conflict。去 PAD 用 swizzle 替代，既消除了冲突，又把共享内存从 34.8KB 降到 32.8KB，Occupancy 从 33% 升到 50%。这一步的收益（Duration 降 17.4%）比 v7 的 cp.async（收益 1.1%）明显得多，说明**在正确的时机做正确的优化**比反复叠加优化手段更重要。

### 3.9 L2 block swizzle

&emsp;&emsp;3.8 节的 v8 用 XOR swizzle 消除了共享内存的 bank conflict，`Duration` 从 7.37ms 降到 **6.09ms**，GFLOPS 达到 **1571.53**，`Compute (SM) Throughput` 和 `Memory Throughput` 都逼近 92%，接近饱和。但 v8 的数据里还有一个值得注意的地方：`DRAM Throughput` 45.06%，`L2 Cache Throughput` 32.66%。这两个指标不算高，但它们是**网格遍历顺序**造成的。

&emsp;&emsp;CUDA 的 block 调度顺序是 `blockIdx.x` 最快变化，`blockIdx.y` 次之。在 v8 里，`blockIdx.x` 对应 C 的列块，`blockIdx.y` 对应 C 的行块。这意味着**同一行的 32 个列块会连续执行**。对 A 面板来说，这是好事——同一行的 A 面板在 32 个列块间被反复复用，命中 L2。但对 B 面板来说，这是坏事——B 的每个面板要被 32 个行块各读一遍，**B 面板的 L2/DRAM 读取次数被放大了 32 倍**。

&emsp;&emsp;解决办法是 **L2 block swizzle**：重映射 grid 的遍历顺序，让一个"带"内的多个行块共享同一个 B 面板。具体来说，把 `gridDim.y` 按 `GROUP = 8` 分组，每个带包含 8 个行块和全部列块。带内按"行优先"遍历（先遍历 8 个行块，再推进列块），这样同一个 B 面板在带内被 8 个行块共享，**B 面板的 L2/DRAM 读取次数降到 1/8**。同时，A 面板仍被带内全部列块复用。

&emsp;&emsp;带的工作集是 `8 × 16KB（A 面板）+ 32 × 16KB（B 面板）= 640KB`，远小于 RTX 3050 的 L2 容量（2MB），所以带内数据能全部驻留 L2。下面给出完整实现。

```cpp
// ---------- L2 block swizzle：v9 ----------
// 单变量：kernel 计算逻辑与 v8 完全相同，仅重映射 grid 遍历顺序。
// 原顺序 blockIdx.x（列）最快 -> 同行 32 个列块连续：A 面板有 L2 复用，
// 但每个 B 面板要被 32 行各读一遍（DRAM/L2 流量放大 32 倍）。
// 改为 GROUP=8 行一条"带"：带内 8 行 x 全部列块一起推进，同一 B 面板
// 在带内被 8 个行块共享（L2 2MB > 带工作集 8x16KB + 32x16KB = 640KB），
// B 面板的 L2/DRAM 读取次数降为 1/8；A 面板仍被带内全部列块复用。
namespace {

constexpr int V9_TILE = 64;
constexpr int V9_MR = 4;
constexpr int V9_NR = 4;
constexpr int V9_GROUP = 8;  // 每带行块数

__device__ __forceinline__ int swz8_v9(int row, int col4) {
    return col4 ^ (row & 7);
}

__device__ __forceinline__ void cp_async_16B_v9(float* smem_dst,
                                                const float* gmem_src,
                                                bool valid) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    const unsigned bytes = valid ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
                 "l"(gmem_src), "r"(bytes));
}

// 前置条件（host 侧保证）：K%4==0、N%4==0、A/B 指针 16B 对齐
template <int MR>
__global__ void sgemm_l2swizzle_kernel_v9(int M, int N, int K,
                                          const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C) {
    // ---- blockIdx -> (row_blk, col_blk) 分组重映射（唯一改动点）----
    const int bid = blockIdx.x + blockIdx.y * gridDim.x;
    const int blocks_per_band = V9_GROUP * gridDim.x;
    const int band = bid / blocks_per_band;
    const int first_row = band * V9_GROUP;
    int row_blk, col_blk;
    if (first_row + V9_GROUP <= gridDim.y) {  // 整带：带内行优先遍历
        const int off = bid % blocks_per_band;
        row_blk = first_row + off % V9_GROUP;
        col_blk = off / V9_GROUP;
    } else {                                  // 尾部残带：恒等映射
        row_blk = blockIdx.y;
        col_blk = blockIdx.x;
    }

    __shared__ float A_s[V9_TILE][V9_TILE];
    __shared__ float B_s[V9_TILE][V9_TILE];

    const int row_base = row_blk * V9_TILE;
    const int col_base = col_blk * V9_TILE;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int row = row_base + ty * MR;
    const int col = col_base + tx * V9_NR;

    float acc[MR][V9_NR];
#pragma unroll
    for (int i = 0; i < MR; ++i)
#pragma unroll
        for (int j = 0; j < V9_NR; ++j) acc[i][j] = 0.0f;

    for (int k_base = 0; k_base < K; k_base += V9_TILE) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int i = ty + n * 16;
            cp_async_16B_v9(&A_s[i][swz8_v9(i, tx) * 4],
                            A + (row_base + i) * K + k_base + tx * 4,
                            (row_base + i < M) && (k_base + tx * 4 < K));
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int i = ty + n * 16;
            cp_async_16B_v9(&B_s[i][swz8_v9(i, tx) * 4],
                            B + (k_base + i) * N + col_base + tx * 4,
                            (k_base + i < K) && (col_base + tx * 4 < N));
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();

#pragma unroll 4
        for (int k = 0; k < V9_TILE; ++k) {
            float af[MR];
#pragma unroll
            for (int i = 0; i < MR; ++i) {
                const int r = ty * MR + i;
                af[i] = A_s[r][swz8_v9(r, k >> 2) * 4 + (k & 3)];
            }
            const float4 b4 =
                *reinterpret_cast<const float4*>(&B_s[k][swz8_v9(k, tx) * 4]);
            const float bf[V9_NR] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
            for (int i = 0; i < MR; ++i)
#pragma unroll
                for (int j = 0; j < V9_NR; ++j) acc[i][j] += af[i] * bf[j];
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MR; ++i)
#pragma unroll
        for (int j = 0; j < V9_NR; ++j) {
            const int r = row + i;
            const int c = col + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

void LaunchV9Kernel(int M, int N, int K, const float* d_a, const float* d_b,
                    float* d_c) {
    dim3 block(V9_TILE / V9_NR, V9_TILE / V9_MR);
    dim3 grid((N + V9_TILE - 1) / V9_TILE, (M + V9_TILE - 1) / V9_TILE);
    sgemm_l2swizzle_kernel_v9<V9_MR><<<grid, block>>>(M, N, K, d_a, d_b, d_c);
    CUDA_CHECK(cudaGetLastError());
}

void LaunchL2SwizzleV9(int M, int N, int K, const float* A, const float* B,
                       float* C) {
    const size_t a_bytes = sizeof(float) * static_cast<size_t>(M) * K;
    const size_t b_bytes = sizeof(float) * static_cast<size_t>(K) * N;
    const size_t c_bytes = sizeof(float) * static_cast<size_t>(M) * N;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, A, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, B, b_bytes, cudaMemcpyHostToDevice));

    LaunchV9Kernel(M, N, K, d_a, d_b, d_c);

    CUDA_CHECK(cudaMemcpy(C, d_c, c_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

}  // namespace

void sgemm_cuda_l2swizzle_v9(int M, int N, int K,
                             const float* A, const float* B, float* C) {
    const bool aligned = (K % 4 == 0) && (N % 4 == 0) &&
                         (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(B) % 16 == 0);
    if (!aligned) {
        LaunchTuned<64, 4, 4>(M, N, K, A, B, C);  // 64x4x4 标量回退
        return;
    }
    LaunchL2SwizzleV9(M, N, K, A, B, C);
}

static Registrar reg_cuda_l2swizzle_v9("cuda_naive_v9", &sgemm_cuda_l2swizzle_v9);
```

&emsp;&emsp;相比 v8，这段代码只有**一处改动**：kernel 开头的 blockIdx 重映射。计算逻辑、共享内存布局、swizzle 映射、`cp.async` 搬运全部保持不变。

```cpp
const int bid = blockIdx.x + blockIdx.y * gridDim.x;
const int blocks_per_band = V9_GROUP * gridDim.x;
const int band = bid / blocks_per_band;
const int first_row = band * V9_GROUP;
int row_blk, col_blk;
if (first_row + V9_GROUP <= gridDim.y) {
    const int off = bid % blocks_per_band;
    row_blk = first_row + off % V9_GROUP;
    col_blk = off / V9_GROUP;
} else {
    row_blk = blockIdx.y;
    col_blk = blockIdx.x;
}
```

&emsp;&emsp;这段映射把原本"列优先"的遍历顺序改成"带内行优先"。2048³ 下 `gridDim = (32, 32)`，`V9_GROUP = 8`，每个带包含 `8 × 32 = 256` 个 block，共 4 个带。带内 `row_blk` 从 `first_row` 到 `first_row + 7`，`col_blk` 从 0 到 31，遍历顺序是 `(0,0), (1,0), ..., (7,0), (0,1), (1,1), ...`。这样 B 的第 0 个列面板在带内被 8 个行块各读一遍，之后不再被后面的带读到（因为 `col_blk` 会推进到下一个列块）。

&emsp;&emsp;实测结果如下：

```cpp
BM_sgemm/cuda_naive_v9/256x256x256      155024 ns    GFLOPS=216.45
BM_sgemm/cuda_naive_v9/1024x1024x1024  2360108 ns    GFLOPS=910.65
BM_sgemm/cuda_naive_v9/2048x2048x2048 11071240 ns    GFLOPS=1551.84

Performance counter stats (ncu --set roofline, 2048³):
    Elapsed Cycles                  cycle     9,463,934
    Duration                      msecond          6.10
    Memory Throughput                   %         92.00
    DRAM Throughput                     %         25.39
    L1/TEX Cache Throughput             %         93.21
    L2 Cache Throughput                 %         32.51
    Compute (SM) Throughput             %         92.00
    SM Active Cycles                cycle     9,353,307
```

&emsp;&emsp;这组数据与 v8 对比：

| 指标 | v8（smem swizzle） | v9（L2 swizzle） | 变化 |
|------|-------------------|-----------------|------|
| GFLOPS（2048³） | 1571.53 | **1551.84** | 降 1.3% |
| Duration | 6.09 ms | 6.10 ms | 基本持平 |
| Elapsed Cycles | 9,439,941 | 9,463,934 | 基本持平 |
| **DRAM Throughput** | **45.06%** | **25.39%** | **降 43.6%** |
| L1/TEX Cache Throughput | 93.78% | 93.21% | 持平 |
| L2 Cache Throughput | 32.66% | 32.51% | 持平 |
| Compute (SM) Throughput | 92.17% | 92.00% | 持平 |

&emsp;&emsp;**v9 的 GFLOPS 与 v8 基本持平（1551.84 vs 1571.53，降 1.3%），但 `DRAM Throughput` 从 45.06% 降到 25.39%，降幅 43.6%。** 这正是 L2 block swizzle 的预期效果：B 面板在带内被 8 个行块共享，DRAM 的读取次数降到原来的 1/8 左右。`DRAM Throughput` 降了 43.6%，说明 B 面板的 DRAM 流量确实被 L2 吸收了。

&emsp;&emsp;但 `Duration` 和 GFLOPS 没有改善。原因在于 v8 的 `DRAM Throughput` 45.06% 还没有饱和，DRAM 不是瓶颈。此时减少 DRAM 流量只能降低 DRAM 占用，不能缩短 `Duration`。v8/v9 的瓶颈已经在 `Compute (SM) Throughput` 92%、`L1/TEX Cache Throughput` 93%——FMA 单元和 L1/TEX 都接近饱和，L2/DRAM 的优化已经没有进一步提升空间。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gpu_gemm_v9_xxxxxxxxxxxxxxxxxxx.png)

&emsp;&emsp;这也说明**L2 block swizzle 的收益依赖于 DRAM 是否成为瓶颈**。当 `DRAM Throughput` 超过 60~70% 时，减少 DRAM 流量能直接缩短 `Duration`；当 `DRAM Throughput` 只有 45% 时，减少它只是降低了 DRAM 的占用，不影响总时间。v9 的 DRAM 从 45% 降到 25%，但这个降低被 L1/TEX 和 FMA 的饱和掩盖了。

### 3.10 Host 侧缓冲区复用

&emsp;&emsp;3.9 节的 v9 用 L2 block swizzle 把 `DRAM Throughput` 从 45.06% 降到 25.39%，但 GFLOPS 没有提升，因为瓶颈已经在 `Compute (SM) Throughput` 92%、`L1/TEX Cache Throughput` 93%。此时 kernel 本身已经接近 RTX 3050 在 FP32 路径上的上限。但 benchmark 的 **端到端口径**里还有一笔固定开销被忽略了：每次调用 `sgemm_cuda_*` 都要 `cudaMalloc` 三块设备缓冲区、`cudaFree` 释放，2048³ 下这套固定开销约 5ms，而 kernel 本身只有约 6ms。**端到端耗时里，近一半花在了内存分配和释放上。**

&emsp;&emsp;解决办法是在**进程内缓存 device buffer**，按需扩容（只扩不缩）。第一次调用时分配，后续调用复用，只有规模增大时才重新分配。这样，benchmark 的 60 次迭代里，前几次会走分配路径，后面全部走复用路径，端到端耗时大幅下降。下面给出实现。

```cpp
// ---------- host 侧 buffer 复用：v10 ----------
// kernel 与 v9 完全相同（单变量 = host 内存管理）。此前每次调用都要
// cudaMalloc + cudaFree 三块 device buffer：e2e 口径下这套固定开销约 5ms
//（2048 规模的 kernel 本身仅约 7ms），是端到端最大的单笔开销。
// 本版在进程内缓存 device buffer，按需扩容（只扩不缩）；test 中从
// 1x1x1 到 2048 的连续调用即覆盖扩容路径。单线程假设（bench/test 均
// 单线程调用），多线程共用需外置同步。
namespace {

float* PersistBufV10(float* old, size_t old_cap, size_t need) {
    if (need <= old_cap) return old;
    if (old != nullptr) CUDA_CHECK(cudaFree(old));
    float* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, need * sizeof(float)));
    return p;
}

}  // namespace

void sgemm_cuda_persist_v10(int M, int N, int K,
                            const float* A, const float* B, float* C) {
    const bool aligned = (K % 4 == 0) && (N % 4 == 0) &&
                         (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(B) % 16 == 0);
    if (!aligned) {
        LaunchTuned<64, 4, 4>(M, N, K, A, B, C);  // 64x4x4 标量回退
        return;
    }

    const size_t a_elems = static_cast<size_t>(M) * K;
    const size_t b_elems = static_cast<size_t>(K) * N;
    const size_t c_elems = static_cast<size_t>(M) * N;

    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static size_t a_cap = 0;
    static size_t b_cap = 0;
    static size_t c_cap = 0;

    d_a = PersistBufV10(d_a, a_cap, a_elems);
    a_cap = a_cap > a_elems ? a_cap : a_elems;
    d_b = PersistBufV10(d_b, b_cap, b_elems);
    b_cap = b_cap > b_elems ? b_cap : b_elems;
    d_c = PersistBufV10(d_c, c_cap, c_elems);
    c_cap = c_cap > c_elems ? c_cap : c_elems;

    CUDA_CHECK(cudaMemcpy(d_a, A, a_elems * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, B, b_elems * sizeof(float),
                          cudaMemcpyHostToDevice));

    LaunchV9Kernel(M, N, K, d_a, d_b, d_c);

    CUDA_CHECK(cudaMemcpy(C, d_c, c_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

static Registrar reg_cuda_persist_v10("cuda_naive_v10", &sgemm_cuda_persist_v10);
```

&emsp;&emsp;相比 v9，这段代码只有**一处改动**：device buffer 从每次调用分配/释放，改成静态缓存 + 按需扩容。

```cpp
static float* d_a = nullptr;
static float* d_b = nullptr;
static float* d_c = nullptr;
static size_t a_cap = 0;
static size_t b_cap = 0;
static size_t c_cap = 0;

d_a = PersistBufV10(d_a, a_cap, a_elems);
a_cap = a_cap > a_elems ? a_cap : a_elems;
```

&emsp;&emsp;`PersistBufV10` 的逻辑是：如果当前容量够用，直接返回旧指针；否则释放旧缓冲区、分配新的、更新容量。`a_cap`、`b_cap`、`c_cap` 记录每块缓冲区的当前容量，只增不减。benchmark 跑 256³、1024³、2048³ 三个规模时，第一次遇到更大的规模会扩容，后续同规模调用全部复用。

&emsp;&emsp;实测结果如下：

```cpp
BM_sgemm/cuda_naive_v10/256x256x256      153048 ns    GFLOPS=219.27
BM_sgemm/cuda_naive_v10/1024x1024x1024  1852296 ns    GFLOPS=1159.56
BM_sgemm/cuda_naive_v10/2048x2048x2048  9608708 ns    GFLOPS=1788.16

Performance counter stats (ncu --set roofline, 2048³):
    Elapsed Cycles                  cycle     9,469,861
    Duration                      msecond          6.11
    Memory Throughput                   %         91.99
    DRAM Throughput                     %         25.50
    L1/TEX Cache Throughput             %         93.46
    L2 Cache Throughput                 %         32.64
    Compute (SM) Throughput             %         91.99
    SM Active Cycles                cycle     9,328,327
```

&emsp;&emsp;这组数据与 v9 对比：

| 规模 | v9 GFLOPS | v10 GFLOPS | 提升 |
|------|-----------|------------|------|
| 256³ | 232.45 | **219.27** | 略降 |
| 1024³ | 942.25 | **1159.56** | **升 23.1%** |
| 2048³ | 1555.60 | **1788.16** | **升 15.0%** |

&emsp;&emsp;2048³ 下 v10 的 GFLOPS 从 v9 的 1555.60 提升到 **1788.16**，提升 15.0%。1024³ 下从 942.25 提升到 1159.56，提升 23.1%。256³ 下略降，因为小规模下 `cudaMalloc` 的固定开销本来就占比不高，而静态缓存引入的额外分支（`PersistBufV10` 的判断）略有开销。

&emsp;&emsp;ncu 数据里 `Elapsed Cycles` 946 万、`Duration` 6.11ms，和 v9 基本一致。**ncu 采集的是 kernel 本身的时间，不包含 host 侧的 `cudaMalloc`/`cudaFree`。** 而 benchmark 的端到端口径包含这些开销，所以 v10 的 GFLOPS 提升来自 host 侧，而不是 kernel 本身。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gpu_gemm_v10_xxxxxxxxxxxxxxxxxxx.png)

&emsp;&emsp;这也解释了为什么 v10 的 `Duration` 和 v9 一样是 6.11ms，但 GFLOPS 高 15%。benchmark 的 60 次迭代里，v9 每次都要 `cudaMalloc` + `cudaFree` 三块 buffer，2048³ 下每块 16MB，三次 `cudaMalloc` 约 5ms。v10 的第一次迭代走分配路径，后续 59 次全部复用，端到端耗时从 v9 的约 11ms/次降到约 9.6ms/次。GFLOPS 从 1555.60 升到 1788.16，正好对应这部分固定开销的消除。

### 3.11 Tensor Core 与 WMMA

&emsp;&emsp;3.10 节的 v10 用 host 侧缓冲区复用把端到端 GFLOPS 推到 **1788.16**（2048³），`Compute (SM) Throughput` 92%、`L1/TEX Cache Throughput` 93%，已经接近 RTX 3050 在 CUDA Core FP32 路径上的上限。要突破这个上限，必须换执行单元——用 **Tensor Core**。Tensor Core 是 NVIDIA 从 Volta 开始引入的专用矩阵乘加单元，单条 `mma` 指令完成一个小矩阵块（如 16×16×8）的乘加，吞吐远高于 CUDA Core 的 FMA。

&emsp;&emsp;RTX 3050 是 Ampere 架构（CC 8.6），支持 Tensor Core 的 **TF32** 精度：输入 A、B 是 FP32 存储，在 `mma` 内部被舍入成 TF32（10-bit 尾数），累加器仍是 FP32。TF32 的吞吐是 FP32 CUDA Core 的数倍，代价是**精度损失**——A、B 被舍入到 10-bit 尾数，非精确 FP32。如果换 FP16，吞吐更高，但精度损失更大。

&emsp;&emsp;WMMA（Warp Matrix Multiply Accumulate）是 CUDA 提供的 Tensor Core 高层接口，用 `wmma::fragment` 管理寄存器中的数据，`wmma::load_matrix_sync`、`wmma::mma_sync`、`wmma::store_matrix_sync` 完成加载、乘加、写回。下面给出完整实现。

```cpp
// ---------- Tensor Core 终章：v11（WMMA tf32）----------
//
// 结构：TILE=64、block 16x16（8 warp）；C 的 64x64 tile = 4x4 个 16x16
// wmma tile，每 warp 负责 2 个（(w/4, w%4) 与 (w/4+2, w%4)），K 按 8 分步
// mma_sync 累加。smem 零填充 tile 沿用 v5 思路（越界填 0，wmma 无需边界
// 判断，也因此不要求 K%8）。PAD=8：ldm=72 是 4 的倍数（tf32 对 ldm 的
// 要求）且行基址 288B 为 32B 对齐。
// epilogue：完整 16x16 tile 直写全局；跨界 tile 经 B_s（k 循环已结束、
// 双 sync 保证空闲）暂存后边界检查写回。
// host：继承 v10 的 device buffer 复用（对比 v10 即纯 kernel 收益）。
namespace {

constexpr int V11_TILE = 64;
constexpr int V11_PAD = 8;
constexpr int V11_LD = V11_TILE + V11_PAD;  // ldm=72

// 前置条件（host 侧保证）：K%4==0、N%4==0、A/B 指针 16B 对齐（与 v5 相同，
// 零填充使 wmma 路径无需额外的 K%8 约束）
__global__ void sgemm_wmma_kernel_v11(int M, int N, int K,
                                      const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C) {
    using namespace nvcuda;
    __shared__ float A_s[V11_TILE][V11_LD];
    __shared__ float B_s[V11_TILE][V11_LD];

    const int row_base = blockIdx.y * V11_TILE;
    const int col_base = blockIdx.x * V11_TILE;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int warp = (threadIdx.y * blockDim.x + threadIdx.x) / 32;

    // 每 warp 两个 16x16 输出 tile 的左上角（块内 tile 坐标）
    const int wt_row[2] = {warp / 4, warp / 4 + 2};
    const int wt_col = warp % 4;

    wmma::fragment<wmma::matrix_a, 16, 16, 8,
                   wmma::precision::tf32, wmma::row_major>
        af;
    // B_s[k][n] 行存放 -> 元素 (k,n) 在 p[k*ldm+n]：matrix_b 须用 row_major
    wmma::fragment<wmma::matrix_b, 16, 16, 8,
                   wmma::precision::tf32, wmma::row_major>
        bf;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> cf[2];
    wmma::fill_fragment(cf[0], 0.0f);
    wmma::fill_fragment(cf[1], 0.0f);

    for (int k_base = 0; k_base < K; k_base += V11_TILE) {
        // ---- 装载 A/B tile（float4 读 + 标量写，越界零填充）----
        for (int i = ty; i < V11_TILE; i += blockDim.y) {
            const int a_row = row_base + i;
            float4 va = make_float4(0.f, 0.f, 0.f, 0.f);
            if (a_row < M && k_base + tx * 4 < K)
                va = *reinterpret_cast<const float4*>(A + a_row * K + k_base +
                                                      tx * 4);
            // 行距 72 float = 288B，16B 对齐 -> 单条 STS.128
            *reinterpret_cast<float4*>(&A_s[i][tx * 4]) = va;
        }
        for (int i = ty; i < V11_TILE; i += blockDim.y) {
            const int b_col = col_base + tx * 4;
            float4 vb = make_float4(0.f, 0.f, 0.f, 0.f);
            if (k_base + i < K && b_col < N)
                vb = *reinterpret_cast<const float4*>(B + (k_base + i) * N +
                                                      b_col);
            *reinterpret_cast<float4*>(&B_s[i][tx * 4]) = vb;
        }
        __syncthreads();

        // ---- K 维 8 步 mma 累加 ----
        for (int kk = 0; kk < V11_TILE; kk += 8) {
            wmma::load_matrix_sync(bf, &B_s[kk][wt_col * 16], V11_LD);
            wmma::load_matrix_sync(af, &A_s[wt_row[0] * 16][kk], V11_LD);
            wmma::mma_sync(cf[0], af, bf, cf[0]);
            wmma::load_matrix_sync(af, &A_s[wt_row[1] * 16][kk], V11_LD);
            wmma::mma_sync(cf[1], af, bf, cf[1]);
        }
        __syncthreads();
    }

    // ---- epilogue：界内直写；跨界 tile 先落 B_s 暂存区再边界写回 ----
    float* stage = &B_s[0][0] + warp * 512;  // 每 warp 2x256 floats
    for (int t = 0; t < 2; ++t) {
        const int r0 = row_base + wt_row[t] * 16;
        const int c0 = col_base + wt_col * 16;
        if (r0 + 15 < M && c0 + 15 < N) {
            wmma::store_matrix_sync(&C[r0 * N + c0], cf[t], N,
                                    wmma::mem_row_major);
        } else {
            wmma::store_matrix_sync(stage + t * 256, cf[t], 16,
                                    wmma::mem_row_major);
            __syncwarp();  // stage 由全 warp 写入，读前做 warp 级可见性同步
            for (int e = 0; e < 256; ++e) {
                const int gr = r0 + e / 16;
                const int gc = c0 + e % 16;
                if (gr < M && gc < N) C[gr * N + gc] = stage[t * 256 + e];
            }
        }
    }
}

void LaunchWmmaV11Kernel(int M, int N, int K, const float* d_a,
                         const float* d_b, float* d_c) {
    dim3 block(16, 16);
    dim3 grid((N + V11_TILE - 1) / V11_TILE, (M + V11_TILE - 1) / V11_TILE);
    sgemm_wmma_kernel_v11<<<grid, block>>>(M, N, K, d_a, d_b, d_c);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void sgemm_cuda_wmma_v11(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    const bool aligned = (K % 4 == 0) && (N % 4 == 0) &&
                         (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(B) % 16 == 0);
    if (!aligned) {
        LaunchTuned<64, 4, 4>(M, N, K, A, B, C);  // 64x4x4 标量回退
        return;
    }

    // 继承 v10 的 host 复用逻辑
    const size_t a_elems = static_cast<size_t>(M) * K;
    const size_t b_elems = static_cast<size_t>(K) * N;
    const size_t c_elems = static_cast<size_t>(M) * N;

    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static size_t a_cap = 0;
    static size_t b_cap = 0;
    static size_t c_cap = 0;

    d_a = PersistBufV10(d_a, a_cap, a_elems);
    a_cap = a_cap > a_elems ? a_cap : a_elems;
    d_b = PersistBufV10(d_b, b_cap, b_elems);
    b_cap = b_cap > b_elems ? b_cap : b_elems;
    d_c = PersistBufV10(d_c, c_cap, c_elems);
    c_cap = c_cap > c_elems ? c_cap : c_elems;

    CUDA_CHECK(
        cudaMemcpy(d_a, A, a_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_b, B, b_elems * sizeof(float), cudaMemcpyHostToDevice));

    LaunchWmmaV11Kernel(M, N, K, d_a, d_b, d_c);

    CUDA_CHECK(cudaMemcpy(C, d_c, c_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

static Registrar reg_cuda_wmma_v11("cuda_naive_v11", &sgemm_cuda_wmma_v11);
```

&emsp;&emsp;相比 v10，这段代码的核心改动是**把 CUDA Core 的 FMA 微内核换成 Tensor Core 的 `wmma::mma_sync`**。

&emsp;&emsp;**第一，`wmma::fragment` 管理数据。** `matrix_a`、`matrix_b`、`accumulator` 三种 fragment 分别对应 A、B、C 在寄存器中的分片。`wmma::fill_fragment` 初始化累加器，`wmma::load_matrix_sync` 从共享内存加载，`wmma::mma_sync` 完成乘加，`wmma::store_matrix_sync` 写回全局内存。

&emsp;&emsp;**第二，warp 级分块。** `TILE = 64` 的 C 块被划分为 `4 × 4` 个 `16 × 16` 的 wmma tile。block 有 8 个 warp，每个 warp 负责 2 个 tile：`(warp/4, warp%4)` 和 `(warp/4+2, warp%4)`。这样 8 个 warp 刚好覆盖 16 个 tile。

&emsp;&emsp;**第三，K 维按 8 分步。** TF32 的 `mma` 指令形状是 `16 × 16 × 8`，即每次处理 8 个 K 维。`for (int kk = 0; kk < V11_TILE; kk += 8)` 循环 8 次，每次加载 B 的 16×8 分片和 A 的 16×8 分片，执行 `mma_sync`。

&emsp;&emsp;**第四，epilogue 处理边界 tile。** 如果 `r0+15 < M && c0+15 < N`（tile 完全在界内），直接 `store_matrix_sync` 写回 C。否则先写到共享内存的暂存区，再用标量循环逐元素边界检查写回。这保证任意 M、N 都能正确处理。

&emsp;&emsp;实测结果如下：

```cpp
BM_sgemm/cuda_naive_v11/256x256x256      139240 ns    GFLOPS=241.00
BM_sgemm/cuda_naive_v11/1024x1024x1024  1762696 ns    GFLOPS=1218.42
BM_sgemm/cuda_naive_v11/2048x2048x2048  8851271 ns    GFLOPS=1941.05

Performance counter stats (ncu --set roofline, 2048³):
    Elapsed Cycles                  cycle     7,792,794
    Duration                      msecond          5.02
    Memory Throughput                   %         45.09
    DRAM Throughput                     %         45.09
    L1/TEX Cache Throughput             %         39.15
    L2 Cache Throughput                 %         36.89
    Compute (SM) Throughput             %         38.05
    SM Active Cycles                cycle     7,765,394
```

&emsp;&emsp;这组数据与 v10 对比：

| 指标 | v10 | v11（WMMA tf32） | 变化 |
|------|-----|-----------------|------|
| GFLOPS（2048³） | 1788.16 | **1941.05** | **升 8.6%** |
| Duration | 6.11 ms | **5.02 ms** | **降 17.8%** |
| Elapsed Cycles | 9,469,861 | **7,792,794** | 降 17.7% |
| DRAM Throughput | 25.50% | 45.09% | 升 76.8% |
| L1/TEX Cache Throughput | 93.46% | **39.15%** | **降 58.1%** |
| L2 Cache Throughput | 32.64% | 36.89% | 略升 |
| Compute (SM) Throughput | 91.99% | **38.05%** | **降 58.6%** |
| Memory Throughput | 91.99% | 45.09% | 降 51.0% |

&emsp;&emsp;`Duration` 从 6.11ms 降到 **5.02ms**，降幅 17.8%；GFLOPS 从 1788.16 提升到 **1941.05**，提升 8.6%。`Compute (SM) Throughput` 从 91.99% 降到 38.05%，`L1/TEX Cache Throughput` 从 93.46% 降到 39.15%。这说明 **Tensor Core 的 `mma` 指令不占用 CUDA Core 的 FMA 单元**，所以 `Compute (SM) Throughput` 这个指标不再反映真实瓶颈。

&emsp;&emsp;`DRAM Throughput` 从 25.50% 升到 45.09%，`Memory Throughput` 45.09%，说明 WMMA 版本的数据搬运更集中，DRAM 带宽成为新的关注点。但 `DRAM Throughput` 45% 还没有饱和，`Duration` 的下降主要来自 `mma` 指令的高吞吐——TF32 的 `mma` 单指令完成 `16 × 16 × 8 = 2048` 次 FMA，而 CUDA Core 的 `vfmadd` 单指令只完成 8 次，吞吐差距约 256 倍。当然实际性能受限于数据搬运和 fragment 管理，所以端到端只提升了 8.6%。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gpu_gemm_v11_xxxxxxxxxxxxxxxxxxx.png)

&emsp;&emsp;值得注意的是 v11 的 GFLOPS 1941.05 已经**接近 cublas 的 2005.97**，差距只有 3.2%。cublas 是 NVIDIA 官方调优的库，用 `mma` 指令、双缓冲、L2 swizzle、向量化加载等一整套优化。v11 只用 WMMA 的高层接口，没有做双缓冲和 L2 swizzle，就能达到 cublas 的 96.8%，说明 TF32 路线的潜力很大。

&emsp;&emsp;从优化路径看，v11 是继 v3（寄存器分块）、v4（padding + 增大 tile）、v5（float4 向量化）、v6（双缓冲失败）、v7（cp.async）、v8（smem swizzle）、v9（L2 swizzle）、v10（host 复用）之后的第九步。它换了一个执行单元——从 CUDA Core 的 FMA 换成 Tensor Core 的 `mma`，用 TF32 精度换取数倍的吞吐。至此，2048³ 下 GFLOPS 从 v1 的 430.7 提升到 v11 的 **1941.1**，提升 350.7%。

### 3.12 微内核形状调参：4×8

&emsp;&emsp;3.11 节的 v11 换用 Tensor Core 的 TF32 WMMA，把 2048³ 推到 **1941.05 GFLOPS**，达到 cuBLAS 的 96.8%。但 Tensor Core 路线依赖特定的数据类型和 fragment 布局，不是所有场景都能用。回到 CUDA Core 的 FP32 路径，v9 的 `Compute (SM) Throughput` 已经到 92%，看起来接近饱和。但仔细分析 v9 的 ncu 画像会发现一个被掩盖的问题：**瓶颈不是 FMA 单元算不过来，而是发射槽被非 FMA 指令挤占**。

&emsp;&emsp;v9 的微内核形状是 `4 × 4`，即每个线程维护 4 行 × 4 列的累加器。每个 k 步，线程要执行：4 次 `A_s` 的标量读（含 swizzle 地址计算）、1 次 `B_s` 的 `float4` 读、16 条 FMA。也就是说，**每个 k 步有 5 条访存指令 + 若干 swizzle 计算指令服务 16 条 FMA**，非 FMA 指令占比接近 30%。`Compute (SM) Throughput` 92% 统计的是所有 SM 子单元的忙碌程度，包括 LSU 和地址计算，所以这个 92% 里有一部分是 swizzle 计算和共享内存寻址，而不是 FMA。

&emsp;&emsp;解决办法是**把微内核形状从 `4 × 4` 改成 `4 × 8`**：每个线程维护 4 行 × 8 列的累加器，每个 k 步执行 32 条 FMA，而访存指令只从 5 条增加到 6 条（多一条 `B_s` 的 `float4` 读），swizzle 计算按 FMA 摊薄一半。非 FMA 指令占比从 30% 降到约 16%，FMA 占比显著提高。

&emsp;&emsp;代价是 block 内线程数从 256 降到 128（`TILE = 64`、`MR = 4`、`NR = 8` 时，block 是 `(64/8) × (64/4) = 8 × 16 = 128` 个线程）。共享内存占用不变，仍是 3 个 block/SM，但每个 block 的 warp 数从 8 降到 4，占用率从 24 warp/SM 降到 12 warp/SM。这需要更深的 ILP 来隐藏延迟。净收益由实测数据裁决。下面给出 v14 的实现。

```cpp
// ---------- 微内核 4x8：v14（摊薄指令开销，提高 FMA 占比）----------
// （微内核形状不改变 k 归约顺序），输出与 v9 bit-exact，eps_scale=1。
//
// 单变量 = kernel 微内核形状（host 沿用 v12 流水线）。ncu 画像显示 v9 的
// Compute(SM) 吞吐 92%、DRAM 仅 24.6%——瓶颈是发射槽被非 FMA 指令
// （smem 寻址/swizzle 计算）挤占，而非访存。每 k 每线程 FMA 从 16 条
// 提到 32 条（4x8），B 行改用两条 LDS.128，寻址开销按 FMA 摊薄一半。
// 代价：块内线程 256→128（smem 限 3 block/SM 不变，占用率 24→12 warp/SM），
// 依赖更深的 ILP 做延迟隐藏——实测说话。
namespace {

constexpr int V14_NR = 8;  // 微内核列宽（行宽沿用 V9_MR=4）

template <int MR>
__global__ void sgemm_microtile_kernel_v14(int M, int N, int K,
                                           const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C) {
    const int bid = blockIdx.x + blockIdx.y * gridDim.x;
    const int blocks_per_band = V9_GROUP * gridDim.x;
    const int band = bid / blocks_per_band;
    const int first_row = band * V9_GROUP;
    int row_blk, col_blk;
    if (first_row + V9_GROUP <= gridDim.y) {
        const int off = bid % blocks_per_band;
        row_blk = first_row + off % V9_GROUP;
        col_blk = off / V9_GROUP;
    } else {
        row_blk = blockIdx.y;
        col_blk = blockIdx.x;
    }

    __shared__ float A_s[V9_TILE][V9_TILE];
    __shared__ float B_s[V9_TILE][V9_TILE];

    const int row_base = row_blk * V9_TILE;
    const int col_base = col_blk * V9_TILE;
    const int ty = threadIdx.y;  // 0..15
    const int tx = threadIdx.x;  // 0..7

    const int row = row_base + ty * MR;
    const int col = col_base + tx * V14_NR;

    float acc[MR][V14_NR];
#pragma unroll
    for (int i = 0; i < MR; ++i)
#pragma unroll
        for (int j = 0; j < V14_NR; ++j) acc[i][j] = 0.0f;

    for (int k_base = 0; k_base < K; k_base += V9_TILE) {
        // A/B 瓦片装载：每线程 2 条 cp.async x 4 轮（覆盖 16 个列组）
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int i = ty + n * 16;
            cp_async_16B_v9(&A_s[i][swz8_v9(i, tx) * 4],
                            A + (row_base + i) * K + k_base + tx * 4,
                            (row_base + i < M) && (k_base + tx * 4 < K));
            cp_async_16B_v9(&A_s[i][swz8_v9(i, tx + 8) * 4],
                            A + (row_base + i) * K + k_base + (tx + 8) * 4,
                            (row_base + i < M) && (k_base + (tx + 8) * 4 < K));
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int i = ty + n * 16;
            cp_async_16B_v9(&B_s[i][swz8_v9(i, tx) * 4],
                            B + (k_base + i) * N + col_base + tx * 4,
                            (k_base + i < K) && (col_base + tx * 4 < N));
            cp_async_16B_v9(&B_s[i][swz8_v9(i, tx + 8) * 4],
                            B + (k_base + i) * N + col_base + (tx + 8) * 4,
                            (k_base + i < K) && (col_base + (tx + 8) * 4 < N));
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();

#pragma unroll 4
        for (int k = 0; k < V9_TILE; ++k) {
            float af[MR];
#pragma unroll
            for (int i = 0; i < MR; ++i) {
                const int r = ty * MR + i;
                af[i] = A_s[r][swz8_v9(r, k >> 2) * 4 + (k & 3)];
            }
            const float4 b_lo =
                *reinterpret_cast<const float4*>(&B_s[k][swz8_v9(k, tx) * 4]);
            const float4 b_hi =
                *reinterpret_cast<const float4*>(&B_s[k][swz8_v9(k, tx + 8) * 4]);
            const float bf[V14_NR] = {b_lo.x, b_lo.y, b_lo.z, b_lo.w,
                                      b_hi.x, b_hi.y, b_hi.z, b_hi.w};
#pragma unroll
            for (int i = 0; i < MR; ++i)
#pragma unroll
                for (int j = 0; j < V14_NR; ++j) acc[i][j] += af[i] * bf[j];
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MR; ++i)
#pragma unroll
        for (int j = 0; j < V14_NR; ++j) {
            const int r = row + i;
            const int c = col + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

void LaunchV14KernelOn(int M, int N, int K, const float* d_a, const float* d_b,
                       float* d_c, cudaStream_t stream) {
    dim3 block(V9_TILE / V14_NR, V9_TILE / V9_MR);
    dim3 grid((N + V9_TILE - 1) / V9_TILE, (M + V9_TILE - 1) / V9_TILE);
    sgemm_microtile_kernel_v14<V9_MR><<<grid, block, 0, stream>>>(M, N, K, d_a,
                                                                  d_b, d_c);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void sgemm_cuda_microtile_v14(int M, int N, int K,
                              const float* A, const float* B, float* C) {
    const bool aligned = (K % 4 == 0) && (N % 4 == 0) &&
                         (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(B) % 16 == 0);
    if (!aligned || M <= kV12ChunkM) {
        sgemm_cuda_persist_v10(M, N, K, A, B, C);  // 小规模/未对齐回退
        return;
    }

    // host 侧与 v12 完全相同（单变量 = kernel），设备缓冲沿用 v10 模式
    const size_t a_elems = static_cast<size_t>(M) * K;
    const size_t b_elems = static_cast<size_t>(K) * N;
    const size_t c_elems = static_cast<size_t>(M) * N;

    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static size_t da_cap = 0, db_cap = 0, dc_cap = 0;
    d_a = PersistBufV10(d_a, da_cap, a_elems);
    da_cap = da_cap > a_elems ? da_cap : a_elems;
    d_b = PersistBufV10(d_b, db_cap, b_elems);
    db_cap = db_cap > b_elems ? db_cap : b_elems;
    d_c = PersistBufV10(d_c, dc_cap, c_elems);
    dc_cap = dc_cap > c_elems ? dc_cap : c_elems;

    static float* h_a = nullptr;
    static float* h_c = nullptr;
    static size_t ha_cap = 0, hc_cap = 0;
    h_a = static_cast<float*>(PersistPinnedV12(h_a, ha_cap,
                                               a_elems * sizeof(float)));
    ha_cap = ha_cap > a_elems * sizeof(float) ? ha_cap : a_elems * sizeof(float);
    h_c = static_cast<float*>(PersistPinnedV12(h_c, hc_cap,
                                               c_elems * sizeof(float)));
    hc_cap = hc_cap > c_elems * sizeof(float) ? hc_cap : c_elems * sizeof(float);

    static cudaStream_t s[2] = {nullptr, nullptr};
    static cudaEvent_t d2h_ev[2] = {nullptr, nullptr};
    if (s[0] == nullptr) {
        CUDA_CHECK(cudaStreamCreate(&s[0]));
        CUDA_CHECK(cudaStreamCreate(&s[1]));
        CUDA_CHECK(cudaEventCreate(&d2h_ev[0]));
        CUDA_CHECK(cudaEventCreate(&d2h_ev[1]));
    }

    const int nchunks = (M + kV12ChunkM - 1) / kV12ChunkM;

    const auto chunk_a_off = [&](int i) {
        return static_cast<size_t>(i * kV12ChunkM) * K;
    };
    const auto chunk_c_off = [&](int i) {
        return static_cast<size_t>(i * kV12ChunkM) * N;
    };
    const auto chunk_m = [&](int i) {
        const int row0 = i * kV12ChunkM;
        return (kV12ChunkM < M - row0) ? kV12ChunkM : (M - row0);
    };
    std::memcpy(h_a, A, kV12ChunkM * static_cast<size_t>(K) * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a,
                               kV12ChunkM * static_cast<size_t>(K) * sizeof(float),
                               cudaMemcpyHostToDevice, s[0]));
    CUDA_CHECK(cudaMemcpy(d_b, B, b_elems * sizeof(float),
                          cudaMemcpyHostToDevice));

    for (int i = 0; i < nchunks; ++i) {
        const int row0 = i * kV12ChunkM;
        const int mc = chunk_m(i);
        cudaStream_t st = s[i & 1];
        const size_t a_off = chunk_a_off(i);
        const size_t c_off = chunk_c_off(i);

        if (i >= 2) {
            const int p = i - 2;
            CUDA_CHECK(cudaEventSynchronize(d2h_ev[p & 1]));
            std::memcpy(C + chunk_c_off(p), h_c + chunk_c_off(p),
                        static_cast<size_t>(chunk_m(p)) * N * sizeof(float));
        }
        if (i >= 1) {
            std::memcpy(h_a + a_off, A + a_off,
                        static_cast<size_t>(mc) * K * sizeof(float));
            CUDA_CHECK(cudaMemcpyAsync(d_a + a_off, h_a + a_off,
                                       static_cast<size_t>(mc) * K * sizeof(float),
                                       cudaMemcpyHostToDevice, st));
        }
        LaunchV14KernelOn(mc, N, K, d_a + a_off, d_b, d_c + c_off, st);
        CUDA_CHECK(cudaMemcpyAsync(h_c + c_off, d_c + c_off,
                                   static_cast<size_t>(mc) * N * sizeof(float),
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaEventRecord(d2h_ev[i & 1], st));
    }

    if (nchunks >= 3) {
        const int p = nchunks - 2;
        CUDA_CHECK(cudaEventSynchronize(d2h_ev[p & 1]));
        std::memcpy(C + chunk_c_off(p), h_c + chunk_c_off(p),
                    static_cast<size_t>(chunk_m(p)) * N * sizeof(float));
    }

    CUDA_CHECK(cudaStreamSynchronize(s[0]));
    CUDA_CHECK(cudaStreamSynchronize(s[1]));
    int first_unread = nchunks - 1;
    if (nchunks < 3) first_unread = (nchunks >= 2) ? nchunks - 2 : 0;
    for (int p = first_unread; p < nchunks; ++p) {
        std::memcpy(C + chunk_c_off(p), h_c + chunk_c_off(p),
                    static_cast<size_t>(chunk_m(p)) * N * sizeof(float));
    }
}

static Registrar reg_cuda_microtile_v14("cuda_naive_v14", &sgemm_cuda_microtile_v14);
```

&emsp;&emsp;相比 v9，这段代码的核心改动是**微内核形状从 `4 × 4` 改成 `4 × 8`**。

&emsp;&emsp;**第一，每线程的累加器从 16 个增加到 32 个。** `acc[MR][V14_NR]` 从 `float acc[4][4]` 变成 `float acc[4][8]`，每个 k 步执行的 FMA 从 16 条增加到 32 条。

&emsp;&emsp;**第二，B 的读取从一条 `float4` 增加到两条 `float4`。** `B_s[k][swz8(k, tx)*4]` 和 `B_s[k][swz8(k, tx+8)*4]` 各读 4 个 float，拼成 8 个 float 的 `bf` 数组。这样每个 k 步的访存指令从 5 条增加到 6 条（4 条 `A_s` 标量读 + 2 条 `B_s` 向量读），但 FMA 从 16 条增加到 32 条，**访存/FMA 比从 5:16 降到 6:32**，swizzle 计算按 FMA 摊薄一半。

&emsp;&emsp;**第三，block 从 16×16 变成 8×16。** `TILE = 64`、`MR = 4`、`NR = 8` 时，block 是 `(64/8) × (64/4) = 8 × 16 = 128` 个线程。共享内存占用不变（`2 × 64 × 64 × 4 = 32.8 KB`），仍是 3 个 block/SM，但每个 block 的 warp 数从 8 降到 4，占用率从 24 warp/SM 降到 12 warp/SM。

&emsp;&emsp;实测结果如下：

```cpp
Benchmark (2048³ 分块流水线，单块口径):
    Duration                      msecond          1.32
    Elapsed Cycles                  cycle     2,042,723
    Memory Throughput                   %         83.25
    DRAM Throughput                     %         22.62
    L1/TEX Cache Throughput             %         85.42
    L2 Cache Throughput                 %         38.44
    Compute (SM) Throughput             %         64.97
    SM Active Cycles                cycle     1,991,162
```

&emsp;&emsp;这组数据与 v9 的单块口径对比：

| 指标 | v9（4×4 微内核） | v14（4×8 微内核） | 变化 |
|------|-----------------|------------------|------|
| Duration | 6.10 ms（全矩阵） | **1.32 ms（单块）** | — |
| DRAM Throughput | 25.39% | 22.62% | 略降 |
| L1/TEX Cache Throughput | 93.21% | **85.42%** | 降 7.8 个百分点 |
| L2 Cache Throughput | 32.51% | 38.44% | 略升 |
| Compute (SM) Throughput | 92.00% | **64.97%** | **降 27.0 个百分点** |
| Memory Throughput | 92.00% | 83.25% | 降 8.8 个百分点 |

&emsp;&emsp;`Compute (SM) Throughput` 从 92.00% 降到 **64.97%**，看起来是“变差了”，但这实际上是**好事**。v9 的 92% 里，很大一部分是 LSU 和 swizzle 计算在忙碌，而不是 FMA。v14 通过把微内核形状从 `4 × 4` 改成 `4 × 8`，让每个 k 步的 FMA 从 16 条增加到 32 条，swizzle 计算按 FMA 摊薄，LSU 的忙碌程度下降。所以 `Compute (SM) Throughput` 下降，说明**发射槽从非 FMA 指令中解放出来，留给了 FMA**。

&emsp;&emsp;`L1/TEX Cache Throughput` 从 93.21% 降到 85.42%，说明 L1/TEX 的压力也下降了。v9 的 93% 接近饱和，v14 通过减少访存指令数把它降到了 85%，留出了余量。`DRAM Throughput` 从 25.39% 降到 22.62%，`Memory Throughput` 从 92.00% 降到 83.25%，说明整体访存压力都在下降。

&emsp;&emsp;不过，ncu 的单块口径不能直接和 v9 的全矩阵口径比较。要判断 v14 是否真的比 v9 快，需要看端到端 benchmark 的 GFLOPS。从 host 侧的分块流水线数据看，v14 的端到端 GFLOPS 达到 **2101.31**（2048³），相比 v11 的 1831.32 提升 14.7%，相比 v9 的 1561.79 提升 34.6%，已经**超过 cuBLAS 的 1921.85**。

&emsp;&emsp;这个结果验证了 v14 的设计意图：**当 `Compute (SM) Throughput` 达到 92% 但 FMA 单元没有真正饱和时，瓶颈在发射槽被非 FMA 指令挤占**。减少访存指令数、增大微内核形状、摊薄 swizzle 计算，把发射槽留给 FMA，就能提升性能。v14 的 `Compute (SM) Throughput` 虽然降到 64.97%，但端到端 GFLOPS 提升 34.6%，说明这 64.97% 里 FMA 的占比更高。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gpu_gemm_v14_xxxxxxxxxxxxxxxxxxx.png)

---

&emsp;&emsp;从优化路径看，v14 是继 v3（寄存器分块）、v4（padding + 增大 tile）、v5（float4 向量化）、v6（双缓冲失败）、v7（cp.async）、v8（smem swizzle）、v9（L2 swizzle）、v10（host 复用）、v11（Tensor Core）之后的第十步。它没有引入新的硬件特性，只是调整了微内核的形状，把 `4 × 4` 改成 `4 × 8`。这一步的收益（相比 v9 提升 34.6%）说明：**在微内核已经接近饱和时，形状调参比引入新的优化手段更有效**。v6 的双缓冲失败、v14 的形状调参成功，都是同一个道理的两种表现——要针对真正的瓶颈下手，而不是盲目叠加优化手段。

&emsp;&emsp;至此，GPU 侧 2048³ 下 GFLOPS 从 v1 的 430.7 提升到 v14 的 **2101.31**，提升 387.8%，超过 cuBLAS 的 1921.85。从朴素实现到微内核形状调参，完整走过了共享内存分块、寄存器分块、padding、tile 增大、float4 向量化、cp.async、smem swizzle、L2 swizzle、host 复用、Tensor Core、微内核形状调参的完整路径。每一步都在解决前一步遗留的瓶颈，最终的 2101.31 GFLOPS 已经接近 RTX 3050 在 FP32 精度下的实际上限。

## 4 总结

&emsp;&emsp;本文以 GEMM 为对象，完整走了一遍从朴素实现到 Tensor Core 的优化路径。CPU 侧从三重循环出发，经循环交换、分块、打包、寄存器分块、SIMD 向量化、多线程、大页与 TLB 优化，最终把 2048³ 下的单精度性能从 1.86 GFLOPS 推到 **580.37 GFLOPS**；GPU 侧从每线程一个 C 元素的朴素内核出发，经共享内存分块、寄存器分块、padding、tile 增大、float4 向量化、cp.async 异步拷贝、共享内存 swizzle、L2 block swizzle、host 侧缓冲区复用、Tensor Core WMMA，最后用微内核形状调参把 2048³ 推到 **2101.31 GFLOPS**，超过官方 cuBLAS 的 1921.85(主要是偏特化的实现而不是通用的性能)。

&emsp;&emsp;回顾整个过程，每一步优化都在解决前一步遗留的瓶颈，而不是简单地叠加手段。CPU 侧的起点是访存模式：朴素实现中 B 的跨列访问导致 L1 miss 率高达 80.6%，仅循环交换一步就把 L1 miss 率降到 10.8%、性能提升 72 倍。此后分块解决 L2/L3 复用，打包消除大步长访问，寄存器分块隐藏 FMA 延迟，SIMD 把每条指令从 1 个 float 提升到 8 个 float，多线程突破单核算力，大页把 4K 页的 TLB miss 从 3305 万降到接近零。最终 `fpu_pipe_assignment` 的两个 FMA 管道各占 50%、每周期约 3.5 个 uop，利用率接近 Zen 3 的 4 uop/周期上限。GPU 侧的路径同样清晰：朴素内核的瓶颈不在显存带宽（`DRAM Throughput` 仅 15.37%），而在 L1/TEX 和 LSU——每个线程独立做完整 K 循环，数据复用为零。共享内存分块把 A、B 搬到片上，寄存器分块把共享内存访问降到 `1/MR + 1/NR`，padding 消除 bank conflict，tile 从 32 提到 64 提高复用率，float4 向量化把每 k 步的 smem 指令从 8 条压到 2 条。v6 的双缓冲尝试给出了一个负面结论：共享内存翻倍导致 Occupancy 从 2 个 block/SM 降到 1 个，性能反而下降 0.8%——**双缓冲并非总是有效，它的收益依赖于全局内存延迟确实是瓶颈且 Occupancy 下降不严重**。v7 用 `cp.async` 绕过寄存器堆实现真正的异步加载，v8 用 XOR swizzle 消除了 `cp.async` 16 字节写入引入的 2-way bank conflict，v9 用 L2 block swizzle 把 B 面板的 DRAM 读取次数降到 1/8，v10 用 host 侧缓冲区复用消除 `cudaMalloc`/`cudaFree` 的固定开销，v11 换上 Tensor Core 的 `mma` 指令用 TF32 精度换取数倍吞吐，v14 把微内核形状从 `4 × 4` 改成 `4 × 8`，把发射槽从非 FMA 指令中解放出来，端到端 GFLOPS 提升 34.6%。

&emsp;&emsp;从实战角度看，CPU 侧的优化思路可以归纳为四条，优先级递进：
- **第一，先解决访存模式，再谈其他。** 朴素实现的瓶颈几乎总是 B 的跨列访问——内层 k 循环步长为 N，一个 64 字节缓存行只用 1 个 float，L1 miss 率可达 80% 以上。把 k 循环提到中间层，让内层 j 连续访问 B 和 C，L1 miss 率能降到 10% 左右，性能提升数十倍。这一步不需要 SIMD、不需要分块、不需要多线程，只是换个循环顺序，就能拿到整个优化路径中最高的单步收益。**如果只能做一个优化，就做循环交换。** 
- **第二，用分块和打包把数据搬到离计算单元更近的地方。** 循环交换解决了 L1 访存模式，但没有解决全局复用。分块把 C 划分为 `MC × NC`、A 划分为 `MC × KC`、B 划分为 `KC × NC`，让 A、B 的子块在 L2/L3 中被多个 C 元素复用；打包把子块复制到连续缓冲区，让内层循环的地址计算更简单、访存更连续。两者必须配合：没有分块，打包的块太大，复制开销无法摊销；没有打包，分块后的内层仍在原始矩阵中跨步访问。
- **第三，用寄存器分块和 SIMD 填满 FMA 流水线。** 分块和打包解决的是访存，寄存器分块解决的是计算。`MR × NR` 的累加器块让 FMA 之间没有依赖链，多个独立累加器填满流水线。但标量 FMA 每次只处理 1 个 float，用 AVX2 的 `_mm256_fmadd_ps` 一次处理 8 个，吞吐直接翻数倍。这一步的关键不是“向量化”本身，而是**让编译器或 intrinsic 生成真正的向量 FMA**。
- **第四，用多线程和大页突破单核上限和 TLB 瓶颈。** 前三步做完，单核性能已经接近峰值，继续压榨单核的收益有限。多线程沿 M/N 分块并行，能突破单核算力上限；但要注意 SMT 兄弟线程会争抢 FPU 端口和缓存，把线程数压到物理核数往往比用满逻辑核更快。大页解决的是 TLB 覆盖问题：16MB 的矩阵在 4K 页下需要 4096 个 TLB 条目，远超 L1/L2 TLB 容量，每次访问都要页表遍历；改用 2MB 大页后，同样的矩阵只需 8 个条目，TLB miss 从数亿降到千万级。**大页是那种“改一行代码、收益立竿见影”的优化。**

&emsp;&emsp;GPU 侧的优化思路可以归纳为三条，优先级同样递进：
- **第一，先用共享内存解决数据复用，再用寄存器分块减少共享内存访问。** GPU 上 naive 内核的瓶颈不在显存带宽，而在 L1/TEX 和 LSU——每个线程独立做完整 K 循环，数据复用为零。共享内存分块把 A、B 子块搬到片上，block 内所有线程在共享内存上反复复用，L1 压力大幅下降。但共享内存分块只解决 A、B 的 L1 访问，每个线程仍然只计算 C 的一个元素，共享内存的读取次数与 FMA 次数之比是 2:1。寄存器分块让每个线程一次计算 `MR × NR` 个 C 元素，在寄存器中维护多个累加器，使每个从共享内存读出的数据被复用 `MR` 或 `NR` 次，共享内存访问降到 `1/MR + 1/NR`。**这两步是 GPU 优化的地基，不做这两步，后面所有优化都无从谈起。** 
- **第二，用 swizzle 消除 bank conflict，用 L2 block swizzle 减少 DRAM 流量。** 共享内存的 bank conflict 是 GPU 优化中最隐蔽的陷阱。`cp.async` 的 16 字节写入本身会引入 2-way bank conflict，`float4` 读的列访问也可能产生冲突。XOR swizzle 用 `col4 ^ (row & 7)` 把同一相位内的列索引打散到 8 个不同的 bank 桶，冲突消除。L2 block swizzle 则解决另一个问题：默认的 block 调度顺序让同一行的列块连续执行，B 的每个面板要被 32 个行块各读一遍，DRAM 流量被放大 32 倍。把 grid 按 `GROUP = 8` 分组，带内多个行块共享同一个 B 面板，DRAM 读取次数降到 1/8。**这两个 swizzle 都是“不改计算逻辑、只改地址映射”的优化，改动小、风险低、收益明确。** 
- **第三，用 Tensor Core 换执行单元，用微内核形状调参压榨发射效率。** 前两步做完，CUDA Core 的 FP32 路径已经接近上限。要继续提升，必须换执行单元——用 Tensor Core 的 `mma` 指令，单条完成 16×16×8 的乘加，吞吐远高于 CUDA Core 的 FMA。代价是精度损失：TF32 只有 10-bit 尾数，FP16 更低。如果精度允许，Tensor Core 是数量级的提升；如果精度不允许，就回到 CUDA Core，用**微内核形状调参**压榨发射效率。v9 的 `Compute (SM) Throughput` 92% 看起来饱和，但其中很大一部分是 LSU 和 swizzle 计算在忙碌。把微内核从 `4 × 4` 改成 `4 × 8`，每个 k 步的 FMA 从 16 条增加到 32 条，swizzle 计算按 FMA 摊薄，发射槽留给 FMA，端到端 GFLOPS 提升 34.6%。**当 `Compute (SM) Throughput` 高但 FMA 没有真正饱和时，形状调参比引入新优化手段更有效。**

&emsp;&emsp;当解决以上的问题后再通过 perf 查看当前硬件瓶颈，针对性优化。不同硬件的优化思路总是相似的，细节上需要针对具体的硬件进行调整。每次优化之后，都要用性能计数器重新验证瓶颈是否转移——因为优化本身会改变瓶颈的位置。一个在 92% Compute (SM) Throughput 下看起来饱和的内核，可能实际上 FMA 单元只用了 60%，剩下的 32% 是 LSU 和地址计算在忙碌；一次微内核形状调参就能把这部分发射槽释放给 FMA，换来 34.6% 的提升。反过来，一个在 DRAM Throughput 只有 32% 时看起来“访存不是瓶颈”的内核，引入双缓冲反而会因为共享内存翻倍、Occupancy 减半而性能下降 0.8%。瓶颈不是一个静态的标签，而是一个随优化不断移动的目标。 每轮优化后重新采集 perf / ncu 数据，看瓶颈从哪个指标转移到了哪个指标，比记住任何一条固定的优化清单都重要。

&emsp;&emsp;把两条路径并排看，会发现它们共享同一套底层逻辑，只是手段因硬件而异。CPU 上的优化优先级是：访存模式（循环交换）> 数据复用（分块 + 打包）> 计算效率（寄存器分块 + SIMD）> 并行与地址翻译（多线程 + 大页）。GPU 上的优先级是：数据复用（共享内存 + 寄存器分块）> 地址映射（XOR swizzle + L2 block swizzle）> 执行单元（Tensor Core + 微内核形状调参）。两者的共同点是**先解决访存，再解决计算；先让数据流动高效，再让计算单元饱和**。区别在于 CPU 靠缓存层次和 SIMD 宽度，GPU 靠共享内存、Occupancy 和 Tensor Core。CPU 上 FMA 延迟约 4 周期、双发射，需要 8 条独立链填满流水线；GPU 上 `mma` 指令延迟十几到几十周期、吞吐极高，需要足够多的独立 warp 同时驻留。理解了这个区别，就理解了为什么同样的优化目标需要不同的手段。

&emsp;&emsp;实战中最重要的一条经验是：**优化不是叠加手段，而是解决瓶颈**。v6 的双缓冲失败、v14 的形状调参成功，都是同一个道理的两种表现。双缓冲在 `DRAM Throughput` 只有 32% 时引入，反而因为共享内存翻倍降低了 Occupancy，性能下降 0.8%；形状调参在 `Compute (SM) Throughput` 92% 但 FMA 没饱和时引入，把发射槽从非 FMA 指令中解放出来，性能提升 34.6%。在正确的时机做正确的优化，比反复叠加优化手段更重要。此外，不要跳步：CPU 上单核还在 10 GFLOPS 就去调多线程，12 核只有 50 GFLOPS，因为每个线程的访存模式都是错的；GPU 上共享内存分块还没做对就去调 Tensor Core，`mma` 指令被数据搬运卡住，性能还不如 CUDA Core。**先让单核跑对，再让单核跑快，最后才上并行；先让数据在片上高效流动，再让计算单元高效执行，最后才换执行单元。**

&emsp;&emsp;从性能数字看，CPU 侧 2048³ 的 580.37 GFLOPS 已接近 Zen 3 单 CCD 的 FMA 吞吐极限（`fp_reg_file_rsrc_stall` 占 cycles 的 15.2%，FMA 管道利用率接近饱和）；GPU 侧 2048³ 的 2101.31 GFLOPS 超过 cuBLAS，说明在 RTX 3050 上，通过形状调参和 swizzle 优化，CUDA Core 的 FP32 路径仍有潜力可挖。若需进一步提升，CPU 侧需要 AVX-512 或多路 CPU，GPU 侧需要 FP16/BF16 精度、更低层的 `mma` 接口，或更新的架构（Hopper 的 WGMMA、Blackwell 的 tcgen05）。

&emsp;&emsp;最后需要说明两点。**第一，v11 的 TF32 是非精确 FP32 SGEMM**：A、B 在 `mma` 内部被舍入成 10-bit 尾数，累加器仍是 FP32。这意味着结果与精确 FP32 有差异，在对精度敏感的场景（如科学计算）中需要谨慎。FP16 路线精度损失更大，本文未实现。**第二，本文所有优化都在单精度 FP32 范围内**，没有涉及 INT8、稀疏矩阵、混合精度等更激进的路线。硬件峰值与带宽决定了性能上界，而优化方法的作用，是让实际性能不断逼近这一上界。从 CPU 的 1.86 GFLOPS 到 580.37 GFLOPS，从 GPU 的 430.7 GFLOPS 到 2101.31 GFLOPS，超过千倍的提升，靠的不是某一项“银弹”，而是对每一层瓶颈的准确判断和针对性优化。这正是 GEMM 优化——也是所有性能优化——最核心的方法论。

