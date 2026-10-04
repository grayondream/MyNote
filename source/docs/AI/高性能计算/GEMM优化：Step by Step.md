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

