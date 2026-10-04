# GEMM优化：Step by Step
---

## 1 前言

&emsp;&emsp;在深度学习模型推理中，GEMM（通用矩阵乘法）是绝大多数计算负载的核心来源。无论是 Transformer 架构中的线性层，还是卷积网络展开后的全连接计算，最终都会归结为大规模矩阵乘法运算。有研究指出，在实用序列长度下，GEMM 操作可占据模型推理总计算量的 69%–99%。因此，GEMM 的优化程度直接决定了模型推理的吞吐、延迟与能效表现，是推理系统设计中绕不开的基础问题。

### 1.1 什么是GEMM

&emsp;&emsp;GEMM = **GE**neral **M**atrix **M**ultiply，通用矩阵乘法。数学形式是：

$$
C = \alpha \cdot A \times B + \beta \cdot C
$$

- $A$ 是 $M \times K$ 的矩阵（M行K列）
- $B$ 是 $K \times N$ 的矩阵（K行N列）
- $C$ 是 $M \times N$ 的矩阵（M行N列）
- $\alpha, \beta$ 是标量系数

&emsp;&emsp;在绝大多数讨论中，取 $\alpha=1, \beta=0$，即最简形式 $C = A \times B$。本文只讨论这种情况，理解之后推广到一般情况是平凡的。 矩阵乘法的定义：$C$ 的第 $i$ 行第 $j$ 列元素为

$$
C[i][j] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]
$$

&emsp;&emsp;**C的每个元素，是A的对应行与B的对应列做点积**（对应位置相乘再求和）。

---- 

&emsp;&emsp;一个 $M \times N \times K$ 的矩阵乘法需要：

- **乘法次数**：$M \times N \times K$
- **加法次数**：$M \times N \times (K-1)$
- **总计浮点运算（FLOP）**：约 $2MNK$

&emsp;&emsp;例如，如果$M=N=K=4096$的Float矩阵计算量为：$2 \times 4096^3 \approx 1.37 \times 10^{11}$ FLOP，即**1370亿次浮点运算**。

### 1.2 CPU硬件影响GEMM速度的因素

&emsp;&emsp;通常PC机器的DDR内存虽然可以二维寻址，但是主流操作系统还是将物理地址统一映射为线性的虚拟地址。也就是无论C语言还是cpp定义的二维数组或者矩阵实际存储在内存中仍然是线性排列的。那么就存在一个问题矩阵应该按照什么顺序排列，这也就是常见的行主序还是列主序：
- 行主序：按照行的顺序优先存储;
- 列主序：按照列的顺序优先存储。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20260925_abf1d6.svg)

&emsp;&emsp;元素 `A[i][j]` 在内存中的地址偏移是 `i * K + j`（K是列数，也叫leading dimension，行距）。

&emsp;&emsp;现代计算机存储结构都是多层的，CPU基本都支持三级缓存。由于计算机领域的局部性原理，现代CPU缓存都会采用优先缓存相邻数据的策略来加速程序。从这一点上就能够看出内存Layout可能对数据访问延迟的影响，如果矩阵按照行主序排列那么当访问第i个元素缓存就会提前预取$i+1$甚至于之后位置的内存数据到缓存中，下次CPU再次访问就不需要再触发内存读写直接读取缓存就可以。而如果按照列主序存储的话每次访问第$i$个元素CPU缓存第$i+1$个元素，但是实际下一次需要访问$i * j + 1$位置的元素导致缓存失效必须触发内存读写操作。

&emsp;&emsp;不仅仅内存会影响性能，CPU执行同样会影响性能。CPU执行一条指令不是"瞬间完成"，而是分成了多个阶段（取指、译码、执行、写回等），像工厂的流水线一样。这引出了两个最重要的性能指标：

- **延迟（Latency）**：一条指令从发射到结果可用的时间，以周期（cycle）为单位。例如FMA指令的延迟通常是4个周期。
- **吞吐量（Throughput）**：每周期能发射几条同类指令。例如现代CPU每周期可发射2条FMA（注意：是"半条/周期"的倒数表述，即0.5 cycle per FMA）。

&emsp;&emsp;另外需要注意上面描述的是通用的情况，但是实际不同CPU不同指令大体上符合上面的描述，实际情况需要根据具体的指令来看，比如FMA指令。如果一条FMA需要4周期才能出结果，但每周期能发射2条，那么只有当前面的FMA还没算完、后面的FMA又需要用到它的结果时，流水线才会停顿。为了填满流水线，你需要**至少4/0.5=8条相互独立的FMA指令**同时在飞。

### 1.3 GPU硬件影响GEMM速度的因素

&emsp;&emsp;与CPU不同，GPU的设计哲学从一开始就不是"降低单条指令的延迟"，而是"用海量并行隐藏延迟"。理解这一点，是理解GPU上GEMM优化的前提。影响GPU上GEMM速度的硬件因素，可以沿着存储层次、执行单元、以及二者的配合方式来展开。

- **存储层次与带宽瓶颈**：GPU同样有多级存储，但其容量与带宽的比例与CPU差异巨大：

  - **全局内存（Global Memory / HBM）**：容量大（数十GB），但延迟高（数百周期），带宽相对有限。GPU上GEMM的第一个瓶颈往往就是全局内存带宽——如果每个线程都直接从全局内存读矩阵元素，算力再强也会被访存拖死。
  - **L2缓存**：所有SM共享，容量比CPU的L3小，但带宽高，用于缓解全局内存压力。
  - **共享内存（Shared Memory / SMEM）**：每个SM（流式多处理器）私有的片上存储，容量小（通常几十KB到上百KB），但延迟极低、带宽极高。GEMM优化的核心手段之一，就是把全局内存中的数据分块（tiling）搬到共享内存，让线程在片上反复复用，减少全局访存次数。
  - **寄存器（Register File）**：每个线程私有，速度最快。高性能GEMM内核会让每个线程在寄存器中缓存一小块结果（如8×8的累加器），在一段循环中只做FMA而不访问其他存储。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20260928_df6028.svg)

&emsp;&emsp;与CPU的行主序/列主序问题类似，GPU上矩阵的存储Layout同样关键。但GPU的应对方式不同：它通过**合并访问（coalesced access）**来保证效率——当一个warp（32个线程）访问全局内存时，如果它们访问的是连续地址，硬件会把这次访问合并成一次宽事务；如果地址分散，则会拆成多次事务，带宽利用率骤降。因此，GPU上的GEMM内核通常要求矩阵按特定Layout排列，或者在加载时通过共享内存做转置，以配合合并访问。

- **执行单元与吞吐量**：GPU的算力来自大量SM，每个SM内有：
  - **CUDA Core**：执行FP32/INT32等常规运算。FMA指令在CUDA Core上执行，延迟通常也是4周期左右，但一个SM每周期可以发射多条FMA（具体数量取决于架构，如Ampere上每个SM每周期可发射64条FP32 FMA）。
  - **Tensor Core**：专门为矩阵乘法设计的单元，一条指令就能完成一个小矩阵块（如16×16×16）的乘加。Tensor Core的吞吐量远高于CUDA Core，是现代GPU上GEMM性能的主要来源。但它的使用有约束：需要特定的数据类型（FP16、BF16、TF32、INT8等）、特定的数据Layout（如wmma/mma所要求的fragment排布），并且需要软件显式调用。
  - **特殊函数单元（SFU）**：处理超越函数等，与GEMM关系不大。

&emsp;&emsp;这里同样存在CPU那节提到的"延迟 vs 吞吐"问题，只是尺度不同。在GPU上，一条Tensor Core MMA指令的延迟可能有十几到几十周期，但吞吐量很高。要填满流水线，就需要**足够多的独立warp**同时驻留在SM上。这引出了GPU特有的概念：
  - **Occupancy（占用率）**：每个SM上活跃warp数与最大支持warp数之比。Occupancy太低，延迟无法被隐藏，执行单元会空闲；Occupancy太高，又可能因为寄存器或共享内存不足而限制每个线程能用的资源，反而降低单线程效率。
  - **Warp调度**：当某个warp因为等待内存或依赖前一条指令而停顿时，调度器立刻切换到另一个就绪warp。GPU正是靠这种"用并行度换延迟隐藏"的方式，让海量线程把高延迟的访存和计算重叠起来。

&emsp;&emsp;与CPU那节的结论对应：CPU上GEMM优化主要关注"如何让流水线不断流"（指令级并行、缓存预取、SIMD），而GPU上GEMM优化主要关注"如何用并行度隐藏延迟、如何让数据在存储层次间高效流动"（分块、共享内存、Tensor Core、Occupancy）。两者共享同一个底层逻辑——局部性原理和延迟隐藏——但具体手段因硬件架构而异。

> 硬件平台：
- RTX 3050
- AMD 5600X
    - Caches (sum of all):         
        - L1d:                       192 KiB (6 instances)
        - L1i:                       192 KiB (6 instances)
        - L2:                        3 MiB (6 instances)
        - L3:                        32 MiB (1 instance)
    - NUMA:                        
        - NUMA 节点：                1
        - NUMA 节点0 CPU：           0-11

## 2 GEMM CPU优化
### 2.1 CPU GEMM性能模型和优化方法

&emsp;&emsp;前言中已经详细描述了 GEMM 的数学形式，并指出单精度浮点运算量为 $2MNK$。要进一步优化 GEMM 性能，仅知道运算量还不够，还需要深入理解 CPU 架构和计算模型。

&emsp;&emsp;硬件设备本身的能力是固定的，无论如何优化都无法超越硬件上限。因此，优化的目标不是“消除”硬件限制，而是在给定硬件上尽可能逼近峰值算力与可用带宽所决定的上界。为此，需要先建立可量化的性能模型。

&emsp;&emsp;最常用的模型是算术强度与 **Roofline 模型**。算术强度定义为浮点运算次数与访存字节数之比：
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

&emsp;&emsp;GEMM 的数学定义为 $C[i,j]=\sum_{k=0}^{K-1}A[i,k]B[k,j]$。最直接的实现就是三重循环，逐元素计算 C：

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

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/deepseek_svg_20261001_adec77.svg)

&emsp;&emsp;这段代码逻辑正确，但性能极差。原因有两个：
- 内层 k 循环访问 `B[k * N + j]` 时，地址步长为 N。每次 k 增加 1，B 的访问就跳到下一行，跨越 N 个元素。若 N 较大，一个缓存行中往往只有一个元素被用到，其余部分被浪费。同时，A 的第 i 行和 B 的第 j 列在 k 循环中被反复读取，但每次只使用一次，没有跨 (i,j) 的复用。整个计算过程中，A 被读取了 N 次，B 被读取了 M 次，总访存量约为 $4MNK+4MNK+4MN$ 字节，而浮点运算量只有 $2MNK$。算术强度约为 $2MNK/(8MNK+4MN)\approx 0.25$ FLOP/Byte，远低于 Roofline 转折点，因此性能受内存带宽限制。
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

&emsp;&emsp;交换循环顺序后，虽然引入了C矩阵的频繁WRW,但是B矩阵不会再跨行读取也大幅度提升了矩阵A的缓存命中，从下面的结果能够看出来这一举措是非常有效的。
![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261001_16ac4c.svg)

&emsp;&emsp;为了证明我们关于缓存的猜想，我们来看下v1和v2两个版本实际运行的缓存命中率。
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

&emsp;&emsp;从 `perf stat` 的采集结果来看，naive_v1 在 2048³ 规模下的 L1-dcache miss 率高达 **80.6%**（96 亿次 miss / 119 亿次 load），而循环交换后的 naive_v2 只有 **10.8%**（6.1 亿次 miss / 56 亿次 load），相差约 7.5 倍。这正是朴素实现中 B 矩阵跨列访问的典型特征：内层 k 循环每次访问 `B[k*N+j]` 都要跨越 N 个元素，一个 64 字节缓存行往往只用到其中 1 个 float，其余 15 个被白白浪费；同时 A 的第 i 行和 B 的第 j 列在 (i,j) 双重循环中被反复加载，没有跨迭代的寄存器复用。循环交换把 k 提到中间层后，内层 j 循环连续访问 B 和 C，缓存行利用率大幅提升，总 load 次数也从 119 亿降到 56 亿。

&emsp;&emsp;更关键的是，访存瓶颈进一步压低了指令级并行。naive_v1 的 IPC 只有 **0.18**（413 亿条指令 / 2300 亿周期），意味着每 5.6 个周期才 retire 一条指令，CPU 绝大部分时间在等内存，FMA 流水线大量空转；naive_v2 的 IPC 回升到 **1.01**，虽然离理论峰值仍有距离，但已是 v1 的 5.6 倍。两者叠加的结果是：2048³ 下 naive_v1 耗时 48.76 秒、GFLOPS 仅 0.354，而 naive_v2 耗时 0.67 秒、GFLOPS 达 25.66，**性能差距被放大到 72 倍**。这组数据清楚地说明，GEMM 优化的第一步不是急着上向量化或分块，而是先通过循环交换把访存模式从“跨列跳跃”改成“连续扫描”——仅此一步就能收回两个数量级的性能。

### 2.4 分块

&emsp;&emsp;要突破全局复用的限制，必须让数据在更靠近计算单元的地方被多次复用。由于矩阵计算本身就具备局部独立性，可以通过分块局部计算最后再合并。把 C 划分为 $MC\times NC$ 的块，A 划分为 $MC\times KC$，B 划分为 $KC\times NC$。对每个 C 块，遍历 K 维的 $KC$ 块，将对应的 A、B 子块加载到缓存中，再在缓存内完成多次乘加。这样，A 子块的每一行和 B 子块的每一列都能在 L2/L3 中被多个 C 元素复用。

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

&emsp;&emsp;这里的三层分块循环对应着不同的缓存级别：最外层 jc、pc、ic 控制 L3/L2 分块，保证 A、B 面板在 L2/L3 中复用；内层 i、p、j 则是分块内的计算。分块大小需要根据缓存容量选择，例如 $MC=64$、$NC=64$、$KC=256$。分块后，算术强度从仅按 DRAM 计算的 $n/6$ 提升到按缓存容量计算的水平，性能通常能再提升 2–5 倍。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/deepseek_svg_20261001_b59684.svg)

&emsp;&emsp;但分块后的内层循环仍然每次从 A 取一个标量、从 B 取一行，数据加载频率较高。如果把 A、B 的子块预先复制到连续缓冲区中，就能让内层循环以更紧凑的方式读取——这正是 packing 的作用。理论上 V2 在缓存读写上几乎已经达到最优，V3 把矩阵分块再计算反而让逻辑更复杂了。

&emsp;实际运行也印证了这一点。V3 的 L2 缓存命中率恶化到了约 16%，perf 采集到的数据如下：

```
 Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v3/2048x2048x2048':

    11,213,735,498      cycles
    11,322,406,868      instructions
     6,859,088,356      L1-dcache-loads
       694,658,089      L1-dcache-load-misses
```

&emsp;L1 miss 率约为 10.1%（694M / 6859M），虽然和 V2 的 10.8% 接近，但 V3 的 L1 load 总数比 V2 多了约 22%（6859M vs 5639M），cycles 也多了约 9%。这说明分块并没有减少访存次数，反而因为三层额外循环的边界判断和地址计算引入了更多指令。**在 V2 已经解决 L1 访存模式的前提下，分块带来的 L2/L3 复用收益不足以抵消这些额外开销，所以 V3 无法超过 V2。**

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

&emsp;&emsp;V3和V4看起来很像但是还是有区别的。V3 和 V4 的区别，用一个比喻来说：V3 像在一个散装仓库里取货，工人每次要拿一个零件，然后走到仓库另一头，从一排跳跃的货架上取一列零件——每取一个就要跨过一大片用不到的货，走的路很长。V4 则是在开工之前，先把要用的 A 子块和 B 子块从散装货架上搬到一块连续的托盘上，之后内层计算只需要在托盘上顺序取货，不用再满仓库跑。搬运本身要花力气，但搬一次能让后续成百上千次取货都变快，这就是打包（packing）的作用。

&emsp;&emsp;从 perf 数据看，V4 的优化点非常具体。V3 的 L1-dcache-loads 是 68.6 亿次，V4 降到 52.4 亿次，少了 23.7%；instructions 从 113.2 亿降到 86.8 亿，少了 23.4%；cycles 从 112.1 亿降到 102.5 亿，少了 8.6%。这三项下降说明 V4 并没有减少真正的计算量——FMA 次数始终是 2048³ ≈ 85.9 亿次——但每条 FMA 周围的地址计算和循环控制指令明显减少了。V3 的内层每次都要重新计算 `B + (pc+p)*N + jc` 这种跨步地址，乘法 `(pc+p)*N` 在循环里反复出现；V4 的地址变成 `Bp + p*n`，基址固定、步长小，编译器更容易做强度削弱，硬件预取器也能识别出连续访问模式。

&emsp;&emsp;有一个反直觉的点值得注意：V4 的 IPC 从 V3 的 1.01 掉到了 0.85，L1 miss 率还从 10.1% 略升到 11.3%。但这不代表 V4 变差了。V4 的 L1 load 总数少了近四分之一，所以绝对 miss 次数反而从 6.95 亿降到 5.91 亿，少了 14.9%。miss 率的分母变小，比率略升是正常的。更重要的是，打包后的连续访问让硬件预取器能提前把下一段数据拉进 L1，即使 miss 率略高，每次 miss 的代价也更低——V3 的跨步访问让预取器无从下手，每次 miss 都是实打实的停顿。IPC 降低也不是性能退化，而是因为省下来的指令主要是地址计算和循环控制这类“辅助指令”，有效 FMA 在总指令中的占比提高了。

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

&emsp;&emsp;在 2048³ 规模下，V5 的 CPU 时间降到 574.7ms，GFLOPS 达到 29.90，在三个规模上全面反超 V2（633.9ms、27.10 GFLOPS）和 V4（690.7ms、24.87 GFLOPS）。V5 比 V2 快约 9.3%，比 V4 快约 17%。这个提升不是靠减少计算量——2048³ 的 FMA 次数始终是 85.9 亿次——而是靠让 CPU 用更少的周期完成同样的工作。V5 的 cycles 是 108.4 亿，V2 约 112 亿，V5 用更少的周期完成了同样的计算。

&emsp;&emsp;最显著的微观指标是 L1 miss 率。V5 的 L1-dcache-loads 是 49.6 亿，L1-dcache-load-misses 是 1.47 亿，miss 率约 **2.96%**。对比 V2 的 10.8%、V3 的 13.5%、V4 的 11.3%，这是一个数量级的改善。原因在于微内核用 `_mm256_loadu_ps` 一次加载 8 个 float，让 B 的缓存行利用率从标量版本的 37.5% 提到接近 100%。每次加载正好覆盖整个缓存行，miss 次数自然断崖式下降。而 V2 虽然通过循环交换解决了 B 的跨列访问问题，但内层仍是标量操作，每个缓存行只用到其中一部分，miss 率停在 10% 以上。这也解释了为什么 V5 能在 cycles 更少的情况下完成同样的计算——它把等待内存的时间转化成了有效的 FMA 发射。

&emsp;&emsp;V5 的 instructions 是 123.6 亿，比 V2 的 103.5 亿多约 19%。多出来的指令主要来自打包（复制 A、B 子块）和微内核的循环控制。但 V5 的 cycles 反而比 V2 少，说明多出来的指令是**高效指令**——它们没有引发额外的停顿，反而让 FMA 单元被喂得更满。IPC 约 1.14，不算高，但比 V3 的 1.01 和 V4 的 0.85 都好，说明 CPU 没有在等内存，也没有在空转。这一点在性能优化里很关键：指令多不等于慢，如果多出来的指令能把流水线填满，反而比指令少但频繁停顿的版本更快。

```cpp
Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v5/2048x2048x2048':

10,844,283,019      cycles                                                                
12,361,764,878      instructions                                                          
    4,961,292,893      L1-dcache-loads                                                       
    147,025,797      L1-dcache-load-misses  
```

&emsp;&emsp;从优化路径看，V2 到 V5 经历了四步：循环交换解决 L1 访存模式，分块改善 L2/L3 复用，打包把子块搬到连续缓冲区，微内核 + intrinsic 把内层访存向量化。每一步单独看收益有限，但叠加起来，V5 在 2048³ 下比 V2 快约 9%、比 V4 快约 17%。L1 miss 率从 10.8% 压到 2.96% 是这一轮优化的核心成果，也是 V5 最终反超的关键。这组数据说明，当访存模式经过循环交换和打包之后，**下一步的瓶颈不再是“读得少”，而是“读得宽”**——用一条向量指令替代八条标量加载，才是把 miss 率压到 3% 以下、把 GFLOPS 推过 29 的直接原因。

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

&emsp;&emsp;调用 AVX2 微内核的主循环与 2.6 类似，只需把 `micro_kernel` 替换为 `micro_kernel_avx2`，并把 NR 改为 8。AVX2 版本通常能在标量微内核基础上再提升 3–6 倍。至此单核性能已接近峰值，剩下的问题是单个核心的算力有限，需要把工作分摊到多个核心上。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261003_30c2ea.svg)

&emsp;&emsp;在 2048³ 规模下，V6 的 CPU 时间降到 285.0ms，GFLOPS 达到 60.28，相比 V5 的 530.5ms、32.39 GFLOPS 快了近一倍，相比 V2 的 513.4ms、33.46 GFLOPS 快了约 80%。这个提升来自微内核的彻底重写：V6 不再依赖编译器自动向量化，而是用 AVX2 intrinsic 显式加载 8 个 float、广播 A 的标量、执行向量 FMA。perf 数据也印证了这一点——L1-dcache-loads 从 V5 的 49.6 亿涨到 83.7 亿，但 L1-dcache-load-misses 从 1.47 亿上升到 4.51 亿的绝对值。从miss率和指令等方面看是劣化了的。

&emsp;&emsp;V6 的性能提升主要来自 **FMA 发射效率**，而不是 L1 miss 率。V6 的 instructions 从 V5 的 123.6 亿涨到 159.6 亿，多了 29%，但 cycles 从 108.4 亿只涨到 112.6 亿，IPC 从 1.14 升到 1.42。这说明 CPU 在执行更多指令的同时，周期数没有同比增加，流水线利用率提高了。

```cpp
 Performance counter stats for './build/bench_gemm --benchmark_filter=BM_sgemm/naive_v6/2048x2048x2048':

    11,261,600,769      cycles                                                                
    15,959,752,975      instructions                                                          
     8,366,425,515      L1-dcache-loads                                                       
       451,003,500      L1-dcache-load-misses 
```

&emsp;&emsp;**为什么 L1 miss 率反而升了？** 因为 V6 的 MR=8、NR=8 微内核一次加载 8 个 B 元素（32 字节），但缓存行是 64 字节，只用到一半。V5 的 NR=8 也是加载 8 个 float，但 V5 的微内核可能被编译器优化成了更紧凑的访存模式。V6 的 `_mm256_loadu_ps` 每次加载 32 字节，跨缓存行的概率比 V5 的标量加载更高，所以 miss 率略升。

&emsp;&emsp;从优化路径看，V6 是第一个把 GFLOPS 推过 60 的版本。它的核心改动是：**用 AVX2 intrinsic 替代标量微内核，用 `_mm256_fmadd_ps` 替代逐元素 FMA，用 `_mm256_set1_ps` 替代标量广播**。这些改动让每条指令处理 8 个 float，FMA 吞吐从标量时代的每周期 1~2 条提升到向量时代的每周期 1 条（但每条算 8 个）。在 5600X 的 Zen 3 架构上，256-bit FMA 会被拆成两个 128-bit uop，所以实际吞吐是每周期 4 个 float 的 FMA——这比 V5 的标量版本每周期 1~2 个 float 快了一倍以上，与实测的 1.8 倍提升吻合。



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

&emsp;&emsp;并行化后，性能通常能接近单核的 N 倍，但受限于共享 L3 和内存带宽，扩展比会略低于核数。若系统为 NUMA 架构，可用 `numactl --localalloc` 或 `OMP_PROC_BIND` 绑定线程到本地节点，进一步减少远端访问。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/deepseek_svg_20261001_d317e3.svg)


&emsp;&emsp;在 2048³ 规模下，V7 的单次端到端耗时是 **66.98ms**，但 CPU 时间只有 **51.77ms**，GFLOPS 达到 **331.84**。这里的 `Time` 和 `CPU` 差异（66.98 vs 51.77）说明多线程带来了额外的调度开销——`real_time` 包含线程启动、同步和等待的墙钟时间，而 `cpu_time` 是各线程实际占用 CPU 的时间总和。在 12 线程下，cpu_time / real_time ≈ 0.77，说明有约 23% 的墙钟时间花在了等待和调度上，这是多线程版本典型的开销。

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

&emsp;&emsp;从吞吐看，V7 的 331.84 GFLOPS 是 V6 单核 63.01 GFLOPS 的 **5.27 倍**。12 核理论上能到 12 倍，但实际只拿到 5.27 倍，原因在 perf 数据里：cycles 是 394.05 亿，instructions 是 441.06 亿，IPC 约 **1.12**，比 V6 单核的 1.42 低。多线程下 IPC 下降是正常的——多个线程共享 L3（32MB）和内存带宽，缓存争用和带宽饱和让每个线程的指令退休效率降低。

&emsp;&emsp;L1-dcache-loads 是 180.2 亿，L1-dcache-misses 是 25.74 亿，miss 率约 **14.3%**。这个数字比 V6 单核的 5.4% 高出不少。原因是 12 个线程同时运行时，每个线程的 `Ap`、`Bp` 缓冲区加起来约 786KB，加上 A、B、C 各 16MB 的共享数据，L1 和 L2 的容量被大量挤占。线程切换时缓存局部性被反复打断，miss 率自然上升。这也解释了为什么 V7 的加速比只有 5.27 倍而不是接近 12 倍——**内存带宽和缓存争用是多线程阶段的主要瓶颈**。

### 2.9 TLB优化
&emsp;&emsp;v2-v7是理论上性能优化能够用到的手段，如果期望在特定机器进行更进一步的极限优化，那么就先需要知道我们的瓶颈在哪儿。上一节，推测是在多线程内存conflict,但是实际上是不是，最好用perf工具具体看下CPU的执行情况。

&emsp;&emsp;我们先看下缓存数据来源和填充情况：

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

&emsp;&emsp;缓存填充数据给出了一个非常明确的结论：V7 的瓶颈既不在内存带宽，也不在 NUMA。在 10.78 亿次数据缓存填充中，**91% 来自本地 L2**，8.9% 来自 L3 或同 CCX 的其他 L2，只有 **0.7% 来自内存**，而远端内存和远端 CCX 缓存均为 0。这意味着绝大多数 L1 miss 都在 L2 就被满足，根本没有触及 DRAM。如果内存带宽是瓶颈，来自内存的填充比例应该在 10% 以上；如果 NUMA 分配有问题，`mem_io_remote` 和 `ext_cache_remote` 也不会是 0。5600X 是单 CCD、单 NUMA 节点，所以 NUMA 优化空间为零，`numactl --interleave` 这类操作不会带来任何收益。真正的问题出在 **L1 到 L2 之间的流量**：每个线程的打包缓冲区 `Ap`、`Bp` 加起来约 64KB，已经超过 L1d 的 32KB 容量，导致微内核读取打包数据时频繁触发 L1 miss，miss 之后又要向 L2 发请求。10.78 亿次 L2 填充对应之前测得的 25.74 亿次 L1 miss，说明约 42% 的 L1 miss 转化成了 L2 请求。这就是当前的核心瓶颈——**不是数据取不回来，而是 L1 装不下打包缓冲区，导致 L1 和 L2 之间的带宽被反复占用**。下一步优化的方向应该从“减少内存流量”转向“减少 L1 到 L2 的流量”：把 `KC` 从 128 降到 64，让 `Ap`、`Bp` 各自缩到 16KB、合计 32KB，刚好能放进 L1；同时把打包缓冲区按 64 字节对齐，减少跨缓存行访问。如果 `KC=64` 后 L1 miss 率明显下降、GFLOPS 提升，就说明这个方向是对的。

&emsp;&emsp;再看下L2请求和L2延迟：
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

&emsp;&emsp;L2 数据把 V7 的瓶颈定位得比缓存填充更精确：**L2 命中率约 85.4%，L2 miss 只有 1.05 亿，但 `l2_fill_pending.l2_fill_busy` 高达 79.4 亿周期，占 cycles 的 19.6%**。这说明瓶颈不是 L2 的命中率——L2 本身工作得很好，85% 的请求都能在 L2 内满足，miss 的那 1.05 亿次里绝大多数也在 L3 命中，真正回内存的只有 747 万次。问题出在 **L2 的请求吞吐量**：28.2 亿次 L2 请求把填充队列（MAB）打满了，`l2_fill_busy` 的 79.4 亿周期意味着 L2 有近 20% 的时间在处理未完成的填充请求，新请求要排队等待。

&emsp;&emsp;这个现象和 L1 数据对得上。L1 有 180 亿次 load，其中约 25.7 亿次 miss（14.3%），这些 miss 几乎全部打到了 L2，构成 28.2 亿次 L2 请求的主体。L1 命中率 84% 本身不算差，但 15.7% 的 miss 率在多线程下被 12 个线程放大，就把 L2 的请求队列压垮了。根因在打包缓冲区的尺寸：每个线程的 `Ap`、`Bp` 加起来约 64KB，超过 L1d 的 32KB，微内核每处理一个 6×8 的 C 块就要从 `Ap` 读 6×KC 个 float、从 `Bp` 读 KC×8 个 float，访存/计算比很高，导致 L1 频繁 miss。KC=128 时，每个微内核要读 1792 个 float 却只计算 48 个 C 元素，这些数据在 L1 里放不下，只能反复向 L2 请求。

&emsp;&emsp;再看TLB覆盖情况：

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

&emsp;&emsp;TLB 数据把 V7 的访存瓶颈又往下推了一层：**L1 DTLB miss 共 4379 万次，其中 3305 万次是 4K 页且 L2 TLB 也 miss（`tlb_reload_4k_l2_miss`），占总 miss 的 75.5%**；相比之下，2M 大页的 L2 TLB miss 只有 15 万次。更关键的是 `ls_tablewalker.dside` 是 3407 万次，说明几乎每一次 4K 页的 L2 TLB miss 都触发了一次硬件页表遍历。这个数字远高于普通计算负载的 TLB miss 水平，说明当前的工作集已经超出了 TLB 的覆盖范围。

&emsp;&emsp;再看FP 管道分配（判断 FMA 发射效率）：

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

&emsp;&emsp;FP 管道数据给出了一个相当明确的结论：**V7 的浮点单元已经接近饱和，FMA 发射效率不再是瓶颈**。`fp_ret_sse_avx_ops.mac_flops` 是 1887 亿次 MAC FLOPs，对应约 943 亿次 MAC 操作。2048³ 的理论 FLOP 是 171.8 亿次，12 线程累加后约 2062 亿次 FLOP，实测的 1887 亿次与此吻合，说明 FMA 指令几乎全部被退休，没有因为流水线停顿而被丢弃或反复执行。更关键的是 `fpu_pipe_assignment` 的分布：pipe0 是 73.2 亿 uop，pipe1 是 45.5 亿 uop，两者合计 118.7 亿，占总 FP uop 的 99.2%，而 pipe2 和 pipe3 分别只有 0.98 亿和 0.73 亿。在 Zen 3 上 FMA 只能在 pipe0 和 pipe1 上执行，pipe2/pipe3 负责其他浮点操作，所以这个分布说明几乎所有浮点操作都是 FMA，而且都压在了两个 FMA 管道上。用 cycles（34.4 亿）去除以 FP uop 总数（约 119.7 亿），得到平均每周期 3.48 个 FP uop，而 Zen 3 的 FP 单元理论峰值是每周期 4 个 128-bit uop，**实测利用率约 87%，已经接近饱和**。

&emsp;&emsp;最后看调度器停顿：
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

&emsp;&emsp;调度器停顿数据把 V7 的瓶颈定位到了**浮点寄存器文件**上。六项停顿事件里，`de_dis_dispatch_token_stalls1.fp_reg_file_rsrc_stall` 是 **48.9 亿周期**，占 cycles（391 亿）的 **12.5%**，是所有停顿事件里最大的一个。相比之下，`load_queue_rsrc_stall` 只有 7.68 亿（2.0%），`store_queue_rsrc_stall` 1.02 亿（0.3%），`fp_sch_rsrc_stall` 和两个整数调度器停顿都只有千万级，可以忽略。这个分布说明，V7 的前端不是被加载队列或存储队列堵住的，而是**浮点寄存器文件不够用**——微内核需要的 YMM 寄存器数量超过了物理寄存器文件的可用条目，调度器只能让指令排队等待。

---

&emsp;&emsp;perf 数据串起来看，V7 的瓶颈已经不在“数据取不回来”，而在“指令发不出去”和“地址翻不动页表”。缓存填充数据排除了内存带宽和 NUMA：91% 的填充来自本地 L2，只有 0.7% 来自内存，远端节点为 0。L2 数据进一步指出，L2 命中率 85.4%、真正回内存的只有 747 万次，但 `l2_fill_pending.l2_fill_busy` 高达 79.4 亿周期（占 19.6%），说明 L2 的填充队列被 28.2 亿次请求打满了。TLB 数据解释了填充队列为什么堵：L1 DTLB miss 4379 万次，其中 3305 万次是 4K 页且 L2 TLB 也 miss，触发 3407 万次页表遍历——TLB 覆盖不了 16MB 矩阵的工作集。FP 管道数据表明浮点单元本身已经跑到 87% 利用率，接近饱和；但调度器停顿数据显示，`fp_reg_file_rsrc_stall` 占了 cycles 的 12.5%，说明 FP 寄存器文件不够分配，微内核的 YMM 寄存器需求超过了硬件可用条目。

据此，下一步优化可以分三个方向推进。**第一，减少 L1 到 L2 的流量**：把 `KC` 从 128 降到 64，让每个线程的 `Ap`、`Bp` 从各 32KB 缩到各 16KB，合计 32KB 刚好放进 L1d，L1 miss 率有望从 14.3% 明显下降，`l2_request_g1.all_no_prefetch` 和 `l2_fill_busy` 也会同步回落。**第二，用 2MB 大页替代 4K 页**：通过 `madvise(MADV_HUGEPAGE)` 或 `mmap` + `MAP_HUGETLB` 把 A、B、C 和打包缓冲区放到大页上，16MB 矩阵只需 8 个 TLB 条目，L2 TLB miss 可以从 3305 万压到接近零，页表遍历次数大幅下降，L2 填充队列的堵塞也会缓解。**第三，降低 FP 寄存器压力**：把 NR 从 8 降到 4 改用 128-bit XMM，或者把 MR 从 6 降到 4，减少微内核同时占用的 YMM 累加器数量。Zen 3 的 256-bit FMA 本来就会拆成两个 128-bit uop，改用 128-bit 不会损失吞吐，反而能缓解寄存器重命名压力，`fp_reg_file_rsrc_stall` 有望从 48.9 亿降到 20 亿以下。这三条路分别对应 L1/L2 流量、TLB 覆盖、寄存器容量三个瓶颈，可以独立尝试，也可以用 `l2_fill_busy`、`tlb_reload_4k_l2_miss`、`fp_reg_file_rsrc_stall` 作为指标分别验证效果。如果三条都做到位，GFLOPS 有望从当前的 291~331 推到 350 以上；如果还想更高，就需要 AVX-512 或多路 CPU，单靠调优已经接近 Zen 3 单 CCD 的上限。

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

&emsp;&emsp;V8 的核心突破在于微内核从 4×8 改为 **4×16**，把独立的 FMA 链从 4 条增加到 8 条。旧版微内核每个 `p` 迭代只有 4 条独立 FMA，而 Zen 3 的 FMA 延迟是 4 周期、双发射，要填满流水线至少需要 8 条在途 FMA，旧版的天花板因此只有峰值的一半。新版用 8 个 YMM 累加器（`c00` 到 `c31`）支撑 8 条独立链，内核零溢出，14 µop/迭代恰好让 FMA 端口先饱和。仅此一步就把 2048³ 的 GFLOPS 从 341 推到 434。同时，代码去掉了旧版“把 A、B、C 整块拷贝到大页”的做法，改为直接写 C，K 方向首轮累加器清零（beta=0 语义），后续轮从 C 加载累加，省掉了 48MB 的拷贝开销；打包面板用 `thread_local` 复用，每个线程只分配一次。

&emsp;&emsp;第二层优化针对 **TLB**。打包面板最初用普通 `malloc`，4K 页下每个面板要扫过 64 个 L1 dTLB 条目，2048³ 下 dTLB miss 高达 5 亿次以上。改成 `MAP_HUGETLB` 的 2MB 大页后，面板的 TLB 覆盖从 64 个条目降到 1 个，`ls_l1_d_tlb_miss.all` 从之前的数亿降到 6032 万，`ls_tablewalker.dside` 降到 3047 万。这一步把 GFLOPS 从 434 推到 505。分块参数也在这一步定型：`MC=128, NC=96, KC=256`，Ap 面板 128×256×4 = 128KB，Bp 面板 256×96×4 = 96KB，合计 224KB 贴近 L2 的 512KB；C tile 128×96×4 = 48KB，pc 多轮之间重读基本命中 L1。最后把线程数从 12 逻辑核压到 6 物理核（SMT 检测），因为 FMA 受限时 SMT 兄弟只会争抢 FPU 端口和缓存，实测 6 物理核下 2048³ 反而比 12 逻辑核快，从 534 提到 576。

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

&emsp;&emsp;最终结果：2048³ 下 **29604181 ns、580.37 GFLOPS**，256³ 和 1024³ 分别是 526 和 576 GFLOPS，前两项超过 OpenBLAS，2048³ 达到 OpenBLAS 的 91%。perf 数据也印证了微内核的健康状态：`fpu_pipe_assignment.total0` 和 `total1` 分别是 181.0 亿和 182.8 亿，几乎完美 50/50，FP 管道每周期约 3.5 个 uop，利用率接近 Zen 3 的 4 uop/周期上限；`fp_reg_file_rsrc_stall` 49.7 亿，占 cycles 的 15.2%，是可接受的水平；L1 miss 率 20.2%，看起来偏高，但 `l2_fill_pending.l2_fill_busy` 只有 87.3 亿，说明 L1 miss 后大部分在 L2 快速命中，没有形成填充队列瓶颈；`mem_io_local` 只有 1143 万次，内存带宽完全不是问题。这组数据说明 V8 已经把单 CCD 的 FMA 吞吐压到了接近极限。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/cpu_gemm_v8_xxxxxxxxxxxxxxx_bench_result.png)

&emsp;&emsp;至此，从朴素实现到多线程 SIMD 微内核的完整路径已经走完。每一步都在解决前一步遗留的瓶颈：循环交换解决 B 的跨步访问，分块解决全局复用，打包解决微内核加载效率，寄存器分块解决 FMA 延迟，SIMD 提升单指令吞吐，多线程突破单核算力限制。硬件峰值和带宽决定了性能上界，而这些优化方法的作用，就是让实际性能不断逼近这一上界。

## 3 GEMM GPU优化

