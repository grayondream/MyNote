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

## 2 GEMM CPU优化

### 2.2 朴素实现

&emsp;&emsp;GEMM 的数学定义为 \(C[i,j]=\sum_{k=0}^{K-1}A[i,k]B[k,j]\)。最直接的实现就是三重循环，逐元素计算 C：

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
- 内层 k 循环访问 `B[k * N + j]` 时，地址步长为 N。每次 k 增加 1，B 的访问就跳到下一行，跨越 N 个元素。若 N 较大，一个缓存行中往往只有一个元素被用到，其余部分被浪费。同时，A 的第 i 行和 B 的第 j 列在 k 循环中被反复读取，但每次只使用一次，没有跨 (i,j) 的复用。整个计算过程中，A 被读取了 N 次，B 被读取了 M 次，总访存量约为 \(4MNK+4MNK+4MN\) 字节，而浮点运算量只有 \(2MNK\)。算术强度约为 \(2MNK/(8MNK+4MN)\approx 0.25\) FLOP/Byte，远低于 Roofline 转折点，因此性能受内存带宽限制。
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

### 2.4 分块

&emsp;&emsp;要突破全局复用的限制，必须让数据在更靠近计算单元的地方被多次复用。由于矩阵计算本身就具备局部独立性，可以通过分块局部计算最后再合并。把 C 划分为 \(MC\times NC\) 的块，A 划分为 \(MC\times KC\)，B 划分为 \(KC\times NC\)。对每个 C 块，遍历 K 维的 \(KC\) 块，将对应的 A、B 子块加载到缓存中，再在缓存内完成多次乘加。这样，A 子块的每一行和 B 子块的每一列都能在 L2/L3 中被多个 C 元素复用。

```c
void gemm_blocked(int M, int N, int K,
                  const float *A, const float *B, float *C,
                  int MC, int NC, int KC) {
    for (int jc = 0; jc < N; jc += NC) {
        int n = (jc + NC <= N) ? NC : N - jc;
        for (int pc = 0; pc < K; pc += KC) {
            int kc = (pc + KC <= K) ? KC : K - pc;
            for (int ic = 0; ic < M; ic += MC) {
                int mc = (ic + MC <= M) ? MC : M - ic;
                for (int i = ic; i < ic + mc; i++) {
                    for (int p = pc; p < pc + kc; p++) {
                        float a = A[i*K+p];
                        const float *Brow = B + p*N + jc;
                        float *Crow = C + i*N + jc;
                        for (int j = 0; j < n; j++)
                            Crow[j] += a * Brow[j];
                    }
                }
            }
        }
    }
}
```

&emsp;&emsp;这里的三层分块循环对应着不同的缓存级别：最外层 jc、pc、ic 控制 L3/L2 分块，保证 A、B 面板在 L2/L3 中复用；内层 i、p、j 则是分块内的计算。分块大小需要根据缓存容量选择，例如 \(MC=64\)、\(NC=64\)、\(KC=256\)。分块后，算术强度从仅按 DRAM 计算的 \(n/6\) 提升到按缓存容量计算的水平，性能通常能再提升 2–5 倍。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/deepseek_svg_20261001_b59684.svg)

&emsp;&emsp;但分块后的内层循环仍然每次从 A 取一个标量、从 B 取一行，数据加载频率较高。如果把 A、B 的子块预先复制到连续缓冲区中，就能让内层循环以更紧凑的方式读取。

### 2.5 打包

&emsp;&emsp;打包是把原本行主序或列主序的子块复制到连续缓冲区中，使得内层循环可以按顺序读取。对 A，把 \(MC\times KC\) 的子块按行打包；对 B，把 \(KC\times NC\) 的子块按列打包成连续面板。这样，内层每次都能取到连续的内存，消除了大步长访问，降低了 TLB 压力，也让硬件预取器能更准确地识别访问模式。

```c
void gemm_packed(int M, int N, int K,
                 const float *A, const float *B, float *C,
                 int MC, int NC, int KC) {
    float *Ap = malloc((size_t)MC*KC*sizeof(float));
    float *Bp = malloc((size_t)KC*NC*sizeof(float));

    for (int jc = 0; jc < N; jc += NC) {
        int n = (jc + NC <= N) ? NC : N - jc;
        for (int pc = 0; pc < K; pc += KC) {
            int kc = (pc + KC <= K) ? KC : K - pc;

            /* 打包 B 的 KC×NC 子块为连续面板 */
            for (int p = 0; p < kc; p++)
                for (int j = 0; j < n; j++)
                    Bp[p*n + j] = B[(pc+p)*N + jc + j];

            for (int ic = 0; ic < M; ic += MC) {
                int mc = (ic + MC <= M) ? MC : M - ic;

                /* 打包 A 的 MC×KC 子块为连续面板 */
                for (int i = 0; i < mc; i++)
                    for (int p = 0; p < kc; p++)
                        Ap[i*kc + p] = A[(ic+i)*K + pc + p];

                for (int i = 0; i < mc; i++) {
                    for (int p = 0; p < kc; p++) {
                        float a = Ap[i*kc + p];
                        const float *Brow = Bp + p*n;
                        float *Crow = C + (ic+i)*N + jc;
                        for (int j = 0; j < n; j++)
                            Crow[j] += a * Brow[j];
                    }
                }
            }
        }
    }
    free(Ap); free(Bp);
}
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/deepseek_svg_20261001_5fc5d7.svg)

&emsp;&emsp;打包后，A 和 B 的子块在内存中连续排列，内层循环的加载模式变得非常规则。此时，代码的访存效率已经接近最优，但内层仍然每次只做一个标量与一行的乘加，FMA 的发射效率还有提升空间。

### 2.6 寄存器分块与微内核

&emsp;&emsp;经过分块和打包，A、B 子块已能以连续方式读取，但内层循环每次仍然只更新 C 的一行，累加器数量不足，FMA 依赖链依然存在。解决办法是取 \(MR\times NR\) 的小块，例如 \(MR=8\)、\(NR=6\)，用 48 个寄存器保存 C 的累加值。微内核沿 K 维循环，每次加载 A 的 \(MR\) 个元素和 B 的 \(NR\) 个元素，执行 \(MR\times NR\) 次 FMA。这 48 个累加器彼此独立，FMA 之间没有依赖链，可以持续填满流水线。

```c
#define MR 8
#define NR 6

void micro_kernel(int kc, int n,
                  const float *Ap, const float *Bp,
                  float *C, int ldc) {
    for (int i = 0; i < MR; i++) {
        float c[NR];
        for (int j = 0; j < NR; j++) c[j] = C[i*ldc + j];
        for (int p = 0; p < kc; p++) {
            float a = Ap[i*kc + p];
            const float *Brow = Bp + p*n;
            for (int j = 0; j < NR; j++)
                c[j] += a * Brow[j];
        }
        for (int j = 0; j < NR; j++) C[i*ldc + j] = c[j];
    }
}
```

&emsp;&emsp;调用微内核的主循环如下：

```c
void gemm_micro(int M, int N, int K,
                const float *A, const float *B, float *C,
                int MC, int NC, int KC) {
    float *Ap = malloc((size_t)MC*KC*sizeof(float));
    float *Bp = malloc((size_t)KC*NC*sizeof(float));

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
                        micro_kernel(kc, n,
                                     Ap + ir*kc,
                                     Bp + jr,
                                     C + (ic+ir)*N + (jc+jr),
                                     N);

                /* 处理 MR 和 NR 的边角 */
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
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/deepseek_svg_20261001_465a90.svg)

&emsp;&emsp;微内核通过 48 个独立累加器填满了 FMA 流水线。若 FMA 延迟为 4 周期、吞吐为 2 条/周期，理论上只需要 8 条独立 FMA，而 48 个累加器提供的并行度远远超过这一要求。此时单核性能已经接近标量峰值，但每个 FMA 仍然只处理一个浮点数。

### 2.7 SIMD 向量化

&emsp;&emsp;现代 x86 CPU 支持 AVX2 或 AVX-512，单条 FMA 指令可以作用在 8 个或 16 个单精度浮点数上。把微内核中的 NR 设为 SIMD 宽度的整数倍，用 `_mm256_fmadd_ps` 等内在函数替代标量 FMA，可以让吞吐成倍提升。以下代码使用 AVX2，编译时需加 `-mavx2 -mfma`。

```c
#include <immintrin.h>

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
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/deepseek_svg_20261001_b692d2.svg)

&emsp;&emsp;调用 AVX2 微内核的主循环与 2.6 类似，只需把 `micro_kernel` 替换为 `micro_kernel_avx2`，并把 NR 改为 8。AVX2 版本通常能在标量微内核基础上再提升 3–6 倍。至此单核性能已接近峰值，剩下的问题是单个核心的算力有限，需要把工作分摊到多个核心上。

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

&emsp;&emsp;至此，从朴素实现到多线程 SIMD 微内核的完整路径已经走完。每一步都在解决前一步遗留的瓶颈：循环交换解决 B 的跨步访问，分块解决全局复用，打包解决微内核加载效率，寄存器分块解决 FMA 延迟，SIMD 提升单指令吞吐，多线程突破单核算力限制。硬件峰值和带宽决定了性能上界，而这些优化方法的作用，就是让实际性能不断逼近这一上界。

## 3 GEMM GPU优化
