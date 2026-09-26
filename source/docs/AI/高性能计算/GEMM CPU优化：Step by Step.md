# GEMM CPU SIMD优化：Step by Step

## 1 前言

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

## 0.5 什么是SIMD

SIMD = **S**ingle **I**nstruction **M**ultiple **D**ata，单指令多数据。普通指令一次处理1个数据：

```cpp
c = a + b;  // 1次加法，1个数据
```

SIMD指令一次处理多个数据（打包成向量）：

```cpp
// 假设有一个4宽的向量加法
[ c0 c1 c2 c3 ] = [ a0 a1 a2 a3 ] + [ b0 b1 b2 b3 ];  // 1条指令，4个数据
```

## 0.6 x86的SIMD发展史

| 指令集 | 宽度 | float个数 | double个数 | 寄存器数量 |
|--------|------|-----------|------------|-----------|
| SSE | 128位 | 4 | 2 | 16 (XMM) |
| AVX | 256位 | 8 | 4 | 16 (YMM) |
| AVX2 | 256位 | 8 | 4 | 16 (YMM) |
| AVX-512 | 512位 | 16 | 8 | 32 (ZMM) |

**AVX2**是我们重点关注的：256位向量，8个float同时运算，16个物理寄存器。AVX-512则把宽度翻倍到512位，寄存器数量也翻倍到32个。

## 0.7 什么是FMA

FMA = **F**used **M**ultiply-**A**dd，融合乘加。普通做法是"先乘后加"两条指令：

```cpp
// 普通：2条指令，中间结果需要一次舍入
t = a * b;
c = c + t;
```

FMA是**一条指令完成乘加**，而且中间结果不单独舍入（更精确）：

```cpp
c = fma(a, b, c);  // 1条指令，计算 a*b+c
```

**这为什么重要**：在GEMM中，核心操作就是"累加"（`sum += a*b`）。用FMA，每个乘加只用一条指令，指令数减半，而且精度更高。现代CPU（Haswell 2013年之后）都支持FMA。

## 0.8 缓存层次

CPU访问内存的速度远比它的计算速度慢。为了弥补，CPU内部有多级缓存：

```
寄存器 (Registers)     : ~几十个,  访问延迟 <1周期, 容量几百字节
L1 缓存 (Level 1)      : 每核私有,  延迟 ~4周期,   容量 32KB
L2 缓存 (Level 2)      : 每核私有,  延迟 ~12周期,  容量 256KB~1MB
L3 缓存 (Level 3)      : 多核共享,  延迟 ~40周期,  容量 8~32MB
主存 (DRAM)            : 全部共享,  延迟 ~200周期, 容量 GB级别
```

**一个直观的比喻**：寄存器是你手边的笔（最快），L1是你桌上的书，L2是你书架上的一层，L3是书架的其他层，主存是图书馆。你要用一本书，图书馆的书要先搬到书架上，再搬到桌上，最后拿在手里才能读。搬运需要时间，所以我们要尽量把数据放在**靠近CPU**的地方反复使用。

## 0.9 缓存行

缓存不是按字节读取的，而是按**缓存行（cache line）**为单位，通常是**64字节**。这意味着即使你只读1个float（4字节），硬件也会把周围64字节一次性搬进缓存。这是好事（如果后续也要用周围的元素）还是坏事（如果周围元素用不到）取决于访问模式。

## 0.10 什么是算术强度

算术强度 = **浮点运算次数 / 内存访问字节数**。

如果算术强度低，意味着你访问内存多但计算少，此时CPU会"饿着"——等待数据从内存来，计算单元空闲。反之算术强度高，计算单元能持续工作。

GEMM的重要特点：如果做得对，**算术强度可以非常高**（大量数据重用），所以GEMM是"计算密集型"而非"内存密集型"。

## 0.11 什么是微内核（Microkernel）

微内核是GEMM的最内层代码块——它执行大部分实际计算。在优化良好的GEMM中，**95%以上的时间花在微内核里**。所以微内核设计决定了性能上限。后续章节的大部分精力都用于设计、优化微内核。

---

# 第1章 朴素实现：从数学定义直接翻译

## 1.1 代码

```cpp
#include <cstddef>

void gemm_naive(int M, int N, int K,
                const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i) {         // 遍历C的每一行
        for (int j = 0; j < N; ++j) {     // 遍历C的每一列
            float sum = 0.0f;              // C[i][j]的累加器
            for (int k = 0; k < K; ++k) { // 点积
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}
```

## 1.2 逐行解释

- `A[i*K + k]`：行主序下 `A[i][k]` 的线性地址。K是A的行距。
- `B[k*N + j]`：行主序下 `B[k][j]` 的线性地址。N是B的行距。
- `sum`：点积累加器。
- 三层循环顺序是 i→j→k。最内层是k，意味着**对C的每个元素，我们扫描一遍A的一行和B的一列**。

## 1.3 为什么这么慢：三个硬件问题

### 问题1：标量执行，SIMD硬件闲置

每个 `sum += A * B` 是标量操作，一次只处理1个float。而你的CPU有AVX2单元，一次能处理8个float。**你只用了SIMD能力的1/8**。

### 问题2：B的访问跨步巨大

最内层循环中，`B[k*N + j]` 随着k增加，地址增加 N 个float。假设N=4096，那么每次k增加，地址跳了 4096×4 = 16KB。

缓存行是64字节（16个float），意味着每次读B的1个float，硬件搬来64字节，但你**只用了其中1/16**（4字节/64字节 = 6.25%的缓存行利用率）。同时，16KB的跳跃通常远超L1容量，所以几乎每次B访问都**从L2/L3甚至主存读取**。

### 问题3：算术强度极低

对C的每个元素，我们做了 2K 次浮点运算（K次乘、K次加），但访问了 2K+1 次内存元素（K个A、K个B、1个C）。**算术强度 ≈ 1 FLOP/元素访问**，远低于缓存能高效供给的水平。

## 1.4 性能

在 Haswell 2.0 GHz 双核上，`4096×4096×4096` 的朴素实现大约需要 **60~100 秒**，只有 **1~3 GFLOPS**，不到理论峰值（128 GFLOPS）的 2%。

---

# 第2章 第一个优化：循环重排

## 2.1 思路：把k循环移出去

朴素实现最内层是k，导致B的访问跨步大。我们换一种循环顺序：i→k→j。这样最内层是j，B和C的访问都变成连续的。

```cpp
void gemm_reorder(int M, int N, int K,
                  const float* A, const float* B, float* C) {
    // 先把C清零（因为后面用+=）
    for (int i = 0; i < M*N; ++i) C[i] = 0.0f;

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a = A[i*K + k];         // 标量：A的一个元素
            for (int j = 0; j < N; ++j) {
                C[i*N + j] += a * B[k*N + j];
            }
        }
    }
}
```

## 2.2 硬件分析

现在最内层是j，`B[k*N + j]` 随着j增加而**连续**增加地址。`C[i*N + j]` 也连续。硬件预取器（prefetcher）能识别这种顺序访问，提前把数据搬到缓存。

但这个版本仍是标量。我们接着向量化它。

## 2.3 性能

顺序访问让缓存效率大幅提高，但因为还是标量，性能约5~10 GFLOPS。

---

# 第3章 引入SIMD：AVX2内建函数

## 3.1 什么是"内建函数"（Intrinsic）

C++编译器提供了一组特殊的函数，可以直接对应到CPU的SIMD指令。它们看起来像普通函数，但实际会编译成一条SIMD指令。

```cpp
#include <immintrin.h>  // AVX/AVX2头文件
```

常见的AVX2内建函数（FP32，256位=8个float）：

| 内建函数 | 含义 | 对应指令 |
|----------|------|----------|
| `_mm256_loadu_ps(p)` | 从地址p加载8个float到YMM寄存器 | vmovups |
| `_mm256_storeu_ps(p, v)` | 把YMM寄存器的8个float存到地址p | vmovups |
| `_mm256_set1_ps(x)` | 用标量x填充8个float | vbroadcastss |
| `_mm256_broadcast_ss(p)` | 从地址p读1个float，广播成8个 | vbroadcastss |
| `_mm256_add_ps(a, b)` | 逐元素相加 | vaddps |
| `_mm256_mul_ps(a, b)` | 逐元素相乘 | vmulps |
| `_mm256_fmadd_ps(a, b, c)` | 计算 a*b+c（逐元素） | vfmadd |

**命名规律**：`_mm256_` 表示256位；`ps` = packed single（打包的单精度浮点）；`u` = unaligned（非对齐，允许地址不是32字节倍数）。

## 3.2 类型

```cpp
__m256  // 表示一个256位YMM寄存器，装着8个float
```

## 3.3 向量化的简单GEMM

```cpp
void gemm_avx2_simple(int M, int N, int K,
                      const float* A, const float* B, float* C) {
    // 清零C
    for (int i = 0; i < M*N; ++i) C[i] = 0.0f;

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            __m256 a = _mm256_set1_ps(A[i*K + k]);  // 广播A的1个元素到8个
            for (int j = 0; j < N; j += 8) {         // 每次处理8个j
                __m256 b = _mm256_loadu_ps(&B[k*N + j]);
                __m256 c = _mm256_loadu_ps(&C[i*N + j]);
                c = _mm256_fmadd_ps(a, b, c);        // c = a*b+c，一次8个
                _mm256_storeu_ps(&C[i*N + j], c);
            }
        }
    }
}
```

## 3.4 逐行解释

- `_mm256_set1_ps(A[i*K + k])`：把A的一个标量复制成8个相同的float，放到YMM寄存器里。为什么？因为我们要让这1个A元素与B的8个元素分别相乘。
- `_mm256_loadu_ps(&B[k*N + j])`：从B读8个连续float。
- `_mm256_loadu_ps(&C[i*N + j])`：读C的8个元素（上次的累加结果）。
- `_mm256_fmadd_ps(a, b, c)`：对8个位置同时执行 `c = a*b+c`。
- `_mm256_storeu_ps(&C[i*N + j], c)`：把结果写回。

## 3.5 硬件分析

- FMA从"1个/指令"变成"8个/指令"，SIMD能力被利用。
- B和C都是连续访问，硬件预取器有效。
- **但问题依然存在**：每个C元素在整个K循环中被读写K次（每次都要 load/modify/store），L1带宽压力巨大。而且A的每个元素只被用了1次（作为标量广播），算术强度仍然不高。

## 3.6 性能

约10~20 GFLOPS。

---

# 第4章 关键突破：微内核与寄存器阻塞

## 4.1 核心思想

前面版本的问题：C的每个元素被反复读写。能不能**让C的元素一直待在寄存器里，直到整个K循环结束才写回内存**？

答案是可以。这就是**寄存器阻塞（register blocking）**。

## 4.2 微内核的规模选择

AVX2有**16个YMM寄存器**。我们要合理分配：

- 一部分用来装C的累加器（越多越好，但受限于总寄存器数）
- 一部分用来临时加载A和B

考虑 $m_r \times n_r$ 的微内核（处理C的一个小方块）：

- $m_r$ = 几行
- $n_r$ = 几列

对于FP32，1个YMM装8个float。若选择 $n_r = 16$，需要2个YMM装一行C。若 $m_r = 6$，则C需要 $6 \times 2 = 12$ 个YMM。剩下4个YMM用于加载。

这是著名的 **6×16 微内核**。

## 4.3 数据打包（Packing）— 为什么需要

微内核要高效加载A和B，但原矩阵A的列访问是跨步的（A[i][k]的相邻k元素，i不同则地址差K）。为了让微内核中所有加载都是连续的，我们**预先重排数据**，把A的一小列和B的一小行打包成连续缓冲区。

微内核期望的数据布局：

- **A面板**：按"列优先"排列，每列6个连续float（即 A[0..5][k] 连续存放）。
- **B面板**：按行排列，每行16个连续float（即 B[k][0..15] 连续存放）。

## 4.4 打包代码

```cpp
#include <algorithm>
#include <cstdlib>

// 把A的一个mc×kc子块打包成"每列6个float连续"的格式
void pack_A(int mc, int kc, const float* A, int lda, float* A_pack) {
    const int MR = 6;
    for (int i = 0; i < mc; i += MR) {
        int m = std::min(MR, mc - i);  // 边界：最后一块可能不足6行
        for (int k = 0; k < kc; ++k) {
            // 前m行是真实数据
            for (int ii = 0; ii < m; ++ii)
                A_pack[i*kc + k*MR + ii] = A[(i+ii)*lda + k];
            // 不足的部分填0，避免微内核里做边界判断
            for (int ii = m; ii < MR; ++ii)
                A_pack[i*kc + k*MR + ii] = 0.0f;
        }
    }
}

// 把B的一个kc×nc子块打包成"每行16个float连续"的格式
void pack_B(int kc, int nc, const float* B, int ldb, float* B_pack) {
    const int NR = 16;
    for (int j = 0; j < nc; j += NR) {
        int n = std::min(NR, nc - j);
        for (int k = 0; k < kc; ++k) {
            for (int jj = 0; jj < n; ++jj)
                B_pack[j*kc + k*NR + jj] = B[k*ldb + j + jj];
            for (int jj = n; jj < NR; ++jj)
                B_pack[j*kc + k*NR + jj] = 0.0f;
        }
    }
}
```

**关键解释**：

- 打包后，`A_pack` 的布局是：第0列（6个float）、第1列、第2列……。微内核访问"第k列的6个float"，地址就是 `A_pack + k*6`，连续访问。
- `B_pack` 的布局是：第0行（16个float）、第1行、第2行……。微内核访问"第k行的16个float"，地址就是 `B_pack + k*16`，连续访问。
- 边界填0是**经典技巧**：与其在微内核里判断是否越界（慢），不如填0让计算照常进行（0乘任何数为0，不影响结果）。

## 4.5 微内核代码

```cpp
// C[0..5][0..15] += A[0..5][0..kc-1] * B[0..kc-1][0..15]
// A_pack: 每列6个float连续；B_pack: 每行16个float连续
// C: 行主序，ldc是C的leading dimension（行距）
void micro_6x16(int kc,
                const float* __restrict__ A_pack,
                const float* __restrict__ B_pack,
                float* __restrict__ C, int ldc) {
    // 1. 把C的12个向量加载到寄存器（初始化累加器）
    __m256 c00 = _mm256_loadu_ps(C + 0*ldc + 0);
    __m256 c01 = _mm256_loadu_ps(C + 0*ldc + 8);
    __m256 c10 = _mm256_loadu_ps(C + 1*ldc + 0);
    __m256 c11 = _mm256_loadu_ps(C + 1*ldc + 8);
    __m256 c20 = _mm256_loadu_ps(C + 2*ldc + 0);
    __m256 c21 = _mm256_loadu_ps(C + 2*ldc + 8);
    __m256 c30 = _mm256_loadu_ps(C + 3*ldc + 0);
    __m256 c31 = _mm256_loadu_ps(C + 3*ldc + 8);
    __m256 c40 = _mm256_loadu_ps(C + 4*ldc + 0);
    __m256 c41 = _mm256_loadu_ps(C + 4*ldc + 8);
    __m256 c50 = _mm256_loadu_ps(C + 5*ldc + 0);
    __m256 c51 = _mm256_loadu_ps(C + 5*ldc + 8);

    // 2. K方向循环，执行外积累加
    for (int k = 0; k < kc; ++k) {
        // 加载B的第k行：16个float = 2个YMM
        __m256 b0 = _mm256_loadu_ps(B_pack + k*16 + 0);
        __m256 b1 = _mm256_loadu_ps(B_pack + k*16 + 8);

        // 加载A的第k列：6个标量，每个广播成8个float
        __m256 a0 = _mm256_broadcast_ss(A_pack + k*6 + 0);
        __m256 a1 = _mm256_broadcast_ss(A_pack + k*6 + 1);
        __m256 a2 = _mm256_broadcast_ss(A_pack + k*6 + 2);
        __m256 a3 = _mm256_broadcast_ss(A_pack + k*6 + 3);
        __m256 a4 = _mm256_broadcast_ss(A_pack + k*6 + 4);
        __m256 a5 = _mm256_broadcast_ss(A_pack + k*6 + 5);

        // 12条FMA：C[i][j] += A[i] * B[j]
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }

    // 3. 把累加器写回C
    _mm256_storeu_ps(C + 0*ldc + 0, c00);
    _mm256_storeu_ps(C + 0*ldc + 8, c01);
    _mm256_storeu_ps(C + 1*ldc + 0, c10);
    _mm256_storeu_ps(C + 1*ldc + 8, c11);
    _mm256_storeu_ps(C + 2*ldc + 0, c20);
    _mm256_storeu_ps(C + 2*ldc + 8, c21);
    _mm256_storeu_ps(C + 3*ldc + 0, c30);
    _mm256_storeu_ps(C + 3*ldc + 8, c31);
    _mm256_storeu_ps(C + 4*ldc + 0, c40);
    _mm256_storeu_ps(C + 4*ldc + 8, c41);
    _mm256_storeu_ps(C + 5*ldc + 0, c50);
    _mm256_storeu_ps(C + 5*ldc + 8, c51);
}
```

## 4.6 微内核的硬件分析

### 4.6.1 寄存器分配

- 12个YMM装C累加器。
- 2个YMM临时装B。
- 理论上需要6个YMM装A，但因为A是广播的结果，实际上编译器可能会复用寄存器（a0到a5可能被优化掉，立即数广播直接进FMA）。所以实际用到的物理寄存器在12~16个之间。

### 4.6.2 延迟隐藏

12条FMA累加链（c00, c01, c10, c11, ...），每一条链内部是依赖的（c00依赖上一次的c00）。12 ≥ 8（隐藏4周期FMA延迟所需的最少独立链），**流水线被充分填满**。

### 4.6.3 加载端口分析

每轮k循环：

- 2条B加载（256位）
- 6条A广播（每条对应1次256位加载，虽然只用了1个float）

总共8次"加载类"操作。现代CPU通常有2个加载端口，每周期2次加载。8次加载需要4周期。

同一轮中有12条FMA，现代CPU每周期可发射2条FMA，需要6周期。

**结论**：加载端口比FMA端口更空闲（4 < 6周期），FMA是瓶颈。这正是我们想要的——微内核应该让FMA满负荷。

### 4.6.4 数据重用

- 每个A标量（广播前是1个float）用于2条FMA（与b0和b1各乘一次）。
- 每个B向量（8个float）用于6条FMA（分别与a0..a5相乘）。

在k循环的每次迭代中，加载14个float（6个A + 16个B，其中16个B是2个YMM），做12×8 = 96次乘加 = 192 FLOP。**算术强度 = 192 / (14×4) ≈ 3.4 FLOP/字节**，已经很高。

## 4.7 性能

微内核本身的计算效率已经接近峰值。但如果直接对整个大矩阵使用它（不做分块），会因缓存不够大而性能受限。**微内核只是第一步，还需要缓存分块**。

---

# 第5章 缓存分块：让数据在正确的层次里

## 5.1 问题：微内核需要的数据从哪来

微内核一次处理 $6 \times 16$ 的C块和对应的A、B数据。当矩阵很大时，A和B的数据无法全部装进缓存。每次进入微内核，A和B都要从内存/L2/L3搬进来。

解决方案：**分块（blocking / tiling）**。把大矩阵切成小块，让每一级小块都能装进对应的缓存层，然后在小的块里反复调用微内核。

## 5.2 三级分块

我们从大到小：

**第一级：L3分块**（$n_c$ 方向）

- 每次选择 $k_c \times n_c$ 的B子面板，使其能装入L3缓存。
- 这样后续的A面板分块可以反复使用这个B子面板。

**第二级：L2分块**（$m_c$ 方向）

- 每次选择 $m_c \times k_c$ 的A子面板，使其能装入L2缓存。
- 这样A子面板在微内核调用之间反复使用。

**第三级：L1/寄存器分块**（微内核）

- 微内核内部 $m_r \times n_r$ 的C块常驻寄存器。
- A的 $m_r \times k_c$ 面板和B的 $k_c \times n_r$ 微面板从L1读取。

## 5.3 参数选择依据

以FP32、L2=256KB、L3=8MB为例：

- $k_c \times n_c \times 4$ 字节 ≤ L3容量的一部分，例如 $256 \times 4096 \times 4 = 4$ MB。
- $m_c \times k_c \times 4$ 字节 ≤ L2容量的一部分，例如 $192 \times 256 \times 4 = 196$ KB，接近256KB。
- 微内核 $6 \times 16$ 已经在寄存器里。

参数不是拍脑袋，而是由缓存大小反推。

## 5.4 完整分块GEMM代码

```cpp
void gemm_avx2(int M, int N, int K,
               const float* A, const float* B, float* C) {
    const int MC = 192;    // L2分块大小（A的行数）
    const int KC = 256;    // K方向分块大小
    const int NC = 4096;   // L3分块大小（B的列数）
    const int MR = 6;      // 微内核行数
    const int NR = 16;     // 微内核列数

    // 打包缓冲区（对齐到64字节，即缓存行大小）
    float* A_pack = (float*)aligned_alloc(64, MC * KC * sizeof(float));
    float* B_pack = (float*)aligned_alloc(64, KC * NC * sizeof(float));

    // 先把C清零（微内核是累加模式）
    for (int i = 0; i < M*N; ++i) C[i] = 0.0f;

    // ---- 最外层：N方向分块（jc）----
    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(NC, N - jc);

        // ---- 中间层：K方向分块（pc）----
        for (int pc = 0; pc < K; pc += KC) {
            int kc = std::min(KC, K - pc);

            // 打包B的子面板：kc × nc
            pack_B(kc, nc, B + pc*N + jc, N, B_pack);

            // ---- 内层：M方向分块（ic）----
            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(MC, M - ic);

                // 打包A的子面板：mc × kc
                pack_A(mc, kc, A + ic*K + pc, K, A_pack);

                // ---- 微内核循环：遍历C子块（mc × nc）----
                for (int jr = 0; jr < nc; jr += NR) {
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = std::min(MR, mc - ir);
                        int nr = std::min(NR, nc - jr);

                        if (mr == MR && nr == NR) {
                            // 满块：调用快速微内核
                            micro_6x16(kc,
                                A_pack + ir*kc,           // A子面板
                                B_pack + jr*kc,           // B子面板
                                C + (ic+ir)*N + (jc+jr),  // C子块
                                N);                        // C的leading dimension
                        } else {
                            // 边界块：回退到标量（教学用，实际应写小微内核）
                            for (int i = 0; i < mr; ++i)
                                for (int j = 0; j < nr; ++j) {
                                    float sum = C[(ic+ir+i)*N + (jc+jr+j)];
                                    for (int k = 0; k < kc; ++k)
                                        sum += A[(ic+ir+i)*K + (pc+k)] *
                                               B[(pc+k)*N + (jc+jr+j)];
                                    C[(ic+ir+i)*N + (jc+jr+j)] = sum;
                                }
                        }
                    }
                }
            }
        }
    }

    free(A_pack);
    free(B_pack);
}
```

## 5.5 分块的硬件解释

- **最外层jc循环**：B的子面板每次被pack一次，然后被内层的所有ic和k循环使用。这就是"B面板在L3中驻留"。
- **中间pc循环**：A的子面板在每次ic块开始时pack一次，被此后的所有jr、ir循环使用。这是"A面板在L2中驻留"。
- **内层微内核**：C的6×16块常驻寄存器。

分块不是"分完就完事"，而是让**每一级数据在被逐出到更慢的层次之前，被尽可能多次地使用**。

## 5.6 打包的性能意义

- **TLB压力**：原始矩阵的相邻行地址差很大（N×4字节），会跨页。打包后的缓冲区是连续的小块，页表项少。
- **单位步长**：微内核内所有A和B的加载都是连续的（或广播式的），可以使用对齐/非对齐加载指令直接操作。

## 5.7 性能

加入分块和打包后，性能从10~20 GFLOPS跃升到 **50~65 GFLOPS**，接近峰值的50%。

---

# 第6章 微内核的进一步调优

## 6.1 循环展开

目前微内核的k循环每次处理1步k。我们可以一次处理4步k来充分利用流水线：

```cpp
for (int k = 0; k < kc; k += 4) {
    // 预加载4步的数据
    __m256 b0_0 = _mm256_loadu_ps(B_pack + (k+0)*16 + 0);
    __m256 b0_1 = _mm256_loadu_ps(B_pack + (k+0)*16 + 8);
    __m256 b1_0 = _mm256_loadu_ps(B_pack + (k+1)*16 + 0);
    __m256 b1_1 = _mm256_loadu_ps(B_pack + (k+1)*16 + 8);
    __m256 b2_0 = _mm256_loadu_ps(B_pack + (k+2)*16 + 0);
    __m256 b2_1 = _mm256_loadu_ps(B_pack + (k+2)*16 + 8);
    __m256 b3_0 = _mm256_loadu_ps(B_pack + (k+3)*16 + 0);
    __m256 b3_1 = _mm256_loadu_ps(B_pack + (k+3)*16 + 8);

    // 每步k做12条FMA
    // ...
}
```

展开的好处：

- 减少循环控制开销（递增、比较、跳转）。
- 给编译器更多**指令级并行（ILP）**空间重排指令。
- 让加载和FMA交错进行，减少流水线气泡。

## 6.2 数据对齐

`_mm256_loadu_ps` 是"非对齐"加载，允许任意地址。但如果我们保证地址**32字节对齐**（YMM宽度），可以使用 `_mm256_load_ps`（对齐加载），在某些CPU上延迟更低。

做法：在打包时把缓冲区起始地址对齐到64字节（`aligned_alloc(64, ...)`），并确保微内核访问偏移是32的倍数。

对于 $n_r = 16$，B_pack每行16个float = 64字节，天然对齐。对于 $m_r = 6$，A_pack每列6个float = 24字节，不是32的倍数。所以A的加载只能非对齐。这是 $m_r=6$ 的一个小缺点。

## 6.3 软件预取

微内核的访问模式虽然是连续的，但在微面板之间会有跳跃。我们可以在k循环中提前预取后面的数据：

```cpp
_mm_prefetch((const char*)(B_pack + (k+8)*16), _MM_HINT_T0);  // 预取到L1
```

**注意**：如果硬件预取器已经能很好地处理，软件预取可能适得其反（产生冗余请求）。一般在打包这种不规则访问中才用。

## 6.4 性能

经过这些调优，性能可达到 **70~85 GFLOPS**。

---

# 第7章 AVX-512微内核

## 7.1 AVX-512的硬件升级

- 向量宽度：512位（16个float）
- 寄存器数量：32个（zmm0~zmm31）

32个寄存器允许**更大的微内核**。典型设计：**16×16** 微内核，用16个ZMM装C，剩下16个ZMM用于A/B加载和预取。

## 7.2 代码

```cpp
#ifdef __AVX512F__
#include <immintrin.h>

// 16×16 微内核
void micro_16x16_avx512(int kc,
                        const float* __restrict__ A_pack,
                        const float* __restrict__ B_pack,
                        float* __restrict__ C, int ldc) {
    __m512 c[16];
    // 加载C的16行，每行16个float（1个ZMM）
    #pragma unroll
    for (int i = 0; i < 16; ++i)
        c[i] = _mm512_loadu_ps(C + i*ldc);

    for (int k = 0; k < kc; ++k) {
        // 加载B的第k行：16个float
        __m512 b = _mm512_loadu_ps(B_pack + k*16);
        // 对A的16个元素，每个广播后与b做FMA
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            __m512 a = _mm512_set1_ps(A_pack[k*16 + i]);
            c[i] = _mm512_fmadd_ps(a, b, c[i]);
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
        _mm512_storeu_ps(C + i*ldc, c[i]);
}
#endif
```

## 7.3 硬件分析

- 16条独立FMA链（c[0]到c[15]），远超延迟隐藏需求。
- 每步k：1次B加载 + 16次A广播 + 16条FMA。
- **潜在瓶颈**：A的16次广播。广播指令占用加载端口。16次广播 + 1次加载 = 17次加载类操作，可能需要8.5周期。而16条FMA只需8周期发射。加载可能成为瓶颈。
- 优化方法：用 `_mm512_broadcastss_ps` 直接从内存广播（合并加载和广播），或者从A_pack加载整行（16个float就是1个ZMM），然后用permute来广播每个元素（但这样更慢）。实际高性能库会用特殊技巧。

## 7.4 频率问题

AVX-512在某些CPU（如早期Skylake-X）上会触发降频（AVX offset），可能降低200~400 MHz。但每次处理的数据翻倍，通常净收益仍然正。需要实测。

## 7.5 编译

```bash
g++ -O3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm
```

---

# 第8章 多线程

## 8.1 并行化策略

GEMM的各层循环都可以并行，但最佳并行位置是**外层jc循环**（N方向分块）。原因：

- 各线程处理不同的C列块，互不干扰。
- 共享的A面板在L3中被多线程复用，缓存效率高。
- B面板不共享，各线程独立打包。

## 8.2 OpenMP版本

```cpp
#include <omp.h>

void gemm_avx2_omp(int M, int N, int K,
                   const float* A, const float* B, float* C) {
    const int MC = 192, KC = 256, NC = 4096;
    const int MR = 6, NR = 16;

    // 并行清零C
    #pragma omp parallel for
    for (int i = 0; i < M*N; ++i) C[i] = 0.0f;

    // 并行区域：每个线程独立分配打包缓冲区
    #pragma omp parallel
    {
        float* A_pack = (float*)aligned_alloc(64, MC * KC * sizeof(float));
        float* B_pack = (float*)aligned_alloc(64, KC * NC * sizeof(float));

        // 只有jc循环被并行化
        #pragma omp for
        for (int jc = 0; jc < N; jc += NC) {
            int nc = std::min(NC, N - jc);
            for (int pc = 0; pc < K; pc += KC) {
                int kc = std::min(KC, K - pc);
                pack_B(kc, nc, B + pc*N + jc, N, B_pack);
                for (int ic = 0; ic < M; ic += MC) {
                    int mc = std::min(MC, M - ic);
                    pack_A(mc, kc, A + ic*K + pc, K, A_pack);
                    for (int jr = 0; jr < nc; jr += NR) {
                        for (int ir = 0; ir < mc; ir += MR) {
                            int mr = std::min(MR, mc - ir);
                            int nr = std::min(NR, nc - jr);
                            if (mr == MR && nr == NR) {
                                micro_6x16(kc, A_pack + ir*kc,
                                          B_pack + jr*kc,
                                          C + (ic+ir)*N + (jc+jr), N);
                            } else {
                                for (int i = 0; i < mr; ++i)
                                    for (int j = 0; j < nr; ++j) {
                                        float sum = C[(ic+ir+i)*N + (jc+jr+j)];
                                        for (int k = 0; k < kc; ++k)
                                            sum += A[(ic+ir+i)*K + (pc+k)] *
                                                   B[(pc+k)*N + (jc+jr+j)];
                                        C[(ic+ir+i)*N + (jc+jr+j)] = sum;
                                    }
                            }
                        }
                    }
                }
            }
        }
        free(A_pack);
        free(B_pack);
    }
}
```

## 8.3 NUMA与内存亲和性

在多CPU插槽（socket）的服务器上，内存被分为多个NUMA节点。每个CPU访问本地节点内存快，跨节点慢。

用 `numactl` 或 `pthread_setaffinity_np` 把线程绑定到特定CPU核，并确保它访问的内存分配在本地节点：

```bash
numactl --cpunodebind=0 --membind=0 ./gemm
```

## 8.4 性能

在双核Haswell上，OpenMP可获得约1.8~2倍加速（含超线程小增益），总性能约 85~105 GFLOPS（相对单核的50~60）。

---

# 第9章 性能测量与调优

## 9.1 测量代码

```cpp
#include <chrono>
#include <cstdio>
#include <cstdlib>

double benchmark_gemm(int M, int N, int K,
                      void (*gemm)(int,int,int,const float*,const float*,float*)) {
    float* A = (float*)aligned_alloc(64, M*K*sizeof(float));
    float* B = (float*)aligned_alloc(64, K*N*sizeof(float));
    float* C = (float*)aligned_alloc(64, M*N*sizeof(float));

    // 初始化（用随机数避免编译器优化掉计算）
    for (int i = 0; i < M*K; ++i) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K*N; ++i) B[i] = (float)rand() / RAND_MAX;

    // 预热（让缓存和频率稳定）
    gemm(M, N, K, A, B, C);

    auto start = std::chrono::high_resolution_clock::now();
    gemm(M, N, K, A, B, C);
    auto end = std::chrono::high_resolution_clock::now();

    double sec = std::chrono::duration<double>(end - start).count();
    double gflops = 2.0 * M * N * K / sec / 1e9;
    printf("M=%d N=%d K=%d: %.2f GFLOPS (%.3f s)\n", M, N, K, gflops, sec);

    free(A); free(B); free(C);
    return gflops;
}

int main() {
    benchmark_gemm(1024, 1024, 1024, gemm_naive);
    benchmark_gemm(1024, 1024, 1024, gemm_avx2_simple);
    benchmark_gemm(4096, 4096, 4096, gemm_avx2);
    benchmark_gemm(4096, 4096, 4096, gemm_avx2_omp);
    return 0;
}
```

## 9.2 测量注意事项

- **预热**：第一次运行包含冷缓存效应，测量第二次的结果。
- **多次测量取平均**：单次测量波动大。
- **验证结果**：用朴素实现验证AVX实现的正确性，误差应小于 $10^{-4}$ 相对值。
- **避免编译优化**：如果结果不被使用，编译器可能把整个计算删掉。把结果打印或做checksum。

## 9.3 正确性验证

```cpp
#include <cmath>

bool verify(int M, int N, int K,
            void (*gemm1)(int,int,int,const float*,const float*,float*),
            void (*gemm2)(int,int,int,const float*,const float*,float*)) {
    float* A = (float*)aligned_alloc(64, M*K*sizeof(float));
    float* B = (float*)aligned_alloc(64, K*N*sizeof(float));
    float* C1 = (float*)aligned_alloc(64, M*N*sizeof(float));
    float* C2 = (float*)aligned_alloc(64, M*N*sizeof(float));

    for (int i = 0; i < M*K; ++i) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K*N; ++i) B[i] = (float)rand() / RAND_MAX;

    gemm1(M, N, K, A, B, C1);  // 参考实现（朴素）
    gemm2(M, N, K, A, B, C2);  // 待验证实现

    double max_err = 0;
    for (int i = 0; i < M*N; ++i) {
        double err = std::fabs(C1[i] - C2[i]) / std::max(1.0, std::fabs(C1[i]));
        if (err > max_err) max_err = err;
    }
    printf("Max relative error: %.3e\n", max_err);

    free(A); free(B); free(C1); free(C2);
    return max_err < 1e-4;
}
```

---

# 第10章 性能阶梯总结

## 10.1 各阶段性能

以下为 Haswell 2.0 GHz 双核、FP32、$4096^3$ 矩阵的典型性能：

| 阶段 | 硬件瓶颈 | 关键技术 | 性能 (GFLOPS) | 相对峰值 |
|------|---------|---------|--------------|---------|
| 朴素三重循环 | 标量执行、B跨步访问 | 无 | 1–3 | ~2% |
| 循环重排 | 标量执行 | i→k→j | 5–10 | ~6% |
| 简单向量化 | C读写带宽 | AVX2 FMA | 10–20 | ~15% |
| AVX2微内核 | FMA延迟隐藏 | 6×16寄存器阻塞 | 40–60* | ~40%* |
| 分块+打包 | L2/L3容量、TLB | 三级分块、数据打包 | 50–65 | ~50% |
| 微内核调优 | 发射带宽、加载端口 | 展开、对齐、调度 | 70–85 | ~65% |
| 多线程 | 单核吞吐上限 | OpenMP | 85–105 | ~80%* |

*多线程的"相对峰值"以多核峰值 128 GFLOPS 为基准。

## 10.2 加速比来源

从 1 GFLOPS 到 100 GFLOPS，总加速比约 **100倍**。分解为：

- **向量化 + 寄存器阻塞**：8~10倍（利用SIMD宽度 + 隐藏FMA延迟）
- **分块 + 打包**：5~8倍（解决内存墙）
- **微内核调优 + 预取 + 多线程**：2~3倍

## 10.3 硬件约束的层级

每一层优化都对应一个硬件约束：

1. **FMA执行单元**：向量化 + 寄存器阻塞让它忙起来。
2. **L1带宽**：微内核的加载模式让它高效。
3. **L2/L3容量**：分块策略把数据放在合适层次。
4. **主存带宽**：打包和预取让它不拖后腿。
5. **多核扩展**：线程并行和NUMA亲和性。

## 10.4 进一步优化方向

- **手写汇编微内核**：控制每一条指令的顺序和寄存器分配，超越编译器。
- **非临时存储**：C的写回用 `_mm256_stream_ps` 绕过缓存。
- **AVX-512与VNNI**：对INT8量化的神经网络推理，VNNI提供4倍吞吐。
- **自动调优**：用工具搜索最佳的MC/KC/NC参数组合（如OpenBLAS的调优脚本）。
- **参考高度优化的库**：BLIS、OpenBLAS、MKL、Eigen的源码，理解工业级实现。

---

# 附录A 编译与运行

```bash
# 基础AVX2编译
g++ -O3 -mavx2 -mfma -fopenmp gemm.cpp -o gemm

# AVX-512
g++ -O3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm

# 性能剖析（Linux）
perf stat -e cycles,instructions,cache-misses ./gemm
perf record -g ./gemm
perf report

# 绑定CPU（多核）
taskset -c 0,1 ./gemm
numactl --cpunodebind=0 --membind=0 ./gemm
```

# 附录B 常见问题

**Q1：为什么微内核选6×16而不是8×8？**
A：AVX2有16个寄存器。8×8需要8个寄存器装C（每个C行8个float用1个YMM），剩余8个用于A/B。8个累加链刚好等于隐藏延迟的临界值，没有余量，容易停顿。6×16用12个装C，余量更大。而且6×16的B重用次数（6次）大于8×8（8次，但B向量短）。

**Q2：为什么打包时要填0？**
A：避免微内核处理边界时做分支判断。分支预测失败代价高，而填0让微内核"以为"是完整块。

**Q3：为什么用 `_mm256_broadcast_ss` 而不是先加载再shuffle？**
A：`broadcast_ss` 直接从内存取1个float并复制，只需1条指令。如果先加载整个YMM再提取，反而多几条指令。

**Q4：为什么分块参数不选最大？**
A：分块越大，缓存命中率可能下降（如果超出缓存容量）；分块越小，调用开销和缓存未命中增加。最佳值需要针对具体CPU调优。

**Q5：我的CPU支持AVX-512吗？**
A：用 `cat /proc/cpuinfo | grep avx512` 查看。注意很多消费级CPU（如Intel 12/13/14代）屏蔽了AVX-512。

**Q6：为什么结果和朴素实现有微小差异？**
A：FMA和不同的加法顺序会导致浮点舍入差异。这是正常的，误差应在 $10^{-6}$ 相对量级。如果误差很大（如 $10^{-2}$），说明有bug。

---

# 结语

从朴素实现到接近峰值性能，GEMM优化经历了：**向量化 → 寄存器阻塞 → 缓存分块 → 打包 → 微内核调优 → 多线程** 六个阶段。每一步都不是"魔法"，而是对CPU硬件层级（FMA执行单元、寄存器、L1/L2/L3缓存、主存）的深入理解和针对性利用。

希望这份文档帮你建立了从"数学定义"到"硬件指令"的完整链路。下一步，建议你：

1. 亲手编译、运行、测量每个版本。
2. 用 `perf` 观察瓶颈在哪里。
3. 阅读 BLIS 或 OpenBLAS 的源码，看看工业级实现是如何处理细节的。
4. 尝试针对你的CPU手写汇编微内核。

理解这些原理后，你不只学会了优化GEMM，更学会了**如何把任何计算密集的代码优化到接近硬件峰值**——这是高性能计算的核心能力。
