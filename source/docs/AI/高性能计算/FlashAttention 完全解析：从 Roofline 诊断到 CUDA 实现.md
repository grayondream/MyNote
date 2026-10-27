# FlashAttention 完全解析：从 Roofline 诊断到 CUDA 实现

**摘要**：FlashAttention通过online softmax使得softmax可以按照tile划分减少全局内存的搬运。本文分三步讲清楚这件事：先用 Roofline 模型说明传统注意力在各代 GPU 上都被“带宽”卡住；再推导 online softmax，看它如何一遍扫描就精确完成 softmax，并把 $N\times N$ 的大矩阵永远留在片上；最后走进 CUDA 实现，看同一套算法在不同硬件上如何被逐级工程化。

---

&emsp;&emsp;FlashAttention的优化思路主要分为两部分，一部分是通过tile划分来减少内存搬运，这部分是HPC常用的手段并不算是特别惊艳的手段；另一部分是通过online softmax使得softmax可以划分tile而不用先全数遍历。所以总的来说核心思想还是HPC领域常见的TILE思路，论文含金量各自评判吧。

## 1. 注意力结构及其性能瓶颈

### 1.1 Attention 的结构与计算量

&emsp;&emsp;**先看数学定义。** 单头缩放点积注意力接收三个矩阵——查询 $Q$、键 $K$、值 $V$，形状都是 $N\times d$。这里 $N$ 是序列长度，$d$ 是每个头的维度。它的定义是：

$$
O = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

&emsp;&emsp;式子里的 $1/\sqrt{d}$ 只是用来把点积的数值范围压稳，防止 softmax 梯度爆炸，不影响计算结构。为了行文简洁，后面我们把它略去。整个算子可以拆成三步：

$$
S = QK^\top \in \mathbb{R}^{N\times N}, \qquad
P = \operatorname{softmax}(S) \in \mathbb{R}^{N\times N}, \qquad
O = PV \in \mathbb{R}^{N\times d}
$$

&emsp;&emsp;**三步各要算多少次？** 第一步 $S = QK^\top$：输出有 $N^2$ 个元素，每个元素是 $Q$ 的一行和 $K$ 的一行做 $d$ 维点积，也就是 $2N^2d$ 次浮点运算。这是一次标准的矩阵乘法（GEMM），完全交给 Tensor Core 干，是三步里效率最高的。第二步 $P = \operatorname{softmax}(S)$：对 $S$ 的每一行独立做 safe softmax——求行最大值 $m$、每个元素减 $m$、取指数、求和、再逐元素除以分母。每行大约 $5N$ 次简单运算，总计约 $5N^2$。**单看数字，当 $d=128$ 时它只占总运算量的约 1%，好像可以忽略。但它不是矩阵乘法**：exp、max、逐元素除法都走 CUDA Core，而 A100 上 FP16 矩阵乘的峰值约 312 TFLOPS，逐元素运算只有约 19 TFLOPS，差了十几倍。所以这 1% 的运算量，可能吃掉 10%–20% 的实际时间——这正是后来 FA2 重点优化的地方（第 3 节会讲）。第三步 $O = PV$：输出 $Nd$ 个元素，每个元素是 $P$ 的一行（长度 $N$）和 $V$ 的一列（长度 $N$）做点积，同样是 $2N^2d$ 次运算，同样交给 Tensor Core。三步加起来：

$$
\text{FLOPs}_{\text{attn}} = 4N^2 d + 5N^2 = (4d + 5)N^2
$$

&emsp;&emsp;其中 $4N^2d$ 来自两次矩阵乘，$5N^2$ 来自 softmax 的逐元素运算（5 为近似常数，随实现细节浮动）。主项 $4N^2d$ 对 $N$ 是二次增长，这就是长上下文成本飙升的算术根源，也是线性注意力、稀疏注意力想攻克的目标。**注意：FlashAttention 没有改变这一项**——它照样做完整的 $O(N^2)$ 计算。

&emsp;&emsp;**比计算量更麻烦的是中间矩阵的体量。** 输入输出 $Q, K, V, O$ 各含 $Nd$ 个元素，随 $N$ 线性增长；但中间矩阵 $S$ 和 $P$ 各含 $N^2$ 个元素，随 $N$ **二次**增长。以 $N=8192$、FP16（每元素 2 字节）算一笔账：

| 张量 | 元素数 | FP16 大小 |
|---|---:|---:|
| $Q, K, V, O$（各） | $8192 \times 128 \approx 1.05\text{M}$ | 2.1 MB |
| $S$ | $8192^2 = 67.1\text{M}$ | **134 MB** |
| $P$ | $8192^2 = 67.1\text{M}$ | **134 MB** |

&emsp;&emsp;一个 $S$ 或 $P$，体量是单个 $Q$ 块的 **64 倍**。而 GPU 的片上高速缓存（SRAM）有多小？A100 每个 SM 只有约 192 KB——连一个完整的 $S$ 都塞不下。所以传统实现别无选择，只能把 $S$ 和 $P$ 放到容量大但速度慢的 HBM 里，在 kernel 之间来回搬。

&emsp;&emsp;**传统实现长什么样：三个独立 kernel。** 标准做法把上面三步拆成三个独立的 CUDA kernel，一个接一个执行。每个 kernel 都完整走一遍“从 HBM 读 → 计算 → 写回 HBM”：

- **Kernel 1（QKᵀ）**：读 $Q, K$，算出 $S$，写回 HBM；
- **Kernel 2（softmax）**：读 $S$，算出 $P$，写回 HBM；
- **Kernel 3（PV）**：读 $P$ 和 $V$，算出 $O$，写回 HBM。

>吐槽一下论文里这种方式真正工程落地没人会这么实现，这种方式明显是违背最基本的GEMM优化的。

&emsp;&emsp;下图画出这条数据流。蓝色是随 $N$ 线性增长的输入/输出张量，红色是 $N^2$ 量级的中间矩阵——它们是 HBM 流量的主要来源。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/svgviewer-outputassssssssssssssssss.svg)

&emsp;&emsp;图中最值得注意的一点：$Q, K, V, O$ 随 $N$ 线性增长，$S$ 和 $P$ 随 $N$ 二次增长。序列越长，后者的占比越压倒前者。**算一笔 HBM 流量的账。** 按 FP16（2 字节/元素）逐个 kernel 记账：

| 阶段 | 读 | 写 | 元素数 |
|---|---|---|---:|
| Kernel 1 | $Q, K$ | $S$ | $2Nd + N^2$ |
| Kernel 2 | $S$ | $P$ | $2N^2$ |
| Kernel 3 | $P, V$ | $O$ | $N^2 + 2Nd$ |
| **合计** | | | $\mathbf{4N^2 + 4Nd}$ |

&emsp;&emsp;换算成字节：

$$
\text{Bytes}_{\text{naive}} = 2(4N^2 + 4Nd) = 8N^2 + 8Nd
$$

&emsp;&emsp;当 $N \gg d$ 时，主导项是 $8N^2$。取 $N=4096, d=128$，总字节约 **138.4 MB**，其中 $S$ 和 $P$ 的读写占了 **97%**。三步计算的数学完全正确，运算量也不大（这个配置下按 $(4d+5)N^2$ 计算约 8.67 GFLOP），问题出在中间结果的落地方式上：$S$ 和 $P$ 必须写到 HBM，下一个 kernel 又必须完整读回来。计算核心在等这些数据的时候只能闲着。

&emsp;&emsp;**但“闲着”是否真的发生，得看硬件参数。** 同样 138.4 MB，在带宽 2 TB/s 的 A100 上要搬 68 μs，在带宽 8 TB/s 的 B200 上只要 17 μs；而同样 8.67 GFLOP 的计算，不同架构的耗时差异更大。所以光看 FLOPs 或光看字节数都不够，必须把两者放进同一个坐标系——这就是 Roofline 模型。

### 1.2 用 Roofline 判断不同架构下的瓶颈

![](https://thumb.wikimedia.org/wikipedia/commons/thumb/b/b1/Example_of_a_Roofline_model.svg/1920px-Example_of_a_Roofline_model.svg.png)

&emsp;&emsp;**先承认一个现实：GPU 的存储是两级的。** 一边是 HBM，容量大（80 GB）、带宽有限、延迟高；另一边是 SRAM（共享内存 + L1），容量极小、带宽极高。具体数字：

| 层级 | A100 | H100 | 特点 |
|---|---|---|---|
| **HBM** | 80 GB / 2.04 TB/s | 80 GB / 3.35 TB/s | 容量大、带宽有限、延迟高 |
| **SRAM** | 每 SM ~192 KB，全芯片 ~20 MB / ~19 TB/s | 每 SM 256 KB，全芯片 ~33 MB / ~33 TB/s | 容量极小、带宽约为 HBM 的 10 倍 |

&emsp;&emsp;**SRAM 带宽是 HBM 的约 10 倍，容量却小约 4000 倍**。任何 kernel 都得先把数据从 HBM 搬进 SRAM 才能计算。计算本身几乎不花时间，**等搬运才花时间**。**Roofline 模型怎么用。** 它用“算术强度” $I$ 来刻画一个 kernel 的性能上限。算术强度的定义很直白：**每搬 1 字节数据，能换来多少次浮点运算**。

$$
I = \frac{\text{FLOPs}}{\text{HBM 访问字节数}}
$$

&emsp;&emsp;硬件的性能上限是：

$$
\text{可达 FLOP/s} = \min\bigl(\text{Peak FLOP/s},\; I \times \text{Peak HBM BW}\bigr)
$$

&emsp;&emsp;这个式子可以这么理解：性能有一个“算力天花板”和一个“带宽天花板”。当 $I$ 很小时，瓶颈是带宽——你搬得不够快，算力再强也没用；当 $I$ 很大时，瓶颈变成算力——数据供给跟得上了，就看算得多快。两个天花板相交的地方叫**脊点（ridge point）**：

$$
I_{\text{ridge}} = \frac{\text{Peak FLOP/s}}{\text{Peak HBM BW}}
$$

- $I < I_{\text{ridge}}$：落在斜线段，**memory-bound**（带宽受限）。执行时间由 $\text{Bytes}/\text{BW}$ 决定，加算力没用。
- $I > I_{\text{ridge}}$：落在平台段，**compute-bound**（算力受限）。执行时间由 $\text{FLOPs}/\text{Peak}$ 决定，减字节没用。

&emsp;&emsp;判断一个算子该往哪个方向优化，第一步就是把它放到 Roofline 图上。**传统 Attention 的算术强度是多少？** 把 1.1 节的数字代进去：

$$
I_{\text{naive}} = \frac{(4d+5)N^2}{8N^2 + 8Nd} = \frac{(4d+5)N}{8(N+d)}
\;\xrightarrow[N \gg d]{}\; \frac{4d+5}{8} \approx \frac{d}{2}
$$

&emsp;&emsp;这个结果很关键：**传统 Attention 的算术强度不随序列长度增长，只正比于头维度 $d$**。序列 $N$ 变长，$N^2$ 项会把 FLOPs 和字节数同时放大，比值不变。也就是说，**序列变长不会让这个算子变得“更划算”**。代入典型值：$d=64$ 时约 32.6 FLOPs/byte，$d=128$ 时约 64.6，$d=256$ 时约 128.6，$d=512$ 时约 256.6。现代 LLM 的头维度通常在 64–128（Llama 系列 128，GPT-3 128），对应算术强度只有 32–65。

&emsp;&emsp;**各代 GPU 的脊点在哪。** 各架构的理论峰值（dense、无稀疏）与脊点：

| GPU | 峰值 bf16 (TFLOPS) | HBM 带宽 (TB/s) | 脊点 $I_{\text{ridge}}$ (FLOPs/byte) |
|---|---:|---:|---:|
| A100 SXM 80GB | 312 | 2.04 | **153** |
| H100 SXM | 989 | 3.35 | **295** |
| H200 SXM | 989 | 4.80 | **206** |
| B200 | ~2250 | 8.00 | **281** |

&emsp;&emsp;取 $N=4096, d=128$、FP16 的典型配置（$I_{\text{naive}} = 64.6$），做完整判定：

| GPU | FLOPs | HBM 字节 | 计算下限 | 带宽下限 | 判定 | 带宽/计算 |
|---|---:|---:|---:|---:|---|---:|
| A100 | 8.67 GFLOP | 138.4 MB | 27.8 μs | **67.8 μs** | memory-bound | 2.44× |
| H100 | 8.67 GFLOP | 138.4 MB | 8.8 μs | **41.3 μs** | memory-bound | 4.69× |
| H200 | 8.67 GFLOP | 138.4 MB | 8.8 μs | **28.8 μs** | memory-bound | 3.27× |
| B200 | 8.67 GFLOP | 138.4 MB | 3.9 μs | **17.3 μs** | memory-bound | 4.43× |

（“计算下限”和“带宽下限”是假设达到理论峰值时的耗时；实际耗时取较大者，再加上各种开销。）

&emsp;&emsp;**两个直接结论。** 第一，在这些架构上，传统 Attention 全是 memory-bound。换成更快的 Tensor Core 也没用——只要字节数不降，执行时间就被带宽天花板卡死。这就是“升级到 H100 后注意力没变快多少”的原因。第二，**架构越新，偏离脊点越远**。从 A100 到 H100，算力提升约 3.2 倍，带宽只提升约 1.6 倍，脊点从 153 涨到 295；而 $I_{\text{naive}}$ 只由 $d$ 决定，稳稳停在 64.6。于是带宽与计算的差距从 2.44 倍扩大到 4.69 倍。**IO 瓶颈在新架构上不是缓解了，而是更严重了。** 这解释了两件事：一是 FlashAttention 在 Hopper 上的收益比 Ampere 更大；二是 FA3 要专门用 TMA、warp specialization 把搬运压到极致——在 H100 上，任何浪费带宽的实现都会被 4.69 倍的差距放大。

&emsp;&emsp;**什么情况下不是 memory-bound？** 公式 $I_{\text{naive}} \approx (4d+5)/8$ 说明，头维度很大时，传统 Attention 可能进入 compute-bound。比如 $d=512$ 时 $I \approx 256.6$，在 A100 上已经超过脊点 153，此时减少字节的收益有限，优化重心该转向计算本身（比如低精度 Tensor Core）。但这一条在现代 LLM 里很少成立：主流模型的 head dim 是 64–128，为了控制参数量和并行度，很少超过 256。对绝大多数真实负载，传统 Attention 稳稳落在 memory-bound 区间。这也提示一件事：**FlashAttention 不是普适最优**，它的收益前提是 $d$ 不大、$N$ 不小。

&emsp;&emsp;**唯一有效的杠杆。** Roofline 分析把优化方向指得很明确：传统 Attention 的 $I \approx (4d+5)/8$，远低于脊点，执行时间由 HBM 字节数决定；加算力不改变瓶颈位置；**唯一能把算子推过脊点的办法，是减少 HBM 字节数，把中间矩阵留在 SRAM 里**。按 1.1 节的账，$S$ 和 $P$ 占了 97% 的字节。如果能让它们完全不出现在 HBM 中，字节数就从 $8N^2 + 8Nd$ 降到约 $8Nd$，算术强度从约 $d/2$ 跳到约 $N/2$——$N=4096$ 时是 2048，足以跨过任何现代 GPU 的脊点。

&emsp;&emsp;这正是 FlashAttention 的设计目标。下一节顺着这条 IO 路径，拆解传统实现的病根，并展示 tiling 和 online softmax 如何在不改变数学结果的前提下，把 $N^2$ 项从 HBM 流量中消掉。

---

## 2. FlashAttention 的原理

### 2.1 核心机制：Tiling 与 Online Softmax

&emsp;&emsp;**总策略一句话：不改变数学，只改变执行顺序。** 第 1 节确定了唯一的有效杠杆——减少 HBM 字节。传统实现的字节几乎全来自 $S$ 和 $P$ 的物化（97%）。所以 FlashAttention 的目标很明确：**让 $N\times N$ 矩阵从出生到消亡都待在 SRAM 里，一步也不写进 HBM**。它不做任何近似，输出和标准注意力逐元素相同。做法是把三个 kernel 合成一个：每个 thread block 负责一个 Q 块，在 SRAM 里完成这个块和所有 K/V 块的 $QK^\top$、softmax、$PV$，最后只把 $O$ 块写回去。听起来不错，但一落地就撞上一个数学障碍。

&emsp;&emsp;**障碍：分块之后，softmax 没法算了。** 标准 safe softmax 是**三遍扫描**：第一遍求整行最大值 $m = \max_j x_j$；第二遍求归一化和 $\ell = \sum_j e^{x_j - m}$；第三遍才输出 $p_j = e^{x_j - m}/\ell$。三遍的前提是整行数据同时可见。分块流式处理时，处理第 $j$ 个 K 块只能看到 $S$ 的一段行片段，**在见到全部 K 之前，根本不知道全局最大值 $m$ 是多少**，自然也算不出 $\ell$。如果退回去把 $S$ 块先算完存到 HBM、再回头做 softmax，那就回到标准实现的老路了，tiling 的收益全没了。所以关键问题是：**能不能只扫一遍数据、不保存历史块，还得到和全局 safe softmax 一模一样的结果？**

&emsp;&emsp;**Online softmax：把“先见全局”换成“边见边修”。** 答案是能。核心思路是维护两个运行中的标量——当前已遍历部分的最大值 $m$ 和指数和 $\ell$——每读入一个新块就增量更新。这两个量在任何时刻都等于“如果现在就结束，全局 safe softmax 会得到的值”。假设已经处理完若干块，累计状态是 $(m, \ell)$：

$$
m = \max_{j \in \text{seen}} x_j, \qquad \ell = \sum_{j \in \text{seen}} e^{x_j - m}
$$

&emsp;&emsp;现在新读入一块 $x^{(b)}$，先算这一块自己的局部统计量 $m_b = \max_j x_j^{(b)}$ 和 $\ell_b = \sum_j e^{x_j^{(b)} - m_b}$。合并后的真实最大值显然是：

$$
m_{\text{new}} = \max(m,\; m_b)
$$

&emsp;&emsp;但 $\ell$ 不能直接相加——旧的 $\ell$ 是以 $m$ 为基准算出来的，新的 $\ell_b$ 是以 $m_b$ 为基准算出来的，**两者的基准不同**。要合并，得先把它们换算到同一个基准 $m_{\text{new}}$ 下：

$$
\ell_{\text{new}} = \ell \cdot e^{m - m_{\text{new}}} + \ell_b \cdot e^{m_b - m_{\text{new}}}
$$

&emsp;&emsp;合并后的 $\ell_{\text{new}}$ 应该等于 $\sum_{j \in \text{seen} \cup b} e^{x_j - m_{\text{new}}}$。拆成旧部分和新部分：

$$
\sum_{j \in \text{seen}} e^{x_j - m_{\text{new}}} = e^{m - m_{\text{new}}} \sum_{j \in \text{seen}} e^{x_j - m} = e^{m - m_{\text{new}}}\,\ell
$$

$$
\sum_{j \in b} e^{x_j - m_{\text{new}}} = e^{m_b - m_{\text{new}}} \sum_{j \in b} e^{x_j - m_b} = e^{m_b - m_{\text{new}}}\,\ell_b
$$

&emsp;&emsp;两项相加，正好是上面的更新式。**恒等式成立，而且和分块方式、分块顺序无关**。系数 $e^{m - m_{\text{new}}}$ 就是所谓的**修正因子（correction factor）**：如果新块没有刷新最大值（$m_b \le m$），那么 $m_{\text{new}} = m$，修正因子是 1，旧状态原样保留、直接累加；如果新块刷新了最大值（$m_b > m$），修正因子小于 1，旧状态被整体“重缩放”到新基准下。整个过程中只需保存 $(m, \ell)$ 两个标量，历史块用完就扔，**一遍扫描就能完成**。

&emsp;&emsp;**从标量到向量：输出累积量也要跟着重缩放。** 注意力的输出不是标量概率，而是概率对 $V$ 的加权和：

$$
O = \sum_j p_j v_j = \frac{1}{\ell}\sum_j e^{x_j - m} v_j
$$

&emsp;&emsp;定义**未归一化输出累积量** $\tilde{O} = \sum_j e^{x_j - m} v_j$（形状 $B_r \times d$），最终输出就是 $O = \tilde{O}/\ell$。和 $\ell$ 一个道理，$\tilde{O}$ 也是以当前 $m$ 为基准的，合并新块时必须先重缩放再累加：

$$
\tilde{O}_{\text{new}} = e^{m - m_{\text{new}}}\,\tilde{O} + e^{m_b - m_{\text{new}}}\,\bigl(P_b V_b\bigr)
$$

&emsp;&emsp;其中 $P_b = e^{S_b - m_b}$ 是当前块的局部未归一化概率。**修正因子同时作用在 $\ell$ 和 $\tilde{O}$ 上，保证两者始终处于同一个基准**，这是正确性的关键。循环结束后做一次除法 $O = \tilde{O}/\ell$，得到的结果和全局 safe softmax 严格等价。整个过程里，$S$ 块和 $P$ 块的生命周期完全在 SRAM 内，从未落盘。

&emsp;&emsp;**完整算法：外层 Q、内层 KV 的双层循环。** 把上面的机制组装起来，前向 kernel 长这样（Python 风格伪代码，省略 batch 和 head 维）：

```python
for i in range(N // B_r):                    # 外层：遍历 Q 块
    Q_i = load(Q[i*B_r : (i+1)*B_r])         # 一次性读入 SRAM
    m, l, O_tilde = -inf, 0, 0               # 运行态，驻留 SRAM
    for j in range(N // B_c):                # 内层：遍历 K/V 块
        K_j, V_j = load(K, V, j)             # 流式读入 SRAM
        S_ij = Q_i @ K_j.T                   # GEMM 1，Tensor Core
        m_b  = rowmax(S_ij)                  # 局部最大值
        m_new = max(m, m_b)
        corr = exp(m - m_new)                # 修正因子
        l    = l * corr + rowsum(exp(S_ij - m_new))
        O_tilde = O_tilde * corr + exp(S_ij - m_new) @ V_j   # GEMM 2
        m = m_new
    O_i = O_tilde / l                        # 循环外做一次归一化
    store(O_i)                               # 唯一一次写回 HBM
```

&emsp;&emsp;第一，**S 和 P 从未作为持久张量存在**：`S_ij` 和 `exp(S_ij - m_new)` 都是循环体内的临时值，用完即弃，编译器可以把它们完全分配在寄存器和 SRAM 里。第二，**归一化被推迟到循环外**：内层循环只做累加，除法只在最后执行一次，这省掉了 $N/B_c$ 倍的逐元素除法——这是 FA2 相对 FA1 的关键改进之一（第 3 节展开）。第三，**修正因子在绝大多数迭代中都是 1**：一旦 $m$ 稳定下来，`corr` 恒等于 1，重缩放退化成空操作，实际开销远小于最坏情况。

&emsp;&emsp;下图展示单个 Q 块在内层循环中的数据流。蓝色是一次性读写，橙色是循环内反复使用但从不离开 SRAM 的数据，红色标注了传统实现与本实现的分野——$S$ 和 $P$ 只在 SRAM 内短暂存在，没有任何指向 HBM 的箭头。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/flash_attention_2026-10-17.svg)

&emsp;&emsp;**状态压缩：为什么存 LSE 而不是 $(m,\ell)$。** 前向结束后，反向传播需要知道每个 Q 行的 softmax 基准，才能重算 $P$。直接存 $(m, \ell)$ 每行要 2 个 FP32 标量；FlashAttention 改存**对数域归一化常数**：

$$
\operatorname{LSE} = m + \ln \ell
$$

&emsp;&emsp;有了它，概率可以直接写成 $p_j = e^{x_j - \operatorname{LSE}}$——**一次减法加一次指数，不需要除法**，而且每行只占 1 个 FP32 标量，HBM 写带宽减半。这个 32 位状态量在前向结束时写回 HBM，规模是 $N$ 个标量（而不是 $N^2$），是反向重算（第 3.2 节）和解码阶段 Split-KV 的共同基础。

### 2.2 IO 复杂度与 Roofline 验证

&emsp;&emsp;**精确算一算 HBM 流量。** 现在可以给 FlashAttention 记账了。外层循环 $N/B_r$ 次，每次读一个 Q 块（$Nd$ 元素总计，每个 Q 只读一次），结束时写一个 O 块（同样 $Nd$）。内层循环里，每个 Q 块要遍历全部 K/V，读取量是 $2Nd$，所以 K/V 的总读取量是：

$$
\frac{N}{B_r} \cdot 2Nd = \frac{2N^2 d}{B_r}
$$

&emsp;&emsp;SRAM 容量 $M$（元素数）决定了 $B_r$ 的上界：Q 块、K/V 块、$S$ 块和 $\tilde{O}$ 要同时驻留，$B_r$ 大致受 $M/d$ 约束。代进去：

$$
\text{IO}_{\text{FA}} = \Theta\!\left(Nd + \frac{N^2 d^2}{M}\right)
$$

&emsp;&emsp;对比标准实现的 $\Theta(N^2 + Nd)$：**主导项从 $N^2$ 变成了 $N^2d^2/M$**。这个替换效果如何，取决于 $d^2/M$ 和 1 谁大谁小。在 A100 上（$M \approx 192\text{ KB} / 2\text{ B} = 98304$ 元素，$d=128$），$d^2/M \approx 1/6$；按 SRAM 允许的最大 Q 块（$B_r \approx 512$）估算，K/V 项约 $2N^2 \cdot 128 / 512 = N^2/2$ 元素，比标准实现里 $S$、$P$ 各 $N^2$ 的读写量低一个数量级。实际实现还有一层额外缓冲：$K$ 和 $V$ 总共 $2Nd$ 个元素（$N=4096, d=128$ 时只有 2 MB），完全装得进 A100 的 40 MB L2，**内层循环的重复读取大量命中 L2 而不是 HBM**，使实际 HBM 流量进一步逼近下界 $Nd$。

| $N$（$d=128$，FP16） | 标准 HBM 流量 | FlashAttention 流量 | 降低倍数 |
|---:|---:|---:|---:|
| 1,024 | 8.4 MB | 1.1 MB | 7.6× |
| 4,096 | 138.4 MB | 16.8 MB | 8.2× |
| 16,384 | 2,147.5 MB | 268.4 MB | 8.0× |

（表中是论文报告的估算值，随 tile 配置和 L2 命中率浮动；降低倍数约为 $N/(4d)$，随序列长度线性增长——序列越长，收益越大。）

&emsp;&emsp;**回到 Roofline：这次跨过脊点了。** 把 1.2 节的框架套到新数字上。$N=4096$、$d=128$ 时，FLOPs 不变（仍是 8.67 GFLOP），但字节数从 138.4 MB 降到 16.8 MB：

$$
I_{\text{FA}} = \frac{8.67 \times 10^9}{1.68 \times 10^7} \approx 516 \ \text{FLOPs/byte}
$$

&emsp;&emsp;对照 1.2 节的脊点表：A100 是 153，H100 是 295，B200 是 281。**516 跨过了所有这些架构的脊点**，算子从 memory-bound 移进 compute-bound。执行时间上直接反映了这一点：A100 上标准实现的带宽下限是 67.8 μs，而 FlashAttention 的计算下限只有 27.8 μs——**瓶颈换了一侧，天花板也跟着换了**。

&emsp;&emsp;**“做了更多计算却更快”怎么解释。** 到这里可以把这个反直觉现象讲清楚了。FlashAttention 的 FLOPs 确实比标准实现多：online softmax 的修正因子、行最大值比较、逐块 rescale 都带来额外运算，总量比标准前向高 10%–20%（反向的重算还会再加，见第 3.2 节）。但这些多出来的 FLOPs 有个共同特征——**它们发生在 SRAM 里，而且分散在原本就在等 HBM 的空闲周期里**。标准实现的 Kernel 2 算术强度只有 1.25，Tensor Core 全程闲置；FlashAttention 把 softmax 融进内层循环后，这些非 matmul 运算可以和 GEMM 的等待时间重叠。用 Roofline 的话说：**多付出的算力落在 compute-bound 区间，而省下的带宽原本是唯一的瓶颈**。这正是 1.2 节那条判据的直接兑现——在 memory-bound 区间，减少字节比减少 FLOPs 更有价值。

&emsp;&emsp;**因果掩码是白送的额外收益。** 自回归训练里注意力矩阵是下三角的，上三角完全被掩码。分块后可以按块判断：完全位于对角线以下的块，跳过掩码判断直接算；完全位于对角线以上的块，**根本不进入内层循环**。后者让 K/V 的总读取量减半，IO 复杂度里的 $N^2$ 项直接砍掉一半，前向实际耗时也相应下降——这是在 tiling 结构上自然得到的，不需要额外的算法设计。

&emsp;&emsp;**反向怎么办。** 前向解决了，但训练要梯度。反向传播需要 $P$ 来算 $dV = P^\top dO$ 和 $dS = dP \odot P'$。如果按标准做法把 $P$ 存下来，$O(N^2)$ 的显存墙立刻回来——前向省下的显存全吐出去，训练配置还是被锁死（第 1.1 节的表格）。FlashAttention 的解法是**不存 $P$，只存 LSE**，反向时从 $Q$、$K$ 和 LSE 在 SRAM 里按块重算 $S$ 和 $P$。这笔交易的代价是反向 FLOPs 约翻倍，收益是激活显存从 $O(N^2)$ 降到 $O(N)$。在 Roofline 框架下它同样划算：重算发生在 SRAM 内的 compute-bound 操作上，而“存储 $P$”的替代方案在反向时还得从 HBM 读 $O(N^2)$ 字节，是 memory-bound 的——**用便宜算力换昂贵带宽**，和本节主线完全一致。反向的具体结构和 CUDA 实现细节留到第 3.2 节。

---

## 3 FlashAttention-2：把跨过脊点之后的浪费收回来

&emsp;&emsp;第 2 节证明了 tiling + online softmax 能把算术强度从约 64.6 抬到约 516，让注意力跨过 A100（153）、H100（295）的脊点，进入 compute-bound。但跨过脊点不等于跑满算力。FlashAttention-1 在 A100 上只达到 25%–40% 的理论峰值，而调优好的 GEMM 能到 80%–90%。FA2 要解决的是跨过脊点之后，compute-bound 区间里那些“不该算的”和“没并行的”。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 680" width="100%" role="img" aria-label="FlashAttention-2 三项优化示意">
  <title>图 3-1 FlashAttention-2：减少非 matmul、序列并行、split-Q</title>
  <desc>左图展示 FA1 内循环每次迭代都执行 B_r×d 的 O_tilde 重缩放，标量工作排在 CUDA Core 上；右图展示 FA2 将重缩放改为条件触发，并把 warp 划分从 split-K 改为 split-Q，同时在 Q 序列维度增加并行。底部列出三项优化及其效果。</desc>
  <defs>
    <marker id="arrow31" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#334155"/>
    </marker>
    <style>
      .t31{font-family:Arial,'Microsoft YaHei',sans-serif;fill:#0f172a}
      .s31{font-family:Arial,'Microsoft YaHei',sans-serif;fill:#475569}
      .box31{stroke:#334155;stroke-width:1.5;rx:8}
      .red31{fill:#fee2e2;stroke:#dc2626;stroke-width:1.8;rx:8}
      .blue31{fill:#dbeafe;stroke:#2563eb;stroke-width:1.5;rx:8}
      .green31{fill:#dcfce7;stroke:#16a34a;stroke-width:1.5;rx:8}
      .orange31{fill:#ffedd5;stroke:#ea580c;stroke-width:1.5;rx:8}
    </style>
  </defs>
  <rect x="0" y="0" width="1120" height="680" fill="#ffffff"/>
  <text x="30" y="36" class="t31" font-size="24" font-weight="700">图 3-1 FA2 的三项优化：把 compute-bound 区间内的排队与并行缺口补上</text>
  <text x="30" y="62" class="s31" font-size="14">FA1 已跨过脊点；FA2 不改变数学，只减少 CUDA Core 上的非 matmul 工作、增加序列并行、减少共享内存往返。</text>

  <rect x="30" y="90" width="500" height="300" fill="#f8fafc" stroke="#cbd5e1"/>
  <text x="50" y="120" class="t31" font-size="18" font-weight="700">FA1 内循环：每次迭代都重缩放 Õ</text>
  <rect x="55" y="150" width="105" height="58" class="blue31"/>
  <text x="107" y="176" text-anchor="middle" class="t31" font-size="14">QKᵀ</text>
  <text x="107" y="194" text-anchor="middle" class="s31" font-size="11">Tensor Core</text>
  <rect x="190" y="150" width="115" height="58" class="orange31"/>
  <text x="247" y="172" text-anchor="middle" class="t31" font-size="13">softmax 更新</text>
  <text x="247" y="190" text-anchor="middle" class="s31" font-size="11">rowmax / exp</text>
  <rect x="335" y="150" width="150" height="58" class="red31"/>
  <text x="410" y="170" text-anchor="middle" class="t31" font-size="13">Õ *= corr</text>
  <text x="410" y="188" text-anchor="middle" class="s31" font-size="11">B_r×d 逐元素，每次迭代</text>
  <rect x="55" y="245" width="105" height="58" class="blue31"/>
  <text x="107" y="271" text-anchor="middle" class="t31" font-size="14">PV</text>
  <text x="107" y="289" text-anchor="middle" class="s31" font-size="11">Tensor Core</text>
  <rect x="190" y="245" width="115" height="58" class="red31"/>
  <text x="247" y="267" text-anchor="middle" class="t31" font-size="12">l 更新 / 除法</text>
  <text x="247" y="285" text-anchor="middle" class="s31" font-size="11">CUDA Core 排队</text>
  <rect x="335" y="245" width="150" height="58" class="red31"/>
  <text x="410" y="267" text-anchor="middle" class="t31" font-size="12">warp 间 split-K</text>
  <text x="410" y="285" text-anchor="middle" class="s31" font-size="11">跨 warp 归约经 SMEM</text>
  <path d="M160 179 L188 179" stroke="#334155" stroke-width="1.5" marker-end="url(#arrow31)"/>
  <path d="M305 179 L333 179" stroke="#334155" stroke-width="1.5" marker-end="url(#arrow31)"/>
  <path d="M410 208 L410 243" stroke="#334155" stroke-width="1.5" marker-end="url(#arrow31)"/>
  <text x="55" y="335" class="s31" font-size="12">瓶颈：标量重缩放和归约把 CUDA Core 排满，Tensor Core 等喂料；</text>
  <text x="55" y="355" class="s31" font-size="12">并行度只来自 batch×head，小 batch 长序列填不满 SM。</text>

  <rect x="560" y="90" width="530" height="300" fill="#f8fafc" stroke="#cbd5e1"/>
  <text x="580" y="120" class="t31" font-size="18" font-weight="700">FA2 内循环：条件重缩放 + split-Q</text>
  <rect x="585" y="150" width="105" height="58" class="blue31"/>
  <text x="637" y="176" text-anchor="middle" class="t31" font-size="14">QKᵀ</text>
  <text x="637" y="194" text-anchor="middle" class="s31" font-size="11">Tensor Core</text>
  <rect x="720" y="150" width="135" height="58" class="orange31"/>
  <text x="787" y="172" text-anchor="middle" class="t31" font-size="13">softmax 更新</text>
  <text x="787" y="190" text-anchor="middle" class="s31" font-size="11">rowmax / exp；corr 多数为 1</text>
  <rect x="885" y="150" width="165" height="58" class="green31"/>
  <text x="967" y="170" text-anchor="middle" class="t31" font-size="13">仅当 m 刷新时 rescale</text>
  <text x="967" y="188" text-anchor="middle" class="s31" font-size="11">非 matmul 工作显著下降</text>
  <rect x="585" y="245" width="105" height="58" class="blue31"/>
  <text x="637" y="271" text-anchor="middle" class="t31" font-size="14">PV</text>
  <text x="637" y="289" text-anchor="middle" class="s31" font-size="11">Tensor Core</text>
  <rect x="720" y="245" width="135" height="58" class="green31"/>
  <text x="787" y="267" text-anchor="middle" class="t31" font-size="12">warp 内 split-Q</text>
  <text x="787" y="285" text-anchor="middle" class="s31" font-size="11">每 warp 负责 Q 的不同行</text>
  <rect x="885" y="245" width="165" height="58" class="green31"/>
  <text x="967" y="267" text-anchor="middle" class="t31" font-size="12">Õ 留在寄存器</text>
  <text x="967" y="285" text-anchor="middle" class="s31" font-size="11">减少 SMEM 往返</text>
  <path d="M690 179 L718 179" stroke="#334155" stroke-width="1.5" marker-end="url(#arrow31)"/>
  <path d="M855 179 L883 179" stroke="#334155" stroke-width="1.5" marker-end="url(#arrow31)"/>
  <path d="M967 208 L967 243" stroke="#334155" stroke-width="1.5" marker-end="url(#arrow31)"/>
  <text x="585" y="335" class="s31" font-size="12">额外优化：Q 序列维度并行——不同 Q 块分到不同 block，小 batch 也能铺满 SM。</text>
  <text x="585" y="355" class="s31" font-size="12">效果：A100 上 25%–40% → 50%–73% 峰值，225 TFLOPs/s（72% MFU）。</text>

  <rect x="30" y="415" width="1060" height="235" fill="#ffffff" stroke="#94a3b8"/>
  <text x="50" y="445" class="t31" font-size="18" font-weight="700">三项优化在 Roofline 上的位置：不移动脊点，只把工作点推向算力平台</text>
  <rect x="55" y="475" width="300" height="140" class="blue31"/>
  <text x="205" y="502" text-anchor="middle" class="t31" font-size="15" font-weight="700">① 减少非 matmul FLOPs</text>
  <text x="75" y="528" class="s31" font-size="12">把 Õ 的重缩放改为条件触发；</text>
  <text x="75" y="550" class="s31" font-size="12">避免每次迭代都做 B_r×d 逐元素乘；</text>
  <text x="75" y="572" class="s31" font-size="12">让 Tensor Core 不等 CUDA Core。</text>
  <rect x="380" y="475" width="300" height="140" class="green31"/>
  <text x="530" y="502" text-anchor="middle" class="t31" font-size="15" font-weight="700">② Q 序列维度并行</text>
  <text x="400" y="528" class="s31" font-size="12">并行度从 batch×head 扩展到 Q 块；</text>
  <text x="400" y="550" class="s31" font-size="12">长序列、小 batch 也能填满 SM；</text>
  <text x="400" y="572" class="s31" font-size="12">不改变算术强度，只提高占用率。</text>
  <rect x="705" y="475" width="360" height="140" class="orange31"/>
  <text x="885" y="502" text-anchor="middle" class="t31" font-size="15" font-weight="700">③ split-Q warp 划分</text>
  <text x="725" y="528" class="s31" font-size="12">从 split-K 改为 split-Q；</text>
  <text x="725" y="550" class="s31" font-size="12">warp 间不在每次迭代交换中间结果；</text>
  <text x="725" y="572" class="s31" font-size="12">减少共享内存通信和归约开销。</text>
  <text x="55" y="635" class="s31" font-size="12">Roofline 效果：FA1 把 I 从约 64.6 抬到约 516，跨过脊点；FA2 让实际吞吐从“刚过脊点”推进到“接近 GEMM 平台”。</text>
</svg>

&emsp;&emsp;第 2 节的内层循环伪代码里，每次迭代都执行 `O_tilde = O_tilde * corr + ...`。这是一次对 $B_r \times d$ 矩阵的逐元素乘法，走 CUDA Core，不走 Tensor Core。在 A100 上，Tensor Core 的矩阵乘吞吐约是标量运算的 16 倍。即使内循环里只有少量非 matmul 工作，它也会成为延迟瓶颈——因为 GEMM 再快，也要等这批标量重缩放做完才能继续。

FA2 的做法是**改变循环的组织方式，把对 $\tilde{O}$ 的重缩放从“每次迭代都做”变为“只在需要时做”**。FA1 在内层每次迭代都无条件缩放 $\tilde{O}$；FA2 把归一化进一步推迟，只在最大值真正刷新时才触发对 $\tilde{O}$ 的 rescale，并且将 $\ell$ 的更新与 $\tilde{O}$ 的解耦，使绝大多数迭代中的标量工作只剩行最大值比较和指数运算，不再触碰 $B_r \times d$ 的累加器。总 FLOPs 的前向主项 $O(N^2 d)$ 不变，但非 matmul 部分被显著压缩。

**沿序列长度维度并行：让“小 batch、长序列”也能填满 SM**

&emsp;&emsp;FA1 的并行度只来自 batch × head。当 batch 小、head 少时（长上下文训练和推理的典型形态），可用的 thread block 数填不满 A100 的 108 个 SM，大量 SM 闲置。FA2 **在 Q 的序列长度维度上增加了一层并行**：不同的 Q 块可以分配到不同的 thread block，即使只有一个 head，也能通过序列切分获得足够的并行粒度。这在 Roofline 框架下不改变算术强度，但把“可并行的计算”真正铺到了硬件上，直接提升 SM 占用率。

**warp 间工作划分：减少共享内存往返**

&emsp;&emsp;FA1 在 thread block 内部沿 K/V 维度切分 warp（split-K），多个 warp 各算一部分 K/V，再跨 warp 归约。跨 warp 归约必须经过共享内存，带来额外的 SRAM 读写。FA2 把 warp 间的划分从 split-K 改成 **split-Q**：每个 warp 负责 Q 块的不同行，独立完成该行的 softmax 统计和 $\tilde{O}$ 累加，warp 之间不需要在每次迭代中交换中间结果。这削减了共享内存通信，也让每个 warp 的 $\tilde{O}$ 累加器真正留在寄存器里。

&emsp;&emsp;三项改动叠加，FA2 在 A100 上达到 **50%–73% 的理论峰值**，端到端训练达到 225 TFLOPs/s（72% MFU），约为 FA1 的 2 倍。在 Roofline 图上，FA2 没有移动脊点的位置，也没有改变 $I \approx 516$ 这个事实——它做的是让 compute-bound 区间里的执行效率从“跨过脊点但跑不满”推进到“几乎和 GEMM 一样满”。

---

## 4 FlashAttention-3：在 Hopper 上把“搬运”和“计算”真正拆开

&emsp;&emsp;FA2 在 A100 上已经把 compute-bound 利用率做到接近 GEMM，但同一套代码搬到 H100 上只达到 **约 35% 利用率**，而 H100 上调优好的 GEMM 能到 80%–90%。原因不在算法，在**执行模型**：FA2 仍然是同步的——加载完 K/V 块，`__syncthreads()`，然后算 QKᵀ，再做 softmax，再算 PV，再进入下一块。TMA 和 Tensor Core 在 Hopper 上是可以异步执行的硬件，FA2 没有用它们，搬运和计算在时间上仍然是串行的。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261026_1a5c5d.svg)

**TMA：把地址计算和搬运从线程指令中剥离**

&emsp;&emsp;FA2 用 `cp.async` 加载 K/V 块，发起线程需要手写地址计算、处理边界。FA3 用 Hopper 的 **TMA（Tensor Memory Accelerator）**：一条 `cp.async.bulk.tensor` 指令由单个线程提交，硬件 descriptor 完成整块 2D tile 的搬移、越界填充和 swizzle，发起线程不阻塞，立刻可以去做别的事。加载不再消耗计算 warp 的寄存器，也不再占用它们的执行槽。

**WGMMA：矩阵乘不再经过寄存器中转**

&emsp;&emsp;FA2 的 `mma.sync` 是 warp 级同步指令，操作数必须先从共享内存用 `ldmatrix` 加载到寄存器，再送入 Tensor Core。FA3 使用 Hopper 的 **WGMMA**（`wgmma.mma_async`）：以 warpgroup（128 线程）为粒度，**直接从共享内存读取操作数**，发起后不阻塞，硬件在后台完成矩阵乘。QKᵀ 和 PV 两个 GEMM 都不再需要 `ldmatrix` 的寄存器往返，共享内存到 Tensor Core 的路径被缩短。

**Warp specialization：让搬运和 softmax 在不同的 warp 上同时发生**

&emsp;&emsp;这是 FA3 结构上最关键的变化。一个 thread block 的 warpgroup 被分为两类：

- **生产者**：只负责用 TMA 把下一块 K/V 搬进多级 SMEM buffer，填满 pipeline，不参与计算；
- **消费者**：只负责从 SMEM 取数、发 WGMMA、做 online softmax，不直接访问 HBM。

&emsp;&emsp;生产者发出 TMA 后不等待，消费者在上一块数据上继续算 GEMM 和 softmax。同时，FA3 还把 **GEMM 和 softmax 在消费者内部交错**：GEMM-I（QKᵀ）的结果一出来，softmax 就开始处理，不必等整个 block 的 GEMM 全部结束再统一做；softmax 产出的 $P$ 立刻送给 GEMM-II（PV）。两条流水在时间上错开，Tensor Core 和 CUDA Core 不再互相等。

**FP8 与“不协调处理”：低精度的收益要配得上误差**

&emsp;&emsp;Hopper 的 FP8 Tensor Core 吞吐是 FP16 的两倍，但 softmax 的指数运算走 SFU，吞吐不随精度提升而翻倍。如果直接把 FP8 套进 attention，GEMM 变快了，softmax 却原地不动，整体收益被吃掉；同时 FP8 的数值范围窄，直接量化 Q/K 会放大离群值误差。

&emsp;&emsp;FA3 的处理是 **block quantization + incoherent processing**：对 Q 和 K 按块做缩放量化，并在量化前施加一个 Hadamard 旋转，把离群值“摊匀”，使 FP8 的表示误差显著下降。论文报告 FP8 路径相对基线 FP8 attention 数值误差降低 **2.6 倍**。效果上，FP8 FA3 在 H100 上达到 **约 1.2 PFLOPS**，而 FP16 路径达到 **约 740 TFLOPS（75% 利用率）**，相对 FA2 在 H100 上提升 **1.5–2.0 倍**，反向提升 **1.5–1.75 倍**。

&emsp;&emsp;从 Roofline 的角度看，FA3 没有改变算法的算术强度下界，它做的是**把硬件异步能力兑现为实际吞吐**：FA2 在 H100 上只跑了脊点以上区间的一小部分，FA3 把这个区间的利用率从 35% 拉到 75%，同时用 FP8 把 Tensor Core 的天花板本身提高了一档。

---

## 5 FlashAttention-4：当 Blackwell 的瓶颈不再是“算得不够快”

&emsp;&emsp;FA3 的出发点是 H100 上 Tensor Core 和搬运单元之间的同步等待。Blackwell（B200/GB200）把这个问题推到了一个新阶段：**Tensor Core 吞吐翻倍，但共享内存带宽、SFU（指数单元）和通用 ALU 基本没有同比例扩展**。从 Hopper 到 Blackwell，BF16 张量核心吞吐增加约 2.25 倍，但 SFU 数量和 SMEM 带宽几乎不变。

&emsp;&emsp;这意味着 attention 的瓶颈在 Blackwell 上发生了**结构性迁移**：前向的瓶颈从 GEMM 转向 **softmax 指数运算**（SFU 不够快），反向的瓶颈转向 **共享内存流量**（SMEM bandwidth 不够）。FA4 的设计不再只是“把 GEMM 跑满”，而是**同时降低非 GEMM 路径上的资源消耗，并让 GEMM 与这些非 GEMM 工作深度重叠**。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 650" width="100%" role="img" aria-label="Blackwell 瓶颈迁移与 FA4 对策">
  <title>图 5-1 Blackwell 的不对称扩展：Tensor Core 翻倍，SFU/SMEM 不翻倍；FA4 逐一移开新短板</title>
  <desc>左侧条形图对比 Hopper 到 Blackwell 的相对吞吐：Tensor Core 约 2.25 倍，SFU 约 1 倍，SMEM 带宽约 1 倍，FMA 约 1 倍。右侧列出 FA4 的四项对策：FMA 多项式指数绕开 SFU、条件 rescale、TMEM 存中间结果、2-CTA MMA 共享操作数。底部标注前向和反向的新瓶颈。</desc>
  <defs>
    <marker id="arrow51" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#334155"/>
    </marker>
    <style>
      .t51{font-family:Arial,'Microsoft YaHei',sans-serif;fill:#0f172a}
      .s51{font-family:Arial,'Microsoft YaHei',sans-serif;fill:#475569}
      .b51{fill:#dbeafe;stroke:#2563eb;stroke-width:1.4;rx:6}
      .r51{fill:#fee2e2;stroke:#dc2626;stroke-width:1.4;rx:6}
      .g51{fill:#dcfce7;stroke:#16a34a;stroke-width:1.4;rx:6}
      .o51{fill:#ffedd5;stroke:#ea580c;stroke-width:1.4;rx:6}
    </style>
  </defs>
  <rect x="0" y="0" width="1120" height="650" fill="#ffffff"/>
  <text x="30" y="36" class="t51" font-size="24" font-weight="700">图 5-1 FA4 面对的不对称扩展：Tensor Core 翻倍，SFU 与 SMEM 几乎不动</text>
  <text x="30" y="62" class="s51" font-size="14">Blackwell 上 attention 的瓶颈从“HBM 字节”迁移到“Tensor Core 周围的功能单元排队”。</text>

  <rect x="30" y="90" width="520" height="330" fill="#f8fafc" stroke="#cbd5e1"/>
  <text x="50" y="120" class="t51" font-size="17" font-weight="700">相对吞吐（Hopper = 1.0）</text>
  <text x="50" y="148" class="s51" font-size="12">条形长度表示 Blackwell/Hopper 的相对资源吞吐</text>
  <g class="s51" font-size="12">
    <text x="55" y="185">Tensor Core</text>
    <rect x="160" y="170" width="300" height="22" fill="#e2e8f0" stroke="#94a3b8"/>
    <rect x="160" y="170" width="270" height="22" fill="#16a34a" opacity="0.85"/>
    <text x="440" y="186" class="t51" font-size="12" font-weight="700">2.25×</text>
    <text x="55" y="235">SFU / exp</text>
    <rect x="160" y="220" width="300" height="22" fill="#e2e8f0" stroke="#94a3b8"/>
    <rect x="160" y="220" width="120" height="22" fill="#dc2626" opacity="0.85"/>
    <text x="290" y="236" class="t51" font-size="12" font-weight="700">≈1.0×</text>
    <text x="55" y="285">SMEM 带宽</text>
    <rect x="160" y="270" width="300" height="22" fill="#e2e8f0" stroke="#94a3b8"/>
    <rect x="160" y="270" width="120" height="22" fill="#dc2626" opacity="0.85"/>
    <text x="290" y="286" class="t51" font-size="12" font-weight="700">≈1.0×</text>
    <text x="55" y="335">FMA / ALU</text>
    <rect x="160" y="320" width="300" height="22" fill="#e2e8f0" stroke="#94a3b8"/>
    <rect x="160" y="320" width="120" height="22" fill="#ea580c" opacity="0.85"/>
    <text x="290" y="336" class="t51" font-size="12" font-weight="700">≈1.0×</text>
  </g>
  <text x="55" y="380" class="s51" font-size="12">结果：GEMM 更快，但 softmax 的 exp 仍排在 SFU；</text>
  <text x="55" y="400" class="s51" font-size="12">反向的中间结果仍压在 SMEM 带宽上。</text>

  <rect x="580" y="90" width="510" height="330" fill="#f8fafc" stroke="#cbd5e1"/>
  <text x="600" y="120" class="t51" font-size="17" font-weight="700">FA4 的四项对策</text>
  <rect x="605" y="140" width="220" height="70" class="b51"/>
  <text x="715" y="164" text-anchor="middle" class="t51" font-size="13" font-weight="700">① FMA 多项式 exp</text>
  <text x="715" y="184" text-anchor="middle" class="s51" font-size="11">把指数从 SFU 迁到 FMA；</text>
  <text x="715" y="200" text-anchor="middle" class="s51" font-size="11">绕开前向 SFU 排队。</text>
  <rect x="845" y="140" width="220" height="70" class="g51"/>
  <text x="955" y="164" text-anchor="middle" class="t51" font-size="13" font-weight="700">② 条件 rescale</text>
  <text x="955" y="184" text-anchor="middle" class="s51" font-size="11">仅当 m 刷新时缩放；</text>
  <text x="955" y="200" text-anchor="middle" class="s51" font-size="11">砍掉约 90% 无效重缩放。</text>
  <rect x="605" y="230" width="220" height="70" class="o51"/>
  <text x="715" y="254" text-anchor="middle" class="t51" font-size="13" font-weight="700">③ Tensor Memory</text>
  <text x="715" y="274" text-anchor="middle" class="s51" font-size="11">中间结果放 TMEM；</text>
  <text x="715" y="290" text-anchor="middle" class="s51" font-size="11">削减反向 SMEM 流量。</text>
  <rect x="845" y="230" width="220" height="70" class="r51"/>
  <text x="955" y="254" text-anchor="middle" class="t51" font-size="13" font-weight="700">④ 2-CTA MMA</text>
  <text x="955" y="274" text-anchor="middle" class="s51" font-size="11">两个 CTA 共享操作数；</text>
  <text x="955" y="290" text-anchor="middle" class="s51" font-size="11">减少 SMEM 重复加载与原子加。</text>
  <text x="605" y="340" class="s51" font-size="12">目标：在 Blackwell 不对称扩展下，让 BF16 仍达到约 71% 利用率；</text>
  <text x="605" y="360" class="s51" font-size="12">低精度路径进一步推到 MXFP8 2.85 PF/s、FP4 2.13× BF16。</text>

  <rect x="30" y="445" width="1060" height="175" fill="#ffffff" stroke="#94a3b8"/>
  <text x="50" y="475" class="t51" font-size="17" font-weight="700">瓶颈迁移：从“搬多少字节”到“Tensor Core 周围谁先见底”</text>
  <rect x="55" y="495" width="300" height="100" class="o51"/>
  <text x="205" y="520" text-anchor="middle" class="t51" font-size="14" font-weight="700">前向新瓶颈：SFU</text>
  <text x="75" y="545" class="s51" font-size="12">Tensor Core 2.25×，SFU 不增；</text>
  <text x="75" y="565" class="s51" font-size="12">exp 排队 → FA4 用 FMA 多项式；</text>
  <text x="75" y="585" class="s51" font-size="12">条件 rescale 再削标量工作。</text>
  <rect x="380" y="495" width="300" height="100" class="r51"/>
  <text x="530" y="520" text-anchor="middle" class="t51" font-size="14" font-weight="700">反向新瓶颈：SMEM 带宽</text>
  <text x="400" y="545" class="s51" font-size="12">dQ/dK/dV 中间结果多；</text>
  <text x="400" y="565" class="s51" font-size="12">TMEM 转移部分读写；</text>
  <text x="400" y="585" class="s51" font-size="12">2-CTA MMA 共享操作数。</text>
  <rect x="705" y="495" width="360" height="100" class="g51"/>
  <text x="885" y="520" text-anchor="middle" class="t51" font-size="14" font-weight="700">Roofline 语言</text>
  <text x="725" y="545" class="s51" font-size="12">FA1 解决 HBM 字节；FA2 贴算力平台；</text>
  <text x="725" y="565" class="s51" font-size="12">FA3 兑现异步；FA4 在 compute-bound</text>
  <text x="725" y="585" class="s51" font-size="12">区间内部重新分配功能单元负载。</text>
</svg>

**前向：用 FMA 单元“仿真”指数，绕开 SFU 瓶颈**

&emsp;&emsp;FP32 的 `exp` 在 Hopper/Blackwell 上走 SFU。当 Tensor Core 越来越快，SFU 成为前向的限制。FA4 把指数函数从 SFU 迁移到 **FMA 单元上的多项式近似**——用若干次乘加逼近 `exp`，精度可控，但不再排队等 SFU。同时引入**条件 softmax 重缩放**：只有当新块真正刷新行最大值时才执行 rescale，避免 90% 的不必要重缩放。两者叠加，前向的 softmax 路径被大幅瘦身，Tensor Core 的喂料速度不再被 SFU 拖住。

**反向：Tensor Memory + 2-CTA MMA，削减共享内存流量**

&emsp;&emsp;反向的瓶颈在 SMEM 带宽。FA4 利用 Blackwell 的 **Tensor Memory（TMEM）** 存储中间结果，把一部分原本要落在 SMEM 上的读写转移到 TMEM；同时使用 **2-CTA MMA 模式**：两个 thread block 共享同一组操作数，减少 SMEM 的重复加载，并把反向中 $dQ$ 的归约从大量 atomic add 转为更结构化的更新。这两项硬件特性的直接效果，是把反向的主要压力从 SMEM 带宽上移走。

**全异步 MMA 与更大 tile**

&emsp;&emsp;FA3 的 WGMMA 已经是异步的，但 FA4 在 Blackwell 上进一步利用**完全异步的 MMA 操作和更大的 tile 尺寸**，重新设计前向和反向的软件流水线，让 Tensor Core、softmax 和内存操作之间的重叠更充分。更大的 tile 同时降低了每 FLOP 的 softmax 标量开销和 SMEM 访问次数。

&emsp;&emsp;结果：FA4 在 B200 上 BF16 达到 **约 1605–1613 TFLOPs/s（71% 利用率）**，比 cuDNN 9.13 快 1.3 倍，比 Triton 快 2.7 倍。在低精度扩展上，后续工作把 MXFP8 前向推到 **2.85 PF/s**、反向 **2 PF/s**，FP4 路径在 GB200 上达到 BF16 前向吞吐的 **2.13 倍**。

&emsp;&emsp;FA4 面对的已经不是“HBM 字节太多”的问题——那件事在第 2 节就解决了。它面对的是**跨过脊点之后，compute-bound 区间内部哪些单元在排队**。Blackwell 把 Tensor Core 和 SFU/SMEM 的扩展速度拉开，FA4 的回答是：把指数搬到 FMA、把 rescale 降到最低、把中间结果搬进 TMEM、把两个 CTA 的操作数共享起来。**瓶颈从“搬多少字节”变成了“在 Tensor Core 周围，哪个功能单元最先见底”**，FA4 逐一移开这些新的短板。

---

## 6 fa1 → fa5 在 RTX 3050 上的实测

### 6.1 实验设置：一张 sm_86 小卡 vs 五个不同靶场的版本

&emsp;&emsp;RTX 3050（sm_86，20 SM）：FP16 Tensor Core 峰值约 18 TFLOPS，显存带宽 224 GB/s，脊点 $I_{\text{ridge}} \approx 80$ FLOPs/byte。8 个配置（B4H8/B1H1/B2H4，N=1024–4096，D=32/64/128，含 causal），FP16 输入输出、FP32 累积。

&emsp;&emsp;**关键在于每个版本的设计靶场不同：**

| 版本 | 设计靶场 | 依赖的关键特性 | 3050（sm_86） |
|---|---|---|---|
| **fa1** | Ampere | `__syncthreads` + 手写 FMA | ✅ 原样可跑 |
| **fa2** | Ampere | `cp.async` 双缓冲 | ✅ 可跑 |
| **fa3** | **Hopper** | **TMA + WGMMA + warp specialization** | ❌ 退回 wmma |
| **fa4** | Ampere 化 FA3 | 多级流水线 + 条件 rescale | ⚠️ 仅 rescale 兑现 |
| **fa5** | Ampere 化 FA4 | Split-KV | ✅ 可跑（纯软件调度） |

&emsp;&emsp;fa3 的两大杠杆 TMA（`cp.async.bulk.tensor`）和 WGMMA（`wgmma.mma_async`）都是 sm_90+ 指令，在 sm_86 上**不存在**，只能退回 `wmma::mma_sync` + 同步加载。**3050 上跑出来的 fa3，只是它 Hopper 版本的退化版。**

### 6.2 实测


![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/flash_attention_latency_by_shape.svg)

&emsp;&emsp;**整条 fa1→fa5 只有 1.6–2.7×，且几乎全部来自 fa2→fa3 这一跳。** 逐跳拆：

&emsp;&emsp;**① fa1 vs fa2：双缓冲没赢。** fa1 在 5/8 配置上更快（6.91 vs 7.35）。原因：fa1 的 $B_c=64$、fa2 的 $B_c=32$，内层循环次数翻倍，per-tile 固定开销翻倍；更关键的是**算/存比太低**——D=64 的 tile 读入约 70 ns，FMA 点积约 300–400 ns，算/存比 ~5，`cp.async` 只是把"等"从 `__syncthreads` 挪到了 `cp_async_wait`，重叠窗口太窄。唯一兑现的是 D=128（12.82 vs 14.73，快 13%），算/存比 ~10 才刚好覆盖延迟。

&emsp;&emsp;**② fa2→fa3：2–3×，唯一能拿到的"代差"。** 一次 `mma.m16n16k16` 顶约 256 次标量 FMA，`QKᵀ` 和 `PV` 都走 Tensor Core，CUDA Core 从点积里解放。但 3050 上 MFU 只有 **7.3%**——这 7.3% 是 wmma 在这张卡的天花板，与 fa3 算法设计无关。**TMA 和 WGMMA 在 sm_86 上根本不存在，fa3 真正杀伤力部分完全作废**。

&emsp;&emsp;**③ fa3→fa4→fa5：边际收益不到 15%。** fa3→fa4 的收益几乎全来自**条件 rescale**（跳过 90% 无效 `wmma::store/load`），多级流水线本身几乎不贡献——没有 TMA，Stage 增加只是把等待堆在队列里。fa4→fa5 的 split-KV 是纯软件调度、不依赖硬件，**是唯一能拿到"设计意图 100%"的版本**，但只在并行度不足时兑现：

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/flash_attention_tflops_by_shape.svg)

&emsp;&emsp;**另外两件与硬件无关的收益完整兑现**：因果掩码在分块结构上自然拿到 2 倍（fa1–fa5 满/因果比全部 0.46–0.52）；split-KV 在 B1H1 这类低并行场景拿 15–20%。

### 6.3 结论

| 版本 | 3050 上实际兑现 | 丢掉的 |
|---|---|---|
| **fa1** | ✅ 100% | 无 |
| **fa2** | ⚠️ 双缓冲仅 D=128 兑现 | 算/存比太低 |
| **fa3** | ⚠️ 只拿到 wmma（~7% MFU） | **TMA + WGMMA 完全作废** |
| **fa4** | ⚠️ 条件 rescale 兑现，流水线无效 | 没有 TMA，Stage 无意义 |
| **fa5** | ✅ split-KV 完整兑现（低并行时） | 无 |

&emsp;&emsp;**3050 上 fa1→fa5 只拿到 1.6–2.7×**，几乎全部来自 fa2→fa3（FMA → Tensor Core），其余版本边际收益 < 15%。这不是实现问题，是**硬件规格与设计靶场错配**。

&emsp;&emsp;**fa3 是"最亏"的版本。** 核心创新 TMA 和 WGMMA 是 Hopper 专属（sm_90+），sm_86 上无法编译成目标指令，只能退回 wmma。**3050 上看到的"fa3 加速"不是 FA3 的加速，是 wmma 的加速**——FA3 论文在 H100 上报告的 1.5–2.0× 是 TMA/WGMMA 兑现后的结果，和这里完全是两笔账。

&emsp;&emsp;**能完整兑现的只有三件与架构解耦的事**：Tensor Core 化（wmma 这一档）、因果掩码、split-KV（低并行时）。这恰好印证前五节的核心论点——FA 系列的收益从来不是"算法更聪明"，而是"把每代 GPU 上最先见底的那个资源挪开"。3050 上最先见底的是 FMA 吞吐，所以 fa3 那跳最猛；3050 上没有 Hopper 的异步单元，fa4 的多级流水线就打水漂。**要突破 wmma 的 7% MFU 天花板，路径不是更多版本，而是换硬件**——这正是 FA3/FA4 的 1.5–2× 和 71% MFU 都只在 H100/B200 上报告的原因。

---

## 7 总结


&emsp;&emsp;FlashAttention 的演进可以用一条比第 4 节更长的因果链来概括：

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261026_e25f33.svg)

&emsp;&emsp;**FA1 问的是“字节搬得太多”。** 传统注意力把 $N \times N$ 矩阵物化到 HBM，算术强度只有约 64.6，远低于 A100 的脊点 153、H100 的 295。tiling 让中间矩阵留在 SRAM，online softmax 让分块遍历仍能得到精确 softmax——HBM 字节降约 8 倍，算术强度升到约 516，**算子从 memory-bound 跨入 compute-bound**。

&emsp;&emsp;**FA2 问的是“跨过脊点之后为什么还跑不满”。** FA1 在 A100 上只到 25%–40% 峰值，因为内循环里非 matmul 的 rescale 在 CUDA Core 上排队，并行度只来自 batch × head，warp 间 split-K 又制造了不必要的共享内存往返。FA2 把这三处逐一收掉，达到 50%–73% 峰值、225 TFLOPs/s（72% MFU），**在 compute-bound 区间内逼近 GEMM 的利用率**。

&emsp;&emsp;**FA3 问的是“Hopper 的异步硬件为什么没用上”。** FA2 在 H100 上仍只有 35%，因为同步执行模型让 TMA 和 Tensor Core 互相等。FA3 用生产者–消费者 warp specialization 把搬运、GEMM、softmax 在时间上拆开，用 TMA 和 WGMMA 去掉寄存器中转，用 FP8 + 不协调处理在控制误差的同时把 Tensor Core 天花板抬高。FP16 到 740 TFLOPS / 75%，FP8 到约 1.2 PFLOPS，相对 FA2 前向 1.5–2.0 倍、反向 1.5–1.75 倍，**把硬件异步能力兑现为实际吞吐**。

&emsp;&emsp;**FA4 问的是“Blackwell 把瓶颈搬到了哪里”。** Tensor Core 翻倍，SFU 和 SMEM 不翻倍，前向的瓶颈变成 softmax 指数，反向变成 SMEM 带宽。FA4 用 FMA 上的多项式指数绕开 SFU，用条件 rescale 砍掉 90% 的重缩放，用 TMEM 和 2-CTA MMA 削减反向 SMEM 流量和 atomic add。B200 上 BF16 达到约 1605 TFLOPs/s / 71%，低精度路径进一步推到 MXFP8 2.85 PF/s、FP4 2.13× BF16，**在不对称扩展下把 GEMM 周围的非 GEMM 短板逐一移开**。

&emsp;&emsp;四个版本共享同一个数学：精确 softmax，不近似。变化的只是**每一代硬件上“最先见底的那个资源”**。FA1 见底的是 HBM 带宽，FA2 见底的是 CUDA Core 上的非 matmul 排队和并行度，FA3 见底的是同步模型下闲置的 TMA/WGMMA，FA4 见底的是 SFU 和 SMEM 带宽。FlashAttention 的工程史，就是不断重新识别“哪个单元最先不够用”，然后在不改变数学的前提下，把工作搬到还有余量的单元上去。总的来说优化思路还是GEMM优化那一套。