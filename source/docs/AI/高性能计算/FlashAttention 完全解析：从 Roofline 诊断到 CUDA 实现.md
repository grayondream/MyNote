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

## 3. CUDA 实现：从 Kernel 骨架到硬件流水线
### 3.1 前向 Kernel：Ampere 基线实现

```cuda
// Q, K, V, O: [B, H, N, D]，FP16 存储
// softmax_lse: [B, H, N]，FP32
template <int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flash_fwd_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float* __restrict__ softmax_lse,
    int B, int H, int N, float scale, bool is_causal)
{
    // grid.x = Q 块数，grid.y = B * H
    const int q_block = blockIdx.x;
    const int bh      = blockIdx.y;
    const int b = bh / H, h = bh % H;

    // 动态共享内存：Q_tile + K_tile + V_tile + S/P 临时区
    extern __shared__ char smem_raw[];
    half* sQ = reinterpret_cast<half*>(smem_raw);
    half* sK = sQ + BLOCK_M * HEAD_DIM;
    half* sV = sK + BLOCK_N * HEAD_DIM;
    float* sP = reinterpret_cast<float*>(sV + BLOCK_N * HEAD_DIM);

    // 运行态：m, l, O_tilde
    float m[BLOCK_M / 32];      // 每个 warp 负责若干行
    float l[BLOCK_M / 32];
    float acc[BLOCK_M / 32][HEAD_DIM / 4];   // 寄存器中的 Õ

    // 一次性加载 Q_tile
    load_q_tile(sQ, Q, b, h, q_block, N);
    init_running_state(m, l, acc);            // m=-inf, l=0, acc=0

    const int kv_start = 0;
    const int kv_end   = is_causal ? (q_block + 1) * BLOCK_M : N;

    for (int kv_block = kv_start; kv_block < kv_end; kv_block += BLOCK_N) {
        cp_async_load_kv(sK, sV, K, V, b, h, kv_block, N);   // cp.async
        cp_async_commit_and_wait();
        __syncthreads();

        compute_qk_and_softmax_update<BLOCK_M, BLOCK_N, HEAD_DIM>(
            sQ, sK, sP, m, l, acc, scale, is_causal, q_block, kv_block);
        __syncthreads();

        accumulate_pv<BLOCK_M, BLOCK_N, HEAD_DIM>(sP, sV, acc);
        __syncthreads();
    }

    // 循环外唯一一次归一化与写回
    normalize_and_store(O, softmax_lse, acc, l, b, h, q_block, N);
}
```

&emsp;&emsp;这个骨架里每个 block 内做三件事：加载 Q（一次）、循环加载 K/V 并更新运行态（$N/B_c$ 次）、最后归一化写回（一次）。`sQ`、`sK`、`sV`、`sP` 全部来自动态共享内存，大小在 launch 时按 `BLOCK_M`、`BLOCK_N`、`HEAD_DIM` 计算。**`S` 和 `P` 没有对应的全局内存缓冲区**——这是整个实现与标准注意力最根本的区别。

&emsp;&emsp;**内层循环：online softmax 的 CUDA 展开。** 内层循环是 2.1 节伪代码的直接翻译，但每一步都要落到具体的 CUDA 原语上。第一步 $S_{ij} = Q_i K_j^\top$，用 `mma.sync` 的 `m16n8k16` 指令，Q 作为 A 操作数、K 作为 B 操作数，累加器是 FP32：

```cuda
// 每个 warp 负责 BLOCK_M/4 行，用 ldmatrix 从共享内存取 fragment
uint32_t a_frag[4], b_frag[2];
ldmatrix_x4(a_frag, sQ + warp_row * HEAD_DIM);
ldmatrix_x2(b_frag, sK + warp_col * HEAD_DIM);

float s_frag[4] = {0.f, 0.f, 0.f, 0.f};
mma_sync_m16n8k16(s_frag, a_frag, b_frag, s_frag);   // D = A * B + C
```

&emsp;&emsp;`s_frag` 是 $S_{ij}$ 的一个 16×8 分片，留在寄存器中，不写回共享内存。第二步做 online softmax 更新，每个 warp 对自己的行分片做行最大值、修正因子和指数和。这里是 FA2 的关键决策——**重缩放推迟**：

```cuda
// 行最大值：warp 内先做 shfl 归约，再跨 warp 用共享内存合并
float m_block = warp_rowmax(s_frag);
float m_new   = fmaxf(m_running, m_block);
float alpha   = __expf(m_running - m_new);      // 修正因子

// 更新 ℓ 和 Õ
l_running = l_running * alpha + warp_rowsum(__expf(s_frag - m_new));
for (int i = 0; i < ACC_SIZE; ++i)
    acc[i] *= alpha;                             // 重缩放历史累积

// 转成 FP16，准备作为 PV 的 A 操作数
half p_frag[4];
for (int i = 0; i < 4; ++i)
    p_frag[i] = __float2half(__expf(s_frag[i] - m_new));

m_running = m_new;
```

&emsp;&emsp;注意 `acc[i] *= alpha` 只在内层循环里执行，而除法 `acc / l` 被完全推迟到循环外。若 `alpha == 1`（最大值未被刷新，绝大多数迭代如此），这次乘法可以跳过——编译器或手写的分支可以把它优化掉。第三步 $\tilde{O} \mathrel{+}= P_{ij} V_j$，`p_frag` 直接作为下一次 `mma.sync` 的 A 操作数，`sV` 作为 B 操作数，累加到 `acc`：

```cuda
uint32_t p_a[2], v_b[2];
pack_p_fragment(p_a, p_frag);                   // 原地重解释为 MMA A fragment
ldmatrix_x2(v_b, sV + warp_col * HEAD_DIM);
mma_sync_m16n8k16(acc, p_a, v_b, acc);          // 累加，不写回共享内存
```

&emsp;&emsp;三步串起来，一次迭代的 HBM 访问只有 `cp.async` 加载的 $K_j, V_j$，`S_{ij}` 和 `P_{ij}` 全程在寄存器和共享内存之间流转。

&emsp;&emsp;**共享内存布局与 bank conflict。** Tensor Core 对操作数布局有严格要求。`ldmatrix` 从共享内存加载分片时，如果同一 warp 内的线程访问了同一个 bank 的不同地址，就会发生 bank conflict。标准行主序布局下，行宽 `HEAD_DIM` 往往不是 32 的倍数，跨行访问直接冲突。最直接的缓解是**行填充**：每行末尾插 padding。更高效的是 **XOR swizzle**，对列索引施加基于行索引的 XOR 变换：

```cuda
// 行主序 tile 的物理列索引：col ^ ((row & mask) << shift)
__device__ inline int swizzled_col(int row, int col) {
    return col ^ ((row & 0x7) << 3);   // 以 8 行为一组，每组错开 8 个 bank
}
```

&emsp;&emsp;`cp.async` 写入和 `ldmatrix` 读取必须使用同一套 swizzle 映射。在 Ampere 及更新架构的 FP16 kernel 中，XOR swizzle 已经取代行填充，在不增加存储开销的前提下把规则的多路冲突转化为无冲突访问。另一个 CUDA 特有的优化是 **$P$ 的重解释**：`p_frag` 不写回共享内存再用 `ldmatrix` 读入，而是在寄存器中直接重解释为 MMA 的 A fragment（`pack_p_fragment`）。这要求 $S$ 的累加器布局与后续 GEMM 的 A 操作数格式兼容，省掉了 $P$ 的一次共享内存往返。

&emsp;&emsp;**因果掩码：三段循环。** 自回归训练中注意力矩阵是下三角的，上三角完全被掩码。分块后可以按块判断，把内层循环拆成三段：

```cuda
// 完全位于对角线以下的块：跳过所有掩码判断
for (int kv = 0; kv < q_block * BLOCK_M; kv += BLOCK_N) {
    // 直接计算，不施加 mask
}

// 与对角线相交的块：施加三角掩码
for (int kv = q_block * BLOCK_M; kv < (q_block + 1) * BLOCK_M; kv += BLOCK_N) {
    // 计算时对 s_frag 施加 mask
}

// 完全位于对角线以上的块：根本不进入循环
```

&emsp;&emsp;第三段不进入循环，意味着 K/V 的总读取量减半。对因果训练，这一项单独带来约 $2\times$ 的加速——它不改变数学，只是在 tiling 结构上自然获得的收益。

### 3.2 Hopper 特化、反向传播与实现路径

&emsp;&emsp;**Hopper 特化：TMA、WGMMA 与 warp specialization。** Ampere 上的基线实现用 `cp.async` + `mma.sync` 已经接近 GEMM 效率，但在 Hopper 上只用到约 35% 的峰值算力。瓶颈在于同步执行模型：加载 warp 发出 `cp.async` 后要等 barrier，计算 warp 才能开始 MMA，搬运和计算无法真正重叠。Hopper 的三项硬件特性改变了这个约束，CUDA 代码的结构也随之变化。

&emsp;&emsp;**TMA** 把搬运从线程指令中剥离。一条 `cp.async.bulk.tensor.2d` 由单个线程提交，硬件 DMA 引擎完成整块 2D tile 的搬移，地址计算、越界填充和共享内存 swizzle 全部在 descriptor 中声明式配置：

```cuda
// 初始化 TMA descriptor（host 端）
CUtensorMap k_desc, v_desc;
cuTensorMapEncodeTiled(&k_desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                       4, K_ptr, dims, strides, box_dims, ...);

// kernel 内：单线程提交，零寄存器开销
if (threadIdx.x == 0) {
    cp_async_bulk_tensor_2d(&sK, &k_desc, coords_k, &barrier);
}
```

&emsp;&emsp;加载线程不再消耗寄存器算地址，也不占用计算 warp 的执行槽。**WGMMA** 以 Warp Group（128 线程）为粒度执行矩阵乘，直接消费共享内存中的操作数，不需要 `ldmatrix` 的寄存器中转：

```cuda
// WGMMA 示意：A 和 B 都来自共享内存，accumulator 在寄存器
wgmma_m64n128k16(acc, sQ_desc, sK_desc, acc, /*scale_D=*/1, /*scale_A=*/1, /*scale_B=*/1);
```

&emsp;&emsp;**Warp specialization** 把线程块内的 warp 划分为生产者和消费者。生产者 warp 持续通过 TMA 发出下一块 K/V 的加载请求，消费者 warpgroup 执行 WGMMA 和 softmax 更新：

```cuda
if (warp_id < NUM_PRODUCER_WARPS) {
    // 生产者：循环发出 TMA 请求
    for (int kv = 0; kv < num_kv_blocks; ++kv) {
        cp_async_bulk_tensor_2d(sK_buf[kv % STAGES], &k_desc, ...);
        cp_async_bulk_tensor_2d(sV_buf[kv % STAGES], &v_desc, ...);
        mbarrier_arrive_expect_tx(&bar[kv % STAGES], BYTES_PER_STAGE);
    }
} else {
    // 消费者：WGMMA + softmax
    for (int kv = 0; kv < num_kv_blocks; ++kv) {
        mbarrier_wait(&bar[kv % STAGES]);
        wgmma_qk(acc_s, sQ, sK_buf[kv % STAGES]);
        online_softmax_update(acc_s, m, l, acc_o);
        wgmma_pv(acc_o, sP, sV_buf[kv % STAGES]);
    }
}
```

&emsp;&emsp;生产者与消费者通过硬件 barrier 同步，搬运和 Tensor Core 计算全程重叠，消除了 FA2 中“加载—等待—计算—再加载”的串行气泡。Colfax 的 CUTLASS Hopper 实现将这一结构完整落地，核心工程挑战是 **layout conformance**：TMA 加载布局、WGMMA 期望的 A/B operand 布局、FP32 accumulator 布局三者并不天然一致，需要用 CUTLASS 的 `Layout` 和 `Tensor` 抽象系统化地表达和转换。该实现在 H100 PCIe 上相比 Ampere 优化的 FA2 版本，观测到 20%–50% 的 FLOPs/s 提升。

&emsp;&emsp;**反向传播的 CUDA 实现。** 反向 kernel 的复杂度显著高于前向，核心约束是**不能保存 $P$**。前向只写回 $O$ 和每行的 LSE（$N$ 个 FP32 标量），反向时从 $Q$、$K$ 和 LSE 在 SRAM 中按块重算 $S$ 与 $P$。kernel 签名直接反映这一设计——**没有 $P$ 的指针**：

```cuda
__global__ void flash_bwd_kernel(
    const half* dO, const half* Q, const half* K, const half* V, const half* O,
    const float* softmax_lse, const float* D,   // D_i = rowsum(dO_i ⊙ O_i)
    half* dQ, half* dK, half* dV,
    int B, int H, int N, float scale, bool is_causal)
```

&emsp;&emsp;进入主反向循环之前，有一个独立的预处理 kernel 计算 $D_i = \operatorname{rowsum}(dO_i \odot O_i)$，供后续所有涉及 $dS$ 的计算复用；如果 LSE 需要梯度，还会调整为 $D'_i = D_i - dLSE_i$，主反向 kernel 结构不变。主反向 kernel 的并行划分决定了梯度归约方式。$(i,j)$ 版本让每个 block 拥有一个 $(Q_i, K_j)$ tile，同时计算 $dQ_i$、$dK_j$、$dV_j$ 的局部贡献：

```cuda
// 重算 S_ij 和 P_ij（不读 HBM 中的 P）
wgmma_qk(acc_s, sQ, sK);
online_recompute_p(acc_s, p, softmax_lse_i);    // 用 LSE 重建 P

// 计算 dV_j 和 dP
wgmma_pv(dv_acc, p_T, dO);                      // dV_j += P_ij^T · dO_i
wgmma_dp(dp_acc, dO, sV_T);                     // dP_ij = dO_i · V_j^T

// 计算 dS_ij，再得到 dQ_i 和 dK_j
compute_ds(ds, dp_acc, p, D_i);                 // dS = (dP - D) ⊙ P
wgmma_dq(dq_acc, ds, sK);                       // dQ_i += dS_ij · K_j
wgmma_dk(dk_acc, ds_T, sQ);                     // dK_j += dS_ij^T · Q_i
```

&emsp;&emsp;$dK_j$ 和 $dV_j$ 沿 $i$ 维跨 block 归约，$dQ_i$ 沿 $j$ 维归约。FA3 的确定性模式用信号量按顺序对 $dQ$ 原地更新，避免 FP32 原子加的累积误差和非确定性。重计算的代价是反向 FLOPs 约翻倍，但激活显存从 $O(N^2)$ 降到 $O(N)$，而且反向 kernel 不需要从 HBM 读取 $P$——用 SRAM 内的便宜算力换掉了 HBM 上的昂贵带宽。

## 4 总结

&emsp;&emsp;FlashAttention 的加速逻辑可以压缩成一条因果链：传统注意力把 \(N\times N\) 矩阵物化到 HBM，使算子落在 memory-bound 区间（算术强度约 64.6，远低于各代 GPU 的脊点）→ 唯一有效的杠杆是减少 HBM 字节 → tiling 让中间矩阵留在 SRAM，online softmax 让 softmax 能在分块遍历中精确完成，两者合起来把三个 kernel 融成一个 → HBM 字节降低约 8 倍，算术强度升到约 516，算子跨过脊点进入 compute-bound → 多做的那些浮点运算落在 SRAM 里、落在原本等带宽的空闲周期里，所以“算得更多反而更快”。
