# KV Cache 原理详解：自回归解码中的键值复用机制

&emsp;&emsp;大语言模型的推理过程是逐 token 串行生成的。这一特性使得朴素实现在每步解码时都要对完整序列重新执行一次前向计算，产生大量冗余。**KV Cache** 是消除该冗余最基础、也最关键的工程手段。

---

## 1. 自回归解码的基本流程

&emsp;&emsp;给定初始 Prompt，模型按如下循环生成文本：

1. 将当前序列输入模型，得到下一个 token 的概率分布；
2. 采样或取 argmax，得到新 token；
3. 将新 token 追加至序列末尾，重复步骤 1，直至遇到结束符或达到长度上限。

&emsp;&emsp;每一步仅输出**一个** token，这是 Decode 阶段串行性的来源。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/kvcache_exanoke_202610272301_svgviewer-output.svg)
**图 1：自回归生成流程。蓝色为原始 Prompt token，橙色为逐步生成的 token。**

&emsp;&emsp;设序列长度在第 $t$ 步为 $t$。若不加任何缓存，第 $t$ 步前向计算的输入长度为 $t$，且其中前 $t-1$ 个 token 与上一步完全相同。

---

## 2. 朴素解码的计算冗余

&emsp;&emsp;以生成 3 个 token 为例，朴素实现的计算分布如下：

| 步 | 输入序列 | 参与计算的 token 数 |
|---|---|---|
| 1 | $[t_1]$ | 1 |
| 2 | $[t_1,t_2]$ | 2 |
| 3 | $[t_1,t_2,t_3]$ | 3 |

&emsp;&emsp;$t_1$ 被计算 3 次，$t_2$ 被计算 2 次。总 token 级计算量为 $1+2+3=6$，而实际只有 3 个 token。


![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/kvcache_exanoke_202610272302_svgviewer-output.svg)
**图 2：朴素实现的重复计算。红色块为被重复计算的历史 token。**

&emsp;&emsp;在真实模型中，每个 token 需经过 $L$ 层 Transformer，每层包含一次完整注意力与前馈网络。冗余计算量随序列长度线性累积，不可忽略。

---

## 3. Self-Attention 的计算结构与可复用性

&emsp;&emsp;设输入隐状态矩阵 $X \in \mathbb{R}^{n \times d}$，其中 $n$ 为序列长度、$d$ 为隐藏维度。每层注意力通过三个线性投影得到：

$$
Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V
$$

&emsp;&emsp;其中 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$。三者的语义分别为：

- **Q（Query）**：当前 token 用于检索的查询向量；
- **K（Key）**：各 token 作为被检索对象的索引向量；
- **V（Value）**：各 token 携带的实际内容向量。

&emsp;&emsp;注意力输出为 $\mathrm{softmax}(QK^\top/\sqrt{d_h})\,V$，即用 Q 与全部 K 的相似度对 V 加权聚合。


![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/kvcache_exanoke_202610272303_svgviewer-output.svg)
**图 3：自注意力中的 Q、K、V 投影。X 的每一行对应一个 token，分别经三个权重矩阵投影。**

&emsp;&emsp;**引理（K/V 的位置稳定性）**：在因果自回归生成中，第 $t$ 步输入序列的前 $t-1$ 个 token 与第 $t-1$ 步完全一致，且因果掩码保证这些位置的隐状态不受后续 token 影响。因此，历史 token 的 $k$ 与 $v$ 在后续所有步中保持不变。据此得到两条结论：

1. **K、V 可缓存**：历史 token 的 $k_i, v_i$ 只需计算一次，之后直接复用；
2. **Q 不可缓存**：每步新增 token 都需要一个全新的 query 去检索历史信息，$Q$ 必须实时计算。

KV Cache 的全部内容即源于此：缓存历史 token 的 K 与 V，避免重复投影。

---

## 4. KV Cache 的机制

&emsp;&emsp;核心策略是以显存换取计算量。每生成一个新 token：

1. 仅对当前 token 计算 $q_t, k_t, v_t$；
2. 将 $k_t, v_t$ 追加至缓存末尾；
3. 以 $q_t$ 与缓存中全部 $K_{1:t}$ 做注意力，对全部 $V_{1:t}$ 加权求和。


![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/kvcache_exanoke_202610272304_svgviewer-output.svg)
**图 4：KV Cache 的存储布局。K 与 V 按序列位置排列，新 token 的 k、v 追加至末尾。**

&emsp;&emsp;值得注意的是，缓存中不包含 Q。query 仅在当前步的注意力计算中使用一次，不具备跨步复用价值。

---

## 5. Prefill 阶段：Prompt 的并行处理

&emsp;&emsp;实际推理分为两个性质迥异的阶段：**Prefill** 与 **Decode**。

&emsp;&emsp;Prefill 发生在收到完整 Prompt 后、开始生成之前。此时模型持有全部 Prompt token，可对其并行计算 $Q$、$K$、$V$，属于计算密集型（compute-bound）操作。


![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/kvcache_exanoke_202610272305_svgviewer-output.svg)
**图 5：Prefill 阶段。Prompt 整体并行前向，Q、K、V 同时得出，K 与 V 写入缓存。**

&emsp;&emsp;该阶段并行度高、算术强度大，GPU 计算单元利用率充分，通常是耗时的主要部分之一（与 Prompt 长度成正比）。其输出除首 token 外，还包括一份已填充的 KV Cache，供后续 Decode 复用。

---

## 6. Decode 阶段：单 token 增量计算

&emsp;&emsp;Prefill 完成后进入 Decode 阶段。此后模型每步仅处理一个新 token，属于访存密集型（memory-bound）操作。

&emsp;&emsp;设缓存中已有 $k_{1:3}, v_{1:3}$，当前待处理 token 为 $t_4$：

1. 对 $t_4$ 计算 $q_4, k_4, v_4$；
2. 将 $k_4, v_4$ 追加至缓存，得到 $K_{1:4}, V_{1:4}$；
3. 计算 $\mathrm{softmax}(q_4K_{1:4}^\top/\sqrt{d_h})\,V_{1:4}$，得到输出并预测 $t_5$。


![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/kvcache_exanoke_202610272306_svgviewer-output.svg)
**图 6：Decode 阶段。每步仅计算单个 token 的 Q、K、V，K/V 追加进缓存，Q 与全部历史 K 做注意力。**

&emsp;&emsp;需要强调的是，Decode 每步的注意力仍需读取全部历史 K、V，因此其开销随序列长度线性增长，但增长的是**访存**而非**计算**——这正是 Decode 阶段受显存带宽制约的根源。

---

## 7. 复杂度与显存开销

### 7.1 计算复杂度

&emsp;&emsp;记序列长度为 $n$、隐藏维度为 $d$、层数为 $L$。单层单步的主要开销来自线性投影与注意力：

| 项目 | 无 KV Cache（第 $t$ 步） | 有 KV Cache（第 $t$ 步） |
|---|---|---|
| QKV 投影 | $O(t \cdot d^2)$ | $O(d^2)$ |
| 注意力打分与聚合 | $O(t^2 \cdot d)$ | $O(t \cdot d)$ |
| 输出投影 / FFN | $O(t \cdot d^2)$ | $O(d^2)$ |

&emsp;&emsp;对 $t = 1 \dots n$ 求和，完整生成 $n$ 个 token 的总开销为：

$$
\text{无缓存}: O(n^2 d^2 + n^3 d)
\qquad\Longrightarrow\qquad
\text{有缓存}: O(n\,d^2 + n^2 d)
$$

&emsp;&emsp;当 $n \gg d$ 时，注意力项占主导，规模从 $O(n^3)$ 降至 $O(n^2)$；当 $n \ll d$ 时，线性层项占主导，规模从 $O(n^2d^2)$ 降至 $O(nd^2)$。两种情形下 KV Cache 均带来数量级上的改善。


![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/kvcache_exanoke_202610272307_svgviewer-output.svg)
**图 7：累计 token 级计算次数示意。无缓存时呈二次增长，有缓存时呈线性增长；真实 FLOPs 还需乘上每 token 的投影与注意力开销。**

### 7.2 显存开销

&emsp;&emsp;KV Cache 的代价体现在显存占用上。设层数为 $L$、注意力头数为 $n_h$、每头维度 $d_h$、序列长度为 $S$、每元素字节数为 $b$，则缓存大小为：

$$
M_{\text{cache}} = 2 \times L \times n_h \times S \times d_h \times b
$$

&emsp;&emsp;其中系数 $2$ 对应 K 与 V 各一份。以 7B 级模型（$L=32$、$n_h=32$、$d_h=128$、FP16 即 $b=2$）、序列长度 $S=4096$ 为例：

$$
2 \times 32 \times 32 \times 4096 \times 128 \times 2 \approx 2\ \text{GiB}
$$

&emsp;&emsp;该数值为**单条请求**的缓存占用。并发服务时，缓存总量随并发数与序列长度线性叠加，往往成为显存容量的主要消耗项，也是限制 batch size 与上下文长度的关键因素。

&emsp;&emsp;这一矛盾直接催生了后续一系列优化方向：以 MQA / GQA 缩减 KV 头数以压缩缓存体积，以 PagedAttention 消除分页式预分配带来的内部碎片。相关内容不在本文范围内。

---

## 8. 小结

1. 大模型推理为自回归逐 token 生成，朴素实现在每步重复计算整个序列；
2. 因果掩码保证了历史 token 的隐状态不受后续 token 影响，因此其 $K$、$V$ 在后续步骤中保持不变；
3. KV Cache 缓存历史 $K$、$V$，每步仅计算新 token 的 $q, k, v$，并将 $k, v$ 追加至缓存；
4. 推理分为 Prefill（并行处理 Prompt，计算密集）与 Decode（逐 token 生成，访存密集）两个阶段；
5. 计算开销由 $O(n^2d^2 + n^3d)$ 降至 $O(nd^2 + n^2d)$；
6. 代价是显存占用随序列长度与并发数线性增长。

KV Cache 是 LLM 推理系统中最基础的优化之一，也是理解后续调度、分页与压缩技术的必要前提。
