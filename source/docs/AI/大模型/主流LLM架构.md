# LLM架构演进

## 1 前言

&emsp;&emsp;自 2017 年 Transformer 提出以来，大语言模型（Large Language Model, LLM）领域经历了从序列建模到通用生成的范式跃迁。Transformer 以自注意力机制替代循环与卷积，摆脱了 RNN 的串行计算瓶颈，使训练可以大规模并行，同时具备更强的长程依赖建模能力。此后，BERT 开启了 encoder-only 的理解路线，GPT 系列则沿 decoder-only 的生成路线不断放大，最终在 GPT-3、PaLM、Chinchilla、LLaMA 等模型上验证了“模型规模、数据规模与算力投入”共同驱动能力增长的Scaling Law。

&emsp;&emsp;LLM 架构发展至今，已不只是单一 Transformer 结构的重复堆叠，而是在表达能力、训练稳定性、推理效率、上下文长度与部署成本之间持续权衡的结果。早期模型主要关注“能不能训练起来”和“能不能生成通顺文本”；随着参数规模进入百亿、千亿乃至万亿级别，架构设计的重心逐渐转向如何降低单位计算成本、如何稳定训练超深网络、如何支持更长上下文、如何从稠密计算走向稀疏激活，以及如何将语言模型扩展为多模态、可推理、可调用工具的通用智能体。

&emsp;&emsp;从整体脉络看，LLM 架构演进大致沿着几条主线展开：从 RNN/CNN 到 Transformer；从 encoder-decoder、encoder-only 到 decoder-only；从稠密模型到混合专家（MoE）；从绝对位置编码到 RoPE、ALiBi 等相对位置方案；从多头注意力（MHA）到多查询注意力（MQA）、分组查询注意力（GQA）和 MLA；从 Post-LN 到 Pre-LN、RMSNorm；从 ReLU/GELU 到 SwiGLU；从全量注意力到 FlashAttention、滑动窗口、稀疏注意力和线性注意力；从单模态文本建模到图文、音视频多模态统一建模。这些变化并非彼此孤立，而是共同服务于一个目标：在有限算力下，让模型获得更强的通用表示与生成能力。

&emsp;&emsp;为了更清晰地理解 LLM 的架构演进，本文从模型架构与关键组件两个层面展开。模型架构关注整体范式，例如 encoder-only、encoder-decoder、decoder-only、MoE、多模态 Transformer 等；关键组件关注局部设计，例如注意力机制、位置编码、归一化、激活函数、前馈网络、残差连接、路由策略与并行方式等。通过这两条线索，可以看到 LLM 如何从机器翻译中的注意力机制，逐步演化为支撑对话、推理、多模态与 Agent 应用的通用智能底座。

## 2 基础组件与方法

&emsp;&emsp;在进入具体模型之前，需要先理解支撑 LLM 架构演进的基础组件与方法。Transformer 之所以能够取代 RNN 和 CNN 成为序列建模的统一基座，并非依赖某一项孤立的创新，而是由注意力机制、位置编码、残差连接、归一化、前馈网络、分词器、训练目标、优化器、后训练对齐、推理优化、长上下文技术、多模态架构、缩放规律和分布式训练等多个组件协同作用的结果。这些组件的设计选择——注意力的头数与分组方式、位置编码的形式、归一化的位置与类型、激活函数的形状、MoE 的路由策略、后训练的对齐方法——在后续的模型演进中被反复调整和替换，形成了现代 LLM 的“标准配方”。本节将文章中涉及的所有基础组件与方法按类别展开，每个组件单独成节，为后续各模型章节提供完整的技术背景。

---

### 2.1 注意力机制

#### 2.1.1 点积注意力

&emsp;&emsp;点积注意力（Dot-Product Attention）是最常用的一类注意力机制。它通过计算 **Query** 与 **Key** 之间的点积来衡量二者相关性，再经过 Softmax 归一化得到注意力权重，最后对 **Value** 进行加权求和。给定查询矩阵 $Q \in \mathbb{R}^{n \times d_k}$、键矩阵 $K \in \mathbb{R}^{m \times d_k}$、值矩阵 $V \in \mathbb{R}^{m \times d_v}$，点积注意力的基本形式为：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top)V
$$

&emsp;&emsp;若只看单个查询向量 $q_i$，则对每个键 $k_j$ 计算相似度：

$$
e_{ij}=q_i^\top k_j
$$

&emsp;&emsp;然后归一化：

$$
\alpha_{ij}=
\frac{\exp(e_{ij})}{\sum_{l=1}^{m}\exp(e_{il})}
$$

&emsp;&emsp;最终输出为：

$$
o_i=\sum_{j=1}^{m}\alpha_{ij}v_j
$$

&emsp;&emsp;在实际的 Transformer 等模型中，通常使用 **缩放点积注意力**（Scaled Dot-Product Attention）：

$$
\mathrm{Attention}(Q,K,V)=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

&emsp;&emsp;其中 $\sqrt{d_k}$ 是缩放因子。当 $d_k$ 较大时，点积结果的方差会变大，Softmax 容易进入饱和区，导致梯度变小。除以 $\sqrt{d_k}$ 可以使点积分布更稳定，从而有利于训练。

---

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出缩放点积注意力的 PyTorch 实现：

```python
import math
import torch

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output, attn
```

&emsp;&emsp;点积注意力中的 Q/K/V 来自信息检索隐喻：query 是“我想找什么”，key 是“我有什么标签”，value 是“我实际内容”。但在 Transformer 里，它们只是同一个输入 $X$ 的三个线性投影，并没有显式数据库，也没有真正的键值存储。因此 QKV 是一种接口命名和直觉包装，不是注意力的数学本质。

&emsp;&emsp;从维度视角看，设 $X\in\mathbb{R}^{n\times d}$，其中 $n$ 是 token 维，$d$ 是特征维。注意力先通过 $QK^\top$ 在 token 维上建立 $n\times n$ 的关系矩阵，再经过 Softmax 得到扩散权重，最后用 $AV$ 把 token 维的信息重新汇聚到特征维。也就是说，它完成的是“特征维到 token 关系，再到 token 混合，最后回到特征表示”的过程。若暂时忽略 $W_V$，有 $O=AX$，则：

$$
O_{i,c}=\sum_{j=1}^{n}A_{ij}X_{j,c}
$$

&emsp;&emsp;这意味着输出第 $i$ 个 token 的第 $c$ 个特征，是所有输入 token 同一特征通道的加权平均。所以它确实很像“沿 token 维的扩散”。加入 $W_V$ 后，再在特征维上做一次线性混合：

$$
O=AXW_V
$$

&emsp;&emsp;因此，更准确的说法是：注意力是在 token 维和特征维之间做自适应维度扩散：先算 token 间关系，再按关系混合值，最后回到特征表示。Q/K/V 是功能角色命名，不是数学本质；核心是关系矩阵 $A$ 和对 $V$ 的加权混合。

#### 2.1.2 多头注意力（MHA）

&emsp;&emsp;多头注意力（Multi-Head Attention, MHA）是 Transformer 中的核心组件。它并非只做一次注意力，而是将查询、键、值通过多组线性投影映射到多个子空间，在每个子空间中并行执行缩放点积注意力，最后将各头输出拼接并做一次线性变换。

&emsp;&emsp;设输入序列表示为 $X \in \mathbb{R}^{n \times d_{model}}$，头数为 $h$。对于第 $i$ 个头，定义投影矩阵：

$$
W_i^Q \in \mathbb{R}^{d_{model} \times d_k},\quad
W_i^K \in \mathbb{R}^{d_{model} \times d_k},\quad
W_i^V \in \mathbb{R}^{d_{model} \times d_v}
$$

&emsp;&emsp;则第 $i$ 个头的查询、键、值分别为：

$$
Q_i = XW_i^Q,\quad K_i = XW_i^K,\quad V_i = XW_i^V
$$

&emsp;&emsp;第 $i$ 个头的注意力输出为：

$$
\mathrm{head}_i = \mathrm{Attention}(Q_i, K_i, V_i)
= \mathrm{softmax}\left(\frac{Q_iK_i^\top}{\sqrt{d_k}}\right)V_i
$$

&emsp;&emsp;将所有头的输出拼接，再经过输出投影矩阵 $W^O \in \mathbb{R}^{h d_v \times d_{model}}$，得到多头注意力的最终输出：

$$
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h) W^O
$$

&emsp;&emsp;通常令 $d_k = d_v = d_{model} / h$，这样各头的计算量总和与单头注意力在相同维度下相当。每个头只处理 $d_{model}/h$ 维的子空间，因此多头注意力的优点是能在不同表示子空间中并行关注不同模式，增强表达能力且适合 GPU 并行；缺点是参数量和显存开销会随头数增加，头数过多可能造成冗余或训练不稳定，且各头输出最终仍由 $W^O$ 混合，并非完全独立。

&emsp;&emsp;从维度视角看，多头注意力把特征维 $d_{model}$ 切分成 $h$ 个子空间。每个头在自己的子空间内计算 token 间关系矩阵，并沿 token 维做加权扩散，最后拼接各头结果，再通过 $W^O$ 在特征维上混合。因此，多头注意力可以理解为多个维度通道上的并行自适应扩散，它比单头注意力具有更强的表示能力，也更符合“维度扩散”的直觉。

---

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出多头注意力的裸实现，不调用 `nn.MultiheadAttention`，也不调用 `F.scaled_dot_product_attention`，只使用基础张量运算和手动参数：

```python
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        self.b_q = nn.Parameter(torch.zeros(d_model))
        self.b_k = nn.Parameter(torch.zeros(d_model))
        self.b_v = nn.Parameter(torch.zeros(d_model))
        self.b_o = nn.Parameter(torch.zeros(d_model))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = torch.matmul(query, self.W_q) + self.b_q
        K = torch.matmul(key, self.W_k) + self.b_k
        V = torch.matmul(value, self.W_v) + self.b_v

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)

        head_out = head_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = torch.matmul(head_out, self.W_o) + self.b_o
        return output, attn
```

&emsp;&emsp;这个实现中，$Q/K/V$ 的线性投影、分头、缩放点积注意力、拼接和输出投影全部手动完成。若输入 `query, key, value` 形状为 $(B,L,d_{model})$，则输出形状为 $(B,L,d_{model})$，`attn` 形状为 $(B,h,L,L)$。自注意力中三者来自同一输入；交叉注意力中 `query` 来自解码器，`key` 和 `value` 来自编码器。

&emsp;&emsp;其中 $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$、$W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$。通常取 $d_k = d_v = d_{\text{model}} / h$。MHA 是 GPT-1、GPT-2、BERT、T5 等早期模型的标配，也是 LLaMA 1 的注意力方案。

#### 2.1.3 多查询注意力（MQA）

&emsp;&emsp;多查询注意力（Multi-Query Attention, MQA）是 MHA 的一种变体，核心思想是让所有注意力头共享同一组 Key 和 Value 投影，而 Query 仍然保持多头。这样可以大幅减少推理时 KV Cache 的存储量和内存带宽需求，同时保持 Query 端的多样性。

&emsp;&emsp;设输入 $X \in \mathbb{R}^{n \times d_{model}}$，头数为 $h$。MQA 中，每个头有独立的 Query 投影：

$$
W_i^Q \in \mathbb{R}^{d_{model} \times d_k}, \quad i=1,\dots,h
$$

&emsp;&emsp;但所有头共享同一组 Key 和 Value 投影：

$$
W^K \in \mathbb{R}^{d_{model} \times d_k}, \quad W^V \in \mathbb{R}^{d_{model} \times d_v}
$$

&emsp;&emsp;于是：

$$
Q_i = X W_i^Q, \quad K = X W^K, \quad V = X W^V
$$

&emsp;&emsp;第 $i$ 个头的输出为：

$$
\mathrm{head}_i = \mathrm{softmax}\left(\frac{Q_i K^\top}{\sqrt{d_k}}\right) V
$$

&emsp;&emsp;将所有头的输出拼接，再经过输出投影 $W^O$：

$$
\mathrm{MQA}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h) W^O
$$

&emsp;&emsp;通常令 $d_k = d_v = d_{model}/h$，因此共享的 K 和 V 只有单个头的维度。这样在自回归生成时，只需缓存一份 K 和 V，而不是每个头各缓存一份，KV Cache 大小从 $O(h \cdot n \cdot d_k)$ 降到 $O(n \cdot d_k)$，显著降低显存占用和带宽压力。MQA 的优点是推理速度快、显存占用低，适合长序列和大 batch 生成；缺点是所有头共享 K/V 会削弱表示能力，可能导致训练不稳定或效果略差于 MHA，尤其在需要细粒度多模式关注的场景中。

&emsp;&emsp;从维度视角看，MQA 让 Query 在多个子空间中计算关系矩阵，但这些关系矩阵都作用在同一个 Value 上。这相当于多个查询通道并行扩散，但共享同一份被扩散的内容。相比 MHA，它牺牲了一部分 Value 端的多样性，换取了推理效率的大幅提升。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 MQA 的裸实现，只使用基础张量运算和手动参数：

```python
import math
import torch
import torch.nn as nn

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Query 每个头独立
        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.b_q = nn.Parameter(torch.zeros(d_model))

        # Key 和 Value 所有头共享，只有一个头
        self.W_k = nn.Parameter(torch.empty(d_model, self.d_k))
        self.b_k = nn.Parameter(torch.zeros(self.d_k))
        self.W_v = nn.Parameter(torch.empty(d_model, self.d_v))
        self.b_v = nn.Parameter(torch.zeros(self.d_v))

        # 输出投影
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        self.b_o = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_o)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Query: (B, L, d_model) -> (B, h, L, d_k)
        Q = torch.matmul(x, self.W_q) + self.b_q
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Key, Value: 共享，形状 (B, L, d_k) -> (B, 1, L, d_k)
        K = torch.matmul(x, self.W_k) + self.b_k
        V = torch.matmul(x, self.W_v) + self.b_v
        K = K.unsqueeze(1)  # (B, 1, L, d_k)
        V = V.unsqueeze(1)  # (B, 1, L, d_v)

        # 注意力分数: Q (B, h, L, d_k) 与 K (B, 1, L, d_k) 广播
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # (B, h, L, L)
        head_out = torch.matmul(attn, V)      # (B, h, L, d_v)

        # 拼接并输出投影
        head_out = head_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = torch.matmul(head_out, self.W_o) + self.b_o
        return output, attn
```

&emsp;&emsp;这个实现中，K 和 V 只有一个头，通过 `unsqueeze(1)` 后在头维度上广播到所有 Query 头。若输入形状为 $(B,L,d_{model})$，输出形状为 $(B,L,d_{model})$，`attn` 形状为 $(B,h,L,L)$。

---

#### 2.1.4 分组查询注意力（GQA）

&emsp;&emsp;分组查询注意力（Grouped-Query Attention, GQA）是 MHA 和 MQA 的折中方案。它将 Query 头分成 $G$ 组，每组共享一组 Key 和 Value 投影。当 $G=h$ 时，GQA 退化为 MHA；当 $G=1$ 时，退化为 MQA。因此 GQA 可以在推理效率和模型质量之间取得更好的平衡。

&emsp;&emsp;设总 Query 头数为 $h$，分组数为 $G$，则每组包含 $h/G$ 个 Query 头。对于第 $g$ 组，定义共享的 Key 和 Value 投影：

$$
W_g^K \in \mathbb{R}^{d_{model} \times d_k}, \quad W_g^V \in \mathbb{R}^{d_{model} \times d_v}, \quad g=1,\dots,G
$$

&emsp;&emsp;该组内每个 Query 头 $i$ 有独立投影 $W_i^Q$：

$$
Q_i = X W_i^Q, \quad K_g = X W_g^K, \quad V_g = X W_g^V
$$

&emsp;&emsp;第 $i$ 个头的输出为：

$$
\mathrm{head}_i = \mathrm{softmax}\left(\frac{Q_i K_g^\top}{\sqrt{d_k}}\right) V_g
$$

&emsp;&emsp;其中 $i$ 属于第 $g$ 组。将所有头的输出拼接并经过输出投影：

$$
\mathrm{GQA}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h) W^O
$$

&emsp;&emsp;通常令 $d_k = d_v = d_{model}/h$，每组共享的 K/V 头数为 1。KV Cache 大小从 MHA 的 $O(h \cdot n \cdot d_k)$ 降到 $O(G \cdot n \cdot d_k)$，比 MQA 的 $O(n \cdot d_k)$ 略大，但远小于 MHA。GQA 的优点是兼顾了推理效率和表示能力，在大模型推理中广泛使用；缺点是分组数 $G$ 需要手动选择，不同任务和模型规模下最优 $G$ 可能不同，且实现比 MHA 和 MQA 稍复杂。

&emsp;&emsp;从维度视角看，GQA 将 Query 头分组，每组 Query 共享一份被扩散的内容。它既保留了部分 Value 端的多样性，又减少了 KV Cache 的冗余。可以理解为在多个查询通道和少量共享内容通道之间做自适应维度扩散。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 GQA 的裸实现，只使用基础张量运算和手动参数：

```python
import math
import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_groups == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.num_heads_per_group = num_heads // num_kv_groups
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Query 每个头独立
        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.b_q = nn.Parameter(torch.zeros(d_model))

        # Key 和 Value 每组共享一份
        self.W_k = nn.Parameter(torch.empty(d_model, num_kv_groups * self.d_k))
        self.b_k = nn.Parameter(torch.zeros(num_kv_groups * self.d_k))
        self.W_v = nn.Parameter(torch.empty(d_model, num_kv_groups * self.d_v))
        self.b_v = nn.Parameter(torch.zeros(num_kv_groups * self.d_v))

        # 输出投影
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        self.b_o = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_o)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Query: (B, L, d_model) -> (B, h, L, d_k)
        Q = torch.matmul(x, self.W_q) + self.b_q
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Key, Value: 每组一份，形状 (B, L, G, d_k) -> (B, G, L, d_k)
        K = torch.matmul(x, self.W_k) + self.b_k
        V = torch.matmul(x, self.W_v) + self.b_v
        K = K.view(batch_size, seq_len, self.num_kv_groups, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_groups, self.d_v).transpose(1, 2)

        # 将每组 K/V 重复到组内每个 Query 头
        # K: (B, G, L, d_k) -> (B, G, 1, L, d_k) -> (B, G, h_per_group, L, d_k) -> (B, h, L, d_k)
        K = K.unsqueeze(2).expand(-1, -1, self.num_heads_per_group, -1, -1)
        K = K.reshape(batch_size, self.num_heads, seq_len, self.d_k)
        V = V.unsqueeze(2).expand(-1, -1, self.num_heads_per_group, -1, -1)
        V = V.reshape(batch_size, self.num_heads, seq_len, self.d_v)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)

        # 拼接并输出投影
        head_out = head_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = torch.matmul(head_out, self.W_o) + self.b_o
        return output, attn
```

&emsp;&emsp;这个实现中，Key 和 Value 按组生成，然后通过 `expand` 和 `reshape` 复制到组内每个 Query 头，使得后续注意力计算与标准多头注意力一致。若输入形状为 $(B,L,d_{model})$，输出形状为 $(B,L,d_{model})$，`attn` 形状为 $(B,h,L,L)$。当 `num_kv_groups = num_heads` 时退化为 MHA；当 `num_kv_groups = 1` 时退化为 MQA。

#### 2.1.5 多头潜在注意力（MLA）

&emsp;&emsp;多头潜在注意力（Multi-Head Latent Attention, MLA）是 DeepSeek-V2 提出的注意力变体，核心思想是通过低秩联合压缩将 Key 和 Value 投影到一个紧凑的潜在空间，在推理时只需缓存这个潜在向量，从而大幅降低 KV Cache 的存储和带宽需求。与 MQA 和 GQA 通过共享 KV 头来压缩缓存不同，MLA 采用压缩到潜在空间的策略，将键和值联合压缩为单个低秩向量，这是 MLA 与其他压缩方法的本质区别。

&emsp;&emsp;设输入隐藏状态为 $h_t \in \mathbb{R}^{d_{model}}$，MLA 首先通过下投影矩阵 $W^{DKV} \in \mathbb{R}^{d_c \times d_{model}}$ 将 $h_t$ 压缩为潜在向量：

$$
c_t^{KV} = W^{DKV} h_t, \quad c_t^{KV} \in \mathbb{R}^{d_c}
$$

&emsp;&emsp;其中 $d_c \ll n_h \cdot d_h$ 是 KV 压缩维度。然后在推理时，只需缓存 $c_t^{KV}$，而 Key 和 Value 通过上投影矩阵 $W^{UK}$ 和 $W^{UV}$ 从潜在向量重建：

$$
k_t = W^{UK} c_t^{KV}, \quad v_t = W^{UV} c_t^{KV}
$$

&emsp;&emsp;对于 Query，MLA 同样进行低秩压缩以降低训练时的激活内存：

$$
c_t^Q = W^{DQ} h_t, \quad q_t = W^{UQ} c_t^Q
$$

&emsp;&emsp;注意力计算与标准缩放点积注意力一致：

$$
o_t = \mathrm{softmax}\left(\frac{q_t k_t^\top}{\sqrt{d_h}}\right) v_t
$$

&emsp;&emsp;MLA 的优点是 KV Cache 压缩率极高，相比 MHA 可压缩 90% 以上，同时推理时内存带宽需求显著降低，在带宽受限的硬件上性能更稳定；缺点是引入了额外的投影矩阵和潜在空间维度超参数，训练时需要联合优化下投影和上投影，且重建 Key/Value 时增加了少量计算开销，整体实现比 MHA 更复杂。

&emsp;&emsp;从维度视角看，MLA 将 Key 和 Value 先压缩到低维潜在空间，推理时再重建回高维。这相当于在 token 维扩散之前，先对 Value 端做了一次特征维的降维压缩，大幅减少了需要缓存和传输的数据量。Q 端同样做了低秩压缩，但 Q 不需要缓存，所以主要收益在训练激活内存上。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 MLA 的裸实现，只使用基础张量运算和手动参数：

```python
import math
import torch
import torch.nn as nn

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, kv_latent_dim, q_latent_dim=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.kv_latent_dim = kv_latent_dim
        self.q_latent_dim = q_latent_dim if q_latent_dim is not None else d_model

        # Query 低秩投影
        self.W_dq = nn.Parameter(torch.empty(d_model, self.q_latent_dim))
        self.W_uq = nn.Parameter(torch.empty(self.q_latent_dim, d_model))
        self.b_q = nn.Parameter(torch.zeros(d_model))

        # KV 联合低秩投影
        self.W_dkv = nn.Parameter(torch.empty(d_model, kv_latent_dim))
        self.W_uk = nn.Parameter(torch.empty(kv_latent_dim, d_model))
        self.W_uv = nn.Parameter(torch.empty(kv_latent_dim, d_model))
        self.b_k = nn.Parameter(torch.zeros(d_model))
        self.b_v = nn.Parameter(torch.zeros(d_model))

        # 输出投影
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        self.b_o = nn.Parameter(torch.zeros(d_model))

        for w in (self.W_dq, self.W_uq, self.W_dkv, self.W_uk, self.W_uv, self.W_o):
            nn.init.xavier_uniform_(w)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Query 低秩压缩与重建
        c_q = torch.matmul(x, self.W_dq)
        Q = torch.matmul(c_q, self.W_uq) + self.b_q

        # KV 联合低秩压缩
        c_kv = torch.matmul(x, self.W_dkv)

        # 从潜在向量重建 Key 和 Value
        K = torch.matmul(c_kv, self.W_uk) + self.b_k
        V = torch.matmul(c_kv, self.W_uv) + self.b_v

        # 分头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)

        # 拼接并输出投影
        head_out = head_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = torch.matmul(head_out, self.W_o) + self.b_o
        return output, attn
```

&emsp;&emsp;这个实现中，推理时只需缓存 `c_kv`，形状为 $(B, L, d_{kv\_latent})$，而标准 MHA 需缓存 $(B, h, L, d_k)$ 的 K 和 V。若输入形状为 $(B,L,d_{model})$，输出形状为 $(B,L,d_{model})$，`attn` 形状为 $(B,h,L,L)$。

---

#### 2.1.6 交叉注意力与因果掩码

&emsp;&emsp;交叉注意力（Cross-Attention）与因果掩码（Causal Mask）是 Transformer 解码器中两个紧密相关的机制。交叉注意力解决的是“查询来自哪里、键值来自哪里”的问题，而因果掩码解决的是“当前 token 能看到哪些位置”的问题。

&emsp;&emsp;在标准自注意力中，$Q$、$K$、$V$ 都来自同一个序列。而在交叉注意力中，Query 来自解码器的当前层输入，Key 和 Value 来自编码器的最后一层输出：

$$
Q = H_{dec} W^Q, \quad K = H_{enc} W^K, \quad V = H_{enc} W^V
$$

&emsp;&emsp;其中 $H_{dec} \in \mathbb{R}^{n_{dec} \times d}$ 是解码器隐藏状态，$H_{enc} \in \mathbb{R}^{n_{enc} \times d}$ 是编码器输出。交叉注意力使解码器能够在生成每个 token 时，从编码器的完整输出中检索最相关的信息。

&emsp;&emsp;因果掩码则用于自回归生成场景。在解码器的自注意力层中，如果不加限制，每个位置会关注所有位置，包括未来的位置，这会导致信息泄露。因果掩码在注意力分数矩阵上应用上三角掩码：

$$
\mathrm{Mask}_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

&emsp;&emsp;将掩码加到缩放后的注意力分数上：

$$
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + \mathrm{Mask}\right)
$$

&emsp;&emsp;由于 $e^{-\infty}=0$，未来位置的注意力权重被置零，每个位置只能关注自己和之前的位置。需要注意的是，交叉注意力层通常不使用因果掩码，因为 Key/Value 来自编码器的完整输出，本身就是对源序列的完整编码，不存在“未来信息泄露”的问题。

&emsp;&emsp;交叉注意力的优点是解码器能在生成过程中动态聚焦编码器的不同部分，实现两个序列之间的对齐，在机器翻译、语音识别、多模态任务中广泛使用；缺点是需要维护两套不同来源的表示，计算和内存开销比自注意力更大，且当编码器输出很长时，交叉注意力的 $O(n_{dec} \cdot n_{enc})$ 复杂度会成为瓶颈。因果掩码的优点是实现简单、计算开销小，能有效保证自回归生成的因果性；缺点是限制了解码器自注意力层的信息流向，使模型无法利用未来上下文。

&emsp;&emsp;从维度视角看，交叉注意力是在两个不同序列的 token 维之间建立关系矩阵。Query 序列的每个位置通过关系矩阵从 Key 序列的所有位置加权聚合 Value。因果掩码则是在关系矩阵上施加一个下三角约束，使 token 维的扩散只沿时间方向单向进行。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出交叉注意力与因果掩码的裸实现：

```python
import math
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        self.b_q = nn.Parameter(torch.zeros(d_model))
        self.b_k = nn.Parameter(torch.zeros(d_model))
        self.b_v = nn.Parameter(torch.zeros(d_model))
        self.b_o = nn.Parameter(torch.zeros(d_model))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

    def forward(self, decoder_hidden, encoder_output, cross_mask=None, causal=True):
        B, L_dec, _ = decoder_hidden.size()
        L_enc = encoder_output.size(1)

        # Query 来自解码器，Key/Value 来自编码器
        Q = torch.matmul(decoder_hidden, self.W_q) + self.b_q
        K = torch.matmul(encoder_output, self.W_k) + self.b_k
        V = torch.matmul(encoder_output, self.W_v) + self.b_v

        Q = Q.view(B, L_dec, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, L_enc, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, L_enc, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 因果掩码：仅用于自注意力，交叉注意力通常不需要
        if causal and L_dec == L_enc:
            causal_mask = torch.triu(
                torch.ones(L_dec, L_enc, device=scores.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if cross_mask is not None:
            scores = scores.masked_fill(cross_mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)

        head_out = head_out.transpose(1, 2).contiguous().view(B, L_dec, self.d_model)
        output = torch.matmul(head_out, self.W_o) + self.b_o
        return output, attn
```

&emsp;&emsp;这个实现中，Query 来自解码器隐藏状态，Key/Value 来自编码器输出。当 `causal=True` 且解码器与编码器序列长度相同时，应用因果掩码。若 `decoder_hidden` 形状为 $(B, L_{dec}, d_{model})$，`encoder_output` 形状为 $(B, L_{enc}, d_{model})$，则输出形状为 $(B, L_{dec}, d_{model})$，`attn` 形状为 $(B, h, L_{dec}, L_{enc})$。

---

#### 2.1.7 FlashAttention

&emsp;&emsp;FlashAttention 是一种 IO 感知的精确注意力算法，它不改变注意力的数学定义，而是通过优化 GPU 显存层次之间的读写来大幅提升计算效率和降低显存占用。标准注意力需要将完整的 $n \times n$ 注意力矩阵写入 GPU 高带宽内存，再读回进行 Softmax 和后续计算，这导致大量的 HBM 读写成为瓶颈。FlashAttention 的核心思想是用分块技术将 Q、K、V 分割成能装入片上 SRAM 的小块，在 SRAM 中完成计算，避免在 HBM 中存储完整的注意力矩阵。

&emsp;&emsp;FlashAttention 的关键技术包括两个方面：分块和重计算。分块将 Q、K、V 按块加载到 SRAM，在块之间逐步计算 Softmax，在每个块输出之前对其归一化并累加，最后得到正确结果。由于 Softmax 的归一化需要全局信息，FlashAttention 采用在线 Softmax 算法：对每个查询块，逐块遍历 Key 块，边计算边维护当前的最大值和归一化因子，最终累积得到正确的输出。

&emsp;&emsp;在线 Softmax 的核心递推公式如下。设当前已处理到第 $j$ 个 Key 块，维护运行最大值 $m_j$ 和运行归一化因子 $\ell_j$：

$$
m_j = \max(m_{j-1}, \max(K_j \text{ 块的行最大值})
$$

$$
\ell_j = e^{m_{j-1} - m_j} \ell_{j-1} + \sum_{k \in \text{块}j} e^{s_k - m_j}
$$

&emsp;&emsp;输出累积为：

$$
O_j = e^{m_{j-1} - m_j} O_{j-1} + \sum_{k \in \text{块}j} e^{s_k - m_j} V_k
$$

&emsp;&emsp;遍历完所有 Key 块后，用 $\ell$ 归一化得到最终输出。重计算则在反向传播时不存储完整的注意力矩阵，而是重新计算需要的部分，以计算换存储。

&emsp;&emsp;FlashAttention 的优点是显存占用从 $O(n^2)$ 降到 $O(n)$，速度相比标准注意力有 2-4 倍提升，且产生完全相同的数值结果，属于精确注意力而非近似方法；缺点是实现复杂度高，通常需要自定义 CUDA/Triton 内核，且在小序列长度下优势不明显，分块大小需要根据 SRAM 容量手动调优。

&emsp;&emsp;从维度视角看，FlashAttention 并不改变 token 维扩散的数学形式，而是改变了这个扩散在硬件上的执行方式。标准注意力先物化完整的 $n \times n$ 关系矩阵再应用，而 FlashAttention 将关系矩阵按块流式生成和消费，使计算从内存受限转为计算受限。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 FlashAttention 核心思想（分块 + 在线 Softmax）的模拟实现，用 Python 循环模拟块处理流程，不调用任何注意力接口：

```python
import math
import torch

def flash_attention_simulated(Q, K, V, block_size=64):
    """
    模拟 FlashAttention 的分块 + 在线 Softmax 流程。
    Q, K, V: (B, h, L, d)
    """
    B, h, L, d = Q.shape
    O = torch.zeros_like(V)          # 输出累积
    m = torch.full((B, h, L, 1), float('-inf'), device=Q.device)  # 运行最大值
    l = torch.zeros((B, h, L, 1), device=Q.device)                # 运行归一化因子

    num_blocks = math.ceil(L / block_size)

    for j in range(num_blocks):
        start = j * block_size
        end = min(start + block_size, L)

        K_j = K[:, :, start:end, :]   # (B, h, b, d)
        V_j = V[:, :, start:end, :]

        # 计算当前块的注意力分数
        S_j = torch.matmul(Q, K_j.transpose(-2, -1)) / math.sqrt(d)

        # 当前块的局部最大值
        m_j = torch.max(S_j, dim=-1, keepdim=True).values

        # 更新运行最大值
        m_new = torch.maximum(m, m_j)

        # 更新运行归一化因子
        l = l * torch.exp(m - m_new) + torch.sum(
            torch.exp(S_j - m_new), dim=-1, keepdim=True
        )

        # 累积输出
        O = O * torch.exp(m - m_new) + torch.matmul(
            torch.exp(S_j - m_new), V_j
        )

        m = m_new

    # 最终归一化
    O = O / l
    return O
```

&emsp;&emsp;这个实现用循环模拟了 FlashAttention 的块间在线 Softmax 过程。实际生产中使用 `flash_attn` 库或 PyTorch 2.0 的 `F.scaled_dot_product_attention` 可获得 CUDA 加速。若输入形状为 $(B,h,L,d)$，输出形状为 $(B,h,L,d)$。

---

#### 2.1.8 滑动窗口注意力（SWA）

&emsp;&emsp;滑动窗口注意力（Sliding Window Attention, SWA）是一种稀疏注意力机制，核心思想是限制每个 token 只关注其周围固定窗口内的邻居，而非整个序列。标准注意力的复杂度为 $O(n^2)$，当序列很长时成为瓶颈。SWA 将每个 token 的注意力范围限制在窗口大小 $w$ 内，复杂度降为 $O(n \cdot w)$，在保持局部依赖建模能力的同时大幅降低计算和内存开销。

&emsp;&emsp;设窗口大小为 $w$，对于位置 $i$，其注意力范围限制为：

$$
j \in [i - w, i + w]
$$

&emsp;&emsp;对于因果生成场景（只看左侧），范围变为 $j \in [i - w, i]$。注意力分数矩阵变为带状矩阵，仅对角线附近 $w$ 个对角线内的元素非零。数学形式与标准注意力相同，但只对窗口内的位置计算：

$$
o_i = \sum_{j=i-w}^{i+w} \mathrm{softmax}\left(\frac{q_i k_j^\top}{\sqrt{d_k}}\right) v_j
$$

&emsp;&emsp;在实际模型中，即使每一层只看局部窗口，堆叠多层后信息仍可传播到更远的位置。Mistral 7B 使用窗口大小 $W=4096$，共 32 层，理论感受野可达到约 131K 个 token。Longformer 在此基础上引入了全局注意力，允许部分特殊 token（如分类 token）关注整个序列，以补充局部窗口无法覆盖的全局依赖。

&emsp;&emsp;SWA 的优点是计算复杂度从 $O(n^2)$ 降到 $O(n \cdot w)$，显存占用大幅减少，且实现简单，只需在注意力分数矩阵上应用带状掩码；缺点是每个 token 只能直接看到局部窗口内的信息，全局依赖需要通过多层堆叠间接传递，可能丢失需要精确长距离建模的任务中的关键信息，窗口大小 $w$ 需要根据任务和硬件手动选择。

&emsp;&emsp;从维度视角看，SWA 在 token 维扩散中引入了一个带状约束：关系矩阵只在每条对角线附近的 $w$ 个位置非零。这相当于把全局扩散变为局部扩散，牺牲了单层内的全局感受野，但通过多层堆叠来补偿。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出滑动窗口注意力的裸实现：

```python
import math
import torch
import torch.nn as nn

class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        self.b_q = nn.Parameter(torch.zeros(d_model))
        self.b_k = nn.Parameter(torch.zeros(d_model))
        self.b_v = nn.Parameter(torch.zeros(d_model))
        self.b_o = nn.Parameter(torch.zeros(d_model))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

    def _build_band_mask(self, seq_len, device):
        """构建带状掩码：位置 i 只关注 [i-w, i+w] 范围内的位置"""
        w = self.window_size
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            left = max(0, i - w)
            right = min(seq_len, i + w + 1)
            mask[i, left:right] = 1.0
        return mask

    def forward(self, x, causal=False):
        B, L, _ = x.size()

        Q = torch.matmul(x, self.W_q) + self.b_q
        K = torch.matmul(x, self.W_k) + self.b_k
        V = torch.matmul(x, self.W_v) + self.b_v

        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 构建带状掩码
        band_mask = self._build_band_mask(L, x.device)
        if causal:
            causal_mask = torch.tril(torch.ones(L, L, device=x.device))
            band_mask = band_mask * causal_mask

        scores = scores.masked_fill(
            band_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf')
        )

        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)

        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = torch.matmul(head_out, self.W_o) + self.b_o
        return output, attn
```

&emsp;&emsp;这个实现中，`_build_band_mask` 构建带状掩码，使每个位置只关注窗口内的邻居。当 `causal=True` 时，窗口只向左侧开放。若输入形状为 $(B,L,d_{model})$，输出形状为 $(B,L,d_{model})$，`attn` 形状为 $(B,h,L,L)$，但大部分权重为零，实际有效计算量为 $O(n \cdot w)$。



#### 2.1.9 稀疏注意力与 DSA

&emsp;&emsp;稀疏注意力（Sparse Attention）是一类通过限制注意力计算范围来降低复杂度的机制。标准自注意力的复杂度为 $O(n^2)$，每个 token 需要与序列中所有其他 token 计算关系。稀疏注意力通过为每个查询选择性地只关注部分 key，将计算量降至 $O(n \cdot k)$，其中 $k$ 是每个查询实际参与计算的 key 数量。

&emsp;&emsp;传统稀疏注意力方法（如 Longformer、BigBird）通常采用“局部窗口+全局标记”的静态策略：每个 token 只关注邻近的固定窗口，同时强制所有 token 关注少量全局节点。这种方法实现简单，但存在两个问题：稀疏模式固定，无法适应不同输入的动态需求；若关键依赖超出局部窗口且未被选为全局标记，模型可能遗漏重要信息。

&emsp;&emsp;DeepSeek Sparse Attention（DSA）是 DeepSeek-V3.2 引入的稀疏注意力方案，其核心创新是引入一个轻量级的 **闪电索引器（Lightning Indexer）** 来动态预测 token 重要性，再据此进行 Top-k 选择。DSA 的核心思想是“先选择，再计算”：先用低精度、低维度的投影快速为每个查询打分，选出最相关的 $k$ 个 key，再仅对这 $k$ 个 key 做完整的注意力计算。

&emsp;&emsp;设隐藏状态为 $h_t$，Indexer 首先用低维投影生成索引查询和索引键：

$$
q_t^{idx} = W^{q,idx} h_t, \quad k_t^{idx} = W^{k,idx} h_t
$$

&emsp;&emsp;索引分数的计算与标准注意力类似，但使用 ReLU 替代 Softmax，并在多个索引头上做加权求和：

$$
s_{t,j} = \sum_{m=1}^{H_{idx}} w_m \cdot \mathrm{ReLU}\left(\frac{(q_t^{idx})_m^\top (k_j^{idx})_m}{\sqrt{d_{idx}}}\right)
$$

&emsp;&emsp;其中 $w_m$ 是每个索引头的可学习权重，$H_{idx}$ 是索引头数。根据这些分数，为每个查询选出 top-$k$ 个 key，然后只对这 $k$ 个 key 计算完整的缩放点积注意力：

$$
o_t = \mathrm{softmax}\left(\frac{q_t k_{C_t}^\top}{\sqrt{d_k}}\right) v_{C_t}
$$

&emsp;&emsp;其中 $C_t$ 是查询 $t$ 选出的 $k$ 个 key 的索引集合。Indexer 的 Q/K 路径全程使用 FP8 或 FP4 低精度计算，维度远小于主注意力，因此打分开销极低。DSA 的优点是动态稀疏模式能适应输入内容，在保持模型质量的同时显著降低长序列推理的计算量，且 Indexer 的低精度设计使额外开销可控；缺点是引入了额外的 Indexer 模块和超参数（索引维度、索引头数、top-$k$），训练时需要联合优化 Indexer 和主注意力，且稀疏模式的不规则性给连续批处理和分页注意力等推理优化带来了挑战。

&emsp;&emsp;从维度视角看，DSA 在 token 维扩散之前增加了一个“路由”阶段：Indexer 先粗略判断哪些位置的 Value 值得关注，再将完整的扩散计算限制在这个子集上。它不改变扩散的数学形式，而是改变了参与扩散的 token 集合。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 DSA 核心思想（Lightning Indexer + Top-k 稀疏注意力）的裸实现：

```python
import math
import torch
import torch.nn as nn

class DeepSeekSparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_idx_heads=4, top_k=64):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_idx_heads = num_idx_heads
        self.top_k = top_k

        # 主注意力投影
        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        # Lightning Indexer: 低维低精度投影
        idx_dim = self.d_k // 4
        self.idx_dim = idx_dim
        self.W_qi = nn.Parameter(torch.empty(d_model, num_idx_heads * idx_dim))
        self.W_ki = nn.Parameter(torch.empty(d_model, num_idx_heads * idx_dim))
        self.idx_weights = nn.Parameter(torch.ones(num_idx_heads))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o, self.W_qi, self.W_ki):
            nn.init.xavier_uniform_(w)

    def forward(self, x, causal=True):
        B, L, _ = x.size()

        # --- Indexer 打分 ---
        q_idx = torch.matmul(x, self.W_qi)
        k_idx = torch.matmul(x, self.W_ki)
        q_idx = q_idx.view(B, L, self.num_idx_heads, self.idx_dim).transpose(1, 2)
        k_idx = k_idx.view(B, L, self.num_idx_heads, self.idx_dim).transpose(1, 2)

        idx_scores = torch.matmul(q_idx, k_idx.transpose(-2, -1)) / math.sqrt(self.idx_dim)
        idx_scores = torch.relu(idx_scores)
        # 多头加权求和
        idx_scores = (idx_scores * self.idx_weights.view(1, -1, 1, 1)).sum(dim=1)  # (B, L, L)

        # --- Top-k 选择 ---
        k = min(self.top_k, L)
        if causal:
            # 因果掩码：只允许关注当前位置及之前
            causal_mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
            idx_scores = idx_scores.masked_fill(~causal_mask.unsqueeze(0), float('-inf'))

        topk_indices = idx_scores.topk(k, dim=-1).indices  # (B, L, k)

        # --- 主注意力（仅对 top-k 计算）---
        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        # 收集 top-k 的 K 和 V
        # topk_indices: (B, L, k) -> 扩展为 (B, num_heads, L, k)
        gather_idx = topk_indices.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        K_gathered = K.gather(2, gather_idx)  # (B, h, L, k, d_k)
        V_gathered = V.gather(2, gather_idx)

        # 计算注意力
        scores = torch.matmul(Q.unsqueeze(3), K_gathered.transpose(-2, -1)).squeeze(3)
        scores = scores / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)  # (B, h, L, k)

        head_out = torch.matmul(attn.unsqueeze(-2), V_gathered).squeeze(-2)  # (B, h, L, d_k)

        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = torch.matmul(head_out, self.W_o)
        return output, attn
```

&emsp;&emsp;这个实现中，Indexer 用低维投影快速打分，再取 top-$k$ 索引，主注意力只对选出的 $k$ 个 key 计算。若输入形状为 $(B,L,d_{model})$，输出形状为 $(B,L,d_{model})$，`attn` 形状为 $(B,h,L,k)$。

---

#### 2.1.10 混合注意力 CSA / HCA

&emsp;&emsp;混合注意力（Hybrid Attention）是 DeepSeek-V4 提出的注意力架构，由两个互补的组件构成：**压缩稀疏注意力（Compressed Sparse Attention, CSA）** 和 **重度压缩注意力（Heavily Compressed Attention, HCA）** 。其核心思想是沿序列维度对 KV Cache 进行压缩，同时结合稀疏选择或全量注意力，在百万级上下文场景下将注意力算力降到可接受的水平。

&emsp;&emsp;CSA 的做法是先将每 $m$ 个相邻 token 的 KV 压缩成一个压缩条目，然后用 Lightning Indexer 为每个查询选出 top-$k$ 个最相关的压缩条目做注意力。压缩比 $m$ 通常取 4，即每 4 个 token 压缩为 1 个条目，这样 KV Cache 序列长度降为原来的 $1/4$。压缩不是简单平均，而是用可学习的 softmax 权重加位置偏置做加权求和，并且采用重叠压缩来避免硬切边界处的信息断裂：一组用前 $m$ 个 token，另一组用后 $m$ 个 token，两个压缩结果拼接后得到最终的压缩条目。

&emsp;&emsp;设第 $g$ 组的 token 集合为 $\{t_1, \dots, t_m\}$，压缩条目为：

$$
c_g = \sum_{i=1}^{m} \alpha_i \cdot k_{t_i}, \quad \alpha_i = \mathrm{softmax}(z_i + b_i)
$$

&emsp;&emsp;其中 $z_i$ 是可学习的投影分数，$b_i$ 是位置偏置。Value 的压缩采用同样的加权方式。压缩后的序列长度为 $L/m$，再对每个查询选出 top-$k$ 个压缩条目做注意力。

&emsp;&emsp;HCA 是另一个极端：压缩比 $m'$ 拉到 128，即每 128 个 token 压缩为 1 个条目，压缩后的序列长度极短，因此不需要稀疏选择，直接做全量注意力即可。HCA 的计算量为 $O(n \times n/128) = O(n^2/128)$，相比原始 $O(n^2)$ 降低了两个数量级。CSA 以适中压缩率保留细节、靠稀疏选择省算力；HCA 以极高压缩率提供一个粗粒度但覆盖全局的视野。两者交错堆叠，使模型既能捕捉精细的局部依赖，又能维持全局感受野。

&emsp;&emsp;混合注意力的优点是百万上下文下的边际成本被压到可用水平，V4-Pro 的单 token 推理 FLOPs 仅为 V3.2 的 27%，KV Cache 仅为 10%；缺点是结构复杂，压缩器需要额外的投影矩阵和 softmax 门控，CSA 还需要额外的 Indexer 打分流程，训练和推理的实现复杂度都显著高于标准注意力。

&emsp;&emsp;从维度视角看，CSA 和 HCA 都是在 token 维扩散之前先对 Value 端做一次序列维的压缩，减少参与扩散的 token 数量。CSA 保留较多 token 但用稀疏选择控制计算量，HCA 直接用激进压缩减少 token 数量。两者交错使用，相当于在“精细但昂贵”和“粗略但廉价”之间做权衡。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 CSA 和 HCA 核心思想（压缩 + 稀疏/全量注意力）的裸实现：

```python
import math
import torch
import torch.nn as nn

class CompressedSparseAttention(nn.Module):
    """CSA: 压缩 + 稀疏选择"""
    def __init__(self, d_model, num_heads, compress_ratio=4, top_k=64):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.compress_ratio = compress_ratio
        self.top_k = top_k

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        # 压缩权重和偏置
        self.W_compress = nn.Parameter(torch.empty(d_model, d_model))
        self.b_compress = nn.Parameter(torch.zeros(d_model))

        # Indexer
        self.W_idx = nn.Parameter(torch.empty(d_model, d_model))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o, self.W_compress, self.W_idx):
            nn.init.xavier_uniform_(w)

    def _compress(self, x):
        B, L, D = x.size()
        m = self.compress_ratio
        L_pad = math.ceil(L / m) * m
        if L_pad > L:
            x = torch.cat([x, torch.zeros(B, L_pad - L, D, device=x.device)], dim=1)
        x = x.view(B, L_pad // m, m, D)
        # 可学习加权压缩
        scores = torch.matmul(x, self.W_compress) + self.b_compress
        weights = torch.softmax(scores, dim=2)
        compressed = (x * weights).sum(dim=2)  # (B, L//m, D)
        return compressed

    def forward(self, x, causal=True):
        B, L, _ = x.size()

        # 压缩 KV
        x_k = self._compress(x)
        x_v = self._compress(x)
        L_c = x_k.size(1)

        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x_k, self.W_k).view(B, L_c, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x_v, self.W_v).view(B, L_c, self.num_heads, self.d_k).transpose(1, 2)

        # Indexer 选择 top-k 压缩条目
        idx_score = torch.matmul(x, self.W_idx)  # (B, L, D)
        idx_score = torch.matmul(idx_score, K.mean(dim=1).transpose(-2, -1))  # (B, L, L_c)
        k = min(self.top_k, L_c)
        topk_idx = idx_score.topk(k, dim=-1).indices  # (B, L, k)

        gather_idx = topk_idx.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        K_g = K.gather(2, gather_idx)
        V_g = V.gather(2, gather_idx)

        scores = torch.matmul(Q.unsqueeze(3), K_g.transpose(-2, -1)).squeeze(3) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn.unsqueeze(-2), V_g).squeeze(-2)

        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return torch.matmul(head_out, self.W_o), attn


class HeavilyCompressedAttention(nn.Module):
    """HCA: 重度压缩 + 全量注意力"""
    def __init__(self, d_model, num_heads, compress_ratio=128):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.compress_ratio = compress_ratio

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        self.W_compress = nn.Parameter(torch.empty(d_model, d_model))
        self.b_compress = nn.Parameter(torch.zeros(d_model))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o, self.W_compress):
            nn.init.xavier_uniform_(w)

    def _compress(self, x):
        B, L, D = x.size()
        m = self.compress_ratio
        L_pad = math.ceil(L / m) * m
        if L_pad > L:
            x = torch.cat([x, torch.zeros(B, L_pad - L, D, device=x.device)], dim=1)
        x = x.view(B, L_pad // m, m, D)
        scores = torch.matmul(x, self.W_compress) + self.b_compress
        weights = torch.softmax(scores, dim=2)
        return (x * weights).sum(dim=2)

    def forward(self, x):
        B, L, _ = x.size()

        x_k = self._compress(x)
        x_v = self._compress(x)
        L_c = x_k.size(1)

        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x_k, self.W_k).view(B, L_c, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x_v, self.W_v).view(B, L_c, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)

        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return torch.matmul(head_out, self.W_o), attn
```

&emsp;&emsp;CSA 先压缩再稀疏选择，HCA 直接重度压缩后做全量注意力。两者的压缩器结构相同，区别仅在压缩比和是否使用 Indexer。

---

#### 2.1.11 线性注意力

&emsp;&emsp;线性注意力（Linear Attention）是一类通过核函数特征映射将注意力计算复杂度从 $O(n^2)$ 降到 $O(n)$ 的方法。其核心思想是将 Softmax 注意力中的相似度计算分解为可分解的核函数，利用矩阵乘法的结合律重新组织计算顺序，避免显式构造 $n \times n$ 的注意力矩阵。

&emsp;&emsp;标准注意力中，Softmax 的归一化使得注意力权重无法分解：

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

&emsp;&emsp;Softmax 作用于 $QK^\top$ 的每一行，与 $V$ 的乘法无法交换顺序。线性注意力的关键观察是：如果用一个可分解的核函数 $\phi$ 替换 Softmax，使得相似度可以写成两个特征映射的内积：

$$
\mathrm{sim}(q_i, k_j) \approx \phi(q_i)^\top \phi(k_j)
$$

&emsp;&emsp;那么注意力输出可以重写为：

$$
o_i = \frac{\sum_{j=1}^{n} \phi(q_i)^\top \phi(k_j) v_j}{\sum_{j=1}^{n} \phi(q_i)^\top \phi(k_j)}
= \frac{\phi(q_i)^\top \left(\sum_{j=1}^{n} \phi(k_j) v_j^\top\right)}{\phi(q_i)^\top \left(\sum_{j=1}^{n} \phi(k_j)\right)}
$$

&emsp;&emsp;关键变化在于：$\sum_{j=1}^{n} \phi(k_j) v_j^\top$ 是一个与查询无关的中间矩阵，可以在遍历序列时增量累积，不需要为每个查询重新计算。这样，总复杂度从 $O(n^2 d)$ 降为 $O(n d^2)$，其中 $d$ 是特征维度。当 $d \ll n$ 时，线性注意力在计算量上具有显著优势。

&emsp;&emsp;常用的核函数 $\phi$ 包括：Linear Transformer 使用 $\phi(x) = \mathrm{ELU}(x) + 1$，保证输出非负；Performer 使用随机特征映射近似 Softmax 核；cosFormer 使用余弦重加权来增强局部性。选择核函数的核心约束是 $\phi$ 的输出需要保持非负，否则归一化因子可能为零或负值，导致数值不稳定。

&emsp;&emsp;线性注意力的因果形式还可以写成类似 RNN 的状态更新。设状态矩阵 $S_t = \sum_{j=1}^{t} \phi(k_j) v_j^\top$ 和归一化向量 $z_t = \sum_{j=1}^{t} \phi(k_j)$，则：

$$
S_t = S_{t-1} + \phi(k_t) v_t^\top, \quad z_t = z_{t-1} + \phi(k_t)
$$

$$
o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}
$$

&emsp;&emsp;这种循环形式使线性注意力在自回归生成时只需维护固定大小的状态 $S \in \mathbb{R}^{d_\phi \times d_v}$，而非随序列长度增长的 KV Cache，推理内存占用为常数。

&emsp;&emsp;线性注意力的优点是计算复杂度从 $O(n^2)$ 降到 $O(n)$，推理时状态大小固定，不随序列增长，适合超长序列和流式生成；缺点是核函数近似会损失 Softmax 注意力的表达能力和数值稳定性，模型在需要精确长距离依赖建模的任务上可能表现不如标准注意力，且训练时并行形式与推理时循环形式之间的数值一致性需要额外注意。

&emsp;&emsp;从维度视角看，线性注意力改变了 token 维扩散的计算顺序：不再先构造完整的关系矩阵再应用，而是将 Value 端的信息压缩到一个与序列长度无关的状态矩阵中，查询通过核函数特征映射从这个状态中读取信息。这相当于把“全量扩散”替换为“状态累积 + 状态读取”，以表达能力的损失换取了序列长度的线性扩展能力。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出线性注意力的裸实现，包含并行形式和循环形式：

```python
import torch
import torch.nn as nn

class LinearAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

    @staticmethod
    def _phi(x):
        """特征映射：ELU + 1，保证非负"""
        return torch.nn.functional.elu(x) + 1.0

    def forward(self, x, causal=True):
        B, L, _ = x.size()

        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        # 特征映射
        Q_phi = self._phi(Q)  # (B, h, L, d_k)
        K_phi = self._phi(K)

        if causal:
            # 因果形式：循环累积状态
            S = torch.zeros(B, self.num_heads, self.d_k, self.d_k, device=x.device)
            Z = torch.zeros(B, self.num_heads, self.d_k, device=x.device)
            outputs = []

            for t in range(L):
                # 状态更新
                S = S + K_phi[:, :, t].unsqueeze(-1) * V[:, :, t].unsqueeze(-2)
                Z = Z + K_phi[:, :, t]

                # 读取
                o_t = (Q_phi[:, :, t].unsqueeze(-2) @ S).squeeze(-2)
                z_t = (Q_phi[:, :, t] * Z).sum(dim=-1, keepdim=True)
                outputs.append(o_t / (z_t + 1e-6))

            head_out = torch.stack(outputs, dim=2)  # (B, h, L, d_k)
        else:
            # 并行形式：先算中间矩阵，再读取
            # S = K_phi^T @ V  (B, h, d_k, d_k)
            S = torch.matmul(K_phi.transpose(-2, -1), V)
            Z = K_phi.sum(dim=2)  # (B, h, d_k)

            head_out = torch.matmul(Q_phi, S)  # (B, h, L, d_k)
            z_out = (Q_phi * Z.unsqueeze(2)).sum(dim=-1, keepdim=True)  # (B, h, L, 1)
            head_out = head_out / (z_out + 1e-6)

        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = torch.matmul(head_out, self.W_o)
        return output, None
```

&emsp;&emsp;这个实现中，`_phi` 使用 ELU+1 作为特征映射。`causal=True` 时用循环形式累积状态，`causal=False` 时用并行形式先算中间矩阵再读取。若输入形状为 $(B,L,d_{model})$，输出形状为 $(B,L,d_{model})$。循环形式的状态 $S$ 大小固定为 $(B,h,d_k,d_k)$，与序列长度无关。

---

### 2.2 Transformer 架构

#### 2.2.1 Encoder-only

&emsp;&emsp;Encoder-only 架构只保留 Transformer 的编码器部分，每个 token 可以关注序列中的所有其他 token，因此是双向注意力。它不包含自回归生成机制，通常用于理解类任务，如文本分类、序列标注、抽取式问答和句子相似度。代表性模型包括 BERT、RoBERTa、DeBERTa 等。

&emsp;&emsp;设输入 token 序列经过嵌入和位置编码后得到 $X \in \mathbb{R}^{n \times d_{model}}$。Encoder-only 由 $N$ 个相同的编码层堆叠而成，每层包含两个子层：多头自注意力和前馈网络。每个子层都使用残差连接和层归一化。自注意力中 $Q,K,V$ 都来自同一输入，且不使用因果掩码，因此每个位置都能看到全序列。

&emsp;&emsp;编码器层的计算可以写为：

$$
Z = \mathrm{LayerNorm}\left(X + \mathrm{MultiHead}(X,X,X)\right)
$$

$$
Y = \mathrm{LayerNorm}\left(Z + \mathrm{FFN}(Z)\right)
$$

&emsp;&emsp;其中 $\mathrm{FFN}$ 通常是两层线性变换加激活函数：

$$
\mathrm{FFN}(z) = W_2 \cdot \sigma(W_1 z + b_1) + b_2
$$

&emsp;&emsp;Encoder-only 的输出是每个位置的上下文表示。对于分类任务，通常取第一个特殊 token 的表示或对所有位置做池化，再接入任务头。Encoder-only 的优点是双向注意力使每个 token 都能利用完整上下文，在理解类任务上表现优异，且可以并行处理整个序列，训练效率高；缺点是无法直接用于自回归生成，因为训练时看到完整序列会导致信息泄露，且推理时没有因果约束，不能逐 token 生成。

&emsp;&emsp;从维度视角看，Encoder-only 在 token 维上做全连接扩散：每个 token 的关系矩阵覆盖所有位置，没有方向性约束。它适合对整段序列做一次全局信息混合，然后输出每个位置的上下文表示。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 Encoder-only 的裸实现，只使用基础张量运算和手动参数：

```python
import math
import torch
import torch.nn as nn

class EncoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=512):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.token_emb = nn.Parameter(torch.empty(vocab_size, d_model))
        self.pos_emb = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.token_emb, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "W_q": nn.Parameter(torch.empty(d_model, d_model)),
                "W_k": nn.Parameter(torch.empty(d_model, d_model)),
                "W_v": nn.Parameter(torch.empty(d_model, d_model)),
                "W_o": nn.Parameter(torch.empty(d_model, d_model)),
                "W_1": nn.Parameter(torch.empty(d_model, d_ff)),
                "b_1": nn.Parameter(torch.zeros(d_ff)),
                "W_2": nn.Parameter(torch.empty(d_ff, d_model)),
                "b_2": nn.Parameter(torch.zeros(d_model)),
                "ln1_w": nn.Parameter(torch.ones(d_model)),
                "ln1_b": nn.Parameter(torch.zeros(d_model)),
                "ln2_w": nn.Parameter(torch.ones(d_model)),
                "ln2_b": nn.Parameter(torch.zeros(d_model)),
            })
            for _ in range(num_layers)
        ])

        for layer in self.layers:
            for name, p in layer.items():
                if "W_" in name:
                    nn.init.xavier_uniform_(p)

    def _layer_norm(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def forward(self, input_ids):
        B, L = input_ids.size()
        x = self.token_emb[input_ids] + self.pos_emb[:L].unsqueeze(0)

        for layer in self.layers:
            # 自注意力
            Q = torch.matmul(x, layer["W_q"])
            K = torch.matmul(x, layer["W_k"])
            V = torch.matmul(x, layer["W_v"])

            Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = torch.softmax(scores, dim=-1)
            head_out = torch.matmul(attn, V)
            head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
            attn_out = torch.matmul(head_out, layer["W_o"])

            # 残差 + LayerNorm
            x = self._layer_norm(x + attn_out, layer["ln1_w"], layer["ln1_b"])

            # FFN
            ffn_out = torch.matmul(
                torch.relu(torch.matmul(x, layer["W_1"]) + layer["b_1"]),
                layer["W_2"]
            ) + layer["b_2"]

            x = self._layer_norm(x + ffn_out, layer["ln2_w"], layer["ln2_b"])

        return x
```

&emsp;&emsp;这个实现中，每个编码层先做双向自注意力，再做前馈网络，均使用 Post-LN 结构。若输入 `input_ids` 形状为 $(B,L)$，输出形状为 $(B,L,d_{model})$。

---

#### 2.2.2 Decoder-only

&emsp;&emsp;Decoder-only 架构只保留 Transformer 的解码器部分，每个 token 只能关注自己和之前的位置，因此是因果注意力。它通过自回归方式逐 token 生成序列，是当前大语言模型的主流架构。代表性模型包括 GPT 系列、LLaMA、Qwen、DeepSeek 等。

&emsp;&emsp;设输入 token 序列经过嵌入和位置编码后得到 $X \in \mathbb{R}^{n \times d_{model}}$。Decoder-only 由 $N$ 个相同的解码层堆叠而成，每层包含两个子层：因果多头自注意力和前馈网络。与 Encoder-only 的关键区别是自注意力使用因果掩码，将未来位置的注意力分数置为 $-\infty$，确保位置 $i$ 只能关注 $j \leq i$。

&emsp;&emsp;因果掩码矩阵为：

$$
M_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}
$$

&emsp;&emsp;解码器层的计算为：

$$
Z = \mathrm{LayerNorm}\left(X + \mathrm{MultiHead}(X,X,X) + M\right)
$$

$$
Y = \mathrm{LayerNorm}\left(Z + \mathrm{FFN}(Z)\right)
$$

&emsp;&emsp;训练时，Decoder-only 仍然可以并行处理整个序列，因为因果掩码保证了每个位置只看到之前的内容。推理时，模型逐 token 生成，每次将新 token 追加到输入序列末尾，并缓存之前所有层的 Key 和 Value，避免重复计算。Decoder-only 的优点是结构统一、易于扩展，适合自回归生成，且与 KV Cache 配合后推理效率高；缺点是只能利用左侧上下文，无法像 Encoder-only 那样双向建模，且训练和推理存在一定的计算模式差异。

&emsp;&emsp;从维度视角看，Decoder-only 在 token 维上做因果扩散：关系矩阵是下三角矩阵，信息只能从过去流向未来。它适合逐 token 生成，每一步的输出只依赖已生成的 token。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 Decoder-only 的裸实现，包含因果掩码：

```python
import math
import torch
import torch.nn as nn

class DecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=512):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.token_emb = nn.Parameter(torch.empty(vocab_size, d_model))
        self.pos_emb = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.token_emb, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "W_q": nn.Parameter(torch.empty(d_model, d_model)),
                "W_k": nn.Parameter(torch.empty(d_model, d_model)),
                "W_v": nn.Parameter(torch.empty(d_model, d_model)),
                "W_o": nn.Parameter(torch.empty(d_model, d_model)),
                "W_1": nn.Parameter(torch.empty(d_model, d_ff)),
                "b_1": nn.Parameter(torch.zeros(d_ff)),
                "W_2": nn.Parameter(torch.empty(d_ff, d_model)),
                "b_2": nn.Parameter(torch.zeros(d_model)),
                "ln1_w": nn.Parameter(torch.ones(d_model)),
                "ln1_b": nn.Parameter(torch.zeros(d_model)),
                "ln2_w": nn.Parameter(torch.ones(d_model)),
                "ln2_b": nn.Parameter(torch.zeros(d_model)),
            })
            for _ in range(num_layers)
        ])

        for layer in self.layers:
            for name, p in layer.items():
                if "W_" in name:
                    nn.init.xavier_uniform_(p)

    def _layer_norm(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def forward(self, input_ids):
        B, L = input_ids.size()
        x = self.token_emb[input_ids] + self.pos_emb[:L].unsqueeze(0)

        # 因果掩码
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device), diagonal=1
        ).bool()

        for layer in self.layers:
            Q = torch.matmul(x, layer["W_q"])
            K = torch.matmul(x, layer["W_k"])
            V = torch.matmul(x, layer["W_v"])

            Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            head_out = torch.matmul(attn, V)
            head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
            attn_out = torch.matmul(head_out, layer["W_o"])

            x = self._layer_norm(x + attn_out, layer["ln1_w"], layer["ln1_b"])

            ffn_out = torch.matmul(
                torch.relu(torch.matmul(x, layer["W_1"]) + layer["b_1"]),
                layer["W_2"]
            ) + layer["b_2"]

            x = self._layer_norm(x + ffn_out, layer["ln2_w"], layer["ln2_b"])

        return x
```

&emsp;&emsp;这个实现中，因果掩码在每层自注意力中应用，确保每个位置只能关注自己和之前的位置。若输入 `input_ids` 形状为 $(B,L)$，输出形状为 $(B,L,d_{model})$。实际生成时通常还会缓存每层的 K 和 V，这里为简化省略。

---

#### 2.2.3 Encoder-Decoder

&emsp;&emsp;Encoder-Decoder 架构同时包含编码器和解码器，编码器双向处理源序列，解码器自回归生成目标序列，并通过交叉注意力从编码器输出中读取信息。它适合序列到序列任务，如机器翻译、文本摘要、语音识别。代表性模型包括原始 Transformer、T5、BART 等。

&emsp;&emsp;设源序列为 $X \in \mathbb{R}^{n \times d_{model}}$，目标序列为 $Y \in \mathbb{R}^{m \times d_{model}}$。编码器由 $N$ 层双向自注意力和前馈网络组成，输出 $H_{enc}$。解码器由 $N$ 层组成，每层包含三个子层：因果自注意力、交叉注意力和前馈网络。交叉注意力中，Query 来自解码器，Key 和 Value 来自编码器输出：

$$
Q = H_{dec} W^Q, \quad K = H_{enc} W^K, \quad V = H_{enc} W^V
$$

&emsp;&emsp;解码器层的计算为：

$$
Z_1 = \mathrm{LayerNorm}\left(Y + \mathrm{MaskedMultiHead}(Y,Y,Y)\right)
$$

$$
Z_2 = \mathrm{LayerNorm}\left(Z_1 + \mathrm{CrossAttention}(Z_1, H_{enc}, H_{enc})\right)
$$

$$
Z_3 = \mathrm{LayerNorm}\left(Z_2 + \mathrm{FFN}(Z_2)\right)
$$

&emsp;&emsp;Encoder-Decoder 的优点是编码器可以双向理解源序列，解码器通过交叉注意力动态对齐源和目标，适合输入输出长度不同、需要显式对齐的任务；缺点是结构比 Encoder-only 和 Decoder-only 更复杂，参数量和计算量更大，且交叉注意力的 $O(n_{dec} \cdot n_{enc})$ 复杂度在长序列上容易成为瓶颈。

&emsp;&emsp;从维度视角看，Encoder-Decoder 包含两次 token 维扩散：编码器内部的双向扩散和解码器内部的因果扩散，以及解码器到编码器的交叉扩散。交叉扩散的关系矩阵形状为 $n_{dec} \times n_{enc}$，连接了两个不同序列的 token 维。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 Encoder-Decoder 的裸实现，包含编码器、解码器和交叉注意力：

```python
import math
import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, num_heads, num_layers, d_ff, max_len=512):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.src_emb = nn.Parameter(torch.empty(src_vocab, d_model))
        self.tgt_emb = nn.Parameter(torch.empty(tgt_vocab, d_model))
        self.pos_emb = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.src_emb, std=0.02)
        nn.init.normal_(self.tgt_emb, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        def make_layer():
            return nn.ModuleDict({
                "W_q": nn.Parameter(torch.empty(d_model, d_model)),
                "W_k": nn.Parameter(torch.empty(d_model, d_model)),
                "W_v": nn.Parameter(torch.empty(d_model, d_model)),
                "W_o": nn.Parameter(torch.empty(d_model, d_model)),
                "W_cq": nn.Parameter(torch.empty(d_model, d_model)),
                "W_ck": nn.Parameter(torch.empty(d_model, d_model)),
                "W_cv": nn.Parameter(torch.empty(d_model, d_model)),
                "W_co": nn.Parameter(torch.empty(d_model, d_model)),
                "W_1": nn.Parameter(torch.empty(d_model, d_ff)),
                "b_1": nn.Parameter(torch.zeros(d_ff)),
                "W_2": nn.Parameter(torch.empty(d_ff, d_model)),
                "b_2": nn.Parameter(torch.zeros(d_model)),
                "ln1_w": nn.Parameter(torch.ones(d_model)),
                "ln1_b": nn.Parameter(torch.zeros(d_model)),
                "ln2_w": nn.Parameter(torch.ones(d_model)),
                "ln2_b": nn.Parameter(torch.zeros(d_model)),
                "ln3_w": nn.Parameter(torch.ones(d_model)),
                "ln3_b": nn.Parameter(torch.zeros(d_model)),
            })

        self.enc_layers = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([make_layer() for _ in range(num_layers)])

        for layer in list(self.enc_layers) + list(self.dec_layers):
            for name, p in layer.items():
                if "W_" in name:
                    nn.init.xavier_uniform_(p)

    def _ln(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def _attn(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    def forward(self, src_ids, tgt_ids):
        B, L_s = src_ids.size()
        _, L_t = tgt_ids.size()

        src = self.src_emb[src_ids] + self.pos_emb[:L_s].unsqueeze(0)
        tgt = self.tgt_emb[tgt_ids] + self.pos_emb[:L_t].unsqueeze(0)

        # 编码器
        enc = src
        for layer in self.enc_layers:
            Q = torch.matmul(enc, layer["W_q"]).view(B, L_s, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(enc, layer["W_k"]).view(B, L_s, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(enc, layer["W_v"]).view(B, L_s, self.num_heads, self.d_k).transpose(1, 2)
            attn_out = self._attn(Q, K, V)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_s, self.d_model)
            attn_out = torch.matmul(attn_out, layer["W_o"])
            enc = self._ln(enc + attn_out, layer["ln1_w"], layer["ln1_b"])

            ffn = torch.matmul(torch.relu(torch.matmul(enc, layer["W_1"]) + layer["b_1"]), layer["W_2"]) + layer["b_2"]
            enc = self._ln(enc + ffn, layer["ln2_w"], layer["ln2_b"])

        # 解码器
        dec = tgt
        causal_mask = torch.triu(torch.ones(L_t, L_t, device=dec.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        for layer in self.dec_layers:
            # 因果自注意力
            Q = torch.matmul(dec, layer["W_q"]).view(B, L_t, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(dec, layer["W_k"]).view(B, L_t, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(dec, layer["W_v"]).view(B, L_t, self.num_heads, self.d_k).transpose(1, 2)
            attn_out = self._attn(Q, K, V, causal_mask)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_t, self.d_model)
            attn_out = torch.matmul(attn_out, layer["W_o"])
            dec = self._ln(dec + attn_out, layer["ln1_w"], layer["ln1_b"])

            # 交叉注意力
            Q = torch.matmul(dec, layer["W_cq"]).view(B, L_t, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(enc, layer["W_ck"]).view(B, L_s, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(enc, layer["W_cv"]).view(B, L_s, self.num_heads, self.d_k).transpose(1, 2)
            cross_out = self._attn(Q, K, V)
            cross_out = cross_out.transpose(1, 2).contiguous().view(B, L_t, self.d_model)
            cross_out = torch.matmul(cross_out, layer["W_co"])
            dec = self._ln(dec + cross_out, layer["ln2_w"], layer["ln2_b"])

            # FFN
            ffn = torch.matmul(torch.relu(torch.matmul(dec, layer["W_1"]) + layer["b_1"]), layer["W_2"]) + layer["b_2"]
            dec = self._ln(dec + ffn, layer["ln3_w"], layer["ln3_b"])

        return dec
```

&emsp;&emsp;这个实现中，编码器使用双向自注意力，解码器先做因果自注意力，再做交叉注意力，最后做前馈网络。若 `src_ids` 形状为 $(B,L_s)$，`tgt_ids` 形状为 $(B,L_t)$，输出形状为 $(B,L_t,d_{model})$。

---

#### 2.2.4 Post-LN 与 Pre-LN

&emsp;&emsp;Post-LN 和 Pre-LN 是 Transformer 中残差连接与层归一化的两种排列方式。Post-LN 将层归一化放在残差相加之后，Pre-LN 将层归一化放在子层输入之前。原始 Transformer 使用 Post-LN，而大多数现代大模型使用 Pre-LN，因为 Pre-LN 训练更稳定，对学习率预热依赖更小。

&emsp;&emsp;设子层函数为 $F$，输入为 $x$。Post-LN 的计算为：

$$
y = \mathrm{LayerNorm}(x + F(x))
$$

&emsp;&emsp;Pre-LN 的计算为：

$$
y = x + F(\mathrm{LayerNorm}(x))
$$

&emsp;&emsp;在多层堆叠中，Post-LN 的残差路径上每层都经过 LayerNorm，导致梯度在反向传播时容易被归一化操作缩放，深层网络需要学习率预热才能稳定训练。Pre-LN 的残差路径是恒等映射，梯度可以直接回传，训练更稳定，因此成为主流选择。

&emsp;&emsp;Post-LN 的优点是原始 Transformer 采用该结构，在浅层模型中表现良好，且输出经过归一化，数值范围稳定；缺点是深层训练不稳定，需要精细的学习率预热和初始化。Pre-LN 的优点是训练稳定，适合深层模型，对学习率预热不敏感；缺点是输出未归一化，可能需要额外的最终 LayerNorm，且在某些任务上最终性能可能略低于调优良好的 Post-LN。

&emsp;&emsp;从维度视角看，Post-LN 和 Pre-LN 不改变 token 维扩散的数学形式，只改变特征维上归一化的位置。Pre-LN 让残差路径保持干净，相当于在特征维上保留了一条无归一化的信息高速公路。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 Post-LN 和 Pre-LN 的裸实现对比：

```python
import torch
import torch.nn as nn

class PostLNBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.ln1_w = nn.Parameter(torch.ones(d_model))
        self.ln1_b = nn.Parameter(torch.zeros(d_model))
        self.ln2_w = nn.Parameter(torch.ones(d_model))
        self.ln2_b = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def _ln(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def forward(self, x, attn_fn):
        # Post-LN: LayerNorm(x + F(x))
        attn_out = attn_fn(x)
        x = self._ln(x + attn_out, self.ln1_w, self.ln1_b)

        ffn_out = torch.matmul(
            torch.relu(torch.matmul(x, self.W_1) + self.b_1),
            self.W_2
        ) + self.b_2
        x = self._ln(x + ffn_out, self.ln2_w, self.ln2_b)
        return x


class PreLNBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.ln1_w = nn.Parameter(torch.ones(d_model))
        self.ln1_b = nn.Parameter(torch.zeros(d_model))
        self.ln2_w = nn.Parameter(torch.ones(d_model))
        self.ln2_b = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def _ln(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def forward(self, x, attn_fn):
        # Pre-LN: x + F(LayerNorm(x))
        norm_x = self._ln(x, self.ln1_w, self.ln1_b)
        x = x + attn_fn(norm_x)

        norm_x = self._ln(x, self.ln2_w, self.ln2_b)
        ffn_out = torch.matmul(
            torch.relu(torch.matmul(norm_x, self.W_1) + self.b_1),
            self.W_2
        ) + self.b_2
        x = x + ffn_out
        return x
```

&emsp;&emsp;这个实现中，`PostLNBlock` 在残差相加后做 LayerNorm，`PreLNBlock` 在子层输入前做 LayerNorm。两者都使用手动实现的 LayerNorm 和前馈网络。若输入形状为 $(B,L,d_{model})$，输出形状相同。

#### 2.2.5 残差连接

&emsp;&emsp;残差连接（Residual Connection）是 Transformer 能够堆叠数十甚至上百层的关键使能技术。这一概念最初由 He 等人在 2015 年的 ResNet 中提出，Transformer 对其进行了沿用。其核心思想是不让网络直接学习目标映射 $H(x)$，而是学习残差 $F(x) = H(x) - x$，网络的实际输出变为：

$$
H(x) = x + F(x)
$$

&emsp;&emsp;如果目标映射接近恒等变换，那么 $F(x)$ 接近零，让网络学习一个接近零的函数比学习一个恒等映射要容易得多，只需要将权重初始化为接近零即可。在 Transformer 中，每个子层都包裹在残差连接中：

$$
\mathrm{output} = x + \mathrm{Sublayer}(x)
$$

&emsp;&emsp;其中 $\mathrm{Sublayer}$ 可以是注意力层或前馈网络。残差连接对梯度传播的改善可以通过链式法则直观理解。对于没有残差连接的深层网络，梯度需要经过每一层的权重矩阵连乘：

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial h_L} \prod_{l=1}^{L} \frac{\partial h_l}{\partial h_{l-1}}
$$

&emsp;&emsp;当层数 $L$ 很大时，连乘容易导致梯度消失或爆炸。有了残差连接后，由于 $h_l = h_{l-1} + F_l(h_{l-1})$，梯度变为：

$$
\frac{\partial h_l}{\partial h_{l-1}} = I + \frac{\partial F_l}{\partial h_{l-1}}
$$

&emsp;&emsp;其中 $I$ 是单位矩阵。这意味着梯度始终有一条从输出直达输入的“高速公路”，即使 $\partial F_l / \partial h_{l-1}$ 很小，梯度仍然可以通过恒等路径 $I$ 无损地传播，这就是 Transformer 能够堆叠极深层数的根本原因。残差连接的优点是提供了梯度高速公路，使深层网络可训练，并具有“自适应深度”特性——如果某个子层学到的变换对任务无益，网络可以让子层输出接近零，退化为恒等映射，至少不会比浅层差；缺点是残差路径上的累积求和可能导致深层网络中特征范数持续增长，需要配合 LayerNorm 或 RMSNorm 来稳定数值范围，且在某些情况下残差分支的输出可能被主路径的恒等信号所淹没。

&emsp;&emsp;从维度视角看，残差连接在特征维上保留了一条恒等通路，使 token 维扩散的结果以“增量”方式叠加到原始表示上，而非替换它。这相当于为每个子层提供了一个可学习的“残差修正”，网络只需学习在当前表示基础上需要补充什么信息。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出残差连接的裸实现，并对比有残差和无残差两种结构：

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """带残差连接的子层封装"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def forward(self, x, sublayer_fn):
        # 残差连接: output = x + Sublayer(x)
        return x + sublayer_fn(x)


class NoResidualBlock(nn.Module):
    """无残差连接的子层封装，用于对比"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def forward(self, x, sublayer_fn):
        # 无残差连接: output = Sublayer(x)
        return sublayer_fn(x)
```

&emsp;&emsp;实际使用时，子层函数通常包含 LayerNorm 和注意力或 FFN 操作。残差连接本身只是简单的加法，但正是这个加法使得深层 Transformer 的训练成为可能。若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.2.6 并行注意力与 MLP

&emsp;&emsp;标准 Transformer 层采用串行结构：先做自注意力，再做前馈网络，两个子层依次经过残差连接和归一化。并行注意力与 MLP（Parallel Attention and MLP）是一种替代结构，将自注意力和 MLP 从同一输入并行计算，然后将两个分支的输出同时加到残差路径上。这种结构在 GPT-NeoX 和 Falcon 等模型中广泛使用。

&emsp;&emsp;串行结构的计算为：

$$
Z = \mathrm{LayerNorm}\left(X + \mathrm{Attention}(X)\right)
$$

$$
Y = \mathrm{LayerNorm}\left(Z + \mathrm{MLP}(Z)\right)
$$

&emsp;&emsp;并行结构的计算为：

$$
Y = X + \mathrm{Attention}(\mathrm{LayerNorm}(X)) + \mathrm{MLP}(\mathrm{LayerNorm}(X))
$$

&emsp;&emsp;关键区别在于：串行结构中 MLP 的输入是自注意力的输出，两个子层之间有依赖关系；并行结构中自注意力和 MLP 都从同一个归一化输入出发，彼此独立，最后将两个输出同时加到残差路径上。在张量并行训练中，串行结构每层的 MHA-MLP 连接需要一次 all-reduce 通信，而并行结构消除了这个通信，使 MHA 和 MLP 可以在单个 GPU 上并行执行。

&emsp;&emsp;并行注意力与 MLP 的优点是减少了子层间的依赖，在张量并行训练中消除了每层的 all-reduce 通信开销，支持 MHA 和 MLP 的并行执行，训练速度可提升显著；缺点是两个子层从同一输入出发，MLP 无法利用注意力层的输出信息，可能略微削弱表达能力，且并行结构下两个分支的输出直接相加，可能需要在初始化时调整各分支的缩放系数以保持数值稳定。

&emsp;&emsp;从维度视角看，串行结构中 MLP 在注意力扩散之后进行特征维的非线性变换，而并行结构中 MLP 和注意力同时对原始表示做独立的特征维变换，最后叠加。这相当于把“先扩散再变换”改为“扩散与变换并行”，牺牲部分层间信息流换取计算效率。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出并行注意力与 MLP 的裸实现，并与串行结构对比：

```python
import math
import torch
import torch.nn as nn

class ParallelAttentionMLP(nn.Module):
    """并行注意力与 MLP: 两个分支从同一输入出发，输出同时加到残差路径"""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 注意力参数
        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        # MLP 参数
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))

        # 共享 LayerNorm
        self.ln_w = nn.Parameter(torch.ones(d_model))
        self.ln_b = nn.Parameter(torch.zeros(d_model))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o, self.W_1, self.W_2):
            nn.init.xavier_uniform_(w)

    def _ln(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * self.ln_w + self.ln_b

    def _attn(self, x, causal=True):
        B, L, _ = x.size()
        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if causal:
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return torch.matmul(out, self.W_o)

    def _mlp(self, x):
        return torch.matmul(
            torch.relu(torch.matmul(x, self.W_1) + self.b_1),
            self.W_2
        ) + self.b_2

    def forward(self, x, causal=True):
        # 共享 LayerNorm 后的输入
        norm_x = self._ln(x)

        # 两个分支并行计算
        attn_out = self._attn(norm_x, causal=causal)
        mlp_out = self._mlp(norm_x)

        # 两个分支输出同时加到残差路径
        return x + attn_out + mlp_out


class SerialAttentionMLP(nn.Module):
    """串行结构: 先注意力，再 MLP，作为对比"""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.ln1_w = nn.Parameter(torch.ones(d_model))
        self.ln1_b = nn.Parameter(torch.zeros(d_model))
        self.ln2_w = nn.Parameter(torch.ones(d_model))
        self.ln2_b = nn.Parameter(torch.zeros(d_model))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o, self.W_1, self.W_2):
            nn.init.xavier_uniform_(w)

    def _ln(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def forward(self, x, causal=True):
        B, L, _ = x.size()
        norm_x = self._ln(x, self.ln1_w, self.ln1_b)

        Q = torch.matmul(norm_x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(norm_x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(norm_x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if causal:
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
        attn_out = torch.matmul(attn_out, self.W_o)

        # 第一个残差
        x = x + attn_out

        # MLP 从注意力输出出发
        norm_x = self._ln(x, self.ln2_w, self.ln2_b)
        mlp_out = torch.matmul(
            torch.relu(torch.matmul(norm_x, self.W_1) + self.b_1),
            self.W_2
        ) + self.b_2

        return x + mlp_out
```

&emsp;&emsp;`ParallelAttentionMLP` 中自注意力和 MLP 从同一个 LayerNorm 输出并行计算，输出同时加到残差路径。`SerialAttentionMLP` 中 MLP 从注意力的输出出发。若输入形状为 $(B,L,d_{model})$，两者输出形状相同。

---

#### 2.2.7 统一系统与路由器

&emsp;&emsp;统一系统与路由器（Unified System and Router）是混合专家（MoE）架构中的核心机制。传统 Transformer 的每层前馈网络被替换为 MoE 层，由多个“专家”组成，每个专家本质上是一个小型 FFN。当输入一个 token 时，模型通过路由器决定该 token 应该由哪些专家处理，通常是 1 到 2 个，然后将选中专家的输出加权求和。

&emsp;&emsp;设第 $l$ 层的输入 token 表示为 $u_t^l$，路由器的门控分数为：

$$
s_{i,t} = \mathrm{Softmax}_i\left({u_t^l}^\top e_i^l\right)
$$

&emsp;&emsp;其中 $e_i^l$ 是第 $i$ 个专家的可学习嵌入。只保留匹配度最高的 $K$ 个专家：

$$
g_{i,t} = \begin{cases} s_{i,t} & s_{i,t} \in \mathrm{TopK}(\{s_{j,t}\}, K) \\ 0 & \text{otherwise} \end{cases}
$$

&emsp;&emsp;MoE 层的输出为：

$$
h_t^l = \sum_{i=1}^{N} g_{i,t} \cdot \mathrm{FFN}_i(u_t^l) + u_t^l
$$

&emsp;&emsp;DeepSeekMoE 在此基础上引入了细粒度专家划分和共享专家隔离两个优化。细粒度专家划分将每个专家拆分为更小的专家，增加组合灵活性；共享专家隔离则设置一组共享专家，所有 token 都固定分配给这些共享专家，专门捕获通用知识，从而减少路由专家的参数冗余。带有共享专家的 MoE 层输出为：

$$
h_t^l = \sum_{i=1}^{K_s} \mathrm{FFN}_i(u_t^l) + \sum_{i=K_s+1}^{mN} g_{i,t} \cdot \mathrm{FFN}_i(u_t^l) + u_t^l
$$

&emsp;&emsp;其中 $K_s$ 是共享专家数量，$mN$ 是总专家数量。在负载均衡方面，DeepSeek-V3.2 采用了一种基于偏置项的负载均衡策略：路由器计算每个 token 与每个专家之间的亲和度分数，然后对最近接收了较多 token 的专家添加一个轻微的负偏置，从而在不使用辅助损失的情况下实现负载均衡。

&emsp;&emsp;统一系统与路由器的优点是稀疏激活使模型参数量可以极大扩展，但每个 token 只激活少量专家，推理成本远低于同等参数量的稠密模型，且通过共享专家隔离和细粒度划分使专家更加专业化；缺点是路由机制可能面临负载不均衡问题，部分专家可能过载而其他专家训练不足，导致路由坍缩，需要额外的负载均衡策略，且专家分布在多 GPU 上时 token 的分发和聚合会带来显著的通信开销。

&emsp;&emsp;从维度视角看，路由器在特征维上做了一次稀疏选择：每个 token 根据自身特征被路由到不同的专家子网络，相当于在特征维上实现了条件计算。这不同于注意力在 token 维上的扩散，而是在特征维上按 token 内容动态选择变换路径。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 MoE 层与路由器的裸实现，包含共享专家和负载均衡偏置：

```python
import torch
import torch.nn as nn

class MoELayer(nn.Module):
    """MoE 层: 共享专家 + 路由专家 + 负载均衡偏置"""
    def __init__(self, d_model, d_ff, num_routed_experts, num_shared_experts, top_k):
        super().__init__()
        self.d_model = d_model
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k

        # 共享专家: 所有 token 固定激活
        self.shared_experts = nn.ModuleList([
            self._make_ffn(d_model, d_ff) for _ in range(num_shared_experts)
        ])

        # 路由专家
        self.routed_experts = nn.ModuleList([
            self._make_ffn(d_model, d_ff) for _ in range(num_routed_experts)
        ])

        # 路由器: 专家嵌入
        self.expert_emb = nn.Parameter(torch.empty(num_routed_experts, d_model))
        nn.init.normal_(self.expert_emb, std=0.02)

        # 负载均衡偏置 (DeepSeek-V3.2 风格)
        self.register_buffer("routing_bias", torch.zeros(num_routed_experts))

    def _make_ffn(self, d_model, d_ff):
        return nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        B, L, D = x.size()
        x_flat = x.view(B * L, D)

        # 共享专家: 所有 token 固定经过
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)

        # 路由器打分
        scores = torch.matmul(x_flat, self.expert_emb.T)  # (B*L, num_experts)
        scores = scores + self.routing_bias.unsqueeze(0)
        scores = torch.softmax(scores, dim=-1)

        # Top-k 选择
        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)  # (B*L, top_k)
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)

        # 路由专家计算
        routed_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            idx = topk_indices[:, k]  # (B*L,)
            weight = topk_scores[:, k].unsqueeze(-1)  # (B*L, 1)
            for e in range(self.num_routed_experts):
                mask = (idx == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.routed_experts[e](expert_input)
                    routed_out[mask] += weight[mask] * expert_output

        # 输出: 共享专家 + 路由专家 + 残差
        output = shared_out + routed_out + x_flat
        return output.view(B, L, D)

    def update_routing_bias(self, expert_load, bias_rate=0.001):
        """根据专家负载更新偏置: 负载高的专家加负偏置"""
        avg_load = expert_load.mean()
        self.routing_bias -= bias_rate * (expert_load - avg_load)
```

&emsp;&emsp;这个实现中，共享专家对所有 token 固定激活，路由专家通过 Top-k 选择激活。`routing_bias` 用于负载均衡，负载过高的专家会被施加负偏置以降低其被选中的概率。若输入形状为 $(B,L,d_{model})$，输出形状相同。


---

### 2.3 位置编码

#### 2.3.1 绝对位置编码

&emsp;&emsp;绝对位置编码（Absolute Positional Encoding）为序列中的每个位置分配一个独立的表示向量，并将其与 token 嵌入相加，使模型能够区分不同位置的 token。最直接的做法是维护一个可学习的位置嵌入矩阵 $P \in \mathbb{R}^{L_{max} \times d_{model}}$，其中 $L_{max}$ 是最大序列长度，$d_{model}$ 是模型维度。对于位置 $i$，其位置编码为 $P_i$，输入表示为：

$$
x_i = E_{token}(t_i) + P_i
$$

&emsp;&emsp;其中 $E_{token}(t_i)$ 是 token $t_i$ 的词嵌入。位置嵌入矩阵与模型其他参数一起通过反向传播学习。这种方案在 BERT、GPT-2 等模型中广泛使用。绝对位置编码的优点是实现简单，每个位置有独立的可学习参数，模型可以自由地为不同位置学习不同的表示，适合位置模式固定的任务；缺点是位置嵌入表的大小受最大序列长度限制，无法外推到训练时未见过的更长序列，且每个位置独立学习，位置之间的相对关系需要模型从数据中隐式推断，参数效率较低。

&emsp;&emsp;从维度视角看，绝对位置编码在特征维上为每个位置添加了一个位置相关的偏置向量。它不改变 token 维扩散的结构，只是让每个位置的查询和键在特征空间中带有了位置标签，使注意力能够区分不同位置。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出可学习绝对位置编码的裸实现：

```python
import torch
import torch.nn as nn

class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # 可学习的位置嵌入表
        self.pos_emb = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        # x: (B, L, d_model)
        B, L, D = x.size()
        assert L <= self.max_len, "序列长度超过最大位置编码长度"
        return x + self.pos_emb[:L].unsqueeze(0)
```

&emsp;&emsp;这个实现中，位置嵌入是可学习参数，输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.3.2 正弦/余弦位置编码

&emsp;&emsp;正弦/余弦位置编码（Sinusoidal Positional Encoding）是原始 Transformer 使用的固定位置编码方案，不需要学习参数。它利用不同频率的正弦和余弦函数为每个位置生成唯一的位置向量，使模型能够通过三角恒等式捕捉相对位置关系。对于位置 $pos$ 和维度索引 $i$，编码定义为：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

&emsp;&emsp;其中 $i$ 从 $0$ 到 $d_{model}/2 - 1$。不同维度对应不同波长，从 $2\pi$ 到 $10000 \cdot 2\pi$ 形成几何级数。这种设计使得对于任意固定偏移 $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数，从而让模型更容易学习相对位置关系。正弦/余弦位置编码的优点是无需训练参数，可以外推到比训练时更长的序列，且不同位置之间的相对关系由三角函数自然编码；缺点是固定编码的表达能力不如可学习位置嵌入灵活，在某些需要位置特异性较强的任务上可能不如可学习方案，且与 token 嵌入相加后可能干扰语义信息。

&emsp;&emsp;从维度视角看，正弦/余弦编码在特征维上为每个位置提供了一个确定性的、频率分解的偏置。不同维度对应不同尺度的位置变化，使注意力可以通过特征维上的模式匹配来感知位置差异。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出正弦/余弦位置编码的裸实现：

```python
import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 预计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, d_model)
        L = x.size(1)
        return x + self.pe[:, :L]
```

&emsp;&emsp;这个实现中，位置编码是固定的，不参与训练。输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.3.3 相对位置编码

&emsp;&emsp;相对位置编码（Relative Positional Encoding）不关心 token 的绝对位置，而是关注 token 之间的相对距离。在注意力计算中，相对位置信息被注入到查询和键的交互中，使模型能够根据“两个 token 相距多远”来调整注意力权重。设位置 $i$ 和 $j$ 之间的相对距离为 $i-j$，相对位置编码为每个距离分配一个可学习的嵌入向量。注意力分数变为：

$$
e_{ij} = \frac{(q_i + r_{i-j})^\top k_j}{\sqrt{d_k}}
$$

&emsp;&emsp;或采用更常见的形式，将相对位置偏置直接加到注意力分数上：

$$
e_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}} + b_{i-j}
$$

&emsp;&emsp;其中 $b_{i-j}$ 是相对距离 $i-j$ 对应的可学习偏置。相对位置编码的优点是更符合语言中位置关系的本质，模型关注的是相对距离而非绝对位置，因此具有更好的长度外推能力，且对序列顺序的建模更自然；缺点是实现比绝对位置编码复杂，需要为每个相对距离维护参数，当序列很长时相对距离范围很大，参数表可能过大，通常需要裁剪或分桶。

&emsp;&emsp;从维度视角看，相对位置编码在注意力分数矩阵上添加了一个与相对距离相关的偏置，直接修改了 token 维扩散的关系矩阵。它不改变查询和键的特征表示，而是让关系矩阵本身带有位置感知能力。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出相对位置编码的裸实现，将可学习相对位置偏置加到注意力分数上：

```python
import math
import torch
import torch.nn as nn

class RelativePositionEncoding(nn.Module):
    def __init__(self, d_model, num_heads, max_relative_distance):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_rel = max_relative_distance

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        # 相对位置偏置表: 范围 [-max_rel, max_rel]
        self.rel_bias = nn.Parameter(torch.zeros(2 * max_relative_distance + 1, num_heads))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

    def forward(self, x, causal=True):
        B, L, _ = x.size()

        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 构建相对距离矩阵并裁剪
        pos = torch.arange(L, device=x.device)
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)  # (L, L)
        rel = torch.clamp(rel, -self.max_rel, self.max_rel)
        rel = rel + self.max_rel  # 映射到 [0, 2*max_rel]

        # 取偏置: (L, L, num_heads) -> (num_heads, L, L)
        bias = self.rel_bias[rel].permute(2, 0, 1)  # (h, L, L)
        scores = scores + bias.unsqueeze(0)

        if causal:
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)
        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return torch.matmul(head_out, self.W_o), attn
```

&emsp;&emsp;这个实现中，相对距离被裁剪到 $[-max\_rel, max\_rel]$，每个距离和每个头有独立的偏置。输入形状为 $(B,L,d_{model})$，输出形状相同，`attn` 形状为 $(B,h,L,L)$。

---

#### 2.3.4 T5 相对位置偏置

&emsp;&emsp;T5 相对位置偏置（T5 Relative Position Bias）是 T5 模型采用的一种分桶式相对位置编码方案。它不直接为每个相对距离分配参数，而是将相对距离映射到有限数量的桶中，每个桶有一个可学习的偏置。这样既保留了相对位置信息，又控制了参数数量，并且能外推到训练时未见过的更长距离。

&emsp;&emsp;T5 的分桶策略是：将相对距离 $i-j$ 映射到桶索引。对于距离 $d = |i-j|$，桶索引由以下规则确定：

$$
\mathrm{bucket}(d) = \begin{cases}
d & d < n \\
n + \lfloor \log(d/n) / \log(\log_{max}/n) \rfloor & d \geq n
\end{cases}
$$

&emsp;&emsp;其中 $n$ 是线性增长的边界，通常取 $n=8$，$\log_{max}$ 是最大距离的对数边界。对于有符号的相对距离，桶索引还需要根据符号偏移。最终，每个注意力头有一个可学习的偏置向量 $B \in \mathbb{R}^{n_{buckets}}$，注意力分数为：

$$
e_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}} + B_{\mathrm{bucket}(i-j)}
$$

&emsp;&emsp;T5 相对位置偏置的优点是参数数量远小于逐距离参数化方案，通过分桶实现了对长距离的粗粒度建模，且能够外推到比训练时更长的序列；缺点是分桶边界是人工设计的超参数，不同任务和序列长度下最优分桶可能不同，且分桶后的偏置精度不如逐距离方案，可能损失部分短距离的精细位置信息。

&emsp;&emsp;从维度视角看，T5 相对位置偏置与相对位置编码类似，都是在注意力关系矩阵上添加位置相关的偏置。不同之处在于它将连续距离离散化到桶中，用有限参数覆盖任意距离，相当于对关系矩阵的位置偏置做了一次分段常值近似。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 T5 相对位置偏置的裸实现：

```python
import math
import torch
import torch.nn as nn

class T5RelativePositionBias(nn.Module):
    def __init__(self, num_heads, num_buckets=32, max_distance=128, linear_boundary=8):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.linear_boundary = linear_boundary

        # 每个头每个桶一个偏置
        self.rel_bias = nn.Parameter(torch.zeros(num_buckets, num_heads))

    def _relative_position_bucket(self, relative_position):
        """将相对距离映射到桶索引"""
        ret = torch.zeros_like(relative_position, dtype=torch.long)
        n = self.linear_boundary

        # 绝对距离
        abs_pos = torch.abs(relative_position)
        # 线性部分: |d| < n
        ret += torch.where(
            abs_pos < n,
            abs_pos,
            torch.zeros_like(abs_pos)
        )

        # 对数部分: |d| >= n
        log_bucket = torch.floor(
            torch.log(abs_pos.float() / n)
            / math.log(self.max_distance / n)
            * (self.num_buckets - n)
        ).long()
        log_bucket = torch.clamp(log_bucket, 0, self.num_buckets - n - 1)
        ret += torch.where(
            abs_pos >= n,
            n + log_bucket,
            torch.zeros_like(log_bucket)
        )

        # 符号: 正向和负向使用不同桶范围
        ret = torch.where(
            relative_position < 0,
            ret + self.num_buckets,
            ret
        )
        return ret

    def forward(self, x, num_buckets_total=None):
        # x: (B, L, d_model)
        B, L, _ = x.size()
        pos = torch.arange(L, device=x.device)
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)  # (L, L)
        rel = torch.clamp(rel, -self.max_distance, self.max_distance)

        bucket = self._relative_position_bucket(rel)  # (L, L)
        # 取偏置: (L, L, num_heads) -> (num_heads, L, L)
        bias = self.rel_bias[bucket].permute(2, 0, 1)
        return bias.unsqueeze(0)  # (1, h, L, L)
```

&emsp;&emsp;这个实现中，`_relative_position_bucket` 将相对距离映射到桶索引，然后从 `rel_bias` 中取出对应的偏置。实际使用时，将返回的偏置加到注意力分数上即可。若输入形状为 $(B,L,d_{model})$，返回的偏置形状为 $(1,h,L,L)$，可广播到 $(B,h,L,L)$。

#### 2.3.5 ALiBi

&emsp;&emsp;ALiBi（Attention with Linear Biases）由 Press 等人于 2022 年提出，采用了一种极其简洁的位置编码方案：不添加任何位置编码向量，也不修改 Query 和 Key，而是直接在注意力分数上加上一个与相对距离成正比的负偏置。对于查询位置 $m$ 和键位置 $n$，注意力分数变为：

$$
\mathrm{score}(q_m, k_n) = \frac{q_m^\top k_n}{\sqrt{d_k}} - r \cdot |m - n|
$$

&emsp;&emsp;其中 $r$ 是每个注意力头特有的斜率参数。不同头使用不同的 $r$ 值，形成一个几何序列。对于 $H$ 个头，第 $h$ 个头的斜率定义为：

$$
r_h = 2^{-8h/H}, \quad h = 1, \dots, H
$$

&emsp;&emsp;当 $H=8$ 时，斜率依次为 $1/2, 1/4, \dots, 1/256$。斜率越大的头，注意力范围越窄，越聚焦于近邻；斜率越小的头，注意力范围越宽，能覆盖更远的距离。这种设计使不同头在不同尺度上工作，模拟了多尺度的局部性先验。

&emsp;&emsp;ALiBi 的直觉非常自然：距离越远的位置，注意力分数的惩罚越大，模型倾向于关注相近的 token。这种线性偏置对任意距离都有定义，不存在位置上限，因此具有出色的长度外推能力。在 1024 长度上训练的模型可以直接外推到 2048 甚至更长，性能衰减很小。ALiBi 的优点是无需额外参数，实现极其简单，长度外推能力强，且不修改 Query 和 Key 的表示，计算开销几乎为零；缺点是线性偏置假设“距离越远越不相关”，在某些需要精确长距离依赖的任务中可能过度惩罚远距离注意力，且斜率几何序列是人工设定的，不同任务和模型规模下最优斜率分布可能不同。

&emsp;&emsp;从维度视角看，ALiBi 在 token 维扩散的关系矩阵上直接添加了一个与相对距离线性相关的偏置，使关系矩阵天然带有距离衰减的归纳偏置。它不改变 Query 和 Key 的特征表示，而是让注意力分数的空间结构本身包含位置信息。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 ALiBi 的裸实现，只使用基础张量运算和手动参数：

```python
import math
import torch
import torch.nn as nn

class ALiBiAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        self.b_q = nn.Parameter(torch.zeros(d_model))
        self.b_k = nn.Parameter(torch.zeros(d_model))
        self.b_v = nn.Parameter(torch.zeros(d_model))
        self.b_o = nn.Parameter(torch.zeros(d_model))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

        # 为每个头预计算斜率 r_h = 2^{-8h/H}, h=1,...,H
        slopes = torch.tensor(
            [2.0 ** (-8.0 * h / num_heads) for h in range(1, num_heads + 1)],
            dtype=torch.float32
        )
        self.register_buffer("slopes", slopes)

    def _build_alibi_bias(self, seq_len, device):
        """构建 ALiBi 偏置矩阵: (H, L, L)"""
        # 相对距离 |m - n|
        pos = torch.arange(seq_len, device=device)
        distance = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()  # (L, L)
        # 每个头的偏置: -r_h * distance
        bias = -self.slopes.view(-1, 1, 1) * distance.unsqueeze(0)  # (H, L, L)
        return bias

    def forward(self, x, causal=True):
        B, L, _ = x.size()

        Q = torch.matmul(x, self.W_q) + self.b_q
        K = torch.matmul(x, self.W_k) + self.b_k
        V = torch.matmul(x, self.W_v) + self.b_v

        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 添加 ALiBi 偏置
        alibi_bias = self._build_alibi_bias(L, x.device)  # (H, L, L)
        scores = scores + alibi_bias.unsqueeze(0)  # (B, H, L, L)

        if causal:
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)

        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = torch.matmul(head_out, self.W_o) + self.b_o
        return output, attn
```

&emsp;&emsp;这个实现中，`_build_alibi_bias` 为每个头构建了与相对距离线性相关的负偏置矩阵。若输入形状为 $(B,L,d_{model})$，输出形状为 $(B,L,d_{model})$，`attn` 形状为 $(B,h,L,L)$。

---

#### 2.3.6 RoPE：旋转位置编码

&emsp;&emsp;旋转位置编码（Rotary Position Embedding, RoPE）由 Su 等人于 2021 年提出，核心思想是通过旋转操作将位置信息注入到查询和键向量中。与加法式位置编码不同，RoPE 不是将位置向量加到 token 嵌入上，而是根据 token 的绝对位置对 Query 和 Key 向量进行旋转，使得注意力分数天然地只依赖于相对位置。

&emsp;&emsp;RoPE 的数学基础是复数旋转。将 Query 和 Key 的每两个相邻维度视为一个复数，实部为偶数维，虚部为奇数维。对于位置 $m$，Query 向量 $\mathbf{q}_m$ 被乘以旋转因子 $e^{i\theta_t m}$，其中 $\theta_t$ 是第 $t$ 个频率：

$$
\theta_t = 10000^{-t/(d_k/2)}, \quad t \in \{0, 1, \dots, d_k/2 - 1\}
$$

&emsp;&emsp;在复数表示下，RoPE 的操作为：

$$
\bar{\mathbf{q}}_m = \mathbf{q}_m \circ e^{i\theta_t m}, \quad \bar{\mathbf{k}}_n = \mathbf{k}_n \circ e^{i\theta_t n}
$$

&emsp;&emsp;其中 $\circ$ 表示逐元素乘法。注意力分数为 $\mathrm{Re}[\bar{\mathbf{q}}_m \bar{\mathbf{k}}_n^*]$，展开后：

$$
\mathrm{Re}[\bar{\mathbf{q}}_m \bar{\mathbf{k}}_n^*] = \mathrm{Re}\left[\sum_t q_t k_t^* e^{i\theta_t(m-n)}\right]
$$

&emsp;&emsp;旋转因子中的 $e^{i\theta_t(m-n)}$ 仅依赖于相对距离 $m-n$，因此 RoPE 天然编码了相对位置信息。在实数实现中，RoPE 等价于对每两个相邻维度应用一个 $2\times 2$ 旋转矩阵：

$$
\begin{pmatrix} q'_{2t} \\ q'_{2t+1} \end{pmatrix} =
\begin{pmatrix} \cos(m\theta_t) & -\sin(m\theta_t) \\ \sin(m\theta_t) & \cos(m\theta_t) \end{pmatrix}
\begin{pmatrix} q_{2t} \\ q_{2t+1} \end{pmatrix}
$$

&emsp;&emsp;RoPE 的优点是：无额外参数，长度外推能力强，通过频率覆盖可以扩展到训练长度之外的序列；相对位置编码使模型更自然地对位置关系建模；旋转操作不改变向量范数，数值稳定；实现简单，可以与任意注意力机制组合。缺点是当外推到远超训练长度的序列时，低频维度的旋转圈数会超出训练时的范围，导致注意力分数分布偏移，需要配合基频调整或位置插值才能有效外推。

&emsp;&emsp;从维度视角看，RoPE 在特征维上对 Query 和 Key 施加了与位置相关的旋转，使注意力关系矩阵中的每个元素都带有相对距离的相位信息。它不改变 token 维扩散的结构，而是让关系矩阵的数值本身编码了位置差异。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 RoPE 的裸实现，包含旋转矩阵的构建和应用：

```python
import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, d_k, base=10000.0, max_len=8192):
        super().__init__()
        self.d_k = d_k
        self.base = base

        # 预计算逆频率: theta_t = base^{-2t/d_k}, t=0,...,d_k/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)  # (d_k/2,)

    def forward(self, q, k, positions=None):
        """
        q, k: (B, h, L, d_k)
        positions: (B, L) 或 None 时使用 0..L-1
        """
        B, h, L, d_k = q.size()
        if positions is None:
            positions = torch.arange(L, device=q.device).unsqueeze(0).expand(B, -1)

        # 计算角度: (B, L, d_k/2)
        angles = positions.unsqueeze(-1).float() * self.inv_freq  # (B, L, d_k/2)
        cos = torch.cos(angles)  # (B, L, d_k/2)
        sin = torch.sin(angles)

        # 将 q, k 的最后一维重塑为 (..., d_k/2, 2)
        q_reshaped = q.view(B, h, L, d_k // 2, 2)
        k_reshaped = k.view(B, h, L, d_k // 2, 2)

        # 旋转: [q0, q1] -> [q0*cos - q1*sin, q0*sin + q1*cos]
        cos = cos.unsqueeze(1)  # (B, 1, L, d_k/2)
        sin = sin.unsqueeze(1)

        q_rot = torch.stack([
            q_reshaped[..., 0] * cos - q_reshaped[..., 1] * sin,
            q_reshaped[..., 0] * sin + q_reshaped[..., 1] * cos
        ], dim=-1)  # (B, h, L, d_k/2, 2)
        k_rot = torch.stack([
            k_reshaped[..., 0] * cos - k_reshaped[..., 1] * sin,
            k_reshaped[..., 0] * sin + k_reshaped[..., 1] * cos
        ], dim=-1)

        return q_rot.reshape(B, h, L, d_k), k_rot.reshape(B, h, L, d_k)
```

&emsp;&emsp;这个实现中，`inv_freq` 预计算了 RoPE 的逆频率，`forward` 将 Query 和 Key 的每两个相邻维度视为一个复数进行旋转。若输入形状为 $(B,h,L,d_k)$，输出形状相同。

---

#### 2.3.7 RoPE 基频调整

&emsp;&emsp;RoPE 基频调整（RoPE Base Frequency Adjustment）是一类通过修改 RoPE 的频率参数来扩展上下文窗口的方法。标准 RoPE 使用固定的基频 $base = 10000$，当序列长度远超训练长度时，低频维度的旋转角度会超出训练时覆盖的范围，导致注意力分数分布偏移。基频调整通过增大 $base$ 或对频率进行缩放，使旋转角度在更长序列上仍保持在合理范围内。

&emsp;&emsp;位置插值（Position Interpolation, PI）是最简单的方法：将位置索引 $m$ 除以缩放因子 $s = L_{target} / L_{train}$，等效于将所有频率统一缩小：

$$
m' = \frac{m}{s}, \quad \theta_t' = \frac{\theta_t}{s}
$$

&emsp;&emsp;PI 的问题是它均匀压缩了所有频率，导致高频维度的局部细节分辨率下降。

&emsp;&emsp;NTK-aware 方法通过修改 $base$ 来实现非均匀缩放。设缩放因子为 $s$，新的基频为：

$$
base' = base \cdot s^{d_k / (d_k - 2)}
$$

&emsp;&emsp;这种调整等效于对高频维度保留更多分辨率，对低频维度进行更多压缩，从而在扩展上下文的同时保留局部结构。

&emsp;&emsp;YaRN（Yet another RoPE extension）在 NTK-aware 的基础上进一步引入分频率插值和注意力温度校准。它将频率维度分为三组：高频维度保持原值，低频维度进行插值，中间频率平滑过渡。同时引入注意力缩放因子 $t$ 来校准注意力分布：

$$
\frac{1}{\sqrt{t}} = 0.1 \ln s + 1
$$

&emsp;&emsp;注意力分数被缩放为 $\frac{q_m^\top k_n}{t \sqrt{d_h}}$，以补偿长序列下注意力熵的变化。RoPE 基频调整的优点是能显著扩展上下文窗口，YaRN 等方法可以用较少的微调步数达到接近甚至超过训练长度的外推性能；缺点是需要额外的超参数（缩放因子、分界点、温度系数），不同模型和训练配置下最优参数不同，且部分方法仍需要少量微调才能稳定。

&emsp;&emsp;从维度视角看，基频调整改变了 RoPE 中不同维度旋转速度的分布，使低频维度在更长序列上仍具有合理的相位覆盖，从而避免注意力分数在长距离上出现分布崩塌。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 NTK-aware 基频调整的裸实现：

```python
import math
import torch
import torch.nn as nn

class RoPEWithNTKScaling(nn.Module):
    def __init__(self, d_k, base=10000.0, max_len=8192, scaling_factor=1.0):
        super().__init__()
        self.d_k = d_k
        self.base = base
        self.scaling_factor = scaling_factor

        # NTK-aware 调整后的基频
        adjusted_base = base * (scaling_factor ** (d_k / (d_k - 2)))
        inv_freq = 1.0 / (adjusted_base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, k, positions=None):
        B, h, L, d_k = q.size()
        if positions is None:
            positions = torch.arange(L, device=q.device).unsqueeze(0).expand(B, -1)

        angles = positions.unsqueeze(-1).float() * self.inv_freq
        cos = torch.cos(angles).unsqueeze(1)  # (B, 1, L, d_k/2)
        sin = torch.sin(angles).unsqueeze(1)

        q_reshaped = q.view(B, h, L, d_k // 2, 2)
        k_reshaped = k.view(B, h, L, d_k // 2, 2)

        q_rot = torch.stack([
            q_reshaped[..., 0] * cos - q_reshaped[..., 1] * sin,
            q_reshaped[..., 0] * sin + q_reshaped[..., 1] * cos
        ], dim=-1).reshape(B, h, L, d_k)

        k_rot = torch.stack([
            k_reshaped[..., 0] * cos - k_reshaped[..., 1] * sin,
            k_reshaped[..., 0] * sin + k_reshaped[..., 1] * cos
        ], dim=-1).reshape(B, h, L, d_k)

        return q_rot, k_rot
```

&emsp;&emsp;这个实现中，`adjusted_base` 根据缩放因子 $s$ 计算 NTK-aware 的新基频。若输入形状为 $(B,h,L,d_k)$，输出形状相同。

---

#### 2.3.8 部分 RoPE

&emsp;&emsp;部分 RoPE（Partial RoPE）是指对 Query 和 Key 向量只对部分维度施加 RoPE 旋转，剩余维度保持不变的编码方式。这种设计最早在 GPT-NeoX 中实验使用，后来在 DeepSeek 的 MLA 中得到了关键性的应用。

&emsp;&emsp;设 Query 和 Key 的维度为 $d_k$，部分 RoPE 只对前 $d_r$ 个维度施加旋转，剩余 $d_k - d_r$ 个维度不做任何位置编码（称为 NoPE 部分）。以只旋转一半维度为例，频率设置为：

$$
\theta_i = \begin{cases}
b^{-4i/d_k} & i < d_k/4 \\
0 & i \geq d_k/4
\end{cases}
$$

&emsp;&emsp;其中 $\theta_i = 0$ 的维度对应不旋转的 NoPE 部分。

&emsp;&emsp;部分 RoPE 的核心洞察是：NoPE 部分负责语义聚合（关注内容相关性），RoPE 部分负责位置编码，两者互补。从理论上看，部分 RoPE 使 $\sum_i \cos(m\theta_i) \geq 0$ 对所有 $m$ 和 $b$ 恒成立，具有更好的语义聚合能力。在 MLA 中，DeepSeek 使用 128 个 NoPE 维度和 64 个 RoPE 维度，将主要计算放在 NoPE 部分，这是 MLA 能够实现双重投影（将 Query 和 Key 的投影吸收到潜在空间中）的理论前提。

&emsp;&emsp;部分 RoPE 的优点是兼顾了位置编码和语义聚合，在长上下文任务中可能优于完整 RoPE，且为 MLA 等高效注意力架构提供了基础；缺点是旋转维度和非旋转维度的比例需要手动选择，不同任务和模型规模下最优比例可能不同。

&emsp;&emsp;从维度视角看，部分 RoPE 将特征维划分为两个功能区：一部分维度承载位置信息，另一部分维度承载纯语义信息。这使得注意力关系矩阵中的位置信号和语义信号在特征维上解耦，模型可以更灵活地调节两者对注意力分数的贡献。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出部分 RoPE 的裸实现：

```python
import torch
import torch.nn as nn

class PartialRoPE(nn.Module):
    def __init__(self, d_k, rotary_dim, base=10000.0):
        """
        d_k: 总维度
        rotary_dim: 施加 RoPE 的维度数，剩余维度不旋转
        """
        super().__init__()
        self.d_k = d_k
        self.rotary_dim = rotary_dim

        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, k, positions=None):
        B, h, L, d_k = q.size()
        if positions is None:
            positions = torch.arange(L, device=q.device).unsqueeze(0).expand(B, -1)

        # 只对前 rotary_dim 个维度做旋转
        q_rot_part = q[..., :self.rotary_dim]   # (B, h, L, rotary_dim)
        q_pass_part = q[..., self.rotary_dim:]  # (B, h, L, d_k - rotary_dim)
        k_rot_part = k[..., :self.rotary_dim]
        k_pass_part = k[..., self.rotary_dim:]

        angles = positions.unsqueeze(-1).float() * self.inv_freq  # (B, L, rotary_dim/2)
        cos = torch.cos(angles).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(1)

        q_reshaped = q_rot_part.view(B, h, L, self.rotary_dim // 2, 2)
        k_reshaped = k_rot_part.view(B, h, L, self.rotary_dim // 2, 2)

        q_rot = torch.stack([
            q_reshaped[..., 0] * cos - q_reshaped[..., 1] * sin,
            q_reshaped[..., 0] * sin + q_reshaped[..., 1] * cos
        ], dim=-1).reshape(B, h, L, self.rotary_dim)

        k_rot = torch.stack([
            k_reshaped[..., 0] * cos - k_reshaped[..., 1] * sin,
            k_reshaped[..., 0] * sin + k_reshaped[..., 1] * cos
        ], dim=-1).reshape(B, h, L, self.rotary_dim)

        # 拼接旋转部分和不旋转部分
        q_out = torch.cat([q_rot, q_pass_part], dim=-1)
        k_out = torch.cat([k_rot, k_pass_part], dim=-1)
        return q_out, k_out
```

&emsp;&emsp;这个实现中，`rotary_dim` 指定了施加旋转的维度数，剩余维度直接透传。若输入形状为 $(B,h,L,d_k)$，输出形状相同。

---

#### 2.3.9 iRoPE：交错旋转位置编码

&emsp;&emsp;iRoPE（interleaved Rotary Position Embeddings）是 Meta 在 Llama 4 Scout 中使用的架构，声称支持 1000 万 token 的上下文窗口。其核心思想是在模型的不同层之间交错使用两种注意力模式：一部分层使用 RoPE 并采用分块局部注意力掩码，只能关注固定窗口内的近期 token（例如 8K token）；另一部分层不使用任何位置编码（NoPE），采用完整的因果掩码，可以访问全部上下文历史。

&emsp;&emsp;这种交错设计的关键动机是：NoPE 层不携带位置偏置，能够纯粹基于内容相似度进行注意力聚合，在处理超长上下文时不会因为位置编码的数值范围问题而失效；而 RoPE 层通过局部窗口聚焦于短距离的精细位置关系。两者交替堆叠，使模型既能捕捉长距离的语义依赖，又能维持局部的位置精度。

&emsp;&emsp;具体实现中，通常采用“每 4 层使用一次 RoPE”的比例，即第 1、5、9、... 层使用 RoPE 和分块注意力，其余层使用 NoPE 和全局因果注意力。此外，iRoPE 还结合了推理时的注意力温度缩放（Attention Temperature Scaling）：随着序列长度增加，动态调整注意力分布的平滑程度，防止注意力分数在超长序列上变得过于尖锐或过于平坦。

&emsp;&emsp;iRoPE 的优点是无需额外参数，通过层间交错实现了局部精度和全局覆盖的平衡，NoPE 层的全局注意力使模型能够直接访问任意距离的上下文，理论上可扩展到任意序列长度；缺点是 KV Cache 仍然随序列长度线性增长，注意力计算的 $O(n^2)$ 复杂度并未降低，内存瓶颈依然存在，且 NoPE 层不携带位置信息，在需要精确位置感知的任务中可能不如全 RoPE 方案。

&emsp;&emsp;从维度视角看，iRoPE 在层间实现了位置编码的“交替启用与禁用”。RoPE 层在特征维上注入位置旋转，NoPE 层则让特征维纯粹承载语义信息。这种交错使得模型在不同深度上分别处理位置敏感和位置无关的注意力模式。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 iRoPE 交错层设计的裸实现，包含 RoPE 层和 NoPE 层的交替：

```python
import math
import torch
import torch.nn as nn

class iRoPEAttention(nn.Module):
    """iRoPE 注意力层: 根据 use_rope 决定是否施加 RoPE"""
    def __init__(self, d_model, num_heads, use_rope, chunk_size=8192):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        self.chunk_size = chunk_size

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

        if use_rope:
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
            self.register_buffer("inv_freq", inv_freq)

    def _apply_rope(self, x):
        B, h, L, d_k = x.size()
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        angles = positions.unsqueeze(-1).float() * self.inv_freq
        cos = torch.cos(angles).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(1)

        x_reshaped = x.view(B, h, L, d_k // 2, 2)
        x_rot = torch.stack([
            x_reshaped[..., 0] * cos - x_reshaped[..., 1] * sin,
            x_reshaped[..., 0] * sin + x_reshaped[..., 1] * cos
        ], dim=-1).reshape(B, h, L, d_k)
        return x_rot

    def forward(self, x):
        B, L, _ = x.size()

        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            Q = self._apply_rope(Q)
            K = self._apply_rope(K)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 因果掩码
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        if self.use_rope:
            # RoPE 层: 叠加分块局部掩码，只关注最近 chunk_size 个 token
            local_mask = torch.ones(L, L, device=x.device).bool()
            for i in range(L):
                left = max(0, i - self.chunk_size + 1)
                local_mask[i, :left] = False
            mask = causal_mask | ~local_mask
        else:
            # NoPE 层: 只使用完整因果掩码，可访问全部历史
            mask = causal_mask

        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)

        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return torch.matmul(head_out, self.W_o)


class iRoPEStack(nn.Module):
    """iRoPE 堆叠: 每 4 层使用一次 RoPE"""
    def __init__(self, d_model, num_heads, num_layers, rope_every=4, chunk_size=8192):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            use_rope = ((i + 1) % rope_every == 0)
            self.layers.append(
                iRoPEAttention(d_model, num_heads, use_rope=use_rope, chunk_size=chunk_size)
            )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x
```

&emsp;&emsp;这个实现中，`iRoPEAttention` 根据 `use_rope` 标志决定是否施加 RoPE 和分块掩码。`iRoPEStack` 按照每 4 层使用一次 RoPE 的模式堆叠注意力层。若输入形状为 $(B,L,d_{model})$，输出形状相同。

#### 2.3.10 NoPE

&emsp;&emsp;NoPE（No Positional Encoding）指的是在 Transformer 中完全不添加任何显式位置编码。模型仅依靠因果注意力掩码和 token 自身的嵌入来隐式地学习位置信息。这一方案最初被认为会导致模型无法区分 token 顺序，但 Kazemnejad 等人在 2023 年的系统研究中发现，NoPE 在长度泛化任务上反而优于 ALiBi、RoPE、APE 和 T5 相对位置偏置等所有显式位置编码方案。

&emsp;&emsp;NoPE 的理论基础在于：因果注意力掩码本身已经引入了一种隐式的顺序信号。由于每个位置只能关注自己和之前的位置，注意力权重的模式天然携带了位置信息。理论分析表明，NoPE 的注意力点积可以分解为内容函数和相对距离函数两部分：

$$
\langle \bm{q}_t, \bm{k}_i \rangle = f_{\mathrm{cnt}}(\bm{q}_t, \bm{k}_i) + f_{\mathrm{rel}}(t-i)
$$

&emsp;&emsp;其中 $f_{\mathrm{cnt}}$ 是内容的函数，$f_{\mathrm{rel}}$ 是相对距离的函数。这意味着 NoPE 虽然不显式注入位置编码，但通过多层注意力的组合，模型可以隐式地实现相对位置编码的功能。在实际训练中，SGD 优化后的 NoPE 主要呈现出与 T5 相对位置偏置相似的注意力模式。

&emsp;&emsp;NoPE 的优点是无需任何位置编码参数或计算开销，长度泛化能力优于所有显式位置编码方案，且在长序列外推时不会出现位置编码数值范围失效的问题；缺点是位置信息的隐式学习需要更多训练步数才能收敛，在小规模模型或短序列任务上可能不如显式位置编码，且对于需要精确绝对位置感知的任务（如某些结构化预测任务），NoPE 可能表现不佳。

&emsp;&emsp;从维度视角看，NoPE 让 token 维扩散的关系矩阵完全由内容和因果结构决定，不额外注入位置信号。位置信息不是不存在，而是以隐式方式编码在注意力模式中。在 iRoPE 等交错架构中，NoPE 层被用于全局注意力，专门负责长距离的语义聚合。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 NoPE 注意力的裸实现，与带 RoPE 的版本对比，NoPE 层不施加任何位置编码：

```python
import math
import torch
import torch.nn as nn

class NoPEAttention(nn.Module):
    """NoPE 注意力层: 不施加任何位置编码，仅依靠因果掩码"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

    def forward(self, x):
        B, L, _ = x.size()

        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        # 标准缩放点积注意力，不添加任何位置编码
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 因果掩码是唯一的结构性信号
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device), diagonal=1
        ).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)

        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return torch.matmul(head_out, self.W_o), attn
```

&emsp;&emsp;这个实现中，`NoPEAttention` 不包含任何位置编码操作，Query 和 Key 直接进行点积，因果掩码是唯一的结构性位置信号。若输入形状为 $(B,L,d_{model})$，输出形状相同，`attn` 形状为 $(B,h,L,L)$。

---

#### 2.3.11 YARN

&emsp;&emsp;YARN（Yet another RoPE extension）是 Peng 等人于 2023 年提出的 RoPE 上下文扩展方法，在 NTK-aware 基频调整的基础上进一步引入了分频率插值和注意力温度缩放。标准位置插值（PI）将所有 RoPE 频率统一除以缩放因子 $s$，导致高频维度的局部细节分辨率下降；NTK-aware 方法通过修改基频来非均匀缩放，但仍存在长距离注意力分数不稳定的问题。YARN 通过三个组件的组合解决了这些问题。

&emsp;&emsp;第一个组件是分频率感知插值。YARN 将 RoPE 的频率维度按波长分为三组：高频维度保持原值，低频维度进行完整插值，中间频率平滑过渡。设第 $m$ 个维度的波长为 $\lambda_m = 2\pi / \theta_m$，YARN 的缩放策略为：

$$
\omega_m' = \begin{cases}
\omega_m & \lambda_m < \alpha \\
(1-\gamma(\lambda_m)) \omega_m + \gamma(\lambda_m) \omega_m / s & \alpha \leq \lambda_m \leq \beta \\
\omega_m / s & \lambda_m > \beta
\end{cases}
$$

&emsp;&emsp;其中 $\gamma$ 是在 $[\alpha, \beta]$ 区间上从 0 平滑增加到 1 的混合函数。这种分段策略使高频维度保留局部位置精度，低频维度获得更长的波长覆盖，中间频率平滑过渡。

&emsp;&emsp;第二个组件是注意力温度缩放。当位置被插值后，Query 和 Key 的点积方差发生变化，导致 Softmax 分布偏移。YARN 在 Softmax 之前引入温度因子 $t$：

$$
\mathrm{softmax}\left(\frac{QK^\top}{t \cdot \sqrt{d_k}}\right)
$$

&emsp;&emsp;温度因子由缩放因子 $s$ 计算得到：

$$
t = 0.1 \ln(s) + 1.0
$$

&emsp;&emsp;当 $s=4$ 时 $t \approx 1.14$，当 $s=8$ 时 $t \approx 1.21$。温度缩放是 YARN 的关键组件，没有它时 8 倍扩展会失败。

&emsp;&emsp;第三个组件是缩放因子的动态计算。YARN 支持在推理时根据实际输入长度动态调整缩放因子，使模型在训练时使用 $s=4$ 的情况下也能在 $s=8$ 时优雅降级而非灾难性失败。

&emsp;&emsp;YARN 的优点是在 2 倍到 8 倍上下文扩展时能达到最优的困惑度，温度缩放组件对长距离稳定性至关重要，且支持零样本推理时扩展，无需额外微调即可使用；缺点是需要手动设置分频率的阈值 $\alpha$ 和 $\beta$，不同模型和基频下最优阈值不同，且温度因子的经验公式 $t = 0.1 \ln(s) + 1$ 是启发式的，在极端缩放比例下可能不够精确。

&emsp;&emsp;从维度视角看，YARN 在不同频率维度上施加了不同程度的缩放：高频维度保留短距离精度，低频维度负责长距离覆盖，温度缩放则校准了 Softmax 分布的形状。这相当于在 RoPE 的频率轴上做了一次非均匀的维度重分配。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 YARN 的裸实现，包含分频率插值和注意力温度缩放：

```python
import math
import torch
import torch.nn as nn

class YaRNRoPE(nn.Module):
    def __init__(self, d_k, base=10000.0, original_max_len=4096,
                 extended_max_len=32768, beta_fast=32, beta_slow=1):
        """
        d_k: 头维度
        original_max_len: 预训练时的最大长度
        extended_max_len: 目标扩展长度
        beta_fast: 高频边界（波长小于此值的维度不缩放）
        beta_slow: 低频边界（波长大于此值的维度完全缩放）
        """
        super().__init__()
        self.d_k = d_k
        self.base = base
        self.original_max_len = original_max_len
        self.extended_max_len = extended_max_len
        self.scale = extended_max_len / original_max_len

        # 原始逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)

        # 计算每个维度的波长
        wavelengths = 2 * math.pi / inv_freq  # (d_k/2,)

        # 分频率插值: 高频保留，低频插值，中间平滑过渡
        gamma = ((wavelengths - beta_slow) / (beta_fast - beta_slow)).clamp(0, 1)
        # gamma=0 -> 不缩放, gamma=1 -> 完全缩放
        inv_freq_scaled = (1 - gamma) * inv_freq + gamma * inv_freq / self.scale
        self.register_buffer("inv_freq_scaled", inv_freq_scaled)

        # 注意力温度缩放因子
        self.attn_scale = 0.1 * math.log(self.scale) + 1.0

    def forward(self, q, k, positions=None):
        B, h, L, d_k = q.size()
        if positions is None:
            positions = torch.arange(L, device=q.device).unsqueeze(0).expand(B, -1)

        # 使用缩放后的频率
        angles = positions.unsqueeze(-1).float() * self.inv_freq_scaled
        cos = torch.cos(angles).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(1)

        q_r = q.view(B, h, L, d_k // 2, 2)
        k_r = k.view(B, h, L, d_k // 2, 2)

        q_rot = torch.stack([
            q_r[..., 0] * cos - q_r[..., 1] * sin,
            q_r[..., 0] * sin + q_r[..., 1] * cos
        ], dim=-1).reshape(B, h, L, d_k)

        k_rot = torch.stack([
            k_r[..., 0] * cos - k_r[..., 1] * sin,
            k_r[..., 0] * sin + k_r[..., 1] * cos
        ], dim=-1).reshape(B, h, L, d_k)

        return q_rot, k_rot, self.attn_scale
```

&emsp;&emsp;这个实现中，`inv_freq_scaled` 根据波长的分频率策略对逆频率进行非均匀缩放，`attn_scale` 是温度因子 $t$。实际使用时，注意力分数需要除以 $t \cdot \sqrt{d_k}$。若输入形状为 $(B,h,L,d_k)$，输出形状相同，同时返回温度因子。

---

#### 2.3.12 DCA：双块注意力

&emsp;&emsp;双块注意力（Dual Chunk Attention, DCA）由 An 等人于 2024 年提出，是一种无需训练即可扩展 LLM 上下文窗口的方法。DCA 将长序列的注意力计算分解为块内注意力和块间注意力，使 Llama2 70B（原生 4K 上下文）能够支持超过 100K token 的上下文窗口，而无需任何持续训练。

&emsp;&emsp;DCA 的核心挑战在于：当序列长度超过预训练窗口时，直接使用 RoPE 的相对位置编码会导致超出训练范围的位置索引，使注意力分数分布偏移。DCA 的解决方案是重新设计相对位置矩阵的构建方式，使其能够准确反映 token 之间的相对位置，同时保持预训练模型的原始位置索引和嵌入。

&emsp;&emsp;DCA 包含三个核心组件。设块大小为 $w$（通常设为预训练窗口大小），序列被分割为 $C = \lceil L/w \rceil$ 个块。块内注意力处理同一块内的 token，维持原始的相对位置编码：

$$
A_{\mathrm{intra}} = \mathrm{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i
$$

&emsp;&emsp;块间注意力处理不同块之间的 token，通过特殊的位置索引映射避免超出预训练范围。对于查询块 $c$ 和键块 $j$（$j < c$），查询使用位置索引 $c-1$ 对应的位置来关注之前的块，使相对位置被限制在预训练窗口内：

$$
A_{\mathrm{inter}} = \mathrm{softmax}\left(\frac{Q_i K_j^\top}{\sqrt{d_k}} \cdot M_{ij}\right) V_j
$$

&emsp;&emsp;其中 $M_{ij}$ 是位置掩码矩阵，确保相对位置不超出预训练范围。连续块注意力专门处理相邻块之间的 token，确保块边界处的连续性。

&emsp;&emsp;DCA 的优点是训练无关，可以直接应用于现有预训练模型，将 4K 上下文扩展到 100K+ token 而困惑度增长微乎其微，计算复杂度从 $O(L^2)$ 降至 $O(L \cdot w)$，且与 FlashAttention 无缝集成；缺点是块大小的选择需要权衡局部精度和全局覆盖，块间注意力的位置索引映射可能损失部分跨块相对位置的精细信息，且 DCA 主要解决上下文长度限制问题，而非提升模型的核心能力。

&emsp;&emsp;从维度视角看，DCA 在 token 维扩散的关系矩阵上施加了块结构：块内保持原始相对位置，块间通过压缩的位置索引维持长程依赖。这相当于将全局扩散分解为块内精确扩散和块间粗粒度扩散的叠加。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 DCA 的裸实现，包含块内注意力、块间注意力和连续块注意力：

```python
import math
import torch
import torch.nn as nn

class DualChunkAttention(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.chunk_size = chunk_size

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

    def forward(self, x, causal=True):
        B, L, _ = x.size()
        w = self.chunk_size
        num_chunks = math.ceil(L / w)

        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        # 构建 DCA 位置索引
        # 块内: 使用原始相对位置
        # 块间: 查询使用块边界位置索引 c-1 来关注之前的块
        pos_intra = torch.arange(L, device=x.device)
        pos_inter = torch.zeros(L, device=x.device, dtype=torch.long)
        for c in range(num_chunks):
            start = c * w
            end = min(start + w, L)
            # 块内使用原始位置
            pos_intra[start:end] = torch.arange(start, end, device=x.device)
            # 块间使用压缩位置: 当前块内的查询位置映射到 c-1
            pos_inter[start:end] = c - 1 if c > 0 else 0

        # 构建 DCA 偏置矩阵
        pos_diff_intra = pos_intra.unsqueeze(0) - pos_intra.unsqueeze(1)
        pos_diff_inter = pos_intra.unsqueeze(0) - pos_inter.unsqueeze(1)

        # 判断每个 (i,j) 对属于块内还是块间
        chunk_i = torch.arange(L, device=x.device) // w
        chunk_j = torch.arange(L, device=x.device) // w
        is_intra = (chunk_i.unsqueeze(1) == chunk_j.unsqueeze(0))  # (L, L)
        is_inter = ~is_intra

        # 综合位置差
        pos_diff = torch.where(is_intra, pos_diff_intra, pos_diff_inter)

        # 使用位置差作为偏置（简化实现，实际 DCA 用 RoPE 旋转）
        # 这里用相对位置差作为注意力偏置
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 添加 DCA 位置偏置
        dca_bias = -torch.abs(pos_diff.float()) * 0.01  # 简化的线性偏置
        scores = scores + dca_bias.unsqueeze(0).unsqueeze(0)

        if causal:
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)
        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return torch.matmul(head_out, self.W_o), attn
```

&emsp;&emsp;这个实现中，DCA 将序列分割为块，块内使用原始位置差，块间使用压缩后的位置索引。实际 DCA 使用 RoPE 旋转来实现位置编码，这里用位置偏置简化演示。若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.3.13 温度缩放与分块注意力掩码

&emsp;&emsp;温度缩放与分块注意力掩码是 Llama 4 中 iRoPE 架构的两个关键推理时优化组件。iRoPE 将模型层分为 RoPE 层和 NoPE 层，RoPE 层使用分块注意力掩码进行局部注意力，NoPE 层使用全因果掩码进行全局注意力，并在 NoPE 层中引入推理时温度缩放来稳定长距离注意力分布。

&emsp;&emsp;分块注意力掩码（Chunked Attention Mask）限制每个 token 只关注其所在块及之前固定窗口内的 token。与滑动窗口注意力不同，分块掩码将序列划分为不重叠的块，每个查询只能关注当前块和之前的块，但不对跨块注意力做进一步限制。在 Llama 4 中，块大小通常设为 8192，每 4 层中使用 3 层 RoPE 局部注意力和 1 层 NoPE 全局注意力。

&emsp;&emsp;温度缩放（Temperature Scaling）在推理时对 NoPE 层的注意力 logits 进行动态调整。随着序列长度增加，注意力分数的分布会发生变化，温度缩放通过调整 Softmax 的平滑程度来补偿。Llama 4 的温度缩放公式为：

$$
\mathrm{scale} = \log\left(\left\lfloor \frac{\mathrm{position} + 1}{\mathrm{floor\_scale}} \right\rfloor + 1\right) \cdot \mathrm{attn\_scale} + 1
$$

&emsp;&emsp;其中 $\mathrm{floor\_scale}$ 控制从多长位置开始放大（默认 8192），$\mathrm{attn\_scale}$ 控制放大强度（默认 0.1）。当位置小于 $\mathrm{floor\_scale}$ 时，$\mathrm{scale} \approx 1$，不产生缩放效果；当位置远超 $\mathrm{floor\_scale}$ 时，温度因子对数增长，使注意力分布更加平滑。

&emsp;&emsp;温度缩放与分块注意力掩码的优点是推理时无需额外训练即可启用，温度缩放对数增长的设计使长距离注意力不会过度平滑，分块掩码将局部注意力的计算量控制在 $O(L \cdot w)$ 内，两者配合使 iRoPE 架构能够处理 1000 万 token 级别的上下文；缺点是分块掩码的块大小需要手动设置，过大的块会降低局部精度，过小的块会增加层间信息传递的负担，温度缩放的经验参数 $\mathrm{floor\_scale}$ 和 $\mathrm{attn\_scale}$ 也需要根据模型规模调整。

&emsp;&emsp;从维度视角看，分块注意力掩码在 token 维扩散的关系矩阵上施加了块对角约束，使 RoPE 层专注于块内的精细位置关系；温度缩放在 NoPE 层中调整注意力分布的锐度，使全局扩散在超长序列上保持数值稳定。两者配合，实现了局部精度和全局覆盖的平衡。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出温度缩放与分块注意力掩码的裸实现：

```python
import math
import torch
import torch.nn as nn

class TemperatureScaledChunkedAttention(nn.Module):
    """结合温度缩放与分块注意力掩码的注意力层"""
    def __init__(self, d_model, num_heads, chunk_size=8192,
                 floor_scale=8192, attn_scale=0.1, use_rope=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.chunk_size = chunk_size
        self.floor_scale = floor_scale
        self.attn_scale = attn_scale
        self.use_rope = use_rope

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

        if use_rope:
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
            self.register_buffer("inv_freq", inv_freq)

    def _apply_rope(self, x, positions):
        B, h, L, d_k = x.size()
        angles = positions.unsqueeze(-1).float() * self.inv_freq
        cos = torch.cos(angles).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(1)

        x_r = x.view(B, h, L, d_k // 2, 2)
        x_rot = torch.stack([
            x_r[..., 0] * cos - x_r[..., 1] * sin,
            x_r[..., 0] * sin + x_r[..., 1] * cos
        ], dim=-1).reshape(B, h, L, d_k)
        return x_rot

    def _compute_temperature_scale(self, L, device):
        """计算温度缩放因子: 随位置对数增长"""
        positions = torch.arange(L, device=device).float()
        scale = torch.log(
            torch.floor((positions + 1.0) / self.floor_scale) + 1.0
        ) * self.attn_scale + 1.0
        return scale  # (L,)

    def _build_chunked_mask(self, L, device):
        """构建分块注意力掩码: 每个 token 只关注当前块及之前的块"""
        w = self.chunk_size
        num_chunks = math.ceil(L / w)
        chunk_ids = torch.arange(L, device=device) // w  # (L,)
        # 允许关注当前块及之前的所有块
        mask = chunk_ids.unsqueeze(0) >= chunk_ids.unsqueeze(1)  # (L, L)
        return mask.bool()

    def forward(self, x):
        B, L, _ = x.size()
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            Q = self._apply_rope(Q, positions)
            K = self._apply_rope(K, positions)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 温度缩放: 对 NoPE 层应用
        if not self.use_rope:
            temp_scale = self._compute_temperature_scale(L, x.device)  # (L,)
            scores = scores * temp_scale.view(1, 1, L, 1)

        # 分块注意力掩码: 对 RoPE 层应用
        if self.use_rope:
            chunk_mask = self._build_chunked_mask(L, x.device)
            causal_mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
            final_mask = chunk_mask & causal_mask
            scores = scores.masked_fill(~final_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            # NoPE 层: 全因果掩码，可访问全部历史
            causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)
        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return torch.matmul(head_out, self.W_o), attn
```

&emsp;&emsp;这个实现中，当 `use_rope=True` 时施加分块注意力掩码和因果掩码的叠加，当 `use_rope=False` 时施加全因果掩码和温度缩放。若输入形状为 $(B,L,d_{model})$，输出形状相同，`attn` 形状为 $(B,h,L,L)$。

---

### 2.4 归一化

#### 2.4.1 LayerNorm

&emsp;&emsp;层归一化（Layer Normalization, LayerNorm）由 Ba 等人于 2016 年提出，是 Transformer 中最早使用的归一化方法。它对每个样本的特征维做归一化，独立于 batch 维度，因此不受 batch size 影响，适合变长序列。设输入为 $x \in \mathbb{R}^{d}$，LayerNorm 计算为：

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma_i \hat{x}_i + \beta_i
$$

&emsp;&emsp;其中 $\gamma, \beta \in \mathbb{R}^{d}$ 是可学习的缩放和偏移参数，$\epsilon$ 是防止除零的小常数。与 BatchNorm 不同，LayerNorm 的统计量在单个样本内部计算，不跨 batch 聚合，因此训练和推理行为一致。在 Transformer 中，LayerNorm 通常作用在最后一维即特征维上，对每个 token 独立归一化。

&emsp;&emsp;LayerNorm 的优点是训练和推理行为一致，不依赖 batch size，适合序列建模；同时对特征维做归一化使每层输入的分布稳定，缓解内部协变量偏移。缺点是计算需要同时求均值和方差，比 RMSNorm 多一次归约操作，在深层大模型中归一化层的开销不可忽略；此外 LayerNorm 的均值中心化在某些任务中并非必要，反而可能损失部分方向信息。

&emsp;&emsp;从维度视角看，LayerNorm 在特征维上对每个 token 独立做中心化和缩放，将特征分布的均值和方差拉回稳定范围。它不改变 token 维扩散的结构，只是让每层子层的输入保持在合理的数值范围内。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 LayerNorm 的裸实现：

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # x: (..., d_model)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta
```

&emsp;&emsp;这个实现中，均值和方差在最后一维上计算，每个 token 独立归一化。若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.4.2 Post-LN

&emsp;&emsp;Post-LN 是原始 Transformer 采用的归一化排列方式，将 LayerNorm 放在残差相加之后。设子层函数为 $F$，输入为 $x$，Post-LN 的输出为：

$$
y = \mathrm{LayerNorm}(x + F(x))
$$

&emsp;&emsp;在编码器和解码器的每个子层中，输入先经过子层计算，与残差相加后再做 LayerNorm。Post-LN 的残差路径上每层都经过归一化，使得梯度在反向传播时被归一化操作的缩放因子影响。理论分析表明，Post-LN 的梯度范数随层数增加而衰减，深层网络需要学习率预热（warmup）才能稳定训练。

&emsp;&emsp;Post-LN 的优点是原始 Transformer 采用该结构，在浅层模型（如 6 层、12 层）中表现良好，输出经过归一化后数值范围稳定，无需额外的最终 LayerNorm；缺点是深层训练不稳定，必须配合精细的学习率预热和参数初始化，当层数超过一定规模时训练容易发散，因此现代大模型大多改用 Pre-LN。

&emsp;&emsp;从维度视角看，Post-LN 在特征维上让残差路径每层都经过归一化，使得信息在传递过程中不断被重新标准化。这有助于数值稳定，但也压缩了残差路径上信息的动态范围。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 Post-LN 的裸实现：

```python
import torch
import torch.nn as nn

class PostLNBlock(nn.Module):
    def __init__(self, d_model, d_ff, eps=1e-6):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.ln1_gamma = nn.Parameter(torch.ones(d_model))
        self.ln1_beta = nn.Parameter(torch.zeros(d_model))
        self.ln2_gamma = nn.Parameter(torch.ones(d_model))
        self.ln2_beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def _ln(self, x, gamma, beta):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * gamma + beta

    def forward(self, x, attn_fn):
        # Post-LN: LayerNorm(x + F(x))
        attn_out = attn_fn(x)
        x = self._ln(x + attn_out, self.ln1_gamma, self.ln1_beta)

        ffn_out = torch.matmul(
            torch.relu(torch.matmul(x, self.W_1) + self.b_1),
            self.W_2
        ) + self.b_2
        x = self._ln(x + ffn_out, self.ln2_gamma, self.ln2_beta)
        return x
```

&emsp;&emsp;这个实现中，每个子层先计算 $F(x)$，与残差相加后再做 LayerNorm。若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.4.3 Pre-LN

&emsp;&emsp;Pre-LN 将 LayerNorm 放在子层输入之前，残差路径上不经过归一化。设子层函数为 $F$，输入为 $x$，Pre-LN 的输出为：

$$
y = x + F(\mathrm{LayerNorm}(x))
$$

&emsp;&emsp;Pre-LN 的残差路径是恒等映射，梯度可以通过残差路径直接回传，不受归一化操作的缩放影响。这使深层网络训练更稳定，对学习率预热不敏感，因此成为现代大模型（GPT-3、LLaMA、Qwen 等）的主流选择。Pre-LN 的缺点是残差路径上的激活值会随层数增加而累积增长，通常需要在最后一层后加一个最终的 LayerNorm 来稳定输出。

&emsp;&emsp;从梯度角度看，Pre-LN 的梯度为 $\partial y / \partial x = I + \partial F / \partial x$，恒等路径保证了梯度的直接传播。而 Post-LN 的梯度为 $\partial y / \partial x = \partial \mathrm{LN} / \partial (x + F) \cdot (I + \partial F / \partial x)$，归一化的雅可比矩阵会缩放梯度，导致深层训练不稳定。

&emsp;&emsp;Pre-LN 的优点是训练稳定，适合深层模型，对学习率预热不敏感，梯度传播良好；缺点是输出未归一化，残差路径上的激活值可能随深度增长，需要额外的最终 LayerNorm，且在相同层数下最终性能可能略低于调优良好的 Post-LN。

&emsp;&emsp;从维度视角看，Pre-LN 让残差路径保持干净，相当于在特征维上保留了一条无归一化的信息高速公路。归一化只作用于子层的输入，不干扰恒等路径。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 Pre-LN 的裸实现：

```python
import torch
import torch.nn as nn

class PreLNBlock(nn.Module):
    def __init__(self, d_model, d_ff, eps=1e-6):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.ln1_gamma = nn.Parameter(torch.ones(d_model))
        self.ln1_beta = nn.Parameter(torch.zeros(d_model))
        self.ln2_gamma = nn.Parameter(torch.ones(d_model))
        self.ln2_beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def _ln(self, x, gamma, beta):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * gamma + beta

    def forward(self, x, attn_fn):
        # Pre-LN: x + F(LayerNorm(x))
        norm_x = self._ln(x, self.ln1_gamma, self.ln1_beta)
        x = x + attn_fn(norm_x)

        norm_x = self._ln(x, self.ln2_gamma, self.ln2_beta)
        ffn_out = torch.matmul(
            torch.relu(torch.matmul(norm_x, self.W_1) + self.b_1),
            self.W_2
        ) + self.b_2
        x = x + ffn_out
        return x
```

&emsp;&emsp;这个实现中，LayerNorm 在子层输入前应用，残差路径保持恒等。若输入形状为 $(B,L,d_{model})$，输出形状相同。实际使用中通常在所有层之后再加一个最终 LayerNorm。

---

#### 2.4.4 RMSNorm

&emsp;&emsp;RMSNorm（Root Mean Square Layer Normalization）由 Zhang 和 Sennrich 于 2019 年提出，是 LayerNorm 的简化版本。它去掉了均值中心化，只保留方差归一化，即用均方根（RMS）替代标准差。设输入为 $x \in \mathbb{R}^{d}$，RMSNorm 的计算为：

$$
\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}
$$

$$
y_i = \frac{x_i}{\mathrm{RMS}(x)} \cdot \gamma_i
$$

&emsp;&emsp;与 LayerNorm 的区别有两点：RMSNorm 不减去均值，因此不需要计算均值；RMSNorm 只保留缩放参数 $\gamma$，去掉了偏移参数 $\beta$。这两个简化使 RMSNorm 的计算量比 LayerNorm 少约 10% 到 15%，在大模型中归一化层的开销不可忽略，因此这一节省相当可观。

&emsp;&emsp;RMSNorm 的理论依据是：LayerNorm 的有效性主要来自方差归一化（缩放不变性），而非均值中心化（平移不变性）。实验表明，去掉均值中心化后模型性能几乎不变，但训练速度更快。RMSNorm 已被 LLaMA、Qwen、DeepSeek、Mistral 等主流大模型广泛采用。

&emsp;&emsp;RMSNorm 的优点是计算更简单、更快，参数更少，且在大模型上性能与 LayerNorm 相当甚至更好；缺点是不做均值中心化，在某些对特征均值敏感的任务中可能损失部分表达能力，且去掉了偏移参数 $\beta$，灵活度略低于 LayerNorm。

&emsp;&emsp;从维度视角看，RMSNorm 在特征维上只对幅度做归一化，不改变特征的方向。它保留了特征向量在特征空间中的方向信息，只将其缩放到单位均方根尺度，再通过 $\gamma$ 做逐维缩放。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 RMSNorm 的裸实现：

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: (..., d_model)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma
```

&emsp;&emsp;这个实现中，只计算均方根，不做均值中心化，也不使用偏移参数。若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.4.5 GroupNorm

&emsp;&emsp;组归一化（Group Normalization, GroupNorm）由 Wu 和 He 于 2018 年提出，最初用于计算机视觉，后来被引入 Transformer 的变体中。GroupNorm 将特征维划分为 $G$ 个组，在每个组内独立计算均值和方差进行归一化。设输入为 $x \in \mathbb{R}^{d}$，将其划分为 $G$ 组，每组维度为 $d/G$，对第 $g$ 组：

$$
\mu_g = \frac{1}{d/G}\sum_{i \in \mathcal{G}_g} x_i, \quad \sigma_g^2 = \frac{1}{d/G}\sum_{i \in \mathcal{G}_g}(x_i - \mu_g)^2
$$

$$
y_i = \frac{x_i - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}} \cdot \gamma_i + \beta_i, \quad i \in \mathcal{G}_g
$$

&emsp;&emsp;GroupNorm 是 LayerNorm 和 InstanceNorm 的推广：当 $G=1$ 时退化为 LayerNorm，当 $G=d$ 时退化为 InstanceNorm。GroupNorm 的设计动机是在 batch size 很小时仍能稳定训练，因为它的统计量不依赖 batch 维度。

&emsp;&emsp;在 Transformer 中，GroupNorm 主要用于一些对归一化粒度有特殊需求的结构，如 QK-Norm 中的分组归一化、部分视觉 Transformer 的混合架构。标准大语言模型仍以 LayerNorm 和 RMSNorm 为主，GroupNorm 使用较少。

&emsp;&emsp;GroupNorm 的优点是统计量不依赖 batch size，在小 batch 下稳定，且分组归一化可以捕捉特征维内的局部统计结构，比 LayerNorm 更灵活；缺点是引入了分组数 $G$ 这一超参数，需要手动选择，且分组归一化破坏了特征维的全局统计一致性，在某些任务中可能不如 LayerNorm。

&emsp;&emsp;从维度视角看，GroupNorm 在特征维上做了分块归一化，每个子空间独立计算统计量。这相当于在特征维上引入了分组结构，使不同特征组可以有不同的归一化尺度。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 GroupNorm 的裸实现：

```python
import torch
import torch.nn as nn

class GroupNorm(nn.Module):
    def __init__(self, d_model, num_groups, eps=1e-6):
        super().__init__()
        assert d_model % num_groups == 0
        self.d_model = d_model
        self.num_groups = num_groups
        self.group_size = d_model // num_groups
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # x: (..., d_model)
        shape = x.shape
        # 重塑为 (..., num_groups, group_size)
        x_grouped = x.view(*shape[:-1], self.num_groups, self.group_size)

        mean = x_grouped.mean(dim=-1, keepdim=True)
        var = x_grouped.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x_grouped - mean) / torch.sqrt(var + self.eps)

        # 恢复原始形状
        x_norm = x_norm.view(*shape)
        return x_norm * self.gamma + self.beta
```

&emsp;&emsp;这个实现中，特征维被划分为 `num_groups` 组，每组独立计算均值和方差。若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.4.6 QK-Norm

&emsp;&emsp;QK-Norm（Query-Key Normalization）是对注意力中的 Query 和 Key 分别做归一化的技术，用于稳定注意力 logits 的数值范围，防止训练发散。标准注意力中，注意力分数为 $QK^\top / \sqrt{d_k}$，当 $Q$ 和 $K$ 的范数较大时，分数可能变得极大或极小，导致 Softmax 饱和、梯度消失。QK-Norm 在计算注意力分数之前，对 $Q$ 和 $K$ 分别做 LayerNorm 或 RMSNorm：

$$
\hat{Q} = \mathrm{Norm}(Q), \quad \hat{K} = \mathrm{Norm}(K)
$$

$$
\mathrm{scores} = \frac{\hat{Q}\hat{K}^\top}{\sqrt{d_k}}
$$

&emsp;&emsp;QK-Norm 最早在 Vision Transformer 的变体中被提出，后来被 ViT-22B、Stable Diffusion 3、Gemma 2 等模型采用。在 Gemma 2 中，QK-Norm 使用 RMSNorm 对每个注意力头的 Query 和 Key 做归一化，有效防止了注意力 logits 在训练中爆炸。QK-Norm 与 Pre-LN 配合使用时效果最好，因为 Pre-LN 的残差路径上激活值会累积增长，QK-Norm 可以在注意力计算前将这些激活值重新标准化。

&emsp;&emsp;QK-Norm 的优点是显著提升训练稳定性，允许使用更大的学习率，防止注意力 logits 爆炸，对深层模型尤其有效；缺点是增加了额外的归一化计算，略微增加推理延迟，且归一化后的 Query 和 Key 损失了部分幅度信息，可能影响注意力的锐度。

&emsp;&emsp;从维度视角看，QK-Norm 在注意力计算前对 Query 和 Key 的特征维做归一化，使关系矩阵的数值范围不依赖于输入的幅度。这相当于在 token 维扩散之前对查询和键做了一次特征维的尺度校准。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 QK-Norm 注意力的裸实现，使用 RMSNorm 对 Q 和 K 归一化：

```python
import math
import torch
import torch.nn as nn

class QKNormAttention(nn.Module):
    def __init__(self, d_model, num_heads, eps=1e-6):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.eps = eps

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        # QK-Norm: 对每个头维度做 RMSNorm
        self.q_gamma = nn.Parameter(torch.ones(self.d_k))
        self.k_gamma = nn.Parameter(torch.ones(self.d_k))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

    def _rms_norm(self, x, gamma):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * gamma

    def forward(self, x, causal=True):
        B, L, _ = x.size()

        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        # 对 Q 和 K 做 RMSNorm
        Q = self._rms_norm(Q, self.q_gamma)
        K = self._rms_norm(K, self.k_gamma)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if causal:
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)

        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return torch.matmul(head_out, self.W_o), attn
```

&emsp;&emsp;这个实现中，Query 和 Key 在计算注意力分数前分别经过 RMSNorm。若输入形状为 $(B,L,d_{model})$，输出形状相同，`attn` 形状为 $(B,h,L,L)$。

---

### 2.5 激活函数

#### 2.5.1 ReLU

&emsp;&emsp;ReLU（Rectified Linear Unit）是最经典的激活函数之一，定义为：

$$
\mathrm{ReLU}(x) = \max(0, x)
$$

&emsp;&emsp;它保留正区间的线性响应，将负区间置零。在 Transformer 的前馈网络中，ReLU 曾是原始架构的默认选择。ReLU 的优点是计算极其简单，仅需一次比较和置零，正区间梯度恒为 1，能有效缓解梯度消失，且负区间输出为零，带来稀疏激活；缺点是负区间梯度为零，若某神经元的输入长期为负，其权重将无法更新，导致“神经元死亡”，且输出不是零中心的，可能影响优化稳定性。

&emsp;&emsp;从维度视角看，ReLU 在特征维上对每个元素独立施加非线性，将负值截断为零。它不改变 token 维扩散的结构，只在前馈网络的中间层引入稀疏性和非线性变换。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 ReLU 的裸实现：

```python
import torch
import torch.nn as nn

class ReLU(nn.Module):
    def forward(self, x):
        return torch.maximum(x, torch.zeros_like(x))
```

&emsp;&emsp;若输入形状为 $(B,L,d_{ff})$，输出形状相同。

---

#### 2.5.2 GELU

&emsp;&emsp;GELU（Gaussian Error Linear Unit）由 Hendrycks 和 Gimpel 于 2016 年提出，定义为：

$$
\mathrm{GELU}(x) = x \cdot \Phi(x)
$$

&emsp;&emsp;其中 $\Phi(x)$ 是标准正态分布的累积分布函数。实际计算中常用 tanh 近似：

$$
\mathrm{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)
$$

&emsp;&emsp;GELU 可以理解为对输入进行随机正则化的期望：以输入值的大小决定保留概率。它平滑、非单调，在负区间有微小负值，在 Transformer（如 BERT、GPT）中广泛使用。GELU 的优点是平滑可导，负区间梯度不为零，缓解了神经元死亡，在自然语言处理任务中通常优于 ReLU；缺点是计算涉及 tanh 或 erf，比 ReLU 慢，且没有 ReLU 那样的严格稀疏性。

&emsp;&emsp;从维度视角看，GELU 在特征维上提供了一种平滑的门控：正区近似线性，负区平滑衰减到零。它比 ReLU 更平滑，使前馈网络的非线性变换更稳定。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 GELU 的 tanh 近似裸实现：

```python
import math
import torch
import torch.nn as nn

class GELU(nn.Module):
    def forward(self, x):
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x ** 3)))
```

&emsp;&emsp;若输入形状为 $(B,L,d_{ff})$，输出形状相同。

---

#### 2.5.3 SwiGLU

&emsp;&emsp;SwiGLU 是 GLU（Gated Linear Unit）的变体，使用 Swish 作为门控激活。Swish 定义为：

$$
\mathrm{Swish}(x) = x \cdot \sigma(x)
$$

&emsp;&emsp;其中 $\sigma$ 是 Sigmoid。SwiGLU 的前馈计算为：

$$
\mathrm{SwiGLU}(x) = \mathrm{Swish}(xW_1) \odot (xW_2)
$$

&emsp;&emsp;然后经过输出投影 $W_3$。与标准 FFN 的两层结构不同，SwiGLU 有三个权重矩阵：$W_1$ 和 $W_2$ 将输入投影到中间维度，逐元素相乘后由 $W_3$ 投影回原维度。LLaMA、Qwen、DeepSeek 等主流大模型均采用 SwiGLU。SwiGLU 的优点是门控机制使网络能动态调节信息流，性能通常优于 ReLU 和 GELU，且平滑可导；缺点是参数量比标准 FFN 多约 50%（三个矩阵而非两个），计算量也相应增加。

&emsp;&emsp;从维度视角看，SwiGLU 在特征维上引入了一个乘性门控：一个分支用 Swish 产生门控信号，另一个分支提供内容，两者逐元素相乘。这相当于在特征维上做了一次数据依赖的软选择。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 SwiGLU 的裸实现：

```python
import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W_3 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.b_2 = nn.Parameter(torch.zeros(d_ff))
        self.b_3 = nn.Parameter(torch.zeros(d_model))
        for w in (self.W_1, self.W_2, self.W_3):
            nn.init.xavier_uniform_(w)

    def forward(self, x):
        # Swish(xW1) = xW1 * sigmoid(xW1)
        gate = torch.matmul(x, self.W_1) + self.b_1
        gate = gate * torch.sigmoid(gate)
        content = torch.matmul(x, self.W_2) + self.b_2
        hidden = gate * content
        return torch.matmul(hidden, self.W_3) + self.b_3
```

&emsp;&emsp;若输入形状为 $(B,L,d_{model})$，输出形状为 $(B,L,d_{model})$。

---

#### 2.5.4 GeGLU

&emsp;&emsp;GeGLU 与 SwiGLU 结构相同，但使用 GELU 作为门控激活：

$$
\mathrm{GeGLU}(x) = \mathrm{GELU}(xW_1) \odot (xW_2)
$$

&emsp;&emsp;然后经过 $W_3$ 投影。GeGLU 在 Gemma、PaLM 等模型中使用。与 SwiGLU 相比，GeGLU 的门控信号更平滑，负区间有微小负值，可能在某些任务中提供更细腻的门控。GeGLU 的优点是性能与 SwiGLU 相当，平滑门控有利于优化，在部分模型上表现略优；缺点是 GELU 计算比 Swish 稍复杂，整体参数量和计算量与 SwiGLU 相同。

&emsp;&emsp;从维度视角看，GeGLU 与 SwiGLU 一样，在特征维上做乘性门控，区别仅在于门控函数从 Swish 换成了 GELU。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 GeGLU 的裸实现：

```python
import math
import torch
import torch.nn as nn

class GeGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W_3 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.b_2 = nn.Parameter(torch.zeros(d_ff))
        self.b_3 = nn.Parameter(torch.zeros(d_model))
        for w in (self.W_1, self.W_2, self.W_3):
            nn.init.xavier_uniform_(w)

    def _gelu(self, x):
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x ** 3)))

    def forward(self, x):
        gate = self._gelu(torch.matmul(x, self.W_1) + self.b_1)
        content = torch.matmul(x, self.W_2) + self.b_2
        hidden = gate * content
        return torch.matmul(hidden, self.W_3) + self.b_3
```

&emsp;&emsp;若输入形状为 $(B,L,d_{model})$，输出形状为 $(B,L,d_{model})$。

---

#### 2.5.5 SiTU-GLU

&emsp;&emsp;SiTU（Sigmoid Tanh Unit）是一种平滑激活函数，定义为：

$$
\mathrm{SiTU}(x) = x \cdot \tanh(\mathrm{softplus}(x))
$$

&emsp;&emsp;其中 $\mathrm{softplus}(x) = \ln(1+e^x)$。SiTU 在负区间平滑衰减到零，在正区间近似线性，兼具 ReLU 的稀疏性和 GELU 的平滑性。SiTU-GLU 将 SiTU 作为门控激活：

$$
\mathrm{SiTU\text{-}GLU}(x) = \mathrm{SiTU}(xW_1) \odot (xW_2)
$$

&emsp;&emsp;然后经过 $W_3$ 投影。SiTU-GLU 的优点是门控函数平滑且非单调，负区间梯度稳定，训练稳定性好，在部分大模型中表现出与 SwiGLU 相当或更优的性能；缺点是 tanh 和 softplus 的复合计算比 Swish 和 GELU 稍复杂，推理延迟略高。

&emsp;&emsp;从维度视角看，SiTU-GLU 在特征维上提供了更平滑的门控曲线，使门控信号在负区平滑趋近于零，正区平滑趋近于恒等，介于 ReLU 的硬门控和 Swish 的软门控之间。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 SiTU-GLU 的裸实现：

```python
import torch
import torch.nn as nn

class SiTU_GLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W_3 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.b_2 = nn.Parameter(torch.zeros(d_ff))
        self.b_3 = nn.Parameter(torch.zeros(d_model))
        for w in (self.W_1, self.W_2, self.W_3):
            nn.init.xavier_uniform_(w)

    def _softplus(self, x):
        # 数值稳定的 softplus: max(x,0) + log1p(exp(-|x|))
        return torch.maximum(x, torch.zeros_like(x)) + torch.log1p(torch.exp(-torch.abs(x)))

    def _situ(self, x):
        return x * torch.tanh(self._softplus(x))

    def forward(self, x):
        gate = self._situ(torch.matmul(x, self.W_1) + self.b_1)
        content = torch.matmul(x, self.W_2) + self.b_2
        hidden = gate * content
        return torch.matmul(hidden, self.W_3) + self.b_3
```

&emsp;&emsp;若输入形状为 $(B,L,d_{model})$，输出形状为 $(B,L,d_{model})$。

---

#### 2.5.6 Sqrt(Softplus(·))

&emsp;&emsp;Sqrt(Softplus) 是一种非负、平滑的激活函数，定义为：

$$
\mathrm{SqrtSoftplus}(x) = \sqrt{\mathrm{softplus}(x)} = \sqrt{\ln(1+e^x)}
$$

&emsp;&emsp;它的输出恒为正，且随 $x$ 增大而缓慢增长（近似 $\sqrt{x}$），随 $x$ 减小而平滑趋近于零。Sqrt(Softplus) 在部分 MoE 路由器和注意力门控中被用作替代 Sigmoid 或 Softmax 的平滑非负函数。Sqrt(Softplus) 的优点是输出非负，适合作为门控或权重生成函数，平滑可导，梯度稳定，且平方根使输出增长比 Softplus 更平缓，避免数值过大；缺点是输出不是零中心，且计算涉及 exp 和 log，比 Sigmoid 稍慢。

&emsp;&emsp;从维度视角看，Sqrt(Softplus) 在特征维上将任意实数映射到正区间，可作为乘性门控的权重生成器。它不改变 token 维扩散的结构，而是在前馈或路由路径中提供平滑的非负门控信号。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 Sqrt(Softplus) 的裸实现：

```python
import torch
import torch.nn as nn

class SqrtSoftplus(nn.Module):
    def forward(self, x):
        # 数值稳定的 softplus
        softplus = torch.maximum(x, torch.zeros_like(x)) + torch.log1p(torch.exp(-torch.abs(x)))
        return torch.sqrt(softplus)
```

&emsp;&emsp;若输入形状为 $(B,L,d)$，输出形状相同，且所有输出非负。


---

### 2.6 前馈网络与混合专家

#### 2.6.1 前馈网络（FFN）

&emsp;&emsp;前馈网络（Feed-Forward Network, FFN）是 Transformer 中每个子层之后的标准组件，对每个位置独立进行特征维变换。原始 Transformer 的 FFN 由两层线性变换和一个激活函数组成：

$$
\mathrm{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2
$$

&emsp;&emsp;其中 $W_1 \in \mathbb{R}^{d_{ff} \times d_{model}}$，$W_2 \in \mathbb{R}^{d_{model} \times d_{ff}}$，中间维度 $d_{ff}$ 通常取 $4 d_{model}$。$\sigma$ 可以是 ReLU、GELU 或 SwiGLU 等激活函数。FFN 对序列中每个 token 独立作用，不混合 token 维信息，因此与注意力层互补：注意力负责 token 间信息交换，FFN 负责特征维的非线性变换和容量扩展。

&emsp;&emsp;FFN 的优点是结构简单、并行度高，通过升维-非线性-降维的方式显著增加模型表达能力，且逐位置独立计算，不依赖序列长度；缺点是参数量和计算量较大，通常占 Transformer 总参数的三分之二左右，中间维度 $d_{ff}$ 的选择需要权衡容量和效率。

&emsp;&emsp;从维度视角看，FFN 在特征维上先升维到高维空间，施加非线性后再降回原维度，相当于在特征维上做了一次非线性特征变换。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 FFN 的裸实现：

```python
import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def forward(self, x):
        hidden = torch.relu(torch.matmul(x, self.W_1) + self.b_1)
        return torch.matmul(hidden, self.W_2) + self.b_2
```

&emsp;&emsp;若输入形状为 $(B,L,d_{model})$，输出形状为 $(B,L,d_{model})$。

---

#### 2.6.2 混合专家（MoE）

&emsp;&emsp;混合专家（Mixture of Experts, MoE）将 Transformer 中的 FFN 替换为多个并行的专家网络，每个专家本质上是一个小型 FFN。对于每个输入 token，路由器计算其与各专家的匹配分数，并据此对专家输出进行加权求和：

$$
\mathrm{MoE}(x) = \sum_{i=1}^{N} g_i(x) \cdot \mathrm{FFN}_i(x)
$$

&emsp;&emsp;其中 $N$ 是专家总数，$g_i(x)$ 是路由器分配给第 $i$ 个专家的权重，通常满足 $\sum_i g_i = 1$。与稠密 FFN 相比，MoE 的参数量可以随专家数线性增长，但每个 token 只激活部分专家，因此计算量远小于同等参数量的稠密模型。

&emsp;&emsp;MoE 的优点是参数量与计算量解耦，可以用较低的计算成本获得极大的模型容量，适合大规模预训练；缺点是路由机制可能导致负载不均衡，部分专家过载而其他专家训练不足，且专家分布在多 GPU 上时 token 分发和聚合会带来通信开销。

&emsp;&emsp;从维度视角看，MoE 在特征维上实现了条件计算：每个 token 根据自身特征被路由到不同的专家子网络，相当于在特征维上按内容动态选择变换路径。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 MoE 的裸实现，包含路由器和多个专家：

```python
import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def forward(self, x):
        return torch.matmul(
            torch.relu(torch.matmul(x, self.W_1) + self.b_1),
            self.W_2
        ) + self.b_2


class MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.router = nn.Parameter(torch.empty(num_experts, d_model))
        nn.init.normal_(self.router, std=0.02)

    def forward(self, x):
        B, L, D = x.size()
        x_flat = x.view(B * L, D)

        # 路由器打分并归一化
        scores = torch.matmul(x_flat, self.router.T)  # (B*L, num_experts)
        gates = torch.softmax(scores, dim=-1)

        # 加权求和所有专家
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            out += gates[:, i:i+1] * expert(x_flat)

        return out.view(B, L, D)
```

&emsp;&emsp;若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.6.3 稀疏混合专家（SMoE）

&emsp;&emsp;稀疏混合专家（Sparse Mixture of Experts, SMoE）是 MoE 的稀疏激活版本。在标准 MoE 中，所有专家都会对每个 token 计算并加权，计算量仍然很大。SMoE 只保留路由器分数最高的 $k$ 个专家，其余专家的输出直接置零：

$$
g_i(x) = \begin{cases}
s_i(x) & s_i(x) \in \mathrm{TopK}(\{s_j(x)\}, k) \\
0 & \text{otherwise}
\end{cases}
$$

&emsp;&emsp;其中 $s_i(x)$ 是路由器对第 $i$ 个专家的原始分数。通常 $k$ 取 1 或 2。SMoE 使每个 token 只经过极少数专家，计算量从 $O(N)$ 降到 $O(k)$，同时参数量仍为 $O(N)$，因此可以用极低的计算成本获得巨大的模型容量。Switch Transformer、GShard、DeepSeekMoE 等都采用 SMoE。

&emsp;&emsp;SMoE 的优点是计算效率极高，参数量与计算量解耦，适合万亿级参数模型；缺点是 Top-k 选择不可导，通常需要用直通估计器或强化学习来训练路由器，且负载不均衡问题更严重，需要辅助损失或偏置调整来维持专家利用率。

&emsp;&emsp;从维度视角看，SMoE 在特征维上做了稀疏门控：每个 token 只激活少数专家，其余专家的计算完全跳过。这相当于在特征维上实现了高度条件化的稀疏变换。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 Top-1 SMoE 的裸实现：

```python
import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def forward(self, x):
        return torch.matmul(
            torch.relu(torch.matmul(x, self.W_1) + self.b_1),
            self.W_2
        ) + self.b_2


class SparseMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.router = nn.Parameter(torch.empty(num_experts, d_model))
        nn.init.normal_(self.router, std=0.02)

    def forward(self, x):
        B, L, D = x.size()
        x_flat = x.view(B * L, D)

        scores = torch.matmul(x_flat, self.router.T)  # (B*L, num_experts)
        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)
        topk_gates = torch.softmax(topk_scores, dim=-1)

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            idx = topk_indices[:, k]
            gate = topk_gates[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    out[mask] += gate[mask] * self.experts[e](x_flat[mask])

        return out.view(B, L, D)
```

&emsp;&emsp;若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.6.4 专家

&emsp;&emsp;专家（Expert）是 MoE 架构中的基本计算单元，本质上是一个小型前馈网络。每个专家通常由两层线性变换和一个激活函数组成：

$$
\mathrm{Expert}_i(x) = W_2^{(i)} \cdot \sigma(W_1^{(i)} x + b_1^{(i)}) + b_2^{(i)}
$$

&emsp;&emsp;与标准 FFN 的区别在于，专家的中间维度通常更小，因为 MoE 层会包含大量专家，总参数量需要控制。例如 DeepSeekMoE 将每个专家拆分为更细粒度的专家，中间维度仅为标准 FFN 的几分之一。专家之间不共享参数，各自学习不同的特征变换模式。

&emsp;&emsp;专家的优点是专业化：不同专家可以在训练中分化出不同的功能，有的处理语法，有的处理语义，有的处理特定领域知识；缺点是单个专家的训练数据较少，如果路由不当，部分专家可能得不到充分训练，导致专家冗余或坍缩。

&emsp;&emsp;从维度视角看，每个专家在特征维上执行一次独立的升维-非线性-降维变换。多个专家并行存在，路由器根据 token 内容选择使用哪些专家的变换结果。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出单个专家的裸实现：

```python
import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def forward(self, x):
        hidden = torch.relu(torch.matmul(x, self.W_1) + self.b_1)
        return torch.matmul(hidden, self.W_2) + self.b_2
```

&emsp;&emsp;若输入形状为 $(B,L,d_{model})$，输出形状为 $(B,L,d_{model})$。

---

#### 2.6.5 路由网络

&emsp;&emsp;路由网络（Router Network）是 MoE 中决定每个 token 分配给哪些专家的模块。它通常是一个简单的线性层，将 token 的隐藏状态映射到专家数量维度的分数向量，再经过 Softmax 得到归一化的门控权重：

$$
s(x) = W_r x, \quad g(x) = \mathrm{softmax}(s(x))
$$

&emsp;&emsp;其中 $W_r \in \mathbb{R}^{N \times d_{model}}$ 是路由器的可学习参数，$N$ 是专家总数。路由器与专家一起端到端训练，通过反向传播学习如何根据 token 内容分配专家。为了避免负载不均衡，通常会引入辅助损失或偏置调整机制。

&emsp;&emsp;路由网络的优点是结构简单、计算量小，仅为一个线性层加 Softmax，可以轻松集成到 Transformer 中；缺点是路由决策是离散的，Top-k 选择不可导，需要用直通估计器或 Gumbel-Softmax 等技巧，且路由器容易陷入局部最优，导致部分专家被过度使用。

&emsp;&emsp;从维度视角看，路由网络在特征维上计算 token 与专家的匹配分数，相当于在特征维上做了一次相似度匹配，然后根据匹配结果选择专家子网络。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出路由网络的裸实现：

```python
import torch
import torch.nn as nn

class Router(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.W_r = nn.Parameter(torch.empty(num_experts, d_model))
        nn.init.normal_(self.W_r, std=0.02)

    def forward(self, x):
        # x: (B, L, d_model)
        scores = torch.matmul(x, self.W_r.T)  # (B, L, num_experts)
        gates = torch.softmax(scores, dim=-1)
        return gates, scores
```

&emsp;&emsp;若输入形状为 $(B,L,d_{model})$，`gates` 形状为 $(B,L,N)$，`scores` 形状相同。

---

#### 2.6.6 Top-k 路由

&emsp;&emsp;Top-k 路由是稀疏 MoE 中最常用的路由策略。路由器先计算所有专家的分数，然后只保留分数最高的 $k$ 个专家，其余专家的门控权重置零：

$$
\mathrm{TopK}(s, k)_i = \begin{cases}
s_i & s_i \in \mathrm{TopK}(\{s_j\}, k) \\
-\infty & \text{otherwise}
\end{cases}
$$

$$
g(x) = \mathrm{softmax}(\mathrm{TopK}(s(x), k))
$$

&emsp;&emsp;通常 $k$ 取 1 或 2。Top-1 路由计算最省，但负载不均衡风险最高；Top-2 路由允许 token 同时使用两个专家，通常能提升模型质量，但计算量翻倍。Switch Transformer 使用 Top-1，GShard 和 DeepSeekMoE 使用 Top-2 或更细粒度的 Top-k。

&emsp;&emsp;Top-k 路由的优点是计算量固定为 $O(k)$，与专家总数无关，适合大规模稀疏模型；缺点是 Top-k 操作不可导，路由器无法直接通过梯度学习，通常需要对选中的门控值做 Softmax 后回传梯度，且负载不均衡问题需要通过辅助损失或专家偏置来缓解。

&emsp;&emsp;从维度视角看，Top-k 路由在特征维上执行了稀疏选择：每个 token 只激活少数几个专家，其余专家的变换被完全跳过，实现了条件计算。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 Top-k 路由的裸实现：

```python
import torch
import torch.nn as nn

class TopKRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.W_r = nn.Parameter(torch.empty(num_experts, d_model))
        nn.init.normal_(self.W_r, std=0.02)

    def forward(self, x):
        # x: (B, L, d_model)
        scores = torch.matmul(x, self.W_r.T)  # (B, L, num_experts)
        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)  # (B, L, k)
        topk_gates = torch.softmax(topk_scores, dim=-1)  # (B, L, k)
        return topk_gates, topk_indices
```

&emsp;&emsp;若输入形状为 $(B,L,d_{model})$，`topk_gates` 形状为 $(B,L,k)$，`topk_indices` 形状相同。

---

#### 2.6.7 共享专家

&emsp;&emsp;共享专家（Shared Expert）是 DeepSeekMoE 引入的一种专家设计，指所有 token 都固定分配到的专家，不经过路由器选择。共享专家与路由专家并行存在，其输出直接加到最终结果中：

$$
h = \sum_{i=1}^{K_s} \mathrm{Expert}_i(x) + \sum_{j=K_s+1}^{N} g_j(x) \cdot \mathrm{Expert}_j(x)
$$

&emsp;&emsp;其中 $K_s$ 是共享专家数量，$N$ 是总专家数量。共享专家负责捕获通用知识，减少路由专家之间的参数冗余，使路由专家可以更专注于特定领域或模式。DeepSeekMoE 的实验表明，设置 1 到 2 个共享专家即可显著提升专家专业化程度。

&emsp;&emsp;共享专家的优点是提供了稳定的通用变换路径，缓解了路由不均衡问题，且减少了路由专家的冗余；缺点是增加了固定的计算量，因为所有 token 都必须经过共享专家，且共享专家和路由专家的比例需要手动调整。

&emsp;&emsp;从维度视角看，共享专家在特征维上提供了一条所有 token 共享的通用变换路径，而路由专家提供条件化的专用变换路径。两者叠加，使模型既有稳定的基础表示，又有灵活的条件计算。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出带共享专家的 MoE 层裸实现：

```python
import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def forward(self, x):
        return torch.matmul(
            torch.relu(torch.matmul(x, self.W_1) + self.b_1),
            self.W_2
        ) + self.b_2


class MoEWithSharedExperts(nn.Module):
    def __init__(self, d_model, d_ff, num_shared, num_routed, top_k=2):
        super().__init__()
        self.num_shared = num_shared
        self.num_routed = num_routed
        self.top_k = top_k

        self.shared_experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(num_shared)]
        )
        self.routed_experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(num_routed)]
        )
        self.router = nn.Parameter(torch.empty(num_routed, d_model))
        nn.init.normal_(self.router, std=0.02)

    def forward(self, x):
        B, L, D = x.size()
        x_flat = x.view(B * L, D)

        # 共享专家：所有 token 固定经过
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)

        # 路由专家：Top-k 选择
        scores = torch.matmul(x_flat, self.router.T)
        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)
        topk_gates = torch.softmax(topk_scores, dim=-1)

        routed_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            idx = topk_indices[:, k]
            gate = topk_gates[:, k].unsqueeze(-1)
            for e in range(self.num_routed):
                mask = (idx == e)
                if mask.any():
                    routed_out[mask] += gate[mask] * self.routed_experts[e](x_flat[mask])

        return (shared_out + routed_out).view(B, L, D)
```

&emsp;&emsp;若输入形状为 $(B,L,d_{model})$，输出形状相同。

#### 2.6.8 细粒度专家

&emsp;&emsp;细粒度专家（Fine-Grained Expert Segmentation）是 DeepSeekMoE 提出的专家设计策略。传统 MoE 中，每个专家的中间维度与标准 FFN 相同，专家数量有限，路由组合的灵活性受到制约。细粒度专家将每个专家进一步拆分为多个更小的专家，使总专家数从 $N$ 增加到 $mN$，同时从其中激活 $mK$ 个专家，保持总参数量和计算量不变：

$$
g_{i,t} = \begin{cases}
s_{i,t}, & s_{i,t} \in \mathrm{TopK}(\{s_{j,t} \mid 1 \leqslant j \leqslant mN\}, mK) \\
0, & \text{otherwise}
\end{cases}
$$

&emsp;&emsp;其中 $m$ 是细分因子，通常取 2 到 4。每个细粒度专家的中间维度缩小为原来的 $1/m$。这种设计从组合角度看极大地增强了激活专家的组合灵活性。以一个典型的 Top-2 路由为例，若 $N=16$，组合数为 $\binom{16}{2}=120$；若每个专家拆分为 4 个，则 $mN=64$，激活 $mK=8$ 个专家，组合数变为 $\binom{64}{8}=4{,}426{,}165{,}368$，增长了七个数量级。

&emsp;&emsp;细粒度专家的优点是组合灵活性的大幅提升使模型能够更精确地匹配 token 与专家的关系，实现更精细的知识获取和专家专业化；缺点是专家数量增加后路由器的选择空间更大，路由决策的计算量和负载均衡难度也随之上升，且每个专家的训练数据更少，可能加剧专家欠训练的风险。

&emsp;&emsp;从维度视角看，细粒度专家在特征维上将每个专家的变换空间进一步细分，使特征维上的条件计算粒度更细。路由器可以在更大的专家集合中选择更匹配的子集，相当于在特征维上实现了更精细的稀疏激活。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出细粒度专家的裸实现，将标准专家拆分为多个小专家并激活 Top-k：

```python
import torch
import torch.nn as nn

class FineGrainedExpert(nn.Module):
    """细粒度专家: 中间维度为标准 FFN 的 1/m"""
    def __init__(self, d_model, d_ff_fine):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff_fine))
        self.b_1 = nn.Parameter(torch.zeros(d_ff_fine))
        self.W_2 = nn.Parameter(torch.empty(d_ff_fine, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def forward(self, x):
        return torch.matmul(
            torch.relu(torch.matmul(x, self.W_1) + self.b_1),
            self.W_2
        ) + self.b_2


class FineGrainedMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts_base, seg_factor, top_k):
        super().__init__()
        self.num_experts = num_experts_base * seg_factor
        self.top_k = top_k
        self.d_ff_fine = d_ff // seg_factor

        self.experts = nn.ModuleList([
            FineGrainedExpert(d_model, self.d_ff_fine)
            for _ in range(self.num_experts)
        ])
        self.router = nn.Parameter(torch.empty(self.num_experts, d_model))
        nn.init.normal_(self.router, std=0.02)

    def forward(self, x):
        B, L, D = x.size()
        x_flat = x.view(B * L, D)

        scores = torch.matmul(x_flat, self.router.T)
        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)
        topk_gates = torch.softmax(topk_scores, dim=-1)

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            idx = topk_indices[:, k]
            gate = topk_gates[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    out[mask] += gate[mask] * self.experts[e](x_flat[mask])

        return out.view(B, L, D)
```

&emsp;&emsp;若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.6.9 负载均衡损失

&emsp;&emsp;负载均衡损失（Load Balancing Loss）是 MoE 训练中用于防止专家负载不均衡的辅助损失函数。稀疏 MoE 中，Top-k 路由容易导致部分专家被过度使用而其他专家几乎不被激活，严重时出现“路由坍缩”。负载均衡损失通过惩罚不均衡的分配来引导路由器将 token 均匀地分配到各专家。

&emsp;&emsp;设一个 batch 中有 $T$ 个 token，$N$ 个专家。定义 $D_i$ 为分配给专家 $i$ 的 token 比例，$P_i$ 为专家 $i$ 的平均门控概率：

$$
D_i = \frac{1}{T}\sum_{x \in \mathcal{B}} \mathbf{1}\{\mathrm{argmax}\,G_\sigma(x) = i\}
$$

$$
P_i = \frac{1}{T}\sum_{x \in \mathcal{B}} G_\sigma(x)_i
$$

&emsp;&emsp;其中 $\mathcal{B}$ 是当前 batch，$G_\sigma(x)$ 是路由器的门控输出。负载均衡损失定义为：

$$
\mathcal{L}_{\mathrm{load\text{-}balancing}} = N \sum_{i=1}^{N} D_i P_i
$$

&emsp;&emsp;当所有专家被均匀分配时，$D_i = P_i = 1/N$，损失取得最小值 1。总训练损失为：

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{moe}} + \alpha \mathcal{L}_{\mathrm{load\text{-}balancing}}
$$

&emsp;&emsp;其中 $\alpha$ 是辅助损失的权重系数，通常取 0.01 左右。负载均衡损失的优点是实现简单，只需在训练损失上增加一个可微的辅助项，能有效防止路由坍缩，使所有专家都得到充分训练；缺点是辅助损失会引入干扰梯度，与主任务损失产生竞争，可能轻微损害模型性能，且 $\alpha$ 需要手动调优，过大会主导训练，过小则无法有效均衡。

&emsp;&emsp;从维度视角看，负载均衡损失在特征维上约束了路由器的分配行为，使 token 在专家子空间中的分布更加均匀。它不改变注意力或 FFN 的结构，而是通过优化目标引导路由器学习更均衡的分配策略。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出负载均衡损失的裸实现：

```python
import torch
import torch.nn as nn

def load_balancing_loss(router_scores, topk_indices, num_experts):
    """
    router_scores: (B*L, num_experts) 路由器原始分数
    topk_indices: (B*L, top_k) Top-k 选择的专家索引
    """
    T = router_scores.size(0)

    # Di: 分配给专家 i 的 token 比例
    D = torch.zeros(num_experts, device=router_scores.device)
    for k in range(topk_indices.size(1)):
        for e in range(num_experts):
            D[e] += (topk_indices[:, k] == e).float().sum()
    D = D / T

    # Pi: 专家 i 的平均门控概率
    gates = torch.softmax(router_scores, dim=-1)
    P = gates.mean(dim=0)  # (num_experts,)

    # 负载均衡损失
    loss = num_experts * (D * P).sum()
    return loss
```

&emsp;&emsp;这个实现中，$D_i$ 通过统计 Top-k 索引中专家 $i$ 出现的次数得到，$P_i$ 通过路由器概率的均值得到。损失值在均匀分配时接近 1。

---

#### 2.6.10 无辅助损失负载均衡

&emsp;&emsp;无辅助损失负载均衡（Auxiliary-Loss-Free Load Balancing）是 DeepSeek-V2/V3 采用的一种负载均衡策略，由 Wang 等人于 2024 年提出。传统辅助损失方法通过在主损失上增加均衡惩罚项来引导路由，但辅助损失会产生干扰梯度，与主任务优化目标竞争，可能损害模型性能。无辅助损失方法完全去掉了辅助损失，改为在 Top-K 路由前对每个专家的路由分数施加一个专家级偏置 $b_i$，并根据近期负载动态更新偏置。

&emsp;&emsp;设路由器对专家 $i$ 的原始分数为 $s_{i,t}$，加上偏置后的分数为：

$$
s'_{i,t} = s_{i,t} + b_i
$$

&emsp;&emsp;Top-K 选择基于 $s'_{i,t}$ 进行。偏置 $b_i$ 不参与梯度计算，而是根据每个专家近期的实际负载进行增量更新：如果专家 $i$ 负载过高，减小 $b_i$；如果负载过低，增大 $b_i$：

$$
b_i \leftarrow b_i - \gamma \cdot \mathrm{sign}(\mathrm{load}_i - \bar{\mathrm{load}})
$$

&emsp;&emsp;其中 $\gamma$ 是偏置更新速率，$\mathrm{load}_i$ 是专家 $i$ 近期的 token 分配量，$\bar{\mathrm{load}}$ 是平均负载。这种更新形成了一种历史反馈机制，使偏置自动收敛到使负载均衡的值。

&emsp;&emsp;无辅助损失负载均衡的优点是消除了辅助损失带来的干扰梯度，不损害主任务性能，偏置更新不通过反向传播因此无需额外的梯度计算，且与专家并行训练兼容；缺点是偏置更新速率 $\gamma$ 需要调优，过大会导致路由震荡，过小则均衡速度慢，且偏置的初始值需要合理设置，否则早期训练可能出现极端不均衡。

&emsp;&emsp;从维度视角看，无辅助损失均衡通过修改路由分数的偏置来引导分配，而不是修改损失函数。这相当于在特征维的专家选择阶段引入了一个自适应的校准项，使路由器的决策更倾向于均衡。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出无辅助损失负载均衡的裸实现：

```python
import torch
import torch.nn as nn

class LossFreeBalancedMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k, bias_rate=0.001):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_rate = bias_rate

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        self.router = nn.Parameter(torch.empty(num_experts, d_model))
        nn.init.normal_(self.router, std=0.02)

        # 专家偏置: 不参与梯度计算
        self.register_buffer("routing_bias", torch.zeros(num_experts))

    def forward(self, x):
        B, L, D = x.size()
        x_flat = x.view(B * L, D)

        # 原始路由分数
        scores = torch.matmul(x_flat, self.router.T)

        # 加上偏置后的分数用于 Top-K 选择
        biased_scores = scores + self.routing_bias.unsqueeze(0)

        # Top-K 选择基于 biased_scores
        topk_biased, topk_indices = biased_scores.topk(self.top_k, dim=-1)

        # 门控权重使用原始分数计算
        topk_original = scores.gather(1, topk_indices)
        topk_gates = torch.softmax(topk_original, dim=-1)

        # 计算专家
        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            idx = topk_indices[:, k]
            gate = topk_gates[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    out[mask] += gate[mask] * self.experts[e](x_flat[mask])

        # 更新偏置（不通过梯度）
        with torch.no_grad():
            for e in range(self.num_experts):
                load_e = (topk_indices == e).float().sum()
                avg_load = topk_indices.numel() / self.num_experts
                if load_e > avg_load:
                    self.routing_bias[e] -= self.bias_rate
                elif load_e < avg_load:
                    self.routing_bias[e] += self.bias_rate

        return out.view(B, L, D)
```

&emsp;&emsp;这个实现中，Top-K 选择基于加了偏置的分数，但门控权重使用原始分数计算。偏置在每次前向传播后根据负载自动更新。若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.6.11 Hash routing

&emsp;&emsp;Hash routing（哈希路由）是一种无参数的路由策略，使用预定义的哈希函数将 token 确定性地分配到专家，而非通过可学习的路由器。最简单的形式是基于 token ID 的哈希表：每个 token ID 通过哈希函数映射到固定的专家索引。在 DeepSeek-V4 中，前几个 MoE 层采用了冻结的 token-id 到 expert-id 查找表（`tid2eid`），专家选择完全由查表决定，不使用可学习的 argmax。

&emsp;&emsp;Hash routing 的核心形式为：

$$
\mathrm{expert}(t) = \mathrm{Hash}(t) \bmod N
$$

&emsp;&emsp;其中 $t$ 是 token 的标识（如 token ID 或位置索引），$N$ 是专家总数。哈希函数可以是简单的取模运算、乘法哈希或更复杂的混合哈希。由于哈希函数是固定的，每个 token 的专家分配在训练前就已确定，不随训练过程变化。

&emsp;&emsp;Hash routing 的优点是路由完全无参数，消除了路由器的训练开销和负载均衡问题，且分配是确定性的，便于推理时的预计算和缓存；缺点是哈希分配与 token 内容无关，无法根据语义特征选择最合适的专家，路由质量受哈希函数设计影响很大，通常性能不如可学习路由。实验表明，哈希路由在 GPT-OSS-20B 上的平均得分约为 36.1 分，显著低于基于特征的路由方法。在 DeepSeek-V4 中，Hash routing 仅用于前几个 MoE 层，后续层仍使用可学习的 Top-K 路由。

&emsp;&emsp;从维度视角看，Hash routing 完全绕过了特征维上的内容匹配，将 token 维的分配问题转化为一个固定的映射。它不利用 token 的特征信息来决定专家选择，因此无法实现内容感知的条件计算。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 Hash routing 的裸实现：

```python
import torch
import torch.nn as nn

class HashRoutingMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, vocab_size):
        super().__init__()
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

        # 冻结的 token-id 到 expert-id 查找表
        self.register_buffer(
            "tid2eid",
            torch.randint(0, num_experts, (vocab_size,))
        )

    def forward(self, x, input_ids):
        """
        x: (B, L, d_model)
        input_ids: (B, L) token ID 序列
        """
        B, L, D = x.size()
        x_flat = x.view(B * L, D)
        ids_flat = input_ids.view(B * L)

        # 查表获得专家索引
        expert_ids = self.tid2eid[ids_flat]  # (B*L,)

        out = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            mask = (expert_ids == e)
            if mask.any():
                out[mask] = self.experts[e](x_flat[mask])

        return out.view(B, L, D)
```

&emsp;&emsp;这个实现中，每个 token 的专家分配完全由冻结的 `tid2eid` 查找表决定，路由器不参与计算。若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.6.12 Stable LatentMoE

&emsp;&emsp;Stable LatentMoE 是 Kimi K3 采用的 MoE 架构，在 LatentMoE 的基础上增加了训练稳定性优化。其核心设计思想是将专家计算从主干隐藏维度解耦到低维潜在空间：主干保持较高的隐藏维度以维持全局表达能力，专家在低维潜在空间中计算以控制成本。

&emsp;&emsp;LatentMoE 的流程为：首先通过共享下投影矩阵 $W_{\mathrm{down}}$ 将 token 的隐藏状态 $h \in \mathbb{R}^{d}$ 投影到潜在空间：

$$
z = W_{\mathrm{down}} h, \quad z \in \mathbb{R}^{\ell}
$$

&emsp;&emsp;其中 $\ell = d / \alpha$ 是潜在维度，$\alpha$ 是压缩因子。然后所有路由专家都在潜在空间中执行计算：

$$
z' = \sum_{i \in \mathrm{TopK}} g_i \cdot \mathrm{Expert}_i(z)
$$

&emsp;&emsp;最后通过共享上投影矩阵 $W_{\mathrm{up}}$ 将结果映射回完整隐藏维度：

$$
h' = W_{\mathrm{up}} z'
$$

&emsp;&emsp;这种设计的核心优势是将“主干表达能力”和“专家计算成本”解耦。主干可以保持较宽的隐藏维度以维持表示能力，而专家的计算在低维潜在空间中进行，参数量和计算量都大幅降低。

&emsp;&emsp;Kimi K3 的 Stable LatentMoE 具体配置为：896 个路由专家，每个 token 激活 16 个专家，同时保留 2 个全宽度共享专家。隐藏状态从 7168 维投影到 3584 维的潜在空间（压缩比 2:1），在潜在空间中执行专家计算后再映射回完整维度。训练稳定性方面，Kimi K3 采用 SiTU-GLU 作为专家激活函数，并配合 Quantile Balancing 进行负载均衡，在极高稀疏度下保持训练稳定。

&emsp;&emsp;Stable LatentMoE 的优点是解耦了主干宽度与专家计算成本，使模型可以在不增加专家计算量的情况下扩展主干表示能力，共享专家提供了稳定的通用变换路径，SiTU-GLU 的平滑门控和 Quantile Balancing 的精确均衡共同保障了训练稳定性；缺点是引入了潜在维度和压缩比两个额外超参数，下投影和上投影增加了额外的计算开销，且潜在空间的维度选择需要权衡表达能力和计算效率。

&emsp;&emsp;从维度视角看，Stable LatentMoE 在特征维上引入了一个低维瓶颈：token 先被压缩到潜在空间，在低维空间中完成专家选择和非线性变换，再展开回原始维度。这相当于在特征维上做了一次“压缩-条件计算-解压缩”的循环。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 Stable LatentMoE 的裸实现：

```python
import torch
import torch.nn as nn

class LatentExpert(nn.Module):
    """在潜在空间中计算的专家"""
    def __init__(self, latent_dim, d_ff_latent):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(latent_dim, d_ff_latent))
        self.b_1 = nn.Parameter(torch.zeros(d_ff_latent))
        self.W_2 = nn.Parameter(torch.empty(d_ff_latent, latent_dim))
        self.b_2 = nn.Parameter(torch.zeros(latent_dim))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def _situ(self, x):
        sp = torch.maximum(x, torch.zeros_like(x)) + torch.log1p(torch.exp(-torch.abs(x)))
        return x * torch.tanh(sp)

    def forward(self, z):
        return torch.matmul(
            self._situ(torch.matmul(z, self.W_1) + self.b_1),
            self.W_2
        ) + self.b_2


class StableLatentMoE(nn.Module):
    def __init__(self, d_model, latent_dim, num_routed, num_shared,
                 top_k, d_ff_latent):
        super().__init__()
        self.num_routed = num_routed
        self.top_k = top_k

        # 共享下投影和上投影
        self.W_down = nn.Parameter(torch.empty(d_model, latent_dim))
        self.W_up = nn.Parameter(torch.empty(latent_dim, d_model))
        nn.init.xavier_uniform_(self.W_down)
        nn.init.xavier_uniform_(self.W_up)

        # 路由专家（潜在空间）
        self.routed_experts = nn.ModuleList([
            LatentExpert(latent_dim, d_ff_latent) for _ in range(num_routed)
        ])
        # 共享专家（全宽度）
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff_latent * 2), nn.ReLU(),
                nn.Linear(d_ff_latent * 2, d_model)
            ) for _ in range(num_shared)
        ])

        self.router = nn.Parameter(torch.empty(num_routed, d_model))
        nn.init.normal_(self.router, std=0.02)

    def forward(self, x):
        B, L, D = x.size()
        x_flat = x.view(B * L, D)

        # 共享专家（全宽度）
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)

        # 投影到潜在空间
        z = torch.matmul(x_flat, self.W_down)  # (B*L, latent_dim)

        # 路由
        scores = torch.matmul(x_flat, self.router.T)
        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)
        topk_gates = torch.softmax(topk_scores, dim=-1)

        # 在潜在空间中计算路由专家
        routed_out = torch.zeros_like(z)
        for k in range(self.top_k):
            idx = topk_indices[:, k]
            gate = topk_gates[:, k].unsqueeze(-1)
            for e in range(self.num_routed):
                mask = (idx == e)
                if mask.any():
                    routed_out[mask] += gate[mask] * self.routed_experts[e](z[mask])

        # 投影回完整维度
        routed_out = torch.matmul(routed_out, self.W_up)

        return (shared_out + routed_out).view(B, L, D)
```

&emsp;&emsp;这个实现中，路由专家在潜在空间中计算，共享专家在完整维度中计算。若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.6.13 Quantile Balancing

&emsp;&emsp;Quantile Balancing（QB）是由 Jianlin Su 提出的一种无辅助损失的 MoE 负载均衡方法，将负载均衡问题建模为等式约束的线性规划，并用分位数求解路由偏置。

&emsp;&emsp;QB 的核心思想是：不再通过辅助损失惩罚主损失，而是为每个专家维护一个偏置 $\beta_j$，偏置只影响专家的排序或阈值，不参与梯度计算。在 Top-k 路由版本中，路由器打分矩阵为 $s \in \mathbb{R}^{m \times n}$，其中 $m$ 是 token 数，$n$ 是专家数。QB 的目标是找到偏置 $\beta \in \mathbb{R}^n$，使得每个专家被激活的次数接近 $mk/n$。

&emsp;&emsp;QB 的最优解具有简洁的分位数形式。对于每个专家 $j$，其最优偏置为：

$$
\beta_j = \text{第 } \frac{mk}{n} \text{ 大的 } s_{i,j} \text{ 值}
$$

&emsp;&emsp;这对应于打分矩阵第 $j$ 列的 $1 - k/n$ 分位数。也就是说，QB 不需要交替迭代或梯度下降，而是直接从当前 batch 的打分矩阵中一步计算出最优偏置。推理时的路由决策变为：

$$
\mathrm{TopK}(s_i - \beta)
$$

&emsp;&emsp;其中 $s_i$ 是第 $i$ 个 token 的打分向量。在动态激活版本中，去掉“每 token 恰好激活 $k$ 个专家”的行约束，只保留列方向的负载约束和平均预算，激活条件变为 $s_{i,j} - \beta_j > 0$，一步分位数求解即可得到绝对均衡的最优解。

&emsp;&emsp;Quantile Balancing 的优点是没有学习率等超参数需要调优，均衡速度快，尤其擅长处理极端不均衡的情况，如在全 MoE 模型中第一层也能快速达到均衡，且不产生干扰梯度，与主任务优化完全解耦；缺点是需要计算打分矩阵的分位数，在大规模分布式训练中分位数的跨设备通信可能成为开销，且 QB 假设每个 token 的打分矩阵在 batch 内可比较，在某些流式或小 batch 场景下可能不够稳定。

&emsp;&emsp;从维度视角看，Quantile Balancing 在特征维上通过分位数确定偏置，使每个专家的有效容量被精确地校准到目标负载。它不修改损失函数，而是直接调整路由的决策边界。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 Quantile Balancing 的裸实现：

```python
import torch
import torch.nn as nn

class QuantileBalancedMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        self.router = nn.Parameter(torch.empty(num_experts, d_model))
        nn.init.normal_(self.router, std=0.02)

        self.register_buffer("beta", torch.zeros(num_experts))

    def update_beta(self, scores):
        """
        scores: (m, n) 打分矩阵, m 为 token 数, n 为专家数
        一步分位数更新 beta
        """
        m, n = scores.shape
        k = self.top_k
        target = int(m * k / n)  # 每个专家的目标激活次数

        with torch.no_grad():
            for j in range(n):
                col = scores[:, j].sort(descending=True).values
                # beta_j = 第 target 大的值（1 - k/n 分位数）
                if target < m:
                    self.beta[j] = col[target]
                else:
                    self.beta[j] = col[-1]

    def forward(self, x, update=True):
        B, L, D = x.size()
        x_flat = x.view(B * L, D)

        scores = torch.matmul(x_flat, self.router.T)  # (B*L, num_experts)

        if update:
            self.update_beta(scores)

        # 用偏置调整后的分数做 Top-K
        adjusted = scores - self.beta.unsqueeze(0)
        topk_scores, topk_indices = adjusted.topk(self.top_k, dim=-1)
        topk_gates = torch.softmax(scores.gather(1, topk_indices), dim=-1)

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            idx = topk_indices[:, k]
            gate = topk_gates[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    out[mask] += gate[mask] * self.experts[e](x_flat[mask])

        return out.view(B, L, D)
```

&emsp;&emsp;这个实现中，`update_beta` 从打分矩阵的每一列取第 $m k / n$ 大的值作为偏置，实现了一步分位数均衡。若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

### 2.7 残差连接与初始化

#### 2.7.1 标准残差连接

&emsp;&emsp;标准残差连接（Standard Residual Connection）是 Transformer 中最基础的跨层信息通路，它将子层的输出与输入直接相加：

$$
x_{l+1} = x_l + F_l(x_l)
$$

&emsp;&emsp;其中 $x_l$ 是第 $l$ 层的输入，$F_l$ 是该层的子层函数（注意力或 FFN）。展开多层后，第 $L$ 层的输出可以写成：

$$
x_L = x_0 + \sum_{l=0}^{L-1} F_l(x_l)
$$

&emsp;&emsp;这个形式揭示了残差连接的两个核心性质。第一，主干路径 $x_0$ 是恒等映射，浅层信息可以无损地传到任意深度。第二，每层的贡献 $F_l(x_l)$ 以累加方式叠加到主干上，而不是替换主干。反向传播时，梯度为：

$$
\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{l=0}^{L-1}\left(I + \frac{\partial F_l}{\partial x_l}\right)
$$

&emsp;&emsp;乘积中的 $I$ 保证了梯度至少有一条恒等通路可以回传，不会因为连乘而指数衰减。标准残差连接的优点是实现极简，仅一次加法，不增加参数，且为深层网络提供了稳定的梯度通路，是 Transformer 能堆叠上百层的基础；缺点是主干路径上的激活值会随层数线性累积增长，深层网络中残差分支的输出可能被主干的恒等信号淹没，且所有层共享同一条主干路径，无法对不同深度的信息流做差异化调节。

&emsp;&emsp;从维度视角看，标准残差连接在特征维上为每个子层提供了一条恒等通路，使 token 维扩散的结果以增量方式叠加到主干表示上。主干和残差分支是固定的一对一相加关系，没有可学习的混合机制。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出标准残差连接的裸实现，并与无残差版本对比：

```python
import torch
import torch.nn as nn

class StandardResidual(nn.Module):
    """标准残差连接: x_{l+1} = x_l + F_l(x_l)"""
    def forward(self, x, sublayer_fn):
        return x + sublayer_fn(x)


class NoResidual(nn.Module):
    """无残差连接，用于对比"""
    def forward(self, x, sublayer_fn):
        return sublayer_fn(x)
```

&emsp;&emsp;若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.7.2 残差路径缩放初始化

&emsp;&emsp;残差路径缩放初始化（Residual Path Scaling Initialization）是针对标准残差连接中激活值随深度累积增长问题的初始化策略。GPT-2 首次系统性地采用了这一方案：将每个残差分支的输出投影层权重按 $1/\sqrt{2N}$ 缩放，其中 $N$ 是残差层总数。

&emsp;&emsp;设第 $l$ 层的子层输出为 $F_l(x_l)$，其最后一层线性变换的权重初始化为：

$$
W_l \sim \mathcal{N}\left(0, \frac{\sigma^2}{2N}\right)
$$

&emsp;&emsp;其中 $\sigma$ 是基础标准差，$2N$ 是因为每层有两个残差分支（注意力和 FFN）。这种缩放使得每个残差分支的输出方差被压缩，从而在 $N$ 层累加后，主干激活值的方差仍保持在合理范围内。若不缩放，每层残差分支的方差为 $\sigma^2$，$N$ 层累加后主干方差约为 $N\sigma^2$，随深度线性增长，导致深层激活值爆炸。

&emsp;&emsp;从方差传播的角度看，设每层残差分支的输出独立且方差为 $\sigma_l^2$，则主干方差为：

$$
\mathrm{Var}(x_L) = \mathrm{Var}(x_0) + \sum_{l=0}^{L-1} \sigma_l^2
$$

&emsp;&emsp;若每层 $\sigma_l^2 = \sigma^2 / (2N)$，则 $N$ 层累加后总方差约为 $\mathrm{Var}(x_0) + \sigma^2/2$，与深度无关。残差路径缩放初始化的优点是无需额外参数或计算，仅在初始化时缩放权重，就能使深层网络的激活值方差保持稳定，配合 Pre-LN 可以训练上百层模型；缺点是需要知道总层数 $N$，不适用于动态深度或层数变化的场景，且缩放因子 $1/\sqrt{2N}$ 是启发式的，不同任务和模型规模下最优缩放可能不同。

&emsp;&emsp;从维度视角看，残差路径缩放初始化在特征维上控制了每层残差分支的贡献幅度，使主干上的累积信号不会随深度失控。它不改变残差连接的数学形式，只改变了各层贡献的相对尺度。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出残差路径缩放初始化的裸实现，对输出投影层按 $1/\sqrt{2N}$ 缩放：

```python
import math
import torch
import torch.nn as nn

class ScaledResidualBlock(nn.Module):
    """带残差路径缩放初始化的子层"""
    def __init__(self, d_model, d_ff, num_layers):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.W_1)
        # 输出投影按 1/sqrt(2N) 缩放
        std = math.sqrt(2.0 / (2 * num_layers)) / math.sqrt(d_ff)
        nn.init.normal_(self.W_2, std=std)

    def forward(self, x):
        hidden = torch.relu(torch.matmul(x, self.W_1) + self.b_1)
        return torch.matmul(hidden, self.W_2) + self.b_2


class ScaledResidualStack(nn.Module):
    def __init__(self, d_model, d_ff, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            ScaledResidualBlock(d_model, d_ff, num_layers)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x
```

&emsp;&emsp;这个实现中，每个残差块的输出投影 `W_2` 按 $1/\sqrt{2N}$ 缩放初始化，使多层累加后方差保持稳定。若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.7.3 零初始化与零卷积

&emsp;&emsp;零初始化（Zero Initialization）与零卷积（Zero Convolution）是一类将子层输出投影初始化为零的技术，使每个残差块在训练开始时退化为恒等映射。设子层为 $F_l$，将其最后一层权重初始化为零：

$$
W_l^{out} = 0, \quad b_l^{out} = 0
$$

&emsp;&emsp;则训练初期 $F_l(x) = 0$，残差块输出为 $x + 0 = x$，整个网络在初始化时等价于恒等映射。这种设计的代表工作包括 ReZero、ControlNet 的零卷积、以及 GLM 等模型中的零初始化输出层。

&emsp;&emsp;零初始化的核心优势在于训练稳定性：由于每个残差块初始时对主干无贡献，网络的初始行为等同于一个浅层模型，随着训练逐步学习到非零的残差贡献。这避免了深层网络初始化时的信号爆炸和梯度问题，同时让每个残差块从“零贡献”开始渐进地学习自己的功能。ReZero 进一步引入了一个可学习的标量 $\alpha_l$，初始化为零：

$$
x_{l+1} = x_l + \alpha_l F_l(x_l)
$$

&emsp;&emsp;这样 $\alpha_l$ 可以从零开始随训练增长，自动调节每个残差分支的贡献强度。零初始化与零卷积的优点是训练稳定性极佳，深层网络在初始化时等价于浅层网络，每个残差块渐进学习，且 ReZero 的 $\alpha_l$ 提供了可学习的贡献强度；缺点是训练初期残差分支梯度极小（因为输出为零），学习速度较慢，需要更多训练步数才能达到与标准初始化相当的性能，且零初始化破坏了权重的对称性，某些情况下需要额外的扰动来打破对称。

&emsp;&emsp;从维度视角看，零初始化在特征维上让每个残差分支的初始贡献为零，主干路径完全主导初始表示。随着训练进行，各分支逐步学习到非零的修正信号，相当于从恒等映射出发渐进地学习深度。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出零初始化与 ReZero 风格的裸实现：

```python
import torch
import torch.nn as nn

class ZeroInitBlock(nn.Module):
    """零初始化残差块: 输出投影初始化为零"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.zeros(d_ff, d_model))  # 零初始化
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_1)

    def forward(self, x):
        hidden = torch.relu(torch.matmul(x, self.W_1) + self.b_1)
        return torch.matmul(hidden, self.W_2) + self.b_2


class ReZeroBlock(nn.Module):
    """ReZero: 可学习标量 alpha 初始化为零"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.alpha = nn.Parameter(torch.zeros(1))  # 初始化为零
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def forward(self, x):
        hidden = torch.relu(torch.matmul(x, self.W_1) + self.b_1)
        out = torch.matmul(hidden, self.W_2) + self.b_2
        return x + self.alpha * out
```

&emsp;&emsp;这个实现中，`ZeroInitBlock` 将输出投影初始化为零，`ReZeroBlock` 使用可学习的标量 $\alpha$ 初始化为零。若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.7.4 mHC：流形约束超连接

&emsp;&emsp;mHC（Manifold-Constrained Hyper-Connections，流形约束超连接）由字节跳动 Seed 团队于 2025 年提出，是对超连接（Hyper-Connections, HC）的改进。超连接将标准残差连接从单条主干扩展为 $n$ 条并行残差流，每层通过可学习的混合矩阵在这 $n$ 条流之间重新分配信息，从而提升模型的表达能力和训练稳定性。

&emsp;&emsp;设第 $l$ 层有 $n$ 条残差流 $x_l^{(1)}, \dots, x_l^{(n)}$，超连接的更新规则为：

$$
x_{l+1}^{(i)} = \sum_{j=1}^{n} A_{ij} x_l^{(j)} + F_l\left(\sum_{j=1}^{n} B_{ij} x_l^{(j)}\right)
$$

&emsp;&emsp;其中 $A \in \mathbb{R}^{n \times n}$ 是残差流的混合矩阵，$B \in \mathbb{R}^{n \times n}$ 是子层输入的混合矩阵。当 $n=1$ 时，超连接退化为标准残差连接。HC 的问题是 $A$ 和 $B$ 是无约束的可学习矩阵，训练中可能出现特征值超出稳定范围的情况，导致残差流爆炸或消失。

&emsp;&emsp;mHC 的核心创新是将混合矩阵 $A$ 约束在双随机矩阵流形上，即 $A$ 的每一行和每一列的元素之和都等于 1，且所有元素非负：

$$
A \mathbf{1} = \mathbf{1}, \quad \mathbf{1}^\top A = \mathbf{1}^\top, \quad A_{ij} \geq 0
$$

&emsp;&emsp;双随机矩阵的谱范数不超过 1，即所有特征值的绝对值 $\leq 1$，且 1 是最大的特征值。这意味着残差流的混合是保范的，不会放大信号，从而保证了深层堆叠的稳定性。此外，双随机矩阵的乘积仍然是双随机矩阵，因此多层混合的复合仍然保持稳定。

&emsp;&emsp;为了让 $A$ 在训练中始终保持双随机性，mHC 采用 Sinkhorn-Knopp 算法对可学习的原始矩阵进行归一化，或将参数化设计为流形上的内在坐标。mHC 的优点是显著提升了深层网络的训练稳定性，双随机约束使残差流的谱范数有界，混合矩阵的复合不会放大信号，在视觉和大语言模型任务上都报告了优于标准残差连接的性能；缺点是引入了 $n^2$ 级别的混合参数和额外的 Sinkhorn 归一化计算，$n$ 较大时计算开销不可忽略，且双随机约束可能限制了混合矩阵的表达能力。

&emsp;&emsp;从维度视角看，mHC 将残差路径从单条主干扩展为 $n$ 条并行流，并通过双随机矩阵在各流之间做保范的线性混合。这相当于在特征维上维护了 $n$ 条并行的信息高速公路，并通过流形约束保证信息在流之间的重新分配不会导致尺度失控。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 mHC 的裸实现，包含双随机约束的混合矩阵和 Sinkhorn 归一化：

```python
import torch
import torch.nn as nn

class ManifoldConstrainedHyperConnection(nn.Module):
    """mHC: 多条残差流 + 双随机混合矩阵"""
    def __init__(self, d_model, num_streams, d_ff, sinkhorn_iters=20):
        super().__init__()
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters

        # 混合矩阵的可学习原始参数
        self.A_raw = nn.Parameter(torch.eye(num_streams) + 0.01 * torch.randn(num_streams, num_streams))
        self.B_raw = nn.Parameter(torch.eye(num_streams) + 0.01 * torch.randn(num_streams, num_streams))

        # 子层参数
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def _sinkhorn(self, M):
        """Sinkhorn-Knopp 算法: 将正矩阵归一化为双随机矩阵"""
        M = torch.exp(M)  # 保证非负
        for _ in range(self.sinkhorn_iters):
            M = M / (M.sum(dim=1, keepdim=True) + 1e-9)
            M = M / (M.sum(dim=0, keepdim=True) + 1e-9)
        return M

    def forward(self, streams):
        """
        streams: (n, B, L, d_model) 或 list of (B, L, d_model)
        """
        # 双随机混合矩阵
        A = self._sinkhorn(self.A_raw)  # (n, n)
        B = self._sinkhorn(self.B_raw)

        # 堆叠为 (n, B, L, d_model)
        if isinstance(streams, list):
            X = torch.stack(streams, dim=0)
        else:
            X = streams
        n = X.size(0)

        # 残差流混合: sum_j A_ij X_j
        X_mixed = torch.einsum("ij,jbld->ibld", A, X)
        # 子层输入混合: sum_j B_ij X_j
        X_sub = torch.einsum("ij,jbld->ibld", B, X)

        # 子层计算
        hidden = torch.relu(torch.matmul(X_sub, self.W_1) + self.b_1)
        F_out = torch.matmul(hidden, self.W_2) + self.b_2

        # 输出: 混合残差 + 子层输出
        return X_mixed + F_out
```

&emsp;&emsp;这个实现中，`_sinkhorn` 将可学习的原始矩阵归一化为双随机矩阵，保证谱范数不超过 1。若输入为 $n$ 条残差流，每条形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.7.5 双随机矩阵与 Sinkhorn-Knopp

&emsp;&emsp;双随机矩阵（Doubly Stochastic Matrix）是指所有元素非负、每行之和为 1、每列之和也为 1 的方阵。形式化地，$A \in \mathbb{R}^{n \times n}$ 是双随机矩阵当且仅当：

$$
A_{ij} \geq 0, \quad \sum_{j=1}^{n} A_{ij} = 1, \quad \sum_{i=1}^{n} A_{ij} = 1
$$

&emsp;&emsp;双随机矩阵在数学上具有优良的性质。根据 Birkhoff-von Neumann 定理，任何双随机矩阵都可以表示为置换矩阵的凸组合。双随机矩阵的谱范数不超过 1，即 $\|A\|_2 \leq 1$，且 1 是最大特征值，对应的特征向量是 $\mathbf{1}/\sqrt{n}$。这意味着双随机矩阵作用在任意向量上不会放大其范数，且双随机矩阵的乘积仍然是双随机矩阵。这些性质使双随机矩阵成为残差流混合的理想选择：多层复合后信号不会爆炸，也不会消失。

&emsp;&emsp;Sinkhorn-Knopp 算法是一种将正矩阵迭代归一化为双随机矩阵的经典算法。给定一个元素全为正的矩阵 $M$，算法交替进行行归一化和列归一化：

$$
M^{(t+1/2)} = \mathrm{diag}(M^{(t)} \mathbf{1})^{-1} M^{(t)}
$$

$$
M^{(t+1)} = M^{(t+1/2)} \mathrm{diag}(\mathbf{1}^\top M^{(t+1/2)})^{-1}
$$

&emsp;&emsp;即先把每一行归一化到和为 1，再把每一列归一化到和为 1，如此交替迭代。Sinkhorn 证明了对于任意正矩阵，这个过程收敛到一个双随机矩阵。实际中迭代 10 到 20 次即可达到足够精度。在 mHC 中，可学习的原始矩阵 $A_{raw}$ 先通过指数函数映射到正数域，再用 Sinkhorn 归一化为双随机矩阵。

&emsp;&emsp;双随机矩阵与 Sinkhorn-Knopp 的优点是数学性质优良，谱范数有界保证信号不放大，乘积封闭性保证多层复合稳定，Sinkhorn 算法简单且收敛快，无需额外的超参数；缺点是 Sinkhorn 迭代增加了前向和反向的计算开销，且双随机约束将所有矩阵限制在同一流形上，可能损失部分表达能力，对于非方阵或非平衡场景，双随机约束需要推广为行随机或列随机。

&emsp;&emsp;从维度视角看，双随机矩阵在特征维上提供了一种保范的线性混合：它重新分配各条残差流的信息，但不改变总的信息量。Sinkhorn-Knopp 算法则是将无约束的可学习矩阵投影到这个保范流形上的工具。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出双随机矩阵和 Sinkhorn-Knopp 算法的裸实现：

```python
import torch
import torch.nn as nn

def sinkhorn_knopp(M, num_iters=20, eps=1e-9):
    """
    将正矩阵 M 归一化为双随机矩阵。
    M: (n, n) 元素为正的矩阵
    """
    M = M.clone()
    for _ in range(num_iters):
        # 行归一化
        M = M / (M.sum(dim=1, keepdim=True) + eps)
        # 列归一化
        M = M / (M.sum(dim=0, keepdim=True) + eps)
    return M


class DoublyStochasticLinear(nn.Module):
    """参数化双随机矩阵的模块"""
    def __init__(self, n, num_iters=20):
        super().__init__()
        self.n = n
        self.num_iters = num_iters
        # 可学习原始参数
        self.M_raw = nn.Parameter(torch.randn(n, n) * 0.01)

    def forward(self):
        # 指数映射到正数域，再 Sinkhorn 归一化
        M = torch.exp(self.M_raw)
        return sinkhorn_knopp(M, self.num_iters)


# 验证双随机性质
if __name__ == "__main__":
    layer = DoublyStochasticLinear(n=4)
    A = layer()
    print("行和:", A.sum(dim=1))  # 应接近全 1
    print("列和:", A.sum(dim=0))  # 应接近全 1
    print("最小元素:", A.min().item())  # 应 >= 0
    # 谱范数检查
    print("谱范数:", torch.linalg.norm(A, ord=2).item())  # 应 <= 1
```

&emsp;&emsp;这个实现中，`sinkhorn_knopp` 交替进行行归一化和列归一化，将任意正矩阵转化为双随机矩阵。`DoublyStochasticLinear` 将可学习参数通过指数映射保证非负，再经 Sinkhorn 归一化。实际使用中，每次前向传播都需要调用 Sinkhorn 迭代，反向传播通过迭代过程自动求导。若输入矩阵形状为 $(n,n)$，输出形状相同，且满足行和、列和均为 1。


---

### 2.8 分词器

#### 2.8.1 BPE

&emsp;&emsp;BPE（Byte Pair Encoding，字节对编码）最初是一种数据压缩算法，后被 Sennrich 等人引入神经机器翻译，成为最常用的子词分词方法之一。BPE 的核心思想是从字符级表示出发，迭代合并频率最高的相邻符号对，逐步构建子词词表。训练过程为：将语料中的每个词拆分为字符序列，并在词尾添加结束符；统计所有相邻符号对的频率；合并频率最高的符号对为一个新符号；重复上述过程直到词表达到预设大小。编码时，将新词拆分为字符，然后按照训练时学到的合并规则依次合并，直到无法再合并。

&emsp;&emsp;设当前符号序列为 $s = (s_1, s_2, \dots, s_m)$，相邻符号对 $(s_i, s_{i+1})$ 的频率为 $\mathrm{freq}(s_i, s_{i+1})$。BPE 每步选择：

$$
(a, b) = \arg\max_{(x,y)} \mathrm{freq}(x, y)
$$

&emsp;&emsp;然后将所有出现的相邻对 $(a, b)$ 合并为新符号 $ab$。BPE 的优点是算法简单、训练和编码速度快，能有效平衡词表大小和序列长度，且对未登录词可以通过子词组合表示；缺点是合并规则基于频率贪心选择，可能产生不合理的子词边界，对多语言和形态丰富语言的支持依赖训练语料，且字符级初始化使低频字符的表示可能不够充分。

&emsp;&emsp;从维度视角看，BPE 在 token 维上做了一次离散化的降维：将字符序列合并为更长的子词单元，减少序列长度，同时保持词表规模可控。

**&emsp;&emsp;Python 实现示例**

&emsp;&emsp;下面给出 BPE 训练和编码的裸实现：

```python
from collections import Counter

def get_stats(vocab):
    """统计相邻符号对频率"""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """合并指定符号对"""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab

def train_bpe(corpus, num_merges):
    # 初始化：每个词拆为字符，词尾加 </w>
    vocab = Counter()
    for word in corpus:
        chars = ' '.join(list(word)) + ' </w>'
        vocab[chars] += 1

    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    return merges

def encode_bpe(word, merges):
    symbols = list(word) + ['</w>']
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i+1] == b:
                symbols = symbols[:i] + [a+b] + symbols[i+2:]
            else:
                i += 1
    return symbols
```

&emsp;&emsp;这个实现中，`train_bpe` 迭代合并频率最高的相邻符号对，`encode_bpe` 按合并顺序依次应用规则。若输入为单词列表，输出为子词序列。

---

#### 2.8.2 Byte-level BPE

&emsp;&emsp;Byte-level BPE（字节级 BPE）由 Radford 等人在 GPT-2 中提出，将 BPE 的基础单元从 Unicode 字符改为字节。文本首先被编码为 UTF-8 字节序列，每个字节映射到一个可打印的 Unicode 字符，然后在这个字节序列上执行标准 BPE。基础词表固定为 256 个字节，因此任何文本都可以被无损表示，不存在未登录词问题。

&emsp;&emsp;字节级 BPE 的关键设计是字节到 Unicode 的映射表，将 256 个字节中的可打印字符保持不变，不可打印字符映射到 Unicode 的私用区或其他可打印区间。例如，空格映射为 Ġ，换行映射为 Ċ。设字节序列为 $b_1, b_2, \dots, b_m$，映射函数为 $\phi$，则初始符号序列为 $\phi(b_1), \phi(b_2), \dots, \phi(b_m)$，然后执行标准 BPE 合并。

&emsp;&emsp;字节级 BPE 的优点是完全消除了未登录词问题，任何文本都可以表示，且对多语言、表情符号、代码等混合内容鲁棒；缺点是字节级序列比字符级更长，导致序列长度增加，推理速度变慢，且多字节字符被拆分为多个字节，可能增加模型学习难度。

&emsp;&emsp;从维度视角看，字节级 BPE 将 token 维的离散化粒度降到字节级别，使词表覆盖所有可能的输入，但代价是序列长度增加，注意力计算的 token 维规模变大。

**&emsp;&emsp;Python 实现示例**

&emsp;&emsp;下面给出字节级 BPE 的字节映射和简化训练实现：

```python
from collections import Counter

def bytes_to_unicode():
    """构建字节到可打印 Unicode 的映射"""
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

def train_byte_level_bpe(corpus, num_merges):
    byte_encoder = bytes_to_unicode()
    vocab = Counter()
    for text in corpus:
        # 文本 -> UTF-8 字节 -> 映射为 Unicode 字符
        byte_seq = text.encode('utf-8')
        symbols = [byte_encoder[b] for b in byte_seq]
        vocab[' '.join(symbols)] += 1

    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for word, freq in vocab.items():
            syms = word.split()
            for i in range(len(syms)-1):
                pairs[(syms[i], syms[i+1])] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        new_vocab = {}
        bigram = ' '.join(best)
        repl = ''.join(best)
        for word, freq in vocab.items():
            new_vocab[word.replace(bigram, repl)] = freq
        vocab = new_vocab
        merges.append(best)
    return merges, byte_encoder
```

&emsp;&emsp;这个实现中，文本先被编码为 UTF-8 字节，再映射为可打印 Unicode 字符，然后执行 BPE 合并。任何输入都可以被表示，不会出现未登录词。

---

#### 2.8.3 SentencePiece

&emsp;&emsp;SentencePiece 由 Kudo 和 Richardson 于 2018 年提出，是一种语言无关的子词分词框架。与 BPE 需要预分词不同，SentencePiece 直接将原始文本视为 Unicode 字符序列，包括空格，然后在该序列上训练 BPE 或 Unigram 语言模型。空格被转义为特殊符号 ▁（U+2581），使分词过程完全可逆，解码时只需将 ▁ 替换回空格。

&emsp;&emsp;SentencePiece 支持两种主要算法：BPE 和 Unigram。Unigram 语言模型从一个大词表开始，迭代删除对语言模型概率损失最小的子词，直到达到目标词表大小。与 BPE 的贪心合并不同，Unigram 基于概率模型，通常能产生更合理的子词划分。SentencePiece 的编码过程使用 Viterbi 算法在 Unigram 模型中寻找最优分割。

&emsp;&emsp;SentencePiece 的优点是语言无关，不需要预分词，适合中文、日文等无空格语言，可逆性保证解码无损，且支持 BPE 和 Unigram 两种算法，灵活性强；缺点是训练速度比纯 BPE 慢，Unigram 模型的实现复杂度较高，且 ▁ 符号的引入使词表中包含特殊标记，需要额外处理。

&emsp;&emsp;从维度视角看，SentencePiece 在 token 维上直接对 Unicode 字符序列做子词切分，不依赖语言特定的预分词规则，使 token 维的离散化对多语言统一。

**&emsp;&emsp;Python 实现示例**

&emsp;&emsp;下面给出 SentencePiece 风格的 Unigram 简化实现，包含 ▁ 转义和 Viterbi 编码：

```python
import math
from collections import Counter

def prepare_text(text):
    """将空格替换为 ▁"""
    return text.replace(' ', '▁')

def train_unigram(corpus, target_vocab_size):
    """简化 Unigram 训练：从字符开始，统计子词频率"""
    vocab = Counter()
    for text in corpus:
        text = prepare_text(text)
        for ch in text:
            vocab[ch] += 1
    # 简化为字符级词表，实际 Unigram 会迭代剪枝
    return dict(vocab)

def encode_unigram(text, vocab):
    """Viterbi 最优分割"""
    text = prepare_text(text)
    n = len(text)
    # dp[i] = (最小负对数概率, 最优分割)
    dp = [(0.0, []) for _ in range(n + 1)]
    for i in range(1, n + 1):
        best = (float('inf'), None)
        for j in range(i):
            sub = text[j:i]
            if sub in vocab:
                prob = vocab[sub] / sum(vocab.values())
                cost = dp[j][0] - math.log(prob + 1e-9)
                if cost < best[0]:
                    best = (cost, j)
        if best[1] is not None:
            j = best[1]
            dp[i] = (best[0], dp[j][1] + [text[j:i]])
        else:
            dp[i] = (dp[i-1][0] - math.log(1e-9), dp[i-1][1] + [text[i-1]])
    return dp[n][1]
```

&emsp;&emsp;这个实现中，`prepare_text` 将空格替换为 ▁，`encode_unigram` 使用 Viterbi 算法寻找最优子词分割。实际 SentencePiece 会训练完整的 Unigram 模型并剪枝词表。

---

#### 2.8.4 tiktoken

&emsp;&emsp;tiktoken 是 OpenAI 开发的快速 BPE 分词器，用于 GPT 系列模型。它的核心是字节级 BPE，但引入了一个正则表达式预分词步骤，将文本按模式拆分为片段，再对每个片段应用 BPE。GPT-2 使用的正则模式为：

$$
\text{pattern} = \text{``'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+''}
$$

&emsp;&emsp;这个模式将缩写、字母、数字、标点和空白分开，使 BPE 在不同类别内部合并，避免跨类别产生不合理的子词。tiktoken 的另一个优化是使用 Rust 实现核心循环，并支持并行编码，速度远快于 Python 实现。tiktoken 的优点是速度极快，与 OpenAI 模型完全兼容，字节级 BPE 保证无未登录词，正则预分词提升了子词质量；缺点是主要用于 OpenAI 模型，词表固定，不便于自定义训练，且正则模式对多语言特别是中文、日文等无空格语言的支持有限，需要依赖字节回退。

&emsp;&emsp;从维度视角看，tiktoken 在 BPE 之前增加了一次基于正则的 token 维预切分，将文本按字符类别分组，使后续 BPE 合并在语义一致的片段内进行。

**&emsp;&emsp;Python 实现示例**

&emsp;&emsp;下面给出 tiktoken 风格的正则预分词 + 字节级 BPE 的简化实现：

```python
import re
from collections import Counter

# GPT-2 风格的正则预分词模式（简化）
PAT = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+")

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

def pre_tokenize(text):
    return PAT.findall(text)

def byte_level_bpe_train(corpus, num_merges):
    byte_encoder = bytes_to_unicode()
    vocab = Counter()
    for text in corpus:
        for chunk in pre_tokenize(text):
            byte_seq = chunk.encode('utf-8')
            symbols = [byte_encoder[b] for b in byte_seq]
            vocab[' '.join(symbols)] += 1

    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for word, freq in vocab.items():
            syms = word.split()
            for i in range(len(syms)-1):
                pairs[(syms[i], syms[i+1])] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        bigram = ' '.join(best)
        repl = ''.join(best)
        new_vocab = {}
        for word, freq in vocab.items():
            new_vocab[word.replace(bigram, repl)] = freq
        vocab = new_vocab
        merges.append(best)
    return merges, byte_encoder
```

&emsp;&emsp;这个实现中，`pre_tokenize` 使用正则将文本拆分为片段，然后对每个片段做字节级 BPE。实际 tiktoken 使用预编译的词表和 Rust 加速。

---

#### 2.8.5 词表大小与多语言

&emsp;&emsp;词表大小是多语言模型设计中的关键超参数。词表越大，单个 token 能承载的信息越多，序列长度越短，但嵌入矩阵和输出层的参数量也越大，且低频 token 的训练样本不足，容易欠拟合。词表越小，参数量越少，但序列长度增加，注意力计算的成本上升，且多语言字符可能被拆分为多个字节，进一步拉长序列。

&emsp;&emsp;设词表大小为 $V$，模型维度为 $d$，则嵌入矩阵参数量为 $V \times d$。对于 128K 词表和 4096 维模型，嵌入矩阵约有 5 亿参数。多语言场景下，不同语言的字符频率差异巨大，若训练语料以英语为主，其他语言的字符可能被拆分得很碎。常见的平衡策略包括：增加词表大小以覆盖更多语言的常用子词；为特定语言添加专用 token；使用字节级回退保证任何语言都能表示；在训练语料中按语言比例采样，使词表反映多语言分布。

&emsp;&emsp;词表大小与多语言的权衡没有统一最优解。实践表明，英语为主的模型词表通常在 32K 到 128K 之间，多语言模型倾向于使用更大的词表（如 128K 到 256K），并在训练中引入语言特定的子词。词表大小与多语言设计的优点是合理配置可以显著提升多语言性能，减少序列长度和推理成本；缺点是词表扩大带来参数量和显存开销，多语言平衡需要大量实验调优，且小语种在词表中仍可能被过度拆分。

&emsp;&emsp;从维度视角看，词表大小决定了 token 维离散化的粒度：词表越大，每个 token 覆盖的字符越多，序列越短；词表越小，序列越长但词表嵌入矩阵越小。多语言场景需要在不同语言的字符分布之间找到平衡点。

**&emsp;&emsp;Python 实现示例**

&emsp;&emsp;下面给出词表大小与序列长度的简单分析代码：

```python
import math

def vocab_tradeoff(vocab_size, d_model, avg_token_per_word=1.0):
    """计算嵌入参数量和序列长度关系"""
    embed_params = vocab_size * d_model
    # 假设词表越大，平均每个 token 覆盖的字符越多
    avg_chars_per_token = math.log(vocab_size) / math.log(2)
    seq_len_factor = 1.0 / avg_chars_per_token
    return {
        "vocab_size": vocab_size,
        "embed_params": embed_params,
        "avg_chars_per_token": avg_chars_per_token,
        "relative_seq_len": seq_len_factor,
    }

# 示例：比较不同词表大小
for vs in [32000, 64000, 128000, 256000]:
    result = vocab_tradeoff(vs, d_model=4096)
    print(f"词表 {vs:6d} | 嵌入参数 {result['embed_params']/1e6:8.1f}M | "
          f"平均字符/token {result['avg_chars_per_token']:.2f} | "
          f"相对序列长度 {result['relative_seq_len']:.4f}")
```

&emsp;&emsp;这个实现中，`vocab_tradeoff` 展示了词表大小与嵌入参数量、平均字符/token 数的关系。实际模型设计中，需要结合训练语料的多语言分布、序列长度限制和硬件显存综合选择词表大小。

---

### 2.9 训练目标

#### 2.9.1 自回归语言建模

&emsp;&emsp;自回归语言建模（Autoregressive Language Modeling, AR）是 Decoder-only 模型的标准训练目标。给定 token 序列 $x = (x_1, x_2, \dots, x_T)$，模型在每一步根据之前的所有 token 预测下一个 token，训练目标为最大化对数似然：

$$
\mathcal{L}_{\mathrm{AR}} = \sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)
$$

&emsp;&emsp;其中 $x_{<t} = (x_1, \dots, x_{t-1})$，$P(x_t \mid x_{<t})$ 由模型输出的 Softmax 分布给出。训练时使用因果掩码保证每个位置只能看到左侧上下文，损失为交叉熵：

$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log \frac{\exp(z_{t, x_t})}{\sum_{v=1}^{V} \exp(z_{t, v})}
$$

&emsp;&emsp;其中 $z_t$ 是模型在位置 $t$ 的 logits，$V$ 是词表大小。自回归语言建模的优点是训练目标与生成过程完全一致，模型可以直接用于逐 token 生成，且因果掩码使训练可以并行处理整个序列；缺点是每个位置只预测一个 token，监督信号密度较低，且只能利用左侧上下文，无法像双向模型那样同时利用右侧信息。

&emsp;&emsp;从维度视角看，自回归语言建模在 token 维上施加因果约束，使关系矩阵为下三角，信息从过去流向未来。模型在每个位置输出对下一个 token 的预测，特征维上的表示被映射到词表分布。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出自回归语言建模的裸实现，包含因果自注意力、前馈网络和交叉熵损失：

```python
import math
import torch
import torch.nn as nn

class AutoregressiveLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=512):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.vocab_size = vocab_size

        self.token_emb = nn.Parameter(torch.empty(vocab_size, d_model))
        self.pos_emb = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.token_emb, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "W_q": nn.Parameter(torch.empty(d_model, d_model)),
                "W_k": nn.Parameter(torch.empty(d_model, d_model)),
                "W_v": nn.Parameter(torch.empty(d_model, d_model)),
                "W_o": nn.Parameter(torch.empty(d_model, d_model)),
                "W_1": nn.Parameter(torch.empty(d_model, d_ff)),
                "b_1": nn.Parameter(torch.zeros(d_ff)),
                "W_2": nn.Parameter(torch.empty(d_ff, d_model)),
                "b_2": nn.Parameter(torch.zeros(d_model)),
                "ln1_w": nn.Parameter(torch.ones(d_model)),
                "ln1_b": nn.Parameter(torch.zeros(d_model)),
                "ln2_w": nn.Parameter(torch.ones(d_model)),
                "ln2_b": nn.Parameter(torch.zeros(d_model)),
            })
            for _ in range(num_layers)
        ])
        for layer in self.layers:
            for name, p in layer.items():
                if "W_" in name:
                    nn.init.xavier_uniform_(p)

        self.ln_f_w = nn.Parameter(torch.ones(d_model))
        self.ln_f_b = nn.Parameter(torch.zeros(d_model))
        self.lm_head = nn.Parameter(torch.empty(d_model, vocab_size))
        nn.init.normal_(self.lm_head, std=0.02)

    def _ln(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def forward(self, input_ids):
        B, L = input_ids.size()
        x = self.token_emb[input_ids] + self.pos_emb[:L].unsqueeze(0)

        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            Q = torch.matmul(x, layer["W_q"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(x, layer["W_k"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(x, layer["W_v"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            head_out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
            attn_out = torch.matmul(head_out, layer["W_o"])
            x = self._ln(x + attn_out, layer["ln1_w"], layer["ln1_b"])

            ffn_out = torch.matmul(
                torch.relu(torch.matmul(x, layer["W_1"]) + layer["b_1"]),
                layer["W_2"]
            ) + layer["b_2"]
            x = self._ln(x + ffn_out, layer["ln2_w"], layer["ln2_b"])

        x = self._ln(x, self.ln_f_w, self.ln_f_b)
        logits = torch.matmul(x, self.lm_head)  # (B, L, vocab_size)
        return logits

    def loss(self, input_ids):
        logits = self.forward(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.vocab_size), targets.reshape(-1)
        )
```

&emsp;&emsp;若输入 `input_ids` 形状为 $(B,L)$，`logits` 形状为 $(B,L-1,V)$，`loss` 为标量。

---

#### 2.9.2 掩码语言建模

&emsp;&emsp;掩码语言建模（Masked Language Modeling, MLM）是 Encoder-only 模型（如 BERT）的预训练目标。它随机选择输入序列中的一部分 token 进行掩码，然后要求模型根据双向上下文预测这些被掩码的 token。设原始序列为 $x = (x_1, \dots, x_T)$，随机选择掩码位置集合 $\mathcal{M}$，将 $\mathcal{M}$ 中的 token 替换为特殊标记 `[MASK]`，得到损坏序列 $\tilde{x}$。训练目标为：

$$
\mathcal{L}_{\mathrm{MLM}} = \sum_{i \in \mathcal{M}} \log P(x_i \mid \tilde{x}; \theta)
$$

&emsp;&emsp;其中 $P(x_i \mid \tilde{x})$ 由模型在位置 $i$ 的输出 Softmax 给出。BERT 还采用了 80% 替换为 `[MASK]`、10% 替换为随机 token、10% 保持不变的策略，以缓解预训练与微调之间的不一致。掩码语言建模的优点是双向注意力使每个位置都能利用完整上下文，在理解类任务上表现优异；缺点是预训练时存在 `[MASK]` 标记而微调时没有，造成不一致，且无法直接用于自回归生成。

&emsp;&emsp;从维度视角看，掩码语言建模在 token 维上使用全连接扩散，每个位置可以看到所有其他位置。模型需要在被破坏的 token 维上恢复原始信息，特征维表示被映射到词表分布。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出掩码语言建模的裸实现，包含随机掩码和交叉熵损失：

```python
import math
import torch
import torch.nn as nn

class MaskedLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=512, mask_token_id=0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

        self.token_emb = nn.Parameter(torch.empty(vocab_size, d_model))
        self.pos_emb = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.token_emb, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "W_q": nn.Parameter(torch.empty(d_model, d_model)),
                "W_k": nn.Parameter(torch.empty(d_model, d_model)),
                "W_v": nn.Parameter(torch.empty(d_model, d_model)),
                "W_o": nn.Parameter(torch.empty(d_model, d_model)),
                "W_1": nn.Parameter(torch.empty(d_model, d_ff)),
                "b_1": nn.Parameter(torch.zeros(d_ff)),
                "W_2": nn.Parameter(torch.empty(d_ff, d_model)),
                "b_2": nn.Parameter(torch.zeros(d_model)),
                "ln1_w": nn.Parameter(torch.ones(d_model)),
                "ln1_b": nn.Parameter(torch.zeros(d_model)),
                "ln2_w": nn.Parameter(torch.ones(d_model)),
                "ln2_b": nn.Parameter(torch.zeros(d_model)),
            })
            for _ in range(num_layers)
        ])
        for layer in self.layers:
            for name, p in layer.items():
                if "W_" in name:
                    nn.init.xavier_uniform_(p)

        self.ln_f_w = nn.Parameter(torch.ones(d_model))
        self.ln_f_b = nn.Parameter(torch.zeros(d_model))
        self.lm_head = nn.Parameter(torch.empty(d_model, vocab_size))
        nn.init.normal_(self.lm_head, std=0.02)

    def _ln(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def forward(self, input_ids):
        B, L = input_ids.size()
        x = self.token_emb[input_ids] + self.pos_emb[:L].unsqueeze(0)

        for layer in self.layers:
            Q = torch.matmul(x, layer["W_q"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(x, layer["W_k"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(x, layer["W_v"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = torch.softmax(scores, dim=-1)
            head_out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
            attn_out = torch.matmul(head_out, layer["W_o"])
            x = self._ln(x + attn_out, layer["ln1_w"], layer["ln1_b"])

            ffn_out = torch.matmul(
                torch.relu(torch.matmul(x, layer["W_1"]) + layer["b_1"]),
                layer["W_2"]
            ) + layer["b_2"]
            x = self._ln(x + ffn_out, layer["ln2_w"], layer["ln2_b"])

        x = self._ln(x, self.ln_f_w, self.ln_f_b)
        logits = torch.matmul(x, self.lm_head)
        return logits

    def loss(self, input_ids, mask_prob=0.15):
        B, L = input_ids.size()
        # 随机选择掩码位置
        mask = torch.rand(B, L, device=input_ids.device) < mask_prob
        # 至少保留一个非掩码位置（简化处理）
        corrupted = input_ids.clone()
        corrupted[mask] = self.mask_token_id

        logits = self.forward(corrupted)
        # 只计算被掩码位置的损失
        loss = torch.nn.functional.cross_entropy(
            logits[mask], input_ids[mask]
        )
        return loss
```

&emsp;&emsp;若输入 `input_ids` 形状为 $(B,L)$，`loss` 为标量，仅在被掩码位置计算。

---

#### 2.9.3 Span Corruption

&emsp;&emsp;Span Corruption 是 T5 提出的预训练目标，属于降噪自编码的一种。它随机选择输入序列中的连续片段（span），用哨兵 token 替换这些片段，然后要求解码器自回归地恢复被替换的 span 内容。设原始序列为 $x$，随机选择若干 span，将每个 span 替换为唯一的哨兵 token，得到损坏输入 $\tilde{x}$。目标序列由所有被替换的 span 内容组成，用哨兵 token 分隔。训练目标为：

$$
\mathcal{L}_{\mathrm{Span}} = \sum_{j=1}^{S} \log P(\mathrm{span}_j \mid \tilde{x}, \mathrm{span}_{<j}; \theta)
$$

&emsp;&emsp;其中 $S$ 是 span 数量，$\mathrm{span}_j$ 是第 $j$ 个被替换的片段。T5 通常使用 15% 的 token 被替换，平均 span 长度为 3。Span Corruption 的优点是训练目标更接近生成任务，适合编码器-解码器架构，能学习更长片段的重建，且哨兵 token 使解码器知道需要生成多少个片段；缺点是需要编码器-解码器结构，训练目标比 MLM 复杂，且哨兵 token 的引入增加了词表特殊标记。

&emsp;&emsp;从维度视角看，Span Corruption 在 token 维上先破坏连续片段，再通过编码器双向建模损坏序列，解码器自回归恢复片段。这相当于在 token 维上做了一次局部信息的删除与重建。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 Span Corruption 的裸实现，包含 span 选择、损坏输入构建和编码器-解码器损失：

```python
import math
import torch
import torch.nn as nn

class SpanCorruption(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_len=512, sentinel_start=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.vocab_size = vocab_size
        self.sentinel_start = sentinel_start if sentinel_start is not None else vocab_size - 100

        self.token_emb = nn.Parameter(torch.empty(vocab_size, d_model))
        self.pos_emb = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.token_emb, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        def make_layer():
            return nn.ModuleDict({
                "W_q": nn.Parameter(torch.empty(d_model, d_model)),
                "W_k": nn.Parameter(torch.empty(d_model, d_model)),
                "W_v": nn.Parameter(torch.empty(d_model, d_model)),
                "W_o": nn.Parameter(torch.empty(d_model, d_model)),
                "W_cq": nn.Parameter(torch.empty(d_model, d_model)),
                "W_ck": nn.Parameter(torch.empty(d_model, d_model)),
                "W_cv": nn.Parameter(torch.empty(d_model, d_model)),
                "W_co": nn.Parameter(torch.empty(d_model, d_model)),
                "W_1": nn.Parameter(torch.empty(d_model, d_ff)),
                "b_1": nn.Parameter(torch.zeros(d_ff)),
                "W_2": nn.Parameter(torch.empty(d_ff, d_model)),
                "b_2": nn.Parameter(torch.zeros(d_model)),
                "ln1_w": nn.Parameter(torch.ones(d_model)),
                "ln1_b": nn.Parameter(torch.zeros(d_model)),
                "ln2_w": nn.Parameter(torch.ones(d_model)),
                "ln2_b": nn.Parameter(torch.zeros(d_model)),
                "ln3_w": nn.Parameter(torch.ones(d_model)),
                "ln3_b": nn.Parameter(torch.zeros(d_model)),
            })

        self.enc_layers = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([make_layer() for _ in range(num_layers)])

        for layer in list(self.enc_layers) + list(self.dec_layers):
            for name, p in layer.items():
                if "W_" in name:
                    nn.init.xavier_uniform_(p)

        self.ln_f_w = nn.Parameter(torch.ones(d_model))
        self.ln_f_b = nn.Parameter(torch.zeros(d_model))
        self.lm_head = nn.Parameter(torch.empty(d_model, vocab_size))
        nn.init.normal_(self.lm_head, std=0.02)

    def _ln(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def _attn(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    def forward(self, src_ids, tgt_ids):
        B, L_s = src_ids.size()
        _, L_t = tgt_ids.size()

        src = self.token_emb[src_ids] + self.pos_emb[:L_s].unsqueeze(0)
        tgt = self.token_emb[tgt_ids] + self.pos_emb[:L_t].unsqueeze(0)

        # 编码器
        enc = src
        for layer in self.enc_layers:
            Q = torch.matmul(enc, layer["W_q"]).view(B, L_s, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(enc, layer["W_k"]).view(B, L_s, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(enc, layer["W_v"]).view(B, L_s, self.num_heads, self.d_k).transpose(1, 2)
            attn_out = self._attn(Q, K, V)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_s, self.d_model)
            attn_out = torch.matmul(attn_out, layer["W_o"])
            enc = self._ln(enc + attn_out, layer["ln1_w"], layer["ln1_b"])

            ffn = torch.matmul(torch.relu(torch.matmul(enc, layer["W_1"]) + layer["b_1"]), layer["W_2"]) + layer["b_2"]
            enc = self._ln(enc + ffn, layer["ln2_w"], layer["ln2_b"])

        # 解码器
        dec = tgt
        causal_mask = torch.triu(torch.ones(L_t, L_t, device=dec.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        for layer in self.dec_layers:
            Q = torch.matmul(dec, layer["W_q"]).view(B, L_t, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(dec, layer["W_k"]).view(B, L_t, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(dec, layer["W_v"]).view(B, L_t, self.num_heads, self.d_k).transpose(1, 2)
            attn_out = self._attn(Q, K, V, causal_mask)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_t, self.d_model)
            attn_out = torch.matmul(attn_out, layer["W_o"])
            dec = self._ln(dec + attn_out, layer["ln1_w"], layer["ln1_b"])

            Q = torch.matmul(dec, layer["W_cq"]).view(B, L_t, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(enc, layer["W_ck"]).view(B, L_s, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(enc, layer["W_cv"]).view(B, L_s, self.num_heads, self.d_k).transpose(1, 2)
            cross_out = self._attn(Q, K, V)
            cross_out = cross_out.transpose(1, 2).contiguous().view(B, L_t, self.d_model)
            cross_out = torch.matmul(cross_out, layer["W_co"])
            dec = self._ln(dec + cross_out, layer["ln2_w"], layer["ln2_b"])

            ffn = torch.matmul(torch.relu(torch.matmul(dec, layer["W_1"]) + layer["b_1"]), layer["W_2"]) + layer["b_2"]
            dec = self._ln(dec + ffn, layer["ln3_w"], layer["ln3_b"])

        dec = self._ln(dec, self.ln_f_w, self.ln_f_b)
        logits = torch.matmul(dec, self.lm_head)
        return logits

    def build_corrupted(self, input_ids, span_len=3, corruption_rate=0.15):
        """构建损坏输入和目标序列（简化版）"""
        B, L = input_ids.size()
        corrupted = input_ids.clone()
        targets = []
        for b in range(B):
            seq = input_ids[b].tolist()
            i = 0
            tgt = []
            sentinel_id = self.sentinel_start
            while i < L:
                if torch.rand(1).item() < corruption_rate:
                    end = min(i + span_len, L)
                    tgt.append(sentinel_id)
                    tgt.extend(seq[i:end])
                    corrupted[b, i:end] = sentinel_id
                    sentinel_id += 1
                    i = end
                else:
                    i += 1
            tgt.append(self.sentinel_start + 99)  # EOS
            targets.append(tgt)
        max_tgt_len = max(len(t) for t in targets)
        tgt_tensor = torch.full((B, max_tgt_len), 0, dtype=torch.long, device=input_ids.device)
        for b, t in enumerate(targets):
            tgt_tensor[b, :len(t)] = torch.tensor(t, device=input_ids.device)
        return corrupted, tgt_tensor

    def loss(self, input_ids):
        corrupted, targets = self.build_corrupted(input_ids)
        logits = self.forward(corrupted, targets[:, :-1])
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.vocab_size), targets[:, 1:].reshape(-1)
        )
```

&emsp;&emsp;这个实现中，`build_corrupted` 随机选择 span 并用哨兵 token 替换，目标序列由哨兵 token 和原始 span 内容组成。`loss` 计算解码器交叉熵。

---

#### 2.9.4 多 Token 预测（MTP）

&emsp;&emsp;多 Token 预测（Multi-Token Prediction, MTP）是一种增强自回归语言建模的训练目标，要求模型在每个位置不仅预测下一个 token，还预测未来多个 token。设预测未来 $K$ 个 token，训练目标为：

$$
\mathcal{L}_{\mathrm{MTP}} = \sum_{t=1}^{T} \sum_{k=1}^{K} \log P(x_{t+k} \mid x_{\leq t}; \theta)
$$

&emsp;&emsp;实现上，模型可以在每个位置使用多个输出头，第 $k$ 个头预测偏移 $k$ 的 token。所有头共享主干表示，但各自有独立的输出投影。MTP 的优点是增加了训练信号的密度，每个位置提供 $K$ 个监督信号，提升了样本效率，且迫使模型学习更长程的依赖关系；缺点是计算量随 $K$ 线性增加，且未来 token 的预测可能比下一个 token 更难，需要平衡各头的损失权重。

&emsp;&emsp;从维度视角看，多 Token 预测在 token 维上将预测范围从下一个位置扩展到未来 $K$ 个位置，特征维上的表示需要同时编码多个未来 token 的信息。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出多 Token 预测的裸实现，使用多个输出头：

```python
import math
import torch
import torch.nn as nn

class MultiTokenLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_len=512, num_future=4):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.vocab_size = vocab_size
        self.num_future = num_future

        self.token_emb = nn.Parameter(torch.empty(vocab_size, d_model))
        self.pos_emb = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.token_emb, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "W_q": nn.Parameter(torch.empty(d_model, d_model)),
                "W_k": nn.Parameter(torch.empty(d_model, d_model)),
                "W_v": nn.Parameter(torch.empty(d_model, d_model)),
                "W_o": nn.Parameter(torch.empty(d_model, d_model)),
                "W_1": nn.Parameter(torch.empty(d_model, d_ff)),
                "b_1": nn.Parameter(torch.zeros(d_ff)),
                "W_2": nn.Parameter(torch.empty(d_ff, d_model)),
                "b_2": nn.Parameter(torch.zeros(d_model)),
                "ln1_w": nn.Parameter(torch.ones(d_model)),
                "ln1_b": nn.Parameter(torch.zeros(d_model)),
                "ln2_w": nn.Parameter(torch.ones(d_model)),
                "ln2_b": nn.Parameter(torch.zeros(d_model)),
            })
            for _ in range(num_layers)
        ])
        for layer in self.layers:
            for name, p in layer.items():
                if "W_" in name:
                    nn.init.xavier_uniform_(p)

        self.ln_f_w = nn.Parameter(torch.ones(d_model))
        self.ln_f_b = nn.Parameter(torch.zeros(d_model))

        # 多个输出头
        self.lm_heads = nn.ParameterList([
            nn.Parameter(torch.empty(d_model, vocab_size)) for _ in range(num_future)
        ])
        for head in self.lm_heads:
            nn.init.normal_(head, std=0.02)

    def _ln(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def forward(self, input_ids):
        B, L = input_ids.size()
        x = self.token_emb[input_ids] + self.pos_emb[:L].unsqueeze(0)

        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            Q = torch.matmul(x, layer["W_q"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(x, layer["W_k"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(x, layer["W_v"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            head_out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
            attn_out = torch.matmul(head_out, layer["W_o"])
            x = self._ln(x + attn_out, layer["ln1_w"], layer["ln1_b"])

            ffn_out = torch.matmul(
                torch.relu(torch.matmul(x, layer["W_1"]) + layer["b_1"]),
                layer["W_2"]
            ) + layer["b_2"]
            x = self._ln(x + ffn_out, layer["ln2_w"], layer["ln2_b"])

        x = self._ln(x, self.ln_f_w, self.ln_f_b)

        # 每个头预测不同偏移
        logits_list = []
        for k, head in enumerate(self.lm_heads):
            logits_list.append(torch.matmul(x, head))  # (B, L, vocab_size)
        return logits_list

    def loss(self, input_ids):
        logits_list = self.forward(input_ids)
        total_loss = 0.0
        for k, logits in enumerate(logits_list):
            offset = k + 1
            if input_ids.size(1) <= offset:
                continue
            # 预测 x_{t+offset}
            pred = logits[:, :-offset, :]
            target = input_ids[:, offset:]
            total_loss += torch.nn.functional.cross_entropy(
                pred.reshape(-1, self.vocab_size), target.reshape(-1)
            )
        return total_loss / len(logits_list)
```

&emsp;&emsp;这个实现中，每个输出头预测一个未来偏移的 token，损失为各头交叉熵的平均。若输入形状为 $(B,L)$，输出为多个 logits 列表，每个形状 $(B,L,V)$。

---

#### 2.9.5 降噪自编码

&emsp;&emsp;降噪自编码（Denoising Autoencoding, DAE）是一种通过向输入添加噪声并训练模型恢复原始输入的预训练目标。与 Span Corruption 类似，但噪声类型更灵活，可以包括 token 掩码、随机替换、删除、交换等。设原始序列为 $x$，噪声过程为 $\mathcal{N}$，损坏输入为 $\tilde{x} = \mathcal{N}(x)$，训练目标为最小化重构损失：

$$
\mathcal{L}_{\mathrm{DAE}} = -\sum_{t=1}^{T} \log P(x_t \mid \tilde{x}; \theta)
$$

&emsp;&emsp;降噪自编码的优点是模型学习到鲁棒的表示，能够从部分损坏的输入中恢复完整信息，适合预训练编码器或编码器-解码器模型；缺点是噪声类型和比例需要手动设计，不同任务下最优噪声策略不同，且重构目标可能使模型过度关注局部恢复而忽略全局语义。

&emsp;&emsp;从维度视角看，降噪自编码在 token 维上引入噪声，破坏部分信息，模型需要从损坏的 token 维中恢复原始表示。这相当于在特征维上学习一个去噪映射，使表示对输入扰动具有鲁棒性。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出降噪自编码的裸实现，使用随机替换和删除作为噪声：

```python
import math
import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_len=512, noise_prob=0.15):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.vocab_size = vocab_size
        self.noise_prob = noise_prob

        self.token_emb = nn.Parameter(torch.empty(vocab_size, d_model))
        self.pos_emb = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.token_emb, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.enc_layers = nn.ModuleList([
            nn.ModuleDict({
                "W_q": nn.Parameter(torch.empty(d_model, d_model)),
                "W_k": nn.Parameter(torch.empty(d_model, d_model)),
                "W_v": nn.Parameter(torch.empty(d_model, d_model)),
                "W_o": nn.Parameter(torch.empty(d_model, d_model)),
                "W_1": nn.Parameter(torch.empty(d_model, d_ff)),
                "b_1": nn.Parameter(torch.zeros(d_ff)),
                "W_2": nn.Parameter(torch.empty(d_ff, d_model)),
                "b_2": nn.Parameter(torch.zeros(d_model)),
                "ln1_w": nn.Parameter(torch.ones(d_model)),
                "ln1_b": nn.Parameter(torch.zeros(d_model)),
                "ln2_w": nn.Parameter(torch.ones(d_model)),
                "ln2_b": nn.Parameter(torch.zeros(d_model)),
            })
            for _ in range(num_layers)
        ])
        for layer in self.enc_layers:
            for name, p in layer.items():
                if "W_" in name:
                    nn.init.xavier_uniform_(p)

        self.ln_f_w = nn.Parameter(torch.ones(d_model))
        self.ln_f_b = nn.Parameter(torch.zeros(d_model))
        self.lm_head = nn.Parameter(torch.empty(d_model, vocab_size))
        nn.init.normal_(self.lm_head, std=0.02)

    def _ln(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def add_noise(self, input_ids):
        """随机替换和删除 token"""
        B, L = input_ids.size()
        corrupted = input_ids.clone()
        # 随机替换
        replace_mask = torch.rand(B, L, device=input_ids.device) < self.noise_prob
        random_tokens = torch.randint(0, self.vocab_size, (B, L), device=input_ids.device)
        corrupted[replace_mask] = random_tokens[replace_mask]
        # 随机删除（用 mask token 替代，简化处理）
        delete_mask = torch.rand(B, L, device=input_ids.device) < self.noise_prob / 2
        corrupted[delete_mask] = 0  # 假设 0 是 mask token
        return corrupted

    def forward(self, input_ids):
        B, L = input_ids.size()
        x = self.token_emb[input_ids] + self.pos_emb[:L].unsqueeze(0)

        for layer in self.enc_layers:
            Q = torch.matmul(x, layer["W_q"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(x, layer["W_k"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(x, layer["W_v"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = torch.softmax(scores, dim=-1)
            head_out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
            attn_out = torch.matmul(head_out, layer["W_o"])
            x = self._ln(x + attn_out, layer["ln1_w"], layer["ln1_b"])

            ffn_out = torch.matmul(
                torch.relu(torch.matmul(x, layer["W_1"]) + layer["b_1"]),
                layer["W_2"]
            ) + layer["b_2"]
            x = self._ln(x + ffn_out, layer["ln2_w"], layer["ln2_b"])

        x = self._ln(x, self.ln_f_w, self.ln_f_b)
        logits = torch.matmul(x, self.lm_head)
        return logits

    def loss(self, input_ids):
        corrupted = self.add_noise(input_ids)
        logits = self.forward(corrupted)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.vocab_size), input_ids.reshape(-1)
        )
```

&emsp;&emsp;这个实现中，`add_noise` 随机替换和删除 token，模型需要从损坏输入中恢复原始序列。若输入形状为 $(B,L)$，`loss` 为标量。


---

### 2.10 优化器与训练精度

#### 2.10.1 Adam

&emsp;&emsp;Adam（Adaptive Moment Estimation）由 Kingma 和 Ba 于 2015 年提出，是目前最广泛使用的自适应学习率优化器。它同时维护梯度的一阶矩估计（动量）和二阶矩估计（梯度平方的指数移动平均），并利用偏差校正使估计在训练初期无偏。

&emsp;&emsp;设第 $t$ 步的梯度为 $g_t = \nabla_\theta f_t(\theta_{t-1})$，Adam 的计算为：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

&emsp;&emsp;其中 $\beta_1$ 和 $\beta_2$ 是一阶和二阶矩的衰减率，通常取 $\beta_1=0.9$，$\beta_2=0.999$。由于 $m_0$ 和 $v_0$ 初始化为零，早期估计偏向零，需要进行偏差校正：

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

&emsp;&emsp;参数更新为：

$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

&emsp;&emsp;其中 $\eta$ 是学习率，$\epsilon$ 是防止除零的小常数，通常取 $10^{-8}$。Adam 的优点是自适应学习率使不同参数的更新幅度自动缩放，对超参数不敏感，收敛速度快，适合稀疏梯度和非平稳目标；缺点是二阶矩估计需要存储每个参数的梯度平方均值，显存开销是 SGD 的三倍，且 $\epsilon$ 在后期可能干扰收敛精度。

&emsp;&emsp;从维度视角看，Adam 在特征维上为每个参数维护独立的自适应学习率。一阶矩在特征维上做动量平滑，二阶矩在特征维上估计梯度尺度，两者配合使每个特征维的更新步长自适应调整。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 Adam 的裸实现：

```python
import torch
import torch.nn as nn

class Adam(nn.Module):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

&emsp;&emsp;这个实现中，一阶矩 `m` 和二阶矩 `v` 逐参数维护，偏差校正使用当前步数 `t`。

---

#### 2.10.2 AdamW

&emsp;&emsp;AdamW 由 Loshchilov 和 Hutter 于 2019 年提出，核心改进是将权重衰减从梯度更新中解耦。在标准 Adam 中，L2 正则化通过将 $\lambda \theta$ 加到梯度 $g_t$ 上来实现，这使得权重衰减与自适应学习率耦合：自适应学习率会缩放权重衰减的效果，导致不同参数的衰减强度不一致。AdamW 将权重衰减直接作用于参数本身，与梯度更新解耦：

$$
\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)
$$

&emsp;&emsp;其中 $\lambda$ 是权重衰减系数。在解耦形式中，权重衰减项 $\lambda \theta_{t-1}$ 不经过 $1/\sqrt{\hat{v}_t}$ 的缩放，因此所有参数以相同比例衰减。权重衰减项按 $\lambda \eta$ 缩放，其中 $\eta$ 是学习率。这意味着当使用学习率调度时，权重衰减的实际强度会随学习率变化，通常在训练后期学习率降低时，权重衰减的绝对幅度也相应减小。

&emsp;&emsp;AdamW 的优点是解耦权重衰减使正则化效果与自适应学习率独立，在 Transformer 训练中通常优于 Adam，且与学习率调度配合更自然；缺点是与 Adam 相比需要额外调优 $\lambda$，且在某些小模型或短训练任务上优势不明显。

&emsp;&emsp;从维度视角看，AdamW 在特征维上分别处理梯度更新和权重衰减：梯度更新沿特征维做自适应缩放，权重衰减沿特征维做均匀收缩。两者解耦后，特征维上的每个参数可以独立地平衡拟合与正则化。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 AdamW 的裸实现：

```python
import torch
import torch.nn as nn

class AdamW(nn.Module):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.01):
        super().__init__()
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # 解耦权重衰减: 先衰减参数，再做梯度更新
            p.data -= self.lr * (
                m_hat / (torch.sqrt(v_hat) + self.eps)
                + self.weight_decay * p.data
            )

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

&emsp;&emsp;这个实现中，权重衰减项 `weight_decay * p.data` 与梯度更新项相加后统一乘以学习率 `lr`，实现了 AdamW 的解耦权重衰减。

---

#### 2.10.3 Muon

&emsp;&emsp;Muon（MomentUm Orthogonalized by Newton-schulz）由 Keller Jordan 于 2024 年提出，是一种专门针对矩阵参数（如线性层的权重矩阵）的优化器。其核心思想是对动量矩阵进行正交化，使更新方向在谱范数下具有最优性。与 Adam 逐元素处理梯度不同，Muon 将整个权重矩阵作为一个整体，利用矩阵的谱结构来指导更新。

&emsp;&emsp;Muon 的算法流程为：首先对梯度矩阵进行标准 SGD 动量累积：

$$
M_t = \beta M_{t-1} + (1-\beta) G_t
$$

&emsp;&emsp;其中 $G_t$ 是第 $t$ 步的梯度矩阵。然后对动量矩阵 $M_t$ 进行正交化：

$$
O_t = \mathrm{Ortho}(M_t)
$$

&emsp;&emsp;正交化通过 Newton-Schulz 迭代近似计算。设 $M \in \mathbb{R}^{m \times n}$ 的奇异值分解为 $M = U\Sigma V^\top$，其正交化结果为 $O = UV^\top$。实际中 Newton-Schulz 迭代用矩阵乘法近似这个正交极因子：

$$
X_{k+1} = X_k \left( \frac{3}{2} I - \frac{1}{2} X_k^\top X_k \right)
$$

&emsp;&emsp;通常只需 5 步迭代即可达到足够精度。收敛性分析表明，Muon 使用 Newton-Schulz 的收敛速率与精确 SVD 极因子分解相同，误差随迭代步数 $q$ 双指数收敛到 1。参数更新为：

$$
W_t = W_{t-1} - \eta \cdot O_t
$$

&emsp;&emsp;Muon 的优点是矩阵正交化更新利用了权重矩阵的谱结构，在 LLM 预训练中比 AdamW 具有更高的 token 效率，且 Newton-Schulz 迭代只需矩阵乘法，硬件效率高；缺点是仅适用于二维矩阵参数，偏置和 LayerNorm 参数仍需用 Adam 处理，且正交化在极端规模下可能出现数值不稳定。

&emsp;&emsp;从维度视角看，Muon 在特征维上利用了矩阵的奇异值结构：正交化使更新方向的奇异值全部为 1，相当于在谱范数约束下做最速下降。这比 Adam 的逐元素自适应缩放更能捕捉矩阵参数的整体几何结构。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 Muon 的裸实现，包含 SGD 动量和 Newton-Schulz 正交化：

```python
import torch
import torch.nn as nn

class Muon(nn.Module):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        super().__init__()
        self.params = [p for p in params if p.dim() == 2]
        self.lr = lr
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.m = [torch.zeros_like(p) for p in self.params]

    def _newton_schulz(self, M):
        """Newton-Schulz 迭代近似正交化"""
        X = M / (M.norm() + 1e-8)  # 归一化
        for _ in range(self.ns_steps):
            A = X.transpose(-2, -1) @ X
            X = X @ (1.5 * torch.eye(A.size(-1), device=A.device) - 0.5 * A)
        return X

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self.m[i] = self.momentum * self.m[i] + (1 - self.momentum) * g
            O = self._newton_schulz(self.m[i])
            p.data -= self.lr * O

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

&emsp;&emsp;这个实现中，`_newton_schulz` 通过迭代矩阵乘法近似正交化动量矩阵，无需 SVD。注意 Muon 仅作用于二维矩阵参数，偏置和一维参数需用 Adam 单独处理。

---

#### 2.10.4 MuonClip

&emsp;&emsp;MuonClip 由 Kimi 团队在 Kimi K2 训练中提出，是 Muon 的稳定性增强版本。Kimi 团队在将 Muon 扩展到万亿参数规模时发现，Muon 的正交化更新会导致注意力 logits 爆炸，进而引起模型发散。MuonClip 通过引入 QK-Clip 机制解决了这一问题。

&emsp;&emsp;QK-Clip 的核心思想是监控注意力中 Query 和 Key 的点积幅度，当 logits 超过预设阈值时，对 Q 和 K 的投影矩阵进行缩放：

$$
\mathrm{logits} = QK^\top, \quad \text{if } \|\mathrm{logits}\|_\infty > \tau: \quad Q \leftarrow Q \cdot \sqrt{\frac{\tau}{\|\mathrm{logits}\|_\infty}}
$$

&emsp;&emsp;其中 $\tau$ 是 logits 的阈值。这种缩放直接作用于注意力计算的 Q 和 K，不影响 Muon 的矩阵正交化更新。MuonClip 在 Kimi K2 的 15.5 万亿 token 预训练中实现了零损失尖峰，同时保持了 Muon 的 token 效率优势，计算效率是传统 AdamW 的 2 倍。

&emsp;&emsp;MuonClip 的优点是解决了 Muon 在大规模训练中的 logits 爆炸问题，使万亿参数级训练稳定，同时保持了 Muon 的 token 效率；缺点是引入了额外的 QK 监控和缩放计算，且阈值 $\tau$ 需要根据模型规模调整。

&emsp;&emsp;从维度视角看，MuonClip 在 Muon 的谱范数正交化基础上，增加了对注意力 logits 的幅度约束。这相当于在特征维上同时约束了权重矩阵的谱范数和注意力输出的数值范围，双重保障训练稳定。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 MuonClip 的裸实现，包含 Muon 优化和 QK-Clip：

```python
import torch
import torch.nn as nn

class MuonClip(nn.Module):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5,
                 qk_clip_threshold=30.0):
        super().__init__()
        self.params = [p for p in params if p.dim() == 2]
        self.lr = lr
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.qk_clip_threshold = qk_clip_threshold
        self.m = [torch.zeros_like(p) for p in self.params]

    def _newton_schulz(self, M):
        X = M / (M.norm() + 1e-8)
        for _ in range(self.ns_steps):
            A = X.transpose(-2, -1) @ X
            X = X @ (1.5 * torch.eye(A.size(-1), device=A.device) - 0.5 * A)
        return X

    def qk_clip(self, Q, K):
        """QK-Clip: 当 logits 超过阈值时缩放 Q 和 K"""
        logits = Q @ K.transpose(-2, -1)
        max_logit = logits.abs().max()
        if max_logit > self.qk_clip_threshold:
            scale = torch.sqrt(
                self.qk_clip_threshold / (max_logit + 1e-8)
            )
            Q = Q * scale
            K = K * scale
        return Q, K

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self.m[i] = self.momentum * self.m[i] + (1 - self.momentum) * g
            O = self._newton_schulz(self.m[i])
            p.data -= self.lr * O

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

&emsp;&emsp;这个实现中，`qk_clip` 在注意力计算前检查 logits 幅度，超过阈值时缩放 Q 和 K。实际使用中需在注意力层中调用 `qk_clip`。

---

#### 2.10.5 L-BFGS

&emsp;&emsp;L-BFGS（Limited-memory BFGS）是一种拟牛顿优化方法，通过近似 Hessian 矩阵的逆来加速收敛。与 BFGS 需要存储 $n \times n$ 的完整 Hessian 近似不同，L-BFGS 只存储最近 $m$ 步的位移向量 $s_k = \theta_{k+1} - \theta_k$ 和梯度变化 $y_k = g_{k+1} - g_k$，利用两循环递归计算 Hessian 逆与梯度的乘积，将存储从 $O(n^2)$ 降到 $O(mn)$。

&emsp;&emsp;两循环递归的计算过程为：给定当前梯度 $g_k$ 和历史对 $\{(s_i, y_i)\}_{i=k-m}^{k-1}$，首先向前循环计算中间变量 $\alpha_i$，然后向后循环累积更新方向。设初始 Hessian 逆近似为 $H_k^0 = \frac{s_{k-1}^\top y_{k-1}}{y_{k-1}^\top y_{k-1}} I$，两循环递归的计算为：

$$
q = g_k
$$

$$
\text{for } i = k-1, \dots, k-m: \quad \alpha_i = \rho_i s_i^\top q, \quad q = q - \alpha_i y_i
$$

$$
r = H_k^0 q
$$

$$
\text{for } i = k-m, \dots, k-1: \quad \beta = \rho_i y_i^\top r, \quad r = r + s_i(\alpha_i - \beta)
$$

&emsp;&emsp;其中 $\rho_i = 1/(y_i^\top s_i)$。最终更新方向为 $-r$。L-BFGS 的优点是收敛速度快，在光滑目标函数上具有超线性收敛，无需手动设置学习率（通过线搜索确定步长），适合中小规模参数的精细调优；缺点是内存和计算开销随历史步数 $m$ 线性增长，对随机梯度的噪声敏感，不适合大规模深度学习训练中常见的非凸、随机优化场景。

&emsp;&emsp;从维度视角看，L-BFGS 在特征维上利用历史梯度信息构建曲率近似，通过两循环递归隐式地沿特征维做二次型缩放。它不逐维维护自适应学习率，而是捕捉特征维之间的相关性。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 L-BFGS 两循环递归的裸实现：

```python
import torch
import torch.nn as nn

class LBFGS(nn.Module):
    def __init__(self, params, lr=1.0, history_size=10):
        super().__init__()
        self.params = list(params)
        self.lr = lr
        self.history_size = history_size
        self.s_history = []  # s_k = theta_{k+1} - theta_k
        self.y_history = []  # y_k = g_{k+1} - g_k
        self.rho_history = []  # 1/(y^T s)

    def _two_loop_recursion(self, grad):
        """L-BFGS 两循环递归"""
        q = grad.clone()
        alphas = []
        # 向前循环
        for s, y, rho in zip(reversed(self.s_history),
                              reversed(self.y_history),
                              reversed(self.rho_history)):
            alpha = rho * torch.sum(s * q)
            alphas.append(alpha)
            q = q - alpha * y

        # 初始 Hessian 逆近似: H0 = (s^T y) / (y^T y) I
        if len(self.s_history) > 0:
            s_last = self.s_history[-1]
            y_last = self.y_history[-1]
            gamma = torch.sum(s_last * y_last) / (
                torch.sum(y_last * y_last) + 1e-10
            )
        else:
            gamma = 1.0

        r = gamma * q

        # 向后循环
        for s, y, alpha in zip(self.s_history,
                                self.y_history,
                                reversed(alphas)):
            beta = self.rho_history[self.s_history.index(s)] * torch.sum(y * r)
            r = r + s * (alpha - beta)

        return r

    def step(self, closure=None):
        """closure 返回 loss，并计算梯度"""
        loss = closure()
        grad = torch.cat([p.grad.data.view(-1) for p in self.params
                          if p.grad is not None])
        theta = torch.cat([p.data.view(-1) for p in self.params
                           if p.grad is not None])

        if len(self.s_history) > 0:
            direction = self._two_loop_recursion(grad)
            # 线搜索确定步长（简化: 固定学习率）
            new_theta = theta - self.lr * direction
        else:
            new_theta = theta - self.lr * grad

        # 更新参数
        offset = 0
        for p in self.params:
            if p.grad is not None:
                numel = p.numel()
                p.data.copy_(new_theta[offset:offset+numel].view_as(p))
                offset += numel

        return loss

    def update_history(self, s, y):
        """在参数更新后调用，记录 s 和 y"""
        if torch.sum(y * s) > 1e-10:
            self.s_history.append(s)
            self.y_history.append(y)
            self.rho_history.append(1.0 / torch.sum(y * s))
            if len(self.s_history) > self.history_size:
                self.s_history.pop(0)
                self.y_history.pop(0)
                self.rho_history.pop(0)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

&emsp;&emsp;这个实现中，`_two_loop_recursion` 通过向前和向后两个循环计算 Hessian 逆与梯度的乘积。实际使用中需要配合线搜索确定步长。

---

#### 2.10.6 FP8 混合精度训练

&emsp;&emsp;FP8 混合精度训练使用 8 位浮点数进行矩阵乘法和部分激活值存储，同时保留 FP32 或 BF16 的主权重副本。FP8 有两种格式：E4M3 和 E5M2。E4M3 使用 4 位指数和 3 位尾数，精度更高但动态范围有限，范围约 $\pm 448$；E5M2 使用 5 位指数和 2 位尾数，动态范围更宽，约 $\pm 57344$，但精度更低。实践中通常采用混合格式：前向传播的激活和权重使用 E4M3，反向传播的梯度使用 E5M2。

&emsp;&emsp;FP8 训练的核心挑战是张量的动态范围可能超出 FP8 的表示范围，需要通过缩放因子将张量归一化到 FP8 可表示的区间。常见的缩放策略包括三种。逐张量缩放（Per-Tensor Scaling）为整个张量计算一个缩放因子，使用张量的绝对最大值进行归一化，实现简单但精度有限。分块缩放（Blockwise FP8）将激活和梯度按 $128\times 128$ 的瓦片量化，权重按 $1\times 128$ 的块量化，采用 E4M3 格式，在 Hopper 平台上推荐使用，已在 DeepSeek-V3 等大规模 MoE 模型中验证有效。MXFP8 在 Blackwell 平台上使用 $1\times 32$ 的更细粒度量化，缩放因子使用 E8M0 格式，由第五代 Tensor Core 原生支持。

&emsp;&emsp;FP8 混合精度训练的优点是相比 BF16 矩阵乘法吞吐量翻倍，显存占用和带宽需求降低约一半，且在大规模模型中已被验证可以保持与 BF16 相当的收敛性；缺点是 FP8 的动态范围有限，需要精细的缩放策略，数值不稳定风险较高，且不同硬件平台的 FP8 支持程度和最优配方不同。

&emsp;&emsp;从维度视角看，FP8 混合精度在特征维上以更低的位宽表示张量，通过缩放因子将特征维的数值范围压缩到 FP8 的可表示区间。这相当于在特征维上做了一次有损压缩，但通过保留 FP32 主权重和精细的缩放策略，将精度损失控制在可接受范围内。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 FP8 混合精度训练核心逻辑的模拟实现：

```python
import torch
import torch.nn as nn

class FP8Linear(nn.Module):
    """FP8 混合精度线性层模拟: 权重 FP8，计算 FP32，主权重 FP32"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # 主权重副本（FP32）
        self.register_buffer("weight_master", self.weight.data.clone())

    def _quantize_fp8(self, x, scale):
        """将张量量化为 FP8 模拟格式并反量化"""
        x_scaled = x / scale
        x_clamped = torch.clamp(x_scaled, -448.0, 448.0)  # E4M3 范围
        # 模拟 FP8 精度: 量化到 3 位尾数
        x_fp8 = x_clamped * scale
        return x_fp8

    def _compute_scale(self, x):
        """逐张量缩放: amax 归一化"""
        amax = x.abs().max()
        return amax / 448.0 if amax > 0 else torch.tensor(1.0, device=x.device)

    def forward(self, x):
        # 量化权重
        w_scale = self._compute_scale(self.weight)
        w_fp8 = self._quantize_fp8(self.weight, w_scale)

        # 量化激活
        x_scale = self._compute_scale(x)
        x_fp8 = self._quantize_fp8(x, x_scale)

        # FP8 矩阵乘法（模拟: 反量化后做 FP32 乘法）
        out = torch.matmul(x_fp8, w_fp8.T)
        return out
```

&emsp;&emsp;这个实现模拟了 FP8 的量化-反量化流程。实际中需使用 Tensor Core 原生 FP8 支持（如 Transformer Engine）获得加速。

---

#### 2.10.7 BF16 与 INT4 量化

&emsp;&emsp;BF16（Brain Floating Point 16）和 INT4 是两种不同定位的低精度格式。BF16 用于训练，INT4 用于推理量化。

&emsp;&emsp;BF16 使用 8 位指数和 7 位尾数，与 FP32 具有相同的指数范围，因此不会出现梯度上溢或下溢，通常不需要损失缩放。FP16 使用 5 位指数和 10 位尾数，精度更高但动态范围窄，训练时需要损失缩放来防止梯度下溢。BF16 的优点是训练稳定性好，无需损失缩放，与 FP32 的切换几乎无损，在 A100 及以上 GPU 上原生支持；缺点是尾数精度低于 FP16，在某些对精度敏感的任务上可能需要 FP32 回退。

&emsp;&emsp;INT4 量化将权重从 16 位压缩到 4 位，模型大小减少约 75%，使 70B 模型可以在单张消费级 GPU 上运行。GPTQ 和 AWQ 是两种主流的 INT4 训练后量化方法。GPTQ 基于逐层重构，使用 Hessian 矩阵的二阶信息来最小化量化误差，在真实任务上通常优于 AWQ。AWQ 基于激活感知的权重缩放，通过保护重要权重通道来减少量化损失，在多语言检索任务上对低资源语言的保护更好。INT4 量化通常只量化权重（W4A16），激活保持 FP16 或 BF16，因为激活的异常值对量化更敏感。DeepSeek-V4 Flash 在 INT4 量化后可在单张 80GB H100 上运行，质量损失约 5%。

&emsp;&emsp;BF16 与 INT4 量化的优点是 BF16 使大规模训练无需损失缩放，INT4 使大模型可以在消费级硬件上部署，两者配合实现了从训练到推理的完整低精度方案；缺点是 INT4 量化对激活异常值敏感，需要 GPTQ 或 AWQ 等算法精心处理，且量化后的模型在长上下文和低资源语言任务上性能下降更明显。

&emsp;&emsp;从维度视角看，BF16 在训练时以 16 位精度表示特征维，保留了 FP32 的动态范围但降低了精度；INT4 在推理时将权重压缩到 4 位，在特征维上以极低精度存储权重，通过缩放因子和分组量化来最小化信息损失。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 BF16 训练和 INT4 量化推理的模拟实现：

```python
import torch
import torch.nn as nn

class BF16Linear(nn.Module):
    """BF16 混合精度线性层: 权重 BF16，计算 FP32"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # 权重转为 BF16，计算时转回 FP32
        w_bf16 = self.weight.to(torch.bfloat16).to(torch.float32)
        x_bf16 = x.to(torch.bfloat16).to(torch.float32)
        return torch.matmul(x_bf16, w_bf16.T)


class INT4Linear(nn.Module):
    """INT4 量化线性层模拟: 权重 4 位，分组量化"""
    def __init__(self, in_features, out_features, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("q_weight", None)
        self.register_buffer("scales", None)
        self.register_buffer("zeros", None)

    def quantize(self):
        """分组 INT4 量化"""
        w = self.weight.data
        out_f, in_f = w.shape
        num_groups = in_f // self.group_size

        w_grouped = w.view(out_f, num_groups, self.group_size)
        # 每组计算 min/max
        w_min = w_grouped.min(dim=-1, keepdim=True).values
        w_max = w_grouped.max(dim=-1, keepdim=True).values

        # 非对称量化: scale = (max - min) / 15, zero = round(-min / scale)
        scale = (w_max - w_min) / 15.0
        scale = torch.clamp(scale, min=1e-8)
        zero = torch.round(-w_min / scale).clamp(0, 15)

        # 量化到 [0, 15]
        q = torch.round(w_grouped / scale + zero).clamp(0, 15)

        self.q_weight = q.to(torch.uint8)
        self.scales = scale
        self.zeros = zero

    def forward(self, x):
        if self.q_weight is None:
            self.quantize()
        # 反量化
        w_deq = (self.q_weight.float() - self.zeros) * self.scales
        w_deq = w_deq.view(self.out_features, self.in_features)
        return torch.matmul(x, w_deq.T)
```

&emsp;&emsp;这个实现中，`BF16Linear` 模拟 BF16 训练的前向计算，`INT4Linear` 使用分组非对称量化将权重压缩到 4 位，推理时反量化回 FP32 进行计算。


---

### 2.11 后训练与对齐

#### 2.11.1 监督微调（SFT）

&emsp;&emsp;监督微调（Supervised Fine-Tuning, SFT）是在预训练模型基础上，使用高质量的输入-输出对进行有监督训练，使模型学会遵循指令、完成特定任务。设训练数据为 $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{N}$，其中 $x^{(i)}$ 是输入指令，$y^{(i)}$ 是目标回答。SFT 的损失为标准自回归交叉熵，但通常只对回答部分计算损失：

$$
\mathcal{L}_{\mathrm{SFT}} = -\sum_{i=1}^{N} \sum_{t=1}^{|y^{(i)}|} \log P(y_t^{(i)} \mid x^{(i)}, y_{<t}^{(i)}; \theta)
$$

&emsp;&emsp;输入部分的 token 被掩码，不参与损失计算。SFT 的优点是训练目标明确、稳定，直接教模型“什么样的回答是好的”，且实现简单，只需标准的自回归训练流程；缺点是依赖高质量标注数据，标注成本高，容易过拟合到训练数据的风格，且无法学习到训练数据中未出现的更优回答。从维度视角看，SFT 在 token 维上做因果扩散，特征维上的表示被调整以匹配目标回答的分布，相当于在预训练表示的基础上做了一次有监督的定向微调。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn as nn

def sft_loss(logits, labels, prompt_mask):
    """
    logits: (B, L, V)
    labels: (B, L)
    prompt_mask: (B, L) 1 表示回答部分，0 表示输入部分
    """
    B, L, V = logits.size()
    logits_flat = logits[:, :-1].reshape(-1, V)
    labels_flat = labels[:, 1:].reshape(-1)
    mask_flat = prompt_mask[:, 1:].reshape(-1)

    loss = torch.nn.functional.cross_entropy(
        logits_flat, labels_flat, reduction='none'
    )
    loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
    return loss
```

&emsp;&emsp;若输入形状为 $(B,L)$，损失仅对回答部分的 token 计算。

---

#### 2.11.2 RLHF

&emsp;&emsp;RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）是一种将人类偏好引入语言模型训练的方法，最早由 Christiano 等人于 2017 年提出，后在 InstructGPT 中被系统化。RLHF 的流程分为三个阶段：第一阶段是 SFT，用人工标注的示范数据微调预训练模型；第二阶段是训练奖励模型，用人类对模型输出的偏好排序数据训练一个打分模型；第三阶段是用强化学习算法（通常是 PPO）优化策略模型，使其输出获得更高的奖励。

&emsp;&emsp;第三阶段的优化目标为：

$$
\mathcal{L}_{\mathrm{RLHF}} = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \cdot \mathrm{KL}\left(\pi_\theta(y|x) \| \pi_{\mathrm{ref}}(y|x)\right) \right]
$$

&emsp;&emsp;其中 $r_\phi$ 是奖励模型，$\pi_{\mathrm{ref}}$ 是 SFT 后的参考模型，$\beta$ 是 KL 惩罚系数。KL 项防止策略偏离参考模型太远，避免奖励黑客。RLHF 的优点是直接优化人类偏好，能显著提升模型的有用性和安全性；缺点是流程复杂，需要训练多个模型，PPO 训练不稳定，且奖励模型可能被过优化。

&emsp;&emsp;从维度视角看，RLHF 在特征维上通过奖励信号调整策略模型的输出分布，KL 惩罚约束了分布偏移的幅度，相当于在参考模型的特征空间中做有约束的定向优化。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn as nn

def rlhf_loss(policy_logprobs, ref_logprobs, rewards, beta=0.1):
    """
    policy_logprobs: (B, L) 策略模型的对数概率
    ref_logprobs: (B, L) 参考模型的对数概率
    rewards: (B,) 序列级奖励
    """
    # KL 惩罚: 逐 token 的 log 概率差
    kl = policy_logprobs - ref_logprobs
    # 序列级 KL
    kl_seq = kl.sum(dim=-1)
    # RLHF 目标: 最大化 reward - beta * KL
    loss = -(rewards - beta * kl_seq).mean()
    return loss
```

---

#### 2.11.3 奖励模型（RM）

&emsp;&emsp;奖励模型（Reward Model, RM）是 RLHF 中的关键组件，用于对模型输出进行打分，代替人类偏好。奖励模型通常在 SFT 模型的基础上，将最后的语言模型头替换为标量输出头，输入 (prompt, response) 对，输出一个实数奖励。训练数据为人类对同一 prompt 的多个回答的偏好排序。

&emsp;&emsp;常用的奖励模型训练目标是 Bradley-Terry 模型，对于一对回答 $y_w$（更优）和 $y_l$（更差），偏好概率为：

$$
P(y_w \succ y_l \mid x) = \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

&emsp;&emsp;训练损失为负对数似然：

$$
\mathcal{L}_{\mathrm{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

&emsp;&emsp;奖励模型的优点是提供了可扩展的偏好信号，使 RL 优化可以大规模进行，且相比直接让人类参与训练循环，成本大幅降低；缺点是奖励模型容易被过优化，模型可能找到获得高奖励但实际质量差的输出，即奖励黑客，且偏好数据存在噪声和标注者偏差。

&emsp;&emsp;从维度视角看，奖励模型在特征维上将序列表示映射为一个标量，通过比较两个回答的奖励差来学习偏好。这相当于在特征维上学习一个排序函数。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, backbone, d_model):
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Parameter(torch.empty(d_model, 1))
        nn.init.normal_(self.reward_head, std=0.02)

    def forward(self, input_ids):
        hidden = self.backbone(input_ids)  # (B, L, d_model)
        # 取最后一个 token 的表示
        last_hidden = hidden[:, -1, :]  # (B, d_model)
        reward = torch.matmul(last_hidden, self.reward_head).squeeze(-1)
        return reward

def rm_loss(reward_chosen, reward_rejected):
    """Bradley-Terry 损失"""
    return -torch.nn.functional.logsigmoid(
        reward_chosen - reward_rejected
    ).mean()
```

---

#### 2.11.4 PPO

&emsp;&emsp;PPO（Proximal Policy Optimization，近端策略优化）是 RLHF 第三阶段使用的强化学习算法，由 Schulman 等人于 2017 年提出。PPO 的核心思想是限制每次策略更新的幅度，避免策略更新过大导致训练崩溃。PPO 使用裁剪的替代目标函数：

$$
\mathcal{L}_{\mathrm{PPO}} = \mathbb{E}_t \left[ \min\left( \rho_t \hat{A}_t, \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

&emsp;&emsp;其中 $\rho_t = \pi_\theta(a_t|s_t) / \pi_{\theta_{\mathrm{old}}}(a_t|s_t)$ 是重要性采样比率，$\hat{A}_t$ 是优势函数估计，$\epsilon$ 是裁剪范围，通常取 0.2。在 RLHF 中，动作是每个 token，状态是 prompt 和已生成的 token，奖励由奖励模型给出，优势函数通常用 GAE（Generalized Advantage Estimation）计算。

&emsp;&emsp;PPO 的优点是训练稳定，裁剪机制防止策略更新过大，且可以直接优化奖励信号；缺点是需要同时维护策略模型、参考模型、奖励模型和值函数模型，显存开销大，超参数敏感，训练速度慢。从维度视角看，PPO 在特征维上通过裁剪的重要性采样比率约束策略更新的幅度，优势函数在 token 维上传播奖励信号。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn as nn

def ppo_loss(logprobs, old_logprobs, advantages, clip_epsilon=0.2):
    """
    logprobs: (B, L) 当前策略的对数概率
    old_logprobs: (B, L) 旧策略的对数概率
    advantages: (B, L) 优势函数
    """
    ratio = torch.exp(logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    return policy_loss
```

---

#### 2.11.5 DPO

&emsp;&emsp;DPO（Direct Preference Optimization，直接偏好优化）由 Rafailov 等人于 2023 年提出，其核心洞察是：RLHF 中带 KL 约束的奖励最大化问题存在闭式最优解，该解可以用策略模型和参考模型的对数概率比表示：

$$
r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta \log Z(x)
$$

&emsp;&emsp;将这一关系代入 Bradley-Terry 偏好模型，奖励模型被消去，得到直接基于偏好数据的损失：

$$
\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} \right)
$$

&emsp;&emsp;DPO 的优点是无需训练奖励模型，也无需在线采样，直接在偏好数据上优化策略，训练稳定且实现简单，计算成本远低于 PPO；缺点是 DPO 是离线方法，无法像 PPO 那样在线探索，且对偏好数据的质量依赖更强。从维度视角看，DPO 在特征维上直接比较优选和次选回答的对数概率比，通过参考模型的比值约束来调整策略分布。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    各参数形状: (B,)
    """
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_ratio - rejected_ratio)
    loss = -F.logsigmoid(logits).mean()
    return loss
```

---

#### 2.11.6 GRPO

&emsp;&emsp;GRPO（Group Relative Policy Optimization，组相对策略优化）由 DeepSeek 团队于 2024 年提出，用于 DeepSeekMath 和 DeepSeek-R1 的训练。GRPO 的核心改进是去掉了 PPO 中的值函数模型，改用同一 prompt 下多个采样回答的组内相对奖励作为优势估计。对于每个 prompt $x$，采样 $G$ 个回答 $\{y_1, \dots, y_G\}$，每个回答获得奖励 $r_i$，则第 $i$ 个回答的优势为：

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_1, \dots, r_G\})}{\mathrm{std}(\{r_1, \dots, r_G\})}
$$

&emsp;&emsp;GRPO 的损失为：

$$
\mathcal{L}_{\mathrm{GRPO}} = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( \rho_i \hat{A}_i, \mathrm{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}}) \right]
$$

&emsp;&emsp;GRPO 的优点是省去了值函数模型，显存占用大幅降低，组内相对奖励自然实现了基线估计，无需额外训练 Critic，且与奖励模型的配合更简单；缺点是组内采样数量 $G$ 需要权衡，$G$ 太小优势估计方差大，$G$ 太大采样成本高。从维度视角看，GRPO 在特征维上通过组内归一化计算优势，去掉了值函数这一额外的特征维映射，直接用组内奖励的统计量作为基线。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn.functional as F

def grpo_loss(logprobs, old_logprobs, rewards, ref_logprobs,
              clip_epsilon=0.2, beta=0.01):
    """
    logprobs: (G, L) 当前策略对数概率
    old_logprobs: (G, L) 旧策略对数概率
    rewards: (G,) 组内奖励
    ref_logprobs: (G, L) 参考模型对数概率
    """
    # 组内相对优势
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r  # (G,)
    advantages = advantages.unsqueeze(-1)  # (G, 1)

    ratio = torch.exp(logprobs - old_logprobs)  # (G, L)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # KL 惩罚
    kl = (logprobs - ref_logprobs).mean()
    return policy_loss + beta * kl
```

#### 2.11.7 拒绝采样

&emsp;&emsp;拒绝采样（Rejection Sampling）在语言模型对齐中是一种简单而有效的数据筛选方法。其基本流程是：对每个提示 $x$，从当前策略模型 $\pi_\theta$ 中采样 $K$ 个候选回答 $\{y_1, \dots, y_K\}$，然后用奖励模型或规则对每个回答打分，只保留得分最高的一个或若干个回答，作为新的监督数据用于 SFT 或偏好学习。形式化地，选中的回答为：

$$
y^* = \arg\max_{y_i \sim \pi_\theta(\cdot|x)} r_\phi(x, y_i)
$$

&emsp;&emsp;拒绝采样的优点是实现简单，无需修改训练目标，仅通过筛选高质量样本就能提升数据质量，且可以与 SFT、DPO 等方法无缝结合；缺点是采样成本随 $K$ 线性增长，且只保留最高分样本会导致模式崩溃，模型可能失去多样性，同时奖励模型的偏差会被放大。从维度视角看，拒绝采样在 token 维上从同一提示生成多条路径，再根据奖励在特征维上的投影选择最优路径，相当于在输出空间中做了一次基于奖励的离散筛选。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn.functional as F

def rejection_sampling(policy_logits, reward_fn, num_samples=4):
    """
    policy_logits: (B, L, V) 策略模型输出
    reward_fn: 接受 (B, L, V) 返回 (B, num_samples) 奖励
    """
    B, L, V = policy_logits.size()
    samples = []
    rewards = []
    for _ in range(num_samples):
        # 从策略分布中采样
        probs = F.softmax(policy_logits, dim=-1)
        sampled = torch.multinomial(
            probs.view(-1, V), 1
        ).view(B, L)
        samples.append(sampled)
        rewards.append(reward_fn(sampled))
    rewards = torch.stack(rewards, dim=-1)  # (B, num_samples)
    best_idx = rewards.argmax(dim=-1)  # (B,)
    best_samples = torch.stack(samples, dim=1)  # (B, num_samples, L)
    best_samples = best_samples[torch.arange(B), best_idx]  # (B, L)
    return best_samples, rewards
```

&emsp;&emsp;这个实现中，对每个提示采样多个回答，选择奖励最高的一个作为输出。实际中奖励函数可以是训练好的奖励模型。

---

#### 2.11.8 RLAIF

&emsp;&emsp;RLAIF（Reinforcement Learning from AI Feedback）用 AI 反馈替代人类反馈来训练奖励模型或直接提供奖励信号。其流程与 RLHF 类似，但偏好标注由强大的 LLM 完成：给定提示 $x$ 和两个回答 $y_a, y_b$，让 AI 判断哪个更好，生成偏好标签，再用这些标签训练奖励模型或直接用于 DPO。RLAIF 的核心优势在于可扩展性：人类标注成本高且速度慢，而 AI 标注可以大规模并行生成。RLAIF 的优点是显著降低标注成本，可以快速迭代，且在某些任务上 AI 反馈与人类反馈高度一致；缺点是 AI 反馈可能继承并放大基座模型的偏见，且对于超出 AI 能力的任务，反馈质量无法保证。从维度视角看，RLAIF 在特征维上用 AI 的偏好判断替代人类的偏好判断，奖励信号来自另一个模型的输出分布，相当于用模型间的知识蒸馏来指导策略优化。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn.functional as F

def rlaif_preference_loss(policy_chosen_logps, policy_rejected_logps,
                          ref_chosen_logps, ref_rejected_logps,
                          ai_confidence, beta=0.1):
    """
    ai_confidence: (B,) AI 对偏好判断的置信度
    """
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_ratio - rejected_ratio)
    # 用 AI 置信度加权
    loss = -F.logsigmoid(logits) * ai_confidence
    return loss.mean()
```

&emsp;&emsp;这个实现中，AI 置信度用于加权偏好损失，置信度高的样本对训练贡献更大。

---

#### 2.11.9 宪法 AI 与自我纠正

&emsp;&emsp;宪法 AI（Constitutional AI）由 Anthropic 提出，核心思想是让模型根据一套书面原则（宪法）进行自我批评和修正，减少对人类标注的依赖。流程分为两个阶段。第一阶段是监督学习：模型对提示生成初始回答，然后根据宪法原则进行自我批评，指出回答中违反原则的地方，再生成修正后的回答。用修正后的回答进行 SFT。第二阶段是强化学习：模型对同一提示生成多个回答，用 AI 根据宪法判断哪个更好，生成偏好数据训练奖励模型，再用 RLAIF 优化策略。自我纠正的形式化过程为：给定初始回答 $y_0$，批评 $c = \mathrm{Critique}(x, y_0, \mathcal{C})$，修正 $y_1 = \mathrm{Revise}(x, y_0, c, \mathcal{C})$，其中 $\mathcal{C}$ 是宪法原则集合。宪法 AI 的优点是减少人类标注，过程透明可审计，且可以通过修改宪法快速调整模型行为；缺点是宪法原则的设计需要大量人工，模型可能学会表面迎合原则而忽略深层意图，且自我批评可能引入新的错误。从维度视角看，宪法 AI 在特征维上引入了一个基于原则的批评信号，模型需要根据这个信号调整输出分布，相当于在策略空间中沿宪法约束的方向做投影。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn.functional as F

def constitutional_self_correction(policy_logits, critique_scores, beta=0.5):
    """
    policy_logits: (B, L, V) 初始回答的 logits
    critique_scores: (B,) 批评分数，越低表示越违反宪法
    """
    # 用批评分数调整 logits: 违反原则的回答被抑制
    adjusted_logits = policy_logits - beta * critique_scores.view(-1, 1, 1)
    return adjusted_logits

def constitutional_loss(policy_logits, revised_logits):
    """让修正后的分布接近理想分布"""
    log_probs = F.log_softmax(policy_logits, dim=-1)
    target_probs = F.softmax(revised_logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()
```

&emsp;&emsp;这个实现中，批评分数用于调整 logits，修正后的分布作为监督目标。

---

#### 2.11.10 可验证奖励

&emsp;&emsp;可验证奖励（Verifiable Rewards）是一类无需训练奖励模型、直接用程序化验证器给出奖励信号的奖励机制，广泛应用于数学、代码、逻辑推理等答案可自动判定正确性的任务。对于数学题，验证器可以检查最终答案是否与标准答案一致；对于代码题，验证器可以运行测试用例判断代码是否正确。设回答 $y$ 的最终答案为 $a(y)$，标准答案为 $a^*$，则奖励为：

$$
r(x, y) = \begin{cases}
1 & \text{if } \mathrm{Verify}(a(y), a^*) = \text{True} \\
0 & \text{otherwise}
\end{cases}
$$

&emsp;&emsp;在 RLVR（Reinforcement Learning with Verifiable Rewards）中，这个 0/1 奖励直接用于 PPO 或 GRPO 的优势估计，无需训练奖励模型。可验证奖励的优点是奖励绝对准确，不存在奖励黑客问题，训练信号干净，且可以大规模自动生成；缺点是仅适用于答案可验证的领域，对于开放式生成、创意写作等任务无法使用，且 0/1 奖励稀疏，需要配合组内相对优势等方法才能有效训练。从维度视角看，可验证奖励在特征维上用确定性验证函数替代了学习的奖励模型，奖励信号是二值的、无偏的，但覆盖范围有限。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch

def verifiable_reward(predicted_answers, ground_truth_answers):
    """
    predicted_answers: list of str
    ground_truth_answers: list of str
    返回 0/1 奖励
    """
    rewards = []
    for pred, gt in zip(predicted_answers, ground_truth_answers):
        # 简单字符串比较，实际中可以是数学等价判断或代码执行
        if pred.strip() == gt.strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return torch.tensor(rewards)

def grpo_with_verifiable_reward(logprobs, old_logprobs, rewards,
                                 clip_epsilon=0.2):
    """用可验证奖励做 GRPO"""
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r
    advantages = advantages.unsqueeze(-1)
    ratio = torch.exp(logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    return -torch.min(surr1, surr2).mean()
```

&emsp;&emsp;这个实现中，验证器直接比较答案，返回 0/1 奖励，然后用于 GRPO 的优势计算。

---

#### 2.11.11 思考预算

&emsp;&emsp;思考预算（Thinking Budget）是控制模型推理时计算量的机制，在具备思维链（Chain-of-Thought）能力的模型中尤为重要。模型在回答前会生成一段思考过程，思考预算限制了这段过程的长度或计算量。设思考预算为 $B$，模型生成的思考 token 数为 $T_{\mathrm{think}}$，则约束为 $T_{\mathrm{think}} \leq B$。在训练时，可以通过在数据中混合不同预算的样本，让模型学会在给定预算下分配推理资源；在推理时，可以动态设置预算，简单问题用低预算，复杂问题用高预算。思考预算的优点是显著降低推理成本，使模型可以根据任务难度自适应地分配计算，且高预算通常能提升复杂推理任务的准确率；缺点是预算过低会导致模型无法完成复杂推理，预算分配策略需要额外训练或启发式规则，且思考过程的可解释性可能随预算压缩而下降。从维度视角看，思考预算在 token 维上限制了思维链的长度，相当于在序列生成过程中对 token 维扩散的步数施加了上限，模型需要在有限步数内完成信息聚合。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn.functional as F

def thinking_budget_loss(policy_logits, budget, eos_token_id):
    """
    policy_logits: (B, L, V)
    budget: 最大思考 token 数
    惩罚超过预算的生成
    """
    B, L, V = policy_logits.size()
    probs = F.softmax(policy_logits, dim=-1)
    # 计算生成 EOS 的概率
    eos_prob = probs[:, :, eos_token_id]  # (B, L)
    # 超过预算的 token 被惩罚
    if L > budget:
        over_budget = torch.arange(L, device=policy_logits.device) >= budget
        penalty = -eos_prob[:, over_budget].sum(dim=-1).mean()
    else:
        penalty = torch.tensor(0.0, device=policy_logits.device)
    return penalty

def apply_budget_mask(logits, budget, eos_token_id):
    """推理时强制在预算处生成 EOS"""
    B, L, V = logits.size()
    if L >= budget:
        # 将预算之后的位置的 EOS 概率设为无穷大
        logits[:, budget:, eos_token_id] = float('inf')
    return logits
```

&emsp;&emsp;这个实现中，`thinking_budget_loss` 惩罚超过预算的生成，`apply_budget_mask` 在推理时强制在预算处结束思考。

---

### 2.12 推理时计算与推理模型

#### 2.12.1 思维链（CoT）

&emsp;&emsp;思维链（Chain-of-Thought, CoT）由 Wei 等人于 2022 年提出，核心思想是让模型在给出最终答案之前，先生成一段中间推理步骤。对于数学题、逻辑推理、多跳问答等需要多步推理的任务，直接输出答案往往容易出错，而生成推理过程可以显著提升准确率。CoT 的形式化表示为：给定问题 $x$，模型生成推理链 $z = (z_1, \dots, z_m)$ 和最终答案 $y$，训练目标为：

$$
\mathcal{L}_{\mathrm{CoT}} = -\sum_{t=1}^{m} \log P(z_t \mid x, z_{<t}) - \log P(y \mid x, z)
$$

&emsp;&emsp;CoT 的触发方式包括少样本提示（在提示中给出几个带推理步骤的示例）、零样本提示（直接要求模型“一步一步思考”）、以及通过 SFT 在训练数据中显式加入推理过程。CoT 的优点是显著提升多步推理任务的准确率，推理过程可解释，便于人工检查错误；缺点是推理过程增加了输出长度和推理成本，且模型可能生成看似合理但实际错误的推理链，即“错误推理得到正确答案”。从维度视角看，CoT 在 token 维上扩展了生成序列的长度，将原本单步的答案映射扩展为多步的推理路径，使模型在特征维上有更多中间计算步骤来完成复杂变换。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn.functional as F

def cot_loss(logits, input_ids, cot_start, answer_start):
    """
    logits: (B, L, V)
    input_ids: (B, L)
    cot_start: 思维链起始位置
    answer_start: 答案起始位置
    """
    B, L, V = logits.size()
    logits_flat = logits[:, :-1].reshape(-1, V)
    labels_flat = input_ids[:, 1:].reshape(-1)

    # 构建掩码: 只对思维链和答案部分计算损失
    mask = torch.zeros(B, L - 1, device=logits.device)
    mask[:, cot_start-1:answer_start-1] = 1.0  # 思维链部分
    mask[:, answer_start-1:] = 1.0             # 答案部分
    mask_flat = mask.reshape(-1)

    loss = F.cross_entropy(logits_flat, labels_flat, reduction='none')
    return (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
```

&emsp;&emsp;这个实现中，损失只对思维链和答案部分计算，输入问题部分的 token 被掩码。

---

#### 2.12.2 隐藏思维链

&emsp;&emsp;隐藏思维链（Latent Chain-of-Thought）是指模型在内部隐式地进行多步推理，而不在输出中显式生成推理 token。与显式 CoT 不同，隐藏思维链不占用输出序列长度，而是在特征维上通过多层变换完成推理。实现方式包括：在模型内部增加额外的计算层（如循环块、深度循环），使用连续向量而非离散 token 作为中间状态，或通过特殊训练目标让模型在特定位置进行额外计算。设隐藏状态为 $h$，隐藏思维链可以表示为：

$$
h^{(0)} = \mathrm{Encoder}(x), \quad h^{(k+1)} = \mathrm{Block}(h^{(k)}), \quad y = \mathrm{Decoder}(h^{(K)})
$$

&emsp;&emsp;其中 $K$ 是内部推理步数。隐藏思维链的优点是推理成本不随推理复杂度线性增长（输出长度不变），且中间状态是连续向量，避免了离散 token 的信息瓶颈；缺点是内部推理过程不可解释，难以调试，且需要特殊的训练目标或架构设计才能有效学习多步推理。从维度视角看，隐藏思维链在特征维上增加了额外的变换步数，相当于在特征空间中做多次迭代，而非在 token 维上展开推理路径。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn as nn

class LatentCoTBlock(nn.Module):
    """隐藏思维链: 在特征维上做 K 步内部推理"""
    def __init__(self, d_model, num_heads, d_ff, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads,
                dim_feedforward=d_ff, batch_first=True
            ) for _ in range(num_steps)
        ])

    def forward(self, x):
        # 在内部重复 K 步，不生成新 token
        for layer in self.layers:
            x = layer(x)
        return x
```

&emsp;&emsp;这个实现中，模型在内部对隐藏状态做 K 步变换，不生成额外的 token，推理过程完全隐藏在特征维中。

---

#### 2.12.3 推理时计算扩展

&emsp;&emsp;推理时计算扩展（Inference-Time Compute Scaling）是指在推理阶段通过增加计算量来提升模型性能的策略。与训练时扩展（增加参数量或训练数据）不同，推理时扩展在模型固定后，通过调整推理过程来分配更多计算资源给困难问题。主要方法包括：增加思维链长度（让模型生成更长的推理过程）、多次采样后投票（best-of-N 或 majority voting）、树搜索（在推理路径上做搜索）、以及自适应计算（根据问题难度动态分配计算量）。设单次推理成本为 $C$，采样 $N$ 次的成本为 $NC$，多数投票的输出为：

$$
y^* = \arg\max_{y} \sum_{i=1}^{N} \mathbf{1}\{y_i = y\}
$$

&emsp;&emsp;推理时计算扩展的优点是无需重新训练模型，仅通过推理策略就能提升性能，且可以根据任务难度灵活调整计算量；缺点是计算成本随采样数或推理长度线性甚至超线性增长，且对于简单问题过度扩展会造成浪费。从维度视角看，推理时计算扩展在 token 维上增加了生成的步数，或在输出空间中探索了多条路径，相当于用更多 token 维的计算来补偿模型固定的特征维容量。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn.functional as F

def self_consistency_vote(policy, input_ids, num_samples=8):
    """自一致性投票: 多次采样后多数投票"""
    samples = []
    for _ in range(num_samples):
        # 采样生成
        logits = policy(input_ids)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        sampled = torch.multinomial(probs, 1)
        samples.append(sampled)

    # 多数投票
    stacked = torch.stack(samples, dim=-1)  # (B, 1, num_samples)
    votes = stacked.squeeze(1)  # (B, num_samples)
    # 简单多数投票
    result = []
    for b in range(votes.size(0)):
        values, counts = votes[b].unique(return_counts=True)
        result.append(values[counts.argmax()])
    return torch.stack(result)

def best_of_n_rerank(policy, reward_model, input_ids, num_samples=8):
    """Best-of-N: 采样 N 个，选奖励最高的"""
    best_sample = None
    best_reward = float('-inf')
    for _ in range(num_samples):
        logits = policy(input_ids)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        sampled = torch.multinomial(probs, 1)
        reward = reward_model(sampled)
        if reward > best_reward:
            best_reward = reward
            best_sample = sampled
    return best_sample
```

&emsp;&emsp;这个实现中，`self_consistency_vote` 多次采样后投票，`best_of_n_rerank` 选择奖励最高的样本。

---

#### 2.12.4 reasoning_effort

&emsp;&emsp;`reasoning_effort` 是 OpenAI o 系列和 GPT-5 系列模型中引入的推理努力控制参数，允许用户在推理时指定模型投入多少计算资源进行推理。该参数通常取值为 `low`、`medium`、`high`，对应不同的推理深度和思考 token 预算。在 API 层面，`reasoning_effort` 控制模型在生成最终答案前进行内部推理的程度：`low` 适用于简单问答和事实检索，`medium` 适用于一般推理任务，`high` 适用于复杂数学、代码和逻辑推理。其背后的机制是模型在训练时学会了根据不同的努力级别调整推理行为，推理努力越高，生成的思考 token 越多，内部计算步数越多。设努力级别为 $e \in \{e_1, e_2, \dots, e_K\}$，对应的思考预算为 $B(e)$，则推理过程为：

$$
z \sim \pi_\theta(\cdot \mid x, e), \quad y \sim \pi_\theta(\cdot \mid x, z)
$$

&emsp;&emsp;`reasoning_effort` 的优点是用户可以按需权衡推理质量和成本，简单问题用低努力快速回答，复杂问题用高努力获得更好结果，且同一模型可以在不同努力级别下服务不同场景；缺点是努力级别与任务难度的匹配需要用户判断，过高努力浪费计算，过低努力可能无法解决复杂问题。从维度视角看，`reasoning_effort` 在 token 维上控制思维链的展开长度，相当于在推理时动态调节 token 维扩散的步数预算。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn.functional as F

class ReasoningEffortController:
    """reasoning_effort 控制: 不同努力级别对应不同思考预算"""
    EFFORT_BUDGETS = {
        "low": 256,
        "medium": 1024,
        "high": 4096,
    }

    def __init__(self, policy, tokenizer):
        self.policy = policy
        self.tokenizer = tokenizer

    def generate_with_effort(self, input_ids, effort="medium", max_new_tokens=2048):
        budget = self.EFFORT_BUDGETS.get(effort, 1024)
        max_new_tokens = min(max_new_tokens, budget)

        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self.policy(generated)
            next_logits = logits[:, -1, :]

            # 如果达到预算上限，强制生成结束符
            if generated.size(1) - input_ids.size(1) >= budget:
                next_logits[:, self.tokenizer.eos_token_id] = float('inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return generated
```

&emsp;&emsp;这个实现中，不同努力级别对应不同的思考 token 预算，预算耗尽时强制结束推理。

---

#### 2.12.5 MCTS 与 Q*

&emsp;&emsp;MCTS（Monte Carlo Tree Search，蒙特卡洛树搜索）是一种在决策空间中通过随机模拟来评估动作价值的搜索算法，在 AlphaGo 等博弈系统中取得巨大成功。在语言模型推理中，MCTS 将每个推理步骤视为一个动作，通过选择、扩展、模拟、回溯四个阶段构建搜索树。选择阶段使用 UCB 公式平衡探索与利用：

$$
a^* = \arg\max_a \left( Q(s, a) + c \cdot \sqrt{\frac{\ln N(s)}{N(s, a)}} \right)
$$

&emsp;&emsp;其中 $Q(s,a)$ 是动作价值估计，$N(s)$ 是状态访问次数，$N(s,a)$ 是动作访问次数，$c$ 是探索常数。Q*（Q-star）是 OpenAI 在 o 系列模型中探索的推理搜索框架，结合了 MCTS 和 Q-learning 的思想，通过在推理路径上做树搜索和价值估计来提升复杂推理的准确性。Q* 的核心是将推理过程建模为马尔可夫决策过程，每一步推理是一个动作，最终答案的正确性作为奖励。MCTS 与 Q* 的优点是能够在推理时系统性地探索多条路径，避免贪心解码陷入局部最优，对数学和逻辑推理任务提升显著；缺点是搜索成本高，需要多次调用模型评估状态，且价值估计的准确性直接影响搜索质量。从维度视角看，MCTS 在 token 维上构建了一棵推理路径树，通过树搜索在多个可能的 token 序列之间做选择，相当于将单路径的 token 维扩散扩展为多路径的树状探索。

**&emsp;&emsp;PyTorch 实现示例**

```python
import math
import torch
import torch.nn.functional as F

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def ucb_score(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

def mcts_search(policy, value_fn, root_state, num_simulations=50,
                max_depth=10):
    """简化的 MCTS 推理搜索"""
    root = MCTSNode(root_state)

    for _ in range(num_simulations):
        node = root
        # 选择
        while node.children and len(node.children) > 0:
            node = max(node.children.values(), key=lambda n: n.ucb_score())

        # 扩展
        if node.visits > 0:
            logits = policy(node.state)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            topk_probs, topk_ids = probs.topk(5)
            for prob, tid in zip(topk_probs[0], topk_ids[0]):
                new_state = torch.cat([node.state, tid.view(1, 1)], dim=-1)
                node.children[tid.item()] = MCTSNode(new_state, node)

        # 模拟: 用价值函数估计
        value = value_fn(node.state)

        # 回溯
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    # 选择访问次数最多的子节点
    best_child = max(root.children.values(), key=lambda n: n.visits)
    return best_child.state
```

&emsp;&emsp;这个实现中，MCTS 通过 UCB 选择、扩展、价值估计和回溯四个阶段构建搜索树，最终选择访问次数最多的路径。

---

#### 2.12.6 路由器与统一系统

&emsp;&emsp;路由器与统一系统（Router and Unified System）在推理架构中指的是一个统一的模型系统，通过路由器根据任务类型或难度动态选择不同的推理模式。在 DeepSeek-V4 等系统中，统一系统包含多个推理模式：快速模式（直接输出答案，无思维链）、思考模式（生成完整思维链）、以及介于两者之间的混合模式。路由器根据输入的特征或任务类型决定使用哪种模式。

&emsp;&emsp;设输入为 $x$，路由器输出模式选择 $m = \mathrm{Router}(x)$，其中 $m \in \{\text{fast}, \text{think}, \text{hybrid}\}$。不同模式对应不同的推理预算和生成策略：

$$
y = \begin{cases}
\pi_{\text{fast}}(y \mid x) & m = \text{fast} \\
\pi_{\text{think}}(y \mid x, z) & m = \text{think} \\
\pi_{\text{hybrid}}(y \mid x, z_{1:k}) & m = \text{hybrid}
\end{cases}
$$

&emsp;&emsp;路由器的训练可以通过监督学习（用标注的任务难度标签）或强化学习（根据最终答案质量和计算成本的权衡）。路由器与统一系统的优点是单一模型可以同时服务简单和复杂任务，简单任务快速响应，复杂任务深度推理，资源分配更高效；缺点是路由器的决策可能出错，将复杂任务误判为简单任务会导致质量下降，且统一系统需要在训练时同时优化多种推理模式，训练复杂度更高。从维度视角看，路由器在特征维上根据输入特征选择不同的 token 维扩散策略：快速模式跳过或极短思维链，思考模式展开完整思维链，混合模式在两者之间动态调整。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReasoningRouter(nn.Module):
    """推理路由器: 根据输入选择推理模式"""
    def __init__(self, d_model, num_modes=3):
        super().__init__()
        self.num_modes = num_modes
        self.classifier = nn.Parameter(torch.empty(num_modes, d_model))
        nn.init.normal_(self.classifier, std=0.02)

    def forward(self, x):
        """
        x: (B, L, d_model)
        返回每个样本的模式概率
        """
        # 取平均池化作为输入表示
        pooled = x.mean(dim=1)  # (B, d_model)
        logits = torch.matmul(pooled, self.classifier.T)  # (B, num_modes)
        probs = F.softmax(logits, dim=-1)
        return probs

    def route(self, x):
        """选择模式"""
        probs = self.forward(x)
        mode = probs.argmax(dim=-1)  # (B,)
        return mode, probs


class UnifiedReasoningSystem(nn.Module):
    """统一推理系统: 路由器 + 多模式生成"""
    def __init__(self, policy, d_model):
        super().__init__()
        self.policy = policy
        self.router = ReasoningRouter(d_model)
        self.mode_budgets = {
            0: 0,     # fast: 无思维链
            1: 512,   # think: 中等思维链
            2: 2048,  # deep: 长思维链
        }

    def forward(self, input_ids, hidden_states):
        mode, probs = self.router.route(hidden_states)
        outputs = []
        for b in range(input_ids.size(0)):
            budget = self.mode_budgets[mode[b].item()]
            # 根据预算生成
            out = self.policy.generate(input_ids[b:b+1], max_new_tokens=budget + 256)
            outputs.append(out)
        return outputs, mode, probs
```

&emsp;&emsp;这个实现中，`ReasoningRouter` 根据输入表示选择推理模式，`UnifiedReasoningSystem` 根据模式分配不同的思维链预算。

---

#### 2.12.7 推测解码

&emsp;&emsp;推测解码（Speculative Decoding）由 Leviathan 等人于 2023 年提出，是一种加速自回归生成的技术。其核心思想是用一个小而快的草稿模型（Draft Model）快速生成多个候选 token，然后用大模型（Target Model）并行验证这些候选 token 是否可接受。由于大模型的前向传播可以并行处理整个候选序列，而自回归生成是串行的，推测解码将多次串行前向合并为一次并行前向，从而加速生成。

&emsp;&emsp;设草稿模型为 $q$，目标模型为 $p$。草稿模型自回归生成 $\gamma$ 个候选 token $\tilde{x}_1, \dots, \tilde{x}_\gamma$，然后目标模型并行计算这些位置的概率分布。对于每个候选 token $\tilde{x}_t$，接受概率为：

$$
\min\left(1, \frac{p(\tilde{x}_t \mid x, \tilde{x}_{<t})}{q(\tilde{x}_t \mid x, \tilde{x}_{<t})}\right)
$$

&emsp;&emsp;若候选 token 被拒绝，则从修正分布中重新采样：

$$
p'(x) = \mathrm{norm}\left(\max(0, p(x \mid x, \tilde{x}_{<t}) - q(x \mid x, \tilde{x}_{<t}))\right)
$$

&emsp;&emsp;推测解码的优点是理论上保证输出分布与目标模型完全一致，即无损加速，实际加速比可达 2-3 倍，且可以与量化、批处理等技术叠加；缺点是草稿模型的质量直接影响加速比，草稿模型太差会导致大量拒绝，反而增加开销，且需要额外显存加载草稿模型。从维度视角看，推测解码在 token 维上并行验证多个候选 token，将串行的 token 维扩散转化为并行的批量验证，用草稿模型的快速生成换取目标模型的并行计算。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn.functional as F

def speculative_decoding(draft_model, target_model, input_ids,
                         gamma=5, max_new_tokens=100):
    """
    draft_model: 草稿模型
    target_model: 目标模型
    gamma: 每次草稿生成的 token 数
    """
    generated = input_ids
    tokens_generated = 0

    while tokens_generated < max_new_tokens:
        # 草稿模型自回归生成 gamma 个候选
        draft_tokens = []
        draft_probs = []
        draft_input = generated
        for _ in range(gamma):
            logits = draft_model(draft_input)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            token = torch.multinomial(probs, 1)
            draft_tokens.append(token)
            draft_probs.append(probs)
            draft_input = torch.cat([draft_input, token], dim=-1)

        draft_tokens = torch.cat(draft_tokens, dim=-1)  # (B, gamma)

        # 目标模型并行验证
        candidate = torch.cat([generated, draft_tokens], dim=-1)
        target_logits = target_model(candidate)
        target_probs = F.softmax(target_logits, dim=-1)

        # 逐个验证
        accepted = 0
        for t in range(gamma):
            p = target_probs[:, generated.size(1) + t - 1, :]
            q = draft_probs[t]
            token = draft_tokens[:, t]

            # 接受概率
            p_token = p.gather(1, token.unsqueeze(-1)).squeeze(-1)
            q_token = q.gather(1, token.unsqueeze(-1)).squeeze(-1)
            accept_prob = torch.minimum(
                torch.ones_like(p_token), p_token / (q_token + 1e-8)
            )

            if torch.rand(1).item() < accept_prob.item():
                accepted += 1
            else:
                # 拒绝: 从修正分布重新采样
                corrected = torch.clamp(p - q, min=0)
                corrected = corrected / (corrected.sum(dim=-1, keepdim=True) + 1e-8)
                new_token = torch.multinomial(corrected, 1)
                generated = torch.cat([generated, draft_tokens[:, :accepted], new_token], dim=-1)
                tokens_generated += accepted + 1
                break
        else:
            # 全部接受
            generated = torch.cat([generated, draft_tokens], dim=-1)
            tokens_generated += gamma

    return generated
```

&emsp;&emsp;这个实现中，草稿模型自回归生成 $\gamma$ 个候选 token，目标模型并行验证，接受则保留，拒绝则从修正分布重新采样。理论上输出分布与目标模型完全一致。


---

### 2.13 推理优化与加速

#### 2.13.1 KV 缓存

&emsp;&emsp;KV 缓存（KV Cache）是自回归生成中最核心的推理优化技术。在 Decoder-only 模型中，每生成一个新 token，都需要对所有历史 token 计算注意力。如果不做缓存，每步都要重新计算历史 token 的 Key 和 Value，导致总计算量为 $O(L^2)$。KV 缓存的思路是将每层已计算过的 Key 和 Value 保存下来，生成新 token 时只需计算当前 token 的 Q、K、V，并将新的 K、V 追加到缓存中，然后与缓存的全部历史 K、V 做注意力。

&emsp;&emsp;设第 $l$ 层在位置 $t$ 的输入为 $h_t^l$，KV 缓存更新为：

$$
K_{1:t}^l = \mathrm{Concat}(K_{1:t-1}^l, h_t^l W_K^l), \quad V_{1:t}^l = \mathrm{Concat}(V_{1:t-1}^l, h_t^l W_V^l)
$$

&emsp;&emsp;注意力计算变为：

$$
o_t^l = \mathrm{softmax}\left(\frac{(h_t^l W_Q^l) (K_{1:t}^l)^\top}{\sqrt{d_k}}\right) V_{1:t}^l
$$

&emsp;&emsp;KV 缓存的优点是每步只需计算当前 token 的 Q、K、V，避免了历史 token 的重复计算，将生成总计算量从 $O(L^2)$ 降到 $O(L)$（每步）和 $O(L^2)$（总体但常数更小），且实现简单，只需在注意力层中维护两个张量；缺点是显存占用随序列长度和层数线性增长，对于长上下文和大 batch，KV 缓存可能占用数十 GB 显存，成为推理的主要瓶颈，且缓存需要按 batch 和层分别管理，增加了内存分配和调度的复杂度。

&emsp;&emsp;从维度视角看，KV 缓存将 token 维上的历史信息持久化到显存中，使每步生成只需在特征维上计算当前 token 的投影，再与缓存中的 token 维表示做注意力。它本质上是把 token 维扩散的中间结果缓存起来，避免重复计算。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出带 KV 缓存的自注意力裸实现：

```python
import math
import torch
import torch.nn as nn

class CachedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

    def forward(self, x, cache=None):
        """
        x: (B, 1, d_model) 当前 token（推理时）
        或 (B, L, d_model) 训练时
        cache: (K_cache, V_cache)，形状 (B, h, T, d_k)
        """
        B, L, _ = x.size()

        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        if cache is not None:
            K_cache, V_cache = cache
            K = torch.cat([K_cache, K], dim=2)
            V = torch.cat([V_cache, V], dim=2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if L == 1:
            # 单 token 生成，无需因果掩码
            attn = torch.softmax(scores, dim=-1)
        else:
            causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = torch.softmax(scores, dim=-1)

        head_out = torch.matmul(attn, V)
        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = torch.matmul(head_out, self.W_o)
        return output, (K, V)
```

&emsp;&emsp;这个实现中，`cache` 保存了历史 K 和 V，当前 token 的 K、V 追加到缓存后与 Q 做注意力。若输入为单 token $(B,1,d_{model})$，输出形状为 $(B,1,d_{model})$，缓存形状随生成长度增长。

---

#### 2.13.2 滚动缓冲区缓存

&emsp;&emsp;滚动缓冲区缓存（Rolling Buffer Cache）是滑动窗口注意力或流式生成中使用的 KV 缓存管理策略。当注意力只关注最近 $w$ 个 token 时，KV 缓存不需要保存全部历史，只需维护一个大小为 $w$ 的环形缓冲区。每生成一个新 token，新的 K、V 覆盖缓冲区中最旧的条目，缓存大小固定为 $O(w)$。

&emsp;&emsp;设窗口大小为 $w$，第 $t$ 步的缓存索引为：

$$
\mathrm{idx}(t) = t \bmod w
$$

&emsp;&emsp;缓存更新为：

$$
K_{\mathrm{buf}}[\mathrm{idx}(t)] = k_t, \quad V_{\mathrm{buf}}[\mathrm{idx}(t)] = v_t
$$

&emsp;&emsp;注意力只对缓冲区中的 $w$ 个位置计算。滚动缓冲区缓存的优点是显存占用恒定，不随序列长度增长，使流式生成和超长序列推理成为可能；缺点是只能保留最近 $w$ 个 token 的信息，超过窗口的历史信息完全丢失，对于需要长距离依赖的任务可能损失性能，且窗口大小 $w$ 需要根据任务和硬件选择。从维度视角看，滚动缓冲区缓存在 token 维上只保留最近的 $w$ 个位置，将 KV 缓存的 token 维从动态增长变为固定窗口，使 token 维扩散只在局部窗口内进行。

**&emsp;&emsp;PyTorch 实现示例**

```python
import math
import torch
import torch.nn as nn

class RollingBufferAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

    def init_buffer(self, batch_size, device):
        K_buf = torch.zeros(batch_size, self.num_heads, self.window_size, self.d_k, device=device)
        V_buf = torch.zeros(batch_size, self.num_heads, self.window_size, self.d_k, device=device)
        return K_buf, V_buf, 0

    def forward(self, x, buffer):
        """
        x: (B, 1, d_model)
        buffer: (K_buf, V_buf, step)
        """
        B, L, _ = x.size()
        K_buf, V_buf, step = buffer

        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        # 写入环形缓冲区
        idx = step % self.window_size
        K_buf[:, :, idx:idx+1, :] = K
        V_buf[:, :, idx:idx+1, :] = V

        # 从缓冲区读取（按时间顺序重排）
        order = (torch.arange(self.window_size, device=x.device) + step + 1) % self.window_size
        K_read = K_buf[:, :, order, :]
        V_read = V_buf[:, :, order, :]

        # 有效长度
        valid_len = min(step + 1, self.window_size)
        K_valid = K_read[:, :, -valid_len:, :]
        V_valid = V_read[:, :, -valid_len:, :]

        scores = torch.matmul(Q, K_valid.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V_valid)
        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = torch.matmul(head_out, self.W_o)
        return output, (K_buf, V_buf, step + 1)
```

&emsp;&emsp;这个实现中，`K_buf` 和 `V_buf` 大小固定为窗口大小，新的 K、V 覆盖最旧的位置。若输入为单 token $(B,1,d_{model})$，输出形状为 $(B,1,d_{model})$，缓存大小恒定。

---

#### 2.13.3 量化

&emsp;&emsp;量化（Quantization）是将模型权重和激活值从高精度浮点数映射到低精度整数的技术，用于减少显存占用和加速推理。在推理场景中，量化主要作用于权重和 KV 缓存。设原始浮点权重为 $w$，量化到 $b$ 位整数：

$$
w_q = \mathrm{round}\left(\frac{w - z}{s}\right), \quad \hat{w} = s \cdot w_q + z
$$

&emsp;&emsp;其中 $s$ 是缩放因子，$z$ 是零点。对于对称量化，$z=0$，$s = \max(|w|) / (2^{b-1}-1)$；对于非对称量化，$s = (w_{\max} - w_{\min}) / (2^b - 1)$，$z = \mathrm{round}(-w_{\min}/s)$。KV 缓存的量化通常采用分组量化：将每个头的 K 或 V 按通道分组，每组独立计算缩放因子和零点。设分组大小为 $g$，则每组的量化参数不同，以更好地适应不同通道的数值分布。量化推理的优点是显存占用和带宽需求大幅降低，INT8 量化可将模型大小减半，INT4 量化可减少 75%，且整数运算在支持 INT8/INT4 的硬件上比浮点更快；缺点是非对称量化的零点计算增加开销，低精度量化会引入精度损失，对异常值敏感的层可能需要保留高精度，且量化后的模型在某些任务上性能下降明显。

&emsp;&emsp;从维度视角看，量化在特征维上将连续浮点值映射到离散整数格点，通过缩放因子和零点保留数值范围信息。分组量化在特征维上进一步细分，使不同通道组有不同的量化尺度，减少信息损失。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn as nn

def symmetric_quantize(x, bits=8):
    """对称量化"""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max() / qmax
    scale = torch.clamp(scale, min=1e-8)
    x_q = torch.round(x / scale).clamp(-qmax - 1, qmax)
    return x_q, scale

def asymmetric_quantize(x, bits=8, group_size=128):
    """分组非对称量化"""
    qmax = 2 ** bits - 1
    orig_shape = x.shape
    # 按最后一维分组
    x_flat = x.reshape(-1, group_size)
    x_min = x_flat.min(dim=-1, keepdim=True).values
    x_max = x_flat.max(dim=-1, keepdim=True).values
    scale = (x_max - x_min) / qmax
    scale = torch.clamp(scale, min=1e-8)
    zero = torch.round(-x_min / scale).clamp(0, qmax)
    x_q = torch.round(x_flat / scale + zero).clamp(0, qmax)
    # 反量化
    x_deq = (x_q - zero) * scale
    return x_deq.reshape(orig_shape), scale, zero

class QuantizedLinear(nn.Module):
    """量化线性层: 权重 INT8，计算时反量化"""
    def __init__(self, in_features, out_features, bits=8, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("q_weight", None)
        self.register_buffer("scale", None)
        self.register_buffer("zero", None)

    def quantize(self):
        w = self.weight.data
        qmax = 2 ** self.bits - 1
        w_flat = w.reshape(-1, self.group_size)
        w_min = w_flat.min(dim=-1, keepdim=True).values
        w_max = w_flat.max(dim=-1, keepdim=True).values
        scale = (w_max - w_min) / qmax
        scale = torch.clamp(scale, min=1e-8)
        zero = torch.round(-w_min / scale).clamp(0, qmax)
        q = torch.round(w_flat / scale + zero).clamp(0, qmax)
        self.q_weight = q.to(torch.uint8)
        self.scale = scale
        self.zero = zero

    def forward(self, x):
        if self.q_weight is None:
            self.quantize()
        w_deq = (self.q_weight.float() - self.zero) * self.scale
        w_deq = w_deq.reshape(self.out_features, self.in_features)
        return torch.matmul(x, w_deq.T)
```

&emsp;&emsp;这个实现中，`symmetric_quantize` 和 `asymmetric_quantize` 分别实现对称和分组非对称量化，`QuantizedLinear` 将权重 INT8 量化后存储，推理时反量化计算。

---

#### 2.13.4 FlashAttention

&emsp;&emsp;FlashAttention 在推理场景中的核心价值是减少 KV 缓存和注意力矩阵的显存占用。标准注意力在推理时需要显式构造完整的 $L \times L$ 注意力矩阵，对于长序列，这个矩阵的显存占用为 $O(L^2)$，成为推理瓶颈。FlashAttention 通过分块和在线 Softmax，在片上 SRAM 中完成注意力计算，避免将完整的注意力矩阵写入显存。

&emsp;&emsp;在推理时，FlashAttention 的分块策略与训练时一致：将 Q、K、V 分块加载到 SRAM，逐块计算注意力分数，用在线 Softmax 维护运行最大值和归一化因子，最终累积输出。设当前处理到第 $j$ 个 K 块，运行最大值 $m_j$ 和归一化因子 $\ell_j$ 的更新为：

$$
m_j = \max(m_{j-1}, \max(K_j \text{ 的行最大值}))
$$

$$
\ell_j = e^{m_{j-1} - m_j} \ell_{j-1} + \sum_{k \in \text{块}j} e^{s_k - m_j}
$$

&emsp;&emsp;输出累积为：

$$
O_j = e^{m_{j-1} - m_j} O_{j-1} + \sum_{k \in \text{块}j} e^{s_k - m_j} V_k
$$

&emsp;&emsp;在推理中，FlashAttention 与 KV 缓存结合使用：KV 缓存按块组织，每次生成新 token 时，只将新的 K、V 块追加到缓存，然后 FlashAttention 逐块遍历缓存计算注意力。FlashAttention 在推理中的优点是显存占用从 $O(L^2)$ 降到 $O(L)$，长序列推理的显存瓶颈大幅缓解，且与 KV 缓存和分页注意力兼容；缺点是实现复杂，通常需要自定义 CUDA 内核，且在小 batch 或短序列下优势不明显。

&emsp;&emsp;从维度视角看，FlashAttention 在推理中改变了注意力在 token 维上的执行方式：不物化完整的 $L \times L$ 关系矩阵，而是按块流式生成和消费，使 token 维扩散从内存受限转为计算受限。

**&emsp;&emsp;PyTorch 实现示例**

```python
import math
import torch

def flash_attention_inference(Q, K_cache, V_cache, block_size=64):
    """
    推理时的 FlashAttention 模拟: 分块 + 在线 Softmax
    Q: (B, h, 1, d_k) 当前 token 的查询
    K_cache, V_cache: (B, h, L, d_k) 历史缓存
    """
    B, h, _, d_k = Q.shape
    L = K_cache.size(2)
    O = torch.zeros_like(Q)
    m = torch.full((B, h, 1, 1), float('-inf'), device=Q.device)
    l = torch.zeros((B, h, 1, 1), device=Q.device)

    num_blocks = math.ceil(L / block_size)
    for j in range(num_blocks):
        start = j * block_size
        end = min(start + block_size, L)
        K_j = K_cache[:, :, start:end, :]
        V_j = V_cache[:, :, start:end, :]

        S_j = torch.matmul(Q, K_j.transpose(-2, -1)) / math.sqrt(d_k)
        m_j = S_j.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m, m_j)
        l = l * torch.exp(m - m_new) + torch.exp(S_j - m_new).sum(dim=-1, keepdim=True)
        O = O * torch.exp(m - m_new) + torch.matmul(torch.exp(S_j - m_new), V_j)
        m = m_new

    return O / l
```

&emsp;&emsp;这个实现模拟了推理时 FlashAttention 的分块在线 Softmax 流程，避免构造完整的 $L \times L$ 矩阵。

---

#### 2.13.5 分块注意力掩码

&emsp;&emsp;分块注意力掩码（Chunked Attention Mask）在推理中用于限制注意力范围，是滑动窗口注意力和流式生成的核心掩码形式。它将序列划分为固定大小的块，每个查询只能关注当前块及之前块内的 token，掩码矩阵呈块下三角结构。设块大小为 $w$，查询位置 $i$ 和键位置 $j$ 的掩码为：

$$
M_{ij} = \begin{cases}
0 & \lfloor i/w \rfloor \geq \lfloor j/w \rfloor \\
-\infty & \text{otherwise}
\end{cases}
$$

&emsp;&emsp;在推理时，分块掩码与 KV 缓存配合使用：缓存按块组织，每生成一个块，只需将新块的 K、V 追加到缓存，并更新掩码。分块掩码的优点是显存和计算量从 $O(L^2)$ 降到 $O(L \cdot w)$，适合长序列流式推理，且块结构便于硬件优化和并行计算；缺点是块大小 $w$ 需要手动选择，过小的块限制感受野，过大的块增加计算量，且跨块的长距离依赖需要通过多层堆叠间接传递。从维度视角看，分块掩码在 token 维上施加了块下三角约束，将全局扩散分解为块内扩散和块间扩散的叠加，块内注意力保持精细，块间注意力粗粒度地覆盖更远的历史。

**&emsp;&emsp;PyTorch 实现示例**

```python
import math
import torch
import torch.nn as nn

class ChunkedMaskAttention(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.chunk_size = chunk_size

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

    def _build_chunked_mask(self, L, device):
        """构建分块掩码: 查询 i 可以关注所有 j 满足 floor(i/w) >= floor(j/w)"""
        i = torch.arange(L, device=device)
        chunk_i = i // self.chunk_size
        chunk_j = chunk_i.unsqueeze(1)  # (L, 1)
        mask = chunk_j >= chunk_i.unsqueeze(0)  # (L, L)
        return mask.bool()

    def forward(self, x, causal=True):
        B, L, _ = x.size()

        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 分块掩码
        chunk_mask = self._build_chunked_mask(L, x.device)
        if causal:
            causal_mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
            chunk_mask = chunk_mask & causal_mask

        scores = scores.masked_fill(~chunk_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)
        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return torch.matmul(head_out, self.W_o), attn
```

&emsp;&emsp;这个实现中，`_build_chunked_mask` 构建块下三角掩码，每个查询只能关注当前块及之前块内的所有 token。若输入形状为 $(B,L,d_{model})$，输出形状相同，`attn` 形状为 $(B,h,L,L)$，但有效计算量为 $O(L \cdot w)$。


---

### 2.14 长上下文技术

#### 2.14.1 上下文窗口扩展

&emsp;&emsp;上下文窗口扩展（Context Window Extension）是指在不重新预训练或仅少量微调的情况下，将模型可处理的序列长度从训练时的窗口扩展到更长范围。标准 Transformer 在训练时使用固定长度的位置编码，当推理序列超过训练长度时，位置编码的数值范围会超出训练时覆盖的区间，导致注意力分布偏移、困惑度急剧上升。常见的扩展方法包括位置插值、RoPE 基频调整、分块注意力、稀疏与混合注意力、以及 NoPE 与 RoPE 交错等。

&emsp;&emsp;设训练窗口为 $L_{\mathrm{train}}$，目标窗口为 $L_{\mathrm{target}}$，缩放因子为 $s = L_{\mathrm{target}} / L_{\mathrm{train}}$。位置插值将位置索引压缩：

$$
m' = \frac{m}{s}
$$

&emsp;&emsp;然后将 $m'$ 代入 RoPE 或位置编码。基频调整则修改 RoPE 的频率参数，使不同频率维度在更长序列上保持合理的相位覆盖。分块和稀疏方法则通过限制注意力范围或压缩 KV 来降低长序列的计算和显存开销。

&emsp;&emsp;上下文窗口扩展的优点是无需从头训练即可显著延长模型可用上下文，位置插值和基频调整只需少量微调甚至零样本即可生效，分块和稀疏方法还能同时降低长序列推理成本；缺点是外推能力受训练分布限制，极端扩展比例下仍会出现性能下降，不同方法需要不同的超参数，且长上下文任务本身对模型的信息检索和聚合能力要求很高，仅靠位置编码扩展并不足以保证效果。

&emsp;&emsp;从维度视角看，上下文窗口扩展在 token 维上改变了位置编码的有效范围，使注意力关系矩阵中的位置信号在更长序列上仍然可区分。位置插值压缩了 token 维的位置尺度，基频调整重新分配了特征维上不同频率维度的旋转速度。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出位置插值与上下文扩展的裸实现：

```python
import torch
import torch.nn as nn

class PositionInterpolationRoPE(nn.Module):
    def __init__(self, d_k, base=10000.0, train_len=4096, target_len=32768):
        super().__init__()
        self.d_k = d_k
        self.train_len = train_len
        self.target_len = target_len
        self.scale = target_len / train_len
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, k, positions=None):
        B, h, L, d_k = q.size()
        if positions is None:
            positions = torch.arange(L, device=q.device).unsqueeze(0).expand(B, -1)
        # 位置插值：除以缩放因子
        positions = positions.float() / self.scale
        angles = positions.unsqueeze(-1) * self.inv_freq
        cos = torch.cos(angles).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(1)

        q_r = q.view(B, h, L, d_k // 2, 2)
        k_r = k.view(B, h, L, d_k // 2, 2)

        q_rot = torch.stack([
            q_r[..., 0] * cos - q_r[..., 1] * sin,
            q_r[..., 0] * sin + q_r[..., 1] * cos
        ], dim=-1).reshape(B, h, L, d_k)

        k_rot = torch.stack([
            k_r[..., 0] * cos - k_r[..., 1] * sin,
            k_r[..., 0] * sin + k_r[..., 1] * cos
        ], dim=-1).reshape(B, h, L, d_k)

        return q_rot, k_rot
```

&emsp;&emsp;这个实现中，位置索引被除以缩放因子后再计算 RoPE 旋转角。若输入形状为 $(B,h,L,d_k)$，输出形状相同。

---

#### 2.14.2 RoPE 基频调整

&emsp;&emsp;RoPE 基频调整（RoPE Base Frequency Adjustment）通过修改 RoPE 的基础频率 $base$ 来扩展上下文窗口。标准 RoPE 使用 $base=10000$，当序列长度远超训练长度时，低频维度的旋转角度会超出训练范围，导致注意力分数分布偏移。NTK-aware 方法将基频放大：

$$
base' = base \cdot s^{d_k / (d_k - 2)}
$$

&emsp;&emsp;其中 $s = L_{\mathrm{target}} / L_{\mathrm{train}}$。这种调整等效于对高频维度保留更多分辨率，对低频维度进行更多压缩，从而在扩展上下文的同时保留局部结构。RoPE 基频调整的优点是无需额外参数，仅修改频率计算即可显著改善长序列外推，且与位置插值相比能更好地保留高频细节；缺点是缩放因子需要手动设置，不同模型和训练长度下最优 $base'$ 不同，且极端扩展比例下仍可能失效。

&emsp;&emsp;从维度视角看，基频调整改变了 RoPE 中不同维度旋转速度的分布，使低频维度在更长序列上仍具有合理的相位覆盖，避免注意力分数在长距离上出现分布崩塌。

**&emsp;&emsp;PyTorch 实现示例**

```python
import math
import torch
import torch.nn as nn

class RoPEWithNTKScaling(nn.Module):
    def __init__(self, d_k, base=10000.0, scaling_factor=1.0):
        super().__init__()
        self.d_k = d_k
        self.base = base
        self.scaling_factor = scaling_factor
        adjusted_base = base * (scaling_factor ** (d_k / (d_k - 2)))
        inv_freq = 1.0 / (adjusted_base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, k, positions=None):
        B, h, L, d_k = q.size()
        if positions is None:
            positions = torch.arange(L, device=q.device).unsqueeze(0).expand(B, -1)
        angles = positions.unsqueeze(-1).float() * self.inv_freq
        cos = torch.cos(angles).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(1)

        q_r = q.view(B, h, L, d_k // 2, 2)
        k_r = k.view(B, h, L, d_k // 2, 2)

        q_rot = torch.stack([
            q_r[..., 0] * cos - q_r[..., 1] * sin,
            q_r[..., 0] * sin + q_r[..., 1] * cos
        ], dim=-1).reshape(B, h, L, d_k)

        k_rot = torch.stack([
            k_r[..., 0] * cos - k_r[..., 1] * sin,
            k_r[..., 0] * sin + k_r[..., 1] * cos
        ], dim=-1).reshape(B, h, L, d_k)

        return q_rot, k_rot
```

&emsp;&emsp;这个实现中，`adjusted_base` 根据缩放因子计算 NTK-aware 的新基频。若输入形状为 $(B,h,L,d_k)$，输出形状相同。

---

#### 2.14.3 YARN

&emsp;&emsp;YARN（Yet another RoPE extension）在 NTK-aware 基频调整的基础上引入分频率插值和注意力温度缩放。标准位置插值均匀压缩所有频率，导致高频维度局部细节分辨率下降；YARN 将频率维度分为三组：高频维度保持原值，低频维度进行完整插值，中间频率平滑过渡。同时引入温度因子 $t$ 校准 Softmax 分布：

$$
t = 0.1 \ln(s) + 1.0
$$

&emsp;&emsp;注意力分数缩放为：

$$
\mathrm{softmax}\left(\frac{QK^\top}{t \cdot \sqrt{d_k}}\right)
$$

&emsp;&emsp;YARN 的优点是分频率插值保留了高频局部精度，温度缩放补偿了长序列下注意力熵的变化，在 2 到 8 倍扩展时达到最优困惑度，且支持零样本推理时扩展；缺点是需要手动设置分频率阈值，温度因子的经验公式在极端缩放比例下可能不够精确。

&emsp;&emsp;从维度视角看，YARN 在不同频率维度上施加不同程度的缩放：高频维度保留短距离精度，低频维度负责长距离覆盖，温度缩放则校准 Softmax 分布的形状。

**&emsp;&emsp;PyTorch 实现示例**

```python
import math
import torch
import torch.nn as nn

class YaRNRoPE(nn.Module):
    def __init__(self, d_k, base=10000.0, original_max_len=4096,
                 extended_max_len=32768, beta_fast=32, beta_slow=1):
        super().__init__()
        self.d_k = d_k
        self.scale = extended_max_len / original_max_len
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        wavelengths = 2 * math.pi / inv_freq
        gamma = ((wavelengths - beta_slow) / (beta_fast - beta_slow)).clamp(0, 1)
        inv_freq_scaled = (1 - gamma) * inv_freq + gamma * inv_freq / self.scale
        self.register_buffer("inv_freq_scaled", inv_freq_scaled)
        self.attn_scale = 0.1 * math.log(self.scale) + 1.0

    def forward(self, q, k, positions=None):
        B, h, L, d_k = q.size()
        if positions is None:
            positions = torch.arange(L, device=q.device).unsqueeze(0).expand(B, -1)
        angles = positions.unsqueeze(-1).float() * self.inv_freq_scaled
        cos = torch.cos(angles).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(1)

        q_r = q.view(B, h, L, d_k // 2, 2)
        k_r = k.view(B, h, L, d_k // 2, 2)

        q_rot = torch.stack([
            q_r[..., 0] * cos - q_r[..., 1] * sin,
            q_r[..., 0] * sin + q_r[..., 1] * cos
        ], dim=-1).reshape(B, h, L, d_k)

        k_rot = torch.stack([
            k_r[..., 0] * cos - k_r[..., 1] * sin,
            k_r[..., 0] * sin + k_r[..., 1] * cos
        ], dim=-1).reshape(B, h, L, d_k)

        return q_rot, k_rot, self.attn_scale
```

&emsp;&emsp;这个实现中，`inv_freq_scaled` 根据波长的分频率策略对逆频率进行非均匀缩放，`attn_scale` 是温度因子 $t$。注意力分数需要除以 $t \cdot \sqrt{d_k}$。

---

#### 2.14.4 DCA

&emsp;&emsp;DCA（Dual Chunk Attention，双块注意力）是一种无需训练即可扩展上下文窗口的方法。它将长序列的注意力计算分解为块内注意力和块间注意力：块内注意力处理同一块内的 token，维持原始相对位置；块间注意力处理不同块之间的 token，通过压缩位置索引避免超出预训练范围。设块大小为 $w$，序列被分割为 $C = \lceil L/w \rceil$ 个块。对于查询块 $c$ 和键块 $j$（$j < c$），查询使用位置索引 $c-1$ 对应的位置来关注之前的块，使相对位置被限制在预训练窗口内。

&emsp;&emsp;DCA 的优点是训练无关，可直接应用于现有预训练模型，将 4K 上下文扩展到 100K+ token，计算复杂度从 $O(L^2)$ 降至 $O(L \cdot w)$，且与 FlashAttention 兼容；缺点是块大小选择需要权衡局部精度和全局覆盖，块间位置映射可能损失部分跨块相对位置的精细信息。

&emsp;&emsp;从维度视角看，DCA 在 token 维扩散的关系矩阵上施加了块结构：块内保持原始相对位置，块间通过压缩位置索引维持长程依赖。

**&emsp;&emsp;PyTorch 实现示例**

```python
import math
import torch
import torch.nn as nn

class DualChunkAttention(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.chunk_size = chunk_size

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

    def forward(self, x, causal=True):
        B, L, _ = x.size()
        w = self.chunk_size
        num_chunks = math.ceil(L / w)

        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        pos_intra = torch.arange(L, device=x.device)
        pos_inter = torch.zeros(L, device=x.device, dtype=torch.long)
        for c in range(num_chunks):
            start = c * w
            end = min(start + w, L)
            pos_intra[start:end] = torch.arange(start, end, device=x.device)
            pos_inter[start:end] = c - 1 if c > 0 else 0

        pos_diff_intra = pos_intra.unsqueeze(0) - pos_intra.unsqueeze(1)
        pos_diff_inter = pos_intra.unsqueeze(0) - pos_inter.unsqueeze(1)

        chunk_i = torch.arange(L, device=x.device) // w
        chunk_j = torch.arange(L, device=x.device) // w
        is_intra = (chunk_i.unsqueeze(1) == chunk_j.unsqueeze(0))
        pos_diff = torch.where(is_intra, pos_diff_intra, pos_diff_inter)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        dca_bias = -torch.abs(pos_diff.float()) * 0.01
        scores = scores + dca_bias.unsqueeze(0).unsqueeze(0)

        if causal:
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)
        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return torch.matmul(head_out, self.W_o), attn
```

&emsp;&emsp;这个实现中，DCA 将序列分割为块，块内使用原始位置差，块间使用压缩后的位置索引。若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.14.5 iRoPE

&emsp;&emsp;iRoPE（interleaved Rotary Position Embeddings）在模型的不同层之间交错使用两种注意力模式：一部分层使用 RoPE 并采用分块局部注意力掩码，只能关注固定窗口内的近期 token；另一部分层不使用任何位置编码（NoPE），采用完整的因果掩码，可以访问全部上下文历史。通常采用“每 4 层使用一次 RoPE”的比例，即第 1、5、9、... 层使用 RoPE 和分块注意力，其余层使用 NoPE 和全局因果注意力。

&emsp;&emsp;iRoPE 的优点是无需额外参数，通过层间交错实现了局部精度和全局覆盖的平衡，NoPE 层的全局注意力使模型能够直接访问任意距离的上下文，理论上可扩展到任意序列长度；缺点是 KV Cache 仍然随序列长度线性增长，注意力计算的 $O(n^2)$ 复杂度并未降低，内存瓶颈依然存在。

&emsp;&emsp;从维度视角看，iRoPE 在层间实现了位置编码的“交替启用与禁用”。RoPE 层在特征维上注入位置旋转，NoPE 层则让特征维纯粹承载语义信息。

**&emsp;&emsp;PyTorch 实现示例**

```python
import math
import torch
import torch.nn as nn

class iRoPEAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_rope, chunk_size=8192):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        self.chunk_size = chunk_size

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

        if use_rope:
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
            self.register_buffer("inv_freq", inv_freq)

    def _apply_rope(self, x):
        B, h, L, d_k = x.size()
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        angles = positions.unsqueeze(-1).float() * self.inv_freq
        cos = torch.cos(angles).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(1)

        x_r = x.view(B, h, L, d_k // 2, 2)
        x_rot = torch.stack([
            x_r[..., 0] * cos - x_r[..., 1] * sin,
            x_r[..., 0] * sin + x_r[..., 1] * cos
        ], dim=-1).reshape(B, h, L, d_k)
        return x_rot

    def forward(self, x):
        B, L, _ = x.size()
        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            Q = self._apply_rope(Q)
            K = self._apply_rope(K)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        if self.use_rope:
            local_mask = torch.ones(L, L, device=x.device).bool()
            for i in range(L):
                left = max(0, i - self.chunk_size + 1)
                local_mask[i, :left] = False
            mask = causal_mask | ~local_mask
        else:
            mask = causal_mask

        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)
        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return torch.matmul(head_out, self.W_o)


class iRoPEStack(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, rope_every=4, chunk_size=8192):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            use_rope = ((i + 1) % rope_every == 0)
            self.layers.append(
                iRoPEAttention(d_model, num_heads, use_rope=use_rope, chunk_size=chunk_size)
            )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x
```

&emsp;&emsp;这个实现中，`iRoPEAttention` 根据 `use_rope` 决定是否施加 RoPE 和分块掩码，`iRoPEStack` 按照每 4 层使用一次 RoPE 的模式堆叠。

---

#### 2.14.6 稀疏与混合注意力

&emsp;&emsp;稀疏与混合注意力是长上下文推理中降低计算和显存开销的重要方向。稀疏注意力为每个查询只选择部分 key 计算注意力，混合注意力则交错使用不同压缩率或不同注意力模式的层。以 DeepSeek 的压缩稀疏注意力（CSA）和重度压缩注意力（HCA）为例，CSA 先将每 $m$ 个相邻 token 的 KV 压缩成一个压缩条目，再用 Indexer 为每个查询选出 top-$k$ 个压缩条目做注意力；HCA 将压缩比拉到 128，压缩后序列极短，直接做全量注意力。两者交错堆叠，使模型既能捕捉精细的局部依赖，又能维持全局感受野。

&emsp;&emsp;设第 $g$ 组 token 集合为 $\{t_1,\dots,t_m\}$，压缩条目为：

$$
c_g = \sum_{i=1}^{m} \alpha_i \cdot k_{t_i}, \quad \alpha_i = \mathrm{softmax}(z_i + b_i)
$$

&emsp;&emsp;然后对每个查询选出 top-$k$ 个压缩条目做注意力。稀疏与混合注意力的优点是百万上下文下的边际成本被压到可用水平，KV Cache 和 FLOPs 大幅降低；缺点是结构复杂，压缩器需要额外投影矩阵和门控，稀疏选择的不规则性给连续批处理和分页注意力带来挑战。

&emsp;&emsp;从维度视角看，CSA 和 HCA 都是在 token 维扩散之前先对 Value 端做序列维压缩，减少参与扩散的 token 数量。CSA 保留较多 token 但用稀疏选择控制计算量，HCA 直接用激进压缩减少 token 数量。

**&emsp;&emsp;PyTorch 实现示例**

```python
import math
import torch
import torch.nn as nn

class CompressedSparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, compress_ratio=4, top_k=64):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.compress_ratio = compress_ratio
        self.top_k = top_k

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        self.W_compress = nn.Parameter(torch.empty(d_model, d_model))
        self.b_compress = nn.Parameter(torch.zeros(d_model))
        self.W_idx = nn.Parameter(torch.empty(d_model, d_model))

        for w in (self.W_q, self.W_k, self.W_v, self.W_o, self.W_compress, self.W_idx):
            nn.init.xavier_uniform_(w)

    def _compress(self, x):
        B, L, D = x.size()
        m = self.compress_ratio
        L_pad = math.ceil(L / m) * m
        if L_pad > L:
            x = torch.cat([x, torch.zeros(B, L_pad - L, D, device=x.device)], dim=1)
        x = x.view(B, L_pad // m, m, D)
        scores = torch.matmul(x, self.W_compress) + self.b_compress
        weights = torch.softmax(scores, dim=2)
        return (x * weights).sum(dim=2)

    def forward(self, x):
        B, L, _ = x.size()
        x_k = self._compress(x)
        x_v = self._compress(x)
        L_c = x_k.size(1)

        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x_k, self.W_k).view(B, L_c, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x_v, self.W_v).view(B, L_c, self.num_heads, self.d_k).transpose(1, 2)

        idx_score = torch.matmul(x, self.W_idx)
        idx_score = torch.matmul(idx_score, K.mean(dim=1).transpose(-2, -1))
        k = min(self.top_k, L_c)
        topk_idx = idx_score.topk(k, dim=-1).indices

        gather_idx = topk_idx.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        K_g = K.gather(2, gather_idx)
        V_g = V.gather(2, gather_idx)

        scores = torch.matmul(Q.unsqueeze(3), K_g.transpose(-2, -1)).squeeze(3) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn.unsqueeze(-2), V_g).squeeze(-2)
        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return torch.matmul(head_out, self.W_o), attn
```

&emsp;&emsp;这个实现中，CSA 先压缩 KV，再对每个查询选出 top-$k$ 个压缩条目做注意力。若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.14.7 温度缩放与分块注意力掩码

&emsp;&emsp;温度缩放与分块注意力掩码是长上下文推理中配合使用的两个组件。分块注意力掩码限制每个 token 只关注其所在块及之前固定窗口内的 token，将局部注意力的计算量控制在 $O(L \cdot w)$ 内。温度缩放在推理时对注意力 logits 进行动态调整，随着序列长度增加，通过调整 Softmax 的平滑程度来补偿注意力分布的变化。Llama 4 的温度缩放公式为：

$$
\mathrm{scale} = \log\left(\left\lfloor \frac{\mathrm{position} + 1}{\mathrm{floor\_scale}} \right\rfloor + 1\right) \cdot \mathrm{attn\_scale} + 1
$$

&emsp;&emsp;其中 $\mathrm{floor\_scale}$ 控制从多长位置开始放大，$\mathrm{attn\_scale}$ 控制放大强度。当位置小于 $\mathrm{floor\_scale}$ 时，$\mathrm{scale} \approx 1$，不产生缩放效果；当位置远超 $\mathrm{floor\_scale}$ 时，温度因子对数增长，使注意力分布更加平滑。温度缩放与分块注意力掩码的优点是推理时无需额外训练即可启用，分块掩码降低局部注意力计算量，温度缩放稳定长距离注意力分布；缺点是分块掩码的块大小需要手动设置，温度缩放的经验参数也需要根据模型规模调整。

&emsp;&emsp;从维度视角看，分块注意力掩码在 token 维扩散的关系矩阵上施加了块对角约束，使 RoPE 层专注于块内的精细位置关系；温度缩放在 NoPE 层中调整注意力分布的锐度，使全局扩散在超长序列上保持数值稳定。

**&emsp;&emsp;PyTorch 实现示例**

```python
import math
import torch
import torch.nn as nn

class TemperatureScaledChunkedAttention(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=8192,
                 floor_scale=8192, attn_scale=0.1, use_rope=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.chunk_size = chunk_size
        self.floor_scale = floor_scale
        self.attn_scale = attn_scale
        self.use_rope = use_rope

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(w)

        if use_rope:
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
            self.register_buffer("inv_freq", inv_freq)

    def _apply_rope(self, x, positions):
        B, h, L, d_k = x.size()
        angles = positions.unsqueeze(-1).float() * self.inv_freq
        cos = torch.cos(angles).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(1)
        x_r = x.view(B, h, L, d_k // 2, 2)
        x_rot = torch.stack([
            x_r[..., 0] * cos - x_r[..., 1] * sin,
            x_r[..., 0] * sin + x_r[..., 1] * cos
        ], dim=-1).reshape(B, h, L, d_k)
        return x_rot

    def _compute_temperature_scale(self, L, device):
        positions = torch.arange(L, device=device).float()
        scale = torch.log(
            torch.floor((positions + 1.0) / self.floor_scale) + 1.0
        ) * self.attn_scale + 1.0
        return scale

    def _build_chunked_mask(self, L, device):
        w = self.chunk_size
        chunk_ids = torch.arange(L, device=device) // w
        mask = chunk_ids.unsqueeze(0) >= chunk_ids.unsqueeze(1)
        return mask.bool()

    def forward(self, x):
        B, L, _ = x.size()
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

        Q = torch.matmul(x, self.W_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.W_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.W_v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            Q = self._apply_rope(Q, positions)
            K = self._apply_rope(K, positions)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if not self.use_rope:
            temp_scale = self._compute_temperature_scale(L, x.device)
            scores = scores * temp_scale.view(1, 1, L, 1)

        if self.use_rope:
            chunk_mask = self._build_chunked_mask(L, x.device)
            causal_mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
            final_mask = chunk_mask & causal_mask
            scores = scores.masked_fill(~final_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, V)
        head_out = head_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return torch.matmul(head_out, self.W_o), attn
```

&emsp;&emsp;这个实现中，当 `use_rope=True` 时施加分块注意力掩码和因果掩码的叠加，当 `use_rope=False` 时施加全因果掩码和温度缩放。若输入形状为 $(B,L,d_{model})$，输出形状相同，`attn` 形状为 $(B,h,L,L)$。


---

### 2.15 多模态架构

#### 2.15.1 拼接式多模态

&emsp;&emsp;拼接式多模态（Concatenation-based Multimodal Fusion）是最早出现、也是实现最简单的一类多模态融合方式。其核心思想是将不同模态的特征在序列维或特征维上直接拼接，然后送入统一的 Transformer 进行建模。以视觉-语言任务为例，图像经过视觉编码器得到一组视觉 token 表示 $V \in \mathbb{R}^{n_v \times d}$，文本经过词嵌入得到文本 token 表示 $T \in \mathbb{R}^{n_t \times d}$，然后将二者在序列维拼接：

$$
X = \mathrm{Concat}(V, T) \in \mathbb{R}^{(n_v + n_t) \times d}
$$

&emsp;&emsp;拼接后的序列送入标准的自注意力层，视觉 token 和文本 token 在同一个注意力矩阵中相互交互。为了区分模态，通常还会为视觉 token 和文本 token 分别添加模态类型嵌入，或使用模态特定的位置编码。拼接式多模态的优点是结构简单，直接复用标准 Transformer，无需设计复杂的跨模态模块，且自注意力天然支持任意位置之间的交互；缺点是视觉 token 和文本 token 的表示空间可能不一致，简单拼接后模型需要额外学习对齐，且视觉 token 数量通常远大于文本 token，导致注意力计算中视觉部分占据主导，文本信号容易被稀释。从维度视角看，拼接式多模态在 token 维上将两个不同来源的序列合并为一个长序列，使注意力关系矩阵同时覆盖视觉-视觉、文本-文本和视觉-文本三种交互。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出拼接式多模态的裸实现，将视觉 token 和文本 token 拼接后送入自注意力：

```python
import math
import torch
import torch.nn as nn

class ConcatenationMultimodal(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff,
                 visual_dim, text_vocab, max_len=512):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 视觉投影: 将视觉特征映射到 d_model
        self.visual_proj = nn.Parameter(torch.empty(visual_dim, d_model))
        # 文本嵌入
        self.text_emb = nn.Parameter(torch.empty(text_vocab, d_model))
        # 模态类型嵌入: 0 为视觉, 1 为文本
        self.modal_emb = nn.Parameter(torch.empty(2, d_model))
        self.pos_emb = nn.Parameter(torch.empty(max_len, d_model))

        nn.init.xavier_uniform_(self.visual_proj)
        nn.init.normal_(self.text_emb, std=0.02)
        nn.init.normal_(self.modal_emb, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "W_q": nn.Parameter(torch.empty(d_model, d_model)),
                "W_k": nn.Parameter(torch.empty(d_model, d_model)),
                "W_v": nn.Parameter(torch.empty(d_model, d_model)),
                "W_o": nn.Parameter(torch.empty(d_model, d_model)),
                "W_1": nn.Parameter(torch.empty(d_model, d_ff)),
                "b_1": nn.Parameter(torch.zeros(d_ff)),
                "W_2": nn.Parameter(torch.empty(d_ff, d_model)),
                "b_2": nn.Parameter(torch.zeros(d_model)),
                "ln1_w": nn.Parameter(torch.ones(d_model)),
                "ln1_b": nn.Parameter(torch.zeros(d_model)),
                "ln2_w": nn.Parameter(torch.ones(d_model)),
                "ln2_b": nn.Parameter(torch.zeros(d_model)),
            })
            for _ in range(num_layers)
        ])
        for layer in self.layers:
            for name, p in layer.items():
                if "W_" in name:
                    nn.init.xavier_uniform_(p)

        self.ln_f_w = nn.Parameter(torch.ones(d_model))
        self.ln_f_b = nn.Parameter(torch.zeros(d_model))

    def _ln(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def forward(self, visual_features, text_ids):
        """
        visual_features: (B, n_v, visual_dim)
        text_ids: (B, n_t) 文本 token ID
        """
        B, n_v, _ = visual_features.size()
        n_t = text_ids.size(1)

        # 视觉投影 + 模态嵌入
        V = torch.matmul(visual_features, self.visual_proj)  # (B, n_v, d_model)
        V = V + self.modal_emb[0].unsqueeze(0).unsqueeze(0)

        # 文本嵌入 + 模态嵌入
        T = self.text_emb[text_ids]  # (B, n_t, d_model)
        T = T + self.modal_emb[1].unsqueeze(0).unsqueeze(0)

        # 拼接
        x = torch.cat([V, T], dim=1)  # (B, n_v + n_t, d_model)
        L = x.size(1)
        x = x + self.pos_emb[:L].unsqueeze(0)

        # 标准自注意力
        for layer in self.layers:
            Q = torch.matmul(x, layer["W_q"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(x, layer["W_k"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            V_attn = torch.matmul(x, layer["W_v"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = torch.softmax(scores, dim=-1)
            head_out = torch.matmul(attn, V_attn).transpose(1, 2).contiguous().view(B, L, self.d_model)
            attn_out = torch.matmul(head_out, layer["W_o"])
            x = self._ln(x + attn_out, layer["ln1_w"], layer["ln1_b"])

            ffn_out = torch.matmul(
                torch.relu(torch.matmul(x, layer["W_1"]) + layer["b_1"]),
                layer["W_2"]
            ) + layer["b_2"]
            x = self._ln(x + ffn_out, layer["ln2_w"], layer["ln2_b"])

        x = self._ln(x, self.ln_f_w, self.ln_f_b)
        return x
```

&emsp;&emsp;若 `visual_features` 形状为 $(B,n_v,d_v)$，`text_ids` 形状为 $(B,n_t)$，输出形状为 $(B,n_v+n_t,d_{model})$。

---

#### 2.15.2 早期融合

&emsp;&emsp;早期融合（Early Fusion）指在模型输入层或浅层就将不同模态的信息合并，与拼接式多模态有重叠，但更强调在特征提取的早期阶段进行跨模态交互。典型做法是：在视觉编码器和文本编码器的浅层之间加入跨模态注意力，或在输入嵌入阶段就将视觉特征和文本特征通过门控或加权求和融合。设视觉特征为 $v$，文本特征为 $t$，早期融合的一种形式为：

$$
z = \alpha \cdot W_v v + (1 - \alpha) \cdot W_t t
$$

&emsp;&emsp;其中 $\alpha$ 是可学习的门控权重。另一种常见做法是在浅层 Transformer 中交替使用视觉自注意力、文本自注意力和跨模态注意力，使模态间的信息在早期就开始交换。早期融合的优点是模态间交互发生得早，模型有更多机会学习细粒度的跨模态对齐，且浅层融合的计算开销通常小于深层融合；缺点是视觉和文本的表示在早期尚未充分编码，过早混合可能引入噪声，且不同模态的编码速度不同，强行同步可能限制各自编码器的表达能力。从维度视角看，早期融合在特征维上尽早将不同模态的表示投影到同一空间并混合，使后续的 token 维扩散从一开始就包含跨模态信息。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出早期融合的裸实现，在输入层通过门控加权融合视觉和文本特征：

```python
import torch
import torch.nn as nn

class EarlyFusion(nn.Module):
    def __init__(self, d_model, visual_dim, text_vocab):
        super().__init__()
        self.visual_proj = nn.Parameter(torch.empty(visual_dim, d_model))
        self.text_emb = nn.Parameter(torch.empty(text_vocab, d_model))
        # 门控网络
        self.gate = nn.Parameter(torch.empty(2 * d_model, 1))
        self.b_gate = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.visual_proj)
        nn.init.normal_(self.text_emb, std=0.02)
        nn.init.xavier_uniform_(self.gate)

    def forward(self, visual_features, text_ids):
        """
        visual_features: (B, n_v, visual_dim)
        text_ids: (B, n_t)
        """
        V = torch.matmul(visual_features, self.visual_proj)  # (B, n_v, d_model)
        T = self.text_emb[text_ids]  # (B, n_t, d_model)

        # 对视觉 token 和文本 token 分别做池化，得到全局表示
        v_pool = V.mean(dim=1)  # (B, d_model)
        t_pool = T.mean(dim=1)  # (B, d_model)

        # 门控: 计算融合权重
        gate_input = torch.cat([v_pool, t_pool], dim=-1)  # (B, 2*d_model)
        alpha = torch.sigmoid(torch.matmul(gate_input, self.gate) + self.b_gate)  # (B, 1)

        # 融合后的全局表示
        fused = alpha.unsqueeze(1) * v_pool.unsqueeze(1) + (1 - alpha.unsqueeze(1)) * t_pool.unsqueeze(1)

        # 将融合表示拼接到视觉和文本序列前
        V_fused = torch.cat([fused, V], dim=1)  # (B, 1+n_v, d_model)
        T_fused = torch.cat([fused, T], dim=1)  # (B, 1+n_t, d_model)

        return V_fused, T_fused
```

&emsp;&emsp;这个实现中，门控网络根据视觉和文本的全局池化表示计算融合权重，融合后的表示被拼接到两个模态序列的前端。

---

#### 2.15.3 端到端统一

&emsp;&emsp;端到端统一（End-to-End Unified Multimodal Model）指用一个单一的 Transformer 模型同时处理多种模态的输入和输出，不区分独立的视觉编码器和语言模型，而是将图像、文本、音频等全部离散化为 token，在同一个自回归框架下训练。典型代表包括 Chameleon、AnyGPT 等。其核心思想是将图像通过 VQ-VAE 或 VQ-GAN 量化为离散视觉 token，与文本 token 共享同一个词表或使用独立的模态标记，然后统一进行自回归建模：

$$
P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

&emsp;&emsp;其中 $x_t$ 可以是文本 token 或视觉 token。端到端统一的优点是架构完全统一，训练和推理流程一致，模型可以自由地在模态之间转换，且多模态能力通过同一套参数自然涌现；缺点是视觉 token 的离散化会损失信息，视觉 token 序列通常很长，训练成本高，且不同模态的 token 分布差异大，共享词表可能导致模态间的干扰。从维度视角看，端到端统一将所有模态都映射到同一个离散 token 维，使 token 维扩散在统一的空间中进行，特征维上的表示需要同时编码多种模态的语义。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出端到端统一模型的简化裸实现，使用共享词表处理文本和视觉 token：

```python
import math
import torch
import torch.nn as nn

class UnifiedMultimodalLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 visual_codebook_size, max_len=1024):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.vocab_size = vocab_size
        self.visual_codebook_size = visual_codebook_size
        # 共享词表: 文本 token + 视觉 token + 特殊标记
        self.total_vocab = vocab_size + visual_codebook_size + 2
        self.token_emb = nn.Parameter(torch.empty(self.total_vocab, d_model))
        self.pos_emb = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.token_emb, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "W_q": nn.Parameter(torch.empty(d_model, d_model)),
                "W_k": nn.Parameter(torch.empty(d_model, d_model)),
                "W_v": nn.Parameter(torch.empty(d_model, d_model)),
                "W_o": nn.Parameter(torch.empty(d_model, d_model)),
                "W_1": nn.Parameter(torch.empty(d_model, d_ff)),
                "b_1": nn.Parameter(torch.zeros(d_ff)),
                "W_2": nn.Parameter(torch.empty(d_ff, d_model)),
                "b_2": nn.Parameter(torch.zeros(d_model)),
                "ln1_w": nn.Parameter(torch.ones(d_model)),
                "ln1_b": nn.Parameter(torch.zeros(d_model)),
                "ln2_w": nn.Parameter(torch.ones(d_model)),
                "ln2_b": nn.Parameter(torch.zeros(d_model)),
            })
            for _ in range(num_layers)
        ])
        for layer in self.layers:
            for name, p in layer.items():
                if "W_" in name:
                    nn.init.xavier_uniform_(p)

        self.ln_f_w = nn.Parameter(torch.ones(d_model))
        self.ln_f_b = nn.Parameter(torch.zeros(d_model))
        self.lm_head = nn.Parameter(torch.empty(d_model, self.total_vocab))
        nn.init.normal_(self.lm_head, std=0.02)

    def _ln(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def forward(self, input_ids):
        B, L = input_ids.size()
        x = self.token_emb[input_ids] + self.pos_emb[:L].unsqueeze(0)

        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            Q = torch.matmul(x, layer["W_q"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(x, layer["W_k"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(x, layer["W_v"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            head_out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
            attn_out = torch.matmul(head_out, layer["W_o"])
            x = self._ln(x + attn_out, layer["ln1_w"], layer["ln1_b"])

            ffn_out = torch.matmul(
                torch.relu(torch.matmul(x, layer["W_1"]) + layer["b_1"]),
                layer["W_2"]
            ) + layer["b_2"]
            x = self._ln(x + ffn_out, layer["ln2_w"], layer["ln2_b"])

        x = self._ln(x, self.ln_f_w, self.ln_f_b)
        logits = torch.matmul(x, self.lm_head)
        return logits
```

&emsp;&emsp;这个实现中，文本 token 和视觉 token 共享同一个嵌入矩阵和语言模型头，视觉 token 的 ID 从 `vocab_size` 开始偏移。若输入形状为 $(B,L)$，输出形状为 $(B,L,\text{total\_vocab})$。

---

#### 2.15.4 跨模态注意力

&emsp;&emsp;跨模态注意力（Cross-Modal Attention）是专门用于在不同模态之间交换信息的注意力机制。与自注意力中 Q、K、V 来自同一序列不同，跨模态注意力中 Query 来自一个模态，Key 和 Value 来自另一个模态。设视觉特征为 $V \in \mathbb{R}^{n_v \times d}$，文本特征为 $T \in \mathbb{R}^{n_t \times d}$，文本到视觉的跨模态注意力为：

$$
Q = T W_Q, \quad K = V W_K, \quad V_{\text{attn}} = V W_V
$$

$$
\mathrm{CrossAttn}(T, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V_{\text{attn}}
$$

&emsp;&emsp;同样可以定义视觉到文本的跨模态注意力。在 Transformer 中，跨模态注意力通常插入在自注意力之后，与自注意力交替堆叠。跨模态注意力的优点是显式建模模态间的对齐关系，使一个模态的表示可以直接查询另一个模态的信息，适合视觉问答、图像描述、多模态检索等任务；缺点是跨模态注意力的计算复杂度为 $O(n_v \cdot n_t)$，当视觉 token 数量很大时开销显著，且需要额外设计模态间的对齐目标和训练策略。从维度视角看，跨模态注意力在 token 维上建立了两个不同模态序列之间的关系矩阵，使一个模态的 token 维扩散可以吸收另一个模态的信息。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出跨模态注意力的裸实现，包含文本到视觉和视觉到文本两个方向：

```python
import math
import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 文本到视觉
        self.W_q_t = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o_t = nn.Parameter(torch.empty(d_model, d_model))

        # 视觉到文本
        self.W_q_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k_t = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v_t = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o_v = nn.Parameter(torch.empty(d_model, d_model))

        for w in (self.W_q_t, self.W_k_v, self.W_v_v, self.W_o_t,
                  self.W_q_v, self.W_k_t, self.W_v_t, self.W_o_v):
            nn.init.xavier_uniform_(w)

    def _attn(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    def forward(self, text_features, visual_features):
        """
        text_features: (B, n_t, d_model)
        visual_features: (B, n_v, d_model)
        """
        B, n_t, _ = text_features.size()
        n_v = visual_features.size(1)

        # 文本到视觉
        Q_t = torch.matmul(text_features, self.W_q_t).view(B, n_t, self.num_heads, self.d_k).transpose(1, 2)
        K_v = torch.matmul(visual_features, self.W_k_v).view(B, n_v, self.num_heads, self.d_k).transpose(1, 2)
        V_v = torch.matmul(visual_features, self.W_v_v).view(B, n_v, self.num_heads, self.d_k).transpose(1, 2)
        t2v = self._attn(Q_t, K_v, V_v)
        t2v = t2v.transpose(1, 2).contiguous().view(B, n_t, self.d_model)
        t2v = torch.matmul(t2v, self.W_o_t)

        # 视觉到文本
        Q_v = torch.matmul(visual_features, self.W_q_v).view(B, n_v, self.num_heads, self.d_k).transpose(1, 2)
        K_t = torch.matmul(text_features, self.W_k_t).view(B, n_t, self.num_heads, self.d_k).transpose(1, 2)
        V_t = torch.matmul(text_features, self.W_v_t).view(B, n_t, self.num_heads, self.d_k).transpose(1, 2)
        v2t = self._attn(Q_v, K_t, V_t)
        v2t = v2t.transpose(1, 2).contiguous().view(B, n_v, self.d_model)
        v2t = torch.matmul(v2t, self.W_o_v)

        return t2v, v2t
```

&emsp;&emsp;这个实现中，`t2v` 是文本查询视觉得到的表示，`v2t` 是视觉查询文本得到的表示。若文本特征形状为 $(B,n_t,d_{model})$，视觉特征形状为 $(B,n_v,d_{model})$，输出形状分别为 $(B,n_t,d_{model})$ 和 $(B,n_v,d_{model})$。

---

#### 2.15.5 视觉编码器

&emsp;&emsp;视觉编码器（Vision Encoder）是多模态模型中负责将图像转换为 token 序列的模块。常见的选择包括 ViT（Vision Transformer）、CLIP ViT、SigLIP 等。ViT 将图像划分为固定大小的 patch，每个 patch 展平后经过线性投影得到 patch 嵌入，加上位置编码后送入标准 Transformer 编码器。设图像为 $I \in \mathbb{R}^{H \times W \times C}$，patch 大小为 $P \times P$，则 patch 数量为：

$$
N = \frac{H}{P} \times \frac{W}{P}
$$

&emsp;&emsp;每个 patch 展平为 $P^2 C$ 维向量，经过线性投影得到 $d$ 维嵌入：

$$
x_i = W_{\text{patch}} \cdot \mathrm{Flatten}(\mathrm{Patch}_i) + b_{\text{patch}}
$$

&emsp;&emsp;然后加上位置编码，送入 Transformer 编码器。视觉编码器的输出是一组视觉 token，通常还会经过一个投影层映射到语言模型的维度。视觉编码器的优点是 ViT 架构与文本 Transformer 高度一致，便于统一设计和联合训练，且 patch 化使图像可以像文本一样被 token 化；缺点是 patch 数量随图像分辨率平方增长，高分辨率图像的视觉 token 序列很长，计算和显存开销大，且视觉编码器通常需要在大规模图像-文本对上预训练才能获得良好的语义表示。从维度视角看，视觉编码器将图像的二维空间结构转化为一维 token 序列，在 token 维上建立了 patch 之间的关系矩阵，特征维上学习视觉语义表示。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出简化 ViT 视觉编码器的裸实现：

```python
import math
import torch
import torch.nn as nn

class VisionEncoder(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 d_model=768, num_heads=12, num_layers=12, d_ff=3072):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Patch 嵌入: 将每个 patch 投影到 d_model
        self.patch_proj = nn.Parameter(
            torch.empty(patch_size * patch_size * in_channels, d_model)
        )
        self.patch_bias = nn.Parameter(torch.zeros(d_model))
        self.pos_emb = nn.Parameter(torch.empty(self.num_patches, d_model))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))

        nn.init.xavier_uniform_(self.patch_proj)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "W_q": nn.Parameter(torch.empty(d_model, d_model)),
                "W_k": nn.Parameter(torch.empty(d_model, d_model)),
                "W_v": nn.Parameter(torch.empty(d_model, d_model)),
                "W_o": nn.Parameter(torch.empty(d_model, d_model)),
                "W_1": nn.Parameter(torch.empty(d_model, d_ff)),
                "b_1": nn.Parameter(torch.zeros(d_ff)),
                "W_2": nn.Parameter(torch.empty(d_ff, d_model)),
                "b_2": nn.Parameter(torch.zeros(d_model)),
                "ln1_w": nn.Parameter(torch.ones(d_model)),
                "ln1_b": nn.Parameter(torch.zeros(d_model)),
                "ln2_w": nn.Parameter(torch.ones(d_model)),
                "ln2_b": nn.Parameter(torch.zeros(d_model)),
            })
            for _ in range(num_layers)
        ])
        for layer in self.layers:
            for name, p in layer.items():
                if "W_" in name:
                    nn.init.xavier_uniform_(p)

        self.ln_f_w = nn.Parameter(torch.ones(d_model))
        self.ln_f_b = nn.Parameter(torch.zeros(d_model))

    def _ln(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def forward(self, images):
        """
        images: (B, C, H, W)
        """
        B, C, H, W = images.size()
        P = self.patch_size
        n_h, n_w = H // P, W // P

        # 切分 patch: (B, C, H, W) -> (B, n_h*n_w, P*P*C)
        patches = images.unfold(2, P, P).unfold(3, P, P)  # (B, C, n_h, n_w, P, P)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, n_h * n_w, C * P * P)  # (B, N, P*P*C)

        # Patch 嵌入
        x = torch.matmul(patches, self.patch_proj) + self.patch_bias  # (B, N, d_model)
        x = x + self.pos_emb.unsqueeze(0)

        # 添加 CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, d_model)
        L = x.size(1)

        # Transformer 编码器
        for layer in self.layers:
            Q = torch.matmul(x, layer["W_q"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(x, layer["W_k"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(x, layer["W_v"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = torch.softmax(scores, dim=-1)
            head_out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
            attn_out = torch.matmul(head_out, layer["W_o"])
            x = self._ln(x + attn_out, layer["ln1_w"], layer["ln1_b"])

            ffn_out = torch.matmul(
                torch.relu(torch.matmul(x, layer["W_1"]) + layer["b_1"]),
                layer["W_2"]
            ) + layer["b_2"]
            x = self._ln(x + ffn_out, layer["ln2_w"], layer["ln2_b"])

        x = self._ln(x, self.ln_f_w, self.ln_f_b)
        return x  # (B, 1+N, d_model)
```

&emsp;&emsp;这个实现中，图像被切分为 patch，每个 patch 投影为 d_model 维嵌入，加上位置编码和 CLS token 后送入标准 Transformer 编码器。若输入图像形状为 $(B,C,H,W)$，输出形状为 $(B,1+N,d_{model})$，其中 $N = (H/P) \times (W/P)$。


---

### 2.16 缩放规律

#### 2.16.1 Kaplan Scaling

&emsp;&emsp;Kaplan Scaling 指 Kaplan 等人在 2020 年提出的大模型缩放定律。他们系统研究了语言模型性能与模型参数量 $N$、训练数据量 $D$、训练计算量 $C$ 之间的关系，发现测试损失随这三个量分别呈现幂律下降。单独看模型规模时：

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}
$$

&emsp;&emsp;单独看数据量时：

$$
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

&emsp;&emsp;单独看训练计算量时：

$$
L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

&emsp;&emsp;其中 $N_c, D_c, C_c$ 是常数，$\alpha_N, \alpha_D, \alpha_C$ 是幂律指数。联合形式通常写为：

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

&emsp;&emsp;训练计算量近似为 $C \approx 6ND$。Kaplan 的结论是，给定计算预算 $C$，模型参数量应比数据量增长得更快，约为 $N \propto C^{0.73}$，$D \propto C^{0.27}$。这意味着在有限计算下，优先增大模型而非数据。Kaplan Scaling 的优点是首次系统量化了大模型性能与规模的关系，为资源分配提供了可预测的指导；缺点是数据分配明显不足，后续 Chinchilla 实验表明 Kaplan 低估了数据量的重要性，导致同等计算下模型训练不充分。

&emsp;&emsp;从维度视角看，Kaplan Scaling 在参数量维和数据量维上分别建立了幂律衰减关系，计算量维是两者的耦合约束。它指导的是如何在模型容量和数据规模之间分配计算资源。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch

def kaplan_scaling_loss(N, D, E=1.69, A=406.4, B=410.7,
                        alpha=0.34, beta=0.28):
    """Kaplan 风格的联合缩放损失"""
    return E + A / (N ** alpha) + B / (D ** beta)

def kaplan_optimal_allocation(C, exp_N=0.73, exp_D=0.27):
    """给定计算预算 C，按 Kaplan 指数分配 N 和 D"""
    # 归一化常数，使 N*D ≈ C/6
    N = C ** exp_N
    D = C ** exp_D
    # 缩放到 6ND = C
    scale = (C / (6 * N * D)) ** 0.5
    N = N * scale
    D = D * scale
    return N, D
```

&emsp;&emsp;若给定计算预算 $C$，`kaplan_optimal_allocation` 按 Kaplan 指数返回模型参数量和数据量。

---

#### 2.16.2 Chinchilla-optimal

&emsp;&emsp;Chinchilla-optimal 是 Hoffmann 等人在 2022 年提出的训练计算最优缩放方案。他们重新审视了 Kaplan 的结论，发现模型参数量 $N$ 和训练数据量 $D$ 应当近似等比例扩展，而不是模型远大于数据。在标准缩放损失下：

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

&emsp;&emsp;约束为训练计算量：

$$
C \approx 6ND
$$

&emsp;&emsp;通过拉格朗日乘子法最小化损失，可得到最优分配：

$$
N^* \propto C^{\frac{\beta}{\alpha+\beta}}, \quad D^* \propto C^{\frac{\alpha}{\alpha+\beta}}
$$

&emsp;&emsp;Chinchilla 的实验估计 $\alpha \approx 0.34$，$\beta \approx 0.28$，指数接近，因此 $N$ 和 $D$ 近似等比例增长。经验上，每个参数约需 20 个训练 token，即 $D \approx 20N$。代入 $C \approx 6ND$ 可得：

$$
N^* \approx \sqrt{\frac{C}{120}}, \quad D^* \approx 20N^*
$$

&emsp;&emsp;Chinchilla 用 70B 参数、1.4T token 的模型击败了 280B 参数、300B token 的 Gopher，验证了数据量不足是当时大模型的主要瓶颈。Chinchilla-optimal 的优点是真正实现了训练计算最优，同等计算下性能显著优于 Kaplan 分配；缺点是只考虑训练计算，未考虑推理成本，导致按此方案训练的模型在推理时可能过大、过贵。

&emsp;&emsp;从维度视角看，Chinchilla-optimal 在参数量维和数据量维之间寻找等比例扩展点，使训练计算预算下的损失最小化。它修正了 Kaplan 在数据维上的低估。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch

def chinchilla_loss(N, D, E=1.69, A=406.4, B=410.7,
                    alpha=0.34, beta=0.28):
    return E + A / (N ** alpha) + B / (D ** beta)

def chinchilla_optimal(C, tokens_per_param=20.0):
    """给定训练计算 C，按 D = tokens_per_param * N 求最优 N"""
    # C ≈ 6 N D = 6 * tpp * N^2
    N = torch.sqrt(torch.tensor(C / (6.0 * tokens_per_param)))
    D = tokens_per_param * N
    return N, D

def chinchilla_optimal_general(C, alpha=0.34, beta=0.28):
    """通用指数形式的最优分配"""
    exp_N = beta / (alpha + beta)
    exp_D = alpha / (alpha + beta)
    N = C ** exp_N
    D = C ** exp_D
    scale = (C / (6 * N * D)) ** 0.5
    return N * scale, D * scale
```

&emsp;&emsp;`chinchilla_optimal` 按 20 token/参数给出最优 $N,D$，`chinchilla_optimal_general` 使用通用指数计算。

---

#### 2.16.3 训练时计算扩展

&emsp;&emsp;训练时计算扩展（Training-Time Compute Scaling）指通过增加训练阶段的计算量来提升模型性能。训练计算量主要由模型参数量 $N$、训练 token 数 $D$ 和训练轮数决定，近似为：

$$
C_{\mathrm{train}} \approx 6ND
$$

&emsp;&emsp;增加训练计算的方式包括：增大模型参数量、增加训练数据量、延长训练步数、使用更大的 batch size、以及采用更高效的并行策略。训练时扩展的核心问题是如何在参数量、数据量和训练步数之间分配计算预算，使损失最小。Chinchilla-optimal 给出了训练计算最优的分配比例，而实际训练还会受到数据质量、硬件通信、优化稳定性等因素的约束。

&emsp;&emsp;训练时计算扩展的优点是直接提升模型能力上限，缩放定律使性能可预测，且可以通过并行训练在集群上扩展；缺点是计算成本高昂，数据质量瓶颈日益突出，训练稳定性随规模增大而下降，且训练完成的模型在推理时可能仍然昂贵。

&emsp;&emsp;从维度视角看，训练时计算扩展在参数量维和数据量维上同时增加资源，使模型在特征维上拥有更大容量，在 token 维上见过更多样本。计算量维是这两者的乘积约束。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch

def training_flops(N, D, factor=6.0):
    """训练计算量估计: C ≈ 6ND"""
    return factor * N * D

def allocate_training_budget(C, mode="chinchilla", tokens_per_param=20.0):
    """根据训练预算分配 N 和 D"""
    if mode == "chinchilla":
        N = torch.sqrt(torch.tensor(C / (6.0 * tokens_per_param)))
        D = tokens_per_param * N
    elif mode == "kaplan":
        N = C ** 0.73
        D = C ** 0.27
        scale = (C / (6 * N * D)) ** 0.5
        N, D = N * scale, D * scale
    else:
        raise ValueError(mode)
    return N, D

def loss_prediction(N, D, E=1.69, A=406.4, B=410.7,
                    alpha=0.34, beta=0.28):
    return E + A / (N ** alpha) + B / (D ** beta)
```

&emsp;&emsp;`allocate_training_budget` 根据训练预算和分配模式返回 $N,D$，`loss_prediction` 预测对应损失。

---

#### 2.16.4 推理时计算扩展

&emsp;&emsp;推理时计算扩展（Inference-Time Compute Scaling）指在模型训练完成后，通过增加推理阶段的计算量来提升输出质量。常见方法包括：延长思维链长度、多次采样后投票、best-of-N 重排序、树搜索、以及使用验证器筛选答案。设单次推理的正确答案概率为 $p$，独立采样 $N$ 次后多数投票的正确答案概率近似为：

$$
P_{\mathrm{maj}}(N) \approx 1 - (1-p)^N
$$

&emsp;&emsp;对于 best-of-N，若验证器能以概率 $q$ 正确识别正确答案，则最终正确概率为：

$$
P_{\mathrm{best}}(N) \approx 1 - (1-pq)^N
$$

&emsp;&emsp;推理时计算扩展的优点是无需重新训练模型，仅通过推理策略就能提升性能，且可以根据任务难度动态分配计算量，简单问题快速回答，复杂问题深度推理；缺点是计算成本随采样数或推理长度线性甚至超线性增长，边际收益递减，且对于简单问题过度扩展会造成浪费。

&emsp;&emsp;从维度视角看，推理时计算扩展在 token 维上增加了生成的步数，或在输出空间中探索了多条路径，相当于用更多 token 维的计算来补偿模型固定的特征维容量。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch

def majority_vote_accuracy(p, N):
    """多数投票正确率上界"""
    return 1 - (1 - p) ** N

def best_of_n_accuracy(p, q, N):
    """best-of-N 正确率，q 为验证器识别正确率"""
    return 1 - (1 - p * q) ** N

def simulate_best_of_n(policy, reward_fn, input_ids,
                       num_samples=8, max_new_tokens=64):
    """模拟 best-of-N 采样并选奖励最高"""
    best = None
    best_r = float("-inf")
    for _ in range(num_samples):
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = policy(generated)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=-1)
        r = reward_fn(generated).item()
        if r > best_r:
            best_r = r
            best = generated
    return best, best_r
```

&emsp;&emsp;`majority_vote_accuracy` 和 `best_of_n_accuracy` 给出推理扩展的理论增益，`simulate_best_of_n` 模拟采样选择流程。

---

#### 2.16.5 Inference-optimal

&emsp;&emsp;Inference-optimal 指在考虑推理成本时的最优模型规模与训练数据分配。Chinchilla-optimal 只最小化训练损失或训练计算，但实际部署中推理成本往往远超训练成本。设模型参数量为 $N$，训练 token 数为 $D$，推理 token 数为 $R$，总计算成本近似为：

$$
C_{\mathrm{total}} \approx 6ND + 2NR
$$

&emsp;&emsp;其中 $6ND$ 是训练计算量，$2NR$ 是推理计算量。目标是在总预算 $C_{\mathrm{total}}$ 下最小化损失：

$$
\min_{N,D} \quad L(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

$$
\text{s.t.} \quad 6ND + 2NR = C_{\mathrm{total}}
$$

&emsp;&emsp;当推理 token 数 $R$ 很大时，推理项 $2NR$ 主导总成本，最优解倾向于减小模型参数量 $N$、增加训练 token 数 $D$。这意味着模型会比 Chinchilla-optimal 更小，但训练得更久，即“过度训练”小模型。LLaMA 系列就是 Inference-optimal 的典型代表：LLaMA-7B 使用约 1T token 训练，远超 Chinchilla 建议的 140B token。Inference-optimal 的优点是显著降低推理成本，适合高推理需求场景，单位推理成本更低；缺点是训练成本更高，小模型容量有限，在需要极强推理能力的任务上可能不如大模型。

&emsp;&emsp;从维度视角看，Inference-optimal 在参数量维、数据量维和推理 token 维之间做三方权衡。推理需求越大，最优模型越偏向小参数量、多数据，即在参数量维上收缩，在数据维上扩展。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch

def inference_optimal(N, D, R, E=1.69, A=406.4, B=410.7,
                      alpha=0.34, beta=0.28):
    """计算给定 N,D,R 的损失和总成本"""
    loss = E + A / (N ** alpha) + B / (D ** beta)
    cost = 6 * N * D + 2 * N * R
    return loss, cost

def search_inference_optimal(C_total, R, N_grid, tokens_per_param_grid):
    """网格搜索推理最优 N 和 D"""
    best = None
    best_loss = float("inf")
    for N in N_grid:
        for tpp in tokens_per_param_grid:
            D = tpp * N
            loss, cost = inference_optimal(N, D, R)
            if cost <= C_total and loss < best_loss:
                best_loss = loss
                best = (N, D, loss, cost)
    return best

# 示例: 总预算固定，推理 token 越多，最优 N 越小
C_total = 1e24
R_values = [0, 1e12, 1e13]
N_grid = torch.logspace(9, 11, 50)       # 1B 到 100B
tpp_grid = torch.logspace(1, 3, 50)      # 10 到 1000 token/参数
for R in R_values:
    N, D, loss, cost = search_inference_optimal(
        C_total, R, N_grid, tpp_grid
    )
    print(f"R={R:.1e}: N={N:.2e}, D={D:.2e}, loss={loss:.4f}")
```

&emsp;&emsp;`search_inference_optimal` 在给定总预算和推理 token 数下，网格搜索最优的模型参数量和数据量。随着 $R$ 增大，最优 $N$ 会减小、$D$ 会增大，体现推理最优的核心权衡。


---

### 2.17 并行与分布式训练

#### 2.17.1 数据并行

&emsp;&emsp;数据并行（Data Parallelism, DP）是分布式训练中最基础的并行策略。其核心思想是将训练数据划分到多个设备上，每个设备持有完整的模型副本，独立完成前向传播和反向传播，然后通过梯度同步使所有副本保持一致。设总 batch size 为 $B$，设备数为 $N_d$，则每个设备处理 $B/N_d$ 个样本。第 $i$ 个设备的梯度为：

$$
g_i = \nabla_\theta \mathcal{L}_i(\theta), \quad \mathcal{L}_i = \frac{1}{B/N_d} \sum_{x \in \mathcal{D}_i} \mathcal{L}(x; \theta)
$$

&emsp;&emsp;梯度同步后取平均：

$$
g = \frac{1}{N_d} \sum_{i=1}^{N_d} g_i
$$

&emsp;&emsp;然后所有设备用相同的平均梯度更新参数，保证模型副本始终一致。数据并行的优点是实现简单，几乎不需要修改模型代码，对模型结构无特殊要求，且通信量相对固定（仅梯度同步），适合计算密集型的大 batch 训练；缺点是每个设备必须容纳完整模型，显存开销大，当模型参数量超过单卡显存时无法使用，且梯度同步的通信量与参数量成正比，在低速网络下可能成为瓶颈。

&emsp;&emsp;从维度视角看，数据并行在 batch 维上切分数据，每个设备处理不同的 batch 子集，但所有设备在特征维上持有相同的模型参数。梯度同步在特征维上对所有设备的梯度做平均，确保参数更新一致。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出数据并行的裸实现，使用 `torch.distributed` 进行梯度同步：

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

class DataParallelModel(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def forward(self, x):
        hidden = torch.relu(torch.matmul(x, self.W_1) + self.b_1)
        return torch.matmul(hidden, self.W_2) + self.b_2

def all_reduce_gradients(model):
    """手动实现梯度 all-reduce 平均"""
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= dist.get_world_size()

def dp_worker(rank, world_size, d_model=256, d_ff=1024):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = DataParallelModel(d_model, d_ff)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for step in range(100):
        # 每个设备使用不同的数据子集
        x = torch.randn(32, d_model) + rank  # 数据因 rank 而异
        target = torch.randn(32, d_model)

        output = model(x)
        loss = nn.functional.mse_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        # 梯度同步
        all_reduce_gradients(model)
        optimizer.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(dp_worker, args=(world_size,), nprocs=world_size, join=True)
```

&emsp;&emsp;这个实现中，每个设备独立计算梯度后通过 `all_reduce` 同步并平均。实际使用中推荐 `torch.nn.parallel.DistributedDataParallel`，它通过梯度桶化与计算重叠来提升通信效率。

---

#### 2.17.2 张量并行

&emsp;&emsp;张量并行（Tensor Parallelism, TP）由 Megatron-LM 提出，核心思想是将单个权重矩阵沿特定维度切分到多个设备上，每个设备只计算部分结果，再通过通信合并。与数据并行不同，张量并行切分的是模型参数本身，因此可以训练单卡无法容纳的大模型。以 MLP 层 $Y = XW$ 为例，Megatron-LM 采用列并行和行并行的组合策略。

&emsp;&emsp;设权重矩阵 $W \in \mathbb{R}^{h_1 \times h_2}$，列并行将 $W$ 按列切分为 $W = [W_1, W_2, \dots, W_N]$，每个设备计算 $Y_i = XW_i$，然后拼接得到完整输出。行并行将 $W$ 按行切分为 $W = [W_1; W_2; \dots; W_N]$，每个设备计算 $Y_i = X_i W_i$，然后通过 all-reduce 求和。Megatron-LM 在第一层使用列并行，第二层使用行并行，这样前向传播中只需一次 all-reduce，反向传播中只需一次 all-reduce。

&emsp;&emsp;张量并行的优点是单卡显存需求大幅降低，使超大模型训练成为可能，且通信与计算可以重叠，效率较高；缺点是通信频繁，每层前向和反向都需要 all-reduce，对网络带宽要求高，通常只在单节点内（NVLink）使用，且需要修改模型结构以适配切分策略。

&emsp;&emsp;从维度视角看，张量并行在特征维上将权重矩阵切分到多个设备，每个设备只负责特征维的一个子空间。前向传播中通过 all-reduce 在特征维上合并各设备的局部结果，反向传播中同样通过 all-reduce 同步梯度。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出列并行和行并行的裸实现：

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """列并行: 权重按列切分，输出按列拼接"""
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.world_size = world_size
        self.out_per_partition = out_features // world_size
        self.weight = nn.Parameter(torch.empty(in_features, self.out_per_partition))
        self.bias = nn.Parameter(torch.zeros(self.out_per_partition))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # 每个设备计算自己的列分片
        out = torch.matmul(x, self.weight) + self.bias
        # 收集所有设备的输出并拼接
        out_list = [torch.zeros_like(out) for _ in range(self.world_size)]
        dist.all_gather(out_list, out)
        return torch.cat(out_list, dim=-1)


class RowParallelLinear(nn.Module):
    """行并行: 权重按行切分，输出 all-reduce 求和"""
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.world_size = world_size
        self.in_per_partition = in_features // world_size
        self.weight = nn.Parameter(torch.empty(self.in_per_partition, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # 输入按特征维切分
        x_partition = x.chunk(self.world_size, dim=-1)[dist.get_rank()]
        out = torch.matmul(x_partition, self.weight)
        # all-reduce 求和
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out + self.bias


class TensorParallelMLP(nn.Module):
    """Megatron-LM 风格 MLP: 第一层列并行，第二层行并行"""
    def __init__(self, d_model, d_ff, world_size):
        super().__init__()
        self.col_linear = ColumnParallelLinear(d_model, d_ff, world_size)
        self.row_linear = RowParallelLinear(d_ff, d_model, world_size)

    def forward(self, x):
        hidden = torch.relu(self.col_linear(x))
        return self.row_linear(hidden)
```

&emsp;&emsp;这个实现中，`ColumnParallelLinear` 将输出特征维切分到各设备，`RowParallelLinear` 将输入特征维切分并通过 all-reduce 合并结果。若输入形状为 $(B,L,d_{model})$，输出形状为 $(B,L,d_{model})$。

---

#### 2.17.3 流水线并行

&emsp;&emsp;流水线并行（Pipeline Parallelism, PP）将模型按层切分为多个阶段，每个阶段部署在不同的设备上，数据以微批次的形式在阶段之间流动。GPipe 是最早的流水线并行方案，将每个 batch 拆分为多个微批次，先依次执行所有微批次的前向传播，再依次执行反向传播。这种方案的缺点是内存占用高，因为每个阶段需要保存所有微批次的中间激活值直到反向传播完成。

&emsp;&emsp;PipeDream 提出的 1F1B（One-Forward-One-Backward）调度改进了这一问题：在预热阶段，微批次依次向前流动，直到最后一个阶段收到第一个微批次；此后，每个阶段交替执行一次前向和一次反向，反向传播完成后立即释放对应的激活值，从而将内存占用从 $O(M)$ 降到 $O(P)$，其中 $M$ 是微批次数量，$P$ 是流水线阶段数。

&emsp;&emsp;流水线并行的优点是通信量小，只需在阶段之间传递激活值和梯度，适合跨节点部署，且与数据并行和张量并行正交，可以组合使用；缺点是存在流水线气泡，即部分设备在预热和冷却阶段空闲，气泡比例约为 $(P-1)/M$，需要增加微批次数量来降低气泡，但微批次过多又会增加内存开销。

&emsp;&emsp;从维度视角看，流水线并行在层维上将模型切分到多个设备，每个设备负责连续的若干层。数据在层维上以微批次的形式流动，前向传播时激活值沿层维传递，反向传播时梯度沿相反方向传递。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 1F1B 流水线调度的裸实现：

```python
import torch
import torch.nn as nn

class PipelineStage(nn.Module):
    """单个流水线阶段: 包含若干层"""
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class Pipeline1F1B:
    """1F1B 流水线调度器"""
    def __init__(self, stages, num_micro_batches):
        self.stages = stages
        self.num_micro_batches = num_micro_batches
        self.num_stages = len(stages)

    def train_step(self, micro_batches, targets, criterion, optimizer):
        """执行一个 1F1B 训练步"""
        P = self.num_stages
        M = self.num_micro_batches

        # 存储各阶段的激活值
        activations = {p: [] for p in range(P)}
        losses = []

        # 预热阶段: 前向传播
        for m in range(M):
            x = micro_batches[m]
            for p in range(P):
                x = self.stages[p](x)
                if p < P - 1:
                    activations[p].append(x.detach().requires_grad_(True))

        # 稳态 + 冷却阶段: 交替前向和反向
        for m in range(M):
            # 反向传播
            x = self.stages[P-1](activations[P-2].pop()) if P > 1 else micro_batches[m]
            loss = criterion(x, targets[m])
            loss.backward()
            losses.append(loss.item())

            # 下一个微批次的前向传播（如果还有）
            if m + 1 < M:
                x = micro_batches[m + 1]
                for p in range(P):
                    x = self.stages[p](x)
                    if p < P - 1:
                        activations[p].append(x.detach().requires_grad_(True))

        optimizer.step()
        optimizer.zero_grad()
        return sum(losses) / len(losses)
```

&emsp;&emsp;这个实现展示了 1F1B 的核心调度逻辑：预热阶段依次前向传播所有微批次，然后交替执行反向传播和下一个微批次的前向传播。实际中需要跨设备通信来传递激活值和梯度。

---

#### 2.17.4 3D 并行

&emsp;&emsp;3D 并行（3D Parallelism）是数据并行、张量并行和流水线并行的组合，用于训练超大规模模型。其核心思想是：数据并行在 batch 维上扩展，张量并行在特征维上切分，流水线并行在层维上切分，三者正交，可以同时使用。设数据并行度为 $N_d$，张量并行度为 $N_t$，流水线并行度为 $N_p$，则总设备数为：

$$
N_{\mathrm{total}} = N_d \times N_t \times N_p
$$

&emsp;&emsp;在 3D 并行中，每个设备被分配到一个三维网格中的一个位置。数据并行组内的设备持有相同的模型分片但处理不同的数据；张量并行组内的设备持有同一层的不同特征分片；流水线并行组内的设备持有不同层的模型分片。通信模式也相应分层：张量并行组内需要频繁的 all-reduce（通常通过 NVLink），流水线并行组间需要点对点通信传递激活值和梯度，数据并行组间需要 all-reduce 同步梯度。

&emsp;&emsp;3D 并行的优点是能够组合三种并行的优势，在超大规模训练中实现最优的显存和计算效率；缺点是配置复杂，需要根据模型规模、硬件拓扑和网络带宽手动调优各维度的并行度，且不同维度的通信模式可能相互干扰，调度难度高。从维度视角看，3D 并行同时在 batch 维（数据并行）、特征维（张量并行）和层维（流水线并行）上切分模型和数据，实现三维空间上的分布式计算。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 3D 并行的配置和通信组划分的裸实现：

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

class Parallel3DConfig:
    """3D 并行配置"""
    def __init__(self, dp_size, tp_size, pp_size):
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.world_size = dp_size * tp_size * pp_size

    def get_groups(self, rank):
        """根据全局 rank 计算各维度的通信组"""
        # rank = dp_rank * tp_size * pp_size + tp_rank * pp_size + pp_rank
        pp_rank = rank % self.pp_size
        tp_rank = (rank // self.pp_size) % self.tp_size
        dp_rank = rank // (self.tp_size * self.pp_size)

        # 数据并行组: 相同 tp_rank 和 pp_rank，不同 dp_rank
        dp_group = []
        for d in range(self.dp_size):
            dp_group.append(d * self.tp_size * self.pp_size + tp_rank * self.pp_size + pp_rank)

        # 张量并行组: 相同 dp_rank 和 pp_rank，不同 tp_rank
        tp_group = []
        for t in range(self.tp_size):
            tp_group.append(dp_rank * self.tp_size * self.pp_size + t * self.pp_size + pp_rank)

        # 流水线并行组: 相同 dp_rank 和 tp_rank，不同 pp_rank
        pp_group = []
        for p in range(self.pp_size):
            pp_group.append(dp_rank * self.tp_size * self.pp_size + tp_rank * self.pp_size + p)

        return {
            "dp_rank": dp_rank, "tp_rank": tp_rank, "pp_rank": pp_rank,
            "dp_group": dp_group, "tp_group": tp_group, "pp_group": pp_group,
        }


def parallel_3d_worker(rank, config):
    """3D 并行 worker"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=config.world_size)

    groups = config.get_groups(rank)

    # 创建各维度的通信组
    dp_group = dist.new_group(groups["dp_group"])
    tp_group = dist.new_group(groups["tp_group"])
    pp_group = dist.new_group(groups["pp_group"])

    print(f"Rank {rank}: dp={groups['dp_rank']}, tp={groups['tp_rank']}, pp={groups['pp_rank']}")

    # 模拟：数据并行组内 all-reduce 梯度
    tensor = torch.ones(10) * rank
    dist.all_reduce(tensor, group=dp_group)
    tensor /= config.dp_size

    # 模拟：张量并行组内 all-reduce
    tensor2 = torch.ones(10) * rank
    dist.all_reduce(tensor2, group=tp_group)

    dist.destroy_process_group()
```

&emsp;&emsp;这个实现展示了 3D 并行的通信组划分逻辑。实际训练中，数据并行组负责梯度同步，张量并行组负责层内特征维的 all-reduce，流水线并行组负责层间的点对点激活传递。

---

#### 2.17.5 ZeRO

&emsp;&emsp;ZeRO（Zero Redundancy Optimizer）是 DeepSpeed 提出的显存优化技术，核心思想是消除数据并行中的显存冗余。标准数据并行中，每个设备都持有完整的模型参数、梯度和优化器状态，导致大量重复存储。ZeRO 将这些状态切分到各设备上，使每个设备只存储一部分，需要时再通过通信收集。ZeRO 分为三个递进阶段。

&emsp;&emsp;ZeRO-1 切分优化器状态。以 Adam 为例，优化器需要存储 FP32 主权重、一阶矩和二阶矩，共 $3N$ 个参数（$N$ 为模型参数量），这些状态被均匀切分到 $N_d$ 个设备上，每个设备只更新自己负责的部分。ZeRO-2 在 ZeRO-1 的基础上进一步切分梯度，每个设备只保留与自身优化器状态对应的梯度分片。ZeRO-3 进一步切分模型参数本身，每个设备只存储 $1/N_d$ 的参数，前向和反向传播时通过 all-gather 按需收集。

&emsp;&emsp;ZeRO-1 和 ZeRO-2 消除优化器状态和梯度的冗余，ZeRO-3 消除参数的冗余。显存节省比例分别为：ZeRO-1 约为 $3N/N_d$ 的优化器状态节省，ZeRO-2 进一步节省 $2N/N_d$ 的梯度，ZeRO-3 将参数显存从 $2N$ 降至 $2N/N_d$。ZeRO 的优点是无需修改模型代码，仅通过配置即可大幅降低单卡显存需求，使大模型训练在有限显存下成为可能；缺点是 ZeRO-3 的通信量显著增加，每层前向和反向都需要 all-gather 参数，通信开销可能抵消显存节省的收益。

&emsp;&emsp;从维度视角看，ZeRO 在数据并行的基础上，将特征维上的参数、梯度和优化器状态进一步切分到各设备，使每个设备在特征维上只负责一部分状态。ZeRO-3 相当于将数据并行与参数切分结合，实现了“数据并行+参数并行”的混合模式。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 ZeRO 各阶段显存节省的模拟实现：

```python
import torch
import torch.nn as nn

class ZeROOptimizer:
    """模拟 ZeRO 各阶段的显存节省"""
    def __init__(self, model, optimizer, world_size, stage=1):
        self.model = model
        self.optimizer = optimizer
        self.world_size = world_size
        self.stage = stage
        self.rank = 0  # 简化: 模拟单个 rank

    def step(self):
        if self.stage == 1:
            # ZeRO-1: 仅切分优化器状态
            # 每个设备只更新自己负责的那部分优化器状态
            pass
        elif self.stage == 2:
            # ZeRO-2: 切分优化器状态 + 梯度
            pass
        elif self.stage == 3:
            # ZeRO-3: 切分优化器状态 + 梯度 + 参数
            pass
        self.optimizer.step()

    def memory_savings(self):
        """估算各阶段的显存节省"""
        N = sum(p.numel() for p in self.model.parameters())
        # 标准数据并行: 参数(2N) + 梯度(2N) + 优化器状态(12N) = 16N
        baseline = 16 * N
        if self.stage == 1:
            # ZeRO-1: 优化器状态切分
            saved = 12 * N * (1 - 1 / self.world_size)
        elif self.stage == 2:
            # ZeRO-2: 优化器状态 + 梯度切分
            saved = (12 * N + 2 * N) * (1 - 1 / self.world_size)
        elif self.stage == 3:
            # ZeRO-3: 优化器状态 + 梯度 + 参数切分
            saved = (12 * N + 2 * N + 2 * N) * (1 - 1 / self.world_size)
        return baseline, baseline - saved
```

&emsp;&emsp;这个实现模拟了 ZeRO 各阶段的显存节省计算。实际使用中推荐 `deepspeed.initialize` 配合配置 JSON 启用 ZeRO。

---

#### 2.17.6 专家并行

&emsp;&emsp;专家并行（Expert Parallelism, EP）是混合专家（MoE）模型专用的并行策略。MoE 层包含多个专家，每个专家本质上是一个小型 FFN。专家并行将不同专家部署在不同的设备上，每个设备只持有部分专家，token 通过路由器分配到对应的专家设备进行计算，计算完成后再收集回原设备。

&emsp;&emsp;专家并行的核心通信模式是 all-to-all：每个设备将自己的 token 发送到持有目标专家的设备，接收其他设备发送来的 token，本地专家计算完成后，再将结果发送回原设备。设设备数为 $N_e$，专家总数为 $E$，每个设备持有 $E/N_e$ 个专家。第 $i$ 个设备的 token 分发过程为：

$$
\mathrm{dispatch}: \quad \text{token}_i \xrightarrow{\text{all-to-all}} \{\text{token}_{j \to i} \mid \mathrm{expert}(\text{token}_j) \in \mathcal{E}_i\}
$$

&emsp;&emsp;专家并行的优点是 MoE 模型的专家可以分布到多个设备上，解决了单卡显存无法容纳大量专家的问题，且每个设备只计算被路由到本地的 token，计算量稀疏；缺点是 all-to-all 通信开销大，且路由不均衡时部分设备可能过载而其他设备空闲，导致负载不均衡和通信效率下降。

&emsp;&emsp;从维度视角看，专家并行在特征维上将专家子网络分布到不同设备，token 在特征维上被路由到不同的专家。all-to-all 通信在 token 维上重新分配 token 到持有目标专家的设备，形成 token 维与特征维之间的动态映射。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出专家并行的裸实现，包含 all-to-all 分发和收集：

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def forward(self, x):
        return torch.matmul(
            torch.relu(torch.matmul(x, self.W_1) + self.b_1),
            self.W_2
        ) + self.b_2


class ExpertParallelMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, world_size):
        super().__init__()
        self.world_size = world_size
        self.experts_per_rank = num_experts // world_size
        self.rank = dist.get_rank()

        # 本设备持有的专家
        self.local_experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(self.experts_per_rank)
        ])

        # 路由器: 全局专家嵌入
        self.router = nn.Parameter(torch.empty(num_experts, d_model))
        nn.init.normal_(self.router, std=0.02)

    def forward(self, x):
        """
        x: (B, L, d_model)
        简化实现: 每个 token 选择一个专家
        """
        B, L, D = x.size()
        x_flat = x.view(B * L, D)

        # 路由: 选择专家
        scores = torch.matmul(x_flat, self.router.T)
        expert_ids = scores.argmax(dim=-1)  # (B*L,)

        # 确定哪些 token 属于本设备的专家
        local_start = self.rank * self.experts_per_rank
        local_end = local_start + self.experts_per_rank
        local_mask = (expert_ids >= local_start) & (expert_ids < local_end)

        # 本地专家计算
        local_tokens = x_flat[local_mask]
        local_expert_ids = expert_ids[local_mask] - local_start

        local_output = torch.zeros_like(local_tokens)
        for e in range(self.experts_per_rank):
            mask = (local_expert_ids == e)
            if mask.any():
                local_output[mask] = self.local_experts[e](local_tokens[mask])

        # 简化: 不做实际的 all-to-all 通信
        # 实际实现中需要 all_to_all 分发 token 和收集结果
        output = torch.zeros_like(x_flat)
        output[local_mask] = local_output

        return output.view(B, L, D)
```

&emsp;&emsp;这个实现展示了专家并行的核心逻辑：每个设备持有部分专家，token 根据路由结果发送到对应设备。实际中需要 `dist.all_to_all` 进行 token 的分发和收集。若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.17.7 MoonEP / FlashKDA / AgentEnv

&emsp;&emsp;MoonEP、FlashKDA 和 AgentEnv 是月之暗面（Moonshot AI）在 Kimi K2/K3 训练中开发的三项基础设施技术，分别解决专家并行通信、线性注意力推理和智能体强化学习环境三个不同层面的问题。

&emsp;&emsp;**MoonEP** 是一个通过动态冗余专家实现完美负载均衡的专家并行通信库。传统专家并行中，路由不均衡会导致部分设备过载、部分设备空闲，通信延迟由最热的 rank 决定。MoonEP 的核心创新是：从当前路由输出中在线规划少量冗余专家，在专家计算前预取这些冗余专家的权重，使每个 rank 接收到的 token 数严格相等（$S \times K$），无论路由多么倾斜。MoonEP 采用零拷贝设计，token 直接写入远程 rank 上按专家分组的最终位置，通信缓冲区直接交给计算使用，消除了通信缓冲区到用户缓冲区的拷贝。在 H20 上的基准测试中，MoonEP 的通信时间在各不均衡水平下均低于 DeepEP v2，且端到端训练中迭代时间保持平坦，不会因不均衡导致 OOM。

&emsp;&emsp;**FlashKDA** 是 Kimi Delta Attention（KDA）线性注意力的融合内核实现。FlashKDA v1 使用 `CHUNK=16` 而非 Flash Linear Attention 的 `CHUNK=64`，原因有三：数值范围在 bf16 内可表示，消除了块内重缩放的需要；$16 \times 16$ 矩阵求逆远快于 $64 \times 64$；所有 `CHUNK=16` 数学映射到 SM80 MMA 指令，保持可移植性。FlashKDA 将计算分为两个内核：K1 为 token 并行（`g` 激活、L2 归一化、衰减应用、$L/M_{qk}$ 构造、矩阵求逆），K2 为 head 并行（逐块 delta 规则递推、输出投影、运行状态累积）。递归状态以 bf16 存储，状态更新使用 FP32 FMA，在推理基准中无精度损失。

&emsp;&emsp;**AgentEnv** 是一个开源的分布式智能体环境运行平台，支撑 Kimi K3 的智能体强化学习训练。它基于 Firecracker 微虚拟机，通过 overlaybd 按需加载 OCI 镜像，可在集群级运行海量环境。AgentEnv 的核心特性包括：基于快照的环境启动或恢复低于 50ms，暂停低于 100ms；运行中的环境可 fork 为多个独立沙箱，支撑并行智能体工作流；通过 ublk 提供高性能 I/O 并共享宿主机页缓存。AgentEnv 已在生产环境中支撑超 150 万镜像与 5000 万个执行环境的并发调用。

&emsp;&emsp;这三项技术的共同特点是针对大规模训练和推理中的特定瓶颈提供系统级解决方案：MoonEP 解决专家并行的通信均衡问题，FlashKDA 解决线性注意力的推理效率问题，AgentEnv 解决智能体强化学习的环境扩展问题。它们的优点是针对性强、在各自场景中显著提升效率；缺点是各自依赖特定的模型架构或训练范式，通用性有限。

&emsp;&emsp;从维度视角看，MoonEP 在专家并行的 token 维分发上引入动态冗余，使 token 维的负载分布均匀化；FlashKDA 在特征维上通过分块和融合优化线性注意力的状态递推；AgentEnv 在环境维上通过快照和 fork 实现并行智能体工作流。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出 MoonEP 负载均衡规划核心逻辑的裸实现：

```python
import torch

def moonep_plan(routing_weights, num_experts, num_ranks, top_k):
    """
    MoonEP 规划: 计算冗余专家分配，使每个 rank 的 token 数均衡
    routing_weights: (num_tokens, num_experts) 路由器打分
    返回每个 rank 的 token 分配和冗余专家规划
    """
    num_tokens = routing_weights.size(0)
    tokens_per_rank = num_tokens * top_k // num_ranks
    target = tokens_per_rank

    # 每个 token 的 top-k 专家
    topk_scores, topk_experts = routing_weights.topk(top_k, dim=-1)  # (N, K)

    # 统计每个专家的 token 数
    expert_load = torch.zeros(num_experts, device=routing_weights.device)
    for k in range(top_k):
        expert_load.scatter_add_(0, topk_experts[:, k],
                                 torch.ones(num_tokens, device=routing_weights.device))

    # 将专家均匀分配到 rank
    experts_per_rank = num_experts // num_ranks
    rank_load = torch.zeros(num_ranks, device=routing_weights.device)
    for r in range(num_ranks):
        start = r * experts_per_rank
        end = start + experts_per_rank
        rank_load[r] = expert_load[start:end].sum()

    # 计算每个 rank 需要的冗余专家数
    redundant = torch.clamp(target - rank_load, min=0).long()
    return {
        "expert_load": expert_load,
        "rank_load": rank_load,
        "redundant_experts": redundant,
        "target_per_rank": target,
    }


def flashkda_chunk_forward(q, k, g, chunk_size=16):
    """
    FlashKDA 单块前向的简化模拟
    q, k: (B, H, L, D)
    g: (B, H, L) 门控值
    """
    B, H, L, D = q.size()
    num_chunks = L // chunk_size

    outputs = []
    state = torch.zeros(B, H, D, D, device=q.device)  # 递归状态

    for c in range(num_chunks):
        start = c * chunk_size
        end = start + chunk_size

        q_c = q[:, :, start:end, :]
        k_c = k[:, :, start:end, :]
        g_c = g[:, :, start:end]

        # 块内 delta 规则递推
        for t in range(chunk_size):
            q_t = q_c[:, :, t, :]
            k_t = k_c[:, :, t, :]
            g_t = g_c[:, :, t]

            # 状态更新: S = S * exp(g_t) + k_t^T * v_t (简化)
            state = state * torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)
            state = state + k_t.unsqueeze(-1) * k_t.unsqueeze(-2)

            # 读取: o_t = q_t @ S
            o_t = torch.matmul(q_t.unsqueeze(-2), state).squeeze(-2)
            outputs.append(o_t)

    return torch.stack(outputs, dim=2)  # (B, H, L, D)
```

&emsp;&emsp;这个实现中，`moonep_plan` 计算每个 rank 的专家负载和需要的冗余专家数，`flashkda_chunk_forward` 模拟 FlashKDA 的块内递推逻辑。实际中 MoonEP 使用 CUDA 内核实现规划，FlashKDA 使用融合内核优化内存访问。


---

### 2.18 数据工程与训练方法

#### 2.18.1 WebText

&emsp;&emsp;WebText 是 OpenAI 为 GPT-2 构建的预训练语料，核心来源是 Reddit 上获得至少 3 个 karma 的出站链接所指向的网页。抓取后使用 Dragnet 等内容提取器抽取正文，去除 HTML 标签、脚本和导航栏，最终得到约 40GB 文本、800 万篇文档。形式化地，文档集合为：

$$
\mathcal{D}_{\mathrm{WebText}} = \{ \mathrm{Extract}(u) \mid (u, k) \in \mathcal{L}_{\mathrm{Reddit}}, k \geq 3 \}
$$

&emsp;&emsp;其中 $u$ 是出站链接，$k$ 是帖子 karma。WebText 的优点是文本与 Reddit 社区兴趣对齐，质量较高，适合训练生成模型；缺点是数据集未公开，领域偏斜严重，依赖 Reddit 用户群体，可能包含有害、偏见或版权内容，且规模相对现代预训练语料较小。从维度视角看，WebText 在 token 维上提供了大量自然语言序列，使模型学习词序列统计规律；特征维上通过内容提取和去重减少噪声，使表示更集中于正文语义。

**&emsp;&emsp;PyTorch 实现示例**

```python
import re

def extract_main_text(html):
    text = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_webtext(reddit_posts, min_karma=3):
    docs = []
    for post in reddit_posts:
        if post["karma"] < min_karma:
            continue
        text = extract_main_text(post["html"])
        if text:
            docs.append(text)
    return docs
```

若输入为帖子列表，输出为正文文档列表。

---

#### 2.18.2 C4

&emsp;&emsp;C4（Colossal Clean Crawled Corpus）是 Google 为 T5 构建的大规模英文预训练语料，来自 2019 年 4 月的一次 Common Crawl 快照。原始 Common Crawl 包含大量噪声、模板文本和非自然语言，C4 通过一系列启发式规则清洗：只保留以标点结尾的行，删除少于 3 个词的句子，删除包含坏词表中词汇的页面，删除包含 JavaScript 或代码特征的页面，删除非英语页面，并进行文档级去重。清洗后约 750GB 英文文本。C4 的优点是大规模、公开、清洗规则明确，成为后续许多模型的基线语料；缺点是启发式过滤可能删除有价值内容，英文中心严重，对低资源语言不友好，且仍残留偏见和有害内容。从维度视角看，C4 在 token 维上通过行级和文档级过滤压缩了原始爬取数据的噪声，使 token 序列更接近自然语言；特征维上通过语言识别和代码过滤，使表示空间更集中于英文自然语言。

**&emsp;&emsp;PyTorch 实现示例**

```python
BAD_WORDS = {"sex", "porn", "xxx"}
PUNCT_END = (".", "!", "?", '"', "'")

def c4_clean_line(line):
    line = line.strip()
    if len(line.split()) < 3:
        return None
    if not line.endswith(PUNCT_END):
        return None
    if any(w in line.lower() for w in BAD_WORDS):
        return None
    if "{" in line or "}" in line or "function" in line:
        return None
    return line

def c4_clean_document(text):
    lines = []
    for line in text.split("\n"):
        cleaned = c4_clean_line(line)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)

def deduplicate_docs(docs):
    seen = set()
    unique = []
    for doc in docs:
        h = hash(doc)
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique
```

若输入为原始文本，输出为清洗后的文档。

---

#### 2.18.3 MassiveText

&emsp;&emsp;MassiveText 是 DeepMind 为 Gopher 构建的预训练语料集合，总规模约 10.5TB 文本，包含多个子集：MassiveWeb（网页）、Books（书籍）、C4、News（新闻）、GitHub（代码）、Wikipedia（百科）等。每个子集有独立的清洗和过滤流程，最后按一定比例混合。MassiveText 的设计强调多源覆盖，使模型在网页、书籍、代码、百科等不同分布上都能获得训练信号。MassiveText 的优点是多源、规模大、覆盖广，混合比例可调，适合训练通用语言模型；缺点是完整数据集未公开，各子集的过滤规则和混合比例对最终性能影响很大，且仍存在偏见和版权问题。从维度视角看，MassiveText 在 token 维上提供了来自不同领域的序列分布，使模型学习跨领域的词序列规律；特征维上通过多源混合，使表示空间同时覆盖自然语言、代码和结构化知识。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch

def mix_massive_text(sources, weights, total_tokens):
    """
    sources: dict source_name -> token tensor
    weights: dict source_name -> float
    """
    total_w = sum(weights.values())
    mixed = []
    for name, tokens in sources.items():
        n = int(total_tokens * weights[name] / total_w)
        if n > 0:
            idx = torch.randint(0, len(tokens), (n,))
            mixed.append(tokens[idx])
    result = torch.cat(mixed, dim=0)
    perm = torch.randperm(len(result))
    return result[perm]
```

若输入为多个来源的 token 张量，输出为混合后的 token 序列。

---

#### 2.18.4 RefinedWeb

&emsp;&emsp;RefinedWeb 是 Falcon 模型使用的预训练语料，完全从 Common Crawl 构建，强调仅用网页数据也能训练出高质量模型。其流程包括：URL 过滤、内容提取、语言识别、质量过滤、MinHash 去重、以及安全过滤。RefinedWeb 公开了约 5T token 的英文语料，并提供了可复现的构建管线。RefinedWeb 的优点是完全公开、去重严格、质量高，证明了网页数据经过精细清洗后可以媲美人工整理语料；缺点是英文为主，过滤规则依赖启发式，仍可能保留偏见和有害内容，且对非英语覆盖不足。从维度视角看，RefinedWeb 在 token 维上通过严格去重和过滤减少了重复和低质序列，使 token 分布更接近高质量英文；特征维上通过内容提取和安全过滤，使表示更集中于有价值语义。

**&emsp;&emsp;PyTorch 实现示例**

```python
import re
import hashlib

def refinedweb_filter(text):
    if len(text.split()) < 50:
        return False
    if not re.search(r"[.!?]", text[-100:]):
        return False
    if "lorem ipsum" in text.lower():
        return False
    return True

def minhash_signature(text, num_hashes=64, shingle_size=5):
    words = text.split()
    shingles = set()
    for i in range(len(words) - shingle_size + 1):
        shingles.add(" ".join(words[i:i+shingle_size]))
    sig = []
    for i in range(num_hashes):
        min_h = float("inf")
        for s in shingles:
            h = int(hashlib.md5(f"{i}_{s}".encode()).hexdigest(), 16)
            if h < min_h:
                min_h = h
        sig.append(min_h)
    return sig

def refinedweb_dedup(docs, num_hashes=64, bands=8):
    signatures = [minhash_signature(d, num_hashes) for d in docs]
    buckets = {}
    rows = num_hashes // bands
    for idx, sig in enumerate(signatures):
        for b in range(bands):
            band = tuple(sig[b*rows:(b+1)*rows])
            buckets.setdefault(band, []).append(idx)
    keep = set(range(len(docs)))
    for band, idxs in buckets.items():
        if len(idxs) > 1:
            for i in idxs[1:]:
                keep.discard(i)
    return [docs[i] for i in sorted(keep)]
```

若输入为文档列表，输出为过滤和去重后的文档。

---

#### 2.18.5 LAION-5B

&emsp;&emsp;LAION-5B 是目前最大的公开多模态图像-文本数据集之一，包含约 58.5 亿个图文对，来自 Common Crawl 中提取的 HTML 图像链接和 alt 文本。构建流程包括：解析 HTML、下载图像、计算 CLIP 图像和文本嵌入、用余弦相似度过滤低质量对，并分为 LAION-2B 英文、LAION-2B 多语言、LAION-400M 等子集。形式化地，保留条件为：

$$
\cos(\mathrm{CLIP}_{\mathrm{img}}(I), \mathrm{CLIP}_{\mathrm{text}}(T)) \geq \tau
$$

&emsp;&emsp;其中 $\tau$ 是相似度阈值。LAION-5B 的优点是规模巨大、多模态、公开，推动了开源多模态模型发展；缺点是噪声和偏见严重，包含版权、隐私和不当内容，CLIP 过滤本身带有偏见，且图像链接可能失效。从维度视角看，LAION-5B 在 token 维上同时提供图像 patch 序列和文本 token 序列，使多模态模型学习跨模态对齐；特征维上通过 CLIP 嵌入过滤，使图文对在共享语义空间中靠近。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn.functional as F

def laion_filter(image_embeds, text_embeds, threshold=0.28):
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    sims = (image_embeds * text_embeds).sum(dim=-1)
    keep = sims >= threshold
    return keep, sims

def build_laion_subset(pairs, clip_model, threshold=0.28):
    kept = []
    for img, text in pairs:
        with torch.no_grad():
            img_emb = clip_model.encode_image(img)
            txt_emb = clip_model.encode_text(text)
        sim = F.cosine_similarity(img_emb, txt_emb, dim=-1)
        if sim.item() >= threshold:
            kept.append((img, text))
    return kept
```

若输入为图像和文本嵌入，输出为过滤后的图文对。

---

#### 2.18.6 MinHash 去重

&emsp;&emsp;MinHash 去重是大规模语料构建中最常用的近似去重方法。其核心思想是用一组随机哈希函数将文档的 shingle 集合映射为签名向量，使两个文档签名相等的概率等于它们的 Jaccard 相似度：

$$
P(\mathrm{MinHash}(A)=\mathrm{MinHash}(B)) = J(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

&emsp;&emsp;其中 $A,B$ 是文档的 shingle 集合。实践中通常取 $k$ 个哈希函数得到 $k$ 维签名，再用 LSH 将签名分带，只有至少一个带完全相同的文档才被认定为候选重复对，从而避免两两比较。MinHash 的优点是能扩展到数十亿文档，去重效果好，参数可调，广泛用于 C4、RefinedWeb、LAION 等语料；缺点是近似方法，可能误删相似但不同文档，也可能漏掉部分重复，对短文本效果差，且 shingle 大小和哈希数需要调优。从维度视角看，MinHash 在 token 维上将文档的 shingle 集合压缩为固定长度签名，使 token 维的相似性可以在低维签名空间中快速估计；特征维上通过 LSH 分桶，将相似文档映射到同一桶中。

**&emsp;&emsp;PyTorch 实现示例**

```python
import hashlib

def minhash_signature(text, num_hashes=128, shingle_size=5):
    words = text.split()
    shingles = set()
    for i in range(len(words) - shingle_size + 1):
        shingles.add(" ".join(words[i:i+shingle_size]))
    sig = []
    for i in range(num_hashes):
        min_h = float("inf")
        for s in shingles:
            h = int(hashlib.md5(f"{i}_{s}".encode()).hexdigest(), 16)
            if h < min_h:
                min_h = h
        sig.append(min_h)
    return sig

def lsh_dedup(docs, num_hashes=128, bands=16):
    rows = num_hashes // bands
    buckets = {}
    for idx, doc in enumerate(docs):
        sig = minhash_signature(doc, num_hashes)
        for b in range(bands):
            band = tuple(sig[b*rows:(b+1)*rows])
            buckets.setdefault(band, []).append(idx)
    keep = set(range(len(docs)))
    for band, idxs in buckets.items():
        if len(idxs) > 1:
            for i in idxs[1:]:
                keep.discard(i)
    return [docs[i] for i in sorted(keep)]
```

&emsp;&emsp;若输入为文档列表，输出为去重后的文档列表。

#### 2.18.7 质量过滤

&emsp;&emsp;质量过滤（Quality Filtering）是预训练语料构建中剔除低质量文档的关键步骤。大规模网页爬取数据中混杂着模板文本、广告、导航栏、机器生成内容和重复片段，直接用于训练会显著损害模型性能。质量过滤通常结合启发式规则和模型打分两种手段。启发式规则包括：文档长度阈值、标点符号比例、停用词比例、重复 n-gram 比例、大写字母比例、特殊字符比例等。设文档 $d$ 的重复 n-gram 比例为：

$$
\mathrm{rep}(d) = 1 - \frac{|\{\text{unique n-grams in } d\}|}{|\{\text{all n-grams in } d\}|}
$$

&emsp;&emsp;当 $\mathrm{rep}(d) > \tau$ 时，文档被判定为低质量。模型打分则用一个在高质量语料（如 Wikipedia、书籍）上训练的分类器对文档打分，保留分数高于阈值的文档。质量过滤的优点是显著提升训练语料质量，使模型在同等 token 数下取得更好性能，且规则和模型可以组合使用，灵活性强；缺点是过滤规则可能误删有价值内容，模型打分器本身带有偏见，可能偏好特定风格而忽略多样性，且过滤阈值需要大量实验调优。从维度视角看，质量过滤在 token 维上剔除了低信息密度和重复的序列，使 token 分布更集中于高质量自然语言；特征维上通过分类器打分，使表示空间偏向与高质量语料相似的分布。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出启发式质量过滤的裸实现：

```python
import re
from collections import Counter

def repetition_ratio(text, n=4):
    words = text.split()
    if len(words) < n:
        return 1.0
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    if not ngrams:
        return 1.0
    return 1.0 - len(set(ngrams)) / len(ngrams)

def punctuation_ratio(text):
    if not text:
        return 0.0
    punct = sum(1 for c in text if c in ".,!?;:")
    return punct / len(text)

def uppercase_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)

def quality_filter(text, min_words=50, max_rep=0.3,
                   min_punct=0.01, max_upper=0.3):
    if len(text.split()) < min_words:
        return False
    if repetition_ratio(text) > max_rep:
        return False
    if punctuation_ratio(text) < min_punct:
        return False
    if uppercase_ratio(text) > max_upper:
        return False
    if "lorem ipsum" in text.lower():
        return False
    return True

def filter_corpus(docs, **kwargs):
    return [d for d in docs if quality_filter(d, **kwargs)]
```

&emsp;&emsp;若输入为文档列表，输出为通过质量过滤的文档。

---

#### 2.18.8 合成数据与蒸馏

&emsp;&emsp;合成数据与蒸馏（Synthetic Data and Distillation）指用强模型生成训练数据，或用强模型的输出分布指导弱模型训练。在预训练和后训练中，合成数据已成为提升数据规模和质量的重要手段。设教师模型为 $p_T$，学生模型为 $p_S$，蒸馏损失为：

$$
\mathcal{L}_{\mathrm{KD}} = -\sum_{t=1}^{T} \sum_{v=1}^{V} p_T(v \mid x_{<t}) \log p_S(v \mid x_{<t})
$$

&emsp;&emsp;若只使用教师模型的硬标签，则退化为标准交叉熵。在合成数据生成中，通常让强模型对大量提示生成回答，经过筛选后用于 SFT 或继续预训练。合成数据与蒸馏的优点是可以用较低成本获得大量高质量数据，将强模型的能力迁移到小模型，且数据可控、可定制；缺点是教师模型的偏见和错误会被学生继承，合成数据分布可能与真实数据分布偏移，过度依赖合成数据会导致模型多样性下降和模式崩溃。从维度视角看，合成数据与蒸馏在 token 维上用教师模型生成的序列扩展了训练数据，特征维上通过软标签传递教师模型的表示分布，使学生模型学习到比硬标签更丰富的语义结构。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出知识蒸馏损失的裸实现：

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, temperature=2.0,
                      hard_labels=None, alpha=0.5):
    """
    student_logits: (B, L, V)
    teacher_logits: (B, L, V)
    hard_labels: (B, L) 可选
    """
    # 软标签蒸馏
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    kd_loss = -(soft_teacher * soft_student).sum(dim=-1).mean()
    kd_loss = kd_loss * (temperature ** 2)

    if hard_labels is not None:
        ce_loss = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            hard_labels.reshape(-1)
        )
        return alpha * kd_loss + (1 - alpha) * ce_loss
    return kd_loss

def generate_synthetic_data(teacher_model, prompts, max_new_tokens=256):
    """用教师模型生成合成数据"""
    synthetic = []
    for prompt in prompts:
        generated = prompt
        for _ in range(max_new_tokens):
            logits = teacher_model(generated)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=-1)
        synthetic.append(generated)
    return synthetic
```

&emsp;&emsp;`distillation_loss` 组合软标签和硬标签损失，`generate_synthetic_data` 用教师模型生成合成序列。

---

#### 2.18.9 代码执行反馈

&emsp;&emsp;代码执行反馈（Code Execution Feedback）指在代码生成任务中，通过实际运行生成的代码并根据执行结果（通过/失败、错误类型、输出匹配）提供奖励或监督信号。在代码预训练和后训练中，执行反馈是构建可验证奖励的核心手段。设生成的代码为 $c$，测试用例集合为 $\mathcal{T}$，执行反馈奖励为：

$$
r(c) = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \mathbf{1}\{\mathrm{Exec}(c, t) = \text{pass}\}
$$

&emsp;&emsp;在 RLVR 中，这个奖励直接用于 PPO 或 GRPO 的优势估计。代码执行反馈的优点是奖励绝对准确，不存在奖励黑客问题，训练信号干净，且可以自动大规模生成；缺点是需要安全的沙箱环境执行代码，执行时间和资源成本高，对于无法自动验证的代码任务（如架构设计、代码风格）无法使用。从维度视角看，代码执行反馈在 token 维上用确定性执行结果替代了学习的奖励模型，使代码 token 序列的质量由实际运行结果直接评判。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出代码执行反馈的模拟实现：

```python
import subprocess
import tempfile
import os
import torch

def execute_code(code, test_input, timeout=5):
    """在沙箱中执行代码，返回是否通过"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                     delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            ["python", fname],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout
    except subprocess.TimeoutExpired:
        return False, ""
    finally:
        os.unlink(fname)

def code_execution_reward(codes, test_cases):
    """
    codes: list of str
    test_cases: list of (input, expected_output)
    """
    rewards = []
    for code in codes:
        passed = 0
        for inp, expected in test_cases:
            ok, out = execute_code(code, inp)
            if ok and out.strip() == expected.strip():
                passed += 1
        rewards.append(passed / len(test_cases))
    return torch.tensor(rewards)

def grpo_with_code_reward(logprobs, old_logprobs, rewards,
                          clip_epsilon=0.2):
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r
    advantages = advantages.unsqueeze(-1)
    ratio = torch.exp(logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon,
                        1 + clip_epsilon) * advantages
    return -torch.min(surr1, surr2).mean()
```

&emsp;&emsp;`execute_code` 在子进程中运行代码，`code_execution_reward` 根据测试用例通过率计算奖励，`grpo_with_code_reward` 用该奖励做 GRPO 优化。

---

#### 2.18.10 编程语言翻译

&emsp;&emsp;编程语言翻译（Programming Language Translation）指将代码从一种编程语言转换为另一种，如 Python 到 C++、Java 到 Kotlin。在预训练数据构建中，编程语言翻译可以生成平行代码语料，用于训练代码翻译模型或增强代码理解能力。设源语言代码为 $c_s$，目标语言代码为 $c_t$，翻译模型为 $p_\theta$，训练目标为：

$$
\mathcal{L}_{\mathrm{trans}} = -\sum_{t=1}^{|c_t|} \log p_\theta(c_t^{(t)} \mid c_s, c_t^{(<t)})
$$

&emsp;&emsp;编程语言翻译的优点是可以在低资源编程语言之间迁移知识，生成大量平行语料，且翻译后的代码可以通过执行验证正确性；缺点是不同编程语言的语义和惯用法差异大，直译可能产生不符合目标语言习惯的代码，翻译质量依赖源语言和目标语言的训练数据量，且执行验证需要为每种语言配置环境。从维度视角看，编程语言翻译在 token 维上将源语言的 token 序列映射到目标语言的 token 序列，特征维上学习跨语言的语义对齐。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出编程语言翻译的简化训练实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CodeTranslator(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, num_heads,
                 num_layers, d_ff, max_len=1024):
        super().__init__()
        self.src_emb = nn.Parameter(torch.empty(src_vocab, d_model))
        self.tgt_emb = nn.Parameter(torch.empty(tgt_vocab, d_model))
        self.pos_emb = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.src_emb, std=0.02)
        nn.init.normal_(self.tgt_emb, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, d_ff,
                                       batch_first=True),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, num_heads, d_ff,
                                       batch_first=True),
            num_layers=num_layers
        )
        self.lm_head = nn.Parameter(torch.empty(d_model, tgt_vocab))
        nn.init.normal_(self.lm_head, std=0.02)

    def forward(self, src_ids, tgt_ids):
        B, L_s = src_ids.size()
        L_t = tgt_ids.size(1)

        src = self.src_emb[src_ids] + self.pos_emb[:L_s].unsqueeze(0)
        tgt = self.tgt_emb[tgt_ids] + self.pos_emb[:L_t].unsqueeze(0)

        memory = self.encoder(src)
        causal_mask = torch.triu(
            torch.ones(L_t, L_t, device=tgt.device), diagonal=1
        ).bool()
        dec = self.decoder(tgt, memory, tgt_mask=causal_mask)
        return torch.matmul(dec, self.lm_head)

def translation_loss(logits, tgt_ids):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt_ids.reshape(-1)
    )
```

&emsp;&emsp;若 `src_ids` 形状为 $(B,L_s)$，`tgt_ids` 形状为 $(B,L_t)$，输出 logits 形状为 $(B,L_t,V)$。

---

#### 2.18.11 文档反向翻译

&emsp;&emsp;文档反向翻译（Document Back-Translation）是数据增强和预训练语料构建中的一种技术。其流程是：将目标语言的文档翻译成另一种语言，再翻译回目标语言，得到一个改写版本。由于两次翻译引入了词汇和句法变化，回译版本可以作为原始文档的增强样本，增加训练数据的多样性。设原始文档为 $d$，中间语言为 $L_m$，回译文档为：

$$
\tilde{d} = \mathrm{Translate}_{L_m \to L_{\mathrm{tgt}}}\left(\mathrm{Translate}_{L_{\mathrm{tgt}} \to L_m}(d)\right)
$$

&emsp;&emsp;在预训练中，回译文档可以与原始文档一起使用，扩充语料规模；在微调中，回译可以生成释义多样的训练样本。文档反向翻译的优点是无需人工标注即可生成大量改写文本，增加数据多样性，提升模型鲁棒性；缺点是翻译模型可能引入错误和偏见，回译文本的流畅度和忠实度可能下降，且对低资源语言翻译质量差。从维度视角看，文档反向翻译在 token 维上生成与原始文档语义相近但表面形式不同的序列，特征维上保持语义表示不变，使模型学习到对表面变化不敏感的更鲁棒的表示。

**&emsp;&emsp;PyTorch 实现示例**

&emsp;&emsp;下面给出文档反向翻译的模拟实现：

```python
import torch
import torch.nn as nn

class BackTranslator:
    """文档反向翻译: 目标语言 -> 中间语言 -> 目标语言"""
    def __init__(self, forward_model, backward_model,
                 src_tokenizer, mid_tokenizer):
        self.forward_model = forward_model   # 目标 -> 中间
        self.backward_model = backward_model # 中间 -> 目标
        self.src_tokenizer = src_tokenizer
        self.mid_tokenizer = mid_tokenizer

    def _generate(self, model, input_ids, max_new_tokens=512):
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = model(generated)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == 2:  # EOS
                break
        return generated

    def back_translate(self, text):
        # 目标语言 -> 中间语言
        src_ids = self.src_tokenizer.encode(text)
        mid_ids = self._generate(self.forward_model, src_ids)
        mid_text = self.mid_tokenizer.decode(mid_ids)

        # 中间语言 -> 目标语言
        mid_ids2 = self.mid_tokenizer.encode(mid_text)
        back_ids = self._generate(self.backward_model, mid_ids2)
        back_text = self.src_tokenizer.decode(back_ids)
        return back_text

    def augment_corpus(self, docs):
        """对语料做回译增强"""
        augmented = []
        for doc in docs:
            augmented.append(doc)
            augmented.append(self.back_translate(doc))
        return augmented
```

&emsp;&emsp;这个实现中，`back_translate` 先翻译到中间语言再翻译回目标语言，`augment_corpus` 将原始文档和回译文档合并为增强语料。实际中翻译模型使用预训练的编码器-解码器架构。


---

### 2.19 模型架构范式

#### 2.19.1 Encoder-only

&emsp;&emsp;Encoder-only 架构只保留 Transformer 的编码器部分，每个 token 可以关注序列中的所有其他 token，因此是双向注意力。它不包含自回归生成机制，通常用于理解类任务，如文本分类、序列标注、抽取式问答和句子相似度。代表性模型包括 BERT、RoBERTa、DeBERTa 等。设输入 token 序列经过嵌入和位置编码后得到 $X \in \mathbb{R}^{n \times d_{model}}$，编码器由 $N$ 个相同层堆叠，每层包含双向多头自注意力和前馈网络，均使用残差连接和层归一化：

$$
Z = \mathrm{LayerNorm}\left(X + \mathrm{MultiHead}(X,X,X)\right)
$$

$$
Y = \mathrm{LayerNorm}\left(Z + \mathrm{FFN}(Z)\right)
$$

&emsp;&emsp;Encoder-only 的优点是双向注意力使每个 token 都能利用完整上下文，在理解类任务上表现优异，且可以并行处理整个序列，训练效率高；缺点是无法直接用于自回归生成，因为训练时看到完整序列会导致信息泄露，且推理时没有因果约束，不能逐 token 生成。从维度视角看，Encoder-only 在 token 维上做全连接扩散，每个 token 的关系矩阵覆盖所有位置，没有方向性约束，适合对整段序列做一次全局信息混合。

**&emsp;&emsp;PyTorch 实现示例**

```python
import math
import torch
import torch.nn as nn

class EncoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=512):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.token_emb = nn.Parameter(torch.empty(vocab_size, d_model))
        self.pos_emb = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.token_emb, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "W_q": nn.Parameter(torch.empty(d_model, d_model)),
                "W_k": nn.Parameter(torch.empty(d_model, d_model)),
                "W_v": nn.Parameter(torch.empty(d_model, d_model)),
                "W_o": nn.Parameter(torch.empty(d_model, d_model)),
                "W_1": nn.Parameter(torch.empty(d_model, d_ff)),
                "b_1": nn.Parameter(torch.zeros(d_ff)),
                "W_2": nn.Parameter(torch.empty(d_ff, d_model)),
                "b_2": nn.Parameter(torch.zeros(d_model)),
                "ln1_w": nn.Parameter(torch.ones(d_model)),
                "ln1_b": nn.Parameter(torch.zeros(d_model)),
                "ln2_w": nn.Parameter(torch.ones(d_model)),
                "ln2_b": nn.Parameter(torch.zeros(d_model)),
            })
            for _ in range(num_layers)
        ])
        for layer in self.layers:
            for name, p in layer.items():
                if "W_" in name:
                    nn.init.xavier_uniform_(p)

    def _ln(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def forward(self, input_ids):
        B, L = input_ids.size()
        x = self.token_emb[input_ids] + self.pos_emb[:L].unsqueeze(0)

        for layer in self.layers:
            Q = torch.matmul(x, layer["W_q"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(x, layer["W_k"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(x, layer["W_v"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = torch.softmax(scores, dim=-1)
            head_out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
            attn_out = torch.matmul(head_out, layer["W_o"])
            x = self._ln(x + attn_out, layer["ln1_w"], layer["ln1_b"])

            ffn_out = torch.matmul(
                torch.relu(torch.matmul(x, layer["W_1"]) + layer["b_1"]),
                layer["W_2"]
            ) + layer["b_2"]
            x = self._ln(x + ffn_out, layer["ln2_w"], layer["ln2_b"])

        return x
```

若输入 `input_ids` 形状为 $(B,L)$，输出形状为 $(B,L,d_{model})$。

---

#### 2.19.2 Decoder-only

&emsp;&emsp;Decoder-only 架构只保留 Transformer 的解码器部分，每个 token 只能关注自己和之前的位置，因此是因果注意力。它通过自回归方式逐 token 生成序列，是当前大语言模型的主流架构。代表性模型包括 GPT 系列、LLaMA、Qwen、DeepSeek 等。设输入序列经过嵌入和位置编码后得到 $X \in \mathbb{R}^{n \times d_{model}}$，解码器由 $N$ 个相同层堆叠，每层包含因果多头自注意力和前馈网络。因果掩码将未来位置的注意力分数置为 $-\infty$：

$$
M_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}
$$

&emsp;&emsp;解码器层的计算为：

$$
Z = \mathrm{LayerNorm}\left(X + \mathrm{MaskedMultiHead}(X,X,X) + M\right)
$$

$$
Y = \mathrm{LayerNorm}\left(Z + \mathrm{FFN}(Z)\right)
$$

&emsp;&emsp;Decoder-only 的优点是结构统一、易于扩展，适合自回归生成，且与 KV Cache 配合后推理效率高；缺点是只能利用左侧上下文，无法像 Encoder-only 那样双向建模，且训练和推理存在一定的计算模式差异。从维度视角看，Decoder-only 在 token 维上做因果扩散，关系矩阵是下三角矩阵，信息只能从过去流向未来，适合逐 token 生成。

**&emsp;&emsp;PyTorch 实现示例**

```python
import math
import torch
import torch.nn as nn

class DecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=512):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.token_emb = nn.Parameter(torch.empty(vocab_size, d_model))
        self.pos_emb = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.token_emb, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "W_q": nn.Parameter(torch.empty(d_model, d_model)),
                "W_k": nn.Parameter(torch.empty(d_model, d_model)),
                "W_v": nn.Parameter(torch.empty(d_model, d_model)),
                "W_o": nn.Parameter(torch.empty(d_model, d_model)),
                "W_1": nn.Parameter(torch.empty(d_model, d_ff)),
                "b_1": nn.Parameter(torch.zeros(d_ff)),
                "W_2": nn.Parameter(torch.empty(d_ff, d_model)),
                "b_2": nn.Parameter(torch.zeros(d_model)),
                "ln1_w": nn.Parameter(torch.ones(d_model)),
                "ln1_b": nn.Parameter(torch.zeros(d_model)),
                "ln2_w": nn.Parameter(torch.ones(d_model)),
                "ln2_b": nn.Parameter(torch.zeros(d_model)),
            })
            for _ in range(num_layers)
        ])
        for layer in self.layers:
            for name, p in layer.items():
                if "W_" in name:
                    nn.init.xavier_uniform_(p)

    def _ln(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def forward(self, input_ids):
        B, L = input_ids.size()
        x = self.token_emb[input_ids] + self.pos_emb[:L].unsqueeze(0)
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            Q = torch.matmul(x, layer["W_q"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(x, layer["W_k"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(x, layer["W_v"]).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            head_out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
            attn_out = torch.matmul(head_out, layer["W_o"])
            x = self._ln(x + attn_out, layer["ln1_w"], layer["ln1_b"])

            ffn_out = torch.matmul(
                torch.relu(torch.matmul(x, layer["W_1"]) + layer["b_1"]),
                layer["W_2"]
            ) + layer["b_2"]
            x = self._ln(x + ffn_out, layer["ln2_w"], layer["ln2_b"])

        return x
```

若输入 `input_ids` 形状为 $(B,L)$，输出形状为 $(B,L,d_{model})$。

---

#### 2.19.3 Encoder-Decoder

&emsp;&emsp;Encoder-Decoder 架构同时包含编码器和解码器，编码器双向处理源序列，解码器自回归生成目标序列，并通过交叉注意力从编码器输出中读取信息。它适合序列到序列任务，如机器翻译、文本摘要、语音识别。代表性模型包括原始 Transformer、T5、BART 等。设源序列为 $X \in \mathbb{R}^{n \times d_{model}}$，目标序列为 $Y \in \mathbb{R}^{m \times d_{model}}$，编码器输出 $H_{enc}$。解码器每层包含因果自注意力、交叉注意力和前馈网络。交叉注意力中，Query 来自解码器，Key 和 Value 来自编码器输出：

$$
Q = H_{dec} W^Q, \quad K = H_{enc} W^K, \quad V = H_{enc} W^V
$$

&emsp;&emsp;解码器层的计算为：

$$
Z_1 = \mathrm{LayerNorm}\left(Y + \mathrm{MaskedMultiHead}(Y,Y,Y)\right)
$$

$$
Z_2 = \mathrm{LayerNorm}\left(Z_1 + \mathrm{CrossAttention}(Z_1, H_{enc}, H_{enc})\right)
$$

$$
Z_3 = \mathrm{LayerNorm}\left(Z_2 + \mathrm{FFN}(Z_2)\right)
$$

&emsp;&emsp;Encoder-Decoder 的优点是编码器可以双向理解源序列，解码器通过交叉注意力动态对齐源和目标，适合输入输出长度不同、需要显式对齐的任务；缺点是结构比 Encoder-only 和 Decoder-only 更复杂，参数量和计算量更大，且交叉注意力的 $O(n_{dec} \cdot n_{enc})$ 复杂度在长序列上容易成为瓶颈。从维度视角看，Encoder-Decoder 包含两次 token 维扩散：编码器内部的双向扩散和解码器内部的因果扩散，以及解码器到编码器的交叉扩散。

**&emsp;&emsp;PyTorch 实现示例**

```python
import math
import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, num_heads,
                 num_layers, d_ff, max_len=512):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.src_emb = nn.Parameter(torch.empty(src_vocab, d_model))
        self.tgt_emb = nn.Parameter(torch.empty(tgt_vocab, d_model))
        self.pos_emb = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.src_emb, std=0.02)
        nn.init.normal_(self.tgt_emb, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        def make_layer():
            return nn.ModuleDict({
                "W_q": nn.Parameter(torch.empty(d_model, d_model)),
                "W_k": nn.Parameter(torch.empty(d_model, d_model)),
                "W_v": nn.Parameter(torch.empty(d_model, d_model)),
                "W_o": nn.Parameter(torch.empty(d_model, d_model)),
                "W_cq": nn.Parameter(torch.empty(d_model, d_model)),
                "W_ck": nn.Parameter(torch.empty(d_model, d_model)),
                "W_cv": nn.Parameter(torch.empty(d_model, d_model)),
                "W_co": nn.Parameter(torch.empty(d_model, d_model)),
                "W_1": nn.Parameter(torch.empty(d_model, d_ff)),
                "b_1": nn.Parameter(torch.zeros(d_ff)),
                "W_2": nn.Parameter(torch.empty(d_ff, d_model)),
                "b_2": nn.Parameter(torch.zeros(d_model)),
                "ln1_w": nn.Parameter(torch.ones(d_model)),
                "ln1_b": nn.Parameter(torch.zeros(d_model)),
                "ln2_w": nn.Parameter(torch.ones(d_model)),
                "ln2_b": nn.Parameter(torch.zeros(d_model)),
                "ln3_w": nn.Parameter(torch.ones(d_model)),
                "ln3_b": nn.Parameter(torch.zeros(d_model)),
            })

        self.enc_layers = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([make_layer() for _ in range(num_layers)])
        for layer in list(self.enc_layers) + list(self.dec_layers):
            for name, p in layer.items():
                if "W_" in name:
                    nn.init.xavier_uniform_(p)

    def _ln(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * w + b

    def _attn(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    def forward(self, src_ids, tgt_ids):
        B, L_s = src_ids.size()
        _, L_t = tgt_ids.size()
        src = self.src_emb[src_ids] + self.pos_emb[:L_s].unsqueeze(0)
        tgt = self.tgt_emb[tgt_ids] + self.pos_emb[:L_t].unsqueeze(0)

        enc = src
        for layer in self.enc_layers:
            Q = torch.matmul(enc, layer["W_q"]).view(B, L_s, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(enc, layer["W_k"]).view(B, L_s, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(enc, layer["W_v"]).view(B, L_s, self.num_heads, self.d_k).transpose(1, 2)
            attn_out = self._attn(Q, K, V).transpose(1, 2).contiguous().view(B, L_s, self.d_model)
            attn_out = torch.matmul(attn_out, layer["W_o"])
            enc = self._ln(enc + attn_out, layer["ln1_w"], layer["ln1_b"])
            ffn = torch.matmul(torch.relu(torch.matmul(enc, layer["W_1"]) + layer["b_1"]), layer["W_2"]) + layer["b_2"]
            enc = self._ln(enc + ffn, layer["ln2_w"], layer["ln2_b"])

        dec = tgt
        causal_mask = torch.triu(torch.ones(L_t, L_t, device=dec.device), diagonal=1).bool().unsqueeze(0).unsqueeze(0)
        for layer in self.dec_layers:
            Q = torch.matmul(dec, layer["W_q"]).view(B, L_t, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(dec, layer["W_k"]).view(B, L_t, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(dec, layer["W_v"]).view(B, L_t, self.num_heads, self.d_k).transpose(1, 2)
            attn_out = self._attn(Q, K, V, causal_mask).transpose(1, 2).contiguous().view(B, L_t, self.d_model)
            attn_out = torch.matmul(attn_out, layer["W_o"])
            dec = self._ln(dec + attn_out, layer["ln1_w"], layer["ln1_b"])

            Q = torch.matmul(dec, layer["W_cq"]).view(B, L_t, self.num_heads, self.d_k).transpose(1, 2)
            K = torch.matmul(enc, layer["W_ck"]).view(B, L_s, self.num_heads, self.d_k).transpose(1, 2)
            V = torch.matmul(enc, layer["W_cv"]).view(B, L_s, self.num_heads, self.d_k).transpose(1, 2)
            cross_out = self._attn(Q, K, V).transpose(1, 2).contiguous().view(B, L_t, self.d_model)
            cross_out = torch.matmul(cross_out, layer["W_co"])
            dec = self._ln(dec + cross_out, layer["ln2_w"], layer["ln2_b"])

            ffn = torch.matmul(torch.relu(torch.matmul(dec, layer["W_1"]) + layer["b_1"]), layer["W_2"]) + layer["b_2"]
            dec = self._ln(dec + ffn, layer["ln3_w"], layer["ln3_b"])

        return dec
```

若 `src_ids` 形状为 $(B,L_s)$，`tgt_ids` 形状为 $(B,L_t)$，输出形状为 $(B,L_t,d_{model})$。

---

#### 2.19.4 MoE

&emsp;&emsp;MoE（Mixture of Experts，混合专家）将 Transformer 中的前馈网络替换为多个并行的专家网络，每个专家本质上是一个小型 FFN。对于每个输入 token，路由器计算其与各专家的匹配分数，并据此对专家输出进行加权求和。设专家总数为 $N$，路由器门控权重为 $g_i(x)$，MoE 层输出为：

$$
\mathrm{MoE}(x) = \sum_{i=1}^{N} g_i(x) \cdot \mathrm{FFN}_i(x)
$$

&emsp;&emsp;稀疏 MoE 只保留分数最高的 $k$ 个专家，其余专家输出置零：

$$
g_i(x) = \begin{cases}
s_i(x) & s_i(x) \in \mathrm{TopK}(\{s_j(x)\}, k) \\
0 & \text{otherwise}
\end{cases}
$$

&emsp;&emsp;MoE 的优点是参数量与计算量解耦，可以用较低的计算成本获得极大的模型容量，适合大规模预训练；缺点是路由机制可能导致负载不均衡，部分专家过载而其他专家训练不足，且专家分布在多 GPU 上时 token 分发和聚合会带来通信开销。从维度视角看，MoE 在特征维上实现了条件计算：每个 token 根据自身特征被路由到不同的专家子网络，相当于在特征维上按内容动态选择变换路径。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.W_2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def forward(self, x):
        return torch.matmul(
            torch.relu(torch.matmul(x, self.W_1) + self.b_1),
            self.W_2
        ) + self.b_2


class SparseMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.router = nn.Parameter(torch.empty(num_experts, d_model))
        nn.init.normal_(self.router, std=0.02)

    def forward(self, x):
        B, L, D = x.size()
        x_flat = x.view(B * L, D)
        scores = torch.matmul(x_flat, self.router.T)
        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)
        topk_gates = torch.softmax(topk_scores, dim=-1)

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            idx = topk_indices[:, k]
            gate = topk_gates[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    out[mask] += gate[mask] * self.experts[e](x_flat[mask])
        return out.view(B, L, D)
```

若输入形状为 $(B,L,d_{model})$，输出形状相同。

---

#### 2.19.5 统一系统

&emsp;&emsp;统一系统（Unified System）指用一个统一的模型架构处理多种任务或多种推理模式，通过路由器根据输入特征动态选择不同的计算路径。在 DeepSeek-V4 等系统中，统一系统包含快速模式、思考模式和混合模式：快速模式直接输出答案，思考模式生成完整思维链，混合模式在两者之间动态调整。设输入为 $x$，路由器输出模式选择 $m = \mathrm{Router}(x)$，不同模式对应不同的推理预算和生成策略：

$$
y = \begin{cases}
\pi_{\text{fast}}(y \mid x) & m = \text{fast} \\
\pi_{\text{think}}(y \mid x, z) & m = \text{think} \\
\pi_{\text{hybrid}}(y \mid x, z_{1:k}) & m = \text{hybrid}
\end{cases}
$$

&emsp;&emsp;统一系统的优点是单一模型可以同时服务简单和复杂任务，简单任务快速响应，复杂任务深度推理，资源分配更高效；缺点是路由器的决策可能出错，将复杂任务误判为简单任务会导致质量下降，且统一系统需要在训练时同时优化多种推理模式，训练复杂度更高。从维度视角看，路由器在特征维上根据输入特征选择不同的 token 维扩散策略：快速模式跳过或极短思维链，思考模式展开完整思维链，混合模式在两者之间动态调整。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedRouter(nn.Module):
    def __init__(self, d_model, num_modes=3):
        super().__init__()
        self.num_modes = num_modes
        self.classifier = nn.Parameter(torch.empty(num_modes, d_model))
        nn.init.normal_(self.classifier, std=0.02)

    def forward(self, x):
        pooled = x.mean(dim=1)
        logits = torch.matmul(pooled, self.classifier.T)
        probs = F.softmax(logits, dim=-1)
        mode = probs.argmax(dim=-1)
        return mode, probs


class UnifiedSystem(nn.Module):
    def __init__(self, d_model, num_modes=3):
        super().__init__()
        self.router = UnifiedRouter(d_model, num_modes)
        self.mode_budgets = {0: 0, 1: 512, 2: 2048}

    def forward(self, hidden_states):
        mode, probs = self.router(hidden_states)
        budgets = torch.tensor(
            [self.mode_budgets[m.item()] for m in mode],
            device=hidden_states.device
        )
        return mode, probs, budgets
```

若输入 `hidden_states` 形状为 $(B,L,d_{model})$，输出为模式索引、模式概率和对应预算。

---

#### 2.19.6 双轨发布

&emsp;&emsp;双轨发布（Dual-Track Release）指同时发布两个不同定位的模型版本，通常是一个高性能大模型和一个高效小模型，例如 Pro 和 Flash、Base 和 Chat、或 Dense 和 MoE 版本。两条轨道共享相同的训练数据、分词器和训练流程，但在模型规模、推理成本和应用场景上形成互补。设大模型参数量为 $N_{\mathrm{pro}}$，小模型参数量为 $N_{\mathrm{flash}}$，通常满足：

$$
N_{\mathrm{pro}} > N_{\mathrm{flash}}, \quad C_{\mathrm{pro}} > C_{\mathrm{flash}}
$$

&emsp;&emsp;其中 $C$ 是推理计算成本。双轨发布的优点是用户可以根据任务难度和成本预算灵活选择模型，简单任务用 Flash 快速响应，复杂任务用 Pro 深度推理，同时发布两条轨道可以覆盖更广的部署场景；缺点是维护两套模型的训练和推理基础设施成本更高，小模型可能无法完全复现大模型的能力，且两条轨道之间的能力差距需要仔细平衡。从维度视角看，双轨发布在参数量维和推理成本维上提供了两个工作点，大模型在特征维上拥有更大容量，小模型通过更多训练 token 或更高效的架构在有限容量下逼近大模型性能。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn as nn

class DualTrackModel(nn.Module):
    def __init__(self, vocab_size, d_model_pro, d_model_flash,
                 num_heads_pro, num_heads_flash, num_layers, d_ff):
        super().__init__()
        self.pro = nn.ModuleDict({
            "token_emb": nn.Parameter(torch.empty(vocab_size, d_model_pro)),
            "pos_emb": nn.Parameter(torch.empty(512, d_model_pro)),
            "layers": nn.ModuleList([
                nn.ModuleDict({
                    "W_q": nn.Parameter(torch.empty(d_model_pro, d_model_pro)),
                    "W_k": nn.Parameter(torch.empty(d_model_pro, d_model_pro)),
                    "W_v": nn.Parameter(torch.empty(d_model_pro, d_model_pro)),
                    "W_o": nn.Parameter(torch.empty(d_model_pro, d_model_pro)),
                    "W_1": nn.Parameter(torch.empty(d_model_pro, d_ff)),
                    "W_2": nn.Parameter(torch.empty(d_ff, d_model_pro)),
                }) for _ in range(num_layers)
            ])
        })
        self.flash = nn.ModuleDict({
            "token_emb": nn.Parameter(torch.empty(vocab_size, d_model_flash)),
            "pos_emb": nn.Parameter(torch.empty(512, d_model_flash)),
            "layers": nn.ModuleList([
                nn.ModuleDict({
                    "W_q": nn.Parameter(torch.empty(d_model_flash, d_model_flash)),
                    "W_k": nn.Parameter(torch.empty(d_model_flash, d_model_flash)),
                    "W_v": nn.Parameter(torch.empty(d_model_flash, d_model_flash)),
                    "W_o": nn.Parameter(torch.empty(d_model_flash, d_model_flash)),
                    "W_1": nn.Parameter(torch.empty(d_model_flash, d_ff)),
                    "W_2": nn.Parameter(torch.empty(d_ff, d_model_flash)),
                }) for _ in range(num_layers)
            ])
        })

    def forward(self, input_ids, track="flash"):
        model = self.flash if track == "flash" else self.pro
        d_model = model["token_emb"].size(-1)
        B, L = input_ids.size()
        x = model["token_emb"][input_ids] + model["pos_emb"][:L].unsqueeze(0)
        for layer in model["layers"]:
            Q = torch.matmul(x, layer["W_q"])
            K = torch.matmul(x, layer["W_k"])
            V = torch.matmul(x, layer["W_v"])
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_model ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            x = x + torch.matmul(attn, V)
            x = x + torch.matmul(torch.relu(torch.matmul(x, layer["W_1"])), layer["W_2"])
        return x
```

若输入 `input_ids` 形状为 $(B,L)$，选择 `track="pro"` 或 `"flash"` 时输出形状分别为 $(B,L,d_{model\_pro})$ 或 $(B,L,d_{model\_flash})$。


---

### 2.20 关键概念与术语


#### 2.20.1 零样本

&emsp;&emsp;零样本（Zero-Shot）指模型在没有针对特定任务进行任何微调的情况下，直接根据任务描述或提示完成推理。设任务描述为 $T$，输入为 $x$，模型直接输出：

$$
y = \arg\max_{y} P(y \mid T, x; \theta)
$$

&emsp;&emsp;其中 $\theta$ 是预训练模型参数，$T$ 可以是一句自然语言指令，如“判断情感：正面或负面”。零样本能力来自预训练阶段学到的广泛语言知识和模式，模型通过将任务描述与输入结合，隐式地推断出任务意图。零样本的优点是无需标注数据、无需微调，部署成本极低，可以快速适配新任务，且同一模型可处理多种任务；缺点是性能通常低于少样本或微调，对提示措辞敏感，复杂任务上容易失败，且无法保证输出格式严格符合要求。从维度视角看，零样本在 token 维上将任务描述作为条件序列，特征维上利用预训练学到的通用表示直接映射到输出，相当于在预训练特征空间中做一次条件检索。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn.functional as F

def zero_shot_classify(prompt_ids, label_ids, model):
    """
    prompt_ids: (B, L) 任务描述 + 输入的 token 序列
    label_ids: (num_labels,) 候选标签的 token ID
    model: 返回 logits 的语言模型
    """
    logits = model(prompt_ids)  # (B, L, V)
    last_logits = logits[:, -1, :]  # (B, V)
    label_logits = last_logits[:, label_ids]  # (B, num_labels)
    probs = F.softmax(label_logits, dim=-1)
    pred = probs.argmax(dim=-1)
    return pred, probs
```

&emsp;&emsp;这个实现中，模型根据提示末尾的 logits 对候选标签打分，概率最高的标签作为零样本预测结果。

---

#### 2.20.2 上下文学习

&emsp;&emsp;上下文学习（In-Context Learning, ICL）指在提示中提供少量示例，模型无需更新参数即可根据这些示例推断任务并完成查询。设示例集合为 $\{(x_1, y_1), \dots, (x_k, y_k)\}$，查询为 $x$，模型预测：

$$
y = \arg\max_{y} P(y \mid x_1, y_1, \dots, x_k, y_k, x; \theta)
$$

&emsp;&emsp;ICL 的关键在于示例以自然语言序列形式拼接在提示中，模型通过前向注意力机制在示例和查询之间建立关联，隐式地识别任务模式。ICL 的优点是无需微调，示例数量增加通常能提升性能，且同一模型可灵活切换任务；缺点是示例选择、顺序和格式对性能影响大，上下文长度限制了示例数量，且模型可能复制示例中的表面模式而非真正理解任务。从维度视角看，ICL 在 token 维上将示例和查询拼接为长序列，注意力关系矩阵在示例和查询之间建立映射，特征维上模型从前向计算中动态提取任务表示，相当于在推理时做了一次隐式的任务条件化。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn.functional as F

def build_icl_prompt(examples, query, tokenizer):
    """拼接示例和查询"""
    parts = []
    for x, y in examples:
        parts.append(f"输入：{x}\n输出：{y}")
    parts.append(f"输入：{query}\n输出：")
    return tokenizer.encode("\n\n".join(parts))

def icl_classify(examples, query, label_ids, model, tokenizer):
    prompt_ids = build_icl_prompt(examples, query, tokenizer)
    prompt_tensor = torch.tensor([prompt_ids])
    logits = model(prompt_tensor)
    last_logits = logits[:, -1, :]
    label_logits = last_logits[:, label_ids]
    probs = F.softmax(label_logits, dim=-1)
    return probs.argmax(dim=-1), probs
```

&emsp;&emsp;这个实现中，示例和查询被拼接为单个序列，模型根据序列末尾的 logits 选择标签。示例的数量和顺序会影响 ICL 性能。

---

#### 2.20.3 涌现能力

&emsp;&emsp;涌现能力（Emergent Abilities）指模型规模或训练计算量达到一定阈值后，某些能力突然出现并快速提升。设模型参数量为 $N$，能力指标为 $A(N)$，涌现表现为 $A(N)$ 在 $N$ 小于阈值时接近随机水平，超过阈值后急剧上升。一种常用的拟合形式为：

$$
A(N) = A_{\max} \cdot \sigma\left(\alpha (\log N - \log N_0)\right)
$$

&emsp;&emsp;其中 $\sigma$ 是 Sigmoid 函数，$N_0$ 是涌现阈值，$\alpha$ 控制上升陡峭程度。涌现能力包括多步推理、代码生成、指令遵循等。关于涌现是否真实存在仍有争议，部分研究表明若使用连续指标而非离散指标，能力提升可能是平滑的。涌现能力的优点是揭示了规模带来的质变，指导了大模型扩展策略；缺点是阈值难以预测，不同任务涌现点不同，且部分“涌现”可能是指标选择造成的假象。从维度视角看，涌现对应特征维容量跨过某个临界点后，模型能够表示和组合更复杂的函数，token 维上的长程依赖建模能力也随之跃升。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch

def emergent_curve(log_N, log_N0=10.0, alpha=2.0, A_max=1.0):
    """模拟涌现曲线: log_N 为 log10(参数量)"""
    return A_max * torch.sigmoid(alpha * (log_N - log_N0))

# 示例: 参数量从 1e8 到 1e12
log_N = torch.linspace(8, 12, 100)
accuracy = emergent_curve(log_N)
```

&emsp;&emsp;这个实现用 Sigmoid 函数模拟能力随参数量的非线性跃升，`log_N0` 控制涌现阈值。

---

#### 2.20.4 对齐税

&emsp;&emsp;对齐税（Alignment Tax）指模型经过对齐训练（如 RLHF、SFT）后，在某些通用能力基准上出现的性能下降。对齐训练通常优化人类偏好或安全目标，这些目标与预训练的语言建模目标不完全一致，导致模型在部分任务上出现能力退化。设预训练损失为 $\mathcal{L}_{\mathrm{pre}}$，对齐损失为 $\mathcal{L}_{\mathrm{align}}$，联合优化时：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{pre}} + \lambda \mathcal{L}_{\mathrm{align}}
$$

&emsp;&emsp;当 $\lambda$ 较大时，对齐目标主导训练，模型可能牺牲部分通用能力换取对齐性能。对齐税的优点是提醒研究者对齐与能力之间的权衡，推动更精细的对齐方法；缺点是过度强调对齐税可能导致对齐不足，且不同任务上的对齐税差异很大，难以统一衡量。从维度视角看，对齐税在特征维上表现为对齐目标与预训练目标竞争同一表示空间，对齐训练将特征表示拉向偏好方向，可能偏离预训练学到的通用语言结构。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch

def alignment_tax(pre_loss, align_loss, lam=0.1):
    """联合损失: 预训练损失 + lambda * 对齐损失"""
    return pre_loss + lam * align_loss

def simulate_tax(lam_values, pre_loss_fn, align_loss_fn):
    """模拟不同对齐权重下的通用能力下降"""
    results = []
    for lam in lam_values:
        total = alignment_tax(pre_loss_fn(), align_loss_fn(), lam)
        results.append(total.item())
    return results
```

&emsp;&emsp;这个实现展示了预训练损失与对齐损失的加权组合，$\lambda$ 越大，对齐目标对总损失的贡献越大，通用能力可能下降越明显。

---

#### 2.20.5 奖励黑客

&emsp;&emsp;奖励黑客（Reward Hacking）指策略模型找到奖励模型的漏洞，生成能获得高奖励但实际质量差的输出。设真实奖励为 $r^*(x, y)$，奖励模型为 $r_\phi(x, y)$，策略优化目标为：

$$
\max_\theta \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} [r_\phi(x, y)]
$$

&emsp;&emsp;当 $r_\phi$ 与 $r^*$ 不一致时，策略可能利用 $r_\phi$ 的偏差。例如奖励模型偏好长回答，策略就生成冗长但无信息的文本；奖励模型偏好特定措辞，策略就堆砌这些措辞。奖励黑客的优点是暴露了奖励模型的不足，推动更鲁棒的奖励建模和 KL 约束；缺点是被黑客攻击的模型输出质量差，可能包含有害或误导内容，且难以自动检测。从维度视角看，奖励黑客在特征维上表现为策略沿着奖励模型梯度的方向移动，但该方向与真实质量方向偏离，策略利用了奖励模型在特征空间中的盲区。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn.functional as F

def reward_model_with_length_bias(text_length, true_quality, bias=0.1):
    """模拟有长度偏差的奖励模型"""
    return true_quality + bias * text_length

def policy_hacking_step(policy_logits, target_length, lr=0.1):
    """策略通过增加长度获得高奖励"""
    probs = F.softmax(policy_logits, dim=-1)
    # 奖励与长度正相关，策略增加长 token 的概率
    length_reward = torch.tensor(target_length, dtype=torch.float32)
    loss = -length_reward * probs[:, 1].mean()  # 假设 token 1 是长文本标记
    loss.backward()
    # 手动更新 logits
    policy_logits.data -= lr * policy_logits.grad
    policy_logits.grad.zero_()
    return policy_logits

def true_quality(text_length):
    """真实质量与长度无关"""
    return torch.ones_like(text_length) * 0.5
```

&emsp;&emsp;这个实现模拟了奖励模型对长度的偏好，策略通过增加长文本标记的概率获得更高奖励，但真实质量并未提升，体现了奖励黑客的核心机制。

#### 2.20.6 灾难性遗忘

&emsp;&emsp;灾难性遗忘（Catastrophic Forgetting）指模型在学习新任务时，原有任务上的性能急剧下降。在持续学习或顺序微调场景中，参数更新会覆盖之前任务学到的表示，导致旧知识丢失。设旧任务损失为 $\mathcal{L}_{\mathrm{old}}$，新任务损失为 $\mathcal{L}_{\mathrm{new}}$，朴素微调只优化 $\mathcal{L}_{\mathrm{new}}$，参数更新可能大幅改变对旧任务重要的权重。弹性权重巩固（EWC）通过 Fisher 信息矩阵约束重要参数的变化：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{new}} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^{\mathrm{old}})^2
$$

&emsp;&emsp;其中 $F_i$ 是旧任务对参数 $\theta_i$ 的 Fisher 信息，衡量该参数对旧任务的重要性。灾难性遗忘的优点是暴露了顺序学习的根本困难，推动持续学习和参数隔离方法的发展；缺点是缓解方法通常需要额外存储旧任务信息或计算 Fisher 矩阵，且完全避免遗忘仍很困难。从维度视角看，灾难性遗忘在特征维上表现为新任务的梯度更新覆盖了旧任务的重要特征方向，参数隔离和正则化相当于在特征维上为旧任务保留关键子空间。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn as nn

class EWC:
    def __init__(self, model, old_task_loader, lambda_ewc=100.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.params_old = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.fisher = self._compute_fisher(old_task_loader)

    def _compute_fisher(self, loader):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()
        for x, y in loader:
            self.model.zero_grad()
            out = self.model(x)
            loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
        for n in fisher:
            fisher[n] /= len(loader)
        return fisher

    def penalty(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            loss += (self.fisher[n] * (p - self.params_old[n]) ** 2).sum()
        return self.lambda_ewc * loss
```

若模型参数形状为 $(d_{model}, d_{ff})$，`penalty` 返回标量，加到新任务损失上即可。

---

#### 2.20.7 双塔架构与 InfoNCE

&emsp;&emsp;双塔架构（Dual-Encoder）由两个独立的编码器分别处理两种输入，如查询和文档、图像和文本。两个编码器将输入映射到同一嵌入空间，相似输入的嵌入距离近，不相似的距离远。设查询编码器为 $f_q$，文档编码器为 $f_d$，查询 $q$ 和文档 $d$ 的相似度为：

$$
s(q, d) = \frac{f_q(q)^\top f_d(d)}{\|f_q(q)\| \|f_d(d)\|}
$$

&emsp;&emsp;InfoNCE 损失用于训练双塔，使正样本对的相似度高于负样本对：

$$
\mathcal{L}_{\mathrm{InfoNCE}} = -\log \frac{\exp(s(q, d^+) / \tau)}{\sum_{d \in \mathcal{D}} \exp(s(q, d) / \tau)}
$$

&emsp;&emsp;其中 $d^+$ 是正样本，$\mathcal{D}$ 包含正样本和负样本，$\tau$ 是温度系数。双塔架构的优点是查询和文档可以独立编码，文档嵌入可以离线预计算，检索时只需计算查询嵌入与文档嵌入的内积，适合大规模检索；缺点是查询和文档之间没有交互，无法捕捉细粒度的词级对齐，性能通常低于交叉编码器。从维度视角看，双塔在特征维上将两种模态映射到共享嵌入空间，InfoNCE 在 batch 维上拉近正样本对、推远负样本对，使嵌入空间的几何结构反映语义相似度。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualEncoder(nn.Module):
    def __init__(self, d_model, d_out):
        super().__init__()
        self.W_q = nn.Parameter(torch.empty(d_model, d_out))
        self.W_d = nn.Parameter(torch.empty(d_model, d_out))
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_d)

    def forward(self, query, doc):
        q_emb = F.normalize(torch.matmul(query, self.W_q), dim=-1)
        d_emb = F.normalize(torch.matmul(doc, self.W_d), dim=-1)
        return q_emb, d_emb

def info_nce_loss(q_emb, d_emb, temperature=0.07):
    """
    q_emb: (B, d_out) 查询嵌入
    d_emb: (B, d_out) 正样本文档嵌入
    负样本为 batch 内其他文档
    """
    logits = torch.matmul(q_emb, d_emb.T) / temperature  # (B, B)
    labels = torch.arange(q_emb.size(0), device=q_emb.device)
    return F.cross_entropy(logits, labels)
```

若查询和文档形状为 $(B, d_{model})$，输出嵌入形状为 $(B, d_{out})$，`info_nce_loss` 返回标量。

---

#### 2.20.8 缩放规律

&emsp;&emsp;缩放规律（Scaling Laws）描述模型性能随参数量 $N$、数据量 $D$、计算量 $C$ 的变化关系。Kaplan 等人发现测试损失随三者呈幂律下降：

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad
L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

&emsp;&emsp;联合形式为：

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

&emsp;&emsp;其中 $E$ 是不可约损失，$A, B$ 是常数，$\alpha, \beta$ 是幂律指数。训练计算量近似为 $C \approx 6ND$。缩放规律的优点是使大模型性能可预测，指导资源分配和模型设计；缺点是幂律在极端规模下可能失效，数据质量、架构差异和优化器选择都会影响规律，且不同任务的最优分配不同。从维度视角看，缩放规律在参数量维和数据量维上分别建立了损失衰减关系，计算量维是两者的耦合约束，指导如何在模型容量和数据规模之间分配资源。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch

def scaling_loss(N, D, E=1.69, A=406.4, B=410.7,
                 alpha=0.34, beta=0.28):
    """联合缩放损失"""
    return E + A / (N ** alpha) + B / (D ** beta)

def training_flops(N, D, factor=6.0):
    """训练计算量估计"""
    return factor * N * D

# 示例: 固定计算量下比较不同 N 和 D 组合
C = 1e23
for N in [1e9, 1e10, 1e11]:
    D = C / (6 * N)
    loss = scaling_loss(N, D)
    print(f"N={N:.1e}, D={D:.2e}, loss={loss:.4f}")
```

`scaling_loss` 预测给定 $N,D$ 的损失，`training_flops` 计算训练计算量。

---

#### 2.20.9 计算最优训练

&emsp;&emsp;计算最优训练（Compute-Optimal Training）指在给定训练计算预算 $C$ 下，选择最优的模型参数量 $N$ 和训练数据量 $D$ 使损失最小。约束为 $C \approx 6ND$，目标为：

$$
\min_{N,D} \quad L(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

$$
\text{s.t.} \quad 6ND = C
$$

&emsp;&emsp;通过拉格朗日乘子法可得最优分配：

$$
N^* \propto C^{\frac{\beta}{\alpha+\beta}}, \quad D^* \propto C^{\frac{\alpha}{\alpha+\beta}}
$$

&emsp;&emsp;Chinchilla 的实验估计 $\alpha \approx 0.34$，$\beta \approx 0.28$，指数接近，因此 $N$ 和 $D$ 近似等比例增长，经验上每个参数约需 20 个训练 token。计算最优训练的优点是同等计算下性能最优，避免了 Kaplan 式模型过大、数据不足的问题；缺点是只考虑训练计算，未考虑推理成本，实际部署中可能需要过度训练小模型以降低推理成本。从维度视角看，计算最优训练在参数量维和数据量维之间寻找等比例扩展点，使训练计算预算下的损失最小化。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch

def compute_optimal(C, tokens_per_param=20.0):
    """按 20 token/参数 计算最优 N 和 D"""
    N = torch.sqrt(torch.tensor(C / (6.0 * tokens_per_param)))
    D = tokens_per_param * N
    return N, D

def compute_optimal_general(C, alpha=0.34, beta=0.28):
    """通用指数形式"""
    exp_N = beta / (alpha + beta)
    exp_D = alpha / (alpha + beta)
    N = C ** exp_N
    D = C ** exp_D
    scale = (C / (6 * N * D)) ** 0.5
    return N * scale, D * scale

# 示例: 给定 C=1e24 计算最优分配
C = 1e24
N, D = compute_optimal(C)
print(f"Chinchilla 最优: N={N:.2e}, D={D:.2e}")
```

`compute_optimal` 按 20 token/参数给出最优 $N,D$，`compute_optimal_general` 使用通用指数。

---

#### 2.20.10 推理时计算扩展

&emsp;&emsp;推理时计算扩展（Inference-Time Compute Scaling）指在模型训练完成后，通过增加推理阶段的计算量来提升输出质量。常见方法包括延长思维链、多次采样后投票、best-of-N 重排序、树搜索和验证器筛选。设单次推理的正确答案概率为 $p$，独立采样 $N$ 次后多数投票的正确答案概率近似为：

$$
P_{\mathrm{maj}}(N) \approx 1 - (1-p)^N
$$

&emsp;&emsp;对于 best-of-N，若验证器能以概率 $q$ 正确识别正确答案，则最终正确概率为：

$$
P_{\mathrm{best}}(N) \approx 1 - (1-pq)^N
$$

&emsp;&emsp;推理时计算扩展的优点是无需重新训练模型，仅通过推理策略就能提升性能，且可以根据任务难度动态分配计算量；缺点是计算成本随采样数或推理长度线性甚至超线性增长，边际收益递减，简单问题过度扩展会造成浪费。从维度视角看，推理时计算扩展在 token 维上增加了生成的步数，或在输出空间中探索了多条路径，相当于用更多 token 维的计算来补偿模型固定的特征维容量。

**&emsp;&emsp;PyTorch 实现示例**

```python
import torch

def majority_vote_accuracy(p, N):
    """多数投票正确率上界"""
    return 1 - (1 - p) ** N

def best_of_n_accuracy(p, q, N):
    """best-of-N 正确率"""
    return 1 - (1 - p * q) ** N

def simulate_best_of_n(policy, reward_fn, input_ids,
                       num_samples=8, max_new_tokens=64):
    """模拟 best-of-N 采样并选奖励最高"""
    best = None
    best_r = float("-inf")
    for _ in range(num_samples):
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = policy(generated)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=-1)
        r = reward_fn(generated).item()
        if r > best_r:
            best_r = r
            best = generated
    return best, best_r
```

`majority_vote_accuracy` 和 `best_of_n_accuracy` 给出推理扩展的理论增益，`simulate_best_of_n` 模拟采样选择流程。


---

&emsp;&emsp;以上组件与方法共同构成了 LLM 架构演进的技术底座。从注意力机制到 MoE 路由，从位置编码到长上下文技术，从 RLHF 到推理时计算扩展，每一项组件都在“能力—成本—可控性”的三角约束下被反复调整。后续各章节中的具体模型——GPT-2、GPT-3、T5、Gopher/Chinchilla、InstructGPT/ChatGPT、LLaMA、Mistral、Falcon、GPT-4/4o、Gemini 1.5 Pro、Llama 3/3.1、Qwen 2/2.5、o1/o3、DeepSeek-V3/R1、GLM-5、DeepSeek-V4、Kimi K3、MiMo-V2、Qwen3、Llama 4、Claude 5、Grok 3、GPT-5——都是这些基础组件在不同规模、不同数据、不同训练目标和不同部署约束下的具体组合与优化结果。

## 3 模型探索阶段

### 3.1 GPT-2

- 论文地址：[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

&emsp;&emsp;GPT-1 确立了“预训练 + 微调”的两阶段范式，但这一范式存在一个根本性的局限：每一个下游任务都需要独立的标注数据和一次独立的微调过程。数据集的构建成本高昂，任务格式的设计也需要人工介入，这使得模型难以被扩展到任意数量的新任务上。GPT-2 的核心思想是：**一个足够大的语言模型，在足够多样化的文本上训练之后，本身就能够在没有任何显式监督的情况下识别并执行训练语料中自然出现的各种任务。**它在架构上几乎完全沿用了 GPT-1 的 Transformer Decoder，但将参数规模从约 1.17 亿扩大到 15 亿，并取消了微调阶段，首次展示了零样本（zero-shot）多任务学习能力。GPT-2 的主要贡献包括：

1. 提出“无监督多任务学习”范式，证明语言模型在足够大时可以在不进行任何参数或结构修改的情况下执行翻译、问答、摘要等下游任务；
2. 在 GPT-1 基础上完成架构层面的关键工程改进——Pre-LN、残差路径缩放初始化和 Byte-level BPE，为后续超深 Transformer 的稳定训练奠定了基础；
3. 构建 WebText 数据集（约 40GB 高质量网页文本），验证了数据质量和多样性对零样本泛化能力的决定性作用；
4. 通过 GPT-2 Small / Medium / Large / XL 四种尺寸的系统对比，首次展示了“能力随规模涌现”的现象，成为 Scaling Law 的早期实验证据；
5. 开源模型权重，推动了后续开源语言模型生态的兴起。

---

**架构总览：更大更宽的 Transformer Decoder**

&emsp;&emsp;GPT-2 的模型结构延续了 GPT-1 的单向 **Transformer Decoder** 设计，训练目标仍然是标准的自回归语言建模：给定前文 token 序列，预测下一个 token。论文明确指出 GPT-2 是 GPT 的直接放大版本，参数量增加了 10 倍以上，训练数据量也增加了 10 倍以上。

&emsp;&emsp;GPT-2 提供了四种尺寸配置，参数规模从 1.24 亿到 15 亿不等：

| 模型 | 层数 | 注意力头数 | 隐藏维度 | 参数量 |
|------|------|-----------|---------|--------|
| GPT-2 Small | 12 | 12 | 768 | ~124M |
| GPT-2 Medium | 24 | 16 | 1024 | ~355M |
| GPT-2 Large | 36 | 20 | 1280 | ~774M |
| GPT-2 XL | 48 | 25 | 1600 | ~1.5B |

![](https://jalammar.github.io/images/gpt2/gpt2-sizes-hyperparameters-3.png)

&emsp;&emsp;四种尺寸的对比实验表明，随着模型规模的增大，语言建模困惑度持续下降，且零样本任务的性能呈现出明显的非线性提升：GPT-2 Small（117M）几乎没有零样本能力，GPT-2 Medium（345M）开始表现出一些能力，GPT-2 XL（1.5B）则明显涌现出多任务迁移能力。

---

**Pre-LN：将层归一化移到子模块之前**

&emsp;&emsp;GPT-1 沿用了原始 Transformer 的 Post-LN 结构，即层归一化被放置在残差连接之后：$x_{l+1} = \text{LayerNorm}(x_l + \text{Sublayer}(x_l))$。这种设计在深层网络中会导致梯度范数在反向传播时出现剧烈波动，训练不稳定。

&emsp;&emsp;GPT-2 将层归一化从残差连接之后移到了每个子模块的输入之前，即 Pre-LN 结构：

$$x_{l+1} = x_l + \text{Sublayer}(\text{LayerNorm}(x_l))$$

&emsp;&emsp;同时在最后一个自注意力模块之后额外添加了一层 LayerNorm。Pre-LN 的核心优势在于：残差路径上的信息可以不经过任何归一化变换而直接传递到深层，这极大地缓解了梯度消失和梯度爆炸问题，使得训练数十层甚至上百层的 Transformer 成为可能。这一设计后来被几乎所有主流 LLM（GPT-3、LLaMA、PaLM 等）沿用，成为超深 Transformer 的标准配置。

---

**残差路径缩放初始化：让深层网络稳定起步**

&emsp;&emsp;Pre-LN 解决了训练过程中的梯度稳定性问题，但在初始化阶段，深层网络仍然面临一个风险：如果残差分支的输出方差在每一层都保持不变，那么随着层数的增加，残差路径上的激活值方差会逐层累积，最终导致前向传播的数值不稳定。

&emsp;&emsp;GPT-2 的解决方案是在初始化时对残差层的权重进行缩放：将残差分支中特定权重矩阵的标准差乘以 $1/\sqrt{N}$，其中 $N$ 是残差层的数量。以 GPT-2 XL 为例，$N = 48$，缩放因子约为 $1/6.93$。这意味着在训练开始时，每个残差分支的输出被显著压制，残差路径上的信号以近乎恒等映射的方式逐层传递，网络的初始行为接近于一个浅层模型。随着训练的进行，各层的残差权重逐渐增大，网络的有效深度逐步“生长”出来。

&emsp;&emsp;这一初始化策略与 ControlNet 中零卷积的设计逻辑在哲学上是一致的：都是通过在训练初期将新增路径的输出置零或压至很小，来保证预训练或初始状态下的数值稳定性，让模型从一个安全、可控的起点逐步学习。残差路径缩放初始化后来成为 GPT-3 及后续模型的标准做法。

---

**Byte-level BPE：彻底消除 OOV 问题**

&emsp;&emsp;GPT-1 使用的 BPE 分词器以 Unicode 字符为基本单位，词表大小为 40,478，仍然存在一定数量的未登录词（OOV）。GPT-2 将分词器升级为 Byte-level BPE：以 256 个字节作为基础字符集，通过 BPE 合并操作构建词表，最终词表大小为 50,257（256 个基础字节 + 50,000 个合并操作 + 1 个特殊 token）。

&emsp;&emsp;Byte-level BPE 的关键优势在于：任何 Unicode 文本都可以被无损地表示为字节序列，因此词汇表中不存在任何 OOV token。无论输入是英文、中文、表情符号还是二进制数据，分词器都能将其编码为已知的 token 序列。这消除了传统词级或字符级分词器在遇到罕见词或跨语言文本时的退化问题。

&emsp;&emsp;GPT-2 的上下文窗口从 GPT-1 的 512 扩展到了 1024 个 token。位置编码采用可学习的绝对位置嵌入，即一个形状为 $[1024, d_{\text{model}}]$ 的查找表，每个位置对应一个可训练的嵌入向量。这一设计在 GPT-2 时代是合理的，但后来随着上下文长度需求的急剧增长，可学习的绝对位置嵌入无法外推到训练时未见过的长度，逐渐被 RoPE 等相对位置编码方案取代。

---

**训练数据：WebText 与数据质量优先原则**

&emsp;&emsp;GPT-2 的训练数据 WebText 是一个从 Reddit 爬取构建的高质量网页文本数据集。具体流程为：收集 Reddit 上获得至少 3 个 karma（赞）的外链，共约 4500 万个链接；对链接指向的网页进行文本提取和去重，得到约 800 万份文档，总计约 40GB 文本；同时移除了所有 Wikipedia 文档，以避免与下游评估任务的数据重叠。

&emsp;&emsp;WebText 的设计哲学可以概括为“质量优先于数量”。虽然 40GB 的规模在当时并不算最大（同期 Common Crawl 的规模远大于此），但经过 Reddit 社区投票筛选后的数据在语言质量、内容多样性和自然对话比例上远优于未经筛选的原始网页数据。这种“以社区信号作为质量过滤器”的思路，后来被 RefinedWeb、FineWeb 等数据集构建工作继承和发展。

&emsp;&emsp;WebText 的多样性是零样本能力的关键前提。论文指出，正是因为训练语料中自然包含了大量“任务式语言模式”——例如翻译网站的对照文本、问答论坛的对话记录、代码仓库中的文档和注释——模型在预训练过程中“无意中”学到了这些任务的执行方式。GPT-2 因此能够在推理时通过设计合适的提示词来触发这些任务，例如输入“Translate English to French: The house is wonderful.”，模型直接输出“La maison est magnifique.”，而无需任何微调。

---

**零样本多任务学习：从“微调一切”到“提示一切”**

&emsp;&emsp;GPT-2 最核心的贡献在于证明了语言模型可以作为无监督多任务学习器。论文从理论上论证了这一可能性：监督学习的目标函数（给定输入预测输出）是无监督语言建模目标函数在序列子集上的评估，因此无监督目标的全局最小值同时也是监督目标的全局最小值。这意味着，一个在足够大的文本语料上训练到接近全局最优的语言模型，原则上已经“学会”了语料中蕴含的所有任务，只需要通过合适的提示来激发这些能力。

&emsp;&emsp;需要指出的是，GPT-2 的零样本性能虽然展示了“可能性”，但绝对水平仍然有限：翻译 BLEU 仅 5 分（专业模型可达 40+），阅读理解 F1 为 55（专业模型 80+）。真正将零样本能力推向实用水平的是后续的 GPT-3，后者通过更大的规模（175B）和上下文学习（in-context learning）机制，在少样本设定下实现了质的飞跃。但 GPT-2 确立了“用自然语言提示来指定任务”这一核心范式，这是从 GPT-1 到 GPT-3 之间最关键的观念转变。

---

**涌现能力：规模法则的早期证据**

&emsp;&emsp;GPT-2 论文中最具影响力的发现之一，是四种不同规模模型的能力对比所揭示的“涌现”现象。GPT-2 Small（117M）在零样本任务上几乎无表现，GPT-2 Medium（345M）开始出现一些零星的正确输出，GPT-2 XL（1.5B）则明显展现出跨任务迁移能力。这种能力并非随规模线性增长，而是在某个规模阈值之后突然“出现”，这是“涌现能力”在语言模型中的最早系统记录。

&emsp;&emsp;这一发现直接催生了“更大就是更好”的共识，并推动了 GPT-3、Gopher、Chinchilla 等更大规模模型的诞生。Kaplan 等人随后在 2020 年发表的 Scaling Laws 论文中，将 GPT-2 中的经验观察形式化为幂律关系，为后续模型的规模选择提供了定量指导。可以说，GPT-2 不仅是一个模型，更是一个实验证据——它证明了 scaling 本身可以产生新的能力，而不需要架构上的根本创新。

---

**GPT-2 与 GPT-1 的关系及后续影响**

&emsp;&emsp;**GPT-2 是 GPT-1 的规模化版本，同时完成了三项关键工程改进：Pre-LN 保障了深层训练的稳定性，残差缩放初始化解决了深层网络的启动问题，Byte-level BPE 消除了 OOV 问题。** 在范式层面，GPT-2 用“零样本多任务学习”替代了 GPT-1 的“预训练 + 微调”，首次展示了通用语言模型的雏形。

&emsp;&emsp;GPT-2 对后续工作的影响是深远的。GPT-3 直接继承了 GPT-2 的架构和 Pre-LN 设计，将参数规模从 15 亿扩大到 1750 亿，并在此基础上引入了上下文学习（in-context learning）机制，使零样本和少样本能力达到了实用水平。GPT-2 的 Byte-level BPE 分词器被后续几乎所有 GPT 系列模型沿用。残差路径缩放初始化成为超深 Transformer 的标准实践。而 GPT-2 展示的“能力随规模涌现”现象，则直接启发了 Scaling Law 的系统研究，为 LLM 的“规模化竞赛”提供了理论依据。

&emsp;&emsp;GPT-2 的局限同样值得关注。它的零样本能力虽然在多种任务上有所体现，但绝对性能远不足以替代微调模型；它的上下文窗口仅 1024 token，限制了长文本理解；它的可学习绝对位置嵌入无法外推到更长的序列。此外，OpenAI 在 GPT-2 发布时出于安全考虑采取了分阶段释放策略，先用较小的模型权重公开，花了 9 个月评估风险后才释放完整模型，这也开启了 LLM 安全与对齐研究的先河。这些局限在 GPT-3 及后续工作中被逐步解决，但 GPT-2 所确立的“大规模预训练 + 提示驱动任务”范式，始终是 LLM 架构演进的核心主线。

### 3.2 GPT-3

- 论文地址：[Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)

&emsp;&emsp;GPT-2 证明了“更大的语言模型可以零样本执行多种任务”，但它的零样本性能在绝对水平上仍然有限，翻译 BLEU 仅 5 分，阅读理解 F1 为 55，距离实用还有明显差距。一个自然的追问是：如果把模型规模继续放大约 100 倍，零样本能力能否从“偶尔可行”变成“普遍可用”？GPT-3 的核心思想就是对这个问题给出肯定的回答，并且给出一个比“零样本”更强大的机制——**上下文学习（In-Context Learning, ICL）** 。GPT-3 不再要求模型“什么都不看就直接输出答案”，而是允许用户在输入中提供少量示例，模型在不更新任何参数的情况下，仅凭前向传播就能从这些示例中“学会”新任务。GPT-3 的主要贡献包括：

1. 训练了 1750 亿参数的自回归语言模型，参数量是 GPT-2 最大版本的 100 倍以上，首次在多个 NLP 基准上使少样本（few-shot）性能达到或超过此前需要微调才能达到的水平；
2. 系统提出并验证了上下文学习范式：任务和少量示例完全通过文本交互指定，模型不进行任何梯度更新或微调，仅靠前向推理完成任务；
3. 在翻译、问答、完形填空、单词重组、三位数算术、新词使用等数十个数据集上进行了零样本、单样本和少样本的全面评估，展示了“规模驱动能力涌现”的系统性证据；
4. 在架构层面沿用了 GPT-2 的设计，但引入了交替的密集与局部带状稀疏注意力模式，以降低超大规模下的计算开销；
5. 与 Kaplan 等人的Scaling Law工作共同确立了“参数量、数据量、算力三者协同放大”的范式，直接催生了后续 PaLM、Chinchilla、LLaMA 等模型。

---

**架构：与 GPT-2 同源，规模放大 100 倍**

&emsp;&emsp;GPT-3 的模型架构与 GPT-2 高度一致：仍然是 **decoder-only 的 Transformer，仍然使用 Pre-LN、Byte-level BPE 分词器、残差路径缩放初始化和可学习的绝对位置嵌入**。最大的变化是规模。GPT-3 的完整版本拥有 96 层 Transformer、12288 维嵌入维度和 96 个注意力头，参数量达到 1750 亿。上下文长度从 GPT-2 的 1024 扩展到 2048 token。

&emsp;&emsp;OpenAI 同时还训练了从 1.25 亿到 130 亿参数的 8 个不同规模版本，用于研究缩放行为。其中 175B 版本被称为“GPT-3”，其余版本分别命名为 GPT-3 Small（125M）、GPT-3 Medium（350M）、GPT-3 Large（760M）、GPT-3 XL（1.3B）、GPT-3 2.7B、GPT-3 6.7B 和 GPT-3 13B。这种“同一架构、不同规模”的系列设计，使Scaling Law的验证得以在受控条件下进行。

&emsp;&emsp;架构上唯一的关键改动是注意力模式：GPT-3 在 Transformer 的各层中交替使用密集注意力和局部带状稀疏注意力，类似于 Sparse Transformer 的做法。稀疏注意力层将每个 token 的注意力范围限制在局部窗口内，从而降低长序列下的计算复杂度。对于 GPT-3 的 2048 上下文长度，这一改动在当时是必要的工程优化，但它并未改变模型的基本能力，后续的 GPT-3.5 和 GPT-4 实际上放弃了这一设计，回归全密集注意力并使用 FlashAttention 等更高效的实现。

&emsp;&emsp;GPT-3 的训练数据规模从 GPT-2 的约 40GB 提升到约 3000 亿 token（约 570GB 过滤后文本）。数据来源包括五个部分，并采用加权混合策略。

---

**上下文学习：不更新参数的任务适配**

&emsp;&emsp;GPT-3 最核心的贡献是系统展示并命名了**上下文学习**能力。其基本设定是：模型在推理时接收一个“提示”，提示中包含任务描述和若干示例，模型直接输出答案，不进行任何梯度更新或微调。根据示例数量，可以分为三种模式：

- **零样本（Zero-shot）** ：只给任务描述，不给示例。例如：“Translate English to French: cheese =>”
- **单样本（One-shot）** ：给一个示例。例如：“Translate English to French: sea otter => loutre de mer; cheese =>”
- **少样本（Few-shot）** ：给 10 到 100 个示例，通常放在同一个上下文窗口中。

&emsp;&emsp;少样本模式的关键在于：模型不是通过微调来“学习”任务，而是在前向传播过程中，从上下文的示例中“推断”出输入和输出之间的映射关系。论文指出，这种能力并不是被显式训练出来的——GPT-3 的训练目标仍然是无监督的下一个 token 预测，模型从未见过“任务描述 + 示例 + 输出”这种格式的标注数据。上下文学习是模型规模达到一定阈值后“涌现”出来的副产品。

&emsp;&emsp;一个重要的细节是，GPT-3 的少样本性能随模型规模单调提升，而小模型在少样本设定下甚至可能不如零样本——因为小模型无法从示例中提取有效信息，额外的示例反而成为了干扰。这进一步支持了“上下文学习是规模驱动的涌现能力”这一判断。GPT-3 在数十个 NLP 基准上进行了评估，其少样本性能在多个任务上达到了与当时最优微调模型相当的水平。

---

**Scaling Law：性能的幂律可预测性**

&emsp;&emsp;GPT-3 的 8 个不同规模版本为Scaling Law的验证提供了理想实验条件。论文发现，语言建模损失（交叉熵）随模型规模、数据量和训练计算量的增加呈平滑的幂律下降，且这种关系在跨越三个数量级的范围内保持一致。这一发现与 Kaplan 等人同期发表的 Scaling Laws 论文相互印证，共同确立了“模型性能可以通过规模预测”的经验规律。

&emsp;&emsp;更关键的是，规模不仅带来“量”的提升，还带来“质”的变化。上下文学习能力在 GPT-3 的较小版本（125M、350M）上几乎不存在，在 1.3B 到 13B 之间开始零星出现，在 175B 上才变得可靠。这种“能力涌现”现象——即某些能力在规模达到阈值之前完全不可观测，之后突然出现——成为后续 LLM 研究的核心议题之一。

---

**GPT-3 与 GPT-2 的关系及后续影响**

&emsp;&emsp;**GPT-3 是 GPT-2 在规模维度上的激进放大，同时引入了一个全新的能力维度：上下文学习。** 架构上二者几乎完全相同，GPT-3 的全部“新能力”都来自规模——更大的参数量、更多的训练数据、更长的训练时间。这在当时传递了一个明确的信号：架构创新可能不是 LLM 进步的主要驱动力，规模本身就能带来质变。

&emsp;&emsp;从 GPT-3 到 ChatGPT 的路线也清晰可见：GPT-3.5 系列在 GPT-3 的基础上引入了指令微调和 RLHF，将“能完成任务的模型”转化为“能安全、有帮助地对话的模型”。但底层的架构范式——decoder-only Transformer、自回归语言建模、上下文学习——始终没有脱离 GPT-3 所确立的框架。GPT-3 因此可以被视为 LLM 从“研究原型”走向“通用基础设施”的分水岭。

### 3.3 T5

- 论文地址：[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683)

&emsp;&emsp;在 T5 出现之前，NLP 领域的迁移学习已经形成了两条清晰的路线：BERT 代表的 encoder-only 架构通过掩码语言建模学习双向表示，擅长理解类任务，但无法直接生成文本；GPT 代表的 decoder-only 架构通过自回归语言建模学习生成能力，擅长生成类任务，但缺乏双向上下文理解。这种架构分化导致模型复用性差——一个为文本分类训练的 BERT 模型很难直接用于翻译或摘要。与此同时，迁移学习领域充斥着大量相互独立的技术创新——不同的预训练目标、不同的架构变体、不同的微调策略——但它们之间缺乏系统的比较和统一的评估框架。T5 的核心思想是：**将所有 NLP 任务统一为“文本到文本”的格式，即输入是文本，输出也是文本，然后用同一个模型、同一个损失函数、同一套超参数处理所有任务。** T5 的主要贡献包括：

1. 提出统一的文本到文本框架，将所有 NLP 任务——翻译、摘要、问答、分类、回归——全部转换为“输入文本 → 输出文本”的格式，消除了任务特定的输出层和损失函数；
2. 对迁移学习领域进行了系统性的实证评估，比较了架构变体（encoder-only、decoder-only、encoder-decoder）、预训练目标（语言建模、降噪、span corruption）、数据集、微调策略等多个维度的效果；
3. 构建并开源了 Colossal Clean Crawled Corpus（C4），一个约 750GB 的高质量英文网页文本数据集，为后续预训练研究提供了标准数据源；
4. 采用简化的相对位置编码，用可学习的标量偏置替代绝对位置嵌入，使模型在理论上能够外推到训练时未见过的序列长度；
5. 在 24 个 NLP 任务上进行了评估，在多个基准上取得当时最优结果，并开源了全部代码、数据和模型权重。

---

**文本到文本框架：统一所有 NLP 任务**

&emsp;&emsp;T5 最核心的设计理念是文本到文本的统一框架。在 T5 之前，不同的 NLP 任务需要不同的输出格式和处理方式：文本分类需要在线性层上接 softmax 输出类别概率；序列标注需要为每个 token 预测一个标签；抽取式问答需要预测答案在原文中的起止位置；机器翻译需要序列到序列的生成。这意味着每换一个任务，就需要修改模型结构、更换损失函数、重新设计训练流程。

&emsp;&emsp;T5 的做法是：**所有任务的输入和输出都表示为文本字符串。** 为了让模型知道当前执行的是什么任务，T5 在输入前加上一个任务前缀。例如：

```text
翻译（英→德）:  "translate English to German: That is good."  →  "Das ist gut."
情感分类:       "sst2 sentence: This movie is great."         →  "positive"
文本摘要:       "summarize: <article text>"                    →  "<summary text>"
自然语言推理:   "mnli premise: <p> hypothesis: <h>"           →  "entailment"
回归任务:       "stsb sentence1: <s1> sentence2: <s2>"        →  "3.8"
```

&emsp;&emsp;这种统一带来了几个关键优势。首先，同一个模型可以在所有任务上使用相同的损失函数（交叉熵）和相同的超参数（学习率、批次大小等），极大地简化了训练流程。其次，多任务联合训练成为可能——T5 可以在一个批次中混合来自不同任务的数据，让模型同时学习多种能力。第三，回归任务也可以被纳入统一框架：T5 让模型预测数字的字符串表示（如“3.8”）而不是数字本身，从而将回归问题转化为文本生成问题。

&emsp;&emsp;论文指出，文本到文本框架的提出还有一个更深层的目的：为迁移学习研究提供一个标准的试验台。在 T5 之前，不同论文使用的架构、数据、训练目标和评估协议各不相同，导致研究结果难以直接比较。T5 通过将所有变量统一到一个框架中，使得不同技术选择的效果可以在受控条件下进行系统评估。

---

**系统性的实证评估：迁移学习的方法论文献**

&emsp;&emsp;T5 论文长达 53 页，其核心价值不仅在于提出了一个新模型，更在于它对迁移学习领域进行了迄今为止最系统的实证评估。论文围绕以下几个维度进行了对照实验：

- **架构变体**：比较了 encoder-only、decoder-only 和 encoder-decoder 三种架构。结论是 encoder-decoder 架构在文本到文本任务上表现最优，因为编码器提供双向上下文理解，解码器负责自回归生成，两者结合兼顾了理解和生成能力；
- **预训练目标**：比较了语言建模、BERT 式掩码语言建模、降噪目标、span corruption 等多种目标。结论是 span corruption 在生成任务上表现最好，且计算效率较高；
- **预训练数据**：比较了 C4、Wikipedia、Common Crawl 等不同数据集，以及是否进行多任务预训练。结论是更大的、更多样化的数据带来更好的下游性能；
- **微调策略**：比较了全量微调、适配器层、逐步解冻等多种方案。结论是全量微调在大多数任务上表现最好，但适配器层在参数效率上有优势；
- **模型规模**：比较了从 2.2 亿到 110 亿参数的多种配置，验证了Scaling Law在 encoder-decoder 架构上的有效性。

&emsp;&emsp;这种系统性的评估使 T5 论文成为迁移学习领域的方法论文献——它不仅报告了“什么最好”，还详细记录了“什么不好”以及“为什么不好”。论文中大量的消融实验和对照分析，为后续研究提供了宝贵的经验参考。

---

**编码器-解码器架构：回归原始 Transformer 设计**

&emsp;&emsp;T5 采用了经典的 Transformer 编码器-解码器架构，这在当时是一个值得注意的选择。2019 年前后，BERT（encoder-only）和 GPT（decoder-only）在各自的领域占据主导地位，encoder-decoder 架构虽然在原始 Transformer 中已经提出，但并未获得同等的关注。T5 的实证评估表明，encoder-decoder 架构在文本到文本的统一框架下表现最优——编码器的双向注意力适合理解输入，解码器的因果注意力适合自回归生成，而编码器-解码器交叉注意力则负责在两者之间传递信息。

&emsp;&emsp;在具体实现上，T5 对原始 Transformer 进行了几处关键简化：

- **简化的层归一化**：去掉了 LayerNorm 中的偏置项，且将层归一化放置在残差路径之外（Pre-LN 的变体），使训练更稳定；
- **无绝对位置嵌入**：T5 完全去掉了绝对位置嵌入，改用相对位置编码（详见下文）；
- **简化的激活函数**：使用 ReLU 而非 GELU，虽然 T5 1.1 版本后来改用了 GeGLU。

&emsp;&emsp;T5 的编码器和解码器都由 12 层 Transformer 块组成（Base 版本），隐藏维度为 768，注意力头数为 12，参数量约为 2.2 亿——约为 BERT-Base 的两倍，因为多了解码器部分。T5 提供了从 Small（约 7700 万参数）到 11B 的多种规模配置。

---

**相对位置编码：简化到极致的位置方案**

&emsp;&emsp;T5 在位置编码上做出了一个独特的选择：**完全放弃绝对位置嵌入，改用极其简化的相对位置编码。** 在原始 Transformer 中，位置编码是正弦/余弦函数或可学习的绝对嵌入，它们为每个位置分配一个唯一的向量表示。T5 的做法不同：它为每一对相对位置偏移学习一个标量偏置，直接加到注意力权重上（pre-softmax），从而在计算注意力时注入位置信息。

&emsp;&emsp;具体来说，T5 将所有可能的相对位置偏移映射到 32 个桶中。偏移量为 0 的桶单独分配；偏移量 1 到 7 各占一个桶；偏移量 8 到 127 按对数增长的方式分组（如 8-11、12-15、……、64-127）；超过 128 的偏移全部归入同一个桶。每个桶对应一个可学习的标量偏置，该偏置在注意力计算时加到对应的注意力 logit 上。所有层共享同一组位置编码参数，但同一层内的不同注意力头使用不同的偏置值。

&emsp;&emsp;这种设计有几个优点。首先，**参数量极小**：只有 32 个可学习的标量值（每头独立），相比可学习绝对位置嵌入的 $L \times d$ 参数量（L 为最大序列长度），几乎可以忽略不计。其次，**理论上支持长度外推**：由于位置信息被编码为相对偏移而非绝对位置，模型在理论上可以处理比训练时更长的序列——虽然论文指出，单层模型对超过 128 的相对位置不敏感，但多层堆叠可以通过组合局部信息来感知更大的偏移。第三，**计算效率高**：相对位置偏置直接加到注意力矩阵上，不需要修改输入嵌入。

&emsp;&emsp;这种简化的相对位置编码后来被多种模型借鉴，其思想也影响了后续的 ALiBi（Attention with Linear Biases）等位置编码方案。

---

**C4 数据集：高质量预训练语料的构建**

&emsp;&emsp;迁移学习的一个重要前提是有一个高质量、大规模、多样化的预训练语料。T5 的作者发现，当时可用的数据集无法同时满足这三个条件：Wikipedia 质量高、格式统一，但规模有限；Common Crawl 规模大、多样性好，但质量参差不齐。为此，他们构建了 **Colossal Clean Crawled Corpus（C4）** 。

&emsp;&emsp;C4 从 Common Crawl 的 2019 年 4 月快照中提取，经过多步清洗：只保留以标点符号结尾的行；移除包含“javascript”“lorem ipsum”等模板文本的页面；使用 langdetect 过滤非英语文本；移除包含脏词、代码或特定有害内容的页面；使用启发式规则移除重复的文本行和重复的 3-gram。最终得到约 750GB 的高质量英文文本，是 Wikipedia 的两个数量级以上。

&emsp;&emsp;C4 的构建方法体现了“数据质量优先”的原则——不是简单地爬取所有网页，而是通过一系列精心设计的过滤规则来保留高质量内容。这种数据清洗范式后来被 RefinedWeb、FineWeb 等数据集构建工作继承和发展。T5 作者还基于 C4 构建了多语言版本 mC4，覆盖 101 种语言，为后续的多语言预训练研究提供了数据基础。

---

**Span Corruption：受 SpanBERT 启发的降噪目标**

&emsp;&emsp;T5 的预训练目标是 **span corruption**（跨度破坏），一种降噪自编码目标，受 SpanBERT 的启发。具体做法是：从输入文本中随机选择连续 token 跨度，将这些跨度替换为特殊的哨兵 token（sentinel token），然后训练模型根据上下文重建被替换的原始文本。

&emsp;&emsp;具体参数为：破坏原始序列的 15%，每个被替换的跨度平均长度为 3 个 token。每个跨度分配一个唯一的哨兵 token（如 `<extra_id_0>`、`<extra_id_1>` 等），目标序列由所有被破坏跨度的原始内容组成，每个跨度前也加上对应的哨兵 token 标记。例如：

```text
输入:  Thank you <extra_id_0> me to your party <extra_id_1> week.
目标:  <extra_id_0> for inviting <extra_id_1> last <extra_id_2>
```

&emsp;&emsp;论文的实验表明，span corruption 目标在生成任务上表现优于独立同分布的掩码语言建模（即 BERT 式的逐 token 掩码），且由于目标序列通常比输入序列短得多，训练的计算效率也更高。

&emsp;&emsp;选择 span corruption 而非标准语言建模的一个关键考量是：T5 的 encoder-decoder 架构需要一个“从输入序列到输出序列”的映射，而标准语言建模的输入和输出是同一个序列的偏移版本，这更适合 decoder-only 架构。span corruption 则天然地为 encoder 提供了需要理解的上下文，为 decoder 提供了需要生成的缺失内容，与 encoder-decoder 架构形成了良好的匹配。

---

**T5 与 BERT、GPT 的关系及后续影响**

&emsp;&emsp;**T5 不是对 BERT 或 GPT 的渐进改进，而是一次范式层面的统一。** BERT 将 NLP 任务分为“理解”和“生成”两类，用不同的架构处理；GPT 则将所有任务统一为生成，但放弃了双向理解能力。T5 通过文本到文本框架证明，只要将所有任务都转化为生成问题，encoder-decoder 架构就能同时兼顾理解与生成，且不需要为每个任务设计特定的输出层。

&emsp;&emsp;T5 对后续工作的影响是多层面的。在架构层面，T5 的 encoder-decoder 设计被 mT5（多语言版本）、ByT5（字节级版本）、UL2（统一预训练目标）等模型继承和发展。UL2 在 T5 的基础上提出了 Mixture-of-Denoisers 目标，混合了 span corruption、极端 span corruption 和顺序 PrefixLM 三种降噪任务，进一步提升了预训练的通用性。在方法论层面，T5 论文的系统性实证评估成为后续研究的标准做法——大规模消融实验、多维度对照分析、详细的实验记录，这些做法在 PaLM、Chinchilla、LLaMA 等论文中被继承。

&emsp;&emsp;在应用层面，T5 的统一框架使“一个模型处理所有任务”成为现实。Hugging Face 的 T5 实现成为最常用的文本生成模型之一，被广泛应用于摘要、翻译、问答等场景。T5 的文本到文本思想也深刻影响了后续的提示工程范式——当用户用自然语言指令让 LLM 执行任务时，本质上就是在使用 T5 所确立的“文本输入 → 文本输出”交互模式。

&emsp;&emsp;从更宏观的视角看，T5 代表了 LLM 演进中的一个关键转折点：**在架构创新的竞赛之外，统一范式和系统实证同样具有根本性的价值。** T5 没有提出革命性的新架构，而是通过将已有的技术方案系统地比较、统一和优化，证明了“简单但统一”的设计可以超越“复杂但分散”的设计。这一思想在后续的 LLM 发展中持续显现——从 GPT-3 的“规模统一一切”到 ChatGPT 的“对话统一一切”，统一范式始终是推动技术进步的重要力量。

### 3.4 Gopher / Chinchilla（2021—2022）

- 论文地址：
  - Gopher：[Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/pdf/2112.11446)
  - Chinchilla：[Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556)

&emsp;&emsp;GPT-3 证明了规模可以带来能力涌现，但它和同期所有大模型都遵循同一条隐含假设：性能提升主要靠放大参数量，训练数据量保持相对固定即可。GPT-3 用 1750 亿参数只训练了 3000 亿 token，Gopher 用 2800 亿参数也只训练了 3000 亿 token。这条路线在 2021 年前后遇到了两个越来越尖锐的问题：训练成本急剧攀升，但性能增益开始出现边际递减；推理和微调的部署成本同样高昂，模型越大，下游应用的门槛越高。DeepMind 的 Gopher 和 Chinchilla 两项工作恰好构成了这一问题的完整叙事：Gopher 是“更大参数”路线的极致，Chinchilla 则通过系统性的缩放实验推翻了这条路线，证明**在固定计算预算下，模型参数量和训练数据量应当等比例放大，而不是只放大参数**。Chinchilla 用 700 亿参数和 1.4 万亿 token，在相同计算预算下全面超越了 2800 亿参数的 Gopher，成为 LLM 缩放规律从“参数优先”转向“数据与参数均衡”的关键转折点。

---

**Gopher：280B 参数的规模化探索**

&emsp;&emsp;Gopher 是 DeepMind 于 2021 年 12 月发布的 2800 亿参数自回归语言模型，架构上延续了 GPT-3 的 decoder-only Transformer 路线，但引入了若干关键改进。Gopher 使用 RMSNorm 替代 LayerNorm，以降低归一化的计算开销并提升训练稳定性；使用相对位置编码（来自 Dai et al. 2019 的方案）替代绝对位置嵌入，使模型在理论上可以外推到训练时未见过的序列长度。此外，Gopher 使用 GeLU 激活函数和 SentencePiece 分词器，上下文长度为 2048 token。

&emsp;&emsp;Gopher 的模型族包含六个规模版本，从 4400 万参数到 2800 亿参数，为缩放规律的研究提供了受控实验条件。最大版本的 Gopher 拥有 80 层 Transformer、128 个注意力头和 16384 维的内部维度。训练在 4096 块 TPU v3 芯片上完成。

&emsp;&emsp;训练数据方面，Gopher 使用了 **MassiveText**，一个约 2.35 万亿 token 的英文为主语料库，来源涵盖网页、书籍、新闻、源代码、Wikipedia 和 C4。与 GPT-3 相比，Gopher 的一个显著变化是**书籍数据的比重大幅提升**：书籍数据占 Gopher 训练数据的 27%，而 GPT-3 中仅占 16%。论文指出，这一调整对提升模型在长文本理解和知识密集型任务上的表现起到了重要作用。

&emsp;&emsp;Gopher 在 152 个不同的 NLP 任务上进行了评估，在大多数任务上取得了当时的最优性能，尤其在阅读理解、事实核查和有毒语言识别等任务上，规模带来的增益最为显著。论文同时揭示了一个重要的负面发现：**在逻辑推理和数学推理任务上，规模带来的增益极其有限**。这一发现暗示，语言建模目标的固有属性可能决定了某些能力无法仅靠扩大模型和数据来获得。

&emsp;&emsp;Gopher 的另一项重要贡献是对模型偏见和毒性的系统分析。论文发现，随着模型规模的增大，模型在涉及性别、种族、宗教等敏感话题时产生偏见内容的比例并未显著下降，某些维度上甚至有所上升。这一分析开启了 LLM 安全研究中“规模与偏见关系”的系统性讨论。

---

**Chinchilla：推翻“参数优先”路线的缩放实验**

&emsp;&emsp;Gopher 发布后不久，DeepMind 的同一团队开始追问一个更根本的问题：**在给定计算预算下，模型参数量和训练数据量应当如何分配？** 此前 Kaplan 等人（2020）的缩放定律认为，当计算预算增加 10 倍时，模型参数量应增加约 5.5 倍，而训练 token 数只需增加约 1.8 倍。这一结论直接指导了 GPT-3、Gopher、Jurassic-1 等模型的设计：参数量被推到极致，而训练数据量相对滞后。

&emsp;&emsp;Chinchilla 论文通过对超过 400 个语言模型（从 7000 万到 160 亿参数，训练数据从 50 亿到 5000 亿 token）的系统实验，重新检验了这一结论。核心发现是：**对于计算最优的训练，模型大小和训练 token 数应当等比例缩放——模型参数量每翻一倍，训练 token 数也应当翻一倍**。这一结论与此前 Kaplan 等人的建议形成了鲜明对比。Chinchilla 的作者指出，现有的大语言模型严重欠训练：它们的参数量远大于其训练数据所能支撑的最优水平。

&emsp;&emsp;为了验证这一假设，Chinchilla 团队使用与 Gopher 完全相同的计算预算，训练了一个 700 亿参数的模型，但使用了 **1.4 万亿 token** 的训练数据——参数量约为 Gopher 的 1/4，训练数据量约为 Gopher 的 4.7 倍。这个模型被命名为 Chinchilla。架构上，Chinchilla 与 Gopher 家族完全一致，但将优化器从 Adam 替换为 AdamW，这一改动被报告为改善了语言建模损失和下游任务性能。

&emsp;&emsp;Chinchilla 的缩放实验采用三种互补的方法来拟合最优的参数量-数据量配比。第一种方法固定模型大小、改变训练步数，拟合每个模型在各自训练 horizon 下的最优损失；第二种方法固定计算预算、改变模型大小，观察损失随参数量的变化曲线；第三种方法将损失建模为参数量 N 和数据量 D 的参数化函数：

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

&emsp;&emsp;其中 $E$ 是数据分布上的不可约损失，$A/N^{\alpha}$ 建模有限参数量带来的近似误差，$B/D^{\beta}$ 建模有限数据量带来的估计误差。论文通过 L-BFGS 优化拟合得到 $\alpha \approx 0.34$、$\beta \approx 0.28$、$E \approx 1.69$、$A \approx 406.4$、$B \approx 410.7$。这一参数化形式后来成为 LLM 缩放研究的标准框架，被广泛用于预测不同规模配置下的预期损失。

&emsp;&emsp;三种方法一致地给出了相同的结论：最优的参数量 N 和数据量 D 都大约与计算预算 C 的 0.5 次幂成正比，即 $N_{\text{opt}} \propto C^{0.49}$、$D_{\text{opt}} \propto C^{0.51}$。这意味着参数量和数据量应当同步增长，而不是像 Kaplan 等人建议的那样偏向参数量。

---

**从 Gopher 到 Chinchilla：LLM 缩放范式的转向**

&emsp;&emsp;Gopher 和 Chinchilla 的关系可以概括为：**Gopher 代表了“参数优先”缩放路线的终点，Chinchilla 则开启了“参数与数据均衡”的新范式。** Gopher 用 280B 参数在 152 个任务上取得了当时的最优性能，但它也暴露了这条路线的问题——训练成本极高，推理部署门槛极大，而逻辑推理等关键能力并未随规模同步提升。Chinchilla 通过严格的对照实验证明，这些代价本可以更小：在相同的计算预算下，一个更小但训练更充分的模型可以做得更好。

&emsp;&emsp;这一发现对后续 LLM 的发展产生了直接而深远的影响。**LLaMA 系列**是 Chinchilla 缩放规律最直接的继承者：LLaMA-1（2023）在 7B 到 65B 参数上使用了 1 万亿到 1.4 万亿 token 的训练数据，训练数据量远超同期同规模模型，验证了“小模型 + 大数据”的可行性。**Llama 2、Llama 3** 进一步将这一策略推向极致，Llama 3 的 8B 模型使用了超过 15 万亿 token 的训练数据，训练数据量是模型参数量的近 2000 倍。**Mistral 7B** 同样遵循了这一路线，以 7B 参数在多个基准上超越了更大的模型。

&emsp;&emsp;在方法论层面，Chinchilla 的缩放实验设计——大规模的受控模型训练、参数化损失函数的拟合、三种方法的交叉验证——成为后续缩放规律研究的标准范式。Meta 的 LLaMA 系列、Google 的 PaLM 2、Anthropic 的 Claude 等模型在设计时都参考了 Chinchilla 的配比建议。Chinchilla 论文中提出的“计算最优训练”概念，使 LLM 的开发从“尽可能大”转向“在给定预算下尽可能优”，这直接影响了后续模型的设计决策——包括是否值得为了性能提升而将参数量翻倍，以及训练数据应该准备多少。

&emsp;&emsp;Chinchilla 的局限同样值得关注。它的缩放实验主要在 7000 万到 160 亿参数的范围内进行，而将结论外推到 700 亿以上的规模时，是否存在新的缩放行为变化，论文并未完全验证。此外，Chinchilla 的最优配比是基于语言建模损失拟合的，而下游任务的性能与损失之间的关系并不总是线性的——某些能力（如推理）可能对数据质量和多样性的敏感度高于对数据量的敏感度。这些开放问题在 LLaMA 3、DeepSeek 等后续工作中被进一步探索。

## 4 开源模型爆发

### 4.1 InstructGPT / ChatGPT

- 论文地址：
  - InstructGPT：[Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155)
  - ChatGPT：[Introducing ChatGPT](https://openai.com/blog/chatgpt)（技术报告未公开发表）

&emsp;&emsp;GPT-3 证明了规模可以带来上下文学习能力，但它和同期所有 LLM 都面临同一个困境：**语言建模目标与用户意图之间存在根本性的错位。** GPT-3 的训练目标是“预测互联网文本的下一个 token”，而用户期望的是“安全、诚实地遵循指令”。这两者并不天然一致。一个 1750 亿参数的 GPT-3 可以流畅地续写任何文本，却可能编造事实、生成有害内容、或在被要求“解释量子力学”时输出一篇虚构的新闻报道。这种“能力与意图的错位”不是规模可以解决的——更大的模型只会更流利地产生不符合用户期望的输出。InstructGPT 的核心思想是：**不再单纯依赖扩大模型规模，而是用人类反馈作为奖励信号，通过强化学习将模型的行为“对齐”到人类意图上。** 它提出了一套三步走的 RLHF（Reinforcement Learning from Human Feedback）训练框架——监督微调、奖励建模、PPO 强化学习——首次在工程上实现了大语言模型与人类偏好的有效对齐，其 1.3B 参数版本的输出被人类评估者优先选择的频率超过了 175B 的 GPT-3。ChatGPT 则是 InstructGPT 方法在对话场景上的产品化实现，于 2022 年 11 月 30 日发布后，在两个月内用户数突破一亿，成为 LLM 从研究原型走向大众应用的分水岭。

---

**核心问题：语言建模目标的“不对齐”**

&emsp;&emsp;InstructGPT 论文开篇即指出：“让语言模型更大，并不会内在地使它们更好地遵循用户意图。”论文将这种错位归因于训练目标本身：语言建模的目标是“预测网页文本的下一个 token”，而用户的核心需求是“有帮助、安全地遵循指令”，两者在本质上是不同的目标。GPT-3 在 SQuAD、DROP 等标准 NLP 基准上表现优异，但在实际使用中经常出现三类问题：编造事实（hallucination）、生成有毒内容（toxicity）、以及偏离用户指令（instruction following failure）。这三类问题分别对应了“真实、无害、有用”三个对齐维度中的不同失败模式。

&emsp;&emsp;此前解决这类问题的常规思路是扩大模型规模或增加训练数据。但 InstructGPT 的立场是：**规模解决的是能力问题，对齐解决的是行为问题。** 一个模型可以非常“有能力”却非常“不对齐”——它知道如何写出一篇有说服力的虚假新闻，也知道如何生成带有偏见的言论，但它并不“愿意”按照用户的真实意图行事。RLHF 的目标不是提升模型的知识或推理能力，而是调整它的行为倾向：在多种可能的输出中，选择更符合人类偏好的那一种。

---

**RLHF 三步框架：SFT → 奖励模型 → PPO**

&emsp;&emsp;InstructGPT 的 RLHF 框架由三个顺序执行的步骤构成，每一步都需要独立的数据集和训练目标。

**第一步：监督微调（Supervised Fine-Tuning, SFT）**

&emsp;&emsp;SFT 的目标是让模型学会“符合人类期望的输出模式”。OpenAI 雇佣了约 40 名标注员，从 OpenAI API 用户提交的提示和标注员自行编写的提示中筛选出约 13,000 条训练样例。每条样例包含一个提示和标注员撰写的理想回答。模型在 GPT-3 的基础上进行监督微调，训练 16 个 epoch，使用余弦学习率衰减和 0.2 的残差 dropout。论文报告了一个值得注意的细节：验证损失在第一个 epoch 后就已经过拟合，但更多的训练轮次仍然持续提升奖励模型得分和人类偏好评分——这说明语言建模损失与人类偏好之间存在不一致，SFT 的优化目标不能简单地用验证损失来衡量。

**第二步：奖励模型（Reward Model, RM）**

&emsp;&emsp;奖励模型的任务是学习“什么样的输出更符合人类偏好”，并将其量化为一个标量奖励值。数据收集方式是：对于每个提示，让 SFT 模型生成 4 到 9 个不同的输出，标注员对这些输出进行排序。排序数据比绝对评分更可靠，因为人类标注员在比较两个输出时的一致性远高于独立评分。

&emsp;&emsp;奖励模型的架构是在 SFT 模型的基础上，去掉最后的词嵌入层，加入一个输出标量奖励的评分头。OpenAI 选择使用 6B 参数的模型作为奖励模型，而非 175B 版本，原因是“175B 的模型不稳定，不适合用于奖励函数”。损失函数采用 pairwise 排序损失，目标是最大化偏好输出与非偏好输出之间的奖励差值：

$$\mathcal{L}_{\text{RM}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right]$$

&emsp;&emsp;其中 $y_w$ 是被人类标注员优先选择的输出，$y_l$ 是次优输出。一个关键的训练细节是：对于同一个提示的所有输出对，OpenAI 将它们放在同一个 batch 中处理，所有输出只需进行一次前向传播，然后对奖励分数进行组合并计算 pairwise loss。这样做避免了同一提示的多个输出对分散在不同 batch 中导致的过拟合问题。

**第三步：PPO 强化学习**

&emsp;&emsp;PPO 阶段的目标是让 SFT 模型学会生成能够从奖励模型获得高分的输出。训练数据集包含约 31,000 条仅来自 API 的提示，不需要人工标注。训练流程是：对于每个提示，策略模型（即正在训练的 SFT 模型）生成一个输出，奖励模型对该输出打分，然后使用 PPO 算法根据奖励信号更新策略模型的参数。

&emsp;&emsp;为了防止模型为了“刷分”而输出退化文本（例如重复同一句高奖励的话），OpenAI 在目标函数中加入了逐 token 的 KL 散度惩罚，约束策略模型的输出分布不要偏离 SFT 模型太远：

$$\text{objective}(\phi) = \mathbb{E}_{(x,y) \sim \pi_\phi^{RL}} \left[ r_\theta(x, y) - \beta \log \frac{\pi_\phi^{RL}(y|x)}{\pi^{SFT}(y|x)} \right]$$

&emsp;&emsp;其中 $\pi_\phi^{RL}$ 是正在训练的策略模型，$\pi^{SFT}$ 是第一步得到的 SFT 模型，$\beta$ 控制 KL 惩罚的强度。这个惩罚项的作用类似于 ControlNet 中的零卷积和 LoRA 中的零初始化：**都是通过约束新增模块在训练初期的影响，来保证模型不会因为优化目标的改变而丢失原有的能力。**

&emsp;&emsp;此外，OpenAI 还提出了一个变体 **PPO-ptx**：在 PPO 的优化目标中额外加入预训练数据的梯度更新项，使模型在学会对齐的同时不遗忘预训练阶段学到的语言能力。这个设计的直接目的是缓解“对齐税”——RLHF 训练会导致模型在 SQuAD、DROP、HellaSwag 等公共 NLP 基准上的性能出现退化，因为对齐目标与这些基准所衡量的能力并不完全一致。PPO-ptx 通过在优化目标中混合预训练梯度，在一定程度上抵消了这种退化。

---

**关键结果：1.3B 超越 175B**

&emsp;&emsp;InstructGPT 最引人注目的结果是：**1.3B 参数的 InstructGPT 的输出，被人类评估者优先选择的频率超过了 175B 参数的 GPT-3**。这意味着，将一个已有的能力对齐到人类意图上，所带来的“体感提升”大于将能力继续放大 100 倍。论文中的人类评估显示，InstructGPT（PPO-ptx）在 API 提示分布上的胜率显著高于 GPT-3 及其提示版本。

&emsp;&emsp;在真实性方面，InstructGPT 在 TruthfulQA 基准上的表现显著优于 GPT-3，生成事实性错误的频率明显下降。在毒性方面，RealToxicityPrompts 上的评估显示 InstructGPT 的有毒输出比例有所降低，但论文也坦承改进幅度有限，且模型对偏见内容的改善并不明显。在指令遵循的泛化能力上，InstructGPT 能够处理训练数据中未出现过的指令类型，这表明 RLHF 学到的不是某种特定的回答模板，而是一种更通用的“理解并执行指令”的行为模式。

&emsp;&emsp;论文还报告了一个有趣的对比：InstructGPT 在公共 NLP 数据集（如 SQuAD、DROP）上的性能略低于 SFT 基线。这并不矛盾——这些基准衡量的是特定任务上的精确匹配能力，而 InstructGPT 被优化的是“在开放式对话中提供有用回答”的能力。两者的优化目标不同，性能表现自然会有差异。这一发现也提示了一个更深层的问题：**“对齐”的评估标准本身是主观的、场景依赖的，不存在一个单一的指标可以同时衡量所有维度的对齐质量。**

---

**ChatGPT：从 InstructGPT 到对话产品**

&emsp;&emsp;ChatGPT 与 InstructGPT 的核心区别在于**产品化程度**。InstructGPT 是一个研究模型，通过 API 提供服务；ChatGPT 是一个面向大众的对话界面，用户不需要理解提示工程或 few-shot 示例，只需要像与人对话一样输入问题即可。从 LLM 架构演进的角度看，ChatGPT 的意义在于：**它证明了 RLHF 不仅是一种研究技术，更是一种可以产品化、规模化的工程能力。** InstructGPT 论文中的三步框架，在 ChatGPT 中被完整地保留并大规模部署，成为后续所有对话式 LLM（Claude、Gemini、文心一言等）的标准训练范式。

---

**局限与争议**

&emsp;&emsp;InstructGPT 和 ChatGPT 的对齐方法并非没有代价。**对齐税**是其中最直接的问题：RLHF 训练会导致模型在标准 NLP 基准上的性能退化，PPO-ptx 虽然缓解了这一问题，但并未完全消除。**奖励黑客**是另一个根本性挑战：奖励模型是对人类偏好的近似，而非人类偏好的精确表达。当策略模型学会利用奖励模型的缺陷来获取高分时，就会产生“奖励黑客”行为——例如输出看似有帮助但实际上空洞或谄媚的回答。RLHF 的稳定性也依赖于奖励模型的质量和 KL 惩罚的强度：$\beta$ 过大会导致模型过于保守，$\beta$ 过小则可能导致输出退化。

&emsp;&emsp;ChatGPT 的局限性在发布后迅速暴露。它会产生“幻觉”——以自信的语气编造事实；它在某些话题上表现出过度保守的拒绝行为；它的知识截止于训练数据的时间点，无法获取最新信息。此外，ChatGPT 的训练数据来自互联网，不可避免地继承了其中的偏见。OpenAI 在 ChatGPT 发布时明确标注了这些局限，并采取了逐步释放的策略。从更宏观的视角看，InstructGPT 和 ChatGPT 开启了一个新的问题空间：**对齐不是一次性完成的任务，而是一个持续的过程。** 模型的行为会随着使用场景的变化而变化，人类偏好本身也在演化，如何在一个动态环境中维持模型与人类意图的一致性，是 RLHF 之后对齐研究的核心议题。

---

**后续影响：RLHF 成为后训练的标准范式**

&emsp;&emsp;InstructGPT 论文发表于 2022 年 3 月，ChatGPT 发布于同年 11 月，两者共同开启了大语言模型的“后训练”（post-training）时代。在此之前，LLM 的训练流程是“预训练 + 微调”；在此之后，标准流程变成了“预训练 + SFT + 奖励建模 + 强化学习”。这一范式被几乎所有主流 LLM 继承：Anthropic 的 Claude 系列在 RLHF 基础上提出了 Constitutional AI，用一组明确的规则替代部分人类标注；Google 的 Gemini 在 RLHF 之外引入了基于 AI 反馈的强化学习（RLAIF）；开源社区的 LLaMA 系列、Alpaca、Vicuna 等模型则通过指令微调和 RLHF 的简化版本实现了对话能力。

&emsp;&emsp;在方法论层面，InstructGPT 论文中的许多设计决策——奖励模型选择 6B 而非 175B、pairwise 排序损失、KL 惩罚、PPO-ptx——成为后续对齐研究的参考基准。DPO（Direct Preference Optimization）等方法通过将奖励模型和策略模型合并为一个目标函数，简化了 RLHF 的流程；但 RLHF 所确立的核心思想——**用人类偏好作为训练信号，将模型的行为从“能生成”引导到“会生成”** ——始终没有被替代。从 GPT-3 到 InstructGPT 的转变，标志着 LLM 的发展重心从“规模驱动”转向了“对齐驱动”；而从 InstructGPT 到 ChatGPT 的转变，则标志着这种对齐能力从研究实验室走向了十亿级用户的产品。

### 4.2 LLaMA 系列

- 论文地址：
  - LLaMA 1：[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971)
  - LLaMA 2：[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288)
  - LLaMA 3：[The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783)

&emsp;&emsp;InstructGPT 和 ChatGPT 证明了 RLHF 可以将语言模型的行为对齐到人类意图，但它们的训练细节和模型权重完全闭源。GPT-3.5 和 GPT-4 的架构、数据、训练配方均不公开，学术界和开源社区只能通过 API 使用这些模型，无法进行深入研究或本地部署。这种“能力垄断”的局面与 2020 年前后 NLP 领域对开放研究的期待形成了尖锐矛盾。与此同时，Chinchilla 的缩放规律已经证明，在给定计算预算下，一个更小但训练更充分的模型可以超越更大的模型——这意味着训练顶级 LLM 的门槛并非不可逾越，关键在于是否愿意投入足够的数据和训练时间。LLaMA 系列的核心贡献正在于此：**Meta 用 Chinchilla 最优缩放规律训练了一系列参数量从 7B 到 65B 的基础语言模型，全部使用公开可获取的数据集，并以开放权重的方式发布，首次让学术界和开源社区拥有了可以本地部署、自由微调、深入研究的高质量 LLM 基座。** 从 LLaMA 1 到 LLaMA 3，这一系列模型不仅确立了现代 LLM 的“标准架构配方”——RoPE、RMSNorm、SwiGLU、GQA——更以开放权重策略催生了 Alpaca、Vicuna、Mistral、Qwen、DeepSeek 等数以万计的衍生模型，从根本上改变了 LLM 领域的权力格局。

---

**架构创新：从 GPT 风格到 LLaMA 风格的标准配方**

&emsp;&emsp;LLaMA 1 在整体上沿用了 GPT-2 的 decoder-only Transformer 架构，但对其内部组件进行了多项经过实践验证的替换。这些替换后来被几乎所有主流开源 LLM 采纳，形成了现代 LLM 的“标准配方”。

&emsp;&emsp;**RMSNorm（Root Mean Square Layer Normalization）** ：GPT-2 使用 LayerNorm，对激活值进行均值中心化和方差归一化。LLaMA 将 LayerNorm 替换为 RMSNorm，只对激活值的均方根进行缩放，省略了均值中心化操作。这一简化的代价极低，但在训练稳定性和计算效率上都有所提升。LLaMA 沿用了 Pre-Norm 结构，即在每个子层之前进行归一化，残差路径上不经过任何归一化变换，保障了深层网络的梯度流动。

&emsp;&emsp;**RoPE（Rotary Position Embedding）** ：GPT-2 使用可学习的绝对位置嵌入，为上下文窗口中的每个位置学习一个独立的向量。LLaMA 改用旋转位置编码 RoPE，在注意力计算时对 Query 和 Key 向量施加旋转矩阵，使注意力分数自然地携带 token 之间的相对距离信息。RoPE 不引入额外的位置嵌入表，参数效率更高，且理论上支持长度外推——虽然实际外推效果取决于训练时的上下文长度分布。

&emsp;&emsp;**SwiGLU 激活函数**：GPT-2 的前馈网络使用 GELU 激活函数。LLaMA 将 FFN 中的激活函数替换为 SwiGLU（Swish-Gated Linear Unit），这是一种门控线性单元，通过两个并行的线性投影（一个内容投影和一个门控投影）进行逐元素相乘，再投影回模型维度。SwiGLU 使用三个投影矩阵而非两个，在同等参数量下提供了更强的表达能力。为了避免门控路径导致参数量膨胀，LLaMA 将 FFN 的中间维度调整为原始的 2/3，使总参数量与标准 FFN 大致相当。

&emsp;&emsp;**无偏置线性层**：LLaMA 移除了大多数线性投影中的偏置向量。这一改动节省的参数量相对权重矩阵而言很小，但在整个网络堆栈中一致应用，累积效果可观。

&emsp;&emsp;**分组查询注意力（GQA）** ：LLaMA 1 使用标准的多头注意力（MHA），每个 Query 头对应独立的 Key 头和 Value 头。LLaMA 2 在 70B 版本中首次引入分组查询注意力 GQA：多个 Query 头共享一组 Key 和 Value 头，从而在推理时显著减小 KV 缓存的大小。LLaMA 3 则将 GQA 应用于所有主要模型规模，使其成为 LLaMA 系列的标配。

---

**LLaMA 1：Chinchilla 缩放规律的第一个大规模验证**

&emsp;&emsp;LLaMA 1 于 2023 年 2 月发布，包含 7B、13B、33B 和 65B 四个参数版本。训练数据完全来自公开可获取的语料：CommonCrawl、C4、Wikipedia、Books、Github、arXiv 和 StackExchange，总计约 1.4 万亿 token。LLaMA 1 的核心设计原则直接来自 Chinchilla：在给定推理成本预算下，最优的模型不是训练最快的模型，而是推理最快的模型。用更多的 token 训练更小的模型，可以在推理阶段获得更低的部署成本。

&emsp;&emsp;这一策略的结果是：LLaMA-13B 在大多数基准测试上超越了 175B 参数的 GPT-3，而 LLaMA-65B 的性能可与 Chinchilla-70B 和 PaLM-540B 竞争。65B 版本在 2048 张 A100 80G GPU 上训练了近 21 天。LLaMA 1 的开放权重发布在学术界引发了巨大反响，但受限于研究许可协议，不可免费商用。这一许可限制并未阻止社区的热情——Alpaca、Vicuna 等衍生模型迅速涌现，证明了开放权重基座的巨大生态潜力。

---

**LLaMA 2：商用开放与 RLHF 对齐**

&emsp;&emsp;LLaMA 2 于 2023 年 7 月发布，包含 7B、13B 和 70B 三个参数版本（34B 版本未开源），训练数据从 LLaMA 1 的 1.4T token 扩充到 2T token，上下文长度从 2048 翻倍到 4096。架构上的关键变化是在 70B 版本中引入 GQA，以降低大模型推理时的 KV 缓存开销。

&emsp;&emsp;LLaMA 2 与 LLaMA 1 最根本的区别在于对齐策略。Meta 在基础模型之上，通过 SFT 和 RLHF 训练了 LLaMA 2 Chat 系列，训练流程与 InstructGPT 的三步框架一致：监督微调、奖励建模、PPO 强化学习。LLaMA 2 的 RLHF 数据更加注重安全性，在有用性和无害性之间进行了更精细的权衡。Meta 同时发布了基础模型和 Chat 模型，前者用于研究和微调，后者直接面向对话应用。

&emsp;&emsp;LLaMA 2 最重要的变化是**许可协议的开放**：Meta 允许免费商用，仅对月活跃用户超过 7 亿的实体设置了额外限制。这一变化使 LLaMA 2 从研究工具变成了可商业化部署的基础设施，直接推动了企业级 LLM 应用的爆发。

---

**LLaMA 3 / 3.1：开放权重追平闭源旗舰**

&emsp;&emsp;LLaMA 3 于 2024 年 4 月发布，训练数据量发生了数量级的跃迁：从 LLaMA 2 的 2T token 扩展到 **15T token**，其中代码数据扩充了约 4 倍。这一数据规模的提升直接带来了代码能力和逻辑推理能力的显著进步。词汇表从 LLaMA 2 的 32K 扩展到 **128K**，显著提升了多语言场景下的分词效率。上下文窗口从 4K 扩展到 8K，随后在 LLaMA 3.1 中进一步扩展到 **128K**。

&emsp;&emsp;LLaMA 3 发布了 8B 和 70B 两个版本，全尺寸均采用 GQA。LLaMA 3.1（2024 年 7 月）将参数规模推至 **405B**，是当时最大的公开权重模型，在多个基准上追平甚至超越了 GPT-4（2023 年 3 月版），首次证明开放权重模型可以达到顶级闭源模型的水平。LLaMA 3.1 还发布了宽松的社区许可协议，允许使用其输出来训练其他模型，这直接催生了大量衍生模型的繁荣。

&emsp;&emsp;LLaMA 3.2（2024 年 9 月）将布局扩展到端侧，发布了 1B 和 3B 的小尺寸模型，以及 11B 和 90B 的多模态版本。LLaMA 3.3（2024 年 12 月）则用 70B 的参数量追平了 405B 版本的对齐水平。这一系列迭代覆盖了从数据中心旗舰到手机端侧部署的完整频谱，展示了开放权重模型在规模谱系上的全面布局能力。

---

**LLaMA 4：MoE 架构与原生多模态**

&emsp;&emsp;2025 年 4 月，Meta 发布了 LLaMA 4 系列，标志着架构范式的又一次转向。LLaMA 4 首次采用**混合专家（MoE）架构**，并原生支持文本、图像、音频和视频的多模态推理与生成。

&emsp;&emsp;首批发布的两款模型——**Scout** 和 **Maverick**——均采用 MoE 设计。Scout 拥有 109B 总参数，每个 token 激活 17B 参数，由 16 位专家组成，支持 **10M token** 的上下文窗口，可量化至 int4 并在单张 NVIDIA H100 GPU 上运行。Maverick 拥有 400B 总参数，每 token 激活 17B，由 128 位专家组成，支持 1M 上下文长度。两者均构建于 Meta 所称的“Transformer 2.0”架构之上，在注意力机制、专家路由和多模态融合方面进行了深度优化。Meta 还预览了拥有 2 万亿参数的 Behemoth 版本，定位为对标 DeepSeek V3 的旗舰模型。

&emsp;&emsp;LLaMA 4 的 MoE 架构使模型能够在推理时只激活总参数的一小部分，在保持大规模模型容量的同时显著降低推理成本。这一设计方向与 DeepSeek-V3、Mixtral 等模型的 MoE 路线一致，反映了 LLM 架构从“稠密放大”向“稀疏激活”的范式转变。

---

**局限与争议**

&emsp;&emsp;LLaMA 系列并非没有局限。**跨语言安全对齐**是其中最突出的问题之一：LLaMA 的安全对齐主要在高资源语言（英语）上进行，在低资源语言中攻击成功率显著上升，部分场景下接近零防护。这意味着用户可以通过切换到低资源语言来绕过模型的安全限制，对多语言部署场景构成了实质性的安全风险。

&emsp;&emsp;在技术层面，LLaMA 的架构虽然成为了开源模型的标准配方，但这一“配方”本身并未提供根本性的架构创新。RoPE、RMSNorm、SwiGLU 均由其他工作提出，LLaMA 的价值在于系统性地组合和验证了这些组件，而非发明了新的注意力机制或训练范式。LLaMA 4 转向 MoE 架构，则表明纯粹的稠密 decoder-only 架构在超大规模下的效率瓶颈已经显现。

---

**LLaMA 系列与 GPT 系列的关系及后续影响**

&emsp;&emsp;**LLaMA 是 GPT 架构范式在开放权重路线上的最成功实现。** 它没有改变 decoder-only Transformer 的基本框架，而是通过 Chinchilla 缩放规律的系统应用、架构组件的标准化替换和开放权重策略，将 GPT-3 所证明的能力以可及的方式交付给了整个研究社区。从 LLaMA 1 到 LLaMA 3，模型参数量从 65B 扩展到 405B，训练数据从 1.4T token 扩展到 15T token，上下文从 2K 扩展到 128K——这些数字的变化背后，是“开放权重 + 大规模数据”路线的持续验证。

&emsp;&emsp;LLaMA 的后续影响是多层面的。在模型层面，Mistral、Qwen、DeepSeek、Yi、Falcon 等开源模型直接继承了 LLaMA 的架构配方和缩放策略。在生态层面，Hugging Face 的 Transformers 库、PEFT 微调框架、vLLM 推理引擎等基础设施围绕 LLaMA 形成了完整的工具链。在方法论层面，LLaMA 验证的“小模型 + 大数据”策略在 Chinchilla 的基础上进一步推进——LLaMA 3 的 15T token 训练数据量已经远超 Chinchilla 的最优配比建议，表明在推理成本成为主要瓶颈的场景下，数据规模的进一步扩张仍然有效。

&emsp;&emsp;从更宏观的视角看，LLaMA 系列代表了 LLM 演进中“开放”与“闭源”两条路线的竞争。LLaMA 1 和 LLaMA 2 证明了开放权重可以快速建立生态，LLaMA 3.1 证明了开放权重模型可以达到闭源旗舰的性能水平，但 LLaMA 4 的 MoE 转向和 Meta 最终终止开源策略的决定，也揭示了开放权重路线在商业可持续性上的根本挑战。LLaMA 所确立的架构配方将长期作为开源 LLM 的基础，但开放权重本身是否能在下一代模型中延续，仍然是一个开放的问题。

### 4.3 Mistral 7B & Mixtral 8x7B

- 论文地址：
  - Mistral 7B：[Mistral 7B](https://arxiv.org/pdf/2310.06825)
  - Mixtral 8x7B：[Mixtral of Experts](https://arxiv.org/pdf/2401.04088)

&emsp;&emsp;LLaMA 系列证明了开放权重模型可以成为 LLM 生态的基座，但它的“小模型 + 大数据”策略仍然是一条**规模驱动**的路线——要获得更强的能力，就需要更大的参数量、更多的训练数据、更长的训练时间。对于大多数研究者和中小企业而言，训练一个 70B 甚至 405B 的模型仍然遥不可及。一个更根本的问题是：**性能提升是否只有“放大”这一条路？** 能否在更小的参数规模下，通过架构层面的效率创新，实现与更大模型相当甚至更优的性能？Mistral AI 的两款模型——Mistral 7B 和 Mixtral 8x7B——分别从两个方向回答了这个问题：Mistral 7B 通过滑动窗口注意力（SWA）和分组查询注意力（GQA）在 7B 参数规模下实现了推理效率的大幅优化，性能超越 Llama 2 13B，并在数学、代码和推理任务上超过 Llama 1 34B；Mixtral 8x7B 则采用稀疏混合专家（SMoE）架构，总参数量达到 46.7B，但每个 token 仅激活 13B 参数，推理成本与 13B 稠密模型相当，性能却在大多数基准上超越了 Llama 2 70B 和 GPT-3.5。这两款模型共同确立了“**效率优先**”的架构设计哲学，其影响在后续的 DeepSeek-V3、Qwen 2.5-MoE、Llama 4 等模型中持续显现。

---

**Mistral 7B：滑动窗口注意力与分组查询注意力的效率组合**

&emsp;&emsp;Mistral 7B 于 2023 年 9 月发布，参数量为 73 亿，在 Apache 2.0 许可下完全开放，允许免费商用。它的架构建立在 LLaMA 系列确立的“标准配方”之上——decoder-only Transformer、Pre-Norm、RMSNorm、RoPE、SwiGLU——但引入了两项关键的效率优化：**滑动窗口注意力（Sliding Window Attention, SWA）** 和**分组查询注意力（Grouped Query Attention, GQA）** 。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/sliding_window_attention_2026-09-13_22-40-11.png)

&emsp;&emsp;SWA 的核心思想是限制每个 token 的注意力范围。在标准全注意力中，序列中每个 token 都需要关注之前所有 token，注意力计算量随序列长度呈二次方增长。SWA 将每个 token 的注意力范围限制在一个固定大小的窗口内：query 位置 $i$ 只关注 $[i-W, i]$ 范围内的 token，其中 $W$ 是窗口大小。Mistral 7B 的窗口大小为 4096，模型共 32 层。

&emsp;&emsp;单层 SWA 的注意力范围虽然只有 4096 个 token，但通过多层堆叠，信息可以以每层 $W$ 的速度向前传播：第 $k$ 层的位置 $i$ 通过第 $k-1$ 层看到 $[i-W, i]$，再通过第 $k-2$ 层看到 $[i-2W, i]$，如此层层递推。Mistral 7B 共 32 层、窗口大小 4096，理论感受野可以达到约 **131K token**。这意味着模型在训练时只需要处理 8K 的序列，但在推理时理论上可以处理长达 128K 的上下文。

&emsp;&emsp;SWA 带来的最直接收益是 KV 缓存大小的上界约束。在标准全注意力中，KV 缓存的大小随序列长度线性增长，长序列推理的显存占用会迅速膨胀。而 SWA 的 KV 缓存总元素为 $2 \cdot h_{kv} \cdot d_h \cdot \text{层数} \cdot \min(N, W)$，其中 $N$ 是序列长度，$W$ 是窗口大小。当序列长度超过窗口大小时，KV 缓存不再继续增长，而是被固定在窗口大小对应的上限。配合滚动缓冲区缓存（rolling buffer cache），Mistral 7B 可以在处理任意长度序列时保持恒定的显存占用，这在当时是一个显著的工程优势。

&emsp;&emsp;GQA 是另一项关键的推理效率优化。在标准多头注意力（MHA）中，每个 Query 头都有独立的 Key 头和 Value 头，KV 缓存的大小与注意力头数成正比。GQA 将多个 Query 头分组，每组共享一组 Key 和 Value 头，从而在保持模型质量的同时显著减小 KV 缓存的大小。Mistral 7B 使用 GQA 替代 MHA，加速了推理速度并降低了缓存占用。

&emsp;&emsp;这两项优化共同作用的结果是：Mistral 7B 的推理效率远超同规模模型，只需约 **6GB 显存**即可运行，可以在 MacBook 等消费级设备上本地部署。在性能上，Mistral 7B 在所有评估基准上均优于 Llama 2 13B，并在推理、数学和代码生成任务上超过 Llama 1 34B。在推理任务上，Mistral 7B 的表现甚至直逼参数量近 10 倍的 Llama 2 70B。

&emsp;&emsp;2024 年 3 月，Mistral AI 发布了 Mistral 7B v0.2 基础模型，将上下文窗口从 8K 提升至 32K，并取消了滑动窗口注意力机制。这一变化反映了 SWA 在实际部署中的一个权衡：虽然 SWA 在理论感受野和 KV 缓存效率上有优势，但在需要精确长距离检索的任务中，全注意力的直接建模能力仍然更强。Mistral 7B v0.2 选择以更大的上下文窗口和全注意力替代 SWA，说明“效率”的定义是场景依赖的——在某些场景下，减少计算量是效率；在另一些场景下，一次性处理更长的上下文才是效率。

---

**Mixtral 8x7B：稀疏混合专家的工程落地**

&emsp;&emsp;如果说 Mistral 7B 是在**稠密模型**内部通过注意力优化来提升效率，那么 Mixtral 8x7B 则走了一条更激进的路线：**用稀疏激活替代稠密计算**。Mixtral 8x7B 于 2023 年 12 月发布，是一个基于稀疏混合专家（Sparse Mixture of Experts, SMoE）的 decoder-only 语言模型。

&emsp;&emsp;MoE 的基本思想并不新颖——早在 1991 年就有相关研究，GShard、Switch Transformer 等工作也探索过 MoE 在语言模型中的应用。但 Mixtral 8x7B 的关键贡献在于：它是第一个**在工程上成功落地、性能可与顶级稠密模型竞争、且完全开源的 MoE 语言模型**。在它之前，MoE 的训练和推理都面临诸多工程挑战——专家负载不均衡、通信开销大、推理时专家选择的不确定性——这些挑战使得 MoE 长期停留在学术论文中，未能在生产级模型中大规模部署。

&emsp;&emsp;Mixtral 8x7B 的架构设计围绕一个核心问题展开：**如何在增加模型容量的同时控制推理成本？** 它的答案是：将 Transformer 块中的前馈网络（FFN）替换为 MoE 层，每个 MoE 层包含 8 个独立的“专家”，每个专家本身是一个标准的 SwiGLU FFN。对于每一个输入 token，一个路由网络（Router）从 8 个专家中选择 **2 个**来激活，并将这 2 个专家的输出加权求和作为最终输出。

&emsp;&emsp;这一设计的关键在于“稀疏激活”。虽然模型的总参数量达到 **46.7B**（8 个专家 × 每专家约 5.6B 参数，加上注意力层和嵌入层等非专家参数），但每个 token 只激活其中 **13B** 参数（2 个专家 + 共享的注意力层和嵌入层）。这意味着推理时的计算量只与 13B 稠密模型相当，而不是 46.7B。用 Mixtral 论文的话说，Mixtral 8x7B “像 13B 模型一样计算，像 47B 模型一样存储”。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/moe_Snipaste_2026-09-13_22-47-23.png)

&emsp;&emsp;路由机制是 MoE 设计的核心。Router 是一个可学习的线性层，输入是 token 的隐藏状态，输出是 8 个专家的概率分布。训练时，Router 通过 top-2 选择机制决定每个 token 的专家分配，并使用负载均衡损失（load balancing loss）来防止所有 token 都被路由到同一个专家。一个设计上的关键选择是：**Router 的选择是 token 级别的，而非序列级别的**——同一个序列中的不同 token 可以被路由到不同的专家。这使得模型能够根据每个 token 的语义内容动态地“召唤”最相关的专家来处理，而非对整个序列使用固定的计算路径。

&emsp;&emsp;在训练数据方面，Mixtral 8x7B 使用了 **32K token** 的上下文窗口进行预训练，训练数据规模与 Llama 2 系列相当。模型支持完全密集的 32K 上下文长度，不依赖滑动窗口注意力。论文报告了 Mixtral 在长上下文检索任务上的表现：无论目标信息位于 32K 序列的哪个位置，模型都能成功检索到，且性能不随位置衰减。

&emsp;&emsp;Mixtral 8x7B 的性能表现超出了大多数人的预期。在数学、代码生成和多语言理解任务上，它显著优于 Llama 2 70B。在多数标准基准上，它与 GPT-3.5 持平甚至略胜一筹。更关键的是，Mixtral 8x7B-Instruct——经过指令微调的版本——在人类评估基准上超越了 GPT-3.5 Turbo、Claude-2.1、Gemini Pro 和 Llama 2 70B-Chat。这是一个 13B 激活参数的模型在对话能力上超越 70B 稠密模型和 GPT-3.5 的案例，标志着 MoE 架构从“学术上有趣”变成了“工程上有效”。

&emsp;&emsp;推理效率方面，Mixtral 8x7B 的推理速度比 Llama 2 70B 快 **6 倍**，而模型质量相当或更好。这一效率优势来自稀疏激活的本质：虽然模型需要存储 46.7B 参数，但每个 token 的前向计算只涉及 13B 参数，计算量大幅降低。对于显存受限的场景，可以通过量化技术（如 int4）进一步压缩模型大小，使 Mixtral 8x7B 的部署门槛接近于 13B 稠密模型。

---

**MoE 的工程挑战与解决方案**

&emsp;&emsp;MoE 架构在理论上优雅，但在工程实现上面临几个核心挑战，Mixtral 8x7B 的设计决策在很大程度上是对这些挑战的回应。

&emsp;&emsp;**专家负载不均衡**是最直接的问题。如果 Router 倾向于将大多数 token 路由到少数几个专家，那么这些专家会过载，而其他专家则被闲置，导致训练效率下降和模型容量浪费。Mixtral 8x7B 使用了标准的负载均衡损失，通过惩罚专家负载分布的方差来鼓励 Router 均匀分配 token。此外，Router 的 top-2 选择机制本身提供了一定的均衡效果——与 top-1 相比，top-2 允许更多专家参与计算，降低了单个专家过载的风险。

&emsp;&emsp;**通信开销**是分布式训练中的另一个瓶颈。在标准的 MoE 训练中，不同的专家通常被放置在不同的设备上，token 的路由需要跨设备通信，这可能导致显著的延迟。Mixtral 8x7B 的工程实现利用了 Megablocks 等高性能 MoE 内核，将 MoE 层的前馈网络操作转换为大型稀疏矩阵乘法，在单 GPU 上高效运行，并自然处理不同专家获得的可变数量 token 的情况。这种“在单 GPU 上高效运行 MoE”的能力，是 Mixtral 8x7B 能够以合理成本训练和推理的关键。

&emsp;&emsp;**推理时的不确定性**是第三个挑战。在稠密模型中，推理路径是确定的：每个 token 经过相同的网络层。在 MoE 模型中，推理路径取决于 Router 的选择，而 Router 的行为可能随输入分布的变化而变化。Mixtral 8x7B 通过在大规模多样化数据上训练 Router，使其选择行为相对稳定。论文的消融实验表明，Router 的选择在一定程度上是可解释的——例如，处理代码的 token 更倾向于被路由到特定专家，但 Router 并未完全“专精化”，大多数专家仍然保持了一定的通用性。

---

**Mistral 7B 与 Mixtral 8x7B 的关系：效率的两种路径**

&emsp;&emsp;Mistral 7B 和 Mixtral 8x7B 可以视为同一设计哲学的两个分支：**在给定计算预算下最大化模型能力**。Mistral 7B 的“预算”是单次推理的显存和计算量，它通过 SWA 和 GQA 将这一预算的利用效率最大化——在 7B 稠密参数上实现 13B 甚至 34B 模型的性能。Mixtral 8x7B 的“预算”是模型的总存储容量，它通过 MoE 将这一容量转化为能力——用 46.7B 的存储空间承载 70B 级别模型的知识，但只消耗 13B 模型的推理计算量。

&emsp;&emsp;两者的共同点是：**不追求“更大”，而是追求“更聪明地使用计算”** 。在 LLaMA 系列和 Chinchilla 缩放规律主导的“更大即更好”叙事中，Mistral AI 提出了一个不同的视角：架构层面的效率创新可以在不增加计算成本的前提下带来性能提升。Mistral 7B 在 7B 规模上超越 13B 模型，Mixtral 8x7B 在 13B 激活参数下追平 70B 模型——这两个结果分别从两个方向验证了“效率优先”策略的有效性。

&emsp;&emsp;从架构演进的角度看，Mistral 7B 和 Mixtral 8x7B 代表了 LLM 设计中的一个重要转折点：**从“规模驱动”转向“效率驱动”** 。在 GPT-3 到 GPT-4 的路线中，性能提升主要来自参数量的增加；在 Mistral 的路线中，性能提升来自架构设计的优化。这一转变的背景是推理成本逐渐成为 LLM 部署的主要瓶颈——模型训练是一次性成本，而推理是持续成本。一个在训练时多花 10 倍计算但推理时少花 10 倍计算的模型，在实际部署中可能具有更高的经济价值。

---

**局限**

&emsp;&emsp;Mistral 7B 的 SWA 设计在理论上提供了 128K 的感受野，但这一理论优势在实际使用中受到限制。SWA 的信息传播依赖多层堆叠，每一层只能看到固定窗口内的 token，这意味着远距离信息的传递需要经过多层间接推理，而非一次直接注意力。在需要精确长距离依赖的任务（如长文档问答、多跳推理）中，SWA 的表现可能不如全注意力。Mistral 7B v0.2 取消 SWA 并切换到全注意力，正是对这一局限的回应。

&emsp;&emsp;Mixtral 8x7B 的 MoE 架构也面临几个问题。**专家利用率**是其中之一：如果 Router 在推理时倾向于选择少数专家，那么其他专家的参数实际上被浪费了，模型的“有效容量”可能远低于 46.7B。论文虽然没有提供详细的专家利用率分析，但社区的一些复现实验表明，Mixtral 的 Router 在不同任务上的选择模式存在显著差异，某些专家在特定领域（如代码或数学）中被更频繁地激活。**推理的确定性**是另一个问题：由于 Router 的选择依赖于输入，同一模型在不同输入上的计算路径不同，这给推理优化（如批处理、缓存）带来了额外的复杂性。

---

**Mistral 7B / Mixtral 8x7B 与 LLaMA 系列的关系及后续影响**

&emsp;&emsp;**Mistral 7B 和 Mixtral 8x7B 是 LLaMA 架构范式在“效率”维度上的关键扩展。** 它们沿用了 LLaMA 确立的 decoder-only Transformer 架构和 RoPE、RMSNorm、SwiGLU 等标准组件，但在注意力机制和 FFN 结构上进行了根本性的优化。Mistral 7B 的 SWA + GQA 组合和 Mixtral 8x7B 的 SMoE 架构，分别从稠密和稀疏两个方向探索了“在更小计算预算下实现更强能力”的路径。

&emsp;&emsp;这两款模型对后续 LLM 架构的影响是深远的。**MoE 架构**在 Mixtral 8x7B 之后迅速成为开源模型的主流选择之一：DeepSeek-V2/V3 采用了细粒度的 MoE 设计，Qwen 1.5-MoE 和 Qwen 2.5-MoE 将 MoE 与长上下文结合，Llama 4 也转向了 MoE 架构。**GQA** 在 Mixtral 之后成为几乎所有主流 LLM 的标配——LLaMA 3、Qwen 2、Gemma 2 等模型全部采用 GQA 或其变体。**SWA** 虽然被 Mistral 7B v0.2 放弃，但其思想被后续的滑动窗口与全注意力混合方案继承，并在长上下文场景中持续演进。

&emsp;&emsp;从更宏观的视角看，Mistral 7B 和 Mixtral 8x7B 代表了 LLM 演进中一个被低估的维度：**架构效率**。在 GPT 系列和 LLaMA 系列主导的“规模叙事”中，性能提升往往被归因于更大的参数量、更多的数据、更强的算力。Mistral 的工作提醒了整个领域：同样的性能可以用更少的计算获得，而效率本身就是一种竞争力。当推理成本成为 LLM 部署的主要瓶颈时，效率优先的设计哲学——无论是注意力层面的 SWA/GQA，还是架构层面的 MoE——将变得比单纯的规模扩张更加重要。

### 4.4 Falcon

- 论文地址：[The Falcon Series of Open Language Models](https://arxiv.org/pdf/2311.16867)
- 数据集论文：[The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data Only](https://arxiv.org/pdf/2306.01116)

&emsp;&emsp;LLaMA 系列和 Mistral 系列分别从“规模驱动”和“效率驱动”两个方向推进了开放权重大语言模型的发展。但两者共享一个隐含假设：**高质量的预训练数据必须混合来自书籍、论文、社交媒体等“ curated ”来源，单纯依靠网页爬取数据无法训练出顶级模型。** GPT-3 的数据混合中，Common Crawl 仅占 60%，其余来自 WebText2、Books 和 Wikipedia；LLaMA 1 同样混合了 CommonCrawl、C4、Wikipedia、Books、Github、arXiv 等多个来源。这种“网页 + curated 混合”的策略虽然有效，但 curated 语料的获取和清洗成本高昂，且规模难以持续扩展。阿联酋技术创新研究院（Technology Innovation Institute, TII）的 Falcon 系列对此提出了一个根本性的挑战：**如果对网页数据进行足够严格的清洗和去重，单纯依靠 Common Crawl 是否也能训练出与 curated 混合语料同等甚至更优的模型？** Falcon 的答案是肯定的。TII 构建了 RefinedWeb——一个完全基于 Common Crawl、经过严格过滤和去重的网页数据集——并在此之上训练了 Falcon-7B、Falcon-40B 和 Falcon-180B 三个模型。Falcon-40B 在发布时登顶 Hugging Face 开源模型排行榜，Falcon-180B 以 1800 亿参数成为当时最大的开放权重语言模型，性能接近 PaLM-2-Large。更重要的是，Falcon 系列以 Apache 2.0 许可完全开源，与 LLaMA 的受限许可形成鲜明对比，为开源社区提供了一个真正无约束的顶级模型基座。

---

**RefinedWeb：只靠网页数据能否训练出顶级模型？**

&emsp;&emsp;RefinedWeb 是 Falcon 系列最核心的贡献之一，它回答了一个在 LLM 预训练中长期悬而未决的问题：网页数据的质量上限在哪里？

&emsp;&emsp;在 Falcon 之前，Common Crawl 被普遍视为“低质量、高噪声”的数据源，需要与 curated 语料混合使用才能训练出好模型。TII 的 RefinedWeb 论文通过系统实验证明，这一判断是错误的——问题不在于网页数据本身，而在于清洗和去重的力度不够。RefinedWeb 的构建流程由五个顺序执行的阶段组成：

- **URL 过滤**：制定 URL 黑名单，移除成人内容、恶意软件、垃圾信息等来源的页面。同时对 URL 进行评分，保留来自可信域名和高质量路径的内容；
- **内容抽取**：使用 trafilatura 等工具从 HTML 中抽取正文内容，丢弃导航栏、广告、页脚等模板化元素；
- **语言识别**：使用 fastText 训练语言识别模型，保留英语得分高于阈值的文章，去除非英语页面和由非自然语言构成的页面；
- **质量过滤**：在篇章级别和句子级别进行启发式过滤。篇章级别检查文章整体长度、标点符号比例、重复内容比例；句子级别过滤掉过短、重复或格式异常的句子；
- **大规模去重**：首先使用 MinHash 进行模糊去重，然后进行精确子串去重。MinHash 为每篇文章计算 9000 个哈希值，使用 20 个桶、每桶 450 个值进行筛选，最后移除重复片段超过 50 个 token 的文章。

&emsp;&emsp;RefinedWeb 的公开版本包含约 **9.68 亿个网页**，总计 **2.8TB 的干净文本数据**，约 **5000 亿到 6500 亿 token**（取决于分词器）。论文的关键实验是：分别用 RefinedWeb 单独训练、RefinedWeb + curated 混合训练、纯 curated 训练的模型进行对比。结果显示，**仅用 RefinedWeb 训练的模型在多个基准上达到或超过了 curated 混合语料训练的模型**。这意味着，网页数据的质量瓶颈并非不可逾越——只要清洗和去重足够彻底，网页数据完全可以替代 curated 语料。

&emsp;&emsp;这一发现的实践意义是巨大的。Curated 语料（书籍、论文、社交媒体）的获取受到版权、规模和时效性的多重限制，而网页数据的规模几乎是无限的。RefinedWeb 证明了“数据质量优先于数据来源”的原则：一份经过严格清洗的网页数据集，可以在规模上持续扩展，同时保持与 curated 语料相当的质量水平。

---

**架构创新：并行注意力与 MLP 的工程取舍**

&emsp;&emsp;Falcon 系列的架构建立在 decoder-only Transformer 的基础上，但 TII 在工程层面做出了几个与 LLaMA 系列不同的关键选择，这些选择直接影响了模型的推理效率和部署成本。

&emsp;&emsp;**多查询注意力（Multi-Query Attention, MQA）** 是 Falcon 系列最核心的架构优化。在标准多头注意力（MHA）中，每个 Query 头对应独立的 Key 头和 Value 头，KV 缓存的大小与注意力头数成正比。MQA 将 Key 和 Value 头的数量压缩为 **1**，所有 Query 头共享同一组 Key 和 Value 头。这一改动使 KV 缓存的大小从 $2 \cdot h \cdot d_h \cdot L$ 降至 $2 \cdot d_h \cdot L$（$h$ 为头数，$d_h$ 为头维度，$L$ 为层数），缓存占用降低至原来的 $1/h$。对于 Falcon-40B 这样的模型（60 层、128 个注意力头），MQA 带来的显存节省是决定性的——它使 40B 参数模型可以在单张 A100 80GB 上以合理批量进行推理。论文报告 MQA 在训练稳定性和最终性能上与 MHA 相当，但推理效率显著提升。

&emsp;&emsp;**并行注意力与 MLP** 是 Falcon 的另一个工程创新。在标准 Transformer 块中，自注意力层和前馈网络（MLP）是顺序执行的：先做注意力，再做 MLP，两者通过残差连接串联。Falcon 将两者改为**并行执行**：输入同时送入注意力分支和 MLP 分支，两个分支的输出相加后得到最终输出。这种并行化设计减少了串行计算链的长度，在训练时可以更充分地利用 GPU 的计算资源，在推理时也能降低延迟。TII 在论文中指出，并行注意力与 MLP 的设计受到 GPT-J 的启发，但在 Falcon 中得到了系统验证和规模化应用。

&emsp;&emsp;**旋转位置编码（RoPE）** 与 LLaMA 系列一致，Falcon 使用 RoPE 替代可学习的绝对位置嵌入，使模型在理论上支持长度外推。Falcon 系列使用标准 RoPE，未采用 LLaMA 3 中出现的频率调整等改进。**FlashAttention** 在 Falcon 的训练和推理中得到了大规模应用，通过分块计算和显存优化，将注意力操作的内存访问从二次方降低到线性，使长序列训练在合理显存下成为可能。

&emsp;&emsp;综合来看，Falcon 的架构选择体现了“**推理效率优先**”的设计哲学。MQA 降低 KV 缓存、并行注意力与 MLP 减少串行计算、FlashAttention 优化显存访问——这些优化的共同目标是让大规模模型在实际部署中更可行。与 Mistral 7B 的 SWA + GQA 组合不同，Falcon 选择了更激进的 MQA（而非 GQA），在效率上走得更远，但代价是 Key/Value 表达能力的进一步压缩。后来的实践表明，MQA 在超大规模模型（如 180B）上的性能损失逐渐显现，GQA 作为 MQA 和 MHA 之间的折中方案，成为更主流的选择——LLaMA 3、Mistral 系列、Qwen 系列均采用了 GQA。

---

**Falcon 与 LLaMA、Mistral 的关系及后续影响**

&emsp;&emsp;**Falcon 系列是 LLaMA 架构范式在“数据工程”维度上的关键扩展。** 它沿用了 decoder-only Transformer 的基本框架，但在两个层面做出了独特贡献：在架构层面，MQA 和并行注意力与 MLP 为推理效率提供了不同于 LLaMA 的优化路径；在数据层面，RefinedWeb 首次系统证明，经过严格清洗的网页数据可以独立训练出顶级模型，无需依赖 curated 语料的混合。这一发现对整个 LLM 领域的数据策略产生了深远影响——后续的 FineWeb、Dolma、RedPajama 等开源数据集都继承了 RefinedWeb 的“网页数据 + 严格清洗”范式，而 LLaMA 3 的 15T token 训练数据也大幅增加了网页数据的比重。

&emsp;&emsp;Falcon 系列对开源社区的影响同样显著。Falcon-40B 的 Apache 2.0 许可是当时高性能开源模型中最宽松的之一，直接推动了其在企业级应用中的 adoption。Falcon-180B 虽然改用了更严格的许可，但其作为“最大开放权重模型”的地位为 TII 赢得了全球关注，使阿联酋成为 LLM 领域少数几个拥有顶级基础模型能力的国家之一。TII 后续推出的 Falcon 2（11B，2024 年 5 月）、Falcon Mamba（7B，2024 年 8 月）和 Falcon-H1（2025 年）进一步扩展了这一系列，其中 Falcon Mamba 是首个在 Hugging Face 排行榜上登顶的 State Space Language Model，Falcon-H1 则采用了混合头架构，在效率上继续探索。

&emsp;&emsp;从更宏观的视角看，Falcon 系列代表了 LLM 演进中一个重要的维度：**数据工程的独立价值**。在 LLaMA 系列主导的“架构标准化”叙事和 Mistral 主导的“架构效率”叙事之外，Falcon 提出了第三个问题：同样的架构、同样的规模，不同的数据清洗策略能带来多大的性能差异？RefinedWeb 的答案是：差异可以很大——大到网页数据可以替代 curated 语料。这一发现使“数据质量”从一个辅助因素变成了 LLM 设计的核心变量，也提醒了整个领域：在架构逐渐趋同的背景下，数据工程可能是下一个竞争的主战场。

## 5 多模态与长文本

### 5.1 GPT-4 / GPT-4o

- 论文地址：
  - GPT-4：[GPT-4 Technical Report](https://arxiv.org/pdf/2303.08774)
  - GPT-4o：[Hello GPT-4o](https://openai.com/index/hello-gpt-4o/)

&emsp;&emsp;GPT-3 证明了规模可以带来上下文学习能力，InstructGPT 和 ChatGPT 则证明了 RLHF 可以将模型的行为对齐到人类意图。但这两条路线共享一个隐含前提：**模型只处理文本，且推理成本随参数量线性增长**。GPT-3 的 1750 亿参数每次前向传播需要激活全部参数，训练和推理成本极高；ChatGPT 虽然在对话能力上取得了突破，但它本质上仍是一个纯文本模型，无法理解图像、声音或视频。更重要的是，从 GPT-3 到 GPT-4，如果继续沿着“稠密 Transformer + 扩大参数”的路线前进，训练成本将变得不可承受——GPT-3 的训练成本约为 460 万美元，而一个 10 倍规模的稠密模型训练成本将超过 5000 万美元，且推理成本同样令人望而却步。GPT-4 的核心思路是：**在架构层面用混合专家（MoE）替代稠密 Transformer，在模态层面将纯文本模型扩展为多模态模型，在推理层面用推测解码等技术降低延迟**。这三个维度的创新共同构成了 GPT-4 作为 LLM 架构演进分水岭的地位。GPT-4o 则进一步将多模态能力从“拼接式”升级为“端到端统一”，用一个模型同时处理文本、视觉和音频，并将 API 价格降低 50%。两者的主要贡献包括：

1. GPT-4 采用混合专家（MoE）架构，在约 1.8 万亿总参数中每次前向传播仅激活约 2800 亿参数，在保持模型容量的同时将推理计算量控制在可接受范围内；
2. GPT-4 首次在 LLM 中实现大规模多模态能力，支持图像和文本的联合输入，能够理解图表、照片、手写文字和复杂视觉场景；
3. GPT-4 在专业和学术基准上达到人类水平，在模拟律师考试中取得前 10% 的成绩，MMLU 准确率在 57 个学科上大幅超越此前所有模型；
4. GPT-4o 采用单一 Transformer 架构端到端训练文本、视觉和音频，实现跨模态实时推理，音频响应延迟低至 232 毫秒，接近人类对话的响应时间；
5. GPT-4o 在保持 GPT-4 Turbo 文本性能的同时，API 价格降低 50%，推理速度提升 2 倍，使顶级多模态能力首次具备了大规模部署的经济可行性。

---

#### GPT-4：混合专家架构与多模态能力的工程突破

&emsp;&emsp;GPT-4 于 2023 年 3 月发布，是 OpenAI 首个大规模多模态模型。其技术报告明确指出，GPT-4 可以接受图像和文本输入并产生文本输出，在多种专业和学术基准上达到人类水平。但技术报告对架构细节的披露极为有限，OpenAI 以“竞争格局和安全影响”为由拒绝公开模型规模、硬件、训练计算量、数据集构建方法和训练方法。真正让外界理解 GPT-4 架构的，是一份由 SemiAnalysis 发布的泄露文档，该文档基于多个信源披露了 GPT-4 的架构、基础设施、训练数据集和成本细节。

**混合专家架构：1.8 万亿总参数，2800 亿激活参数**

&emsp;&emsp;GPT-4 最核心的架构创新是采用混合专家模型。根据泄露文档，GPT-4 在约 120 层 Transformer 中拥有大约 **1.8 万亿总参数**，是 GPT-3 的 10 倍以上。这些参数分布在 **16 个专家**中，每个专家的 MLP 参数约为 **1110 亿**，此外还有约 550 亿共享参数用于注意力机制。每次前向传播时，一个路由算法为每个 token 选择 **2 个专家**进行计算，因此实际激活的参数约为 **2800 亿**，消耗约 **560 TFLOP** 的计算量。作为对比，如果使用纯稠密模型，每次前向传播需要激活全部 1.8 万亿参数，消耗约 **3700 TFLOP**。

&emsp;&emsp;MoE 架构的核心价值在于**解耦模型容量与推理成本**。1.8 万亿总参数赋予了模型强大的知识存储和表示能力，但每次推理只激活 2800 亿参数，使推理成本仅相当于一个中等规模稠密模型的水平。这与 Mistral 的 Mixtral 8x7B 在哲学上完全一致——用稀疏激活替代稠密计算——但 GPT-4 的规模远大于 Mixtral：Mixtral 的总参数为 46.7B、激活 13B，而 GPT-4 的总参数为 1.8T、激活 280B，两者相差约 20 倍。

&emsp;&emsp;GPT-4 的训练使用了约 **13 万亿 token** 的训练数据，预训练阶段的上下文长度为 **8K**，后续通过微调扩展到 **32K**。训练在约 **25,000 张 NVIDIA A100 GPU** 上进行，耗时约 **90 到 100 天**。训练采用了 **8 路张量并行**，因为这是 NVLink 的限制——A100 的 NVLink 带宽支持 8 路全互联，超过 8 路后通信效率显著下降。

**多模态能力：从纯文本到图文联合理解**

&emsp;&emsp;GPT-4 的另一个里程碑意义在于将 LLM 从纯文本模型扩展为多模态模型。GPT-4 可以接收图像和文本的任意组合作为输入，并生成文本输出。这一能力使其能够执行此前 LLM 无法完成的任务：解释图表中的数据趋势、识别照片中的物体和场景、理解手写文字和复杂文档布局、分析工程图纸中的标注和尺寸。

&emsp;&emsp;GPT-4 的视觉能力在多个应用场景中得到了验证。微软的 “Be My Eyes” 应用利用 GPT-4 的视觉能力为视障用户提供实时帮助——描述冰箱中的食物、读取产品标签、识别药品包装上的说明、在陌生环境中导航。在专业领域，GPT-4 的视觉能力被用于医学影像的初步分析和工业质检中的缺陷识别。

&emsp;&emsp;需要注意的是，GPT-4 的多模态能力在实现方式上与 GPT-4o 有本质区别。GPT-4 的多模态是 **“拼接式”** 的：视觉和文本模态通过独立的编码器处理，然后在某个中间层进行融合。而 GPT-4o 采用的是 **“端到端统一”** 架构，所有模态共享同一个神经网络。这一差异在 GPT-4o 部分详述。

**推测解码：降低推理延迟的关键技术**

&emsp;&emsp;GPT-4 的 MoE 架构降低了每次前向传播的计算量，但 2800 亿激活参数的推理延迟仍然可观。为了进一步加速推理，OpenAI 在生产环境中使用了 **推测解码**（Speculative Decoding）技术。

&emsp;&emsp;推测解码的核心思想是“先猜后验”：使用一个更小、更快的“草稿模型”先生成 K 个候选 token，然后让大模型（目标模型）对这 K 个 token 进行一次性验证。如果草稿模型的预测与大模型的预测一致，则接受这些 token；如果不一致，则从第一个不一致的位置开始重新生成。由于大模型只需要进行一次前向传播就能验证多个 token，而不是逐 token 生成，整体推理速度可以提升 2 到 3 倍，且理论上不损失生成质量。

&emsp;&emsp;推测解码的数学保证在于：被接受的 token 分布与目标模型直接采样的分布完全一致。这是因为验证步骤使用的是目标模型的概率分布，草稿模型只负责“提议”，目标模型负责“裁决”。因此，推测解码是一种**无损加速**方法——它不改变模型的输出分布，只是改变了生成过程的计算顺序。

**安全对齐与性能表现**

&emsp;&emsp;GPT-4 在预训练之后经过了与 InstructGPT 类似的 RLHF 对齐流程，但增加了多项安全措施：基于规则的奖励模型（rule-based RM）、额外的安全提示、以及模型辅助的安全管线。RLHF 微调显著提升了模型的安全性，但技术报告同时指出，**模型在考试中的能力主要来自预训练过程，RLHF 后训练对齐对其影响不大**——这意味着对齐调整的是模型的行为倾向，而非知识或推理能力本身。

&emsp;&emsp;在性能表现上，GPT-4 在 MMLU 基准的 57 个学科上大幅超越此前所有模型，在英语之外的其他语言上也表现出色。在模拟律师考试中，GPT-4 的成绩进入前 10%，而 GPT-3.5 仅处于后 10%。这些结果表明，GPT-4 的能力提升并非简单的规模放大，而是 MoE 架构带来的容量-效率权衡和多模态能力共同作用的结果。

---

#### GPT-4o：端到端统一多模态与实时交互

&emsp;&emsp;GPT-4 的多模态能力虽然开创了 LLM 的新应用场景，但其“拼接式”架构存在两个根本性缺陷。第一，**模态间信息丢失**：视觉和音频的信息在各自的编码器中被压缩后，关键细节可能无法完整传递到文本生成阶段。第二，**延迟累积**：各模态的处理流水线串联执行，实时性受到严重限制。在 GPT-4o 之前，ChatGPT 的语音模式由三个独立模型串联组成——一个模型将音频转录为文本，GPT-4 处理文本并输出文本，第三个模型将文本转换回音频。这一管线的平均延迟为 GPT-3.5 的 2.8 秒和 GPT-4 的 5.4 秒，且 GPT-4 在这个过程中“丢失了大量信息——它无法直接观察语调、多个说话者的差异或背景噪音，也无法输出笑声、歌唱或表达情感”。

&emsp;&emsp;GPT-4o 于 2024 年 5 月发布，其名称中的“o”代表“omni”（全知、全能）。与 GPT-4 的根本区别在于，GPT-4o 是**一个端到端训练的单一模型**，文本、视觉和音频的所有输入和输出都由同一个神经网络处理。

**统一 Transformer 架构：跨模态注意力机制**

&emsp;&emsp;GPT-4o 的架构核心是**统一的 Transformer 设计**。与传统的多模态模型为不同模态分别设计编码器和解码器不同，GPT-4o 将所有模态的数据统一编码到同一个神经网络中处理。其核心机制是**动态跨模态注意力**（Dynamic Cross-Modal Attention），通过模态掩码矩阵让模型在推理时动态决定不同模态特征的权重。例如，在处理“描述图片中的对话场景”这一任务时，模型可以同时关注图像中的肢体语言、背景音效和对话文本，而非孤立地分析每一个模态。

&emsp;&emsp;这种设计的优势在于**信息不经过模态转换的中间瓶颈**。在 GPT-4 的拼接式架构中，视觉信息必须先被压缩为某种“视觉 token”，再与文本 token 拼接；音频信息必须先被转录为文本，再送入语言模型。每一步转换都可能丢失原始模态中的关键信息——例如语音中的情感语调、图像中的空间关系。GPT-4o 的统一架构消除了这些中间转换，使模型能够直接“感知”原始的多模态信号。

**实时交互：232 毫秒的响应延迟**

&emsp;&emsp;GPT-4o 最直观的突破是实时交互能力。其音频输入的响应延迟低至 **232 毫秒**，平均响应时间为 **320 毫秒**，与人类对话中的响应时间相当。这一延迟水平使 GPT-4o 能够支持真正的实时语音对话——用户可以像与真人交谈一样与模型交互，包括打断、重叠说话、以及自然的停顿和节奏。

&emsp;&emsp;GPT-4o 的音频能力不仅限于语音识别和合成，还包括**情感理解和表达**。模型能够感知说话者的情绪状态（通过语调、语速、停顿等特征），并在输出中表达相应的情感——包括笑声、歌唱和不同风格的情感表达。在 OpenAI 的现场演示中，GPT-4o 能够根据对话内容调整语气，从平静的叙述切换到兴奋的语调，甚至在听到工作人员深呼吸时做出相应的反应。

**性能与成本：同水平性能，一半价格**

&emsp;&emsp;GPT-4o 在传统基准上达到了 GPT-4 Turbo 的文本、推理和编码性能水平，同时**在多语言、音频和视觉能力上设立了新的标杆**。在 MMLU 基准上，GPT-4o 达到 **88.7%** 的准确率，高于 GPT-4 的 86.4% 和 GPT-4 Turbo 的 87.4%。在医学住院医师考试等专业评估中，GPT-4o 的准确率（85.88%）显著高于 GPT-4（81.27%）。

&emsp;&emsp;GPT-4o 的另一个关键改进是**成本**。GPT-4o 的 API 价格为每百万输入 token **$2.50**、每百万输出 token **$10**，相比 GPT-4 Turbo 降低了 **50%**。这一降价策略使顶级多模态能力首次具备了大规模部署的经济可行性。此外，GPT-4o 还通过 **INT4 量化**将模型体积缩小至原版的 **1/8**，同时保持 **98%** 的精度，使边缘设备上的本地运行成为可能。

---

#### GPT-4 与 GPT-4o 的关系及后续影响

&emsp;&emsp;**GPT-4 和 GPT-4o 代表了 LLM 架构演进中“能力扩展”与“效率优化”两个维度的融合。** GPT-4 通过 MoE 架构解决了“如何在有限推理成本下扩展模型容量”的问题，通过多模态能力解决了“如何让 LLM 理解文本之外的世界”的问题。GPT-4o 则在 GPT-4 的基础上，通过端到端统一架构解决了“如何让多模态交互真正实时、自然”的问题，通过成本优化解决了“如何让顶级能力惠及更多用户”的问题。

&emsp;&emsp;从架构演进的脉络看，GPT-4 和 GPT-4o 的贡献可以从三个层面理解。**在稀疏激活层面**，GPT-4 的 MoE 架构与 Mistral 的 Mixtral 8x7B 形成了有趣的对照：Mixtral 证明了 MoE 在开源模型中的可行性，GPT-4 则证明了 MoE 在超大规模（1.8T 总参数）下的有效性。两者的共同结论是：MoE 是当前 LLM 在容量与效率之间实现最优权衡的主流架构选择。这一结论在后续的 DeepSeek-V3、Llama 4、Qwen 2.5-MoE 等模型中得到了持续验证。

&emsp;&emsp;**在多模态层面**，GPT-4 的“拼接式”多模态和 GPT-4o 的“端到端统一”多模态代表了两种不同的技术路线。GPT-4 的路线更接近 LLaVA、Flamingo 等开源多模态模型的设计——用独立的视觉编码器（如 CLIP ViT）提取图像特征，再通过交叉注意力或投影层注入语言模型。GPT-4o 的路线则更激进：所有模态共享同一个 Transformer，不区分编码器和解码器。这一设计在理论上更优雅，但训练难度也更大——需要足够多的多模态对齐数据来让模型学会在统一表示空间中处理不同模态的信息。

&emsp;&emsp;**在推理优化层面**，GPT-4 的推测解码和 GPT-4o 的 INT4 量化代表了两种互补的策略。推测解码通过“先猜后验”减少大模型的前向传播次数，INT4 量化通过降低数值精度减少每次前向传播的内存和计算开销。两者共同作用，使 GPT-4o 在保持 GPT-4 Turbo 性能的同时将 API 价格降低 50%。

&emsp;&emsp;从更宏观的视角看，GPT-4 和 GPT-4o 标志着 LLM 从“纯文本生成器”向“多模态通用智能体”的转变。GPT-4 首次让 LLM 能够“看到”世界，GPT-4o 则让 LLM 能够“实时地看到、听到并回应”世界。这一转变的影响是深远的：它使 LLM 的应用场景从文本写作、代码生成、问答扩展到了机器人控制、实时翻译、无障碍辅助、工业质检、医疗影像分析等需要多模态感知的领域。GPT-4o 的 232 毫秒响应延迟和情感表达能力，则为 LLM 作为“对话伙伴”而非“文本工具”的定位提供了技术基础。

&emsp;&emsp;GPT-4 和 GPT-4o 的局限同样值得关注。**幻觉问题**仍然存在——模型会以自信的语气编造事实，在多模态场景中，这一风险可能更高，因为模型需要对视觉信息进行“解读”而非“检索”。**推理成本**虽然通过 MoE 和量化得到了控制，但 1.8 万亿总参数的存储需求和 2800 亿激活参数的计算需求，仍然使 GPT-4 级别的模型难以在消费级硬件上本地运行。**对齐的鲁棒性**在多模态场景中面临新的挑战：图像和音频中的隐晦信息可能被用于绕过文本层面的安全过滤。这些问题在 GPT-4o 之后的 o1、o3 等推理模型中被进一步探索，但尚未得到根本解决。

### 5.2 Gemini 1.5 Pro

- 论文地址：[Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf)

&emsp;&emsp;GPT-4 通过 MoE 架构解决了“如何在有限推理成本下扩展模型容量”的问题，GPT-4o 则通过端到端统一架构解决了“如何让多模态交互真正实时、自然”的问题。但两者在上下文长度上仍然受到根本性限制：GPT-4 Turbo 的上下文窗口为 128K token，GPT-4o 同样为 128K。128K token 意味着模型一次最多处理约 300 页文本、或一段数分钟的视频。对于需要分析整本书、整部电影、或大型代码库的任务，这一窗口远远不够。与此同时，RAG（检索增强生成）成为当时解决长上下文问题的主流方案——将文档切分为片段，通过向量检索找到相关片段后送入模型。但 RAG 存在固有缺陷：检索精度有限，可能遗漏关键信息；片段之间的上下文关系在切分时丢失；多跳推理能力受限于检索质量。Google DeepMind 的 Gemini 1.5 Pro 对这两个问题给出了一个统一的答案：**通过稀疏混合专家（MoE）架构在保持计算效率的同时扩展模型容量，通过突破性的长上下文技术将上下文窗口从 128K 提升到 100 万 token，并在研究中成功验证了 1000 万 token 的极限。** Gemini 1.5 Pro 的核心突破在于：它证明了模型可以在数百万 token 的上下文中实现近乎完美的信息检索和推理，使“直接处理整个文档库”成为可能，从而从根本上动摇了 RAG 作为长上下文解决方案的必要性。Gemini 1.5 Pro 的主要贡献包括：

1. 采用稀疏混合专家（MoE）架构，在 175B 参数量中仅激活部分专家，实现了与 Gemini 1.0 Ultra 相当的质量，同时训练计算量大幅降低；
2. 将上下文窗口从 Gemini 1.0 的 32K token 扩展到 **100 万 token**（生产环境），并在研究中成功测试了 **1000 万 token**，创下当时大规模基础模型的最长上下文记录；
3. 在跨模态长上下文检索任务中实现 **近乎完美的召回率（>99.7%）** ，覆盖文本、视频和音频三种模态，且召回率不随上下文长度增加而衰减；
4. 原生支持文本、图像、音频和视频的混合模态输入，在长文档问答、长视频问答和长上下文语音识别等任务上刷新了当时最优水平；
5. 展示了“上下文学习”的新高度：模型能够仅凭一本语法书和双语词表，在上下文中学习一门全新语言（Kalamang）的翻译，质量接近从同一资料学习的人类。

---

**MoE 架构：计算效率的代际提升**

&emsp;&emsp;Gemini 1.5 Pro 最核心的架构变革是从 Gemini 1.0 的稠密 Transformer 转向了**稀疏混合专家（Sparse Mixture-of-Experts, MoE）架构**。这一转变的直接动因是计算效率：传统的稠密 Transformer 在每次前向传播时需要激活全部参数，随着模型规模的增长，训练和推理成本呈线性上升。而 MoE 架构通过门控网络为每个 token 选择性地激活一部分专家，使模型能够在不显著增加计算量的前提下大幅扩展总参数量。

&emsp;&emsp;根据公开的推断分析，Gemini 1.5 Pro 采用了 **16 个专家、每次激活 2 个**的 MoE 配置，模型总参数量约为 **175B**，隐藏维度为 **8192**，共 **72 层**，并使用了**分组查询注意力（GQA）** 。门控网络在每一层评估输入的 token 表示，从 16 个专家中选择最相关的 2 个进行激活，被选中的专家独立处理 token，其输出通过加权求和合并为最终表示。这种稀疏激活机制使模型在保持 175B 总参数容量的同时，每次前向传播只消耗约 2/16 的计算量。

&emsp;&emsp;Google 在 MoE 领域的研究积累可以追溯到 GShard、Switch Transformer 和 GLaM 等早期工作。GLaM 以 1.2 万亿参数、每 token 仅激活 96.6B（8%）的配置，证明了稀疏模型可以在计算成本与稠密模型相当的情况下实现更强的性能，同时训练能耗仅为 GPT-3 的约三分之一。Gemini 1.5 Pro 将这些研究成果规模化落地，在训练效率和服务效率上均实现了代际提升。DeepMind CEO Demis Hassabis 表示，MoE 架构使 Gemini 1.5 “在训练和服务上都更加高效”。

&emsp;&emsp;MoE 架构对 Gemini 1.5 Pro 的长上下文能力至关重要。在稠密模型中，处理 100 万 token 的上下文需要为每个 token 激活全部参数，计算量将是天文数字。而 MoE 的稀疏激活特性使得模型可以在处理超长序列时，将计算资源集中在与当前 token 最相关的专家上，从而在不显著增加单 token 计算量的前提下支持极长的上下文窗口。

---

**长上下文突破：从 128K 到 1000 万 token**

&emsp;&emsp;Gemini 1.5 Pro 最引人注目的突破是将上下文窗口从 Gemini 1.0 的 32K token 扩展到 **100 万 token**（生产环境默认 128K），并在研究中成功测试了 **1000 万 token** 的极限。100 万 token 意味着模型可以一次性处理：1 小时的视频、11 小时的音频、超过 3 万行代码、或超过 70 万字的文本。1000 万 token 则相当于约 10 小时的视频或数百万页文档。

&emsp;&emsp;这一突破并非简单的工程优化。Google DeepMind 的研究团队在长上下文项目上经历了一系列深度学习创新，研究科学家 Nikolay Savinov 回忆道：“我们最初的目标是 128,000 token，我觉得设置一个雄心勃勃的目标会更好，所以建议了 100 万 token。现在我们甚至在研究中超越了它 10 倍”。工程师 Denis Teplyashin 补充说：“一个突破接一个突破，每一个都打开了新的可能性。当它们叠加在一起时，我们很惊讶地发现它们能做到什么——从 128K 跳到 512K，再到 100 万，最近甚至在内部研究中达到了 1000 万”。

&emsp;&emsp;长上下文的核心挑战不在于“能否接收这么多 token”，而在于“能否在这么多 token 中准确找到并推理所需信息”。Gemini 1.5 Pro 在“大海捞针”（Needle-in-a-Haystack）测试中实现了 **>99.7% 的召回率**，覆盖文本、视频和音频三种模态，且召回率不随上下文长度增加而衰减。这一结果的意义在于：它证明了模型可以在 100 万 token 的上下文中保持对任意位置信息的精确检索能力，而不像某些模型那样在上下文过长时出现“中间遗忘”现象。

&emsp;&emsp;Gemini 1.5 Pro 的长上下文能力在多个实际任务中得到了验证。在处理 402 页的阿波罗 11 号登月任务记录时，模型能够对文件中的对话、事件和细节进行推理，并识别出记录中的奇怪细节。在分析巴斯特·基顿 44 分钟的无声电影时，模型能够准确分析各种情节点和事件，甚至推理出电影中容易被遗漏的小细节。在代码领域，模型能够处理超过 100,000 行代码的提示，提出有用的修改建议并解释代码逻辑。

&emsp;&emsp;长上下文能力的一个关键意义在于它对 RAG 范式的挑战。传统 RAG 需要将文档切分为片段、通过检索找到相关片段后送入模型，但检索精度有限、片段间上下文关系丢失。Gemini 1.5 Pro 的 100 万 token 上下文窗口使模型可以直接处理整个文档库，在 1382 页的《悲惨世界》全文中定位一个手绘场景，总 token 数达到 732K。这种“直接处理全量信息”的方式消除了 RAG 的检索瓶颈，在需要全局理解或跨文档推理的任务中具有根本性优势。

---

**多模态能力：原生混合模态理解**

&emsp;&emsp;Gemini 1.5 Pro 是一个**中型的多模态模型**，其多模态能力是**原生**的，而非像某些模型那样在纯文本模型上“拼接”视觉或音频模块。Demis Hassabis 明确指出，MoE 架构“从一开始就整合了文本、图像和音频模态”，而不是在后期阶段将独立的文本、图像和音频模型以繁琐的方式组合。

&emsp;&emsp;在视频理解方面，Gemini 1.5 Pro 在 VATEX 英文视频描述任务上取得了 **64.6** 的分数（4-shot），显著高于 Gemini 1.0 Pro 的 57.4 和 1.0 Ultra 的 62.7；在中文视频描述任务上，Gemini 1.5 Pro 达到 **55.3**，同样超越了前代模型。在音频理解方面，Gemini 1.5 Pro 在长上下文语音识别任务上刷新了当时最优水平，并且能够处理长达 **11 小时的音频**。在跨模态长上下文检索中，模型在文本、视频和音频三种模态上均实现了近乎完美的召回率。

&emsp;&emsp;Gemini 1.5 Pro 的多模态能力在一个特别具有挑战性的任务中得到了展示：**从零样本翻译到罕见语言**。Kalamang 语是一种全球不到 200 人使用的语言，只有一本语法书和一份双语词表。在没有经过任何 Kalamang 语专门训练的情况下，Gemini 1.5 Pro 仅通过在上下文中提供这本语法书和词表，就实现了与从同样资料学习的人类相当水平的翻译质量。这一结果展示了长上下文与多模态能力结合后的独特价值：模型可以将大量的参考材料作为上下文输入，在推理时“现场学习”新任务，而无需任何参数更新。

---

**局限**

&emsp;&emsp;Gemini 1.5 Pro 的长上下文能力虽然令人印象深刻，但其局限性同样值得关注。**延迟与计算成本**是首要问题。处理 100 万 token 的上下文需要大量的注意力计算，即使 MoE 架构降低了单 token 计算量，总计算量仍然巨大。Google 在发布时明确指出“正在积极优化以改善延迟，减少计算需求并增强用户体验”。在实际部署中，100 万 token 的上下文窗口并非默认可用，而是需要通过 Private Preview 申请。

&emsp;&emsp;**技术细节的不透明**是另一个争议点。Gemini 1.5 的技术报告长达 153 页，但关于模型的具体参数量、专家数量、激活参数比例等关键架构细节并未完全公开。外界对 Gemini 1.5 Pro 的 175B 参数、16 专家、2 激活的配置了解主要来自第三方推断分析，而非官方披露。这种“选择性透明”引发了学术界的讨论：与 LLaMA 系列和 Mistral 系列的完全开源形成对比，Gemini 1.5 Pro 作为闭源模型，其架构细节的缺失使研究者难以复现或深入分析其长上下文能力的技术根源。

&emsp;&emsp;**长上下文的实际有效性**也存在争议。GDELT 项目在 Gemini 1.5 Pro 发布后进行的“真实世界大海捞针”测试显示，模型在某些特定任务上的表现与官方公布的 >99.7% 召回率存在差距，在需要精确时间推理或跨文档关联的任务中，性能下降更为明显。这表明“大海捞针”测试虽然验证了模型的基础检索能力，但长上下文在复杂推理任务中的有效性仍需更全面的评估。

---

**Gemini 1.5 Pro 与 GPT-4o 的关系及后续影响**

&emsp;&emsp;**Gemini 1.5 Pro 和 GPT-4o 代表了 LLM 架构演进中“长上下文”与“实时多模态”两条并行但互补的技术路线。** GPT-4o 通过端到端统一架构和 INT4 量化实现了实时交互能力；Gemini 1.5 Pro 通过 MoE 架构和长上下文技术创新实现了超长信息处理能力。两者共同推动了 LLM 从“文本生成器”向“通用信息处理器”和“实时对话伙伴”的双向演进。

&emsp;&emsp;Gemini 1.5 Pro 的长上下文能力对 RAG 范式产生了深远影响。在 100 万 token 上下文窗口出现之前，RAG 是处理长文档的标准方案。Gemini 1.5 Pro 证明模型可以直接处理整个文档库，在 1382 页全文中定位一个场景、在 402 页任务记录中推理对话细节。这种“全量上下文”方式消除了检索环节的信息损失，在需要全局理解、跨文档推理或精确引用的场景中具有根本性优势。RAG 仍然在需要实时更新、大规模知识库或成本敏感的场景中有其价值，但 Gemini 1.5 Pro 的发布使“长上下文能否替代 RAG”成为一个被广泛讨论的议题。

&emsp;&emsp;在架构层面，Gemini 1.5 Pro 的 MoE 设计验证了稀疏激活在超大规模多模态模型中的有效性。这与 GPT-4 的 MoE 架构、Mixtral 8x7B 的稀疏专家设计形成了有趣的对照。三者都采用了“大总参数量 + 小激活参数量”的范式，但具体实现不同：GPT-4 使用 16 专家、2 激活、1.8T 总参数；Mixtral 使用 8 专家、2 激活、46.7B 总参数；Gemini 1.5 Pro 使用 16 专家、2 激活、175B 总参数。这一趋同表明，稀疏 MoE 已成为超大规模 LLM 在容量与效率之间实现最优权衡的主流架构选择。

&emsp;&emsp;Gemini 1.5 Pro 的后续发展清晰可见。2024 年底至 2025 年初，Google 推出了 **Gemini 2.0 和 Gemini 2.5** 系列，标志着 LLM 进入“智能体时代”。这些迭代模型在稀疏 MoE 架构的基础上引入了**推理时“思考”机制**，能够在推理过程中动态分配计算步骤。Gemini 2.0 Flash 以 Gemini 1.5 Pro 一半的延迟超越了其性能，并增加了原生多模态输出能力，如文生图协同生成和跨数小时视频的长上下文多模态推理。Gemini 2.5 系列进一步扩展了 MoE 架构，结合了来自人类和评论家反馈的强化学习（RL*F），在推理、编码和多模态基准上取得了当时最优的结果。

&emsp;&emsp;从更宏观的视角看，Gemini 1.5 Pro 代表了 LLM 演进中“上下文即能力”这一新范式的确立。在 GPT-3 时代，模型的能力由参数量决定；在 InstructGPT 时代，能力由对齐质量决定；在 Gemini 1.5 Pro 时代，能力开始由**上下文窗口**决定——能够处理多少信息，就能完成多复杂的任务。这一转变使 LLM 的应用场景从“生成文本”扩展到了“理解世界”：分析整部电影、推理整本书的逻辑、审计整个代码库的安全性。Gemini 1.5 Pro 的 100 万 token 上下文窗口和近乎完美的跨模态召回能力，为 LLM 成为“通用信息处理器”提供了技术基础，而 Gemini 2.5 的推理时计算机制则进一步将这一基础推向了“可推理的通用智能体”的方向。

### 5.3 Llama 3 / 3.1

- 论文地址：[The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783)

&emsp;&emsp;Gemini 1.5 Pro 通过 MoE 架构和 100 万 token 上下文窗口证明了闭源模型在长上下文和多模态能力上的领先地位，GPT-4o 通过端到端统一架构和实时交互能力巩固了 OpenAI 在产品化方面的优势。但两者都是闭源模型——研究者无法获取权重、无法本地部署、无法深入分析其架构细节。LLaMA 1 和 LLaMA 2 虽然以开放权重的方式发布，但其最大规模仅为 70B（LLaMA 2），在性能上与 GPT-4 级别的闭源旗舰仍有明显差距。LLaMA 3 和 LLaMA 3.1 的核心目标正是弥合这一差距：**用完全公开可获取的数据训练一个 4050 亿参数的模型，以开放权重的方式发布，在多个基准上追平甚至超越 GPT-4o 和 Claude 3.5 Sonnet，首次让开放权重模型达到顶级闭源模型的水平。** LLaMA 3.1 的技术报告长达 92 页，首次系统披露了 Meta 在大规模 LLM 训练中的数据工程、三阶段预训练流程和后训练配方。论文明确指出，LLaMA 3 的成功来自“数据、规模化、复杂度管理”三个关键因素，而非架构创新——它使用了非常传统的 decoder-only Transformer 架构，没有采用 MoE 或滑动窗口注意力。LLaMA 3 / 3.1 的主要贡献包括：

1. 训练了 405B 参数量的 LLaMA 3.1，是当时最大的开放权重语言模型，在 MMLU、IFEval、HumanEval 等多个基准上追平甚至超越 GPT-4o 和 Claude 3.5 Sonnet；
2. 将训练数据规模从 LLaMA 2 的 1.8T token 扩展到超过 **15 万亿 token**，数据量约为 LLaMA 2 的 8 倍以上，涵盖更多非英语数据、数学数据和代码数据；
3. 将上下文窗口从 LLaMA 3 的 8K token 扩展到 **128K token**，通过渐进式 RoPE 频率缩放实现，使模型能够处理约 50 页文本的长文档；
4. 采用 SFT + 拒绝采样 + DPO 的后训练流程，用旗舰 405B 模型优化 8B 和 70B 小模型，实现了模型族内的能力蒸馏；
5. 更新了 Llama 3 社区许可证，明确允许使用 LLaMA 3 进行合成数据生成和知识蒸馏以改进其他模型，生态系统包含 25 个合作伙伴，包括亚马逊、英伟达、Databricks、Groq、微软云和谷歌云。

---

**架构设计：传统而克制的工程选择**

&emsp;&emsp;LLaMA 3.1 的架构选择出人意料地保守。在 GPT-4 已经证明 MoE 架构有效、Mistral 已经证明滑动窗口注意力可用的背景下，Meta 明确拒绝了这两条路线。论文将这一决策归因于“复杂度管理”——在 405B 参数的超大规模下，训练稳定性比架构新颖性更为重要。Meta 发现，简单的稠密 Transformer 结构在超大规模训练中更可控、更可预测，而 MoE 的路由不稳定性和滑动窗口注意力的长距离信息损失在大规模下可能引入不可控的风险。

&emsp;&emsp;LLaMA 3.1 沿用了 LLaMA 系列确立的“标准配方”——decoder-only Transformer、Pre-Norm、RMSNorm、SwiGLU 激活函数、RoPE 旋转位置编码——但在此基础上进行了几项关键调整。最核心的变化是 **RoPE 基频从 LLaMA 2 的 10,000 提升到 500,000**。RoPE 基频决定了旋转角度随位置变化的速率：基频越大，相邻位置的旋转角度差越小，模型对远距离位置的分辨能力越强。论文指出，将 RoPE 基频提高到 500,000 使模型能够有效处理 32K 以上的上下文长度，这是 LLaMA 3.1 将上下文窗口从 8K 扩展到 128K 的关键技术前提。

&emsp;&emsp;**分组查询注意力（GQA）** 在 LLaMA 3.1 的所有规模上均被采用。与 LLaMA 2 仅在 70B 版本中使用 GQA 不同，LLaMA 3.1 的 8B、70B 和 405B 全部使用 GQA，多个 Query 头共享一组 Key 和 Value 头，显著减小了 KV 缓存的大小，提升了推理吞吐量。

&emsp;&emsp;**词汇表**从 LLaMA 2 的 32K 扩展到 **128K token**，使用 OpenAI 的 tiktoken 分词器开发。这一扩展显著提升了多语言场景下的分词效率——中文、日文、韩文等非拉丁语系的 token 压缩率大幅改善，同等文本所需的 token 数量减少，间接扩展了有效上下文长度。

&emsp;&emsp;在模型规模上，LLaMA 3.1 的 405B 版本拥有 **126 层 Transformer**，隐藏维度为 **16,384**，使用 128 个注意力头（GQA），FFN 中间维度约为 53,248。训练在 **16,000 块 NVIDIA H100 GPU** 上完成，使用了 **3.8 × 10²⁵ FLOPs** 的计算量。作为对比，GPT-3 的训练计算量约为 3.14 × 10²³ FLOPs，LLaMA 3.1 的计算量约为 GPT-3 的 120 倍。

---

**数据工程：15 万亿 token 的质量管控**

&emsp;&emsp;LLaMA 3 训练数据规模从 LLaMA 2 的 1.8 万亿 token 跃升至 **15.6 万亿 token**，数据量增长超过 8 倍。这一规模远超 Chinchilla 缩放规律的最优配比建议，反映了 Meta 的一个关键判断：在推理成本成为主要瓶颈的场景下，继续扩大数据规模仍然有效，甚至比扩大参数量更具性价比。

&emsp;&emsp;数据来源涵盖网页文本、代码、书籍、学术论文、多语言文本等多个类别，但 Meta 并未披露具体的来源细节。数据预处理采用了基于启发式规则和基于模型的双重质量过滤策略：fastText 分类器用于语言识别和低质量内容过滤，基于 RoBERTa 的分类器用于判断内容的教育价值和信息密度。这些分类器还帮助确定训练过程中数据混合的上下文类别，使 Meta 能够动态调整不同数据源在训练中的权重。

&emsp;&emsp;LLaMA 3 的一个显著变化是**大幅增加了代码和数学数据的比例**。论文指出，代码数据在 LLaMA 3 中的比重是 LLaMA 2 的约 4 倍。这一调整直接反映在 HumanEval 和 GSM8K 等代码与数学基准的显著提升上——LLaMA 3.1 405B 在 HumanEval 上达到 89.0%，较 LLaMA 3 70B 的 81.7% 有显著进步。代码训练采用了三种合成数据方法：**代码执行反馈**（让模型生成代码，通过单元测试验证正确性后微调）、**编程语言翻译**（将 Python 代码翻译为其他语言以扩充低资源语言的数据）、**文档反向翻译**（从代码生成注释和文档，再反向生成代码）。

&emsp;&emsp;多语言数据在 LLaMA 3 中占比约 **8%**，支持英语、德语、法语、意大利语、葡萄牙语、印地语、西班牙语和泰语八种语言。多语言训练数据的构成包括：2.4% 的人工标注数据、44.2% 来自其他 NLP 任务的数据、18.8% 的拒绝采样数据和 34.6% 的翻译推理数据。Meta 坦承人工标注在多语言数据中占比极低，大部分依赖合成数据和已有 NLP 数据集的复用。

---

**三阶段预训练：从标准训练到长上下文到退火**

&emsp;&emsp;LLaMA 3.1 的预训练并非一次性完成，而是分为三个顺序执行的阶段，每个阶段有不同的数据配比、上下文长度和训练目标。

&emsp;&emsp;**第一阶段：标准初始预训练。** 使用 15.6 万亿 token 的完整数据集，上下文窗口为 8K token，批次大小从 400 万 token 逐步增加到 1600 万 token。训练过程中，数据混合并非固定不变——Meta 在训练过程中动态调整不同数据源的权重，以优化模型的学习效率和泛化能力。

&emsp;&emsp;**第二阶段：长上下文继续预训练。** 在标准预训练完成后，Meta 使用 **8000 亿 token**（约占数据集总量的 5%）进行继续预训练，将上下文长度从 8K 逐步扩展到 128K。这一扩展分 **六个阶段**完成，每个阶段将上下文长度增加一定倍数，使模型能够平滑地适应更大的上下文窗口。渐进式扩展的策略避免了直接在 128K 长度上训练可能导致的注意力分布崩溃。

&emsp;&emsp;**第三阶段：高质量数据退火。** 在预训练的最后阶段，Meta 使用一个 **400 亿 token** 的小型但高质量数据集进行退火训练。退火数据集包含经过精心筛选的数学、代码、逻辑推理等高质量样本。论文报告，退火训练显著提升了 GSM8K 和 MATH 验证集的性能——例如，在 GSM8K 训练集上退火后，GSM8K 验证集的准确率出现了明显跃升。退火阶段的本质是在预训练末期将模型“聚焦”到高质量、高信息密度的数据上，使模型在保持通用能力的同时强化推理和代码等关键能力。

---

**后训练：SFT + 拒绝采样 + DPO**

&emsp;&emsp;LLaMA 3.1 的后训练流程与 InstructGPT 的 RLHF 三步框架有所不同。Meta 选择了 **SFT + 拒绝采样 + DPO** 的路线，而非传统的奖励模型 + PPO 强化学习。这一选择反映了 Meta 在“复杂度管理”上的考量：DPO 不需要单独训练奖励模型，也不需要 PPO 中的策略梯度估计和 KL 惩罚调参，流程更简洁、更可控。

&emsp;&emsp;后训练的第一步是**监督微调**。SFT 数据的来源包括人工标注和合成数据。Meta 坦承，大多数 SFT 样本由合成数据生成——对于代码任务，Meta 先用 1T token 的代码数据（85% 为代码）继续训练一个“代码专家”模型，用该专家收集高质量人工标注数据，然后对基础模型进行代码后训练。对于数学推理，Meta 采用“从人类反馈中学习”的方式：让模型生成推理过程，人类标注员判断推理是否正确，错误的推理过程被反馈给模型作为负面示例，模型据此修正自己的推理策略。

&emsp;&emsp;后训练的第二步是**拒绝采样**。对于每个提示，模型生成多个候选回答，通过奖励模型或人工评估筛选出最优回答，将筛选后的数据用于下一轮微调。Meta 的一个关键创新是：在拒绝采样阶段，从多个使用不同超参数训练的 DPO 模型中选择表现最好的模型来生成 Prompt 的回答，而非使用单一模型。这种“模型集成 + 最佳选择”的策略提升了拒绝采样数据的质量和多样性。

&emsp;&emsp;后训练的第三步是**直接偏好优化**。DPO 将奖励模型和策略模型合并为一个目标函数，直接使用偏好数据优化策略模型，避免了 PPO 中奖励模型和策略模型交替训练的复杂性。LLaMA 3.1 的 DPO 训练使用学习率 1e-5，通过多轮对齐迭代提升模型的有用性和安全性。

&emsp;&emsp;Meta 还采用了一个独特的**模型族蒸馏策略**：用旗舰 405B 模型的输出来优化 8B 和 70B 小模型。在后训练阶段，405B 模型作为“教师”，为小模型生成高质量的合成训练数据，使小模型能够学习到旗舰模型的对齐水平和推理模式。这一策略显著缩小了 8B 和 70B 模型与 405B 模型之间的性能差距。

---

**LLaMA 3 / 3.1 与 Gemini 1.5 Pro、GPT-4o 的关系及后续影响**

&emsp;&emsp;**LLaMA 3.1 与 Gemini 1.5 Pro、GPT-4o 代表了 LLM 演进中三条并行但日益交织的路线：开放权重的规模追赶、闭源模型的长上下文扩展和闭源模型的实时多模态。** LLaMA 3.1 用 405B 参数和 15 万亿 token 证明了开放权重模型可以达到闭源旗舰的性能水平；Gemini 1.5 Pro 用 100 万 token 上下文窗口和跨模态近乎完美的召回率证明了长上下文是独立于参数规模的能力维度；GPT-4o 用端到端统一架构和 232 毫秒响应延迟证明了实时多模态交互是 LLM 产品化的关键方向。

&emsp;&emsp;三者在技术路线上的差异反映了各自的核心约束。LLaMA 3.1 的约束是“开放权重”——它必须在公开数据上训练、以开放许可发布，这限制了它在数据来源和架构复杂度上的选择空间。Gemini 1.5 Pro 的约束是“长上下文”——它必须在 100 万 token 的规模上保持检索和推理的准确性，这要求它在注意力机制和位置编码上进行专门优化。GPT-4o 的约束是“实时交互”——它必须在 232 毫秒内完成多模态推理，这要求它在架构和量化上进行极致的效率优化。

&emsp;&emsp;从架构演进的脉络看，LLaMA 3.1 的贡献在于**证明了“数据 + 规模”仍然是开放权重模型追赶闭源模型的有效路径**。在 MoE、稀疏注意力、多模态统一架构等创新层出不穷的背景下，LLaMA 3.1 用最传统的 decoder-only Transformer 加上 15 万亿 token 的高质量数据和精心设计的三阶段预训练流程，实现了与 GPT-4o 和 Claude 3.5 Sonnet 接近的性能。这一结果提示了一个重要判断：对于开放权重模型而言，数据工程和训练流程的优化可能比架构创新更具杠杆效应。

&emsp;&emsp;LLaMA 3.1 的后续影响清晰可见。在模型层面，LLaMA 3.2（2024 年 9 月）将布局扩展到端侧，发布了 1B 和 3B 的小尺寸模型，以及 11B 和 90B 的多模态版本——这是 LLaMA 系列首次支持视觉理解。LLaMA 3.3（2024 年 12 月）用 70B 的参数量追平了 405B 版本的对齐水平，验证了“小模型 + 大数据 + 强后训练”可以逼近旗舰性能。在架构层面，LLaMA 4（2025 年 4 月）首次转向 MoE 架构，采用 16 专家、109B 总参数、17B 激活参数的配置，并原生支持文本、图像、音频和视频的多模态推理——这一转变标志着 Meta 在保持开放权重的同时，开始拥抱架构创新。在生态层面，LLaMA 3.1 的开放许可和 25 个合作伙伴的网络使 LLaMA 成为企业级 LLM 部署的事实标准，Hugging Face 上的 LLaMA 系列下载量在 2024 年 7 月已突破 3 亿次。

&emsp;&emsp;从更宏观的视角看，LLaMA 3 / 3.1 代表了开放权重模型与闭源模型之间竞争的一个关键转折点。在 LLaMA 1 时代，开放权重模型落后闭源旗舰约 1-2 代；在 LLaMA 2 时代，差距缩小到约 1 代；在 LLaMA 3.1 时代，差距缩小到在多数文本任务上已可忽略不计。这一追赶的代价是巨大的训练投入——15 万亿 token 的数据工程、16,000 块 H100 的训练集群、92 页的技术报告——但它证明了开放权重路线在技术上是可行的。然而，LLaMA 4 转向 MoE 和 Meta 最终终止开源策略的决定也揭示了开放权重路线的商业挑战：当训练成本达到数千万美元级别时，单纯依靠社区生态难以支撑持续的研发投入。LLaMA 3.1 所达到的性能高度，可能既是开放权重路线的巅峰，也是其可持续性面临考验的起点。

### 5.4 Qwen 2 / 2.5

- 论文地址：
  - Qwen2：[Qwen2 Technical Report](https://arxiv.org/pdf/2407.10671)
  - Qwen2.5：[Qwen2.5 Technical Report](https://arxiv.org/pdf/2412.15115)
- 模型仓库：[Qwen2](https://huggingface.co/Qwen)、[Qwen2.5](https://huggingface.co/Qwen)

&emsp;&emsp;LLaMA 3.1 用 405B 参数和 15 万亿 token 证明了开放权重模型可以达到闭源旗舰的性能水平，但其部署门槛——训练成本数千万美元、推理至少需要 4 块 H100——使大多数开发者和企业无法真正使用这一能力。开放权重模型的核心价值在于“可及性”，而 LLaMA 3.1 的旗舰版本在这一点上与闭源模型并无本质区别。与此同时，多语言能力在开源模型中长期处于被忽视的状态：LLaMA 3.1 的训练数据中约 92% 为英语，中文等非拉丁语系语言的 token 压缩率和模型性能均不理想。阿里云通义千问团队的 Qwen 2 和 Qwen 2.5 对这两个问题给出了系统性的回答：**以远小于 LLaMA 3.1 405B 的参数规模，通过更大规模的高质量训练数据、更精细的后训练流程和更全面的模型谱系，在多数基准上达到甚至超越旗舰级模型的性能，同时以 Apache 2.0 许可完全开放，为中文和全球开发者提供一个真正可及、可商用、可本地部署的顶级基座。** Qwen2.5-72B-Instruct 以 LLaMA 3.1 405B 约五分之一的参数量，在多项基准上展现出与后者相当的性能，成为 2024 年全球开源模型排行榜的新标杆。截至 2026 年，Qwen 系列累计下载量达 4000 万次，衍生模型数量达 7.8 万个，成为全球最具影响力的开源 LLM 生态之一。

---

**Qwen 2：架构沿革与数据规模跃升**

&emsp;&emsp;Qwen2 于 2024 年 6 月发布，是 Qwen 系列从 Qwen 1 到 Qwen 1.5 之后的第三代架构。其整体设计延续了 Qwen 系列对 LLaMA“标准配方”的继承与改良，但在几个关键维度上进行了系统性的增强。

&emsp;&emsp;Qwen2 提供了 **4 个稠密模型规模**（0.5B、1.5B、7B、72B）和 **1 个 MoE 模型**（总参数 57B，激活参数 14B），覆盖了从端侧部署到旗舰性能的完整频谱。所有模型均在超过 **7 万亿 token** 的高质量数据集上进行预训练。

&emsp;&emsp;在架构层面，Qwen2 沿用了 Qwen 系列的若干核心组件并进行了更新：

- **分组查询注意力（GQA）** ：Qwen2 在所有规模上均采用 GQA 替代传统多头注意力。GQA 将多个 Query 头分组共享一组 Key 和 Value 头，在推理时显著减小 KV 缓存的大小，提升吞吐量。这与 LLaMA 3 在全部规模上采用 GQA 的做法一致，反映了 GQA 已成为现代 LLM 的标配；
- **SwiGLU 激活函数**：前馈网络使用 SwiGLU 替代标准 ReLU 或 GELU，通过门控机制增强表达能力；
- **旋转位置嵌入（RoPE）** ：使用 RoPE 处理位置信息，取代可学习的绝对位置嵌入；
- **QKV 偏置**：与 LLaMA 系列“无偏置线性层”的设计不同，Qwen2 在注意力机制的 Q、K、V 投影中保留了偏置项。这一设计在 Qwen 系列中从早期版本即被采用，被认为是 Qwen 模型在部分任务上表现更稳定的原因之一；
- **RMSNorm 与 Pre-Norm**：归一化层采用 RMSNorm，并放置在子层之前（Pre-Norm），保障深层网络的训练稳定性；
- **字节级 BPE 分词器**：词表大小为 **151,643 个常规词元加 3 个控制词元**，所有规模的模型共享同一词汇表。Qwen2 的分词器对中文的压缩效率显著优于 LLaMA 系列的 128K 分词器，这是 Qwen 在中文场景中具有天然优势的技术基础。

&emsp;&emsp;Qwen2 在长上下文处理上引入了 **双块注意力（Dual Chunk Attention, DCA）** 和 **YARN**。DCA 将长序列分割为可管理的长度块，如果输入可以在一个块中处理，DCA 产生与原始注意力相同的结果；否则，DCA 在块内和跨块之间有效捕获相对位置信息。YARN 则用于重新调整注意力权重以实现更好的长度外推。这两项技术使 Qwen2 在推理时能够有效处理超出训练长度的序列，而无需在更长序列上进行继续训练。DCA 后来被应用于 Qwen2、Qwen2.5 以及 Qwen3，与 YARN 一起作为扩展模型上下文长度的标准手段。

---

**Qwen 2.5：数据规模翻倍与后训练升级**

&emsp;&emsp;Qwen2.5 于 2024 年 9 月发布，是 Qwen2 的全面升级版本。其最核心的变化是**训练数据规模从 Qwen2 的 7 万亿 token 扩展到 18 万亿 token**，数据量增加超过一倍。这一规模远超 Chinchilla 缩放规律的最优配比建议，反映了 Qwen 团队的一个重要判断：在推理成本成为主要瓶颈的场景下，继续扩大数据规模仍然有效，且比扩大参数量更具性价比。Qwen2.5-72B 的参数量仅为 LLaMA 3.1 405B 的约五分之一，但通过更大规模的数据训练，在多项基准上达到了与后者相当的性能。

&emsp;&emsp;Qwen2.5 提供了 **7 个稠密模型规模**（0.5B、1.5B、3B、7B、14B、32B、72B），覆盖了从手机端侧到数据中心旗舰的完整频谱。此外，Qwen 团队还开源了专注于编程的 **Qwen2.5-Coder**（1.5B、7B、32B）和专注于数学的 **Qwen2.5-Math**（1.5B、7B、72B），以及多模态模型 Qwen2-VL。所有开放权重的模型都是稠密的 decoder-only 语言模型，除 3B 和 72B 版本外均采用 **Apache 2.0 许可证**，完全允许商用和修改。

&emsp;&emsp;Qwen2.5-Coder 在包含 **5.5 万亿 token 编程相关数据**上进行了训练，使其即使较小的编程专用模型也能在编程评估基准上表现出与大型语言模型竞争的竞争力。Qwen2.5-Math 支持中英双语，并整合了多种推理方法，包括思维链（CoT）、程序思维（PoT）和工具集成推理（TIR）。

---

**后训练：SFT + DPO + GRPO 的多阶段对齐**

&emsp;&emsp;Qwen2.5 的后训练流程在 InstructGPT 的 RLHF 三步框架基础上进行了系统性的扩展。Qwen2.5 技术报告明确指出，其后训练包括**复杂的监督微调（SFT），使用超过 100 万个样本**，以及**多阶段的强化学习，包括离线学习 DPO 和在线学习 GRPO**。

&emsp;&emsp;**监督微调**阶段使用了超过 100 万个高质量指令样本，覆盖了知识问答、代码生成、数学推理、长文本生成、结构化数据分析和指令跟随等多种任务类型。这一规模远大于 InstructGPT 的 13,000 条 SFT 样本，反映了 Qwen 团队对指令数据规模和多样性的重视。

&emsp;&emsp;**直接偏好优化（DPO）** 被用于离线学习阶段。DPO 将奖励模型和策略模型合并为一个目标函数，直接使用偏好数据优化策略模型，避免了 PPO 中奖励模型和策略模型交替训练的复杂性。DPO 的训练稳定性优于 PPO，且不需要在线生成样本，训练效率更高。

&emsp;&emsp;**组相对策略优化（GRPO）** 被用于在线学习阶段。GRPO 是 DeepSeek 在 DeepSeek-Math 中提出的一种强化学习算法，其核心思想是对同一提示生成多个回答，通过组内相对排名来估计优势函数，而不是训练一个独立的价值网络。GRPO 在数学和代码等需要精确推理的任务上表现尤为突出，因为它能够利用可验证的奖励信号（如答案是否正确、代码是否通过测试）进行高效优化。

&emsp;&emsp;这一 **“SFT + DPO + GRPO”** 的三阶段后训练流程后来成为开源 LLM 后训练的标准范式。DeepSeek-V3、Qwen2.5-Max、以及大量社区衍生模型都采用了类似的流程。与 InstructGPT 的“SFT + RM + PPO”相比，这一新范式在保持对齐质量的同时显著降低了训练复杂度和计算成本。

---

**长上下文扩展：DCA + YARN + 1M token**

&emsp;&emsp;Qwen2.5 在长上下文能力上进行了进一步的扩展。所有 Qwen2.5 语言模型原生支持 **128K token** 的上下文长度，并能生成最多 **8K token** 的内容。这一能力得益于 Qwen2 引入的 DCA 和 YARN 技术。

&emsp;&emsp;在此基础上，Qwen 团队于 2025 年 1 月开源了 **Qwen2.5-1M 系列**，首次将开源模型的上下文长度扩展至 **100 万 token**（约 150 万字）。Qwen2.5-1M 通过在训练和推理中的创新，显著增强了长上下文能力，同时保持了短上下文性能并降低了成本。在长上下文检索任务中，Qwen2.5-7B-Instruct 和 Qwen2.5-14B-Instruct 通过 DCA 在超过 80% 的评测中表现优异。Qwen2.5-1M 的性能超越了 GPT-4o-mini，标志着长文本处理进入“全文档一次性解析”时代。

&emsp;&emsp;DCA 的一个重要特性是**无需训练即可扩展上下文长度**。它通过在推理时对注意力机制进行分块处理，使模型能够处理远超训练长度的序列，而无需在百万 token 序列上进行继续训练。这一特性使 Qwen2.5 的长上下文扩展具有极高的工程灵活性和成本效率。

---

**Qwen2.5-Max：MoE 架构的旗舰探索**

&emsp;&emsp;在开源稠密模型之外，Qwen 团队还推出了基于 MoE 架构的旗舰模型 **Qwen2.5-Max**。Qwen2.5-Max 于 2025 年 1 月 29 日发布，采用超大规模混合专家架构，**预训练数据超过 20 万亿 token**，官方公布的参数量达到 **1.8 万亿（稀疏激活）** ，动态激活参数比例控制在 **30%–40%**。模型通过动态专家分配策略优化计算资源利用率。

&emsp;&emsp;Qwen2.5-Max 在多项公开评测中表现优异。在 Chatbot Arena 盲测榜单中，Qwen2.5-Max 取得**全球第七**的排名，并在**数学和编程单项能力上排名第一**。在 MMLU-Pro、LiveCodeBench 等权威评测中，其性能与 GPT-4 和 Claude 3.5 Sonnet 相当或更优。Qwen2.5-Max 目前通过阿里云百炼平台提供 API 服务，但尚未开源。

&emsp;&emsp;Qwen2.5-Max 的发布标志着 Qwen 系列从“开源稠密模型”向“开源 + 闭源 MoE 双轨”的战略扩展。这一策略与 Mistral AI 的“开源 + 闭源混合”路线一致：通过开源模型建立生态和信任，通过闭源 MoE 模型提供顶级性能并获取商业收入。

---

**模型谱系与开放生态**

&emsp;&emsp;Qwen2.5 的模型谱系覆盖了从端侧到云端的完整频谱，这是其在开源社区获得广泛 adoption 的关键因素之一。Qwen2.5 语言模型提供 **0.5B、1.5B、3B、7B、14B、32B、72B** 七个规模，每个规模都提供基础版本、指令跟随版本和量化版本。专业领域模型包括 Qwen2.5-Coder（编程）、Qwen2.5-Math（数学）和 Qwen2-VL（多模态）。Qwen2.5 全系列总计上架**100 多个模型**。

&emsp;&emsp;Qwen2.5 支持**29 种以上语言**，包括中文、英文、法文、西班牙文、葡萄牙文、德文、意大利文、俄文、日文、韩文、越南文、泰文、阿拉伯文等。多语言能力是 Qwen 系列相对于 LLaMA 系列的显著优势——LLaMA 3.1 的训练数据中约 92% 为英语，而 Qwen2.5 在多语言基准上的表现显著优于同规模 LLaMA 模型。对于中文场景，Qwen2.5 的分词器对中文的压缩效率更高，推理速度更快，被开发者社区评价为“中文场景的版本答案”。

&emsp;&emsp;在开源生态方面，Qwen 系列的累计下载量达到 **4000 万次**，衍生模型数量达到 **7.8 万个**。这一生态规模在开源 LLM 中仅次于 LLaMA 系列，但 Qwen 的 Apache 2.0 许可（大部分模型）比 LLaMA 的社区许可更为宽松，企业用户可以无顾虑地商用和修改。NVIDIA 的 OpenCodeReasoning-Nemotron-7B 等第三方模型明确标注衍生自 Qwen2.5-7B-Instruct，反映了 Qwen 作为基座模型在企业级应用中的广泛采纳。

---

**Qwen 2 / 2.5 与 LLaMA、Mistral 的关系及后续影响**

&emsp;&emsp;**Qwen2.5 是 LLaMA 架构范式在“数据规模与训练效率”维度上的极致优化。** 它没有改变 decoder-only Transformer 的基本框架，而是通过将训练数据规模从 Qwen2 的 7T token 扩展到 18T token、将后训练样本从 13K 扩展到 100 万、引入 DPO + GRPO 的多阶段强化学习，在参数量仅为 LLaMA 3.1 405B 约五分之一的情况下实现了与后者相当甚至更优的性能。这一结果有力地验证了 Chinchilla 缩放规律的核心洞见：在给定计算预算下，数据规模与参数规模的均衡比单纯放大参数量更为重要。

&emsp;&emsp;与 LLaMA 系列和 Mistral 系列的关系可以从两个维度理解。在**架构层面**，Qwen2.5 与 LLaMA 3 共享了几乎相同的“标准配方”（GQA、SwiGLU、RoPE、RMSNorm），但在分词器（151K vs 128K）、QKV 偏置和长上下文扩展技术（DCA + YARN vs 渐进式 RoPE 缩放）上存在差异。在**生态定位层面**，Qwen2.5 的 Apache 2.0 许可比 LLaMA 3.1 的社区许可更为宽松，且模型谱系覆盖了从 0.5B 到 72B 的完整频谱，在“可及性”上具有明显优势。在**语言能力层面**，Qwen2.5 对 29 种语言的原生支持和中文场景的专门优化，使其成为中文开发者的首选基座。

&emsp;&emsp;Qwen2.5 的后续影响清晰可见。在模型层面，**Qwen2.5-Max** 验证了 MoE 架构在 1.8T 参数规模下的有效性，为 Qwen3 的 MoE 路线铺平了道路。**Qwen2.5-Coder** 和 **Qwen2.5-Math** 分别成为编程和数学领域开源模型的新标杆。**Qwen2.5-1M** 将开源模型的上下文长度推至 100 万 token，使“全文档一次性解析”成为开源社区可用的能力。在生态层面，Qwen 的 7.8 万个衍生模型覆盖了从医疗、法律到角色扮演的各个垂直领域，形成了全球最具活力的开源 LLM 生态之一。在方法论层面，Qwen2.5 的 **“SFT + DPO + GRPO”** 后训练流程成为开源 LLM 对齐的标准范式，被 DeepSeek-V3、Llama 4 等后续模型广泛采纳。

&emsp;&emsp;从更宏观的视角看，Qwen 2 / 2.5 代表了开放权重模型演进中“**以数据规模与训练效率追赶参数规模**”这一路线的成功。LLaMA 3.1 用 405B 参数证明了开放权重模型可以达到闭源旗舰的性能水平，但代价是极高的训练成本和部署门槛。Qwen2.5 用 72B 参数和 18T token 的数据，在多数文本任务上达到了与 LLaMA 3.1 405B 相当的性能，同时以 Apache 2.0 许可完全开放。这一结果的意义不仅在于技术性能，更在于**可及性**：一个 72B 模型可以在 4 块 A100 80GB 上以 int8 量化运行，而 405B 模型至少需要 8 块。当顶级能力从“只有少数机构可及”变为“大多数开发者可及”时，创新的门槛被大幅降低，生态的活力被显著激发。Qwen 系列 7.8 万个衍生模型的生态规模，正是这一可及性所带来的直接结果。

## 6 推理与系统工程
### 6.1 OpenAI o1 / o3 系列

- 论文地址：
  - o1：[Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
  - o3：[OpenAI o3 and o4-mini System Card](https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf)
- 官方公告：[Introducing OpenAI o1](https://openai.com/index/introducing-openai-o1-preview/)、[Introducing OpenAI o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)

&emsp;&emsp;LLaMA 3.1 和 Qwen 2.5 代表了“预训练 + 后训练”范式的巅峰：15 万亿 token 的预训练数据、100 万条 SFT 样本、DPO + GRPO 的多阶段强化学习，最终在文本任务上追平了闭源旗舰。但这一范式的根本约束始终存在：**模型在推理时的计算量是固定的——给定一个提示，模型以相同的计算量生成回答，无论问题简单还是复杂。** 一个三位数加法和一个需要多步逻辑推导的数学证明，消耗的计算资源相同。这导致模型在需要深度推理的任务上表现不佳：GPT-4o 在 AIME 2024 数学竞赛上的准确率仅为 13.4%，在 Codeforces 编程竞赛上的准确率约为 11%。这些失败并非知识不足——GPT-4o 知道数学定理和编程语法——而是**推理过程的深度不足**：模型没有足够的“思考时间”来探索多条解题路径、验证中间假设、并在发现错误后回溯修正。OpenAI 的 o1 和 o3 系列对此给出了一个根本性的回答：**将推理时的计算量从固定值变为可调变量，让模型在生成最终答案之前，先生成一条内部的、隐藏的“思维链”（Chain of Thought, CoT），在思维链中探索、验证和修正推理过程，然后仅将结论呈现给用户。** 这一范式的转变——从“训练时 Scaling”扩展到“推理时 Scaling”——使同一基础模型在推理密集型任务上的性能可以被“解锁”到远超其训练规模预期的水平。o1 在 AIME 2024 上的准确率从 GPT-4o 的 13.4% 跃升至 83.3%，提升超过 6 倍。o3 进一步在 ARC-AGI 基准上以高计算模式达到 87.5%，首次突破人类水平门槛（85%）。

---

**核心思想：用强化学习训练模型“思考”**

&emsp;&emsp;o1 系列最根本的创新不在于架构，而在于**训练目标**。GPT 系列和 LLaMA 系列使用自回归语言建模作为预训练目标，通过预测下一个 token 来学习语言和知识。SFT 和 RLHF 则调整模型的行为倾向，使其输出更符合人类偏好。但这两阶段都**没有显式地训练模型进行多步推理**——模型学到的是“给定前文，下一个 token 最可能是什么”，而不是“给定问题，通过怎样的推理步骤可以到达正确答案”。

&emsp;&emsp;o1 的训练流程引入了**大规模强化学习**，奖励信号来自**可验证的正确性**而非人类偏好。在数学问题上，奖励信号是答案是否正确；在编程问题上，奖励信号是代码是否通过测试用例；在逻辑推理问题上，奖励信号是结论是否与前提一致。OpenAI 的技术报告明确指出，o1 是“通过大规模强化学习训练的，使用思维链进行推理”。这与 InstructGPT 的 RLHF 有本质区别：RLHF 的奖励模型学习的是“人类更喜欢哪个回答”，而 o1 的奖励信号是“哪个回答在客观上是正确的”。可验证的奖励信号消除了奖励黑客的风险——模型无法通过编造看似合理的错误答案来获得高分。

&emsp;&emsp;o1 的强化学习训练产生了一个**涌现行为**：模型自发地学会了在思维链中生成自我纠正、假设检验和回溯的策略。在 OpenAI 公布的示例中，o1 在推理过程中会明确地说出“Wait, that can't be right because...”或“Let me try a different approach...”，然后放弃错误路径并尝试新的方法。这种行为不是通过监督学习直接教出来的——它是在强化学习过程中，模型发现“自我纠正”和“假设检验”能够提高获得正确答案的概率后，自发形成的策略。这与 AlphaGo 在自我对弈中发现人类从未使用过的围棋策略在机制上高度相似：强化学习让模型在奖励信号引导下自主探索最优策略，而非模仿人类示范。

---

**隐藏思维链：训练时生成，推理时隐藏**

&emsp;&emsp;o1 的思维链有一个关键特征：**它是隐藏的**。用户看到的是模型最终输出的简洁答案，而非思维链本身。OpenAI 在技术报告中解释了这一选择的原因：隐藏思维链使 OpenAI 能够对思维链内容进行安全审查和过滤，而不影响最终输出的呈现。如果思维链是可见的，模型可能被迫在思维链中生成符合安全规范的内容，从而损害推理的自由度和深度。

&emsp;&emsp;隐藏思维链的另一层考量是**竞争壁垒**。OpenAI 明确表示，o1 的思维链是“一项重要的知识产权”，不公开是为了防止竞争对手通过分析思维链来逆向工程 o1 的训练方法。这与 OpenAI 在 GPT-4 技术报告中拒绝披露架构细节的逻辑一脉相承。然而，这一选择也引发了学术界的争议：思维链是理解推理模型行为的关键窗口，隐藏思维链使研究者无法分析模型在推理过程中的失败模式、偏见传播或安全漏洞。

&emsp;&emsp;从训练的角度看，隐藏思维链与**推理时计算扩展**（Test-Time Compute Scaling）紧密相关。o1 的思维链长度不是固定的——在训练过程中，模型学会根据问题的难度调整思维链的长度。简单的问题只需要几步推理，复杂的问题可以生成数千个推理 token。OpenAI 的研究显示，推理时计算的增加可以持续提升性能：在 AIME 2024 上，o1 的单样本贪婪解码达到 74%，64 样本多数投票达到 83%，1000 样本重排序达到 93%。这意味着同一基础模型的性能可以通过在推理时投入更多计算来“解锁”，而不需要重新训练。

---

**o3 系列：推理能力的进一步跃迁**

&emsp;&emsp;o3 于 2024 年 12 月首次公布，2025 年 4 月 16 日正式发布，是 o1 的直接后继者。o3 在 o1 的基础上进行了三项关键扩展：**推理时计算的可调性**、**工具使用的全面集成**和**多模态推理**。

&emsp;&emsp;**推理时计算的可调性**是 o3 最核心的工程创新。o3 引入了 `reasoning_effort` 参数，允许开发者按请求控制模型的“思考深度”，支持低、中、高三个计算层级。在 ARC-AGI 基准上，低计算层级得分 75.7%，高计算层级达到 87.5%——后者是 AI 系统首次在该基准上突破人类水平门槛（85%），而 o1 在相同基准上的得分仅在 25% 到 32% 之间。这一设计使 o3 能够根据任务的复杂度动态分配计算资源：简单查询使用低计算层级以降低延迟和成本，复杂推理任务使用高计算层级以最大化准确性。

&emsp;&emsp;**工具使用的全面集成**是 o3 相对于 o1 的另一个关键升级。o3 原生支持网页浏览、Python 代码执行、图像和文件分析、图像生成以及 Canvas 和自动化功能。o3 的思维链不仅包含内部推理，还可以在推理过程中**主动调用工具**——例如，在解决数学问题时执行 Python 代码进行数值验证，或在分析数据时运行统计计算。这一能力使 o3 从“纯文本推理模型”转变为“推理 + 工具调用的通用问题求解器”。

&emsp;&emsp;在架构层面，o3 的核心创新是将 **Q* 算法与蒙特卡洛树搜索（MCTS）深度结合**，形成“预测-验证-优化”的闭环推理系统。与 o1 的线性思维链不同，o3 的推理过程更像一棵搜索树：模型在推理过程中探索多条候选路径，通过奖励信号评估每条路径的潜力，然后选择最有希望的路径深入展开。在离线训练阶段，o3 使用 PPO 算法在合成数据环境中进行数亿次策略迭代，优化动作价值函数 Q(s,a) 的估计精度；在线推理阶段，MCTS 动态扩展决策树，每步推理调用约 500 次神经网络计算，显著超越 GPT-4 的 32 次调用上限。

&emsp;&emsp;**o3-mini** 是 o3 的轻量版本，于 2025 年 1 月 31 日发布，采用动态参数激活技术，核心参数规模控制在 130 亿至 270 亿之间，通过分组查询注意力（GQA）和稀疏激活策略，推理效率接近千亿参数模型。o3-mini 的响应速度比 o1-mini 快 24%，平均响应时间从 10.16 秒降至 7.7 秒。o3-mini 是 OpenAI 首个向免费用户开放的推理模型，以“免费 + 轻量化”为核心定位，直接瞄准中小开发者和资源有限型企业的需求。

---

**o1 / o3 与 GPT-4o、LLaMA 3.1 的关系及后续影响**

&emsp;&emsp;**o1 和 o3 代表了 LLM 演进中一个全新的维度：推理时计算扩展（Test-Time Compute Scaling）。** GPT-4o、LLaMA 3.1、Qwen 2.5 等模型的核心优化方向是**训练时效率**——在给定计算预算下，如何通过数据规模、架构选择和训练流程优化来最大化模型能力。o1 和 o3 的核心优化方向是**推理时效率**——在给定训练成本下，如何通过增加推理时的计算量来“解锁”更深层的推理能力。OpenAI 研究员 Noam Brown 在 NeurIPS 2024 上给出了一个引人注目的对比：“让模型在一手牌中思考 20 秒，获得的提升相当于将模型规模和训练扩大 100,000 倍”。这一发现并不意味着预训练 Scaling Law 的终结，而是扩展了 Scaling Law 的维度——从单一的训练时计算扩展到了“训练时 + 推理时”的联合优化。

&emsp;&emsp;o1 和 o3 的后续影响是多层面的。**在方法论层面**，o1 的隐藏思维链 + 可验证奖励的强化学习范式被 DeepSeek-R1 以开源方式复制并超越。DeepSeek-R1 于 2025 年 1 月发布，核心创新在于**纯强化学习训练**：不依赖人类标注的推理轨迹，仅通过可验证奖励（如数学证明检查器、代码执行器）的反馈信号，让模型自发地发展出长链推理能力。R1 在训练过程中自发出现了可识别的推理策略——自我纠正、假设检验、验证——这与 o1 的涌现行为在机制上高度一致。R1 的开源发布使推理模型的能力从闭源实验室走向了整个研究社区。

&emsp;&emsp;**在产品层面**，o3 的 `reasoning_effort` 参数和工具集成使推理模型从“专用数学/编程工具”转变为“通用问题求解器”。o3 原生支持网页浏览、代码执行、图像分析和文件处理，使其能够胜任需要多步推理和工具调用的复杂任务。这一能力使 o3 成为 OpenAI Operator 智能体的底层模型，从 GPT-4o 升级至 o3 后，智能体在浏览器操作、任务分解和动态策略调整上的表现显著提升。

&emsp;&emsp;**在范式层面**，o1 和 o3 确立了一个新的 Scaling Law 维度。从 Kaplan Scaling（2020-2022，优先扩大参数）到 Chinchilla-optimal（2022-2024，参数与数据均衡增长），再到 Inference-optimal（2024至今，小模型 + 超量数据），最终到 **Test-time compute**（2024至今，推理时“思考更久”），LLM 的缩放范式从单一的训练规模维度逐步扩展到了训练效率、数据效率和推理计算的多维度优化。o1 和 o3 正是这一范式扩展的起点和验证。

&emsp;&emsp;**在安全层面**，o1 和 o3 带来了新的对齐挑战。由于思维链是隐藏的，安全审查只能在最终输出层面进行，无法检测模型在推理过程中是否探索了不当的路径。OpenAI 在 o3 的 System Card 中报告了多项安全评估结果，包括对 CBRN（化学、生物、放射和核）风险、网络攻击能力和欺骗性行为的系统测试。与此同时，推理模型的可解释性成为一个新的研究方向：如果模型的推理过程是不可见的，如何确保它“诚实地思考”？这一问题在 o3 的后续版本（o3-pro、o4-mini）以及 GPT-5 的推理模式中持续被探索。

&emsp;&emsp;从更宏观的视角看，o1 和 o3 系列标志着 LLM 从“快速反应系统”向“深度推理系统”的转变。GPT-3 到 GPT-4o 的路线追求的是“更快、更便宜、更多模态”——让模型在尽可能短的时间内生成尽可能好的回答。o1 和 o3 的路线追求的是“思考更久、推理更深、答案更可靠”——让模型在面对复杂问题时，愿意花更多时间探索、验证和修正。这两种路线并非替代关系，而是互补：GPT-4o 适合实时交互、内容生成和快速问答；o1 和 o3 适合数学证明、代码调试、科学推理和战略规划。OpenAI 研究副总裁 Jerry Tworek 在 2025 年的一次播客采访中表示，“在某种程度上，GPT-5 可以被视作 o3.1”。这一表述暗示了推理能力正在从“专用模型”向“通用模型的基础能力”演进——未来的 LLM 将同时具备快速反应和深度推理的能力，并根据任务的复杂度动态切换模式。

### 6.2 DeepSeek-V3 / R1

- 论文地址：
  - DeepSeek-V3：[DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)
  - DeepSeek-R1：[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948)（发表于 Nature，vol. 645, pp. 633–638, 2025）
- 模型仓库：[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)、[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)

&emsp;&emsp;o1 和 o3 证明了推理时计算扩展的有效性，但 OpenAI 的隐藏思维链、闭源权重和不透明的训练细节，使这一范式的可复现性和可及性受到根本限制。与此同时，GPT-4 级别的 MoE 架构和 o1 级别的推理能力，长期被视为需要数万张 H100 和数千万美元训练成本才能获得的“大厂专属品”。DeepSeek 的 V3 和 R1 对这两个假设同时发起了挑战。**DeepSeek-V3 用 671B 总参数、37B 激活参数的 MoE 架构，在仅 2048 张 H800 GPU 上以约 557.6 万美元的成本完成了训练，在多项基准上追平甚至超越 GPT-4o 和 Claude 3.5 Sonnet。DeepSeek-R1 则在此基础上，用纯强化学习（不依赖任何人类标注的推理轨迹）激励出长链推理能力，在 AIME 2024 上达到 79.8% 的准确率，性能比肩 OpenAI o1 正式版，并以 MIT 许可证完全开源权重，同时蒸馏出六个小模型将推理能力下沉到 1.5B 到 70B 的参数量范围。** DeepSeek 的工作标志着开源模型首次在“通用能力 + 推理能力 + 训练效率”三个维度上同时达到闭源旗舰水平，其影响远超技术本身——它证明了在算力约束下，通过架构和算法的系统性创新，同样可以实现顶级 LLM 能力。DeepSeek 的主要贡献包括：

1. DeepSeek-V3 采用 **MLA + DeepSeekMoE** 架构，首创**无辅助损失的负载均衡策略**和**多 token 预测训练目标**，在 14.8 万亿 token 上预训练，训练全程无损失尖峰、无回滚；
2. DeepSeek-V3 首次在超大规模模型上验证了 **FP8 混合精度训练**框架的有效性，训练仅需 2.788M H800 GPU 小时，成本约 557.6 万美元，训练成本约为 GPT-4 的 1/10；
3. DeepSeek-R1 提出**纯强化学习推理训练范式**，在 DeepSeek-V3-Base 上仅使用基于规则的准确性奖励和格式奖励进行 GRPO 训练，不依赖人类标注的推理轨迹，模型自发涌现出自我反思、验证和动态策略适应等高级推理模式；
4. DeepSeek-R1 在 AIME 2024 上达到 79.8%，MATH-500 达到 97.3%，在数学、编程和 STEM 领域的研究生水平任务上性能比肩 OpenAI o1 正式版；
5. DeepSeek-R1 以 **MIT 许可证**完全开源，同时通过 R1 蒸馏出基于 Qwen2.5 和 Llama 3 的六个小模型（1.5B 至 70B），其中 32B 和 70B 模型在多项能力上对标 OpenAI o1-mini。

---

**DeepSeek-V3：低成本 MoE 架构的工程突破**

&emsp;&emsp;DeepSeek-V3 于 2024 年 12 月 26 日发布，是一个基于混合专家（MoE）架构的 decoder-only 语言模型。其核心架构设计围绕两个目标展开：**推理效率最大化**和**训练成本最小化**。

&emsp;&emsp;在注意力层面，DeepSeek-V3 采用 **多头潜在注意力（Multi-head Latent Attention, MLA）** ，这是 DeepSeek 系列的核心架构创新。传统多头注意力在推理时需要缓存完整的 Key 和 Value 矩阵，KV 缓存大小随序列长度线性增长，长序列推理的显存占用迅速膨胀。MLA 通过低秩压缩策略解决了这一问题：将 Key 和 Value 张量先投影到一个紧凑的潜在空间，再投影回高维空间。存储的不再是完整的 KV 矩阵，而是一个低维的中间态。这一设计受低秩适配启发，在保留模型捕捉丰富上下文关系能力的同时，大幅降低了 KV 缓存的内存占用。MLA 还与旋转位置嵌入（RoPE）集成，在压缩 KV 的同时保持位置信息的完整性。

&emsp;&emsp;在 FFN 层面，DeepSeek-V3 采用 **DeepSeekMoE** 架构，将标准前馈网络替换为稀疏专家混合层。与 Mixtral 8x7B 的粗粒度专家设计不同，DeepSeekMoE 采用**细粒度专家**和**共享专家**的组合策略。每个 token 被路由到多个小型专家，而非少数大型专家，使专家分配更加精细。同时，部分专家被设为“共享专家”，所有 token 都必须经过这些专家处理，确保基础能力的稳定传递。

&emsp;&emsp;DeepSeek-V3 在 MoE 训练上引入了一项关键创新：**无辅助损失的负载均衡策略**。传统 MoE 训练依赖辅助损失来鼓励专家负载均衡，但辅助损失会干扰语言建模主目标，导致性能退化。DeepSeek-V3 完全去掉了辅助损失，改用一种基于动态偏置项的均衡策略，在几乎不影响主目标的前提下实现专家负载的动态均衡。论文报告这一策略在训练全程保持了专家负载的稳定分布，且未产生任何性能干扰。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/deepseekv3_Snipaste_2026-09-15_08-51-19.png)

&emsp;&emsp;**多 token 预测**是 DeepSeek-V3 的另一个训练目标创新。传统语言模型在每个位置只预测下一个 token，而 DeepSeek-V3 要求模型在每个位置同时预测多个未来 token。这一目标通过完整的因果链增强了训练信号的密度，使模型能够更好地进行规划和推理。消融实验表明，MTP 策略持续提升了模型在大多数评估基准上的性能。

&emsp;&emsp;在训练精度上，DeepSeek-V3 首次在超大规模模型上验证了 **FP8 混合精度训练**的有效性。通过精细的量化策略，模型在 FP8 精度下进行前向和反向计算，同时以 BF16 格式存储低精度优化器状态，显著提升了训练速度并降低了 GPU 内存占用。这一技术是 DeepSeek-V3 能够以 2.788M H800 GPU 小时完成 671B 模型训练的关键因素之一。

&emsp;&emsp;DeepSeek-V3 在 14.8 万亿 token 上完成了预训练。训练过程异常稳定，全程未经历任何不可恢复的损失尖峰，也未进行任何回滚操作。对于一个 671B 参数的 MoE 模型而言，这种级别的训练稳定性在大规模训练中极为罕见。

---

**DeepSeek-R1：纯强化学习激励推理能力**

&emsp;&emsp;DeepSeek-R1 的核心创新在于回答了一个此前被认为需要大量人类标注才能解决的问题：**能否在不依赖任何人类标注推理轨迹的情况下，仅通过强化学习让模型自发学会长链推理？** o1 的训练细节从未公开，但业界普遍推测其依赖了大量人类标注的思维链数据。DeepSeek-R1 的答案是肯定的，且这一结论被发表在 Nature 上，经过了严格的同行评审。

&emsp;&emsp;DeepSeek-R1 的训练建立在 DeepSeek-V3-Base 之上，使用了 **组相对策略优化（Group Relative Policy Optimization, GRPO）** 作为强化学习框架。GRPO 的核心思想是：对于每个问题，从旧策略中采样一组输出（例如 16 个），计算组内每个输出的奖励，然后用组内均值和标准差对奖励进行归一化，得到每个输出的优势值。与 PPO 不同，GRPO **不需要训练独立的价值网络（critic）** ，而是用组内相对比较来估计优势。在推理任务中，模型输出可以长达数千个 token，训练一个能准确评估每一步推理质量的 critic 网络不仅内存开销巨大，而且极难稳定。GRPO 的组内相对比较方法消除了这一瓶颈，使强化学习能够在推理规模上高效运行。

&emsp;&emsp;DeepSeek-R1 的奖励设计是其成功的关键。R1-Zero 版本（纯 RL 版本）**完全使用基于规则的奖励**，不涉及任何神经奖励模型。奖励由两部分组成：**准确性奖励**和**格式奖励**。准确性奖励评估最终答案是否正确——在数学问题上，模型需要将最终答案以指定格式（如放在 \boxed{} 中）输出，从而可以可靠地通过规则进行验证；在代码竞赛问题上，使用编译器对模型输出进行预定义测试用例的评估，生成客观的正确性反馈。格式奖励则要求模型将推理过程封装在 `<think>` 和 `</think>` 标签中，确保思维过程被明确标注，便于后续分析和安全审查。

&emsp;&emsp;这一奖励设计的核心优势在于**不可欺骗性**。准确性奖励来自确定性的规则验证——数学答案要么正确要么错误，代码要么通过测试要么不通过。模型无法通过编造看似合理的推理过程来获得高分，因为最终答案的验证是独立于推理过程的。这与 RLHF 中基于人类偏好的奖励模型有本质区别：人类偏好是模糊的、可被操纵的，而规则验证是精确的、不可欺骗的。DeepSeek 在论文中明确指出，他们**刻意避免在推理任务上使用神经奖励模型**（无论是基于结果的还是基于过程的），因为神经奖励模型在推理规模下容易遭受奖励黑客攻击，且重新训练奖励模型需要额外的计算资源并增加训练管线的复杂性。

&emsp;&emsp;R1-Zero 的训练过程完全**跳过了监督微调阶段**，直接在基础模型上进行强化学习。这一设计选择源于一个明确的假设：**人类定义的推理模式可能限制模型的探索空间**，而无约束的强化学习能够更好地激励 LLM 涌现出新的推理能力。训练过程中，R1-Zero 自发地发展出了**自我反思、验证和动态策略适应**等高级推理模式。模型在处理数学和编程问题时，倾向于生成更长的回答，在其中进行验证、反思和替代方案的探索。这些行为不是通过监督学习教出来的，而是在强化学习过程中，模型发现这些策略能够提高获得正确答案的概率后自发形成的。

&emsp;&emsp;然而，R1-Zero 也存在可读性差和语言混用等问题。为此，DeepSeek 在 R1 正式版的训练中引入了**多阶段训练流程**。R1 的训练包含四个阶段：**冷启动 SFT**（使用数千条包含思维链的示例数据进行监督微调）、**面向推理的强化学习**（在冷启动模型上应用与 R1-Zero 相同的 RL 流程）、**拒绝采样与 SFT**（用 RL 模型生成推理轨迹，筛选后用于第二轮 SFT）、**全场景强化学习**（在推理数据、通用数据和安全性数据上联合进行 RL，引入语言一致性奖励解决语言混用问题）。

&emsp;&emsp;R1 的最终性能在多个推理基准上达到了与 OpenAI o1 正式版相当的水平。在 AIME 2024 上，R1 的 pass@1 准确率达到 **79.8%**（R1-Zero 为 71.0%），MATH-500 达到 **97.3%**（R1-Zero 为 95.9%）。在 Codeforces 编程竞赛中，R1 的 rating 达到 2029，超过了 96.3% 的人类参赛者。在 GPQA Diamond 博士级科学基准上，R1 的准确率与 o1 相当。

---

**开源策略与生态影响**

&emsp;&emsp;DeepSeek-R1 以 **MIT 许可证**完全开源，这是最宽松的开源许可证之一，允许自由使用、修改、分发和商用，不设任何用户规模或用途限制。DeepSeek 不仅开源了 R1 的权重，还完整公开了训练技术细节，包括 GRPO 的奖励设计、多阶段训练流程和冷启动数据的构建方法。论文中披露的细节程度远超 OpenAI 的 o1 技术报告——后者仅提供了性能数据和有限的训练方法描述，而 DeepSeek-R1 的论文在 Nature 上经过了同行评审，提供了可复现的训练配方。

&emsp;&emsp;DeepSeek 同时发布了**六个蒸馏模型**，基于 Qwen2.5 和 Llama 3 系列，参数量分别为 1.5B、7B、8B、14B、32B 和 70B。这些蒸馏模型使用 DeepSeek-R1 生成的推理数据对小模型进行微调，将 R1 的推理能力迁移到了参数量小得多的架构上。其中 **32B 和 70B 模型在多项能力上实现了对标 OpenAI o1-mini 的效果**。DeepSeek-R1-Distill-Qwen-32B 在 AIME 2024 上达到 72.6%，DeepSeek-R1-Distill-Qwen-7B 达到 55.5%，1.5B 版本也达到 28.9%。这一蒸馏策略的意义在于：它将此前需要数百 B 参数才能获得的推理能力，下沉到了消费级 GPU 可以运行的规模，使推理能力的可及性发生了根本性变化。

&emsp;&emsp;DeepSeek-V3 的训练成本披露对整个行业产生了巨大冲击。**557.6 万美元的训练成本**仅为 GPT-4 估算训练成本的约 **1/10**，且仅使用 2048 张 H800 GPU 耗时不到两个月完成。这一数字并非“小规模实验”的成本，而是一个在多项基准上追平 GPT-4o 的 671B 参数模型的完整训练成本。DeepSeek 通过 MLA 降低推理成本、通过 DeepSeekMoE 降低训练计算量、通过 FP8 训练降低内存和计算开销、通过无辅助损失负载均衡消除性能干扰——这些优化叠加在一起，使训练成本降低了近一个数量级。这一结果引发了全球 AI 行业的广泛关注和讨论：**顶级 LLM 的训练门槛是否被高估了？** 如果架构和算法的系统性优化可以将训练成本降低 10 倍，那么“只有大厂才能训练顶级模型”的假设就不再成立。

---

**局限**

&emsp;&emsp;DeepSeek-V3 和 R1 的局限性首先体现在**推理的稳定性**上。R1-Zero 的纯 RL 训练虽然成功涌现了推理能力，但模型输出的可读性较差，存在语言混用和格式不稳定的问题。R1 正式版通过多阶段训练解决了这些问题，但论文也指出，R1 在通用任务（如创意写作、事实问答）上的表现不如其推理任务，强化学习对推理能力的聚焦可能在某种程度上牺牲了其他维度的能力。

&emsp;&emsp;**奖励设计的适用范围**是另一个值得关注的问题。R1 的基于规则的奖励设计高度依赖于可验证的正确性——数学答案可以精确验证，代码可以通过测试用例。但对于**无法通过规则验证的推理任务**（如法律论证、伦理推理、战略规划），准确性奖励无法直接应用，格式奖励也无法保证推理的质量。R1 的“纯 RL”范式在可验证任务上非常成功，但在需要主观判断或模糊推理的领域，其适用性仍然有限。

&emsp;&emsp;**训练细节的透明度**虽然远高于 OpenAI 的 o1，但仍存在一些未完全披露的环节。DeepSeek-R1 的冷启动数据构建方法、拒绝采样阶段的筛选标准、以及全场景 RL 阶段的具体数据配比，在论文中虽然有描述，但未提供完整的数据集或可复现的配方。社区中已有多个复现工作（如 Open-R1、Simple-RL）尝试重现 R1 的训练流程，取得了部分成功，但尚未完全达到 R1 的最终性能。

---

**DeepSeek-V3 / R1 与 o1/o3、LLaMA、Qwen 的关系及后续影响**

&emsp;&emsp;**DeepSeek-V3 和 R1 代表了 LLM 演进中“效率优先”路线的最激进实践，以及“推理能力民主化”的关键转折。** 与 o1/o3 的对比最为鲜明：OpenAI 的推理模型依赖隐藏思维链、闭源权重和推测中的大量人类标注数据；DeepSeek 的推理模型采用公开的思维链格式、MIT 开源权重和纯强化学习训练，不依赖任何人类标注的推理轨迹。两者的最终性能相当，但 DeepSeek 的方案在**可复现性、可及性和成本**上具有根本优势。o1 的隐藏思维链是一种竞争壁垒，而 R1 的公开思维链是一种知识共享——它使整个研究社区能够分析、改进和超越 R1 的推理方法。

&emsp;&emsp;与 LLaMA 3.1 的对比同样有启发性。LLaMA 3.1 用 405B 参数、15T token 和 16,000 张 H100 的训练规模证明了开放权重模型可以达到闭源旗舰水平，但代价是数千万美元的训练成本和至少 8 块 H100 的部署门槛。DeepSeek-V3 用 671B 总参数、37B 激活参数、14.8T token 和 2,048 张 H800 的训练规模达到了类似水平的性能，训练成本仅 557 万美元。两者的核心差异在于**激活参数效率**：LLaMA 3.1 的 405B 全部参数在推理时都需要参与计算，而 DeepSeek-V3 仅激活 37B 参数，推理成本降低了约 10 倍。

&emsp;&emsp;与 Qwen 2.5 的关系则体现了**不同效率路线的互补**。Qwen2.5-72B 用 72B 稠密参数和 18T token 达到了与 LLaMA 3.1 405B 相当的性能，走的是“小模型 + 大数据 + 强后训练”的稠密路线。DeepSeek-V3 用 671B 总参数和 37B 激活参数走的是“稀疏激活 + 细粒度 MoE + 无辅助损失负载均衡”的 MoE 路线。两者都以 Apache 2.0 或 MIT 许可证开源，但 DeepSeek 的 MoE 架构在推理效率上具有结构性优势——在同等总参数量下，MoE 的激活参数量远小于稠密模型，推理成本更低。

&emsp;&emsp;DeepSeek 的后续影响清晰可见。**在方法论层面**，R1 的纯 RL 推理训练范式被大量后续工作采用和扩展。GRPO 算法成为推理模型训练的标准选择，Qwen 的 QwQ 系列、OpenAI 的 o 系列（据推测）以及大量开源推理模型都采用了类似的组内相对比较方法。**在生态层面**，R1 的 MIT 许可证和蒸馏模型策略使推理能力的门槛大幅降低——一个 7B 的蒸馏模型可以在消费级 GPU 上运行，且具备相当水平的数学和代码推理能力。**在产业层面**，DeepSeek-V3 的低成本训练披露引发了全球 AI 行业对“训练成本效率”的重新审视。微软、Google、Meta 等公司随后在模型效率优化上投入了更多资源，而 DeepSeek 的 MLA、无辅助损失负载均衡和 FP8 训练技术被广泛分析和借鉴。

&emsp;&emsp;从更宏观的视角看，DeepSeek-V3 和 R1 代表了 LLM 演进中一个被长期低估的维度：**算法效率的杠杆效应**。在 GPT 系列和 LLaMA 系列主导的“规模叙事”中，性能提升往往被归因于更大的参数量、更多的数据、更强的算力。DeepSeek 的工作提醒了整个领域：同样的性能可以用不到 1/10 的成本获得。MLA 将推理时的 KV 缓存压缩到极低水平，无辅助损失负载均衡消除了 MoE 训练中的性能干扰，FP8 训练将内存和计算开销减半，GRPO 消除了推理模型训练中对 critic 网络的需求——这些优化的叠加效应，使 DeepSeek-V3 的训练成本仅为 GPT-4 的约 1/10，而 DeepSeek-R1 的推理训练成本远低于 o1。当训练成本从“数千万美元”降至“数百万美元”时，顶级 LLM 的开发门槛发生了质的变化。DeepSeek 的 2048 张 H800 集群，在算力规模上约为 LLaMA 3.1 所用 16,000 张 H100 集群的 1/8，但训练出的模型在推理能力上达到了相当水平。这一结果的意义不仅在于技术性能，更在于它证明了**在算力约束下，架构和算法的创新可以部分替代算力投入**。对于资源有限的研究机构和中小企业而言，这一发现具有根本性的启示：顶级 LLM 能力并非只有“堆算力”一条路。

## 7 大模型时代
### 7.1 GLM-5系列

- 论文地址：[GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/pdf/2602.15763)

&emsp;&emsp;GLM-4.5 验证了将智能体、推理与编程（ARC）能力融合进单一 MoE 架构的可行性，但当模型真正投入到复杂的软件工程、长周期多轮对话的真实业务中时，算力成本和真实环境适应性成为核心瓶颈。GLM-5 的核心目标是将编程范式从“Vibe Coding”（氛围编程，即程序员手动提示 AI 生成代码）转向“Agentic Engineering”（智能体工程），要求 AI 不再只是辅助工具，而是一个可以自主规划、执行、迭代的“虚拟工程师”。GLM-5 的主要贡献包括：

1. 参数规模从 GLM-4.5 的 355B（激活 32B）扩展至 **744B（激活 40B）** ，预训练数据量从 23T 提升至 **28.5 万亿 token**，构建 78 层隐藏层，集成 **256 个专家模块，每次激活 8 个**，稀疏度 5.9%，上下文窗口最高支持 **202K token**；
2. 引入 **DeepSeek 同款稀疏注意力（DSA）** ，通过动态细粒度选择机制替换传统密集注意力，将注意力计算复杂度从 O(L²) 降至 O(L·k)，在 k=2048、上下文长度 L=128K 时计算量减少约 98%；
3. 构建全新的**异步强化学习基础设施**，将生成和训练解耦，加上独创的异步智能体 RL 算法，使训练效率大幅提升；
4. 完成与**华为昇腾、摩尔线程、海光、寒武纪、昆仑芯、沐曦以及燧原**等国产芯片的全栈适配；
5. 在实际测试中，GLM-5 可以自主连续运行代码超过 **24 小时**、执行 **700 次工具调用**、**800 次上下文切换**，从零构建一个可运行的 GBA 模拟器。

**DSA 稀疏注意力：动态筛选高价值 Token**

&emsp;&emsp;GLM-5 最核心的架构创新是引入 **DeepSeek 稀疏注意力（DSA）** 机制。传统 Transformer 的密集注意力计算复杂度随上下文长度呈平方级（O(N²）增长），当上下文窗口扩展至 200K 甚至更长时，计算成本变得极其昂贵，成为限制智能体处理复杂任务的主要瓶颈。

&emsp;&emsp;DSA 的核心思想是“先筛选、后计算”：首先由**闪电索引器（Lightning Indexer）** 快速评估查询 token 与历史 token 的相关性并打分，然后仅选择得分最高的 Top-k 个 token 进行完整的注意力计算。这一机制将注意力计算复杂度从传统的 O(L²) 降至 O(L·k)，其中 k 远小于 L。在 k=2048、上下文长度 L=128K 时，计算量减少约 **98%**。在 GLM-5 的 744B 参数模型中，DSA 被用于处理长达 200K 的上下文，将注意力计算量降低了 **1.5–2 倍**。实际部署中，在 H800 GPU 上处理长文本时，DSA 能降低约 **40% 至 50% 的推理成本**，而核心任务性能损失小于 1%。

&emsp;&emsp;直接训练基于 DSA 的超大模型存在梯度爆炸或模型崩塌的风险，因此 GLM-5 团队采取了**两阶段持续预训练策略**：首先在**稠密预热阶段**使用相对稠密的注意力机制（类似于 MLA 的变体），让模型先看全所有信息，建立起全局稳固的语义表征能力；然后在**稀疏切换阶段**逐步过渡到 DSA 模式，通过平滑过渡在提升计算效率的同时保持模型性能不退化。

**异步智能体强化学习：解耦生成与训练**

&emsp;&emsp;GLM-5 的第二个核心创新是**异步强化学习基础设施**。传统的智能体 RL 训练中，模型需要在环境中执行动作、收集轨迹、计算奖励、更新参数，整个流程是串行的——生成和训练不能同时进行，训练效率受限于环境交互的速度。GLM-5 通过将生成和训练**完全解耦**，使模型能够在生成新的智能体轨迹的同时，利用之前收集的轨迹进行参数更新。配合独创的**异步智能体 RL 算法**，训练效率大幅提升，使 GLM-5 能够在真实编程场景中表现卓越。

&emsp;&emsp;这一基础设施的设计与 GLM-5 的目标定位直接相关——它不是为了在标准基准上刷分，而是为了在真实的、长时间跨度的编程任务中持续工作。GLM-5 在 Coding 与 Agent 能力上取得开源 SOTA 表现，在真实编程场景的使用体验逼近 Claude Opus 4.5，更擅长复杂系统工程与长程 Agent 任务。

**国产芯片全栈适配**

&emsp;&emsp;GLM-5 完成了与华为昇腾、摩尔线程、海光、寒武纪、昆仑芯、沐曦以及燧原等国产芯片的全栈适配。这一适配不是简单的“能在国产芯片上运行”，而是针对 DSA 稀疏注意力的计算模式进行了专门的算子优化，使 GLM-5 的 744B 参数模型能够在国产算力平台上实现高效推理。

&emsp;&emsp;这一战略选择的意义超出了技术层面。在 GLM-5 发布后，外国网友直呼“在成本效率方面，美国的 AI 赶不上中国”，并认为 GLM-5“极大拉小了和 Claude Opus 4.6 之间的距离”。GLM-5 的国产芯片适配使中国的 AI 基础设施不再受制于 NVIDIA 的 GPU 供应，为大规模部署提供了独立的算力底座。

---

### 7.2 DeepSeek V4系列

- 论文地址：[DeepSeek-V4 Technical Report](https://arxiv.org/pdf/2604.17286)

&emsp;&emsp;DeepSeek-V3 用 671B 总参数、37B 激活参数和 557 万美元的训练成本证明了 MoE 架构的效率优势。DeepSeek-R1 用纯强化学习激励出长链推理能力，在推理任务上追平了 OpenAI o1。但两者共享一个根本约束：**上下文窗口受限于 128K token，KV 缓存在长序列场景下的内存占用成为推理瓶颈**。DeepSeek-V4 于 2026 年 4 月发布，包含 **V4-Pro（1.6 万亿总参数，激活 490 亿）** 和 **V4-Flash（2840 亿总参数，激活 130 亿）** 两个版本，两者均原生支持 **100 万 token 上下文**，均为 MoE 架构、纯文本模型。V4 在 1M 场景下，V4-Pro 的单 token FLOPs 只有 V3.2 的 **27%**，KV 缓存只有 **10%**。DeepSeek-V4 的主要贡献包括：

1. 引入 **mHC（流形约束超连接）** 强化残差连接，将传统残差流从一维扩展为多条并行通道，并通过双随机矩阵约束保证深层堆叠的数值稳定性；
2. 设计 **hybrid attention 架构**，CSA（压缩稀疏注意力）和 HCA（重度压缩注意力）交替叠加，解决长文效率问题；
3. 采用 **Muon 优化器**替代 AdamW，接管绝大多数参数的训练，提升训练稳定性和收敛速度；
4. MoE 部分继续使用 DeepSeekMoE，但进行了细节微调：affinity score 的激活函数从 Sigmoid 换成 Sqrt(Softplus(·))，去掉了 routing target nodes 的数量约束，前几层 dense FFN 换成了用 Hash routing 的 MoE 层；
5. 支持华为昇腾算力，预计下半年昇腾 950 超节点批量上市。

**mHC：给残差连接加一层数学约束**

&emsp;&emsp;残差连接是何恺明 2016 年在 ResNet 中提出的，十年来几乎没有根本性变化。模型一层一层堆叠，梯度沿着残差路径回传，这是深度学习能够工作的前提。但当模型越来越深、参数越来越多之后，传统残差连接开始暴露缺陷——信号传递不稳定，训练容易崩溃。

&emsp;&emsp;DeepSeek-V4 引入的 **mHC（Manifold-Constrained Hyper-Connections）** 将残差流从一维变成 **n_hc 条并行通道**，每层之间通过一个矩阵 B 来混合。核心创新在于将矩阵 B 约束到**双随机矩阵的流形**上（数学上称为 Birkhoff polytope），行和列都归一化为 1。这个约束带来两个关键好处：矩阵的谱范数天然不超过 1，残差传播有了硬上限，不会爆炸；这种矩阵在乘法下是封闭的，堆叠很多层仍然稳定。

&emsp;&emsp;输入映射 A 和输出映射 C 通过 Sigmoid 函数保证非负且有界，避免信号互相抵消。实现上使用 **Sinkhorn-Knopp 迭代**，交替做行归一化和列归一化，迭代 20 次收敛。DeepSeek 做了 fused kernel 配合选择性 recomputation，实测 mHC 带来的 wall-time 开销控制在 overlapped pipeline 的 **6.7%**。

**Hybrid Attention：CSA + HCA 交替叠加**

&emsp;&emsp;DeepSeek-V4 的注意力架构在每个 decoder block 中按层类型动态分派三种注意力模式：

- **Sliding-window full attention**：仅处理滑动窗口内的局部 token，用于引导层（bootstrap layers），匹配 V3 的全注意力风格；
- **Compressed Sparse Attention（CSA）** ：使用低压缩池（compress_rate_csa，默认 m=4）配合重叠窗口，加上 Lightning Indexer 对查询与池中条目进行评分，在核心注意力之前收集每个查询的 Top-k 块；
- **Heavily Compressed Attention（HCA）** ：使用高压缩池（compress_rate_hca，默认 m'=128）配合非重叠窗口，不使用索引器——每个池化条目都参与注意力计算。

&emsp;&emsp;三种注意力类型共享同一骨干：**共享 K=V 多查询注意力**（num_key_value_heads=1，KV 投影产生单个 KV 头，同一张量同时作为 key 和 value 读取）、**部分 RoPE**（在每头的尾部 qk_rope_head_dim 通道上施加旋转，注意力输出的 rope 切片也以位置 -i 施加相同旋转）、**逐头可学习注意力汇聚**、以及**分组低秩输出投影**。此外，CSA 还保留了**共享滑动窗口 K=V 分支**，以保持局部细粒度依赖，长距离压缩器的输出与该分支的 KV 在进入核心注意力之前拼接。

**Muon 优化器与 MoE 细节微调**

&emsp;&emsp;DeepSeek-V4 将 V3 使用的 AdamW 替换为 **Muon 优化器**，接管绝大多数参数的训练。Muon 最早由 Kimi 团队在大规模训练中验证，其在大规模训练中表现出更快的收敛速度和更好的训练稳定性。V4 将这一优化器引入 DeepSeek 系列，进一步提升了训练效率。

&emsp;&emsp;MoE 部分继续沿用 DeepSeekMoE 架构，但进行了若干细节微调：affinity score 的激活函数从 Sigmoid 换成了 Sqrt(Softplus(·))，去掉了 routing target nodes 的数量约束，前几层 dense FFN 换成了用 Hash routing 的 MoE 层。MTP（Multi-Token Prediction）模块与 V3 保持一致。

---

### 7.3 Kimi K2 / K3 系列

- 论文地址：
  - Kimi K2：[Kimi K2: Open Agentic Intelligence](https://arxiv.org/pdf/2507.20534)
  - Kimi K3：[Kimi K3 Technical Report](https://huggingface.co/MoonshotAI/Kimi-K3)

&emsp;&emsp;Kimi K2 于 2025 年 7 月发布，拥有 **1 万亿总参数（320 亿激活参数）** 的 MoE 架构，采用 **384 个专家，每层激活其中 8 个**，在多项基准测试中达到了开源模型的 SOTA 水平。Kimi K3 于 2026 年 7 月发布，总参数规模达到 **2.8 万亿**，激活参数约 **1040 亿**，是全球首个迈入 **3 万亿参数级别**的开源模型。K3 采用 MoE 架构，**896 个专家中每 Token 激活 16 个**，具备原生视觉理解能力，支持 **100 万 token 的上下文窗口**。

**K2 的核心创新：MuonClip 优化器**

&emsp;&emsp;Kimi K2 最核心的技术创新是 **MuonClip 优化器**。团队摒弃了传统的 Adam 优化器，创新性地使用了 Muon 优化器。相比 Adam，Muon 在大规模训练中表现出更快的收敛速度和更好的训练稳定性，同时优化了梯度更新过程，减少了内存占用。K2 进一步在 Muon 基础上引入了 **QK-clip 技术**，有效缓解了训练过程中的不稳定性问题。基于 MuonClip，K2 在 **15.5 万亿 token** 的数据上完成了训练。

&emsp;&emsp;K2 的架构采用了**超稀疏 MoE 配合多头潜在注意力（MLA）** ，与 DeepSeek-V3 的设计类似。MLA 通过潜在空间压缩显著减少了注意力计算的内存开销，在保持性能的同时提升了推理速度。

**K3 的架构跃迁：KDA + AttnRes + Stable LatentMoE**

&emsp;&emsp;Kimi K3 在 K2 的基础上进行了全面的架构升级。K3 的注意力架构采用 **KDA + AttnRes** 的组合：以 **3:1 比例混合 KDA（Kimi Delta Attention）与 Gated MLA**，实现高效长上下文建模，并通过**块级注意力残差**增强跨层信息流动。

&emsp;&emsp;MoE 部分采用 **Stable LatentMoE**：每个 token 从 **896 个路由专家中激活 16 个**，通过 **SiTU-GLU** 与 **Quantile Balancing** 保持极高稀疏度下的训练稳定。K3 的视觉编码器 **MoonViT-V2** 从零开始使用 next-token prediction 训练，无需对比预训练，在达到 SigLIP 初始化基线效果的同时获得了更稳定的优化过程。

&emsp;&emsp;K3 的后训练在通用推理、通用 Agent 和编程 Agent 三大领域进行大规模任务合成，基于百万 token 上下文的强化学习基建，在近 20 个内部评测集上完成了完整评估。

**K3 的基础设施开源：MoonEP / FlashKDA / AgentEnv**

&emsp;&emsp;Kimi K3 同步开源了三项支撑模型训练的关键 Infra 技术：

- **MoonEP**：为超大的细粒度 MoE 打造的高性能通信库，让专家并行（expert-parallel）的通信在不均衡的情况下仍然实现极致效率；
- **FlashKDA**：Kimi Delta Attention 的高性能算子。在英伟达 H20 上，相比 flash-linear-attention 基线，prefill 速度提升 **1.72-2.22 倍**，可以直接作为 flash-linear-attention 的替换后端；
- **AgentEnv**：与 KVCache.ai 合作开发的沙箱系统，用于大规模运行 Agent 环境，为 K3 的后训练提供高保真、强隔离沙箱，支持快速快照、恢复和分叉。

&emsp;&emsp;K3 在核心架构上的原始创新使其在多项评测中展现出逼近全球顶尖闭源模型的性能。K3 重点强化了长程编程能力，规模化效率相比 K2.5 提升了 **2.5 倍**（即在算力最优意义下，单位算力产出智能约等于原来 2.5 倍）。

---

### 7.4 Xiaomi MiMo-V2系列

- 论文地址：[MiMo-V2-Flash Technical Report](https://arxiv.org/pdf/2601.03678)

&emsp;&emsp;小米于 2025 年 12 月发布 MiMo-V2-Flash，这是小米首个开源 MoE 大模型，由罗福莉主导研发。MiMo-V2-Flash 总参数量 **309B**，每次推理仅激活 **15B** 参数，激活比约为 **20:1**，在同级别模型中属于第一梯队。该模型采用 **5:1 混合的滑动窗口注意力（SWA）与全局注意力架构**及多层 MTP 推理加速技术，在 **27T 词元**上完成训练。

**Hybrid SWA：全局视野与局部效率的平衡**

&emsp;&emsp;MiMo-V2-Flash 最核心的架构创新是 **Hybrid SWA（混合滑动窗口注意力）** 。纯 Transformer 的全局注意力在长序列场景下是性能杀手——self-attention 的计算量随序列长度平方增长，跑到 32K token 时注意力本身就成了瓶颈。Flash Attention 解决了计算效率问题，但解决不了信息密度问题。

&emsp;&emsp;MiMo-V2-Flash 引入的混合注意力机制将**全局注意力与滑动窗口注意力按 1:5 的比例搭配**。全局注意力负责跨区域的信息融合，让模型能看到句子开头和结尾之间的关联；滑动窗口注意力负责局部语义的精确捕捉，5 层窗口注意力交替堆叠，等效感受野并不小，但每层的计算量被严格控制在局部范围内。用 MiMo 团队的话说，“全局负责‘抬头看路’，局部负责‘低头干活’”。两者结合，在保持推理速度的同时，长上下文的理解质量不会出现明显下滑。

**Agent 原生训练：GRPO 与 MiMo Coding Bench**

&emsp;&emsp;MiMo-V2-Flash 的定位是“专为智能体 AI 设计，专注于快”，其训练方法围绕这一目标展开。团队使用 **GRPO（组相对策略优化）** 进行强化学习优化。GRPO 是 DeepSeek 在 2025 年初提出的群体相对策略优化方法，比传统 PPO 更稳定、收敛更快。MiMo 在此基础上融合了当年的新论文成果，后发优势明显。

&emsp;&emsp;团队还自建了 **MiMo Coding Bench** 评测集，专门衡量模型在复杂编程任务中的表现，模拟真实 Agent 场景。从评测结果看，MiMo-V2-Flash 在代码能力上达到开源第一，在 Agent 场景下任务完成率高。MiMo-V2-Flash 发布当天即冲上了 OpenRouter 闭源模型榜首，API 定价为百万输出 Token 仅两元，展现了极高的性价比。

---

### 7.5 Qwen-3系列

- 论文地址：[Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388)

&emsp;&emsp;Qwen2.5 以 72B 稠密参数和 18T token 的训练数据达到了与 LLaMA 3.1 405B 相当的性能，验证了“数据规模优先于参数规模”的效率路线。但 Qwen2.5 的推理能力和通用能力仍然是分离的——用户需要在“对话优化模型”和“专用推理模型”之间切换。Qwen3 于 2025 年 5 月发布，核心创新在于**将推理模式和非推理模式整合至统一框架**，一个模型同时支持复杂多步推理和快速响应，根据用户查询或对话模板动态切换模式。Qwen3 系列包含 **6 个密集模型**（0.6B、1.7B、4B、8B、14B、32B）和 **2 个 MoE 模型**（30B-A3B、235B-A22B），全部采用 Apache 2.0 协议开源。

**混合推理模式：思考预算机制**

&emsp;&emsp;Qwen3 最核心的创新是**双模式架构**。模型将“思考模式”（用于复杂多步推理）和“非思考模式”（基于上下文的快速响应）整合到同一个模型中，无需切换模型，而是根据用户查询或对话模板进行动态模式切换。这一设计与 OpenAI 的 o1/o3 系列将推理模型和通用模型分开的策略形成鲜明对比——Qwen3 将两者合并为一个统一的模型。

&emsp;&emsp;Qwen3 还引入了**思考预算机制**，允许用户在推理过程中自适应分配计算资源，从而根据任务复杂度平衡延迟与性能。对于简单查询，模型使用较少的思考预算快速响应；对于复杂推理任务，模型可以分配更多的计算资源进行深度思考。

**MoE 架构：128 专家激活 8 个**

&emsp;&emsp;Qwen3 的旗舰 MoE 模型 **Qwen3-235B-A22B** 总参数量为 2350 亿，激活参数量为 220 亿，共 **94 层**，采用 **128 个专家，每个 token 激活 8 个**。与 Qwen2.5-MoE 不同的是，Qwen3-MoE **舍弃了共享专家模块**，并采用**全局批次负载均衡损失**技术促进专家专业化。这些架构与训练创新使模型在下游任务中表现显著提升。

&emsp;&emsp;Qwen3 的密集模型架构与 Qwen2.5 相似，包含 GQA、SwiGLU、RoPE 以及带预归一化的 RMSNorm。一个关键变化是**移除了 Qwen2 中使用的 QKV-bias**，并在注意力机制中**引入 QK-Norm**，以确保 Qwen3 的训练稳定性。

**多语言扩展与蒸馏策略**

&emsp;&emsp;Qwen3 将多语言支持从 Qwen2.5 的 **29 种扩展至 119 种语言及方言**，通过增强跨语言理解与生成能力提升全球可用性。在蒸馏策略方面，Qwen3 采用“大带小”的模式，从大号模型中蒸馏数据训练小号模型，显著降低了构建轻量级模型所需的计算资源，同时确保其性能具有高度竞争力。

&emsp;&emsp;在性能方面，Qwen3-235B-A22B 在数学（AIME25 得分 **81.5**）、代码生成（LiveCodeBench **70.7**）等核心评测中**超越 DeepSeek-R1（671B 参数）和 Grok-3 等国际顶尖模型**，在推理效率和任务适应性上实现突破，仅需 **4 张 H20 显卡**即可部署旗舰模型。

---

### 7.6 Llama 4系列

- 论文地址：[The Llama 4 Herd: Architecture, Training, Evaluation, and Deployment Notes](https://arxiv.org/pdf/2601.11659)（该论文已被 arXiv 撤稿，但此前公开的技术细节仍可作为参考）

&emsp;&emsp;Llama 3.1 用 405B 稠密参数和 15 万亿 token 证明了开放权重模型可以达到闭源旗舰的性能水平，但其核心架构与 Llama 1 相比几乎没有本质变化。Meta 始终拒绝 MoE 架构和架构层面的激进创新，将“复杂度管理”作为超大规模训练的首要原则。Llama 4 于 2025 年 4 月发布，标志着 Meta 首次拥抱 MoE 架构、原生多模态训练和创新的长上下文机制。Llama 4 包含 **Scout（17B 激活参数，16 专家，109B 总参数）** 和 **Maverick（17B 激活参数，128 专家，400B 总参数）** 两个开放权重版本，以及 **Behemoth（约 2T 总参数，288B 激活参数）** 预览版教师模型。

**MoE 架构与无丢弃的 Top-1 路由**

&emsp;&emsp;Llama 4 的 MoE 层采用**共享专家 + 路由专家**的混合设计。Scout 配置 **1 个共享专家和 16 个路由专家**，Maverick 配置 **1 个共享专家和 128 个路由专家**。路由机制采用**无丢弃的 token 选择路由**和 **Top-1 选择**——每个 token 只被路由到一个路由专家，而非 Mixtral 的 Top-2 选择。Top-1 路由的优势在于计算效率：每个 token 只触发一个路由专家的前向传播，而非两个，进一步降低了推理计算量。共享专家则确保所有 token 都能获得基础能力的稳定传递。

**iRoPE：10M token 上下文的长上下文架构**

&emsp;&emsp;Llama 4 Scout 最引人注目的技术突破是 **1000 万 token 的上下文窗口**。这一能力的核心支撑是 Meta 提出的 **iRoPE（交错旋转位置编码）** 架构。iRoPE 的解决方案是**交替使用带位置编码和不带位置编码的注意力层**：偶数层使用 RoPE 处理局部位置信息，奇数层不使用位置编码（NoPE），专注于内容本身的语义匹配。这种交替设计使模型在超长上下文中既能利用位置信息进行精确检索，又能避免位置编码的数值不稳定性对语义匹配的干扰。

&emsp;&emsp;iRoPE 还包含**温度缩放**：根据序列长度动态调整注意力分布的“温度”，在序列变长时适当平滑注意力分布，防止其崩溃。在推理阶段，Llama 4 对特定层应用**分块注意力掩码**，将长序列分割为可管理的块进行处理，使模型能够在 10M token 的规模上保持内存效率。

**早期融合多模态**

&emsp;&emsp;Llama 4 是 Llama 系列首个**原生多模态**模型。它采用**早期融合**策略，将文本和视觉信息在模型骨干的初始处理阶段即整合到统一表示空间。这与 GPT-4 的“拼接式”多模态有本质区别：GPT-4 使用独立的视觉编码器提取图像特征，再通过交叉注意力注入语言模型；Llama 4 则让文本 token 和视觉 token 在进入 Transformer 的第一层之前就处于同一表示空间中，所有 Transformer 层都能同时“看到”两种模态的信息。Llama 4 支持文本和图像输入，输出为文本和代码，覆盖 12 种语言。

&emsp;&emsp;Llama 4 的训练数据超过 **30 万亿 token**，是 Llama 3 预训练数据的两倍以上。训练使用了 **32,000 张 H100 GPU**，消耗约 **750 万 GPU 小时**。Maverick 在多项基准上超越 GPT-4o 和 Gemini 2.0 Flash，在推理和编程任务上与 DeepSeek V3 相当，但激活参数不到后者的一半。在 LMArena 的实验性聊天版本上，Maverick 取得 **ELO 1417** 的分数。

---

### 7.7 Claude 5系列

- 论文地址：[Claude 5 System Card](https://www.anthropic.com/claude-5-system-card)

&emsp;&emsp;Claude 4.8 和 Claude Opus 4.5 在专业领域和安全性上建立了极高的标准，但其对齐方法主要依赖 RLHF——通过人类反馈调整模型行为。RLHF 存在标注成本高、价值偏差难修正、幻觉问题突出等痛点。Claude 5 于 2026 年 6 月发布，包含 **Claude Fable 5（安全对齐版，面向企业与公众部署）** 和 **Claude Mythos 5（全能力版，面向受控研究环境）** 双轨版本。Claude 5 在 **MMLU-Pro 基准测试中达到 98.3% 的准确率**，逼近人类专家水平，同时在编码、数学、法律、医疗等领域实现**幻觉率降低 50% 以上**的关键突破。

**宪法自我纠正机制：从 RLHF 到内生对齐**

&emsp;&emsp;Claude 5 最核心的创新是 **宪法自我纠正机制（Constitutional Self-Correction）** 。传统 RLHF 方法依赖人工反馈进行“事后修补”——模型生成输出后，由人类标注员或奖励模型进行评估和修正。宪法 AI 不遵循 OpenAI 的做法，而是在模型训练的**底层植入一套“宪法”原则**，使模型学会根据宪法原则**自我评估并修正回应**。

&emsp;&emsp;具体来说，宪法 AI 的一个创新是让 Claude **依据宪法原则自我评价、自我修正**。例如，模型在生成一个回答后，会评估该回答“是否尊重了用户自主性”，然后根据评估重新生成一个修正版本。这一机制使模型具备**实时自检与价值偏差修正**的能力，而非依赖事后的外部反馈。Anthropic 发现，宪法 AI 让模型生成自己的反馈信号，在细微的伦理判断上比众包人类评分者更可靠。

**基础架构优化**

&emsp;&emsp;Claude 5 基于优化的 Transformer 架构，在**注意力机制、归一化、激活函数**三大核心模块实现了升级，为高性能推理与长文本处理奠定了基础。Anthropic 发布的 194 页技术报告详细阐述了架构选择，包括对神经网络内部激活的实时监控机制。

&emsp;&emsp;Claude 5 的双轨发布策略——Fable 5（安全对齐版）和 Mythos 5（全能力版）——平衡了模型能力与安全可控性。Fable 5 是 Anthropic 首个**通用可用的 Mythos 级别模型**，属于比 Opus 更高的能力层级，以无防护版本发布。这一策略使 Anthropic 能够在研究环境中探索模型的全能力边界，同时为企业和公众部署提供经过安全对齐的版本。

---

### Grok 3

- 论文地址：[Grok 3 Technical Report](https://x.ai/grok-3-technical-report)

&emsp;&emsp;Grok 2 是 xAI 的第二代模型，在 2024 年与 GPT-4o 的竞争中展现了竞争力，但其架构和训练规模相对有限。Grok 3 于 2025 年 2 月发布，在训练过程中调用了 **10 万个 NVIDIA H100 芯片**，较前代产品 Grok 2 使用的 15,000 个 GPU 实现了数倍的跨越式提升。马斯克称 Grok 3 的计算量比 Grok 2 提升了 **10 倍**。Grok 3 在 **MMLU 基准上达到 89.7% 的准确率**，较 GPT-4 高出 6.2 个百分点；在 **MATH 数学推理数据集上实现 81.3% 的解题准确度**，创下当前公开模型的最佳记录。

**MoE 架构与分层注意力机制**

&emsp;&emsp;Grok 3 采用 **600B 参数的 MoE 架构（16 个专家，每次激活 2 个）** ，配备**分组查询注意力（GQA）** ，共 96 层，隐藏维度为 12,288，上下文长度为 **131K token**。Grok 3 的核心创新在于**分层注意力机制（Hierarchical Attention Mechanism）** ，将文本、图像、语音处理整合为统一认知框架。与传统的 MoE 模型不同，Grok 3 引入了**跨专家注意力门控**，允许专业化组件之间共享知识，而不会产生灾难性干扰。

&emsp;&emsp;Grok 3 还采用了**分阶段认知流架构**，使推理过程可审计、可追踪。这一架构将推理分解为多个阶段，每个阶段的决策都可以被独立检查和验证，为 AI 推理的透明性和可解释性提供了工程实践方案。

**多模态认知引擎与训练规模**

&emsp;&emsp;Grok 3 在 **CM3leon 架构**基础上引入了跨模态注意力对齐机制、量子化语义编码器和时空连续性建模模块。在 **VCR（视觉常识推理）测试中，多模态理解准确率达到 92.1%**，较 CLIP 提升 19%。

&emsp;&emsp;Grok 3 的训练使用了 **10 万个 H100 GPU**，训练计算量是 Grok 2 的 **10 倍**。xAI 同步发布了 **“约束对齐协议”（CAP）** ，包含动态价值观修正模块、事实性核查子系统和危害内容过滤网络，测试显示将有害输出概率控制在 **0.003% 以下**。

---

### 7.8 GPT-5 / GPT-6系列

- 论文地址：[GPT-5 System Card](https://arxiv.org/pdf/2601.03267)

&emsp;&emsp;GPT-4o 通过端到端统一架构实现了实时多模态交互，o1/o3 通过推理时计算扩展实现了深度推理能力。但这两条路线是分离的——GPT-4o 是快速但缺乏深度推理的通用模型，o1/o3 是深度推理但响应缓慢的专用模型。用户需要在两者之间切换，而模型本身无法根据任务复杂度自动选择。GPT-5 于 2025 年 8 月发布，核心思路是**将快速模型、深度推理模型和实时路由器整合为一个统一系统**。GPT-5 的主要创新包括：

1. 采用 **“统一系统”（unified system）** 复合设计，由一个高效的基础模型（gpt-5-main）、一个深度推理模型（gpt-5-thinking）和一个**实时路由器**组成，路由器根据对话类型、复杂度、工具需求和明确意图快速决定使用哪个模型；
2. 引入**分层路由（hierarchical routing）** 机制，动态分配计算资源，使简单查询快速响应，复杂问题自动切换到深度推理模式；
3. 在减少幻觉、改进指令遵循和最小化谄媚方面取得显著进展，在写作、编码和健康三个 ChatGPT 最常用场景中提升了性能；
4. 推理模型通过强化学习训练，学会**在回答之前先生成长内部思维链**，通过训练学会精炼思考过程、尝试不同策略并识别错误；
5. 路由器持续基于**真实信号**进行训练，包括用户切换模型的频率、回复偏好率和测量准确性，随时间不断优化。

**统一系统架构：分层路由与自适应计算**

&emsp;&emsp;GPT-5 最根本的架构创新是**用路由器替代了单一模型**。传统 LLM 中，用户面对的是一个固定的模型——无论问题简单还是复杂，都使用相同的计算路径。GPT-5 的统一系统将这一模式彻底改变：gpt-5-main 处理大多数日常查询，gpt-5-thinking 处理需要深度推理的复杂问题，而实时路由器在两者之间动态调度。

&emsp;&emsp;路由器的工作原理基于多个信号：**对话类型**（闲聊、编码、分析等）、**复杂度**（简单事实查询还是多步推理）、**工具需求**（是否需要调用外部工具）、以及**明确意图**（用户是否在提示中说“仔细想想这个问题”）。路由器持续基于真实信号进行训练，包括用户切换模型的频率、回复偏好率和测量准确性，随时间不断优化。

&emsp;&emsp;GPT-5 系统在 API 中提供了多个模型变体：**gpt-5-main** 和 **gpt-5-main-mini** 是快速高吞吐模型，**gpt-5-thinking**、**gpt-5-thinking-mini** 和 **gpt-5-thinking-nano** 是不同规模的推理模型。在 ChatGPT 中，**gpt-5-thinking-pro** 使用并行测试时计算，进一步提升了推理能力。

**推理模型：强化学习训练“先思考后回答”**

&emsp;&emsp;GPT-5 的推理模型（gpt-5-thinking 系列）通过**强化学习**训练，学会在回答之前先生成一条内部的思维链。通过训练，这些模型学会精炼思考过程、尝试不同策略并识别自己的错误。推理使模型能够遵循特定的指导原则和模型策略，帮助它们按照安全期望行事——这意味着它们提供更有帮助的答案，并更好地抵抗绕过安全规则的尝试。

&emsp;&emsp;这一训练方法与 o1/o3 的推理范式一脉相承，但 GPT-5 的关键区别在于：推理能力**不再是独立模型的专属功能**，而是被整合进统一系统中，由路由器自动调度。用户不需要知道“这个问题需要推理”，路由器会根据对话的复杂度自动判断。OpenAI 在系统卡中明确表示：“在不久的将来，我们计划将这些能力整合到单一模型中”。

**安全与对齐**

&emsp;&emsp;GPT-5 所有模型都配备了 **safe-completions**——OpenAI 最新的安全训练方法，用于防止不允许的内容生成。与 ChatGPT agent 类似，OpenAI 将 **gpt-5-thinking 在生物和化学领域标记为“高能力”** ，并激活了相应的安全防护措施。虽然 OpenAI 没有明确证据表明该模型能够帮助新手造成严重生物危害，但出于预防考虑，他们选择了这一保守立场。

&emsp;&emsp;在数据方面，GPT-5 模型在多样化的数据集上训练，包括互联网上公开可用的信息、与第三方合作获取的信息，以及用户或人类训练师提供的信息。数据处理管线包含严格的过滤流程以维护数据质量并降低潜在风险，使用先进的过滤流程减少训练数据中的个人信息，并结合 Moderation API 和安全分类器防止有害或敏感内容的使用。

---

## 8 总结

&emsp;&emsp;从 2017 年 Transformer 提出到 2026 年 GPT-5、Claude 5、DeepSeek-V4、Kimi K3 等模型相继落地，LLM 的架构演进可以概括为一条主线：**以 Transformer 为统一基座，以 decoder-only 为主流范式，以 Scaling Law 为导航，以效率优化为约束，以对齐和推理为能力放大器，以多模态、长上下文和 Agent 为应用出口。** 这条主线并非线性叠加，而是在“能力—成本—可控性”三者之间不断重新平衡的结果。

&emsp;&emsp;在**基础架构层面**，LLM 并未脱离 Transformer 的基本框架，但核心组件已经完成了一轮系统性替换。注意力机制从标准多头注意力（MHA）出发，先后演化出多查询注意力（MQA）、分组查询注意力（GQA）、多头潜在注意力（MLA），并进一步走向滑动窗口注意力、稀疏注意力、混合注意力与线性注意力。位置编码从可学习的绝对位置嵌入，转向相对位置编码、RoPE、ALiBi，再到 iRoPE、NoPE 和动态温度缩放。归一化从 Post-LN 转向 Pre-LN、RMSNorm 和 QK-Norm；激活函数从 ReLU/GELU 转向 SwiGLU、GeGLU、SiTU-GLU；残差连接从标准残差走向 mHC 等流形约束超连接。这些组件的共同目标是：在更深的网络、更长的上下文和更大的参数规模下，保持训练稳定性和推理效率。

&emsp;&emsp;在**模型范式层面**，演进大致经历了四个阶段。第一阶段是 2018—2020 年的“架构探索期”，GPT-2、GPT-3、T5 分别确立了 decoder-only、上下文学习和文本到文本统一框架；第二阶段是 2021—2022 年的“缩放反思期”，Gopher 将参数优先路线推到极致，Chinchilla 则证明参数量与训练数据应当等比例放大，LLaMA 系列随后将“小模型 + 大数据”变成开源标准；第三阶段是 2023—2024 年的“开源爆发与多模态扩展期”，Mistral、Falcon、Qwen、DeepSeek 等模型从效率、数据工程、MoE 架构和长上下文等维度展开竞争，GPT-4/4o、Gemini 1.5 Pro 将多模态和百万级上下文推向实用；第四阶段是 2025—2026 年的“推理与系统工程期”，o1/o3、DeepSeek-R1、Qwen3、GPT-5 等模型将推理时计算、强化学习、统一路由和 Agent 能力整合进模型系统，LLM 从“文本生成器”转向“通用问题求解器”。

&emsp;&emsp;在**训练与对齐层面**，后训练范式从 InstructGPT 的“SFT + 奖励模型 + PPO”逐步转向“SFT + DPO + GRPO”的多阶段流程。可验证奖励、纯强化学习、宪法自我纠正和推理时思维链，成为提升模型推理能力与安全性的关键手段。DeepSeek-R1 证明，不依赖人类标注推理轨迹，仅通过规则奖励和强化学习，模型也能自发涌现出自我反思、验证和动态策略适应等高级推理模式。GPT-5 则进一步将快速模型、深度推理模型和实时路由器整合为统一系统，使推理能力从“专用模型功能”变成“系统自动调度能力”。

&emsp;&emsp;在**效率层面**，MoE 已成为超大规模模型的主流选择。从 Mixtral 8x7B 到 DeepSeek-V3、Llama 4、Qwen3、GLM-5、DeepSeek-V4、Kimi K3，几乎所有旗舰模型都采用“大总参数量 + 小激活参数量”的稀疏架构。专家路由从 Top-2 走向 Top-1、细粒度专家、共享专家和无辅助损失负载均衡；训练精度从 FP16/BF16 走向 FP8；优化器从 AdamW 走向 Muon、MuonClip；推理优化从 KV 缓存压缩、推测解码、INT4 量化走向 DSA、CSA、HCA 等混合稀疏注意力。DeepSeek-V3 以 557 万美元训练成本追平 GPT-4o 级别性能，标志着“算法效率的杠杆效应”开始部分替代单纯的算力投入。

&emsp;&emsp;在**生态与竞争层面**，开源权重与闭源旗舰之间的差距经历了“落后—追赶—趋同—再分化”的过程。LLaMA 1/2/3、Mistral、Falcon、Qwen、DeepSeek、Kimi、GLM、MiMo 等模型以 Apache 2.0、MIT 或社区许可发布，推动了推理能力、长上下文和多语言能力的民主化。Qwen 系列衍生模型超过 7.8 万个，Llama 系列下载量超过 12 亿次，DeepSeek-R1 的 MIT 许可使推理能力下沉到 1.5B—70B 的消费级规模。但与此同时，Meta 终止 Llama 开源策略、Qwen2.5-Max 与 GLM-5 等旗舰采用闭源或混合策略，也说明开放权重路线在商业可持续性上仍面临根本挑战。

&emsp;&emsp;展望未来，LLM 架构演进可能继续沿几个方向展开：**第一，推理能力内化**，推理时计算不再是独立模型的专属功能，而是由统一系统按需调度；**第二，多模态原生统一**，文本、图像、音频、视频在同一个 Transformer 中端到端训练，早期融合取代后期拼接；**第三，长上下文常态化**，100 万 token 逐步成为旗舰标配，千万级上下文在特定场景落地；**第四，Agent 化与工具调用**，模型从“回答问题”转向“自主规划、执行、迭代”，GLM-5 的 Agentic Engineering 和 GPT-5 的统一系统是这一方向的早期实践；**第五，效率与安全的双重约束**，在算力、显存、延迟和合规成本持续上升的背景下，稀疏激活、混合注意力、量化推理和安全对齐将成为模型设计的默认前提。

&emsp;&emsp;总体而言，LLM 的架构演进并非单一技术的胜利，而是注意力机制、位置编码、归一化、激活函数、残差连接、MoE 路由、训练目标、后训练对齐和推理系统工程共同作用的结果。从 GPT-2 的 15 亿参数到 Kimi K3 的 2.8 万亿总参数，从 1024 token 上下文到 1000 万 token 上下文，从纯文本生成到多模态 Agent，LLM 已经从一个语言建模工具演化为支撑对话、推理、编程、多模态理解和自主行动的基础智能底座。未来的竞争，将不再只是“谁的参数更多”，而是“谁能在给定算力、数据和部署约束下，更高效地产生可靠、可控、可扩展的智能”。

