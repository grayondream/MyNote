# LLM架构演进

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

#### 2.1.1 缩放点积注意力

&emsp;&emsp;注意力机制是 Transformer 的核心。给定查询向量 $Q$、键向量 $K$ 和值向量 $V$，缩放点积注意力的定义为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

&emsp;&emsp;其中 $Q \in \mathbb{R}^{n \times d_k}$、$K \in \mathbb{R}^{m \times d_k}$、$V \in \mathbb{R}^{m \times d_v}$，$n$ 是查询序列长度，$m$ 是键值序列长度，$d_k$ 是键的维度。除以 $\sqrt{d_k}$ 的目的是防止点积在维度较高时数值过大，导致 softmax 梯度消失。注意力分数矩阵的每一行经过 softmax 后，表示当前查询位置对所有键位置的关注权重，最终输出是值向量的加权和。

#### 2.1.2 多头注意力（MHA）

&emsp;&emsp;单一注意力头只能学习一种“相关性模式”。为了让模型同时关注不同类型的关系——语法依赖、语义相似、位置邻近——Transformer 引入多头注意力：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

&emsp;&emsp;其中 $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$、$W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$。通常取 $d_k = d_v = d_{\text{model}} / h$。MHA 是 GPT-1、GPT-2、BERT、T5 等早期模型的标配，也是 LLaMA 1 的注意力方案。

#### 2.1.3 多查询注意力（MQA）

&emsp;&emsp;MHA 在推理时，每个 Query 头都需要独立的 Key 头和 Value 头，KV 缓存大小与注意力头数 $h$ 成正比。MQA 将所有 Query 头共享同一组 Key 和 Value 头，KV 缓存大小降至原来的 $1/h$。MQA 由 Shazeer 于 2019 年提出，被 Falcon 系列大规模采用。代价是 Key/Value 表达能力的压缩，在超大规模模型上可能出现性能损失。

#### 2.1.4 分组查询注意力（GQA）

&emsp;&emsp;GQA 将 Query 头分组，每组共享一组 Key 和 Value 头，在 MQA 和 MHA 之间取得折中。GQA 被 LLaMA 2/3、Mistral、Qwen 2/2.5/3、Llama 4、DeepSeek 等几乎所有主流开源模型采用，成为现代 LLM 的标配。LLaMA 3.1 的 8B、70B 和 405B 全部使用 GQA；Qwen3 的密集模型也包含 GQA、SwiGLU、RoPE 以及带预归一化的 RMSNorm。

#### 2.1.5 多头潜在注意力（MLA）

&emsp;&emsp;MLA 是 DeepSeek 系列提出的压缩式注意力。它将 Key 和 Value 张量先投影到一个低维潜在空间，再投影回高维空间，存储的是低维中间态而非完整 KV 矩阵。这一设计受低秩适配启发，在保留模型捕捉丰富上下文关系能力的同时，大幅降低 KV 缓存的内存占用。MLA 还与旋转位置嵌入（RoPE）集成，在压缩 KV 的同时保持位置信息的完整性。MLA 是 DeepSeek-V3 能够以 37B 激活参数高效推理的关键技术之一，也被 Kimi K2 等模型采用。

#### 2.1.6 交叉注意力与因果掩码

&emsp;&emsp;交叉注意力（Cross-Attention）中，Query 来自解码器的中间表示，Key 和 Value 来自编码器的输出。它用于 encoder-decoder 架构，使解码器在生成每个 token 时能够“查询”编码器的全部输入信息。LDM/Stable Diffusion 中，交叉注意力的 Query 来自 UNet 的中间特征，Key 和 Value 来自文本嵌入，使图像生成能够根据文本提示进行条件控制。

&emsp;&emsp;因果掩码（Causal Mask）用于 decoder-only 架构的自注意力。它确保位置 $i$ 只能关注位置 $\leq i$ 的 token，从而支持自回归生成。具体实现是在 softmax 之前将未来位置的注意力 logit 设为 $-\infty$，使 softmax 后的权重为零。

#### 2.1.7 FlashAttention

&emsp;&emsp;FlashAttention 是一种注意力计算的显存优化实现。它通过分块计算和重计算，将注意力操作的内存访问从二次方降低到线性，避免显式存储完整的 $n \times n$ 注意力矩阵。FlashAttention 在 Falcon、LLaMA、Qwen 等模型的训练和推理中得到大规模应用，使长序列训练在合理显存下成为可能。

#### 2.1.8 滑动窗口注意力（SWA）

&emsp;&emsp;SWA 将每个 token 的注意力范围限制在一个固定大小的窗口内：query 位置 $i$ 只关注 $[i-W, i]$ 范围内的 token，其中 $W$ 是窗口大小。Mistral 7B 的窗口大小为 4096，模型共 32 层，理论感受野可达约 131K token。SWA 带来的最直接收益是 KV 缓存大小的上界约束：当序列长度超过窗口大小时，KV 缓存不再继续增长，而是被固定在窗口大小对应的上限。配合滚动缓冲区缓存（rolling buffer cache），Mistral 7B 可以在处理任意长度序列时保持恒定的显存占用。MiMo-V2-Flash 采用 5:1 混合的滑动窗口注意力与全局注意力架构。

#### 2.1.9 稀疏注意力与 DSA

&emsp;&emsp;稀疏注意力通过只计算部分注意力权重来降低复杂度。GLM-5 引入 DeepSeek 同款稀疏注意力（DSA），核心思想是“先筛选、后计算”：首先由闪电索引器（Lightning Indexer）快速评估查询 token 与历史 token 的相关性并打分，然后仅选择得分最高的 Top-k 个 token 进行完整的注意力计算。在 k=2048、上下文长度 L=128K 时，计算量减少约 98%。在 GLM-5 的 744B 参数模型中，DSA 被用于处理长达 200K 的上下文，将注意力计算量降低了 1.5–2 倍，推理成本降低约 40% 至 50%。

#### 2.1.10 混合注意力 CSA / HCA

&emsp;&emsp;DeepSeek-V4 设计 hybrid attention 架构，CSA（压缩稀疏注意力）和 HCA（重度压缩注意力）交替叠加。CSA 使用低压缩池配合重叠窗口，加上 Lightning Indexer 对查询与池中条目进行评分，在核心注意力之前收集每个查询的 Top-k 块；HCA 使用高压缩池配合非重叠窗口，不使用索引器，每个池化条目都参与注意力计算。三种注意力类型共享同一骨干：共享 K=V 多查询注意力、部分 RoPE、逐头可学习注意力汇聚、分组低秩输出投影。CSA 还保留了共享滑动窗口 K=V 分支，以保持局部细粒度依赖。

#### 2.1.11 线性注意力

&emsp;&emsp;线性注意力通过将 softmax 注意力分解为核函数形式，将计算复杂度从 $O(n^2)$ 降至 $O(n)$。Kimi K3 的 KDA（Kimi Delta Attention）即属于高效长上下文建模的线性注意力变体，以 3:1 比例混合 KDA 与 Gated MLA，并通过块级注意力残差增强跨层信息流动。

---

### 2.2 Transformer 架构

#### 2.2.1 Encoder-only

&emsp;&emsp;Encoder-only 架构使用双向注意力，擅长理解类任务（分类、抽取式问答），但无法直接生成文本。BERT 是 encoder-only 的代表。它通过掩码语言建模学习双向表示，在 GLUE 等理解基准上取得了当时最优结果。

#### 2.2.2 Decoder-only

&emsp;&emsp;Decoder-only 架构使用因果注意力，擅长生成类任务，是当前 LLM 的主流范式。GPT 系列、LLaMA 系列、Mistral、Qwen、DeepSeek、Kimi、GLM 等几乎全部采用 decoder-only。其训练目标是自回归语言建模：给定前文 token 序列，预测下一个 token。Decoder-only 在自回归语言建模目标下天然支持任意任务的统一表示，且训练和推理流程最简单、最容易规模化。

#### 2.2.3 Encoder-Decoder

&emsp;&emsp;Encoder-Decoder 架构中，编码器双向理解输入，解码器自回归生成输出，适合序列到序列任务。原始 Transformer、T5、BART 是代表。T5 通过文本到文本框架证明，encoder-decoder 架构在统一框架下表现最优——编码器的双向注意力适合理解输入，解码器的因果注意力适合自回归生成，交叉注意力负责在两者之间传递信息。但后续的大规模通用模型几乎全部回归 decoder-only。

#### 2.2.4 Post-LN 与 Pre-LN

&emsp;&emsp;原始 Transformer 使用 Post-LN：$x_{l+1} = \text{LayerNorm}(x_l + \text{Sublayer}(x_l))$。这种设计在深层网络中会导致梯度范数剧烈波动，训练不稳定。GPT-2 将层归一化移到每个子模块的输入之前，即 Pre-LN：$x_{l+1} = x_l + \text{Sublayer}(\text{LayerNorm}(x_l))$。Pre-LN 使残差路径上的信息可以不经过任何归一化变换而直接传递到深层，极大缓解了梯度消失和梯度爆炸问题，使训练数十层甚至上百层的 Transformer 成为可能。此后，几乎所有主流 LLM 都采用 Pre-LN 或其变体。

#### 2.2.5 残差连接

&emsp;&emsp;残差连接由何恺明 2016 年在 ResNet 中提出，是深度学习能够工作的前提。模型一层一层堆叠，梯度沿着残差路径回传。在 Transformer 中，每个子模块（注意力、FFN）都包裹在残差连接中。标准残差为 $x_{l+1} = x_l + f(x_l)$。当模型越来越深、参数越来越多之后，传统残差连接开始暴露缺陷——信号传递不稳定，训练容易崩溃。DeepSeek-V4 引入 mHC（流形约束超连接）强化残差连接（见 2.7）。

#### 2.2.6 并行注意力与 MLP

&emsp;&emsp;在标准 Transformer 块中，自注意力层和前馈网络是顺序执行的：先做注意力，再做 MLP，两者通过残差连接串联。Falcon 将两者改为并行执行：输入同时送入注意力分支和 MLP 分支，两个分支的输出相加后得到最终输出。这种并行化设计减少了串行计算链的长度，在训练时可以更充分地利用 GPU 的计算资源，在推理时也能降低延迟。TII 在论文中指出，并行注意力与 MLP 的设计受到 GPT-J 的启发，但在 Falcon 中得到了系统验证和规模化应用。

#### 2.2.7 统一系统与路由器

&emsp;&emsp;GPT-5 采用“统一系统”（unified system）复合设计，由一个高效的基础模型（gpt-5-main）、一个深度推理模型（gpt-5-thinking）和一个实时路由器组成。路由器根据对话类型、复杂度、工具需求和明确意图快速决定使用哪个模型。这一设计用路由器替代了单一模型，使简单查询快速响应，复杂问题自动切换到深度推理模式。路由器持续基于真实信号进行训练，包括用户切换模型的频率、回复偏好率和测量准确性，随时间不断优化。

---

### 2.3 位置编码

#### 2.3.1 绝对位置编码

&emsp;&emsp;Transformer 的注意力机制本身是置换不变的。为了注入位置信息，Transformer 在输入嵌入中加入了位置编码。绝对位置编码为序列中的每个位置分配一个唯一的表示。GPT-2、GPT-3 和早期的 BERT 使用可学习的绝对位置嵌入：为每个位置学习一个向量 $p_{pos}$，输入嵌入为 $h_0 = \text{TokenEmbed}(x) + \text{PositionEmbed}(pos)$。这种方案简单有效，但无法外推到训练时未见过的长度。

#### 2.3.2 正弦/余弦位置编码

&emsp;&emsp;原始 Transformer 使用固定公式生成位置编码：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

&emsp;&emsp;这种编码不需要训练参数，不同位置之间的相对关系可以通过线性变换表示，理论上可以外推，但实际外推效果有限。

#### 2.3.3 相对位置编码

&emsp;&emsp;相对位置编码不关心 token 的绝对位置，而是关心两个 token 之间的距离。这一设计更符合语言建模的直觉，且理论上支持长度外推。相对位置编码经历了从“学习标量偏置”到“旋转位置编码”的演进。

#### 2.3.4 T5 相对位置偏置

&emsp;&emsp;T5 为每一对相对位置偏移学习一个标量偏置，直接加到注意力权重上（pre-softmax）：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V
$$

&emsp;&emsp;T5 将所有可能的相对位置偏移映射到 32 个桶中，每个桶对应一个可学习的标量偏置。参数量极小，计算效率高，但表达能力有限。

#### 2.3.5 ALiBi

&emsp;&emsp;ALiBi（Attention with Linear Biases）不学习任何位置参数，而是直接根据 Query 和 Key 之间的距离施加一个线性偏置：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + m \cdot \text{distance}\right)V
$$

&emsp;&emsp;其中 $m$ 是每个注意力头固定的斜率。ALiBi 不引入可学习参数，训练时使用较短序列，推理时可以直接外推到更长序列。BLOOM 等模型采用过 ALiBi，但在 LLaMA 系列崛起后被 RoPE 取代。

#### 2.3.6 RoPE：旋转位置编码

&emsp;&emsp;RoPE 通过旋转矩阵将位置信息注入 Query 和 Key 向量，使注意力分数自然地携带相对位置信息。对于位置 $m$ 的 Query 向量 $q_m$ 和位置 $n$ 的 Key 向量 $k_n$，RoPE 对它们施加与位置相关的旋转：

$$
q_m' = R_m q_m, \quad k_n' = R_n k_n
$$

&emsp;&emsp;注意力分数为 $(q_m')^T k_n' = q_m^T R_{n-m} k_n$，只依赖于相对位置 $n-m$。在实际实现中，RoPE 将 $d$ 维向量分成 $d/2$ 对，每对施加一个二维旋转：

$$
\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} =
\begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

&emsp;&emsp;其中 $\theta_i = 10000^{-2i/d}$ 是频率。LLaMA 1/2/3、Qwen 2/2.5/3、Mistral、DeepSeek-V3、Kimi、GLM 等几乎所有开源模型都使用 RoPE 或其变体。

#### 2.3.7 RoPE 基频调整

&emsp;&emsp;RoPE 的长度外推能力可以通过调整基频来增强。LLaMA 2 使用基频 10,000，LLaMA 3.1 将基频提升到 500,000，使模型能够有效处理 32K 以上的上下文。基频越大，相邻位置的旋转角度差越小，模型对远距离位置的分辨能力越强。

#### 2.3.8 部分 RoPE

&emsp;&emsp;DeepSeek-V4 采用部分 RoPE，仅在每头的尾部通道上施加旋转，其余通道保持无位置编码，以平衡位置精度和语义匹配。注意力输出的 rope 切片也以位置 $-i$ 施加相同旋转。

#### 2.3.9 iRoPE：交错旋转位置编码

&emsp;&emsp;Llama 4 Scout 提出 iRoPE，交替使用带位置编码和不带位置编码的注意力层：偶数层使用 RoPE 处理局部位置信息，奇数层不使用位置编码（NoPE），专注于内容本身的语义匹配。这种交替设计使模型在超长上下文中既能利用位置信息进行精确检索，又能避免位置编码的数值不稳定性对语义匹配的干扰。

#### 2.3.10 NoPE

&emsp;&emsp;NoPE（No Position Encoding）即不使用位置编码的注意力层。Llama 4 的 iRoPE 中，奇数层不使用位置编码，专注于内容本身的语义匹配。NoPE 层在超长上下文中可以避免位置编码的数值不稳定性，但单独使用可能损失位置感知能力，通常与 RoPE 层交替使用。

#### 2.3.11 YARN

&emsp;&emsp;YARN 用于重新调整注意力权重以实现更好的长度外推。Qwen2 引入 DCA 和 YARN，使模型在推理时能够有效处理超出训练长度的序列，而无需在更长序列上进行继续训练。YARN 与 DCA 一起作为扩展模型上下文长度的标准手段。

#### 2.3.12 DCA：双块注意力

&emsp;&emsp;DCA（Dual Chunk Attention）将长序列分割为可管理的长度块，如果输入可以在一个块中处理，DCA 产生与原始注意力相同的结果；否则，DCA 在块内和跨块之间有效捕获相对位置信息。DCA 的一个重要特性是无需训练即可扩展上下文长度。它被应用于 Qwen2、Qwen2.5 以及 Qwen3，使 Qwen2.5-1M 系列首次将开源模型的上下文长度扩展至 100 万 token。

#### 2.3.13 温度缩放与分块注意力掩码

&emsp;&emsp;iRoPE 还包含温度缩放：根据序列长度动态调整注意力分布的“温度”，在序列变长时适当平滑注意力分布，防止其崩溃。在推理阶段，Llama 4 对特定层应用分块注意力掩码，将长序列分割为可管理的块进行处理，使模型能够在 10M token 的规模上保持内存效率。

---

### 2.4 归一化

#### 2.4.1 LayerNorm

&emsp;&emsp;LayerNorm 对激活值进行均值中心化和方差归一化。原始 Transformer 使用 Post-LN，GPT-1 沿用。LayerNorm 在 NLP 中比 BatchNorm 更稳定，因为序列长度可变，BatchNorm 的统计量不稳定。

#### 2.4.2 Post-LN

&emsp;&emsp;Post-LN 将层归一化放置在残差连接之后：$x_{l+1} = \text{LayerNorm}(x_l + \text{Sublayer}(x_l))$。这种设计在深层网络中会导致梯度范数在反向传播时出现剧烈波动，训练不稳定。

#### 2.4.3 Pre-LN

&emsp;&emsp;Pre-LN 将层归一化移到每个子模块的输入之前：$x_{l+1} = x_l + \text{Sublayer}(\text{LayerNorm}(x_l))$。Pre-LN 使残差路径上的信息可以不经过任何归一化变换而直接传递到深层，极大缓解了梯度消失和梯度爆炸问题。GPT-2 首次大规模采用，后续几乎所有主流 LLM 沿用。

#### 2.4.4 RMSNorm

&emsp;&emsp;RMSNorm 只对激活值的均方根进行缩放，省略了均值中心化操作。这一简化的代价极低，但在训练稳定性和计算效率上都有所提升。LLaMA 将 LayerNorm 替换为 RMSNorm，并沿用了 Pre-Norm 结构。Qwen3 的密集模型也包含带预归一化的 RMSNorm。

#### 2.4.5 GroupNorm

&emsp;&emsp;GroupNorm 在 DDPM 的 UNet 中替代 BatchNorm。因为在去噪采样时，Batch Size 通常很小甚至为 1，BatchNorm 在这种情况下极不稳定，而 GroupNorm 的表现非常鲁棒。Stable Diffusion 的 UNet 中，每个层级包含两个 ResNet 块，结构为 Conv → GroupNorm → SiLU → Conv。

#### 2.4.6 QK-Norm

&emsp;&emsp;Qwen3 在注意力机制中引入 QK-Norm，以确保训练稳定性。Qwen3 还移除了 Qwen2 中使用的 QKV-bias。QK-Norm 对 Query 和 Key 进行归一化，防止注意力 logit 在训练过程中过大，提升大规模训练的稳定性。

---

### 2.5 激活函数

#### 2.5.1 ReLU

&emsp;&emsp;原始 Transformer 的 FFN 使用 ReLU：$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$。T5 使用 ReLU 而非 GELU，虽然 T5 1.1 版本后来改用了 GeGLU。

#### 2.5.2 GELU

&emsp;&emsp;GPT-2 的前馈网络使用 GELU 激活函数。Gopher 使用 GeLU 激活函数和 SentencePiece 分词器。GELU 是 ReLU 的平滑近似，在 Transformer 中被广泛使用。

#### 2.5.3 SwiGLU

&emsp;&emsp;SwiGLU（Swish-Gated Linear Unit）是一种门控线性单元，通过两个并行的线性投影（一个内容投影和一个门控投影）进行逐元素相乘，再投影回模型维度。SwiGLU 使用三个投影矩阵而非两个，在同等参数量下提供了更强的表达能力。为了避免门控路径导致参数量膨胀，LLaMA 将 FFN 的中间维度调整为原始的 2/3。LLaMA 1/2/3、Mistral、Qwen、DeepSeek 等模型均采用 SwiGLU。

#### 2.5.4 GeGLU

&emsp;&emsp;GeGLU 是 GELU 的门控变体，与 SwiGLU 类似，但使用 GELU 作为门控激活函数。T5 1.1 版本使用了 GeGLU。

#### 2.5.5 SiTU-GLU

&emsp;&emsp;Kimi K3 的 Stable LatentMoE 使用 SiTU-GLU 与 Quantile Balancing 保持极高稀疏度下的训练稳定。

#### 2.5.6 Sqrt(Softplus(·))

&emsp;&emsp;DeepSeek-V4 的 MoE 部分将 affinity score 的激活函数从 Sigmoid 换成 Sqrt(Softplus(·))，去掉了 routing target nodes 的数量约束。

---

### 2.6 前馈网络与混合专家

#### 2.6.1 前馈网络（FFN）

&emsp;&emsp;前馈网络是一个两层 MLP，对每个位置独立应用：

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

&emsp;&emsp;中间维度通常为 $d_{\text{model}}$ 的 4 倍。FFN 为模型提供了非线性变换能力，是 Transformer 中参数量最大的部分之一。

#### 2.6.2 混合专家（MoE）

&emsp;&emsp;MoE 将 Transformer 块中的前馈网络替换为 MoE 层，每个 MoE 层包含多个独立的“专家”，每个专家本身是一个标准的 FFN。对于每一个输入 token，一个路由网络从专家中选择部分来激活，并将这些专家的输出加权求和作为最终输出。MoE 的核心价值在于解耦模型容量与推理成本：用巨大的总参数量承载知识，用极小的激活参数量控制计算开销。

#### 2.6.3 稀疏混合专家（SMoE）

&emsp;&emsp;SMoE 是 MoE 的稀疏激活版本。Mixtral 8x7B 是第一个在工程上成功落地、性能可与顶级稠密模型竞争、且完全开源的 MoE 语言模型。它总参数量 46.7B，每个 token 仅激活 13B 参数，推理成本与 13B 稠密模型相当，性能却在大多数基准上超越了 Llama 2 70B 和 GPT-3.5。

#### 2.6.4 专家

&emsp;&emsp;每个专家是一个独立的 FFN。DeepSeekMoE 采用细粒度专家和共享专家的组合策略。每个 token 被路由到多个小型专家，而非少数大型专家。部分专家被设为“共享专家”，所有 token 都必须经过这些专家处理，确保基础能力的稳定传递。Llama 4 的 MoE 层采用共享专家 + 路由专家的混合设计：Scout 配置 1 个共享专家和 16 个路由专家，Maverick 配置 1 个共享专家和 128 个路由专家。

#### 2.6.5 路由网络

&emsp;&emsp;Router 是一个可学习的线性层，输入是 token 的隐藏状态，输出是专家的概率分布。Router 的选择是 token 级别的，而非序列级别的——同一个序列中的不同 token 可以被路由到不同的专家。Mixtral 使用 top-2 选择机制，Llama 4 使用 Top-1 选择，DeepSeek-V3 使用细粒度专家和共享专家，Kimi K3 每个 token 从 896 个路由专家中激活 16 个。

#### 2.6.6 Top-k 路由

&emsp;&emsp;Top-k 路由选择概率最高的 k 个专家。Mixtral 使用 Top-2，Llama 4 使用 Top-1，Qwen3-235B-A22B 每个 token 激活 8 个专家，Kimi K3 激活 16 个。Top-1 路由的优势在于计算效率：每个 token 只触发一个路由专家的前向传播，进一步降低了推理计算量。

#### 2.6.7 共享专家

&emsp;&emsp;共享专家是所有 token 都必须经过的专家，确保基础能力的稳定传递。DeepSeekMoE 采用共享专家，Llama 4 也配置了 1 个共享专家。Qwen3-MoE 则舍弃了共享专家模块，并采用全局批次负载均衡损失技术促进专家专业化。

#### 2.6.8 细粒度专家

&emsp;&emsp;DeepSeekMoE 采用细粒度专家，将标准前馈网络替换为稀疏专家混合层。与 Mixtral 8x7B 的粗粒度专家设计不同，DeepSeekMoE 采用细粒度专家和共享专家的组合策略，使专家分配更加精细。

#### 2.6.9 负载均衡损失

&emsp;&emsp;负载均衡损失通过惩罚专家负载分布的方差来鼓励 Router 均匀分配 token，防止所有 token 都被路由到同一个专家。Mixtral 8x7B 使用了标准的负载均衡损失。

#### 2.6.10 无辅助损失负载均衡

&emsp;&emsp;DeepSeek-V3 完全去掉了辅助损失，改用一种基于动态偏置项的均衡策略，在几乎不影响主目标的前提下实现专家负载的动态均衡。论文报告这一策略在训练全程保持了专家负载的稳定分布，且未产生任何性能干扰。

#### 2.6.11 Hash routing

&emsp;&emsp;DeepSeek-V4 的前几层 dense FFN 换成了用 Hash routing 的 MoE 层。

#### 2.6.12 Stable LatentMoE

&emsp;&emsp;Kimi K3 的 MoE 部分采用 Stable LatentMoE：每个 token 从 896 个路由专家中激活 16 个，通过 SiTU-GLU 与 Quantile Balancing 保持极高稀疏度下的训练稳定。

#### 2.6.13 Quantile Balancing

&emsp;&emsp;Quantile Balancing 是 Kimi K3 用于保持极高稀疏度下训练稳定的技术，与 SiTU-GLU 配合使用。

---

### 2.7 残差连接与初始化

#### 2.7.1 标准残差连接

&emsp;&emsp;标准残差为 $x_{l+1} = x_l + f(x_l)$。残差连接使梯度可以直接回传，是深层网络训练的基础。

#### 2.7.2 残差路径缩放初始化

&emsp;&emsp;GPT-2 在初始化时对残差层的权重进行缩放：将残差分支中特定权重矩阵的标准差乘以 $1/\sqrt{N}$，其中 $N$ 是残差层的数量。以 GPT-2 XL 为例，$N = 48$，缩放因子约为 $1/6.93$。这意味着在训练开始时，每个残差分支的输出被显著压制，残差路径上的信号以近乎恒等映射的方式逐层传递，网络的初始行为接近于一个浅层模型。随着训练的进行，各层的残差权重逐渐增大，网络的有效深度逐步“生长”出来。这一初始化策略后来成为 GPT-3 及后续模型的标准做法。

#### 2.7.3 零初始化与零卷积

&emsp;&emsp;ControlNet 的核心设计是零卷积：一个 $1 \times 1$ 的卷积层，权重和偏置全部初始化为零。这意味着在训练的第一步，无论控制条件是什么，零卷积的输出都是零，ControlNet 对锁定副本的影响完全为零。LoRA 对两个低秩矩阵采用不对称初始化策略：矩阵 $A$ 使用高斯分布随机初始化，矩阵 $B$ 初始化为全零矩阵。在这一初始化下 $\Delta W = BA = 0$，模型的前向行为与未添加 LoRA 时的预训练模型完全一致。这种“从零开始、渐进生长”的机制，使训练过程天然稳定。

#### 2.7.4 mHC：流形约束超连接

&emsp;&emsp;DeepSeek-V4 引入 mHC（Manifold-Constrained Hyper-Connections），将残差流从一维变成 $n_{hc}$ 条并行通道，每层之间通过一个矩阵 $B$ 来混合。核心创新在于将矩阵 $B$ 约束到双随机矩阵的流形上（Birkhoff polytope），行和列都归一化为 1。这个约束带来两个关键好处：矩阵的谱范数天然不超过 1，残差传播有了硬上限，不会爆炸；这种矩阵在乘法下是封闭的，堆叠很多层仍然稳定。输入映射 $A$ 和输出映射 $C$ 通过 Sigmoid 函数保证非负且有界，避免信号互相抵消。

#### 2.7.5 双随机矩阵与 Sinkhorn-Knopp

&emsp;&emsp;双随机矩阵是行和列都归一化为 1 的矩阵。实现上使用 Sinkhorn-Knopp 迭代，交替做行归一化和列归一化，迭代 20 次收敛。DeepSeek 做了 fused kernel 配合选择性 recomputation，实测 mHC 带来的 wall-time 开销控制在 overlapped pipeline 的 6.7%。

---

### 2.8 分词器

#### 2.8.1 BPE

&emsp;&emsp;BPE（Byte Pair Encoding）通过迭代合并频率最高的字节对来构建词表。GPT-1 使用的 BPE 分词器以 Unicode 字符为基本单位，词表大小为 40,478，仍然存在一定数量的未登录词（OOV）。

#### 2.8.2 Byte-level BPE

&emsp;&emsp;GPT-2 将分词器升级为 Byte-level BPE：以 256 个字节作为基础字符集，通过 BPE 合并操作构建词表，最终词表大小为 50,257。Byte-level BPE 的关键优势在于：任何 Unicode 文本都可以被无损地表示为字节序列，因此词汇表中不存在任何 OOV token。GPT-2 的 Byte-level BPE 分词器被后续几乎所有 GPT 系列模型沿用。

#### 2.8.3 SentencePiece

&emsp;&emsp;Gopher 使用 SentencePiece 分词器。SentencePiece 将文本视为 Unicode 字符序列，支持直接从原始文本训练分词器，无需预分词。

#### 2.8.4 tiktoken

&emsp;&emsp;LLaMA 3 的词汇表从 LLaMA 2 的 32K 扩展到 128K token，使用 OpenAI 的 tiktoken 分词器开发。这一扩展显著提升了多语言场景下的分词效率——中文、日文、韩文等非拉丁语系的 token 压缩率大幅改善，同等文本所需的 token 数量减少，间接扩展了有效上下文长度。

#### 2.8.5 词表大小与多语言

&emsp;&emsp;Qwen2 的词表大小为 151,643 个常规词元加 3 个控制词元，所有规模的模型共享同一词汇表。Qwen2 的分词器对中文的压缩效率显著优于 LLaMA 系列的 128K 分词器。Qwen3 将多语言支持从 Qwen2.5 的 29 种扩展至 119 种语言及方言。

---

### 2.9 训练目标

#### 2.9.1 自回归语言建模

&emsp;&emsp;自回归语言建模是 GPT 系列和 LLaMA 系列的核心预训练目标：给定前文 token 序列，预测下一个 token。训练损失为交叉熵。GPT-3 的训练目标仍然是无监督的下一个 token 预测，模型从未见过“任务描述 + 示例 + 输出”这种格式的标注数据。

#### 2.9.2 掩码语言建模

&emsp;&emsp;BERT 使用掩码语言建模：随机遮蔽输入中的部分 token，训练模型根据上下文重建被遮蔽的 token。它提供双向上下文理解，适合理解类任务，但无法直接生成文本。

#### 2.9.3 Span Corruption

&emsp;&emsp;T5 的预训练目标是 span corruption，受 SpanBERT 启发。从输入文本中随机选择连续 token 跨度，将这些跨度替换为特殊的哨兵 token，然后训练模型根据上下文重建被替换的原始文本。具体参数为：破坏原始序列的 15%，每个被替换的跨度平均长度为 3 个 token。span corruption 在生成任务上表现优于独立同分布的掩码语言建模，且由于目标序列通常比输入序列短得多，训练的计算效率也更高。

#### 2.9.4 多 Token 预测（MTP）

&emsp;&emsp;DeepSeek-V3 要求模型在每个位置同时预测多个未来 token。这一目标通过完整的因果链增强了训练信号的密度，使模型能够更好地进行规划和推理。消融实验表明，MTP 策略持续提升了模型在大多数评估基准上的性能。MiMo-V2-Flash 也采用多层 MTP 推理加速技术。

#### 2.9.5 降噪自编码

&emsp;&emsp;T5 的 span corruption 是一种降噪自编码目标。UL2 在 T5 的基础上提出了 Mixture-of-Denoisers 目标，混合了 span corruption、极端 span corruption 和顺序 PrefixLM 三种降噪任务，进一步提升了预训练的通用性。

---

### 2.10 优化器与训练精度

#### 2.10.1 Adam

&emsp;&emsp;Adam 是常用的自适应优化器。Gopher 使用 Adam。Kimi K2 摒弃了传统的 Adam 优化器，创新性地使用了 Muon 优化器。

#### 2.10.2 AdamW

&emsp;&emsp;Chinchilla 将优化器从 Adam 替换为 AdamW，这一改动被报告为改善了语言建模损失和下游任务性能。DeepSeek-V4 将 V3 使用的 AdamW 替换为 Muon 优化器。

#### 2.10.3 Muon

&emsp;&emsp;Muon 最早由 Kimi 团队在大规模训练中验证，其在大规模训练中表现出更快的收敛速度和更好的训练稳定性，同时优化了梯度更新过程，减少了内存占用。DeepSeek-V4 采用 Muon 优化器替代 AdamW，接管绝大多数参数的训练。

#### 2.10.4 MuonClip

&emsp;&emsp;Kimi K2 在 Muon 基础上引入了 QK-clip 技术，形成 MuonClip 优化器，有效缓解了训练过程中的不稳定性问题。基于 MuonClip，K2 在 15.5 万亿 token 的数据上完成了训练。

#### 2.10.5 L-BFGS

&emsp;&emsp;Chinchilla 的缩放实验采用 L-BFGS 优化拟合损失函数的参数。

#### 2.10.6 FP8 混合精度训练

&emsp;&emsp;DeepSeek-V3 首次在超大规模模型上验证了 FP8 混合精度训练的有效性。通过精细的量化策略，模型在 FP8 精度下进行前向和反向计算，同时以 BF16 格式存储低精度优化器状态，显著提升了训练速度并降低了 GPU 内存占用。这一技术是 DeepSeek-V3 能够以 2.788M H800 GPU 小时完成 671B 模型训练的关键因素之一。

#### 2.10.7 BF16 与 INT4 量化

&emsp;&emsp;GPT-4o 通过 INT4 量化将模型体积缩小至原版的 1/8，同时保持 98% 的精度，使边缘设备上的本地运行成为可能。LLaMA 3.1 405B 使用 int8 量化需要约 400GB 显存。Mixtral 8x7B 可通过 int4 量化进一步压缩模型大小。

---

### 2.11 后训练与对齐

#### 2.11.1 监督微调（SFT）

&emsp;&emsp;InstructGPT 的 SFT 阶段使用约 13,000 条训练样例，模型在 GPT-3 的基础上进行监督微调，训练 16 个 epoch。Qwen2.5 的 SFT 阶段使用了超过 100 万个高质量指令样本。LLaMA 3.1 的 SFT 数据来源包括人工标注和合成数据。

#### 2.11.2 RLHF

&emsp;&emsp;RLHF（Reinforcement Learning from Human Feedback）由 InstructGPT 提出，三步框架为 SFT → 奖励模型 → PPO。RLHF 的目标不是提升模型的知识或推理能力，而是调整它的行为倾向：在多种可能的输出中，选择更符合人类偏好的那一种。

#### 2.11.3 奖励模型（RM）

&emsp;&emsp;奖励模型的任务是学习“什么样的输出更符合人类偏好”，并将其量化为一个标量奖励值。OpenAI 选择使用 6B 参数的模型作为奖励模型，而非 175B 版本。损失函数采用 pairwise 排序损失：

$$
\mathcal{L}_{\text{RM}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right]
$$

#### 2.11.4 PPO

&emsp;&emsp;PPO 阶段的目标是让 SFT 模型学会生成能够从奖励模型获得高分的输出。OpenAI 在目标函数中加入了逐 token 的 KL 散度惩罚，约束策略模型的输出分布不要偏离 SFT 模型太远：

$$
\text{objective}(\phi) = \mathbb{E}_{(x,y) \sim \pi_\phi^{RL}} \left[ r_\theta(x, y) - \beta \log \frac{\pi_\phi^{RL}(y|x)}{\pi^{SFT}(y|x)} \right]
$$

&emsp;&emsp;此外，OpenAI 还提出了 PPO-ptx 变体，在 PPO 的优化目标中额外加入预训练数据的梯度更新项，缓解“对齐税”。

#### 2.11.5 DPO

&emsp;&emsp;DPO（Direct Preference Optimization）将奖励模型和策略模型合并为一个目标函数，直接使用偏好数据优化策略模型，避免了 PPO 中奖励模型和策略模型交替训练的复杂性。DPO 的训练稳定性优于 PPO，且不需要在线生成样本，训练效率更高。Qwen2.5 使用 DPO 进行离线学习。

#### 2.11.6 GRPO

&emsp;&emsp;GRPO（Group Relative Policy Optimization）由 DeepSeek 在 DeepSeek-Math 中提出。其核心思想是：对于每个问题，从旧策略中采样一组输出，计算组内每个输出的奖励，然后用组内均值和标准差对奖励进行归一化，得到每个输出的优势值。与 PPO 不同，GRPO 不需要训练独立的价值网络（critic），而是用组内相对比较来估计优势。GRPO 在数学和代码等需要精确推理的任务上表现尤为突出。Qwen2.5 使用 GRPO 进行在线学习，MiMo-V2-Flash 也使用 GRPO 进行强化学习优化。

#### 2.11.7 拒绝采样

&emsp;&emsp;LLaMA 3.1 后训练第二步是拒绝采样。对于每个提示，模型生成多个候选回答，通过奖励模型或人工评估筛选出最优回答，将筛选后的数据用于下一轮微调。Meta 的一个关键创新是：在拒绝采样阶段，从多个使用不同超参数训练的 DPO 模型中选择表现最好的模型来生成 Prompt 的回答。

#### 2.11.8 RLAIF

&emsp;&emsp;Google 的 Gemini 在 RLHF 之外引入了基于 AI 反馈的强化学习（RLAIF）。

#### 2.11.9 宪法 AI 与自我纠正

&emsp;&emsp;Anthropic 的 Claude 系列在 RLHF 基础上提出了 Constitutional AI，用一组明确的规则替代部分人类标注。Claude 5 最核心的创新是宪法自我纠正机制：在模型训练的底层植入一套“宪法”原则，使模型学会根据宪法原则自我评估并修正回应。这一机制使模型具备实时自检与价值偏差修正的能力。

#### 2.11.10 可验证奖励

&emsp;&emsp;DeepSeek-R1 的奖励设计完全使用基于规则的奖励，不涉及任何神经奖励模型。奖励由两部分组成：准确性奖励和格式奖励。准确性奖励评估最终答案是否正确——在数学问题上，模型需要将最终答案以指定格式输出；在代码竞赛问题上，使用编译器对模型输出进行预定义测试用例的评估。这一奖励设计的核心优势在于不可欺骗性。

#### 2.11.11 思考预算

&emsp;&emsp;Qwen3 引入思考预算机制，允许用户在推理过程中自适应分配计算资源，从而根据任务复杂度平衡延迟与性能。

---

### 2.12 推理时计算与推理模型

#### 2.12.1 思维链（CoT）

&emsp;&emsp;o1 和 o3 系列将推理时的计算量从固定值变为可调变量，让模型在生成最终答案之前，先生成一条内部的、隐藏的“思维链”，在思维链中探索、验证和修正推理过程，然后仅将结论呈现给用户。

#### 2.12.2 隐藏思维链

&emsp;&emsp;o1 的思维链是隐藏的。用户看到的是模型最终输出的简洁答案，而非思维链本身。隐藏思维链使 OpenAI 能够对思维链内容进行安全审查和过滤，同时也构成竞争壁垒。DeepSeek-R1 则采用公开的思维链格式。

#### 2.12.3 推理时计算扩展

&emsp;&emsp;o1 和 o3 代表了推理时计算扩展（Test-Time Compute Scaling）。OpenAI 研究员 Noam Brown 指出：“让模型在一手牌中思考 20 秒，获得的提升相当于将模型规模和训练扩大 100,000 倍。”在 AIME 2024 上，o1 的单样本贪婪解码达到 74%，64 样本多数投票达到 83%，1000 样本重排序达到 93%。

#### 2.12.4 reasoning_effort

&emsp;&emsp;o3 引入了 reasoning_effort 参数，允许开发者按请求控制模型的“思考深度”，支持低、中、高三个计算层级。在 ARC-AGI 基准上，低计算层级得分 75.7%，高计算层级达到 87.5%。

#### 2.12.5 MCTS 与 Q*

&emsp;&emsp;o3 的核心创新是将 Q* 算法与蒙特卡洛树搜索（MCTS）深度结合，形成“预测-验证-优化”的闭环推理系统。与 o1 的线性思维链不同，o3 的推理过程更像一棵搜索树：模型在推理过程中探索多条候选路径，通过奖励信号评估每条路径的潜力，然后选择最有希望的路径深入展开。

#### 2.12.6 路由器与统一系统

&emsp;&emsp;GPT-5 采用统一系统，由 gpt-5-main、gpt-5-thinking 和实时路由器组成。路由器根据对话类型、复杂度、工具需求和明确意图快速决定使用哪个模型。路由器持续基于真实信号进行训练。

#### 2.12.7 推测解码

&emsp;&emsp;推测解码的核心思想是“先猜后验”：使用一个更小、更快的“草稿模型”先生成 K 个候选 token，然后让大模型对这 K 个 token 进行一次性验证。由于大模型只需要进行一次前向传播就能验证多个 token，整体推理速度可以提升 2 到 3 倍，且理论上不损失生成质量。

---

### 2.13 推理优化与加速

#### 2.13.1 KV 缓存

&emsp;&emsp;KV 缓存是自回归推理中缓存历史 Key 和 Value 的机制。MHA 的 KV 缓存大小与注意力头数成正比；MQA 将所有 Query 头共享同一组 KV 头；GQA 将 Query 头分组共享 KV 头；MLA 通过低秩压缩降低 KV 缓存的内存占用。

#### 2.13.2 滚动缓冲区缓存

&emsp;&emsp;Mistral 7B 配合滚动缓冲区缓存，可以在处理任意长度序列时保持恒定的显存占用。

#### 2.13.3 量化

&emsp;&emsp;GPT-4o 通过 INT4 量化将模型体积缩小至原版的 1/8，同时保持 98% 的精度。FP8 混合精度训练被 DeepSeek-V3 验证。LLaMA 3.1 405B 使用 int8 量化需要约 400GB 显存。

#### 2.13.4 FlashAttention

&emsp;&emsp;FlashAttention 通过分块计算和显存优化，将注意力操作的内存访问从二次方降低到线性，使长序列训练在合理显存下成为可能。Falcon 的训练和推理中大规模应用了 FlashAttention。

#### 2.13.5 分块注意力掩码

&emsp;&emsp;Llama 4 在推理阶段对特定层应用分块注意力掩码，将长序列分割为可管理的块进行处理，使模型能够在 10M token 的规模上保持内存效率。

---

### 2.14 长上下文技术

#### 2.14.1 上下文窗口扩展

&emsp;&emsp;从 GPT-2 的 1024 token 到 GPT-3 的 2048，再到 GPT-4 的 32K、LLaMA 3.1 的 128K、Gemini 1.5 Pro 的 100 万 token、Llama 4 Scout 的 1000 万 token，上下文窗口持续扩展。

#### 2.14.2 RoPE 基频调整

&emsp;&emsp;LLaMA 3.1 将 RoPE 基频从 10,000 提升到 500,000，使模型能够有效处理 32K 以上的上下文长度。

#### 2.14.3 YARN

&emsp;&emsp;YARN 用于重新调整注意力权重以实现更好的长度外推。Qwen2 引入 DCA 和 YARN，使模型在推理时能够有效处理超出训练长度的序列。

#### 2.14.4 DCA

&emsp;&emsp;DCA（Dual Chunk Attention）将长序列分割为可管理的长度块，在块内和跨块之间有效捕获相对位置信息。DCA 无需训练即可扩展上下文长度。

#### 2.14.5 iRoPE

&emsp;&emsp;Llama 4 的 iRoPE 交替使用带位置编码和不带位置编码的注意力层，使模型在超长上下文中既能利用位置信息进行精确检索，又能避免位置编码的数值不稳定性。

#### 2.14.6 稀疏与混合注意力

&emsp;&emsp;GLM-5 的 DSA、DeepSeek-V4 的 CSA + HCA、MiMo-V2 的 5:1 SWA + 全局注意力混合、Kimi K3 的 KDA + Gated MLA 3:1 混合，都是在保持长上下文理解能力的同时降低注意力计算量。

#### 2.14.7 温度缩放与分块注意力掩码

&emsp;&emsp;iRoPE 包含温度缩放，根据序列长度动态调整注意力分布的温度。Llama 4 在推理阶段对特定层应用分块注意力掩码。

---

### 2.15 多模态架构

#### 2.15.1 拼接式多模态

&emsp;&emsp;GPT-4 的多模态是“拼接式”的：视觉和文本模态通过独立的编码器处理，然后在某个中间层进行融合。LLaVA、Flamingo 等开源多模态模型也采用类似设计——用独立的视觉编码器（如 CLIP ViT）提取图像特征，再通过交叉注意力或投影层注入语言模型。

#### 2.15.2 早期融合

&emsp;&emsp;Llama 4 采用早期融合策略，将文本和视觉信息在模型骨干的初始处理阶段即整合到统一表示空间。文本 token 和视觉 token 在进入 Transformer 的第一层之前就处于同一表示空间中，所有 Transformer 层都能同时“看到”两种模态的信息。

#### 2.15.3 端到端统一

&emsp;&emsp;GPT-4o 是端到端训练的单一模型，文本、视觉和音频的所有输入和输出都由同一个神经网络处理。其核心机制是动态跨模态注意力，通过模态掩码矩阵让模型在推理时动态决定不同模态特征的权重。

#### 2.15.4 跨模态注意力

&emsp;&emsp;GPT-4o 的动态跨模态注意力让模型在推理时动态决定不同模态特征的权重。Grok 3 在 CM3leon 架构基础上引入了跨模态注意力对齐机制、量子化语义编码器和时空连续性建模模块。

#### 2.15.5 视觉编码器

&emsp;&emsp;CLIP 采用双塔架构，包含一个图像编码器和一个文本编码器，两者相互独立但输出被投影到同一个高维语义空间。Kimi K3 的视觉编码器 MoonViT-V2 从零开始使用 next-token prediction 训练，无需对比预训练，在达到 SigLIP 初始化基线效果的同时获得了更稳定的优化过程。

---

### 2.16 缩放规律

#### 2.16.1 Kaplan Scaling

&emsp;&emsp;Kaplan 等人（2020）的缩放定律认为，当计算预算增加 10 倍时，模型参数量应增加约 5.5 倍，而训练 token 数只需增加约 1.8 倍。这一结论直接指导了 GPT-3、Gopher、Jurassic-1 等模型的设计。

#### 2.16.2 Chinchilla-optimal

&emsp;&emsp;Chinchilla 论文通过对超过 400 个语言模型的系统实验，重新检验了 Kaplan 的结论。核心发现是：对于计算最优的训练，模型大小和训练 token 数应当等比例缩放——模型参数量每翻一倍，训练 token 数也应当翻一倍。Chinchilla 用 700 亿参数和 1.4 万亿 token，在相同计算预算下全面超越了 2800 亿参数的 Gopher。

#### 2.16.3 训练时计算扩展

&emsp;&emsp;GPT-3、Gopher、Chinchilla 等模型的核心优化方向是训练时效率——在给定计算预算下，如何通过数据规模、架构选择和训练流程优化来最大化模型能力。

#### 2.16.4 推理时计算扩展

&emsp;&emsp;o1 和 o3 的核心优化方向是推理时效率——在给定训练成本下，如何通过增加推理时的计算量来“解锁”更深层的推理能力。从 Kaplan Scaling 到 Chinchilla-optimal，再到 Inference-optimal，最终到 Test-time compute，LLM 的缩放范式从单一的训练规模维度逐步扩展到了训练效率、数据效率和推理计算的多维度优化。

#### 2.16.5 Inference-optimal

&emsp;&emsp;在推理成本成为主要瓶颈的场景下，继续扩大数据规模仍然有效，甚至比扩大参数量更具性价比。Qwen2.5-72B 以 LLaMA 3.1 405B 约五分之一的参数量，在多项基准上展现出与后者相当的性能。

---

### 2.17 并行与分布式训练

#### 2.17.1 数据并行

&emsp;&emsp;数据并行将同一模型复制到多个设备，每个设备处理不同批次的数据，然后汇总梯度。它是分布式训练的基础。

#### 2.17.2 张量并行

&emsp;&emsp;张量并行将单个 Transformer 层的权重矩阵切分到多个设备上。GPT-4 训练采用了 8 路张量并行，因为这是 NVLink 的限制——A100 的 NVLink 带宽支持 8 路全互联，超过 8 路后通信效率显著下降。

#### 2.17.3 流水线并行

&emsp;&emsp;流水线并行将模型的不同层放置在不同的设备上，数据以微批次的形式在设备之间流水线式传递。Falcon-180B 训练使用了 3D 并行策略（张量并行 + 流水线并行 + 数据并行）和 ZeRO 优化器。

#### 2.17.4 3D 并行

&emsp;&emsp;3D 并行同时使用张量并行、流水线并行和数据并行。Falcon-180B 在 4096 张 GPU 上使用 3D 并行策略和 ZeRO 优化器来管理 180B 参数的分布式训练。

#### 2.17.5 ZeRO

&emsp;&emsp;ZeRO 通过分片优化器状态、梯度和参数来降低单设备显存占用。Falcon-180B 训练使用了 ZeRO 优化器。

#### 2.17.6 专家并行

&emsp;&emsp;MoE 训练中，不同的专家通常被放置在不同的设备上，token 的路由需要跨设备通信。Kimi K3 开源了 MoonEP，为超大的细粒度 MoE 打造的高性能通信库，让专家并行的通信在不均衡的情况下仍然实现极致效率。

#### 2.17.7 MoonEP / FlashKDA / AgentEnv

&emsp;&emsp;Kimi K3 同步开源了三项支撑模型训练的关键 Infra 技术：MoonEP（高性能通信库）、FlashKDA（Kimi Delta Attention 的高性能算子，在英伟达 H20 上 prefill 速度提升 1.72-2.22 倍）、AgentEnv（与 KVCache.ai 合作开发的沙箱系统，用于大规模运行 Agent 环境）。

---

### 2.18 数据工程与训练方法

#### 2.18.1 WebText

&emsp;&emsp;GPT-2 的训练数据 WebText 是从 Reddit 爬取构建的高质量网页文本数据集，约 40GB。设计哲学是“质量优先于数量”。

#### 2.18.2 C4

&emsp;&emsp;T5 构建并开源了 Colossal Clean Crawled Corpus（C4），约 750GB 的高质量英文网页文本数据集。C4 从 Common Crawl 的 2019 年 4 月快照中提取，经过多步清洗。

#### 2.18.3 MassiveText

&emsp;&emsp;Gopher 使用了 MassiveText，一个约 2.35 万亿 token 的英文为主语料库，来源涵盖网页、书籍、新闻、源代码、Wikipedia 和 C4。

#### 2.18.4 RefinedWeb

&emsp;&emsp;TII 构建了 RefinedWeb——一个完全基于 Common Crawl、经过严格过滤和去重的网页数据集，包含约 9.68 亿个网页，总计 2.8TB 的干净文本数据，约 5000 亿到 6500 亿 token。仅用 RefinedWeb 训练的模型在多个基准上达到或超过了 curated 混合语料训练的模型。

#### 2.18.5 LAION-5B

&emsp;&emsp;Stable Diffusion 使用了 LAION-5B 作为训练数据。

#### 2.18.6 MinHash 去重

&emsp;&emsp;RefinedWeb 使用 MinHash 进行模糊去重：为每篇文章计算 9000 个哈希值，使用 20 个桶、每桶 450 个值进行筛选，最后移除重复片段超过 50 个 token 的文章。

#### 2.18.7 质量过滤

&emsp;&emsp;LLaMA 3 数据预处理采用了基于启发式规则和基于模型的双重质量过滤策略：fastText 分类器用于语言识别和低质量内容过滤，基于 RoBERTa 的分类器用于判断内容的教育价值和信息密度。

#### 2.18.8 合成数据与蒸馏

&emsp;&emsp;LLaMA 3 代码训练采用三种合成数据方法：代码执行反馈、编程语言翻译、文档反向翻译。LLaMA 3.1 采用模型族蒸馏策略：用旗舰 405B 模型的输出来优化 8B 和 70B 小模型。DeepSeek-R1 蒸馏出六个小模型，将推理能力迁移到 1.5B 到 70B 的参数量范围。

#### 2.18.9 代码执行反馈

&emsp;&emsp;让模型生成代码，通过单元测试验证正确性后微调。

#### 2.18.10 编程语言翻译

&emsp;&emsp;将 Python 代码翻译为其他语言以扩充低资源语言的数据。

#### 2.18.11 文档反向翻译

&emsp;&emsp;从代码生成注释和文档，再反向生成代码。

---

### 2.19 模型架构范式

#### 2.19.1 Encoder-only

&emsp;&emsp;BERT 代表，双向注意力，擅长理解类任务。

#### 2.19.2 Decoder-only

&emsp;&emsp;GPT 系列、LLaMA 系列、Mistral、Qwen、DeepSeek、Kimi、GLM 等，因果注意力，擅长生成类任务，是当前 LLM 的主流范式。

#### 2.19.3 Encoder-Decoder

&emsp;&emsp;原始 Transformer、T5、BART 代表，编码器双向理解输入，解码器自回归生成输出。

#### 2.19.4 MoE

&emsp;&emsp;MoE 将 FFN 替换为多个专家，通过路由网络为每个 token 选择性地激活部分专家。MoE 已成为超大规模模型的主流选择。

#### 2.19.5 统一系统

&emsp;&emsp;GPT-5 采用统一系统，由快速模型、深度推理模型和实时路由器组成。

#### 2.19.6 双轨发布

&emsp;&emsp;Claude 5 包含 Claude Fable 5（安全对齐版）和 Claude Mythos 5（全能力版）双轨版本。

---

### 2.20 关键概念与术语

#### 2.20.1 零样本

&emsp;&emsp;GPT-2 首次展示了零样本多任务学习能力：模型在没有任何显式监督的情况下识别并执行训练语料中自然出现的各种任务。

#### 2.20.2 上下文学习

&emsp;&emsp;GPT-3 系统展示并命名了上下文学习能力：模型在推理时接收一个“提示”，提示中包含任务描述和若干示例，模型直接输出答案，不进行任何梯度更新或微调。

#### 2.20.3 涌现能力

&emsp;&emsp;GPT-2 论文中首次系统记录了“涌现”现象：某些能力在规模达到阈值之前完全不可观测，之后突然出现。

#### 2.20.4 对齐税

&emsp;&emsp;RLHF 训练会导致模型在标准 NLP 基准上的性能退化，PPO-ptx 虽然缓解了这一问题，但并未完全消除。

#### 2.20.5 奖励黑客

&emsp;&emsp;奖励模型是对人类偏好的近似，而非人类偏好的精确表达。当策略模型学会利用奖励模型的缺陷来获取高分时，就会产生“奖励黑客”行为。

#### 2.20.6 灾难性遗忘

&emsp;&emsp;ControlNet 的训练过程中，零卷积的初始化策略和锁定副本的设计确保训练初期 ControlNet 的输出为零，原始 UNet 完全按照预训练的方式工作，模型不会经历“突然被大量噪声干扰”的灾难性遗忘。

#### 2.20.7 双塔架构与 InfoNCE

&emsp;&emsp;CLIP 采用双塔架构，包含一个图像编码器和一个文本编码器。对比学习预训练使用 InfoNCE 损失，本质上是把图文匹配任务转化成了一个批内检索问题。温度参数 $\tau$ 初始化为 0.07，可学习，控制相似度分布的尖锐程度。

#### 2.20.8 缩放规律

&emsp;&emsp;语言建模损失随模型规模、数据量和训练计算量的增加呈平滑的幂律下降。

#### 2.20.9 计算最优训练

&emsp;&emsp;Chinchilla 论文中提出的“计算最优训练”概念，使 LLM 的开发从“尽可能大”转向“在给定预算下尽可能优”。

#### 2.20.10 推理时计算扩展

&emsp;&emsp;从单一的训练时计算扩展到了“训练时 + 推理时”的联合优化。

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

