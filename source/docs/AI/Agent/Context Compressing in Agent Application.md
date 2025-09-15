# Context Compressing in Agent Application
## 1 IN-CONTEXT AUTOENCODER FOR CONTEXT COMPRESSION IN A LARGE LANGUAGE MODEL
>论文地址：[https://arxiv.org/pdf/2307.06945  ](https://arxiv.org/pdf/2307.06945  )
>代码：[https://github.com/getao/icae/tree/main  ](https://github.com/getao/icae/tree/main  )

### 1.1 模型结构
&emsp;&emsp;现有大语言模型面临几个核心问题：
- 大语言模型（LLMs）通常受限于上下文长度（context window）——输入太长就会导致计算资源消耗高、内存需求大。
- 多数解决方案是改变模型架构（比如稀疏注意力、滑动窗口、长距离注意力等），但这些方案在上下文长度极长时仍然性能下降。

&emsp;&emsp;为了解决这个问题，论文提出了一个不同思路：把长的上下文压缩成一个短的“记忆槽”（memory slots），只让模型在推理时看到这些压缩后的记忆槽，而非完整上下文，从而节省计算与内存开销。

![](https://github.com/getao/icae/raw/main/icae_demo.png  )
&emsp;&emsp;ICAE 的设计可以分成两部分：编码器（encoder）、解码器（decoder）。
- 编码器
  - 基于已有的 LLM，使用 LoRA（Low-Rank Adaptation）方式做轻量调整，以及新增 memory tokens 的 embedding。 
  - 输入一个完整上下文（长度为 L 的 token 序列 w₁…w_L），在其后附加 k 个 memory tokens m₁…m_k（k ≪ L），然后通过编码器处理，输出这 k 个 memory slots（mf₁ … mf_k）。这些 memory slots 是对整个上下文的压缩表示。 
- 解码器
  - 解码器就是保持不变的目标 LLM，其作用是“读取”这些 memory slots（加上某些额外 prompt 或任务输入）来完成任务，比如重构原文、继续文本、回答问题等。也就是说，解码器并不看到完整上下文，只看到压缩后的 memory slots。 

&emsp;&emsp;训练分为两个步骤：
- 预训练：双目标优化
  - 通过海量文本（如 The Pile）训练，让记忆槽具备 “恢复上下文” 和 “泛化表征” 能力：
  - **自编码目标（$\mathcal{L}_{AE}$）**：从记忆槽中恢复原上下文 $c$，即最大化 $P(c|\tilde{m_1},...,\tilde{m_k};\Theta_{LLM})$，需在记忆槽后追加特殊 token“[AE]” 标识任务。该目标无需标注，可利用海量数据，确保记忆槽保留上下文核心信息。
  - **文本续写目标（$\mathcal{L}_{LM}$）**：从记忆槽中预测上下文的续写内容 $o=(w_{L+1},...,w_{L+N})$，即最大化 $P(o|\tilde{m_1},...,\tilde{m_k};\Theta_{LLM})$。缓解单一 AE 目标的过拟合问题，提升记忆槽的泛化能力。
  - 预训练损失为双目标加权：$\mathcal{L}_{pretrain}=\lambda\mathcal{L}_{AE}+(1-\lambda)\mathcal{L}_{LM}$，其中 $\lambda=0.4\sim0.6$ 时效果最优。

- 指令微调：适配实际任务
为让记忆槽能与多样化 prompt 交互，使用自建的 PWC 数据集（Prompt-with-Context）微调：
  - **PWC 数据集**：含 24 万训练样本、1.8 万测试样本，每个样本为（上下文、prompt、响应）三元组，由 GPT-4 生成，覆盖不同上下文长度（多数超 500token）与任务类型（摘要、问答、关键词提取等）。
  - **微调目标**：最大化 “基于记忆槽 + prompt 生成正确响应” 的概率，即 $\mathcal{L}_{FT}=\max_{\Theta_{LoRA},e_m}P(r_1...r_n|m_1...m_k,p_1...p_m;\Theta_{LLM})$，增强记忆槽的任务适配性。

### 1.2 代码分析
&emsp;&emsp;从模型上看，ICAE 是基于已有的 LLM 构建的，具体来说，是在 LLM 基础上应用了 LoRA 技术，新增了 memory tokens 的 embedding。
```python
self.icae = base_llama.LlamaForCausalLM.from_pretrained(...)  # 基础LLM
self.icae = get_peft_model(self.icae, lora_config)  # 应用LoRA
```
&emsp;&emsp;同时通过新增的 memory tokens 嵌入层，模型可以学习到 memory slots 的表示。
```python
# 记忆token嵌入层（可学习参数）
self.memory_token_embed = nn.Embedding(model_args.mem_size+3, self.dim)

# 生成记忆槽
compress_outputs = self.icae(inputs_embeds=autoencoder_input_embedding, enable_lora=True)
memory_embedding = compress_outputs[memory_mask].view(batch_size, self.model_args.mem_size, -1)
```
&emsp;&emsp;生成阶段直接复用基础 LLM 的解码器，将记忆槽与 prompt 拼接后作为输入，无需修改原模型结构，确保兼容性。
```python
# 拼接记忆槽和prompt嵌入
decoder_input_embeddings = torch.cat((memory_embedding, prompt_answer_embs), dim=1)
# 调用原LLM生成响应
decoder_outputs = self.icae(inputs_embeds=decoder_input_embeddings)
```

## 2 Compressed Context Memory For Online Language Model Interaction
>论文地址：[https://arxiv.org/abs/2312.03414  ](https://arxiv.org/abs/2312.03414  )
>代码：[https://github.com/snu-mllab/context-memory  ](https://github.com/snu-mllab/context-memory  )

### 2.1 模型
&emsp;&emsp;现有LLM交互中面临几个问题：
- 在在线交互（对话、个性化、多任务学习等）中，语言模型的上下文随着时间持续累积。Transformer 的 self-attention 机制对完整上下文（context）的 key/value (KV) 缓存与计算开销随着时间线性或更坏地增长，导致内存使用大、延迟高、吞吐低。
- 现有的上下文压缩方法大多是针对固定长度 context（prompt / 演示集）设计的，不太适合上下文随着时间不断新增、需要动态处理的情形。

&emsp;&emsp;论文提出了一种叫 Compressed Context Memory (CCM) 的机制，用来在 inference 时动态压缩累积的 context KV，节省资源，同时尽量保持性能。
![](https://github.com/snu-mllab/Context-Memory/blob/main/image/main.png?raw=true)

&emsp;&emsp;CCM提出动态压缩上下文内存框架，通过持续压缩积累的 KV 对，在有限内存中支持在线推理，核心包含 3 大模块：
- 上下文压缩机制：基于⟨COMP⟩token 的 KV 压缩
  - 压缩对象：直接压缩注意力 KV 对（而非 token 嵌入），兼容 Transformer 层内并行性；
  - 专用压缩 token：引入⟨COMP⟩token，训练模型将当前新增上下文\(c(t)\)与历史压缩内存\(Mem(t-1)\)的信息，压缩到⟨COMP⟩token 的 KV 对中，得到压缩特征\(h(t) \in \mathbb{R}^{2 \times L \times d}\)（L为模型层数，d为隐藏层维度）；
  - 压缩过程：\(h(t) = g_{comp}(Mem(t-1), c(t))\)，仅需前向计算，无需递归。
- 动态内存更新，设计可微、可并行的内存更新函数\(g_{update}\)，适配不同场景：
  - CCM-concat：将新的 h(t) 直接连接到旧的 memory 中，memory 随时间增长。
  - CCM-merge：将新的 h(t) 与旧 memory 按加权平均合并（例如算术平均或指数移动平均），memory 保持固定大小。
- 高效训练：并行化 + 条件 LoRA
  - 并行化训练：递归压缩过程 “展开” 为单轮并行前向计算，避免递归导致的训练耗时与梯度传播误差，训练速度比 RMT/AutoCompressor 快 7 倍；
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/cmm_KV_PARALLEL.png)
  - 条件 LoRA:为了训练压缩模块，引入一个 conditional LoRA adapter，只在 ⟨COMP⟩ token 上作用，不修改原模型的主参数 θ，而新增一些低秩参数 ∆θ 用来学习压缩操作。这样避免 overfitting 模型在没有 context 的情况下也能“猜”输出的问题。

&emsp;&emsp;推理阶段：
- 在实际用的时候，随着新 context 的到来，不断用压缩函数更新 memory，并在生成响应时只用 memory + 当前输入，而不是整个历史 context。这样 attention 的计算成本和内存成本都大幅下降。
arXiv
- 在 streaming 的设置里，还结合 sliding window（最近的上下文窗口）与压缩 oldest tokens 的操作来控制 KV cache 的大小。

### 2.2 源码分析
&emsp;&emsp;在注意力机制中通过 sum_mask 和 sum_attn_mask 实现键值对的动态合并与压缩。
```python
# 合并压缩键值对（用于动态更新内存）
if sum_attn_mask is not None:
    sum_attn_mask = sum_attn_mask.to(key_states.dtype)
    # 计算压缩后的键值平均值
    key_comp_avg = torch.matmul(sum_attn_mask.unsqueeze(1), key_states)
    value_comp_avg = torch.matmul(sum_attn_mask.unsqueeze(1), value_states)
    # 用掩码控制原始键值与压缩键值的融合
    no_sum_mask = (1 - sum_mask).to(key_states.dtype).unsqueeze(1).unsqueeze(-1)
    key_states = no_sum_mask * key_states + key_comp_avg  # 动态更新key内存
    value_states = no_sum_mask * value_states + value_comp_avg  # 动态更新value内存
```

&emsp;&emsp;在注意力层中，q_proj、k_proj、v_proj、o_proj 均使用 LinearMask，并在 forward 中传入 comp_mask 控制 LoRA 的条件激活：
```python
class LinearMask(nn.Linear):
    """Linear function with compression mask as an argument. 
       The mask is used for conditional LoRA at src/peft_custom/lora.py-Linear()-forward().
    """

    def forward(self, input: Tensor, comp_mask=None) -> Tensor:
        return F.linear(input, self.weight, self.bias)


self.q_proj = LinearMask(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = LinearMask(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = LinearMask(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = LinearMask(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                               max_position_embeddings=self.max_position_embeddings)
```

## 3 LoCoCo: Dropping In Convolutions for Long Context Compression
>论文地址：[https://arxiv.org/pdf/2406.05317](https://arxiv.org/pdf/2406.05317)
>代码地址：[https://github.com/VITA-Group/LoCoCo](https://github.com/VITA-Group/LoCoCo)

### 3.1 模型
&emsp;&emsp;大型语言模型（LLMs）在文本生成、代码合成、问答等任务中表现优异，但处理长上下文序列时面临KV 缓存内存瓶颈：Transformer 的注意力计算需缓存键（Key）和值（Value），其大小随上下文长度线性增长，易耗尽 GPU 内存。现有方法存在明显缺陷：
- StreamingLLM：通过 “注意力 sink” 和局部窗口限制 receptive field，会丢失中间上下文（“lost in the middle”），无法利用全序列信息；
- H₂O：基于注意力分数启发式丢弃 KV 对，难以扩展到训练上下文长度以外的序列；
- LongLoRA：需修改预训练 LLM 架构，且在小上下文场景下性能受损；
- 注意力近似方法（如稀疏注意力、低秩近似）：无法缓解推理阶段的 KV 缓存爆炸问题。

&emsp;&emsp;LoCoCo 的核心目标是用固定大小 KV 缓存实现长上下文高效处理，兼顾推理与微调阶段，且支持 “即插即用”（无需修改原有 LLM 架构），核心设计包括三部分：
- 数据驱动的自适应 KV 融合:区别于传统 “启发式丢弃 KV 对”，LoCoCo 通过1D 卷积核动态计算混合权重，融合历史 KV 缓存与新输入 token，最小化上下文信息丢失，保证注意力建模准确性。
- 分段注意力与卷积压缩：
  - 分段注意力（Segment-Level Attention）:将长序列划分为长度为 B 的段，逐段处理：
    - 每段生成当前的 Q、K、V；
    - 注意力计算时结合历史 KV 缓存与当前段 KV；
    - 若累计 KV 长度超过固定缓存大小 M，触发卷积压缩。
  - 卷积 token 压缩器,通过 1D 卷积层生成融合权重:
    - 输入：历史 KV 缓存（维度 d×M）与当前段 KV（维度 d×B）拼接，共 2d×(M+B) 维度；
    - 卷积层输出 M×(M+B) 维度的权重矩阵 W，经归一化后得到历史 KV 的权重\(\tilde{w}_{i,j}\)和当前 KV 的权重\(w_{i,j}\)；
    - 融合更新：\(\tilde{k}_i = \sum_j w_{i,j}k_j + \sum_j \tilde{w}_{i,j}\tilde{k}_j\)（V 的更新同理），最终缓存大小保持为 M。
- 即插即用集成:
  - 推理阶段：在预训练 LLM 的注意力层顶部插入卷积压缩器，仅用少量校准数据微调压缩器，预训练权重不变，可随时移除压缩器切换回全序列模式；
  - 微调阶段：结合位置插值（positional interpolation）扩展上下文，插入卷积压缩器并添加 LoRA 适配器，仅微调压缩器、LoRA 及嵌入 / 归一化层，无需全量微调。

### 3.2 源码分析
&emsp;&emsp;通过卷积层动态学习融合权重，结合残差权重实现自适应融合。
- 卷积压缩器定义：
```python
layers = []
hidden_size, in_size = self.mem_compress * args.expand, dim_kv  # dim_kv = 2*head_dim（K+V）
for i in range(args.n_convlayer):
    if i != 0:
        in_size = hidden_size
    if i == args.n_convlayer-1:
        hidden_size = self.mem_compress  # 最终输出mem_compress个权重
    # 1D卷积层（论文中的核心组件，用于学习融合权重）
    layers.append(nn.Conv1d(
        in_channels=in_size, 
        out_channels=hidden_size, 
        kernel_size=args.kernel_size, 
        padding=int((args.kernel_size-1)//2)  # 保持序列长度
    ))
    if args.hidden_act is not None:
        layers.append(ACT2FN[args.hidden_act])  # 非线性激活（如ReLU）
self.layers = nn.Sequential(*layers)  # 卷积网络用于生成融合权重
```
- 自适应权重融合（结合卷积权重与残差权重）：
```python
# 卷积网络生成权重（数据驱动部分）
x = torch.cat([non_heavy_hitter_key_states, non_heavy_hitter_value_states], dim=-1)  # 拼接K和V（2d通道）
x = rearrange(x, 'b h s d -> (b h) d s', b=bsz, h=n_head)  # 适配Conv1d输入格式
x = self.layers(x)  # 卷积计算权重
x = nn.functional.softmax(x, dim=-1)  # 归一化（论文公式7）
weight = rearrange(x, '(b h) m s -> b h m s', b=bsz)  # 恢复维度

# 融合残差权重（确保重要信息不丢失）
residual_weight = self.make_residual_weight(non_heavy_hitter_hh_scores)  # 基于注意力分数的残差权重
weight = residual_weight * (1-self.normalizer) + weight * self.normalizer  # 自适应融合（论文未明确但代码新增的稳定机制）
```

&emsp;&emsp;模块化设计`memory_saver`，通过独立接口处理 KV 缓存，不侵入原模型结构。
&emsp;&emsp;通过固定缓存大小、划分重令牌与非重令牌，实现低内存开销的长序列处理。
- 固定缓存大小控制
```python
self.mem_size = args.mem_size  # 总缓存大小（固定值，如512）
self.mem_compress = int(args.mem_size * (1-args.hh_keep_rate))  # 可压缩部分大小
self.keep_hh_size = self.mem_size - self.mem_compress  # 保留的重令牌大小
```
- 重令牌保留 + 非重令牌压缩（降低计算开销）
```python
def partition_past_key_values(self, past_key_states, past_value_states, hh_scores):
    # 1. 划分重令牌（heavy hitter）：保留注意力分数高的token
    _, hh_idxs = torch.topk(hh_scores[..., :-self.local_len], self.keep_hh_size-self.local_len, dim=-1)
    mask_bottom = zeros.scatter(-1, hh_idxs, True)  # 重令牌掩码
    if self.local_len > 0:
        mask_bottom[:, :, -self.local_len:] = True  # 额外保留最近的local_len个token（局部性保障）
    
    # 2. 分离重令牌KV和非重令牌KV
    heavy_hitter_key_states = torch.masked_select(past_key_states, mask_bottom).reshape(...)  # 直接保留，不压缩
    non_heavy_hitter_key_states = torch.masked_select(past_key_states, ~mask_bottom).reshape(...)  # 需要压缩
    return (heavy_hitter_key_states, ...), (non_heavy_hitter_key_states, ...)
```
## 4 Selective Context / “Compressing Context to Enhance Inference Efficiency”