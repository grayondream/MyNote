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

![](https://github.com/snu-mllab/Context-Memory/blob/main/image/main.png  )

### 2.2 源码分析
润色下