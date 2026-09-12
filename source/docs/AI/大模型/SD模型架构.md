# SD模型发展历程

&emsp;&emsp;自2022年Stable Diffusion开源以来，文本到图像生成领域经历了前所未有的技术爆发。作为潜在扩散模型（Latent Diffusion Model, LDM）最具代表性的实现，Stable Diffusion将高维像素空间的扩散过程迁移至计算效率更高的低维潜在空间，在保持生成质量的前提下将显存占用与计算量降低超过1000倍，首次让消费级GPU用户能够实现本地部署的文生图能力。

&emsp;&emsp;SD模型发展至今也不仅仅是单一模型的成功，而是不断迭代升级，不断提升质量。时至今日几乎重塑了AI绘画的生态。为了更加清晰的理解AI绘画模型的架构，本文从架构演进、性能优化和生态落地多个维度数理SD的技术发展脉络。

## 1 DDPM

- 论文地址：[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)

&emsp;&emsp;DDPM 的核心思想是：先把图像一步步加噪，直到近似变成纯高斯噪声；再训练一个神经网络学习“倒放”这个过程，从噪声一步步去噪，最终生成图像。DDPM 的主要贡献包括：

1. 证明扩散模型可以生成高质量图像；
2. 提出简单的“预测噪声”参数化；
3. 建立扩散模型与去噪分数匹配、退火 Langevin 动力学之间的联系；
4. 给出简化的训练目标，稳定且效果好；
5. 从有损压缩和自回归角度解释扩散模型的行为。

**前向过程：把图像逐渐变成噪声**

&emsp;&emsp;设真实图像为 $x_0$。前向过程是一个固定的马尔可夫链，每一步按方差 $\beta_t$ 加入高斯噪声：

$$
q(x_t|x_{t-1})
=
\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I)
=
\mathcal{N}(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I)
$$

其中 $\alpha_t=1-\beta_t$。它有两个关键特点：

- 没有可学习参数，完全固定；
- 任意时刻 $x_t$ 可以闭式采样：

$$
q(x_t|x_0)
=
\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)
$$

其中

$$
\bar{\alpha}_t=\prod_{s=1}^t\alpha_s
$$

等价地：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,
\quad
\epsilon\sim\mathcal{N}(0,I)
$$

&emsp;&emsp;当 $T=1000$、$\beta_t$ 从 $10^{-4}$ 线性增加到 $0.02$ 时，$x_T$ 已经非常接近标准高斯噪声 $\mathcal{N}(0,I)$。

---

**逆向过程：学习一步步去噪**

&emsp;&emsp;DDPM 学习一个逆向马尔可夫链：

$$
p_\theta(x_{t-1}|x_t)
=
\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))
$$

&emsp;&emsp;对于 $t=1$ 的边界情况，采样时通常只取均值、不加噪声。

&emsp;&emsp;DDPM 不让网络学习方差，而是把方差固定为不随时间学习的超参数，例如：

$$
\Sigma_\theta(x_t,t)=\sigma_t^2 I
$$

&emsp;&emsp;常用选择是 $\sigma_t^2=\beta_t$，或

$$
\sigma_t^2
=
\tilde{\beta}_t
=
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
$$

&emsp;&emsp;真正需要学习的是均值 $\mu_\theta$。DDPM 不直接预测均值，而是让网络预测 $x_t$ 中累积的噪声。令网络为 $\epsilon_\theta(x_t,t)$，它预测的是从 $x_0$ 到 $x_t$ 累积加入的噪声 $\epsilon$，而不是第 $t$ 步单独加入的噪声。于是均值可写成：

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)
\right)
$$

由于 $\beta_t=1-\alpha_t$，也可写为：

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)
\right)
$$

&emsp;&emsp;采样时：

$$
x_{t-1}
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)
\right)
+
\sigma_t z
$$

其中 $z\sim\mathcal{N}(0,I)$。当 $t>1$ 时加入噪声；当 $t=1$ 时取 $z=0$。

&emsp;&emsp;这个更新与 score 有关。更准确地说：

$$
\nabla_{x_t}\log q(x_t)
\approx
-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar{\alpha}_t}}
$$

&emsp;&emsp;因此 $\epsilon_\theta$ 可以理解为噪声方向或 score 的负方向，而不是直接的数据密度梯度方向。它与 Langevin 动力学中的 score 项有密切联系。

---

**训练目标：简单到只有 MSE**

&emsp;&emsp;DDPM 实际使用的简化训练目标为：

$$
L_{\text{simple}}(\theta)
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\left\|
\epsilon-\epsilon_\theta
\left(
\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,t
\right)
\right\|^2
\right]
$$

训练过程：

```text
重复：
  x0 ~ 数据分布
  t ~ Uniform(1, ..., T)
  ε ~ N(0, I)
  对以下损失做梯度下降：
    || ε - εθ(√(ᾱ_t) x0 + √(1-ᾱ_t) ε, t) ||²
直到收敛
```

&emsp;&emsp;随机选一张图、随机选一个时间步、随机加噪，然后让网络预测“我加进去的累积噪声是什么”。这个目标不是原始 VLB 本身，而是 VLB 去掉/重加权后的简化目标。原始 VLB 中不同时间步的权重不同，通常小 $t$ 的权重较大；$L_{\text{simple}}$ 对各个 $t$ 近似均匀加权，因此相对于原始 VLB，它降低了小 $t$ 的相对权重，使训练更强调大 $t$、高噪声阶段的困难去噪任务。论文发现这种简化最终样本质量更好。

---

**采样：从纯噪声倒放出图像**

&emsp;&emsp;训练完成后，生成图像就是逆向扩散：

```text
x_T ~ N(0, I)
for t = T, ..., 1:
    z ~ N(0, I) 如果 t > 1，否则 z = 0
    x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * εθ(x_t, t)) + σ_t z
return x_0
```

&emsp;&emsp;这需要 $T$ 次网络前向。DDPM 原文取 $T=1000$，所以采样较慢。后续 DDIM、Latent Diffusion、Stable Diffusion 等工作都在此基础上大幅加速，或在潜空间中进行扩散。

**网络架构**

&emsp;&emsp;在 DDPM 中，使用的网络结构是基于医学图像分割的 UNet 架构，但为了适应扩散模型的特殊需求，Ho 等人对其进行了关键的改造。普通的 UNet 只需要输入一张图像并输出一张图像。而 DDPM 的 UNet 不仅要输入当前时间步的噪声图像 $x_{t}$，还必须输入当前所处的时间步长 $t$。这是因为网络必须知道当前的噪声有多大，才能准确预测出对应的噪声。DDPM 的 UNet 实现核心包含以下 四个关键改造与模块：
- **timestep嵌入**：由于时间步 $t$ 是一个标量（如 1 到 1000 之间的整数），不能直接和图像特征拼接。DDPM 借鉴了 Transformer 的做法：使用 正弦/余弦位置编码 (Sinusoidal Position Embedding) 将标量 $t$ 映射为一个高维向量。再通过几层全连接层（MLP）将这个时间向量放大，使其维度与 UNet 各个特征图的通道数一致。如何融合：在 UNet 的每一个残差块（Residual Block）中，将这个时间向量通过相加或尺度缩放（Scale-Shift）的方式融合到图像特征中。
- **残差块 (Residual Block)** 与群组归一化 (GroupNorm)传统的 UNet 常用普通的卷积和 BatchNorm，而 DDPM 换成了更强的结构：使用 ResNet 风格的残差结构（Conv -> GroupNorm -> SiLU -> Conv）。放弃 BatchNorm，改用 GroupNorm（群组归一化）。因为在去噪采样时，Batch Size 通常很小甚至为 1，BatchNorm 在这种情况下极不稳定，而 GroupNorm 的表现非常鲁棒。
- **自注意力机制** (Self-Attention)为了让模型具备全局建模能力（比如生成合理的动物肢体结构或全局对称的图案），DDPM 在 UNet 的低分辨率特征图层（通常是 $16\times16$ 或 $8\times8$ 的分辨率处）引入了 Self-Attention 块。它和 Transformer 中的自注意力相同，将特征图展平为序列，在通道维度上计算 Q、K、V。这极大地增强了网络捕获长距离图像特征依赖的能力。
- 经典的 U 型跳跃连接 (Skip Connections)保留了 UNet 最核心的特征：在下采样（Encoder）时保存每一层的特征图，在上采样（Decoder）时，通过通道拼接 (Channel Concatenation) 将 Encoder 的特征与 Decoder 的特征融合。这确保了图像的高频细节（如边缘、纹理）不会在压缩过程中丢失。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/ddpm_html_20260912_8b9708.html.png)

## 2 CLIP

- 论文地址：[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020)

&emsp;&emsp;CLIP 的核心思想同样非常朴素：传统视觉模型靠人工标注的固定类别做监督，类别一换就得重新标注、重新训练；CLIP 反过来，直接从互联网上爬取的海量“图像—文本”对中学习，让模型自己学会“哪段文字描述了哪张图”。一旦学会这种图文对齐能力，面对新的分类任务时，只需把类别名写成自然语言描述，让模型去匹配即可，完全不需要额外标注数据。CLIP 的主要贡献包括：

1. 证明用自然语言监督信号可以训练出强大的可迁移视觉模型；
2. 提出简单且可扩展的对比学习预训练目标，在 4 亿图文对上从头训练；
3. 实现真正的零样本迁移，在 ImageNet 上无需使用其 128 万训练样本即可匹配原始 ResNet-50 的准确率；
4. 展现出极强的鲁棒性，对图像分布偏移的泛化能力比同等精度的监督模型高出 40%～50%；
5. 开源代码与预训练权重，成为后续多模态模型（如 DALL·E、Stable Diffusion、LLaVA）的核心基础组件。

---

**模型架构：双塔结构与共享语义空间**

&emsp;&emsp;CLIP 采用双塔架构，包含一个图像编码器和一个文本编码器，两者相互独立但输出被投影到同一个高维语义空间。图像编码器支持 ResNet 和 Vision Transformer（ViT）两种架构：ResNet 版本通过 5 个阶段的卷积与全局平均池化得到图像特征向量；ViT 版本将图像切分为 16×16 的 patch，经线性投影后送入多层 Transformer，最终取 [CLS] token 的输出作为图像特征。文本编码器基于 Transformer 架构，输入文本经过分词后进入多层 Transformer 编码器，同样取序列中 [CLS] 位置的输出作为文本特征向量。两个编码器的输出维度被统一投影到相同的嵌入空间，使得图像向量和文本向量可以直接做点积或余弦相似度比较。所有编码器从头训练，不使用任何预训练权重。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/clip_textencoder_imageencoder_2026-09-12_16-18-23.png)

---

**对比学习预训练：把正样本拉近、负样本推远**

&emsp;&emsp;CLIP 的预训练目标是图文对比学习。对于一个包含 $N$ 个图文对的批次，模型计算 $N \times N$ 的相似度矩阵：第 $i$ 行第 $j$ 列的元素表示第 $i$ 张图像与第 $j$ 段文本的相似度。对角线上的 $N$ 个元素是正样本对（图像与其真实配对文本），其余 $N^2 - N$ 个元素都是负样本对。训练目标就是让对角线上的相似度尽可能高，非对角线上的相似度尽可能低。

&emsp;&emsp;具体实现时，先对图像特征和文本特征做 L2 归一化，然后计算相似度矩阵，再乘以一个可学习的温度系数 $\tau$ 的指数，最后在行方向和列方向分别做交叉熵损失并取平均：

$$
L_{\text{CLIP}}
=
\frac{1}{2}
\left(
L_{\text{image}\to\text{text}} + L_{\text{text}\to\text{image}}
\right)
$$

&emsp;&emsp;其中 $L_{\text{image}\to\text{text}}$ 的含义是：对每一张图像，模型需要从 $N$ 段文本中找出与它真正配对的那一段；$L_{\text{text}\to\text{image}}$ 则是对每一段文本，从 $N$ 张图像中找出真正配对的那一张。这种对称设计被称为 InfoNCE 损失，本质上是把图文匹配任务转化成了一个批内检索问题。温度参数 $\tau$ 初始化为 0.07，可学习，控制相似度分布的尖锐程度：$\tau$ 越小，正负样本之间的区分压力越大。批次大小 $N$ 越大，负样本越多，对比信号越强，因此 CLIP 在训练时使用了极大的批次规模。

---

**零样本迁移：用自然语言“提示”分类器**

&emsp;&emsp;训练完成后，CLIP 不需要任何微调就能执行图像分类。对于 $K$ 个候选类别，为每个类别构造一段文本提示，例如对于类别 “cat”，构造 “a photo of a cat”；对于 “dog”，构造 “a photo of a dog”。将所有提示文本送入文本编码器，得到 $K$ 个文本特征向量。再将待分类图像送入图像编码器，得到图像特征向量。计算图像特征与每个文本特征的余弦相似度，取相似度最高的类别作为预测结果。

&emsp;&emsp;这种“用文本提示做分类器”的方式有一个重要细节：类别名单独使用时往往是一个孤立的词，而 CLIP 的文本编码器是在完整句子上训练的，因此直接输入 “cat” 的效果不如输入 “a photo of a cat”。论文中系统比较了多种提示模板，发现合适的提示工程（prompt engineering）对零样本准确率有显著影响。此外，还可以对多个提示模板的文本特征取平均，进一步提升鲁棒性。这种零样本能力使得 CLIP 可以在 OCR、动作识别、地理定位、细粒度分类等 30 多个数据集上直接迁移，大多数任务上无需任何额外训练就能与完全监督的基线相媲美。

---

**训练目标：批内对比，简单到只有交叉熵**

&emsp;&emsp;CLIP 的训练过程可以概括为：

```text
重复：
  从数据集中取一批 N 个图文对 (x_i, t_i)
  图像特征 v_i = normalize(ImageEncoder(x_i))
  文本特征 u_i = normalize(TextEncoder(t_i))
  计算相似度矩阵 S[i][j] = v_i · u_j * exp(τ)
  对行做交叉熵损失：让 S[i][i] 最大
  对列做交叉熵损失：让 S[i][i] 最大
  对两个损失取平均，做梯度下降
直到收敛
```

&emsp;&emsp;这个目标简单到只有交叉熵和归一化，但它的威力来自两个因素：第一，数据规模极大，4 亿图文对覆盖了极其广泛的视觉概念；第二，批次内负样本数量随批次增大而线性增长，模型被迫学会非常精细的图文对齐。与 DDPM 的 MSE 目标类似，CLIP 的训练目标本身并不复杂，真正的关键是大规模数据和对比学习的可扩展性。

---

**影响与后续发展**

&emsp;&emsp;CLIP 之后，图文对比预训练成为多模态领域的标准范式。ALIGN 用更大规模的数据验证了“噪声图文对也能训练出好模型”；SigLIP 用 Sigmoid 损失替代 Softmax 交叉熵，在更小批次下也能有效训练；DALL·E 2 和            Stable Diffusion 使用 CLIP 的文本编码器将提示词映射为条件向量；LLaVA 等视觉语言模型将 CLIP 视觉编码器与大型语言模型对接。CLIP 的双塔架构和零样本能力，使其成为连接视觉与语言的核心基础设施。


## 3 LDM / Stable Diffusion

- 论文地址：[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752)

&emsp;&emsp;DDPM 证明了扩散模型可以生成高质量图像，但它直接在像素空间中进行扩散，高分辨率图像的训练和推理代价极为高昂。LDM 的核心思想是：不在像素空间上做扩散，而是在一个经过感知压缩的潜在空间上做扩散。先用一个自编码器把图像压缩成低维潜在表示，然后在这个潜在空间中训练扩散模型，生成时再从潜在表示解码回像素图像。这相当于把扩散模型从“像素去噪”变成了“压缩表示去噪”。LDM 的主要贡献包括：

1. 提出在预训练自编码器的潜在空间中训练扩散模型，首次在复杂度降低与细节保留之间达到接近最优的平衡点；
2. 引入交叉注意力层，将扩散模型转化为支持文本、边界框等通用条件输入的灵活生成器；
3. 在图像修复和类别条件图像合成上达到当时最优性能，在无条件生成、文本到图像合成和超分辨率任务上具有高度竞争力；
4. 相比像素空间扩散模型大幅降低了计算需求，使高分辨率图像合成在有限计算资源下成为可能。

---

**感知压缩：把图像压缩到潜在空间**

&emsp;&emsp;LDM 的第一阶段是训练一个自编码器，将像素图像压缩为潜在表示。给定图像 $x \in \mathbb{R}^{H \times W \times 3}$，编码器 $\mathcal{E}$ 将其映射为潜在表示 $z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c}$，其中 $h = H/2^m$，$w = W/2^m$，下采样因子 $f = 2^m$。解码器 $\mathcal{D}$ 则负责从潜在表示重建图像 $\tilde{x} = \mathcal{D}(z)$。

&emsp;&emsp;这个自编码器通常采用 VAE 架构。以 Stable Diffusion 使用的 sd-vae-ft-mse 为例：编码器将 $512 \times 512 \times 3$ 的 RGB 图像压缩为 $64 \times 64 \times 4$ 的潜在表示，下采样因子为 8，通道数仅为 4。训练完成后自编码器被冻结，扩散过程完全在这个压缩空间中运行。为了保持潜在空间的尺度稳定，实际使用时会乘以一个缩放因子（如 0.18215），送入 UNet 前乘以它，取出后再除回去。

&emsp;&emsp;自编码器的训练目标包含重建损失和 KL 正则化项。重建损失确保解码图像与原始图像尽可能一致，KL 项则约束潜在表示的分布，防止潜在空间过度拟合训练数据。

---

**潜在空间中的前向过程与逆向过程**

&emsp;&emsp;设经过编码器得到的潜在表示为 $z_0 = \mathcal{E}(x_0)$。LDM 的前向过程与 DDPM 完全相同，只不过是在潜在空间上进行的：

$$
q(z_t|z_{t-1}) = \mathcal{N}(z_t;\sqrt{\alpha_t}z_{t-1},(1-\alpha_t)I)
$$

&emsp;&emsp;任意时刻的闭式采样为：

$$
z_t = \sqrt{\bar{\alpha}_t}z_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
$$

&emsp;&emsp;其中 $\alpha_t = 1-\beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。DDPM 的噪声调度、参数化方式和采样公式都可以直接迁移到潜在空间上。

&emsp;&emsp;逆向过程同样学习一个去噪网络 $\epsilon_\theta(z_t,t)$ 来预测加入的噪声。采样时从标准高斯噪声 $z_T \sim \mathcal{N}(0,I)$ 出发，逐步去噪得到 $z_0$，最后通过解码器得到图像：$\tilde{x}_0 = \mathcal{D}(z_0)$。

---

**条件生成：交叉注意力注入文本信息**

&emsp;&emsp;无条件扩散模型只能随机生成图像，要让模型根据文本提示生成指定内容，需要引入条件信息。LDM 的解决方案是在 UNet 的去噪网络中插入交叉注意力层。文本提示首先经过一个冻结的文本编码器（Stable Diffusion 中通常使用 CLIP 文本编码器）得到文本嵌入，然后在 UNet 的各个空间分辨率层级上，通过交叉注意力将文本信息注入到潜在特征中。

&emsp;&emsp;具体来说，交叉注意力层的 Query 来自 UNet 的中间特征，Key 和 Value 来自文本嵌入。这意味着 UNet 的每一层特征图都在“询问”文本嵌入中哪些 token 与当前空间位置最相关。在 Stable Diffusion 的 UNet 中，交叉注意力层被放置在 $32 \times 32$、$16 \times 16$ 和 $8 \times 8$ 三个分辨率层级上，每个层级都有对应的 SpatialTransformer 模块来同时执行自注意力和交叉注意力。

&emsp;&emsp;除了交叉注意力，LDM 还支持通过拼接（concatenation）的方式注入条件信息，例如用于图像修复的掩码、超分辨率任务中的低分辨率图像等。这种设计使得同一个 UNet 可以灵活地处理多种条件生成任务。

---

**训练目标：在潜在空间中预测噪声**

&emsp;&emsp;LDM 的训练目标与 DDPM 的简化目标完全一致，只不过是在潜在空间上进行的：

$$
L_{\text{LDM}}(\theta)
=
\mathbb{E}_{z \sim \mathcal{E}(x), \epsilon \sim \mathcal{N}(0,I), t}
\left[
\left\|
\epsilon - \epsilon_\theta(z_t, t)
\right\|_2^2
\right]
$$

&emsp;&emsp;其中 $z_t = \sqrt{\bar{\alpha}_t}z_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，$z_0 = \mathcal{E}(x_0)$。这个目标可以进一步改写为潜在表示上的重建形式：

$$
L_{\text{LDM}}(\theta)
=
\mathbb{E}_{t,z_0,\epsilon}
\left[
\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}
\left\|
z_0 - \hat{z}_0
\right\|_2^2
\right]
$$

&emsp;&emsp;其中 $\hat{z}_0$ 是从预测噪声 $\epsilon_\theta(z_t,t)$ 反推得到的潜在表示估计。

&emsp;&emsp;训练过程可以概括为：

```text
重复：
  x0 ~ 数据分布
  z0 = E(x0)                      # 编码到潜在空间
  t ~ Uniform(1, ..., T)
  ε ~ N(0, I)
  对以下损失做梯度下降：
    || ε - εθ(√(ᾱ_t) z0 + √(1-ᾱ_t) ε, t) ||²
直到收敛
```

&emsp;&emsp;这个目标与 DDPM 的 $L_{\text{simple}}$ 在形式上是相同的，区别在于输入从像素图像 $x_0$ 变成了潜在表示 $z_0 = \mathcal{E}(x_0)$。整个训练过程中自编码器 $\mathcal{E}$ 和 $\mathcal{D}$ 保持冻结，只有 UNet $\epsilon_\theta$ 被更新。

---

**采样：DDIM 加速与潜在空间解码**

&emsp;&emsp;训练完成后，生成图像就是逆向扩散加上解码的过程。虽然可以直接使用 DDPM 的 1000 步采样，但 LDM 在实践中通常使用 DDIM 采样来加速。DDIM 是一种确定性采样方法，与 DDPM 共享相同的训练目标，但可以在 50 到 100 步内生成高质量样本。

&emsp;&emsp;采样过程如下：

```text
z_T ~ N(0, I)
for t = T, ..., 1:
    ε_pred = εθ(z_t, t)                      # 预测噪声
    z_{t-1} = 1/√α_t * (z_t - (1-α_t)/√(1-ᾱ_t) * ε_pred) + σ_t z
z_0 → x_0 = D(z_0)                           # 解码回像素空间
return x_0
```

&emsp;&emsp;其中 $z \sim \mathcal{N}(0,I)$ 在 $t>1$ 时加入，$t=1$ 时取 $z=0$。采样完成后，将潜在表示 $z_0$ 送入解码器 $\mathcal{D}$ 即可得到最终的像素图像。

---

**Stable Diffusion 的 UNet 架构细节**

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/ldm_stable_diffusion_2026-09-12_21-21-11.png)

&emsp;&emsp;Stable Diffusion 使用的 UNet 在 DDPM 的 UNet 基础上有几处关键改造，以适应潜在空间扩散和文本条件生成的需求：

- **输入与输出**：输入是 $4 \times 64 \times 64$ 的潜在表示，输出同样是 $4 \times 64 \times 64$ 的预测噪声。UNet 的通道数从 320 开始，经过 640、1280 逐层翻倍，整体参数量约为 8.6 亿。

- **时间步嵌入**：与 DDPM 相同，使用正弦位置编码将时间步 $t$ 映射为高维向量，再通过 MLP 融合到每个残差块中。网络通过这种方式知道当前去噪进行到了哪一步，从而调整去噪策略。

- **ResNet 块与 GroupNorm**：每个层级包含两个 ResNet 块，结构为 Conv → GroupNorm → SiLU → Conv，使用 GroupNorm 而非 BatchNorm，因为采样时批次大小通常很小，GroupNorm 在这种场景下更稳定。

- **SpatialTransformer 模块**：在 $32 \times 32$、$16 \times 16$、$8 \times 8$ 三个层级上，每个 ResNet 块之后接一个 SpatialTransformer。这个模块同时包含自注意力和交叉注意力：自注意力让模型捕捉空间上的长距离依赖，交叉注意力则负责从文本嵌入中提取与当前空间位置相关的语义信息。

- **跳跃连接**：沿用 UNet 的经典设计，编码器每一层的特征通过通道拼接传递到解码器对应层，确保高频细节在压缩过程中不会丢失。

- **文本编码器**：使用冻结的 CLIP ViT-L/14 文本编码器，输入为 77 个 BPE token，输出维度为 768 的文本嵌入序列，供交叉注意力层使用。

---

**LDM 与 Stable Diffusion 的关系及后续影响**

&emsp;&emsp;Stable Diffusion 是 LDM 的一种具体实现。所有 Stable Diffusion 模型都是 LDM，但并非所有 LDM 都是 Stable Diffusion。Stable Diffusion 相对于原论文的关键变化包括：使用更大的训练数据（LAION-5B）、更大的 UNet（约 8.6 亿参数）、以及更成熟的文本编码器配置。

&emsp;&emsp;LDM 的“潜在空间扩散 + 交叉注意力条件生成”范式深刻影响了后续工作。SDXL 引入双文本编码器和更大的 UNet 骨干；ControlNet 在 UNet 编码器上复制一份可训练分支，通过零卷积实现精确的空间条件控制；LoRA 通过低秩适配微调交叉注意力层的权重，使个性化生成变得轻量而高效。LDM 将扩散模型从“像素空间的高成本实验”变成了“消费级硬件上可运行的实用工具”，这是它成为生成式 AI 里程碑式工作的根本原因。

## 4 SDXL

- 论文地址：[SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/pdf/2307.01952)

&emsp;&emsp;LDM 证明了“潜在空间扩散 + 交叉注意力条件生成”范式可行，但 Stable Diffusion 1.5/2.1 在 1024×1024 分辨率下的生成质量和提示跟随能力仍有明显不足。SDXL 的核心思路是：不改变扩散范式本身，而是从模型规模、文本条件质量、训练信号设计和采样后处理四个维度对 LDM 进行全面增强。它在保持潜空间扩散框架的前提下，把 UNet 骨干扩大了约 3 倍，引入双文本编码器丰富文本理解，设计微条件机制弥补训练数据的预处理缺陷，并加入精炼模型进一步提升最终图像的视觉保真度。SDXL 的主要贡献包括：

1. 将 UNet 骨干扩大到 2.6B 参数，通过更密集的注意力块和更大的交叉注意力上下文显著提升模型容量；
2. 引入双文本编码器（CLIP ViT-L + OpenCLIP ViT-bigG），将两个编码器的特征拼接后注入交叉注意力，同时将 OpenCLIP 的池化嵌入额外注入各卷积块；
3. 设计尺寸微条件和裁剪微条件，使模型能够感知训练样本的原始分辨率和裁剪位置，从而在推理时通过指定“原始尺寸”和“无裁剪”条件生成更高质量的图像；
4. 支持多宽高比训练，模型原生支持从 512×512 到 2048×512 的多种分辨率，无需后处理裁剪；
5. 引入精炼模型（Refiner），在基础模型生成潜在表示后进行低噪声加噪-去噪，提升细节和背景质量。

---

**架构与规模：更深的 UNet 与更密集的注意力**

&emsp;&emsp;SDXL 的 UNet 相比 Stable Diffusion 1.5/2.1 有显著扩展。在架构上，SDXL 参考 Simple Diffusion 的做法，将 Transformer 计算集中到 UNet 的低层级特征上，采用异构的 Transformer 块分布：最高特征层级不放置 Transformer 块，中间层级放置 2 个块，最低层级放置 10 个块，同时去掉了最低的 8 倍下采样层级。这种设计使模型参数量达到 2.6B，而 SD 1.5 的 UNet 仅有约 860M。

&emsp;&emsp;文本编码器方面，SDXL 使用了两个独立的文本编码器：CLIP ViT-L/14 和 OpenCLIP ViT-bigG。两个编码器的 token 嵌入被拼接后投影到交叉注意力维度，拼接后的上下文维度从 SD 1.5 的 768 提升到 2048。此外，OpenCLIP ViT-bigG 的池化文本嵌入（CLS embedding）被单独投影后，通过加法注入到 UNet 的所有卷积块中，为模型提供全局的文本语义信息。文本编码器的总参数量达到 817M。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/sdxl_e_2026-09-12_21-31-32.png)

---

**微条件：让模型感知训练数据的预处理历史**

&emsp;&emsp;LDM 范式要求训练图像有一个最低分辨率，因为图像需要先经过 VAE 编码器压缩到潜在空间。Stable Diffusion 1.4/1.5 的做法是丢弃所有短边小于 512 像素的图像，但这会损失大量训练数据。SDXL 的训练数据中，若丢弃所有低于预训练分辨率的样本，会损失约 39% 的数据。另一种做法是将低分辨率图像放大，但这会引入上采样伪影，导致生成图像模糊。

&emsp;&emsp;SDXL 的解决方案是尺寸微条件：将图像缩放前的原始高度和宽度作为额外的条件输入，通过傅里叶特征编码后与时间步嵌入相加，注入 UNet。这样模型能够“知道”当前训练样本的原始分辨率是多少，从而学会在低分辨率样本上不产生模糊，在高分辨率样本上保持细节。推理时，用户可以指定“原始尺寸”条件来控制生成图像的视觉风格和细节层次——将尺寸条件设得较小会使生成图像趋向于更简洁的构图，设得较大则会增强细节。

&emsp;&emsp;裁剪微条件解决的是另一个问题：训练时通常使用随机裁剪，这会导致图像主体偏离中心。SDXL 将裁剪坐标 $(c_{\text{top}}, c_{\text{left}})$ 也通过傅里叶编码注入 UNet，使模型学会“补偿”裁剪带来的偏移。推理时，将裁剪条件设为 $(0,0)$ 即可获得主体居中的生成结果。

---

**多宽高比训练：原生支持非方形图像**

&emsp;&emsp;Stable Diffusion 的标准输出是 512×512 或 1024×1024 的方形图像，但现实中的图像以横向（16:9）和纵向（9:16）居多。SDXL 在微调阶段采用数据分桶策略：将训练数据按宽高比分组，每个批次内的图像来自同一桶，桶的像素总数保持在 1024² 附近，高度和宽度按 64 的倍数调整。模型同时接收桶的目标尺寸作为条件，从而在推理时可以直接生成指定宽高比的图像，无需生成后再裁剪。

---

**精炼模型：两阶段生成流程**

&nbsp;&emsp;&emsp;SDXL 的生成流程采用两阶段设计。第一阶段由基础模型（Base）完成从纯噪声到潜在表示的完整去噪过程，生成带有一定噪声的潜在表示。第二阶段由精炼模型（Refiner）对基础模型的输出进行低噪声加噪-去噪处理。Refiner 使用与 Base 相同的架构和文本编码器配置，但专门在前 200 个离散噪声尺度上训练，只处理低噪声阶段。

&emsp;&emsp;Refiner 的作用是提升细节质量，尤其是背景细节和面部区域。论文的用户研究表明，使用 Refiner 后的 SDXL 在人类偏好评估中显著优于 Base 单独使用和 SD 1.5/2.1：SDXL 加 Refiner 的胜率为 48.44%，SDXL Base 为 36.93%，SD 1.5 为 7.91%，SD 2.1 为 6.71%。值得注意的是，在 FID 和 CLIP 分数等传统指标上，SDXL 的改进并不明显，这与用户偏好形成了对比，说明传统指标可能无法完全反映生成质量的提升。

&emsp;&emsp;Refiner 是可选的，Base 模型可以单独使用。但使用 Refiner 需要额外加载一个模型到显存中，这对消费级 GPU 是一个负担。论文在 Future Work 中也指出，未来应研究单阶段方案以替代两阶段流程。

---

**训练目标与采样**

&emsp;&emsp;SDXL 的训练目标与 LDM 完全一致，仍然是潜在空间中的噪声预测 MSE 损失：

$$
L_{\text{SDXL}}(\theta)
=
\mathbb{E}_{z_0, \epsilon \sim \mathcal{N}(0,I), t}
\left[
\left\|
\epsilon - \epsilon_\theta(z_t, t \mid \tau)
\right\|_2^2
\right]
$$

&emsp;&emsp;其中 $\tau$ 表示文本条件，包括双编码器的拼接嵌入和池化嵌入。所有微条件向量（尺寸、裁剪、目标尺寸）在傅里叶编码后与时间步嵌入拼接，在每个 UNet 块中注入模型。

&emsp;&emsp;采样时使用 DDIM 或 Euler 等求解器，通常需要约 50 步。SDXL 推荐使用 1024×1024 分辨率，也支持 768×768 和 512×512，但效果不如 1024。SDXL 还支持为两个文本编码器分别传入不同的提示，甚至可以将同一个提示的不同部分分别传入两个编码器。

---

**SDXL 与 LDM 的关系及后续影响**

&emsp;&emsp;**SDXL 是 LDM 范式在规模和质量上的全面升级。它没有改变 LDM 的核心架构——仍然是 VAE + 潜空间扩散 + 交叉注意力条件生成——而是通过放大模型、丰富文本条件、增加训练信号和引入精炼阶段**，将这一范式的潜力推到了新的高度。

&emsp;&emsp;SDXL 之后，基于它的生态迅速扩展：SDXL Turbo 通过对抗扩散蒸馏将推理步数压缩到 1-4 步；LCM-LoRA 提供了 SDXL 的潜在一致性适配器，使少步生成更加灵活；ControlNet 和 IP-Adapter 等控制与个性化方法也迅速适配到 SDXL 架构上。SDXL 成为开源文生图社区事实上的基础模型，在 Hugging Face 上的下载量超过 200 万次，衍生出数千个社区微调版本。

## 5 ControlNet

- 论文地址：[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543)

&emsp;&emsp;SDXL 把文生图的生成质量推到了新的高度，但即便是最精细的提示词，也无法精确描述人物四肢的具体角度、背景中物体的确切位置、每一缕光线的照射方向。文本描述天然是粗粒度的，而图像生成往往需要像素级的空间控制。ControlNet 的核心思想是：不修改、不重新训练已经训练好的扩散模型，而是在它旁边“嫁接”一个可训练的控制分支，通过一种特殊的“零卷积”连接，让这个分支从零开始逐渐学会如何按照额外的空间条件（边缘图、深度图、人体姿态等）来引导生成过程。ControlNet 的主要贡献包括：

1. 提出一种能够为任意预训练文生图扩散模型添加空间条件控制的通用神经网络架构，无需改变原始模型的拓扑结构；
2. 设计“零卷积”作为连接层，其权重初始化为零，确保训练初期控制分支不会对预训练模型的生成能力造成任何干扰；
3. 通过“锁定副本 + 可训练副本”的双分支设计，使 ControlNet 能够在极小数据集（< 50k）和极大数据集（> 1M）上均表现出稳定的训练鲁棒性；
4. 在边缘、深度、分割、人体姿态等多种空间控制条件下验证了架构的有效性，且支持多条件组合与无提示生成；
5. 获得 ICCV 2023 最佳论文奖，成为可控图像生成领域事实上的标准方案。

---

**问题背景：文本控制的局限性**

&emsp;&emsp;Stable Diffusion 等文生图模型通过交叉注意力接收文本条件，但文本提示在空间精度上存在天然上限。用户可以用提示词描述“一个举起右手的人”，却无法精确指定右手应该举起多少度、手臂与躯干的夹角是多少。这种粗粒度控制阻碍了 AI 绘图在需要精确构图、姿态迁移、线稿上色等专业场景中的应用。在 ControlNet 之前，一些工作尝试通过在 UNet 的交叉注意力层中注入额外条件来弥补，但这些方法要么需要修改原始网络结构，要么在训练数据有限时容易破坏预训练模型的生成能力。

---

**核心架构：锁定副本与可训练副本**

&emsp;&emsp;ControlNet 的架构设计围绕一个关键约束展开：既要让模型学会新的空间控制能力，又要确保预训练模型的生成质量不被破坏。解决方案是将原始扩散模型的编码器复制为两份：一份“锁定副本”（locked copy），权重完全冻结，保留从数十亿张图像中学到的生成知识；一份“可训练副本”（trainable copy），权重初始化为锁定副本的值，但在训练过程中接收控制条件并更新参数。

&emsp;&emsp;锁定副本和可训练副本之间通过“零卷积”层连接。对于 UNet 编码器中的每一个 block，可训练副本的输入除了接收来自上一层的特征外，还额外接收控制条件图像（如边缘图、姿态图等）经过编码后的特征。可训练副本的输出经过零卷积后，逐元素加到锁定副本对应解码器 block 的输出上。这样，原始 UNet 的前向路径完全不受影响，控制信息通过侧路注入。

&emsp;&emsp;在 Stable Diffusion 的实现中，ControlNet 的可训练副本仅复制 UNet 的编码器部分（包括中间块），解码器部分不复制，以节省参数量。整个 ControlNet 结构由 14 个可训练 block 和对应的零卷积层组成，参数规模约为原始 UNet 的一半。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/controlNet_Snipaste_2026-09-12_21-43-19.png)

---

**零卷积：从零开始的渐进学习**

&emsp;&emsp;零卷积是 ControlNet 最核心的设计。它是一个 $1 \times 1$ 的卷积层，权重和偏置全部初始化为零。这意味着在训练的第一步，无论控制条件是什么，零卷积的输出都是零，ControlNet 对锁定副本的影响完全为零。原始 UNet 的行为与未添加 ControlNet 时完全一致，预训练模型的生成能力不受任何干扰。

&emsp;&emsp;零卷积能够正常训练的原因在于梯度流动的方向。考虑一个简单的线性层 $y = wx + b$，当 $w = 0$ 且输入 $x \neq 0$ 时：

$$
\frac{\partial y}{\partial w} = x \neq 0, \quad \frac{\partial y}{\partial b} = 1 \neq 0
$$

&emsp;&emsp;虽然此时 $\frac{\partial y}{\partial x} = w = 0$，意味着零卷积层暂时不会将梯度回传给控制分支的上游，但权重 $w$ 和偏置 $b$ 本身仍然能够通过梯度下降被更新。经过若干次迭代后，$w$ 不再为零，$\frac{\partial y}{\partial x}$ 也随之非零，梯度得以正常地向控制分支传播。这种“从零开始、渐进生长”的机制，使 ControlNet 的训练过程天然稳定，不会因为随机初始化的控制分支而给预训练模型注入有害噪声。

---

**条件编码：支持多种空间控制类型**

&emsp;&emsp;ControlNet 的条件输入可以是多种空间控制信号。论文中验证的控制类型包括：

- **Canny 边缘**：通过 Canny 算子从参考图像中提取边缘图，控制生成图像中物体的轮廓位置和形状；
- **HED 软边缘**：使用 HED 边缘检测器提取更柔和的边界，适合需要保留更多渐变和纹理细节的场景；
- **深度图**：通过 MiDaS 等深度估计器得到场景的深度信息，控制生成图像中物体的远近关系和空间层次；
- **人体姿态**：通过 OpenPose 提取人体关键点骨架，精确控制生成人物的姿态和动作；
- **语义分割**：通过 ADE20K 等分割模型得到场景的语义标签图，控制生成图像中不同区域的内容类别；
- **线稿**：从手绘草图或线稿中提取轮廓，用于线稿上色和风格化生成。

&emsp;&emsp;每种控制条件都对应一个独立训练的 ControlNet 模型，但共享相同的架构和训练框架。条件图像首先经过一个轻量的编码网络（通常由几个卷积层组成）映射为与 UNet 特征图维度匹配的特征，然后输入可训练副本。

---

**训练目标：与扩散模型完全一致的噪声预测**

&emsp;&emsp;ControlNet 的训练目标与 Stable Diffusion 的原始训练目标在形式上完全相同，仍然是在潜在空间中预测噪声，只不过去噪网络的输入额外包含了控制条件。损失函数为：

$$
\mathcal{L}_{\text{ControlNet}}
=
\mathbb{E}_{z_0, t, c_t, c_f, \epsilon}
\left[
\left\|
\epsilon - \epsilon_\theta(z_t, t, c_t, c_f)
\right\|_2^2
\right]
$$

&emsp;&emsp;其中 $z_0 = \mathcal{E}(x_0)$ 是潜在表示，$c_t$ 是文本条件，$c_f$ 是任务特定的空间条件（如边缘图、姿态图等），$\epsilon_\theta$ 是包含 ControlNet 分支的完整去噪网络。训练时锁定副本的参数冻结不更新，只有可训练副本和零卷积层的参数被优化。

&emsp;&emsp;训练过程可以概括为：

```text
重复：
  x0 ~ 数据分布
  z0 = E(x0)                              # 编码到潜在空间
  t ~ Uniform(1, ..., T)
  ε ~ N(0, I)
  控制条件 cf 从 x0 中提取（如 Canny 边缘）
  锁定副本参数冻结
  对以下损失做梯度下降，只更新可训练副本和零卷积：
    || ε - εθ(√(ᾱ_t) z0 + √(1-ᾱ_t) ε, t, ct, cf) ||²
直到收敛
```

&emsp;&emsp;论文使用 AdamW 优化器，学习率建议在 $1 \times 10^{-5}$ 到 $5 \times 10^{-5}$ 之间，小数据集使用较小的学习率。训练过程中还会以一定概率随机丢弃文本条件（通常为 10%），以支持无提示的控制生成。

---

**训练稳定性与数据效率**

&emsp;&emsp;ControlNet 在训练稳定性方面的一个显著特点是：即使使用极小规模的数据集（少于 50k 样本，甚至 1k 样本），也不会破坏预训练模型的生成能力。这直接得益于零卷积的初始化策略和锁定副本的设计。在训练初期，ControlNet 的输出为零，原始 UNet 完全按照预训练的方式工作；随着训练进行，零卷积的权重逐渐偏离零，控制信号以渐进的方式注入，模型不会经历“突然被大量噪声干扰”的灾难性遗忘。

&emsp;&emsp;论文还报告了一个有趣的现象：ControlNet 的训练损失曲线有时会出现“突然收敛”（sudden convergence）——在某个训练步数之前损失下降缓慢，之后突然快速下降。这可能对应于模型突然“学会”了如何将控制条件与去噪过程对齐。

---

**与 T2I-Adapter 的对比**

&emsp;&emsp;T2I-Adapter 是与 ControlNet 同期提出的另一种条件控制方案。两者的共同点是都采用轻量化的旁路设计，不修改原始 UNet 的结构，训练参数少、成本低。主要区别在于：

&emsp;&emsp;ControlNet 通过“锁定副本 + 可训练副本”的深度复制获得了更强的控制能力，尤其适合需要像素级精度的任务（如姿态控制、线稿上色）。但代价是参数更多、计算量更大——每个 ControlNet 模型都需要复制 UNet 的编码器，且在去噪过程的每一步都需要运行整个控制分支。T2I-Adapter 则更轻量：它的旁路网络尺寸更小，且可以在整个去噪过程中只运行一次，将条件特征缓存后复用，因此在移动端和实时场景中更具效率优势。论文中的对比实验显示，在细粒度控制任务上 ControlNet 的 FID 表现更好，而 T2I-Adapter 在效率敏感的场景中更有竞争力。

---

**多条件组合与无提示控制**

&emsp;&emsp;ControlNet 支持通过多个 ControlNet 模型同时施加不同类型的控制。例如，可以同时使用 Canny 边缘控制轮廓、OpenPose 控制姿态、深度图控制空间层次，三种条件分别通过各自的零卷积层注入同一个 UNet。由于每个 ControlNet 的零卷积输出是逐元素相加的，多个控制信号可以自然叠加。

&emsp;&emsp;此外，ControlNet 在设计时通过随机丢弃文本条件，使模型能够独立于文本提示工作。这意味着用户可以在不提供任何提示词的情况下，仅凭一张边缘图或姿态图生成图像，控制信号本身就构成了完整的条件。

---

**与 LDM/SDXL 的关系及后续影响**

&emsp;&emsp;**ControlNet 是 LDM 范式在“可控性”维度上的关键扩展。它没有改变 LDM 的核心——仍然是 VAE + 潜在空间扩散 + 交叉注意力文本条件——而是在此基础上增加了一个并行的空间条件注入通路。** 这种设计使得 ControlNet 可以即插即用地适配任何基于 LDM 的模型：从 SD 1.5 到 SDXL，从文本到图像到图像到图像，都可以使用相同架构的 ControlNet 进行条件控制。

&emsp;&emsp;ControlNet 之后，可控生成领域迅速扩展：ControlNet-XS 通过双向通信减少参数和计算开销；ControlNet++ 引入像素级循环一致性约束提升控制精度；IP-Adapter 将图像提示通过解耦的交叉注意力注入，与 ControlNet 形成互补。在应用层面，ControlNet 已经渗透到 AI 绘图的几乎所有专业工作流中——线稿上色、姿态迁移、建筑草图渲染、老照片修复、创意二维码生成——成为生成式 AI 从“随机生成”走向“精确控制”的关键基础设施。

## 6 LoRA

- 论文地址：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)

&emsp;&emsp;全量微调是让大模型适配下游任务的传统范式，但随着模型参数规模激增，这一范式的代价变得难以承受。以 GPT-3 175B 为例，每个下游任务都需要独立存储和部署一份完整的 175B 参数副本，对显存和存储的要求远超多数研究者和企业的可及范围。已有的参数高效微调方案各有缺陷：Adapter 会在推理时引入额外延迟，Prefix-Tuning 会占用输入序列长度且优化困难。LoRA 的核心思想建立在这样一个假设之上：**预训练模型在适配下游任务时，权重矩阵的更新量 $\Delta W$ 具有很低的内在秩，因此可以将这个更新量分解为两个小得多的矩阵的乘积，只训练这两个小矩阵，而保持原始权重完全冻结。** LoRA 的主要贡献包括：

1. 提出低秩分解的参数化方法，将权重更新表示为 $\Delta W = BA$，其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，秩 $r \ll \min(d,k)$，可训练参数减少至原来的万分之一；
2. 设计 $A$ 随机高斯初始化、$B$ 零初始化的策略，确保训练初期 $\Delta W = BA = 0$，模型行为与预训练模型完全一致，避免了微调初期的性能突变；
3. 在 GPT-3 175B 上验证，LoRA 的可训练参数减少 10,000 倍，GPU 显存需求降低约 3 倍，同时模型质量持平或优于全量微调，且推理时无额外延迟；
4. 训练完成后可将 $\Delta W = BA$ 合并回原始权重 $W = W_0 + BA$，推理阶段与原始模型在计算路径上完全一致，不引入任何额外的计算开销；
5. 提供经验性分析，揭示语言模型适配中的秩亏缺现象，解释了为何极低秩（如 $r=1$ 或 $r=2$）就能在 $d=12288$ 的权重矩阵上有效工作。

---

**低秩分解：用两个小矩阵表示大更新**

&emsp;&emsp;设预训练模型中的某个权重矩阵为 $W_0 \in \mathbb{R}^{d \times k}$。全量微调直接更新 $W_0$，得到 $W' = W_0 + \Delta W$，其中 $\Delta W$ 与 $W_0$ 维度相同。LoRA 的做法是将 $\Delta W$ 分解为两个低秩矩阵的乘积：

$$
\Delta W = BA
$$

&emsp;&emsp;其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，秩 $r \ll \min(d,k)$。经过 LoRA 微调后的权重为：

$$
W' = W_0 + BA
$$

&emsp;&emsp;前向传播时，对于输入 $x$，输出为：

$$
h = W_0 x + \Delta W x = W_0 x + BA x
$$

&emsp;&emsp;训练过程中 $W_0$ 保持冻结，不接收梯度更新，只有 $A$ 和 $B$ 被优化。由于 $r$ 远小于 $d$ 和 $k$，$(A,B)$ 的参数总量 $r(d+k)$ 远小于 $\Delta W$ 的参数总量 $dk$。以 GPT-3 为例，当 $d = 12288$、$r = 2$ 时，可训练参数从 $1.5 \times 10^8$ 量级降至 $5 \times 10^4$ 量级。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/lora_Snipaste_2026-09-12_21-57-38.png)

&emsp;&emsp;在 Transformer 架构中，LoRA 通常被应用于自注意力模块的查询、键、值投影权重 $W_q, W_k, W_v \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$，以及输出投影权重 $W_o$。论文中通过消融实验比较了在不同权重矩阵上应用 LoRA 的效果，发现同时适配 $W_q$ 和 $W_v$ 在多数任务上表现最优。

---

**初始化策略：从零开始，保证训练稳定**

&emsp;&emsp;LoRA 对两个低秩矩阵采用了不对称的初始化策略。矩阵 $A$ 使用均值为零、方差为 $\sigma^2$ 的高斯分布进行随机初始化，矩阵 $B$ 初始化为全零矩阵。在这一初始化下：

$$
\Delta W = BA = 0
$$

&emsp;&emsp;这意味着在训练的第一步，LoRA 分支的输出完全为零，模型的前向行为与未添加 LoRA 时的预训练模型完全一致。原始模型的生成或推理能力不受任何干扰，微调过程从“零偏移”开始，随着训练逐步学习到适配下游任务的更新量。这种初始化策略与 ControlNet 中零卷积的设计逻辑一脉相承：都是通过将新增分支的初始输出置零，来保证预训练模型的既有能力在训练初期不被破坏。

&emsp;&emsp;将 $B$ 初始化为零而非随机值，还有一个梯度流动层面的考虑。如果 $B$ 和 $A$ 都随机初始化，$\Delta W$ 在第一步就是一个随机的大矩阵，会给预训练模型注入噪声。而 $B=0$ 时，虽然 $\frac{\partial \mathcal{L}}{\partial A}$ 在第一步可能为零，但 $\frac{\partial \mathcal{L}}{\partial B}$ 非零，$B$ 从零开始学习；一旦 $B$ 偏离零，$A$ 的梯度也随之非零，两个矩阵以渐进的方式共同学习。

---

**训练与推理：无额外延迟的适配**

&emsp;&emsp;LoRA 的训练过程可以概括为：

```text
重复：
  x, y ~ 下游任务数据
  冻结 W0，只更新 A 和 B
  前向传播：h = W0 x + BA x
  对以下损失做梯度下降（只更新 A 和 B）：
    L(h, y)
直到收敛
```

&emsp;&emsp;训练完成后的推理阶段，可以将 $BA$ 合并回原始权重：

$$
W = W_0 + BA
$$

&emsp;&emsp;合并后，模型的推理计算路径与原始预训练模型完全相同，不引入任何额外的网络层或计算步骤。这与 Adapter 或 Prefix-Tuning 形成鲜明对比：Adapter 在推理时需要额外经过插入的小型网络模块，Prefix-Tuning 需要额外的 KV 缓存，两者都会增加推理延迟。LoRA 的“训练时独立、推理时合并”特性，使其成为唯一在推理阶段完全零开销的参数高效微调方法。

&emsp;&emsp;在实际使用中，也可以选择不合并 $BA$，而是在推理时动态加载多个 LoRA 适配器，每个适配器对应一个不同的下游任务。由于每个 LoRA 只包含 $r(d+k)$ 个参数，一个适配器的文件大小通常只有几 MB 到几十 MB，远小于完整模型的数 GB。这使得一个基础模型可以同时服务多个任务，只需在推理时切换对应的 LoRA 权重即可。

---

**秩的选择：低秩足以捕获任务适配**

&emsp;&emsp;LoRA 的核心假设是 $\Delta W$ 具有低内在秩。论文通过实验验证了这一假设：在 GPT-3 175B 上，将 $r$ 设为 1 或 2，就能在多数任务上达到与全量微调相当的性能，而 $d$ 高达 12288。进一步的分析表明，增加 $r$ 并不能带来持续的收益：当 $r$ 从 1 增加到 64 时，某些任务的性能先升后降，说明过大的秩反而可能导致过拟合。

&emsp;&emsp;这一发现揭示了一个深层事实：大模型在下游适配时的权重更新，本质上是在一个极低维的子空间内进行的。预训练模型已经在大规模数据上学到了丰富的通用表示，下游任务适配只需要在这个表示空间中进行小幅调整，而不需要重新学习大量的新参数。LoRA 的秩 $r$ 正是这个“适配子空间”的维度，它通常只需要 4 到 64 就足够，远小于原始权重的维度。

---

**与 Adapter、Prefix-Tuning 的对比**

&emsp;&emsp;在 LoRA 之前，参数高效微调领域已有 Adapter 和 Prefix-Tuning 两条主要技术路线。Adapter 在 Transformer 层之间插入小型瓶颈网络，Prefix-Tuning 在输入序列前添加可训练的前缀向量。三者的对比如下：

- **可训练参数量**：LoRA 与 Adapter 相近，均可在原模型参数的 0.01%–1% 范围内；Prefix-Tuning 的可训练参数更少，但受限于前缀长度。
- **推理延迟**：LoRA 零额外延迟（可合并），Adapter 有显著延迟（batch size=1 时增加 20%–30%），Prefix-Tuning 占用输入序列长度。
- **训练稳定性**：LoRA 的零初始化策略使其在训练初期完全不影响预训练模型，稳定性最好；Adapter 和 Prefix-Tuning 在训练初期可能对模型行为造成扰动。
- **多任务部署**：LoRA 适配器可独立存储和动态切换，文件大小通常为几 MB；Adapter 需要为每个任务存储额外的网络模块。

&emsp;&emsp;论文中的实验表明，在相同可训练参数量下，LoRA 在几乎所有基准上均优于 Prefix-Tuning，与 Adapter 性能相近但参数量更少。

---

**在扩散模型中的应用：Stable Diffusion 的 LoRA**

&emsp;&emsp;LoRA 最初为语言模型设计，但其低秩适配的思想迅速被迁移到扩散模型中，成为 Stable Diffusion 生态中最主流的微调方式。在 Stable Diffusion 中，LoRA 通常被应用于 UNet 的交叉注意力层和自注意力层中的投影权重，训练参数量仅为原始 UNet 的 0.1%–1%，适配器文件大小通常只有几 MB 到几十 MB。用户可以在消费级 GPU（如 Tesla T4、V100）上，用少量自定义图像和对应的文本标注对 Stable Diffusion 进行风格化或主题化微调。

&emsp;&emsp;Stable Diffusion 的 LoRA 生态催生了“模型即插件”的范式：基础模型（SD 1.5、SDXL）作为通用生成引擎，各种 LoRA 适配器作为可插拔的风格、角色、画风模块，用户可以在提示词中通过权重控制多个 LoRA 的叠加效果。这一范式极大地降低了定制化图像生成的门槛。

---

**后续影响与变体**

&emsp;&emsp;LoRA 已成为参数高效微调领域事实上的标准方案。其后续发展包括：

- **QLoRA**：将预训练模型量化为 4-bit，在此基础上应用 LoRA，进一步将显存需求降低到可在单张消费级 GPU 上微调 65B 模型的程度；
- **LoRA+**：为 $A$ 和 $B$ 设置不同的学习率，加速收敛；
- **DoRA**：将权重更新分解为幅度和方向两部分，对方向部分应用 LoRA，提升微调效果；
- **LoRA 的秩自适应**：根据任务难度动态调整 $r$，在简单任务上用极小秩，在复杂任务上适当增大。

&emsp;&emsp;从更宏观的视角看，LoRA 与 ControlNet 共同构成了扩散模型可控生成的两大支柱：ControlNet 解决的是“空间结构控制”，LoRA 解决的是“风格与语义控制”。两者在 Stable Diffusion 工作流中经常配合使用——ControlNet 固定构图，LoRA 注入风格，文本提示提供语义引导，三者叠加形成了当前 AI 绘图领域最成熟的控制体系。

## 7 DiT

- 论文地址：[Scalable Diffusion Models with Transformers](https://arxiv.org/pdf/2212.09748)

&emsp;&emsp;到 LDM 为止，扩散模型的主流骨干网络始终是 U-Net。U-Net 的卷积结构天然适合图像，但它本质上是一种局部算子，捕获长距离依赖需要依赖堆叠和下采样。与此同时，Transformer 在 NLP 和视觉识别领域已经证明了自己是更可扩展的架构——ViT 的成功表明，只要数据足够，标准 Transformer 在图像理解上可以超越卷积网络。一个自然的问题是：扩散模型的 U-Net 骨干，究竟是必要的，还是仅仅是一种历史惯性？DiT 的核心思想就是回答这个问题：把 U-Net 替换成标准 Transformer，让扩散模型像 ViT 一样从架构统一中受益。DiT 的主要贡献包括：

1. 提出 Diffusion Transformer（DiT），在 LDM 框架内用 Transformer 替换 U-Net 骨干，证明 U-Net 的归纳偏置对扩散模型性能并非不可或缺；
2. 设计 patchify 输入层，将 VAE 潜表示切分为 patch 序列，使标准 ViT 架构可以直接用于扩散去噪；
3. 系统比较了四种条件注入方式（in-context、cross-attention、adaLN、adaLN-Zero），证明 adaLN-Zero 在计算效率和生成质量上均最优；
4. 通过以 Gflops 为衡量标准的缩放分析，首次证明扩散模型同样遵循“模型复杂度越高、FID 越低”的缩放规律；
5. DiT-XL/2 在 ImageNet 256×256 上取得当时最优的 FID 2.27，超越所有基于 U-Net 的扩散模型。

---

**Patchify：把潜表示切分成 token 序列**

&emsp;&emsp;DiT 整体沿用 LDM 的两阶段框架：第一阶段由一个冻结的 VAE 编码器将图像 $x \in \mathbb{R}^{H \times W \times 3}$ 压缩为潜表示 $z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c}$，第二阶段在潜空间中训练去噪网络。与 LDM 的唯一区别在于，去噪网络从 UNet 换成了标准 Transformer。

&emsp;&emsp;Transformer 的输入是一个 token 序列，而潜表示 $z$ 是二维空间特征图。DiT 的解决方案是 patchify：将 $z$ 切分为 $p \times p$ 的非重叠 patch，每个 patch 展平后经过一个线性层投影为维度 $d$ 的 token。设 $h = w = 64$、$c = 4$，当 $p = 2$ 时得到 $32 \times 32 = 1024$ 个 token；当 $p = 8$ 时得到 $8 \times 8 = 64$ 个 token。patch 大小 $p$ 是 DiT 的关键超参数：$p$ 越小，token 数量越多，计算量（Gflops）越大，但模型可处理的细节层次也越高。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/dit_Snipaste_2026-09-12_22-02-39.png)

&emsp;&emsp;token 序列在进入 Transformer 块之前，会加上标准的正弦-余弦频率位置编码，使模型能够感知每个 patch 在空间中的原始位置。

---

**DiT 块设计：四种条件注入方式的对比**

&emsp;&emsp;扩散模型与普通图像分类网络的关键区别在于，去噪网络必须接收额外条件信息：当前时间步 $t$，以及可选的类别标签或文本描述。DiT 系统比较了四种将条件信息注入 Transformer 块的方式：

&emsp;&emsp;**In-context 条件**：将 $t$ 和类别标签 $y$ 的嵌入向量作为两个额外的 token 直接拼接到图像 token 序列的前面，然后让标准 ViT 块处理整个序列。训练完成后，在最终输出中移除这两个条件 token。这种方式不修改 Transformer 块结构，引入的额外 Gflops 可以忽略不计。

&emsp;&emsp;**Cross-attention 条件**：将 $t$ 和 $y$ 的嵌入拼接为长度 2 的独立序列，在标准自注意力块之后增加一个多头交叉注意力层，让图像 token 去“查询”条件嵌入。这与 LDM 中注入文本条件的方式类似。缺点是交叉注意力引入了约 15% 的额外 Gflops，是四种方式中计算开销最大的。

&emsp;&emsp;**adaLN 条件**：不直接学习维度级的缩放和偏移参数 $\gamma$ 和 $\beta$，而是用一个 MLP 从 $t$ 和 $y$ 的嵌入之和中回归出 $\gamma$ 和 $\beta$，然后对 LayerNorm 的输出做自适应调制。adaLN 是四种方式中 Gflops 增加最少的，且是唯一对所有 token 施加相同调制的条件机制。

&emsp;&emsp;**adaLN-Zero 条件**：在 adaLN 的基础上，将每个 DiT 块中回归出的缩放参数 $\gamma$ 初始化为零，同时对残差连接前的输出也做零初始化。这使得每个 DiT 块在训练开始时等价于恒等映射，模型行为与“无条件的 Transformer”完全一致，随着训练逐步学习条件调制。

&emsp;&emsp;论文的实验结论明确：adaLN-Zero 在所有条件注入方式中表现最优，且计算开销最低。一个后续的理论分析进一步确认，零初始化是 adaLN-Zero 性能优势的最关键因素，它带来了一种“渐进式”的条件注入过程，使训练过程更加稳定。

---

**训练目标：与 LDM 完全一致，只是骨干变了**

&emsp;&emsp;DiT 的训练目标与 LDM 的潜在空间噪声预测 MSE 完全相同：

$$
L_{\text{DiT}}(\theta)
=
\mathbb{E}_{z_0, \epsilon \sim \mathcal{N}(0,I), t}
\left[
\left\|
\epsilon - \epsilon_\theta(z_t, t, y)
\right\|_2^2
\right]
$$

&emsp;&emsp;其中 $z_0 = \mathcal{E}(x_0)$ 是 VAE 潜表示，$y$ 是类别标签（在条件生成实验中）。训练时 VAE 保持冻结，只有 Transformer 骨干 $\epsilon_\theta$ 被更新。这个目标与 DDPM 的 $L_{\text{simple}}$ 在形式上是相同的，区别仅在于去噪网络的架构从 UNet 换成了 Transformer。

---

**缩放行为：Gflops 越高，FID 越低**

&emsp;&emsp;DiT 最核心的实验发现是：扩散模型同样遵循清晰的缩放规律。论文以 Gflops 作为模型复杂度的统一度量，系统比较了不同深度、宽度和 patch 大小的 DiT 变体。结果显示，随着 Gflops 的增加，FID 持续下降，且这种趋势在 DiT-S、DiT-B、DiT-L、DiT-XL 四个规模层级上一致成立。

&emsp;&emsp;这一发现的意义在于：U-Net 的架构选择本身并不是扩散模型性能的瓶颈。只要把模型容量放大，Transformer 可以比 U-Net 更有效地利用新增的计算量。DiT-XL/2 以 118.6 Gflops 的复杂度，在 ImageNet 256×256 上取得了 FID 2.27，超过了同等或更高 Gflops 的所有 U-Net 基线。

&emsp;&emsp;DiT 的设计空间可以概括为三个维度：patch 大小 $p$（控制 token 数量）、Transformer 的深度和宽度（控制每 token 的计算量）、以及注意力头的数量。论文提供了 DiT-S/2、DiT-B/2、DiT-L/2、DiT-XL/2 等一系列配置，其中 DiT-XL/2 表示最大的模型使用 $p=2$。

---

**实验结果与关键发现**

&emsp;&emsp;DiT 在 ImageNet 256×256 和 512×512 的类别条件生成任务上进行了全面评估。核心结果包括：

- **条件注入方式**：adaLN-Zero 显著优于 in-context、cross-attention 和 vanilla adaLN，且 Gflops 开销最低；
- **模型规模**：DiT-XL/2 在 256×256 上达到 FID 2.27，在 512×512 上达到 FID 3.04，均为当时最优；
- **patch 大小**：在相同模型规模下，$p=2$ 的 DiT 优于 $p=4$ 和 $p=8$，因为更多的 token 让模型有更细粒度的空间处理能力；
- **与 U-Net 的对比**：在相似 Gflops 下，DiT 的 FID 优于 ADM、LDM 等 U-Net 基线，且随着规模增大，优势扩大。

&emsp;&emsp;一个值得注意的细节是，DiT 的训练不需要任何 U-Net 特有的技巧（如多尺度特征融合、跳跃连接），标准 Transformer 加上 adaLN-Zero 条件注入就能达到最优性能。这进一步支持了论文的核心论点：扩散模型的性能更多取决于模型容量和训练规模，而非骨干网络的具体拓扑结构。

---

**与 U-Net 的关系及后续影响**

&emsp;&emsp;DiT 并没有否定 U-Net 的有效性，而是揭示了 U-Net 的归纳偏置在足够大的模型和数据规模下并不是必需的。卷积的局部性在数据有限时是一种有用的先验，但当模型容量和训练数据足够充分时，Transformer 的全局注意力机制可以更有效地利用计算资源。

&emsp;&emsp;DiT 对后续工作产生了深远影响：

- **Sora**：OpenAI 的视频生成模型 Sora 在 DiT 基础上引入时空 patch，将二维图像 patch 扩展为三维时空 patch，使 Transformer 能够同时处理空间和时间维度；
- **SiT**：谢赛宁团队在 DiT 基础上提出 Scalable Interpolant Transformers，用更灵活的插值框架替代标准扩散过程，在质量和速度上均有所提升；
- **Stable Diffusion 3 / MMDiT**：SD3 采用了基于 DiT 改进的多模态扩散 Transformer（MMDiT）架构，用 Transformer 替代了 SDXL 中的 UNet，验证了 DiT 范式在文本到图像生成中的有效性；
- **PixArt-α**：在 DiT 基础上引入更高效的训练策略和文本条件注入，大幅降低了训练成本。

&emsp;&emsp;**DiT 与 LDM 的关系可以概括为：LDM 证明了“在潜空间做扩散”是对的，DiT 证明了“用 Transformer 做骨干”也是对的**。两者结合，构成了当前大规模生成模型（从图像到视频）的主流技术底座。

## 8 Flow Matching

- 论文地址：[Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)

&emsp;&emsp;DiT 证明了扩散模型的骨干网络可以从 U-Net 换成 Transformer，但扩散过程本身仍然受限于两个根本问题：第一，前向加噪过程是预先定义的、不可学习的，模型只能被动地学习如何逆转这个固定的噪声路径；第二，这个路径通常是弯曲的，意味着逆向采样需要很多步才能达到足够的精度。Flow Matching 的核心思想是跳出“加噪-去噪”的框架，转而直接学习一个连续时间变换，让概率质量沿着一条可设计的、通常更直的路径从噪声分布流向数据分布。它建立在连续归一化流（Continuous Normalizing Flows, CNF）的理论基础上，但通过条件概率路径的构造，避开了 CNF 训练中高昂的仿真代价，使 CNF 首次可以在 ImageNet 规模上训练。Flow Matching 的主要贡献包括：

1. 提出 Flow Matching 训练目标，一种无需仿真即可训练连续归一化流的方法，通过回归固定条件概率路径的向量场来学习 CNF；
2. 证明 Flow Matching 与一般的高斯概率路径族兼容，扩散模型的路径只是其中的一个特例，且在这个特例下 Flow Matching 比标准扩散训练更稳定、更鲁棒；
3. 引入最优传输（Optimal Transport, OT）位移插值来定义条件概率路径，得到的路径比扩散路径更高效，训练和采样都更快，泛化能力也更好；
4. 在 ImageNet 上使用 Flow Matching 训练的 CNF 在似然和样本质量两个维度上均取得当时最优，且可以使用现成的数值 ODE 求解器进行快速可靠的采样；
5. 为后续的 Rectified Flow、Stochastic Interpolants 等工作奠定了理论基础，最终推动了 Stable Diffusion 3、FLUX 等模型采用流匹配作为生成框架。

---

**连续归一化流：用 ODE 代替离散变换**

&emsp;&emsp;要理解 Flow Matching，需要先理解连续归一化流。传统归一化流通过一系列精心设计的可逆变换层将简单分布映射到复杂分布，每一层都必须满足严格的可逆性约束，这严重限制了模型的设计空间。CNF 采取了一种截然不同的方法：用一个连续时间的常微分方程来定义变换。

&emsp;&emsp;具体来说，CNF 定义了一个时间相关的向量场 $v_t: \mathbb{R}^d \to \mathbb{R}^d$，数据点在这个向量场的引导下沿着如下 ODE 流动：

$$
\frac{d\mathbf{x}_t}{dt} = v_t(\mathbf{x}_t), \quad t \in [0,1]
$$

&emsp;&emsp;给定初始分布 $p_0$（通常是标准高斯分布），求解这个 ODE 可以得到任意时刻 $t$ 的分布 $p_t$。如果向量场 $v_t$ 满足适当的 Lipschitz 连续性条件，ODE 的解存在且唯一，因此这个过程是完全可逆的：从 $p_1$（数据分布）出发，沿着 $-v_t$ 积分，可以回到 $p_0$。这种双向性是 CNF 的一个重要特征。

&emsp;&emsp;CNF 的核心优势在于它用一个连续的动力学系统替代了离散的层叠结构，因此不再需要严格的架构约束来保证可逆性。但代价是训练变得困难：标准的最大似然训练需要通过 ODE 求解器进行反向传播，计算成本极高，难以扩展到大规模数据。Flow Matching 正是为了解决这个矛盾而提出的。

---

**Flow Matching 目标：回归条件向量场**

&emsp;&emsp;Flow Matching 的核心创新在于找到了一种无需仿真即可训练 CNF 的目标函数。设 $p_t$ 是从 $p_0$ 到 $p_1$ 的概率路径，$u_t(x)$ 是生成这个路径的真实向量场（即满足连续性方程 $\partial_t p_t + \nabla \cdot (p_t u_t) = 0$ 的向量场）。如果能够直接回归 $u_t$，就得到了一个正确的 CNF：

$$
\mathcal{L}_{\text{FM}}(\theta)
=
\mathbb{E}_{t \sim \mathcal{U}[0,1], x \sim p_t}
\left[
\left\|
v_\theta(x, t) - u_t(x)
\right\|_2^2
\right]
$$

&emsp;&emsp;但问题在于，$p_t$ 和 $u_t$ 都是未知的，无法直接计算。Flow Matching 的解决方案是引入条件变量 $z$，将边缘概率路径和边缘向量场分解为条件版本。给定一个条件 $z$（在图像生成中通常对应一个数据样本 $x_1$），定义条件概率路径 $p_t(x|z)$ 和对应的条件向量场 $u_t(x|z)$，使得边缘分布满足 $p_t(x) = \int p_t(x|z) q(z) dz$。Flow Matching 论文证明了一个关键定理：**回归条件向量场与回归边缘向量场产生相同的梯度**：

$$
\nabla_\theta \mathcal{L}_{\text{FM}}(\theta)
=
\nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)
$$

&emsp;&emsp;其中条件流匹配（Conditional Flow Matching, CFM）目标为：

$$
\mathcal{L}_{\text{CFM}}(\theta)
=
\mathbb{E}_{t \sim \mathcal{U}[0,1], z \sim q(z), x \sim p_t(\cdot|z)}
\left[
\left\|
v_\theta(x, t) - u_t(x|z)
\right\|_2^2
\right]
$$

&emsp;&emsp;这个定理的实践意义是根本性的：它意味着不需要知道边缘向量场 $u_t$，只需要设计一个条件概率路径 $p_t(x|z)$，并计算对应的条件向量场 $u_t(x|z)$，就可以训练 CNF。而条件路径的设计是完全自由的，只要满足 $p_0(x|z) = \mathcal{N}(0,I)$（在 $t=0$ 时所有条件都从同一噪声分布出发）和 $p_1(x|z) = \delta(x - x_1)$（在 $t=1$ 时集中在数据点上）这两个边界条件即可。

---

**条件概率路径：从高斯到数据点的直线插值**

&emsp;&emsp;Flow Matching 论文中最简单也最常用的条件概率路径是**高斯条件路径**：

$$
p_t(x|z) = \mathcal{N}(x; \mu_t(z), \sigma_t^2 I)
$$

&emsp;&emsp;其中 $\mu_t(z)$ 和 $\sigma_t$ 是时间相关的均值和标准差函数，满足 $\mu_0(z) = 0$、$\sigma_0 = 1$（初始为标准高斯）以及 $\mu_1(z) = z$、$\sigma_1 = 0$（终点为数据点 $z$）。一个特别自然的选择是：

$$
\mu_t(z) = t z, \quad \sigma_t = 1 - t
$$

&emsp;&emsp;这意味着在时刻 $t$，条件分布是一个以 $t z$ 为中心、方差为 $(1-t)^2$ 的高斯分布。当 $t=0$ 时，均值为零、方差为一，即标准高斯；当 $t=1$ 时，方差为零、均值为 $z$，即确定性地到达数据点。

&emsp;&emsp;在这个路径下，条件向量场可以闭式求出。由 $\mathbf{x}_t = t z + (1-t) \epsilon$（其中 $\epsilon \sim \mathcal{N}(0,I)$），对时间求导得到：

$$
u_t(x|z) = \frac{d}{dt} \mathbb{E}[x_t | x_1 = z] = z - \epsilon
$$

&emsp;&emsp;这个条件向量场有一个非常直观的解释：它就是从噪声 $\epsilon$ 指向数据 $z$ 的方向向量，且在整个时间区间内保持恒定。这就是 Flow Matching 中“直线路径”的来源——条件轨迹是直线，因此条件向量场不依赖于时间 $t$。训练目标因此可以简化为：

$$
\mathcal{L}_{\text{CFM}}(\theta)
=
\mathbb{E}_{t, z, \epsilon}
\left[
\left\|
v_\theta(t z + (1-t)\epsilon, t) - (z - \epsilon)
\right\|_2^2
\right]
$$

&emsp;&emsp;训练过程可以概括为：

```text
重复：
  z ~ 数据分布
  t ~ Uniform(0, 1)
  ε ~ N(0, I)
  对以下损失做梯度下降：
    || vθ(t * z + (1-t) * ε, t) - (z - ε) ||²
直到收敛
```

&emsp;&emsp;与 DDPM 的 $L_{\text{simple}}$ 对比：DDPM 让网络预测“我加进去的噪声是什么”，Flow Matching 让网络预测“从噪声指向数据的方向是什么”。两者的形式相似，但 Flow Matching 的目标不涉及任何噪声调度参数（如 $\bar{\alpha}_t$），时间 $t$ 在 $[0,1]$ 上均匀采样，且条件向量场不依赖于 $t$，因此训练更加简洁。

---

**最优传输路径：让边缘路径也变直**

&emsp;&emsp;上述条件路径虽然简单，但条件轨迹的直线性并不意味着边缘路径也是直的。边缘向量场 $u_t(x) = \mathbb{E}[u_t(x|z) | x_t = x]$ 是对所有可能的数据点 $z$ 的条件向量场的加权平均，这个平均会导致边缘路径出现弯曲。路径越弯，采样时需要越多的 ODE 求解步数才能达到足够的精度。

&emsp;&emsp;Flow Matching 论文引入**最优传输位移插值**来改善这一问题。最优传输的核心思想是寻找从噪声分布到数据分布的最“经济”的搬运方案，使得总搬运距离最小。在条件路径的构造中，使用最优传输意味着将每个噪声样本 $\epsilon$ 与数据点 $z$ 配对时，选择的配对方案应该使得 $\|z - \epsilon\|$ 尽可能小。具体来说，条件路径变为：

$$
\mu_t(z) = t z, \quad \sigma_t = 1 - t
$$

&emsp;&emsp;但配对方式从“随机配对”（每个 $\epsilon$ 与随机采样的 $z$ 配对）变为“最优配对”（通过求解小批量最优传输问题，找到使总距离最小的配对）。使用最优传输配对后，边缘路径中的“交叉”现象大幅减少，路径变得更加笔直。

&emsp;&emsp;论文的实验结果明确：使用 OT-CFM（Optimal Transport Conditional Flow Matching）在 ImageNet 上训练的模型，不仅在 FID 上优于标准 CFM，而且采样所需的 ODE 步数更少。这直接验证了“路径越直，采样越快”的直觉。

---

**采样：从噪声出发，沿向量场积分**

&emsp;&emsp;训练完成后，生成过程就是从标准高斯噪声出发，求解 ODE：

$$
\frac{d\mathbf{x}_t}{dt} = v_\theta(\mathbf{x}_t, t), \quad \mathbf{x}_0 \sim \mathcal{N}(0,I)
$$

&emsp;&emsp;从 $t=0$ 积分到 $t=1$，得到的数据分布样本 $\mathbf{x}_1$ 就是生成结果。采样过程可以概括为：

```text
x0 ~ N(0, I)
for t = 0, Δt, 2Δt, ..., 1:
    x_{t+Δt} = x_t + vθ(x_t, t) * Δt
return x1
```

&emsp;&emsp;与扩散模型不同，Flow Matching 使用的是确定性的 ODE 采样，每一步的更新量仅由当前向量场决定，不涉及随机噪声注入。这使得采样过程可以搭配标准的高阶 ODE 求解器（如 Runge-Kutta 方法）来提高精度。当路径足够直时，甚至可以使用 Euler 方法以很少的步数（如 10 到 50 步）生成高质量样本，而 DDPM 通常需要 1000 步、DDIM 需要 50 到 100 步。

---

**与扩散模型的关系**

&emsp;&emsp;Flow Matching 与扩散模型的关系可以精确地表述为：**扩散模型是 Flow Matching 的一个特例**。扩散模型的前向过程 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ 定义了一族高斯条件概率路径，其中均值函数为 $\mu_t(z) = \sqrt{\bar{\alpha}_t} z$、标准差函数为 $\sigma_t = \sqrt{1-\bar{\alpha}_t}$。这恰好符合 Flow Matching 框架中条件概率路径的一般形式 $\mathcal{N}(x; \mu_t(z), \sigma_t^2 I)$，只不过均值和标准差的函数形式与 Flow Matching 中常用的线性插值不同。

&emsp;&emsp;因此，Flow Matching 论文指出，使用扩散路径的 Flow Matching 训练目标与标准扩散训练目标在本质上一致，但 Flow Matching 的表述更加统一和简洁。更重要的是，Flow Matching 的实验发现，即使用扩散路径，Flow Matching 的训练也比标准扩散训练更稳定、更鲁棒。这暗示了标准扩散训练中某些看似必要的技巧（如噪声调度的精细设计、VLB 中的加权项）可能并非最优选择。

&emsp;&emsp;两者最根本的区别在于设计哲学：扩散模型的前向过程是预先定义的、不可学习的，模型只能被动地学习如何逆转它；Flow Matching 则把“从噪声到数据的路径”本身作为一个可设计的对象，允许研究者选择更直、更高效的路径。这种从“学习逆转固定路径”到“设计并学习最优路径”的转变，是 Flow Matching 在效率上超越扩散模型的理论根源。

---

**与 Rectified Flow 的区别**

&emsp;&emsp;Rectified Flow 是 Flow Matching 同期发展的一个密切相关的工作，两者经常被混淆。Rectified Flow 的核心思想是通过“拉直”过程来改善 ODE 路径：先用标准方法训练一个流模型，然后用这个模型生成配对样本，再用这些配对样本重新训练一个更直的流模型，如此迭代。这个过程称为“reflow”，它可以逐步减小路径的曲率，最终实现单步或极少步生成。

&emsp;&emsp;两者的关键区别在于：Flow Matching 通过设计条件路径（如最优传输插值）来获得较直的边缘路径，是一次性的设计选择；Rectified Flow 则通过迭代的 reflow 过程来逐步拉直路径，是一个多阶段的优化过程。在实践中，Flow Matching 的 OT-CFM 变体和 Rectified Flow 经常结合使用——先用 OT-CFM 训练一个较好的初始流，再用 reflow 进一步拉直。

---

**实际应用：Stable Diffusion 3 与 FLUX**

&emsp;&emsp;Flow Matching 从理论走向大规模实践的关键一步是 Stable Diffusion 3。SD3 采用了基于 DiT 改进的多模态扩散 Transformer（MMDiT）架构，但将生成框架从扩散模型替换为 Flow Matching。SD3 使用条件流匹配目标进行训练，并在采样时使用 Euler 求解器，以较少的步数生成高质量图像。Stable Diffusion 3 的成功证明了 Flow Matching 不仅在理论上有优势，在实际的大规模文本到图像生成中也能达到甚至超越扩散模型的水平。

&emsp;&emsp;FLUX 是另一个采用 Flow Matching 的大型文生图模型，由 Black Forest Labs 开发。FLUX 使用了与 SD3 类似的 MMDiT 架构和 Flow Matching 训练框架，但在模型规模、训练数据和架构细节上做了进一步优化，在生成质量和提示跟随能力上达到了新的高度。

&emsp;&emsp;Flow Matching 还被广泛应用于视频生成、语音合成、机器人轨迹规划等领域。在视频生成中，Flow Matching 的确定性 ODE 采样比扩散模型的随机采样更加稳定，有利于生成时序连贯的视频。在机器人领域，Flow Matching 被用于学习从噪声到动作轨迹的连续变换，展现出比扩散策略更高的采样效率。

---

**后续影响与局限**

&emsp;&emsp;Flow Matching 的理论框架统一了扩散模型、连续归一化流和最优传输三个看似独立的研究方向，为生成建模提供了一个更加清晰和灵活的范式。它的后续发展包括：

- **Stochastic Interpolants**：将 Flow Matching 从 ODE 推广到 SDE，允许在采样过程中注入可控的随机性，在效率和多样性之间提供更灵活的权衡；
- **Shortcut Models**：在 Flow Matching 基础上引入自一致性约束，训练模型在不同时间步之间进行跳跃，实现更少步数的生成；
- **MeanFlow**：通过在拉直的 Rectified Flow 轨迹上训练平均速度场，实现单步生成；
- **Consistency Flow Matching**：将一致性模型的思想引入 Flow Matching，使模型在训练过程中逐步学习从任意时间步直接映射到数据。

&emsp;&emsp;Flow Matching 的局限在于：它的采样仍然是确定性的，这意味着对于同一初始噪声，生成的样本是确定的。虽然这带来了更好的可控性和稳定性，但也意味着在需要多样性的场景中，必须通过改变初始噪声来获得不同的输出。相比之下，扩散模型的随机采样天然具有多样性。此外，Flow Matching 的理论优势在低维数据上最为明显，在高维图像数据中，条件路径的直线性与边缘路径的直线性之间的差距仍然存在，如何进一步缩小这个差距是当前研究的一个活跃方向。

&emsp;&emsp;从更宏观的视角看，Flow Matching 代表了生成模型从“加噪-去噪”范式向“路径设计”范式的一次重要转变。它告诉我们，生成模型的核心不在于噪声本身，而在于如何设计一条从简单分布到复杂分布的高效路径。这一思想正在深刻影响着下一代生成模型的设计。

## 9 Stable Diffusion 3

- 论文地址：[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206)

&emsp;&emsp;DiT 证明了 Transformer 可以作为扩散模型的有效骨干，Flow Matching 提供了比扩散更简洁、更高效的路径设计框架。但这两条线索在 SDXL 时代尚未真正汇合——SDXL 仍然使用 UNet 骨干和标准扩散目标。Stable Diffusion 3 的核心工作，正是把 DiT 的 Transformer 架构与 Rectified Flow 的流匹配框架结合起来，并针对文本到图像生成的特殊需求，设计了一种全新的多模态骨干网络。SD3 的主要贡献包括：

1. 提出 MMDiT（Multimodal Diffusion Transformer）架构，对文本和图像两种模态使用独立的权重集，但通过联合注意力机制实现双向信息流动，显著提升文本理解和文字渲染能力；
2. 将 Rectified Flow 引入大规模文生图训练，并提出一种偏向中间时间步的重新加权采样策略，使 RF 模型在增加采样步数时性能持续提升，而不会像传统 RF 那样在步数增多后性能下降；
3. 建立清晰的缩放规律，验证损失与自动图像对齐指标（GenEval）和人类偏好评分（ELO）高度相关，为模型规模选择提供了可靠依据；
4. 训练了从 8 亿到 80 亿参数的模型系列，最大模型在视觉美观度、提示遵循和文字渲染方面全面超越 DALL·E 3、Midjourney v6 等同期系统；
5. 公开实验数据、代码和模型权重，延续 Stability AI 的开源路线，推动了流匹配 Transformer 在开源社区的普及。

---

**从 DiT 到 MMDiT：独立权重与双向注意力**

&emsp;&emsp;DiT 的设计假设去噪网络的输入是单一模态的潜表示，条件信息（时间步、类别标签或文本嵌入）作为额外的 token 或调制信号注入。但在文生图任务中，文本和图像是两种本质不同的模态：文本是离散的、语义密集的 token 序列，图像是连续的、空间结构化的潜表示。如果简单地将文本 token 和图像 token 拼接后送入同一组 Transformer 权重，模型可能难以同时建模两种模态的独特特征。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/sd3_Snipaste_2026-09-12_22-11-45.png)

&emsp;&emsp;SD3 的解决方案是 MMDiT：对文本和图像分别使用独立的权重集。具体来说，文本 token 经过一组线性投影和 LayerNorm 参数，图像 token 经过另一组独立的投影和归一化参数。但在注意力计算时，两个模态的序列被拼接在一起，进行联合的全局自注意力。这意味着文本 token 可以“看到”图像 token，图像 token 也可以“看到”文本 token，信息在两个模态之间双向流动。注意力输出后再分别通过各自模态的独立前馈网络和残差连接。

&emsp;&emsp;这种设计的关键优势在于：每个模态在自己的特征空间中独立工作，避免了共享权重可能带来的模态特异性信息损失；同时，联合注意力让两个模态在语义层面对齐，使模型能够精确地将文本描述中的概念映射到图像中的空间位置。SD3 论文报告，MMDiT 在训练过程中的视觉保真度和文本对齐度均优于 UViT 和 DiT 等骨干网络。

---

**文本条件注入：三种编码器的融合**

&emsp;&emsp;SD3 使用了三种文本编码器来提取文本表示：两个 CLIP 模型（CLIP ViT-L 和 OpenCLIP ViT-bigG）和一个 T5 模型。CLIP 编码器擅长捕捉视觉-语义对齐，T5 则提供了更强的语言建模能力，尤其在处理复杂句式和长文本描述时更为可靠。三个编码器的输出经过投影后融合为统一的文本嵌入序列，其中一部分用于 MMDiT 的联合注意力，另一部分通过池化后与时间步嵌入拼接，用于自适应 LayerNorm 的调制。

&emsp;&emsp;这种多编码器融合的设计，使 SD3 在文字渲染和复杂提示遵循任务上有了质的提升。SD3 能够准确生成图像中的英文单词，且文字与背景的融合更加自然，这在之前的文生图模型中是一个显著的短板。

---

**Rectified Flow 与重新加权采样**

&emsp;&emsp;SD3 采用 Rectified Flow 作为生成框架。在 Rectified Flow 中，数据和噪声在训练期间以线性轨迹连接：给定数据点 $z_1$ 和噪声 $z_0 \sim \mathcal{N}(0,I)$，中间时刻的样本为 $z_t = t z_1 + (1-t) z_0$，目标向量场为 $v = z_1 - z_0$。训练目标为：

$$
\mathcal{L}_{\text{RF}}(\theta)
=
\mathbb{E}_{t, z_0, z_1}
\left[
\left\|
v_\theta(z_t, t) - (z_1 - z_0)
\right\|_2^2
\right]
$$

&emsp;&emsp;但 SD3 发现，标准 RF 训练在少步采样时表现良好，随着采样步数增加，其相对性能反而下降——这与扩散模型的行为相反。SD3 的解决方案是引入一种偏向中间时间步的重新加权采样策略。作者假设，轨迹的中间部分对应更具挑战性的预测任务，因为此时数据信号和噪声信号都不可忽略，模型需要同时理解两者。通过对时间步采样分布进行重新加权，给中间区域更多权重，SD3 的 RF 变体在增加采样步数时性能持续提升，而不会出现标准 RF 的性能退化。

&emsp;&emsp;这一改进的实践意义是显著的：SD3 可以在 50 步采样下生成 1024×1024 分辨率的高质量图像，而无需依赖 DDIM 或 DPM-Solver 等后处理加速技巧。

---

**缩放规律：验证损失作为可靠指标**

&emsp;&emsp;SD3 进行了一项大规模的缩放研究，训练了从 15 个 block、4.5 亿参数到 38 个 block、80 亿参数的模型系列。核心发现是：随着模型规模和训练步数的增加，验证损失呈现平滑下降趋势，且验证损失与 GenEval 自动对齐指标和 ELO 人类偏好评分之间存在强相关性。

&emsp;&emsp;这一发现的意义在于，它提供了一种可预测的模型选择策略：研究者可以通过验证损失来评估不同规模配置的预期性能，而不需要为每个配置都进行昂贵的人类评估。SD3 的最大模型 SD3-8B 在人类偏好评估中，视觉美观度几乎胜过所有同期模型，语义理解平均胜率超过 60%，文字渲染能力对 Midjourney V6 的胜率超过 80%，对 DALL·E 3 的胜率接近 70%。

---

**与 SDXL 的关系及后续影响**

&emsp;&emsp;SD3 与 SDXL 的关系可以概括为：SDXL 把 LDM 范式在 UNet 框架下推到了极限，SD3 则同时更换了骨干网络和生成框架——从 UNet 换到 MMDiT，从扩散换到 Rectified Flow。这不是对 SDXL 的渐进改进，而是一次范式级别的更新。

&emsp;&emsp;SD3 对后续工作产生了深远影响。其 MMDiT 架构被 FLUX 直接继承并进一步放大，FLUX 在 SD3 的基础上增加了模型规模和训练数据，成为开源文生图的新标杆。SD3 的 Rectified Flow 训练策略也被后续的流匹配模型广泛采用。更重要的是，SD3 论文中建立的缩放分析方法——用验证损失预测生成质量——正在成为生成模型研究的一种标准实践，使模型开发从“试错”走向“可预测的工程”。

&emsp;&emsp;SD3 的局限在于其 MMDiT 架构虽然支持多模态，但当前主要验证的是文本-图像两种模态。论文中提到的“易于扩展到视频等模态”仍是一个需要后续工作验证的命题。此外，SD3 的 8B 模型虽然可以在 24GB 显存的消费级 GPU 上运行，但推理成本仍然显著高于 SDXL。如何在保持 MMDiT 性能优势的同时降低部署门槛，是 SD3 之后开源社区持续关注的问题。

## 10 FLUX.1 Kontext

- 论文地址：[FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space](https://arxiv.org/pdf/2506.15742)

&emsp;&emsp;FLUX.1 证明了流匹配 Transformer 在文生图任务上可以达到新的高度，但它的生成过程仍然是从纯噪声出发、一次性生成完整图像。现实中的图像创作往往不是“从零生成”，而是“在已有图像上进行有控制的修改”——换掉背景、改变物体颜色、保持角色一致性进行多轮编辑、将某个风格迁移到新场景中。这些任务要求模型既能理解输入图像的语义内容，又能执行自然语言指令指定的局部或全局修改。FLUX.1 Kontext 的核心思想是：不在文生图模型之外单独构建编辑管线，而是通过一种统一的序列拼接方法，让同一个流匹配模型同时具备生成和编辑能力。它将文本指令和参考图像的 token 拼接到目标图像的 token 序列中，让模型在去噪过程中同时“看到”指令和参考内容，从而生成既符合指令又保持上下文一致性的新图像。FLUX.1 Kontext 的主要贡献包括：

1. 提出一种统一的序列拼接方法，将文本指令 token 和参考图像 token 与目标图像 token 拼接为单一序列，用同一个流匹配模型同时处理文本到图像生成和图像到图像编辑任务；
2. 复用 FLUX.1 的双流-单流混合 Transformer 骨干，在单流块中让上下文 token 和目标图像 token 通过联合注意力进行信息融合，无需额外的编辑专用模块；
3. 引入 3D 旋转位置编码的扩展，为参考图像 token 分配独立的时间维度索引，使模型能够区分目标图像和多个上下文来源；
4. 实现强一致性多轮编辑，模型在连续编辑过程中对角色、物体和风格的视觉漂移极小，适合迭代式精修工作流；
5. 发布 KontextBench 基准，包含 1026 个图像-提示对，覆盖局部编辑、全局编辑、角色参考、风格参考和文字编辑五大任务类别，为统一图像编辑模型提供了系统评估标准。

---

**统一序列拼接：把编辑变成生成的一个特例**

&emsp;&emsp;FLUX.1 Kontext 最核心的设计是一个看似简单但极为有效的思路：无论是文生图还是图像编辑，都可以表示为“给定一些上下文 token，生成目标图像 token”的问题。

&emsp;&emsp;对于文生图任务，序列中只有文本指令 token，没有参考图像 token。模型从纯噪声出发，在文本条件的引导下生成目标图像。对于图像编辑任务，参考图像首先经过 VAE 编码器压缩为潜表示，然后切分为 patch token，与文本指令 token 一起拼接到目标图像 token 序列的前面。目标图像 token 本身在训练时由真实图像加噪得到，在推理时从纯噪声出发。模型的目标是预测一个速度向量，使得目标图像 token 沿着从噪声到清晰图像的路径流动，同时参考图像 token 和指令 token 在整个过程中保持不变，仅作为上下文条件参与注意力计算。

&emsp;&emsp;这种设计的关键优势在于：模型不需要为编辑任务单独学习一套参数或结构。编辑能力来自于模型在训练中见到的“参考图像 + 指令 + 目标图像”三元组数据，而推理时的行为与文生图完全一致——都是求解一个从噪声到图像的 ODE。唯一的区别是序列中多了参考图像 token。这意味着 FLUX.1 Kontext 可以处理任意组合的上下文：一个参考图像、多个参考图像、纯文本指令，甚至没有任何指令的纯图像补全。

---

**架构：复用 FLUX.1 骨干，扩展位置编码**

&emsp;&emsp;FLUX.1 Kontext 的 Transformer 骨干与 FLUX.1 完全一致：19 层双流块 + 38 层单流块，hidden size 为 3072，24 个注意力头，guidance embedding 开启。双流块中，文本 token 和图像 token 分别使用独立的权重集进行模态特异化处理，通过交叉注意力交换信息。单流块中，所有 token 拼接为统一序列，使用共享权重进行联合注意力计算和前馈处理。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob/imgs/flux_kontext_Snipaste_2026-09-12_22-20-36.png)

&emsp;&emsp;关键的架构扩展在于 3D 旋转位置编码的索引方案。FLUX.1 中，图像 token 的位置索引为 \((\hat{h}, \hat{w})\)，文本 token 统一为 \((0,0,0)\)。FLUX.1 Kontext 将位置编码扩展为三维 \((t, \hat{h}, \hat{w})\)：目标图像 token 的 \(t\) 维度设为零或对应其空间位置，而第 \(i\) 个参考图像 token 的 \(t\) 维度设为 \(i\)。这样，模型可以通过 \(t\) 维度的差异区分目标图像和不同来源的参考图像，同时在 \(\hat{h}\) 和 \(\hat{w}\) 维度上保持空间位置信息。

&emsp;&emsp;这种位置编码设计的一个直接好处是：模型可以自然地处理多个参考图像。如果用户提供了两张参考图——比如一张角色图和一张场景图——它们分别被赋予 \(t=1\) 和 \(t=2\) 的索引，模型能够独立地“看到”每一张参考图，并在生成目标图像时同时参考两者的语义内容，而不会将两张参考图的 token 混淆。

---

**训练目标：带上下文的流匹配**

&emsp;&emsp;FLUX.1 Kontext 的训练目标是一个带上下文条件的流匹配损失：

$$
\mathcal{L}_\theta
=
\mathbb{E}_{t \sim p(t),\, x,\, y,\, c}
\left[
\left\|
v_\theta(z_t, t, y, c) - (\epsilon - x)
\right\|_2^2
\right]
$$

&emsp;&emsp;其中 \(x\) 是目标图像的潜在表示，\(y\) 是文本指令，\(c\) 是参考图像（可以为空），\(z_t\) 是目标图像在时间 \(t\) 的插值状态，\(\epsilon\) 是高斯噪声。速度目标 \(v = \epsilon - x\) 与 FLUX.1 的流匹配目标一致：它表示从数据点 \(x\) 指向噪声 \(\epsilon\) 的方向。模型在训练时学习在给定指令 \(y\) 和上下文 \(c\) 的条件下，如何将目标图像从噪声逐步“搬运”到清晰图像。

&emsp;&emsp;与标准流匹配训练的一个关键区别是：参考图像 token 在训练过程中始终是清晰图像的编码，不参与加噪。只有目标图像 token 经历从噪声到数据的插值过程。这使得模型能够明确区分“需要生成的内容”和“作为参考的内容”，避免参考图像的信息在去噪过程中被意外破坏。

---

**KontextBench：系统评估统一编辑能力**

&emsp;&emsp;为了系统评估统一图像生成与编辑模型的性能，FLUX.1 Kontext 论文发布了 KontextBench 基准，包含 1026 个图像-提示对，覆盖五个任务类别：

- **局部编辑**：对图像的特定区域进行修改，同时保持周围上下文不变，例如改变物体颜色或替换背景；
- **全局编辑**：对整个图像进行风格或光照层面的全局变换；
- **角色参考**：给定一张角色参考图，在新场景中生成同一角色，保持外观一致性；
- **风格参考**：给定一张风格参考图，将目标图像渲染为相同的视觉风格；
- **文字编辑**：在图像中添加、替换或移除文字内容，同时保持排版自然。

&emsp;&emsp;评估结果显示，FLUX.1 Kontext 在单轮编辑质量和多轮编辑一致性两个维度上均表现优异。尤其是在多轮编辑场景中，模型对角色和物体的保持能力显著优于此前的编辑模型，视觉漂移极小。模型还在生成速度上具有明显优势，能够支持交互式编辑和快速原型设计工作流。

---

**与 ControlNet、InstructPix2Pix 的关系**

&emsp;&emsp;FLUX.1 Kontext 的编辑能力与已有的条件控制方法有本质区别。ControlNet 通过额外的空间条件（如边缘图、姿态图）来引导生成，但它的控制信号是结构性的，无法理解“把红色汽车换成蓝色”这样的语义指令。InstructPix2Pix 通过微调让模型学会遵循编辑指令，但它的编辑能力局限于训练中见过的指令类型，泛化能力有限。

&emsp;&emsp;FLUX.1 Kontext 的路线更接近“上下文学习”的范式：它不要求用户提供 mask 或结构控制图，也不要求模型为每种编辑类型单独训练。模型在训练中见到的是“指令 + 参考图像 → 目标图像”的通用格式，因此能够泛化到训练中未见过的编辑类型。这种设计使得 FLUX.1 Kontext 的操作门槛极低——用户只需要上传一张图和输入一句话，就可以完成过去需要多步手工操作才能实现的编辑。

---

**局限与后续影响**

&emsp;&emsp;FLUX.1 Kontext 的局限在于：它仍然是一个基于流匹配的迭代去噪模型，编辑过程需要多步采样，虽然比传统扩散模型快，但距离真正实时的交互式编辑仍有差距。此外，模型对参考图像的利用依赖于序列拼接和注意力机制，当参考图像的语义与目标图像差异过大时，模型可能难以在保持参考内容一致性的同时生成合理的输出。论文中提到的“零微调角色一致性”在一些极端姿态或视角变化下仍然可能出现偏差。

&emsp;&emsp;从更宏观的视角看，FLUX.1 Kontext 代表了生成模型从“单次生成”向“迭代编辑”的转变。它证明了流匹配 Transformer 不仅可以用于从零生成图像，还可以作为一个通用的图像变换引擎，在保持上下文一致性的前提下执行任意自然语言指令指定的修改。这一方向正在被后续工作继续推进：更高分辨率的编辑、视频编辑、以及多模态编辑（同时修改图像和音频或文本）都是 Kontext 架构的自然延伸。FLUX.1 Kontext 的序列拼接范式也为多模态生成模型的设计提供了一个简洁而强大的参考模板——不需要复杂的模态专用模块，只需要将所有信息表示为 token 序列，让注意力机制自己学会如何融合。

## 11 总结

&emsp;&emsp;从 DDPM 到 FLUX.1 Kontext，模型发展大致沿着“扩散范式建立—潜空间与文本条件引入—规模化与可控性扩展—Transformer 骨干与流匹配—统一生成与编辑”的路径演进：**DDPM 奠定加噪去噪基础，CLIP 提供图文对齐，LDM/Stable Diffusion 将扩散迁入潜空间并用交叉注意力注入文本，SDXL 通过扩大 UNet、双文本编码器和微条件提升质量，ControlNet 与 LoRA 分别实现空间控制和轻量微调，DiT 用 Transformer 替代 UNet 并验证缩放规律，Flow Matching 把生成转化为可设计的概率路径并统一扩散模型，SD3 以 MMDiT 和 Rectified Flow 完成范式更新，FLUX.1 进一步优化架构，最终 FLUX.1 Kontext 以统一序列拼接将生成与编辑融合，整体趋势是从像素到潜空间、从 UNet 到 Transformer、从扩散到流匹配、从单一生成走向可控多模态统一。**

