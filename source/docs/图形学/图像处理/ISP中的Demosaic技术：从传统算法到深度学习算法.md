# 从传统算法到深度学习算法：ISP中的Demosaic技术
**摘要**
**关键字**：Demosaic；传统算法；深度学习算法；ISP；图像恢复；颜色插值

## 1 Demosaic技术简介
### 1.1 背景
&emsp;&emsp;在了解Demosaic之前我们需要了解下为什么需要Demosaic技术。要生成一幅彩色图像，每个像素位置至少需要采集红、绿、蓝三个颜色通道的信息。一种方案是在光路中使用分光棱镜，将入射光投射到三个独立的传感器上。在每个传感器前加装特定颜色的滤光片，即可获取三通道的全分辨率彩色图像。但这种方案成本高昂，不仅需要三个电荷耦合器件（CCD）传感器，还对传感器的机械对齐精度提出极高要求。更具成本效益的方案是在单个传感器前加装彩色滤光阵列，使每个像素仅采集一个颜色通道的信息，再通过插值算法恢复缺失的另外两个颜色通道信息。这种技术称为Demosaic（去马赛克）技术。Demosaic技术的核心任务是根据采样到的单通道数据，估计出每个像素位置的完整RGB信息，从而生成高质量的彩色图像。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/image_process_algo_isp_digit_image_ccd.png)

&emsp;&emsp;目前最常用的彩色滤光阵列是拜耳滤波器（Bayer Filter），其排列方式如图所示。拜耳滤波器采用2x2的重复单元，其中包含两个绿色通道、一个红色通道和一个蓝色通道。这种设计基于人眼对绿色更敏感的特性，从而提高图像的亮度分辨率。由于每个像素位置仅采集一个颜色通道的信息，Demosaic算法需要通过插值方法估计出缺失的颜色信息，以生成完整的RGB图像。
![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Bayer_pattern_on_sensor_profile.svg/500px-Bayer_pattern_on_sensor_profile.svg.png)

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/image_process_algo_isp_digit_image_byer_pattern.png)

&emsp;&emsp;多说一句，Byer Filter是最常用的采样模式，也就意味着在其他场景也存在其他类型的采样模型，比如X-Trans Filter、EXR Filter等，详细细节可以参考[Bayer_filter](https://en.wikipedia.org/wiki/Bayer_filter)。

### 1.2 Demosaic技术
&emsp;&emsp;Demosaic的核心是从单通道数据中估计出缺失的颜色通道信息，是一个欠采样重建问题。设全分辨率彩色图像为\(S=(R, G, B)\)，其对应的拜耳模式采样数据为\(z_{S}=(z_{R}, z_{G}, z_{B})\)，则去马赛克问题包含两个相互关联的插值任务：一是梅花形网格插值，即补全绿色通道中缺失的半数像素；二是矩形网格插值，即补全红、蓝通道中缺失的四分之三像素。尽管这两类插值问题均可通过双线性插值、边缘导向插值等经典图像插值技术解决，但去马赛克的核心挑战在于**联合利用通道内与通道间的相关性**，从而降低图像重建误差。
![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Colorful_spring_garden_Bayer_%2B_RGB.png/250px-Colorful_spring_garden_Bayer_%2B_RGB.png)
**空间域统计**
&emsp;&emsp;已有多项研究通过实验方法对颜色通道间的相关性进行建模，相关研究涵盖小波域与空间域等不同维度。这些研究的核心结论可归纳为**恒色调假设**，该假设也是绝大多数去马赛克算法的理论基础。在色彩科学中，色调是感知色彩的三大属性之一，另外两个属性为明度和饱和度(色调通常可由颜色分量的比值定义)。在恒色调假设中，颜色通道间的相关性通过色差或色比函数的平滑性来表征。尽管这一启发式假设在相关文献中被广泛应用，但需注意的是，恒色调假设的有效性高度依赖于数据集特性。

&emsp;&emsp;在贝叶斯框架下，后验概率分布\(P(S | z_{S}, \mathcal{H})\)可通过贝叶斯公式与似然模型\(P(z_{S} | S, \mathcal{H})\)和先验模型\(P(S | \mathcal{H})\)建立关联，公式如下：
\[P\left(S | z_{S}, \mathcal{H}\right)=\frac{P\left(z_{S} | S, \mathcal{H}\right) P(S | \mathcal{H})}{P\left(z_{S} | \mathcal{H}\right)} \quad (1)\]
&emsp;&emsp;其中\(\mathcal{H}\)代表所采用的模型假设。

&emsp;&emsp;该统计建模方法可灵活纳入各类不确定性因素，例如，噪声干扰可通过似然项进行建模，多种基于最小均方误差（MMSE）的算法均可基于这一思路推导得出。相比之下，先验模型\(P(S | \mathcal{H})\)反映了对图像光谱-空间相关性的先验知识。与其他图像逆问题类似，先验模型的构建往往是算法设计的核心，其直接决定了算法性能与计算成本的权衡关系。

&emsp;&emsp;一种简单的建模策略是忽略通道间相关性，对各颜色通道独立建模，例如采用马尔可夫随机场或自然图像统计特性。尽管这种方法较为简便，但仅考虑通道内相关性的建模方式本质上对应一种假设模型\(H_{intra}\)。另一种更常用的策略是将联合概率分布\(P(S)\)分解为\(P(G)\)与\(P(R, B | G)\)的乘积，进一步可将\(P(R, B | G)\)简化为\(P(R | G) P(B | G)=P(d_{R G}) P(d_{B G})\)，其中\(d_{R G}=R-G\)、\(d_{B G}=B-G\)为通道间的色差。这种将联合概率分布进行序列式分解的建模方式被记为\(H_{inter }^{seq }\)，是现有多数去马赛克算法的理论基石。此外，也可采用并行或迭代方式对通道间相关性进行建模，这类建模方式被记为\(H_{inter }^{para }\)。

**频域确定性**
&emsp;&emsp;去马赛克问题的另一种思路是，将彩色滤光阵列采样数据视为对全分辨率彩色图像\(S=(R, G, B)\)的降采样处理。根据彩色滤光阵列的采样模式，全分辨率图像可转换为马赛克采样数据\(z\)，其数学表达式为：
\[z=\sum_{S=R, G, B} z_{S}=\sum_{S=R, G, B} M_{S} S \quad (2)\]
&emsp;&emsp;式中\(z_{R}\)、\(z_{G}\)、\(z_{B}\)分别为红、绿、蓝通道的降采样数据，掩模矩阵\(M_{S}\)用于表征彩色滤光阵列的采样模式。例如，在拜耳模式的红色像素位置，掩模矩阵取值为\([M_{R}, M_{G}, M_{B}]=[1, 0, 0]\)。

&emsp;&emsp;对于拜耳彩色滤光阵列，掩模矩阵可通过余弦函数显式表达：
\[
\begin{aligned}
z_{R}(i, j)&=M_{R}(i, j) R(i, j)=\frac{1}{4}(1-\cos \pi i)(1+\cos \pi j) R(i, j) \\
z_{G}(i, j)&=M_{G}(i, j) G(i, j)=\frac{1}{2}(1+\cos \pi i \cos \pi j) G(i, j)\\
z_{B}(i, j)&=M_{B}(i, j) B(i, j)=\frac{1}{4}(1+\cos \pi i)(1-\cos \pi j) B(i, j)
\end{aligned} \quad (3)
\]
&emsp;&emsp;其中\((i, j)\)表示像素坐标，坐标原点为\((0,0)\)。图2展示了马赛克采样数据\(z\)及各通道采样分量\(z_{R}\)、\(z_{G}\)、\(z_{B}\)的分布特征。

&emsp;&emsp;从频域角度分析去马赛克问题的一大优势在于，可直接借鉴传统数字信号处理领域的丰富理论工具。例如，滤波器组理论可用于去马赛克算法的性能分析。近期研究表明，红、绿、蓝通道降采样数据的傅里叶变换，是全分辨率图像傅里叶变换的缩放与周期延拓结果，且红、蓝通道的频谱混叠现象比绿色通道更为严重。基于这一结论，针对亮度和色度分量设计的经典抗混叠滤波器，在柯达PhotoCD数据集上可实现优异性能，且计算成本适中。此外，频域建模方法可扩展至任意类型的彩色滤光阵列模式。

&emsp;&emsp;两种方法各有优劣，需根据实际场景权衡选择。正如恒色调假设在通道间相关性较弱时不再成立，频域中的通道间频谱混叠也会使线性滤波器的设计面临挑战。与确定性建模中可能出现的过拟合问题类似，统计建模中先验模型的构建同样不可避免地需要进行近似处理。算法的最终选择取决于理论模型与观测数据的匹配度，以及内存占用、计算复杂度等工程约束条件。本文的综述目的并非评判不同算法的优劣——尽管文中也呈现了大量仿真实验结果，而是旨在阐明各类算法的异同点，从而深化对去马赛克问题的理解。

### 1.3 评价标准
&emsp;&emsp;Demosaic算法的评价标准需从客观可量化指标和主观视觉感知两个维度综合判定，二者互补且各有侧重：客观指标保证算法的技术性能可复现、可对比，主观评价则贴合人类视觉的实际体验（也是图像最终的使用场景）。
#### 1.3.4 客观评价

**均方误差（MSE）：基础误差指标，PSNR的前置计算**
&emsp;&emsp;**单通道MSE公式**。最基础的像素值偏差量化指标，反映单通道（如亮度Y、色度U/V、RGB单通道）的平均平方误差，**值越小，像素还原越精准**。
$$
\text{MSE} = \frac{1}{H×W} \sum_{i=0}^{H-1}\sum_{j=0}^{W-1} \left[ I(i,j) - K(i,j) \right]^2
$$
&emsp;&emsp;**多通道平均MSE（RGB/ Lab）**。Demosaic需评价整体色彩还原，取三通道MSE的算术平均值：
$$
\text{MSE}_{avg} = \frac{\text{MSE}_R + \text{MSE}_G + \text{MSE}_B}{3}
$$

**峰值信噪比（PSNR）：最常用的误差评价指标**
&emsp;&emsp;**单通道PSNR公式**。基于MSE的归一化指标，单位为**分贝（dB）**，**值越高，图像还原精度越高**，8位图像（$L=256$）为工程默认值，公式可简化。
通用公式：
$$
\text{PSNR} = 10 \log_{10} \left( \frac{(L-1)^2}{\text{MSE}} \right)
$$
&emsp;&emsp;8位图像简化公式（$L-1=255$）：
$$
\text{PSNR} = 10 \log_{10} \left( \frac{255^2}{\text{MSE}} \right) = 20 \log_{10} \left( \frac{255}{\sqrt{\text{MSE}}} \right)
$$
&emsp;&emsp;**多通道PSNR（RGB/YCbCr）**。
- RGB通道：直接用**多通道平均MSE**代入公式计算$\text{PSNR}_{RGB}$；
- YCbCr通道（Demosaic优选）：分别计算亮度通道$\text{PSNR}_Y$、色度通道$\text{PSNR}_{Cb}、\text{PSNR}_{Cr}$，**重点关注$\text{PSNR}_Y$**（人类视觉对亮度更敏感），色度通道仅作辅助。

**结构相似性指数（SSIM）：兼顾结构的相似度指标**
&emsp;&emsp;弥补PSNR仅关注像素点误差、忽略图像结构的缺陷，从**亮度、对比度、结构**三个维度衡量相似度，取值范围**0~1**，**越接近1，图像结构和色彩还原越贴合参考图**。
&emsp;&emsp;**单像素对的SSIM**。对参考图和输出图的任意像素窗口（Demosaic常用**3×3窗口**，适配插值的局部相关性），计算窗口内的SSIM：
$$
\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$
&emsp;&emsp;其中：
- $\mu_x、\mu_y$：参考图窗口$x$、输出图窗口$y$的**像素均值**（反映亮度）；
- $\sigma_x^2、\sigma_y^2$：窗口$x、y$的**像素方差**（反映对比度）；
- $\sigma_{xy}$：窗口$x、y$的**像素协方差**（反映结构相关性）；
- $C_1、C_2$：极小的常数，避免分母为0，工程默认值：$C_1=(0.01L)^2$，$C_2=(0.03L)^2$（8位图像$C_1=6.5025$，$C_2=58.5225$）。
&emsp;&emsp;**整图平均SSIM（MSSIM）**。Demosaic需评价整图效果，计算所有3×3窗口的SSIM算术平均值，即**平均结构相似性指数（MSSIM）**，为工程通用指标：
$$
\text{MSSIM} = \frac{1}{H×W} \sum_{i=0}^{H-1}\sum_{j=0}^{W-1} \text{SSIM}(I(i,j),K(i,j))
$$
&emsp;&emsp;**多通道SSIM**。RGB三通道分别计算$\text{SSIM}_R、\text{SSIM}_G、\text{SSIM}_B$，取平均值为$\text{SSIM}_{RGB}$；同理YCbCr通道重点计算$\text{SSIM}_Y$。

**CIE Lab色彩偏差（ΔE*ab）：专属色彩还原的精准指标**
&emsp;&emsp;PSNR和SSIM均基于RGB/YCbCr空间，与人类视觉的色彩感知存在偏差，**ΔE*ab**基于**CIE Lab色彩空间**（贴合人类视觉的均匀色彩空间），直接量化还原色彩与真实色彩的**视觉偏差**，**值越小，色彩还原越精准**。
&emsp;&emsp;**ΔE*ab核心公式**。对逐像素的CIE Lab三通道值，计算色彩欧氏距离（即色彩偏差）：
$$
\Delta E_{ab}^* = \sqrt{(L_I^* - L_K^*)^2 + (a_I^* - a_K^*)^2 + (b_I^* - b_K^*)^2}
$$
其中：$L_I^*/a_I^*/b_I^*$为参考图像素的Lab值，$L_K^*/a_K^*/b_K^*$为输出图像素的Lab值。
&emsp;&emsp;**整图平均ΔE*ab**。计算所有像素ΔE*ab的算术平均值，为Demosaic色彩还原的核心评价指标：
$$
\Delta E_{avg}^* = \frac{1}{H×W} \sum_{i=0}^{H-1}\sum_{j=0}^{W-1} \Delta E_{ab}^*(i,j)
$$


#### 1.3.5 主观评价

&emsp;&emsp;Demosaic算法的主观评价是以人类视觉感知特性为核心、遵循标准化测试规范的综合体验验收，是算法落地各类实际场景的最终判定标准，其通过专业组与普通组结合的双盲测试形式，在标准校色显示和中性光照环境下，对覆盖人像、风景、纹理等Demosaic典型挑战场景的测试图进行评价，核心围绕伪影可见度、色彩还原感知、细节视觉表现三大关键维度并结合整体观感做加权评分，重点关注摩尔纹、拉链伪影等各类伪影的视觉干扰性、色彩贴合现实场景的自然度与一致性、发丝/边缘等高频细节的可分辨性与清晰性，而非单纯对应客观量化指标，最终以贴合人眼真实观察体验、无视觉不适且适配实际使用场景为评价核心，弥补客观指标无法覆盖的人类视觉非线性感知特性偏差。
- 模糊：图像的高频信息被削弱，导致细节损失。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/image_process_algo_isp_digit_image_blur.png)
- 拉链效应：在边缘区域出现交替的亮暗条纹，类似拉链的形状。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/image_process_algo_isp_digit_image_zipper.png)
- 伪彩色：图像在 Demosaic 过程中，由于缺少对颜色信息的处理，导致颜色通道之间存在伪影。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/image_process_algo_isp_digit_image_false_color.jpg)
- 混叠：当 Demosaic 过程中，由于采样点与相邻像素的颜色值差异较大，导致混叠现象的出现。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/image_process_algo_isp_digit_image_aliaing.jpg)


## 2 传统Demosaic算法
## 3 深度学习Demosaic算法


## 参考文献
- [Bayer_filter](https://en.wikipedia.org/wiki/Bayer_filter)
- 