# ORB算法实现原理源码分析

&emsp;&emsp;摘要：本文主要描述ORB算法原理以及opencv中ORB算法的实现。
&emsp;&emsp;关键字：ORB,FAST,BRIEF

> &emsp;&emsp;ORB算法ICCV论文：[ORB：an efficient alternative to sift or surf](http://www.gwylab.com/download/ORB_2012.pdf)。

## 1 ORB算法简介

&emsp;&emsp;ORB(Oriented Fast and Rotated BRIEF)是一种快速局部特征点提取和快速计算局部特征点描述子算法。该算法分为两个部分：特征点提取和特征点描述子提取。其中特征掉提取是根据FAST改进而来的oFAST；而特征点描述子提取是根据BRIEF(Binary Robust IndependentElementary Feature)特征描述算法改进而来。

$$ORB=Oriented\quad Fast(keypoint)+Rotated\quad BRIEF(descriptor)$$

## 2 算法原理
### 2.1 oFAST: FAST Keypoint Orientation
**FAST**
&emsp;&emsp;FAST算法计算性能比较好但是不具备尺度不变性，旋转不变性以及检测到的特征点没有旋转方向。oFAST正是针对这些缺点改进FAST而来。
**FAST：Features from Accelerated Segment Test**
>&emsp;&emsp;FAST：[Features from Accelerated Segment Test (FAST)](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV1011/AV1FeaturefromAcceleratedSegmentTest.pdf)

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/fast.png)

&emsp;&emsp;FAST在进行特征点提取是根据当前点邻域内的点的差值作为特征点的筛选标准，它假定如果一个点与周围邻域内足够多的点的差值够大则认为是一个特征点：
	- 选择像素$p$，该像素的像素值为$I_{p}$，确定一个筛选的阈值$T$（测试集参考值20%）；
	- 计算以像素$p$为圆心3为半径确定16个像素点的灰度值和圆心$p$的灰度值$I_{p}$的差值，如果存在连续$n$个点（算法的第一个版本的n取值为12）满足$I_x-I_p>|t|$（$I_x$表示以$p$为圆心的点的灰度值，$t$为根据$T$计算出的偏移量），则认为点$p$可以作为一个候选点，否则剔除；
		- 对于16个点都计算差值时间复杂度过高，因此FAST采用即特征点过滤的方式：先判断圆上1,5,9,13号4个点中如果至少3个点满足特征点初选的要求在进行逐个计算，否则终止计算；
	- 遍历图像中每一个像素，重复上述操作，直到遍历结束；

&emsp;&emsp;上面的筛选方式是FAST初版的筛选方式，但是该方式有一些基本的缺陷：
1. 邻域内点数量$n$过小，容易导致筛选的点过多，降低计算效率；
2. 筛选的点完全取决于图像中点的分布情况；
3. 筛选出的点部分点集中在一起。

&emsp;&emsp;问题1,2可以通过机器学习的方式来解决，问题3通过飞机大致抑制解决。
&emsp;&emsp;机器学习的解决方式是针对应用场景训练一个决策树，利用该决策树进行筛选：
>&emsp;&emsp;[Machine learning for high-speed corner detection](https://www.edwardrosten.com/work/rosten_2006_machine.pdf)

- 确定训练集；
- 对于训练集中每一张图像运行FAST算法筛选出图像中的特征点构成集合$P$；
- 对于筛选出的每一个特征点，将其邻域内的16个像素存储为一个一维向量；

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/fastm.png)

- 对于每一个一维的邻域向量中的像素值$I_{p\rightarrow x},x\in{1,...,16}$将其通过下面的规则映射到3中状态 darker than $I_p$， brighter then $I_p$或者和$I_p$相似；
$$
\begin{equation}
S_{p\rightarrow x}\left\{
\begin{array}{ll}
d,I_{p\rightarrow x}&\le I_p - t\quad(darker)\\
s,I_p-t&\lt I_{p\rightarrow x}\lt I_p+t\quad(similar)\\
b,I_p+t&\le I_{p\rightarrow x}\quad (brighter)
\end{array}\right.
\end{equation}
$$
- 对于给定的$x$可以将集合$P$分为三类$P_d,P_s,P_b$，即$P_b=\{p\in P:S_{p\rightarrow x=b}\}$；
- 定义一个布尔变量$K_p$，表示如果$p$为角点则真，否则为假（由于有训练集所以GT我们都是已知的）；
- 使用ID3决策树分类器以$K_p$查询每个子集，以训练出正确特征点的分类器；
	- 决策树使用熵最小化来逼近，类似交叉熵。
$$
\begin{equation}
\begin{aligned}
H(P)&=(c+\overline{c})log_2(c+\overline{c})-clog_2c-\overline{c}log_2\overline{c}\\
c&=|\{p|K_p\quad is\quad true\}|（角点数量）\\
\overline{c}&=|\{p|K_p\quad is\quad false\}|（非角点的数量）
\end{aligned}
\end{equation}
$$
- 递归应用熵最小化来处理所有的集合，直到熵值为0终止计算，得到的决策树就可以用来进行特征点筛选。

>&emsp;&emsp;[ID3 Descision Tree](https://athena.ecs.csus.edu/~mei/177/ID3_Algorithm.pdf)

&emsp;&emsp;问题3可以用非极大值抑制来解决：
- 针对每个点计算打分函数$V$，打分函数的输出是通过周围16个像素的差分和
- 去除$v$值较低的点即可。
$$
\begin{equation}
V=max\left\{
\begin{array}{ll}
\sum(I_s-I_p),if\quad(I_s-I_p)\gt t\\
\sum(I_p-I_s),if\quad(I_p-I_s)\gt t
\end{array}\right.
\end{equation}
$$

**oFAST**
>&emsp;&emsp;oFAST采用的是FAST-9，邻域半径为9。

&emsp;&emsp;oFAST使用多尺度金字塔解决FAST不具备尺度不变性的问题，使用矩来确定特征点的方向，来解决其不具备旋转不变形的问题。
&emsp;&emsp;多尺度金字塔就是将给定的图像以一个给定的尺度进行缩放，来生成多层不同尺度的图像$I_1,I_2,...,I_n$。这些图像每张图像的尺寸不同但是相邻两层图像间的宽高比例相同。然后针对多个尺度的图像应用FAST算法筛选特征点。
&emsp;&emsp;oFAST使用强度质心（intensity centroid，强度质心假设角的强度偏离其中心，并且该向量可用于估算方向）来确定特征点的方向，即计算特征点以$r$为半径范围内的质心，特征点坐标到质心形成一个向量作为特征点的方向。矩的定义
$$
m_{pq}=\sum_{x,y\in r}x^p y^q I(x,y)
$$
&emsp;&emsp;矩的质心为（这里计算是限定的$r$的邻域内）：
$$
C=(\frac{m_{10}}{m_{00}},\frac{m_{01}}{m_{00}})
$$
&emsp;&emsp;特征点的方向为：
$$
\theta=atan2(m_{01},m_{10})
$$

&emsp;&emsp;oFast会使用Harris角点检测对筛选出的特征点进行角点度量，然后根据度量值将特征点排序，最后取出top-N得到目标特征点。

### 2.2 rBRIEF
**BRIEF**

>>&emsp;&emsp;[BRIEF: Binary Robust Independent Elementary Features⋆](https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf)

&emsp;&emsp;BRIEF是一种对已经检测到的特征点进行描述的算法，该算法生成一种二进制描述子来描述已知的特征点。这些描述子可以用来进行特征点匹配等操作。BRIEF摒弃了利用区域直方图描述特征点的传统方法，采用二进制、位异或运算处理，大大的加快了特征点描述符建立的速度，同时也极大的降低了特征点匹配的时间，是一种非常快速的特征点描述子算法。
&emsp;&emsp;BRIEF的目标是得到一个二进制串，该串描述了特征点的特性。描述子生成的方式：
1. 为了减少噪声干扰，首先对图像进行高斯滤波（方差为2，高斯窗口为$9\times 9$）；
2. 然后以特征点为中心，取$S\times S$大小的邻域窗口，在窗口内以一定方式选取一对点$p,q$，比较两个像素点的大小：
	a. 如果$I_p>I_q$，则当前位二进制值为1；
	b. 否则为0；
3. 不断循环，直到生成目标长度的二进制串（ORB采用256长度的二进制串）。

&emsp;&emsp;上面第二步，选取点的采样方式可以为：
- $p,q$都符合$[-\frac{S}{2},\frac{S}{2}]$的均匀采样；
- $p,q$都符合各向同性的高斯分布$[0,\frac{1}{25}S^2]$采样；
- $p$符合高斯分布$[0,\frac{1}{25}S^2]$采样，$q$符合$[0,\frac{1}{100}S^2]$采样，采样方式是首先在原点处为$p$采样，然后以$p$为中心为$q$采样；
- $p,q$在空间量化极坐标下的离散位置处进行随机采样；
- $p=(0,0)^T,q$在空间量化极坐标下的离散位置处进行随机采样；

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/brief.png)

&emsp;&emsp;BRIEF存在不具备旋转不变性，不具备尺度不变性和对噪声敏感的问题。

**steered BRIEF**
&emsp;&emsp;rBRIEF通过将筛选的特征点旋转一定的角度再计算对应的描述子来得到旋转不变性。比如点集$S$：
$$
\begin{equation}
S=\left(
\begin{array}{ll}
x_1,...,x_n\\
y_1,...,y_n
\end{array}\right)
\end{equation}
$$
&emsp;&emsp;通过旋转矩阵$R_{\theta}$（这一角度并不是固定的而是预先以2π/30为增量构建的模式查找表获取的）旋转得到$S_{\theta}$然后再点集$S_{\theta}$上计算描述子：
$$
S_{\theta}=R_{\theta}S
$$

&emsp;&emsp;上述方式虽然能够得到旋转不变性但是这种方式计算得到的描述子在不同特征点之间的区分度不是很大。因此为了解决该问题，进一步采用统计学习的方式来重新选择点的集合。
1. 构建300K个特征点训练集（ORB采用从PASCAL 2006 构建训练集）；
2. 在300K个特征点的每个点的$31\times 31$邻域内按照$M$种方法取点对，比较点对的大小形成$300K\times M$的二进制矩阵$Q$。矩阵的每一列表示对应点的二进制描述；
3. 按距离平均值 0.5 对测试进行排序，形成向量$T$。
4. 贪心搜索
	a. 将第一个测试放入结果向量 R 并将其从 T 中删除；
	b. 从 T 中获取下一个测试，并将其与 R 中的所有测试进行比较。如果其绝对相关性大于阈值，则丢弃它； 否则将其添加到 R；
	c. 重复上一步，直到$R$中有 256 个测试。如果少于 256，则提高阈值并重试。

## 3 ORB opencv实现
```cpp
// nfeatures		期望检测到的特征点数
// scaleFactor		多尺度金字塔的尺度
// nlevels			金字塔层数
// edgeThreshold	边界阈值，靠近边缘edgeThreshold以内的像素是不检测特征点的
// firstLevel		金字塔层的起始层
// wat_k			产生rBRIEF描述子点对的个数
// scoreType		角点响应函数，默认使用HARRIS对角点特征进行排序
// patchSize		邻域大小
// fastThreshold
Ptr<ORB> ORB::create(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, int wta_k, int scoreType, int patchSize, int fastThreshold){
    CV_Assert(firstLevel >= 0);
    return makePtr<ORB_Impl>(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, wta_k, scoreType, patchSize, fastThreshold);
}
```

&emsp;&emsp;ORB的具体实现直接看```detectAndCompute```就可以，```FeatureExtractor```也提供了```detect```和```compute```的接口，但是区别不大:
```cpp
/** Compute the ORB_Impl features and descriptors on an image
 * @param img the image to compute the features and descriptors on
 * @param mask the mask to apply
 * @param keypoints the resulting keypoints
 * @param descriptors the resulting descriptors
 * @param do_keypoints if true, the keypoints are computed, otherwise used as an input
 * @param do_descriptors if true, also computes the descriptors
 */
void ORB_Impl::detectAndCompute( InputArray _image, InputArray _mask, std::vector<KeyPoint>& keypoints, OutputArray _descriptors, bool useProvidedKeypoints )
```

&emsp;&emsp;下面的代码在```opencv中的modules/feature2d/orb.cpp```中，并且删除了部分不影响实际流程的宏处理比如OpenCL，以及部分Assert等相关宏。
```cpp
void ORB_Impl::detectAndCompute( InputArray _image, InputArray _mask, std::vector<KeyPoint>& keypoints,
                                 OutputArray _descriptors, bool useProvidedKeypoints ) {
    CV_Assert(patchSize >= 2);
    bool do_keypoints = !useProvidedKeypoints;
    bool do_descriptors = _descriptors.needed();
    if( (!do_keypoints && !do_descriptors) || _image.empty() )
        return;

    //ROI handling
    const int HARRIS_BLOCK_SIZE = 9;
    int halfPatchSize = patchSize / 2;
    // sqrt(2.0) is for handling patch rotation
    int descPatchSize = cvCeil(halfPatchSize*sqrt(2.0));
    int border = std::max(edgeThreshold, std::max(descPatchSize, HARRIS_BLOCK_SIZE/2))+1;

    Mat image = _image.getMat(), mask = _mask.getMat();
    if( image.type() != CV_8UC1 )
        cvtColor(_image, image, COLOR_BGR2GRAY);			//ORB采用点邻域半径内的像素和当前像素的灰度差来表征特征点，颜色信息对于特征描述意义不大

    int i, level, nLevels = this->nlevels, nkeypoints = (int)keypoints.size();
    bool sortedByLevel = true;

    std::vector<Rect> layerInfo(nLevels);			//当前层图像的rect相对于firstLevel层的位置，如果当前层的图像比firstLevel层大则采用当前图源尺寸
    std::vector<int> layerOfs(nLevels);				//当前层的数据index，用于OpenCL优化
    std::vector<float> layerScale(nLevels);			//保存从第0层到第nLevels层缩放的尺度
    Mat imagePyramid, maskPyramid;
    float level0_inv_scale = 1.0f / getScale(0, firstLevel, scaleFactor);	//第一层的缩放尺度
    size_t level0_width = (size_t)cvRound(image.cols * level0_inv_scale);	//第一层图像的宽度
    size_t level0_height = (size_t)cvRound(image.rows * level0_inv_scale);	//第一层图像的高度
    Size bufSize((int)alignSize(level0_width + border*2, 16), 0);  			//第一层图像对齐后的rowbyteperrow,+2border是为方便做HARRIS做的padding

    int level_dy = (int)level0_height + border*2;
    Point level_ofs(0, 0);
	//这里只计算每一层的尺寸并不进行实际的金字塔层的构造工作，下面的一系列计算都是为了防止越界，以及金字塔特征层是在一张Mat上而不是一个vector<Mat>
    for( level = 0; level < nLevels; level++ )
    {
		//getScale->(float)std::pow(scaleFactor, (double)(level - firstLevel))金字塔层第0层就是原图，层数越高，图像的尺度多样性越丰富，图像越小
        float scale = getScale(level, firstLevel, scaleFactor);
        layerScale[level] = scale;
        float inv_scale = 1.0f / scale;				//从这里能够看出第一层图像是最大的然后逐层减小
        Size sz(cvRound(image.cols * inv_scale), cvRound(image.rows * inv_scale));	//当前层图像的尺寸
        Size wholeSize(sz.width + border*2, sz.height + border*2);		//padding边界尺寸之后的实际尺寸
        if( level_ofs.x + wholeSize.width > bufSize.width )
        {//如果当前行存在足够的空间则优先放在当前行，由于图像是从大到小生成的y方向始终是足够的
            level_ofs = Point(0, level_ofs.y + level_dy);
            level_dy = wholeSize.height;
        }

        Rect linfo(level_ofs.x + border, level_ofs.y + border, sz.width, sz.height);
        layerInfo[level] = linfo;
        layerOfs[level] = linfo.y*bufSize.width + linfo.x;
        level_ofs.x += wholeSize.width;
    }
	
    bufSize.height = level_ofs.y + level_dy;
    imagePyramid.create(bufSize, CV_8U);
    if( !mask.empty() )
        maskPyramid.create(bufSize, CV_8U);

    Mat prevImg = image, prevMask = mask;
    // Pre-compute the scale pyramids 构造金字塔层，下
    for (level = 0; level < nLevels; ++level)
    {//从上面计算的尺寸中取出相应的尺寸缩放图像生成对应的金字塔
        Rect linfo = layerInfo[level];
        Size sz(linfo.width, linfo.height);
        Size wholeSize(sz.width + border*2, sz.height + border*2);
        Rect wholeLinfo = Rect(linfo.x - border, linfo.y - border, wholeSize.width, wholeSize.height);
        Mat extImg = imagePyramid(wholeLinfo), extMask;
        Mat currImg = extImg(Rect(border, border, sz.width, sz.height)), currMask;

        if( !mask.empty() )
        {
            extMask = maskPyramid(wholeLinfo);
            currMask = extMask(Rect(border, border, sz.width, sz.height));
        }

        // Compute the resized image 这里生成的图像边界padding的并不是一个固定的像素值而是原图像的镜像，这样能够最大程度的保留图像的信息避免额外引入的噪声的干扰
        if( level != firstLevel )
        {
            resize(prevImg, currImg, sz, 0, 0, INTER_LINEAR_EXACT);
            if( !mask.empty() )
            {
                resize(prevMask, currMask, sz, 0, 0, INTER_LINEAR_EXACT);
                if( level > firstLevel )
                    threshold(currMask, currMask, 254, 0, THRESH_TOZERO);
            }

            copyMakeBorder(currImg, extImg, border, border, border, border,
                           BORDER_REFLECT_101+BORDER_ISOLATED);
            if (!mask.empty())
                copyMakeBorder(currMask, extMask, border, border, border, border,
                               BORDER_CONSTANT+BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image, extImg, border, border, border, border,
                           BORDER_REFLECT_101);
            if( !mask.empty() )
                copyMakeBorder(mask, extMask, border, border, border, border,
                               BORDER_CONSTANT+BORDER_ISOLATED);
        }
        if (level > firstLevel)
        {
            prevImg = currImg;
            prevMask = currMask;
        }
    }

	// Get keypoints, those will be far enough from the border that no check will be required for the descriptor
	computeKeyPoints(imagePyramid, uimagePyramid, maskPyramid,
					 layerInfo, ulayerInfo, layerScale, keypoints,
					 nfeatures, scaleFactor, edgeThreshold, patchSize, scoreType, useOCL, fastThreshold);

    if( do_descriptors )
    {
        int dsize = descriptorSize();	//固定dsize=32byte就是特征的256位的二进制位描述
        nkeypoints = (int)keypoints.size();
        if( nkeypoints == 0 )
        {
            _descriptors.release();
            return;
        }

        _descriptors.create(nkeypoints, dsize, CV_8U);	//这里创建的是一个nxndesc大小的矩阵表示n个特征点的特征描述
        std::vector<Point> pattern;

        const int npoints = 512;						//256对点
        Point patternbuf[npoints];
        const Point* pattern0 = (const Point*)bit_pattern_31_;	//bit_pattern_32_是通过PASCAL 2006数据集训练好的pattern，这部分数据是硬编码。是patchSize为31时，在其邻域内选取的256个点对
        if( patchSize != 31 )
        {
            pattern0 = patternbuf;
            makeRandomPattern(patchSize, patternbuf, npoints);		//如果patchSize不为31的话会随机生成筛选的点对，而且随机数的种子是固定的保证筛选的位置都是固定的，不然同一张图不同时刻输入得到特征点不一样
        }

        CV_Assert( wta_k == 2 || wta_k == 3 || wta_k == 4 );
        if( wta_k == 2 )
            std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));
        else
        {
            int ntuples = descriptorSize()*4;
            initializeOrbPattern(pattern0, pattern, ntuples, wta_k, npoints);
        }

        for( level = 0; level < nLevels; level++ )
        {
            // preprocess the resized image
            Mat workingMat = imagePyramid(layerInfo[level]);
			//对每一层的图像进行高斯模糊
            //boxFilter(working_mat, working_mat, working_mat.depth(), Size(5,5), Point(-1,-1), true, BORDER_REFLECT_101);
            GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);
        }
		
        {
            Mat descriptors = _descriptors.getMat();
            computeOrbDescriptors(imagePyramid, layerInfo, layerScale,
                                  keypoints, descriptors, pattern, dsize, wta_k);
        }
    }
}
```
&emsp;&emsp;生成的金字塔是一整张大图，所有的图像会放到一张图像中如果几张小图能够放在同一行则是有限并排存放而不是总是按照顺序存放，如下：

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/par.png)

```cpp
static void computeKeyPoints(const Mat& imagePyramid, const UMat& uimagePyramid, const Mat& maskPyramid, const std::vector<Rect>& layerInfo, const UMat& ulayerInfo, const std::vector<float>& layerScale, std::vector<KeyPoint>& allKeypoints, int nfeatures, double scaleFactor, int edgeThreshold, int patchSize, int scoreType, bool useOCL, int fastThreshold  ) {
    int i, nkeypoints, level, nlevels = (int)layerInfo.size();
    std::vector<int> nfeaturesPerLevel(nlevels);

    // fill the extractors and descriptors for the corresponding scales
    float factor = (float)(1.0 / scaleFactor);
    float ndesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)std::pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
	//从这里能够看出检测出的n个特征点是按比例分配在多层金字塔上的，其总数为nfeatures
    for( level = 0; level < nlevels-1; level++ ) {
        nfeaturesPerLevel[level] = cvRound(ndesiredFeaturesPerScale);
        sumFeatures += nfeaturesPerLevel[level];
        ndesiredFeaturesPerScale *= factor;		//第一层的特征点数最多，越往上越少
    }
	
    nfeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);
    // Make sure we forget about what is too close to the boundary
    //edge_threshold_ = std::max(edge_threshold_, patch_size_/2 + kKernelWidth / 2 + 2);

    // pre-compute the end of a row in a circular patch
    int halfPatchSize = patchSize / 2;
    std::vector<int> umax(halfPatchSize + 2);

    int v, v0, vmax = cvFloor(halfPatchSize * std::sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(halfPatchSize * std::sqrt(2.f) / 2);
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(std::sqrt((double)halfPatchSize * halfPatchSize - v * v));

    // Make sure we are symmetric
    for (v = halfPatchSize, v0 = 0; v >= vmin; --v){
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }

    allKeypoints.clear();
    std::vector<KeyPoint> keypoints;
    std::vector<int> counters(nlevels);
    keypoints.reserve(nfeaturesPerLevel[0]*2);

    for( level = 0; level < nlevels; level++ )
    {
        int featuresNum = nfeaturesPerLevel[level];
        Mat img = imagePyramid(layerInfo[level]);
        Mat mask = maskPyramid.empty() ? Mat() : maskPyramid(layerInfo[level]);

        // Detect FAST features, 20 is a good threshold
        {使用fast进行角点检测
			Ptr<FastFeatureDetector> fd = FastFeatureDetector::create(fastThreshold, true);
			fd->detect(img, keypoints, mask);
        }

        // 移除边界附近的角点特征
        KeyPointsFilter::runByImageBorder(keypoints, img.size(), edgeThreshold);
        // Keep more points than necessary as FAST does not give amazing corners
        KeyPointsFilter::retainBest(keypoints, scoreType == ORB_Impl::HARRIS_SCORE ? 2 * featuresNum : featuresNum);

        nkeypoints = (int)keypoints.size();
        counters[level] = nkeypoints;

        float sf = layerScale[level];
        for( i = 0; i < nkeypoints; i++ ){
            keypoints[i].octave = level;
            keypoints[i].size = patchSize*sf;
        }

        std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(allKeypoints));
    }

    std::vector<Vec3i> ukeypoints_buf;
    nkeypoints = (int)allKeypoints.size();
    if(nkeypoints == 0){
        return;
    }
	
    Mat responses;
    // 使用Harris筛选FAST检测到的特征点
    if( scoreType == ORB_Impl::HARRIS_SCORE ) {
        HarrisResponses(imagePyramid, layerInfo, allKeypoints, 7, HARRIS_K);

        std::vector<KeyPoint> newAllKeypoints;
        newAllKeypoints.reserve(nfeaturesPerLevel[0]*nlevels);
        int offset = 0;
        for( level = 0; level < nlevels; level++ ) {
            int featuresNum = nfeaturesPerLevel[level];
            nkeypoints = counters[level];
            keypoints.resize(nkeypoints);
            std::copy(allKeypoints.begin() + offset,
                      allKeypoints.begin() + offset + nkeypoints,
                      keypoints.begin());
            offset += nkeypoints;

            //cull to the final desired level, using the new Harris scores.
            KeyPointsFilter::retainBest(keypoints, featuresNum);

            std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(newAllKeypoints));
        }
        std::swap(allKeypoints, newAllKeypoints);
    }

    nkeypoints = (int)allKeypoints.size();
    {	//通过质心计算当前特征点的角度fastAtan2((float)m_01, (float)m_10);
        ICAngles(imagePyramid, layerInfo, allKeypoints, umax, halfPatchSize);
    }

    for( i = 0; i < nkeypoints; i++ ) {			//将每一层经过缩放过的角点还原到原图上
        float scale = layerScale[allKeypoints[i].octave];
        allKeypoints[i].pt *= scale;
    }
}
```
&emsp;&emsp;下面两张图分别是用Harris过滤前的结果，第二张图的角点比第一张图稍微少一些但是不明显：
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/bfp.png)
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/afterfp.png)

&emsp;&emsp;下面是经过计算的点的角度，图中线相对于x轴正方向的夹角就是角点的角度：
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/afterfp_angle.png)

&emsp;&emsp;计算特征点的描述子：
```cpp
static void
computeOrbDescriptors( const Mat& imagePyramid, const std::vector<Rect>& layerInfo,
                       const std::vector<float>& layerScale, std::vector<KeyPoint>& keypoints,
                       Mat& descriptors, const std::vector<Point>& _pattern, int dsize, int wta_k )
{
    int step = (int)imagePyramid.step;
    int j, i, nkeypoints = (int)keypoints.size();
    for( j = 0; j < nkeypoints; j++ ){
        const KeyPoint& kpt = keypoints[j];
        const Rect& layer = layerInfo[kpt.octave];
        float scale = 1.f/layerScale[kpt.octave];
        float angle = kpt.angle;
        angle *= (float)(CV_PI/180.f);
        float a = (float)cos(angle), b = (float)sin(angle);
        const uchar* center = &imagePyramid.at<uchar>(cvRound(kpt.pt.y*scale) + layer.y, cvRound(kpt.pt.x*scale) + layer.x);
        float x, y;
        int ix, iy;
        const Point* pattern = &_pattern[0];
        uchar* desc = descriptors.ptr<uchar>(j);

    #if 1
        #define GET_VALUE(idx) \
               (x = pattern[idx].x*a - pattern[idx].y*b, \
                y = pattern[idx].x*b + pattern[idx].y*a, \
                ix = cvRound(x), \
                iy = cvRound(y), \
                *(center + iy*step + ix) )
    #else
	//省略
    #endif

        if( wta_k == 2 ){
            for (i = 0; i < dsize; ++i, pattern += 16){
                int t0, t1, val;
                t0 = GET_VALUE(0); t1 = GET_VALUE(1);
                val = t0 < t1;
                t0 = GET_VALUE(2); t1 = GET_VALUE(3);
                val |= (t0 < t1) << 1;
                t0 = GET_VALUE(4); t1 = GET_VALUE(5);
                val |= (t0 < t1) << 2;
                t0 = GET_VALUE(6); t1 = GET_VALUE(7);
                val |= (t0 < t1) << 3;
                t0 = GET_VALUE(8); t1 = GET_VALUE(9);
                val |= (t0 < t1) << 4;
                t0 = GET_VALUE(10); t1 = GET_VALUE(11);
                val |= (t0 < t1) << 5;
                t0 = GET_VALUE(12); t1 = GET_VALUE(13);
                val |= (t0 < t1) << 6;
                t0 = GET_VALUE(14); t1 = GET_VALUE(15);
                val |= (t0 < t1) << 7;

                desc[i] = (uchar)val;
            }
        }
        else if( wta_k == 3 ){
            for (i = 0; i < dsize; ++i, pattern += 12){
                int t0, t1, t2, val;
                t0 = GET_VALUE(0); t1 = GET_VALUE(1); t2 = GET_VALUE(2);
                val = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0);

                t0 = GET_VALUE(3); t1 = GET_VALUE(4); t2 = GET_VALUE(5);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 2;

                t0 = GET_VALUE(6); t1 = GET_VALUE(7); t2 = GET_VALUE(8);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 4;

                t0 = GET_VALUE(9); t1 = GET_VALUE(10); t2 = GET_VALUE(11);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 6;

                desc[i] = (uchar)val;
            }
        }
        else if( wta_k == 4 ){
            for (i = 0; i < dsize; ++i, pattern += 16){
                int t0, t1, t2, t3, u, v, k, val;
                t0 = GET_VALUE(0); t1 = GET_VALUE(1);
                t2 = GET_VALUE(2); t3 = GET_VALUE(3);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val = k;

                t0 = GET_VALUE(4); t1 = GET_VALUE(5);
                t2 = GET_VALUE(6); t3 = GET_VALUE(7);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val |= k << 2;

                t0 = GET_VALUE(8); t1 = GET_VALUE(9);
                t2 = GET_VALUE(10); t3 = GET_VALUE(11);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val |= k << 4;

                t0 = GET_VALUE(12); t1 = GET_VALUE(13);
                t2 = GET_VALUE(14); t3 = GET_VALUE(15);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val |= k << 6;

                desc[i] = (uchar)val;
            }
        }
        else
            CV_Error( Error::StsBadSize, "Wrong wta_k. It can be only 2, 3 or 4." );
        #undef GET_VALUE
    }
}
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/kps.png)

## 4 参考文献
- [ORB：an efficient alternative to sift or surf](http://www.gwylab.com/download/ORB_2012.pdf)
- [Features from Accelerated Segment Test (FAST)](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV1011/AV1FeaturefromAcceleratedSegmentTest.pdf)
- [Machine learning for high-speed corner detection](https://www.edwardrosten.com/work/rosten_2006_machine.pdf)
- [ID3 Descision Tree](https://athena.ecs.csus.edu/~mei/177/ID3_Algorithm.pdf)
- [BRIEF: Binary Robust Independent Elementary Features⋆](https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf)