# 1 视频压缩
>&emsp;&emsp;为什么需要对视频进行编码？

&emsp;&emsp;视频数据本身就是序列图像，视频中每一帧图像的每一个像素都需要一个甚至于多个字节存储，如果视频文件直接存储rawdata到文件中则其文件大小会异常大。比如1920x1080的8bit视频，FPS按照30帧计算1个小时的影片则需要1920x1080x3600x30x3=671,846,400,000byte=625Gbyte。这样的文件大小即便对于当前价格比较亲民的硬盘来说也时不小的负担。因此需要利用编码技术对视频数据分析编码，去除其中的冗余，减小数据量，减轻网络宽带和硬件设备的压力。

>&emsp;&emsp;怎么对视频进行编码？

&emsp;&emsp;视频编码的基本原理是利用视频数据中空域、时域和统计冗余，构建不同区域之间的相关性，将内容参数化存储。
- 空域冗余：即当前帧图像中不同区域出现重复或者相似内容，可通过某种映射构建二者的关系，只需要保留前者以及映射关系就可以推算出后者。视频编码中使用帧内预测编码技术进行处理。
- 时域冗余：即不同帧间的有轻微的偏差，可以通过某种映射关系构建二者的关系，仅需要保留前一帧的大部分内容，以及后一帧的少部分内容，以及相关的映射关系就可以完整构建出第二帧。
- 统计冗余：即数据中统计上的不均匀性导致某些值不断出现，部分值呈现冗余性，利用熵编码技术将经常出现的值赋予短码，不经常出现的值赋予长码来压缩数据。

# 2 AVC/H264简介
&emsp;&emsp;H264是由ITU-T（International Telecommunication Union）和ISO/IEC（International Organisation for Standardisation / International Electrotechnical Commission）发布的工业化数字视频编解码标准。该标准由 ITU-T 和 ISO/IEC 联合开发，将解决所有视频应用，包括低比特率无线应用、标清和高清广播电视、互联网视频流、高清传输 DVD 内容，以及用于数字影院应用的最高质量视频。
- [H264/AVC-Standard PDF](https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-H.264-201602-S!!PDF-E&type=items)
- [Overview of H.264/AVC Video Coding Standard](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.92.4000&rep=rep1&type=pdf)

# 
## 3.1 H264 帧间预测
## 3.2 H264 帧内预测

## 3.3 H264 CABAC
## 3.4 H364 环路滤波
## 3.5 Picture Management 
## 3.6 Transform and Quantization

![](img/history-h264.png)