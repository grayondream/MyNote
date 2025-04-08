# 媒体格式——JPEG
## 1 简介
&emsp;&emsp;JPEG（Joint Picture Expert Group）编解码标准是由国际标准化组织(ISO)和CCITT联合制定的静态图象有损压缩编码标准（标准也定义了无损压缩内容但是大多数系统都不支持）。JPEG是一种编解码标准不是一种文件格式，其对应的文件格式有JIF，JPEG/JFIF，JPEG/EXIF等。其中JIF（JPEG Interchange Format）是早期的JPEG文件格式，但是由于一些缺陷而没有大规模使用，JPEG/JFIF和JPEG/EXIF为在JIF的基础上的两种继任者（在JIF的基础上，去除JIF中部分字段，添加一些新的字段）。
&emsp;&emsp;JPEG图像文件通常可能的扩展名有```.jpg,.jpeg,.jpe,.jif,jfif,jfi```，而其中三个字母的扩展名是因为早期的DOS系统采用8.3的文件格式，即仅仅支持三个字母的扩展名[JPG vs. JPEG image formats](https://stackoverflow.com/questions/23424399/jpg-vs-jpeg-image-formats)。JPEG/JFIF和JPEG/EXIF无法通过扩展名区分，只能读取文件的metadata区分。
- JPEG：编解码标准；
- JIF（JPEG Interchange Format）:JPEG标准中定义的一种文件格式但是按照标准难以实现且在色彩空间定义，像素宽高比定义等方面存在缺陷而没有大范围的被使用；
- JFIF（JPEG File Interchange Format）:为了解决JIF存在的问题发展的标准文件格式之一，通常用于网络图像的传输；
- EXIF（Exchangeable image file format）:JIF的另一"扩展"，可以存储一些设备相关的元信息（照片拍摄的时间、厂商等）多用于摄像设备，比如智能机拍摄的图像格式通常就是EXIF。

&emsp;&emsp;JPEG图像的优劣：
- 优势：
  - 兼容性：现如今的操作系统基本都支持JPEG图像格式；
  - 尺寸：JPEG采用有损压缩，去除图像中人眼无法清晰分辨的部分信息以保证在不损失图像画质的前提下尽可能的压缩图像大小；
  - 后处理：更容易后处理。
- 劣势：
  - 画质：有损压缩图像在某些场景下容易出现比较明显的细节丢失。丢失如此多的数据可能会导致色调分离——颜色之间的平滑过渡丢失，使图像看起来更加块状和突兀。 它还可能导致出现伪影——边缘混叠、光晕或噪点——这会严重影响图像质量。 摄影师可以通过以原始格式保存照片来避免伪影和分色的潜在缺陷。

&emsp;&emsp;下面三张图像分别为图像质量为1,50,99的JPEG图像，能够看到其中质量为1时图像已经有比较明显的失真了。
>&emsp;&emsp;JPEG质量用来控制JPEG图像画质和图像大小的参数，具体是根据JPEG中的DCT决定，一般值越大画质越好，文件越大。并且不同的实现方式效果可能不同。
>&emsp;&emsp;据我所知，ffmpeg，windows和mac(Mac系统API写入的JPEG文件即便质量参数为0也不会有很明显的失真)的jpeg实现有差异。

|质量1|质量50|质量99|
|-|-|-|
|![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/001.jpeg)|![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/050.jpeg)|![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/099.jpeg)|

&emsp;&emsp;由于现如今JPEG主要是JPEG/JFIF和JPEG/EXIF两种格式，因此下面主要描述这两种格式（下面两张图都为10x10大小的jpeg图像）。

|JFIF|EXIF|
|-|-|
|![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/jfif.jpeg)|![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/exif.jpg)|

## 2 JFIF

>&emsp;&emsp;

## 3 Exif
## 参考文献
- [Everything you need to know about JPEG files](https://www.adobe.com/creativecloud/file-types/image/raster/jpeg-file.html)
- [jpeg.org](https://jpeg.org/)

- [wc3-JPEG](https://www.w3.org/Graphics/JPEG/)
- 