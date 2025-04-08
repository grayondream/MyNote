# 数字视频格式——YUV和RGB
## 1 YUV格式
### 1.1 YUV简介
&emsp;&emsp;YUV是一种颜色编码方式，类似于RGB颜色编码方式。YUV将亮度和色度分离，使用Y（明亮度）、U和V（色度、浓度）三个分量表示一个颜色。三个分量中UV分量只有颜色信息，如果图像只有Y分量图像就是黑白图像。一般见到YPbPr、YUV、Y'UV、YCbCr等专有名词描述的都可以成为YUV，不同的是他们使用的具体场景不同（YUV和Y'UV通常用来编码电视的模拟信号、YCbCr用来描述数字影像），在开发的过程中不需要严格区分他们。
&emsp;&emsp;YUV利用人对图像的亮度信息更加敏感的特点，可以有针对性的对UV分量进行降采样。同时，YUV格式能够有效的兼容彩色电视和黑白电视，在过去黑白电视更加普及的情况下该种方式能够保证兼容性。YUV编码格式相比于RGB编码格式更适合于网络编码传输，但是需要注意的是显示设备的相识模式依然为RGB模式，只是中间需要一次转换。

### 1.2 不同的YUV采样格式
&emsp;&emsp;YUV采样格式有：YUV444、YUV440、YUV422、YUV420和YUV411。采样格式是指按照比例降采样色度分量即UV分量，多个像素中公用一个UV分量来减少带宽。YUVABC格式中表示第一行数据中Y和UV分量采样比例为A:B，第二行数据的Y和UV的采样比例为A:C，而UV采样比例一直是1:1，以此方式不断重复。
- 4:4:4表示完全取样。
- 4:4:0表示1:1的水平取样，垂直2:1采样。
- 4:2:2表示2:1的水平取样，垂直1:1采样。
- 4:2:0表示2:1的水平取样，垂直2:1采样。
- 4:1:1表示4:1的水平取样，垂直1:1采样。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/yuv.png)

&emsp;&emsp;也就是说对于16像素8bit（每个像素的一个分量占用8bit）的图像：
- YUV444占用3×16=48个字节；
- YUV440占用16+2×2×4=32个字节，是YUV444的$\frac{2}{3}$；
- YUV422占用16+2×2×4=32个字节，是YUV444的$\frac{2}{3}$；
- YUV420占用16+2×2×2=24个字节，是YUV444的$\frac{1}{2}$；
- YUV411占用16+2×2×2=24个字节，是YUV444的$\frac{1}{2}$；

&emsp;&emsp;为了验证该猜想可以使用ffmpeg命令解码jpeg图像来查看大小：
```bash
#将图像缩放成100*100解码成对应的格式，将其中的422换成对应的采样格式就可以
ffmpeg -i .\1.jpg -s 100x100 -pix_fmt yuv422p 100x100_yuv422p.yuv
```
&emsp;&emsp;对于YUV422格式图像的大小应该为$100×100×3×\frac{2}{3}=20000$字节。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/yuvsize.PNG)

### 1.3 YUV存储格式
&emsp;&emsp;YUV 数据有两种存储格式：平面格式（planar format）和打包格式（packed format）。
- planar format：先连续存储所有像素点的 Y，紧接着存储所有像素点的 U，随后是所有像素点的 V。
- packed format：每个像素点的 Y、U、V 是连续交错存储的。

### 1.4 YUV存储格式
&emsp;&emsp;YUV根据不同的存储格式和不同的采样格式分为多种格式比如NV12，NV21，YUYV等格式，具体的格式后续用到的话会在这里补充。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/yuv420.PNG)

### 1.5 读取YUV数据分离YUV分量
&emsp;&emsp;这里使用YUV420P数据进行分离读取。
```c
void split_yuv420p_yuv2file(const char *yuvfile, int height, int width, const char *yfile, const char *ufile, const char *vfile){
	FILE *yuvfp = fopen(yuvfile, "rb");
	FILE *yfp = fopen(yfile, "wb");
	FILE *ufp = fopen(ufile, "wb");
	FILE *vfp = fopen(vfile, "wb");
	if (!(yuvfp && yfp && ufp && vfp)) {
		return;
	}

	long long size = width * height;
	unsigned char *buffer = (unsigned char*)malloc(size * 3 / 2);
	
	fread(buffer, 1, size, yuvfp);
	fwrite(buffer, 1, size, yfp);
	fwrite(buffer + size, 1, size / 4, ufp);
	fwrite(buffer + size + size / 4, 1, size / 4, vfp);

	free(buffer);
	fclose(ufp);
	fclose(vfp);
	fclose(yfp);
	fclose(yuvfp);
}
```

&emsp;&emsp;下面是解析出来的图像的yuv数据和Y分量数据。
|![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/yuv420pimg.PNG)|![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/yuv420pimgy.PNG)|
--|--

## 2 RGB格式
### 2.1 简介
&emsp;&emsp;RGB颜色模式就是使用R、G、B三原色表示颜色的一种编码格式，屏幕的显示方式基本都是RGB显示。RGB格式若按照编码存储方式可以分为RGB555、RGB565、RGB24(RGB888)和RGB32。格式不同是因为在构成一个像素的不同颜色所占的位数以及位数比例不同。

### 2.2 不同的格式
- RGB555(高彩色)
  - R = color & 0x7C00, (获取高字节的5个bit)  
  - G = color & 0x03E0, (获取中间5个bit)
  - B = color & 0x001F, (获取低字节5个bit)
- RGB565(高彩色)
  - R = color & 0xF800, (获取高字节的5个bit)
  - G = color & 0x07E0, (获取中间6个bit)
  - B = color & 0x001F, (获取低字节5个bit)
- RGB24(真彩色)
  - R = color & 0x000000FF
  - G = color & 0x0000FF00
  - B = color & 0x00FF0000
- RGB32(真彩色)
  - 低8位保留
    - R = color & 0x0000FF00
    - G = color & 0x00FF0000,
    - B = color & 0xFF000000,
  - 低8位为ALPHA值
    - R = color & 0x0000FF00,
    - G = color & 0x00FF0000,
    - B = color & 0xFF000000,
    - A = color & 0x000000FF

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/rgb.drawio.svg)

## 3 RGB和YUV转换关系

&emsp;&emsp;RGB和YUV的转换关系了解一下就行，具体实现有很多种为了保证画面没有色差和丢失的计算方式，可以参考[YUV](https://zh.wikipedia.org/wiki/YUV)。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/rgb2yuv.PNG)

## 4 参考
- [yuv-wiki](https://en.wikipedia.org/wiki/YUV)
- [YUV详解](https://blog.csdn.net/u014260892/article/details/51883723)
- [一文理解YUV](https://zhuanlan.zhihu.com/p/75735751)
- [YUV色彩格式总结](https://zhuanlan.zhihu.com/p/51394272)
- [如何理解 YUV ？](https://zhuanlan.zhihu.com/p/85620611)
- [视音频数据处理入门：RGB、YUV像素数据处理](https://blog.csdn.net/leixiaohua1020/article/details/50534150)