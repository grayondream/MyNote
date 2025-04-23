# 封装格式之GIF
## 1 GIF图简介
&emsp;&emsp;GIF(Graphics Interchange Format)图像格式是Compuserve与1987年开发的一种图像文件格式。该图像本身可以存储静态图和动态图，但如今该图像主要被用来存储动态图，且在大部分系统上都支持。但是相对于webp这些新式的动态图格式，其在颜色质量和压缩率上的表现相对不如人意。
&emsp;&emsp;该图像表达图像和一般的图像直接存储图像内容不同，而是通过一个颜色表映射来表达对应的图像的内容。也就是说图像中存在一张颜色表存储图像中出现的颜色，然后每一帧的图像通过颜色索引来表示颜色。GIF支持的最大颜色表数量为8bit256色，所以一般的动态图能够看到图像中存在明显的颜色梯度变化的效应。另外，GIF用LZW无损压缩算法压缩图像数据。
&emsp;&emsp;GIF图像存在两个版本87a和89a，本文首先就87a熟悉GIF图像的格式随后在说明89a和87a的区别。

## 2 GIF 图像格式
>&emsp;&emsp;[spec-gif87](https://www.w3.org/Graphics/GIF/spec-gif87.txt)
### 2.1 GIF 87a
&emsp;&emsp;下图是87aGIF图像的基本结构，每一块的具体内容如下：
- GIF Signature：标识当前图像为GIF；
- Screen Descriptor：描述图像的尺寸等信息；
- Global Color Table：全局颜色表；
- 图像单元：GIF图中存在多帧图像，每一帧就是一个图像单元；
- GIF Terminator：图像结束块。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gif-87a.drawio.svg)

&emsp;&emsp;下面每一小节会使用下面这张348x288的87aGIF对比具体每一个单元的值。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/87a.gif)

#### 2.1.1 GIF Signature
&emsp;&emsp;表明当前图像是一个合法的GIF图像在87a版本中是6个字节的固定值```GIF87a```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gif-signature.png)
#### 2.1.2 Screen Descriptor
&emsp;&emsp;Screen Descriptor描述一个GIF图像的基本信息比如图像的宽高，颜色表，背景颜色等内容
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/87a_screen_desc.drawio.svg)

&emsp;&emsp;Screen Descriptor前四个字节时图像的宽度和高度，分别占两个字节，之后的一个字节时图像的基本信息：
- [0, 3)bit pixel：pixel + 1图像的颜色表数量所占的位数，即$2^(pixel + 1)$为颜色表中颜色的最大数量；
- [3, 4)bit 是一个固定的0；
- [4, 7)bit cr：cr + 1表示图像的颜色深度；
- [7, 7]bit M：等于1时表示全局颜色表紧跟Screen Descriptor，当为0时背景色索引无意义。

&emsp;&emsp;之后的1个字节为背景色的索引，如果M为1则后面紧跟的就是全局颜色表。整个Screen Descriptor占6字节。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gif-screen-desc.png)

&emsp;&emsp;从二进制内容看图像的宽度为0x015c(348),高度为0x0120(288)。标志位为0xF7(11110111)，即M=1，cr=7，pixel=7，图像颜色表数量为$2^8$=256颜色，深度为7+1=8位深度。Screen Descriptor后面256x3（RGB三个字节）=768个字节就是整个颜色表。
&emsp;&emsp;GIF的颜色表有两张：全局和局部的，全局通过上面的字段M决定是否存在，1表示使用全局的即所有图像公用一个颜色表，0表示每个图像自己有一个独立的颜色表。颜色表中每个颜色有三个值RGB，分别占1个字节(即每个颜色的值域为0-255)，总共占3个字节。白色为(255,255,255)，黑色为(0,0,0)。

> &emsp;&emsp;GIF文档中还描述了当机器支持小于8bit的处理方式，但是现在的这种情况很少见不多描述。

#### 2.1.3 Image Descriptor 
&emsp;&emsp;Image Descriptor描述了每一帧图像的而具体信息比如图像的位置宽高，局部颜色表等内容。因为有些GIF为了优化图像的大小每一帧存储的是相较于前一帧或者第一帧的差，图像的宽高描述的就是这片子区域在整张图像中的位置。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/87a-image_descriptor.drawio.svg)

&emsp;&emsp;每个Image Descriptor的开头是一个字节的固定字符```0x2c```对应的ASCII码为```,```，该字段本身没有任何含义，只是作为Image Descriptor开头的标记。
&emsp;&emsp;Image Descriptor中的当前帧图像的位置和高宽是相对于Screen Descriptor中的宽高来说的。而最后一个字节类似于Screen Descriptor中的标志位字段，该字段描述了当前帧图像是否使用局部调色板（即颜色表）（局部调色板是相对于Screen Descriptor中的全局而言的）以及局部调色板的颜色数量。局部调色板和全局调色板的布局方式类似，都是rgb分别占用1个字节顺序存放在描述子之后。
- ```M```(1bit)：表示当前帧是否使用局部调色板；
- ```I```(1bit)：表示当前帧图像数据存储方式，如果为1则为交织顺序存储，0表示顺序存储。
- ```pixel```(3bit)：表示当前局部调色板的颜色数量，只有M为1时有效。

&emsp;&emsp;Image Descriptor之后紧跟的就是图像的数据，这里的颜色数据是当前帧图像使用的颜色表中的索引。
&emsp;&emsp;上面也看到了图像有两种存储方式，顺序存储是按照行优先存储，从左到右，从上到下顺序存储。另一种是角质存储，即```I```设为1。交织存储分为4个pass，每个pass根据不同的间隔存储数据：
- pass1：每间隔8行从第0行取数据；
- pass2：每间隔8行从第4行取数据；
- pass3：每间隔4行从第2行取数据；
- pass4：每间隔2行从第1行取数据。

&emsp;&emsp;这个数据并不是直接存储的，为了减小文件大小GIF使用LZW算法对数据进行压缩。
>&emsp;&emsp;LZW算法采用了一种先进的串表压缩，将每个第一次出现的串放在一个串表中，用一个数字来表示串，压缩文件只存贮数字，则不存贮串，从而使图象文件的压缩效率得到较大的提高。

&emsp;&emsp;Screen Descriptor的字节数为6字节，加上256x3字节的颜色表，我们从305开始往下找```0x2c```。然后从下图能够看出第一帧图的的区域为```(0,0,348, 288)```，标志位为0，未使用局部颜色表，并采用顺序存储。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gif-image-descrpiptor.png)

#### 2.1.4 GIF Terminator
&emsp;&emsp;GIF结束块是1个字节的固定值```0x3B```，ASCII码为```;```，当解码器读取到该值就明白EOF（End of File），后面的内容就会忽略。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gif-terminator.png)

#### 2.1.5 GIF 扩展块
&emsp;&emsp;GIF扩展块为GIF提供了更多的灵活性，让GIF包含更多的额外信息。扩展块的第一个字节为标记符```0x21```即```!```，跟进的一个字节是扩展块的功能编码号，随后的一个字节是下面数据域的字节数，之后便是功能需要的数据（最大256字节），而byte count和func data bytes组成的块可能重复多次。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/87a-gif-ext.drawio.svg)

### 2.2 GIF 89a
>&emsp;&emsp;[spec-gif89a](https://www.w3.org/Graphics/GIF/spec-gif89a.txt)

&emsp;&emsp;89a是针对87a的升级版本，相比于后者增加了一些额外的控制块更加精确的控制GIF播放。现在常见的GIF图都是89a，下面将使用下面这张10x10（小图看数据比较方便）的纯色GIF图像来理解GIF图像的基本格式。该图像包含7帧纯颜色图像（```0xff0000,0xffa500,0xffff00,0x00ff00,0x7fff,0x0000ff,0x8b00ff```），每两帧之间的间隔为0.5s。小图数据量小更加能够看清楚原图中得到具体内容。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/89a_1010.gif)

&emsp;&emsp;下面这张图是上面的GIF图的结构，左侧为块结构，右侧为图像的二进制数据（图中的指针指向二进制块的起始位置），89a支持的部分块在该图像中没有。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/89agif.drawio.svg)

#### 2.2.1 Block
&emsp;&emsp;GIF中Block的基本结构为block size+data，结构很简单就是一个带长度的数组，其基本结构类似C的结构体定义：
```c
struct gif_block{
  uint8_t data_size_in_bytes;
  uint8_t pdata;
}
```
&emsp;&emsp;当块大小为0时，是没有后面的数据项的。

#### 2.2.2 GIF Header
&emsp;&emsp;前6个字节为GIF Header，其中前三个字节为GIF表示，如果为GIF图像固定为```GIF```，后面三个字节为版本号，比如```87a```和```89a```。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gif-89a-signature.png)

#### 2.2.3 Screen Descriptor
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/89a_screen_desc.drawio.svg)

&emsp;&emsp;89a版本的屏幕描述符和87a差不多，区别是多了一个字节和字段。89a版本的屏幕描述符栈7个字节，多一个字节描述屏幕像素宽高比，以及在屏幕标志位中第四个bit用来表示后面的颜色表是否经过排序，1表示有序，0表示无序。
- [0, 3)bit pixel：pi<!--  -->xel + 1图像的颜色表数量所占的位数，即$2^(pixel + 1)$为颜色表中颜色的数量；
- [3, 4)bit s(Sort Flag)：表示后面的颜色表是否有序，一般而言颜色表的排序的依据为颜色表中颜色的使用频率，当需要排序时频率越高的越在前码字越短；
- [4, 7)bit cr(Color ResoluTion)：cr + 1表示图像的颜色深度；
- [7, 7]bit M(Global Color Table Flag)：等于1时表示全局颜色表紧跟Screen Descriptor，当为0时背景色索引无意义。

&emsp;&emsp;图像像素宽高比存储的是一个```[0,255]```的值，当值为0时表示没有值，不可用，非0时的计算方式如下，也就是说支持最宽的图像比为1:4，最高的为4:1。
$$
 Aspect Ratio = \frac{(Pixel Aspect Ratio + 15)}{64}
$$

&emsp;&emsp;如果有全局调色板，全局调色板的存储方式和87a相同，不再赘述。

&emsp;&emsp;从下面的数据中能够看出宽高都为```0x000a```即10，标志位为```0xF2```，即```11110010```（M=1，cr=7，s=0，pixel=2），pxiel为2表示GIF中颜色至少需要3bit保存，即最多8个颜色，当前的GIF为7个颜色。之后就是全局调色板，每个颜色占3个字节。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gif-89a-screen-descriptor.png)

#### 2.2.4 Application Extension
&emsp;&emsp;描述应用信息，块标识为```0xFF```。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gif-89a-screen-descriptor.png)

- 前两个字节分别为块的起始标记和块标识，起始标记为固定的```0x21```，当前块标识为```0xFF```；
- 之后的一个字节为块大小，不包含Application Data部分，当前块该值为固定的11；
- 之后便是8字节的应用标识，为ASCII码；
- 之后的3个字节为应用的标识验证。

&emsp;&emsp;这里的应用扩展块的block size为固定的11(```0x0B```)，这里的应用标识为```NETSCAPE2.0```，应用的标识验证为```0103```，而Application Data长度为0，即没有Application Data，最后为终结符。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gif-89a-application-ext.png)

#### 2.2.5 Graphic Control Extension
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/89a-gce.drawio.svg)
&emsp;&emsp;图像控制块是在89a中新添加的，主要描述每一帧图像的帧间隔等控制信息。
- ```Extension Introducer```：图像控制块的起始标记，固定为```0x21```；
- ```Graphic Control Lable```：图像控制块的标识，固定为```0xf9```；
- ```Packet Fields```：控制块的标志位：
  - 预留位 3bit：暂时无意义；
  - Disposal Method：
    - ```0```：不指定。解码器会将整个画布清空用当前帧替换；
    - ```1```：不处置。当前帧需要绘制的内容区域会覆盖上一帧需要绘制的区域；
    - ```2```：恢复到背景色。直接将当前帧绘制到已经绘制的画布上，也就是说只会覆盖上一帧基本都被保留除非被当前帧覆盖；
    - ```3```：恢复到前一帧。需要将非当前帧绘制的区域回复对应的参考帧（一般为首帧），并将需要绘制内容重新绘制；
    - ```4-7```：预留；

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/dispose.png)

  - 用户输入标记（User Input Flag） 1bit：如果设置GIF播放会根据用户的输入交互，交互的事件由应用程序决定，一般为鼠标单击等。（交互的含义是用户触发了相关的事件GIF就继续播放），如果GIF同时定义了delay Time和用户输入标记为1则无论哪一个事件先到达都会播放下一帧；
  - 透明颜色标志位 1bit：置位表示使用透明颜色；
- Delay Time：当前帧图像的图像延迟，最小精度为0.01s。

&emsp;&emsp;能够看到每一帧图像前都有一个GCB控制块，Block Size为固定的4个字节，标识Flag全为0，帧延迟为```0x0032```即```50x0.01s=500ms```，透明颜色索引为```255```。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gif-89a-gcb.png)

>&emsp;&emsp;两种不同的处置方式。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/1.gif)
<center>
<figure>
<img src="https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/0.png" />
<img src="https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/1.png" />
<img src="https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/2.png" />
<img src="https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/3.png" />
</figure>
</center>

<center>
<figure>
<img src="https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/0.png" />
<img src="https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/1.png" />
<img src="https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/2.png" />
<img src="https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/3.png" />
</figure>
</center>

#### 2.2.6 Image Descriptor
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/89a-image_descriptor.drawio.svg)

&emsp;&emsp;89a的帧描述符和87a的帧描述符基本一致都占用10个字节，只不过标志位稍有区别。89a中添加了1个字段```s```，表示局部颜色表是否排序：
- ```M```(1bit)：表示当前帧是否使用局部调色板；
- ```I```(1bit)：表示当前帧图像数据存储方式，如果为1则为交织顺序存储，0表示顺序存储。
- ```s```(1bit)：局部颜色表是否有序，排序的原则和全局颜色表类似；
- ```r```(2bit)：预留位；
- ```pixel```(3bit)：表示当前局部调色板的颜色数量，只有M为1时有效。

&emsp;&emsp;下图中将所有每一帧图像的GCB和Image Descriptor都标注出来了，稍微有点儿乱。开头为```0x2c```，图像的坐标为```(0,0,10,10)```，所有的flag为0，使用全局颜色表因此没有局部颜色表。后面紧跟的就是LZW数据了。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/gif-89a-image-descrpiptor.png)

#### 2.2.7 图像数据
&emsp;&emsp;GIF图像数据是基于颜色表的索引，时间存储的数据是经过LZW算法压缩过的，存储的方式和87a相同。数据域每个数据块的的结构如下：
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/lzw-data.drawio.svg)

- LZW Minimum Code Size：LZW算法压缩时采用的最小码字位数。
- block size：后面的数据大小；
- data：lzw压缩的数据；
- 块结束标记，始终为0。
  
&emsp;&emsp;下图中的数据的LZW最小码字位数为2，数据大小为8byte，之后的8个字节都是实际的图像数据经过LZW压缩后的内容。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/git-89a-lzw.png)


#### 2.2.8 Comment Extension
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/89a-cme.drawio.svg)

&emsp;&emsp;Comment Extension允许你将 ASCII 文本嵌入到 GIF 文件，有时被用来图像描述、图像信贷或其他人类可读的元数据,如图像捕获的 GPS 定位。Comment Extension是可选的，可以在GIF图像中出现多次，一般会被解码器忽略。

- Extension Introducer：固定的```0x21```；
- Comment Label：块标志符```0xfe```；
- Comment Data：数据域。

#### 2.2.9 Plain Text Extension
&emsp;&emsp;Plain Text Extension包含了希望渲染的文字信息，块中会描绘希望渲染的文字的位置和每个字符的cell信息。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/89a-gif-plain-text-extension.drawio.svg)
- Extension Introducer，1个字节：固定的```0x21```；
- Plain Text Label，1个字节：块标识，```0x01```；
- 块大小，1个字节：固定的值12，即```0xc```；
- 下面8个字节分别为坐标即```[left, top, width, height]```；
- Character Cell Width：每个字符cell的宽度；
- Character Cell Height：每个字符cell的高度；
- Text Foreground Color Index：前景颜色索引；
- Text Background Color Index：背景颜色索引；
- Plain Text Data：是一个sub-blocks，每一个最多255字节；
- Terminator。

#### 2.2.10 其他扩展块即信息
&emsp;&emsp;Opt表示可选，Req表示必须。
```
Block Name                  Required   Label       Ext.   Vers.
Application Extension       Opt. (*)   0xFF (255)  yes    89a
Comment Extension           Opt. (*)   0xFE (254)  yes    89a
Global Color Table          Opt. (1)   none        no     87a
Graphic Control Extension   Opt. (*)   0xF9 (249)  yes    89a
Header                      Req. (1)   none        no     N/A
Image Descriptor            Opt. (*)   0x2C (044)  no     87a (89a)
Local Color Table           Opt. (*)   none        no     87a
Logical Screen Descriptor   Req. (1)   none        no     87a (89a)
Plain Text Extension        Opt. (*)   0x01 (001)  yes    89a
Trailer                     Req. (1)   0x3B (059)  no     87a

Unlabeled Blocks
Header                      Req. (1)   none        no     N/A
Logical Screen Descriptor   Req. (1)   none        no     87a (89a)
Global Color Table          Opt. (1)   none        no     87a
Local Color Table           Opt. (*)   none        no     87a

Graphic-Rendering Blocks
Plain Text Extension        Opt. (*)   0x01 (001)  yes    89a
Image Descriptor            Opt. (*)   0x2C (044)  no     87a (89a)

Control Blocks
Graphic Control Extension   Opt. (*)   0xF9 (249)  yes    89a

Special Purpose Blocks
Trailer                     Req. (1)   0x3B (059)  no     87a
Comment Extension           Opt. (*)   0xFE (254)  yes    89a
Application Extension       Opt. (*)   0xFF (255)  yes    89a
```

#### 2.2.11 Trailer
&emsp;&emsp;同87a为固定的```0x3B```。

#### 2.2.12 GIF支持的其他选项
>&emsp;&emsp;非官方标准，但是大部分GIF都支持。
&emsp;&emsp;**循环次数**：GIF图支持设置循环次数，如果该标志设置为0则表示无限循环；
&emsp;&emsp;**颜色抖动**：GIF可以通过颜色抖动算法来减少GIF中的颜色表来所见文件尺寸，但是颜色都读算法仅仅对于颜色变换比较连续的图像比较友好，对于颜色变换比较平滑的效果较差。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/dithering.gif)
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/dither2.gif)

&emsp;&emsp;**隔行采样**：顾名思义。

## 3 尝试C语言手写一个GIF图像
&emsp;&emsp;上面简略了解了下GIF图像的存储格式，下面尝试用C写一张包含RGB三个纯色280x280GIF图像，每一帧图像都是纯色图像，每一帧子图像都不等于原图大小，且图像延时间隔分别为0.5ms。
>&emsp;&emsp;GIF图像中的数据基本都可以用C的fwrite写入，但是对于lzw压缩需要借助额外的lzw库[lzw-compress-lib](https://github.com/jefftime/lzw)，该库的使用很简单加入到项目就可以。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/result1.gif)

&emsp;&emsp;首先是各种结构的定义，结构很简单，按照上面的结构创建就行。部分内容为了方便写入将uint16拆分为两个uint8。
```c
typedef unsigned __int8  uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
typedef uint8_t byte;
extern "C" {
	typedef struct gif_block {
		byte size;
		byte *pdata;
	}gif_block;

	typedef struct gif_header {
		char gif_sig[3];
		char gif_ver[3];
	}gif_header;
	//extension block header
	typedef struct gif_ext_header {
		byte introducer;
		byte label;
	}gif_ext_header;
	//screen descriptor 
	typedef struct gif_screen_decriptor {
		byte width_l;
		byte width_h;
		byte height_l;
		byte height_h;
		byte flag_pixel : 3;
		byte flag_s : 1;
		byte flag_cr : 3;
		byte flag_M : 1;
		byte bk_color_index;
		byte aspect_ratio;
	}gif_screen_decriptor;
	//application extension
	typedef struct gif_app_ext {
		gif_ext_header header;
		byte size;
		char identifier[8];
		byte authentication_code[3];
		gif_block app_data;
		byte terminator;
	}gif_app_ext;

	//graphic control extension
	typedef struct gif_gce{
		gif_ext_header header;
		byte size;
		byte transport_used : 1;
		byte input_flag : 1;
		byte disposal_method : 3;
		byte reversed : 3;
		byte delay_time_l;
		byte delay_time_h;
		byte transparent_color_index;
		byte terminator;
	}gif_gce;

	typedef struct gif_image_descriptor {
		byte introductor;
		byte left_l;
		byte left_h;
		byte right_l;
		byte right_h;
		byte width_l;
		byte width_h;
		byte height_l;
		byte height_h;
		byte flag_pixel : 3;
		byte flag_r : 2;
		byte flag_s : 1;
		byte flag_I : 1;
		byte flag_M : 1;
	};

	typedef struct gif_trailer{
		byte trailer;
	}gif_trailer;
}
```

&emsp;&emsp;实现部分也很简单，基本每一个内容都有注释。
```c
//写入颜色表
static void write_color_table(FILE *fp, uint32_t *color_table, int len) {
	for (int i = 0; i < len; i++) {
		uint32_t current_color = color_table[i];
		uint8_t r = (current_color & 0xFF0000) >> 16;
		uint8_t g = (current_color & 0x00FF00) >> 8;
		uint8_t b = (current_color & 0x0000FF);

		fwrite(&r, sizeof(uint8_t), 1, fp);
		fwrite(&g, sizeof(uint8_t), 1, fp);
		fwrite(&b, sizeof(uint8_t), 1, fp);
	}
}
//将数据压缩并写入文件中
static void write_lzw_data_block(uint8_t bit, unsigned char *pdata, unsigned long len, FILE*fp) {
	//首先写入LZW Minimum Code Size
	fwrite(&bit, sizeof(bit), 1, fp);
	//中间的lzw数据是多个data block，每个block最多256字节
	unsigned long writed_size = 0;
	while (writed_size < len) {
		if ((0xff + writed_size) > len) {
			//未写入的数据小于255，不需要额外的块存储
			uint8_t terminator = 0;
			unsigned long left_size = len - writed_size;
			fwrite(&left_size, sizeof(uint8_t), 1, fp);
			fwrite(pdata + writed_size, left_size, 1, fp);
			writed_size += left_size;
		}	else {
			uint8_t size = 0xff;
			fwrite(&size, sizeof(size), 1, fp);
			fwrite(pdata + writed_size, size, 1, fp);
			writed_size += 0xff;
		}
	}

	//最后需要写入Terminator
	uint8_t terminator = 0;
	fwrite(&terminator, sizeof(terminator), 1, fp);
}

static void write_gif_in_hand(const char *filename) {
	FILE *fp = fopen(filename, "wb");
	if (nullptr == fp) {
		fprintf(stderr, "can not open file %s\n", filename);
	}
	uint16_t width = 280, height = 280;
	//header为固定的GIF89a
	gif_header header = { {0x47, 0x49, 0x46}, {0x38, 0x39, 0x61} };
	fwrite(&header, sizeof(header), 1, fp);

	//宽高100x100，flag为11110010(0xF2)，表示颜色深度为8，采用全局颜色表，颜色数量最多为8，背景色和宽高比不使用
	gif_screen_decriptor scrn_desc = { width, width >> 8, height, height >> 8, 0x02, 0x00, 0x07, 0x01};
	fwrite(&scrn_desc, sizeof(scrn_desc), 1, fp);

	//这里采用的颜色表比较简单，就是rgb三种颜色，额外包含黑色
	const int color_number = 8;
	uint32_t color_table[color_number] = {
		0XFF0000, // 赤
		0XFFA500, // 橙
		0XFFFF00, // 黄
		0X00FF00, // 绿
		0X007FFF, // 青
		0X0000FF, // 蓝
		0X8B00FF, // 紫
		0X000000  // 黑 
	};
	write_color_table(fp, color_table, sizeof(color_table) / sizeof(color_table[0]));
	
	//appliction这里用一种比较简单的方式写入，也可以用上面定义的gif_app_ext，但是稍显麻烦
	uint8_t gif_application_extension[] = { 0x21, 0xFF, 0x0B, 0x4E, 0x45, 0x54, 0x53, 0x43, 0x41, 0x50, 0x45, 0x32, 0x2E, 0x30, 0x03, 0x01, 0x00, 0x00, 0x00 };
	fwrite(gif_application_extension, sizeof(uint8_t), sizeof(gif_application_extension) / sizeof(gif_application_extension[0]), fp);

	for (int i = 0; i < color_number - 1; i++) {
    //每一帧的left和right不同，一直在变化。
		uint16_t image_width = width / (color_number - 1), image_height = height / (color_number - 1);
		uint16_t image_left = i * image_width, image_right = i * image_height;
		gif_gce gce = { 0x21, 0xf9, 0x04, 0, 0, 0, 0, 0x32, 0x32 >> 8, 0xff, 0 };
		fwrite(&gce, sizeof(gce), 1, fp);

		gif_image_descriptor img_desc = { 0x2c, image_left, image_left >> 8, image_right, image_right >> 8, image_width, image_width >> 8, image_height, image_height >> 8, 0, 0, 0, 0, 0 };
		fwrite(&img_desc, sizeof(img_desc), 1, fp);

		//颜色数据，对应的颜色采用对应的索引即可，比如第一帧为0，第二帧为1
		uint8_t* pdata = (uint8_t*)malloc(sizeof(uint8_t) * image_height * image_width);
		memset(pdata, i, sizeof(uint8_t) * image_height * image_width);
		unsigned char *compressed_data = nullptr;
		unsigned long compressed_size = 0;
		lzw_compress_gif(3, image_height * image_width, pdata, &compressed_size, &compressed_data);
		write_lzw_data_block(0x03, compressed_data, compressed_size, fp);
		free(pdata);
	}
  //写入文件终结符。
	gif_trailer trailer = { 0x3B };
	fwrite(&trailer, sizeof(trailer), 1, fp);
	if (fp != nullptr) {
		fclose(fp);
	}
}
```

## 参考文档
- [GIF-wiki](https://en.wikipedia.org/wiki/GIF)
- [w3-GIF-87a](https://www.w3.org/Graphics/GIF/spec-gif87.txt)
- [w3-GIF-89a](https://www.w3.org/Graphics/GIF/spec-gif89a.txt)
- [手写GIF](https://www.ihubin.com/blog/audio-video-basic-18-generate-gif-by-hand/)
- [GIF dispose](http://www.theimage.com/animation/pages/disposal2.html)