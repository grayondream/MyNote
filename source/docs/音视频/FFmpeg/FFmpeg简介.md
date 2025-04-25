# FFmpeg简介

&emsp;&emsp;**摘要**：前一段时间熟悉了下FFmpeg主流程源码实现，对FFmpeg的整体框架有了个大概的认识，因此在此做一个笔记，希望以比较容易理解的文字描述FFmpeg本身的结构，加深对FFmpeg的框架进行梳理加深理解，如果文章中有纰漏或者错误欢迎指出。本文描述了FFmpeg编解码框架的工程结构，基本构成以及大体的调用流程。因为FFmpeg的滤镜是相对独立的一个模块，因此在此不会进行描述。
&emsp;&emsp;**关键字**：FFmpeg,Framework
&emsp;&emsp;**阅读须知**：阅读本文前，你首先需要了解最基本的音视频处理相关的知识，对于这些知识你至少需要最基本的了解，比如知道什么是容器，什么是编解码器，以及大概的工作流程即可。
&emsp;&emsp;FFmepg是一个用C语言实现的多媒体封装、解封转、编解码开源框架，支持了多种IO协议操作，媒体封装格式的封装与解封装以及编解码格式编解码器（包括硬解和软解）。任何软件都可以在FFmpeg的License范围内合理地基于FFmpeg进行开发。FFmpeg有两种开源协议：
- GPL，该协议是具有传染性的，如果使用了GPL部分的代码（FFmpeg可以配置是否开关这部分代码）对应的软件也必须开源否则有法律风险；
- LGPL，允许以动态发布的形式使用，即将FFmpeg编译为动态库使用，但是修改到了FFmpeg部分的代码，修改的部分也需要开源，一般商业软件都会采用这种方式来进行商业软件的开发。

>&emsp;&emsp;FFmpeg is the leading multimedia framework, able to decode, encode, transcode, mux, demux, stream, filter and play pretty much anything that humans and machines have created. It supports the most obscure ancient formats up to the cutting edge. No matter if they were designed by some standards committee, the community or a corporation. It is also highly portable: FFmpeg compiles, runs, and passes our testing infrastructure FATE across Linux, Mac OS X, Microsoft Windows, the BSDs, Solaris, etc. under a wide variety of build environments, machine architectures, and configurations.

## 1 FFmpeg工程
&emsp;&emsp;本小节简单描述下FFmpeg的工程结构相关的内容，以期对FFmpeg工程本身的基本构成有一个基本的认识。
### 1.1 FFmpeg工程结构

&emsp;&emsp;FFmpeg本身的目录结构比较清晰，我们从目录名称中基本就能看出该目录下可能包含哪些文件具体用来干什么。
- ```.```：当前目录下存储的是一些编译和项目相关的配置文件，比如Makefile，License等；
- ```compat```:兼容文件；
- ```doc```:文档，以及一些FFmpeg使用的示例，如果学习FFmpeg的话强烈建议阅读示例；
- ```ffbuild```:编译相关的一些文件，比如依赖选项等等；
- ```fftools```:可以编译成可执行文件的一些工具实现，比如```ffplay,ffmpeg,ffprobe```等工具;
- ```libavcodec```:编解码核心，编解码相关的文件都存放在这里，比如```h264dec.c```等；
- ```libavdevice```:设备相关，比如DShow等；
- ```libavfilter```:滤镜特效处理；
- ```libavformat```:IO操作以及封装格式的封装和转封装等处理；
- ```libavutil```:工具库，比如一些基本的字符串操作，图像操作等；
- ```libavpostproc```:一些效果后处理相关的内容，一般通过filter处理；
- ```libswresample```:音频重采样处理；
- ```libswscale```:视频缩放、颜色空间转换以及色调映射等；
- ```presets```:编解码器的配置文件，参考[FFmpeg-Present-files](http://ffmpeg.org/ffmpeg-all.html#Preset-files)
- ```tests```:测试示例；
- ```tools```:一些简单的工具。

## 2 FFmpeg架构
### 2.1 FFmpeg的总体架构
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/ffmpeg-framework.drawio.svg)

&emsp;&emsp;FFmpeg各个模块是互相独立的，都可以单独使用，比如解封装器只用来对媒体进行解封装或者封装拿到编码器的裸流，或者编解码器直接对裸流数据进行编解码，亦或者使用工具集对已经解码完的数据尽兴处理。
&emsp;&emsp;编解码模块支持多种不同编解码器，所有的编解码器所使用的参数和当前编解码器相关的Context都是使用```AVCodecContext```描述。而FFmpeg中每个具体的解码器都有一个静态的```AVCodec```描述当前解码器如何解码，这个是有一套统一的接口来定义的。上层拿到```AVCodecContext```和```AVCodec```就可以初始化解码器进行解码了，只不过使用FFmpeg提供的解码接口更加方便。FFmpeg并没有硬件解码器归类的```AVCodec```下面，而是在其下层另外规定了一套```AVHWAccel```，通过```AVCodec```来描述该硬件解码器。
&emsp;&emsp;封装和解封装支持多种不同的媒体文件类型，FFmpeg中讲一个文件抽象为```AVFormatContext```，而内部分别将输入流和输出流分别抽象为```AVInputFormat,AVOutputFormat```。```AVInputFormat,AVOutputFormat```用来描述当前媒体文件的相关参数以及对媒体文件进行封装和解封装，而具体的操作通过AVIO来进行。AVIO抽象了具体的文件IO操作，类似编解码器每种类型的输入流都有各自的描述，封装器和解封装器同理。
&emsp;&emsp;工具集也是独立的，只是一些工具函数的集合。
&emsp;&emsp;滤镜用来对裸数据进行一些特效上的处理。（本文不会过多讨论滤镜）

### 2.2 代码结构
&emsp;&emsp;FFmpeg虽然是用C语言写的但是其基本的实现思想是按照OOP的思想实现的，每个具体的格式都有自己的Context和描述类然后通过函数指针来描述具体实例的实际实现，也就是上面描述的```Context->Context->Context->....>Implementation```这种形式，为了对当前处理的对象统一抽象就会有一个Context来描述。而每个Context都有一个```AVClass```和```opaue```来描述当前结构的参数和独有的一些数据，通过这种方式保持了接口的统一的同时，又能兼顾差异性。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/ffmpeg-class-hei.drawio.svg)

### 2.3 调用流程
&emsp;&emsp;FFmpeg的核心就是封装/解封装和解码那一套，下面的流程图是一个大概，有一部分调用被省略了。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/ffmpeg-call-tree.drawio.svg)


