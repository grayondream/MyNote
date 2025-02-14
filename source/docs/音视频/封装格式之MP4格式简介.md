# 封装格式之MP4格式简介

&emsp;&emsp;**摘要**：MP4格式是比较常用的视频封装格式，本文主要描述mp4格式的具体描述以及相关的分析工具。
&emsp;&emsp;**关键字**：MP4

## 1 MP4格式简介
> &emsp;&emsp;ISO/IEC 14496-12:2004：[Information technology — Coding of audio-visual objects — Part 12:ISO base media file format](https://web.archive.org/web/20180219054429/http://l.web.umkc.edu/lizhu/teaching/2016sp.video-communication/ref/mp4.pdf)
>&emsp;&emsp;ISO/IEC 14496-14:2003: [Information technology — Coding of audio-visual objects — Part 14:MP4 file format](https://github.com/OpenAnsible/rust-mp4/raw/master/docs/ISO_IEC_14496-14_2003-11-15.pdf)

&emsp;&emsp;MP4文件格式即ISO/IEC 14496-14:2003，是冲QuickTime文件格式发展而来，是ISO/IEC 14496-12:2004（ISO Base Media File format）的一个具体实例。MP4的实现有两个版本：第一个版本是ISO/IEC 14496-1:2001（MPEG-4 Part 1 (Systems), First edition），发布于2001年；第二个版本ISO/IEC 14496-14:2003（MPEG-4 Part 14 (MP4 file format), Second edition）发布于2003年对前者进行了改进。

&emsp;&emsp;MP4文件格式是一种标准的数字多媒体容器格式，主要用来存储数字音频和数字视频数据，支持多种音频编码数据，视频编码数据以及其他需要嵌入的额外数据，也可以存储静态图像和字幕。MP4 文件可以包含格式标准定义的元数据，还可以包含可扩展元数据平台 (XMP) 元数据。另外，MP4文件的扩展名一般为```.mp4```，```.m4a, .m4p, .m4b, .m4r, .m4v```也是MP4格式的扩展名，只是不同场景使用不同的扩展名（比如```m4a```通常用于仅仅包含音频流的文件）。

&emsp;&emsp;需要注意的是下面的文件格式严格的说应该是ISO Base Media File Format的描述，MP4只是其中的一个实现而已。比如3GP，mov等格式ISO Base Media File Format的实现，都是以box存储数据的方式，而MP4有两个版本的实现mp41和mp42。

## 2 MP4文件格式基本单元
&emsp;&emsp;MP4文件格式中的所有数据都是通过Box（或者Atom）描述，Box是可以嵌套的。Box由Box Header和Box Body描述，Box Header描述Box的大小类型以及一些描述符（网络字节序，大端），Box Body就是当前Box内含的数据。标准中Box定义的伪代码如下：

```c
aligned(8) class Box (unsigned int(32) boxtype, optional unsigned int(8)[16] extended_type) {
    unsigned int(32) size;
    unsigned int(32) type = boxtype;
    if (size==1) {
        unsigned int(64) largesize;
    } else if (size==0) {
        // box extends to end of file
    }
    if (boxtype==‘uuid’) {
        unsigned int(8)[16] usertype = extended_type;
    }
} 
```
- ```size```：32bit描述Box的大小，```size=sizeof(Box Header) + sizeof(Box Body)```。针对不同情况有不同的扩展方式：
  - ```size==1```：真实的大小为64bit的```largesize```：
  - ```size==0```：表明当前Box是文件的最后一个Box，Box Body就是文件剩下的内容；
- ```type```：描述当前Box的类型（MP4中如果找到无法识别的Box会忽略掉），标准格式是4个字符，比如```moov```，如果是用户定义的类型的话固定为```uuid```，而类型由```usertype```描述。

&emsp;&emsp;另一种Box的扩展————FullBox，在原来Box的基础上添加了Box的版本和flag描述：
```c
aligned(8) class FullBox(unsigned int(32) boxtype, unsigned int(8) v, bit(24) f) extends Box(boxtype) {
    unsigned int(8) version = v;
    bit(24) flags = f;
} 
```

- ```version```：描述当前Box的版本；
- ```flags```：顾名思义，标志符描述当前Box。


>&emsp;&emsp;可以在[mp4ra.org]([](https://mp4ra.org/#/))中查看已经注册的Box类型。另外可以用[MP4Info](https://github.com/shihuade/MP4Info)、[mp4explorer](https://github.com/ChayimFriedman2/mp4explorer/releases/tag/v1.1.0)和[Online MP4 file parser](https://www.onlinemp4parser.com/)来解析MP4的Box。

## 3 一些重要的Box
&emsp;&emsp;ISO媒体文件通过Box索引和存储数据，并且数据和索引是分开存放的，从下面的图中能够看到trak存储了索引信息，而具体的数据存储在mdat中。
&emsp;&emsp;下面描述的Box都是基于Box的基本格式，描述Box Body的详细内容。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/e4c01b99f8c0256cbbff2ba3e42eb668.png)

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/76b080bcf3e04831f4be0014ce50966a.png)

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/e4c01b99f8c0256cbbff2ba3e42eb668.png)


### 3.1 ftyp
```c
//类型：ftype
//容器：当前文件
//强制：必需
//数量：1
aligned(8) class FileTypeBox extends Box(‘ftyp’) {
    unsigned int(32) major_brand;
    unsigned int(32) minor_version;
    unsigned int(32) compatible_brands[]; // to end of the box
} 
```
&emsp;&emsp;```ftyp```（File Type Box）在文件中的位置尽可能的靠前帮助解封装器识别文件的类型以及兼容版本。```ftyp```描述了文件的主要规范，解封装器尽量使用对应的规范解复用文件，而次要版本仅供参考，不得用于确定文件是否符合标准。 它可能允许更精确地识别主要规范，以进行检查、调试或改进解码。

- ```major_brand```：当前文件的主要兼容格式，解封装器最好使用改规范解复用文件；
- ```minor_version```：次要版本；
- ```compatible_brands```：当前文件兼容的规范列表。

&emsp;&emsp;```ftyp```Box的大小为32字节，主规范为```isom```，次版本号为```512```，兼容规范为```isom,ios2,avc1,mp41```。比如：
```c
[ftyp] size=8+24
  major_brand = isom
  minor_version = 200
  compatible_brand = isom iso2 avc1 mp41
```
### 3.2 moov
```c
//类型：moov
//容器：当前文件
//强制：必需
//数量：1
aligned(8) class MovieBox extends Box(‘moov’){
}
```

&emsp;&emsp;```moov```（Movie Box）存储媒体文件的元数据，是一个容器box，即具体信息由Box中的子Box描述。一般情况下在文件中的位置接近文件头或者文件尾，大多数情况下在```ftyp```后面。

### 3.3 mdat
```c
//类型：mdat
//容器：当前文件
//强制：非必需
//数量：任何数量
aligned(8) class MediaDataBox extends Box(‘mdat’) { 
    bit(8) data[];
}
```
&emsp;&emsp;```mdat```（Media Data Box）包含实际经过编码的媒体流数据，其具体的媒体数据类型是通过其他Box索引描述，也就是说媒体数据```mdat```即便没有Box Header也能正常解析。

### 3.4 mvhd
```c
//类型：mvhd
//容器：moov
//强制：必需
//数量：1
aligned(8) class MovieHeaderBox extends FullBox(‘mvhd’, version, 0) { 
    if (version==1) { 
        unsigned int(64) creation_time;
        unsigned int(64) modification_time;
        unsigned int(32) timescale;
        unsigned int(64) duration;
    } else { // version==0 
        unsigned int(32) creation_time;
        unsigned int(32) modification_time;
        unsigned int(32) timescale;
        unsigned int(32) duration;
    }

    template int(32) rate = 0x00010000; // typically 1.0 
    template int(16) volume = 0x0100; // typically, full volume 
    const bit(16) reserved = 0; 
    const unsigned int(32)[2] reserved = 0; 
    template int(32)[9] matrix = { 0x00010000,0,0,0,0x00010000,0,0,0,0x40000000 }; // Unity matrix
    bit(32)[6] pre_defined = 0; 
    unsigned int(32) next_track_ID; 
}
```

&emsp;&emsp;```mvhd```（Movie Header Box）是一个FullBox存储描述媒体文件信息的元信息，和具体的媒体数据不相关的内容。
- ```version```：版本，0或者1
- ```creation_time```：一个64位int表示创建时间（in seconds since midnight, Jan. 1, 1904, in UTC time）；
- ```modification_time```：一个64位int表示修改时间（in seconds since midnight, Jan. 1, 1904, in UTC time）；
- ```timescale```：1s的时间尺度，比如60表示时间间隔为$\frac{1}{60}$s；
- ```duration```：表示当前媒体的时长，根据文件中的track的信息推导出来，等于时间最长的track的duration；
- ```rate```：推荐的播放速率，32位整数，高16位、低16位分别代表整数部分、小数部分，1.0表示正常播放速率；
- ```volume```：推荐的音量，16位整数，高8位、低8位分别代表整数部分、小数部分，1.0表示正常播放音量；
- ```matrix```：视频的转换矩阵，默认是个单位矩阵；
- ```next_track_ID```：表示一个值用于将要添加到此演示文稿中的下一个轨道的轨道 ID。零不是有效的轨道 ID 值。 ```next_track_ID```的值应大于正在使用的最大 ```track-ID```。 如果该值等于或大于```0xffff```（最大 32 位），并且要添加新的媒体轨道，则必须在文件中搜索未使用的轨道标识符。比如：
```c
  [mvhd] size=12+96
    version = 0
    flags = 0 
    creation_time = 2022-11-06T03:40:48.000Z
    modification_time = 2022-11-06T03:40:48.000Z
    timescale = 600
    duration = 63250
    rate = 1
    volume = 1
    next_track_ID = 3
```

### 3.5 trak

```c
//类型：trak
//容器：moov
//强制：必需
//数量：大于等于1
aligned(8) class TrackBox extends Box(‘trak’) {
}
```

&emsp;&emsp;```trak```（Track）是一个容器Box，本身不包含任何内容，其含义由内部的Box定义。媒体文件中可能包含多条Track，每一条Track是独立的，都有自己的时间和空间信息，比如音频和视频可以单独分为两个Track存放，而单个视频流也可以分为多个Track存放。

### 3.6 tkhd
```c
//类型：tkhd
//容器：trak
//强制：必需
//数量：1
aligned(8) class TrackHeaderBox extends FullBox(‘tkhd’, version, flags){ 
    if (version==1) { 
        unsigned int(64) creation_time;
        unsigned int(64) modification_time;
        unsigned int(32) track_ID;
        const unsigned int(32) reserved = 0; 
        unsigned int(64) duration;
    } else { // version==0 
        unsigned int(32) creation_time;
        unsigned int(32) modification_time;
        unsigned int(32) track_ID;
        const unsigned int(32) reserved = 0; 
        unsigned int(32) duration; 
    }

    const unsigned int(32)[2] reserved = 0; 
    template int(16) layer = 0; 
    template int(16) alternate_group = 0; 
    template int(16) volume = {if track_is_audio 0x0100 else 0}; 
    const unsigned int(16) reserved = 0; 
    template int(32)[9] matrix= { 0x00010000,0,0,0,0x00010000,0,0,0,0x40000000 }; // unity matrix
    unsigned int(32) width; 
    unsigned int(32) height;
}
```

&emsp;&emsp;```tkhd```（Track Header)是用来存储当前Track的描述信息，一个Track只有一个```tkhd```。媒体轨道的轨道头标志的默认值为 7（track_enabled、track_in_movie、track_in_preview）。 如果在演示中所有轨道既没有设置 track_in_movie 也没有设置 track_in_preview，则所有轨道都应被视为在所有轨道上都设置了两个标志。Hint轨道应将轨道头标志设置为 0，以便在本地播放和预览时忽略它们。
- ```version```：0或者1表示版本；
- ```flag```：24bit的标志位：
  - ```Track_enabled(0x000001)```：表明当前Track可用，如果disable了应该设置为0；
  - ```Track_in_movie(0x000002)```：表明当前Track用于播放；
  - ```Track_in_preview(0x000004)```：表明当前Track用于预览；
- ```creation_time```：一个64位int表示创建时间（in seconds since midnight, Jan. 1, 1904, in UTC time）；
- ```modification_time```：一个64位int表示修改时间（in seconds since midnight, Jan. 1, 1904, in UTC time）；
- ```track_ID```：当前Track的ID，是唯一表示，一个文件中不应该存在两个Track_ID相同的Track；
- ```reserved```：预留；
- ```duration```：当前轨道的时长，如果有轨道的编辑列表则值为所有编辑列表持续时间之和；否则为样本持续时间之和。如果无法确认则时间设置为0xfffffff。另外时间需要根据```mvhd```中的```time_scale```计算；
- ```layer```：视频轨道的叠加顺序，数字越小越靠近观看者，比如1比2靠上，0比1靠上；
- ```alternate_group```：当前Track的分组ID，0表示当前轨道不和任何一个轨道在同一组，任何时候只能播放一个组中的一个Track。
- ```volume```：推荐的音量，16位整数，高8位、低8位分别代表整数部分、小数部分，1.0表示正常播放音量；
- ```matrix```：视频的转换矩阵，默认是个单位矩阵；
- ```width```：表示当前轨道视频宽度的浮点数（高16位为整数部分，低16位为小数部分），在对轨道进行操作前图像都会缩放到当前的size；
- ```height```：表示当前轨道视频高度的浮点数（高16位为整数部分，低16位为小数部分），在对轨道进行操作前图像都会缩放到当前的size。

```c
//这个示例视频有两个track一个视频流一个是音频流
[tkhd] size=12+80, flags=3
    id = 1
    duration = 63250
    width = 640.000000
    height = 360.000000

[tkhd] size=12+80, flags=1
    id = 2
    duration = 63223
    width = 0.000000
    height = 0.000000
```

### 3.7 edts
```c
//类型：edts
//容器：trak
//强制：非必需
//数量：0或1
aligned(8) class EditBox extends Box(‘edts’) {
} 
```

&emsp;&emsp;```edts```（Edit Box）是一个容器Box。描述了播放媒体流时的映射关系，如果为空就按照track中的时间一一映射播放，设定的话就按照```elst```中的时间播放。

### 3.8 elst
```c
//类型：elst
//容器：edts
//强制：非必需
//数量：0或1
aligned(8) class EditListBox extends FullBox(‘elst’, version, 0) {
    unsigned int(32) entry_count;
    for (i=1; i <= entry_count; i++) {
        if (version==1) {
            unsigned int(64) segment_duration;
            int(64) media_time;
        } else { // version==0
            unsigned int(32) segment_duration;
            int(32) media_time;
        }
        
        int(16) media_rate_integer;
        int(16) media_rate_fraction = 0;
    }
} 
```
&emsp;&emsp;```elst```（Edit List）是一个数组，数组中每一项存储了一定的显示规则，比如开始时间，持续时长，以及速率等。加入我们希望视频开始10s画面不懂，然后播放从第0s开始播放30s，```elst```应该为
```c
Entry-count = 2
Segment-duration = 10 seconds Media-Time = -1 Media-Rate = 1
Segment-duration = 30 seconds Media-Time =  0 Media-Rate = 1
```
- ```version```：版本信息，0或者1；
- ```entry_point```：当前列表的项目数；
- ```segment_duration```：以```mvhd```中的```time_scale```表示的当前编辑项持续的时长；
- ```media_time```：包含此编辑段的媒体内的开始时间（以```mdhd```中的```time_scale```为单位）。 如果此字段设置为 –1，则为空编辑。 轨道中的最后一个编辑永远不会是空编辑；
- ```media_rate```：edit段的速率为0的话，画面停止。画面会在```media_time```点上停止```segment_duration```时间。否则这个值始终为1。

>&emsp;&emsp;需要注意的是虽然上面的字段中有```media_rate_integer```和```media_rate_fraction```，但是标准里只描述了```media_time```。通过对比二进制位我认为标准里的```media_time```就是```media_rate_integer```。
>media_rate specifies the relative rate at which to play the media corresponding to this edit segment. If this value is 0, then the edit is specifying a ‘dwell’: the media at media-time is presented for the segment-duration. Otherwise this field shall contain the value 1.

### 3.9 mdia
```c
//类型：mdia
//容器：trak
//强制：必需
//数量：1
aligned(8) class MediaBox extends Box(‘mdia’) {
} 
```

&emsp;&emsp;Media Box是一个容器Box存储当前轨道的媒体信息。

### 3.10 mdhd
```c
//类型：mdhd
//容器：mdia
//强制：必需
//数量：1
aligned(8) class MediaHeaderBox extends FullBox(‘mdhd’, version, 0) { 
    if (version==1) { 
        unsigned int(64) creation_time;
        unsigned int(64) modification_time;
        unsigned int(32) timescale;
        unsigned int(64) duration;
    } else { // version==0 
        unsigned int(32) creation_time;
        unsigned int(32) modification_time;
        unsigned int(32) timescale;
        unsigned int(32) duration;
    }
    bit(1) pad = 0; 
    unsigned int(5)[3] language; // ISO-639-2/T language code 
    unsigned int(16) pre_defined = 0;
}
```

&emsp;&emsp;```mdhd```（Media Header Box）存储了媒体的时长等元数据。
- ```version```：版本号0或者1；
- ```creation_time```：一个64/32位int表示修改时间（in seconds since midnight, Jan. 1, 1904, in UTC time）；
- ```modification_time```：一个64/32位int表示修改时间（in seconds since midnight, Jan. 1, 1904, in UTC time）；
- ```timescale```：当前媒体流1s所包含的可读，即一个刻度表示$\frac{1}{time_scale}$s；
- ```duration```：以```timescale```表示的时长；
- ```language``：当前track的媒体的语言代码。 三字符代码集参见 ISO 639-2/T。 每个字符都被打包为它的 ASCII 值和 0x60 之间的差异。 由于代码仅限于三个小写字母，因此这些值是严格的正数。

### 3.11 hdlr
```c
//类型：mdia
//容器：mdia或者meta
//强制：必需
//数量：1
aligned(8) class HandlerBox extends FullBox(‘hdlr’, version = 0, 0) { 
    unsigned int(32) pre_defined = 0; 
    unsigned int(32) handler_type;
    const unsigned int(32)[3] reserved = 0; 
    string name;
}
```

&emsp;&emsp;```hdlr```（Handler Reference）声明了轨道中的媒体数据呈现的过程，因此也声明了轨道中媒体的性质。 例如，视频轨道将由视频处理程序处理。此框在 Meta Box 中出现时，声明了“Meta Box”内容的结构或格式。
- ```version```：当前box的版本；
- ```handler_type```：
  - 当```hdlr```在```mdia```时为四个字符表示当前track的类型：，可取值```vide,soun,hint```分别表示视频，音频和hint Track；
  - 在```meta```Box中时表示当前Box的内容格式；
- ```name```：一个以```\0```为结尾的UTF-8字符串，该字符串对track进行描述。

### 3.12 minf

```c
//类型：minf
//容器：mdia
//强制：必需
//数量：1
aligned(8) class MediaInformationBox extends Box(‘minf’) {
}
```

&emsp;&emsp;```minf```（Media Information）是一个容器Box，包含Track数据的索引等信息。

### 3.13 vmhd,smhd,hmhd,nmhd
```c
//类型：vmhd,smhd,hmhd,nmhd
//容器：minf
//强制：必需
//数量：1
```
&emsp;&emsp;```vmhd,smhd,hmhd,nmhd```这四个Box都是存储当前媒体数据的描述信息。只是针对不同类型的数据采用不同的box，比如视频就是```vmhd```,音频就是```smhd```。

**vmhd**
```c
aligned(8) class VideoMediaHeaderBox extends FullBox(‘vmhd’, version = 0, 1) { 
    template unsigned int(16) graphicsmode = 0; // copy, see below 
    template unsigned int(16)[3] opcolor = {0, 0, 0};
}
```

**vmhd**
```c
aligned(8) class SoundMediaHeaderBox extends FullBox(‘smhd’, version = 0, 0) { 
    template int(16) balance = 0; 
    const unsigned int(16) reserved = 0;
}
```

**vmhd**
```c
aligned(8) class HintMediaHeaderBox extends FullBox(‘hmhd’, version = 0, 0) { 
    unsigned int(16) maxPDUsize;
    unsigned int(16) avgPDUsize;
    unsigned int(32) maxbitrate;
    unsigned int(32) avgbitrate;
    unsigned int(32) reserved = 0; 
}
```

**vmhd**
```c
aligned(8) class NullMediaHeaderBox extends FullBox(’nmhd’, version = 0, flags) {
}
```

### 3.14 stbl
```c
//类型：stbl
//容器：minf
//强制：必需
//数量：1
aligned(8) class SampleTableBox extends Box(‘stbl’) { }
```

&emsp;&emsp;媒体文件中的数据是通过索引访问的，具体数据存储在```mdat```中，而这个访问的索引就是```stbl```（Sample Table Box）。```stbl```是一个容器Box，具体内容由内部的Box描述。

### 3.15 stsd
&emsp;&emsp;```stsd```给出sample的描述信息，这里面包含了在解码阶段需要用到的任意初始化信息，比如编码等。对于不同Track类型的数据存储的描述信息也不同。

```c
//类型：stsd
//容器：stbl
//强制：必需
//数量：1
aligned(8) abstract class SampleEntry (unsigned int(32) format) extends Box(format){
    const unsigned int(8)[6] reserved = 0; 
    unsigned int(16) data_reference_index;
}

class HintSampleEntry() extends SampleEntry (protocol) { 
    unsigned int(8) data [];
} // Visual Sequences

class VisualSampleEntry(codingname) extends SampleEntry (codingname){ 
    unsigned int(16) pre_defined = 0; 
    const unsigned int(16) reserved = 0; 
    unsigned int(32)[3] pre_defined = 0; 
    unsigned int(16) width; 
    unsigned int(16) height;
    template unsigned int(32) horizresolution = 0x00480000; // 72 dpi 
    template unsigned int(32) vertresolution = 0x00480000; // 72 dpi 
    const unsigned int(32) reserved = 0; 
    template unsigned int(16) frame_count = 1; 
    string[32] compressorname;
    template unsigned int(16) depth = 0x0018; 
    int(16) pre_defined = -1;
} // Audio Sequences

class AudioSampleEntry(codingname) extends SampleEntry (codingname){ 
    const unsigned int(32)[2] reserved = 0; 
    template unsigned int(16) channelcount = 2; 
    template unsigned int(16) samplesize = 16; 
    unsigned int(16) pre_defined = 0; 
    const unsigned int(16) reserved = 0 ;
    template unsigned int(32) samplerate = {timescale of media}<<16; 
}

aligned(8) class SampleDescriptionBox (unsigned int(32) handler_type) extends FullBox('stsd', 0, 0){ 
    int i ;
    unsigned int(32) entry_count; 
    for (i = 1 ; i u entry_count ; i++){ 
        switch (handler_type){
        case ‘soun’: // for audio tracks 
            AudioSampleEntry(); break;
        case ‘vide’: // for video tracks 
            VisualSampleEntry(); break;
        case ‘hint’: // Hint track 
            HintSampleEntry(); break;
        } 
    } 
}
```

&emsp;&emsp;```stsd```存储了对应Track相对应的编码信息，每种类型的Box都有对应的```SampleEntry```（字段含义就不说了比较明显从单词含以上也能判断），而具体的类型根据```hdlr```中的```handle type```判断。而具体的Box是继承自对应编码Box的，比如```avc1```是AVC的编码描述的Box，如果视频信息使用AVC编码的```SampleEntry```就应该继承自```avc1```Box。另外从结构定义也能看出一个Track可以由多个描述，而每个描述之间通过```formatname```和```protocol```唯一标识，比如avc1,mp4a等。

### 3.16 stts
&emsp;&emsp;

```c
//类型：stts
//容器：stbl
//强制：必需
//数量：1
aligned(8) class TimeToSampleBox extends FullBox(’stts’, version = 0, 0) { 
    unsigned int(32) entry_count; 
    int i;
    for (i=0; i < entry_count; i++) { 
        unsigned int(32) sample_count; 
        unsigned int(32) sample_delta;
    }
}
```

&emsp;&emsp;```stts```包含了DTS到sample number的映射表，主要用来推导每个帧的时长。从上面的定义能够看出```stts```的内容就是一个列表，列表的每一项分别：
- ```sample_count```：当前sample的个数，每个sample的时长为```sample_delta```；
- ```sample_delta```：当前sample以```mdhd```中的```timescale```为刻度描述的时长；

&emsp;&emsp;对于恒定帧率的视频一般只有一个项，比如```"sample_count":2530,"sample_delta":512```则整个视频流的时长为```2530x512```，timescale=12288，换算下来为105s，帧率为$\frac{12288}{512}$为24帧。而对于可变帧率的视频一般都有多项。

### 3.17 stss
```c
//类型：stss
//容器：stbl
//强制：非必需
//数量：0 or 1
aligned(8) class SyncSampleBox extends FullBox(‘stss’, version = 0, 0) { 
    unsigned int(32) entry_count; 
    int i;
    for (i=0; i < entry_count; i++) { 
        unsigned int(32) sample_number;
    } 
}
```
&emsp;&emsp;```stss```（Sync Sample Box）存储文件中关键帧所在的sample序号。如果没有stss的话，所有的sample中都是关键帧。

### 3.18 ctts

```c
//类型：ctts
//容器：stbl
//强制：非必需
//数量：0 or 1
aligned(8) class CompositionOffsetBox extends FullBox(‘ctts’, version = 0, 0) { 
    unsigned int(32) entry_count; 
    int i;
    for (i=0; i < entry_count; i++) { 
        unsigned int(32) sample_count; 
        unsigned int(32) sample_offset;
    } 
}
```

&emsp;&emsp;```ctts```（Composition Time To Sample Box）存储从解码（dts）到渲染（pts）之间的差值。对于只有I帧、P帧的视频来说，解码顺序、渲染顺序是一致的，此时，```ctts```没必要存在。对于存在B帧的视频来说，```ctts```就需要存在了。当PTS、DTS不相等时，就需要```ctts```了，公式为$CT(n) = DT(n) + CTTS(n) $（DTS可以根据```stts```获得）。
### 3.19 stsc

```c
//类型：stsc
//容器：stbl
//强制：必需
//数量：1
aligned(8) class SampleToChunkBox extends FullBox(‘stsc’, version = 0, 0) { 
    unsigned int(32) entry_count;
    for (i=1; i u entry_count; i++) { 
        unsigned int(32) first_chunk; 
        unsigned int(32) samples_per_chunk; 
        unsigned int(32) sample_description_index;
    } 
}
```

&emsp;&emsp;sample 以 chunk 为单位分成多个组。chunk的size可以是不同的，chunk里面的sample的size也可以是不同的。可以看到内容就是一个数组，每一项都有表示chunk的映射：
- ```first_chunk```：第一个chunk的索引（以1开始）
- ```samples_per_chunk```：当前chunk的sample数；
- ```sample_description_index```：```stsd```中描述信息对应的索引。

&emsp;&emsp;需要注意的是该表项表示```[entry[i],entry[i+1])```之间的chunk拥有相同的```samples_per_chunk```和```sample_description_index```。

### 3.20 stsz
```c
//类型：stsz
//容器：stbl
//强制：必需
//数量：1
aligned(8) class SampleSizeBox extends FullBox(‘stsz’, version = 0, 0) { 
    unsigned int(32) sample_size; 
    unsigned int(32) sample_count;
    if (sample_size==0) { 
        for (i=1; i <= sample_count; i++) { 
            unsigned int(32) entry_size; 
        }
    } 
}
```

&emsp;&emsp;```stsz```(Sample Size Box)存储了每个Sample的大小，视频的话就是每一帧的大小（还有另外一种结构stsz2）：
- ```sample_size```：默认的sample大小（单位是byte），通常为0。如果sample_size不为0，那么，所有的sample都是同样的大小。如果sample_size为0，那么，sample的大小可能不一样；
- ```sample_count```：当前track里面的sample数目。如果 sample_size==0，那么，sample_count 等于下面entry的条目；
- ```entry_size```：单个sample的大小（如果sample_size==0的话）。


### 3.21 stco
```c
//类型：stco
//容器：stbl
//强制：必需
//数量：1
aligned(8) class ChunkOffsetBox extends FullBox(‘stco’, version = 0, 0) { 
    unsigned int(32) entry_count;
    for (i=1; i <= entry_count; i++) { 
        unsigned int(32) chunk_offset;
    } 
}
```

&emsp;&emsp;chunk在文件中的偏移量。针对小文件、大文件，有两种不同的box类型，分别是stco、co64，它们的结构是一样的，只是字段长度不同。```chunk_offset```指的是在文件本身中的 offset，而不是某个box内部的偏移。在构建mp4文件的时候，需要特别注意 moov 所处的位置，它对于```chunk_offset```的值是有影响的。有一些MP4文件的```moov```在文件末尾，为了优化首帧速度，需要将 moov 移到文件前面，此时，需要对```chunk_offset```进行改写。