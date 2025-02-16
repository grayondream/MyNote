# 封装格式之FLV格式
&emsp;&emsp;**摘要**：本文详细描述了FLV格式的基本组成，对FLV格式每个部分的组成进行了详细的分析和解释，并详细描述了每个字段的含义。
&emsp;&emsp;**关键字**：FLV

## 1 简介
&emsp;&emsp;FLV流媒体格式是sorenson公司开发的一种视频格式，全称为Flash Video。 它的出现有效地解决了视频文件导入Flash后，使导出的SWF文件体积庞大，不能在网络上很好的使用等缺点。由于其视频文件体积轻、封装播放简单等优点，使得其非常合适在网络上传输，目前主流的视频网站无一例外支持FLV流媒体格式进行视频播放。

>&emsp;&emsp;2020年12月31日，Chrome作为最后一个宣布将不再支持使用Flash的应用程序浏览器，flv视频均无法透过Google Chrome收看，除开BiliBili、优酷等视频网站以外的视频网站均停止使用flv作为视频格式。

## 2 FLV文件格式
&emsp;&emsp;FLV整个文件由两部分组成Header和Body，Header描述了FLV文件的基本信息，Body存储了流数据。Body中由一个个Tag组成。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/7344c43def376b3393aa65376389db23.png)

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/386bdbe68be2c1d9dba44998c68b3f92.png)


### 2.1 FLV Header
&emsp;&emsp;FLV Header长度为9个字节，前三个字节是固定的```FLV```三个字符，表示当前文件的标签，第四个字节是当前文件的版本比如0或者1，第五个字节被分为了4个部分，前5个bit位作为预留，不使用的话写0；接下来的3个bit位每一位表示是否有对应的流，次序分别为音频，预留，视频，为1则表示当前文件有对应的流；最后4个字节表示当前Header

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/c86044d9bf58d6251349c027536404e3.png)

### 2.2 FLV Body
&emsp;&emsp;FLV Body紧跟在FLV Header之后，即FLV Header中的```dataoffset```也是FLV Body的起始位置。FLV Body由一系列的Back-Pointer和Tag组成，交错存储，大概的结构如下：
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/078c289eb373aecd0acf08ebea91611a.png)

&emsp;&emsp;Back-Pointer就是一个4字节的区域，存储前一个Tag的长度，第一个Tag没有前任，一定是0。另外，FLV的Tag包含音频、视频或脚本元数据、可选的加密元数据和 payload。最基础的为FLV Tag，还有其他的Audio Tag,Vidoe Tag, Data Tag等。

#### 2.2.1 FLV Tag
&emsp;&emsp;FLV Tag也有Header+Data组成，基本结构如下：
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/563396d8e15e14e58a1dfc27c51062c5.png)

- Reserved(2 bit)：用于FMS的保留字段, 值为0；
- Filter(1bit)：指示packet是否需要预处理：
  - 0 = 不需要预处理；
  - 1 = packet 在渲染前需要预处理(例如解密)；
  - 未加密文件中此值为0，加密文件中此值为1；
- TagType(5bit)：表示当前Tag的类型：
  - 8：音频；
  - 9：视频：
  - 18：脚本数据；
- DataSize（3byte）：Tag中除通用头外的长度，即Header+Data字段的长度 (等于Tag总长度 – 11，即StreamID以下的数据长度，不包含StreamID)；
- Timestamp（3byte）：当前Tag的解码时间戳 (DTS)，单位是毫秒。FLV文件中第一个Tag的DTS总为0；
- TimestampExtended（1byte）：和Timestamp字段一起构成一个32 位值, 此字段为高 8 位，单位毫秒；
- StreamID（3byte）：总是为0；
- 上面的数据是一定有的，下面的数据是根据当前Tag类型或者其他一些属性来决定的：
  - Tag Header：
    - TagType为8，则为AudioTagHeader；
    - TagType为9，则为VideoTagHeader；
  - Filter为1，有EncryptionHeader；
  - Filter为1，则有FilterParams；
  - Data：
    - TagType为8，则为音频数据；
    - TagType为9，则为视频数据；
    - TagType为18，则为脚本数据。

#### 2.2.2 Audio Tag
&emsp;&emsp;Audio Tag包括AudioTagHeader和AudioTagBody两部分组成。
##### 2.2.2.1 Audio Tag Header
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/bccb9c0252826a6d88d04edf6f17fab8.png)

&emsp;&emsp;Audio Tag Header主要描述了音频的基本参数，比如采样率等：
- SoundFormat：音频格式；
  - 0: Linear PCM, platform endian；
  - 1: ADPCM；
  - 2: MP3；
  - 3: Linear PCM, little endian；
  - 4: Nellymoser 16-kHz mono；
  - 5: Nellymoser 8-kHz mono；
  - 6: Nellymoser；
  - 7: G.711 A-law logarithmic PCM；
  - 8: G.711 mu-law logarithmic PCM 9 = reserved；
  - 10: AAC；
  - 11: Speex；
  - 14: MP3 8-Khz；
  - 15: Device-specific sound；
- SoundRate：采样率，FLV支持的采样率比较有限,AAC总为3:
  - 0: 5.5 kHz；
  - 1: 11 kHz；
  - 2: 22 kHz；
  - 3: 44 kHz；
- SoundSize：采样位深，此参数仅适用未压缩格式，压缩格式总在内部被解码为16位：
  - 0: 8位；
  - 1: 16位；
- SoundType：声道数：
  - 0: 单声道；
  - 1: 立体声；
- AACPacketType：AAC帧类型。仅当声音格式为 10 时，存在此字段：
  - 0: AAC sequence header；
  - 1: AAC raw。

##### 2.2.2.2 Audio Tag Body
&emsp;&emsp;音频数据段即AUDIODATA，根据是否加密可以存储加密数据即EncryptedBody或者AudioTagBody。AudioTagBody存储的数据根据当前格式不同而不同，如果是AAC即SoundFormat=10，则存储AACAUDIODATA，否则就是具体格式的数据。
&emsp;&emsp;AACAUDIODATA的存储结构根据是否设置AACPacketType而不同，0则存储的AudioSpecificConfig，否则直接存储AAC的数据。

#### 2.2.3 Video Tag
&emsp;&emsp;Video Tag 包含 VideoTagHeader 和 VideoTagBody 两部分。
##### 2.2.3.1 VideoTagHeader
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/d429b28a803bc5492bcc47b611df3fe4.png)

&emsp;&emsp;
- FrameType：帧类型；
  - 1: keyframe (for AVC, a seekable frame)；
  - 2: inter frame (for AVC, a non-seekable frame)；
  - 3: disposable inter frame (H.263 only)；
  - 4: generated keyframe (reserved for server use only)；
  - 5: video info/command frame；
- CodecID：编解码器ID：
  - 1: JPEG (currently unused)；
  - 2: Sorenson H.263；
  - 3: Screen video；
  - 4: On2 VP6；
  - 5: On2 VP6 with alpha channel；
  - 6: Screen video version 2；
  - 7: AVC；
- AVCPacketType：AVC帧类型只有AVC编码才有：
  - 0: AVC sequence header；
  - 1: AVC NALU；
  - 2: AVC end of sequence (lower level NALU sequence ender is not required or supported);
- CompositionTime：PTS与DTS的时间偏移值，单位ms，记作CTS，只有编码器为AVC才有。

##### 2.2.3.2 VideoTagBody
&emsp;&emsp;同AudioTagBody，区分加密和非加密。而非加密的VideoTagBody根据编解码器类型不同存储的数据不同：
- FrameType == 5：UI8；
- CodecID == 2：H263VIDEOPACKET；
- CodecID == 3：SCREENVIDEOPACKET；
- CodecID == 4：VP6FLVVIDEOPACKET；
- CodecID == 5：VP6FLVALPHAVIDEOPACKET；
- CodecID == 6：SCREENV2VIDEOPACKET；
- CodecID == 7：AVCVIDEOPACKET。

#### 2.2.4 Data Tags
&emsp;&emsp;数据 Tag 封装了单一方法，此方法通常在 Flash 播放器中的网络流对象上被调用。数据 Tag 包含方法名和一组参数。这部分就不详细说明了具体参考abode的标准。

## 3 简单查看下FLV的结构
```c
FLV file version 1
  Contains audio tags: Yes
  Contains video tags: Yes
  Data offset: 9

Prev tag size: 0
Tag type: 18 - Script data object
  Data size: 1195
  Timestamp: 0
  Timestamp extended: 0
  StreamID: 0

Prev tag size: 1206
Tag type: 9 - Video data
  Data size: 54
  Timestamp: 0
  Timestamp extended: 0
  StreamID: 0
  Video tag:
    Frame type: 1 - keyframe (for AVC, a seekable frame)
    Codec ID: 7 - AVC
    AVC video tag:
      AVC packet type: 0 - AVC sequence header
      AVC composition time: 0
      AVC nalu length: 23330847
```
## 4 参考文献
- [Flash Video Format](http://download.macromedia.com/f4v/video_file_format_spec_v10_1.pdf)