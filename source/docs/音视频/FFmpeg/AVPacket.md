# AVPacket
&emsp;&emsp;**摘要**：```AVPacket```是FFmpeg中表示编码的结构体，是FFmpeg最重要的结构体之一。本篇文章针对FFmpeg源码理解```AVPacket```的作用，相关的结构定义以及一些操作API的具体实现。
&emsp;&emsp;**关键字**：```AVPacket```、```AVFrameSideData```
&emsp;&emsp;**注意**：阅读本文前你需要了解基本的视频数据格式比如YUV420P，RGBA8等，以及音频数据格式，比如FLTP，S16等。并且了解FFmpeg的基本解码流程，以及```AVPacket```的简单使用。

## 1 ```AVPacket```
### 1.1 ```AVPacket```简介
&emsp;&emsp;```AVPacket```是FFmpeg中存储压缩数据的结构体，一般从媒体文件中解封装出来的数据或者编码器编码得到的压缩数据都存储在```AVPacket```中，且一个```AVPacket```只存储一帧压缩的视频数据或者一段压缩的音频数据。```AVPacket```的数据通常是通过```AVBufferRef```管理的因此申请释放要是用对应的API。另外，```sizeof(AVPacket)```并不是abi稳定的，如果有新增的字段会在结构体定义的尾部添加，因此在使用```sizeof(AVPacket)```时三思。

```c
typedef struct AVPacket {
    AVBufferRef *buf;               //管理data域的AVBufferRef（引用计数），如果为NULL，表明数据并不是通过引用计数管理的
    int64_t pts;                    //以AVStream->time_base为单位的显示时间戳，如果为默认值AV_NOPTS_VALUE表明无法从码流中读取时间戳信息
    int64_t dts;                    //以AVStream->time_base为单位的解码时间戳，如果为默认值AV_NOPTS_VALUE表明无法从码流中读取时间戳信息
    uint8_t *data;                  //压缩的数据
    int   size;                     //数据大小
    int   stream_index;             //当前packet所属流的索引，一般视频流为0，音频流大于0
    /**
     * A combination of AV_PKT_FLAG values
     */
    int   flags;                   //AV_PKT_FLAG_* 表明当前帧是否为关键帧，数据无效或者需要被丢弃
    AVPacketSideData *side_data;    //容器携带的额外信息
    int side_data_elems;            //sidedata的数量
    int64_t duration;               //以AVStream->time_base为单位的帧时长，即pts_{i+1}-pts_{i}，0表示未知
    int64_t pos;                    //当前packet在流中的位置（bytes），-1表示未知
    void *opaque;                   //用户的私有数据，一般传递给解码器或编码器

    /**
     * AVBufferRef for free use by the API user. FFmpeg will never check the
     * contents of the buffer ref. FFmpeg calls av_buffer_unref() on it when
     * the packet is unreferenced. av_packet_copy_props() calls create a new
     * reference with av_buffer_ref() for the target packet's opaque_ref field.
     *
     * This is unrelated to the opaque field, although it serves a similar
     * purpose.
     */
    AVBufferRef *opaque_ref;        //管理用户数据的AVBufferRef，如果希望通过AVBufferRef管理opaque数据可以指定，内部会正确的拷贝和释放
    AVRational time_base;           //时间基
} AVPacket;
```

### 1.2 ```AVPacket```相关API
- ```void av_init_packet(AVPacket *pkt)```：初始化```AVPacket```，将其中的参数设置为默认值，相关的```AVBufferRef```会直接被置为NULL；
- ```AVPacket *av_packet_alloc(void)```：创建一个```AVPacket```，只是申请了```AVPacket```结构体的内存，其中的数据域依然为空；
- ```void av_packet_free(AVPacket **pkt)```：释放```AVPacket```，其实就是释放```data```，以及结构本身；
- ```int av_new_packet(AVPacket *pkt, int size)```：按照需要申请```size```大小的数据域，但是实际返回的大小是```size+AV_INPUT_BUFFER_PADDING_SIZE```，另外注意只是申请内存，因此pkt必须有效不为空；
- ```void av_shrink_packet(AVPacket *pkt, int size)```：将packet中末尾```AV_INPUT_BUFFER_PADDING_SIZE```大小的内存填充0；
- ```int av_grow_packet(AVPacket *pkt, int grow_by)```：扩充内存；
- ```int av_packet_from_data(AVPacket *pkt, uint8_t *data, int size)```：根据给定的```data```创建packet，并且假定数据域的大小为```size + AV_INPUT_BUFFER_PADDING_SIZE```。该数据是通过```AVBufferRef```管理的也就是说```AVBufferRef接管了```data```，其他地方不应该再操作这块内存；
- ```int av_packet_copy_props(AVPacket *dst, const AVPacket *src)```：将```src```属性（除了数据域）拷贝到```dst```；
- ```void av_packet_unref(AVPacket *pkt)```：使用```AVBufferRef```管理的内存引用计数-1，且释放sidedata；
- ```int av_packet_ref(AVPacket *dst, const AVPacket *src)```：将```src```所有值拷贝到```dst```，如果```src```内存是通过```AVBufferRef```管理二者```data```依然相同；
- ```AVPacket *av_packet_clone(const AVPacket *src)```：```av_packet_alloc+av_packet_ref```；
- ```void av_packet_move_ref(AVPacket *dst, AVPacket *src)```：``dst=std::move(src)```，处理完```src```无效；
- ```int av_packet_make_refcounted(AVPacket *pkt)```：如果packet数据不是通过```AVBufferRef```管理的则创建一份并拷贝内存，需要注意如果输入的```pkt->data```是堆上的内存不释放的话就会内存泄漏；
- ```int av_packet_make_writable(AVPacket *pkt)```：```int av_packet_make_writable(AVPacket *pkt)```：根据```pkt->size```申请对应大小的内存使其可写；
- ```void av_packet_rescale_ts(AVPacket *pkt, AVRational src_tb, AVRational dst_tb)```：将packet的时间戳按照源和目标进行转换。


## 2 ```AVPacketSideData```
### 2.1 结构定义
&emsp;&emsp;```AVPacketSideData```携带容器的额外数据，结构定义非常简单：
```c
typedef struct AVPacketSideData {
    uint8_t *data;
    size_t   size;
    enum AVPacketSideDataType type;
} AVPacketSideData;
```
&emsp;&emsp;FFmpeg支持的sidedata如下：
```c
const char *av_packet_side_data_name(enum AVPacketSideDataType type)
{
    switch(type) {
    case AV_PKT_DATA_PALETTE:                    return "Palette";
    case AV_PKT_DATA_NEW_EXTRADATA:              return "New Extradata";
    case AV_PKT_DATA_PARAM_CHANGE:               return "Param Change";
    case AV_PKT_DATA_H263_MB_INFO:               return "H263 MB Info";
    case AV_PKT_DATA_REPLAYGAIN:                 return "Replay Gain";
    case AV_PKT_DATA_DISPLAYMATRIX:              return "Display Matrix";
    case AV_PKT_DATA_STEREO3D:                   return "Stereo 3D";
    case AV_PKT_DATA_AUDIO_SERVICE_TYPE:         return "Audio Service Type";
    case AV_PKT_DATA_QUALITY_STATS:              return "Quality stats";
    case AV_PKT_DATA_FALLBACK_TRACK:             return "Fallback track";
    case AV_PKT_DATA_CPB_PROPERTIES:             return "CPB properties";
    case AV_PKT_DATA_SKIP_SAMPLES:               return "Skip Samples";
    case AV_PKT_DATA_JP_DUALMONO:                return "JP Dual Mono";
    case AV_PKT_DATA_STRINGS_METADATA:           return "Strings Metadata";
    case AV_PKT_DATA_SUBTITLE_POSITION:          return "Subtitle Position";
    case AV_PKT_DATA_MATROSKA_BLOCKADDITIONAL:   return "Matroska BlockAdditional";
    case AV_PKT_DATA_WEBVTT_IDENTIFIER:          return "WebVTT ID";
    case AV_PKT_DATA_WEBVTT_SETTINGS:            return "WebVTT Settings";
    case AV_PKT_DATA_METADATA_UPDATE:            return "Metadata Update";
    case AV_PKT_DATA_MPEGTS_STREAM_ID:           return "MPEGTS Stream ID";
    case AV_PKT_DATA_MASTERING_DISPLAY_METADATA: return "Mastering display metadata";
    case AV_PKT_DATA_CONTENT_LIGHT_LEVEL:        return "Content light level metadata";
    case AV_PKT_DATA_SPHERICAL:                  return "Spherical Mapping";
    case AV_PKT_DATA_A53_CC:                     return "A53 Closed Captions";
    case AV_PKT_DATA_ENCRYPTION_INIT_INFO:       return "Encryption initialization data";
    case AV_PKT_DATA_ENCRYPTION_INFO:            return "Encryption info";
    case AV_PKT_DATA_AFD:                        return "Active Format Description data";
    case AV_PKT_DATA_PRFT:                       return "Producer Reference Time";
    case AV_PKT_DATA_ICC_PROFILE:                return "ICC Profile";
    case AV_PKT_DATA_DOVI_CONF:                  return "DOVI configuration record";
    case AV_PKT_DATA_S12M_TIMECODE:              return "SMPTE ST 12-1:2014 timecode";
    case AV_PKT_DATA_DYNAMIC_HDR10_PLUS:         return "HDR10+ Dynamic Metadata (SMPTE 2094-40)";
    }
    return NULL;
}
```

### 2.2 操作API
- ```void av_packet_free_side_data(AVPacket *pkt)```：释放对应的sidedata数据；
- ```int av_packet_add_side_data(AVPacket *pkt, enum AVPacketSideDataType type, uint8_t *data, size_t size)```：向packet的尾部插入对应类型的sidedata数据，内部会直接接管```data```，如果packet中已经存在同类型的数据会先释放再插入；
- ```uint8_t *av_packet_new_side_data(AVPacket *pkt, enum AVPacketSideDataType type, size_t size)```：申请对应类型的sidedata并使用```av_packet_add_side_data```插入；
- ```uint8_t *av_packet_get_side_data(const AVPacket *pkt, enum AVPacketSideDataType type, size_t *size)```：获取对应类型sidedata的数据域，返回的指针只应该访问，如果要改写需要拷贝；
- ```uint8_t *av_packet_pack_dictionary(AVDictionary *dict, size_t *size)```：将```AVDictioary```中的数据放到一整块内存并返回；
- ```int av_packet_unpack_dictionary(const uint8_t *data, size_t size, AVDictionary **dict)```：从一整块数据中解析出一个```AVDictionary```；
- ```int av_packet_shrink_side_data(AVPacket *pkt, enum AVPacketSideDataType type, size_t size)```：只会将对应类型的sidedata的size域改为输入参数```size```。

>&emsp;&emsp;```AVPacketList```已经废弃了