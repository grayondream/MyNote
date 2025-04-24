# AVFrame

&emsp;&emsp;**摘要**：```AVFrame```是FFmpeg中表示裸数据的结构体，是FFmpeg最重要的结构体之一。本篇文章针对FFmpeg源码理解```AVFrame```的作用，相关的结构定义以及一些操作API的具体实现。
&emsp;&emsp;**关键字**：```AVFrame```、```AVFrameSideData```
&emsp;&emsp;**注意**：阅读本文前你需要了解基本的视频数据格式比如YUV420P，RGBA8等，以及音频数据格式，比如FLTP，S16等。并且了解FFmpeg的基本解码流程，以及```AVFrame```的简单使用。

## 1. ```AVFrame```简介
&emsp;&emsp;FFmpeg中解码的裸数据都是通过```AVFrame```存储的，因此理解```AVFrame```的具体实现对于使用FFmpeg有比较大的帮助。```AVFrame```是一个复合的结构体，他可以存储音频数据或者视频数据。但是因为音频和视频数据的参数不兼容比如宽高和采样率等，```AVFrame```中会保留两者参数的定义，以至于结构体略显臃肿（同时包含了音频和视频的参数定义）。
&emsp;&emsp;FFmpeg解码一个视频时，会先通过解封装器对视频解封装得到编码的流数据```AVPacket```，再将该流数据送给解码器进行解码，解码出来的裸数据就会存储在```AVFrame```中返回。而一个```AVFrame```中只有一帧画面或者一段音频数据。

## 2 ```AVFrame```
## 2.1 ```AVFrame```结构定义
&emsp;&emsp;```AVFrame```有一套自己的操作API，必须通过相关的api进行创建(```av_frame_alloc```)和释放（```av_frame_free```）。因为```AVFrame```中的内存时通过```AVBufferPool```进行管理的。这就意味着```AVFrame```的内存是通过引用计数管理的，可以重复使用。另外需要注意的```AVFrame```的abi并不稳定这就意味着```sizeof(AVFrame)```并不固定，后续更新可能会直接在```AVFrame```结构体定义的尾部添加成员导致其值改变。
&emsp;&emsp;```AVFrame```（在```libavutil/avframe.h```中）的结构定义较为复杂，这里就不罗列代码而是直接根据代码定义解释所有成员的含义：
- ```uint8_t *data[AV_NUM_DATA_POINTERS]```：存储了音频和视频的raw数据。raw数据的组织方式是按照片来进行存储的。
  - 视频数据：
    - 如果是packed的数据，就只有一片，比如rgba等数据就会存储在```data[0]```，其他几个指针全部置空；
    - 如果是planner数据，需要根据planner的数量存储，比如YUV420P数据存在三片分别为Y、U和V通道分别存储在```data[0]、data[1]、data[2]```中，而NV12数据有两片，Y单独一片，UV数据一片，分别存储在```data[0]和data[1]```中；
  - 音频数据：音频数据同样也分packed和planner：
    - packed数据，只有一片；
    - planner数据，片数和通道数挂钩，双通道数据```data[0]```和```data[1]```就分别存储两个通道的数据；
  - 如果需要存储的数据超过AV_NUM_DATA_POINTERS(8)个通道，额外的数据就需要存储在```extended_data```中；
- ```int linesize[AV_NUM_DATA_POINTERS]```：```linesize```和```data```是一一对应的，存储当前片数据一行或者当前数据帧的大小。数值一般为了性能都是进行过对齐的：
  - 视频数据：视频数据的```linesize```存储了当前片数据一行所占的字节数即(bytes per row)，一片的实际数据大小就是```width*linesize[index]```，比如对于576x432的RGBA数据，```linesize[0]=align(576x4)=2304```，而对于YUV420P```linesize[0]==width```；
  - 音频数据：存储当前片音频的数据大小，所有片的大小相同都是```linesize[0]```，其他域置空；
- ```uint8_t **extended_data```：音频数据，如果音频数据是planner且通道数大于8则需要通过```extended_data```找到多余的数据，同时包含所有数据；其他情况其指向和```data```相同；
- ```int width,height```：视频数据的宽高，对于音频数据无意义；
- ```int nb_samples```：音频数据的sample数量，对于视频数据无意义；
- ```int format```：当前数据的格式，视频时```AVPixelFormat```，音频是```AVSampleFormat```；
- ```int key_frame```：当前帧是否为关键帧；
- ```AVPictureType pict_type```：当前帧的类型，比如IBP帧等；
- ```AVRational sample_aspect_ratio```：SAR，视频的帧的采样比，0/1表示未知；
>视频中有DAR、PAR和SAR三种比例：
>  - ```SAR```：（Storage Aspect Ratio）：存储在本地的视频帧的宽高比；
>  - ```DAR```：（Display Aspect Ratio）实际显示的宽高比，DAR=SAR x PAR；
>  - ```PAR```：（Pixel Aspect Ratio）像素宽高比，一般而言像素都是正方形的即1：1，但是也不绝对；
- ```int64_t pts```：当前帧以```time_base```为单位的显示时间戳；
- ```int64_t dts```：当前帧以```time_base```为单位的解码时间戳；
- ```int64_t pkt_dts```：当前帧对应的```AVPacket```的pts；
- ```int coded_picture_number```：码流中帧的序列号；
- ```int display_picture_number```：帧的显示序列号；
- ```int quality```：图像质量，取值```[1, FF_LAMBDA_MAX]```；
- ```void *opaque```：用户的私有数据指针，内部会直接透传；
- ```int repeat_pict```：解码时表明当前帧额外延迟的时间，计算方式为```extra_delay=repeat_pict/(2 x fps)```，实际帧间隔时间为额外间隔+帧率间隔；
- ```int interlaced_frame```：当前帧图像是否为隔行采样模式，取值0/1；
- ```int top_field_first```：如果当前帧为隔行采样，表明是否先显示顶部的行；
- ```int palette_has_changed```：对于支持调色板的格式，表示调色板是否发生变化；
- ```int64_t reordered_opaque```：
- ```int sample_rate```：音频的采样率，比如8000、44100等；
- ```AVBufferRef buf[AV_NUM_DATA_POINTERS]```：对当前帧中```data```的内存管理的```AVBufferRef```，如果为空则表示当前帧内存不是通过该方式管理的；
- ```AVBufferRef **extended_buf```：对于planner的音频数据超过```AV_NUM_DATA_POINTERS```个片的数据会用```extended_data```存储，这个字段就是对应管理大于```AV_NUM_DATA_POINTERS```通道数的```extended_data```的；
- ```int nb_extended_buf````：```extended_buf```中字段的项数；
- ```AVFrameSideData **side_data```：额外的数据，比如motion vector解码成功就是存储在此项；
- ```int nb_side_data```：```side_data```的项数；
- ```int flags```：当前帧的标志位表明当前帧的状态；
- ```enum AVColorRange color_range;enum AVColorPrimaries color_primaries;enum AVColorTransferCharacteristic color_trc;enum AVColorSpace colorspace;enum AVChromaLocation chroma_location;```：当前帧颜色空间相关信息；
- ```int64_t best_effort_timestamp```：通过启发式算法计算出来的pts（编码无用，解码时由解码器设置）
- ```int64_t pkt_pos```：最后一个送入解码器的帧在文件中的偏移量；
- ```int64_t pkt_duration```：当前帧的时长；
- ```AVDictionary *metadata```：元数据，编码时由用户设置，解码时由libavcodec设置；
- ```int decode_error_flags```：解码错误的标志符，(```FF_DECODE_ERROR_xxx```)；
- ```int channels```：音频通道数，废弃；
- ```int pkt_size```：当前packet编码数据的大小，只有解码有用；
- ```AVBufferRef *hw_frames_ctx```：对于使用硬件加速的解码帧，指向对应```AVHWFramesContext```；
- ```AVBufferRef *opaque_ref```：用户数据，但是看代码基本没有用到；
- ```size_t crop_top;size_t crop_bottom;size_t crop_left;size_t crop_right;```：当前帧的矩形区域，其他都丢弃；
- ```AVBufferRef *private_ref```：内部使用的数据，外部不应该关心。
- ```AVChannelLayout ch_layout```：当前音频数据的存储格式，比如单通道，双通道等，旧版本这个值时int；
### 2.2 ```AVFrame```API实现
&&emsp;&emsp;```AVFrame```的一定要通过相关的API进行操作，除非是读，修改内存相关的操作如果不使用对应的API可能导致```AVBufferRef```释放内存不正确。

- ```AVFrame *av_frame_alloc(void)```：创建一个```AVFrame```，会对内存清零，并将部分参数设置为默认值，需要注意的是这里只是申请了```AVFrame```，内部的数据还是空的，对应的释放API为```av_frame_free```；
- ```void av_frame_unref(AVFrame *frame)```：```AVFrame```中所有通过```AVBufferRef```管理的内存都引用计数-1，并将对应的```AVBufferRef```释放，最后将所有参数设置为默认值；
- ```void av_frame_free(AVFrame **frame)```：释放整个```AVFrame```；
- ```int av_frame_get_buffer(AVFrame *frame, int align)```：根据当前音频和视频帧的参数填充```data```域；
- ```int av_frame_ref(AVFrame *dst, const AVFrame *src)```：将```src```内的数据和参数拷贝到```dst```，实际的数据是指向相同的```data```只是通过不同的```AVBufferRef```管理；
- ```AVFrame *av_frame_clone(const AVFrame *src)```：拷贝```AVFrame```，和```av_frame_ref```的区别是函数内创建目标```AVFrame```；
- ```void av_frame_move_ref(AVFrame *dst, AVFrame *src)```：move后```src```会被置空；
- ```int av_frame_is_writable(AVFrame *frame)```：检查当前帧中的```extended_buf```和```buf```是否可写；
- ```int av_frame_make_writable(AVFrame *frame)```：主要就是通过```av_frame_get_buffer```申请内存；
- ```int av_frame_copy_props(AVFrame *dst, const AVFrame *src)```：拷贝参数；
- ```AVBufferRef *av_frame_get_plane_buffer(AVFrame *frame, int plane)```：返回期望获得的plane的```AVBufferRef```；
- ```int av_frame_apply_cropping(AVFrame *frame, int flags)```：应用指定的裁剪参数，并不会释放对应的内存而是将数据指针和宽高设置为对应的值。

## 3 ```AVFrameSideData```
> &emsp;&emsp;sidedata就是解码过程中的一些中间数据，比如运动向量等

&emsp;&emsp;```AVFrameSideData```的结构体比较简单，
### 3.1 ```AVFrameSideData```结构定义
```c
typedef struct AVFrameSideData {
    enum AVFrameSideDataType type;
    uint8_t *data;          //实际的数据域，具体的数据类型和存储方式通过type解析
    size_t   size;          //数据大小
    AVDictionary *metadata; //元数据
    AVBufferRef *buf;      
} AVFrameSideData;
```

&emsp;&emsp;FFmpeg支持的数据如下，
```c
const char *av_frame_side_data_name(enum AVFrameSideDataType type)
{
    switch(type) {
    case AV_FRAME_DATA_PANSCAN:         return "AVPanScan";
    case AV_FRAME_DATA_A53_CC:          return "ATSC A53 Part 4 Closed Captions";
    case AV_FRAME_DATA_STEREO3D:        return "Stereo 3D";
    case AV_FRAME_DATA_MATRIXENCODING:  return "AVMatrixEncoding";
    case AV_FRAME_DATA_DOWNMIX_INFO:    return "Metadata relevant to a downmix procedure";
    case AV_FRAME_DATA_REPLAYGAIN:      return "AVReplayGain";
    case AV_FRAME_DATA_DISPLAYMATRIX:   return "3x3 displaymatrix";
    case AV_FRAME_DATA_AFD:             return "Active format description";
    case AV_FRAME_DATA_MOTION_VECTORS:  return "Motion vectors";
    case AV_FRAME_DATA_SKIP_SAMPLES:    return "Skip samples";
    case AV_FRAME_DATA_AUDIO_SERVICE_TYPE:          return "Audio service type";
    case AV_FRAME_DATA_MASTERING_DISPLAY_METADATA:  return "Mastering display metadata";
    case AV_FRAME_DATA_CONTENT_LIGHT_LEVEL:         return "Content light level metadata";
    case AV_FRAME_DATA_GOP_TIMECODE:                return "GOP timecode";
    case AV_FRAME_DATA_S12M_TIMECODE:               return "SMPTE 12-1 timecode";
    case AV_FRAME_DATA_SPHERICAL:                   return "Spherical Mapping";
    case AV_FRAME_DATA_ICC_PROFILE:                 return "ICC profile";
    case AV_FRAME_DATA_DYNAMIC_HDR_PLUS: return "HDR Dynamic Metadata SMPTE2094-40 (HDR10+)";
    case AV_FRAME_DATA_REGIONS_OF_INTEREST: return "Regions Of Interest";
    case AV_FRAME_DATA_VIDEO_ENC_PARAMS:            return "Video encoding parameters";
    case AV_FRAME_DATA_SEI_UNREGISTERED:            return "H.26[45] User Data Unregistered SEI message";
    case AV_FRAME_DATA_FILM_GRAIN_PARAMS:           return "Film grain parameters";
    case AV_FRAME_DATA_DETECTION_BBOXES:            return "Bounding boxes for object detection and classification";
    }
    return NULL;
}
```
&emsp;&emsp;如果要解码对应的数据时，设置解码相关参数然后从```AVFrame```中使用相关API拿出数据即可：
```c
  av_dict_set(&opts, "flags2", "+export_mvs", 0);
  ret = avcodec_open2(dec_ctx, dec, &opts);
  while(){
    //send_packet
    while(){
      //receive avframe
      sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
    }
  }
```
&emsp;&emsp;比如motion vector数据如下：
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/mv.png)

### 3.2 ```AVFrameSideData```相关API
- ```const char *av_frame_side_data_name(enum AVFrameSideDataType type)```：获取对应类型的sidedata的字符串描述；
- ```void av_frame_remove_side_data(AVFrame *frame, enum AVFrameSideDataType type)```：释放当前```AVFrame```中对应的sidedata的数据；
- ```AVFrameSideData *av_frame_get_side_data(const AVFrame *frame, enum AVFrameSideDataType type)```：从```AVFrame```中获取对应类型的sidedata；
- ```AVFrameSideData *av_frame_new_side_data(AVFrame *frame, enum AVFrameSideDataType type, size_t size)```：创建对应类型的sidedata并返回。
