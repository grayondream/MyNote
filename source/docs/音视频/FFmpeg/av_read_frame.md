# av_read_frame

&emsp;&emsp;**摘要**：本文主要描述了FFmpeg中用于读取未解码数据接口```av_read_frame```的具体调用流程，详细描述了该接口被调用时所作的具体工作。
&emsp;&emsp;**关键字**：```ffmpeg```、```av_read_frame```
&emsp;&emsp;**读者须知**：读者需要了解FFmpeg的基本使用流程，以及一些FFmpeg的基本常识，了解FFmpegIO相关的内容，以及大致的解码流程。
&emsp;&emsp;使用FFmpeg解码视频时需要主动打开解码器获得解码器相关的Context，然后直接调用```av_read_frame```读取```AVPacket```码流数据送到FFmpeg中进行解码即可。本文主要通过源码了解FFmpeg搜索和打开解码器的基本实现原理和FFmpeg具体的解码流程。


## 1 av_read_frame
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/av_read_frame.drawio.svg)

&emsp;&emsp;```av_read_frame```用于从已经打开的文件中读取未经过解码的码流```AVPacket```，对于视频帧就是一帧的压缩帧，对于音频帧如果音频是固定大小的话则可以是多帧，否则也是一帧。```av_read_frame```内部读取码流时调用```avpriv_packet_list_get```和```av_read_frame_internal```。
&emsp;&emsp;```avpriv_packet_list_get```比较简单就是从当前媒体的PackList中取出一帧。```av_read_frame```的函数实现比较长，其大致流程为：
1. 调用```ff_read_packet```读取一帧码流；
2. 如果1步骤失败则调用```parse_packet```刷新解析器，否则继续到步骤3；
3. 如果当前context需要更新解码器context，则将internal的解码器context更新到stream的解码器context；
4. 如果成功拿到预期的帧则下一步，否则跳转到步骤1；
5. 后续的工作就是解析元数据，计算需要丢弃的数据大小等。

&emsp;&emsp;```ff_read_packet```会先检查缓冲区是否有帧没有的话就会调用```s->iformat->read_packet```即对应个是的解析码流的函数进行解码。每个FFmpeg支持的格式都有一个FormatContext描述对应格式的信息以及解析对应格式的函数指针，比如下面是mov的格式描述：
```c
static const AVClass mov_class = {
    .class_name = "mov,mp4,m4a,3gp,3g2,mj2",
    .item_name  = av_default_item_name,
    .option     = mov_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const AVInputFormat ff_mov_demuxer = {
    .name           = "mov,mp4,m4a,3gp,3g2,mj2",
    .long_name      = NULL_IF_CONFIG_SMALL("QuickTime / MOV"),
    .priv_class     = &mov_class,
    .priv_data_size = sizeof(MOVContext),
    .extensions     = "mov,mp4,m4a,3gp,3g2,mj2,psp,m4b,ism,ismv,isma,f4v",
    .flags_internal = FF_FMT_INIT_CLEANUP,
    .read_probe     = mov_probe,
    .read_header    = mov_read_header,
    .read_packet    = mov_read_packet,
    .read_close     = mov_read_close,
    .read_seek      = mov_read_seek,
    .flags          = AVFMT_NO_BYTE_SEEK | AVFMT_SEEK_TO_PTS,
};
```

&emsp;&emsp;从上面的过程能够看出```av_read_frame```完全是同步的操作，可能是比较耗时的，因为如果一直拿不到帧就会一直遍历当前媒体文件的buffer，因此一般建议开一个线程读取Packet。
