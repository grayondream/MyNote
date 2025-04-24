# AVFormat

&emsp;&emsp;**摘要**：```AVStream```是FFmpeg中一个流数据的抽象，是FFmpeg比较重要的结构体之一。本篇文章针对FFmpeg源码理解```AVStream```的作用，相关的结构定义以及一些操作API的具体实现。
&emsp;&emsp;**关键字**：```AVStream```
&emsp;&emsp;**注意**：阅读本文前你需要知道基本的视频封装格式中音视频流的存储方式。

## 1 ```AVStream```
### 1.1 ```AVStream```简介
&emsp;&emsp;一般封装格式中会存在多种类型的数据：音频数据，视频数据，字幕数据，用户数据等。一般情况下一个视频文件都会有一个视频流，多个（大于等于0）音频流，多个（大于等于0）字幕流。（当然对于纯音频文件没有视频流也是正常的）比如网上下载的一些电影可能由多个语言（英语流，汉语流），多种字幕（中文字幕、英文字幕）供选择。```AVStream```就是针对封装格式中数据流的抽象，描述了当前流的基本信息。如果视频包含2个流（音频流和视频流），在```AVFormatContext```中的```streams```字段中，就会有两个```AVStream```分表示两个流。

### 1.2 ```AVStream```定义
&emsp;&emsp;在FFmpeg 5.0以前只有一个```AVStream```结构体，而在之后其实现进行了调整。FFmpeg内部有一个```FFStream```是不会对外暴露的，该结构体持有一个```AVStream```，在使用时FFmpeg只对用户暴露```AVStream```这样结构体就显得更精简避免用户接收到一些非必须的参数。因为```AVStream```始终在```FFStream```的首部，因此总是能够正确的找到对应的实例。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/avstream_ffstream.drawio.svg)

&emsp;&emsp;```AVStream```结构体本身比较简单，主要描述当前流的基本信息，比如帧率，宽高等，下面过一下每个成员的含义（部分成员参数都是解封装时有用户解封装器自行探测，封装时需要用户指定，下面不再一一赘述）：
- ```const AVClass *av_class;```：当前结构体的参数描述，用户可以通过```AVOption```相关接口设置参数；
- ```int index;```：当前流在```AVFormatContext```中的索引；
- ```int id;```：媒体文件相关的流id；
- ```AVCodecParameters *codecpar;```：解码器相关的参数，比如当前流的类型，解码器id等；
- ```void *priv_data;```：私有数据，可以传递给封装器；
- ```AVRational time_base;```：当前流的时间单位，```AVPacket```和```AVFrame```的时间戳都是以此为单位；
- ```int64_t start_time;```：当前流的开始时间，并不是所有的流的开始时间都是0；
- ```int64_t duration;```：以```time_base```为单位的流时长；
- ```int64_t nb_frames;```：当前流的帧数；
- ```int disposition;```：当前流的效果（```AV_DISPOSITION_*```），比如特效等；
- ```enum AVDiscard discard;```：表明当前流是否弃置的状态；
- ```AVRational sample_aspect_ratio;```：视频帧的SAR；
- ```AVDictionary *metadata;```：元数据；
- ```AVRational avg_frame_rate;```：当前流的平均帧率，对于CFR视频和帧率相同，VFR视频则不同；
- ```AVPacket attached_pic;```：对于```disposition```的为```AV_DISPOSITION_ATTACHED_PIC```流，包含一张图像，比如音频专辑的封面；
- ```AVPacketSideData *side_data;```：留的额外数据，和```AVPacket```中的sidedata不一定有交集；
- ```int nb_side_data;```：sidedata的数量，每个类型占一项；
- ```int event_flags;```：指定解码或者编码时的动作（```AVSTREAM_EVENT_FLAG_*```）；
- ```AVRational r_frame_rate;```：真实码率，即最低的码率，对于VFR视频由最低帧率和最高帧率；
- ```int pts_wrap_bits;```：pts的位数，解复用时使用。

### 1.3 ```FFStream```定义
&emsp;&emsp;```FFStream```是FFmpeg内部的流实现，其第一个成员就是``AVStream```，因此总是能够通过```AVStream```找到对应的```FFStream```。
- ```AVStream pub;```：向外公开的```AVStream```，内部使用时直接将```AVStream```地址强转为```FFStream```即可；
- ```int order;```：codec是否支持reorder，1表示支持，现在主流codec（264，265）都支持；
- ```struct AVBSFContext *bsfc;```：流上的filter相关的context，只有编码会使用；
- ```int bitstream_checked;```：当前流的数据是否需要check，主要是check编码的数据流；
- ```struct AVCodecContext *avctx;```：当前流对应的解码器的context；
- ```int avctx_inited;```当前解码器的context是否被初始化，1表示yes；
- ```struct { struct AVBSFContext *bsf; int inited; } extract_extradata;```：解码额外数据的结构体描述；
- ```int need_context_update;```：是否需要根据解码器的```codecpar```更新流的```codecpar```，需要的话内部会更新；
- ```int is_intra_only;```：当前解码器的帧是不是只有关键帧，比如MJPEG；
- ```FFFrac *priv_pts;```：```FFFrac```结构体三个（```int64_t val, num, den;```）```int64_t```用来表示精确的时间戳，保存着当前流最后一个frame的pts；
- ```struct FFStreamInfo *info;```：内部使用的一些参数，大概了解就行；
- ```AVIndexEntry *index_entries;```：seek是使用的帧索引表，用于快速索引关键帧；
- ```int nb_index_entries;```：帧索引表的项数；
- ```unsigned int index_entries_allocated_size;```：index内存大小；
- ```int64_t interleaver_chunk_size;```：chunk的大小；
- ```int64_t interleaver_chunk_duration;```：chunk的时长；
- ```int request_probe;```：probe的状态，-1表示finished，0表示没有进行，其他值表示以 request_probe 为接受的最低分数执行探测（FFmpeg内会对流进行打分）。
- ```int skip_to_keyframe;```：是否丢弃除了关键帧的内容；
- ```int skip_samples;```：开始解码时应该丢弃的sample数量；
- ```int64_t start_skip_samples;```：从0开始到当前帧的数据都应该跳过，此值不应该为负数；
- ```int64_t first_discard_sample;```：指定音频sample之前的帧都应该被丢弃；
- ```int64_t last_discard_sample;```：指定音频sample之后的帧都应该被丢弃；
- ```int nb_decoded_frames;```：内部已经解码的帧数量，用于内部跟踪解码的进度；
- ```int64_t mux_ts_offset;```：mux的时间戳偏移；
- ```int64_t lowest_ts_allowed;```：track中允许的最小时间戳，一般由muxer写入；
- ```int64_t pts_wrap_reference;```：内部用于校正时间戳的数据；
- ```int pts_wrap_behavior;```：校正时间戳的行为，比如：·```AV_PTS_WRAP_ADD_OFFSET```和```AV_PTS_WRAP_SUB_OFFSET```；
- ```int update_initial_durations_done```：内部标志位避免```update_initial_durations```被执行两次；
- ```int64_t pts_reorder_error[MAX_REORDER_DELAY+1];```：内部用于生成pts的数据；
- ```uint8_t pts_reorder_error_count[MAX_REORDER_DELAY+1];```：上一项的数量；
- ```int64_t pts_buffer[MAX_REORDER_DELAY+1];```：内部用于生成pts的数据；
- ```int inject_global_side_data;```：bool值表示是否注入全局sidedata；
- ```AVRational display_aspect_ratio;```：DAR；
- ```AVProbeData probe_data;```：探测的数据，FFmpeg可以指定```probe_size```长度；
- ```int64_t last_IP_pts;```：最后的IP（不太清楚什么意思），看起来只有[NUT](https://wiki.multimedia.cx/index.php/NUT)格式用到；
- ```int last_IP_duration;```：最后的IP（同上）的时长；
- ```enum AVStreamParseType need_parsing;```：```av_read_frame```取帧数据时的解析类型；
- ```struct AVCodecParserContext *parser;```：```av_read_frame```取帧时解析器的context；
- ```int codec_info_nb_frames;```：通过```avformat_find_stream_info```找到的帧数量；
- ```int stream_identifier;```：流id，主要是MPEG-TS个使用；
- ```int64_t first_dts;```：第一帧的dts；
- ```int64_t cur_dts;```：当前帧的dts；

## 2 ```AVInputFormat```
&emsp;&emsp;```AVInputFormat```存储解封转的文件信息，是固定的，一般定义在对应解封装文件中，比如mov格式的定义在```libavformat\mov.c```文件中，如果需要自己实现解封装器将就需要定义对应的格式：
```c
const AVInputFormat ff_mov_demuxer = {
    .name           = "mov,mp4,m4a,3gp,3g2,mj2",
    .long_name      = NULL_IF_CONFIG_SMALL("QuickTime / MOV"),
    .priv_class     = &mov_class,
    .priv_data_size = sizeof(MOVContext),
    .extensions     = "mov,mp4,m4a,3gp,3g2,mj2,psp,m4b,ism,ismv,isma,f4v,avif",
    .flags_internal = FF_FMT_INIT_CLEANUP,
    .read_probe     = mov_probe,
    .read_header    = mov_read_header,
    .read_packet    = mov_read_packet,
    .read_close     = mov_read_close,
    .read_seek      = mov_read_seek,
    .flags          = AVFMT_NO_BYTE_SEEK | AVFMT_SEEK_TO_PTS | AVFMT_SHOW_IDS,
};
```

- ```const char *name;```：由```,```隔开的格式名称；
- ```const char *long_name;```：全称；
- ```int flags;```：操作文件个标志符，比如是否允许按照bytes seek等；
- ```const char *extensions;```：扩展名；
- ```const struct AVCodecTag *const *codec_tag```：解码器类型；
- ```const AVClasss *priv_class;```：私有的选项；
- ```const char *mime_type;```：```,```隔开的mime_type；
- ```int raw_codec_id;```：裸数据的demuxer的codec id；
- ```int priv_data_size;```：私有数据的大小，一般为对应格式的Context，比如mov格式为```MOVContext```；
- ```int flag_internal;```：内部的标志符；
- ```int (*read_probe)(const AVProbeData *);```：探测当前文件是那个类型的文件的函数指针；
- ```int (*read_header)(struct AVFormatContext *);```：读取格式header，初始化```AVFormatContext```的函数指针；
- ```int (*read_packet)(struct AVFormatContext *, AVPacket *pkt);```：从文件中读取一个packet的函数指针；
- ```int (*read_close)(struct AVFormatContext *);```：关闭流，但是不涉及对应流的释放；
- ```int (*read_seek)(struct AVFormatContext *, int stream_index, int64_t timestamp, int flags);```：seek到对应的位置；
- ```int64_t (*read_timestamp)(struct AVFormatContext *s, int stream_index, int64_t *pos, int64_t pos_limit);```：获取下一个时间戳；
- ```int (*read_play)(struct AVFormatContext *);```：开始或者继续读，只有网络文件有效；
- ```int (*read_pause)(struct AVFormatContext *);```：暂停读，只有网络文件有效；
- ```int (*read_seek2)(struct AVFormatContext *s, int stream_index, int64_t min_ts, int64_t ts, int64_t max_ts, int flags);```：seek到对应的时间戳；
- ```int (*get_device_list)(struct AVFormatContext *s, struct AVDeviceInfoList *device_list);```：获取设备列表。


## 3 ```AVOutputFormat```
&emsp;&emsp;```AVOutputFormat```是对外的结构，是描述封装文件的信息和相关操作函数指针的结构。FFmpeg内部使用的是```FFOutputFormat```，其及结构和```AVStream```和```FFStream```类似：
```c
typedef struct FFOutputFormat {
    AVOutputFormat p;
};
```

&emsp;&emsp;比如mp4格式的描述如下：
```c
const FFOutputFormat ff_mp4_muxer = {
    .p.name            = "mp4",
    .p.long_name       = NULL_IF_CONFIG_SMALL("MP4 (MPEG-4 Part 14)"),
    .p.mime_type       = "video/mp4",
    .p.extensions      = "mp4",
    .priv_data_size    = sizeof(MOVMuxContext),
    .p.audio_codec     = AV_CODEC_ID_AAC,
    .p.video_codec     = CONFIG_LIBX264_ENCODER ?
                         AV_CODEC_ID_H264 : AV_CODEC_ID_MPEG4,
    .init              = mov_init,
    .write_header      = mov_write_header,
    .write_packet      = mov_write_packet,
    .write_trailer     = mov_write_trailer,
    .deinit            = mov_free,
    .p.flags           = AVFMT_GLOBALHEADER | AVFMT_ALLOW_FLUSH | AVFMT_TS_NEGATIVE,
    .p.codec_tag       = mp4_codec_tags_list,
    .check_bitstream   = mov_check_bitstream,
    .p.priv_class      = &mov_isobmff_muxer_class,
};
```
### 3.1 ```AVOutputFormat```
&emsp;&emsp;```AVOutputFormat```是FFmpeg写封装文件是描述当前文件信息的结构。
- ```const char *name```：当前文件格式的简称；
- ```cosnt char *long_name```：详细名称；
- ```const char *mime_type```：mime_type；
- ```enum AVCodecID audio_codec```：默认的音频编码器的类型；
- ```enum AVCodecID video_codec```：默认的视频编码器的类型；
- ```enum AVCodecID subtitle_codec```：默认的字幕编码器的类型；
- ```int flags```:```AVFMT_```描述的标志符；
- ```const struct AVCodecTag *const *codec_tag```：当前格式支持的编解码器的列表；
- ```cosnt AVClass *priv_class```：对应格式私有的Context；
### 3.2 ```FFOutputFormat```
&emsp;&emsp;```FFOutputFormat```是```AVOutputFormat```在FFmpeg内部使用的形式：
- ```AVOutputFormat p```：描述文件的属性；
- ```int priv_data_size```：私有数据的大小；
- ```int flags_internal```：内部的标志符，```FF_FMT_FLAG```；
- ```int (*write_header)(AVFormatContext *);```：写头的函数指针；
    /**
     * Write a packet. If AVFMT_ALLOW_FLUSH is set in flags,
     * pkt can be NULL in order to flush data buffered in the muxer.
     * When flushing, return 0 if there still is more data to flush,
     * or 1 if everything was flushed and there is no more buffered
     * data.
     */
- ```int (*write_packet)(AVFormatContext *, AVPacket *pkt);```：写入一个packet，如果设置了刷新的标志符且pkt为空则会刷新缓冲区；
- ```int (*write_trailer)(AVFormatContext *);```：写文件尾部；
- ```int (*interleave_packet)(AVFormatContext *s, AVPacket *pkt, int flush, int has_packet);```：交错写文件的函数指针，默认是按照dts交错写的；
- ```int (*query_codec)(enum AVCodecID id, int std_compliance);```：检查对应的当前容器是否支持对应的codec；
- ```void (*get_output_timestamp)(AVFormatContext *s, int stream, int64_t *dts, int64_t *wall);```：
- ```int (*control_message)(AVFormatContext *s, int type, void *data, size_t data_size);```：发送消息的回调；
- ```int (*write_uncoded_frame)(AVFormatContext *, int stream_index, AVFrame **frame, unsigned flags);```：写rawdata；
- ```int (*get_device_list)(AVFormatContext *s, struct AVDeviceInfoList *device_list);```：返回设备列表；
- ```int (*init)(AVFormatContext *);```：初始化AVFormatContext```；
- ```void (*deinit)(AVFormatContext *);```：deinit；
- ```int (*check_bitstream)(AVFormatContext *s, AVStream *st, const AVPacket *pkt);```：检查流的正确性、

## 3 ```AVIOContext```
### 3.1 ```AVIOContext```
&emsp;&emsp;```AVIOContext```是FFmpeg中直接读写流的结构体，也就是说openfile，readfile，writefile等操作都是通过这个结构体进行的。先简单看下结构体成员的描述：
- ```const AVClass *av_class;```：当前类的option列表；
- ```unsigned char *buffer;```：内部会维持一个缓冲区，这个就是buffer的起始地址；
- ```int buffer_size```：内部维持的buffer的大小；
- ```unsigned char *buf_ptr```：当前buffer数据消费的地址；
- ```unsinged char *buf_end```：已读数据的地址，因为buffer内会有空闲区，这个地址比```buffer+buffer_size```小是正常的；
- ```void *opaque```：私有的类指针；
- ```int (*read_packet)(void *opaque, uint8_t *buff, int buf_size)```：读packet的函数指针；
- ```int (*write_packet)(void *opaque, uint8_t *buf, int buf_size)```：写packet的函数指针；
- ```int64_t pos```：读取到文件的位置；
- ```int eof_reached```：是否读完文件；
- ```int write_flags```：是否写文件，1表示写文件；
- ```int max_packet_size```：刷新缓冲区的上限；
- ```int min_packet_size```：刷新缓冲区的下限；
- ```unsigned long checksum```：校验和；
- ```unsigned char *checksum_ptr```：计算校验和内存的起始地址；
- ```unsigned long (*update_checksum)(unsigned long checksum, const uint8_t *buf, unsigned int size);```：用户指定的更新校验和的指针；
- ```int (*read_pause)(void *opaque, int pause);```：暂停读，网络数据会使用到；
- ```int64_t (*read_seek)(void *opaque, int stream_index, int64_t timestamp, int flags);```：seek；
- ```int seekable;```：当前流的seek模式（```AVIO_SEEKABLE_```）；
- ```int direct;```：立即写/立即读；
- ```const char *protocol_whitelist;```：协议白名单；
- ```const char *protocol_blacklist;```：协议黑名单；
- ```int (*write_data_type)(void *opaque, uint8_t *buf, int buf_size, enum AVIODataMarkerType type, int64_t time);```：```write_packet```中调用的callback；
- ```int ignore_boundary_point;```：If set, don't call write_data_type separately for AVIO_DATA_MARKER_BOUNDARY_POINT, but ignore them and treat them as AVIO_DATA_MARKER_UNKNOWN (to avoid needlessl small chunks of data returned from the callback).
- ```unsigned char *buf_ptr_max;```：写buffer中逆向seek的最远位置指针；
- ```int64_t bytes_read;```：读入的数据量，只读；
- ```int64_t bytes_written;```：写入的数据量，只读；

### 3.2 相关API

## 2 ```AVFormatContext```
### 2.1 简介
&emsp;&emsp;```AVFormatContext```是FFmpeg中媒体格式的抽象，使用FFmpeg打开一个API之后就会得到一个```AVFormatContext```类型的结构体，该结构体包含了当前媒体文件的相关信息。在FFmpeg中和读写文件直接相关的操作都是通过该结构完成，比如解封装，封装都是通过该结构体完成的。

### 2.2 详情
&emsp;&emsp;下面过一下```AVFormatContext```中结构体成员：
- ```const AVClass *av_class```：当前封装格式的参数列表，key是固定写在对应文件中，可以通过```AVOption```相关的操作API设置和获取。比如在```libavformat\mov.c```中就定义了mp4文件的选项列表```mov_options```；
- ```const struct AVInputFormat *iformat;```：输入文件的context，解封装使用（解封装时不需要用户输入，```avformat_open_input```会设置）；
- ```const struct AVOutputFormat *oformat;```：写文件的context，封装使用；
- ```void *priv_data;```：一个```AVOption```类型的私有数据，mux时由```avformat_write_header()```设置，demux时由```avformat_open_input()```设置;
- ```AVIOContext *pb;```
- ```int ctx_flags```：流的属性标志，```AVFMTCTX_```；
- ```unsigned int nb_streams;```：当前文件中流的数量，即```AVStream```数组的大小；
- ```AVStream **streams;```：流的数组，解封装时由```avformat_open_input```设置，封装时由用户设置；
- ```char *url;```：文件路径/网路流的地址；
- ```int64_t start_time```：第一帧的起始时间，仅仅解封装有用；
- ```int64_t duration```：当前文件的时长，一般为最长的流的长度；
- ```int64_t bit_rate```：当前文件的总码率，内部会自动计算；
- ```unsigned int packet_size```：
- ```int max_delay```：
- ```int flags```：读写文件时的标志，```AVFMT_FLAG_```；
- ```int64_t probesize```：搜索流的最大字节数，对于有些mp4文件moov写在文件末尾，较小的probesize就会导致识别不出文件类型；
- ```int64_t max_analysz_duration```；打开文件搜索流时读取的最大时长；
- ```const uint6_t *key```；
- ```int keylen```：key的长度；
- ```AVProgram **programs```：
- ```enum AVCodecID video_codec_id```：视频编解码器的id；
- ```enum AVCodecID audio_coded_id```：音频编解码器的id；
- ```enum AVCodecID subtitle_codec_id;```：字幕的编解码id；
- ```unsigned int max_index_size;```：