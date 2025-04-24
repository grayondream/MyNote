# FFmpeg中mov解封装

- **摘要**：之前在[Mp4格式详解](https://blog.csdn.net/GrayOnDream/article/details/127815260)中详细描述了Mp4文件格式的具体布局方式。为了更加深入理解mp4文件格式，本文记录了ffmpeg中解封装mp4文件的基本实现。
- **关键字**:```mov```、```FFmpeg```、```mp4````

## 1 简介
&emsp;&emsp;```mp4```文件格式是现如今网络上最常见的视频文件格式，其和```mov```等格式相同都是IOS Base File Format的实现版本，其文件格式都是基于box。
![MP4](https://img-blog.csdnimg.cn/4fcbec9b5f2e4abc8177f817bef10f49.png)

## 2 ff_mov_demuxer
&emsp;&emsp;在FFmpeg中mp4文件解封装的实现在```libavformat\mov.c```文件中。在FFmpeg中每个封装格式都一个描述当前封装格式的结构体和其选项的```AVClass```，mp4个是对应的结构体分别为```ff_mov_demuxer,mov_class```。
&emsp;&emsp;```mov_class```描述了mp4文件的基本选项信息，```mov_options```是一个FFmpeg中内部定义的key-value列表，其中定义了FFmpeg中的基本选项。
&emsp;&emsp;```ff_mov_demuxer```描述如何解封装一个mp4文件的，以及一些基本信息。该结构包含文件扩展名，选项列表，解封装的函数指针，标志位等信息。解封装时```AVFormatContext```都是通过操作函数指针读取文件信息，解封装文件。
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

&emsp;&emsp;```ff_mov_class```每一项的具体含义：
- ```name```：由```,```隔开的格式名称；
- ```long_name```：全称；
- ```priv_class```：私有的选项；
- ```priv_data_size```：私有数据的大小，一般为对应格式的Context，比如mov格式为```MOVContext```；
- ```extensions```：扩展名，能够看到```mov```,```mp4````等格式公用同一个解封装器；
- ```flag_internal```：内部的标志符；
- ```const char *mime_type;```：```,```隔开的mime_type；
- ```read_probe```：探测当前文件是那个类型的文件的函数指针，在```avformat_find_stream_info```用来探测当前文件是否为```mp4```文件，以及相关流信息；
- ```read_header```：读取格式header，初始化```AVFormatContext```的函数指针，```avformat_open_input```时调用，用来读取文件的基本信息；
- ```read_packet```：从文件中读取一个packet的函数指针，读取未解码的数据流的函数指针，在```av_read_frame```时调用；
- ```read_close```：关闭流，但是不涉及对应流的释放；
- ```read_seek```：seek到对应的位置，```av_seek_frame```时调用来seek到对应的位置；
- ```flags;```：操作文件个标志符，比如是否允许按照bytes seek等。

&emsp;&emsp;解封装的基本流程与用```AVFormatContext```解封装的基本流程相同:
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/movdemux.svg)

## 3 解封装具体流程
### 3.1 解封装涉及的结构体
&emsp;&emsp;mp4解封装涉及的结构体比较多，这里挑选几个重点说下。```mov.c```定义了基础box```MOVAtom```的基本结构定义，以及其他接个editlist相关的结构，比如```MOVStts,MOVCtts,MOVElst```等
```c
typedef struct MOVAtom {
    uint32_t type;
    int64_t size; /* total size (excluding the size and type fields) */
} MOVAtom;
```
&emsp;&emsp;以及一些他描述mp4流，轨道索引等信息的结构体，比如```MOVStreamContext```,```MOVFragmentIndex```,```MOVContext```。

### 3.1 ```mov_probe```
&emsp;&emsp;```mov_probe```会返回一个分值，该分值表示当前文件为对应文件格式的分值，分值越高该文件为对应格式的概率越高。在probe时，FFmpeg会根据文件的具体格式进行分辨，```mov```格式就是检测是否存在某个box，如果无法检测到文件符合对应格式，就会退而求其次以扩展名作为依据。所以就会出现有时候检测错误的情况，比如一个随机的mp3文件，其扩展名为mp3，FFmpeg会根据mp3文件进行解封装。解码时并不是每一片都能解码成功，有几率部分片段能够解码正常，但是解码出来的数据是异常的。

```c
#define AVPROBE_SCORE_EXTENSION  50 ///< score for file extension
#define AVPROBE_SCORE_MIME       75 ///< score for file mime type
#define AVPROBE_SCORE_MAX       100 ///< maximum score
```

&emsp;&emsp;```mov_probe```探测流文件的伪代码如下，这里输入被简化为指针```p````：
```c
int mov_probe(int *p){
    int score = 0, offset = 0, moov_offset = -1;
    while(1){
        int64_t size = AV_RB32(p + offset);              //从当前流的位置读取当前box的大小，伪代码不考虑largesize的情况
        char tag[4] = AV_RL32(p + offset+ 4)            //从接下来的内存中读取tag
        switch(tag){
            case "moov":moov_offset = offset + 4;
            case "mdat":
            case "pnot":
            case "udta":
            case "ftyp":
                if(tag == "ftyp" && tag in ["jp2 " "jpx " "jxl "]){
                    score = std::max(score, 5);
                }else{
                    score = AVPROBE_SCORE_MAX;
                }
                break;
            case "ediw":
            case "wide":
            case "junk":
            case "pict":
                score = std::max(score , AVPROBE_SCORE_MAX - 5);break;
            case "skip":
            case "uuid":
            case "prfl":
                score = std::max(score, AVPROBE_SCORE_EXTENSION);break;
        }
        offset += size
    }

    if(score > AVPROBE_SCORE_MAX - 50 && moov_offset != -1){
        /* moov atom in the header - we should make sure that this is not a
         * MOV-packed MPEG-PS */
        offset = moov_offset;

        while (offset < (len(p) - 16)) { /* Sufficient space */
               /* We found an actual hdlr atom */
            if (AV_RL32(p->buf + offset     ) == MKTAG('h','d','l','r') &&
                AV_RL32(p->buf + offset +  8) == MKTAG('m','h','l','r') &&
                AV_RL32(p->buf + offset + 12) == MKTAG('M','P','E','G')) {
                av_log(NULL, AV_LOG_WARNING, "Found media data tag MPEG indicating this is a MOV-packed MPEG-PS.\n");
                /* We found a media handler reference atom describing an
                 * MPEG-PS-in-MOV, return a
                 * low score to force expanding the probe window until
                 * mpegps_probe finds what it needs */
                return 5;
            } else {
                /* Keep looking */
                offset += 2;
            }
        }
    }
    return score;
}
```
&emsp;&emsp;

### 3.2 ```mov_read_header```
&emsp;&emsp;```mov_read_header```是在```avforamt_open_input```时调用，解析mp4文件的基本信息。经过此操作基本上从```AVForamtContext```中和封装格式相关的信息比如```iformat```，流信息等基本上都已经检测到。
&emsp;&emsp;```mov_read_header```的实现过程。首先，校验一些参数，不符合要求就会返回错误。然后调用```mov_read_default```解析流文件中的atom box，从box中读取相关的信息写入到```MOVContext```中。

```c
    /* check MOV header *///不断嵌套读，直到读到moov box未知
    do {
        if (mov->moov_retry)
            avio_seek(pb, 0, SEEK_SET);
        if ((err = mov_read_default(mov, pb, atom)) < 0) {
            av_log(s, AV_LOG_ERROR, "error reading header\n");
            return err;
        }
    } while ((pb->seekable & AVIO_SEEKABLE_NORMAL) && !mov->found_moov && !mov->moov_retry++);
```

&emsp;&emsp;```mov_read_default```中就是不断嵌套读当前atom box内部的所有box然后解析，根据type判断是否为符合要求的box，符合的话就会调用对应的解析函数去解析。结合上面的```do...while```可以理解这里采用的是深度优先的解析方式。具体的解析函数是下面的一个静态函数表格，函数内会通过for循环去寻找是否为符合要求的box然后解析。说实话这样效率感人，索引表感觉更合理。

```c
static const MOVParseTableEntry mov_default_parse_table[] = {
{ MKTAG('A','C','L','R'), mov_read_aclr },
{ MKTAG('A','P','R','G'), mov_read_avid },
{ MKTAG('A','A','L','P'), mov_read_avid },
{ MKTAG('A','R','E','S'), mov_read_ares },
{ MKTAG('a','v','s','s'), mov_read_avss },
...
{ MKTAG('m','d','c','v'), mov_read_mdcv },
{ MKTAG('c','l','l','i'), mov_read_clli },
{ MKTAG('d','v','c','C'), mov_read_dvcc_dvvc },
{ MKTAG('d','v','v','C'), mov_read_dvcc_dvvc },
{ 0, NULL }
};}
```

&emsp;&emsp;经过上面的步骤，流的基本信息已经存储在```MOVContext```和```MOVStreamContext```中，之后就是将解析出来的信息进行处理或者写到```AVFormatContext```中。比如从```side_data```中读取转换矩阵，然后解析当前视频的旋转角，读取chatper，timebase等等。

### 3.3 ```mov_read_packet```
&emsp;&emsp;```mov_read_packet```会在```avformat_find_stream_info```和```av_read_frame```内被调用。前者只会调用几次用来确认流数据的详细信息，而后是是从流中读取未解码的数据。

&emsp;&emsp;首先会调用```mov_find_next_sample```根据当前读取的sample，以及其他时间戳相关的信息解析出下一帧要读取的时间戳。并进行一些size相关的检查，校正要读取的sample的大小以及改变全局的索引（FFMpeg内部的iformat有保存全部的pos索引来表示当前读取到的位置）。
```c
    sample = mov_find_next_sample(s, &st);
    if (!sample || (mov->next_root_atom && sample->pos > mov->next_root_atom)) {
        if (!mov->next_root_atom)
            return AVERROR_EOF;
        if ((ret = mov_switch_root(s, mov->next_root_atom, -1)) < 0)
            return ret;
        goto retry;
    }
```

&emsp;&emsp;然后就是根据标志位来判断当前packet是否要丢弃，调用```av_get_packet```读取数据，在进行一些size上的校正后，调用```avio_read```直接读文件。而具体的读取当然不是一次性读完，因此mov中的数据是按照box存储的，因此会一直读取到满足预期的大小或者报错为止。

```c
if (st->codecpar->codec_id == AV_CODEC_ID_EIA_608 && sample->size > 8)
    ret = get_eia608_packet(sc->pb, pkt, sample->size);
else
    ret = av_get_packet(sc->pb, pkt, sample->size);
```

&emsp;&emsp;最后就是填充```packet```的sidedata，以及更新```ctts,stsc```等相关的索引，以及一些善后的工作。

### 3.4 ```mov_read_seek```
&emsp;&emsp;```seek```的实现比较简单，大部分为计算时间戳，更新索引，调整```ctts,stsc```索引等内容。

### 3.5 ```mov_read_close```
&emsp;&emsp;```mov_read_close```是在```avformat_close_input```时调用，其实现比较简单就是关闭流释放各种context。
```c
static int mov_read_close(AVFormatContext *s)
{
    MOVContext *mov = s->priv_data;
    int i, j;

    for (i = 0; i < s->nb_streams; i++) {
        AVStream *st = s->streams[i];
        MOVStreamContext *sc = st->priv_data;

        if (!sc)
            continue;

        av_freep(&sc->ctts_data);
        for (j = 0; j < sc->drefs_count; j++) {
            av_freep(&sc->drefs[j].path);
            av_freep(&sc->drefs[j].dir);
        }
        av_freep(&sc->drefs);

        sc->drefs_count = 0;

        if (!sc->pb_is_copied)
            ff_format_io_close(s, &sc->pb);         //内部就是调用io_close

        sc->pb = NULL;
        av_freep(&sc->chunk_offsets);
        av_freep(&sc->stsc_data);
        av_freep(&sc->sample_sizes);
        av_freep(&sc->keyframes);
        av_freep(&sc->stts_data);
        av_freep(&sc->sdtp_data);
        av_freep(&sc->stps_data);
        av_freep(&sc->elst_data);
        av_freep(&sc->rap_group);
        av_freep(&sc->display_matrix);
        av_freep(&sc->index_ranges);

        if (sc->extradata)
            for (j = 0; j < sc->stsd_count; j++)
                av_free(sc->extradata[j]);
        av_freep(&sc->extradata);
        av_freep(&sc->extradata_size);

        mov_free_encryption_index(&sc->cenc.encryption_index);
        av_encryption_info_free(sc->cenc.default_encrypted_sample);
        av_aes_ctr_free(sc->cenc.aes_ctr);

        av_freep(&sc->stereo3d);
        av_freep(&sc->spherical);
        av_freep(&sc->mastering);
        av_freep(&sc->coll);
    }

    av_freep(&mov->dv_demux);
    avformat_free_context(mov->dv_fctx);
    mov->dv_fctx = NULL;

    if (mov->meta_keys) {
        for (i = 1; i < mov->meta_keys_count; i++) {
            av_freep(&mov->meta_keys[i]);
        }
        av_freep(&mov->meta_keys);
    }

    av_freep(&mov->trex_data);
    av_freep(&mov->bitrates);

    for (i = 0; i < mov->frag_index.nb_items; i++) {
        MOVFragmentStreamInfo *frag = mov->frag_index.item[i].stream_info;
        for (j = 0; j < mov->frag_index.item[i].nb_stream_info; j++) {
            mov_free_encryption_index(&frag[j].encryption_index);
        }
        av_freep(&mov->frag_index.item[i].stream_info);
    }
    av_freep(&mov->frag_index.item);

    av_freep(&mov->aes_decrypt);
    av_freep(&mov->chapter_tracks);

    return 0;
}
```