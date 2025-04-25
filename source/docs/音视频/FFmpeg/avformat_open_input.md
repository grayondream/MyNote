# avformat_open_input

&emsp;&emsp;**摘要**：本文主要描述了FFmpeg中用于打开文件接口```avformat_open_input```的具体调用流程，详细描述了该接口被调用时所作的具体工作。
&emsp;&emsp;**关键字**：```ffmpeg```、```avformat_open_input```
&emsp;&emsp;**注意**：读者需要了解FFmpeg的基本使用流程，以及一些FFmpeg的基本常识，了解FFmpegIO相关的内容，以及大致的解码流程。

## 1 ```avformat_open_input```大致流程
&emsp;&emsp;在了解```avformat_open_input```的具体实现之前，我们先简单看下具体的函数声明和使用方式。```avformat_open_input```函数调用时会检测一部分当前格式的信息，更多的信息需要调用```avformat_find_stream_info```获取更加准确的信息。在函数调用时可以强制指定对应格式的格式，即参数```AVInputFormat```，否则FFmpeg内部会根据扩展名，数据格式等进行检测。另外也可以设置options，该option控制了探测码流的一些标志位，一般情况下不会设置，但是有些情况下需要根据具体的场景设置，比如有些素材普通的方式检测不到就需要设置```probesize```才能准确的检测到码流的属性等。

```c
/**
 * Open an input stream and read the header. The codecs are not opened.
 * The stream must be closed with avformat_close_input().
 *
 * @param ps Pointer to user-supplied AVFormatContext (allocated by avformat_alloc_context).
 *           May be a pointer to NULL, in which case an AVFormatContext is allocated by this
 *           function and written into ps.
 *           Note that a user-supplied AVFormatContext will be freed on failure.
 * @param url URL of the stream to open.
 * @param fmt If non-NULL, this parameter forces a specific input format.
 *            Otherwise the format is autodetected.
 * @param options  A dictionary filled with AVFormatContext and demuxer-private options.
 *                 On return this parameter will be destroyed and replaced with a dict containing
 *                 options that were not found. May be NULL.
 *
 * @return 0 on success, a negative AVERROR on failure.
 *
 * @note If you want to use custom IO, preallocate the format context and set its pb field.
 */
int avformat_open_input(AVFormatContext **ps, const char *url, const AVInputFormat *fmt, AVDictionary **options);
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/avformat_open_input.drawio.svg)

&emsp;&emsp;从上面的流程图（图中省略了ID3V2 metadata获取的相关内容）中我们能够看出```avformat_open_input```的主要工作就是打开文件流然后探测文件的码流检测出当前文件的格式。

## 2 调用流程详情
&emsp;&emsp;```avformat_alloc_context```，如果用户指定的```AVFormatContext```指针为空内部就会创建一个。而该函数主要的工作就是利用```av_malloc```申请一个```AVFormatContext```然后设置默认值。```av_opt_set_dict```就是设置基本的参数到当前的Context中。
&emsp;&emsp;```init_input```主要工作就是打开输入流，并探测对应的流内容，对流进行打分，分支越高对应格式的可能性越高否则越低。打开流的工作就是利用avio的打开流，根据不同的流类型会调用不同的打开API，文件就是调用```open```。在进行流探测是分别调用```av_probe_input_buffer2```和```av_probe_input_format2```对码流进行打分，前者最终也是调用后者实现的。
```c
static int init_input(AVFormatContext *s, const char *filename,
                      AVDictionary **options)
{
    int ret;
    AVProbeData pd = { filename, NULL, 0 };
    int score = AVPROBE_SCORE_RETRY;

    if (s->pb) {
        s->flags |= AVFMT_FLAG_CUSTOM_IO;
        if (!s->iformat)
            return av_probe_input_buffer2(s->pb, &s->iformat, filename,
                                         s, 0, s->format_probesize);
        else if (s->iformat->flags & AVFMT_NOFILE)
            av_log(s, AV_LOG_WARNING, "Custom AVIOContext makes no sense and "
                                      "will be ignored with AVFMT_NOFILE format.\n");
        return 0;
    }
    //文件流已经被打开过了就直接调用av_probe_input_format2进行格式检测
    if ((s->iformat && s->iformat->flags & AVFMT_NOFILE) ||
        (!s->iformat && (s->iformat = av_probe_input_format2(&pd, 0, &score))))
        return score;
    //打开文件流
    if ((ret = s->io_open(s, &s->pb, filename, AVIO_FLAG_READ | s->avio_flags, options)) < 0)
        return ret;

    if (s->iformat)
        return 0;
    return av_probe_input_buffer2(s->pb, &s->iformat, filename,
                                 s, 0, s->format_probesize);
}
```

&emsp;&emsp;```av_probe_input_format3```内通过遍历一个全局的```demuxer_list```表格，利用每种封装格式的探测指针对流进行打分，选择分值最高的作为最终的流格式。```demuxer_list```是一个自动生成的全局数组，如果没有编译FFmpeg是不会看到该数组的。
```c
static const AVInputFormat *demuxer_list[] = {
    &ff_aa_demuxer,
    &ff_aac_demuxer,
    &ff_aax_demuxer
    ...
    &ff_libmodplug_demuxer,
    NULL 
};
```

&emsp;&emsp;```av_probe_input_format3```的核心代码如下，遍历已知的格式列表，如果探测不到就会常使用mime_type和扩展名作为依据。
```c
while ((fmt1 = av_demuxer_iterate(&i))) {
    if (!is_opened == !(fmt1->flags & AVFMT_NOFILE) && strcmp(fmt1->name, "image2"))
        continue;
    score = 0;
    if (fmt1->read_probe) {
        score = fmt1->read_probe(&lpd);
        if (score)
            av_log(NULL, AV_LOG_TRACE, "Probing %s score:%d size:%d\n", fmt1->name, score, lpd.buf_size);
        if (fmt1->extensions && av_match_ext(lpd.filename, fmt1->extensions)) {
            switch (nodat) {
            case NO_ID3:
                score = FFMAX(score, 1);
                break;
            case ID3_GREATER_PROBE:
            case ID3_ALMOST_GREATER_PROBE:
                score = FFMAX(score, AVPROBE_SCORE_EXTENSION / 2 - 1);
                break;
            case ID3_GREATER_MAX_PROBE:
                score = FFMAX(score, AVPROBE_SCORE_EXTENSION);
                break;
            }
        }
    } else if (fmt1->extensions) {
        if (av_match_ext(lpd.filename, fmt1->extensions))
            score = AVPROBE_SCORE_EXTENSION;
    }
    if (av_match_name(lpd.mime_type, fmt1->mime_type)) {
        if (AVPROBE_SCORE_MIME > score) {
            av_log(NULL, AV_LOG_DEBUG, "Probing %s score:%d increased to %d due to MIME type\n", fmt1->name, score, AVPROBE_SCORE_MIME);
            score = AVPROBE_SCORE_MIME;
        }
    }
    if (score > score_max) {
        score_max = score;
        fmt       = fmt1;
    } else if (score == score_max)
        fmt = NULL;
}
```

&emsp;&emsp;随后就是调用```read_header```取读取一些基本的参数，比如流的数量，时长等。其他代码就是一些检查类以及参数更新类代码了。

## 3 参考文献
- [ID3](https://en.wikipedia.org/wiki/ID3)
- [FFmpeg源代码简单分析：avformat_open_input()](https://blog.csdn.net/leixiaohua1020/article/details/44064715)