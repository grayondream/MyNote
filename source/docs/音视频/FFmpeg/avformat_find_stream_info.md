# avformat_find_stream_info

&emsp;&emsp;**摘要**：在使用FFmpeg库时通常使用```avformat_find_stream_info```相关函数来探测流的基本信息，为了更加深入理解FFmpeg的基本流程，本文根据FFmpeg 5.0的源码详细描述了该函数的具体实现。
&emsp;&emsp;**关键字**：FFmpeg
&emsp;&emsp;**读者须知**：读者需要了解FFmpeg的基本使用流程，以及一些FFmpeg的基本常识，了解FFmpegIO相关的内容，以及大致的解码流程。

## 1 ```avformat_find_stream_info```
&emsp;&emsp;```avformat_open_input```的主要工作时打开流并且对流进行初步的检测设置一些当前流的基本属性。而```avformat_find_stream_info```在调用时其耗时是比较久做的流信息探测的工作更多。```avformat_find_stream_info```比仅仅会读取文件来获取流信息，而且会尝试进行部分解码，根据```AVPcaket```和```AVFrame```更加准确的判断流的基本信息。部分场景下虽然跳过解码过程可以提升```avformat_find_stream_info```的执行速度，但是也会导致流探测的信息不准确。
```c
/**
 * Read packets of a media file to get stream information. This
 * is useful for file formats with no headers such as MPEG. This
 * function also computes the real framerate in case of MPEG-2 repeat
 * frame mode.
 * The logical file position is not changed by this function;
 * examined packets may be buffered for later processing.
 *
 * @param ic media file handle
 * @param options  If non-NULL, an ic.nb_streams long array of pointers to
 *                 dictionaries, where i-th member contains options for
 *                 codec corresponding to i-th stream.
 *                 On return each dictionary will be filled with options that were not found.
 * @return >=0 if OK, AVERROR_xxx on error
 *
 * @note this function isn't guaranteed to open all the codecs, so
 *       options being non-empty at return is a perfectly normal behavior.
 *
 * @todo Let the user decide somehow what information is needed so that
 *       we do not waste time getting stuff the user does not need.
 */
int avformat_find_stream_info(AVFormatContext *ic, AVDictionary **options);
```

&emsp;&emsp;```avformat_find_stream_info```的最终目的就是获取目标流的详细信息如果拿不到就会尝试获取码流```AVPacket```，从码流中解析相关流信息，如果码流中也拿不到就会尝试解码从```AVFrame```中获取。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/avformat_find_stream_info.drawio.svg)

## 2 详细调用过程
&emsp;&emsp;```avformat_find_stream_info```代码的内容非常难以阅读，这里不贴对应的代码了，我们简单分析下。
&emsp;&emsp;在进行探测前下面这部分代码用来设置探测的时长，可以看到部分时长是和格式强相关的，应该和文件本身关系比较大。
```c
max_stream_analyze_duration = max_analyze_duration;
max_subtitle_analyze_duration = max_analyze_duration;
if (!max_analyze_duration) {
    max_stream_analyze_duration =
    max_analyze_duration        = 5*AV_TIME_BASE;
    max_subtitle_analyze_duration = 30*AV_TIME_BASE;
    if (!strcmp(ic->iformat->name, "flv"))
        max_stream_analyze_duration = 90*AV_TIME_BASE;
    if (!strcmp(ic->iformat->name, "mpeg") || !strcmp(ic->iformat->name, "mpegts"))
        max_stream_analyze_duration = 7*AV_TIME_BASE;
}
```
&emsp;&emsp;然后就是循环遍历打开每个流的解码器，为后续解码做准备，这部分就是调用```avcodec_find_decoder```和```avcodec_open2```查找解码器和打开解码器，具体实现等到讲解码时再说。
&emsp;&emsp;之后便是不断循环检测，循环内部会不断判断当前是否已经成功获取完整的流信息，成功或者超出探测时长上限的话就结束；否则就会从读取```AVPacket```和解码帧```AVFrame```，读取```AVPacket```和解码```AVFrame```分别是调用```ff_read_packet```和```avcodec_send_packet,avcodec_receive_frame```进行码流的读取和解码。
&emsp;&emsp;```estimate_timings```用于估算当前媒体文件的时长，根据不同的格式会采取不同的方式：
1. ```mpeg```和```mpegts```会采用pts的方式估算，即调用```estimate_timings_from_pts```估算；
2. 如果非情况1且流本身已经探测到一部分时长信息，则调用```fill_all_stream_timings```根据已有的时长相关信息进行统一；
3. 其他情况调用```estimate_timings_from_bit_rate```，利用文件码流大小和码率估算；

&emsp;&emsp;```estimate_timings_from_pts```核心就是累加```AVPacket```的时长，如下：
```c
for (;;) {
    if (read_size >= DURATION_MAX_READ_SIZE << (FFMAX(retry - 1, 0)))
        break;

    do {
        ret = ff_read_packet(ic, pkt);
    } while (ret == AVERROR(EAGAIN));
    if (ret != 0)
        break;
    read_size += pkt->size;
    st         = ic->streams[pkt->stream_index];
    if (pkt->pts != AV_NOPTS_VALUE &&
        (st->start_time != AV_NOPTS_VALUE ||
            st->internal->first_dts  != AV_NOPTS_VALUE)) {
        if (pkt->duration == 0) {
            ff_compute_frame_duration(ic, &num, &den, st, st->internal->parser, pkt);
            if (den && num) {
                pkt->duration = av_rescale_rnd(1,
                                    num * (int64_t) st->time_base.den,
                                    den * (int64_t) st->time_base.num,
                                    AV_ROUND_DOWN);
            }
        }
        duration = pkt->pts + pkt->duration;
        found_duration = 1;
        if (st->start_time != AV_NOPTS_VALUE)
            duration -= st->start_time;
        else
            duration -= st->internal->first_dts;
        if (duration > 0) {
            if (st->duration == AV_NOPTS_VALUE || st->internal->info->last_duration<= 0 ||
                (st->duration < duration && FFABS(duration - st->internal->info->last_duration) < 60LL*st->time_base.den / st->time_base.num))
                st->duration = duration;
            st->internal->info->last_duration = duration;
        }
    }
    av_packet_unref(pkt);
}
```
&emsp;&emsp;而```estimate_timings_from_bit_rate```代码比较简单就是```duration=filesize/bitrate```。
```c
 duration = av_rescale(filesize, 8LL * st->time_base.den,
                                          ic->bit_rate *
                                          (int64_t) st->time_base.num);
                    st->duration = duration;
```
&emsp;&emsp;最后根据已经探测的的信息统一对时间等信息进行统一。

## 3 参考文献
- [FFmpeg源代码简单分析：avformat_find_stream_info()](https://blog.csdn.net/leixiaohua1020/article/details/44084321)