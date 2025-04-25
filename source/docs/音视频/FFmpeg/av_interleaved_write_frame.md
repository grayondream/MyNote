# av_interleaved_write_frame

&emsp;&emsp;**摘要**：本文主要详细描述FFmpeg中封装时写packet到媒体文件的函数```av_interleaved_write_frame```的实现。
&emsp;&emsp;**关键字**：```av_interleaved_write_frame```
&emsp;&emsp;**读者须知**：读者需要熟悉ffmpeg的基本使用。

## 1 基本调用流程
&emsp;&emsp;```av_interleaved_write_frame```的基本调用流程图如下。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/av_interleaved_write_frame.drawio.svg)

&emsp;&emsp;首先就是根据输入数据是否为空选择调用的函数，如果为空就会调用```interleaved_write_packet```刷新数据，否则调用```write_packets_common```写数据。
&emsp;&emsp;```write_packets_common```中,```check_packet```检查输入的数据和期望写入的媒体流是否能够对上。```prepare_input_packet```对输入数据进行修正，如果pts和dts其中之一为NOPTS则设置为对方的值，以及如果设置了is_intra_only则每一帧都会设置标志位```AV_PKT_FLAG_KEY```。而```check_bitstream```就是调用```s->oformat->check_bitstream```检查流是否符合对应的格式。最后才是调用```write_packet_common```进行写数据。如果有设置filter的话就调用```write_packets_from_bsfs```处理。

&emsp;&emsp;```write_packet_common```会根据输入的参数是否需要交织存储来调用具体的函数写packet。非交织的情况下就会调用```write_packet```，该函数内部实际调用的```s->oformat->write_packet```和```s->oformat->write_uncoded_frame```写文件，后者处理裸流。
&emsp;&emsp;```interleaved_write_packet```内，如果AVOuputFormat设置了对应的函数指针则直接调用```s->oformat->interleave_packet```写文件，否则就用FFmpeg提供的```ff_interleave_packet_per_dts```。我们重点看下这个函数实现。

## 2 ```ff_interleave_packet_per_dts```
>&emsp;&emsp;音视频交织就是，将音频数据和视频数据存储到文件时，按照几帧音频几帧视频的方式存储，这样在处理流数据时就不会发生频繁的seek导致一些性能问题。音视频交织的视频对于网络播放也比较友好。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/av_interleaved.drawio.svg)

&emsp;&emsp;```ff_interleave_packet_per_dts```只是针对当前的两个流的packet的时间戳进行比较避免在文件存储过程中距离太远导致解封转时要频繁seek文件。最终封装文件写入到磁盘还是需要```write_packet```。该函数首先将送入的pkt插入到缓存队列中，然后在从当前缓存队列中选出一帧返回调用```write_packet```进行写入。
&emsp;&emsp;在看```ff_interleave_add_packet```函数的实现之前，我们先简单看下帧比较函数```interleave_compare_dts```的实现，该函数用来比较两个packet的dts。如果非音频流就是调用的```av_compare_ts```进行比较，否则会根据当前音频流是否有preload去除preload的偏移：
```c
int preload  = st ->codecpar->codec_type == AVMEDIA_TYPE_AUDIO;
int preload2 = st2->codecpar->codec_type == AVMEDIA_TYPE_AUDIO;
if (preload != preload2) {
    int64_t ts, ts2;
    preload  *= s->audio_preload;
    preload2 *= s->audio_preload;
    //preload不同时需要减掉preload的偏移
    ts = av_rescale_q(pkt ->dts, st ->time_base, AV_TIME_BASE_Q) - preload;
    ts2= av_rescale_q(next->dts, st2->time_base, AV_TIME_BASE_Q) - preload2;
    if (ts == ts2) {
        ts  = ((uint64_t)pkt ->dts*st ->time_base.num*AV_TIME_BASE - (uint64_t)preload *st ->time_base.den)*st2->time_base.den
            - ((uint64_t)next->dts*st2->time_base.num*AV_TIME_BASE - (uint64_t)preload2*st2->time_base.den)*st ->time_base.den;
        ts2 = 0;
    }
    comp = (ts2 > ts) - (ts2 < ts);
}
```
&emsp;&emsp;重点就是下面的代码，从当前buffer中找到当前帧的插入位置然后插入到packet的链表中。
```c
if (st->internal->last_in_packet_buffer) {
    next_point = &(st->internal->last_in_packet_buffer->next);
} else {
    next_point = &s->internal->packet_buffer;
}
//省略部分代码.......
if (*next_point) {
    if (chunked && !(pkt->flags & CHUNK_START))
        goto next_non_null;

    if (compare(s, &s->internal->packet_buffer_end->pkt, pkt)) {
        while (   *next_point
                && ((chunked && !((*next_point)->pkt.flags&CHUNK_START))
                    || !compare(s, &(*next_point)->pkt, pkt)))
            next_point = &(*next_point)->next;
        if (*next_point)
            goto next_non_null;
    } else {
        next_point = &(s->internal->packet_buffer_end->next);
    }
}
```
&emsp;&emsp;插入成功后回到```ff_interleave_packet_per_dts```中，从当前的packet链表的头结点拿到一阵返回给```write_packet```写入。
