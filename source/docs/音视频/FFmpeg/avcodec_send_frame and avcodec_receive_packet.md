# avcodec_send_frame and avcodec_receive_packet

&emsp;&emsp;**摘要**：本文主要描述了FFmpeg中用于编码的接口的具体调用流程，详细描述了该接口被调用时所作的具体工作。
&emsp;&emsp;**关键字**：```ffmpeg```、```avcodec_send_frame```、```avcodec_receive_packet```
&emsp;&emsp;**读者须知**：读者需要了解FFmpeg的基本使用流程，以及一些FFmpeg的基本常识，了解FFmpegIO相关的内容，以及大致的解码流程。

## 1 ```avcodec_send_frame```
&emsp;&emsp;```avcodec_send_frame```用于在编码时将一帧raw数据发送给编码器，其基本的调用流程比较简单，主要工作就是将输入的数据ref到Internal Frame上。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/avcodec_send_frame.drawio.svg)

&emsp;&emsp;```avcodec_send_frame```首先检查当前的codec是不是编码器且是否打开，并且检查codec中的buffer是否有数据没有，有的话就意味着上一帧的数据还没处理完需要等待这一帧处理完才能继续发送。
```cpp
    if (!avcodec_is_open(avctx) || !av_codec_is_encoder(avctx->codec))
        return AVERROR(EINVAL);

    if (avci->draining)
        return AVERROR_EOF;

    if (avci->buffer_frame->data[0])
        return AVERROR(EAGAIN);
```
&emsp;&emsp;然后是根据输入的frame是否为空来设置标志位，如果为空就表示是最后一帧数据后续的数据就无效了。能够看到在最后如果codec中的packet buffer是空的就会尝试获取一帧packet。
```cpp
    if (!frame) {
        avci->draining = 1;
    } else {
        ret = encode_send_frame_internal(avctx, frame);
        if (ret < 0)
            return ret;
    }

    if (!avci->buffer_pkt->data && !avci->buffer_pkt->side_data) {
        ret = encode_receive_packet_internal(avctx, avci->buffer_pkt);
        if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF)
            return ret;
    }
```
&emsp;&emsp;```encode_send_frame_internal```比较简单，主要就是针对音频数据进行参数检查并对数据进行填充，最后调用```av_frame_ref```将输入的数据的引用计数+1、

## 2 ```avcodec_receive_packet```
### 2.1 基本流程
&emsp;&emsp;```avcodec_receive_packet```相对复杂一点点儿，下面是其调用流程图。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/avcodec_receive_packet.drawio.svg)

&emsp;&emsp;首先是检查当前codec是否为编码器并且是否打开，如果是就继续。然后检查codec中的packet buffer是否有数据有的话就直接返回了，不然就会调用```encode_receive_packet_internal```。
```cpp
int attribute_align_arg avcodec_receive_packet(AVCodecContext *avctx, AVPacket *avpkt){
    AVCodecInternal *avci = avctx->internal;
    int ret;

    av_packet_unref(avpkt);

    if (!avcodec_is_open(avctx) || !av_codec_is_encoder(avctx->codec))
        return AVERROR(EINVAL);

    if (avci->buffer_pkt->data || avci->buffer_pkt->side_data) {
        av_packet_move_ref(avpkt, avci->buffer_pkt);
    } else {
        ret = encode_receive_packet_internal(avctx, avpkt);
        if (ret < 0)
            return ret;
    }

    return 0;
}
```
&emsp;&emsp;```encode_receive_packet_internal```首先就是参数检查，然后根据codec的函数指针设置看调用哪个流程获取编码流。```encode_simple_receive_packet```就是个while循环调用```encode_simple_internal```直到获取编码数据或者出错为止。
```c
    if (avctx->codec->receive_packet) {
        ret = avctx->codec->receive_packet(avctx, avpkt);
        if (ret < 0)
            av_packet_unref(avpkt);
        else
            // Encoders must always return ref-counted buffers.
            // Side-data only packets have no data and can be not ref-counted.
            av_assert0(!avpkt->data || avpkt->buf);
    } else
        ret = encode_simple_receive_packet(avctx, avpkt);
```

&emsp;&emsp;```encode_simple_internal```除了前面一大坨参数检查，主要救赎下面这块儿，看是利用多线程编码还是利用codec的encode接口编码。
```c
    if (CONFIG_FRAME_THREAD_ENCODER &&
        avci->frame_thread_encoder && (avctx->active_thread_type & FF_THREAD_FRAME))
        /* This might modify frame, but it doesn't matter, because
         * the frame properties used below are not used for video
         * (due to the delay inherent in frame threaded encoding, it makes
         *  no sense to use the properties of the current frame anyway). */
        ret = ff_thread_video_encode_frame(avctx, avpkt, frame, &got_packet);
    else {
        ret = avctx->codec->encode2(avctx, avpkt, frame, &got_packet);
        if (avctx->codec->type == AVMEDIA_TYPE_VIDEO && !ret && got_packet &&
            !(avctx->codec->capabilities & AV_CODEC_CAP_DELAY))
            avpkt->pts = avpkt->dts = frame->pts;
    }
```

### 2.2 多线程
&emsp;&emsp;编码的线程和解码的线程一样都是在```avcodec_open2```时创建的，编码是调用```ff_frame_thread_encoder_init```创建的，其中主要就是调用pthread的接口创建线程和相关的参数，可以看到其工作的函数为```static void * attribute_align_arg worker(void *v)```，编码过程中有多个线程每个线程都运行一个worker任务，通过信号量来进行消息的同步。该任务中最终会调用```avctx->codec->encode2```对数据进行编码。而所有的数据交互都是通过```ThreadContext```进行的，无论是输入数还是输出的数据还是消息同步都是通过该Context进行的。
```c
typedef struct{
    AVCodecContext *parent_avctx;
    pthread_mutex_t buffer_mutex;

    pthread_mutex_t task_fifo_mutex; /* Used to guard (next_)task_index */
    pthread_cond_t task_fifo_cond;

    unsigned max_tasks;
    Task tasks[BUFFER_SIZE];
    pthread_mutex_t finished_task_mutex; /* Guards tasks[i].finished */
    pthread_cond_t finished_task_cond;

    unsigned next_task_index;
    unsigned task_index;
    unsigned finished_task_index;

    pthread_t worker[MAX_THREADS];
    atomic_int exit;
} ThreadContext;
```
&emsp;&emsp;当数据到达时主线程会先拷贝数据然后发送信号量signal给任务线程，任务线程拿到消息后编码完成后给主线程发信号finish，主线程取走数据。