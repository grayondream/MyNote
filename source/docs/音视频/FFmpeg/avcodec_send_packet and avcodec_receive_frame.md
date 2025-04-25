# avcodec_send_packet and avcodec_receive_frame

&emsp;&emsp;**摘要**：本文主要描述了FFmpeg中用于解码的接口的具体调用流程，详细描述了该接口被调用时所作的具体工作。
&emsp;&emsp;**关键字**：```ffmpeg```、```avcodec_send_packet```、```avcodec_receive_frame```
&emsp;&emsp;**读者须知**：读者需要了解FFmpeg的基本使用流程，以及一些FFmpeg的基本常识，了解FFmpegIO相关的内容，以及大致的解码流程。

&emsp;&emsp;```avcodec_send_packet```接口将```AVPacket```数据发送给解码器进行解码，然后通过```avcodec_receive_frame```获取数据。

## 1 ```avcodec_send_packet```
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/avcodec_send_packet.drawio.svg)

&emsp;&emsp;```avcodec_send_packet```首先是检查解码器的合法性以及数据是否为空，如果输入数据和Context符合要求就会删除```AVcodecContext->internal->buffer_pkt```中缓存的一帧码流数据，将输入的Packet拷贝到该buffer上。```av_bsf_send_packet```只是拷贝增加输入的Packet引用计数到```AVBSFInternal->buffer_pkt```，最后如果缓存的```buffer_frame```是空的就会调用```decode_receive_frame_internal```解码帧，该过程根据配置项可谓同步也可为异步。

### 1.1 ```decode_receive_frame_internal```
&emsp;&emsp;```decode_receive_frame_internal```内就是真正的调用解码流程，如果解码器的```receive_frame```函数指针不为空就直接调用解码器的```receive_frame```进行解码该过程是同步的。否则就会调用```decode_simple_receive_frame```进行解码。解码完成后需要根据解码的数据和当前解码器Context的一些pts相关的值计算当前帧的具体pts和dts，另外如果有指定```FrameDecodeData```还会调用后处理流程```fdd->post_process```进行解码。

### 1.2 ```decode_simple_receive_frame```
&emsp;&emsp;```decode_simple_receive_frame```主要是调用```decode_simple_internal```进行解码。这里使用的Packet就是前面存储在```AVBSFInternal```中的```buffer_pkt```。然后就是实际调用解码的流程，如果没有配置解码线程就直接调用每个解码器对应的函数指针的```avctx->codec->decode```直接同步拿到帧。否则就会调用```ff_thread_decode_frame```进行多线程解码。
&emsp;&emsp;FFmpeg中每种格式，解码器等都有自己的描述结构，比如下面是gif的解码器描述。
```c
static const AVClass decoder_class = {
    .class_name = "gif decoder",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
    .category   = AV_CLASS_CATEGORY_DECODER,
};

const AVCodec ff_gif_decoder = {
    .name           = "gif",
    .long_name      = NULL_IF_CONFIG_SMALL("GIF (Graphics Interchange Format)"),
    .type           = AVMEDIA_TYPE_VIDEO,
    .id             = AV_CODEC_ID_GIF,
    .priv_data_size = sizeof(GifState),
    .init           = gif_decode_init,
    .close          = gif_decode_close,
    .decode         = gif_decode_frame,
    .capabilities   = AV_CODEC_CAP_DR1,
    .caps_internal  = FF_CODEC_CAP_INIT_THREADSAFE |
                      FF_CODEC_CAP_INIT_CLEANUP,
    .priv_class     = &decoder_class,
};
```
&emsp;&emsp;```ff_thread_decode_frame```内都是通过锁和条件变量进行同步的。首先根据当前的状态获取一个解码线程的Context，然后将当前的Packet提交到该线程上，提交就是将一帧数据增加引用让解码Context的```avpkt```也占用输入帧的引用计数，提交完成就会发送信号通知在等待的解码线程启动。
&emsp;&emsp;解码线程起始在```avcodec_open2```的时候就已经创建好了，在wait数据。具体的执行函数就是```frame_worker_thread```，该函数内就是调用```codec->decode```进行解码解码完成后就会发送通知到```ff_thread_decode_frame```中取解码完的帧。令条件```if (!p->avctx->thread_safe_callbacks && (
         p->avctx->get_format != avcodec_default_get_format ||
         p->avctx->get_buffer2 != avcodec_default_get_buffer2))```为A，如果A为true则当前线程是会被阻塞的，完全就是同步运行，否则就是多线程的。

```c
if (!p->avctx->thread_safe_callbacks && (
         p->avctx->get_format != avcodec_default_get_format ||
         p->avctx->get_buffer2 != avcodec_default_get_buffer2)) {
        while (atomic_load(&p->state) != STATE_SETUP_FINISHED && atomic_load(&p->state) != STATE_INPUT_READY) {
            int call_done = 1;
            pthread_mutex_lock(&p->progress_mutex);
            while (atomic_load(&p->state) == STATE_SETTING_UP)
                pthread_cond_wait(&p->progress_cond, &p->progress_mutex);

            switch (atomic_load_explicit(&p->state, memory_order_acquire)) {
            case STATE_GET_BUFFER:
                p->result = ff_get_buffer(p->avctx, p->requested_frame, p->requested_flags);
                break;
            case STATE_GET_FORMAT:
                p->result_format = ff_get_format(p->avctx, p->available_formats);
                break;
            default:
                call_done = 0;
                break;
            }
            if (call_done) {
                atomic_store(&p->state, STATE_SETTING_UP);
                pthread_cond_signal(&p->progress_cond);
            }
            pthread_mutex_unlock(&p->progress_mutex);
        }
    }
```
## 2 ```avcodec_receive_frame```
&emsp;&emsp;```avcodec_receive_frame```比较简单先检查```buffer_frame```有没有数据，有的话就直接返回，没有即调用```decode_receive_frame_internal```进行解码。