# avcodec_open2

&emsp;&emsp;**摘要**：本文主要描述了FFmpeg中用于打开编解码器接口```avcodec_open2```的具体调用流程，详细描述了该接口被调用时所作的具体工作。
&emsp;&emsp;**关键字**：```ffmpeg```、```avcodec_open2```
&emsp;&emsp;**注意**：读者需要了解FFmpeg的基本使用流程，以及一些FFmpeg的基本常识，了解FFmpegIO相关的内容，以及大致的解码流程。

## 1 avcodec_open2大致流程
&emsp;&emsp;打开codec（codec和编码器统称为codec）前需要通过```avcodec_find_decoder```找到具体的codec，该函数的实现本身就是遍历FFmpeg内部的一个全局```codec_list```，该列表存储了目前FFmpeg设置成的所有codec和编码器的```AVCodec```指针。
&emsp;&emsp;找到codec```AVCodec```之后需要自己通过```avcodec_alloc_context3```创建一个```AVCodecContext```的对象，该对象描述了当前codec的上下文，包含了一些流相关的信息，而```AVCodec```仅仅描述codec本身和流无关。一切准备好之后就需要调用```avcodec_open2```打开codec。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/avcodec_open2.drawio.svg)

## 2 调用流程详情
&emsp;&emsp;从上面的流程中能够看出来```avcodec_open2```主要做了三件事情：
1. 参数检查与设置；
2. 初始化codec线程；
3. 初始化codec。

&emsp;&emsp;**参数设置**基本上都是一个基本涉及解码过程的参数。首先是进行一些基本的参数设置与检查然后对codec的加锁，该锁是一个全局锁，所以这部分是线程安全的。FFmpeg使用的锁和线程相关都是```pthread```。
```cpp
static AVMutex codec_mutex = AV_MUTEX_INITIALIZER;
```

&emsp;&emsp;中间还有一些基本的参数设置与检查就不细细描述了。
&emsp;&emsp;**初始化codec**时会创建一个```AVCodecDescriptor```codec描述，这个也是从一个内部的全局表格```codec_descriptors```中搜索得到的。之后会根据当前codec的类型分别调用```ff_encode_preinit```和```ff_decode_preinit```做一些基本的初始化，这里面也是对当前codec的一些基本参数设置和一些和codec本身相关的对象的创建。而最终的codec初始化就是调用具体的codec初始化函数指针进行。
&emsp;&emsp;```avctx->codec->init```就是一个函数指针，每个codec对象都有初始化，编解码相关的接口，这里直接调用的是具体codec内部的函数指针。比如H264解码的```AVCodec```的指针为如下所示。
```c
const AVCodec ff_h264_decoder = {
    .name                  = "h264",
    .long_name             = NULL_IF_CONFIG_SMALL("H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10"),
    .type                  = AVMEDIA_TYPE_VIDEO,
    .id                    = AV_CODEC_ID_H264,
    .priv_data_size        = sizeof(H264Context),
    .init                  = h264_decode_init,
    .close                 = h264_decode_end,
    .decode                = h264_decode_frame,
    //....    
}
```
&emsp;&emsp;**线程初始化**。```ff_thread_init```用于初始化codec运行时的解码线程内部会创建多个线程的context并初始化，初始化最终调用的是```pthread_***_init```接口进行初始化。
```c
err = init_pthread(fctx, thread_ctx_offsets);
if (err < 0) {
    free_pthread(fctx, thread_ctx_offsets);
    av_freep(&avctx->internal->thread_ctx);
    return err;
}

fctx->async_lock = 1;
fctx->delaying = 1;

if (codec->type == AVMEDIA_TYPE_VIDEO)
    avctx->delay = src->thread_count - 1;

fctx->threads = av_mallocz_array(thread_count, sizeof(PerThreadContext));
if (!fctx->threads) {
    err = AVERROR(ENOMEM);
    goto error;
}

for (; i < thread_count; ) {
    PerThreadContext *p  = &fctx->threads[i];
    int first = !i;

    err = init_thread(p, &i, fctx, avctx, src, codec, first);
    if (err < 0)
        goto error;
}
```

## 3 其他细节
&emsp;&emsp;```avcodec_open2```参数部分针对解码器和编码器的不同有区分，具体的代码就是下面这部分
```c
    if (av_codec_is_decoder(avctx->codec)) {
        if (!avctx->bit_rate)
            //这里的码率获取是根据数据源的不同而不同的，非音频流都是返回avtx的bitrate。而音频流则需要根据当前的解码器参数进行计算避免参数不一致
            avctx->bit_rate = get_bit_rate(avctx);
        /* validate channel layout from the decoder */
        if (avctx->channel_layout) {
            int channels = av_get_channel_layout_nb_channels(avctx->channel_layout);
            if (!avctx->channels)
                avctx->channels = channels;
            else if (channels != avctx->channels) { 
                char buf[512];
                av_get_channel_layout_string(buf, sizeof(buf), -1, avctx->channel_layout);
                av_log(avctx, AV_LOG_WARNING,
                       "Channel layout '%s' with %d channels does not match specified number of channels %d: "
                       "ignoring specified channel layout\n",
                       buf, channels, avctx->channels);
                avctx->channel_layout = 0;
            }
        }
        if (avctx->channels && avctx->channels < 0 ||
            avctx->channels > FF_SANE_NB_CHANNELS) {
            ret = AVERROR(EINVAL);
            goto free_and_end;
        }
        if (avctx->bits_per_coded_sample < 0) {
            ret = AVERROR(EINVAL);
            goto free_and_end;
        }

#if FF_API_AVCTX_TIMEBASE
        if (avctx->framerate.num > 0 && avctx->framerate.den > 0)
            avctx->time_base = av_inv_q(av_mul_q(avctx->framerate, (AVRational){avctx->ticks_per_frame, 1}));
#endif
    }
```
&emsp;&emsp;预先初始化时解码器和编码器分别调用的```ff_decode_preinit```和```ff_encode_preinit```。
&emsp;&emsp;```ff_decode_preinit```主要就是初始化```AVBSFContext```，其他都是对参数进行校验。
&emsp;&emsp;```ff_encode_preinit```中会检查编码器和编码器Context的参数是不是能够对上如果对不上就会尝试校正，这些参数包括音频通道数，视频色彩范围，音频采样率、音频layout等等。

