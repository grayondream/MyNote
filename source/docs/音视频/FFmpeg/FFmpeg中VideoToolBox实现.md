# FFmpeg中VideoToolBox实现

&emsp;&emsp;**摘要**：本文描述了FFmpeg中```videotoobox```解码器如何进行解码工作，如何将一个编码的码流解码为最终的裸流。
&emsp;&emsp;**关键字**：videotoobox,decoder,ffmpeg
&emsp;&emsp;VideoToolbox 是一个低级框架，提供对硬件编码器和解码器的直接访问。 它提供视频压缩和解压缩服务，以及存储在 CoreVideo 像素缓冲区中的光栅图像格式之间的转换服务。 这些服务以会话对象（压缩、解压缩和像素传输）的形式提供，并作为 Core Foundation (CF) 类型输出。 VideoToolbox支持H.263, H.264, HEVC, MPEG-1, MPEG-2, MPEG-4 Part 2, ProRes解码，H.264, HEVC, ProRes编码，最新的版本似乎也支持了VP9解码。
## 1 主流程
### 1.1 涉及的Context
&emsp;&emsp;FFmpeg中每个解码器都有自己的Context描述，该描述按照约定的格式描述对应的解码器参数和解码器的处理函数指针。FFmpeg中的VideoToolbox解码器主要实现代码在```libavcodec/videotoobox.{h,c}```中，其中针对每一种支持的解码格式定义了一个独立的Context，比如```ff_h263_videotoolbox_hwaccel,ff_h263_videotoolbox_hwaccel,ff_h264_videotoolbox_hwaccel,...```等，只是实现上有差异，我们主要关注其中一个即可，这里主要关注```ff_h264_videotoolbox_hwaccel```。
```c
const AVHWAccel ff_h264_videotoolbox_hwaccel = {
    .name           = "h264_videotoolbox",
    .type           = AVMEDIA_TYPE_VIDEO,
    .id             = AV_CODEC_ID_H264,
    .pix_fmt        = AV_PIX_FMT_VIDEOTOOLBOX,
    .alloc_frame    = ff_videotoolbox_alloc_frame,
    .start_frame    = ff_videotoolbox_h264_start_frame,
    .decode_slice   = ff_videotoolbox_h264_decode_slice,
    .decode_params  = videotoolbox_h264_decode_params,
    .end_frame      = videotoolbox_h264_end_frame,
    .frame_params   = ff_videotoolbox_frame_params,
    .init           = ff_videotoolbox_common_init,
    .uninit         = ff_videotoolbox_uninit,
    .priv_data_size = sizeof(VTContext),
};

```
&emsp;&emsp;该结构中定义了：
- 解码器的名称；
- 解码数据的类型；
- 解码器ID；
- 硬件解码的格式；
- 申请一个硬件相关的帧结构的函数指针；
- 解码开始前针对帧进行内存拷贝之类的操作；
- 解码数据；
- 解析解码器需要的参数比如sps等；
- 送帧结束后的后处理；
- 初始化硬件解码器；
- 销毁硬件解码器；
- 当前硬件解码器的描述结构。

&emsp;&emsp;```ff_h264_videotoolbox_hwaccel```是存储在```hw_configs```中的，运行时遍历该列表寻找期望的硬件解码器。所以解码工作是先经过FFmpeg内的```ff_h264_decoder```解码器再进入硬件解码器的。
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
    .capabilities          = /*AV_CODEC_CAP_DRAW_HORIZ_BAND |*/ AV_CODEC_CAP_DR1 |
                             AV_CODEC_CAP_DELAY | AV_CODEC_CAP_SLICE_THREADS |
                             AV_CODEC_CAP_FRAME_THREADS,
    .hw_configs            = (const AVCodecHWConfigInternal *const []) {
#if CONFIG_H264_DXVA2_HWACCEL
                               HWACCEL_DXVA2(h264),
#endif
#if CONFIG_H264_D3D11VA_HWACCEL
                               HWACCEL_D3D11VA(h264),
#endif
#if CONFIG_H264_D3D11VA2_HWACCEL
                               HWACCEL_D3D11VA2(h264),
#endif
#if CONFIG_H264_NVDEC_HWACCEL
                               HWACCEL_NVDEC(h264),
#endif
#if CONFIG_H264_VAAPI_HWACCEL
                               HWACCEL_VAAPI(h264),
#endif
#if CONFIG_H264_VDPAU_HWACCEL
                               HWACCEL_VDPAU(h264),
#endif
#if CONFIG_H264_VIDEOTOOLBOX_HWACCEL
                               HWACCEL_VIDEOTOOLBOX(h264),
#endif
                               NULL
                           },
    .caps_internal         = FF_CODEC_CAP_INIT_THREADSAFE | FF_CODEC_CAP_EXPORTS_CROPPING |
                             FF_CODEC_CAP_ALLOCATE_PROGRESS | FF_CODEC_CAP_INIT_CLEANUP,
    .flush                 = h264_decode_flush,
    .update_thread_context = ONLY_IF_THREADS_ENABLED(ff_h264_update_thread_context),
    .update_thread_context_for_user = ONLY_IF_THREADS_ENABLED(ff_h264_update_thread_context_for_user),
    .profiles              = NULL_IF_CONFIG_SMALL(ff_h264_profiles),
    .priv_class            = &h264_class,
};
```
&emsp;&emsp;```VTContext```VT解码过程中描述VT的Context。
```c
typedef struct VTContext {
    // The current bitstream buffer.
    uint8_t                     *bitstream;
    // The current size of the bitstream.
    int                         bitstream_size;
    // The reference size used for fast reallocation.
    int                         allocated_size;
    // The core video buffer
    CVImageBufferRef            frame;
    // Current dummy frames context (depends on exact CVImageBufferRef params).
    struct AVBufferRef         *cached_hw_frames_ctx;
    // Non-NULL if the new hwaccel API is used. This is only a separate struct
    // to ease compatibility with the old API.
    struct AVVideotoolboxContext *vt_ctx;

    // Current H264 parameters (used to trigger decoder restart on SPS changes).
    uint8_t                     sps[3];
    bool                        reconfig_needed;
    void *logctx;
} VTContext;
```
### 1.2 主要流程
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/vt.drawio.svg)

## 2 每个步骤的具体实现
### 2.1```ff_videotoolbox_common_init```
&emsp;&emsp;```ff_videotoolbox_common_init```在初始化解码器时调用，一般是在```avcodec_open2```时初始化硬件解码器。一般FFmpeg为了更加准确的探测当前视频的媒体信息，在```avformat_find_stream_info```时就会初始化解码器解码少部分的帧来进行流媒体信息探测。
&emsp;&emsp;初始化时首先就时申请VT的Context内存，并设置一些参数，实际上只设置了VT的callback函数和PixFormat。之后及时根据需要初始化```AVHWFramesContext```，主要就是申请内存并设置帧格式比如宽高，格式等等。
&emsp;&emsp;最后就是调用```videotoolbox_start```创建VT的Session，创建的过程比较简单就是直接调用Apple的API创建Session，需要重点关注的是如何设置的。具体的实现函数为```videotoolbox_decoder_config_create```，其中设置硬件加速的配置时写死的，无法进行配置。另外就是从当前的CodecCteonxt中取出sps等信息送给解码器，如果没有这些信息，解码器是无法准确识别出时间戳信息的。sps和pps的解析是由FFmpeg完成的。
```c
    switch (codec_type) {
    case kCMVideoCodecType_MPEG4Video :
        if (avctx->extradata_size)
            data = videotoolbox_esds_extradata_create(avctx);
        if (data)
            CFDictionarySetValue(avc_info, CFSTR("esds"), data);
        break;
    case kCMVideoCodecType_H264 :
        data = ff_videotoolbox_avcc_extradata_create(avctx);
        if (data)
            CFDictionarySetValue(avc_info, CFSTR("avcC"), data);
        break;
    case kCMVideoCodecType_HEVC :
        data = ff_videotoolbox_hvcc_extradata_create(avctx);
        if (data)
            CFDictionarySetValue(avc_info, CFSTR("hvcC"), data);
        break;
#if CONFIG_VP9_VIDEOTOOLBOX_HWACCEL
    case kCMVideoCodecType_VP9 :
        data = ff_videotoolbox_vpcc_extradata_create(avctx);
        if (data)
            CFDictionarySetValue(avc_info, CFSTR("vpcC"), data);
        break;
#endif
    default:
        break;
    }
```
&emsp;&emsp;解码callback的实现比较简单就是Retain一下CVPixelBuffer。
```c
static void videotoolbox_decoder_callback(void *opaque,
                                          void *sourceFrameRefCon,
                                          OSStatus status,
                                          VTDecodeInfoFlags flags,
                                          CVImageBufferRef image_buffer,
                                          CMTime pts,
                                          CMTime duration)
{
    VTContext *vtctx = opaque;

    if (vtctx->frame) {
        CVPixelBufferRelease(vtctx->frame);
        vtctx->frame = NULL;
    }

    if (!image_buffer) {
        av_log(vtctx->logctx,  AV_LOG_DEBUG,
               "vt decoder cb: output image buffer is null: %i\n", status);
        return;
    }

    vtctx->frame = CVPixelBufferRetain(image_buffer);
}
```

### 2.2 ```videotoolbox_h264_decode_params```和```ff_videotoolbox_frame_params```
&emsp;&esmp;```videotoolbox_h264_decode_params```主要的工作就是将上层解码出来额sps和pps信息拷贝到VTContext中。
```c
case H264_NAL_SPS: {
    GetBitContext tmp_gb = nal->gb;
    if (avctx->hwaccel && avctx->hwaccel->decode_params) {
        ret = avctx->hwaccel->decode_params(avctx,
                                            nal->type,
                                            nal->raw_data,
                                            nal->raw_size);
        if (ret < 0)
            goto end;
    }
    if (ff_h264_decode_seq_parameter_set(&tmp_gb, avctx, &h->ps, 0) >= 0)
        break;
    av_log(h->avctx, AV_LOG_DEBUG,
            "SPS decoding failure, trying again with the complete NAL\n");
    init_get_bits8(&tmp_gb, nal->raw_data + 1, nal->raw_size - 1);
    if (ff_h264_decode_seq_parameter_set(&tmp_gb, avctx, &h->ps, 0) >= 0)
        break;
    ff_h264_decode_seq_parameter_set(&nal->gb, avctx, &h->ps, 1);
    break;
```

&emsp;&emsp;```ff_videotoolbox_frame_params```比较简单就是将CodecContext中的参数传递给HWFramesContext。

## ```ff_videotoolbox_alloc_frame,ff_videotoolbox_h264_start_frame,ff_videotoolbox_h264_decode_slice,videotoolbox_h264_end_frame```
&emsp;&emsp;这几个函数每一帧都会调用，顺序是```alloc_frame->start_frame->decode_frame->end_frame```。
&emsp;&emsp;```ff_videotoolbox_alloc_frame```用来申请一块内存，此时的内存只是一块儿裸内存只是将release函数指针设置成了VT的release指针，还未与CVPixelBuffer绑定，绑定是在解码器的Callback中进行的。
&emsp;&emsp;```ff_videotoolbox_h264_start_frame```主要就是将上层传下来的stream数据流拷贝到VTContext中。
&emsp;&emsp;```videotoolbox_common_decode_slice```也是拷贝数据流。
&emsp;&emsp;```videotoolbox_h264_end_frame```才是具体将数据送给解码器的地方，核心的地方就是```videotoolbox_session_decode_frame```，这里送给解码器的数据流就上上面拷贝的数据流，需要注意的是在初始化时的callback中只是做了拷贝内存其他什么也没有做。这是因为在这里调用了```VTDecompressionSessionWaitForAsynchronousFrames```等待异步解码完成，能够保证上一帧解码完成后才送下一帧数据。

### 2.3 ```ff_videotoolbox_uninit```
&emsp;&emsp;```ff_videotoolbox_uninit```比较简单就是释放解码器的Context和缓存中的内存。

## 
- [Apple Documentation——VideoToolbox](https://developer.apple.com/documentation/videotoolbox)
