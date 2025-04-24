# FFplay源码分析
## 1 ffplay 基本架构
### 1.1 视频解码播放的基本流程
&emsp;&emsp;ffmpeg视频解码播放的基本流程如下图所示：
1. 首先对网络媒体数据流进行解封装得到一般的视频封装格式比如MP4等，如果是本地播放的媒体文件就不需要解协议；
2. 然后对视频媒体文件进行解封装，得到未经过解码的视频、音频或者字幕流数据，在ffmpeg中得到的是```AVPacket```；
3. 然后分别对字幕、音频和视频数据进行解码，分别得到字幕、PCM数据和YUV数据；
4. 由于不同数据体积不同解码速度不同，视频解码相对比较慢，如果解码完就立马播放就会出现音频和视频播放不一致的情况因此需要进行音画同步；
5. 最后将音频视频数据分别输出到对应的设备完成播放。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/decode.drawio.svg)

&emsp;&emsp;ffplay是使用SDL（一种跨平台的音视频播放框架）进行具体平台的音视频播放，其使用方式和windows的API很像，比如创建窗口、打开音频输出、处理事件循环等。我们不用太关注于该框架的具体细节，大部分情况下根据函数或者变量名称就能判断涉及SDL的相关内容的含义。
&emsp;&emsp;ffplay内部使用多个线程完成音视频的解复用和解码过程，保证不同数据解码直接互不干扰。

### 1.2 ffplay的基本代码框架
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/ffplay_struct.drawio.svg)

&emsp;&emsp;上图是ffplay的基本框架图，ffplay处理音频的基本流程是创建三个线程分别解封装
- ffplay中参数传递使用一个```VideoState```传递，其中包含了各个部分运行时所需要的所有参数，比较庞大；
- ```event_loop```主要处理用户输入事件，比如切换显示模式，调整音量大小，控制播放seek等操作；
- ```video_refresh```是真正刷新视频画面的函数。
- ```stream_open```中初始化需要使用的队列、clock以及相关参数，并创建相关的线程比如音视频解复用线程、音视频解码线程。

## 2 ffplay中使用到的数据结构
### 2.1 AVPacket队列
```cpp
//avpackt的队列节点，serial表示当前packet的序号
typedef struct MyAVPacketList {
    AVPacket *pkt;
    int serial;
} MyAVPacketList;

//avpacket的队列的head节点，也就是说PacketQueue是一个带有头结点的队列，该头结点中定义了一些队列相关的metadata以及队列使用到的锁和条件变量（保证队列的线程安全）
typedef struct PacketQueue {
    AVFifoBuffer *pkt_list;         //ffmpeg实现的FIFO缓冲区
    int nb_packets;                 //当前队列中avpcket的数量
    int size;                       //队列中所有数据的总字节数
    int64_t duration;               //队列所有及诶大的时长之和
    int abort_request;              //是否终止对队列的操作，用于安全快速的退出
    int serial;                     //序列号
    SDL_mutex *mutex;               //保证线程安全额锁
    SDL_cond *cond;                 //读写的条件变量
} PacketQueue;
```

&emsp;```PacketQueue```是一个线程安全FIFO队列，数据节点是```MyAVPacketList```，内部使用AVFifoBuffer实现数据的存取， ```abort_request```控制队列的状态，```muxte,cond```进行同步和临界区保护。下面是```PacketQueue```一系列的操作api：
- ```packet_queue_put_private```：队列添加packet的具体实现，基本逻辑为检查队列的大小如果不足则扩张，扩张的规则比较简单就是增加一个packet的大小，然后将packet写入到队列中；
- ```packet_queue_put```：写入一个packet，会对packet进行复制，实际上写入的是实现是```packet_queue_put_private```；
- ```packet_queue_put_nullpacket```：写入一个空packet，具体是调用```packet_queue_put```实现，那和普通包有何区别，怀疑是为了保障代码的兼容性，因为旧的api就是构造一个空包写入；
- ```packet_queue_init```：初始化队列，设置关于队列的一些状态信息、锁和信号量；
- ```packet_queue_flush```：刷新队列，将队列中的数据读取并free，并设置相关的状态到初始状态；
- ```packet_queue_destroy```:销毁队列，主要是刷新队列并销毁锁和信号量；
- ```packet_queue_abort```：暂停队列，即设置```abort_request```；
- ```packet_queue_start```：开始队列，清除标志位```abort_request```并自增```serial```；
- ```packet_queue_get```：从队列中取出一个packet。

&emsp;&emsp;队列的实现是一个普通的队列，需要注意的细节就是```serial```的更新时机，在放入同一个队列时队列会将自己的serial号赋值给对应的packet包，并且只有触发```packet_queue_start```和```packet_queue_flush```两个事件时才会更新```serial```。```serial```能够用来区分不同时刻解封装得到的packet是不是连续的包，如果播放时突然暂停，再resume的话serial更新，上一个packet和下一个packet就不是连续的。
```cpp
static int packet_queue_put_private(PacketQueue *q, AVPacket *pkt){
    MyAVPacketList pkt1;
    //...
    pkt1.serial = q->serial;
    //...
}

static void packet_queue_start(PacketQueue *q){
    //...
    q->serial++;
    //...
}

static void packet_queue_flush(PacketQueue *q){
    MyAVPacketList pkt1;
    //...
    q->serial++;
    //...
}
```
### 2.2 AVFrame队列
```cpp
//解码出来的avframe数据存储节点
typedef struct Frame {
    AVFrame *frame;         //解码的音频或者视频数据
    AVSubtitle sub;         //解码的字母数据
    int serial;
    double pts;           /* presentation timestamp for the frame */
    double duration;      /* estimated duration of the frame */
    int64_t pos;          /* byte position of the frame in the input file */
    int width;
    int height;
    int format;
    AVRational sar;         //采样宽高比
    int uploaded;           //当前帧是否已经已经上屏，若果已经上屏则不会再刷新
    int flip_v;             //控制是否垂直翻转
} Frame;

//avframe队列的head节点
typedef struct FrameQueue {
    Frame queue[FRAME_QUEUE_SIZE];      //固定队列，环形缓冲区
    int rindex;                         //读索引，队头
    int windex;                         //写索引，队尾
    int size;                           //当前队列中节点个数
    int max_size;                       //最大允许存储的节点个数，方便区分是否full
    int keep_last;                      //是否要保留最后一个读节点，1的话最后一个节点就不会被覆盖
    int rindex_shown;                   //当前节点是否已经显示
    SDL_mutex *mutex;                   //锁
    SDL_cond *cond;                     //条件变量
    PacketQueue *pktq;                  //关联的packet队列
} FrameQueue;
```

&emsp;&emsp;```Frame```是同时能够表示音频、视频、字母的大杂烩，同时包含了一些状态信息来表示当前Frame的具体状态。```FrameQueue```是使用存储```Frame```节点的线程安全队列，该线程安全队列并没有类似```PacketQueue```使用```AVFifoBuffer```实现，而是用栈上的数组实现线程安全的环形队列。```FrameQueue```的实现一切未高效率让路，一方面使用静态数组（最大不超过16个节点）不用考虑动态内存的问题，减少比不要的内存操作的小号，另一方面队列的lock力度比较小，针对rindex和wrindx都未进行加锁，因为程序能够保证两个线程独立的更新和读取对应的值。

```FrameQueue```的一些操作函数：
- ```frame_queue_init```：初始化avframe队列，实现很简单主要是初始化队列的内存和metadata以及锁等，另外初始化的时候会使用```av_frame_alloc```提前分配好对应的frame的内存；
- ```frame_queue_unref_item```：销毁```Frame```节点中的数据；
- ```frame_queue_destory```：销毁队列，主要工作是销毁frame以及销毁锁以及条件变量
- ```frame_queue_signal```：线程安全的发送signal；
- ```frame_queue_peek```：根据当前的读索引获取一个frame，rindex_shown的作用是如果当前节点已经被读取则跳过读取下一个；
- ```frame_queue_peek_next```：相比于```frame_queue_peek```，读取下一个节点的数据；
- ```frame_queue_peek_last```：返回rindex指向的节点，无论是否被读取过；
- ```frame_queue_peek_writable```：检查当前队列的wrindex指向的内存是否可写，如果可写则获取对应数据的指针，如果不可写的阻塞wait，否则返回对应节点的指针；
- ```frame_queue_peek_readable```：逻辑和```frame_queue_peek_writable```类似，检查队列是否为空，空则wait；否则返回可读的frame的指针；
- ```frame_queue_push```：仅仅更新windex和size，因为再使用```frame_queue_peek_writable```时节点的指针已经移交用户，用户需要将数据存储到其中，这里只需要更新索引即可；
- ```frame_queue_next```：逻辑与```frame_queue_push```类似，只更新索引；
- ```frame_queue_nb_remaining```：返回当前队列中尚未显示的节点的数量；
- ```frame_queue_last_pos```：获取上一次显示的位置。

### 2.3 时钟
&emsp;&emsp;ffplay中有三种时钟，音频，视频和系统外部时钟，主要用于音画同步。

```cpp
typedef struct Clock {
    double pts;             // 当前帧(待播放)显示时间戳，播放后，当前帧变成上一帧
    //c->pts_drift = c->pts - time;
    double pts_drift;       /* clock base minus time at which we updated the clock */
    double last_updated;    // 当前时钟(如视频时钟)最后一次更新时间，也可称当前时钟时间
    double speed;           // 时钟速度控制，用于控制播放速度
    int serial;             // 播放序列，所谓播放序列就是一段连续的播放动作，一个seek操作会启动一段新的播放序列
    int paused;             // 暂停标志
    int *queue_serial;      /* pointer to the current packet queue serial, used for obsolete clock detection */
} Clock;
```

- ```set_clock_at```：设置clock的基本参数;
- ```set_clock```：设置clock的pts和serial；
- ```set_clock_speed```：设置speed；
- ```init_clock```：初始化clock。
- ```get_clock```：获取当前的时间，如果暂停状态则返回的是pts，播放状态则返回的是```pts_drift + time - (time - last_updated) * (1 - speed)```(```time```是系统时间)；
- ```get_master_sync_type```：获取当前的同步类型；
- ```get_master_clock```：获取主同步的时钟的时间；
- ```check_external_clock_speed```：检查并设置外部时钟的speed，会根据预定的参数和当前时钟的参数设置。
- ```sync_clock_to_slave```：将同步的主时钟同步到slave时钟上。

### 2.4 参数
&emsp;&emsp;音频的参数。
```cpp
typedef struct AudioParams {
    int freq;
    int channels;
    int64_t channel_layout;
    enum AVSampleFormat fmt;
    int frame_size;
    int bytes_per_sec;
} AudioParams;
```

## 3 ffplay基本实现
&emsp;&emsp;ffplay的基本流程上面已经能够看到，下面主要描述一些实现细节。ffplay中我们主要关注```stream_open```和```video_fresh```两个函数，这两个函数是主要的线程创建入口和刷新视频的入口。
&emsp;&emsp;```stream_open```中申请VideoState结构体的内存，初始化相关的参数，初始化音频、视频和字幕的AVPacket的队列以及对应的AVFrame的队列，和视频、音频、外部时钟，并将一些参数合法化比如音量限制在0-100之间，最后开启```read_thread```线程。

### 3.1 ```read_thread```线程
&emsp;&emsp;```read_thread```是ffplay的解复用线程，其基本流程很简单就是我们使用ffmpeg时解复用的基本流程：打开媒体文件```AVFormatContext```→设置```AVFormatContext```解复用的参数→查找媒体文件中的媒体流（音频、视频和字幕）→寻找每个流对应的流的索引→打开音频、视频和字幕流对应的解码线程→然后便是循环利用```av_read_frame```解复用读取```AVPacket```。

```cpp
//去除一些细节的主流程
avfomat_alloc_context();
avformat_open_input();
av_dict_set();
avformat_find_stream_info();
avformat_seek_file();
//for search
//得到音频、字幕、视频的索引
av_find_best_stream();
//打开音频、视频、字幕的解码线程
stream_component_open();
for(;;){
    av_read_frame();
    if(isseek){
        avformat_seek_file();
    }
    packet_queue_put();
}
```

&emsp;&emsp;在搜索媒体流对应的流时，首先通过```for```循环```AVFormatContext```中的所有流找到对应的流的第一个流的索引，然后使用```av_find_best_stream```确定最终的流的索引。```av_find_best_stream```中的实现也是通过遍历媒体中的流寻找对应的流的索引，ffplay实现中第一次手动搜寻是期望期望一个参考索引，然后将该参考索引传递给```av_find_best_stream```寻找最佳的流。ffmpeg中最佳流是根据各种启发式确定的最有可能是用户期望的，并尝试找到对应的流的最佳的解码器，一般都会返回用户指定的流的索引。
&emsp;&emsp;解复用线程主循环主要就是读取一帧的处理seek事件将一帧的packet入队。seek的基本逻辑是如果当前状态是播放状态就正常seek然后清空队列，如果是暂停状态seek完不仅仅要清空队列还会显示刷新一帧的画面（具体的步骤是先恢复播放状态播放一帧再暂停），同步外部时钟，不然seek完画面还是seek前的画面。在解复用时会控制队列的大小，如果队列已经full则会阻塞10ms。在解复用入队是会计算当前packet的时间是否在播放范围内，如果不在则跳过。在```read_thread```主循环中会控制seek，退出、循环播放逻辑。

```cpp
stream_start_time = ic->streams[pkt->stream_index]->start_time;
pkt_ts = pkt->pts == AV_NOPTS_VALUE ? pkt->dts : pkt->pts;
pkt_in_play_range = duration == AV_NOPTS_VALUE || (pkt_ts - (stream_start_time != AV_NOPTS_VALUE ? stream_start_time : 0)) * av_q2d(ic->streams[pkt->stream_index]->time_base) - (double)(start_time != AV_NOPTS_VALUE ? start_time : 0) / 1000000 <= ((double)duration / 1000000);
//下面是网络上找的解析函数功能和上面的代码相同相对俩说要清晰很多
int64_t get_stream_start_time(AVFormatContext* ic, int index) {
    int64_t stream_start_time = ic->streams[index]->start_time;
    return stream_start_time != AV_NOPTS_VALUE ? stream_start_time : 0;
}

int64_t get_pkt_ts(AVPacket* pkt) {//ts: timestamp（时间戳）的缩写
    return pkt->pts == AV_NOPTS_VALUE ? pkt->dts : pkt->pts;
}

double ts_as_second(int64_t ts，AVFormatContext* ic，int index) {
    return ts * av_q2d(ic->streams[index]->time_base);
} 

double get_ic_start_time(AVFormatContext* ic) {//ic中的时间单位是us
    return (start_time != AV_NOPTS_VALUE ? start_time : 0) / 1000000;
}
int is_pkt_in_play_range(AVFormatContext* ic, AVPacket* pkt) {
    if (duration == AV_NOPTS_VALUE) //如果当前流无法计算总时长，按无限时长处理
        return 1;

    //计算pkt相对stream位置
    int64_t stream_ts = get_pkt_ts(pkt) - get_stream_start_time(ic, pkt->stream_index);
    double stream_ts_s = ts_as_second(stream_ts, ic, pkt->stream_index);

    //计算pkt相对ic位置
    double ic_ts = stream_ts_s - get_ic_start_time(ic);

    //是否在时间范围内,duration是一个全局变量
    return ic_ts <= ((double)duration / 1000000);
}
```
&emsp;&emsp;```read_thread```中是通过```stream_component_open```打开三个流的解码线程，其基本流程：创建解码器上下文环境→查找解码器→打开解码器→初始化```Decoder```打开对应的解码线程。在打开解码器时会将解码器相关的参数同步到```VideoState```中，```decoder_init```就是简单的更新```VideoState```中```Decoder```中的参数，```decoder_start```会针对三个流分别调用分别创建```audio_thread```、```video_thread```和```subtitle_thread```线程。
```cpp
//去除一些细节的主流程
    avcodec_alloc_context3();
    avcodec_parameters_to_context();
    avcodec_find_decoder();
    avcodec_open2();
    //打开三个流的解码线程
    decoder_init();
    decoder_start()
```

### 3.2 流解码线程
&emsp;&emsp;流解码线程就是上面提到的```audio_thread```、```video_thread```和```subtitle_thread```三个线程。

**video_thread**
&emsp;&emsp;```video_thread```解码线程的基本流程很简单，取帧使用```get_video_frame```，帧的入队使用```queue_picture```该函数的实现很简单就是将frame入队并设置相关的参数调整窗口大小。
```cpp
AVFrame *frame = av_frame_alloc();
for(;;){
    ret = get_video_frame(is, frame);
    //滤镜处理
    duration = (frame_rate.num && frame_rate.den ? av_q2d((AVRational){frame_rate.den, frame_rate.num}) : 0);
    pts = (frame->pts == AV_NOPTS_VALUE) ? NAN : frame->pts * av_q2d(tb);
    ret = queue_picture(is, frame, pts, duration, frame->pkt_pos, is->viddec.pkt_serial);
    av_frame_unref(frame);
}

av_frame_free(&frame);
```

&emsp;&emsp;```get_video_frame```取帧的函数实际实现是```decoder_decode_frame```，取帧后会进行丢帧的处理，这个后面描述。另外视频、音频和字幕都是使用这个函数解码帧。
```cpp
decoder_decode_frame
//丢帧逻辑，暂时省略
```
**audio_thread**
&emsp;&emsp;```audio_thread```的基本逻辑和video基本相同，只是帧入队的参数设置比较简单。取帧的具体实现同样使用```get_video_frame```函数。
```cpp
//设置队列节点的metadata并将解码出的avframe写入队列中
af->pts = (frame->pts == AV_NOPTS_VALUE) ? NAN : frame->pts * av_q2d(tb);
af->pos = frame->pkt_pos;
af->serial = is->auddec.pkt_serial;
af->duration = av_q2d((AVRational){frame->nb_samples, frame->sample_rate});

av_frame_move_ref(af->frame, frame);
frame_queue_push(&is->sampq);
```

**subtitle_thread**
&emsp;&emsp;```subtitle_thread```更简单，基本流程和```video_thread```相同，唯一的区别是不需要管理AVFrame的内存，原因是```Frame```中使用的是```AVSubtitle```变量而不是指针，大概是字母数据占用的内存比较小的缘故。
```cpp
sp->pts = pts;
sp->serial = is->subdec.pkt_serial;
sp->width = is->subdec.avctx->width;
sp->height = is->subdec.avctx->height;
sp->uploaded = 0;
```

**decoder_decode_frame**
&emsp;&emsp;视频、音频和字幕流的解码都是复用的这个逻辑，其中基本框架相同，音频和视频的区别是设定pts的方式不同，而字幕是使用```avcodec_decode_subtile2```这个函数进行解码。该函数实现的基本框架是两个while循环，首先是取帧的函数调用，第二个循环用来从队列中取出一个packet并保证当前packet和当前播放的序列相同，防止不同序列的serail混杂。最后是将packet送到解码器中。

```cpp
for(;;){
    if (d->queue->serial == d->pkt_serial) {
        do{
            //暂停等逻辑的控制
            avcodec_receive_frame()
            //设定对应帧的pts
        }while (ret != AVERROR(EAGAIN));  
        do{
            packet_queue_get()
            if (old_serial != d->pkt_serial) {
                avcodec_flush_buffers(d->avctx);
            }

            if (d->queue->serial == d->pkt_serial)
                break;
        }while(1);

         if (d->avctx->codec_type == AVMEDIA_TYPE_SUBTITLE) {
             avcodec_decode_subtitle2()
         }else{
             avcodec_send_packet()
         }
    }
}
```

### 3.3 画面刷新

#### 3.3.1 ```video_refresh```
&emsp;&emsp;视频画面的刷新视频画面，下面的图是刷新视频画面的大致流程图，图中省略了显示结束后的更新clock的描述。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/video_refresh.drawio.svg)

&emsp;&emsp;在刷新视频时，会从frame队列中取出两帧画面```lastvp```和```vp```对比两帧的```serail```是否相同，不相同则回到开头直到取出相同```serial```的帧。然后就是计算上一帧画面显示了多长时间，如果并未显示足够则继续显示否则取出下一帧。这里计算两帧之间的显示时差是通过```vp_duration```计算，其实现就是两个针对的pts的差值，得到两帧的显示差值后会根据传入的```remaining_time```计算出新的```remaing_time```。传入的```remaining_time```除了第一次调用使用的默认值0.01，后面都是上一次画面刷新计算出来的值。
```cpp
 /* compute nominal last_duration */
last_duration = vp_duration(is, lastvp, vp);
delay = compute_target_delay(last_duration, is);

time= av_gettime_relative()/1000000.0;
if (time < is->frame_timer + delay) {
    *remaining_time = FFMIN(is->frame_timer + delay - time, *remaining_time);
    goto display;
}
```
&emsp;&emsp;在决定好要显示当前帧后则需要更新当前显示帧的timer，能够看到```frame_timer```就是每一帧显示的真实时刻值。
```cpp
is->frame_timer += delay;
if (delay > 0 && time - is->frame_timer > AV_SYNC_THRESHOLD_MAX)
    is->frame_timer = time;
```

&emsp;&emsp;```video_refresh```中还会进行字幕的显示处理，字幕的显示处理相比视频简单很多主要就是检查序列号是否相同，还会检查当前显示的字幕的pts和视频的pts是否满足条件丢帧的条件，如果满足则会将字幕区域清空。下面是判断清空字幕显示区域的判断条件，sp是当前帧字幕，sp2是上一帧字幕。
```cpp
 if (sp->serial != is->subtitleq.serial
|| (is->vidclk.pts > (sp->pts + ((float) sp->sub.end_display_time / 1000)))
|| (sp2 && is->vidclk.pts > (sp2->pts + ((float) sp2->sub.start_display_time / 1000))))
```
&emsp;&emsp;```video_display```是显示视频画面的接口，在显示视频画面时会调用```video_image_display```函数显示一帧的画面。```video_image_display```会显示字幕帧以及调用```upload_texture```显示视频画面，```upload_texture```中会调用```sws_scale```对画面进行缩放然后上屏。

```cpp
sp = frame_queue_peek(&is->subpq);
//如果字幕显示的时间未过时则显示
if (vp->pts >= sp->pts + ((float) sp->sub.start_display_time / 1000)) {}
```

#### 3.3.2 播放音频
&emsp;&emsp;ffplay中打开SDL音频是在```audio_open```中，该函数在创建解码线程前被调用。```sdl_audio_callback```是sdl取帧缓冲的回调函数，该函数中就是将不断从```audio_buf```（VideoState维护的一个缓冲）获取数据，然后拷贝到SDL的缓冲中。如果```audio_buf```中没有数据则会调用```audio_decode_frame```从帧队列中取出数据。该数据除了会同步pts外还会根据目标的音频的参数将取出的音频数据进行```swr_convert```重采样。

```cpp
//更新音频的clock
audio_clock0 = is->audio_clock;
/* update the audio clock with the pts */
if (!isnan(af->pts))
    is->audio_clock = af->pts + (double) af->frame->nb_samples / af->frame->sample_rate;
else
    is->audio_clock = NAN;
is->audio_clock_serial = af->serial;
```

### 3.4 音画同步
&emsp;&emsp;不同的数据解码的数据不同为了保证能够按照视频的参数显示图像因此需要进行同步。既然是同步，那就会有同步的对象，向谁同步，因此就有三种同步对象音频、视频和外部时钟（ffplay默认是同步音频）。
```cpp
enum {
    AV_SYNC_AUDIO_MASTER, /* default choice */
    AV_SYNC_VIDEO_MASTER,
    AV_SYNC_EXTERNAL_CLOCK, /* synchronize to an external clock */
};
```

#### 3.4.1 视频同步音频
&emsp;&emsp;视频的丢帧逻辑在```video_refresh```中，下面是其中关于音画同步的代码的详细注释。视频同步其他时钟的基本思路是当视频相比其他时钟慢时则丢帧加快播放，如果视频较快则等待。
```cpp
//省略部分代码
/* called to display each frame */
//如果当前帧的serial和上一帧的serial不同则更新frame_timer
if (lastvp->serial != vp->serial)
    is->frame_timer = av_gettime_relative() / 1000000.0;

/* compute nominal last_duration */
//计算上一帧持续的时间
last_duration = vp_duration(is, lastvp, vp);
//计算上一帧真正的持续时间
delay = compute_target_delay(last_duration, is);
//获取系统时刻
time= av_gettime_relative()/1000000.0;
//如果系统时刻小于当前播放序列的frame_timer+delay。即上一帧并未显示够时长，则更新remaing_time直接显示当前帧
if (time < is->frame_timer + delay) {
    *remaining_time = FFMIN(is->frame_timer + delay - time, *remaining_time);
    goto display;
}

//上一帧已经结束，更新当前帧的frame_timer
is->frame_timer += delay;
//如果系统的时钟和当前帧的时钟偏差过大则更新frame_timer,AV_SYNC_THRESHOLD_MAX为0.1
if (delay > 0 && time - is->frame_timer > AV_SYNC_THRESHOLD_MAX)
    is->frame_timer = time;
//省略部分代码
if (frame_queue_nb_remaining(&is->pictq) > 1) {
    //丢帧的逻辑
    Frame *nextvp = frame_queue_peek_next(&is->pictq);
    //计算当前帧的显示时长
    duration = vp_duration(is, vp, nextvp);
    //这里的关键起始就这一句time > is->frame_timer + duration，当前的系统时刻大于需要显示的时间点则丢帧
    if(!is->step && 
       (framedrop>0 || (framedrop && get_master_sync_type(is) != AV_SYNC_VIDEO_MASTER)) && time > is->frame_timer + duration){
        is->frame_drops_late++;
        frame_queue_next(&is->pictq);
        goto retry;
    }
}
//省略部分代码
```

&emsp;&emsp;下面是会计算当前帧的显示时长，其实就是两帧数据的pts的差值。
```cpp
static double vp_duration(VideoState *is, Frame *vp, Frame *nextvp) {
    if (vp->serial == nextvp->serial) {
        double duration = nextvp->pts - vp->pts;
        if (isnan(duration) || duration <= 0 || duration > is->max_frame_duration)
            return vp->duration;
        else
            return duration;
    } else {
        return 0.0;
    }
}
```

&emsp;&emsp;下面是计算delay的具体实现。需要注意的是同步是只视频和其他master的差值在一个区间内即可，而不是完全相同，当超过该区间时才会进行同步。并且不同的超过的处理方式不同，从代码中能够看到如果大于```sync_threshold```且大于```AV_SYNC_FRAMEDUP_THRESHOLD```返回的是```delay + diff```，而```diff >= sync_threshold```返回的是2 * delay```这样做的原因是防止返回的delay过小或者过大。

```cpp
static double compute_target_delay(double delay, VideoState *is)
{
    double sync_threshold, diff = 0;

    /* update delay to follow master synchronisation source */
    if (get_master_sync_type(is) != AV_SYNC_VIDEO_MASTER) {
        /* if video is slave, we try to correct big delays by duplicating or deleting a frame */
        diff = get_clock(&is->vidclk) - get_master_clock(is);
        /* skip or repeat frame. We take into account the delay to compute the threshold. I still don't know if it is the best guess */
        /* no AV sync correction is done if below the minimum AV sync threshold */
        //#define AV_SYNC_THRESHOLD_MIN 0.04
        /* AV sync correction is done if above the maximum AV sync threshold */
        //#define AV_SYNC_THRESHOLD_MAX 0.1
        sync_threshold = FFMAX(AV_SYNC_THRESHOLD_MIN, FFMIN(AV_SYNC_THRESHOLD_MAX, delay));
        //max_frame_duration是在打开媒体文件时进行计算的
        //is->max_frame_duration = (ic->iformat->flags & AVFMT_TS_DISCONT) ? 10.0 : 3600.0;
        if (!isnan(diff) && fabs(diff) < is->max_frame_duration) {
            if (diff <= -sync_threshold)
                //视频相比master延后，需要丢帧
                delay = FFMAX(0, delay + diff);
            else if (diff >= sync_threshold && delay > AV_SYNC_FRAMEDUP_THRESHOLD)
                //视频比master超前且大于AV_SYNC_FRAMEDUP_THRESHOLD（0.1）需要等待
                delay = delay + diff;
            else if (diff >= sync_threshold)
                //视频比master超前且小于AV_SYNC_FRAMEDUP_THRESHOLD（0.1）需要等待
                delay = 2 * delay;
        }
    }

    av_log(NULL, AV_LOG_TRACE, "video: delay=%0.3f A-V=%f\n",
            delay, -diff);

    return delay;
}
```
#### 3.4.2 音频同步视频
&emsp;&emsp;音频同步视频的主要代码在```audio_decode_frame```中，其基本过程比较简单。和视频的处理方式不同不是通过丢弃帧来实现而是通过拉长或者压缩给定的音频帧来实现同步。基本流程是计算期望的音频样本数量，如果当前帧的验本数量不等于则创建对应的采样context对音频进行采样。而关键就是获取到实际需要的sample的数量，该工作在```synchronize_audio```中完成。
```cpp
    //计算期望的sample数量
    wanted_nb_samples = synchronize_audio(is, af->frame->nb_samples);
    //如果target的参数和源音频的参数不同则会对音频进行重采样，这里是创建context
    if (af->frame->format        != is->audio_src.fmt            ||
        dec_channel_layout       != is->audio_src.channel_layout ||
        af->frame->sample_rate   != is->audio_src.freq           ||
        (wanted_nb_samples       != af->frame->nb_samples && !is->swr_ctx)) {
        //省略创建音频swr_context的代码
    }

    //对音频进行重采样
    if (is->swr_ctx) {
        //省略计算帧大小的代码
        if (wanted_nb_samples != af->frame->nb_samples) {
            //设置对音频进行补偿的一些参数
            if (swr_set_compensation(is->swr_ctx, (wanted_nb_samples - af->frame->nb_samples) * is->audio_tgt.freq / af->frame->sample_rate,
                                        wanted_nb_samples * is->audio_tgt.freq / af->frame->sample_rate) < 0) {
                return -1;
            }
        }
        av_fast_malloc(&is->audio_buf1, &is->audio_buf1_size, out_size);
        if (!is->audio_buf1)
            return AVERROR(ENOMEM);
        //对音频捷星重采样
        len2 = swr_convert(is->swr_ctx, out, out_count, in, af->frame->nb_samples);
        if (len2 < 0) {
            av_log(NULL, AV_LOG_ERROR, "swr_convert() failed\n");
            return -1;
        }
        if (len2 == out_count) {
            av_log(NULL, AV_LOG_WARNING, "audio buffer is probably too small\n");
            if (swr_init(is->swr_ctx) < 0)
                swr_free(&is->swr_ctx);
        }
        is->audio_buf = is->audio_buf1;
        resampled_data_size = len2 * is->audio_tgt.channels * av_get_bytes_per_sample(is->audio_tgt.fmt);
    } else {
        is->audio_buf = af->frame->data[0];
        resampled_data_size = data_size;
    }

    //更新音频的clock
    audio_clock0 = is->audio_clock;
    /* update the audio clock with the pts */
    if (!isnan(af->pts))
        is->audio_clock = af->pts + (double) af->frame->nb_samples / af->frame->sample_rate;
    else
        is->audio_clock = NAN;
    is->audio_clock_serial = af->serial;

    return resampled_data_size;
```

&emsp;&emsp;```synchronize_audio```负责根据与video clock的差值计算出合适的目标样本数，通过样本数控制音频输出速度。


```cpp
//大概是0.794
 is->audio_diff_avg_coef  = exp(log(0.01) / AUDIO_DIFF_AVG_NB);
 is->audio_diff_avg_count = 0;
        /* since we do not have a precise anough audio FIFO fullness,
           we correct audio sync only if larger than this threshold */
is->audio_diff_threshold = (double)(is->audio_hw_buf_size) / is->audio_tgt.bytes_per_sec;
```

&emsp;&emsp;```synchronize_audio```中会通过累加的偏差值估算当前音频的偏差，然后更新期望的sample数量。并且每次```wanted_nb_samples - nb_samples```的差值不会太大（默认是10%）防止每次同步音频的变化过于明显，此次未同步完的下次继续同步。

```cpp
/* return the wanted number of samples to get better sync if sync_type is video
 * or external master clock */
static int synchronize_audio(VideoState *is, int nb_samples)
{
    int wanted_nb_samples = nb_samples;

    /* if not master, then we try to remove or add samples to correct the clock */
    if (get_master_sync_type(is) != AV_SYNC_AUDIO_MASTER) {
        double diff, avg_diff;
        int min_nb_samples, max_nb_samples;
        //计算音频clock和master clock的差值
        diff = get_clock(&is->audclk) - get_master_clock(is);
        //diff小于AV_NOSYNC_THRESHOLD则计算
        if (!isnan(diff) && fabs(diff) < AV_NOSYNC_THRESHOLD) {
            //计算差值的累加值，通过audio_diff_avg_coef作为系数来控制其他帧的占比，加重当前帧在累加值中的比重
            is->audio_diff_cum = diff + is->audio_diff_avg_coef * is->audio_diff_cum;
            //如果出现偏差超过了20帧才会计算新值，不然使用传入的nb_samples
            if (is->audio_diff_avg_count < AUDIO_DIFF_AVG_NB) {
                /* not enough measures to have a correct estimate */
                is->audio_diff_avg_count++;
            } else {
                /* estimate the A-V difference */
                //通过累加的偏差值估算平均偏差
                avg_diff = is->audio_diff_cum * (1.0 - is->audio_diff_avg_coef);
                //如果平均偏差大于阈值就更新samples
                if (fabs(avg_diff) >= is->audio_diff_threshold) {
                    wanted_nb_samples = nb_samples + (int)(diff * is->audio_src.freq);
                    //这里会对sample的数量进行限制，防止每次同步偏差太大导致太明显的音频变化
                    min_nb_samples = ((nb_samples * (100 - SAMPLE_CORRECTION_PERCENT_MAX) / 100));
                    max_nb_samples = ((nb_samples * (100 + SAMPLE_CORRECTION_PERCENT_MAX) / 100));
                    wanted_nb_samples = av_clip(wanted_nb_samples, min_nb_samples, max_nb_samples);
                }
                av_log(NULL, AV_LOG_TRACE, "diff=%f adiff=%f sample_diff=%d apts=%0.3f %f\n",
                        diff, avg_diff, wanted_nb_samples - nb_samples,
                        is->audio_clock, is->audio_diff_threshold);
            }
        } else {
            /* too big difference : may be initial PTS errors, so reset A-V filter */
            is->audio_diff_avg_count = 0;
            is->audio_diff_cum       = 0;
        }
    }

    return wanted_nb_samples;
}
```

#### 3.4.3 同步到外部时钟
&emsp;&emsp;同步到外部时钟的逻辑会复用前面提到的同步到音频和同步到视频两个策略，唯一不同的是此次两个都要进行同步。那关键就是主时钟如何保证自己的准时。
```cpp
static void update_video_pts(VideoState *is, double pts, int64_t pos, int serial) {
    /* update current video pts */
    set_clock(&is->vidclk, pts, serial);
    sync_clock_to_slave(&is->extclk, &is->vidclk);
}
```

&emsp;&emsp;ffplay的策略是假定每一次更新音频或者视频时，该媒体流的clock是准时的，就调用```sync_clock_to_slave```将slave的clock同步到master的clock。证明这个也很简单，就是典型的数学归纳的场景，当播放第一帧时，音频视频的时钟自然是准时的，此时将音频或者视频的时钟同步到外部时钟成立，当播放第k-1帧时，通过同步的时钟就是第k帧的时钟，此时将第k帧的时钟同步到外部时钟成立。结论成立。

```cpp
/* Let's assume the audio driver that is used by SDL has two periods. */
    //下面的代码在sdl_audio_callback中
    if (!isnan(is->audio_clock)) {
        set_clock_at(&is->audclk, is->audio_clock - (double)(2 * is->audio_hw_buf_size + is->audio_write_buf_size) / is->audio_tgt.bytes_per_sec, is->audio_clock_serial, audio_callback_time / 1000000.0);
        sync_clock_to_slave(&is->extclk, &is->audclk);
    }

    //下面的代码在video_refresh中
    if (!isnan(vp->pts))
        update_video_pts(is, vp->pts, vp->pos, vp->serial);
```
