# UrlContext和URLProtocol

&emsp;&emsp;**摘要**：本文描述FFmpeg中URLContext和URLProtocal的实现。
&emsp;&emsp;**关键字**：URLContext、URLProtocal

&emsp;&emsp;FFmpeg中```URLProtocol```是具体的协议的抽象，其中定义了对应协议的抽象，其中包含了具体协议的操作函数指针。而```URLContext```是对协议操作的抽象，描述了当前协议的读写状态。和其他结构体一样，FFmpeg内部针对每一个协议都有一个static的结构体，该结构体描述了对应协议的操作。
&emsp;&emsp;另外FFmpeg中有个list保存了所有```URLProtocol```的指针，类似```AVCodec```都是定义好的静态变量。
&emsp;&emsp;该list存储在```libavformat\protocol_list.c```文件中，需要注意的是该文件是通过脚本生成的，如果新拉的代码应该看不到，configure再make之后就能看到改文件了。

```c
static const URLProtocol * const url_protocols[] = {
    &ff_async_protocol,
    //...
    &ff_file_protocol,
    &ff_ftp_protocol,
    //...
    &ff_unix_protocol,
    NULL 
};
```
## 1 ```URLContext```
### 1.1 ```URLContext```
&emsp;&emsp;```URLContext```是对IO操作的抽象，类似以```AVCodecContext```，其中了当前媒体操作包含的基本信息 ，描述了当前IO操作的参数。使用过程中，```URLContext```作为AVIO的一个成员用来操作文件流。
```c
typedef struct URLContext {
    const AVClass *av_class;    /**< information for av_log(). Set by url_open(). */
    const struct URLProtocol *prot;
    void *priv_data;
    char *filename;             /**< specified URL */
    int flags;
    int max_packet_size;        /**< if non zero, the stream is packetized with this max packet size */
    int is_streamed;            /**< true if streamed (no seek possible), default = false */
    int is_connected;           //是否连接，网络流
    AVIOInterruptCB interrupt_callback;     //io终止时的callback
    int64_t rw_timeout;         /**< maximum time to wait for (network) read/write operation completion, in mcs */
    const char *protocol_whitelist;     //白名单
    const char *protocol_blacklist;     //黑名单
    int min_packet_size;        /**< if non zero, the stream is packetized with this min packet size */
} URLContext;
```

### 1.2 操作API概要
&emsp;&emsp;下面简单列举一些操作API来说明```URLContext```如何在FFmpeg中使用：
- ```int ffurl_alloc(URLContext **puc, const char *filename, int flags, const AVIOInterruptCB *int_cb)```：根据当前的文件名创建```URLContext```，第一步在```url_protocols```中根据文件名搜索对应的```URLProtocol```，然后初始化```URLContext```的默认参数，不会做额外多余的动作，简单说就是search->malloc->init（仅仅参数）；
- ```int ffurl_open_whitelist(URLContext **puc, const char *filename, int flags,
               const AVIOInterruptCB *int_cb, AVDictionary **options,
               const char *whitelist, const char* blacklist,
               URLContext *parent);
- ```int ffurl_connect(URLContext *uc, AVDictionary **options);```：打开链接，内部主要就是调用```URLProtocol```的```url_open2```函数打开链接然后将```is_connected```设置为1，根据流的类型设置```is_stream```；
- ```int ffurl_accept(URLContext *s, URLContext **c);```：accept链接，直接调用的协议的```url_accept```，一般只会用于网络请求，对于普通的本地IO基本没用；
- ```int ffurl_write(URLContext *h, const unsigned char *buf, int size);```：调用协议的```url_write```写文件。

&emsp;&emsp;从上面的流程大概能够看出```URLContext```大部分操作都是直接调用的```URLProtocol```的函数指针，额外做一些参数检查与匹配。

## 2 ```URLProtocal```
&emsp;&emsp;```URLProtocal```是具体的IO的描述类，类似于具体的```AVCodec```指针，每个类型的IO操作都有其对应的静态对象指针。```URLProtocal```中用函数指针来表示当前文件操作需要调用的具体操作。
```c
typedef struct URLProtocol {
    const char *name;           //协议名，比如http
    int     (*url_open)( URLContext *h, const char *url, int flags);
    /**
     * 下面的函数就是协议连接的api
     * This callback is to be used by protocols which open further nested
     * protocols. options are then to be passed to ffurl_open_whitelist()
     * or ffurl_connect() for those nested protocols.
     */
    int     (*url_open2)(URLContext *h, const char *url, int flags, AVDictionary **options);
    int     (*url_accept)(URLContext *s, URLContext **c);
    int     (*url_handshake)(URLContext *c);

    /**
     * 下面就是直接读写数据操作的接口
     * Read data from the protocol.
     * If data is immediately available (even less than size), EOF is
     * reached or an error occurs (including EINTR), return immediately.
     * Otherwise:
     * In non-blocking mode, return AVERROR(EAGAIN) immediately.
     * In blocking mode, wait for data/EOF/error with a short timeout (0.1s),
     * and return AVERROR(EAGAIN) on timeout.
     * Checking interrupt_callback, looping on EINTR and EAGAIN and until
     * enough data has been read is left to the calling function; see
     * retry_transfer_wrapper in avio.c.
     */
    int     (*url_read)( URLContext *h, unsigned char *buf, int size);          //读数据
    int     (*url_write)(URLContext *h, const unsigned char *buf, int size);    //写数据
    int64_t (*url_seek)( URLContext *h, int64_t pos, int whence);               //seek
    int     (*url_close)(URLContext *h);                                        //关闭
    int (*url_read_pause)(URLContext *h, int pause);                            //暂停读，网络流
    int64_t (*url_read_seek)(URLContext *h, int stream_index, int64_t timestamp, int flags);
    int (*url_get_file_handle)(URLContext *h);                                  //获取文件的handle
    int (*url_get_multi_file_handle)(URLContext *h, int **handles, int *numhandles);
    int (*url_get_short_seek)(URLContext *h);                                   //
    int (*url_shutdown)(URLContext *h, int flags);                              //关闭
    const AVClass *priv_data_class;                                             //私有数据
    int priv_data_size;                                                         //私有数据的大小
    int flags;                                                                  //标志
    int (*url_check)(URLContext *h, int mask);
    int (*url_open_dir)(URLContext *h);
    int (*url_read_dir)(URLContext *h, AVIODirEntry **next);
    int (*url_close_dir)(URLContext *h);
    int (*url_delete)(URLContext *h);
    int (*url_move)(URLContext *h_src, URLContext *h_dst);
    const char *default_whitelist;
} URLProtocol;
```

&emsp;&emsp;FFmpeg中每个IO协议都有一个对应的```URLProtocol```描述该协议IO的具体操作，比如文件IO就定义在```libavformat/file.c```中，最终实际调用的都是文件IO那套接口。
```c
const URLProtocol ff_file_protocol = {
    .name                = "file",
    .url_open            = file_open,
    .url_read            = file_read,
    .url_write           = file_write,
    .url_seek            = file_seek,
    .url_close           = file_close,
    .url_get_file_handle = file_get_handle,
    .url_check           = file_check,
    .url_delete          = file_delete,
    .url_move            = file_move,
    .priv_data_size      = sizeof(FileContext),
    .priv_data_class     = &file_class,
    .url_open_dir        = file_open_dir,
    .url_read_dir        = file_read_dir,
    .url_close_dir       = file_close_dir,
    .default_whitelist   = "file,crypto,data"
};
```
