# FFmpeg中的内存buffer

摘要：FFmpeg中大多数数据存储比如```AVFrame```,```AVPacket```都是通过```AVBufferRef```管理的，而承载数据的结构为```AVBuffer```。本文主要通过FFmpeg源码来分析下FFmpeg中```AVBuffer```相关的实现。
关键字：```AVBuffer```、```AVBufferPool```
## 1. ```AVBufferRef```
### 1.1 ```AVBuffer```结构定义
&emsp;&emsp;```AVBuffer```声明在```libavutil/buffer_internal.h```文件中，而相关的操作函数定义在```libavutil/buffer.c```中。先简单看下```AVBuffer```的结构：
```c
struct AVBuffer {
    uint8_t *data;          /**< data described by this buffer */
    size_t size;                /**< size of data in bytes */
    atomic_uint refcount;   //number of existing AVBufferRef instances referring to this **buffer**
    void (*free)(void *opaque, uint8_t *data);//a callback for freeing the data
    void *opaque;//an opaque pointer, to be used by the freeing callback
    int flags;//A combination of AV_BUFFER_FLAG_*
    int flags_internal;//A combination of BUFFER_FLAG_*
};
```
&emsp;&emsp;该结构比较简单，就是一个含有引用计数的数据类型：
- ```data```：buffer中的数据指针；
- ```size```：数据的大小，即```data```中数据的大小；
- ```refcount```：引用计数，无需多说，当引用计数为0时销毁对应的内存。该变量的操作是原子的，ffmpeg内部针对不同的编译期和平台实现了一套源自变量，具体就深入了，理解意思就行；
- ```free```：释放内存的函数指针，如果不指定的话会使用默认的函数指针```av_buffer_default_free```释放内存；
- ```opaque```：user-defined的指针，用户可以通过该指针将数据传递给```free```函数；
- ```flags```：目前只有一个值```AV_BUFFER_FLAG_READONLY```；
- ```flags_internal```：目前只有一个值```BUFFER_FLAG_REALLOCATABLE```；

### 1.2 ```AVBufferRef```结构定义
&emsp;&emsp;```AVBufferRef```可以看做```AVBuffer```的一个句柄，用来操作```AVBuffer```：
```c
typedef struct AVBufferRef {
    AVBuffer *buffer;
    /**
     * The data buffer. It is considered writable if and only if
     * this is the only reference to the buffer, in which case
     * av_buffer_is_writable() returns 1.
     */
    uint8_t *data;
    size_t   size;//Size of data in bytes.
} AVBufferRef;
```
&emsp;&emsp;```AVBufferRef```结构比较简单，不详细描述，主要注意```data```字段是指向其成员```buffer.data```的。

### 1.3 操作函数
- ```AVBufferRef *av_buffer_create(uint8_t *data, size_t size, void (*free)(void *opaque, uint8_t *data), void *opaque, int flags)```：该函数用来创建一个```AVBufferRef```，具体就是申请内存函数根据参数初始化各个成员。需要注意的是返回的指针和其成员```buffer```是在堆上的，以及```AVBuferRef::data == AVBufferRef::buffer::data```；
- ```AVBufferRef *av_buffer_alloc(size_t size)```：通过```av_buffer_create```创建对象，只不过参数都是默认值；
- ```AVBufferRef *av_buffer_allocz(size_t size)```：相比```av_buffer_alloc```只是对内存进行了0初始化；
- ```AVBufferRef *av_buffer_ref(AVBufferRef *buf)```:FFmpeg中以```_ref```结尾的API都是引用计数+1的含义，相反```_unref```就是引用计数-1。但是需要注意两点：
  - 这里不是单纯的引用计数+1，而是```malloc```了一个```AVBufferRef```作为返回值，然后浅拷贝输入参数；
  - 仅仅引用计数是原子的，类似```shared_ptr```，对象本身不线程安全；
- ```void av_buffer_unref(AVBufferRef **buf)```：引用计数-1，释放内存，调用```free```释放```data```内存；
- ```int av_buffer_is_writable(const AVBufferRef *buf)```：当```flags```设置了```AV_BUFFER_FLAG_READONLY```时始终不可写，否则只有引用计数为1时才可写；
- ```int av_buffer_make_writable(AVBufferRef **pbuf)```：实现就是```copy-on-write```，将```pbuf```复制一份避免写共享的内存影响其他对象；
- ```int av_buffer_realloc(AVBufferRef **pbuf, size_t size)```：重新申请内存，如果传入的```*pbuf```为空则```create```一份。当输入的对象不可写或者不是```BUFFER_FLAG_REALLOCATABLE```时会拷贝一份再```realloc```；
- ```int av_buffer_replace(AVBufferRef **pdst, AVBufferRef *src)```：可以简单的理解就是```*pds=*src```，当```pdst```和```src```指向同一个```buffer```时，什么也不会做，实现类似C++中对象的拷贝构造函数；

## 2. AVBufferRef
### 2.1 结构定义
&emsp;&emsp;```AVBufferPool```是一个单链表，用来管理其中的```AVBuffer```。
```c
typedef struct BufferPoolEntry {
    uint8_t *data;
    /*
     * Backups of the original opaque/free of the AVBuffer corresponding to
     * data. They will be used to free the buffer when the pool is freed.
     */
    void *opaque;
    void (*free)(void *opaque, uint8_t *data);
    AVBufferPool *pool;
    struct BufferPoolEntry *next;
} BufferPoolEntry;
```
&emsp;&emsp;从结构定义中可以看到```BufferPollEntry```就是链表中的节点用来管理对应的```AVBufferRef```。但是仔细看又发现其中并没有```AVBuffer```的指针节点，而是保存了```opaque```和```free```函数指针，因为有这两个值我们就可以很顺利的释放对应的```AVBuffer```，而pool中又保存了对应的allocate的函数指针能够创建对象。
- ```data```：指向```AVBuffer```的地址，因为没有保存```AVBuffer```的地址所以需要一个指针来指向数据；
- ```opaque```：实现中```BufferPoolEntry::opaque->AVBuffer::opaque->BufferPoolEntry```，这样能够保证通过```AVBuffer```调用释放函数时找到管理自己的handle；
- ```free```：释放函数指针，实际上是固定的```pool_release_buffer```；
- ```pool```：直接指向当前的内存池；
- ```next```：链表的节点指针；

```c
struct AVBufferPool {
    AVMutex mutex;
    BufferPoolEntry *pool;
    /*
     * This is used to track when the pool is to be freed.
     * The pointer to the pool itself held by the caller is considered to
     * be one reference. Each buffer requested by the caller increases refcount
     * by one, returning the buffer to the pool decreases it by one.
     * refcount reaches zero when the buffer has been uninited AND all the
     * buffers have been released, then it's safe to free the pool and all
     * the buffers in it.
     */
    atomic_uint refcount;

    size_t size;
    void *opaque;
    AVBufferRef* (*alloc)(size_t size);
    AVBufferRef* (*alloc2)(void *opaque, size_t size);
    void         (*pool_free)(void *opaque);
};
```
&emsp;&emsp;```AVBufferPool```就是内存池的管理对象：
- ```mutex```：线程安全用的锁；
- ```opaque```：```pool_free```函数指针的第一个参数；
- ```alloc```：默认会被设置成```av_buffer_alloc```；
- ```alloc2```：自定义的分配函数，申请```AVBufferRef```时优先使用，没有指定则使用```alloc```；
- ```pool_free```：释放内存池的回调；
- ```size```：单个对象的大小，即整个内存池管理的对象大小是相同的；
- ```refcount```：当前从内存池中分配但是并没有在内存池链表中的节点的引用计数之和。
 
### 2.2 接口实现
- ```AVBufferPool *av_buffer_pool_init2(size_t size, void *opaque, AVBufferRef* (*alloc)(void *opaque, size_t size), void (*pool_free)(void *opaque))```：初始化pool的链表，根据参数设置相应的成员，```alloc2```会设置输入的参数```alloc```，而- ```alloc```会设置成```av_buffer_alloc```；
- ```AVBufferPool *av_buffer_pool_init(size_t size, AVBufferRef* (*alloc)(size_t size))```：只会申请pool的内存设置相关参数，如果alloc为空则pool中的```alloc```设置为```av_buffer_alloc```；
- ```void av_buffer_pool_uninit(AVBufferPool **ppool)```：销毁pool，如果引用计数为1则销毁对象（不知道为什么命名没有类似```_unref```，可能因为没有```ref```吧）；
- ```AVBufferRef *av_buffer_pool_get(AVBufferPool *pool)```：获取一个```AVBufferRef```该内存是通过pool管理的。


### 2.3 内存管理
&emsp;&emsp;```AVBufferPool```是一个以单链表形式实现的栈式内存池。其基本过程就是如果链表非空则出栈头结点，否则申请内存时就创建一个```AVBUfferRef```返回给用户，用户释放时就会将节点入栈到头结点，并且申请和释放内存是线程安全的。```AVBufferPool```就是一个空闲链表栈，通过指定对应的```AVBufferRef```的释放函数为```pool_release_buffer```来对内存进行管理。
&emsp;&emsp;对于一个刚初始化的内存池，连续申请两个Buffer就是下面这种状态：
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/avbuffer_1.png)
&emsp;&emsp;连续申请3个buffer，再释放2个就是下面这种状态（红色为链表的连接线）：
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/avbuffer_2.png)