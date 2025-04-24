# FFmpeg中的内存分配和释放

>&emsp;&emsp;参考的源码为FFmpeg5.0，commit-id为641c434。
>&emsp;&emsp;简单看一下FFmpeg中申请和释放内存的相关函数的具体实现。在解析源码的时候会移除部分内容方便阅读。

## 1 基本的内存分配和释放
&emsp;&emsp;FFmpeg中内存申请和分配的实现都是对```malloc```和```free```的包装，基本都在```libavutil/mem.c```文件中。

### 1.1 ```av_malloc```、```av_mallocz```和```av_calloc```
&emsp;&emsp;```av_malloc```：申请```size```大小的内存。
&emsp;&emsp;```av_mallocz```：申请```size```大小的内存并初始化为0。
&emsp;&emsp;```av_calloc```：申请n个元素大小的内存。
```c
void *av_malloc(size_t size){
    void *ptr = NULL;
    //检查传入的size是否大于最大值
    if (size > atomic_load_explicit(&max_alloc_size, memory_order_relaxed))
        return NULL;
    //内存对齐
#if HAVE_POSIX_MEMALIGN
    if (size) //OS X on SDK 10.6 has a broken posix_memalign implementation
    if (posix_memalign(&ptr, ALIGN, size))
        ptr = NULL;
#elif HAVE_ALIGNED_MALLOC
    ptr = _aligned_malloc(size, ALIGN);
#elif HAVE_MEMALIGN
#ifndef __DJGPP__
    ptr = memalign(ALIGN, size);
#else
    ptr = memalign(size, ALIGN);
#endif
#else
    ptr = malloc(size);
#endif
    if(!ptr && !size) {
        size = 1;
        ptr= av_malloc(1);
    }
#if CONFIG_MEMORY_POISONING
    if (ptr)
        memset(ptr, FF_MEMORY_POISON, size);
#endif
    return ptr;
}
```

&emsp;&emsp;上面是FFmpeg中的```av_malloc```的源码，能够看到首先会检查传入的```size```是否大于默认的最大值，如果大于则返回NULL。其中默认的最大值```max_alloc_size```是一个静态变量。```atomic_load_explicit```就是以原子方式加载并返回指向的原子变量的当前值。
```c
static atomic_size_t max_alloc_size = ATOMIC_VAR_INIT(INT_MAX);
```
&emsp;&emsp;之后会根据不同的配置以不同的内存对齐方式分配内存。内存对齐对访存友好、缓存友好、原子性等优点，具体可参考[purpose-of-memory-alignment](https://stackoverflow.com/questions/381244/purpose-of-memory-alignment)。```posix_memalign```、```_aligned_malloc```、```memalign```就是不同平台以不同的```ALIGN```的内存对齐方式分配内存。默认的```ALIGN```根据是否支持avx512分别为32和16。
```c
#define ALIGN (HAVE_AVX512 ? 64 : (HAVE_AVX ? 32 : 16))
    /* Why 64?
     * Indeed, we should align it:
     *   on  4 for 386
     *   on 16 for 486
     *   on 32 for 586, PPro - K6-III
     *   on 64 for K7 (maybe for P3 too).
     * Because L1 and L2 caches are aligned on those values.
     * But I don't want to code such logic here!
     */
    /* Why 32?
     * For AVX ASM. SSE / NEON needs only 16.
     * Why not larger? Because I did not see a difference in benchmarks ...
     */
    /* benchmarks with P3
     * memalign(64) + 1          3071, 3051, 3032
     * memalign(64) + 2          3051, 3032, 3041
     * memalign(64) + 4          2911, 2896, 2915
     * memalign(64) + 8          2545, 2554, 2550
     * memalign(64) + 16         2543, 2572, 2563
     * memalign(64) + 32         2546, 2545, 2571
     * memalign(64) + 64         2570, 2533, 2558
     *
     * BTW, malloc seems to do 8-byte alignment by default here.
     */
```
&emsp;&emsp;之后会检查插入的```size```为0时，返回的内存时不是真的是```NULL```，是的话就会重新调用```av_malloc```分配1byte的内存。下面是该处的commit message。
```
Only add 1 byte to av_malloc(0) when it actually returned NULL
```
&emsp;&emsp;最后会感觉是否设置了```CONFIG_MEMORY_POISONING```将申请到的内存设置为默认值```FF_MEMORY_POISON(0x2a)```。
&emsp;&emsp;从上面的实现可以看出，我们需要关心的就是内存对齐，即传入的```size```实际的内存可能比```size```大。以及传入```size```为0时，得到的内存不一定为NULL，可能为1byte的内存的指针。
```c
void *av_mallocz(size_t size){
    void *ptr = av_malloc(size);
    if (ptr)
        memset(ptr, 0, size);
    return ptr;
}
```
&emsp;&emsp;```av_mallocz```的实现就是调用了```av_malloc```只不过初始化为0，。
```c
void *av_calloc(size_t nmemb, size_t size){
    size_t result;
    if (size_mult(nmemb, size, &result) < 0)
        return NULL;
    return av_mallocz(result);
}
```
&emsp;&emsp;```av_calloc```实现比较简单不详述。
### 1.2 ```av_realloc```
&emsp;&emsp;```av_realloc```：类似```realloc```，相比而言做了边检检查、对齐和初始化工作。
```c
void *av_realloc(void *ptr, size_t size){
    void *ret;
    if (size > atomic_load_explicit(&max_alloc_size, memory_order_relaxed))
        return NULL;

#if HAVE_ALIGNED_MALLOC
    ret = _aligned_realloc(ptr, size + !size, ALIGN);
#else
    ret = realloc(ptr, size + !size);
#endif
#if CONFIG_MEMORY_POISONING
    if (ret && !ptr)
        memset(ret, FF_MEMORY_POISON, size);
#endif
    return ret;
}
```
&emsp;&emsp;```av_realloc```的实现和```av_malloc```相比基本流程类似：首先检查传入的```size```是否超出默认的值；然后根据是否需要对齐分别采用对应版本的api重新分配内存，这里有点儿不同的是```size + !size```保证传入的尺寸至少不会为0，因为```realloc```传入0等同于```free```；最后根据需要设置内存。

### 1.3  ```av_realloc_f```
&emsp;&emsp;```av_realloc_f```：realloc分配n个对应元素的大小的内存。
```c
void *av_realloc_f(void *ptr, size_t nelem, size_t elsize)
{
    size_t size;
    void *r;

    if (size_mult(elsize, nelem, &size)) {
        av_free(ptr);
        return NULL;
    }
    r = av_realloc(ptr, size);
    if (!r)
        av_free(ptr);
    return r;
}
```

&emsp;&emsp;具体时间调用```av_realloc```实现，区别是```av_realloc```返回的是指定大小内存的指针，而```av_realloc_f```返回n个大小为```elsize```的元素的内存指针。首先计算实际需要分配的内存大小；然后调用```av_realloc```分配内存；最后检查如果失败则调用```av_free```释放内存。释放的原因注释中提到了：
```c
 It frees the input block in case of failure, thus avoiding the memory leak with the classic
```

&emsp;&emsp;而```size_mult```就是用来计算实际的内存大小的。除了计算```a*b```外还做了越界检查。
```c
static int size_mult(size_t a, size_t b, size_t *r){
    size_t t;

#if (!defined(__INTEL_COMPILER) && AV_GCC_VERSION_AT_LEAST(5,1)) || AV_HAS_BUILTIN(__builtin_mul_overflow)
    if (__builtin_mul_overflow(a, b, &t))
        return AVERROR(EINVAL);
#else
    t = a * b;
    /* Hack inspired from glibc: don't try the division if nelem and elsize
     * are both less than sqrt(SIZE_MAX). */
    if ((a | b) >= ((size_t)1 << (sizeof(size_t) * 4)) && a && t / a != b)
        return AVERROR(EINVAL);
#endif
    *r = t;
    return 0;
}
```

### 1.4 ```av_reallocp```
&emsp;&emsp;```av_reallocp```：功能和```av_realloc```类似只是实现不同。
```c
int av_reallocp(void *ptr, size_t size){
    void *val;

    if (!size) {
        av_freep(ptr);
        return 0;
    }

    memcpy(&val, ptr, sizeof(val));
    val = av_realloc(val, size);

    if (!val) {
        av_freep(ptr);
        return AVERROR(ENOMEM);
    }

    memcpy(ptr, &val, sizeof(val));
    return 0;
}
```
&emsp;&emsp;在阅读```av_reallocp```的实现之前，首先需要明确的是```av_reallocp```接受的```p```是一个二级指针，即如果你希望申请内存的指针为```ptr```，传入的应该是```&ptr```。比如下面是FFmpeg中调用```av_reallocp```的例子：
```c
char *src = NULL;
err = av_reallocp(&src, len);
```
&emsp;&emsp;了解到这一点阅读就会轻松很多。首先会检查```size```是否为0，为0就释放内存，然后将```ptr```的值拷贝到临时变量```&val```中，此时```val```和```*ptr```指向同一块内存，然后通过```av_realloc```申请内存，最后检查是否成功以及将```&val```拷贝回```ptr```。
&emsp;&emsp;需要注意的是FFmpeg中的内存管理的api中后缀带p的都是类似的实现。
### 1.5 ```av_malloc_array```和```av_mallocz_array```
&emsp;&emsp;```av_malloc_array```：申请n个元素大小的内存；
&emsp;&emsp;```av_mallocz_array```：申请n个元素大小的内存并初始化。
```c
void *av_malloc_array(size_t nmemb, size_t size){
    size_t result;
    if (size_mult(nmemb, size, &result) < 0)
        return NULL;
    return av_malloc(result);
}

#if FF_API_AV_MALLOCZ_ARRAY
void *av_mallocz_array(size_t nmemb, size_t size){
    size_t result;
    if (size_mult(nmemb, size, &result) < 0)
        return NULL;
    return av_mallocz(result);
}
```
### 1.6 ```av_realloc_array```和```av_reallocp_array```
&emsp;&emsp;```av_realloc_array```：realloc n个元素大小的内存；
&emsp;&emsp;```av_reallocp_array```：realloc n个元素大小的内存；
```c
void *av_realloc_array(void *ptr, size_t nmemb, size_t size){
    size_t result;
    if (size_mult(nmemb, size, &result) < 0)
        return NULL;
    return av_realloc(ptr, result);
}

int av_reallocp_array(void *ptr, size_t nmemb, size_t size){
    void *val;

    memcpy(&val, ptr, sizeof(val));
    val = av_realloc_f(val, nmemb, size);
    memcpy(ptr, &val, sizeof(val));
    if (!val && nmemb && size)
        return AVERROR(ENOMEM);

    return 0;
}
```
&emsp;&emsp;二者的功能类似，只是实现不同，```av_reallocp_array```是使用上面```av_reallocp```类似的实现方式实现。
### 1.7 ```av_free```和```av_freep```
&emsp;&emsp;```av_free```：释放内存；
&emsp;&emsp;```av_freep```：释放内存并将指针置为NULL；
```c
void av_free(void *ptr){
#if HAVE_ALIGNED_MALLOC
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void av_freep(void *arg){
    void *val;

    memcpy(&val, arg, sizeof(val));
    memcpy(arg, &(void *){ NULL }, sizeof(val));
    av_free(val);
}
```
&emsp;&emsp;```av_free```实际调用的就是对齐版本的```free```和普通版本的```free```实现。而```av_freep```实现调用的是```av_free```，并将指针置为NULL。
### 1.8 ```av_memdup```、```av_strdup```和```av_strndup```
&emsp;&emsp;```av_strdup```：拷贝字符串；
&emsp;&emsp;```av_strndup```：拷贝字符串的len长度的值；
&emsp;&emsp;```av_memdup```：内存复制。
```c
char *av_strdup(const char *s){
    char *ptr = NULL;
    if (s) {
        size_t len = strlen(s) + 1;
        ptr = av_realloc(NULL, len);
        if (ptr)
            memcpy(ptr, s, len);
    }
    return ptr;
}

char *av_strndup(const char *s, size_t len){
    char *ret = NULL, *end;

    if (!s)
        return NULL;

    end = memchr(s, 0, len);
    if (end)
        len = end - s;

    ret = av_realloc(NULL, len + 1);
    if (!ret)
        return NULL;

    memcpy(ret, s, len);
    ret[len] = 0;
    return ret;
}

void *av_memdup(const void *p, size_t size){
    void *ptr = NULL;
    if (p) {
        ptr = av_malloc(size);
        if (ptr)
            memcpy(ptr, p, size);
    }
    return ptr;
}
```
&emsp;&emsp;```av_strdup```和```av_strndup```的实现比较简单，主要区别是```av_strndup```需要考虑末尾的```\0```的问题。```av_memdup```就是简单的申请然后拷贝操作。

### 1.9 ```av_dynarray_add_nofree```、```av_dynarray_add```和```av_dynarray2_add```
&emsp;&emsp;```av_dynarray_add```：在列表末尾添加元素，失败释放内存；
&emsp;&emsp;```av_dynarray_add_nofree```：在列表末尾添加元素，失败不释放内存；
&emsp;&emsp;```av_dynarray2_add```：在列表末尾添加元素，失败释放内存；
```c
void av_dynarray_add(void *tab_ptr, int *nb_ptr, void *elem){
    void **tab;
    memcpy(&tab, tab_ptr, sizeof(tab));

    FF_DYNARRAY_ADD(INT_MAX, sizeof(*tab), tab, *nb_ptr, {
        tab[*nb_ptr] = elem;
        memcpy(tab_ptr, &tab, sizeof(tab));
    }, {
        *nb_ptr = 0;
        av_freep(tab_ptr);
    });
}
```
&emsp;&emsp;```av_dynarray_add```的实现类似上面提到的带p的函数的实现，传入的```tab_ptr```实际上是一个三级指针此时的```(void**)*tabptr```和```tab```指向通一块内存。
```c
const char **pix_fmts = NULL;
av_dynarray_add(&pix_fmts, &nb_pix_fmts, (void *)pix_name);
```
&emsp;&emsp;而```FF_DYNARRAY_ADD```宏，比较难看懂，我们可以把该宏改写成函数的伪代码看下：
```c
typedef (*function)();
void ff_dynarray_add_macro(int av_size_max, int av_elt_size, void **av_array, int av_size, function av_success, function av_failure){
    do{
        size_t av_size_new = av_size;
        //如果av_size是2的n次幂则返回0
        if((av_size & (av_size - 1)) == 0){
            av_size_new = av_size ? av_size << 1 : 1;
            //判断是否超出最大容量
            if(av_size_new > (av_size_max / av_elt_size)){
                av_size_new = 0;
            }else{
                void *av_array_new = av_realloc(av_array, av_size_new * av_elt_size);
                if(av_array_new == NULL){
                    av_size_new = 0;
                }else{
                    av_array = av_array_new;
                }
            }
        }

        if(av_size_new != 0){
            av_sucess();
            av_size++;
        }else{
            av_failture();
        }
    }while(0);
}
```
&emsp;&emsp;改写后并不是标准的C语言，比如```av_array```以及```av_size```的修改不会生效，function仅仅是个占位符表示相关操作而已，把```ff_dynarray_add_macro```看成宏就可以。现在看的话其实```FF_DYNARRAY_ADD```就很简单，首先检查数组的元素数量是不是2的幂次方是的话则调用```av_realloc```重新分配2倍的内存，成功的话就调用sucess相关的操作这里是在末尾添加元素并更新大小，失败则调用fail相关操作，这里是释放内存。

```c
int av_dynarray_add_nofree(void *tab_ptr, int *nb_ptr, void *elem){
    void **tab;
    memcpy(&tab, tab_ptr, sizeof(tab));

    FF_DYNARRAY_ADD(INT_MAX, sizeof(*tab), tab, *nb_ptr, {
        tab[*nb_ptr] = elem;
        memcpy(tab_ptr, &tab, sizeof(tab));
    }, {
        return AVERROR(ENOMEM);
    });
    return 0;
}
```
&emsp;&emsp;```av_dynarray_add_nofree```和```av_dynarray_add```唯一的区别就是失败不释放内存。
```c
void *av_dynarray2_add(void **tab_ptr, int *nb_ptr, size_t elem_size,
                       const uint8_t *elem_data){
    uint8_t *tab_elem_data = NULL;

    FF_DYNARRAY_ADD(INT_MAX, elem_size, *tab_ptr, *nb_ptr, {
        tab_elem_data = (uint8_t *)*tab_ptr + (*nb_ptr) * elem_size;
        if (elem_data)
            memcpy(tab_elem_data, elem_data, elem_size);
        else if (CONFIG_MEMORY_POISONING)
            memset(tab_elem_data, FF_MEMORY_POISON, elem_size);
    }, {
        av_freep(tab_ptr);
        *nb_ptr = 0;
    });
    return tab_elem_data;
}
```
&emsp;&emsp;```av_dynarray2_add```的实现类似，只不过是通过内存拷贝的方式将数据拷贝到对应元素的内存处并返回对应节点的指针。

### 1.10 ```av_fast_realloc```、```av_fast_malloc```和```av_fast_mallocz```
&emsp;&emsp;```av_fast_realloc```：快速重新分配；
&emsp;&emsp;```av_fast_malloc```：快速分配内存;
&emsp;&emsp;```av_fast_mallocz```：快速分配内存并初始化为0；
```c
void *av_fast_realloc(void *ptr, unsigned int *size, size_t min_size){
    size_t max_size;

    if (min_size <= *size)
        return ptr;

    max_size = atomic_load_explicit(&max_alloc_size, memory_order_relaxed);

    if (min_size > max_size) {
        *size = 0;
        return NULL;
    }

    min_size = FFMIN(max_size, FFMAX(min_size + min_size / 16 + 32, min_size));

    ptr = av_realloc(ptr, min_size);
    /* we could set this to the unmodified min_size but this is safer
     * if the user lost the ptr and uses NULL now
     */
    if (!ptr)
        min_size = 0;

    *size = min_size;

    return ptr;
}

static inline void fast_malloc(void *ptr, unsigned int *size, size_t min_size, int zero_realloc){
    size_t max_size;
    void *val;

    memcpy(&val, ptr, sizeof(val));
    if (min_size <= *size) {
        av_assert0(val || !min_size);
        return;
    }

    max_size = atomic_load_explicit(&max_alloc_size, memory_order_relaxed);

    if (min_size > max_size) {
        av_freep(ptr);
        *size = 0;
        return;
    }
    min_size = FFMIN(max_size, FFMAX(min_size + min_size / 16 + 32, min_size));
    av_freep(ptr);
    val = zero_realloc ? av_mallocz(min_size) : av_malloc(min_size);
    memcpy(ptr, &val, sizeof(val));
    if (!val)
        min_size = 0;
    *size = min_size;
    return;
}

void av_fast_malloc(void *ptr, unsigned int *size, size_t min_size){
    fast_malloc(ptr, size, min_size, 0);
}

void av_fast_mallocz(void *ptr, unsigned int *size, size_t min_size){
    fast_malloc(ptr, size, min_size, 1);
}
```
&emsp;&emsp;```av_fast_malloc```、```av_fast_mallocz```和```av_fast_realloc```都是复用现有的内存，即如果需要的尺寸小于现在的尺寸则不分配，否则额外多分配一些内存返回。
```c
    if (min_size > max_size) {
        *size = 0;
        return NULL;
    }
```
### 1.11 ```av_memcpy_backptr```
&emsp;&emsp;```av_memcpy_backptr```：将```[dst - back, dst - 1]```的内容拷贝到```dst```不断重复直到填满```[dst, dst + cnt)```，具体看下图就很容易明白。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/av_memcpy_backptr.drawio.svg)

```c
void av_memcpy_backptr(uint8_t *dst, int back, int cnt){
    const uint8_t *src = &dst[-back];
    if (!back)
        return;

    if (back == 1) {
        memset(dst, *src, cnt);
    } else if (back == 2) {
        fill16(dst, cnt);
    } else if (back == 3) {
        fill24(dst, cnt);
    } else if (back == 4) {
        fill32(dst, cnt);
    } else {
        if (cnt >= 16) {
            int blocklen = back;
            while (cnt > blocklen) {
                memcpy(dst, src, blocklen);
                dst       += blocklen;
                cnt       -= blocklen;
                blocklen <<= 1;
            }
            memcpy(dst, src, cnt);
            return;
        }
        if (cnt >= 8) {
            AV_COPY32U(dst,     src);
            AV_COPY32U(dst + 4, src + 4);
            src += 8;
            dst += 8;
            cnt -= 8;
        }
        if (cnt >= 4) {
            AV_COPY32U(dst, src);
            src += 4;
            dst += 4;
            cnt -= 4;
        }
        if (cnt >= 2) {
            AV_COPY16U(dst, src);
            src += 2;
            dst += 2;
            cnt -= 2;
        }
        if (cnt)
            *dst = *src;
    }
}
```
&emsp;&emsp;上面的代码可以拆开一部分一部分看，```fill*```相关的函数分别是优化8, 16, 24, 32bit拷贝，其基本思路类似都是将对应的值在32bit中重复多次然后整个拷贝32byte的数据，下面逐个看具体的实现。
**8bit**
&emsp;&emsp;直接调用```memset```。
```c
  memset(dst, *src, cnt);
```
**16bit**
&emsp;&emsp;```AV_RN16```就是两个数分别存储在32bit数字的低16位中的高8和低8位中，简化就是```dst[-1] << 8 | dst[-2]```。然后将低16位的值重复到高16位，整个32bit中```[dst-2, dst-1]```重复了两次。下面第一个```while```循环每次按照4字节即32bit处理数据赋值，```AV_WN32```负责将32bit还原到8bit中，最后将多余的位设置为```dst[-2]```。
```c
static void fill16(uint8_t *dst, int len){
    uint32_t v = AV_RN16(dst - 2);

    v |= v << 16;

    while (len >= 4) {
        AV_WN32(dst, v);
        dst += 4;
        len -= 4;
    }

    while (len--) {
        *dst = dst[-2];
        dst++;
    }
}
```
**24bit**
&emsp;&emsp;24bit处理稍微麻烦点儿，现将```dst[-3],dst[-2],dst[-1]```分别放置到32bit内存的```[0:8],[8:16],[16:24]```。因为24bit数据对于32bit数据有余，因此使用3个32bit数据放置4个24bit数，即```a,b,c```96bit中```[dst-3, dst-1]```重复了4次。之后的```while```循环处理和上面的类似。
```c
static void fill24(uint8_t *dst, int len){
#if HAVE_BIGENDIAN
    uint32_t v = AV_RB24(dst - 3);
    uint32_t a = v << 8  | v >> 16;
    uint32_t b = v << 16 | v >> 8;
    uint32_t c = v << 24 | v;
#else
    uint32_t v = AV_RL24(dst - 3);
    uint32_t a = v       | v << 24;
    uint32_t b = v >> 8  | v << 16;
    uint32_t c = v >> 16 | v << 8;
#endif

    while (len >= 12) {
        AV_WN32(dst,     a);
        AV_WN32(dst + 4, b);
        AV_WN32(dst + 8, c);
        dst += 12;
        len -= 12;
    }

    if (len >= 4) {
        AV_WN32(dst, a);
        dst += 4;
        len -= 4;
    }

    if (len >= 4) {
        AV_WN32(dst, b);
        dst += 4;
        len -= 4;
    }

    while (len--) {
        *dst = dst[-3];
        dst++;
    }
}
```
**32bit**
&emsp;&emsp;32bit比较简单，因为一个32bit刚好能放下32bit，直接处理即可。
```c
static void fill32(uint8_t *dst, int len){
    uint32_t v = AV_RN32(dst - 4);

#if HAVE_FAST_64BIT
    uint64_t v2= v + ((uint64_t)v<<32);
    while (len >= 32) {
        AV_WN64(dst   , v2);
        AV_WN64(dst+ 8, v2);
        AV_WN64(dst+16, v2);
        AV_WN64(dst+24, v2);
        dst += 32;
        len -= 32;
    }
#endif

    while (len >= 4) {
        AV_WN32(dst, v);
        dst += 4;
        len -= 4;
    }

    while (len--) {
        *dst = dst[-4];
        dst++;
    }
}
```
&emsp;&emsp;下面的大于16字节的就直接每次拷贝16字节；大于8字小于16字节节每次拷贝4字节，拷贝两次；大于4字节小于8字节每次拷贝4字节；大于2字节每次拷贝2字节；小于2字节每次将```[dst - back]```赋值给当前的```[dst]```。
## 2 reference
- [内存对齐的目的](https://stackoverflow.com/questions/381244/purpose-of-memory-alignment)