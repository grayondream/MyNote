# AVOption

&emsp;&emsp;**摘要**：本文通过阅读FFmpeg源码来理解FFmpeg中AVOption的实现原理和具体的使用方式。
&emsp;&emsp;**关键字**：AVClss,AVOption

## 1 ```AVOption```结构

&emsp;&emsp;```AVOption```是FFmpeg中设置参数的一个基本抽象结构。因为FFmpeg是一个支持多种封装解封装器，编解码器的框架，而不同的外部库需要的参数各不相同，因此利用```AVOption```来封装一个基本的key-value结构来获取和设置对应模块的参数。```AVOption```本身就是一个key-value项，可以理解为C++中```map```中的项```pair```，而其中```name```就是```key```，```default_val```就是```value```。而在实际使用中所有的参数是存储在```AVClass```中的```AVOption```数组中，而需要设置参数的模块会在Context结构体开头设置一个```AVClass```的指针来表示当前模块的参数，FFmpeg通过搜索该数组来获取和设置对应模块的参数。另外，并不是所有的参数都会存储在```AVOption```中，有些参数对应的Context本身就有对应的成员，因此```AVOption```只需要指定该成员相对应该Context首部地址的偏移即可进行设置和读取，也就是说通过```AVOption```设置和直接用```Context->param = value```的形式是一致的。

```c
typedef struct AVOption {
    const char *name;           //参数名称
    const char *help;           //简短的帮助信息，因为是char因此无法支持中文等语言
    int offset;                 //选项相对于结构体首部的偏移量
    enum AVOptionType type;     //选项的类型
    union {
        int64_t i64;
        double dbl;
        const char *str;
        AVRational q;
    } default_val;              //选项的默认值
    double min;                 //选项可允许的最小值
    double max;                 //选项可允许的最大值
    int flags;                  //标志符
    const char *unit;           //当前选项所属的逻辑单元，可以为NULL
} AVOption;
```

- ```default_value```：当前选项的值是一个64bit的```union```，这基本能够表示所有的数值参数和字符串参数；

- ```type```：当前选项的类型，如下面的```AVOptionType```的定义，不仅仅输数据类型也涵盖了一些FFmpeg中用到的选项类型；
  
  ```c
  enum AVOptionType{
    AV_OPT_TYPE_FLAGS,
    AV_OPT_TYPE_INT,
    AV_OPT_TYPE_INT64,
    AV_OPT_TYPE_DOUBLE,
    AV_OPT_TYPE_FLOAT,
    AV_OPT_TYPE_STRING,
    AV_OPT_TYPE_RATIONAL,
    AV_OPT_TYPE_BINARY,  ///< offset must point to a pointer immediately followed by an int for the length
    AV_OPT_TYPE_DICT,
    AV_OPT_TYPE_UINT64,
    AV_OPT_TYPE_CONST,
    AV_OPT_TYPE_IMAGE_SIZE, ///< offset must point to two consecutive integers
    AV_OPT_TYPE_PIXEL_FMT,
    AV_OPT_TYPE_SAMPLE_FMT,
    AV_OPT_TYPE_VIDEO_RATE, ///< offset must point to AVRational
    AV_OPT_TYPE_DURATION,
    AV_OPT_TYPE_COLOR,
    AV_OPT_TYPE_CHANNEL_LAYOUT,
    AV_OPT_TYPE_BOOL,
  };
  ```

- ```offset```：如上面描述，这个```offset```是相对于包含```AVClass```的Context的首地址的偏移；

- ```min,max```：选项取值范围，需要注意的是这里声明的是double；

- ```flags```：```flag```描述当前选项的用处，如下定义；
  
  ```c
  #define AV_OPT_FLAG_ENCODING_PARAM  1   ///< a generic parameter which can be set by the user for muxing or encoding
  #define AV_OPT_FLAG_DECODING_PARAM  2   ///< a generic parameter which can be set by the user for demuxing or decoding
  #define AV_OPT_FLAG_AUDIO_PARAM     8
  #define AV_OPT_FLAG_VIDEO_PARAM     16
  #define AV_OPT_FLAG_SUBTITLE_PARAM  32
  //The option is intended for exporting values to the caller.
  #define AV_OPT_FLAG_EXPORT          64
  //The option may not be set through the AVOptions API, only read. This flag only makes sense when AV_OPT_FLAG_EXPORT is also set.
  #define AV_OPT_FLAG_READONLY        128
  #define AV_OPT_FLAG_BSF_PARAM       (1<<8) ///< a generic parameter which can be set by the user for bit stream filtering
  #define AV_OPT_FLAG_RUNTIME_PARAM   (1<<15) ///< a generic parameter which can be set by the user at runtime
  #define AV_OPT_FLAG_FILTERING_PARAM (1<<16) ///< a generic parameter which can be set by the user for filtering
  #define AV_OPT_FLAG_DEPRECATED      (1<<17) ///< set if option is deprecated, users should refer to AVOption.help text for more information
  #define AV_OPT_FLAG_CHILD_CONSTS    (1<<18) ///< set if option constants can also reside in child objects
  ```

## 2 ```AVClass```结构

&emsp;&emsp;由于设置```AVOption```都是通过```AVClass```进行的，因此有必要了解下```AVClass```的结构。```AVClass```是FFmpeg中描述一个```Context```的抽象，FFmpeg中的```Context```的第一个成员就是一个```AVClass```的指针来描述当前```Context```的基本信息和相关的参数，设置参数也是通过这个指针进行的。比如```AVFormatContext```：

```c
typedef struct AVFormatContext {
    //A class for logging and @ref avoptions. Set by avformat_alloc_context(). Exports (de)muxer private options if they exist.
    const AVClass *av_class;
    //省略大部分代码
}AVFormatContext;
```

```c
typedef struct AVClass {
    const char* class_name;                            //AVClass所属类的名称
    const char* (*item_name)(void* ctx);               //获取AVClass所属类名称的函数指针，有些实现会直接返回AVClass->class_name
    const struct AVOption *option;                     //当前类的参数，没有就置为NULL
    int version;                                       //当前字段创建的版本，可用于版本控制，This is used to allow fields to be added without requiring major version bumps everywhere.
    int log_level_offset_offset;                       //AVClass所属结构体中log_level_offset相对于其首地址的偏移，0表示没有该成员
    int parent_log_context_offset;                     //当前Context中存储parent context的偏移量
    void* (*child_next)(void *obj, void *prev);        //AVOptions中下一个可用的参数
    AVClassCategory category;                          //当前类的类别
    AVClassCategory (*get_category)(void* ctx);        //获取当前Context类别的函数指针      
    //查询对应选项的范围，虽然定义了但是FFmpeg源码中好像没有API用到
    int (*query_ranges)(struct AVOptionRanges **, void *obj, const char *key, int flags);
    /**
     * Iterate over the AVClasses corresponding to potential AVOptions-enabled children.
     * @param iter pointer to opaque iteration state. The caller must initialize *iter to NULL before the first call.
     * @return AVClass for the next AVOptions-enabled child or NULL if there are no more such children.
     * @note The difference between child_next and this is that child_next iterates over _already existing_ objects, while child_class_iterate iterates over _all possible_ children.
    const struct AVClass* (*child_class_iterate)(void **iter);
} AVClass;
```
&emsp;&emsp;```AVClassCategory```定义如下：
```c
typedef enum {
    AV_CLASS_CATEGORY_NA = 0,
    AV_CLASS_CATEGORY_INPUT,
    AV_CLASS_CATEGORY_OUTPUT,
    AV_CLASS_CATEGORY_MUXER,
    AV_CLASS_CATEGORY_DEMUXER,
    AV_CLASS_CATEGORY_ENCODER,
    AV_CLASS_CATEGORY_DECODER,
    AV_CLASS_CATEGORY_FILTER,
    AV_CLASS_CATEGORY_BITSTREAM_FILTER,
    AV_CLASS_CATEGORY_SWSCALER,
    AV_CLASS_CATEGORY_SWRESAMPLER,
    AV_CLASS_CATEGORY_DEVICE_VIDEO_OUTPUT = 40,
    AV_CLASS_CATEGORY_DEVICE_VIDEO_INPUT,
    AV_CLASS_CATEGORY_DEVICE_AUDIO_OUTPUT,
    AV_CLASS_CATEGORY_DEVICE_AUDIO_INPUT,
    AV_CLASS_CATEGORY_DEVICE_OUTPUT,
    AV_CLASS_CATEGORY_DEVICE_INPUT,
    AV_CLASS_CATEGORY_NB  ///< not part of ABI/API
}AVClassCategory;
```
&emsp;&emsp;FFmpeg源码在```libavutil/tests/opt.c```中有示例程序```TestContext```，下面只截取其中一部分。从下面可以看到```TestContext```第一个成员便是```AVClass```的指针，然后下面定义了所有选项的数组并将该数组传递给```AVClass```的```option```成员。而其中选项不同类型初始化方式也不同有些是直接写值，有些是通过偏移写```TestContext```的成员变量的值：
```c
typedef struct TestContext {
    const AVClass *class;
    int num;
    int flags;
    AVRational rational;
    enum AVPixelFormat pix_fmt;
    char *escape;
} TestContext;

#define OFFSET(x) offsetof(TestContext, x)

#define TEST_FLAG_COOL 01
#define TEST_FLAG_LAME 02
#define TEST_FLAG_MU   04

static const AVOption test_options[]= {
    {"num",        "set num",            OFFSET(num),            AV_OPT_TYPE_INT,            { .i64 = 0 },                      0,       100, 1 },
    {"rational",   "set rational",       OFFSET(rational),       AV_OPT_TYPE_RATIONAL,       { .dbl = 1 },                      0,        10, 1 },
    {"escape",     "set escape str",     OFFSET(escape),         AV_OPT_TYPE_STRING,         { .str = "\\=," },          CHAR_MIN,  CHAR_MAX, 1 },
    {"flags",      "set flags",          OFFSET(flags),          AV_OPT_TYPE_FLAGS,          { .i64 = 1 },                      0,   INT_MAX, 1, "flags" },
    {"cool",       "set cool flag",      0,                      AV_OPT_TYPE_CONST,          { .i64 = TEST_FLAG_COOL },   INT_MIN,   INT_MAX, 1, "flags" },
    {"pix_fmt",    "set pixfmt",         OFFSET(pix_fmt),        AV_OPT_TYPE_PIXEL_FMT,      { .i64 = AV_PIX_FMT_0BGR },       -1,   INT_MAX, 1 },
};

static const char *test_get_name(void *ctx)
{
    return "test";
}

static const AVClass test_class = {
    .class_name = "TestContext",
    .item_name  = test_get_name,
    .option     = test_options,
};
```

## 3 ```AVOptionRange```及相关的API
**AVOption结构**
```c
typedef struct AVOptionRange {
    const char *str;						//字符串描述
    double value_min, value_max;			//对于字符串表示最大最小长度。对于尺寸，这表示多组件情况下的最小/最大像素数或宽度/高度。
    double component_min, component_max;	//对于字符串，这表示字符的 unicode 范围，0-127 限制为 ASCII。
    int is_range;							//是不是一个范围，0的话就是一个单纯的数值
} AVOptionRange;

//List of AVOptionRange structs.
typedef struct AVOptionRanges {
    AVOptionRange **range;					//二维数组，数组中元素的数目为nb_ranges * nb_components
    int nb_ranges;
    int nb_components;						//每个component有nb_ranges个AVOptionRange
} AVOptionRanges;
```

&emsp;&emsp;下面是FFmpeg代码中给出的示例：
```c
 /* AV_OPT_TYPE_IMAGE_SIZE:
  * component index 0: range of pixel count (width * height).
  * component index 1: range of width.
  * component index 2: range of height.
  */
int range_index, component_index;
AVOptionRanges *ranges;
AVOptionRange *range[3]; //may require more than 3 in the future.
av_opt_query_ranges(&ranges, obj, key, AV_OPT_MULTI_COMPONENT_RANGE);
for (range_index = 0; range_index < ranges->nb_ranges; range_index++) {
    for (component_index = 0; component_index < ranges->nb_components; component_index++)
        range[component_index] = ranges->range[ranges->nb_ranges * component_index + range_index];
    //do something with range here.
}
av_opt_freep_ranges(&ranges);
```

**相关API描述**：
- **```int av_opt_query_ranges_default(AVOptionRanges **, void *obj, const char *key, int flags)```**：获取对应的```key```的取值范围，实现中只有一个```AVOptionRange```项：
```c
    ranges->nb_ranges = 1;
    ranges->nb_components = 1;
    range->is_range = 1;
```
- **```int av_opt_query_ranges(AVOptionRanges **ranges_arg, void *obj, const char *key, int flags)```**：具体实现调用的是```obj```中的```query_ranges```，如果未定义则使用默认的```av_opt_query_ranges_default```；
- **```void av_opt_freep_ranges(AVOptionRanges **ranges)```**：释放```AVOptionRanges```；
- **

## 4 ```AVOption```相关API
&emsp;&emsp;```AVOption```相关API的第一个参数都需要是一个结构体指针，该结构的第一个成员必须是```AVClass```的指针。

- **```const AVOption *av_opt_next(const void *obj, const AVOption *prev)```**：返回相对于```prev```的下一个```AVOption```。实现比较简单既然有了```prev```且```AVOption```存储在数组中那么```prev```自增即可，只是多了一些边界检查；
	- ```obj```：第一个成员为```AVClass```指针的结构体；
	- ```prev```：需要寻找选项的上一个选项，如果为NULL则返回第一个；
	- 返回值：```prev```下一个选项；
```c
const AVOption *av_opt_next(const void *obj, const AVOption *last){
    const AVClass *class;
    if (!obj)
        return NULL;
    class = *(const AVClass**)obj;
    if (!last && class && class->option && class->option[0].name)
        return class->option;
    if (last && last[1].name)
        return ++last;
    return NULL;
}
```

- **```void *av_opt_child_next(void *obj, void *prev)```**：使用```AVClass```中的```child_next```返回下一个可用的参数选型；
```c
void *av_opt_child_next(void *obj, void *prev){
    const AVClass *c = *(AVClass **)obj;
    if (c->child_next)
        return c->child_next(obj, prev);
    return NULL;
}
```

- **```const AVClass *av_opt_child_class_iterate(const AVClass *parent, void * *iter)```**：搜寻子类的```AVClass```，c中没有子类的概念，这里的子类由```AVClass```中```child_class_iterate```定义，具体实现也是调用的```child_class_iterate```取搜索的；
	- ```iter```：存储搜索标志位的地址；
- **```const AVOption *av_opt_find2(void *obj, const char *name, const char *unit, int opt_flags, int search_flags, void * *target_obj)```**：在```obj```中搜索目标参数：
	- 搜索成功的条件：```(string(o->name) == string(name)) && ((o->flags & opt_flags) == opt_flags) && ((!unit && o->type != AV_OPT_TYPE_CONST) || (unit  && o->type == AV_OPT_TYPE_CONST && o->unit && string(o->unit) == string(unit))))```
	- ```obj```：应该是一个第一个成员指针为```AVClass```的结构体；
	- ```name```：希望搜索的参数的名称；
	- ```unit```：搜索的参数所属的逻辑单元；
	- ```opt_flags```：期望搜索的参数的标志位；
	- ```search_flags```：搜索的目标的标志位，如果设置了```AV_OPT_SEARCH_CHILDREN```则搜索的是由```child_class_iterate```定义的子类；
	- ```target_obj```：搜索的目标，如果未设置```AV_OPT_SEARCH_CHILDREN```则为NULL或者输入的```obj```，否则为对应的子类的指针或者NULL（是否为NULL根据```search_flags```是否设置了```AV_OPT_SEARCH_FAKE_OBJ```而定）；
	- **```const AVOption *av_opt_find(void *obj, const char *name, const char *unit, int opt_flags, int search_flags)```**：调用```av_opt_find2```实现，只不过无法搜索子类的选项；
```c
const AVOption *av_opt_find2(void *obj, const char *name, const char *unit, int opt_flags, int search_flags, void **target_obj){
	//下面是简化的代码
    c= *(AVClass**)obj;
    if (search_flags & AV_OPT_SEARCH_CHILDREN) {
        if (search_flags & AV_OPT_SEARCH_FAKE_OBJ) {
            void *iter = NULL;
            const AVClass *child;
            while (child = av_opt_child_class_iterate(c, &iter))
                if (o = av_opt_find2(&child, name, unit, opt_flags, search_flags, NULL))
                    return o;
        } else {
            void *child = NULL;
            while (child = av_opt_child_next(obj, child))
                if (o = av_opt_find2(child, name, unit, opt_flags, search_flags, target_obj))
                    return o;
        }
    }

    while (o = av_opt_next(obj, o)) {
		if(符合条件 && target_obj)
                if (!(search_flags & AV_OPT_SEARCH_FAKE_OBJ))
                    *target_obj = obj;
                else
                    *target_obj = NULL;
            }
            return o;
        }
    }
    return NULL;
}
```

- **```av_opt_set(void *obj, const char *name, const char *val, int search_flags)```**：在```obj```中搜索```name```选项并设置对应的参数，如果对应的参数是只读参数或者搜索失败会返回错误；
	- ```obj```：第一个成员为```AVClass```的结构体指针；
	- ```name```：希望设置参数的选项的key；
	- ```val```：key对应的值，以字符串的形式存储，具体的参数类型内部会根据```AVOption```的类型来进行转换；
	- ```search_flags```：搜索目标选项的标志位，内部搜索通过```av_opt_find2```实现；
```c
int av_opt_set(void *obj, const char *name, const char *val, int search_flags){
    int ret = 0;
    void *dst, *target_obj;
    const AVOption *o = av_opt_find2(obj, name, NULL, 0, search_flags, &target_obj);
	//然后将val根据类型转换成对应的值设置进o
}
```
- **```void av_opt_free(void *obj)```**：释放所有的```AVOption```，只有```AV_OPT_TYPE_STRING|AV_OPT_TYPE_BINARY|AV_OPT_TYPE_DICT```三中类型需要释放；
- **```int av_opt_copy(void *dst, const void *src)```**：拷贝一份；
- 其他设置参数的API，下面这些API基本都是设置对应的类型的具体实现，和```av_opt_set```的实现类似，先搜索再设置，只不过有些需要通过偏移量写入到```obj```的成员变量里，具体的作用从API的名称也能看出来不再赘述：
	- **```av_opt_set_defaults(void *s)```**：设置```s```中所有的```AVOption```值为默认值，```s```的以一个指针应该为```AVClass```；
	- **```av_opt_set_defaults2(void *s, int mask, int flags)```**：仅仅设置```(opt->flags & mask) == flags```的选项的默认值，```av_opt_set_defaults```也是调用```av_opt_set_defaults2```实现的；
	- **```av_set_options_string(void *ctx, const char *opts, const char *key_val_sep, const char *pairs_sep)```**：设置一系列参数；
		- ```ctx```：第一个成员为```AVClass```的结构体指针；
		- ```opts```：多个键值对，键值对之间以```paris_seq```为分隔符，键值对内以```key_val_sep```为分隔符，比如```width=30;height=40;bitrate=3000```；
		- ```key_val_sep```：分隔key-val的字符串，比如```key=value```中分隔符为```=```；
		- ```pairs_sep```：分隔一对键值对的字符串，比如```key1=val1;key2=val2```中分隔符为```;```；
	- **```int av_opt_set_from_string(void *ctx, const char *opts, const char *const *shorthand, const char *key_val_sep, const char *pairs_sep)```**：基本参数和定义和```av_set_options_string```类似，不同的是多了一个```shorthand```参数：
		- ```shorthand```：多个字符串的数组，如果```opts```中的前几个键值对未解析出key，则会使用```shorthand```中的值作为key。比如```opts="5:hello:size=pal",shorthand={ "num", "string", NULL };```则设置的结果是```num=5,string=hello```；
	- **```int av_opt_set_dict(void *obj, struct AVDictionary * *options)```**：遍历```options```内的值然后将值一一设置进obj；
	- **```int av_opt_set_dict2(void *obj, struct AVDictionary **options, int search_flags)```**：类似```av_opt_set_dict```，```search_flags```作用于```av_opt_set```；
	- **```int av_opt_set_int     (void *obj, const char *name, int64_t     val, int search_flags)```**
	- **```int av_opt_set_double  (void *obj, const char *name, double      val, int search_flags)```**
	- **```int av_opt_set_q       (void *obj, const char *name, AVRational  val, int search_flags)```**
	- **```int av_opt_set_bin     (void *obj, const char *name, const uint8_t *val, int size, int search_flags)```**
	- **```int av_opt_set_image_size(void *obj, const char *name, int w, int h, int search_flags)```**
	- **```int av_opt_set_pixel_fmt (void *obj, const char *name, enum AVPixelFormat fmt, int search_flags)```**
	- **```int av_opt_set_sample_fmt(void *obj, const char *name, enum AVSampleFormat fmt, int search_flags)```**
	- **```int av_opt_set_video_rate(void *obj, const char *name, AVRational val, int search_flags)```**
	- **```int av_opt_set_channel_layout(void *obj, const char *name, int64_t ch_layout, int search_flags)```**
	- **```int av_opt_set_dict_val(void *obj, const char *name, const AVDictionary *val, int search_flags)```**
- **```int av_opt_get         (void *obj, const char *name, int search_flags, uint8_t  * *out_val)```**：获取的实现就是通过```avav_opt_find2```搜索然后再将数据转成字符串传出；
- 其他获取的API：
	- **```int av_opt_flag_is_set(void *obj, const char *field_name, const char *flag_name)```**：获取某个flag是否设置，具体实现就是先搜索然后通过```av_opt_get_int```获取值；
		- ```obj```：第一个成员为```AVClass```的结构体指针；
		- ```field_name```：对应flag所属的标签；
		- ```flag_name```：具体的flag的key；
	- **```int av_opt_get_key_value(const char * *ropts, const char *key_val_sep, const char *pairs_sep, unsigned flags, char * *rkey, char * *rval)```**：从给定的字符串键值对中解析出第一个键值对并返回；
	- **```int av_opt_get_int     (void *obj, const char *name, int search_flags, int64_t    *out_val)```**
	- **```int av_opt_get_double  (void *obj, const char *name, int search_flags, double     *out_val)```**
	- **```int av_opt_get_q       (void *obj, const char *name, int search_flags, AVRational *out_val)```**
	- **```int av_opt_get_image_size(void *obj, const char *name, int search_flags, int *w_out, int *h_out)```**
	- **```int av_opt_get_pixel_fmt (void *obj, const char *name, int search_flags, enum AVPixelFormat *out_fmt)```**
	- **```int av_opt_get_sample_fmt(void *obj, const char *name, int search_flags, enum AVSampleFormat *out_fmt)```**
	- **```int av_opt_get_video_rate(void *obj, const char *name, int search_flags, AVRational *out_val)```**
	- **```int av_opt_get_channel_layout(void *obj, const char *name, int search_flags, int64_t *ch_layout)```**
	- **```int av_opt_get_dict_val(void *obj, const char *name, int search_flags, AVDictionary * *out_val)```**
	- **```void *av_opt_ptr(const AVClass *class, void *obj, const char *name)```**：返回```AVClass```所属Context中对应成员的地址，比如返回```AVCodecContext->width```的地址；
	- **```int av_opt_is_set_to_default_by_name(void *obj, const char *name, int search_flags)```**：通过```name```搜索然后通过```av_opt_is_set_to_default```获取；
	- **```int av_opt_serialize(void *obj, int opt_flags, int flags, char * *buffer, const char key_val_sep, const char pairs_sep)```**：将```obj```中的选项序列化为一个字符串，字符串中键值对间的间隔为```pairs_sep```，键值对内的间隔为```key_val_sep```；
- 其他API：
	- **```av_opt_show2(void *obj, void *av_log_obj, int req_flags, int rej_flags)```**：通过```av_log```将当前context支持的选项输出，默认输出到控制台，可以通过FFmpeg内的```av_log_callback```重定向输出：
		- ```obj```：是```AVClass```所属类的指针比如```AVFormatContext```；
		- ```av_log_obj```：一般情况置空，可以传入对应的Context的指针来显示更丰富的内容；
		- ```req_flags```和```rej_flags```**：用来过滤需要显示的内容，```req_flags```是显示需要显示的，```rej_flags```是不显示对应的内容。
	- 下面几个API在头文件中有定义但是看不到实现，从注释看和```av_opt_set```类似，只不过会将结果写会最后一个指针来校验是否写正确：
		- **```int av_opt_eval_flags (void *obj, const AVOption *o, const char *val, int        *flags_out)```**
		- **```av_opt_eval_int   (void *obj, const AVOption *o, const char *val, int        *int_out)```**
		- **```int av_opt_eval_int64 (void *obj, const AVOption *o, const char *val, int64_t    *int64_out)```**
		- **```int av_opt_eval_float (void *obj, const AVOption *o, const char *val, float      *float_out)```**
		- **```int av_opt_eval_double(void *obj, const AVOption *o, const char *val, double     *double_out)```**
		- **```int av_opt_eval_q     (void *obj, const AVOption *o, const char *val, AVRational *q_out)```**