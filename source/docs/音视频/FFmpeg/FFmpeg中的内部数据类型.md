# FFmpeg中的内部数据类型

&emsp;&emsp;摘要：简要描述FFmpeg内部定义的数据类型和相关操作的API及其实现。
&emsp;&emsp;关键字：AVDictionary,
## 1 ```AVDictionary```及相关API
&emsp;&emsp;```AVDictionary```虽然名字里是有字典的单词，但是只是一个单纯的键值对数组。
```c
typedef struct AVDictionaryEntry {
    char *key;
    char *value;
} AVDictionaryEntry;

struct AVDictionary {
    int count;
    AVDictionaryEntry *elems;
};

typedef struct AVDictionary AVDictionary;
```

&emsp;&emsp;相关API实现：
- **```int av_dict_count(const AVDictionary *m)```**：实际上就是返回```AVDictionary->count```；
- **```AVDictionaryEntry *av_dict_get(const AVDictionary *m, const char *key, const AVDictionaryEntry *prev, int flags)```**：从```prev``开始搜索下一个符合搜索要求的项，具体实现中是通过字符串一个一个比较实现的：
	- ```m,key```：搜索的对象和目标key；
	- ```prev```：开始搜索的位置；
	- ```flags```：搜索的flag，如果设置```AV_DICT_MATCH_CASE```则忽略大小写。如果设置```AV_DICT_IGNORE_SUFFIX```则忽略已经匹配好的字符串的后缀，比如```format1```能够匹配上```format```；
- **```int av_dict_set(AVDictionary * *pm, const char *key, const char *value, int flags)```**：这个api准确的描述应该是```av_dict_set_or_add```也就是说能找到则尝试更改，否则添加新的项：
	- ```flags```：配置对```key```和```value```的内存管理，默认会复制对应的字符串，需要注意的是如果设置了类似```AV_DICT_DONT_STRDUP_KEY```也就是将外部申请的字符串的所有权交给了```pm```，内部在释放时会free的；
- **```int av_dict_set_int(AVDictionary * *pm, const char *key, int64_t value, int flags)```**：实际上就是将```value转换成string然后通过```av_dict_set```设置；
- **```int av_dict_parse_string(AVDictionary * *pm, const char *str, const char *key_val_sep, const char *pairs_sep, int flags)```**：根据字符串```str```解析键值对创建```AVDictoionary```：
	- ```str```：一个以```key_val_sep```作为键值对内的分隔符，```pairs_sep```作为键值对间分隔符的字符串，比如```rate=1000;fmt=1;width=100```；
	- ```key_val_sep```：键值对内部的分隔符，对于上面字符串即```=```；
	- ```pairs_sep```：键值对间的分隔符，对于上面的字符串为```;```；
	- ```flags```：控制生成字典的标志符；
- **```void av_dict_free(AVDictionary * *pm)```**：释放字典，是通过FFmpeg内的释放函数释放的也就是说申请的内存最好也是通过FFmpeg内部的函数进行处理不然可能出现无法对齐而释放不正确的问题；
- **```int av_dict_copy(AVDictionary * *dst, const AVDictionary *src, int flags)```**：不断```av_dict_get```然后```av_dict_set```：
- **```int av_dict_get_string(const AVDictionary *m, char * *buffer, const char key_val_sep, const char pairs_sep)```**：将字典解析为字符串，和```av_dict_parse_string```互为逆操作；
- **```int avpriv_dict_set_timestamp(AVDictionary * *dict, const char *key, int64_t timestamp)```**：将```timestamp```格式化为字符串设置。

## 2 ```AVRational```及相关API
&emsp;&emsp;FFmpeg中由于部分浮点小数运算是高精度的，如果采用```float,double```来存储的话多次运算之后无法避免的会产生精度误差从而影响输出的结果。因此FFmpeg采用小数的分数形式来表示对应的小数，直到最后需要使用```float,double```的时候再将其转换成对应的形式能尽可能的降低精度损失。下面是FFmpeg中小数表示的定义形式：
```c
typedef struct AVRational{
    int num; ///< Numerator
    int den; ///< Denominator
} AVRational;
```

&emsp;&emsp;相关API操作：
- **```int av_reduce(int *dst_num, int *dst_den, int64_t num, int64_t den, int64_t max)```**：将输入的``AVRational```的分子分母限制在max，即输出的```dst_num,dst_den```的值都小于```max```，一般用来化简，如果```max```太小肯定有精度损失，一般用设置为```INT_MAX```之类即可；
- **```AVRational av_mul_q(AVRational b, AVRational c)```**：$\frac{b_n*c_n}{b_d*c_d}$；
- **```AVRational av_div_q(AVRational b, AVRational c)```**：$\frac{b_n*c_d}{b_d*c_n}$，利用```av_mul_q```实现；
- **```AVRational av_add_q(AVRational b, AVRational c)```**：$\frac{b_n*c_d+c_n*b_d}{c_d*c_d}$；
- **```AVRational av_sub_q(AVRational b, AVRational c)```**：$\frac{b_n*c_d-c_n*b_d}{c_d*c_d}$；
- **```AVRational av_d2q(double d, int max)```**：将小数```d```转换成```AVRational```类型；
- **```int av_nearer_q(AVRational q, AVRational q1, AVRational q2)```**：比较```q```和```q1,q2```哪个更接近；
  - 0表示差值绝对值相同；
  - -1表示更接近```q2```；
  - 1表示更接近```q1```；
- **```int av_find_nearest_q_idx(AVRational q, const AVRational* q_list)```**：寻找列表中更接近```q```的有理数的索引，判定标准为```av_nearer_q(q, q_list[i], q_list[nearest_q_idx]) > 0```；
- **```uint32_t av_q2intfloat(AVRational q)```**：转为IEEE 32-bit float；
- **```AVRational av_gcd_q(AVRational a, AVRational b, int max_den, AVRational def)```**：返回的值分子分母分别为输入的两个有理数分子分母的最大公约数，如果分母大于```max_den```则返回的是```def```。

