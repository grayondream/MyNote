# Folly-FBString

&emsp;&emsp;**摘要：**工作已经有一段时间了，总的感觉实际工作中所使用到的C++相关的内容对实际提升C++没有太大帮助。基本都是搭积木做需求，前一段时间主要以看书为主，但是总感觉缺点什么。因此打算阅读一遍folly库的实现来提升自己的C++水平。
&emsp;&emsp;**关键字**：fbstring,string,fbstring_core

## 1 FBString简介
>&emsp;&emsp;fbstring is a drop-in replacement for std::string. The main benefit of fbstring is significantly increased performance on virtually all important primitives. This is achieved by using a three-tiered storage strategy and by cooperating with the memory allocator. In particular, fbstring is designed to detect use of jemalloc and cooperate with it to achieve significant improvements in speed and memory usage.

>&emsp;&emsp;fbstring supports 32- and 64-bit and little- and big-endian architectures.

&emsp;&emsp;FBString是facebook内部使用的基础库的string组件，为了达到更好的性能采用了多种存储策略，优化不同场景的性能。FBString完全兼容```std::string```，同时支持```jemalloc```更快的分配内存，减少磁盘碎片，加快并发情况下的速度和性能。

&emsp;&emsp;**实现细节**：
- 三种存储策略；
- 与```std::string```100%兼容。
- COW 存储时对于引用计数线程安全。
- 对 Jemalloc 友好。如果检测到使用```jemalloc```，那么将使用```jemalloc```的一些非标准扩展接口来提高性能。
- find()使用简化版的Boyer-Moore algorithm。在查找成功的情况下，相对于```string::find()```有 30 倍的性能提升。在查找失败的情况下也有 1.5 倍的性能提升。
- 可以与```std::string```互相转换。

## 2 FBString的实现
### 2.1 存储策略
&emsp;&emsp;FBString中存储字符串数据通过```fbstring_core```实现，提供了基本的数据操作的接口，```fbstring_basic```本身只是实现了用户需要的接口而已。```fbstring_core```实现三种不同的存储策略，并且同时支持大小端存储：
- SSO：小字符串直接使用栈内存（小于等于23个字符）；
- Eager Copy：中长度字符（大于23个字符，小于等于255个字符）总是使用堆内存并且总是拷贝，行为类似```std::string```；
- COW：长字符（大于255个字符）使用引用计数和COW计数避免不必要的拷贝操作；

```cpp
struct MediumLarge {
  Char* data_;
  size_t size_;
  size_t capacity_;

  size_t capacity() const {
    return kIsLittleEndian ? capacity_ & capacityExtractMask : capacity_ >> 2;
  }

  void setCapacity(size_t cap, Category cat) {
    capacity_ = kIsLittleEndian
        ? cap | (static_cast<size_t>(cat) << kCategoryShift)
        : (cap << 2) | static_cast<size_t>(cat);
  }
};

union {
    uint8_t bytes_[sizeof(MediumLarge)]; // For accessing the last byte.
    Char small_[sizeof(MediumLarge) / sizeof(Char)];
    MediumLarge ml_;
};

```
**SSO**
&emsp;&emsp;上面的匿名结构中```small_```中存储SSO的字符串，从上面的代码中能够看到SSO的大小依赖于当前机器的位宽和```size_t```的长度，其长度可能是```12~24```个字节，而最后一个字节用来存储当前字符串的长度，因此容量为```11~23```个字符。
&emsp;&emsp;当前SSO的长度和大小端就存储在最后一个字节中，另外SSO中实际存储的是剩余的空间大小而不是实际的大小。当前系统具体为大端还是小端是根据编译器的内置宏来判断，然后再尾部设置不同的Mask来标识。
```c
constexpr auto kIsLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
constexpr static size_t lastChar = sizeof(MediumLarge) - 1;
constexpr static size_t maxSmallSize = lastChar / sizeof(Char);
constexpr static uint8_t categoryExtractMask = kIsLittleEndian ? 0xC0 : 0x3;
size_t smallSize() const {
    assert(category() == Category::isSmall);
    constexpr auto shift = kIsLittleEndian ? 0 : 2;
    auto smallShifted = static_cast<size_t>(small_[maxSmallSize]) >> shift;
    assert(static_cast<size_t>(maxSmallSize) >= smallShifted);
    return static_cast<size_t>(maxSmallSize) - smallShifted;
}

Category category() const {
    // works for both big-endian and little-endian
    return static_cast<Category>(bytes_[lastChar] & categoryExtractMask);
}
```

&emsp;&emsp;```category_type```根据大小端的不同值不同，需要获取当前类型是直接和Mask形成掩码得到的就是当前字符串的类型。
```c
typedef uint8_t category_type;
enum class Category : category_type {
    isSmall = 0,
    isMedium = kIsLittleEndian ? 0x80 : 0x2,
    isLarge = kIsLittleEndian ? 0x40 : 0x1,
};
```

**Eager Copy**和**COW**
&emsp;&emsp;二者的存储结构基本相同，区别是后者会通过引用计数管理，而前者行为类似```std::string```。

### 2.2 构造和拷贝
&emsp;&emsp;```fbstring_core```进行操作时都会区分当前的字符串类型而后调用具体的处理函数进行处理：
```c
  fbstring_core( const Char* const data, const size_t size, bool disableSSO = FBSTRING_DISABLE_SSO) { 
    if (!disableSSO && size <= maxSmallSize) {
      initSmall(data, size);
    } else if (size <= maxMediumSize) {
      initMedium(data, size);
    } else {
      initLarge(data, size);
    }
    assert(this->size() == size);
    assert(size == 0 || memcmp(this->data(), data, size * sizeof(Char)) == 0);
  }
```

**SSO**
&emsp;&emsp;SSO进行初始化时就是直接将输入的字符串指针拷贝到```small_```中。

**Eager Copy**
&emsp;&emsp;Eager Copy创建是直接分配一块内存将原字符串内存拷贝到申请的内存上。
```c
template <class Char>
FOLLY_NOINLINE void fbstring_core<Char>::copyMedium(const fbstring_core& rhs) {
  // Medium strings are copied eagerly. Don't forget to allocate
  // one extra Char for the null terminator.
  auto const allocSize = goodMallocSize((1 + rhs.ml_.size_) * sizeof(Char));
  ml_.data_ = static_cast<Char*>(checkedMalloc(allocSize));
  // Also copies terminator.
  fbstring_detail::podCopy(
      rhs.ml_.data_, rhs.ml_.data_ + rhs.ml_.size_ + 1, ml_.data_);
  ml_.size_ = rhs.ml_.size_;
  ml_.setCapacity(allocSize / sizeof(Char) - 1, Category::isMedium);
  assert(category() == Category::isMedium);
}
```
**COW**
&emsp;&emsp;COW情况下的内存并不是直接申请的而是通过内部提供的```Refcount```获取的，该类实现了一个可引用计数管理的内存。
```c
template <class Char>
FOLLY_NOINLINE void fbstring_core<Char>::initLarge(
    const Char* const data, const size_t size) {
  // Large strings are allocated differently
  size_t effectiveCapacity = size;
  auto const newRC = RefCounted::create(data, &effectiveCapacity);
  ml_.data_ = newRC->data_;
  ml_.size_ = size;
  ml_.setCapacity(effectiveCapacity, Category::isLarge);
  ml_.data_[size] = '\0';
}
```

&emsp;&emsp;```RefCount```就是一个带有引用计数的内存块，其引用计数在整个内存的开头。因为引用计数和内存地址的偏移是固定的，我们总是能够通过数据指针找到引用计数的位置。
```c
  struct RefCounted {
    std::atomic<size_t> refCount_;
    Char data_[1];
    static RefCounted* create(size_t* size) {
        size_t capacityBytes;
        if (!folly::checked_add(&capacityBytes, *size, size_t(1))) {
            throw_exception(std::length_error(""));
        }
        if (!folly::checked_muladd( &capacityBytes, capacityBytes, sizeof(Char), getDataOffset())) {
            throw_exception(std::length_error(""));
        }
        const size_t allocSize = goodMallocSize(capacityBytes);
        auto result = static_cast<RefCounted*>(checkedMalloc(allocSize));
        result->refCount_.store(1, std::memory_order_release);
        *size = (allocSize - getDataOffset()) / sizeof(Char) - 1;
        return result;
    }
  }

```
&emsp;&emsp;拷贝时只是增加引用计数，只有真的需要拷贝时才进行拷贝。
```c
template <class Char>
FOLLY_NOINLINE void fbstring_core<Char>::copyLarge(const fbstring_core& rhs) {
  // Large strings are just refcounted
  ml_ = rhs.ml_;
  RefCounted::incrementRefs(ml_.data_);
  assert(category() == Category::isLarge && size() == rhs.size());
}
```
&emsp;&emsp;拷贝的时机发生在用户请求可修改的元素引用时，因为内部不知道用户会那地址干什么，只能拷贝避免内存数据错误。
```c
template <class Char>
inline Char* fbstring_core<Char>::mutableDataLarge() {
  assert(category() == Category::isLarge);
  if (RefCounted::refs(ml_.data_) > 1) { // Ensure unique.
    unshare();
  }
  return ml_.data_;
}

template <class Char>
FOLLY_NOINLINE void fbstring_core<Char>::unshare(size_t minCapacity) {
  assert(category() == Category::isLarge);
  size_t effectiveCapacity = std::max(minCapacity, ml_.capacity());
  auto const newRC = RefCounted::create(&effectiveCapacity);
  // If this fails, someone placed the wrong capacity in an
  // fbstring.
  assert(effectiveCapacity >= ml_.capacity());
  // Also copies terminator.
  fbstring_detail::podCopy(ml_.data_, ml_.data_ + ml_.size_ + 1, newRC->data_);
  RefCounted::decrementRefs(ml_.data_);
  ml_.data_ = newRC->data_;
  ml_.setCapacity(effectiveCapacity, Category::isLarge);
  // size_ remains unchanged.
}
```
## 3. 一些优化的细节
### 3.1 快速拷贝
&emsp;&emsp;因为SSO占用的长度可能为```sizeof(size_t)```的3倍及其以下，在拷贝SSO时根据当前内存是否对齐直接将对应的内存转换为对应宽度的```size_t```进行拷贝，一次可以处理多个字节，如果```sizeof(size_t)```为8就是一次拷贝8个字符。另外这里检查内存是否是对齐的是根据内对地址的低位是否为0来判断的，因此以对应位宽对其的内存低位一定为0。

```c
// Small strings are bitblitted
template <class Char>
inline void fbstring_core<Char>::initSmall(
    const Char* const data, const size_t size) {
// If data is aligned, use fast word-wise copying. Otherwise,
// use conservative memcpy.
// The word-wise path reads bytes which are outside the range of
// the string, and makes ASan unhappy, so we disable it when
// compiling with ASan.
#ifndef FOLLY_SANITIZE_ADDRESS
  if ((reinterpret_cast<size_t>(data) & (sizeof(size_t) - 1)) == 0) {
    const size_t byteSize = size * sizeof(Char);
    constexpr size_t wordWidth = sizeof(size_t);
    switch ((byteSize + wordWidth - 1) / wordWidth) { // Number of words.
      case 3:
        ml_.capacity_ = reinterpret_cast<const size_t*>(data)[2];
        FOLLY_FALLTHROUGH;
      case 2:
        ml_.size_ = reinterpret_cast<const size_t*>(data)[1];
        FOLLY_FALLTHROUGH;
      case 1:
        ml_.data_ = *reinterpret_cast<Char**>(const_cast<Char*>(data));
        FOLLY_FALLTHROUGH;
      case 0:
        break;
    }
  } else
#endif
  {
    if (size != 0) {
      fbstring_detail::podCopy(data, data + size, small_);
    }
  }
  setSmallSize(size);
}
```

### 3.2 循环展开
&emsp;&emsp;展开循环可以让编译器优化需要执行的指令，不限于优化为```rep```或者SSE等快速的指令。
```c
template <class Pod, class T>
inline void podFill(Pod* b, Pod* e, T c) {
  assert(b && e && b <= e);
  constexpr auto kUseMemset = sizeof(T) == 1;
  if /* constexpr */ (kUseMemset) {
    memset(b, c, size_t(e - b));
  } else {
    auto const ee = b + ((e - b) & ~7u);
    for (; b != ee; b += 8) {
      b[0] = c;
      b[1] = c;
      b[2] = c;
      b[3] = c;
      b[4] = c;
      b[5] = c;
      b[6] = c;
      b[7] = c;
    }
    // Leftovers
    for (; b != e; ++b) {
      *b = c;
    }
  }
}
```

### 3.3 __builtin_expect
&emsp;&emsp;通过编译器支持的```__builtin_expect```可以让编译器更好的优化代码。
```c
#define FOLLY_BUILTIN_EXPECT(exp, c) __builtin_expect(static_cast<bool>(exp), c)
#define FOLLY_LIKELY(...) FOLLY_BUILTIN_EXPECT((__VA_ARGS__), 1)
static RefCounted* create(const Char* data, size_t* size) {
    const size_t effectiveSize = *size;
    auto result = create(size);
    if (FOLLY_LIKELY(effectiveSize > 0)) {
    fbstring_detail::podCopy(data, data + effectiveSize, result->data_);
    }
    return result;
}
```

### 3.3 memory_order
&emsp;&emsp;```std::atomic<size_t> refCount_```进行原子操作的 c++ memory model :
- store，设置引用数为 1 : ```std::memory_order_release```；
- load，获取当前共享字符串的引用数: ```std::memory_order_acquire```；
- add/sub。增加/减少一个引用 : ```std::memory_order_acq_rel```。

### 3.4 malloc or realloc
&emsp;&emsp;避免实际要拷贝的内存很小而导致realloc的性能开销。
```c
FOLLY_MALLOC_CHECKED_MALLOC FOLLY_NOINLINE inline void* smartRealloc(
    void* p,
    const size_t currentSize,
    const size_t currentCapacity,
    const size_t newCapacity) {
  assert(p);
  assert(currentSize <= currentCapacity && currentCapacity < newCapacity);

  auto const slack = currentCapacity - currentSize;
  if (slack * 2 > currentSize) {
    // Too much slack, malloc-copy-free cycle:
    auto const result = checkedMalloc(newCapacity);
    std::memcpy(result, p, currentSize);
    free(p);
    return result;
  }
  // If there's not too much slack, we realloc in hope of coalescing
  return checkedRealloc(p, newCapacity);
}
```

## 4 参考文献
- [C++ folly库解读（一） Fbstring —— 一个完美替代std::string的库](https://zhuanlan.zhihu.com/p/348614098)
- [FBString](https://github.com/facebook/folly/blob/main/folly/docs/FBString.md)
- [FBString分析与使用FBString简介](https://cloud.tencent.com/developer/article/1329539)