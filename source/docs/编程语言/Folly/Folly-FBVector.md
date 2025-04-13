# Folly-FBVector
&emsp;&emsp;**摘要：** 工作已经有一段时间了，总的感觉实际工作中所使用到的C++相关的内容对实际提升C++没有太大帮助。基本都是搭积木做需求，前一段时间主要以看书为主，但是总感觉缺点什么。因此打算阅读一遍folly库的实现来提升自己的C++水平。本文主要描述了folly库中的fbvector的实现思想和细节。
&emsp;&emsp;**关键字**：FBVector

## 1 简介
&emsp;&emsp;```folly```中实现了一套可以替代```std::vector```的```FBVector```。```FBVector```相比```std::vector```有绝对的性能优势，主要归功于以下几点：

### 1.1 内存处理策略
&emsp;&emsp;当当前容器的内存不足时，```std::vector```采用```std::min(max_size(), std::max(current_cap + 1, 2 * current_cap))```（参考llvm实现），即当前使用对象个数的2倍内存来进行扩容，这也是大多是编译器容器采用的比例。但是该系数存在一个一个问题，即缓存不友好。
&emsp;&emsp;假如第一次期望的内存大小为$k$，你们$n$次扩容的内存大小分别为（当系数为$a,a>1$）$k,a^1k,a^2k,...,a^{n - 1}k$，那么$n$次分配的内存总和为
$$
k+ak+a^2k+...+a^{n-1}k=\frac{a^n-1}{a-1}
$$
&emsp;&emsp;为了让后续中的某一次内存分配时能够利用之前已经分配的内存，那么就应该有
$$k+ak+a^2k+...+a^{n-1}k=\frac{a^nk-1}{a-1} \ge a^nk$$
&emsp;&emsp;即$\frac{1}{a^nk}+a\le 2$，即当$n\rightarrow \infty$时，$a\le 2$即可，但是当$a=2$时，只有$n\rightarrow$等式才成立，现实中不可能，因此$1 \lt a \lt 2$才能保证在后续内存扩增时使用到之前分配的内存。
&emsp;&emsp;```FBVector```采用的系数为1.5，配合上```jemalloc```的```inplcae-malloc```，```FBVector```的内存策略缓存更友好。

### 1.2 Relocatable Object
&emsp;&emsp;很多时候大家使用 c++ 时，会保守地把 c++ 的变量假设为不可重定位的。而不可重定位的对象在进行移动时需要进行重新创建拷贝对象、析构原对象的操作，假如 ```vector```的元素有这一层限制的话，会一定程度上降低移动元素时的效率。但实际很多情况下c++的对象都是可重定位的，例外只有：
- 类中使用了内部对象的指针；
- 有外部的观察者观察着该对象且持有其指针。

>&emsp;&emsp;可重定位对象这个词感觉不是很合理，更准确的应该是BitwiseMoveable，可参考https://groups.google.com/a/isocpp.org/g/std-proposals/c/4Wwpi4EUGlg。

>&emsp;&emsp;I'd like to raise the issue of the relocatable types one more time. It is well known concept used under different names in the following libraries: Folly (IsRelocatable), BDE (IsBitwiseMoveable), EASTL (has_trivial_relocate), Qt (Q_MOVABLE_TYPE).
It is natural generalization of the trivially copyable types concept which can be used for the optimization of objects relocation.

&emsp;&emsp;```FBVector```实现了```IsRelocatable```来针对可重定位对象特化。

## 2 small_vector
### 2.1 简介
&emsp;&emsp;```folly::small_vector```是folly针对小数组的优化的实现，不同于```std::array```，当用户向数组中插入元素导致数组大小超过最初设置的元素个数就会退化为堆上分配。```folly::small_vector```基本兼容```std::vector```的实现，当数据量较少时，元素会被分配在栈上保证性能，否则会使用堆内存存储元素。
&emsp;&emsp;```small_vector```使用场景如下：
- Short-lived stack vectors with few elements (or maybe with a usually-known number of elements), if you want to avoid malloc.
- If the vector(s) are usually under a known size and lookups are very common, you'll save an extra cache miss in the common case when the data is kept in-place.
- You have billions of these vectors and don't want to waste space on std::vector's capacity tracking. This vector lets malloc track our allocation capacity. (Note that this slows down the insertion/reallocation code paths significantly; if you need those to be fast you should use fbvector.)

### 2.2 实现
&emsp;&emsp;```small_vector```是优化的思路和```fbstring```的SSO优化类似，对于小内存存储在栈上避免内存分配所带来的开销。先看下```small_vector```的声明，
```c
template <class Value, std::size_t RequestedMaxInline, class InPolicy>
struct small_vector_base {}

template <class Value, std::size_t RequestedMaxInline = 1, class Policy = void>
class small_vector
    : public detail::small_vector_base<Value, RequestedMaxInline, Policy>::type {};
```
**类结构**
&emsp;&emsp;```small_vector```类定义的三个参数中，```Value```表示期望存储的对象类型，```RequestedMaxInline```表示期望在栈上存储的大小，实际上使用的是```MaxInline{ RequestedMaxInline == 0 ? 0 : constexpr_max(kSizeOfValuePtr / kSizeOfValue, RequestedMaxInline)};```，第三个模板参数用来判断是否使用堆内存。```fbvector```支持可选的堆内存，比如官方给出的example：
``` Cpp
    // With space for 32 in situ unique pointers, and only using a
    // 4-byte size_type.
    small_vector<std::unique_ptr<int>, 32, uint32_t> v;
    // A inline vector of up to 256 ints which will not use the heap.
    small_vector<int, 256, NoHeap> v;
    // Same as the above, but making the size_type smaller too.
    small_vector<int, 256, NoHeap, uint16_t> v;
```

&emsp;&emsp;再深入了解```fbvector```的内部内存结构之前，我们先来看下类的三个模板参数如何影响类的结构。
&emsp;&emsp;```Value```即类存储的类成员类型，主要用于type_traits。
&emsp;&emsp;```RequestedMaxInline```即期望在栈上分配的对象的个数。取值的下限是机器字长所能存储的元素个数。
```c
static constexpr std::size_t MaxInline{
      RequestedMaxInline == 0 ? 0 : constexpr_max(kSizeOfValuePtr / kSizeOfValue, RequestedMaxInline)};
```
&emsp;&emsp;```InPolicy```内存分配大小的策略，而如果使用堆存储的话实际的size的决策是在```IntegralSizePolicy```中实现的。

**类内存结构**
```small_vector```的类内存结构如下，通过一个union来保存两个指针，分别代表在堆上和栈上分配内存的两种情况。
```c
  union Data {
    explicit Data() { pdata_.heap_ = nullptr; }
    PointerType pdata_;
    InlineStorageType storage_;
  }
```
&emsp;&emsp;为了理解```small_vector```如何组织内存，我们直接看```push_back```的实现。从下面的代码能够看出```push_back```实际上调用的```emplace_back```。
```c
  template <class... Args>
  reference emplace_back(Args&&... args) {
    auto isize_ = this->getInternalSize();
    if (isize_ < MaxInline) {
      new (u.buffer() + isize_) value_type(std::forward<Args>(args)...);
      this->incrementSize(1);
      return *(u.buffer() + isize_);
    }
    if (!BaseType::kShouldUseHeap) {
      throw_exception<std::length_error>("max_size exceeded in small_vector");
    }
    auto size_ = size();
    auto capacity_ = capacity();
    if (capacity_ == size_) {
      // Any of args may be references into the vector.
      // When we are reallocating, we have to be careful to construct the new
      // element before modifying the data in the old buffer.
      makeSize(
          size_ + 1,
          [&](void* p) { new (p) value_type(std::forward<Args>(args)...); },
          size_);
    } else {
      // We know the vector is stored in the heap.
      new (u.heap() + size_) value_type(std::forward<Args>(args)...);
    }
    this->incrementSize(1);
    return *(u.heap() + size_);
  }

  void push_back(value_type&& t) { emplace_back(std::move(t)); }

  void push_back(value_type const& t) { emplace_back(t); }
```
### 2.3 参考文献
- [is_relocatable type trait](https://groups.google.com/a/isocpp.org/g/std-proposals/c/4Wwpi4EUGlg)