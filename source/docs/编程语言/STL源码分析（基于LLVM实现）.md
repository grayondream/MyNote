


## 1 ```allocator```
### 1.1 ```allocator```
#### 1.1.1 简介
&emsp;&emsp;```allocator```是STL中对一个堆内存分配器，是对内存申请工作的一个封装，将内存的申请和成员的构造抽象开来方便控制。基本上，C++标准库中的容器的默认分配器都是```allocator```。在C++中分配器是通过模板参数的方式指定给对应的容器，默认就是```allocator```，用户自己也可以实现自己的内存管理类，对堆的内存进行有效的管理也可以将对应的分配器指定给容器使用（前提是接口保持一致）。
```cpp
    std::allocator<int> alloc1;
    // demonstrating the few directly usable members
    static_assert(std::is_same_v<int, decltype(alloc1)::value_type>);
    int* p1 = alloc1.allocate(1); // space for one int
    alloc1.deallocate(p1, 1);     // and it is gone
```

#### 1.1.2 ```allocator```的实现
&emsp;&emsp;先简单看下```allocator```的声明，其中[_LIBCPP_TEMPLATE_VIS](https://libcxx.llvm.org//DesignDocs/VisibilityMacros.html)是不同版本编译期的可见性宏，标准库的源码中有大量类似的宏，我们大概直到意思即可不用深究。
```cpp
template <class _Tp>
class _LIBCPP_TEMPLATE_VIS allocator
    : private __non_trivial_if<!is_void<_Tp>::value, allocator<_Tp> >
```
&emsp;&emsp;```allocator```继承的```__non_trivial_if```是一个空类，该类利用cpp的CRTP实现编译期的多态。该类由一个偏特化版本区别是带有non-trivial的构造函数。在```allocator```中非```void```类型都是匹配第二个，```void```类型匹配第一个。而模板的第二个参数```_Unique```是为了保持菱形继承过程中的ABI稳定而设置的。
```cpp
template <bool _Cond, class _Unique>
struct __non_trivial_if { };

template <class _Unique>
struct __non_trivial_if<true, _Unique> {
    _LIBCPP_INLINE_VISIBILITY
    _LIBCPP_CONSTEXPR __non_trivial_if() _NOEXCEPT { }
};
```

**allocate函数**
&emsp;&emsp;```allocate```函数是用来分配堆内存，可以看到内部实现调用了```operator new```。该函数实现的基本逻辑就是检查当前希望分配的大小是否可满足（```allocator_traits```下面会详细描述，暂时就理解为一个类型系统），不满足就会抛出异常，否则会调用对应的分配函数进行分配。```__libcpp_is_constant_evaluated```用来判断当前函数是否为```constexpr```。```libcpp_alocate```的区别内部依然调用了```::operator new实现，只是有内存对齐。另外，这里不用```new```，而是使用```::operator new```应该是为了避免用户承载了```new```的实现而导致无法真正分配到内存。

>&emsp;&emsp;```_VSTD```类似于```std```，也是一个名字空间。
&emsp;&emsp;```_VSTD```is now an alias for ```std```instead of ```std::_LIBCPP_ABI_NAMESPACE```.
  This is technically not a functional change, except for folks that might have been
  using ```_VSTD```in creative ways (which has never been officially supported).   ————来源[libcxx/docs/ReleaseNotes.rs](https://reviews.llvm.org/rG453620f55ea38186cdf165b1ca2deb6c6b226132)


```cpp
_LIBCPP_NODISCARD_AFTER_CXX17 _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX20
    _Tp* allocate(size_t __n) {
        if (__n > allocator_traits<allocator>::max_size(*this))
            __throw_bad_array_new_length();
        if (__libcpp_is_constant_evaluated()) {
            return static_cast<_Tp*>(::operator new(__n * sizeof(_Tp)));
        } else {
            return static_cast<_Tp*>(_VSTD::__libcpp_allocate(__n * sizeof(_Tp), _LIBCPP_ALIGNOF(_Tp)));
        }
    }
```

**deallocate**
&emsp;&emsp;```deallocate```是用来释放内存的，其实现和```allocate```基本类似。
```cpp
_LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX20
    void deallocate(_Tp* __p, size_t __n) _NOEXCEPT {
        if (__libcpp_is_constant_evaluated()) {
            ::operator delete(__p);
        } else {
            _VSTD::__libcpp_deallocate((void*)__p, __n * sizeof(_Tp), _LIBCPP_ALIGNOF(_Tp));
        }
    }
```

**construct**和**destroy**
&emsp;&emsp;```construct```（仅仅在对应内存上构造对象）和```destroy```（仅仅调用析构函数）函数的实现比较简单就是直接显式调用构造函数和析构函数。
```cpp
template <class _Up, class... _Args>
    _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_INLINE_VISIBILITY
    void construct(_Up* __p, _Args&&... __args) {
        ::new ((void*)__p) _Up(_VSTD::forward<_Args>(__args)...);
    }

    _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_INLINE_VISIBILITY
    void destroy(pointer __p) {
        __p->~_Tp();
    }
```

**rebind**
&emsp;&emsp;获取另一个类型的```allocator```，对于带有节点的数据结构，当前```allocator```只能管理节点数据的内存但是对于节点本身是无法管理的，使用这样的方式能够让二者的管理统一。
```cpp
template <class _Up>
    struct _LIBCPP_DEPRECATED_IN_CXX17 rebind {
        typedef allocator<_Up> other;
    };
```
### 1.2 ```allocator```特化版本
#### 1.2.1 ```const allocator```
&emsp;&emsp;```const```特化版本主要是```allocate```时返回的是```const```的函数指针，且在释放时会通过```const_cast```去除```const```进行释放。
```cpp
const _Tp* allocate(size_t __n) {
    if (__n > allocator_traits<allocator>::max_size(*this))
        __throw_bad_array_new_length();
    if (__libcpp_is_constant_evaluated()) {
        return static_cast<const _Tp*>(::operator new(__n * sizeof(_Tp)));
    } else {
        return static_cast<const _Tp*>(_VSTD::__libcpp_allocate(__n * sizeof(_Tp), _LIBCPP_ALIGNOF(_Tp)));
    }
}

void deallocate(const _Tp* __p, size_t __n) {
    if (__libcpp_is_constant_evaluated()) {
        ::operator delete(const_cast<_Tp*>(__p));
    } else {
        _VSTD::__libcpp_deallocate((void*) const_cast<_Tp *>(__p), __n * sizeof(_Tp), _LIBCPP_ALIGNOF(_Tp));
    }
}
```

#### 1.2.2 ```void allocator```
&emsp;&emsp;```allocator```的```void```特化版本就是一个空壳子，什么也没有。
```cpp
template <>
class _LIBCPP_TEMPLATE_VIS allocator<void>
{
#if _LIBCPP_STD_VER <= 17 || defined(_LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_MEMBERS)
public:
    _LIBCPP_DEPRECATED_IN_CXX17 typedef void*             pointer;
    _LIBCPP_DEPRECATED_IN_CXX17 typedef const void*       const_pointer;
    _LIBCPP_DEPRECATED_IN_CXX17 typedef void              value_type;

    template <class _Up> struct _LIBCPP_DEPRECATED_IN_CXX17 rebind {typedef allocator<_Up> other;};
#endif
};
```

### 1.3 类型萃取```allocator_traits```
&emsp;&emsp;```allocator_traits```将当前类型的类型抽象到一个类中，其实现就是一个类型集合。```allocator_traits```额外提供了一些关于内存分配的函数，比如```max_size```等。

```cpp
template <class _Alloc>
struct _LIBCPP_TEMPLATE_VIS allocator_traits
{
    using allocator_type = _Alloc;
    using value_type = typename allocator_type::value_type;
    using pointer = typename __pointer<value_type, allocator_type>::type;
    using const_pointer = typename __const_pointer<value_type, pointer, allocator_type>::type;
    using void_pointer = typename __void_pointer<pointer, allocator_type>::type;
    using const_void_pointer = typename __const_void_pointer<pointer, allocator_type>::type;
    using difference_type = typename __alloc_traits_difference_type<allocator_type, pointer>::type;
    using size_type = typename __size_type<allocator_type, difference_type>::type;
    using propagate_on_container_copy_assignment = typename __propagate_on_container_copy_assignment<allocator_type>::type;
    using propagate_on_container_move_assignment = typename __propagate_on_container_move_assignment<allocator_type>::type;
    using propagate_on_container_swap = typename __propagate_on_container_swap<allocator_type>::type;
    using is_always_equal = typename __is_always_equal<allocator_type>::type;
};
```

```cpp
    template <class _Ap = _Alloc, class = void, class =
        __enable_if_t<!__has_max_size<const _Ap>::value> >
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX20
    static size_type max_size(const allocator_type&) _NOEXCEPT {
        return numeric_limits<size_type>::max() / sizeof(value_type);
    }
```

## 2 string和string_view

&emsp;&emsp;```std::string```的实现实际上就是```basic_string```，我们经常使用的就是两种```typedef basic_string<char>    string;```和```typedef basic_string<wchar_t>wstring```。```std::string```使用的分配器是```allocator<char>```，就是简单的```new char```，而```trait```使用的是```char_traits```。

```cpp
template <class _CharT, class _Traits = char_traits<_CharT>, class _Allocator = allocator<_CharT> >
class _LIBCPP_TEMPLATE_VIS basic_string;

using string = basic_string<char>;
```

### 2.1 ```char_traits```
&emsp;&emsp;```char_traits```就是```char```类型的萃取器，抽象了和当前类型相关的类型集，以及类型相关的一些操作。

```cpp
struct _LIBCPP_DEPRECATED_("char_traits<T> for T not equal to char, wchar_t, char8_t, char16_t or char32_t is non-standard and is provided for a temporary period. It will be removed in LLVM 18, so please migrate off of it.")
    char_traits
{
    using char_type  = _CharT;
    using int_type   = int;
    using off_type   = streamoff;
    using pos_type   = streampos;
    using state_type = mbstate_t;
};
```
&emsp;&emsp;下面这些实现实际上是通用的```char_traits```的实现，因为使用的for-loop性能上完全依赖于编译器优化，如果编译器无法判断两块内存是否overlap的话，对于```trival type```性能可能不如```memcpy```等整块内存拷贝或者move的API。从源码的注释我们也能看到这个类是不建议使用的，应该使用对应的特化版本，特化版本比如```char_traits<char>```使用了```std::copy```等API可以对操作进行加速。

```cpp
char_traits<T> for T not equal to char, wchar_t, char8_t, char16_t or char32_t is non-standard and is provided for a temporary period. It will be removed in LLVM 18, so please migrate off of it
```
**assign lt eq**
&emsp;&emsp;这三个函数就是对```operator =,operator == ,operator <```的封装。
```cpp
static inline void _LIBCPP_CONSTEXPR_SINCE_CXX17
        assign(char_type& __c1, const char_type& __c2) _NOEXCEPT {__c1 = __c2;}
static inline _LIBCPP_CONSTEXPR bool eq(char_type __c1, char_type __c2) _NOEXCEPT
    {return __c1 == __c2;}
static inline _LIBCPP_CONSTEXPR bool lt(char_type __c1, char_type __c2) _NOEXCEPT
    {return __c1 < __c2;}
```

**compare**
&emsp;&emsp;遍历当前字符串，逐个字符比较，需要保证输入的字符串长度至少为```n```负责会越界。
```cpp
static _LIBCPP_CONSTEXPR_SINCE_CXX17 int compare(const char_type* __s1, const char_type* __s2, size_t __n) {
        for (; __n; --__n, ++__s1, ++__s2){
            if (lt(*__s1, *__s2))
                return -1;
            if (lt(*__s2, *__s1))
                return 1;
        }
        return 0;
    }
```

**length**
&emsp;&emsp;获取字符串的长度，遍历字符串，以```char_type(0)```为结束点，实现和```strlen```相同。
```cpp
_LIBCPP_INLINE_VISIBILITY static _LIBCPP_CONSTEXPR_SINCE_CXX17 size_t length(const char_type* __s) {
        size_t __len = 0;
        for (; !eq(*__s, char_type(0)); ++__s)
            ++__len;
        return __len;
    }
```

**find**
&emsp;&emsp;从头到尾遍历字符串，直到找到对应的值为止。
```cpp
_LIBCPP_INLINE_VISIBILITY static _LIBCPP_CONSTEXPR_SINCE_CXX17
    const char_type* find(const char_type* __s, size_t __n, const char_type& __a) {
        for (; __n; --__n){
            if (eq(*__s, __a))
                return __s;
            ++__s;
        }
        return nullptr;
    }
```

**move**：
&emsp;&emsp;字符串移动。可以看到针对当前输入的两块内存的地址相对位置选择了不同方向的拷贝动作，这样做是为了正确处理存在内存交叉的两块内存的字符串。
```cpp
static char_type*       move(char_type* __s1, const char_type* __s2, size_t __n) {
    if (__n == 0) return __s1;
    char_type* __r = __s1;
    if (__s1 < __s2){
        for (; __n; --__n, ++__s1, ++__s2)
            assign(*__s1, *__s2);
    }
    else if (__s2 < __s1){
        __s1 += __n;
        __s2 += __n;
        for (; __n; --__n)
            assign(*--__s1, *--__s2);
    }
    return __r;
}
```

**copy**
&emsp;&emsp;字符串拷贝，这里的动作基本上和```move```相同但是并没有进行地址交错的处理。因为```move```的语义是移动，旧的丢弃，而```copy```的语义是拷贝，旧的应该保留。
```cpp
_LIBCPP_INLINE_VISIBILITY static _LIBCPP_CONSTEXPR_SINCE_CXX20 char_type*       copy(char_type* __s1, const char_type* __s2, size_t __n) {
    if (!__libcpp_is_constant_evaluated()) {
        _LIBCPP_ASSERT(__s2 < __s1 || __s2 >= __s1+__n, "char_traits::copy overlapped range");
    }
    char_type* __r = __s1;
    for (; __n; --__n, ++__s1, ++__s2)
        assign(*__s1, *__s2);
    return __r;
}
```

### 2.2 ```string```

#### 2.2.1 内存布局
&emsp;&emsp;```std::string```中内存布局根据```_LIBCPP_ABI_ALTERNATE_STRING_LAYOUT```有所不同，就是下面列的这种，```__data__```域在类开头，这样在某些需要对齐的场景比较有优势。另一种是将```__data___```等四个成员完全反向陈列。
```cpp
    struct __long{
        struct _LIBCPP_PACKED {
            size_type __is_long_ : 1;
            size_type __cap_ : sizeof(size_type) * CHAR_BIT - 1;
        };
        size_type __size_;
        pointer   __data_;
    };
    enum {__min_cap = (sizeof(__long) - 1)/sizeof(value_type) > 2 ? (sizeof(__long) - 1)/sizeof(value_type) : 2};
    struct __short{
        struct _LIBCPP_PACKED {
            unsigned char __is_long_ : 1;
            unsigned char __size_ : 7;
        };
        char __padding_[sizeof(value_type) - 1];
        value_type __data_[__min_cap];
    };

    struct __raw{
        size_type __words[__n_words];
    };

    struct __rep{
        union
        {
            __long  __l;
            __short __s;
            __raw   __r;
        };
    };

    __compressed_pair<__rep, allocator_type> __r_;
```
&emsp;&emsp;从上面的定义中可以看出，```string```针对不同场景的优化，对于小字符串（一般为15或23）直接存储在栈上即SSO优化，而对于大字符串会通过堆来存储。```__compressed_pair```是一种可以利用编译期一些优化的组合类，可以理解为```pair```。


**```__compressed_pair```**
&emsp;&emsp;```__compressed_pair```其实就是两个变量的集合，类似```pair```，只不过针对不同的场景进行了优化。将标准库中的代码简化之后主要的代码就是下面这段。对于可继承的类型，直接会通过继承的方式构造，这样可以对于空基类可以触发EBO(https://en.cppreference.com/w/cpp/language/ebo)，节省一部分内存；而对于不可继承的类型直接就创建一个成员。
```c
template <class _Tp, int _Idx, bool _CanBeEmptyBase = is_empty<_Tp>::value && !__libcpp_is_final<_Tp>::value>
struct __compressed_pair_elem {
private:
    _Tp __value_;
};

template <class _Tp, int _Idx>
struct __compressed_pair_elem<_Tp, _Idx, true> : private _Tp {};

template <class _T1, class _T2>
class __compressed_pair : private __compressed_pair_elem<_T1, 0>, private __compressed_pair_elem<_T2, 1> {};
```


#### 2.2.2 ```string```的构造与析构
**构造**
&emsp;&emsp;```string```的第一种构造默认初始化，调用```__default_init```创建一个空的```__rep```，没有进行其他操作。第二种就是调用```__init```创建内存。
&emsp;&emsp;首先检查当前输出参数的长度是否符合SSO优化的前提，满足的话直接使用short版本，不满足的话再通过```allocator```分配内存。最后通过```char_traits::copy```将字符拷贝到内存上。
```cpp

template <class _CharT, class _Traits, class _Allocator>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void basic_string<_CharT, _Traits, _Allocator>::__init(const value_type* __s, size_type __sz, size_type __reserve){
    if (__libcpp_is_constant_evaluated())
        __r_.first() = __rep();
    if (__reserve > max_size()) //长度过长
        __throw_length_error();
    pointer __p;
    if (__fits_in_sso(__reserve)){//是否满足sso优化的条件
        __set_short_size(__sz);
        __p = __get_short_pointer();
    }
    else{
        auto __allocation = std::__allocate_at_least(__alloc(), __recommend(__reserve) + 1);
        __p = __allocation.ptr;
        __begin_lifetime(__p, __allocation.count);
        __set_long_pointer(__p);    //设置long.__data__
        __set_long_cap(__allocation.count);//设置long.__cap__
        __set_long_size(__sz);              //设置long.__sz__
    }
    traits_type::copy(std::__to_address(__p), __s, __sz);//拷贝内存，逐个拷贝
    traits_type::assign(__p[__sz], value_type());//末尾插入\0
}
```
&emsp;&emsp;能够注意到申请内存时，实际申请的大小是```__recommend(__reserve) + 1```，起始就是字节对齐的长度。
```cpp
 size_type __recommend(size_type __s) _NOEXCEPT{
    if (__s < __min_cap) {
        if (__libcpp_is_constant_evaluated())
            return static_cast<size_type>(__min_cap);
        else
            return static_cast<size_type>(__min_cap) - 1;
    }
    //__alignment == 16
    size_type __guess = __align_it<sizeof(value_type) < __alignment ?
                    __alignment/sizeof(value_type) : 1 > (__s+1) - 1;
    if (__guess == __min_cap) ++__guess;
    return __guess;
}
```
&emsp;&emsp;另外移动拷贝构造就是直接接管原字符的内存，并在原字符内存上构建一个空字符串。之后输入的字符串的```allocator```和当前字符串不同时才会逐个拷贝，也就是说```move```后的源字符串是会失效的，这也符合```move```的语义。而对于```const _CharT* __s```输入的构造函数和拷贝构造函数直接就是调用的```__init```申请拷贝内存。
```cpp
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 
  basic_string(basic_string&& __str, const allocator_type& __a) : __r_(__default_init_tag(), __a) {
    if (__str.__is_long() && __a != __str.__alloc()) // copy, not move
    __init(std::__to_address(__str.__get_long_pointer()), __str.__get_long_size());
    else {
    if (__libcpp_is_constant_evaluated())
        __r_.first() = __rep();
    __r_.first() = __str.__r_.first();
    __str.__default_init();
    }
    std::__debug_db_insert_c(this);
    if (__is_long())
    std::__debug_db_swap(this, &__str);
}
```
**析构**
&emsp;&emsp;析构就比较简单了，如果是长字符，就会调用```deallocate```释放内存。
```cpp
basic_string<_CharT, _Traits, _Allocator>::~basic_string(){
    std::__debug_db_erase_c(this);
    if (__is_long())
        __alloc_traits::deallocate(__alloc(), __get_long_pointer(), __get_long_cap());
}
```

#### 2.2.3 ```string```的一些操作函数
**扩容**
&emsp;&emsp;```string```的扩容是通过```__grow_by_and_replace```进行的，另一个实现```__grow_by```也是一样的逻辑，基本的流程是：
1. 检查大小是否超出最大值；
2. 计算期望得到的cap大小，值为```std::max(2 * old_cap, new_size)```；
3. 通过```allocator```分配一块儿新内存；
4. 将旧内存中的值拷贝到新内存中；
5. 如果旧内存为堆分配的则释放内存；
6. 设置一些参数，结尾插入空字符。

```cpp
void
basic_string<_CharT, _Traits, _Allocator>::__grow_by_and_replace
    (size_type __old_cap, size_type __delta_cap, size_type __old_sz,
     size_type __n_copy,  size_type __n_del,     size_type __n_add, const value_type* __p_new_stuff){
    size_type __ms = max_size();
    if (__delta_cap > __ms - __old_cap - 1)
        __throw_length_error();
    pointer __old_p = __get_pointer();
    size_type __cap = __old_cap < __ms / 2 - __alignment ?
                          __recommend(std::max(__old_cap + __delta_cap, 2 * __old_cap)) :
                          __ms - 1;
    auto __allocation = std::__allocate_at_least(__alloc(), __cap + 1);
    pointer __p = __allocation.ptr;
    __begin_lifetime(__p, __allocation.count);
    std::__debug_db_invalidate_all(this);
    if (__n_copy != 0)
        traits_type::copy(std::__to_address(__p),
                          std::__to_address(__old_p), __n_copy);
    if (__n_add != 0)
        traits_type::copy(std::__to_address(__p) + __n_copy, __p_new_stuff, __n_add);
    size_type __sec_cp_sz = __old_sz - __n_del - __n_copy;
    if (__sec_cp_sz != 0)
        traits_type::copy(std::__to_address(__p) + __n_copy + __n_add,
                          std::__to_address(__old_p) + __n_copy + __n_del, __sec_cp_sz);
    if (__old_cap+1 != __min_cap || __libcpp_is_constant_evaluated())
        __alloc_traits::deallocate(__alloc(), __old_p, __old_cap+1);
    __set_long_pointer(__p);
    __set_long_cap(__allocation.count);
    __old_sz = __n_copy + __n_add + __sec_cp_sz;
    __set_long_size(__old_sz);
    traits_type::assign(__p[__old_sz], value_type());
}
```

**缩容**
&emsp;&emsp;```string```缩容有两种：一种是调用```__null_terminate_at```插入空字符，实际的大小并未发生变化；另一种是调用```__shrink_or_extend```(实际实现就是申请新内存拷贝旧的，释放旧的)，```__cap__```会变化但是```__sz```不会变化。
```cpp
basic_string& __null_terminate_at(value_type* __p, size_type __newsz) {
      __set_size(__newsz);
      __invalidate_iterators_past(__newsz);
      traits_type::assign(__p[__newsz], value_type());
      return *this;
}
```

&emsp;&emsp;其他的实现就是在```assign,move,copy```以及以上函数的基础上操作，就不再罗列。算法相关的函数等看算法的时候再说。

### 2.3 ```string_view```
&emsp;&emsp;```string_view```用来持有字符串，也就是说他只是一个wrapper，并不对持有的字符串负责，也就没有拷贝析构等操作，相比于```string```性能上要好不少。
```cpp
template<class _CharT, class _Traits = char_traits<_CharT> >
class _LIBCPP_TEMPLATE_VIS basic_string_view;

typedef basic_string_view<char>     string_view;
```

&emsp;&emsp;```string_view```的结构非常简单，就是一个指针和长度，在进行构造时也仅仅是浅拷贝，将对应的指针传递给成员。也就是说在使用```string_view```时要对齐管理的内存及其小心，避免在内存已经失效时还用```string_view```访问。
>It is the programmer's responsibility to ensure that std::string_view does not outlive the pointed-to character array
```cpp
template<class _CharT, class _Traits>
class basic_string_view {
private:
    const   value_type* __data_;
    size_type           __size_;
};

basic_string_view(const _CharT* __s, size_type __len) _NOEXCEPT
        : __data_(__s), __size_(__len){}
```

## 3 vector
&emsp;&emsp;```vector```是标准库中的连续存储的容器，也就是容器中说任意两个索引上相邻的元素的地址也是相邻的，可以通过索引随机访问。```vector```中的元素默认是通过堆内存管理的，在进行空间分配时一般会比时机需求的空间要大，即```capacity_size```，这样能够避免在插入元素时频繁申请内存导致的性能问题（如果频繁申请内存导致页置换的话还是很耗时的）。
&emsp;&emsp;先看下容器的定义，和其他容器一样都是一个模板类。```_Tp```就是类型，而```_Allocator```是进行内存管理的分配器，默认分配器就是通过```::operator new```和```::operator delete```申请和释放内存的。
```cpp
template <class _Tp, class _Allocator /* = allocator<_Tp> */>
class _LIBCPP_TEMPLATE_VIS vector
```

&emsp;&emsp;```vector```的内存布局比较简单如下图，有三个指针分别指向了对应的开始地址，已使用部分的尾地址，申请到的内存的尾地址，```[__begin_, __end_)```之间是已经使用的内存部分，```[__end_, __end_cap_)```是申请了但是未使用的部分（保留这一部分是为了避免插入元素时频繁allocate而可能出现的性能问题。）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8c2a8cde31bf1369a443d75e903ff984.png)


```cpp
private:
    pointer __begin_ = nullptr;
    pointer __end_ = nullptr;
    __compressed_pair<pointer, allocator_type> __end_cap_ =
        __compressed_pair<pointer, allocator_type>(nullptr, __default_init_tag());
```

**构造和销毁**
&emsp;&emsp;```vector```的构造比较简单，就是先通过```allocator```申请一块内存，然后通过```for```循环逐个构造对象。构造时通过```for```循环实现，由于没有利用CPU的一些SMID指令的优化，必然效率不是很好。
```cpp
vector(size_type __n, const value_type& __x, const allocator_type& __a) : __end_cap_(nullptr, __a){
    std::__debug_db_insert_c(this);
    if (__n > 0){
        __vallocate(__n);
        __construct_at_end(__n, __x);
    }
}

vector<_Tp, _Allocator>::__construct_at_end(size_type __n, const_reference __x){
    _ConstructTransaction __tx(*this, __n);
    const_pointer __new_end = __tx.__new_end_;
    for (pointer __pos = __tx.__pos_; __pos != __new_end; __tx.__pos_ = ++__pos) {
        __alloc_traits::construct(this->__alloc(), std::__to_address(__pos), __x);
    }
}
```
&emsp;&emsp;销毁就比较直接，通过一个包装器```__destroy_vector```，先```clear```再调用```deallocate```释放内存。
```cpp
__vec_.__clear();
__alloc_traits::deallocate(__vec_.__alloc(), __vec_.__begin_, __vec_.capacity());;
```

**clear**
&emsp;&emsp;```clear```函数只会针对析构容器中的每一个函数并不会释放当前容器中的内存。因此在进行容器释放时需要注意，如果期望释放内存的话可以通过```vector().swap(vec)```的方式或者在调用```clear```之后调用```shrink_to_fit	```调整内存大小。
```cpp
void clear() _NOEXCEPT{
    size_type __old_size = size();
    __clear();
    __annotate_shrink(__old_size);  //看源码里面什么也不会做
    std::__debug_db_invalidate_all(this);
}

void __clear() _NOEXCEPT {__base_destruct_at_end(this->__begin_);}
void __base_destruct_at_end(pointer __new_last) _NOEXCEPT {
    pointer __soon_to_be_end = this->__end_;
    while (__new_last != __soon_to_be_end)          //依然是一个完整的循环析构
        __alloc_traits::destroy(__alloc(), std::__to_address(--__soon_to_be_end));
    this->__end_ = __new_last;
}
```

**push_back**
&emsp;&emsp;```push_back```时，如果当前有足够的的大小则会在尾部构建一个对象，扩容的大小是按照现有大小的2倍来，即```std::min(max_size(), std::max(current_cap + 1, 2 * current_cap))```，简单的理解就是在条件允许的情况下扩容2倍。
```cpp
void vector<_Tp, _Allocator>::__push_back_slow_path(_Up&& __x){
    allocator_type& __a = this->__alloc();
    //__split_buffer就是一个包装器
    __split_buffer<value_type, allocator_type&> __v(__recommend(size() + 1), size(), __a);
    // __v.push_back(std::forward<_Up>(__x));
    __alloc_traits::construct(__a, std::__to_address(__v.__end_), std::forward<_Up>(__x));
    __v.__end_++;
    __swap_out_circular_buffer(__v);//这个函数没有干什么就是将__v中的size设置给当前的vector
}
```

**emplace_back**
&emsp;&emsp;```emplace_back```和```push_back```基本相同都是向容器中插入元素，如果对于插入```vector::value_type```类型的对象二者是没有区别的，```push_back```也实现了右值的重载，不存在```push_back```对于右值会多次拷贝的情况。主要的区别是```emplace_back```通过可变参数模板将参数直接传递给了构建器也就意味着同样的代码```emplace_back```直接在对应的内存上构建对象，而相比之下```push_back```是先构建再拷贝。
```cpp
void vector<_Tp, _Allocator>::emplace_back(_Args&&... __args)
```

**resize**
&emsp;&emsp;```resize```的实现比较直接，内存小了就扩容，大了就析构但是并不释放内存。
```cpp
void vector<_Tp, _Allocator>::resize(size_type __sz, const_reference __x){
    size_type __cs = size();
    if (__cs < __sz)
        this->__append(__sz - __cs, __x);
    else if (__cs > __sz)
        this->__destruct_at_end(this->__begin_ + __sz);//只会析构对象，不会释放内存
}
```

**shrink_to_fit**
&emsp;&emsp;
```cpp
vector<bool, _Allocator>::shrink_to_fit() _NOEXCEPT{
    if (__external_cap_to_internal(size()) > __cap()){
        vector(*this, allocator_type(__alloc())).swap(*this);
    }
}
```

>```vector<bool>```基本上被建议放弃使用了，所以就不深入了。
## 4 array

&emsp;&emsp;```array```的实现比较简单就是一个简单的栈数组的包装器。就不详细描述了。
```cpp
template <class _Tp, size_t _Size>
struct _LIBCPP_TEMPLATE_VIS array
{
    // types:
    typedef array __self;
    typedef _Tp                                   value_type;
    typedef value_type&                           reference;
    typedef const value_type&                     const_reference;
    typedef value_type*                           iterator;
    typedef const value_type*                     const_iterator;
    typedef value_type*                           pointer;
    typedef const value_type*                     const_pointer;
    typedef size_t                                size_type;
    typedef ptrdiff_t                             difference_type;
    typedef _VSTD::reverse_iterator<iterator>       reverse_iterator;
    typedef _VSTD::reverse_iterator<const_iterator> const_reverse_iterator;

    _Tp __elems_[_Size];
};
```

## 5 ```iterator```
&emsp;&emsp;迭代器是一种抽象设计模式，旨在不暴露实现细节的情况下按照某种特定的顺序访问类内的数据。简单的说，就类似数组和访问数组的指针的关系。这样就可以将实际承载数据的容器和处理容器的一些算法分离开，算法只需要通过迭代器访问类内的数据完成一些通用的算法处理。但是由于高度抽象没有足够的细节，算法处理只能根据有限的信息做通用处理没法针对特定的场景选择更加优秀的处理方式。一些容器比如```string```就实现了自己的一套算法处理API。
&emsp;&emsp;迭代器是用来访问容器中数据成员的抽象接口，它的行为类似于指针对象，可以读写对应索引的成员，并进行```--,++```等操作改变索引。
### 5.1 ```iterator```
&emsp;&emsp;STL的迭代器就是规定了6大类型的迭代器以及迭代器对应的```iterator_traits```，每个迭代器的访问规则都是固定的，比如前向迭代器可以```++```，随机访问迭代器可以```+n```。具体的迭代器实现和具体的容器实现是强相关的，每个迭代器的实现就有具体的容器负责。当外部希望通过迭代器操作容器的元素时只需要根据规定的语义访问即可，如果期望了解嗲到期存放的类型等信息可以使用```iterator_traits```获取，这样就保证了内部的实现对外是完全隔离的。

### 5.2 ```iterator traits```
&emsp;&emsp;STL中有6类迭代器：输入迭代器，输出迭代器，前向迭代器，双向迭代器，连续存储迭代器（C++20新增的），随机访问迭代器。
- 输入迭代器：索引对内的内存可读，比如```istream_iterator```；
- 输出迭代器：索引对应的内存可写，比如```ostream_iterator```；
- 前向迭代器：数据可读写，索引的方向只能向前，可以```++```，但是不能向后不支持```--```；
- 双向迭代器：在前向迭代器的基础上，支持```--```；
- 随机访问迭代器：功能完全和指针相同，可以通过随机指针随机访问容器内的元素，比如```iter+n```访问相对当前位置后第```n```个元素。
- 连序存储迭代器：通过该迭代器访问的元素都保证数据存储是连续的。

&emsp;&emsp;下面是libcxx中不同类型迭代器的标识，这些类虽然是空类，但是是不同的类型就可以通过模板推导的方式选择正确的迭代器。
```cpp
struct _LIBCPP_TEMPLATE_VIS input_iterator_tag {};
struct _LIBCPP_TEMPLATE_VIS output_iterator_tag {};
struct _LIBCPP_TEMPLATE_VIS forward_iterator_tag       : public input_iterator_tag {};
struct _LIBCPP_TEMPLATE_VIS bidirectional_iterator_tag : public forward_iterator_tag {};
struct _LIBCPP_TEMPLATE_VIS random_access_iterator_tag : public bidirectional_iterator_tag {};
#if _LIBCPP_STD_VER >= 20
struct _LIBCPP_TEMPLATE_VIS contiguous_iterator_tag    : public random_access_iterator_tag {};
#endif
```

&emsp;&emsp;STL是迭代器类型的类型萃取器，可以根据当前迭代器的类型推断出一些容器迭代器中需要的类型。STL要求所有迭代器都应该提供这五个基本的类型以方便萃取器获取具体的类型，参考[iterator_traits](https://zh.cppreference.com/w/cpp/iterator/iterator_traits)
```cpp
template <class _Iter>
struct __iterator_traits_impl<_Iter, true>
{
    typedef typename _Iter::difference_type   difference_type;
    typedef typename _Iter::value_type        value_type;
    typedef typename _Iter::pointer           pointer;
    typedef typename _Iter::reference         reference;
    typedef typename _Iter::iterator_category iterator_category;
};
//指针的特化版本
struct _LIBCPP_TEMPLATE_VIS iterator_traits<_Tp*>{
    typedef ptrdiff_t difference_type;
    typedef __remove_cv_t<_Tp> value_type;
    typedef _Tp* pointer;
    typedef _Tp& reference;
    typedef random_access_iterator_tag iterator_category;
#if _LIBCPP_STD_VER >= 20
    typedef contiguous_iterator_tag    iterator_concept;
#endif
};
```

#### 5.3 不同迭代器的实现
##### 5.3.1 容器的迭代器
###### 5.3.1.1 ```__wrap_iter```
&emsp;&emsp;```vector```的迭代器时```__wrap_iter<pointer>```就是一个指针的包装器，将指针的一系列操作封装了一遍，也就是```vector```的迭代器就是一个指针没有什么特别之处。而```array```的迭代器直接就是一个```value_type*```类型，就是一个裸指针。
```cpp
template <class _Iter>
class __wrap_iter
{
public:
    typedef _Iter                                                      iterator_type;
    typedef typename iterator_traits<iterator_type>::value_type        value_type;
    typedef typename iterator_traits<iterator_type>::difference_type   difference_type;
    typedef typename iterator_traits<iterator_type>::pointer           pointer;
    typedef typename iterator_traits<iterator_type>::reference         reference;
    typedef typename iterator_traits<iterator_type>::iterator_category iterator_category;

private:
    iterator_type __i_;

    _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __wrap_iter& operator+=(difference_type __n) _NOEXCEPT{
        _LIBCPP_DEBUG_ASSERT(__get_const_db()->__addable(this, __n), "Attempted to add/subtract an iterator outside its valid range");
        __i_ += __n;
        return *this;
    }
};
```
&emsp;&emsp;另外需要注意的是```vector```的```end()```的返回值就是```__end_```指向的位置，该位置是一个未知的地址，只能作为结束位置的标识符不能访问对应的内存。

###### 5.3.1.2 ```list_iterator```
&emsp;&emsp;```list_iterator```的实现是和```list```的实现相关的，是对根据链表的指针访问节点的一个包装，因为链表只能通过当前节点访问前向节点和后向节点，支持随机访问本身意义不大，因此是双向访问节点。实现也比较简单，不详述。
```cpp
template <class _Tp, class _VoidPtr>
class _LIBCPP_TEMPLATE_VIS __list_iterator
{
    typedef __list_node_pointer_traits<_Tp, _VoidPtr> _NodeTraits;
    typedef typename _NodeTraits::__link_pointer __link_pointer;

    __link_pointer __ptr_;

    _LIBCPP_INLINE_VISIBILITY
    explicit __list_iterator(__link_pointer __p, const void* __c) _NOEXCEPT
        : __ptr_(__p){
        (void)__c;
    }

    template<class, class> friend class list;
    template<class, class> friend class __list_imp;
    template<class, class> friend class __list_const_iterator;
public:
    typedef bidirectional_iterator_tag       iterator_category;
    typedef _Tp                              value_type;
    typedef value_type&                      reference;
    typedef __rebind_pointer_t<_VoidPtr, value_type> pointer;
    typedef typename pointer_traits<pointer>::difference_type difference_type;
    __list_iterator& operator++(){
        __ptr_ = __ptr_->__next_;
        return *this;
    }

    __list_iterator operator++(int) {__list_iterator __t(*this); ++(*this); return __t;}

    __list_iterator& operator--(){
        __ptr_ = __ptr_->__prev_;
        return *this;
    }
}
```

###### 5.3.1.3 ```map_iterator```
&emsp;&emsp;```map```STL内部实现是红黑树，而```map_iterator```是一个包装器具体的是实现```__tree_iterator```。比如```operator++```的实现就是访问平衡数的下一个节点。
```cpp
template <class _EndNodePtr, class _NodePtr>
inline _LIBCPP_INLINE_VISIBILITY _EndNodePtr __tree_next_iter(_NodePtr __x) _NOEXCEPT{
    _LIBCPP_ASSERT(__x != nullptr, "node shouldn't be null");
    if (__x->__right_ != nullptr)
        return static_cast<_EndNodePtr>(_VSTD::__tree_min(__x->__right_));
    while (!_VSTD::__tree_is_left_child(__x))
        __x = __x->__parent_unsafe();
    return static_cast<_EndNodePtr>(__x->__parent_);
}
```

&emsp;&emsp;而```unordered_map```的迭代器实现由```__hash_table```提供，实现比较复杂在了解```hash_table```的实现时再详细描述。

##### 5.3.2 迭代器包装器
###### 5.3.2.1 ```reverse_iterator```

&emsp;&emsp;```reverse_iterator```是一个迭代器的包装器，，并不是一个真正的迭代器。```reverse_iterator```可以相对于当前迭代器反方向访问元素，它将当前迭代器的操作方向完全逆转，+变成-，-变成+，不详述，代码比较简单。
```cpp
template <class _Iter>
class _LIBCPP_TEMPLATE_VIS reverse_iterator
#if _LIBCPP_STD_VER <= 14 || !defined(_LIBCPP_ABI_NO_ITERATOR_BASES)
    : public iterator<typename iterator_traits<_Iter>::iterator_category,
                      typename iterator_traits<_Iter>::value_type,
                      typename iterator_traits<_Iter>::difference_type,
                      typename iterator_traits<_Iter>::pointer,
                      typename iterator_traits<_Iter>::reference>
#endif
{
_LIBCPP_SUPPRESS_DEPRECATED_POP
private:
#ifndef _LIBCPP_ABI_NO_ITERATOR_BASES
    _Iter __t_; // no longer used as of LWG #2360, not removed due to ABI break
#endif

#if _LIBCPP_STD_VER >= 20
    static_assert(__is_cpp17_bidirectional_iterator<_Iter>::value || bidirectional_iterator<_Iter>,
        "reverse_iterator<It> requires It to be a bidirectional iterator.");
#endif // _LIBCPP_STD_VER >= 20

protected:
    _Iter current;
        _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX17
    reverse_iterator& operator++() {--current; return *this;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX17
    reverse_iterator operator++(int) {reverse_iterator __tmp(*this); --current; return __tmp;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX17
    reverse_iterator& operator--() {++current; return *this;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX17
    reverse_iterator operator--(int) {reverse_iterator __tmp(*this); ++current; return __tmp;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX17
    reverse_iterator operator+(difference_type __n) const {return reverse_iterator(current - __n);}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX17
    reverse_iterator& operator+=(difference_type __n) {current -= __n; return *this;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX17
    reverse_iterator operator-(difference_type __n) const {return reverse_iterator(current + __n);}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX17
    reverse_iterator& operator-=(difference_type __n) {current += __n; return *this;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX17
    reference operator[](difference_type __n) const {return *(*this + __n);}
};
```

###### 5.3.2.2 ```back_inserter```、```back_insert_iterator```、```insert_iterator```、```front_insert_iterator```、```insert_iterator```和```inserter```
&emsp;&emsp;后向插入迭代器，在给```back_inserter```赋值是就会调用容器的```push_back```函数插入元素。也就是说只有```push_back```接口的容器才能使用```back_inserter```。
```cpp
 _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 back_insert_iterator& operator=(const typename _Container::value_type& __value){
    container->push_back(__value); return *this;
}

template <class _Container>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 back_insert_iterator<_Container>
back_inserter(_Container& __x){
    return back_insert_iterator<_Container>(__x);
}
```

&emsp;&emsp;```insert_iterator```和```front_insert_iterator```通过调用容器的```push_front```函数实现前插。
```cpp
_LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX20 front_insert_iterator& operator=(const typename _Container::value_type& __value)
{container->push_front(__value); return *this;}
```

&emsp;&emsp;```insert_iterator```和```inserter```通过调用```insert```函数来实现插入元素。
```cpp
_LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX20 insert_iterator& operator=(const typename _Container::value_type& __value)
{iter = container->insert(iter, __value); ++iter; return *this;}
```
# 3 参考文献
- [iterator_traits](https://zh.cppreference.com/w/cpp/iterator/iterator_traits)
- [深入理解STL源码(2) 迭代器(Iterators)和Traits](http://ibillxia.github.io/blog/2014/06/21/stl-source-insight-2-iterators-and-traits/)
- [STL迭代器 基础篇](https://zhuanlan.zhihu.com/p/94629092)
- [iterator](https://en.cppreference.com/w/cpp/iterator)
- [stackoverflow——push_back vs emplace_back](https://stackoverflow.com/questions/4303513/push-back-vs-emplace-back)
- [Proposed Wording for Placement Insert](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2642.pdf)
- [string](https://en.cppreference.com/w/cpp/string/basic_string)
- [string_view](https://en.cppreference.com/w/cpp/string/basic_string_view)
- [cppreference-allocator](https://en.cppreference.com/w/cpp/memory/allocator)
- [llvm-review-vstd](https://reviews.llvm.org/D55517)
- [Difference between "destroy" "destructor" "deallocate" in std::allocator?](https://stackoverflow.com/questions/26667026/difference-between-destroy-destructor-deallocate-in-stdallocator)
- [STL源码阅读小记(一)——Allocator](https://blog.csdn.net/ninesnow_c/article/details/121553229)