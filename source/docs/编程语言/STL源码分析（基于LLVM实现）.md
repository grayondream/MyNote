


## 1 ```allocator```
### 1 ```allocator```
#### 1.1 简介
&emsp;&emsp;```allocator```是STL中对一个堆内存分配器，是对内存申请工作的一个封装，将内存的申请和成员的构造抽象开来方便控制。基本上，C++标准库中的容器的默认分配器都是```allocator```。在C++中分配器是通过模板参数的方式指定给对应的容器，默认就是```allocator```，用户自己也可以实现自己的内存管理类，对堆的内存进行有效的管理也可以将对应的分配器指定给容器使用（前提是接口保持一致）。
```cpp
    std::allocator<int> alloc1;
    // demonstrating the few directly usable members
    static_assert(std::is_same_v<int, decltype(alloc1)::value_type>);
    int* p1 = alloc1.allocate(1); // space for one int
    alloc1.deallocate(p1, 1);     // and it is gone
```

#### 1.2 ```allocator```的实现
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
### 2 ```allocator```特化版本
#### 2.1 ```const allocator```
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

#### 2.2 ```void allocator```
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

### 3 类型萃取```allocator_traits```
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
### 3 参考文献
- [cppreference-allocator](https://en.cppreference.com/w/cpp/memory/allocator)
- [llvm-review-vstd](https://reviews.llvm.org/D55517)
- [Difference between "destroy" "destructor" "deallocate" in std::allocator?](https://stackoverflow.com/questions/26667026/difference-between-destroy-destructor-deallocate-in-stdallocator)
- [STL源码阅读小记(一)——Allocator](https://blog.csdn.net/ninesnow_c/article/details/121553229)