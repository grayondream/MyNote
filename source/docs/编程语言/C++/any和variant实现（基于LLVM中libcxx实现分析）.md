# any和variant实现（基于LLVM中libcxx实现分析）
&emsp;&emsp;本文根据LLVM中libcxx的实现，分析了std::any和std::variant的具体实现。
## 1 std::any
### 1.1 简介
&emsp;&emsp;在 C++17 标准中，```std::any``` 提供了一种类型安全的方式来存储任意类型的值。它使用类型擦除（type erasure）技术实现，使得一个对象可以包含任何类型的值而不需要提前知道该类型。```std::any```的使用比较简单。
```cpp
#include <any>
#include <iostream>
 
int main()
{
    std::cout << std::boolalpha;
 
    // any type
    std::any a = 1;
    std::cout << a.type().name() << ": " << std::any_cast<int>(a) << '\n';
    a = 3.14;
    std::cout << a.type().name() << ": " << std::any_cast<double>(a) << '\n';
    a = true;
    std::cout << a.type().name() << ": " << std::any_cast<bool>(a) << '\n';
 
    // bad cast
    try
    {
        a = 1;
        std::cout << std::any_cast<float>(a) << '\n';
    }
    catch (const std::bad_any_cast& e)
    {
        std::cout << e.what() << '\n';
    }
 
    // has value
    a = 2;
    if (a.has_value())
        std::cout << a.type().name() << ": " << std::any_cast<int>(a) << '\n';
 
    // reset
    a.reset();
    if (!a.has_value())
        std::cout << "no value\n";
 
    // pointer to contained data
    a = 3;
    int* i = std::any_cast<int>(&a);
    std::cout << *i << '\n';
}
```

### 1.2 实现
**内存处理**
&emsp;&emsp;```std::any```的实现简单的说来就是通过一个指针存储数据和一个额外的类型信息来保留类型。
```cpp
class any{
  union _Storage {
    _LIBCPP_HIDE_FROM_ABI constexpr _Storage() : __ptr(nullptr) {}
    void* __ptr;
    __any_imp::_Buffer __buf;
  };

  _HandleFuncPtr __h_ = nullptr;
  _Storage __s_;
};
```

&emsp;&emsp;```__h_```存储不同情况对应处理的函数指针信息，```__s_```存储具体的内存，根据不同情况使用不同的存储方式。和常规的SSO优化类似小于某个大小的内存直接存储在栈上，否则用堆。这里的大小是对齐到3倍机器字长。
```cpp
using _Buffer = aligned_storage_t<3 * sizeof(void*), alignof(void*)>;
```

&emsp;&emsp;不同大小的对象使用不同的handler对数据进行操作，栈内存使用```SmallHandle```，堆内存使用```LargeHandle```，二者唯一的区别只是操作的内存不同，一个操作```ptr```,一个操作```buf```
```cpp
template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS _SmallHandler {
  _LIBCPP_HIDE_FROM_ABI static void*
  __handle(_Action __act, any const* __this, any* __other, type_info const* __info, const void* __fallback_info) {
    switch (__act) {
    case _Action::_Destroy:
      __destroy(const_cast<any&>(*__this));
      return nullptr;
    case _Action::_Copy:
      __copy(*__this, *__other);
      return nullptr;
    case _Action::_Move:
      __move(const_cast<any&>(*__this), *__other);
      return nullptr;
    case _Action::_Get:
      return __get(const_cast<any&>(*__this), __info, __fallback_info);
    case _Action::_TypeInfo:
      return __type_info();
    }
    __libcpp_unreachable();
  }

private:
  _LIBCPP_HIDE_FROM_ABI static void __destroy(any& __this) {
    typedef allocator<_Tp> _Alloc;
    typedef allocator_traits<_Alloc> _ATraits;
    _Alloc __a;
    _Tp* __p = static_cast<_Tp*>(static_cast<void*>(&__this.__s_.__buf));
    _ATraits::destroy(__a, __p);
    __this.__h_ = nullptr;
  }
};
//而对应的largeHandle的销毁不仅仅要释放对象还需要销毁内存
template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS _LargeHandler {
_LIBCPP_HIDE_FROM_ABI static void __destroy(any& __this) {
    typedef allocator<_Tp> _Alloc;
    typedef allocator_traits<_Alloc> _ATraits;
    _Alloc __a;
    _Tp* __p = static_cast<_Tp*>(__this.__s_.__ptr);
    _ATraits::destroy(__a, __p);
    _ATraits::deallocate(__a, __p, 1);
    __this.__h_ = nullptr;
  }
};
```
&emsp;&emsp;具体使用哪种类型的handle则是在构造时根据类型确定的，```_IsSmallObject```用来判断是否是小对象，小对象则使用```_SmallHandler```,否则使用```_LargeHandler```。
```cpp
template <class _Tp>
using _IsSmallObject =
    integral_constant<bool,
                      sizeof(_Tp) <= sizeof(_Buffer) && alignof(_Buffer) % alignof(_Tp) == 0 &&
                          is_nothrow_move_constructible<_Tp>::value >;

template <class _Tp>
using _Handler = conditional_t< _IsSmallObject<_Tp>::value, _SmallHandler<_Tp>, _LargeHandler<_Tp>>;


template <class _ValueType, class _Tp, class>
any::any(_ValueType&& __v) : __h_(nullptr) {
  __any_imp::_Handler<_Tp>::__create(*this, std::forward<_ValueType>(__v));
}
```

**类型信息**
&emsp;&emsp;对于开启了RTTI的场景，比较简单直接返回当前对象的typeinfo即可。
```cpp
  _LIBCPP_HIDE_FROM_ABI static void* __type_info() {
#  if !defined(_LIBCPP_HAS_NO_RTTI)
    return const_cast<void*>(static_cast<void const*>(&typeid(_Tp)));
#  else
    return nullptr;
#  endif
  }
```

### 1.3 自己实现
&emsp;&emsp;说实话LLVM的实现感觉很丑，这里通过继承来实现类型擦除会好看很多（实现其实不全，拷贝构造等都没有实现，但是基本功能是OK的）。
```cpp
#include <iostream>
#include <exception>
#include <type_traits>
#include <memory>
#include <utility>
class AnyCastError : public std::exception {
public:
    const char* what() const noexcept override {
        return "Bad cast in Any";
    }
};

template<class T>
class AnyImpl {
public:
    using Buffer = std::aligned_storage_t<3 * sizeof(void*), alignof(void*)>;
    using IsSmallTrivialObject = std::integral_constant<bool, sizeof(T) <= sizeof(Buffer)&&
        alignof(Buffer) % alignof(T) == 0 &&
        std::is_nothrow_move_constructible<T>::value >;
                                                        
public:
    union Storage {
        void* ptr;
        Buffer buffer;

        Storage() : ptr(nullptr) {}
        ~Storage() {}
    };
};

struct HolderBase {
public:
    virtual ~HolderBase() = default;
    virtual std::unique_ptr<HolderBase> clone() const = 0;
    virtual const std::type_info *typeInfo() const = 0;
};

template<class T>
struct Holder : public HolderBase {
public:
    Holder(T&& value) {
        if constexpr (AnyImpl<T>::IsSmallTrivialObject::value) {
            new(&_storage.buffer)T(std::move(value));
        }
        else {
            _storage.ptr = new T(std::move(value));
        }
    }

    ~Holder() {
        if constexpr (AnyImpl<T>::IsSmallTrivialObject::value) {
            reinterpret_cast<T*>(&_storage.buffer)->~T();
        }
        else {
            delete static_cast<T*>(_storage.ptr);
        }
    }

    virtual std::unique_ptr<HolderBase> clone() const override {
        return std::make_unique<Holder<T>>(getValue());
    }

    T getValue() const {
        if constexpr (AnyImpl<T>::IsSmallTrivialObject::value) {
            return *reinterpret_cast<const T*>(&_storage.buffer);
        }
        else {
            return *static_cast<T*>(_storage.ptr);
        }
    }

    virtual const std::type_info* typeInfo() const override {
        return &typeid(T);
    }

public:
    AnyImpl<T>::Storage _storage;
};

class Any {
public:
    Any() {
        _holder = nullptr;
    }

    template<class T>
    Any(T&& v) {
        _holder = std::make_unique<Holder<T>>(std::forward<T>(v));
    }

    bool hasValue() const {
        return !!_holder;
    }

    template<class T>
    T getValue() {
        return hasValue() ? static_cast<Holder<T>*>(_holder.get())->getValue() : T();
    }

    const std::type_info* typeInfo() const {
        return hasValue() ? _holder->typeInfo() : &typeid(int);
    }
private:
    std::unique_ptr<HolderBase> _holder{};
};

int main(int argc, char **argv){
    try {
        Any a = 42; // 存储 int
        std::cout << "Value: " << a.getValue<int>() << ", Type: " << a.typeInfo()->name() << std::endl;

        Any b = std::string("Hello"); // 存储 string
        std::cout << "Value: " << b.getValue<std::string>() << ", Type: " << b.typeInfo()->name() << std::endl;

        // 测试未存储值的情况
        Any emptyAny;
        std::cout << "Has Value: " << emptyAny.hasValue() << std::endl;
        std::cout << "Type Info: " << emptyAny.typeInfo()->name() << std::endl;

    }
    catch (const AnyCastError& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
```
## 2 std::variant