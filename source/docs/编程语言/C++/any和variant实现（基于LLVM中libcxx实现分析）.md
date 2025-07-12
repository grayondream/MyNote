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
### 2.1 简介
&emsp;&emsp;```std::variant```是C++17引入的一个类模板，用于在编译时存储不同类型的值。它的实现基于类型擦除和联合体，能够存储任意指定的类型，并且在访问时可以自动进行类型检查和转换。
```cpp
#include <cassert>
#include <iostream>
#include <string>
#include <variant>
 
int main()
{
    std::variant<int, float> v, w;
    v = 42; // v contains int
    int i = std::get<int>(v);
    assert(42 == i); // succeeds
    w = std::get<int>(v);
    w = std::get<0>(v); // same effect as the previous line
    w = v; // same effect as the previous line
 
//  std::get<double>(v); // error: no double in [int, float]
//  std::get<3>(v);      // error: valid index values are 0 and 1
 
    try
    {
        std::get<float>(w); // w contains int, not float: will throw
    }
    catch (const std::bad_variant_access& ex)
    {
        std::cout << ex.what() << '\n';
    }
 
    using namespace std::literals;
 
    std::variant<std::string> x("abc");
    // converting constructors work when unambiguous
    x = "def"; // converting assignment also works when unambiguous
 
    std::variant<std::string, void const*> y("abc");
    // casts to void const* when passed a char const*
    assert(std::holds_alternative<void const*>(y)); // succeeds
    y = "xyz"s;
    assert(std::holds_alternative<std::string>(y)); // succeeds
}
```

### 2.2 实现
&emsp;&emsp;简单看了下llvm的实现，看的头疼就去看msvc的实现发现msvc的实现了，就不折磨自己了，下面的分析来自与msvc。首先是对象存储，对象存储是使用union实现的，而为什么不在charbuffer上使用placement new是因为c++17不支持constexpr，该特性c++20才支持。

&emsp;&emsp;能够看到Variant_storage中使用union递归来存储对象，并且使用了constexpr来确保在编译时就能够确定对象的大小。这里的代码是trivial对象的代码实现，对于非trivial的代码区别是添加了一个显式的构造函数来确保对象能够正常析构。
```cpp
template <class _First, class... _Rest>
class _Variant_storage_<true, _First, _Rest...> { // Storage for variant alternatives (trivially destructible case)
public:
    static constexpr size_t _Size = 1 + sizeof...(_Rest);
    union {
        remove_cv_t<_First> _Head;
        _Variant_storage<_Rest...> _Tail;
    };

    _CONSTEXPR20 _Variant_storage_() noexcept {} // no initialization (no active member)

    template <class... _Types>
    constexpr explicit _Variant_storage_(integral_constant<size_t, 0>, _Types&&... _Args) noexcept(
        is_nothrow_constructible_v<_First, _Types...>)
        : _Head(static_cast<_Types&&>(_Args)...) {} // initialize _Head with _Args...

    template <size_t _Idx, class... _Types, enable_if_t<(_Idx > 0), int> = 0>
    constexpr explicit _Variant_storage_(integral_constant<size_t, _Idx>, _Types&&... _Args) noexcept(
        is_nothrow_constructible_v<_Variant_storage<_Rest...>, integral_constant<size_t, _Idx - 1>, _Types...>)
        : _Tail(integral_constant<size_t, _Idx - 1>{}, static_cast<_Types&&>(_Args)...) {} // initialize _Tail (recurse)

    _NODISCARD constexpr _First& _Get() & noexcept {
        return _Head;
    }
    _NODISCARD constexpr const _First& _Get() const& noexcept {
        return _Head;
    }
    _NODISCARD constexpr _First&& _Get() && noexcept {
        return _STD move(_Head);
    }
    _NODISCARD constexpr const _First&& _Get() const&& noexcept {
        return _STD move(_Head);
    }
};
```

&emsp;&emsp;元素获取也是通过递归实现的，只不过好粗暴啊。
```cpp
template <size_t _Idx, class _Storage>
_NODISCARD constexpr decltype(auto) _Variant_raw_get(_Storage&& _Obj) noexcept {
    // access the _Idx-th element of a _Variant_storage
    if constexpr (_Idx == 0) {
        return static_cast<_Storage&&>(_Obj)._Get();
    } else if constexpr (_Idx == 1) {
        return static_cast<_Storage&&>(_Obj)._Tail._Get();
    } else if constexpr (_Idx == 2) {
        return static_cast<_Storage&&>(_Obj)._Tail._Tail._Get();
    } else if constexpr (_Idx == 3) {
        return static_cast<_Storage&&>(_Obj)._Tail._Tail._Tail._Get();
    } else if constexpr (_Idx == 4) {
        return static_cast<_Storage&&>(_Obj)._Tail._Tail._Tail._Tail._Get();
    } else if constexpr (_Idx == 5) {
        return static_cast<_Storage&&>(_Obj)._Tail._Tail._Tail._Tail._Tail._Get();
    } else if constexpr (_Idx == 6) {
        return static_cast<_Storage&&>(_Obj)._Tail._Tail._Tail._Tail._Tail._Tail._Get();
    } else if constexpr (_Idx == 7) {
        return static_cast<_Storage&&>(_Obj)._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Get();
    } else if constexpr (_Idx < 16) {
        return _STD _Variant_raw_get<_Idx - 8>(
            static_cast<_Storage&&>(_Obj)._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail);
    } else if constexpr (_Idx < 32) {
        return _STD _Variant_raw_get<_Idx - 16>(
            static_cast<_Storage&&>(_Obj)
                ._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail);
    } else if constexpr (_Idx < 64) {
        return _STD _Variant_raw_get<_Idx - 32>(
            static_cast<_Storage&&>(_Obj)
                ._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail
                ._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail);
    } else { // _Idx >= 64
        return _STD _Variant_raw_get<_Idx - 64>(
            static_cast<_Storage&&>(_Obj)
                ._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail
                ._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail
                ._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail
                ._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail._Tail);
    }
}
```

&emsp;&emsp;存储方式搞清楚了，下来看一下如何获取元素，获取元素有两种方式，第一种是根据类型获取，第二种是根据索引获取。
```cpp
_EXPORT_STD template <size_t _Idx, class... _Types>
_NODISCARD constexpr auto get_if(variant<_Types...>* _Ptr) noexcept {
    // get the address of *_Ptr's contained value if it holds alternative _Idx
    static_assert(_Idx < sizeof...(_Types), "variant index out of bounds");
    return _Ptr && _Ptr->index() == _Idx ? _STD addressof(_STD _Variant_raw_get<_Idx>(_Ptr->_Storage())) : nullptr;
}
```
&emsp;&emsp;最后是析构，比较简单，单纯的递归搜索析构。
```cpp
 _CONSTEXPR20 void _Destroy() noexcept { // destroy the contained value, if any
     if constexpr (!conjunction_v<is_trivially_destructible<_Types>...>) {
         _STD _Variant_raw_visit(index(), _Storage(), [](auto _Ref) noexcept {
             if constexpr (decltype(_Ref)::_Idx != variant_npos) {
                 using _Indexed_value_type = _Remove_cvref_t<decltype(_Ref._Val)>;
                 _Ref._Val.~_Indexed_value_type();
             }
         });
     }
 }
```

### 2.3 自己实现
&emsp;&emsp;参考上面的代码，我们自己实现一个，原理比较简单就是递归union，由于递归union无法处理析构问题所以需要我们自己处理。其实还有一种实现方式是通过buffer实现，但是该实现无法实现constexpr版本，因此推荐使用union实现。
```cpp
#include <iostream>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <variant>


template <typename... Ts>
struct VariantStorage {
    // ���ػ�������û�и������͵����
};

template <typename Head, typename... Ts>
struct VariantStorage<Head, Ts...> {
    union {
        std::remove_cv_t<Head> value;
        VariantStorage<Ts...> next;
    };

    constexpr VariantStorage() {};
    constexpr ~VariantStorage() noexcept {}
    constexpr VariantStorage(VariantStorage&&) = default;
    constexpr VariantStorage(const VariantStorage&) = default;
    constexpr VariantStorage& operator=(VariantStorage&&) = default;
    constexpr VariantStorage& operator=(const VariantStorage&) = default;
   
};

// �ػ�������ֹ�ݹ�
template<size_t index, typename Head, typename ...Ts>
constexpr auto& get(VariantStorage<Head, Ts...>& v) {
    if constexpr (index == 0) {
        return v.value;
    }else {
        return get<index - 1>(v.next);
    }
}

template <typename T>
constexpr T& get(const VariantStorage<>&) {
    throw std::bad_variant_access();  // �׳��쳣
}

template<typename T, typename Head, typename ...Ts>
constexpr auto& get(VariantStorage<Head, Ts...>& v) {
    using Type = std::remove_cvref_t<T>;
    if constexpr (std::is_same_v<Type, Head>) {
        return v.value;
    }else {
        return get<T>(v.next);
    }
}

template<class T, typename Head, typename... Ts>
constexpr void assign(VariantStorage<Head, Ts...>& v, T&& vv) {
    using Type = std::remove_cvref_t<T>;
    if constexpr (std::is_same_v<T, Head>) {
        *new(&v.value)Type = std::forward<T>(vv);
    }else {
        assign(v.next, std::forward<T>(vv));
    }
}

template <size_t id, typename Head, typename... Ts>
constexpr void destroy(VariantStorage<Head, Ts...>& v) {
    if constexpr (id == 0) {
        v.value.~Head();
    }
    else {
        destroy<id - 1>(v._next);
    }
}

template <typename... Ts>
constexpr bool is_all_trivial_v = false;

template <>
constexpr bool is_all_trivial_v<> = true;

template <typename T, typename... Ts>
constexpr bool is_all_trivial_v<T, Ts...> = std::is_trivial_v<T> && is_all_trivial_v<Ts...>;

constexpr static auto kBadIndex = ~((size_t)0);

template<size_t index, typename U, typename T, typename ...Ts>
struct FindPosition {
    constexpr static size_t value = std::is_same_v<U, T> ? index : FindPosition<index + 1, U, Ts...>::value;
};

template<size_t index, typename U, typename T>
struct FindPosition<index, U, T> {
    constexpr static auto value = std::is_same_v<T, U> ? index : kBadIndex;
};

template<typename U, typename... Ts>
constexpr auto variant_get_index = FindPosition<0, U, Ts...>::value;

template <typename... Ts>
struct Variant {
    constexpr static auto is_all_trivial = is_all_trivial_v<Ts...>;
    constexpr static auto size = sizeof...(Ts);

    Variant() {};
    Variant(const Variant<Ts...>& rhs) = default;
    Variant(Variant<Ts...>&& rhs) = default;
    Variant& operator=(const Variant<Ts...>& rhs) = default;
    Variant& operator=(Variant<Ts...>&& rhs) = default;
    ~Variant() {
        if constexpr (is_all_trivial) {
            destroy(storage);
            index = kBadIndex;
        }
    }
    template <typename T>
        requires(!std::is_same_v<Variant, std::remove_cvref_t<T>>) Variant(T&& rhs) {
        assign(storage, std::forward<T>(rhs));
        index = variant_get_index<T, Ts...>;
    }

    template <typename T>
        requires(!std::is_same_v<Variant, std::remove_cvref_t<T>>)
    auto& operator=(T&& rhs) {
        destroy<index>(storage);
        assign(storage, std::forward<T>(rhs));
        index = variant_get_index<T, Ts...>;
        return *this;
    }

    size_t index;
    VariantStorage<Ts...> storage;
};

template<size_t index, typename ...Ts>
constexpr auto& get(Variant<Ts...>& v) {
    return get<index>(v.storage);
}

template<typename T, typename ...Ts>
constexpr auto& get(Variant<Ts...>& v) {
    return get<T>(v.storage);
}

struct ClassA {
public:
    ~ClassA() {
        printf("destro\n");
    }
};

// ʾ��ʹ��
int main() {
    VariantStorage<int, double, std::string> v;
    static_assert(std::is_same_v<int, decltype(v.value)>);
    static_assert(std::is_same_v<double, decltype(v.next.value)>);
    static_assert(std::is_same_v<std::string, decltype(v.next.next.value)>);

    get<0>(v) = 42;
    std::cout << get<0>(v) << "\n";

    assign(v, std::string("avc"));
    std::cout << get<std::string>(v) << "\n";

    std::cout << variant_get_index<std::string, std::string, int, double> << std::endl;
    std::cout << variant_get_index<int, std::string, int, double> << std::endl;
    std::cout << variant_get_index<double, std::string, int, double> << std::endl;
    std::cout << variant_get_index<char, std::string, int, double> << std::endl;

    Variant<int, double, std::string> a1 = 2.2;
    std::cout << get<double>(a1) << std::endl;
    std::cout << get<1>(a1) << std::endl;

    Variant<int, ClassA> a2 = ClassA();

    return 0;
}
```