# ```magic_enum```实现原理

## 1 简介
&emsp;&emsp;```magic_enum```是一个非常好用的枚举库，它可以将枚举值转换为字符串，也可以将字符串转换为枚举值。
- [magic_enum](https://github.com/Neargye/magic_enum)

>&emsp;&emsp;Header-only C++17 library provides static reflection for enums, work with any enum type without any macro or boilerplate code.

&emsp;&emsp;```magic_enum```的使用方式比较简单：
- 将枚举值转换为字符串
```cpp
enum class Color { Red, Green, Blue};
std::string str = magic_enum::enum_name(Color::Red); // "Red"
```
- 将字符串转换为枚举值
```cpp
Color color = magic_enum::enum_cast<Color>("Red").value(); // Color::Red
```
- 获取枚举值的数量
```cpp
size_t size = magic_enum::enum_count<Color>(); // 3
```
- 获取枚举值的所有值
```cpp
std::vector<Color> colors = magic_enum::enum_values<Color>(); // {Color::Red, Color::Green, Color::Blue}
```
- 获取枚举值的所有名称等。
```cpp
std::vector<std::string> names = magic_enum::enum_names<Color>(); // {"Red", "Green", "Blue"}
```

## 2 原理
### 2.1 获取类型字符串
&emsp;&emsp;```magic_enum```是编译期解析编译器的函数签名来实现的。不同编译器都有预定义的宏，该宏可以用来获取函数的参数类型、返回类型等信息。如果模板参数有枚举类型，那么就可以通过解析该签名获取具体的枚举类型。

&emsp;&emsp;比如对于枚举类型```Type```，其对应的签名为：
```cpp
enum class Type : int { None, Value};
template<auto V>
constexpr std::string_view sig() {
    return __FUNCSIG__;
}
std::cout << sig<Type::None>() << std::endl;
```

&emsp;&emsp;上面的输出为：
```
class std::basic_string_view<char,struct std::char_traits<char> > __cdecl sig<Type::None>(void)
```

&emsp;&emsp;可以看到，该签名中包含了枚举类型```Type```的信息，我们就可以通过解析该字符串拿到具体的枚举类型。另外，上面的示例中的```__FUNCSIG__```是VC++编译器的宏，其他编译器的宏可能不同，比如clang的宏为```__PRETTY_FUNCTION__```，gcc的宏也为```__PRETTY_FUNCTION__```。

&emsp;&emsp;而对应到```magic_enum```的实现中，就是通过解析函数签名来获取枚举类型的。
```cpp
template <typename E>
constexpr auto n() noexcept {
  static_assert(is_enum_v<E>, "magic_enum::detail::n requires enum type.");

  if constexpr (supported<E>::value) {
#if defined(MAGIC_ENUM_GET_TYPE_NAME_BUILTIN)
    constexpr auto name_ptr = MAGIC_ENUM_GET_TYPE_NAME_BUILTIN(E);
    constexpr auto name = name_ptr ? str_view{name_ptr, std::char_traits<char>::length(name_ptr)} : str_view{};
#elif defined(__clang__)
    str_view name;
    if constexpr (sizeof(__PRETTY_FUNCTION__) == sizeof(__FUNCTION__)) {
      static_assert(always_false_v<E>, "magic_enum::detail::n requires __PRETTY_FUNCTION__.");
      return str_view{};
    } else {
      name.size_ = sizeof(__PRETTY_FUNCTION__) - 36;
      name.str_ = __PRETTY_FUNCTION__ + 34;
    }
#elif defined(__GNUC__)
    auto name = str_view{__PRETTY_FUNCTION__, sizeof(__PRETTY_FUNCTION__) - 1};
    if constexpr (sizeof(__PRETTY_FUNCTION__) == sizeof(__FUNCTION__)) {
      static_assert(always_false_v<E>, "magic_enum::detail::n requires __PRETTY_FUNCTION__.");
      return str_view{};
    } else if (name.str_[name.size_ - 1] == ']') {
      name.size_ -= 50;
      name.str_ += 49;
    } else {
      name.size_ -= 40;
      name.str_ += 37;
    }
#elif defined(_MSC_VER)
    // CLI/C++ workaround (see https://github.com/Neargye/magic_enum/issues/284).
    str_view name;
    name.str_ = __FUNCSIG__;
    name.str_ += 40;
    name.size_ += sizeof(__FUNCSIG__) - 57;
#else
    auto name = str_view{};
#endif
    //省略部分代码
}
```

### 2.2 字符串到枚举类型
&emsp;&emsp;字符串到枚举类型的转换比较简单，就是通过遍历枚举类型的所有值，然后比较字符串是否相等，如果相等就返回对应的枚举值。比如```magic_enum```的实现中，就是通过遍历枚举类型的所有值，然后比较字符串是否相等，如果相等就返回对应的枚举值。
```cpp
template <typename E, detail::enum_subtype S = detail::subtype_v<E>>
[[nodiscard]] constexpr auto enum_name(E value) noexcept -> detail::enable_if_t<E, string_view> {
  using D = std::decay_t<E>;
  static_assert(detail::is_reflected_v<D, S>, "magic_enum requires enum implementation and valid max and min.");

  if (const auto i = enum_index<D, S>(value)) {
    return detail::names_v<D, S>[*i];
  }
  return {};
}
```

&emsp;&emsp;那么如何获取o到枚举类型的所有值呢？```magic_enum```的实现中，就是预定义一个最大值和最小值，然后通过遍历枚举类型的所有值，然后比较枚举值是否在最大值和最小值之间，如果在就返回对应的枚举值。
```cpp
// Enum value must be greater or equals than MAGIC_ENUM_RANGE_MIN. By default MAGIC_ENUM_RANGE_MIN = -128.
// If need another min range for all enum types by default, redefine the macro MAGIC_ENUM_RANGE_MIN.
#if !defined(MAGIC_ENUM_RANGE_MIN)
#  define MAGIC_ENUM_RANGE_MIN -128
#endif

// Enum value must be less or equals than MAGIC_ENUM_RANGE_MAX. By default MAGIC_ENUM_RANGE_MAX = 127.
// If need another max range for all enum types by default, redefine the macro MAGIC_ENUM_RANGE_MAX.
#if !defined(MAGIC_ENUM_RANGE_MAX)
#  define MAGIC_ENUM_RANGE_MAX 127
#endif
```

&emsp;&emsp;比如预定义最大值为3，最小值为-2，强制转换后的输出为，能够看到我们很容易从下面的输出中获取到合法的枚举值的所有值。之后的事情就比较简单了，就是比较字符串是否相等，如果相等就返回对应的枚举值。
```cpp
Type)0xfffffffffffffffe
Type)0xffffffffffffffff
None
Value
Type)0x2
```

## 3 简化版本
&emsp;&emsp;```magic_enum```的实现比较复杂，但是我们可以通过简化版本来实现类似的功能。比如我们可以通过预定义一个最大值和最小值，然后通过遍历枚举类型的所有值，然后比较枚举值是否在最大值和最小值之间，如果在就返回对应的枚举值。
&emsp;&emsp;下面的实现是MSVC版本，其他编译器只需要替换对应的宏和索引即可。
```cpp
#include <string>
#include <iostream>
#include <string_view>
#include <array>

constexpr std::size_t kEnumMaxIndex = 3;
constexpr std::size_t kEnumMinIndex = -2;

constexpr auto EnumDefaultSize() {
    return kEnumMaxIndex - kEnumMinIndex;
}

template<auto V>
constexpr std::string_view sig() {
    return __FUNCSIG__;
}

template<auto V>
constexpr std::string_view EnumName() {
    auto str = sig<V>();
    return std::string_view(str.data() + 84, str.size() - 91);
}

template<class T>
constexpr auto EnumNames() {
    constexpr auto sz = EnumDefaultSize();
    std::array<std::string_view, sz> arr;
    [&arr] <std::size_t... I>(std::index_sequence<I...>) {
        ((arr[I] = EnumName<static_cast<T>(I + kEnumMinIndex)>()), ...);
    }(std::make_index_sequence<sz>{});

    return arr;
}

template<typename T>
constexpr std::string_view EnumName(const T v) {
    return EnumNames<T>()[static_cast<std::size_t>(v) - kEnumMinIndex];
}

enum class Type : int {
    None,
    Value
};

int main(int argc, char** argv) {
    auto v = Type::None;
    std::cout << sig<Type::None>() << std::endl;
    std::cout << EnumName<Type::None>() << std::endl;
    std::cout << EnumName(v)<<std::endl;
    auto arr = EnumNames<Type>();
    std::cout << std::endl;
    for (auto i = 0; i < arr.size(); i++) {
        std::cout << arr[i] << std::endl;
    }

    return 0;
}
```