# C++的lambda表达式原理
## 1 简介
&emsp;&emsp;在 C++11 标准中，引入了 Lambda 表达式这一重要特性，为开发者提供了更简洁、灵活的编程方式。Lambda 表达式，也被称为匿名函数，允许在代码中直接定义一个可调用的代码单元，而无需显式地定义一个命名函数。这一特性极大地增强了 C++ 的表达能力，尤其在函数式编程和算法应用中发挥了重要作用。本文将深入探讨 C++ Lambda 表达式的语法、特性、应用场景以及其背后的实现原理。

&emsp;&emsp;ambda 表达式可以理解为一个匿名的内联函数，它具有一个返回类型、一个参数列表和一个函数体。与普通函数不同的是，Lambda 表达式通常使用尾置返回类型。其基本语法形式如下：
```cpp
[capture list](parameter list) -> return type { function body }
```

- capture list：捕获列表，用于指定 Lambda 表达式所在函数中定义的局部变量列表，这些变量可以在 Lambda 表达式内部使用。​捕获可以进行值捕获，引用捕获，可变捕获，以及前几种的混合捕获。
- parameter list：参数列表，与普通函数的参数列表类似，用于接收调用时传递的参数。​
- return type：返回类型，指定 Lambda 表达式的返回值类型。在某些情况下，编译器可以自动推断返回类型，此时可以省略该部分。​
- function body：函数体，包含了 Lambda 表达式实际执行的代码逻辑。

```cpp
std::vector<int> numbers = {5, 2, 8, 1, 9};
std::sort(numbers.begin(), numbers.end(), [](int a, int b) { return a < b; });
for (int num : numbers) {
    std::cout << num << " "; // 输出：1 2 5 8 9
}
```

## 2 原理
&emsp;&emsp;当编写一个 Lambda 表达式时，编译器会将其转换为一个未命名类的未命名对象，这个类重载了函数调用运算符operator()。例如，考虑以下 Lambda 表达式：
```cpp
int main(){
    auto lambda = [](int a, int b) { return a + b; };
  	lambda(1,2);
}
```

&emsp;&emsp;编译器大致会将其转换为类似如下的代码。可以看到主要就是创建了一个匿名类并且重载类对应的```operator()```函数，同时重载```operator*```。```__invoke```函数提供了一种方式，使得 lambda 表达式可以被转换为函数指针类型。这对于需要将 lambda 作为参数传递给接受函数指针的函数非常重要。例如，某些标准库函数或算法可能需要一个函数指针作为参数。
```cpp
int main()
{
    
  class __lambda_3_19
  {
    public: 
    inline /*constexpr */ int operator()(int a, int b) const
    {
      return a + b;
    }
    
    using retType_3_19 = int (*)(int, int);
    inline constexpr operator retType_3_19 () const noexcept
    {
      return __invoke;
    };
    
    private: 
    static inline /*constexpr */ int __invoke(int a, int b)
    {
      return __lambda_3_19{}.operator()(a, b);
    }
    
    
  };
  
  __lambda_3_19 lambda = __lambda_3_19{};
  lambda.operator()(1, 2);
  return 0;
}
```

**值捕获**
&emsp;&emsp;编译器会为捕获的变量在匿名类中创建相应的数据成员，并在构造函数中使用捕获变量的值进行初始化。例如：
```cpp
int main(){
  	int a = 1;
 	int b = 2;
    auto lambda = [=]() { return a + b; };
  	lambda();
}
```

&emsp;&emsp;对应生成的代码中添加了两个成员，并且在匿名类初始化时初始化这两个成员。
```cpp
int main()
{
  int a = 1;
  int b = 2;
    
  class __lambda_5_19
  {
    public: 
    inline /*constexpr */ int operator()() const
    {
      return a + b;
    }
    
    private: 
    int a;
    int b;
    
    public:
    __lambda_5_19(int & _a, int & _b)
    : a{_a}
    , b{_b}
    {}
    
  };
  
  __lambda_5_19 lambda = __lambda_5_19{a, b};
  lambda.operator()();
  return 0;
}
```
**引用捕获**
&emsp;&emsp;引用捕获同理，区别是成员变成了引用。
```cpp
int main(){
  	int a = 1;
 	int b = 2;
    auto lambda = [&]() { return a + b; };
  	lambda();
}
```

```cpp
int main()
{
  int a = 1;
  int b = 2;
    
  class __lambda_5_19
  {
    public: 
    inline /*constexpr */ int operator()() const
    {
      return a + b;
    }
    
    private: 
    int & a;
    int & b;
    
    public:
    __lambda_5_19(int & _a, int & _b)
    : a{_a}
    , b{_b}
    {}
    
  };
  
  __lambda_5_19 lambda = __lambda_5_19{a, b};
  lambda.operator()();
  return 0;
}
```

## 3 使用lambda需要注意的点
&emsp;&emsp;使用lambda时需要注意
1. 捕获的变量的生命周期，捕获的变量在 lambda 执行时必须有效。如果 lambda 的生命周期超过了捕获变量的有效性，将导致未定义行为。
2. 尽管 lambda 通常是轻量级的，但如果捕获的变量较多或较复杂，可能会导致额外的内存开销。在性能关键的场景中，要注意这一点。
