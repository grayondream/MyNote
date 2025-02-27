
# C++静态传递

## 1 反射
&emsp;&emsp;反射是一种强大的工具，它可以在运行时检查对象的类型和属性。反射可以分为动态反射和静态反射。

&emsp;&emsp;静态反射是指在编译时就可以检查对象的类型和属性，而不需要在运行时进行反射。静态反射可以通过模板元编程来实现。静态反射主要发生在编译期间，编译器会根据模板参数的类型来生成对应的代码，无法运行时检查。

&emsp;&emsp;动态反射是指在运行时可以检查对象的类型和属性，而不需要在编译时进行反射。动态反射可以通过运行时类型信息来实现。

&emsp;&emsp;静态反射和动态反射的区别在于，静态反射是在编译时进行的，而动态反射是在运行时进行的。静态反射的优点是性能较高，缺点是灵活性较低；动态反射的优点是灵活性较高，缺点是性能较低。

## 2 静态反射
&emsp;&emsp;静态反射实现比较简单，就是在对象构造时将相关的元数据记录下来，然后在需要的时候进行查询。通常的实现方式是通过宏和模板元来实现，避免大量重复代码。比如：
```cpp
enum class Type{
    INT,
    STRING,
    FLOAT,
};

struct TypeInfo{
    std::string name{};
    enum Type type{}; 
    int offset{};
};

struct Person{
public:

private:
    std::string name;
    int age;
    std::unordered_map<std::string, TypeInfo> _memInfo{};
};


```