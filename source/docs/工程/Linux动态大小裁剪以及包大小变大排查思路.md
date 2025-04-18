# Linux动态大小裁剪以及包大小变大排查思路

## 1 动态库裁剪
&emsp;&emsp;库分为动态库和静态库，动态库是在程序运行时才加载，静态库是在编译时就加载到程序中。动态库的大小通常比静态库小，因为动态库只包含了程序需要的函数和数据，而静态库则包含了所有的函数和数据。静态库可以理解为引入源码编译，链接器在链接过程中会自动分析需要可不需要的代码进行删除裁剪。因此静态库不存在包大小问题（除了特定平台生成静态库过大导致无法生成库文件的问题）。

&emsp;&emsp;动态库裁剪的思路很简单：
1. 通过工具或者编译选项删除不必要的数据和代码；
2. 只导出需要的函数和数据；
3. 关闭不必要的语言特性，如C++的异常处理等；
4. 优化代码，比如能用```constexpr```实现的尽量用```constexpr```实现；

### 1.1 代码层面
&emsp;&emsp;首先代码层面，需要尽可能确保不同模块之间的耦合度低，避免出现循环依赖的情况。其次，需要尽可能减少代码的重复，避免出现冗余代码的情况。最后，需要尽可能减少代码的复杂度，避免出现复杂的算法和数据结构的情况。对于一些能够用```constexpr```实现的功能，尽量用```constexpr```实现，这样可以减少动态库的大小。

&emsp;&emsp;C++中容易导致C++膨胀的代码：
1. 模板函数和模板类。模板函数和模板类在实例化时都会有一个对应版本的实例，如果任何函数都通过编译器的默认推导来实例化很容易导致膨胀。因此模板函数和模板类应该尽量避免使用默认推导，尽可能显示推导能减少实例化版本。因此可以使用类型擦除和显示实例化来解决模板膨胀的问题。
2. 内联函数。内联函数在编译时会被展开，因此内联函数的代码会被复制到调用处，这样会导致代码膨胀。因此内联函数应该尽量避免使用，除非函数的代码量很小。但是这一条对于现代C++ inline的含义已经发生了变化，inline优化基本完全由C++编译器自动优化。
3. 宏。宏在编译时会被替换，因此宏的代码会被复制到调用处，这样会导致代码膨胀。因此宏应该尽量避免使用，除非宏的代码量很小。
4. 异常处理。异常处理会导致代码膨胀，因为异常处理需要在运行时进行，因此异常处理会导致代码膨胀。因此异常处理应该尽量避免使用，除非异常处理的代码量很小。异常处理通常需要存储异常栈回溯相关的信息，因此容易导致代码膨胀。
5. RTTI。RTTI 允许在运行时获取对象的类型信息。 RTTI 需要在代码中插入额外的类型信息，这会增加二进制文件的大小。
6. 虚函数表。虚函数表是一个指针数组，它包含了虚函数的地址。虚函数表需要在运行时进行查找，这会增加二进制文件的大小。但是一般情况下，虚函数表的大小是固定的，因此虚函数表的大小并不是二进制膨胀的主要原因。

### 1.2 编译选项
&emsp;&emsp;通过编译选项可以控制编译器的行为，从而控制编译过程中的优化和裁剪。编译选项通常是通过编译器的命令行参数来设置的。常用的降低二进制大小的编译选项有：
1. 优化等级，在编译动态库时，使用 -O2 或 -O3 优化级别。 这些优化级别可以使编译器生成更紧凑的代码，从而减小动态库的大小。或者使用```-Os```之类平衡性能和大小的选项。
2. 代码裁剪。
   1. ```-function-sections```：将每个函数放入单独的代码段。
   2. ```-gc-sections```：在链接时删除未使用的代码段。
   3. ```-Wl,--gc-sections```：在链接时删除未使用的代码段。
3. LTO。使用链接时优化（Link-Time Optimization, LTO）可以进一步减小动态库的大小。 LTO 允许编译器在链接时进行全局优化，从而消除冗余代码和数据。
   1. ```-flto```：启用 LTO 优化。
   2. ```-fwhole-program```：启用 LTO 优化。

### 1.3 导出符号
&emsp;&emsp;导出符号是指动态库中可以被其他模块（例如可执行文件或其他动态库）访问的函数和变量。 换句话说，它们是库的公共接口。默认情况下，在 Linux 系统中，使用 GCC 或 Clang 编译动态库时，所有非 static 的函数和全局变量都会被导出。 这通常会导致导出过多的符号，增加库的大小。导出符号越多，库的大小越大。 通过只导出必要的符号，可以显著减小库的大小。

&emsp;&emsp;控制导出符号不同编译器提供的方式不同，但是一般来说，有以下几种方式：
1. 通过导出文件指定导出的符号列表；
2. 代码中通过标记来标记需要导出的函数。

```cpp
#ifndef MY_LIBRARY_EXPORT_H
#define MY_LIBRARY_EXPORT_H

#ifdef _WIN32
  #ifdef MY_LIBRARY_BUILD
    #define MY_EXPORT __declspec(dllexport)
  #else
    #define MY_EXPORT __declspec(dllimport)
  #endif
#elif defined(__GNUC__)
  #define MY_EXPORT __attribute__((visibility("default")))
#else
  #define MY_EXPORT
#endif

#endif // MY_LIBRARY_EXPORT_H
```

### 1.4 ```strip```
&emsp;&emsp;通常情况下，二进制产物会包含一些调试信息，比如符号表、调试符号等。这些信息对于调试和分析二进制文件非常有用，但是它们通常不会被用于发布版本。因此，在发布版本中，通常会使用```strip```工具来去除这些调试信息，从而减小二进制文件的大小。
- 不可逆操作： ```strip``` 命令会直接修改文件，并且无法恢复。 因此，在运行 ```strip``` 命令之前，请务必备份文件。
- 影响调试： 移除符号表和调试信息会使调试变得更加困难。 如果需要调试程序，请不要运行 ```strip``` 命令。
- 发布版本： ```strip``` 命令通常用于发布最终版本的程序，以减小文件大小并提高安全性。
- 调试信息分离： 可以使用 ```--only-keep-debug``` 和 ```--add-gnu-debuglink``` 选项将调试信息分离到单独的文件中。 这样可以在不影响程序运行的情况下进行调试。

## 2 实验
### 2.1 测试代码和环境
&emsp;&emsp;我们的测试环境是：
```
Linux DESKTOP-JLHBOB4 4.4.0-19041-Microsoft #4355-Microsoft Thu Apr 12 17:37:00 PST 2024 x86_64 x86_64 x86_64 GNU/Linux
g++ (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
```

&emsp;&emsp;测试代码如下，分别是一个头文件和一个源文件编译成so库：
```cpp
// my_lib.h
#ifndef MY_LARGE_LIBRARY_H
#define MY_LARGE_LIBRARY_H

#include <iostream>
#include <vector>

// 用于控制导出符号，可以参考之前的通用 EXPORT 宏
#ifdef _WIN32
  #ifdef MY_LARGE_LIBRARY_BUILD
    #define MY_LARGE_LIBRARY_API __declspec(dllexport)
  #else
    #define MY_LARGE_LIBRARY_API __declspec(dllimport)
  #endif
#elif defined(__GNUC__)
  #define MY_LARGE_LIBRARY_API __attribute__((visibility("default")))
#else
  #define MY_LARGE_LIBRARY_API
#endif


// 模板类
template <typename T>
class MY_LARGE_LIBRARY_API MyTemplateClass {
public:
    MyTemplateClass(T value);
    T getValue() const;
private:
    T m_value;
};

// 内联函数
inline int MY_LARGE_LIBRARY_API inlineFunction(int x) {
    return x * x * x; // 复杂的计算，增加内联的代价
}

// 虚基类
class MY_LARGE_LIBRARY_API BaseClass {
public:
    BaseClass(int id);
    virtual ~BaseClass();
    virtual int calculate() const;
    int getId() const;
protected:
    int m_id;
};

// 派生类
class MY_LARGE_LIBRARY_API DerivedClass : public BaseClass {
public:
    DerivedClass(int id, double factor);
    ~DerivedClass() override;
    int calculate() const override;
private:
    double m_factor;
};

// 一个导出函数，使用了上述的类和函数
MY_LARGE_LIBRARY_API int processData(const std::vector<int>& data);

#endif // MY_LARGE_LIBRARY_H
```

```cpp
// my_lib.cpp
#include "Mylib.hpp"
#include <numeric> // std::accumulate

// 模板类的实现
template <typename T>
MyTemplateClass<T>::MyTemplateClass(T value) : m_value(value) {}

template <typename T>
T MyTemplateClass<T>::getValue() const {
    return m_value;
}

// 显式实例化一些常用的模板类型，减少编译单元间的重复实例化
template class MY_LARGE_LIBRARY_API MyTemplateClass<int>;
template class MY_LARGE_LIBRARY_API MyTemplateClass<double>;

// 基类的实现
BaseClass::BaseClass(int id) : m_id(id) {}

BaseClass::~BaseClass() {}

int BaseClass::calculate() const {
    return m_id * 2;
}

int BaseClass::getId() const {
    return m_id;
}

// 派生类的实现
DerivedClass::DerivedClass(int id, double factor) : BaseClass(id), m_factor(factor) {}

DerivedClass::~DerivedClass() {}

int DerivedClass::calculate() const {
    return static_cast<int>(m_id * m_factor * 3);
}

// processData 函数的实现
int processData(const std::vector<int>& data) {
    int sum = std::accumulate(data.begin(), data.end(), 0);
    int inlinedResult = inlineFunction(sum);
    MyTemplateClass<int> templateObject(inlinedResult);
    BaseClass* baseObject = new DerivedClass(sum, 2.5);
    int finalResult = templateObject.getValue() + baseObject->calculate();
    delete baseObject;
    return finalResult;
}
```

### 2.1.2 不同操作对二进制大小的影响

|默认| -O1 | -O2| -O3| -Os|符号|section|lto|whole|rtti|异常|debug|strip| 包大小（Byte）| 
| --- | --- | --- | ---|--- | --- |---|---|---|---|---|---|---|---|
|   √  |     |     |||||||||||57400|
|   √  |  √   |     |||||||||||53752|
|   √  |     |  √   |||||||||||53560|
|   √  |     |     |√||||||||||54784|
|   √  |     |     ||√|||||||||53464|
|   √  |     |     ||√|√||||||||53480|
|   √  |     |     ||√|√|√|||||||53936|
|   √  |     |     ||√|√|√|√||||||23120|
|   √  |     |     ||√|√|√|√|√|||||10408|
|   √  |     |     ||√|√|√|√|√|√||||10016|
|   √  |     |     ||√|√|√|√|√|√|√|||10016|
|   √  |     |     ||√|√|√|√|√|√|√|√||9640|
|   √  |     |     ||√|√|√|√|√|√|√|√|√|6008|


&emsp;&emsp;下面是不同配置的详细说明：
- 默认配置：使用默认的编译选项和编译方式，不进行任何裁剪和优化。
  - ```g++ -fPIC -shared Mylib.cpp -g -DMY_LARGE_LIBRARY_BUILD -o mylib.so```
- 使用不同优化选项对比，具体```-O0```、```-O1```、```-O2```、```-O3```。
- 隐藏符号：使用```-fvisibility=hidden```选项隐藏所有符号。
  - ```g++ -fPIC -shared Mylib.cpp -g -DMY_LARGE_LIBRARY_BUILD -o mylibos_hidden.so -fvisibility=hidden -Os```
- 独立section裁剪：使用```-ffunction-sections```和```-fdata-sections```选项将每个函数和数据放入单独的代码段和数据段。
  - ```g++ -fPIC -shared Mylib.cpp -g -DMY_LARGE_LIBRARY_BUILD -o mylibos_sections.so -ffunction-sections -fdata-sections -Os```
- ```lto```
  - ```g++ -fPIC -shared Mylib.cpp -g -DMY_LARGE_LIBRARY_BUILD -o mylibos_sections_lto.so -ffunction-sections -fdata-sections -Os -Wl,--gc-sections -flto```
- 更激进的优化：```-fwhole-program```
  - ```g++ -fPIC -shared Mylib.cpp -g -DMY_LARGE_LIBRARY_BUILD -o mylibos_sections_lto_whole.so -ffunction-sections -fdata-sections -Os -Wl,--gc-sections -flto -fwhole-program```
- 禁用RTTI:```-fno-rtti```
  - ```g++ -fPIC -shared Mylib.cpp -g -DMY_LARGE_LIBRARY_BUILD -o mylibos_sections_lto_whole_nortti.so -ffunction-sections -fdata-sections -Os -Wl,--gc-sections -flto -fwhole-program -fno-rtti```
- 禁用异常```-fno-exceptions```
  - ```g++ -fPIC -shared Mylib.cpp -g -DMY_LARGE_LIBRARY_BUILD -o mylibos_sections_lto_whole_nortti_noex.so -ffunction-sections -fdata-sections -Os -Wl,--gc-sections -flto -fwhole-program -fno-rtti -fno-exceptions```
- 分离调试信息:```-gsplit-dwarf```
  - ```g++ -fPIC -shared Mylib.cpp -g -DMY_LARGE_LIBRARY_BUILD -o mylibos_sections_lto_whole_nortti_noex_debuginfo.so -ffunction-sections -fdata-sections -Os -Wl,--gc-sections -flto -fwhole-program -fno-rtti -fno-exceptions -gsplit-dwarf```
- 删除无用的信息:```strip```
  - ```strip -g -x -s mylib.so```

&emsp;&emsp;从上面的结果来看我们上面大部分操作都可以减少二进制，而且效果明显，我们的库从最开始的57400Byte减少到了6008Byte。能够看到成效是非常明显的。但是本来预期能够降低包大小的操作没有降低包大小的同时，反而增加了包大小这是为什么。


>&emsp;&emsp;实际工程中往往限制导出符号比较能够降低包大小，上面的实验没有降低包大小的原因是因为我们的测试代码非常简单函数太少，因此包大小的优化效果不是很明显。以及一些其他参数没有降低包大小的原因也是因为我们的测试代码比较简单。

## 2.1 包大小排查思路
&emsp;&emsp;下面我们就简单排查下。
&emsp;&emsp;根据上面的数据我们能够看到有两个选项导致了包大小变大，分别是```-O3```和```gc-sections```，前者是因为该选项更倾向于优化性能而牺牲存储空间，因此已经有明确的结论不需要我们去排查。但是我们期望```gc-sections```等选项带来的是包大小优化，但是事实却不是如此。
&emsp;&emsp;首先，对于一个二进制动态库，其有不同的section组成，为了确认包大小变大的原因我们首先要做的是确认是哪个section变大了。因此我们使用```objdump -h```工具拆分二进制包来确认哪个部分增大了。下面是拆分得到的结果：
```
27 .debug_aranges     00000080 0000000000000000 DEBUG
30 .debug_line        000005f1 0000000000000000 DEBUG
31 .debug_str         00003bbe 0000000000000000 DEBUG
33 .debug_ranges      00000180 0000000000000000 DEBUG

27 .debug_aranges     00000110 0000000000000000 DEBUG
30 .debug_line        0000055f 0000000000000000 DEBUG
31 .debug_str         00003bae 0000000000000000 DEBUG
33 .debug_ranges      000000f0 0000000000000000 DEBUG
```

&emsp;&emsp;从上面的拆包能够看到增加的主要是调试信息。而这部分调试信息在后续的```strip```中已经被删除了，因此影响我们最终产物大小的额外因素已经被排除了。如果希望知道具体增大了什么可以通过相关的提取对应section的信息来确认哪一部分增大了。

&emsp;&emsp;上面的排查路径其实不是很典型，因为一般情况下包大小都是因为代码引起的

&emsp;&emsp;下面简单描述下如何排查包大小问题：
1. 首先，对比的产物一定是相同编译参数下的最终产物，使用两个带调试信息的不同编译参数的包对比没有意义(因此排查的前提是代码相同编译参数不同或者编译参数相同代码更改)；
2. 准备好后，使用```objdump -h```分析不同section的大小，来确认方向：
   - 不同section对应不同的数据，一般情况下比较容易出现增大的是data和text段
   - ```.text```: 代码段，包含可执行指令。 如果包大小增加主要是 .text section 变大，则需要关注代码优化。
   - ```.rodata```: 只读数据段，包含字符串常量、只读变量等。 大量的字符串常量或嵌入式资源会增加此 section 的大小。
   - ```.data```: 已初始化数据段，包含已初始化的全局变量和静态变量。 大的静态数组或全局变量会增加此 section 的大小。
3. 明确具体包大小变化比较大的section后，可以尝试对比代码变动来初步确定变大的根本原因，如果无法确定则继续；
4. 使用命令```nm -CS <your_binary> | sort -rnk1 ```对代码段和数据段进行排序，然后对比不同版本之间的差异。
5. 找到差异的具体部分之后再使用```objdump -d```反汇编并对比源码来确认最终原因。

>emsp;&emsp;需要注意的是，有些博客会推荐使用```bloaty```，个人建议如果能够通过该工具排查发现数据异常，推荐直接使用linux native的工具链。（在实际项目中发现```bloaty```似乎统计的不是很准确。）
