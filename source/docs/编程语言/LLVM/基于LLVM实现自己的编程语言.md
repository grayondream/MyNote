# 基于LLVM实现自己的编程语言

&emsp;&emsp;以前实现一门自己的编程语言，往往需要自行完成前端解析、后端优化等一系列复杂工作。如今有了 LLVM，这件事方便了许多。LLVM 为语言设计者提供了从源码到机器码的完整基础设施：设计者只需实现前端，把自己的语言翻译成 LLVM IR，就能复用 LLVM 提供的大量后端优化能力。如果希望进一步优化语言，LLVM 也支持编写自定义 Pass，Rust 就是采用这种方式的典型例子。因此，利用 LLVM 实现编程语言的主要障碍，已经转变为如何设计一门好的语言。

&emsp;&emsp;但“只写前端”这句话容易让人低估工作量。LLVM 负责的是**指令级**的通用问题：指令选择、寄存器分配、指令调度、目标文件格式；它不负责源语言的类型身份、名字修饰、对象布局、模块可见性、运行时库和错误模型。这些语义必须由前端完整地“翻译”为 LLVM IR 能够表达的约定。本文沿着“语言设计 → 词法 → 语法 → 语义 → 类型与布局 → 代码生成 → 优化与链接 → 工程化”的顺序，分析一套可运行的语言实现是如何搭起来的。

- [LLVM-Cdotdot](https://github.com/grayondream/Cdotdot)

## 1 定义语言
### 1.1 先确定“不做什么”

&emsp;&emsp;语言设计的第一步不是写文法，而是确定契约：程序如何执行、内存如何管理、类型如何相容、错误如何表达、代码如何组织、如何与外部世界交互。LLVM 只关心最后一步生成的 IR 是否合法，前面这些问题它一概不回答。一个常见的失败路径是：语法很快就写完了，却在实现语义时发现类型系统、ABI 和运行时互相打架，只能不断打补丁。

&emsp;&emsp;CDD 的做法是先写“设计原则”和“非目标”。它明确定位为系统级语言：保留 C 的显式内存模型与 C 兼容 FFI，去掉预处理器和旧式数值类型，提供模块、命名空间、类、单继承、模板、lambda、编译期求值、原子与线程，标准库用 CDD 自己编写。与此同时，它明确声明自己**不是**内存安全语言，**不是** C++ 的平替，也**不是** ABI 稳定的跨平台语言。这些“不做”与“做”同样重要，因为它们直接决定了前端需要实现多少语义。

&emsp;&emsp;把设计决策写清楚，还能反过来约束实现。例如“定宽数值类型”意味着词法分析必须把字面量后缀解析成具体类型，语义分析必须实现整型提升与常用算术转换，代码生成必须按位宽选择 `i8`/`i16`/`i32`/`i64`/`i128`；“显式内存管理、无 GC”意味着语言不需要在 IR 中插入垃圾回收屏障，但必须把 `malloc`/`free` 的调用约定和所有权规则交给标准库与程序员。每一条语言承诺，最终都会落到某个编译阶段的某段代码上。

### 1.2 用一张表固定语言契约

&emsp;&emsp;在动手实现前，可以先把关键决策列成表。下表是 CDD 的契约摘要，也是后文分析的地图。

| 设计维度 | CDD 的选择 | 对实现的影响 |
|---|---|---|
| 执行模型 | 原生编译，无虚拟机 | 直接生成目标机器码，需要目标三元组与 ABI |
| 类型系统 | 静态、名义类型、定宽数值 | 前端保留完整类型身份，不能只映射为 LLVM 位宽 |
| 内存模型 | 手动管理，无 GC | 不生成 GC 屏障；标准库提供分配器与容器 |
| 泛型 | 模板 + 单态化 | 语义分析阶段实例化，代码生成看到具体类型 |
| 编译期能力 | `compile_time` / `compile_only` | 语义分析内嵌常量求值器与反射 |
| 模块组织 | `module` / `import` / `namespace` | 驱动层需要模块加载器与可见性规则 |
| 外部交互 | C ABI、内联汇编、原子、线程 | 名字修饰必须为 C 符号留出裸名通道 |
| 错误模型 | 诊断码 + 状态值/panic | 无异常，IR 中不需要异常表 |

&emsp;&emsp;表中“对实现的影响”一列值得反复看。很多语言实现之所以在后期重构，是因为早期只把类型当成“位宽 + 有无符号”，等到要做重载、模板、调试信息、反射时才发现类型身份已经丢失。CDD 的 `Type` 层次、`TypeContext` 单例以及 `src/ast/Mangle.cpp` 中的类型编码，本质上都是在维护“源语言类型”与“LLVM 类型”之间的多对一映射。

### 1.3 一个贯穿全文的样例

&emsp;&emsp;下面这段程序覆盖了类、成员函数、模板、`compile_time` 函数、模块导入和标准库输出，是 CDD README 中的语言导览，也是后文分析符号与实例化时的样本。

```cpp
module tour;

class Counter {
public:
    int32 count;
    void inc() { this->count = this->count + 1; }
};

template<typename T> struct Box { T value; };

compile_time int32 twice(int32 x) { return x * 2; }

int32 main() {
    Counter c;
    c.count = 0;
    c.inc();

    Box<int32> b;
    b.value = twice(c.count);
    println("{}", b.value);
    return 0;
}
```

&emsp;&emsp;把文件命名为 `tour.cdd` 后编译运行，输出为 `2`。这段程序在 IR 层会留下几个很有代表性的痕迹：类 `Counter` 变成 `%Counter = type { i32 }`，成员函数变成带隐式 `this` 的自由函数 `@inc_Counterptr`，模板实例变成具名结构 `%"Box$int32"`，`compile_time` 函数在实参为常量时被语义分析折叠、在实参为运行时值时保留为普通调用，`println` 最终落到 `printf` 一族的外部声明。后文会逐一解释这些名字是怎么来的。

## 2 LLVM实现编程语言的基本流程

### 2.1 三段式与职责边界

&emsp;&emsp;LLVM 的经典结构可以概括为“前端 → IR → 优化器/后端”。前端把源语言翻译成 LLVM IR；IR 是带类型的 SSA 中间表示；优化器在 IR 上做与源语言无关的变换；后端把 IR 降到目标机器的指令、寄存器与目标文件。下图把这个边界画了出来。

<p align="center">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 260" width="760" height="260" style="max-width:100%;height:auto" font-family="sans-serif">
  <rect x="20" y="30" width="200" height="160" rx="8" fill="#eaf2f8" stroke="#2980b9" stroke-width="2"/>
  <text x="120" y="55" text-anchor="middle" font-size="14" font-weight="bold" fill="#21618c">前端（你负责）</text>
  <text x="120" y="82" text-anchor="middle" font-size="12">词法 / 语法 / 语义</text>
  <text x="120" y="104" text-anchor="middle" font-size="12">类型检查 / 符号表</text>
  <text x="120" y="126" text-anchor="middle" font-size="12">高层语义 → 低层语义</text>
  <text x="120" y="158" text-anchor="middle" font-size="11" fill="#555">输出：LLVM IR</text>

  <rect x="280" y="30" width="180" height="160" rx="8" fill="#e8f8f5" stroke="#27ae60" stroke-width="2"/>
  <text x="370" y="55" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e8449">LLVM IR</text>
  <text x="370" y="82" text-anchor="middle" font-size="12">带类型的 SSA</text>
  <text x="370" y="104" text-anchor="middle" font-size="12">显式控制流 / 内存</text>
  <text x="370" y="126" text-anchor="middle" font-size="12">与源语言解耦</text>
  <text x="370" y="158" text-anchor="middle" font-size="11" fill="#555">优化器的输入</text>

  <rect x="520" y="30" width="220" height="160" rx="8" fill="#fdf2e9" stroke="#d35400" stroke-width="2"/>
  <text x="630" y="55" text-anchor="middle" font-size="14" font-weight="bold" fill="#a04000">优化器 + 后端</text>
  <text x="630" y="82" text-anchor="middle" font-size="12">Pass 管线</text>
  <text x="630" y="104" text-anchor="middle" font-size="12">指令选择 / 寄存器分配</text>
  <text x="630" y="126" text-anchor="middle" font-size="12">调度 / 目标文件</text>
  <text x="630" y="158" text-anchor="middle" font-size="11" fill="#555">输出：.o / 可执行文件</text>

  <line x1="220" y1="110" x2="272" y2="110" stroke="#555" stroke-width="2" marker-end="url(#a1)"/>
  <line x1="460" y1="110" x2="512" y2="110" stroke="#555" stroke-width="2" marker-end="url(#a1)"/>
  <text x="380" y="225" text-anchor="middle" font-size="12" fill="#c0392b">LLVM 不替你决定：类型身份、名字修饰、ABI、运行时、模块可见性</text>
  <defs>
    <marker id="a1" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#555"/>
    </marker>
  </defs>
</svg>
</p>
<p align="center"><em>图 1　LLVM 三段式与职责边界</em></p>

&emsp;&emsp;这张图最重要的信息不是三个方框，而是最后一行红字。LLVM IR 是“低层但通用”的表示：它能表达整数、浮点、指针、聚合、基本块和 SSA 值，却不知道什么是类、模块、重载或模板。前端必须把所有高层概念降到这些通用构件上。例如，类方法没有“方法”这个概念，只能降级为第一个参数是 `this` 指针的普通函数；名字修饰必须由前端完成，否则链接器无法区分同名重载。

### 2.2 CDD 的真实编译管线

&emsp;&emsp;把上图落到 CDD 的代码上，可以得到一条更具体的管线。它比教科书版本多了模块预加载、标准库注入、编译期求值和链接器编排。

<p align="center">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 250" width="900" height="250" style="max-width:100%;height:auto" font-family="sans-serif">
  <rect x="15" y="40" width="112" height="108" rx="7" fill="#f4f6f7" stroke="#7f8c8d" stroke-width="2"/>
  <text x="71" y="68" text-anchor="middle" font-size="12" font-weight="bold">.cdd 源文件</text>
  <text x="71" y="90" text-anchor="middle" font-size="10">module / import</text>
  <text x="71" y="108" text-anchor="middle" font-size="10">UTF-8 文本</text>

  <rect x="145" y="40" width="112" height="108" rx="7" fill="#eaf2f8" stroke="#2980b9" stroke-width="2"/>
  <text x="201" y="68" text-anchor="middle" font-size="12" font-weight="bold" fill="#21618c">Lexer</text>
  <text x="201" y="90" text-anchor="middle" font-size="10">Token 流</text>
  <text x="201" y="108" text-anchor="middle" font-size="10">位置 / 字面量</text>

  <rect x="275" y="40" width="112" height="108" rx="7" fill="#eaf2f8" stroke="#2980b9" stroke-width="2"/>
  <text x="331" y="68" text-anchor="middle" font-size="12" font-weight="bold" fill="#21618c">Parser</text>
  <text x="331" y="90" text-anchor="middle" font-size="10">AST</text>
  <text x="331" y="108" text-anchor="middle" font-size="10">源码位置</text>

  <rect x="405" y="40" width="112" height="108" rx="7" fill="#e8f8f5" stroke="#27ae60" stroke-width="2"/>
  <text x="461" y="68" text-anchor="middle" font-size="12" font-weight="bold" fill="#1e8449">Sema</text>
  <text x="461" y="90" text-anchor="middle" font-size="10">类型 / 符号</text>
  <text x="461" y="108" text-anchor="middle" font-size="10">模板 / CT 求值</text>

  <rect x="535" y="40" width="112" height="108" rx="7" fill="#e8f8f5" stroke="#27ae60" stroke-width="2"/>
  <text x="591" y="68" text-anchor="middle" font-size="12" font-weight="bold" fill="#1e8449">Codegen</text>
  <text x="591" y="90" text-anchor="middle" font-size="10">IRBuilder</text>
  <text x="591" y="108" text-anchor="middle" font-size="10">布局 / 修饰</text>

  <rect x="665" y="40" width="112" height="108" rx="7" fill="#fdf2e9" stroke="#d35400" stroke-width="2"/>
  <text x="721" y="68" text-anchor="middle" font-size="12" font-weight="bold" fill="#a04000">PassBuilder</text>
  <text x="721" y="90" text-anchor="middle" font-size="10">O0/O1/O2/O3</text>
  <text x="721" y="108" text-anchor="middle" font-size="10">默认管线</text>

  <rect x="795" y="40" width="90" height="108" rx="7" fill="#fdf2e9" stroke="#d35400" stroke-width="2"/>
  <text x="840" y="68" text-anchor="middle" font-size="11" font-weight="bold" fill="#a04000">Target + Link</text>
  <text x="840" y="90" text-anchor="middle" font-size="10">.o 文件</text>
  <text x="840" y="108" text-anchor="middle" font-size="10">系统链接器</text>

  <line x1="127" y1="94" x2="143" y2="94" stroke="#555" stroke-width="2" marker-end="url(#a2)"/>
  <line x1="257" y1="94" x2="273" y2="94" stroke="#555" stroke-width="2" marker-end="url(#a2)"/>
  <line x1="387" y1="94" x2="403" y2="94" stroke="#555" stroke-width="2" marker-end="url(#a2)"/>
  <line x1="517" y1="94" x2="533" y2="94" stroke="#555" stroke-width="2" marker-end="url(#a2)"/>
  <line x1="647" y1="94" x2="663" y2="94" stroke="#555" stroke-width="2" marker-end="url(#a2)"/>
  <line x1="777" y1="94" x2="793" y2="94" stroke="#555" stroke-width="2" marker-end="url(#a2)"/>

  <text x="205" y="190" text-anchor="middle" font-size="11" fill="#555">诊断先于构建：词法/语法/语义错误在此终止</text>
  <text x="600" y="190" text-anchor="middle" font-size="11" fill="#555">测试在进程内用 LLJIT 直接执行 IR，不依赖链接器</text>
  <text x="450" y="220" text-anchor="middle" font-size="12" fill="#c0392b">模块预加载在 Parser 之前；std.c 预置层在 Sema 之前</text>
  <defs>
    <marker id="a2" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#555"/>
    </marker>
  </defs>
</svg>
</p>
<p align="center"><em>图 2　CDD 的编译管线与关键产物</em></p>

&emsp;&emsp;这条管线有一个容易被忽略的细节：**模块预加载发生在解析主文件之前**。`CompilerDriver::compileFile` 先扫描 token 中的 `import` 说明符，解析被导入的 `.cdd` 文件并收集其声明，然后才解析主文件，最后把这些声明插到主翻译单元的最前面。原因是 `S s;` 这样的声明在解析阶段就需要知道 `S` 是结构体还是类型别名；如果等到语义分析再加载，解析器已经把它当成了未知类型。这个顺序是模块化语言必须付的工程成本，也是“先设计、再实现”的一个具体体现。

### 2.3 代码目录与职责

&emsp;&emsp;CDD 的目录划分基本对应上图的每个阶段，读代码时可以按这个映射定位。

| 路径 | 职责 | 关键类型/函数 |
|---|---|---|
| `src/frontend/` | 词法、Token 流、递归下降与 Pratt 解析 | `Lexer`、`Parser`、`Token` |
| `src/ast/` | AST 节点、类型、聚合布局、名称修饰 | `ExprAST`、`Type`、`LayoutBuilder`、`mangleFunction` |
| `src/sema/` | 名称/类型解析、重载、模板、编译期求值 | `SemanticAnalyzer`、`CompileTimeEvaluator` |
| `src/codegen/` | LLVM 上下文、作用域、类型降低、运行时辅助 | `CodegenContext` |
| `src/driver/` | CLI、模块加载、标准库注入、优化与链接 | `CompilerDriver`、`ModuleLoader`、`Linker` |
| `libs/std/` | 用 CDD 编写的标准库 | `std.core`、`std.string`、`std.vec` 等 |

&emsp;&emsp;这种划分让“语言语义”和“LLVM 机制”保持距离：`sema` 的代码不直接使用 `IRBuilder`/`Module` 等代码生成 API；虽然 AST 头文件因为 `codegen` 接口间接引入了 `llvm::Value`，但语义分析本身不依赖 LLVM 的指令构造。只有 `CompileTimeEvaluator` 为了 `size_of`/`offset_of` 这类布局查询，显式使用 LLVM 的 `DataLayout`/`Type`。`codegen` 则集中引入 `IRBuilder`、`Module`、`TargetMachine`。当你自己实现语言时，也建议尽早立下这条边界，否则 LLVM 类型会渗透到语义层，最终无法做常量折叠、模板实例化或独立测试。

## 3 前端：从字符到抽象语法树

### 3.1 词法分析

&emsp;&emsp;词法分析的任务是把字符流切成 Token，并尽可能早地消灭歧义：关键字与标识符如何区分、`=` 与 `==` 如何最长匹配、`1.5f` 是浮点还是整数、字符串里的转义如何处理。CDD 使用手写扫描器而不是生成器，原因是语言的字面量形式很多（定宽后缀、原始字符串、多行字符串、UTF-8 前缀、十六进制浮点），手写更容易做错误恢复和精确位置。

&emsp;&emsp;Token 本身不仅记录“类别 + 文本”，还携带解析后的字面量值、源文件、行列号和字面量种类。这些信息是诊断与语义分析的基础。

```cpp
// 摘自 src/frontend/Token.h（省略部分枚举成员）
enum class TokenType : int32_t {
    TOKEN_IDENTIFIER,
    TOKEN_NUMBER,
    // ... 关键字、运算符、分隔符
    TOKEN_EOS
};

using TokenValue =
    std::variant<std::monostate, unsigned __int128, double, char, std::string>;

class Token {
public:
    TokenType type{};
    std::string lexeme{};       // 源文本
    TokenValue value{};         // 解析后的字面量
    std::string filename{};
    int line{};
    int column{};
    LiteralKind literalKind{LiteralKind::None};
    StringLiteralPrefix stringPrefix{StringLiteralPrefix::None};
    bool isMultilineString{};
};
```

&emsp;&emsp;扫描器的主分派是一个“看首字符决定走哪条路”的循环。标识符允许 UTF-8 多字节序列，数字允许前导小数点，字符串有普通、原始、多行和带前缀四种形态；运算符则按最长匹配逐字符探测。

```cpp
// 摘自 src/frontend/Lexer.cpp，略有删节
Token Lexer::scanToken() {
    // 先记录 token 的起始位置，诊断指向首字符而不是末字符
    m_tokenStartLine = m_lineNum;
    m_tokenStartCol = m_colNum + 1;

    const char ch = peek();
    if (std::isalpha(ch) or ch == '_' or (ch & 0x80) != 0) {
        return scanIdentifier();          // 含 UTF-8 标识符
    } else if (std::isdigit(ch) ||
               (ch == '.' && std::isdigit(peekNext()))) {
        return scanNumber();              // 支持 0b/0o/0x、后缀、十六进制浮点
    } else if (ch == '"') {
        return scanString();
    } else if (ch == '\'') {
        return scanChar();
    }

    Token token;
    switch (ch) {
    case '+':
        if (peekNext() == '+')      token = makeToken(TokenType::TOKEN_PLUS_PLUS, lexeme());
        else if (peekNext() == '=') token = makeToken(TokenType::TOKEN_PLUS_EQ, lexeme());
        else                        token = makeToken(TokenType::TOKEN_PLUS, lexeme());
        break;
    case '-':
        if (peekNext() == '>')      token = makeToken(TokenType::TOKEN_ARROW, lexeme());
        // ... 其余运算符同理
        break;
    }
    return token;
}
```

&emsp;&emsp;字面量的数值处理有两个值得借鉴的工程决定。第一，整型字面量统一用 `unsigned __int128` 累加，超过 128 位才报 `E0003`，避免在 `long long` 上触发有符号溢出 UB；第二，错误不是抛异常，而是把 `Diagnostic` 收进列表并继续扫描。这样一次编译可以报告多个错误，编辑器集成也更友好。

```cpp
// 摘自 src/frontend/Lexer.cpp：128 位累加与溢出诊断
bool overflowed = false;
const unsigned __int128 maxVal = ~(unsigned __int128)0;
auto addDigit = [&](unsigned __int128& acc, unsigned base, unsigned digit) {
    if (overflowed) return;
    if (acc > (maxVal - digit) / base) {
        overflowed = true;
        report(DiagnosticCode::LexIntegerOverflow,
               "integer literal overflows 128 bits",
               m_tokenStartLine, m_tokenStartCol);
    } else {
        acc = acc * base + digit;
    }
};
```

<p align="center">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 230" width="780" height="230" style="max-width:100%;height:auto" font-family="sans-serif">
  <rect x="20" y="22" width="740" height="42" rx="6" fill="#f8f9f9" stroke="#95a5a6" stroke-width="1.5"/>
  <text x="40" y="49" font-family="monospace" font-size="16" fill="#2c3e50">int32 x = a + 42;</text>
  <text x="700" y="49" text-anchor="end" font-size="11" fill="#7f8c8d">源文本</text>

  <rect x="20" y="90" width="96" height="52" rx="5" fill="#eaf2f8" stroke="#2980b9"/>
  <text x="68" y="112" text-anchor="middle" font-family="monospace" font-size="12">int32</text>
  <text x="68" y="130" text-anchor="middle" font-size="9" fill="#555">KW / 类型</text>

  <rect x="126" y="90" width="96" height="52" rx="5" fill="#e8f8f5" stroke="#27ae60"/>
  <text x="174" y="112" text-anchor="middle" font-family="monospace" font-size="12">x</text>
  <text x="174" y="130" text-anchor="middle" font-size="9" fill="#555">IDENT</text>

  <rect x="232" y="90" width="96" height="52" rx="5" fill="#fef9e7" stroke="#f1c40f"/>
  <text x="280" y="112" text-anchor="middle" font-family="monospace" font-size="12">=</text>
  <text x="280" y="130" text-anchor="middle" font-size="9" fill="#555">ASSIGN</text>

  <rect x="338" y="90" width="96" height="52" rx="5" fill="#e8f8f5" stroke="#27ae60"/>
  <text x="386" y="112" text-anchor="middle" font-family="monospace" font-size="12">a</text>
  <text x="386" y="130" text-anchor="middle" font-size="9" fill="#555">IDENT</text>

  <rect x="444" y="90" width="96" height="52" rx="5" fill="#fef9e7" stroke="#f1c40f"/>
  <text x="492" y="112" text-anchor="middle" font-family="monospace" font-size="12">+</text>
  <text x="492" y="130" text-anchor="middle" font-size="9" fill="#555">PLUS</text>

  <rect x="550" y="90" width="96" height="52" rx="5" fill="#fdedec" stroke="#c0392b"/>
  <text x="598" y="112" text-anchor="middle" font-family="monospace" font-size="12">42</text>
  <text x="598" y="130" text-anchor="middle" font-size="9" fill="#555">NUMBER (u128)</text>

  <rect x="656" y="90" width="104" height="52" rx="5" fill="#f4f6f7" stroke="#7f8c8d"/>
  <text x="708" y="112" text-anchor="middle" font-family="monospace" font-size="12">;</text>
  <text x="708" y="130" text-anchor="middle" font-size="9" fill="#555">SEMICOLON</text>

  <line x1="68" y1="64" x2="68" y2="88" stroke="#aaa" stroke-width="1"/>
  <line x1="174" y1="64" x2="174" y2="88" stroke="#aaa" stroke-width="1"/>
  <line x1="280" y1="64" x2="280" y2="88" stroke="#aaa" stroke-width="1"/>
  <line x1="386" y1="64" x2="386" y2="88" stroke="#aaa" stroke-width="1"/>
  <line x1="492" y1="64" x2="492" y2="88" stroke="#aaa" stroke-width="1"/>
  <line x1="598" y1="64" x2="598" y2="88" stroke="#aaa" stroke-width="1"/>
  <line x1="708" y1="64" x2="708" y2="88" stroke="#aaa" stroke-width="1"/>

  <text x="390" y="190" text-anchor="middle" font-size="11" fill="#555">最长匹配：`==` 不会被切成两个 `=`；`1.5f` 是一个浮点 Token</text>
  <text x="390" y="212" text-anchor="middle" font-size="11" fill="#555">每个 Token 携带 file / line / column，错误可以直接指到源码</text>
</svg>
</p>
<p align="center"><em>图 3　字符流到 Token 流：类别、字面量与位置</em></p>

### 3.2 语法分析

&emsp;&emsp;CDD 的解析器是“递归下降 + Pratt”组合：声明和语句用递归下降，因为文法层次清晰；表达式用 Pratt（优先级爬升），因为二元运算符、赋值、三元和逗号运算符的优先级与结合性可以集中到一张表里，避免为每个优先级写一层函数。

&emsp;&emsp;表达式解析的核心是 `parseExpr(minPrec)`：先解析一元表达式与后缀，然后只要当前运算符的优先级不低于 `minPrec`，就把它归约为一个二元节点。左结合运算符让右操作数用 `prec + 1` 继续爬升，右结合运算符（赋值）用 `prec`，这样同一套循环就能表达两种结合性。

```cpp
// 摘自 src/frontend/Parser.cpp，删去了三元、逗号和赋值分支
std::unique_ptr<ExprAST> Parser::parseExpr(int minPrec) {
    auto token = peek();
    if (!token) return nullptr;
    const Token* startTok = &*token;

    auto lhs = parseUnary();
    if (!lhs) return nullptr;
    applyLocation(lhs.get(), startTok);
    lhs = parsePostfix(std::move(lhs));

    while (!eof()) {
        auto opToken = peek();
        int prec = getPrecedence(opToken->type);
        if (prec < minPrec) break;

        if (prec > 0) {
            auto binOp = tokenTypeToBinaryOp(opToken->type);
            advance();
            // 左结合：右操作数从更高优先级开始；右结合则传 prec
            int nextPrec = isRightAssociative(opToken->type) ? prec : prec + 1;
            auto rhs = parseExpr(nextPrec);
            lhs = std::make_unique<BinaryExprAST>(
                binOp, std::move(lhs), std::move(rhs));
            applyLocation(lhs.get(), startTok);
            continue;
        }
        break;
    }
    return lhs;
}
```

&emsp;&emsp;语句层的递归下降更接近文法本身。以 `if` 为例，解析器只做三件事：匹配关键字、解析括号中的条件、解析 then/else 分支；每个节点都写入源码位置，供后续诊断使用。

```cpp
// 摘自 src/frontend/Parser.cpp
std::unique_ptr<StmtAST> Parser::parseIfStmt() {
    auto ifToken = match(TokenType::TOKEN_IF);

    expect(TokenType::TOKEN_LPAREN, "expected '(' after 'if'");
    auto cond = parseExpr();
    expect(TokenType::TOKEN_RPAREN, "expected ')' after condition");

    auto thenStmt = parseStmt();
    std::unique_ptr<StmtAST> elseStmt;
    if (match(TokenType::TOKEN_ELSE)) {
        elseStmt = parseStmt();
    }

    auto stmt = std::make_unique<IfStmtAST>(
        std::move(cond), std::move(thenStmt), std::move(elseStmt));
    stmt->setLocation(ifToken->filename, ifToken->line, ifToken->column);
    return stmt;
}
```

&emsp;&emsp;AST 采用“节点自己知道如何生成代码”的面向对象风格。基类 `ExprAST` 只保留三样东西：语义分析写入的 `type`、左值标记 `isLValue`，以及纯虚的 `codegen`。这种设计的优点是新增节点时只需实现一个方法；缺点是 AST 头文件必须知道 `llvm::Value`（`Expr.h` 直接包含 `llvm/IR/Value.h`），LLVM 类型因此渗透到前端数据结构中。若想彻底隔离，可以把 `codegen` 抽成独立的 visitor，代价是多一层间接与更多样板代码。

```cpp
// 摘自 src/ast/Expr.h
class ExprAST : public ASTNode {
public:
    Type* type{nullptr};
    bool isLValue{false};

    virtual llvm::Value* codegen(CodegenContext& ctx) = 0;
    virtual bool isBitFieldAccess() const { return false; }
};
```

&emsp;&emsp;下图给出表达式 `a + b * 2` 的 AST。Pratt 解析的结果不是“先算加法”，而是乘法位于更深的子树；这正是优先级在树结构中的体现，也是后续代码生成能自然得到正确求值顺序的原因。

<p align="center">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 300" width="720" height="300" style="max-width:100%;height:auto" font-family="sans-serif">
  <rect x="310" y="20" width="100" height="48" rx="8" fill="#eaf2f8" stroke="#2980b9" stroke-width="2"/>
  <text x="360" y="50" text-anchor="middle" font-family="monospace" font-size="18" fill="#21618c">+</text>
  <text x="360" y="12" text-anchor="middle" font-size="10" fill="#555">BinaryExprAST / 优先级低</text>

  <rect x="150" y="120" width="100" height="48" rx="8" fill="#e8f8f5" stroke="#27ae60" stroke-width="2"/>
  <text x="200" y="150" text-anchor="middle" font-family="monospace" font-size="18" fill="#1e8449">a</text>
  <text x="200" y="112" text-anchor="middle" font-size="10" fill="#555">Number/Var</text>

  <rect x="470" y="120" width="100" height="48" rx="8" fill="#eaf2f8" stroke="#2980b9" stroke-width="2"/>
  <text x="520" y="150" text-anchor="middle" font-family="monospace" font-size="18" fill="#21618c">*</text>
  <text x="520" y="112" text-anchor="middle" font-size="10" fill="#555">优先级高，先归约</text>

  <rect x="400" y="220" width="100" height="48" rx="8" fill="#e8f8f5" stroke="#27ae60" stroke-width="2"/>
  <text x="450" y="250" text-anchor="middle" font-family="monospace" font-size="18" fill="#1e8449">b</text>

  <rect x="560" y="220" width="100" height="48" rx="8" fill="#fdedec" stroke="#c0392b" stroke-width="2"/>
  <text x="610" y="250" text-anchor="middle" font-family="monospace" font-size="18" fill="#a93226">2</text>

  <line x1="330" y1="68" x2="220" y2="118" stroke="#555" stroke-width="2"/>
  <line x1="390" y1="68" x2="500" y2="118" stroke="#555" stroke-width="2"/>
  <line x1="500" y1="168" x2="460" y2="218" stroke="#555" stroke-width="2"/>
  <line x1="540" y1="168" x2="590" y2="218" stroke="#555" stroke-width="2"/>
  <text x="250" y="100" font-size="10" fill="#7f8c8d">左</text>
  <text x="455" y="100" font-size="10" fill="#7f8c8d">右</text>

  <text x="360" y="290" text-anchor="middle" font-size="11" fill="#555">Pratt：`*` 的 binding power 高于 `+`，所以 `b * 2` 成为 `+` 的右子树</text>
</svg>
</p>
<p align="center"><em>图 4　表达式 AST：优先级与结合性体现在树形结构中</em></p>

### 3.3 语义分析

&emsp;&emsp;语法分析只保证“形状正确”，语义分析才决定“意思是否成立”。CDD 的 `SemanticAnalyzer` 承担了名称解析、作用域与可见性、类型检查、整型提升与常用算术转换、重载解析、默认参数、模板实例化、`compile_time` 求值、初始化检查、未使用变量检查、弃用诊断等大量工作。它遍历 AST，把每个表达式的 `type` 和 `isLValue` 写回节点，代码生成因此可以假设“类型已知、名字已解析”。

&emsp;&emsp;作用域是一条从内到外的链：块 → 函数 → 命名空间/模块 → 全局。声明只写入当前层，查找沿 `parent` 向上；如果当前层有同名符号，它会遮蔽外层。`Scope` 的实现很朴素，但足以支撑词法作用域、模块可见性和 `namespace` 限定名。

```cpp
// 摘自 src/ast/Symbol.h
class Scope {
public:
    Scope(Scope* parent) : parent(parent) {}
    Symbol* lookup(const std::string& name) const;
    OverloadSet* lookupOverload(const std::string& name) const;
    bool declare(const std::string& name, Symbol* sym);

public:
    std::unordered_map<std::string, OverloadSet> symbols;
    Scope* parent;
};
```

<p align="center">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 660 300" width="660" height="300" style="max-width:100%;height:auto" font-family="sans-serif">
  <rect x="30" y="24" width="280" height="54" rx="8" fill="#f4f6f7" stroke="#7f8c8d" stroke-width="2"/>
  <text x="170" y="48" text-anchor="middle" font-size="13" font-weight="bold">全局作用域</text>
  <text x="170" y="66" text-anchor="middle" font-size="10">printf / malloc / 全局变量</text>

  <rect x="70" y="98" width="280" height="54" rx="8" fill="#eaf2f8" stroke="#2980b9" stroke-width="2"/>
  <text x="210" y="122" text-anchor="middle" font-size="13" font-weight="bold" fill="#21618c">模块 / 命名空间</text>
  <text x="210" y="140" text-anchor="middle" font-size="10">std::println / export 声明</text>

  <rect x="110" y="172" width="280" height="54" rx="8" fill="#e8f8f5" stroke="#27ae60" stroke-width="2"/>
  <text x="250" y="196" text-anchor="middle" font-size="13" font-weight="bold" fill="#1e8449">函数作用域</text>
  <text x="250" y="214" text-anchor="middle" font-size="10">参数 n / 局部变量 sum</text>

  <rect x="150" y="246" width="280" height="42" rx="8" fill="#fef9e7" stroke="#f1c40f" stroke-width="2"/>
  <text x="290" y="272" text-anchor="middle" font-size="13" font-weight="bold" fill="#7d6608">块作用域</text>

  <line x1="170" y1="78" x2="190" y2="96" stroke="#888" stroke-width="2" marker-end="url(#a5)"/>
  <line x1="210" y1="152" x2="230" y2="170" stroke="#888" stroke-width="2" marker-end="url(#a5)"/>
  <line x1="250" y1="226" x2="270" y2="244" stroke="#888" stroke-width="2" marker-end="url(#a5)"/>

  <text x="540" y="118" text-anchor="middle" font-size="11" fill="#555">查找：未命中 → 父作用域</text>
  <text x="540" y="142" text-anchor="middle" font-size="11" fill="#555">声明：只写当前层</text>
  <text x="540" y="166" text-anchor="middle" font-size="11" fill="#555">遮蔽：内层优先</text>
  <text x="540" y="210" text-anchor="middle" font-size="11" fill="#c0392b">模块未 export 的符号</text>
  <text x="540" y="232" text-anchor="middle" font-size="11" fill="#c0392b">对外不可见</text>
  <defs>
    <marker id="a5" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#888"/>
    </marker>
  </defs>
</svg>
</p>
<p align="center"><em>图 5　作用域链：声明向内写，查找向外走</em></p>

&emsp;&emsp;类型检查的核心是“运算符期望什么、两个操作数如何统一”。CDD 实现 C 风格的整型提升与常用算术转换：窄于 `int32` 的整数先提升到 `int32`，再在 `int32`/`int64`/`uint64`/`float64` 等类型之间选择公共类型；无符号与有符号混合时按转换规则处理。下面的代码展示了算术、比较和位运算的分派逻辑，错误会带源码位置写入诊断列表。

```cpp
// 摘自 src/sema/SemanticAnalyzer.cpp，删去部分分支
Type* SemanticAnalyzer::checkBinaryTypes(BinaryOp op, Type* left, Type* right,
                                         ExprAST& node) {
    if (!left || !right) return nullptr;

    switch (op) {
    case BinaryOp::Add:
    case BinaryOp::Sub:
    case BinaryOp::Mul:
    case BinaryOp::Div:
    case BinaryOp::Mod:
        if (isArithmeticType(left) && isArithmeticType(right)) {
            return getCommonType(left, right);       // 整型提升 + 常用算术转换
        }
        if (isPointerOrArray(left) && isIntegerType(right)) return left;
        if (isIntegerType(left) && isPointerOrArray(right)) return right;
        emitError("invalid operands to binary '" + binaryOpToString(op) + "'", node);
        return nullptr;

    case BinaryOp::Eq:
    case BinaryOp::NotEq:
        if (isArithmeticType(left) && isArithmeticType(right)) {
            return typeCtx->getInt32();
        }
        // str == / != 的字节比较分支从略；string 不参与比较
        if (!typesCompatible(left, right)) {
            emitError("comparison of incompatible types", node);
            return nullptr;
        }
        return typeCtx->getInt32();

    case BinaryOp::Lt:
    case BinaryOp::Gt:
    case BinaryOp::Le:
    case BinaryOp::Ge:
        if (isArithmeticType(left) && isArithmeticType(right)) {
            return typeCtx->getInt32();              // 比较结果为 bool/整型
        }
        if (!typesCompatible(left, right)) {
            emitError("comparison of incompatible types", node);
            return nullptr;
        }
        return typeCtx->getInt32();

    case BinaryOp::BitAnd:
    case BinaryOp::BitOr:
    case BinaryOp::BitXor:
        if (isIntegerType(left) && isIntegerType(right)) {
            return getCommonType(left, right);
        }
        emitError("bitwise operator applied to non-integer types", node);
        return nullptr;
    default:
        return nullptr;
    }
}
```

&emsp;&emsp;重载解析是另一块必须自己实现的语义。CDD 为每个函数生成名字修饰后的符号，并在调用点按实参类型计算转换代价：精确匹配优于整型提升，提升优于截断，指针与数组的退化单独处理。语义分析成功后，调用节点会记录被选中的修饰名，代码生成直接按该名字发 `call`，不再做重载判断。这种“语义阶段决策、代码生成执行”的分工，能让后端保持简单，也方便在错误信息里给出候选函数。

&emsp;&emsp;CDD 还做了两项容易被忽略的检查：**确定赋值分析**（变量是否可能在赋值前被读取）和**未使用变量分析**。前者沿 `if/else` 做路径敏感的交集，在循环处保守处理；后者给局部变量打标记。这些分析不改变代码生成，却显著提升语言可用性，而且实现成本远低于做完整的流敏感类型系统。它们说明：语义分析不仅是“让程序合法”，也是“让错误尽早暴露”。

## 4 中端：类型、布局、符号与编译期

### 4.1 类型降低

&emsp;&emsp;代码生成面对的第一个问题是：源语言的 `Type` 如何变成 `llvm::Type`。这不是简单的枚举映射，因为源语言有名义类型、别名、切片、可选值、结果值、字符串视图等复合类型，而 LLVM 只有整数、浮点、指针、数组、结构体和函数。CDD 的 `getLLVMType` 是主要的降低入口，AST 节点基本都通过它取 LLVM 类型。

```cpp
// 摘自 src/codegen/CodegenContext.cpp，删去 Struct/Class/Union 的布局细节
llvm::Type* CodegenContext::getLLVMType(Type* type) {
    if (!type) return llvm::Type::getVoidTy(*context);

    switch (type->kind) {
    case TypeKind::Bool:    return llvm::Type::getInt1Ty(*context);
    case TypeKind::Int8:    return llvm::Type::getInt8Ty(*context);
    case TypeKind::Int32:   return llvm::Type::getInt32Ty(*context);
    case TypeKind::Int64:   return llvm::Type::getInt64Ty(*context);
    case TypeKind::Int128:  return llvm::Type::getInt128Ty(*context);
    case TypeKind::Float32: return llvm::Type::getFloatTy(*context);
    case TypeKind::Float64: return llvm::Type::getDoubleTy(*context);
    case TypeKind::Pointer:
        // 不透明指针：不再递归 pointee，避免自引用结构无限展开
        return llvm::PointerType::get(*context, 0);
    case TypeKind::Array: {
        auto* at = static_cast<ArrayType*>(type);
        return llvm::ArrayType::get(getLLVMType(at->elementType), at->size);
    }
    case TypeKind::Str: {
        // 所有 str 共享一个具名结构，保证跨函数边界类型身份一致
        auto* st = llvm::StructType::getTypeByName(*context, "str");
        return st ? st : llvm::StructType::create(*context,
            {llvm::PointerType::get(*context, 0),
             llvm::Type::getInt64Ty(*context)}, "str");
    }
    case TypeKind::String: {
        auto* st = llvm::StructType::getTypeByName(*context, "string");
        return st ? st : llvm::StructType::create(*context,
            {llvm::PointerType::get(*context, 0),
             llvm::Type::getInt64Ty(*context),
             llvm::Type::getInt64Ty(*context)}, "string");
    }
    default: return llvm::Type::getInt32Ty(*context);
    }
}
```

&emsp;&emsp;这里有两个关键设计。第一，**具名结构只创建一次**。如果每次调用 `StructType::create` 都新建一个 `{ptr, i64}`，LLVM 会把它们视为不同的类型，虽然位布局相同，但函数签名、`select`、`phi` 的合法性都会变得脆弱。CDD 用 `StructType::getTypeByName` 复用已有定义，保证 `str`、`string`、`Slice` 在模块内唯一。第二，**指针是不透明的**。LLVM 15 起逐步默认使用 `ptr`，不再编码 pointee 类型；这对前端是好事，因为自引用结构 `Node* next` 不会在类型降低时递归爆栈。

<p align="center">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 360" width="820" height="360" style="max-width:100%;height:auto" font-family="sans-serif">
  <text x="200" y="28" text-anchor="middle" font-size="13" font-weight="bold" fill="#21618c">CDD 源类型</text>
  <text x="620" y="28" text-anchor="middle" font-size="13" font-weight="bold" fill="#a04000">LLVM 类型</text>

  <rect x="40" y="48" width="320" height="34" rx="5" fill="#eaf2f8" stroke="#2980b9"/>
  <text x="200" y="70" text-anchor="middle" font-family="monospace" font-size="12">bool / int32 / float64</text>
  <rect x="470" y="48" width="310" height="34" rx="5" fill="#fdf2e9" stroke="#d35400"/>
  <text x="625" y="70" text-anchor="middle" font-family="monospace" font-size="12">i1 / i32 / double</text>

  <rect x="40" y="94" width="320" height="34" rx="5" fill="#eaf2f8" stroke="#2980b9"/>
  <text x="200" y="116" text-anchor="middle" font-family="monospace" font-size="12">int32*</text>
  <rect x="470" y="94" width="310" height="34" rx="5" fill="#fdf2e9" stroke="#d35400"/>
  <text x="625" y="116" text-anchor="middle" font-family="monospace" font-size="12">ptr（opaque）</text>

  <rect x="40" y="140" width="320" height="34" rx="5" fill="#eaf2f8" stroke="#2980b9"/>
  <text x="200" y="162" text-anchor="middle" font-family="monospace" font-size="12">int32[4]</text>
  <rect x="470" y="140" width="310" height="34" rx="5" fill="#fdf2e9" stroke="#d35400"/>
  <text x="625" y="162" text-anchor="middle" font-family="monospace" font-size="12">[4 x i32]</text>

  <rect x="40" y="186" width="320" height="34" rx="5" fill="#e8f8f5" stroke="#27ae60"/>
  <text x="200" y="208" text-anchor="middle" font-family="monospace" font-size="12">struct Counter { int32 count; }</text>
  <rect x="470" y="186" width="310" height="34" rx="5" fill="#e8f8f5" stroke="#27ae60"/>
  <text x="625" y="208" text-anchor="middle" font-family="monospace" font-size="12">%Counter = type { i32 }</text>

  <rect x="40" y="232" width="320" height="34" rx="5" fill="#fef9e7" stroke="#f1c40f"/>
  <text x="200" y="254" text-anchor="middle" font-family="monospace" font-size="12">T[] / str / string_view</text>
  <rect x="470" y="232" width="310" height="34" rx="5" fill="#fef9e7" stroke="#f1c40f"/>
  <text x="625" y="254" text-anchor="middle" font-family="monospace" font-size="12">%Slice / %str = type { ptr, i64 }</text>

  <rect x="40" y="278" width="320" height="34" rx="5" fill="#fdedec" stroke="#c0392b"/>
  <text x="200" y="300" text-anchor="middle" font-family="monospace" font-size="12">string</text>
  <rect x="470" y="278" width="310" height="34" rx="5" fill="#fdedec" stroke="#c0392b"/>
  <text x="625" y="300" text-anchor="middle" font-family="monospace" font-size="12">%string = type { ptr, i64, i64 }</text>

  <line x1="360" y1="65" x2="462" y2="65" stroke="#888" stroke-width="1.5" marker-end="url(#a6)"/>
  <line x1="360" y1="111" x2="462" y2="111" stroke="#888" stroke-width="1.5" marker-end="url(#a6)"/>
  <line x1="360" y1="157" x2="462" y2="157" stroke="#888" stroke-width="1.5" marker-end="url(#a6)"/>
  <line x1="360" y1="203" x2="462" y2="203" stroke="#888" stroke-width="1.5" marker-end="url(#a6)"/>
  <line x1="360" y1="249" x2="462" y2="249" stroke="#888" stroke-width="1.5" marker-end="url(#a6)"/>
  <line x1="360" y1="295" x2="462" y2="295" stroke="#888" stroke-width="1.5" marker-end="url(#a6)"/>

  <text x="410" y="340" text-anchor="middle" font-size="11" fill="#c0392b">多对一：源类型身份由前端维护，LLVM 只看到布局</text>
  <defs>
    <marker id="a6" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#888"/>
    </marker>
  </defs>
</svg>
</p>
<p align="center"><em>图 6　类型降低：从源语言身份到 LLVM 布局</em></p>

### 4.2 聚合布局与 ABI

&emsp;&emsp;结构体、类、联合体的布局不能各写一份。CDD 把布局计算集中到 `LayoutBuilder`：按声明顺序排列字段，遵循自然对齐，基类子对象占槽 0，联合体取最大成员块，`packed`/`align` 注解通过显式填充字段实现。`getLLVMType` 与 `offset_of` 编译期查询消费同一份结果，因此“源码里看到的偏移”和“IR 里的偏移”不会分叉。

```cpp
// 摘自 src/ast/LayoutBuilder.h
struct LayoutResult {
    std::vector<LayoutField> fields; // 含基类槽 0 与显式 padding
    bool isPacked{false};
    uint64_t size{0};                // getTypeAllocSize 语义
    uint64_t align{0};               // ABI 对齐
    std::unordered_map<std::string, uint64_t> fieldOffsets;
    std::vector<llvm::Type*> fieldTypes() const;
};
```

&emsp;&emsp;布局之后是调用约定。CDD 当前把标量直接按 LLVM 标量传递，聚合按 LLVM 聚合值传递，尚未显式标注 `byval`/`sret`/`signext`/`zeroext`，而是依赖 LLVM 的默认 lowering。这在“自己玩”的语言里完全够用，但一旦要和 C 库、其他编译器或不同优化级别互操作，就必须把 ABI 冻结成规范，并用 ABI 一致性测试锁住。`docs/spec/abi.md` 把这一点标为 `[plan]`，是务实的做法：先让语言跑起来，再逐步收紧边界。

&emsp;&emsp;对系统级语言而言，ABI 还包括位域、联合体、枚举底层类型、原子类型和线程局部存储。CDD 的位域访问通过“读存储单元 → 掩码 → 移位 → 符号扩展”的读改写序列实现；`atomic T` 通过同宽整数载体调用 LLVM 的 `atomic load/store/atomicrmw`；`thread_local` 直接映射为 LLVM 的 thread-local 全局。这些细节不会出现在语法里，却决定了生成代码能否与操作系统和 C 运行时正确对话。

### 4.3 名称修饰与链接

&emsp;&emsp;LLVM 的符号表是扁平的字符串表，没有重载、命名空间或类。前端必须把源语言的名字编码成唯一且稳定的字符串，这就是名称修饰（name mangling）。CDD 的规则很克制：C 链接的 `extern` 符号保持裸名；无参数函数保持原名；其余函数在名字后追加参数类型编码。

```cpp
// 摘自 src/ast/Mangle.cpp
std::string mangleFunction(const std::string& name,
                           const std::vector<Type*>& paramTypes) {
    if (isCName(name)) return name;          // extern "C"：直接使用 C 符号
    if (paramTypes.empty()) return name;     // 无参函数不做修饰
    std::string result = name;
    for (auto* type : paramTypes) {
        result += "_" + typeToMangled(type);
    }
    return result;
}
```

&emsp;&emsp;配合 `typeToMangled`，`int32 add(int32, int32)` 变成 `add_int32_int32`，类方法 `Counter::inc()` 变成 `inc_Counterptr`（隐式 `this` 作为第一个参数），模板实例 `Box<int32>` 变成 `Box$int32`。这些名字在 `tour.cdd` 的 IR 中可以直接看到（节选）：

```llvm
%Counter = type { i32 }
%"Box$int32" = type { i32 }

define void @inc_Counterptr(ptr %this) { ... }
define i32 @twice_int32(i32 %x) { ... }
```

&emsp;&emsp;名称修饰必须满足三个条件：**唯一**（不同重载不冲突）、**稳定**（同一份源码在不同编译单元生成相同符号）、**可链接**（与 C ABI 的边界清晰）。模板实例的名字还要单射：`Box<Box<int32>>` 不能和 `Box<int32>` 编码成同一个字符串。CDD 用 `$` 分隔模板名与实参，用类型名的递归拼写保证单射；代价是符号可读性下降，但调试器与 `llvm-nm` 仍能读懂大致结构。

### 4.4 编译期求值

&emsp;&emsp;LLVM 本身不执行源语言代码，所以常量求值必须由前端自己完成；许多语言用 `constexpr` 表达这类能力，CDD 则用 `compile_time`。`CompileTimeEvaluator` 解释 AST 的常量子集：字面量、算术、位运算、比较、逻辑、三元、常量数组/结构体成员访问、`size_of`/`align_of`/`offset_of`、字符串比较与拼接，并设置递归深度上限防止编译期爆栈。求值成功的结果写回表达式节点（`ctInt`/`ctFloat`/`ctHandled`），代码生成直接发射 LLVM 常量，运行时不再重复求值。

```cpp
// 摘自 src/sema/CompileTimeEvaluator.h
std::optional<ConstValue> eval(ExprAST* expr, ASTNode& at);
std::vector<ReflectField> evalReflectFields(Type* t, ASTNode& at);
std::optional<std::pair<uint64_t, uint64_t>> layoutSizeAlign(Type* t);
```

&emsp;&emsp;`compile_time` 还承担条件编译与反射查询。`compile_time.if (compile_time.target.os == "linux")` 在语义分析阶段求值，未选中分支不参与类型检查与代码生成；`compile_time.size_of(T)`、`fields_of(T)`、`attributes_of(T)` 则把布局与注解信息变成常量或只读表。与 `#if` 相比，它的优势是仍然经过完整语法与作用域检查，不会出现“预处理后源码已不是语言”的问题。

&emsp;&emsp;这种折叠是“能折则折”，不是强制的。当 `compile_time` 函数的实参是运行期值时，语义分析不会报错，而是保留普通调用；`tour.cdd` 里的 `twice(c.count)` 就属于这种情况，IR 中可以看到真实的 `call i32 @twice_int32(i32 %loadtmp)`。反过来，`compile_time int32 answer = twice(3);` 会在 `answer` 的初始化处直接写入常量 `42`。

&emsp;&emsp;模板与编译期求值相互配合。CDD 采用惰性单态化：`template<typename T> struct Box` 在解析阶段只登记模板，首次使用 `Box<int32>` 时才克隆声明、替换类型参数、生成实例并追加到翻译单元尾部。从未使用的模板不会产生任何符号。下面的 `tour.cdd` IR 片段就是这种策略的直接结果：只有 `Box<int32>` 被实例化，`Box<float64>` 不存在。

```llvm
%"Box$int32" = type { i32 }
```

<p align="center">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 230" width="760" height="230" style="max-width:100%;height:auto" font-family="sans-serif">
  <rect x="30" y="40" width="180" height="64" rx="8" fill="#eaf2f8" stroke="#2980b9" stroke-width="2"/>
  <text x="120" y="66" text-anchor="middle" font-size="12" font-weight="bold" fill="#21618c">AST + 常量表达式</text>
  <text x="120" y="88" text-anchor="middle" font-size="10">twice(3) / size_of(Point)</text>

  <rect x="290" y="40" width="180" height="64" rx="8" fill="#e8f8f5" stroke="#27ae60" stroke-width="2"/>
  <text x="380" y="66" text-anchor="middle" font-size="12" font-weight="bold" fill="#1e8449">CompileTimeEvaluator</text>
  <text x="380" y="88" text-anchor="middle" font-size="10">解释执行，深度受限</text>

  <rect x="550" y="40" width="180" height="64" rx="8" fill="#fdf2e9" stroke="#d35400" stroke-width="2"/>
  <text x="640" y="66" text-anchor="middle" font-size="12" font-weight="bold" fill="#a04000">节点写入常量</text>
  <text x="640" y="88" text-anchor="middle" font-size="10">ctInt / ctFloat / foldedValue</text>

  <line x1="210" y1="72" x2="282" y2="72" stroke="#555" stroke-width="2" marker-end="url(#a7)"/>
  <line x1="470" y1="72" x2="542" y2="72" stroke="#555" stroke-width="2" marker-end="url(#a7)"/>

  <rect x="290" y="150" width="180" height="48" rx="8" fill="#fdedec" stroke="#c0392b" stroke-width="2"/>
  <text x="380" y="179" text-anchor="middle" font-size="12" font-weight="bold" fill="#a93226">代码生成</text>
  <line x1="640" y1="104" x2="470" y2="168" stroke="#c0392b" stroke-width="2" marker-end="url(#a7r)"/>
  <text x="565" y="145" font-size="10" fill="#c0392b">llvm::Constant</text>
  <text x="380" y="220" text-anchor="middle" font-size="11" fill="#555">求值发生在 sema，codegen 只负责把结果变成常量，运行时无额外指令</text>
  <defs>
    <marker id="a7" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#555"/>
    </marker>
    <marker id="a7r" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#c0392b"/>
    </marker>
  </defs>
</svg>
</p>
<p align="center"><em>图 7　编译期求值：语义阶段折叠，代码生成阶段落常量</em></p>

## 5 后端：生成 LLVM IR

### 5.1 CodegenContext：IR 的“工作台”

&emsp;&emsp;`CodegenContext` 持有一个 `LLVMContext`、一个 `Module`、一个 `IRBuilder`、一个 `DIBuilder` 和一组作用域。它相当于代码生成阶段的工作台：AST 节点通过 `ctx.getBuilder()` 插入指令，通过 `ctx.getModule()` 查找/创建全局符号，通过 `pushScope`/`declareVariable` 维护局部名字，通过 `getLLVMType` 做类型降低。

```cpp
// 摘自 src/codegen/CodegenContext.cpp
CodegenContext::CodegenContext()
    : context(std::make_unique<llvm::LLVMContext>()),
      builder(*context),
      module(std::make_unique<llvm::Module>("cdd", *context)),
      diBuilder(std::make_unique<llvm::DIBuilder>(*module)) {
    pushScope();
}
```

&emsp;&emsp;CDD 采用“局部变量先 `alloca`、再 `store`/`load`”的朴素策略，而不是一开始就构造 SSA。这大幅降低了代码生成的复杂度：赋值、取地址、循环变量、`switch` 分支都不需要维护 phi 节点。优化器的栈槽提升（`mem2reg`/`SROA`）会在 `-O1` 以上把可提升的 `alloca` 变成 SSA 寄存器，并自动插入 `phi`。这也是 LLVM 教程推荐的入门路径：**先正确，再优化**。

### 5.2 表达式与左值

&emsp;&emsp;表达式代码生成的关键约定是“左值返回地址，右值返回值”。`VariableExprAST::codegen` 返回 `alloca` 地址而不是加载后的值；需要值时，调用方通过 `emitRValue`/`readLValue` 统一加载。这样赋值 `x = y`、取地址 `&x`、自增 `x++` 都能复用同一套地址计算，不必为每种上下文写特殊逻辑。

```cpp
// 摘自 src/ast/Expr.cpp
static llvm::Value* readLValue(CodegenContext& ctx, ExprAST& e, llvm::Value* v) {
    if (!v) return nullptr;
    if (!e.isLValue) return v;
    if (e.isBitFieldAccess()) {
        auto& ma = static_cast<MemberAccessExprAST&>(e);
        return ctx.loadBitFieldValue(v, ma.bitOffset, ma.bitWidth,
                                     e.type, ma.unitBits);
    }
    return ctx.loadValue(v, e.type);
}
```

&emsp;&emsp;二元运算的代码生成先做类型统一，再选择整数/浮点指令。无符号类型使用 `udiv`/`urem`/`lshr`/`icmp ult`，有符号类型使用 `sdiv`/`srem`/`ashr`/`icmp slt`；浮点走 `fadd`/`fmul`/`fcmp`。这段分派是“源语言语义”在 IR 层最直接的体现，也是错误最容易出现的地方：把无符号除法写成 `sdiv`，或者把窄整数提升漏掉，都会生成能运行但结果错误的代码。

```cpp
// 摘自 src/ast/Expr.cpp
lhs = ctx.castValue(lhs, leftAst, opType);
rhs = ctx.castValue(rhs, rightAst, opType);
bool isFloat = opType->isFloatingPointTy();
bool unsignedOp = (leftPtr || rightPtr) ||
                  (resultAstTy && isUnsignedArithmeticType(resultAstTy));

switch (op) {
case BinaryOp::Add:
    return isFloat ? builder.CreateFAdd(lhs, rhs, "addtmp")
                   : builder.CreateAdd(lhs, rhs, "addtmp");
case BinaryOp::Div:
    return isFloat ? builder.CreateFDiv(lhs, rhs, "divtmp")
                   : (unsignedOp ? builder.CreateUDiv(lhs, rhs, "divtmp")
                                 : builder.CreateSDiv(lhs, rhs, "divtmp"));
case BinaryOp::Lt:
    return isFloat ? builder.CreateFCmpOLT(lhs, rhs, "lttmp")
                   : (unsignedOp ? builder.CreateICmpULT(lhs, rhs, "lttmp")
                                 : builder.CreateICmpSLT(lhs, rhs, "lttmp"));
// ... 其余运算符
}
```

&emsp;&emsp;`&&` 与 `||` 需要短路语义，不能直接发 `and`/`or` 指令。CDD 为它们创建 `log.rhs`、`log.short`、`log.merge` 三个基本块，用 `br i1` 控制求值路径，最后用 `phi` 合并结果。指针比较、字符串比较、三元表达式也各自有专门路径；其中字符串比较先比长度再 `memcmp`，并且用分支而不是 `select`，避免长度不等时读越界。

### 5.3 控制流

&emsp;&emsp;控制流的代码生成本质上是“把结构化语句翻译成基本块图”。`if` 创建 then/else/merge 三块；`while` 创建 cond/body/end 三块；`for` 多一个 inc 块；`switch` 用 `switch` 指令或条件链。循环的 `break`/`continue` 通过上下文中的块栈解析，`defer` 则在离开作用域或跳转前统一发射。

```cpp
// 摘自 src/ast/Stmt.cpp
llvm::Value* WhileStmtAST::codegen(CodegenContext& ctx) {
    llvm::Function* func = ctx.getBuilder().GetInsertBlock()->getParent();
    llvm::BasicBlock* condBB =
        llvm::BasicBlock::Create(ctx.getContext(), "while.cond", func);
    llvm::BasicBlock* bodyBB =
        llvm::BasicBlock::Create(ctx.getContext(), "while.body", func);
    llvm::BasicBlock* endBB =
        llvm::BasicBlock::Create(ctx.getContext(), "while.end", func);

    ctx.pushBreakBlock(endBB);
    ctx.pushContinueBlock(condBB);

    ctx.getBuilder().CreateBr(condBB);
    ctx.getBuilder().SetInsertPoint(condBB);

    llvm::Value* condVal = evalCondition(ctx, *cond);
    ctx.getBuilder().CreateCondBr(condVal, bodyBB, endBB);

    ctx.getBuilder().SetInsertPoint(bodyBB);
    body->codegen(ctx);
    if (!ctx.getBuilder().GetInsertBlock()->getTerminator()) {
        ctx.getBuilder().CreateBr(condBB);
    }

    ctx.popContinueBlock();
    ctx.popBreakBlock();
    ctx.getBuilder().SetInsertPoint(endBB);
    return nullptr;
}
```

<p align="center">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 265" width="780" height="265" style="max-width:100%;height:auto" font-family="sans-serif">
  <rect x="20" y="95" width="120" height="54" rx="6" fill="#f4f6f7" stroke="#7f8c8d" stroke-width="2"/>
  <text x="80" y="118" text-anchor="middle" font-family="monospace" font-size="12">entry</text>
  <text x="80" y="136" text-anchor="middle" font-size="10">i = 1</text>

  <rect x="190" y="95" width="150" height="54" rx="6" fill="#eaf2f8" stroke="#2980b9" stroke-width="2"/>
  <text x="265" y="118" text-anchor="middle" font-family="monospace" font-size="12">while.cond</text>
  <text x="265" y="136" text-anchor="middle" font-size="10">icmp sle i, n</text>

  <rect x="390" y="95" width="150" height="54" rx="6" fill="#e8f8f5" stroke="#27ae60" stroke-width="2"/>
  <text x="465" y="118" text-anchor="middle" font-family="monospace" font-size="12">while.body</text>
  <text x="465" y="136" text-anchor="middle" font-size="10">sum += i; i++</text>

  <rect x="590" y="95" width="150" height="54" rx="6" fill="#fdf2e9" stroke="#d35400" stroke-width="2"/>
  <text x="665" y="118" text-anchor="middle" font-family="monospace" font-size="12">while.end</text>
  <text x="665" y="136" text-anchor="middle" font-size="10">return sum</text>

  <line x1="140" y1="122" x2="182" y2="122" stroke="#555" stroke-width="2" marker-end="url(#a8)"/>
  <line x1="340" y1="112" x2="382" y2="112" stroke="#27ae60" stroke-width="2" marker-end="url(#a8g)"/>
  <text x="360" y="104" text-anchor="middle" font-size="10" fill="#1e8449">true</text>
  <path d="M340 132 L340 178 L665 178 L665 149" fill="none" stroke="#c0392b" stroke-width="2" marker-end="url(#a8r)"/>
  <text x="500" y="196" text-anchor="middle" font-size="10" fill="#a93226">false</text>
  <path d="M465 149 L465 218 L265 218 L265 149" fill="none" stroke="#555" stroke-width="2" marker-end="url(#a8)"/>
  <text x="365" y="238" text-anchor="middle" font-size="10" fill="#555">br while.cond</text>

  <defs>
    <marker id="a8" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#555"/>
    </marker>
    <marker id="a8g" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#27ae60"/>
    </marker>
    <marker id="a8r" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#c0392b"/>
    </marker>
  </defs>
</svg>
</p>
<p align="center"><em>图 8　`while` 到基本块 CFG：cond 是唯一的回边入口</em></p>

&emsp;&emsp;CDD 还在 `for` 的代码生成里专门开了一个作用域，原因很具体：`for (int32 i = 0; ...)` 的 `i` 必须只在循环内可见；如果复用函数作用域，两个同名循环会绑定到同一个符号，第二个循环读到上一个循环的结束值，条件立刻为假，循环体根本不执行。这类“语义正确但实现细节错误”的 bug 很难通过阅读语法发现，必须靠测试覆盖。

### 5.4 函数与翻译单元

&emsp;&emsp;函数代码生成分两步。`TranslationUnitAST::codegen` 先预扫所有结构体布局，再预扫所有函数签名（`codegenPrototype`），最后才遍历生成函数体。预扫签名的目的是让前向调用、模板实例方法、互递归函数在调用点都能找到 `llvm::Function`，而不必要求源码按定义顺序排列。

```cpp
// 摘自 src/ast/Decl.cpp
llvm::Value* FunctionDeclAST::codegen(CodegenContext& ctx) {
    // 签名已由 codegenPrototype 创建；复用它
    llvm::Function* function = ctx.getModule().getFunction(mangledName);
    if (!function) function = static_cast<llvm::Function*>(codegenPrototype(ctx));
    if (function->size() > 0) return function;   // 函数体只生成一次

    llvm::BasicBlock* bb =
        llvm::BasicBlock::Create(ctx.getContext(), "entry", function);
    ctx.getBuilder().SetInsertPoint(bb);
    ctx.pushScope();

    unsigned idx = 0;
    for (auto& param : params) {
        llvm::Argument* arg = function->getArg(idx);
        llvm::AllocaInst* alloca = ctx.getBuilder().CreateAlloca(
            ctx.getLLVMType(param->type), nullptr, param->name);
        ctx.getBuilder().CreateStore(arg, alloca);
        ctx.declareVariable(param->name, alloca, param->type);
        idx++;
    }

    body->codegen(ctx);

    if (!ctx.getBuilder().GetInsertBlock()->getTerminator()) {
        if (returnType->kind == TypeKind::Void)
            ctx.getBuilder().CreateRetVoid();
        else
            ctx.getBuilder().CreateUnreachable();
    }
    ctx.popScope();
    return function;
}
```

&emsp;&emsp;“函数体只生成一次”的幂等守卫看似多余，实际是模板与结构体预扫共同作用的结果：同一方法可能先因结构体或模板实例的预扫出码一次，再因主循环到达一次；如果没有 `function->size() > 0` 判断，就会重复插入基本块，IR 校验直接失败。类似地，结构体布局也按“模板实例优先、基类优先”的顺序出码，避免派生类在基类布局尚未存在时引用基类子对象。

### 5.5 运行时辅助函数

&emsp;&emsp;有些语义不适合在每条指令里内联展开，例如 UTF-8 字符计数、动态格式化、字符串查找/切分、拥有型字符串的深拷贝与释放。CDD 的 `CodegenContext` 采用**惰性合成**：第一次用到某个辅助函数时，用 `IRBuilder` 现场生成一个内部链接的函数，之后复用同一个 `llvm::Function*`。这样不需要额外的自定义运行时库，也不会为每个程序都链接一套用不到的代码；辅助函数内部仍会调用 `malloc`、`memcpy`、`snprintf` 等 libc 符号。

```cpp
// 摘自 src/codegen/CodegenContext.h
llvm::Function* getUtf8CharCountFn();
llvm::Function* getFormatDynFn();
llvm::Function* getStrFindFn(bool reverse);
llvm::Function* getSplitFn();
llvm::Function* getStringCopyFn();
llvm::Function* getStringAssignFn();
```

&emsp;&emsp;`string` 的辅助函数尤其能说明“前端必须自己定义运行时语义”。`string` 是拥有型值 `{ptr, len, cap}`，赋值、传参、返回都可能涉及深拷贝；如果直接按 LLVM 聚合值拷贝，两个 `string` 会共享同一块缓冲区，析构时就会双 `free`。CDD 因此在赋值/初始化处调用 `stringCopy` 或 `stringAssign`，后者带自赋值守卫、释放旧缓冲、再深拷贝。这些逻辑无法交给 LLVM，因为 LLVM 不知道“这个指针拥有内存”。

&emsp;&emsp;最终的可执行文件还需要系统运行时。`CompilerDriver` 在链接时追加 `mimalloc`（内置分配器）、`m`（数学库）、`pthread`（线程）和 `atomic`（16 字节原子可能降级为 libatomic 调用），再通过系统 `cc` 完成链接。对于 C FFI，标准库 `std.c` 在语义分析前被注入，声明 `printf`、`malloc`、`memcpy` 等外部符号；这些符号走裸名，不参与名称修饰。

### 5.6 优化与目标代码

&emsp;&emsp;`cdd -S` 输出的是未优化的 IR，局部变量仍是 `alloca`/`load`/`store`。加上 `-O2` 后，驱动使用 LLVM 的新 PassManager 构建默认管线，栈槽提升（`mem2reg`/`SROA`）把可提升的局部变量变成 SSA，`instcombine`、`inline`、`gvn`、循环优化等依次运行。下面两段 IR 取自同一个 `sum_to`：第一段是前端直接生成的结果，第二段是 `opt -passes='default<O2>'` 之后的结果。

```llvm
; 前端直接生成：每个局部变量一个栈槽
define i32 @sum_to_int32(i32 %n) {
entry:
  %n1 = alloca i32, align 4
  store i32 %n, ptr %n1, align 4
  %sum = alloca i32, align 4
  store i32 0, ptr %sum, align 4
  %i = alloca i32, align 4
  store i32 1, ptr %i, align 4
  br label %for.cond

for.cond:
  %loadtmp = load i32, ptr %i, align 4
  %loadtmp2 = load i32, ptr %n1, align 4
  %letmp = icmp sle i32 %loadtmp, %loadtmp2
  br i1 %letmp, label %for.body, label %for.end

for.body:
  %loadtmp3 = load i32, ptr %sum, align 4
  %loadtmp4 = load i32, ptr %i, align 4
  %addtmp = add i32 %loadtmp3, %loadtmp4
  store i32 %addtmp, ptr %sum, align 4
  br label %for.inc
; ... for.inc / for.end
}
```

```llvm
; O2 之后：栈槽提升为 SSA，循环归纳变量与累加器变成 phi
define i32 @sum_to_int32(i32 %n) {
entry:
  %letmp.not4 = icmp slt i32 %n, 1
  br i1 %letmp.not4, label %for.end, label %for.body

for.body:
  %i.06 = phi i32 [ %addtmp6, %for.body ], [ 1, %entry ]
  %sum.05 = phi i32 [ %addtmp, %for.body ], [ 0, %entry ]
  %addtmp = add i32 %i.06, %sum.05
  %addtmp6 = add i32 %i.06, 1
  %letmp.not = icmp sgt i32 %addtmp6, %n
  br i1 %letmp.not, label %for.end, label %for.body

for.end:
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %addtmp, %for.body ]
  ret i32 %sum.0.lcssa
}
```

&emsp;&emsp;优化管线的接入非常短，核心就是注册分析管理器、构建默认管线、运行：

```cpp
// 摘自 src/driver/CompilerDriver.cpp
llvm::LoopAnalysisManager LAM;
llvm::FunctionAnalysisManager FAM;
llvm::CGSCCAnalysisManager CGAM;
llvm::ModuleAnalysisManager MAM;

llvm::PassBuilder PB;
PB.registerModuleAnalyses(MAM);
PB.registerCGSCCAnalyses(CGAM);
PB.registerFunctionAnalyses(FAM);
PB.registerLoopAnalyses(LAM);
PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

llvm::OptimizationLevel level = llvm::OptimizationLevel::O2;
llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(level);
MPM.run(codegenCtx.getModule(), MAM);
```

<p align="center">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 220" width="880" height="220" style="max-width:100%;height:auto" font-family="sans-serif">
  <rect x="15" y="50" width="105" height="70" rx="6" fill="#eaf2f8" stroke="#2980b9" stroke-width="2"/>
  <text x="67" y="78" text-anchor="middle" font-size="11" font-weight="bold" fill="#21618c">前端 IR</text>
  <text x="67" y="98" text-anchor="middle" font-size="9">alloca / load</text>

  <rect x="140" y="50" width="105" height="70" rx="6" fill="#e8f8f5" stroke="#27ae60" stroke-width="2"/>
  <text x="192" y="78" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e8449">mem2reg</text>
  <text x="192" y="98" text-anchor="middle" font-size="9">栈槽 → SSA</text>

  <rect x="265" y="50" width="105" height="70" rx="6" fill="#e8f8f5" stroke="#27ae60" stroke-width="2"/>
  <text x="317" y="78" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e8449">instcombine</text>
  <text x="317" y="98" text-anchor="middle" font-size="9">常量折叠</text>

  <rect x="390" y="50" width="105" height="70" rx="6" fill="#e8f8f5" stroke="#27ae60" stroke-width="2"/>
  <text x="442" y="78" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e8449">inline</text>
  <text x="442" y="98" text-anchor="middle" font-size="9">小函数内联</text>

  <rect x="515" y="50" width="105" height="70" rx="6" fill="#fef9e7" stroke="#f1c40f" stroke-width="2"/>
  <text x="567" y="78" text-anchor="middle" font-size="11" font-weight="bold" fill="#7d6608">loop opts</text>
  <text x="567" y="98" text-anchor="middle" font-size="9">LICM / unroll</text>

  <rect x="640" y="50" width="105" height="70" rx="6" fill="#fef9e7" stroke="#f1c40f" stroke-width="2"/>
  <text x="692" y="78" text-anchor="middle" font-size="11" font-weight="bold" fill="#7d6608">GVN / DCE</text>
  <text x="692" y="98" text-anchor="middle" font-size="9">冗余消除</text>

  <rect x="765" y="50" width="100" height="70" rx="6" fill="#fdf2e9" stroke="#d35400" stroke-width="2"/>
  <text x="815" y="78" text-anchor="middle" font-size="11" font-weight="bold" fill="#a04000">目标代码</text>
  <text x="815" y="98" text-anchor="middle" font-size="9">.o / 汇编</text>

  <line x1="120" y1="85" x2="138" y2="85" stroke="#555" stroke-width="2" marker-end="url(#a9)"/>
  <line x1="245" y1="85" x2="263" y2="85" stroke="#555" stroke-width="2" marker-end="url(#a9)"/>
  <line x1="370" y1="85" x2="388" y2="85" stroke="#555" stroke-width="2" marker-end="url(#a9)"/>
  <line x1="495" y1="85" x2="513" y2="85" stroke="#555" stroke-width="2" marker-end="url(#a9)"/>
  <line x1="620" y1="85" x2="638" y2="85" stroke="#555" stroke-width="2" marker-end="url(#a9)"/>
  <line x1="745" y1="85" x2="763" y2="85" stroke="#555" stroke-width="2" marker-end="url(#a9)"/>

  <text x="440" y="165" text-anchor="middle" font-size="11" fill="#555">O0：跳过优化直接发射；O2/O3：默认管线；自定义 Pass 可挂在 ModulePassManager 上</text>
  <text x="440" y="190" text-anchor="middle" font-size="11" fill="#c0392b">注意：优化级别不是“越高越好”，它会改变调试体验与编译时间</text>
  <defs>
    <marker id="a9" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#555"/>
    </marker>
  </defs>
</svg>
</p>
<p align="center"><em>图 9　优化管线（示意）：从栈式 IR 到 SSA，再到目标代码</em></p>

&emsp;&emsp;如果语言需要自定义优化（例如把特定内建函数降级、做领域专用的别名分析），可以注册自己的 `ModulePass`/`FunctionPass`，插在默认管线之前或之后。CDD 当前只使用默认管线，但驱动已经具备 `PassBuilder` 与分析管理器，扩展成本很低。需要提醒的是，LLVM 的 PassManager API 在不同大版本间有变化（例如新 PM 取代旧 PM、`TargetMachine` 创建接口接受 `llvm::Triple`），代码里因此出现 `#if LLVM_VERSION_MAJOR >= 19` 这类分支；跟随 LLVM 版本升级是长期维护的一部分。

```cpp
// 摘自 src/driver/CompilerDriver.cpp：跨 LLVM 版本的 API 适配
#if LLVM_VERSION_MAJOR >= 19
    module.setTargetTriple(llvmTriple);
#else
    module.setTargetTriple(llvmTriple.str());
#endif

#if LLVM_VERSION_MAJOR >= 20
    hostFeatures = llvm::sys::getHostCPUFeatures();
#else
    llvm::sys::getHostCPUFeatures(hostFeatures);
#endif
```

## 6 工程化：驱动、模块与测试

### 6.1 驱动编排

&emsp;&emsp;编译器不是“词法 + 语法 + 代码生成”三个函数的拼接，而是一条有顺序、有失败分支的流水线。`CompilerDriver::compileFile` 把这条流水线写得很直白：读文件 → 词法 → 模块预加载 → 解析 → 模块名校验 → 注入 `std.c` → 语义分析 → 代码生成 → 可选优化 → 目标文件 → 链接。每一步失败都立即返回，诊断统一格式化。

```cpp
// 摘自 src/driver/CompilerDriver.cpp，删去模块路径与错误处理细节
Lexer lexer(inputFile, content);
auto tokens = lexer.tokenize();

Parser parser(tokens);
auto ast = parser.parse();

SemanticAnalyzer analyzer;
analyzer.setBuildConfig(false, optLevel > 0 ? "O" + std::to_string(optLevel) : "O0");
analyzer.analyze(*ast);
if (!analyzer.getErrors().empty()) return 1;

CodegenContext codegenCtx;
codegenCtx.setSourceFile(inputFile);
ast->codegen(codegenCtx);
codegenCtx.finalizeDebugInfo();

if (optLevel > 0) {
    // 构建 PassBuilder 默认管线并运行
}

if (!emitObjectFile(codegenCtx.getModule(), objectPath)) return 1;
Linker linker;
return linker.link({objectPath}, outputFile, linkLibs, linkPaths);
```

&emsp;&emsp;驱动还负责“编译器自己不该知道的事”：标准库目录如何查找、`import` 的相对路径如何解析、`-M`/`-L`/`-l` 如何传给链接器、`--no-prelude` 如何关闭 `std.c` 注入。把这些放在驱动层而不是语义层，能让语言核心保持纯粹，也方便测试时替换模块路径。

### 6.2 模块与标准库

&emsp;&emsp;CDD 的模块系统用点分模块名映射到目录：`import std.math;` 解析为 `std/math.cdd`，模块名必须与文件路径一致；`module std.math;` 声明后，只有 `export`/`public` 的声明对外可见。`ModuleLoader` 负责查找、解析、去重和拼接声明；`StdPrelude` 在语义分析前注入 `std.c` 的 libc 绑定层。这样标准库既可以用 CDD 编写，又不需要编译器内嵌大量声明。

```cpp
module std.math;

export float64 square(float64 x) { return x * x; }
```

&emsp;&emsp;上面的 `square` 只是形式示例；真实的 `std.math` 还包含 `sqrt`、三角函数与常量，并通过 `std.c` 绑定 `libm`。

&emsp;&emsp;标准库放在 `libs/std/`，通过 `CDD_STD_DIR`、安装目录或可执行文件相对路径查找。容器、字符串、文件、线程都用 CDD 自身实现，编译器只提供内建类型与少量内建函数。这种“自举式”组织方式有一个隐性好处：语言缺什么，写标准库时立刻会暴露；标准库需要的特性，也会反过来推动语言演进。

### 6.3 测试策略

&emsp;&emsp;编译器测试不能只靠“编译一个 hello world”。CDD 的测试分为三层：前端单元测试（词法、语法、诊断码）、语义/代码生成测试（IR 结构、布局、重载解析）、端到端测试（用 LLJIT 直接执行 IR 或运行生成的可执行文件）。端到端测试的关键是验证 `main` 的返回值，而不是比对完整 IR 文本，因为 IR 会随 LLVM 版本变化。

```cpp
// 摘自 tests/e2e/test_arithmetic_conversions.cpp，删去错误处理
Lexer lexer(file, source);
auto tokens = lexer.tokenize();
Parser parser(tokens);
auto ast = parser.parse();

SemanticAnalyzer analyzer;
analyzer.analyze(*ast);

CodegenContext ctx;
ast->codegen(ctx);

std::string err;
llvm::raw_string_ostream os(err);
if (llvm::verifyModule(ctx.getModule(), &os)) return -1;

auto jit = llvm::orc::LLJITBuilder().create();
auto ts = llvm::orc::ThreadSafeModule(ctx.takeModule(), ctx.takeContext());
(*jit)->addIRModule(std::move(ts));
auto sym = (*jit)->lookup("main");
return ((int (*)())(intptr_t)sym->getValue())();
```

<p align="center">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 300" width="640" height="300" style="max-width:100%;height:auto" font-family="sans-serif">
  <rect x="190" y="24" width="260" height="58" rx="6" fill="#fdedec" stroke="#c0392b" stroke-width="2"/>
  <text x="320" y="48" text-anchor="middle" font-size="12" font-weight="bold" fill="#a93226">端到端</text>
  <text x="320" y="68" text-anchor="middle" font-size="10">LLJIT 执行 / 可执行文件退出码</text>

  <rect x="120" y="100" width="400" height="58" rx="6" fill="#fef9e7" stroke="#f1c40f" stroke-width="2"/>
  <text x="320" y="124" text-anchor="middle" font-size="12" font-weight="bold" fill="#7d6608">集成</text>
  <text x="320" y="144" text-anchor="middle" font-size="10">verifyModule + IR 结构 / 布局 / 名称修饰</text>

  <rect x="50" y="176" width="540" height="58" rx="6" fill="#e8f8f5" stroke="#27ae60" stroke-width="2"/>
  <text x="320" y="200" text-anchor="middle" font-size="12" font-weight="bold" fill="#1e8449">单元</text>
  <text x="320" y="220" text-anchor="middle" font-size="10">Lexer / Parser / Sema / LayoutBuilder / Mangle</text>

  <text x="320" y="272" text-anchor="middle" font-size="12" fill="#555">1900+ 个测试：越往下越快、越往上越接近真实程序</text>
  <text x="320" y="292" text-anchor="middle" font-size="10" fill="#888">先写失败测试，再实现语义；IR 文本断言只用于稳定结构</text>
</svg>
</p>
<p align="center"><em>图 10　编译器测试金字塔：单元、集成、端到端</em></p>

&emsp;&emsp;测试之外，诊断也是工程化的一部分。CDD 给每个错误分配稳定编号（`E0xxx` 词法、`E1xxx` 语法、`E2xxx` 语义、`W3xxx` 警告），并保留源码位置、严重级别和可选修复建议。稳定编号让文档、测试和 IDE 可以引用同一条错误，而不是依赖易变的错误文本。

## 7 实践路线与常见陷阱

&emsp;&emsp;如果要从零实现一门基于 LLVM 的语言，建议按“能跑通的最小闭环”推进，而不是按教科书的模块顺序逐个写完再联调。下面的路线在 CDD 的演进中反复被验证。

| 阶段 | 目标 | 最小验收 |
|---|---|---|
| 1 | 词法 + 表达式解析 | `1 + 2 * 3` 能生成 AST 并打印 |
| 2 | 最小代码生成 | `int32 main() { return 42; }` 能编译运行 |
| 3 | 变量、赋值、算术 | 局部变量与四则运算结果正确 |
| 4 | 控制流 | `if`/`while`/`for` 的 CFG 与 `break`/`continue` 正确 |
| 5 | 函数与调用 | 参数、返回值、前向调用、递归 |
| 6 | 聚合与指针 | 结构体、数组、指针、`sizeof`/`offsetof` |
| 7 | 语义增强 | 类型检查、重载、名字修饰、诊断 |
| 8 | 模块与运行时 | `import`、标准库、C FFI、链接 |
| 9 | 优化与调试 | `-O2`、调试信息、IR 校验 |
| 10 | 进阶特性 | 泛型、闭包、编译期求值、原子/线程 |

&emsp;&emsp;下面这些陷阱在 CDD 的代码和提交历史里都能找到对应痕迹，值得提前规避。

- **类型身份丢失**：只把源类型映射为 `i32`/`i64`，等到重载、模板、调试信息时才发现无法区分 `int32` 与 `uint32`、`A` 与 `B`。解决方法是保留独立的 `Type` 对象，LLVM 类型只是它的 lowering 结果。
- **有符号/无符号指令选错**：`sdiv` 与 `udiv`、`ashr` 与 `lshr`、`icmp slt` 与 `icmp ult` 必须由源类型决定，不能凭 LLVM 类型猜。
- **窄整数提升遗漏**：`uint8 + uint8` 在 C 语义下先提升为 `int`；如果直接 `add i8`，高位会丢失。CDD 在 sema 计算公共类型，在 codegen 用 `zext`/`sext` 显式提升。
- **`alloca` 与 SSA 混淆**：入门阶段用 `alloca` 是对的，但不要手工维护 phi；把 `mem2reg` 留给优化器，先保证语义正确。
- **聚合值拷贝别名**：含指针/拥有型字段的结构体不能按 LLVM 聚合值直接拷贝；`string` 的双 `free` 就是这么产生的。需要字段感知的拷贝/释放助手。
- **名字修饰不稳定**：同一函数在不同编译单元生成不同符号，链接时就会找不到。修饰规则必须只依赖源语言类型，不依赖指针地址、遍历顺序或临时 ID。
- **模块加载顺序错误**：类型声明必须在解析使用点之前可见；CDD 因此在解析主文件前预加载 `import`。
- **LLVM 版本 API 漂移**：`TargetMachine`、`Triple`、`getHostCPUFeatures`、指针类型等接口在不同版本有差异，用 `LLVM_VERSION_MAJOR` 分支隔离，并锁定 CI 的 LLVM 版本。
- **只测“能编译”，不测“语义正确”**：一定要有端到端返回值测试和 IR 校验（`verifyModule`），否则错误会以“能跑但结果不对”的形式潜伏。

&emsp;&emsp;构建与验证命令可以保持很短：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$(llvm-config --cmakedir)"
cmake --build build -j"$(nproc)"

./build/bin/cdd -S demo.cdd                 # 查看前端生成的 IR
./build/bin/cdd demo.cdd -o demo && ./demo  # 编译并运行
ctest --test-dir build -j"$(nproc)"         # 跑测试
```

&emsp;&emsp;如果链接阶段出现 `spdlog`/`fmt`、`ncurses`/`tinfo` 之类的未定义符号，通常不是编译器代码的问题，而是构建环境里多个包管理器提供了不同版本的库。统一 `fmt_DIR`、`spdlog_DIR` 与 LLVM 的 CMake 配置，或者使用项目提供的 `vcpkg` 配置，往往比改代码更有效。

## 8 结语

&emsp;&emsp;LLVM 把“生成机器码”这件苦活变成了可复用的基础设施，但它没有、也不应该替语言设计者做决定。CDD 的实现说明：真正的工作量集中在前端与中端——把源语言的类型、作用域、重载、模板、布局、名字和运行时语义，准确地降级为 LLVM IR 能够表达的约定。词法与语法决定语言“长什么样”，语义分析决定语言“是什么意思”，代码生成决定这些意思“如何变成机器指令”，而驱动、模块、测试与诊断决定这门语言“能不能长期维护”。

&emsp;&emsp;对想动手实现语言的人来说，最稳妥的路径是：先写清楚语言契约，再做一个能编译 `return 42` 的最小闭环；然后按“变量 → 控制流 → 函数 → 聚合 → 模块 → 优化”的顺序逐步加特性，每加一个特性就补端到端测试。LLVM 的 IR 校验器、`opt`、`llc` 和 LLJIT 会让这条路径比想象中短得多。真正需要耐心的是语义细节：一个有符号除法、一次窄整数提升、一个字符串所有权规则，往往比一整套 Pass 更能决定语言是否可靠。

&emsp;&emsp;当语言能够稳定地编译自己编写的标准库时，它才真正从“玩具”变成了“工具”。CDD 用 `libs/std/` 里的 CDD 代码验证了这一点：编译器提供能力，标准库检验能力，测试锁定能力。三者形成闭环，LLVM 只是这条闭环上最可靠的一环。
