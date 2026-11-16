# LLVM IR

&emsp;&emsp;LLVM IR是LLVM编译器中前端和后端之间的中间表示。前端部分（如clang）将编程语言从文本源码翻译成IR,后端优化器再将IR降低成机器汇编。IR的设计思想是软件设计中典型的抽象思路，在原来的编译器前端和后端之间抽象出一层IR,该IR包含控制流图，显示类型等基本的程序结构，又同时剥离了语言的具体语法细节和目标及其的硬件细节，让不同模块可以专注自己的工作。比如现如今如果期望实现自己的编程语言完全可以使用LLVM实现前端，后端完全复用LLVM的后端优化。比如编程原神——Rust初期就是这样快速推进的。

## 1. LLVM IR 简介

### 1.1 IR 在编译流程中的位置

&emsp;&emsp;我们平时写一个C程序，编译器将代码编译成可执行文件，中间发生的事情远比想象中多。以Clang为例，整个流程大致是这样的：

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261116_390a16.svg)

&emsp;&emsp;Clang前端负责把C源码翻译成LLVM IR。这一步主要进行词法分析、语法分析、语义分析，以及生成IR。IR生成之后，优化器（opt）会在IR上运行一系列Pass，对程序进行等价重写来优化布局。最后，后端（llc）把IR降为目标平台的汇编代码，再交给汇编器和链接器生成最终的可执行文件可执行文件。整个过程中IR充当的是前端和后端的桥阶层。

&emsp;&emsp;这个桥阶层将编译器按层抽象开，前端只需要面向 IR 生成代码，不必关心最终跑在 x86 还是 ARM 上；后端只需要消费 IR，不必区分源码是 C、C++ 还是 Rust。相比于传统的编译器将前端和后端耦合在一起，每增加一种源语言或者一种目标架构，都需要重新写一遍完整的编译器而言，IR作为前后端的契约极大的简化了编译器的设计和实现。这样优化器只需要在 IR 这一层实现一次，就能同时惠及所有源语言和所有目标架构。

### 1.2 IR 的三种形式

&emsp;&emsp;LLVM IR在实际使用中有三种存在形式，它们之间可以无损转换：
- **文本形式**，文件扩展名是 `.ll`。这是人类可读的格式，适合教学、调试和手写实验。
- **二进制形式**，也就是 bitcode，文件扩展名是 `.bc`。这是紧凑的序列化格式，适合链接时优化和分发。你平时用 `clang -c` 编译出来的 `.o` 文件里，如果启用了 LTO，里面装的就是 bitcode。
- **内存中的 C++ 对象**。LLVM 提供了一套 C++ API，`Module`、`Function`、`Instruction` 等都是这套 API 里的类。Pass 和 JIT 直接操作的就是这些对象。文本和二进制形式，本质上只是这些对象在磁盘上的序列化表示。

&emsp;&emsp;比如示例程序：
```cpp
#include <stdio.h>

int a;
int add();
int compute(int a, int b, int c) {
    printf("");
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

int add(){ return 0;}
```

&emsp;&emsp;可以通过命令`clang++-22 -O0 -S -emit-llvm main.cpp -o main.ll `得到对应的ir文本：
```cpp
; ModuleID = 'compute.c'
source_filename = "compute.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@a = dso_local global i32 0, align 4
@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local noundef i32 @_Z7computeiii(i32 noundef %0, i32 noundef %1, i32 noundef %2) #0 {
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store i32 %0, ptr %5, align 4
  store i32 %1, ptr %6, align 4
  store i32 %2, ptr %7, align 4
  %8 = call i32 (ptr, ...) @printf(ptr noundef @.str)
  %9 = load i32, ptr %5, align 4
  %10 = load i32, ptr %6, align 4
  %11 = icmp sgt i32 %9, %10
  br i1 %11, label %12, label %14

12:                                               ; preds = %3
  %13 = load i32, ptr %5, align 4
  store i32 %13, ptr %4, align 4
  br label %16

14:                                               ; preds = %3
  %15 = load i32, ptr %6, align 4
  store i32 %15, ptr %4, align 4
  br label %16

16:                                               ; preds = %14, %12
  %17 = load i32, ptr %4, align 4
  ret i32 %17
}

declare i32 @printf(ptr noundef, ...) #1

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z3addv() #2 {
  ret i32 0
}

attributes #0 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"Ubuntu clang version 22.1.2 (1ubuntu1)"}
```
&emsp;&emsp;而文本与二进制之间可以用 `llvm-as` 和 `llvm-dis` 互相转换：

```bash
llvm-as main.ll -o main.bc      # 文本 → 二进制
llvm-dis main.bc -o main.ll     # 二进制 → 文本
```

&emsp;&emsp;如果期望直接生成bitcode,对应的命令是：

```bash
clang -O0 -emit-llvm -c src.c -o src.bc
```

>`-O0` 表示不做优化。这一点很重要——本文后续会看到，`-O0` 生成的 IR 和优化后的 IR 差异非常大，前者保留了大量的栈操作，后者则干净得多。我们先从 `-O0` 的版本开始看，因为它更接近 C 源码的原始结构。

## 2. LLVM IR 骨架：Module → Function → BasicBlock → Instruction

&emsp;&emsp;上一章我们从整体上认识了 LLVM IR，知道了它在编译流程中的位置，也看到了一个 C 函数对应的 IR 长什么样。这一张来详细描述下LLVM IR中的具体内容。一个 `.ll` 文件在结构上只有四层：`Module`、`Function`、`BasicBlock`、`Instruction`。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261116_ff5351.svg)

### 2.1 Module：顶层容器

&emsp;&emsp;一个 `.ll` 文件对应一个 `Module`。可以把 `Module` 理解成"一个编译单元"——它包含了这个文件里所有的全局信息、函数声明和函数定义。

&emsp;&emsp;我们先用 `clang -O0 -emit-llvm -S compute.c -o compute.ll` 生成上一章那个 `compute` 函数的 IR，然后打开 `compute.ll`。文件的开头是这样的：

```llvm
; ModuleID = 'compute.c'
source_filename = "compute.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
```

&emsp;&emsp;这四行就是 `Module` 的头部信息。第一行 `ModuleID` 是一个注释，记录了这个模块是从哪个文件来的，仅供调试参考。第二行 `source_filename` 也是信息性的，用于诊断信息。真正重要的是后两行：`target datalayout` 和 `target triple`。

&emsp;&emsp;`target triple` 描述的是目标平台，格式通常是 `<架构>-<厂商>-<操作系统>`，比如这里的 `x86_64-pc-linux-gnu` 表示 x86-64 架构、未知厂商、Linux 系统、GNU 环境。`target datalayout` 则用一串紧凑的字段描述了更细的内存布局信息——端序是小端（`e`）还是大端（`E`），指针是多少位、对齐是多少，整数和浮点的默认对齐是多少，等等。这个字段在优化和代码生成时非常重要，如果省略了，某些Pass会报警告甚至拒绝运行。

&emsp;&emsp;除了头部信息，`Module` 里还可以包含三类顶层内容：

- **全局变量和常量**：以 `@` 开头，比如 `@a = dso_local global i32 0, align 4`。
- **函数声明**：用 `declare` 开头，只有函数签名，没有函数体。比如 `declare i32 @printf(ptr, ...)`。
- **函数定义**：用 `define` 开头，既有函数签名，又有函数体。

&emsp;&emsp;这里的 `@` 是一个约定：模块级符号（全局变量、函数名）用 `@` 开头，函数内部的局部符号（虚拟寄存器、基本块标签）用 `%` 开头。看到 `@` 就知道它是全局的，看到 `%` 就知道它是某个函数内部的——这个约定贯穿整个 IR 语法。

&emsp;&emsp;文件末尾一堆以 `!` 开头的元数据，比如 `!llvm.module.flags`、`!llvm.ident`。这些是模块级的附加信息，不影响程序的语义。

### 2.2 Function：可调用单元

&emsp;&emsp;`Module` 里最重要的内容就是函数。`Function` 以 `@` 开头，是模块级可见的符号。函数分为两种：声明（`declare`）和定义（`define`）。声明只告诉编译器"有这么个函数，签名是这样的"，不包含函数体；定义则给出完整的函数体。

```llvm
define dso_local noundef i32 @_Z7computeiii(i32 noundef %0, i32 noundef %1, i32 noundef %2) #0 {
```

&emsp;&emsp;这一行包含了不少信息，逐部分拆开看：

- `define`：表示这是一个函数定义。
- `dso_local`：这是一个链接属性（linkage），表示这个符号在最终链接单元内可以解析，不需要走 GOT/PLT。这是现代 Clang 在 Linux 上常见的修饰符。
- `i32`：函数的返回类型。
- `@_Z7computeiii`：函数名，这是一个典型的cpp签名。
- `(i32 noundef %0, i32 noundef %1, i32 noundef %2)`：参数列表。每个参数都有自己的类型（`i32`）、参数属性（`noundef`，表示这个参数不会是未定义值）和对应的名字（`%0`、`%1`、`%2`）。
- `#0`：属性组编号。Clang 会把一组函数属性（比如 `noinline`、`nounwind`）打包成一个编号，在文件末尾用 `attributes #0 = { ... }` 定义。
- `{ ... }`：函数体，由若干个基本块组成。

&emsp;&emsp;声明和定义的语法基本一样，只是声明以 `declare` 开头，并且没有函数体：

```llvm
declare i32 @printf(ptr noundef, ...) #1
```

&emsp;&emsp;这行声明表示有一个叫 `printf` 的函数，返回 `i32`，第一个参数是 `ptr`（指针），后面是可变参数。

&emsp;&emsp;函数内部的局部标识符（`%a`、`%b`、`%c`、`entry` 等）只在函数作用域内有效。不同的函数可以有同名的局部标识符，它们互不干扰。这一点和 C 语言里不同函数可以有同名局部变量是一样的道理。

### 2.3 BasicBlock：控制流的基本单位

&emsp;&emsp;函数体由若干个**基本块**（`BasicBlock`）组成。基本块是控制流的基本单位——它是一段顺序执行的指令序列，中间没有分支，也没有跳转的入口。

&emsp;&emsp;一个基本块以 `label:` 开始，以**恰好一条终结指令**结束。终结指令（terminator）是 `ret`、`br`、`switch`、`unreachable` 这些能改变控制流的指令。如果一个基本块里在 `br` 之后又写了一条 `ret`，IR 验证器会直接报错——一个基本块不允许有第二条终结指令。

&emsp;&emsp;对应的 `-O0` IR（省略了 `alloca`/`load`/`store` 的细节）大致是这样的，如果熟悉汇编的话很容易读懂这一段代码，就是典型的compare跳转逻辑：

```llvm
  %11 = icmp sgt i32 %9, %10
  br i1 %11, label %12, label %14

12:                                               ; preds = %3
  %13 = load i32, ptr %5, align 4
  store i32 %13, ptr %4, align 4
  br label %16

14:                                               ; preds = %3
  %15 = load i32, ptr %6, align 4
  store i32 %15, ptr %4, align 4
  br label %16

16:                                               ; preds = %14, %12
  %17 = load i32, ptr %4, align 4
  ret i32 %17
```

&emsp;&emsp;把基本块看作节点、把终结指令里的跳转目标看作边，就得到了函数的**控制流图**（CFG）。控制流图是后面所有数据流分析和优化的基础，理解它是读懂优化过程的关键。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261116_c5f534.svg)

### 2.4 Instruction：指令即值

&emsp;&emsp;基本块里是一条一条的指令。大多数指令的形式是这样的：

```llvm
%result = opcode operand1, operand2, ...
```

&emsp;&emsp;等号左边是一个虚拟寄存器（`%result`），等号右边是指令本身。**指令产生一个 SSA 值**，这个值可以被后续指令使用。这就是 LLVM IR 的"指令即值"（instruction is value）的设计。一段典型的+/的ir如下：

```llvm
%add = add nsw i32 %0, %1
%div = sdiv i32 %2, %3
```

&emsp;&emsp;`%add = add nsw i32 %0, %1` 的意思是：把 `%0` 和 `%1` 相加，结果保存在 `%add` 里。`nsw` 是 `add` 的一个修饰符（`no signed wrap`），表示"假设这个加法不会发生有符号溢出"。这给优化器提供了额外的推理依据，但如果实际溢出了，行为是未定义的。`%div = sdiv i32 %2, %3` 同理，`sdiv` 是有符号除法。

&emsp;&emsp;不是所有指令都产生普通值。`store`、`br`、`ret`、`switch`、`unreachable` 这些指令就没有等号左边的结果。它们要么是终结指令（控制流的一部分），要么是只产生副作用（比如写内存）的指令。`alloca` 是一个特殊的存在：它会返回一个指针值，但它的语义是"在栈上分配空间"，所以在 `-O0` 的 IR 里，`alloca` 的结果通常被 `store` 和 `load` 使用。

&emsp;&emsp;从 C++ 对象模型的角度看，`Function`、`BasicBlock`、`Instruction` 都继承自 `Value`，同时指令又是 `User`——它使用其他 `Value` 作为操作数。这种"既是值又是使用者"的双重身份，是 IR 里所有数据流分析的基石。在文本 IR 里，它表现为：`%0` 在某一行被定义，在后续行被引用。当你看到 `%0` 时，只要往上找唯一的那次定义，就知道它是什么值——这正是 SSA 带来的便利，后续会详细展开。

&emsp;&emsp;指令的类型系统是强制的。`add nsw i32 %0, %1` 里的 `i32` 不是可有可无的装饰，它声明了这条指令操作数的类型，也声明了结果的类型。两个操作数必须是 `i32`，不能一个是 `i32`、一个是 `i64`。如果确实需要混合宽度，必须先写显式的 `zext` 或 `sext` 指令做扩展。。

## 3. SSA：LLVM IR的核心约束

&emsp;&emsp;上一章拆开了 LLVM IR 的骨架，知道了 `Module`、`Function`、`BasicBlock`、`Instruction` 这四层结构。不过在讲指令的时候，注意的话能够发现目前生成的IR里到处都是 `alloca`、`load`、`store`，而优化后的 IR 里这些指令全都不见了？

>下面使用命令`clang++-22 -O3 -emit-llvm -S compute.c -o compute.ll`生成，相比上面的命令区别是优化等级。

```cpp
; ModuleID = 'compute.c'
source_filename = "compute.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@a = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @_Z7computeiii(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = tail call i32 @llvm.smax.i32(i32 %0, i32 %1)
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @_Z3addv() local_unnamed_addr #0 {
  ret i32 0
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}
!llvm.errno.tbaa = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Ubuntu clang version 22.1.2 (1ubuntu1)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
```

&emsp;&emsp;这个问题的答案就藏在 SSA 里。SSA 是 LLVM IR 最核心的约束。

### 3.1 SSA

&emsp;&emsp;SSA 的全称是 Static Single Assignment，直译过来是"静态单赋值"。名字里的关键词是"单赋值"：**每个虚拟寄存器只被赋值一次**。这句话听起来很简单，但它和C语言里的习惯完全不同。在 C 里，一个变量可以反复赋值：

```c
int x = 1;
x = x + 1;
x = x * 2;
```

&emsp;&emsp;`x` 被赋值了三次，每次赋的值都不一样。而在 SSA 形式的 IR 里，同样的事情会变成这样：

```llvm
%1 = alloca i32, align 4
%2 = alloca i32, align 4
store i32 0, ptr %1, align 4
store i32 1, ptr %2, align 4
%3 = load i32, ptr %2, align 4
%4 = add nsw i32 %3, 1
store i32 %4, ptr %2, align 4
%5 = load i32, ptr %2, align 4
%6 = mul nsw i32 %5, 2
store i32 %6, ptr %2, align 4
%7 = load i32, ptr %2, align 4
```

&emsp;&emsp;注意看：没有哪个变量被赋值了两次。每一次赋值都产生一个新的名字（`%2`、`%4`、`%6`），旧名字的值永远不会改变。这就是 SSA 的核心思想——**变量不再是"可以反复覆写的容器"，而是"一次性的值定义"**。

### 3.2 优化器与SSA

&emsp;&emsp;优化器的工作本质上是分析数据在程序里的流动，然后在不改变程序语义的前提下改写代码。在非 SSA 的形式下，分析数据流是一件很麻烦的事情——当看到 `x` 的时候，不知道它当前的值是从哪里来的，得往回追溯，一路上还要考虑各种分支汇合的情况。而在 SSA 形式下，这件事变得简单了。因为每个值只定义一次，所以当看到 `%2` 的时候，往上找唯一的那次定义，就知道它是什么值。**定义–使用链（def-use chain）是显式且唯一的**。这个性质让很多优化变得非常直接：

- **常量传播**：如果 `%1 = add i32 1, 0` 的右边全是常量，那就可以直接算出结果，把 `%1` 替换成常量。不用考虑 `%1` 会不会在后面被改掉——它不可能被改掉。
- **死代码消除**：如果某个定义没有任何使用者，那它就可以直接删掉。不用考虑它有没有副作用——大部分算术指令没有副作用。
- **全局值编号**：如果两个计算的操作数完全一样，那它们就是同一个值，可以合并。在 SSA 下，判断"操作数完全一样"只需要比较两个虚拟寄存器是不是同一个。
- **寄存器分配**：SSA 形式天然对应活跃变量分析，活跃区间清楚，寄存器分配的冲突图更容易构建。

&emsp;&emsp;可以这么说：SSA 把数据流分析从"追踪变量状态"变成了"遍历图上的边"。这是 LLVM IR 选择 SSA 作为核心表示的根本原因——**优化的便利性优先于人类书写的直觉**。

### 3.3 `mem2reg`与SSA

&emsp;&emsp;上一节用 `clang -O0` 生成的的 IR里，`x` 这个变量被实现成了一个 `alloca` 出来的栈槽：先 `store` 写入，再 `load` 读出。由于C 的变量是可以反复赋值的，而 SSA 不允许，所以 `-O0` 的Clang干脆把变量放到栈上，用内存来模拟"可反复赋值"的语义。这样一来，IR 里就没有任何变量被赋值两次——每次 `store` 都是对内存的写入，不是对虚拟寄存器的赋值。这是合法的 SSA 形式（内存操作不受 SSA 约束），但它并不是"真正的 SSA"。虚拟寄存器层面确实满足单赋值，但数据流被藏在内存里，优化器看不出 `%1` 和 `%2` 之间的关系。

&emsp;&emsp;`mem2reg` 这个 Pass 的职责就是把这种"伪 SSA"变成"真 SSA"。它的名字是 "memory to register" 的缩写——把内存操作提升为寄存器操作。用法很简单：

```bash
//这里如果只用-O0的话opt阶段还是不会优化因此要加上-disable-O0-optnone
clang-22 -O0 -Xclang -disable-O0-optnone -S -emit-llvm main.c -o main.ll
opt -S -passes=mem2reg main.ll -o compute.mem2reg.ll 
```

&emsp;&emsp;运行完之后，再看 `compute.mem2reg.ll`，内容变成了这样：

```llvm
define dso_local i32 @main() #0 {
  %1 = add nsw i32 1, 1
  %2 = mul nsw i32 %1, 2
  ret i32 %2
}
```

&emsp;&emsp;所有的 `alloca`、`load`、`store` 都不见了。`sum` 这个变量被提升成了虚拟寄存器 `%add`，直接参与后续计算。这才是真正的 SSA 形式——数据流清清楚楚地写在指令之间的引用关系里，没有任何内存中介。**`mem2reg` 的适用条件**是：这个 `alloca` 出来的变量必须满足几个条件——它的地址没有被泄露出去（没有传给函数、没有存到别的地方）、它的所有读写都能被静态地追踪到。满足这些条件的 `alloca` 就可以被安全地提升。不满足的（比如取过地址传给别的函数）就只能留在内存里。

&emsp;&emsp;既然 `mem2reg` 这么好，为什么 Clang 在 `-O0` 下不直接生成 SSA 形式的 IR 呢？原因在于，从 C 源码直接生成 SSA 形式需要做支配关系分析、插入 `phi` 节点，这个过程本身比较复杂，而且调试的时候不容易对应回源码。`-O0` 的目标是"生成快、调试友好"，所以 Clang 选择先生成朴素的栈式 IR，把 SSA 化的工作交给 `mem2reg`。

### 3.4 phi：控制流汇合处的选择

&emsp;&emsp;`compute` 这个函数太简单了，它只有一个基本块，没有分支。在只有一个基本块的函数里，SSA 化很直接——把 `alloca` 换成虚拟寄存器就行，不需要处理任何汇合。考虑下面这个 C 函数：

```c
int foo(int a, int b) {
    if (a > b) {
        return a + b + 2;
    } else {
        return a * b * 17;
    }
}
```

&emsp;&emsp;它有两个分支，分别计算不同的结果，然后返回。用 `clang -O0` 生成 IR，再跑一遍 `mem2reg`，会得到类似这样的结果：

```llvm
define dso_local i32 @foo(i32 noundef %0, i32 noundef %1) #0 {
  %3 = icmp sgt i32 %0, %1
  br i1 %3, label %4, label %7

4:                                                ; preds = %2
  %5 = add nsw i32 %0, %1
  %6 = add nsw i32 %5, 2
  br label %10

7:                                                ; preds = %2
  %8 = mul nsw i32 %0, %1
  %9 = mul nsw i32 %8, 17
  br label %10

10:                                               ; preds = %7, %4
  %.0 = phi i32 [ %6, %4 ], [ %9, %7 ]
  ret i32 %.0
}
```

&emsp;&emsp;注意 `return` 这个基本块的第一条指令：

```llvm
%.0 = phi i32 [ %6, %4 ], [ %9, %7 ]
```

&emsp;&emsp;这条指令就是 `phi`。它的语义是：**根据控制流是从哪个前驱到达的，选择对应的值**。这就是一个lable和取值的映射表pi指令根据实际情况选择：如果是从 `%4` 跳过来的，就取 `%6` 的值；如果是从 `%7` 跳过来的，就取 `%9` 的值。

&emsp;&emsp;为什么需要 `phi`？`phi` 是 SSA 形式在多前驱汇合点上的值合并器：因为 SSA 要求每个虚拟寄存器只能被定义一次，而分支汇合处又需要得到一个“取决于从哪个前驱路径进入”的值，所以 `phi` 在汇合基本块开头用 `[值, 前驱基本块]` 列表显式声明每个前驱对应的取值，从而在不违反单赋值约束的前提下把控制流路径信息编码进数据流，它本身不产生机器码，只是 IR 层的选择语义。

&emsp;&emsp;`phi` 的语法有一点特殊：

```llvm
%result = phi <类型> [ <值1>, <标签1> ], [ <值2>, <标签2> ], ...
```

&emsp;&emsp;方括号里的每一对 `[值, 标签]` 是一个"incoming value"，表示"如果控制流从标签对应的前驱到达，就取这个值"。一个 `phi` 必须为它的每一个前驱都提供一个 incoming value，少了或多了都会验证失败。

&emsp;&emsp;`phi` 的位置也有限制：**必须位于基本块的最开头**，先于任何非 `phi` 指令。如果有多个 `phi`，它们要挨在一起，排在最前面。这是 SSA 形式的一个技术性要求，原因是 `phi` 的语义是"在进入基本块的入口处同时选择"，如果允许它在其他指令之后，那"同时"的语义就没法保证了。

&emsp;&emsp;`phi` 不仅在分支汇合处出现，在循环里也会出现。考虑一个简单的计数循环：

```c
int sum_to_n(int n) {
    int sum = 0;
    for (int i = 1; i <= n; i++) {
        sum += i;
    }
    return sum;
}
```

&emsp;&emsp;这个函数的 IR 里，循环头会有类似这样的 `phi`：

```llvm
define dso_local i32 @sum_to_n(i32 noundef %0) #0 {
  br label %2

2:                                                ; preds = %6, %1
  %.01 = phi i32 [ 0, %1 ], [ %5, %6 ]
  %.0 = phi i32 [ 1, %1 ], [ %7, %6 ]
  %3 = icmp sle i32 %.0, %0
  br i1 %3, label %4, label %8

4:                                                ; preds = %2
  %5 = add nsw i32 %.01, %.0
  br label %6

6:                                                ; preds = %4
  %7 = add nsw i32 %.0, 1
  br label %2, !llvm.loop !6

8:                                                ; preds = %2
  ret i32 %.01
}
```

&emsp;&emsp;在这个 `sum_to_n` 的 IR 中，循环头 `%2` 有两个 `phi`：`%.0` 对应循环变量，语义是第一次从 `%1` 进入循环头时取 1，之后每次从回边基本块 `%6` 回到 `%2` 时取 `%7`（即上一轮 `%.0` 加 1 的结果）；`%.01` 对应累加和，语义是第一次从 `%1` 进入时取 0，之后每次从 `%6` 回边进入时取 `%5`（即上一轮 `%.01 + %.0` 的累加结果）。循环头用 `%3 = icmp sle i32 %.0, %0` 判断是否继续：为真进入 `%4` 执行累加，再到 `%6` 递增并回边；为假进入 `%8` 返回 `%.01`。这就是循环迭代变量和累加变量在 SSA 中通过 `phi` 跨迭代传递值的表达方式。

### 3.5 phi 从哪来：支配关系与支配边界

&emsp;&emsp;`phi` 的插入位置有严格的理论依据，这个依据就是**支配关系**（dominance）。先定义一下支配关系：如果从函数入口到基本块 B 的每一条路径都经过基本块 A，我们就说 A **支配** B。直观地理解，A 是 B 的"必经之路"。函数入口支配所有基本块；每个基本块都支配它自己；一个基本块可能有多个支配者，其中离它最近的那个叫"直接支配者"（immediate dominator），所有直接支配关系连起来就是一棵**支配树**（dominator tree）。

&emsp;&emsp;支配关系为什么和 `phi` 有关？考虑一个变量在多个基本块里被定义，这些定义要汇合到某个基本块里使用。如果这些定义点互相支配（比如只有一个定义点，或者所有定义点都在同一条直线上），那就不需要 `phi`——沿着唯一定义点往下走就行了。但如果多个定义点互不支配（比如分别在不同的分支里），那它们汇合的地方就需要 `phi` 来"选择"。

&emsp;&emsp;具体来说，`phi` 需要插入在**支配边界**（dominance frontier）上。支配边界的定义是：节点 X 的支配边界是这样一个集合——对于集合里的每个节点 Y，X 支配 Y 的某个前驱，但 X 不严格支配 Y 本身。换句话说，Y 是 X 的"势力范围"之外的第一个节点，控制流在 Y 这里"逃出"了 X 的支配。

&emsp;&emsp;这个定义听起来有点绕，但它的直观含义是：**`phi` 要插在"多个定义点已经无法互相支配、但值需要汇合"的地方**。经典算法是 Cytron 等人在 1991 年提出的，大致思路是：先求出每个定义点的支配边界，然后迭代地沿着支配边界插入 `phi`，直到不动点。

