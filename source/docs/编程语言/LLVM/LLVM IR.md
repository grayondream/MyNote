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

## 4. 类型系统：IR 的强类型骨架

### 4.1 类型
&emsp;&emsp;之前的IR 例子里到处都是 `i32`——`phi i32 [ ... ]`、`add nsw i32 %a, %b`、`icmp sgt i32 %a, %b`。这些是LLVM IR中值的类型。LLVM IR 是**强类型**的，每个值都有显式类型，指令的操作数类型必须匹配，没有隐式转换。


&emsp;&emsp;**LLVM IR 里的每一个值都有类型，每条指令都显式写出它操作的值的类型，类型不匹配就是错误**。

&emsp;&emsp;这一点和 C 很不一样。在 C 里，`1 + 2` 这种写法，编译器会自己推断类型、自己做整型提升、自己做隐式转换。你写 `int x = 1.5;`，编译器默默帮你截断了。而在 LLVM IR 里，这种"默默"是不存在的。如果期望 `i32` 当 `i64` 用？必须显式写一条 `zext` 或 `sext`。期望把 `double` 截断成 `float`？必须显式写一条 `fptrunc`。

&emsp;&emsp;这么做是因为 LLVM IR 是给机器看的，不是给人看的。优化器和后端需要在任何时刻都能确定一个值的宽度、符号性和语义，不能有模糊地带。如果允许隐式转换，那优化器每做一步变换都要重新推断类型，很容易出错。显式类型让优化器的工作变得确定且局部化——每条指令自己就说清楚了它在干什么。

&emsp;&emsp;LLVM IR 的类型可以分成几大类。
- **整数**：`i1`、`i8`、`i16`、`i32`、`i64`，以及任意位宽的 `iN`。注意 `i1` 是布尔类型，不是 `bool`——LLVM IR 里没有单独的布尔类型，`i1` 就是 1 位整数。
- **浮点**：`half`（16 位）、`float`（32 位）、`double`（64 位），以及 `fp128`、`x86_fp80` 这些扩展类型。平时用得最多的是 `float` 和 `double`。
- **`void`**：只出现在函数返回类型的位置，表示"不返回任何值"。`void` 不能作为变量类型，也不能作为参数类型。
- **`label`**：基本块标签的类型，只用在 `br`、`switch` 这类控制流指令里。
- **数组**：`[N x T]`，表示 N 个类型为 T 的元素。比如 `[4 x i32]` 是 4 个 `i32` 组成的数组，`[10 x i8]` 常用于字符串。数组长度是类型的一部分，`[4 x i32]` 和 `[5 x i32]` 是两种不同的类型。
- **结构体**：`{T1, T2, ...}`，表示按顺序排列的若干个字段。比如 `{i32, i64, ptr}` 是三个字段组成的结构体。结构体默认按字段的自然对齐插入填充，也可以在 `{` 前面加 `packed` 取消填充，比如 `<{i8, i32}>`。
- **指针**：现代 LLVM 里指针统一写作 `ptr`，可以带地址空间，比如 `ptr addrspace(1)`。

>&emsp;&emsp;有一个关于结构体的细节值得注意：**结构体类型是按"形状"比较的，不是按名字比较的**。`{i32, i32}` 和另一个 `{i32, i32}` 是同一个类型，即使它们在源码里来自两个不同的 `struct` 声明。这一点和 C 的类型系统很不一样——C 里两个不同名字的 struct 即使字段完全一样，也是不同的类型。LLVM IR 不做这种区分，它只关心内存布局。

**指针：opaque pointer**

&emsp;&emsp;现代 LLVM IR 使用**不透明指针**（opaque pointer）：指针统一写作 `ptr`，不再携带“指向什么类型”的信息。访问内存时，由指令本身说明类型：

```llvm
%v = load i32, ptr %p
store i64 %x, ptr %p
```

&emsp;&emsp;这里的读法是：**指针只说“在哪里”，指令说“访问或索引的是什么类型”** 。`load i32, ptr %p` 表示从 `%p` 读出 `i32`；`store i64 %x, ptr %p` 表示把 `i64` 写入 `%p`。同一个 `ptr` 可以在不同地方以不同类型访问，只要内存里确实是那个类型，就是合法的。这种设计让指针类型变得简单：`ptr` 就是 `ptr`，`ptr` 转 `ptr` 就是它自己，不需要额外的转换指令，类型比较也不需要再比较“指向什么”。对读 IR 的人来说也更直观——**看到 `ptr`，就看指令里写的类型**。

&emsp;&emsp;注意 `load` 和 `store` 的写法：访问类型只出现一次。`load i32, ptr %p` 里的 `i32` 是这次读取的类型，`ptr` 是地址；`store i64 %x, ptr %p` 里的 `i64` 是这次写入的类型。类型和地址各司其职，读起来没有歧义。

### 4.2 内存与地址：alloca、load、store、GEP

&emsp;&emsp;LLVM IR 里直接操作内存的指令有四类：`alloca`、`load`、`store`、`getelementptr`。

**`alloca`**
&emsp;&emsp;`alloca` 在当前函数的栈帧上分配对象，返回一个 `ptr`：

```llvm
%p = alloca i32, align 4
```

&emsp;&emsp;这行指令分配一个 `i32` 大小的栈空间，返回指向它的 `ptr`，对齐为 4 字节。`alloca` 也可以分配数组：`%arr = alloca [10 x i32], align 4`。

**`store/load`**

&emsp;&emsp;`load` 从指针读出值，`store` 把值写入指针指向的位置。两者都显式写出访问的类型：

```llvm
%v = load i32, ptr %p, align 4
store i32 42, ptr %p, align 4
```

&emsp;&emsp;这两条指令是配对的——`load` 读什么类型，`store` 就得写什么类型。如果 `store` 写的是 `i64`，`load` 读的是 `i32`，那是未定义行为。

**`getelementpr`**
&emsp;&emsp;`getelementptr`（通常简写成 GEP）是 IR 中唯一"只算地址、不访存"的地址计算指令。它的名字有点长，但理解起来不难：**它做的是指针算术和字段索引，返回一个新指针，但不读取任何内存**。

&emsp;&emsp;GEP 的语法是：

```llvm
%result = getelementptr <被索引的类型>, ptr %base, <索引1>, <索引2>, ...
```

&emsp;&emsp;索引的含义取决于被索引的类型。第一个索引总是"跨过指针指向的对象"，后续索引进入聚合体内部。先看第一个例子是结构体字段访问。假设有一个结构体 `%struct.S = type { i32, i64, ptr }`，我们有一个指向它的指针 `%s`：

```llvm
%field = getelementptr %struct.S, ptr %s, i32 0, i32 1
```

&emsp;&emsp;这里有两个索引。第一个 `i32 0` 表示"不移动 `%s` 本身，留在它指向的对象上"——这是 GEP 的惯例，第一个索引为 0 表示不改变基址，从对象本身的第 0 个元素开始。第二个 `i32 1` 表示"取该对象的第 1 个字段"，也就是 `i64` 那个字段。结果是得到一个指向 `%s` 所指数值体第 1 个字段的指针。

&emsp;&emsp;如果是指针算术，就只需要一个索引：

```llvm
%next = getelementptr i32, ptr %p, i64 1
```

&emsp;&emsp;这行指令的意思是：`%p` 是一个指向 `i32` 的指针，把它往后移动 1 个 `i32` 的距离（4 字节），得到 `%next`。如果 `%p` 是一个 `[10 x i32]` 数组的起始地址，`%next` 就是第 1 个元素的地址。注意 GEP 并不关心内存里到底有没有合法数据，它只是做地址运算——这就是"只算地址、不访存"的含义。

&emsp;&emsp;`inbounds` 是一个常见的 GEP 修饰符，写在 `getelementptr` 后面：

```llvm
%next = getelementptr inbounds i32, ptr %p, i64 1
```

&emsp;&emsp;它的语义是"这个地址计算结果不会溢出分配对象的边界"。这个承诺给优化器提供了额外的推理依据——比如它可以假设 `%next` 一定在某个合法对象内部，可以据此做别名分析。但承诺是有代价的：如果实际计算结果确实溢出了边界，行为是未定义的。手写 IR 时用 `inbounds` 要小心，确保语义成立；编译器生成的 IR 里 `inbounds` 用得很多，因为前端知道它生成的地址访问是合法的。

&emsp;&emsp;GEP 是 LLVM IR 里最容易出错的指令之一，初学者经常把第一个索引和第二个索引搞混，或者以为 GEP 会读取内存。记住一句话就不会错：**GEP 只做地址运算，不读取任何内存；第一个索引总是跨过指针本身，第二个及之后的索引才进入聚合体内部**。

### 4.3 datalayout 

&emsp;&emsp;第 2 章讲 `Module` 头部时提到过 `target datalayout`，它"描述内存布局"。他描述了内存中数据的布局。同样的 IR，在不同的目标平台上，类型的大小和对齐可能不同。比如 `i64` 在 32 位平台上的对齐可能是 4 字节，在 64 位平台上是 8 字节。指针的大小更不用说，32 位平台上 `ptr` 是 4 字节，64 位平台上是 8 字节。`datalayout` 就是用来把这些信息告诉优化器和后端的。

&emsp;&emsp;`datalayout` 是一串用 `-` 分隔的字段，每个字段描述一个方面的布局。我们看一个典型的：

```text
e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128
```

&emsp;&emsp;这一串看起来像天书，但拆开看并不复杂：

- `e`：端序，小端（little-endian）。大端是 `E`。
- `m:e`：函数指针的 mangling 方式。
- `p270:32:32` 等：地址空间 270、271、272 的指针大小和对齐。这几个是特殊的地址空间，用于某些特定用途。
- `i64:64`：`i64` 的对齐是 64 位。
- `f80:128`：`x86_fp80` 的对齐是 128 位。
- `n8:16:32:64`：原生整数宽度是 8、16、32、64。
- `S128`：栈对齐是 128 位。

&emsp;&emsp;这些信息看起来琐碎，但它们直接影响了结构体的内存布局。举个例子，假设有这样一个结构体：

```llvm
%struct.S = type { i8, i32 }
```

&emsp;&emsp;在 `datalayout` 声明 `i32` 对齐为 4 字节的平台上，这个结构体的大小是 8 字节——`i8` 占 1 字节，后面填充 3 字节，再放 `i32` 占 4 字节。但如果用了 `packed` 结构体 `<{i8, i32}>`，取消填充，大小就变成 5 字节。

&emsp;&emsp;这个差异直接影响了 GEP 计算出的字节偏移。`getelementptr %struct.S, ptr %s, i32 0, i32 1` 在非 packed 结构体上算出来的是偏移 4，在 packed 结构体上算出来的是偏移 1。如果省略了 `datalayout`，优化器就不知道该怎么算，某些 Pass（比如 SROA、别名分析）会直接报警告甚至拒绝运行。

&emsp;&emsp;所以记住：**`datalayout` 不是可选的装饰，它是类型系统的一部分**。读别人的 IR 时，先看 `datalayout`，才知道那些结构体布局和 GEP 偏移到底是什么意思。

## 5 C++的LLVM IR转换

&emsp;&emsp;前面四章讲的规则对 C 和 C++ 都成立，但 C++ 在 IR 里还留下了几样 C 里没有的东西：名字修饰（name mangling）、`this` 指针、虚函数表、异常处理表、模板实例化。这些东西有一个共同点——**它们大多是前端的约定，IR 本身并不知道"这是 C++"**。IR 里没有"类"这种类型，没有"虚调用"这种指令，也没有"模板"这个概念，所有 C++ 语义都要被拆解成 IR 已有的原语再拼回去。

&emsp;&emsp;这一章用一组可以实际编译的小例子逐项验证这件事。所有 IR 都由本机的 clang 22.1.2 生成，命令统一为：

```bash
clang++-22 -std=c++20 -O0 -Xclang -disable-O0-optnone -S -emit-llvm ch5_1_types.cpp -o ch5_1_types.ll
```

&emsp;&emsp;沿用第 2 章的选择，`-O0` 保留了最朴素的栈式 IR，方便把 C++ 语句和指令一一对应。个别地方再补一份 `-O2` 的产物做对照，命令只是把 `-O0 -Xclang -disable-O0-optnone` 换成 `-O2`。

&emsp;&emsp;源文件和生成文件的对应关系（都在本章配套目录下）：

- 5.1 基础类型和算术操作：`ch5_1_types.cpp` → `ch5_1_types.ll`、`ch5_1_types.O2.ll`
- 5.2 函数：`ch5_2_functions.cpp` → `ch5_2_functions.ll`
- 5.3 控制流：`ch5_3_control.cpp` → `ch5_3_control.ll`、`ch5_3_control.O2.ll`
- 5.4 类和结构体内存布局：`ch5_4_class_layout.cpp` → `ch5_4_class_layout.ll`
- 5.5 继承、多态和虚函数：`ch5_5_inheritance.cpp` → `ch5_5_inheritance.ll`、`ch5_5_inheritance.O2.ll`
- 5.6 C++ 异常和 RAII：`ch5_6_exceptions.cpp` → `ch5_6_exceptions.ll`
- 5.7 模板：`ch5_7_templates.cpp` → `ch5_7_templates.ll`、`ch5_7_templates.O2.ll`

&emsp;&emsp;文中引用的指令和类型都能在上面这些 `.ll` 文件里原样找到（行尾的中文注释是为讲解另加的），建议对照阅读。

### 5.1 基础类型和算术操作

&emsp;&emsp;第 4 章说过 IR 是强类型的，每种 C++ 类型都对应一个确定的 IR 类型。这张表把常见类型列出来：

| C++ 类型 | IR 类型 | 说明 |
| --- | --- | --- |
| `bool` | `i1`（寄存器）/ `i8`（内存） | 内存里占 1 字节，读出来要 `trunc` 成 `i1` 才能参与运算 |
| `char`、`signed char`、`unsigned char` | `i8` | 只有 1 字节，运算时先提升到 `i32` |
| `short`、`unsigned short` | `i16` | 同样要先提升 |
| `int`、`unsigned int` | `i32` | 最常用的整数 |
| `long`、`long long` | `i64` | LP64 平台上都是 8 字节 |
| `float` / `double` | `float` / `double` | IR 里直接叫 `float`/`double`，不叫 `f32`/`f64` |
| 指针、引用 | `ptr` | 引用在 IR 里和指针同型，只是多了 `nonnull` 等属性 |
| `size_t`、`ptrdiff_t` | `i64` | 64 位平台上就是 64 位整数 |

&emsp;&emsp;先从最简单的比较函数开始：

```cpp
bool less_than(int a, int b) { return a < b; }
```

```llvm
define dso_local noundef zeroext i1 @_Z9less_thanii(i32 noundef %0, i32 noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %5 = load i32, ptr %3, align 4
  %6 = load i32, ptr %4, align 4
  %7 = icmp slt i32 %5, %6
  ret i1 %7
}
```

&emsp;&emsp;这里出现了三个第 4 章没细讲的细节。

&emsp;&emsp;第一，返回类型是 `zeroext i1` 而不是 `i1`。`i1` 在寄存器里只占一位，但 C++ 的 `bool` 在 ABI 里占一个字节，`zeroext` 的意思是"返回值的高位会被清零"，保证调用方把 `i1` 当 `i8` 用时看到的是 0 或 1，而不是一个带垃圾位的字节。对应的还有 `signext`，用在 `char`、`short` 这种有符号小整数上——5.1 后面的 `byte_inc` 就会看到。

&emsp;&emsp;第二，`icmp slt` 里的 `slt` 是 "signed less than"。整数比较一律用 `icmp`，有符号和无符号的差别体现在条件码上：`slt`/`sgt`/`sle`/`sge` 是有符号，`ult`/`ugt`/`ule`/`uge` 是无符号，`eq`/`ne` 不分符号。

&emsp;&emsp;第三，函数入口的两条 `alloca` 加两次 `store` 是 `-O0` 的标志性形状——形参也被当作局部变量放在栈上，每次使用都 `load` 一遍。这是第 3 章讲过的"伪 SSA"，`mem2reg` 之后这两个栈槽会消失。

&emsp;&emsp;再看有符号和无符号在算术上的差别：

```cpp
int add_signed(int a, int b) { return a + b; }
unsigned add_unsigned(unsigned a, unsigned b) { return a + b; }
```

```llvm
; add_signed 的核心指令
%7 = add nsw i32 %5, %6

; add_unsigned 的核心指令
%7 = add i32 %5, %6
```

&emsp;&emsp;两条指令的操作数类型都是 `i32`，区别只在 `nsw`。`nsw` 是 "no signed wrap"，它把"这个加法不会溢出"这个承诺写进 IR。反过来读：如果实际溢出了，程序行为未定义。这正是 C++ 标准对**有符号**整数溢出的规定，所以前端敢加 `nsw`；而**无符号**溢出在 C++ 里是良定义的回绕，前端就不能加，只能写一条朴素的 `add`。

&emsp;&emsp;除法、取余同理，符号性直接体现在操作码里：

```cpp
int sdiv(int a, int b) { return a / b; }
unsigned udiv(unsigned a, unsigned b) { return a / b; }
int smod(int a, int b) { return a % b; }
unsigned umod(unsigned a, unsigned b) { return a % b; }
```

```llvm
%7 = sdiv i32 %5, %6
%7 = udiv i32 %5, %6
%7 = srem i32 %5, %6
%7 = urem i32 %5, %6
```

&emsp;&emsp;注意取余是 `srem`/`urem`（remainder），没有 `smod` 这个名字。LLVM 里 `rem` 的语义是"结果符号跟随被除数"，和 C++ 的 `%` 一致。

&emsp;&emsp;接下来是最能体现"没有隐式转换"的部分——类型宽度转换。在 C++ 里写 `(long long)x`、`(int)y`、`(float)z`，IR 里都必须有一条显式指令：

```cpp
long long widen(int x) { return (long long)x; }
int narrow(long long x) { return (int)x; }
unsigned reinterpret_width(int x) { return (unsigned)x; }

double int_to_double(int x) { return (double)x; }
int double_to_int(double d) { return (int)d; }
unsigned double_to_unsigned(double d) { return (unsigned)d; }
float double_to_float(double d) { return (float)d; }
```

```llvm
%4 = sext i32 %3 to i64          ; int -> long long，符号扩展
%4 = trunc i64 %3 to i32         ; long long -> int，截断
%4 = sitofp i32 %3 to double     ; 有符号整数 -> 浮点
%4 = fptosi double %3 to i32     ; 浮点 -> 有符号整数
%4 = fptoui double %3 to i32     ; 浮点 -> 无符号整数
%4 = fptrunc double %3 to float  ; 双精度 -> 单精度
```

&emsp;&emsp;把这几条凑成一张转换表就很好记了：

- 整数之间变宽：有符号用 `sext`，无符号用 `zext`；变窄统一用 `trunc`。
- 整数与浮点之间：整数→浮点用 `sitofp`/`uitofp`，浮点→整数用 `fptosi`/`fptoui`，同样按符号性二分。
- 浮点之间：变宽用 `fpext`，变窄用 `fptrunc`。

&emsp;&emsp;有一个有意思的地方：`unsigned reinterpret_width(int x)` 返回 `unsigned`，但 IR 里**什么都不用做**——`int` 和 `unsigned` 在 IR 里是同一种类型 `i32`，符号性只存在于操作码和条件码里，不存在于类型里。所以这个函数在 `-O0` 下直接 `ret i32 %3`，在 `-O2` 下连参数都直接返回了（同时多了个 `returned` 属性）：

```llvm
define dso_local noundef i32 @_Z17reinterpret_widthi(i32 noundef returned %0) local_unnamed_addr #0 {
  ret i32 %0
}
```

&emsp;&emsp;再看一个混合宽度的例子，这是 C++ 隐式整型提升在 IR 里的样子：

```cpp
int mixed_width(int a, long long b) { return a + b; }
```

```llvm
%5 = load i32, ptr %3, align 4
%6 = sext i32 %5 to i64
%7 = load i64, ptr %4, align 8
%8 = add nsw i64 %6, %7
%9 = trunc i64 %8 to i32
ret i32 %9
```

&emsp;&emsp;源码里只有一行 `a + b`，前端替我们做了三件事：把 `a` 从 `i32` 扩展成 `i64`、在 `i64` 上做加法、把结果截断回 `i32` 作为返回值。**C++ 的隐式转换规则是前端负责的，IR 只呈现结果**——第 4 章说"IR 没有隐式转换"，指的就是这个：转换本身必须变成显式指令，一条都不能省。

&emsp;&emsp;小整数的处理更明显。`signed char` 和 `short` 参与运算时，C++ 会先把它们提升到 `int`，于是 IR 里出现"扩展—在 `i32` 上算—截断回来"的三步：

```cpp
signed char byte_inc(signed char v) { return v + 1; }
short short_add(short a, short b) { return a + b; }
```

```llvm
define dso_local noundef signext i8 @_Z8byte_inca(i8 noundef signext %0) #0 {
  %2 = alloca i8, align 1
  store i8 %0, ptr %2, align 1
  %3 = load i8, ptr %2, align 1
  %4 = sext i8 %3 to i32
  %5 = add nsw i32 %4, 1
  %6 = trunc i32 %5 to i8
  ret i8 %6
}

define dso_local noundef signext i16 @_Z9short_addss(i16 noundef signext %0, i16 noundef signext %1) #0 {
  %3 = alloca i16, align 2
  %4 = alloca i16, align 2
  store i16 %0, ptr %3, align 2
  store i16 %1, ptr %4, align 2
  %5 = load i16, ptr %3, align 2
  %6 = sext i16 %5 to i32
  %7 = load i16, ptr %4, align 2
  %8 = sext i16 %7 to i32
  %9 = add nsw i32 %6, %8
  %10 = trunc i32 %9 to i16
  ret i16 %10
}
```

&emsp;&emsp;注意 `short_add` 里**没有** `add i16`。IR 里依然可以写 `i16` 的加法，but 前端不会生成它，因为 C++ 的语义要求先在 `int` 里算完再截断。如果谁手写 IR 直接做 `add i16`，那在做优化分析和常量传播时就会和 C++ 的真实语义对不上。这类"源码语义决定 IR 形状"的细节，是读前端产物时最容易踩的坑。

&emsp;&emsp;浮点运算有三种指令族，`fadd`、`fsub`、`fmul`、`fdiv`、`frem`，以及比较用的 `fcmp`。看一个除法：

```cpp
float float_math(float a, float b) { return a / b; }
```

```llvm
%7 = fdiv float %5, %6
```

&emsp;&emsp;`float` 和 `double` 在 IR 里就是独立的类型，`fdiv float` 和 `fdiv double` 是两条不同的指令。浮点比较用 `fcmp`，条件码比整数多得多，因为要考虑 NaN：

```llvm
%7 = fcmp ogt double %5, %6
```

&emsp;&emsp;`ogt` 里开头的 `o` 表示 "ordered"，即"两个操作数都不是 NaN 且 `%5 > %6`"。与之相对的是 `u` 系（unordered），NaN 参与时结果为真。C++ 的 `>` 默认生成 ordered 版本，所以 `maxv<double>` 里是 `fcmp ogt`，而 `maxv<int>` 里是 `icmp sgt`——同一份模板代码，实例化到不同类型上就变成了两套完全不同的指令。

&emsp;&emsp;还有一个值得注意的指令：`a * b + 1.5` 没有生成 `fmul` 加 `fadd`，而是生成了一条乘加：

```cpp
double double_math(double a, double b) { return a * b + 1.5; }
```

```llvm
%7 = call double @llvm.fmuladd.f64(double %5, double %6, double 1.500000e+00)
```

&emsp;&emsp;`llvm.fmuladd` 的语义是"允许融合"，而不是"必须融合"：后端如果支持 FMA 指令就融合成一条，不支持就拆回乘加两条。这样前端不用关心目标平台有没有 FMA，把决定权交给后端。注意这个优化在 `-O0` 就发生了，因为它属于**前端的表达式识别**，不需要跑任何 Pass。

&emsp;&emsp;最后对比一下 `-O2` 的产物，感受一下优化器做了哪些事：

```llvm
define dso_local noundef zeroext i1 @_Z9less_thanii(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp slt i32 %0, %1
  ret i1 %3
}

define dso_local noundef i32 @_Z11mixed_widthix(i32 noundef %0, i64 noundef %1) local_unnamed_addr #0 {
  %3 = trunc i64 %1 to i32
  %4 = add i32 %0, %3
  ret i32 %4
}
```

&emsp;&emsp;`alloca`、`load`、`store` 全没了，参数在虚拟寄存器里直接用。更有意思的是 `mixed_width`：优化器发现"`sext` 到 `i64`、加法、`trunc` 回 `i32`"这套操作等价于"直接把 `b` 截断成 `i32` 再相加"，于是把三条指令压成了两条，而且去掉了 `nsw`——因为 `trunc` 之后再相加，即使溢出也不会触发未定义行为，`nsw` 反而是个多余的约束。**优化器在做的，就是在保证语义等价的前提下，用尽可能少的指令表达同一件事。**

### 5.2 函数

&emsp;&emsp;函数是 C++ 落到 IR 上最容易看懂、也最容易踩坑的部分。先看名字。C++ 支持重载，也就是说 `add(int, int)` 和 `add(double, double)` 可以同时存在。IR 里的符号名是扁平的，不能有两个 `@add`，所以前端必须在编译期把参数类型编进名字里，这就是**名字修饰**（name mangling）：

```cpp
int add(int a, int b) { return a + b; }
double add(double a, double b) { return a + b; }
```

```llvm
define dso_local noundef i32 @_Z3addii(i32 noundef %0, i32 noundef %1) #0 {
define dso_local noundef double @_Z3adddd(double noundef %0, double noundef %1) #0 {
```

&emsp;&emsp;Linux 上用的是 Itanium ABI 的修饰规则，`_Z3addii` 可以这样拆：`_Z` 是前缀，`3` 是函数名的长度，`add` 是函数名，后面每个字符对应一个参数类型——`i` 是 `int`，`d` 是 `double`。常用的编码有这么几个：

| 编码 | 类型 | 编码 | 类型 |
| --- | --- | --- | --- |
| `i` | `int` | `j` | `unsigned int` |
| `l` | `long` | `m` | `unsigned long` |
| `x` | `long long` | `y` | `unsigned long long` |
| `f` | `float` | `d` | `double` |
| `b` | `bool` | `c` | `char` |
| `s` | `short` | `t` | `unsigned short` |
| `P` | 指针 | `R` | 引用 |
| `K` | `const` | `v` | `void`（参数表为空） |

&emsp;&emsp;所以 `_Z3addii` 就是 `add(int, int)`，`_Z3adddd` 就是 `add(double, double)`。这也是为什么改了函数签名之后，即使函数体没动，符号名也会变——它本来就该变，因为那是两个不同的函数。

&emsp;&emsp;如果不想要修饰（比如要给 C 或者别的语言调用），就写 `extern "C"`：

```cpp
extern "C" int c_add(int a, int b) { return a + b; }
```

```llvm
define dso_local i32 @c_add(i32 noundef %0, i32 noundef %1) #0 {
```

&emsp;&emsp;符号名就是 `c_add` 本身，没有 `_Z` 前缀。这也解释了为什么 `extern "C"` 的函数不能重载——名字里没有类型信息，两个 `c_add` 会直接冲突。

&emsp;&emsp;默认参数是**调用方**的事，被调函数根本不知道有默认值：

```cpp
int with_default(int a, int b = 42) { return a + b; }
int caller() { return with_default(1); }
```

```llvm
define dso_local noundef i32 @_Z12with_defaultii(i32 noundef %0, i32 noundef %1) #0 {
  ; ... a + b ...
}

define dso_local noundef i32 @_Z6callerv() #0 {
  %1 = call noundef i32 @_Z12with_defaultii(i32 noundef 1, i32 noundef 42)
  ret i32 %1
}
```

&emsp;&emsp;被调函数有两个参数，没有"可选参数"这种概念；`caller` 里直接把 42 填了进去。默认参数是纯粹的前端语法糖，到了 IR 这一层已经不存在了。

&emsp;&emsp;引用和指针在 IR 里是同一个类型 `ptr`，区别体现在**附加属性**上：

```cpp
void take_ref(int &x) { x += 1; }
void take_ptr(int *x) { *x += 1; }
```

```llvm
define dso_local void @_Z8take_refRi(ptr noundef nonnull align 4 dereferenceable(4) %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8, !nonnull !6, !align !7
  %4 = load i32, ptr %3, align 4
  %5 = add nsw i32 %4, 1
  store i32 %5, ptr %3, align 4
  ret void
}

define dso_local void @_Z8take_ptrPi(ptr noundef %0) #0 {
  ; ... 同样的四条指令 ...
}
```

&emsp;&emsp;两个函数的指令序列一模一样，差别全在参数上的属性：

- `nonnull`：这个指针不会是空指针。C++ 的引用必须绑定到一个对象上，所以前端敢这么标。
- `dereferenceable(4)`：从指针开始至少 4 个字节是可读写的。这里 4 就是 `int` 的大小。
- `align 4`：指针指向的地址至少 4 字节对齐。

&emsp;&emsp;这些属性在 `-O0` 下还会以元数据的形式附加在 `load` 上（`!nonnull`、`!align`）。它们不是"注释"，而是**给优化器的承诺**：有了 `nonnull`，优化器可以删掉空指针检查；有了 `dereferenceable`，可以把访问提升到更早的位置。承诺如果被违反，行为是未定义的。

&emsp;&emsp;函数返回大对象时，会出现一个叫 `sret` 的参数：

```cpp
struct Big {
    long long a, b, c, d;
};

Big make_big(long long seed) { return Big{seed, seed, seed, seed}; }
```

```llvm
%struct.Big = type { i64, i64, i64, i64 }

define dso_local void @_Z8make_bigx(ptr dead_on_unwind noalias writable sret(%struct.Big) align 8 %0, i64 noundef %1) #0 {
  %3 = alloca i64, align 8
  store i64 %1, ptr %3, align 8
  %4 = getelementptr inbounds nuw %struct.Big, ptr %0, i32 0, i32 0
  %5 = load i64, ptr %3, align 8
  store i64 %5, ptr %4, align 8
  %6 = getelementptr inbounds nuw %struct.Big, ptr %0, i32 0, i32 1
  %7 = load i64, ptr %3, align 8
  store i64 %7, ptr %6, align 8
  %8 = getelementptr inbounds nuw %struct.Big, ptr %0, i32 0, i32 2
  %9 = load i64, ptr %3, align 8
  store i64 %9, ptr %8, align 8
  %10 = getelementptr inbounds nuw %struct.Big, ptr %0, i32 0, i32 3
  %11 = load i64, ptr %3, align 8
  store i64 %11, ptr %10, align 8
  ret void
}
```

&emsp;&emsp;函数签名变了：返回类型成了 `void`，多了一个 `ptr` 类型的第一个参数 `%0`。这就是 **sret**（structure return）约定——大对象由**调用方**在栈上分配空间，把地址作为隐藏参数传进来，被调函数往这个地址里填数据，最后 `ret void`。之所以要这样，是因为 x86-64 的 SysV ABI 里，只有不超过 16 字节的结构体能塞进 `rax`/`rdx` 返回；`Big` 有 32 字节，只能走内存。

&emsp;&emsp;`sret` 后面还跟了一串修饰：`noalias` 表示这块内存不会和别的参数别名；`writable` 表示被调函数会写它；`dead_on_unwind` 表示如果发生了异常展开，调用方不需要销毁这个返回值。这三个词都是给优化器和异常处理机制看的。

&emsp;&emsp;参数按值传递大对象时也有对应的机制——`byval`：

```cpp
Big pass_and_return(Big v) {
    v.a += 1;
    return v;
}
```

```llvm
define dso_local void @_Z15pass_and_return3Big(ptr dead_on_unwind noalias writable sret(%struct.Big) align 8 %0, ptr noundef byval(%struct.Big) align 8 %1) #0 {
  %3 = getelementptr inbounds nuw %struct.Big, ptr %1, i32 0, i32 0
  %4 = load i64, ptr %3, align 8
  %5 = add nsw i64 %4, 1
  store i64 %5, ptr %3, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 %1, i64 32, i1 false)
  ret void
}
```

&emsp;&emsp;`byval(%struct.Big)` 的意思是：调用方传进来的是一个指针，但被调函数看到的应该是**一份副本**——ABI 规定这份副本由被调函数自己按 `byval` 的语义处理（可能是调用方在栈上传，也可能是被调函数拷一份）。函数体里直接通过 `%1` 读写这块副本，最后用 `memcpy` 32 字节复制到 `sret` 指向的返回空间里。C++ 的"按值传参、修改不影响调用方"这条语义，在 IR 层就是 `byval` 加上一次拷贝。

&emsp;&emsp;`static` 修饰的全局变量和函数内静态变量，在 IR 里是**模块级全局变量**，但链接属性是 `internal`：

```cpp
static int counter = 0;
int next_counter() { return ++counter; }
```

```llvm
@_ZL7counter = internal global i32 0, align 4

define dso_local noundef i32 @_Z12next_counterv() #0 {
  %1 = load i32, ptr @_ZL7counter, align 4
  %2 = add nsw i32 %1, 1
  store i32 %2, ptr @_ZL7counter, align 4
  ret i32 %2
}
```

&emsp;&emsp;名字里的 `L` 表示内部链接（`internal`），别的编译单元看不到这个符号，链接器也不会把它导出。函数体里没有 `alloca`，直接对全局变量 `load`/`store`——因为它本来就不在栈上。

&emsp;&emsp;最后补一句属性组。每个函数定义后面都有个 `#0`，文件末尾用 `attributes #0 = { ... }` 定义了一堆东西：`mustprogress`、`noinline`、`nounwind`、`uwtable`，以及 `"frame-pointer"="all"`、`"target-cpu"="x86-64"` 这些字符串属性。`noinline` 是 `-O0` 加上去的（阻止优化器内联），`nounwind` 表示这个函数不会抛异常，`uwtable` 表示要生成 `.eh_frame` 里的展开信息——即使函数自己不处理异常，栈展开时也需要知道怎么恢复它的栈帧。这些属性在优化阶段会被大量使用，比如 `nounwind` 能让优化器放心地把调用移到别的位置。

### 5.3 控制流

&emsp;&emsp;第 2 章用 `compute` 看过 `if` 对应的 `icmp` + `br`。这一节把常见的控制流结构都过一遍，重点看它们如何被拆成基本块，以及 `-O0` 和 `-O2` 的差别。

&emsp;&emsp;先看带返回值的 `if`：

```cpp
int abs_val(int x) {
    if (x < 0) {
        return -x;
    }
    return x;
}
```

```llvm
define dso_local noundef i32 @_Z7abs_vali(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  %4 = load i32, ptr %3, align 4
  %5 = icmp slt i32 %4, 0
  br i1 %5, label %6, label %9

6:                                                ; preds = %1
  %7 = load i32, ptr %3, align 4
  %8 = sub nsw i32 0, %7
  store i32 %8, ptr %2, align 4
  br label %11

9:                                                ; preds = %1
  %10 = load i32, ptr %3, align 4
  store i32 %10, ptr %2, align 4
  br label %11

11:                                               ; preds = %9, %6
  %12 = load i32, ptr %2, align 4
  ret i32 %12
}
```

&emsp;&emsp;注意 `%2` 这个 `alloca`：它不是在源码里声明的变量，而是前端为"返回值"准备的槽位。两个分支都把结果 `store` 进 `%2`，然后在汇合块 `%11` 里统一 `load` 出来 `ret`。这是 `-O0` 的一种典型形状——**把控制流汇合变成对同一个栈槽的多次写**，从而避免在 `-O0` 阶段就引入 `phi`。`mem2reg` 跑完之后，这个槽位才会变成 `phi`。

&emsp;&emsp;`switch` 有专门的指令：

```cpp
int classify(int x) {
    switch (x) {
    case 0: return 10;
    case 1: return 20;
    case 2: return 30;
    case 3: return 40;
    default: return -1;
    }
}
```

```llvm
%4 = load i32, ptr %3, align 4
switch i32 %4, label %9 [
  i32 0, label %5
  i32 1, label %6
  i32 2, label %7
  i32 3, label %8
]
```

&emsp;&emsp;`switch` 的语法是 `switch <值>, label <默认目标> [ <case值>, label <目标> ... ]`。`default` 分支就是紧跟 `switch` 的那个默认标签。和 `br` 一样，`switch` 也是终结指令，它后面不能再有别的指令。

&emsp;&emsp;`switch` 是 IR 层的一条"多路分支"原语，具体怎么降级成机器码由后端决定——目标平台有跳转表就用跳转表，没有就退化成比较链。而在优化阶段，如果 case 值很密集，优化器甚至可能把它改成别的形状。看 `-O2` 的版本：

```llvm
define dso_local noundef range(i32 -1, 41) i32 @_Z8classifyi(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i32 %0, 4
  %3 = mul nsw i32 %0, 10
  %4 = add nsw i32 %3, 10
  %5 = select i1 %2, i32 %4, i32 -1
  ret i32 %5
}
```

&emsp;&emsp;四个 case 加一个默认值，被压成了四条指令：`icmp ult` 判断是否落在 `[0,4)` 区间里（无符号比较顺便把负数也排除了），`mul` 加 `add` 直接算出 `10 * x + 10`，最后 `select` 在"算出来的值"和 `-1` 之间二选一。优化器发现这几个 case 的返回值恰好是等差数列，于是把整个 `switch` 换成了一个乘法和一次条件选择。这就是"IR 是给优化器看的"的最好例子——**源码里写的是分支，优化后可能一条分支都不剩**。

&emsp;&emsp;循环是基本块组织最复杂的部分。先看 `for`：

```cpp
int sum_to_n(int n) {
    int sum = 0;
    for (int i = 1; i <= n; i++) {
        sum += i;
    }
    return sum;
}
```

```llvm
define dso_local noundef i32 @_Z8sum_to_ni(i32 noundef %0) #0 {
  %2 = alloca i32, align 4        ; n
  %3 = alloca i32, align 4        ; sum
  %4 = alloca i32, align 4        ; i
  store i32 %0, ptr %2, align 4
  store i32 0, ptr %3, align 4
  store i32 1, ptr %4, align 4
  br label %5

5:                                                ; preds = %13, %1
  %6 = load i32, ptr %4, align 4
  %7 = load i32, ptr %2, align 4
  %8 = icmp sle i32 %6, %7
  br i1 %8, label %9, label %16

9:                                                ; preds = %5
  %10 = load i32, ptr %4, align 4
  %11 = load i32, ptr %3, align 4
  %12 = add nsw i32 %11, %10
  store i32 %12, ptr %3, align 4
  br label %13

13:                                               ; preds = %9
  %14 = load i32, ptr %4, align 4
  %15 = add nsw i32 %14, 1
  store i32 %15, ptr %4, align 4
  br label %5, !llvm.loop !6

16:                                               ; preds = %5
  %17 = load i32, ptr %3, align 4
  ret i32 %17
}
```

&emsp;&emsp;四类基本块各自有了分工：

- **入口块**（无标签，`%1`）：分配三个栈槽，初始化，然后无条件 `br` 到循环头。
- **循环头** `%5`：判断条件 `i <= n`，真则进循环体，假则退出。它有两个前驱——入口块和回边块 `%13`。
- **循环体** `%9`：`sum += i`，然后跳转到"步进块"`%13`。
- **步进块** `%13`：`i++`，然后 `br` 回循环头。这条是**回边**（back edge），所以附带了 `!llvm.loop !6` 元数据。

&emsp;&emsp;`!llvm.loop` 是循环的元数据，`!6 = distinct !{!6, !7}`、`!7 = !{!"llvm.loop.mustprogress"}` 的意思是"这个循环保证会推进"。这个标记来自函数的 `mustprogress` 属性（C++ 的循环如果条件永远为真而又没有副作用，标准允许它不终结）。`distinct` 表示每个循环的元数据互相独立。后续的循环优化（向量化、`LICM`、`unroll`）都要读这些元数据。

&emsp;&emsp;为什么要有单独的步进块？因为 `continue` 需要跳到这里，而不是跳到循环头。看一个同时有 `continue` 和 `break` 的循环：

```cpp
int skip_odd(int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        if (i % 2 == 0) {
            continue;
        }
        if (i > 100) {
            break;
        }
        sum += i;
    }
    return sum;
}
```

```llvm
5:                                                ; 循环头
  %6 = load i32, ptr %4, align 4
  %7 = load i32, ptr %2, align 4
  %8 = icmp slt i32 %6, %7
  br i1 %8, label %9, label %25

9:                                                ; 循环体
  %10 = load i32, ptr %4, align 4
  %11 = srem i32 %10, 2
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %13, label %14

13:                                               ; continue 分支
  br label %22

14:                                               ; 内层 if
  %15 = load i32, ptr %4, align 4
  %16 = icmp sgt i32 %15, 100
  br i1 %16, label %17, label %18

17:                                               ; break 分支
  br label %25

18:                                               ; sum += i
  ...
  br label %22

22:                                               ; 步进块：i++
  %23 = load i32, ptr %4, align 4
  %24 = add nsw i32 %23, 1
  store i32 %24, ptr %4, align 4
  br label %5, !llvm.loop !10

25:                                               ; 循环出口
  ...
```

&emsp;&emsp;`continue` 是一条约 `br` 到 `%22` 的跳转，`break` 是一条约 `br` 到 `%25` 的跳转。C++ 里 `for`、`while`、`break`、`continue`、`goto` 这些结构，到 IR 层全部统一成了基本块之间的边——**IR 只认识 `br` 和 `switch` 这两种控制流**，剩下的都是前端的组织方式。

&emsp;&emsp;`while` 和 `do-while` 的差别只在条件判断的位置：

```cpp
int count_down(int n) {          // while：先判断后执行
    int acc = 0;
    while (n > 0) {
        acc += n;
        n--;
    }
    return acc;
}

int do_while(int n) {            // do-while：先执行后判断
    int acc = 0;
    do {
        acc += 2;
        n--;
    } while (n > 0);
    return acc;
}
```

```llvm
; while 的循环头 %4 既判断条件，也是回边目标
4:                                                ; preds = %7, %1
  %5 = load i32, ptr %2, align 4
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %13

7:                                                ; 循环体，末尾 br 回 %4
  ...
  br label %4, !llvm.loop !8

; do-while 的循环体 %4 先执行，条件在 %9 里判断
4:                                                ; preds = %9, %1
  ; ... acc += 2; n--; ...
  br label %9

9:                                                ; 条件块
  %10 = load i32, ptr %2, align 4
  %11 = icmp sgt i32 %10, 0
  br i1 %11, label %4, label %12, !llvm.loop !9
```

&emsp;&emsp;`do-while` 的结构有个特点：条件块 `%9` 的 `br` 是"条件回边"——真则回到 `%4` 继续循环，假则退出。`!llvm.loop` 附着在这条条件 `br` 上而不是无条件 `br` 上。另外，`do-while` 的循环体至少有**一条**从入口进来的路径，所以它是唯一不需要"首次条件判断"就能保证执行的循环形式。

&emsp;&emsp;接下来是短路求值，这是 C++ 里唯一一处"表达式求值顺序影响控制流"的地方：

```cpp
bool short_circuit(int a, int b) {
    return side_effect(a) && side_effect(b);
}
```

```llvm
define dso_local noundef zeroext i1 @_Z13short_circuitii(i32 noundef %0, i32 noundef %1) #1 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %5 = load i32, ptr %3, align 4
  %6 = call noundef i32 @_Z11side_effecti(i32 noundef %5)
  %7 = icmp ne i32 %6, 0
  br i1 %7, label %8, label %12

8:                                                ; preds = %2
  %9 = load i32, ptr %4, align 4
  %10 = call noundef i32 @_Z11side_effecti(i32 noundef %9)
  %11 = icmp ne i32 %10, 0
  br label %12

12:                                               ; preds = %8, %2
  %13 = phi i1 [ false, %2 ], [ %11, %8 ]
  ret i1 %13
}
```

&emsp;&emsp;`a && b` 被拆成了两个块：如果第一次调用返回 0，直接跳到 `%12`，第二段根本不会执行；否则在 `%8` 里执行第二次调用。汇合点用 `phi` 选择结果：从 `%2` 来就是 `false`，从 `%8` 来就是第二次比较的结果。

&emsp;&emsp;这里值得停下来看一眼：**`-O0` 下居然出现了 `phi`**。第 3 章说过 `-O0` 通常用栈槽来避开 `phi`，但 `&&`、`||`、`?:` 这类"表达式级"的汇合是个例外。原因是这些表达式的结果没有对应的源码变量可存，前端干脆用 `phi` 表达"两条路径各给一个值"。也就是说，`phi` 并不只出现在 `mem2reg` 之后，前端在需要的时候也会直接生成。

&emsp;&emsp;三目运算符同理：

```cpp
int choose(int a, int b, bool flag) { return flag ? a : b; }
```

```llvm
%9 = trunc i8 %8 to i1
br i1 %9, label %10, label %12

10:                                               ; preds = %3
  %11 = load i32, ptr %4, align 4
  br label %14

12:                                               ; preds = %3
  %13 = load i32, ptr %5, align 4
  br label %14

14:                                               ; preds = %12, %10
  %15 = phi i32 [ %11, %10 ], [ %13, %12 ]
  ret i32 %15
```

&emsp;&emsp;注意这里 `bool` 参数的处理：函数签名里是 `i1 noundef zeroext %2`，进函数后先 `zext i1 %2 to i8` 存进 `alloca i8`，用的时候再 `trunc i8 %8 to i1`。这就是 5.1 说的"寄存器里是 `i1`，内存里是 `i8`"的完整往返。

&emsp;&emsp;到了 `-O2`，三目运算符会被压成一条 `select`：

```llvm
define dso_local noundef i32 @_Z6chooseiib(i32 noundef %0, i32 noundef %1, i1 noundef zeroext %2) local_unnamed_addr #0 {
  %4 = select i1 %2, i32 %0, i32 %1
  ret i32 %4
}
```

&emsp;&emsp;`select` 和 `phi` 的区别值得记一下：**`phi` 依赖控制流，`select` 不依赖**。`phi` 只能出现在基本块开头，它的取值由"从哪个前驱来"决定；`select` 是一条普通指令，两个值都已经算好，只是根据条件挑一个。所以只要两个分支都没有副作用（或者副作用已经被提到前面），优化器就倾向于把 `phi` 换成 `select`，因为 `select` 不会产生新的基本块，后续优化更简单。

&emsp;&emsp;最后看一个让人吃惊的 `-O2` 产物。`sum_to_n` 的循环没了：

```llvm
define dso_local noundef i32 @_Z8sum_to_ni(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp slt i32 %0, 1
  br i1 %2, label %14, label %3

3:                                                ; preds = %1
  %4 = shl nuw i32 %0, 1
  %5 = add nsw i32 %0, -1
  %6 = zext nneg i32 %5 to i33
  %7 = add nsw i32 %0, -2
  %8 = zext i32 %7 to i33
  %9 = mul i33 %6, %8
  %10 = lshr i33 %9, 1
  %11 = trunc nuw i33 %10 to i32
  %12 = add i32 %4, %11
  %13 = add i32 %12, -1
  br label %14

14:                                               ; preds = %3, %1
  %15 = phi i32 [ 0, %1 ], [ %13, %3 ]
  ret i32 %15
}
```

&emsp;&emsp;优化器识别出这是等差求和，直接用闭式公式 `n * (n + 1) / 2` 替换了整个循环。但它不能真的写成这个表达式——中间结果会溢出。所以优化器把中间计算提升到 `i33`：`zext` 到更宽的类型，乘完再 `trunc` 回来，这样乘法永远不会溢出，`nsw` 就可以安全地去掉。`i33` 这种"奇怪"的位宽在 IR 里完全合法，`iN` 支持任意位宽正是为了这种场景。

&emsp;&emsp;顺带提一句 `shl nuw i32 %0, 1` 就是 `n * 2`，`nuw` 是 "no unsigned wrap"；`zext nneg` 里的 `nneg` 表示"这个值非负，扩展不会改变数值"。这些都是优化器在更精确地描述它掌握的信息，好让后面的 Pass 继续利用。

### 5.4 类和结构体内存布局

&emsp;&emsp;IR 里没有 `class` 也没有 `struct` 类型，类最终只会变两样东西：一个**聚合类型**（描述对象的内存布局），和一组**普通函数**（成员函数）。先看内存布局：

```cpp
struct Padded {
    char c;
    int i;
};

struct Packed {
    char c;
    int i;
} __attribute__((packed));

struct Nested {
    char tag;
    Padded inner;
    double value;
};

Padded g_padded;
Packed g_packed;
Nested g_nested;
```

```llvm
%struct.Padded = type { i8, i32 }
%struct.Packed = type <{ i8, i32 }>
%struct.Nested = type { i8, %struct.Padded, double }

@g_padded = dso_local global %struct.Padded zeroinitializer, align 4
@g_packed = dso_local global %struct.Packed zeroinitializer, align 1
@g_nested = dso_local global %struct.Nested zeroinitializer, align 8
```

&emsp;&emsp;三个类型的字节布局可以算出来：

- `Padded`：`i8` 在偏移 0，`i32` 要求 4 字节对齐，所以偏移 1~3 填充，`i32` 在偏移 4。总大小 8，对齐 4。全局变量的 `align 4` 就是结构体的对齐。
- `Packed`：`<{ ... }>` 是 packed 结构体的写法，取消所有填充。`i8` 在 0，`i32` 在 1，总大小 5，对齐 1（全局变量上是 `align 1`）。
- `Nested`：`tag` 在 0，`inner` 是 `Padded`（对齐 4）所以从 4 开始，占 4~11，`value` 是 `double`（对齐 8）所以从 16 开始，占 16~23。总大小 24，对齐 8——尾部补了 4 字节让整体大小对齐到 8。

&emsp;&emsp;这些数字可以直接在 IR 里验证，因为 `sizeof` 是编译期常量：

```llvm
define dso_local noundef i32 @_Z11padded_sizev() #0 {
  ret i32 8
}

define dso_local noundef i32 @_Z11packed_sizev() #0 {
  ret i32 5
}

define dso_local noundef i32 @_Z11nested_sizev() #0 {
  ret i32 24
}
```

&emsp;&emsp;`sizeof` 在 IR 层不存在，它已经在编译期算完了。反过来，这也意味着**读 IR 时如果看到 `i32 8`，不一定知道它是什么**——它可能是 `sizeof`，可能是数组长度，也可能只是普通的常量。类型信息一旦被折叠成常量就不可恢复了。

&emsp;&emsp;现在看类。成员函数不占对象空间，只占代码空间：

```cpp
class Point {
  public:
    Point(int x, int y);
    int sum() const;
    void move(int dx, int dy);
    int getX() const { return x_; }

    static int count;

  private:
    int x_;
    int y_;
};
```

```llvm
%class.Point = type { i32, i32 }

@_ZN5Point5countE = dso_local global i32 0, align 4
```

&emsp;&emsp;`Point` 有两个 `int` 成员，所以类型是 `{ i32, i32 }`，大小 8（`point_size()` 返回 `ret i32 8`）。`static int count` 不在对象里，而是**模块级全局变量** `@_ZN5Point5countE`——它的生命周期和程序一样，和具体对象无关。这就是 C++ 里"静态成员属于类而不属于对象"的 IR 表达。

&emsp;&emsp;成员函数就是多了一个 `this` 参数的普通函数：

```llvm
define dso_local noundef i32 @_ZNK5Point3sumEv(ptr noundef nonnull align 4 dereferenceable(8) %0) #0 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %class.Point, ptr %3, i32 0, i32 0
  %5 = load i32, ptr %4, align 4
  %6 = getelementptr inbounds nuw %class.Point, ptr %3, i32 0, i32 1
  %7 = load i32, ptr %6, align 4
  %8 = add nsw i32 %5, %7
  ret i32 %8
}
```

&emsp;&emsp;三件事需要说明。

&emsp;&emsp;**第一，`%0` 就是 `this`。** 它被标成 `nonnull align 4 dereferenceable(8)`——`dereferenceable(8)` 正好是 `sizeof(Point)`，因为成员函数可以访问整个对象。函数末尾还有个 `align 2`，表示这个函数的地址按 2 字节对齐（来自类的对齐要求）。

&emsp;&emsp;**第二，成员访问就是 GEP 加 `load`。** `getelementptr inbounds nuw %class.Point, ptr %3, i32 0, i32 0` 里第一个 `i32 0` 是"不移动基址"，第二个 `i32 0` 是"取第 0 个字段"。这就是第 4 章讲 GEP 时说的"第一个索引永远为 0，后续索引才进入聚合体内部"。`x_` 和 `y_` 的偏移分别是 0 和 4，这里没有出现"偏移 4"这样的数字——**GEP 用字段索引而不是字节偏移**，字节偏移由类型布局和 `datalayout` 推导出来。

&emsp;&emsp;**第三，名字里编进了 `const` 和类名。** `_ZNK5Point3sumEv` 拆开来是 `_ZN`（嵌套名开始）、`K`（const 成员函数）、`5Point`（类名）、`3sum`（方法名）、`E`（嵌套名结束）、`v`（无参数）。`const` 成员函数在名字里带一个 `K`，因为 `Point::sum() const` 和 `Point::sum()` 是两个不同的重载。

&emsp;&emsp;构造函数也一样，只是名字里带 `C1`/`C2`：

```llvm
define dso_local void @_ZN5PointC2Eii(ptr noundef nonnull align 4 dereferenceable(8) %0, i32 noundef %1, i32 noundef %2) unnamed_addr #0 align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store ptr %0, ptr %4, align 8
  store i32 %1, ptr %5, align 4
  store i32 %2, ptr %6, align 4
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr inbounds nuw %class.Point, ptr %7, i32 0, i32 0
  %9 = load i32, ptr %5, align 4
  store i32 %9, ptr %8, align 4
  %10 = getelementptr inbounds nuw %class.Point, ptr %7, i32 0, i32 1
  %11 = load i32, ptr %6, align 4
  store i32 %11, ptr %10, align 4
  %12 = load i32, ptr @_ZN5Point5countE, align 4
  %13 = add nsw i32 %12, 1
  store i32 %13, ptr @_ZN5Point5countE, align 4
  ret void
}
```

&emsp;&emsp;构造函数返回 `void`，第一个参数是 `this`，函数体是"往 `this` 的各个字段里 `store` 初始值"。编译器还会额外声明一个 `C1` 版本：

```llvm
@_ZN5PointC1Eii = dso_local unnamed_addr alias void (ptr, i32, i32), ptr @_ZN5PointC2Eii
```

&emsp;&emsp;`C1` 是"完整对象构造函数"，`C2` 是"基类子对象构造函数"。在大多数情况下它们的代码完全一样，所以 clang 用 `alias` 让两个符号指向同一份代码，避免重复生成。析构函数那边有一整套：

| 后缀 | 含义 |
| --- | --- |
| `C1` | 完整对象的构造函数 |
| `C2` | 基类子对象的构造函数 |
| `D0` | 删除析构函数（deleting destructor）：析构 + `operator delete` |
| `D1` | 完整对象的析构函数 |
| `D2` | 基类子对象的析构函数 |

&emsp;&emsp;5.5 讲虚析构的时候会看到 `D0` 和 `D1` 分别被用在什么地方。

&emsp;&emsp;最后看一下 `const` 成员函数在 IR 里到底有什么不同。答案有点反直觉：**指令层面完全一样**。`sum()` 和 `move()` 的函数体没有任何"只读"标记，唯一的区别是名字里多了个 `K`，以及调用点要保证 `this` 指向的是 const 对象。`const` 是纯编译期的约束，到 IR 层只剩名字修饰。只有在 `-O2` 之后，优化器看懂了函数不写内存，才会把 `this` 参数标成 `readonly`：

```llvm
define dso_local noundef i32 @_ZNK4Base5valueEv(ptr noundef nonnull readonly align 8 captures(none) dereferenceable(12) %0) unnamed_addr #0 align 2 {
```

&emsp;&emsp;这是本章反复出现的主题：**C++ 的很多语义到 IR 层只剩下"约定"，需要优化器自己推断**。`const` 如此，`constexpr`、`noexcept`、`[[nodiscard]]` 也都是如此。

### 5.5 继承、多态和虚函数

&emsp;&emsp;虚函数是 C++ 里最"不 C"的东西，也是 IR 里痕迹最明显的东西。先看布局：

```cpp
struct Base {
    virtual int value() const;
    virtual ~Base();
    int b = 1;
};

struct Derived : Base {
    int value() const override;
    ~Derived() override;
    int d = 2;
};
```

```llvm
%struct.Base = type <{ ptr, i32, [4 x i8] }>
%struct.Base.base = type <{ ptr, i32 }>
%struct.Derived = type { %struct.Base.base, i32 }
```

&emsp;&emsp;这三个类型需要仔细读。

&emsp;&emsp;`Base` 里有一个**隐式的 vptr**，占 8 字节放在偏移 0 处——这就是多态的物理基础。`int b` 在偏移 8。字段加起来只有 12 字节，但 `Base` 的对齐是 8（因为含指针），所以要补 4 字节到 16。clang 把这 4 字节显式写成一个 `[4 x i8]` 字段，整体再用 `<{ ... }>` 的 packed 形式声明——这样"字段布局"和"对象大小"就分开了：真正有意义的字段只占 12 字节，`[4 x i8]` 只是补齐，而 C++ 允许派生类复用这块尾部填充。

&emsp;&emsp;`%struct.Base.base` 是"基类子对象"的布局，只有 `{ ptr, i32 }`，12 字节，**不含尾部填充**。所以 `Derived` 的布局是 `{ %struct.Base.base, i32 }`——基类子对象占 0~11，`d` 直接放在偏移 12，正好用掉了 `Base` 的那 4 字节填充。

&emsp;&emsp;结果很漂亮：`sizeof(Base)` 和 `sizeof(Derived)` 都是 16。派生类加了一个 `int` 成员却没有变大，因为基类的尾部填充被复用了。这就是 IR 里那两个 `ret` 的含义：

```llvm
define dso_local noundef i32 @_Z9base_sizev() #0 {
  ret i32 16
}

define dso_local noundef i32 @_Z12derived_sizev() #0 {
  ret i32 16
}
```

&emsp;&emsp;函数签名里的 `dereferenceable(12)` 也对应这一点：`Base` 的 `this` 可以保证访问 12 字节（vptr + `b`），但那 4 字节填充不保证，因为派生类可能在里面放东西。

&emsp;&emsp;接下来是虚函数表。它就是一个模块级的常量数组：

```llvm
@_ZTV4Base = dso_local unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI4Base, ptr @_ZNK4Base5valueEv, ptr @_ZN4BaseD1Ev, ptr @_ZN4BaseD0Ev] }, align 8

@_ZTV7Derived = dso_local unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI7Derived, ptr @_ZNK7Derived5valueEv, ptr @_ZN7DerivedD1Ev, ptr @_ZN7DerivedD0Ev] }, align 8
```

&emsp;&emsp;`_ZTV` 前缀是 "vtable" 的修饰形式，后面的 `4Base`/`7Derived` 是类名。每个 vtable 有 5 个槽位：

| 槽位 | 内容 | 说明 |
| --- | --- | --- |
| 0 | `null` | offset-to-top，单继承里是 0（多继承时用于 `this` 调整） |
| 1 | `@_ZTI4Base` | RTTI 指针，指向 typeinfo 对象 |
| 2 | `@_ZNK4Base5valueEv` | 第 1 个虚函数 |
| 3 | `@_ZN4BaseD1Ev` | 完整对象析构函数 |
| 4 | `@_ZN4BaseD0Ev` | 删除析构函数 |

&emsp;&emsp;对象的 vptr 并不指向数组开头，而是指向**槽位 2**，也就是"第一个虚函数"的位置。这样 `vptr[0]` 就是 `value()`，`vptr[-1]` 是 RTTI，`vptr[-2]` 是 offset-to-top。构造函数里可以看到这个偏移：

```llvm
define linkonce_odr dso_local void @_ZN7DerivedC2Ev(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @_ZN4BaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %3) #5
  store ptr getelementptr inbounds inrange(-16, 24) ({ [5 x ptr] }, ptr @_ZTV7Derived, i32 0, i32 0, i32 2), ptr %3, align 8
  %4 = getelementptr inbounds nuw %struct.Derived, ptr %3, i32 0, i32 1
  store i32 2, ptr %4, align 4
  ret void
}
```

&emsp;&emsp;这个构造函数分三步：先调用基类构造函数，再**改写 vptr**，最后初始化自己的成员。改写 vptr 是必须的——`Base` 的构造函数会把 vptr 设成 `@_ZTV4Base`，`Derived` 的构造函数要把它覆盖成 `@_ZTV7Derived`。这解释了一个经典问题：**构造函数里调用虚函数不会走派生类版本**，因为那时候 vptr 还指向基类的表。

&emsp;&emsp;`getelementptr ... i32 0, i32 2` 的作用是取数组第 2 个元素的地址，也就是 vptr 应该存的值。`inrange(-16, 24)` 是告诉优化器"这个 GEP 的索引只会在 -2 到 3 之间"，因为 vtable 一共有 5 个槽，而 vptr 指向中间。

&emsp;&emsp;虚调用是本章最值得看的一段 IR：

```cpp
int call_virtual(Base *p) { return p->value(); }
```

```llvm
define dso_local noundef i32 @_Z12call_virtualP4Base(ptr noundef %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr inbounds ptr, ptr %4, i64 0
  %6 = load ptr, ptr %5, align 8
  %7 = call noundef i32 %6(ptr noundef nonnull align 8 dereferenceable(12) %3)
  ret i32 %7
}
```

&emsp;&emsp;IR 里**没有"虚调用"这种指令**。`p->value()` 被拆成了四步：

1. `%4 = load ptr, ptr %3` —— 从对象里读出 vptr。
2. `%5 = getelementptr inbounds ptr, ptr %4, i64 0` —— 算出 `vptr + 0`，即第一个虚函数槽的地址。
3. `%6 = load ptr, ptr %5` —— 从表里读出真正的函数指针。
4. `%7 = call i32 %6(this)` —— 间接调用。

&emsp;&emsp;所以一次虚调用 = 两次额外内存读 + 一次间接跳转。开销就在这里，所谓"虚函数比普通函数慢"指的就是这三次间接。同时也解释了为什么虚调用难以内联：编译器在编译 `call_virtual` 时不知道 `p` 的实际类型，`%6` 是个运行期才知道的值。

&emsp;&emsp;注意第 2 步的 `i64 0`。如果调用的是第二个虚函数，这里就是 `i64 1`。`delete` 走的是删除析构函数，槽位是 2：

```llvm
define dso_local void @_Z7destroyP4Base(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = icmp eq ptr %3, null
  br i1 %4, label %9, label %5

5:                                                ; preds = %1
  %6 = load ptr, ptr %3, align 8
  %7 = getelementptr inbounds ptr, ptr %6, i64 2
  %8 = load ptr, ptr %7, align 8
  call void %8(ptr noundef nonnull align 8 dereferenceable(12) %3) #5
  br label %9

9:                                                ; preds = %5, %1
  ret void
}
```

&emsp;&emsp;`delete p` 先判空（C++ 规定 `delete nullptr` 是空操作），再从 `vptr[2]` 取出删除析构函数并调用。删除析构函数 `D0` 的实现是这样的：

```llvm
define dso_local void @_ZN7DerivedD0Ev(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #0 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @_ZN7DerivedD1Ev(ptr noundef nonnull align 8 dereferenceable(16) %3) #5
  call void @_ZdlPvm(ptr noundef %3, i64 noundef 16) #6
  ret void
}
```

&emsp;&emsp;先调用 `D1` 析构（它会先析构派生类部分，再调用基类析构），再用 `_ZdlPvm` 释放内存，`i64 16` 就是 `sizeof(Derived)`。整个链是：`delete` → `vptr[2]` → `Derived::D0` → `Derived::D1` → `Base::~Base` → `operator delete(ptr, 16)`。基类的 `D0` 位置被派生类的 `D0` 覆盖了，这就是"虚析构函数保证正确释放"的机制。

&emsp;&emsp;`D1` 和 `D2` 的分工也能从 IR 里看出来：

```llvm
define dso_local void @_ZN7DerivedD2Ev(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #0 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @_ZN4BaseD2Ev(ptr noundef nonnull align 8 dereferenceable(12) %3) #5
  ret void
}
```

&emsp;&emsp;`D2` 只负责析构"基类子对象"的部分，所以它做的唯一事情就是调用基类的 `D2`。派生类自己的成员如果有析构函数，会在这之前插进去。`D1` 则是"完整对象"版本，通常就是 `D2` 加一个 `alias`——本项目里两个析构函数都是空的，所以直接合并成了一个符号。

&emsp;&emsp;顺带看看 RTTI。`@_ZTI4Base` 和 `@_ZTI7Derived` 是 typeinfo 对象：

```llvm
@_ZTI4Base = dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS4Base }, align 8

@_ZTI7Derived = dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS7Derived, ptr @_ZTI4Base }, align 8

@_ZTS7Derived = dso_local constant [9 x i8] c"7Derived\00", align 1
```

&emsp;&emsp;`_ZTI` 是 typeinfo，`_ZTS` 是类名字符串。`Base` 的 typeinfo 是 `{ vtable指针, 名字 }`，`Derived` 的是 `{ vtable指针, 名字, 基类 typeinfo指针 }`。那个 `getelementptr ... i64 2` 是在取 `__si_class_type_info` 的 vtable 中的第三个槽（偏移 16 字节），这是 C++ ABI 规定的布局。`dynamic_cast` 和 `typeid` 在运行期就是靠这套结构工作的——它们不在 IR 里生成特殊指令，而是调用 `__dynamic_cast` 之类的运行时函数。

&emsp;&emsp;最后对比 `-O2`：

```llvm
define dso_local noundef i32 @_Z12call_virtualP4Base(ptr noundef %0) local_unnamed_addr #4 {
  %2 = load ptr, ptr %0, align 8, !tbaa !13
  %3 = load ptr, ptr %2, align 8
  %4 = tail call noundef i32 %3(ptr noundef nonnull align 8 dereferenceable(12) %0)
  ret i32 %4
}
```

&emsp;&emsp;`alloca` 和两次 `store`/`load` 没了，但**两次 `load` 和间接调用还在**——因为 `p` 的静态类型是 `Base*`，优化器没有任何依据知道它实际指向 `Derived`。这就是所谓"去虚化"（devirtualization）的边界：只有当优化器能证明对象的动态类型（比如对象是局部变量、刚 `new` 出来、或者类被标了 `final`）时，才会把间接调用换成直接调用。同一个文件里 `make_derived()` 返回 `new Derived()`，如果调用点内联了，`Derived` 的 `value()` 就能被直接调用；如果没内联，就只能老老实实走 vtable。

&emsp;&emsp;另外注意 `-O2` 下 `@_ZN7DerivedD1Ev` 变成了 `alias ... ptr @_ZN4BaseD2Ev`，vtable 里的 `D1` 槽也直接填了 `_ZN4BaseD2Ev`。这是因为两个析构函数的机器码完全一样，链接器做了**相同代码折叠**（ICF）。这不是语言语义的要求，而是纯粹的体积优化。

### 5.6 C++异常和RAII

&emsp;&emsp;异常是 C++ 唯一一个能"从函数中间跳出去"的控制流，所以它在 IR 里需要专门的机制。核心变化只有一条：**可能抛异常的调用从 `call` 变成 `invoke`**。

```cpp
extern int may_throw(int);

int handle() {
    try {
        return may_throw(1);
    } catch (int e) {
        return e + 100;
    }
}
```

```llvm
define dso_local noundef i32 @_Z6handlev() #0 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = invoke noundef i32 @_Z9may_throwi(i32 noundef 1)
          to label %6 unwind label %7

6:                                                ; preds = %0
  store i32 %5, ptr %1, align 4
  br label %22

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %9 = extractvalue { ptr, i32 } %8, 0
  store ptr %9, ptr %2, align 8
  %10 = extractvalue { ptr, i32 } %8, 1
  store i32 %10, ptr %3, align 4
  br label %11

11:                                               ; preds = %7
  %12 = load i32, ptr %3, align 4
  %13 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi) #6
  %14 = icmp eq i32 %12, %13
  br i1 %14, label %15, label %24

15:                                               ; preds = %11
  %16 = load ptr, ptr %2, align 8
  %17 = call ptr @__cxa_begin_catch(ptr %16) #6
  %18 = load i32, ptr %17, align 4
  store i32 %18, ptr %4, align 4
  %19 = load i32, ptr %4, align 4
  %20 = add nsw i32 %19, 100
  store i32 %20, ptr %1, align 4
  call void @__cxa_end_catch() #6
  br label %22

21:                                               ; No predecessors!
  call void @llvm.trap()
  unreachable

22:                                               ; preds = %15, %6
  %23 = load i32, ptr %1, align 4
  ret i32 %23

24:                                               ; preds = %11
  %25 = load ptr, ptr %2, align 8
  %26 = load i32, ptr %3, align 4
  %27 = insertvalue { ptr, i32 } poison, ptr %25, 0
  %28 = insertvalue { ptr, i32 } %27, i32 %26, 1
  resume { ptr, i32 } %28
}
```

&emsp;&emsp;这一段信息量很大，逐块读。

&emsp;&emsp;**函数级的变化**：函数定义末尾多了 `personality ptr @__gxx_personality_v0`。personality 函数是栈展开时由 unwinder 回调的"人格函数"（personality routine），负责判断某个栈帧能不能处理这个异常。`__gxx_personality_v0` 是 Itanium ABI 的 C++ 实现，它会去读 `.gcc_except_table` 里的 LSDA（语言相关的数据区），那张表记录着"哪段代码能捕获什么类型、对应的 handler 在哪"。**这张表不在 IR 里**——IR 只能告诉 llc "这里有异常处理"，具体的表由后端生成，放在单独的 section 里。

&emsp;&emsp;**`invoke` 的两个后继**：`to label %6 unwind label %7`。正常返回走 `%6`，抛异常走 `%7`。这和 `call` 的最大区别是它**显式地把"异常"建模成了一条控制流边**——CFG 里从此多了一条边，所有基于 CFG 的分析（支配关系、`phi` 插入、活跃变量）都会把异常路径算进去。这也是为什么开了异常之后代码往往更慢：不是异常贵，而是**正常路径上多了这些边，优化空间被压缩了**。

&emsp;&emsp;**`landingpad`**：异常路径的第一个基本块必须以 `landingpad` 开头，它的结果是一个 `{ ptr, i32 }` 结构体：

```llvm
%8 = landingpad { ptr, i32 }
        catch ptr @_ZTIi
```

- `ptr` 是异常对象（这里就是 `int` 类型的那个对象）的地址；
- `i32` 是 **selector**，一个由 personality 函数分配的编号，用来区分"是哪个 handler"。

后面的 `catch ptr @_ZTIi` 表示"这个 landingpad 处理 C++ 类型 `int` 的异常"，`@_ZTIi` 就是 `int` 的 typeinfo（5.5 讲过的那套 RTTI 结构）。

&emsp;&emsp;**类型匹配是手写的**：`llvm.eh.typeid.for.p0(@_ZTIi)` 返回 `@_ZTIi` 对应的 selector 值，然后和 landingpad 给出的 selector 比较：

```llvm
%13 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi) #6
%14 = icmp eq i32 %12, %13
br i1 %14, label %15, label %24
```

&emsp;&emsp;相等就进 handler `%15`，不等就跳 `%24` 重新抛出。注意这里用的是 selector 的整数比较，而不是指针比较——selector 是运行时分配的，同一个 typeinfo 在不同栈帧上拿到的编号可能不同，所以必须用 `eh.typeid.for` 换算。

&emsp;&emsp;**handler 的边界由运行时函数维护**：`__cxa_begin_catch` 通知运行时"我开始处理这个异常了"，`__cxa_end_catch` 通知"处理完了，可以释放异常对象了"。这两个调用必须成对出现，而且 `end_catch` 自己也可能抛（比如异常的析构函数抛），所以它在 `catch(...) + throw;` 的场景里会以 `invoke` 的形式出现。

&emsp;&emsp;**`resume` 是"继续往上抛"**：`%24` 把 landingpad 拆出来的两个值重新拼成 `{ ptr, i32 }`，然后用 `resume` 交给上层栈帧。`insertvalue ... poison ...` 是构造聚合值的标准写法——`poison` 表示"初值无所谓，马上就会被覆盖"。`resume` 和 `ret` 一样是终结指令，它之后不可能有代码。

&emsp;&emsp;`%21` 那个块标着 `; No predecessors!`，函数体里是 `llvm.trap()` 加 `unreachable`。这是 clang 为"C++ 语义上不可能到达"的位置生成的兜底块，`unreachable` 告诉优化器"走到这里就是未定义行为"，优化器可以据此删掉相关的分支。

&emsp;&emsp;现在看 RAII。这是异常机制里最实用的一环：

```cpp
struct Res {
    int id;
    explicit Res(int i);
    ~Res();
};

int cleanup_demo(int n) {
    Res r(n);
    return may_throw(n);
}
```

```llvm
define dso_local noundef i32 @_Z12cleanup_demoi(i32 noundef %0) #0 personality ptr @__gxx_personality_v0 {
  %2 = alloca i32, align 4
  %3 = alloca %struct.Res, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %6 = load i32, ptr %2, align 4
  call void @_ZN3ResC1Ei(ptr noundef nonnull align 4 dereferenceable(4) %3, i32 noundef %6)
  %7 = load i32, ptr %2, align 4
  %8 = invoke noundef i32 @_Z9may_throwi(i32 noundef %7)
          to label %9 unwind label %10

9:                                                ; preds = %1
  call void @_ZN3ResD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %3) #6
  ret i32 %8

10:                                               ; preds = %1
  %11 = landingpad { ptr, i32 }
          cleanup
  %12 = extractvalue { ptr, i32 } %11, 0
  store ptr %12, ptr %4, align 8
  %13 = extractvalue { ptr, i32 } %11, 1
  store i32 %13, ptr %5, align 4
  call void @_ZN3ResD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %3) #6
  br label %14

14:                                               ; preds = %10
  %15 = load ptr, ptr %4, align 8
  %16 = load i32, ptr %5, align 4
  %17 = insertvalue { ptr, i32 } poison, ptr %15, 0
  %18 = insertvalue { ptr, i32 } %17, i32 %16, 1
  resume { ptr, i32 } %18
}
```

&emsp;&emsp;关键在于**析构函数的调用出现了两次**：

- 正常路径 `%9`：`invoke` 正常返回后，`call @_ZN3ResD1Ev` 析构局部对象，`ret`。
- 异常路径 `%10`：`landingpad ... cleanup` 不匹配任何 catch 类型（`cleanup` 的意思是"只做清理，不吃异常"），调用同一个析构函数，然后 `resume` 继续往上抛。

&emsp;&emsp;这里没有任何"finally"或者"析构块"的概念——**编译器就是把析构函数的调用复制到了每一条退出路径上**。函数里有几个 `return`、几条可能抛异常的边，就有几份析构调用。这也是为什么 RAII 的开销是可控的：析构函数本身是一次普通调用，异常路径上的那份额外代码只在真正抛异常时才执行，正常路径上没有额外分支。

&emsp;&emsp;`cleanup` 和 5.6 开头的 `catch ptr @_ZTIi` 是对应的两种 landingpad 动作：`cleanup` 只清理、不捕获，异常继续传播；`catch` 会"消费"这个异常，进入 handler 之后由 `__cxa_begin_catch`/`__cxa_end_catch` 管理生命周期。

&emsp;&emsp;还有一个细节：`Res` 的析构函数在文件里是 `declare` 的，而且带 `nounwind`：

```llvm
declare void @_ZN3ResD1Ev(ptr noundef nonnull align 4 dereferenceable(4)) unnamed_addr #4

attributes #4 = { nounwind "frame-pointer"="all" ... }
```

&emsp;&emsp;`nounwind` 不是随便加的——C++11 起析构函数默认是 `noexcept` 的，如果在栈展开过程中析构函数又抛出异常，程序会直接 `std::terminate`。所以析构调用可以放心地放在 landingpad 之后：**它不会破坏正在传播的异常**。

&emsp;&emsp;最后看 `catch(...)` 加重抛：

```cpp
int rethrow_demo() {
    try {
        may_throw(2);
    } catch (...) {
        throw;
    }
    return 0;
}
```

```llvm
5:                                                ; preds = %0
  %6 = landingpad { ptr, i32 }
          catch ptr null
  ; ... 保存异常对象和 selector ...

9:                                                ; preds = %5
  %10 = load ptr, ptr %1, align 8
  %11 = call ptr @__cxa_begin_catch(ptr %10) #6
  invoke void @__cxa_rethrow() #7
          to label %26 unwind label %12

12:                                               ; preds = %9
  %13 = landingpad { ptr, i32 }
          cleanup
  ; ... 保存 ...
  invoke void @__cxa_end_catch()
          to label %16 unwind label %23

23:                                               ; preds = %12
  %24 = landingpad { ptr, i32 }
          catch ptr null
  %25 = extractvalue { ptr, i32 } %24, 0
  call void @__clang_call_terminate(ptr %25) #8
  unreachable
```

&emsp;&emsp;几个新东西：`catch ptr null` 就是 `catch (...)`，`null` 的 typeinfo 表示"匹配任何类型"。`__cxa_rethrow` 标记为 `noreturn`，但它仍然可能抛（重抛本身要构造新的异常状态），所以用 `invoke`。最里面的 `%23` 是一个只做 `__clang_call_terminate` 的 landpad——如果 `__cxa_end_catch` 也抛了，说明异常处理机制本身出了问题，只能直接终止程序：

```llvm
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) #5 comdat {
  %2 = call ptr @__cxa_begin_catch(ptr %0) #6
  call void @_ZSt9terminatev() #8
  unreachable
}
```

&emsp;&emsp;这一串"landingpad 里套 landingpad"的结构不难理解，它就是把 C++ 标准里"异常处理过程中不得再抛"的规则**逐条落实成了控制流**。每一层可能抛的地方都配一个 landingpad，最坏情况下走到 `terminate`。

&emsp;&emsp;如果编译时加 `-fno-exceptions`，这一整套东西都不会出现：`invoke` 变回 `call`，`landingpad`、`personality`、`resume` 全部消失，`throw` 语句直接变成对 `std::terminate` 或 `abort` 的调用。**异常在 IR 里是可选的**，这也是 LLVM 能同时服务 C 和 C++ 的原因。

### 5.7 模板

&emsp;&emsp;模板是 C++ 里最"元"的机制，但它在 IR 里留下的痕迹最少——因为**模板在编译期就消失了**。IR 里没有任何"模板函数"或者"模板类型"，只有一个个已经具体化的普通函数和普通类型。

```cpp
template <typename T> T maxv(T a, T b) { return a > b ? a : b; }

int use_max_int() { return maxv(1, 2); }
double use_max_double() { return maxv(1.5, 2.5); }
```

```llvm
define linkonce_odr dso_local noundef i32 @_Z4maxvIiET_S0_S0_(i32 noundef %0, i32 noundef %1) #0 comdat {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %5 = load i32, ptr %3, align 4
  %6 = load i32, ptr %4, align 4
  %7 = icmp sgt i32 %5, %6
  br i1 %7, label %8, label %10
  ; ...
}

define linkonce_odr dso_local noundef double @_Z4maxvIdET_S0_S0_(double noundef %0, double noundef %1) #0 comdat {
  ; ... 同样的结构，但比较换成了 fcmp ogt ...
  %7 = fcmp ogt double %5, %6
}
```

&emsp;&emsp;两份函数体长得几乎一样，但**没有任何共享**——`maxv<int>` 是一个独立的 `i32` 函数，`maxv<double>` 是一个独立的 `double` 函数，指令、寄存器、类型全都不一样。这就是所谓的"模板实例化"：编译器为每组模板实参各生成一份代码。

&emsp;&emsp;名字修饰比普通函数复杂一些：`_Z4maxvIiET_S0_S0_` 拆开是 `_Z` + `4maxv` + `I...E`（模板实参列表：`i` 表示 `T = int`）+ `T_`（返回类型 `T`）+ `S0_` + `S0_`（两个参数，类型都是 `T`）。`S0_` 是**替换**（substitution）：Itanium ABI 允许用 `S<n>_` 指代前面已经出现过的类型，避免重复拼写。`T_` 里的 `T` 表示"模板参数"本身，`S0_` 指向第一个模板实参。这种压缩方式让复杂模板的名字不至于爆炸。

&emsp;&emsp;`linkonce_odr` 和 `comdat` 是给链接器看的：模板定义在头文件里，每个包含它的编译单元都会生成一份 `maxv<int>`。如果每个 `.o` 都导出一份强定义，链接时就会符号重复。所以前端把这类函数标成 `linkonce_odr`（可以合并、只保留一份），并放进一个 `comdat` 组。链接器看到多个同名 comdat 组，随便留一个，剩下的丢掉。

&emsp;&emsp;显式特化则是一个普通函数，没有 `linkonce_odr`：

```cpp
template <> const char *maxv<const char *>(const char *a, const char *b) {
    return strcmp(a, b) > 0 ? a : b;
}
```

```llvm
define dso_local noundef ptr @_Z4maxvIPKcET_S2_S2_(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = call i32 @strcmp(ptr noundef %5, ptr noundef %6) #4
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %11
  ; ...
}
```

&emsp;&emsp;名字里的 `IPKcE` 表示 `T = const char *`（`P` 是指针，`K` 是 const，`c` 是 char）。注意它的函数体完全不同：`const char*` 不能直接用 `>` 比较（那比的是指针地址），所以特化版本调用了 `strcmp`。**模板的"泛型"是编译期展开的，每组实参都可以有不同的实现**，这一点和 Java 的泛型擦除正好相反。

&emsp;&emsp;类模板也会被完全展开成具体的聚合类型：

```cpp
template <typename T, int N> struct FixedArray {
    T data[N];
    int size() const { return N; }
    T &at(int i) { return data[i]; }
};

int use_array() {
    FixedArray<int, 4> arr{};
    arr.at(0) = 7;
    return arr.size() + arr.data[0];
}
```

```llvm
%struct.FixedArray = type { [4 x i32] }

define linkonce_odr dso_local noundef nonnull align 4 dereferenceable(4) ptr @_ZN10FixedArrayIiLi4EE2atEi(ptr noundef nonnull align 4 dereferenceable(16) %0, i32 noundef %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  store ptr %0, ptr %3, align 8
  store i32 %1, ptr %4, align 4
  %5 = load ptr, ptr %3, align 8
  %6 = getelementptr inbounds nuw %struct.FixedArray, ptr %5, i32 0, i32 0
  %7 = load i32, ptr %4, align 4
  %8 = sext i32 %7 to i64
  %9 = getelementptr inbounds [4 x i32], ptr %6, i64 0, i64 %8
  ret ptr %9
}

define linkonce_odr dso_local noundef i32 @_ZNK10FixedArrayIiLi4EE4sizeEv(ptr noundef nonnull align 4 dereferenceable(16) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret i32 4
}
```

&emsp;&emsp;三个值得注意的点。

&emsp;&emsp;**类型里只有 4 没有 N。** `%struct.FixedArray = type { [4 x i32] }`——`N` 已经被代入，数组长度成了类型的一部分（第 4 章说过 `[4 x i32]` 和 `[5 x i32]` 是两个不同的类型）。IR 里不存在"长度是参数"的数组。

&emsp;&emsp;**`size()` 返回常量。** `ret i32 4`——`N` 不占任何存储，它只是编译期的值。这解释了为什么 `FixedArray<T, N>` 的"长度"是零开销的：它根本没进对象。

&emsp;&emsp;**`at()` 里的 GEP 有两级。** 第一级 `getelementptr %struct.FixedArray, ptr %5, i32 0, i32 0` 取到数组字段本身，第二级 `getelementptr [4 x i32], ptr %6, i64 0, i64 %8` 用变量索引取元素。第一级的 `i64 0` 是"不动数组起始地址"，第二级的 `i64 %8` 才是真正的下标。变量下标必须先 `sext` 到 `i64`（GEP 的索引类型固定是 `i64` 系）。

&emsp;&emsp;名字里的 `Li4E` 是"字面量 `int` 4"的编码，所以 `_ZN10FixedArrayIiLi4EE2atEi` 读作"`FixedArray<int, 4>::at(int)`"。模板实参无论是类型还是值，都会编进符号名——这就是为什么两套模板实参会产生两个不同的符号，链接时不会冲突。

&emsp;&emsp;最后看 `-O2`，模板的痕迹几乎被抹干净了：

```llvm
define dso_local noundef i32 @_Z11use_max_intv() local_unnamed_addr #2 {
  ret i32 2
}

define dso_local noundef double @_Z14use_max_doublev() local_unnamed_addr #2 {
  ret double 2.500000e+00
}

define dso_local noundef i32 @_Z9use_arrayv() local_unnamed_addr #2 {
  ret i32 11
}
```

&emsp;&emsp;`maxv(1, 2)` 被内联后在编译期算出了 2，`maxv(1.5, 2.5)` 算出了 2.5，`use_array()` 里那个局部数组被完全 SROA 掉、`at(0) = 7` 和 `size() + data[0]` 折叠成 11。`_Z4maxvIiET_S0_S0_` 和 `_Z4maxvIdET_S0_S0_` 这两个函数在 `-O2` 的文件里**整个消失了**，因为没有任何调用者了，死代码消除把它们删了。

&emsp;&emsp;这说明模板代码和普通代码在优化器眼里没有任何区别：实例化出来的都是普通函数，能内联就内联，能常量折叠就折叠。所谓"模板的零开销抽象"，本质上是**前端把泛型展开成具体代码，优化器再把这些具体代码优化掉**。

&emsp;&emsp;把这七个小节串起来看，C++ 到 LLVM IR 的翻译其实遵循一个统一的套路：**语言特性要么落地成 IR 已有的原语（类型、指令、全局变量），要么落地成前端生成的辅助符号（vtable、typeinfo、mangled name），要么落地成运行库调用（`__cxa_*`、`operator new`）**。IR 本身保持极简：没有类、没有虚函数、没有异常、没有模板，只有模块、函数、基本块、指令、类型和全局变量。正是这种"极简 + 约定"的分工，让 LLVM 能同时作为 C、C++、Rust、Swift 乃至自研语言的后端，而优化器只需要认识那几十条指令。

## 6 总结

&emsp;&emsp;LLVM IR 可以用一句话概括：**它是一种为优化和代码生成而设计的、强类型的、SSA 形式的中间语言；它的价值不在于"能表达多少语法"，而在于"用多小的代价表达语义"**。

1. **IR 是契约，不是语法树。** 它刻意丢掉了源码的语法细节，也刻意丢掉了目标机器的硬件细节。丢掉的东西由前后端各自补回，这正是它能在几十年里同时服务 C、C++、Rust、Swift、Julia 和你自己的语言的原因。
2. **SSA 是为了分析，不是为了好看。** "每个值只定义一次"让 def-use 链唯一、让常量传播和死代码消除变成局部判断、让寄存器分配有清晰的活跃区间。`-O0` 里那些 `alloca/load/store` 只是权宜之计，`mem2reg` 之后才显出 IR 的真面目。
3. **类型必须显式。** 没有隐式转换，没有"自动提升"。C++ 的 `int + long long` 在 IR 里一定要有 `sext` 和 `trunc`；`double → float` 一定要有 `fptrunc`。这看起来啰嗦，但它让每条指令的语义都是自足的。
4. **优化器靠的是承诺。** `nsw`、`nuw`、`inbounds`、`nonnull`、`dereferenceable`、`mustprogress`、`llvm.loop`——这些修饰和元数据都是前端写给优化器的保证书。承诺越多，优化越猛；承诺一旦被违反，程序就是未定义行为。
5. **C++ 的所有高级特性都能被"降级"。** 类降级成聚合类型加普通函数，虚函数降级成表加间接调用，异常降级成 `invoke` 加 `landingpad` 加运行库，模板在前端就被展开干净。IR 里一个都不剩——**能被降级的抽象，才是零开销的抽象**。
6. **同一份 IR 既能优化也能直接执行。** `lli` 可以解释执行它，JIT 可以把它编译进正在运行的进程，`llc` 可以把它变成汇编。文本、bitcode、内存对象三种形式之间的无损转换，让它能同时充当调试载体和分发格式。

&emsp;&emsp;回到最初那个问题：为什么要有 IR？因为编译器的复杂度不会消失，只会转移。IR 把"理解源语言"和"理解目标机器"这两件最复杂的事隔开，让它们可以各自演进，中间用一份足够简单、又足够精确的表示连接起来。LLVM IR 的成功不在于它多么完备，而在于它**在该简单的地方足够简单（几十条指令、七种类型构造），在该精确的地方足够精确（显式类型、SSA、对齐与布局）**。理解了这一点，再看任何一份 `.ll`，都不会觉得它是一堆天书了。
