# C++ 反汇编简要

&emsp;&emsp;**摘要**：本文主要描述x86_64机器中C++代码在汇编中的具体代码。
&emsp;&emsp;**关键字**：cpp,IA32,asm
&emsp;&emsp;**注意**：本书假定你拥有基本的C++软件开发能力，能够理解基本的C++代码。并且熟悉汇编代码，了解基本的取址模式并且熟悉IA32指令集（文中会对IA32的部分指令集进行描述，但是不会过于详细的深入）。

## 1 前言
&emsp;&emsp;C/C++都需要经过编译器变成对应的机器码，通常编译器对程序员是个黑盒子。有些时候我们可能会纠结编译器会不会进行RVO，EBO等优化，以及一些在我们看起来应该正常的代码因为一些UB的行为被C++编译器优化成了不可预期的代码。这时候如果我们了解具体代码是如何编译成对应的二进制机器码对我们查具体的问题非常有益。另一种场景，在开发软件时，线上环境能够复现的问题，我们本地可能是无法复现的。这就需要我们根据线上的堆栈分析具体的原因。而在一些场景下线上堆栈可能是被破坏了的，这就需要我们根据汇编判断线上的堆栈的正确性，从而帮助我们高效快速的分析问题。
&emsp;&emsp;本文主要参考《C++反汇编逆向分析计数揭秘》这本书，在Linux环境下探索C++反汇编的代码，更加深入理解C++工程。

**测试环境**：
>Linux Ubuntu 5.15.0-69-generic #76~20.04.1-Ubuntu SMP Mon Mar 20 15:54:19 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
>clang version: Ubuntu clang version 13.0.1-2ubuntu2~20.04

## 2 基本的数据类型
### 2.1 数值类型
**无符号整数**
&emsp;&emsp;无符号数都是直接存储在内存中，唯一需要注意的是不同机器的存储方式不同。
>- 大端：高位存储高位（Windows，Linux）；
>- 小端：高位存储低位（OSX）；

**有符号整数**
&emsp;&emsp;有符号数和无符号数的区别是，有符号数的最高位表示当前数正/负。无符号数和有符号数能够表示的数值范围大小相同，只是具体能够表示的范围不同。如果数值为正数，则代码中的数值和无符号数无区别；如果为负数，则代码中的数值存储是按照补码存储（起始正数也是补码，不过正数的补码是其自身，这样做是为了方便利用加法计算减法）。
>&emsp;&emsp;补码：数的所有位取反+1。

**单精度浮点类型和双精度浮点类型**
&emsp;&emsp;浮点类型在内存中表示是按照[IEEE 754](https://en.wikipedia.org/wiki/IEEE_754)标准存储的，```float```和```double```而这表示方式差不多，只是表示的范围有差异。因为按照IEEE 754存储浮点是由精度误差的，因此在比较浮点类型的大小是不能用```==```，而是```val - val2 < elps```。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/be21f310dc8cb88d6b48734471e394b6.png)


**布尔类型**
&emsp;&emsp;在内存中就是0或者1。

==**实验**==
&emsp;&emsp;参考的C++代码：
```cpp
#include <cstdint>

int main(int argc, char **argv){
    uint32_t a = 12345678;
    uint64_t a1 = 12345678;

    int32_t b = 12345678;
    int64_t b1 = 12345678;
    int64_t b2 = -1245678;
    uint64_t b3 = 12345678912345678;
    uint64_t b4 = 1;

    char c = 'a';
    short d = 12;
    float e = 1.678;
    double f = 1.678;
    bool g = 1;
}
```
&emsp;&emsp;反汇编代码（代码中省略了不需要我们关注的内容）：
>&emsp;&emsp;本文中的反汇编代码中也会省略很多我们不必要关注的内容，避免干扰我们分析问题。

```
0000000000401110 <main>:
  401110:       55                      push   %rbp
  401111:       48 89 e5                mov    %rsp,%rbp
  401114:       89 7d fc                mov    %edi,-0x4(%rbp)
  401117:       48 89 75 f0             mov    %rsi,-0x10(%rbp)
  40111b:       c7 45 ec 4e 61 bc 00    movl   $0xbc614e,-0x14(%rbp)    ;uint32_t a = 12345678;
  401122:       48 c7 45 e0 4e 61 bc    movq   $0xbc614e,-0x20(%rbp)    ;uint64_t a1 = 12345678;
  401129:       00
  40112a:       c7 45 dc 4e 61 bc 00    movl   $0xbc614e,-0x24(%rbp)    ;int32_t b = 12345678;
  401131:       48 c7 45 d0 4e 61 bc    movq   $0xbc614e,-0x30(%rbp)    ;int64_t b1 = 12345678;
  401138:       00
  401139:       48 c7 45 c8 12 fe ec    movq   $0xffffffffffecfe12,-0x38(%rbp)  ;int64_t b2 = -1245678;
  401140:       ff
  401141:       48 b8 4e d6 14 5e 54    movabs $0x2bdc545e14d64e,%rax   
  401148:       dc 2b 00
  40114b:       48 89 45 c0             mov    %rax,-0x40(%rbp)         ;uint64_t b3 = 12345678912345678;
  40114f:       48 c7 45 b8 01 00 00    movq   $0x1,-0x48(%rbp)         ;uint64_t b4 = 1;
  401156:       00
  401157:       c6 45 b7 61             movb   $0x61,-0x49(%rbp)        ;char c = 'a';
  40115b:       66 c7 45 b4 0c 00       movw   $0xc,-0x4c(%rbp)         ;short d = 12;
  401161:       f3 0f 10 05 a7 0e 00    movss  0xea7(%rip),%xmm0        ## 402010 <_IO_stdin_used+0x10>
  401168:       00
  401169:       f3 0f 11 45 b0          movss  %xmm0,-0x50(%rbp)        ;
  40116e:       f2 0f 10 05 92 0e 00    movsd  0xe92(%rip),%xmm0        ## 402008 <_IO_stdin_used+0x8>
  401175:       00
  401176:       f2 0f 11 45 a8          movsd  %xmm0,-0x58(%rbp)
  40117b:       c6 45 a7 01             movb   $0x1,-0x59(%rbp)
  40117f:       31 c0                   xor    %eax,%eax
  401181:       5d                      pop    %rbp
  401182:       c3                      retq
  401183:       66 2e 0f 1f 84 00 00    nopw   %cs:0x0(%rax,%rax,1)
  40118a:       00 00 00
  40118d:       0f 1f 00                nopl   (%rax)

0000000000402000 <_IO_stdin_used>:
  402000:       01 00                   add    %eax,(%rax)
  402002:       02 00                   add    (%rax),%al
  402004:       00 00                   add    %al,(%rax)
  402006:       00 00                   add    %al,(%rax)
  402008:       0c 02                   or     $0x2,%al
  40200a:       2b 87 16 d9 fa 3f       sub    0x3ffad916(%rdi),%eax
  402010:       b4 c8                   mov    $0xc8,%ah
  402012:       d6                      (bad)
  402013:       3f                      (bad)
```

&emsp;&emsp;从上面的反汇编中我们能够看出，```int32_t```的有符号和无符号在内存中存储方式相同都是```0xbc614e```（并且从内存存储方式能够看出第大端存储），而负数是通过补码的方式存储的即```0xFFFF FFFF FFFF FFFF FF43 9EB2```。
&emsp;&emsp;浮点数1.68对应的float为```0x3FD6C8B4```存储在```rodata```段中的```0x402010```，double为```0x3FFAD916872B020C```存储在```rodata```段中的```0x402008```（直接看内存）。我们可以尝试根据```0x3FD6C8B4```反推一下具体在内存中的浮点数的值。```0x3FD6C8B4->0 01111111 10101101100100010110100->0 127 10101101100100010110100```$+1^{127-127} + 2^{-1} + 2^{-3}+2^{-5}+2^{-6}+2^{-8}+2^{-9}+2^{-12}+2^{-16}+2^{-18}+2^{-19}+2^{-21}=+1.6779999732971191$。可以看到实际存储的数值是有精度误差的，这也是为什么浮点不能```==```。
>&emsp;&emsp;另外能够注意到代码中用到的不是一般的通用寄存器而是mmx寄存器。这是因为MMX是在最初的浮点寄存器ST上扩展而来。

### 2.2 字符串
&emsp;&emsp;C++中的字符串是以```\0```为结束符的一段内存。唯一需要注意的是不同编码的字符串每个字符占用的字节数不同，ascii每个字符占1个字节，unicode每个占2个字节等等。
==**实验**==
```cpp
int main(){
    const char *ptr = "hello";
    const wchar_t *p2 = L"hello";
}
```
&emsp;&emsp;C++中字符串是常量因此存储在```rodata```中，反汇编中```68 65 6c 6c 6f```就是```hello```，而```p2```中每个字符占2个字节。
```
0000000000401110 <main>:
    ;...
    401111:       48 89 e5                mov    %rsp,%rbp
    401114:       48 b8 04 20 40 00 00    movabs $0x402004,%rax
    40111b:       00 00 00
    40111e:       48 89 45 f8             mov    %rax,-0x8(%rbp)
    401122:       48 b8 0c 20 40 00 00    movabs $0x40200c,%rax
    ;...

0000000000402000 <_IO_stdin_used>:
    402004:       68 65 6c 6c 6f          pushq  $0x6f6c6c65
    402009:       00 00                   add    %al,(%rax)
    40200b:       00 68 00                add    %ch,0x0(%rax)
    40200e:       00 00                   add    %al,(%rax)
    402010:       65 00 00                add    %al,%gs:(%rax)
    402013:       00 6c 00 00             add    %ch,0x0(%rax,%rax,1)
    402017:       00 6c 00 00             add    %ch,0x0(%rax,%rax,1)
    40201b:       00 6f 00                add    %ch,0x0(%rdi)
    40201e:       00 00                   add    %al,(%rax)
    402020:       00 00                   add    %al,(%rax)
```

### 2.3 地址，指针，引用和常量
**指针**
&emsp;&emsp;地址就是进程地址空间中的一个索引，而指针变量，就是存储一块地址内容的变量。所以其重点是其本身是一个变量，只不过存储的内容是一个地址索引而已。一个指针的大小根据系统位数不同而不同，32bit机器指针大小为4字节，64bit机器指针大小为8字节。而解释这块地址的内容的方式是根据其类型而来的，因此在对指针变量进行```++```等操作时，其结果是根据具体的类型而来的：
```cpp
ptr + n = 当前ptr地址+sizeof(Type) * n
```
**引用**
&emsp;&emsp;引用是变量的别名，其实现和指针基本相同。可以认为，引用和指针是相同的东西（实际使用中也差不多如此），只是编译器实现是隐藏了很多东西。比如引用计算```++```是对原数据的计算，而不是向指针那行对指针变量的操作。

**常量**
&emsp;&emsp;需要注意的是C++中的常量只是语法上的常量，并不是编译期常量也不是运行期常量。编译期常量应该使用```constexpr```，运行期常量在不使用```const_cast```进行强制转换时此语义是可以保证的。

==**实验**==
```cpp
int main(int argc, char **argv){
    int a = 1;
    int *p = &a;
    p+=1;
    char *pc = (char*)p;
    pc+=1;

    int &b = a;
    b = 2;

    const int c = 33;
    b = c;
    b = *p + c;
    return 0;
}
```
&emsp;&emsp;从下面的汇编代码能够看出来，指针和引用的区别就是指针在进行操作时需要使用```*```解引用才能操作对应内存中的值，而引用直接能操作内存中的值。另外```const```值的立即数在预处理时就被替换了，对于一些复杂的场景，```const```在汇编中和普通变量没有区别，不可变是由编译器语法限制的。
```
0000000000401110 <main>:
  401110:       55                      push   %rbp
  401111:       48 89 e5                mov    %rsp,%rbp
  401114:       c7 45 fc 00 00 00 00    movl   $0x0,-0x4(%rbp)
  40111b:       89 7d f8                mov    %edi,-0x8(%rbp)
  40111e:       48 89 75 f0             mov    %rsi,-0x10(%rbp)
  401122:       c7 45 ec 01 00 00 00    movl   $0x1,-0x14(%rbp)     ;a = 1
  401129:       48 8d 45 ec             lea    -0x14(%rbp),%rax
  40112d:       48 89 45 e0             mov    %rax,-0x20(%rbp)     ;int *p = &a;
  401131:       48 8b 45 e0             mov    -0x20(%rbp),%rax
  401135:       48 83 c0 04             add    $0x4,%rax            
  401139:       48 89 45 e0             mov    %rax,-0x20(%rbp)     ;p=+1
  40113d:       48 8b 45 e0             mov    -0x20(%rbp),%rax
  401141:       48 89 45 d8             mov    %rax,-0x28(%rbp)     ;char *pc = (char*)p;
  401145:       48 8b 45 d8             mov    -0x28(%rbp),%rax
  401149:       48 83 c0 01             add    $0x1,%rax
  40114d:       48 89 45 d8             mov    %rax,-0x28(%rbp)     ;pc+=1
  401151:       48 8d 45 ec             lea    -0x14(%rbp),%rax
  401155:       48 89 45 d0             mov    %rax,-0x30(%rbp)     ;int &b = a;
  401159:       48 8b 45 d0             mov    -0x30(%rbp),%rax
  40115d:       c7 00 02 00 00 00       movl   $0x2,(%rax)          ;b = 2;
  401163:       c7 45 cc 21 00 00 00    movl   $0x21,-0x34(%rbp)    ;const int c = 33;
  40116a:       48 8b 45 d0             mov    -0x30(%rbp),%rax   
  40116e:       c7 00 21 00 00 00       movl   $0x21,(%rax)         ;b = c = 33;
  401174:       48 8b 45 e0             mov    -0x20(%rbp),%rax     
  401178:       8b 08                   mov    (%rax),%ecx          ;eax = *p;
  40117a:       83 c1 21                add    $0x21,%ecx
  40117d:       48 8b 45 d0             mov    -0x30(%rbp),%rax
  401181:       89 08                   mov    %ecx,(%rax)          ;b = *p + c
  401183:       31 c0                   xor    %eax,%eax
  401185:       5d                      pop    %rbp
```

## 3 MSVC和gcc的入口函数
&emsp;&emsp;可参考[程序员自我修养阅读笔记——运行库](https://blog.csdn.net/GrayOnDream/article/details/122799550)

## 4 表达式求值
### 4.1 算法表达式
**加法,减法和乘法，自增自减**
&emsp;&emsp;加法，减法和乘法都有对应的计算指令，比如```add,sub,mul,imul```实现，比较简单。需要注意的是乘法计算中如果乘以的是2的幂数，则编译器会将对应的计算优化为位移运算。自增和自减也比较简单，主要关注后置自增和自减，实际上实现是把表达式拆分了。

==**实验一**==

```cpp
#include <cstdio>
int main(){
    int a = 4096;
    int b = 3;
    int c = 1;
    c = a + b + (1 + 2);
    a = a - ++b;
    b = a * b++ * 4;
    return 0;
}
```
&emsp;&emsp;下面的反汇编比较简单，能够看到都是使用对应的指令实现的。乘法中对于2幂次数利用左移进行了优化。
```
0000000000401110 <main>:
  401110:       55                      push   %rbp
  401111:       48 89 e5                mov    %rsp,%rbp
  401114:       c7 45 fc 00 00 00 00    movl   $0x0,-0x4(%rbp)
  40111b:       c7 45 f8 00 10 00 00    movl   $0x1000,-0x8(%rbp)   ;a = 4096
  401122:       c7 45 f4 03 00 00 00    movl   $0x3,-0xc(%rbp)      ;b = 3
  401129:       c7 45 f0 01 00 00 00    movl   $0x1,-0x10(%rbp)     ;c = 1
  401130:       8b 45 f8                mov    -0x8(%rbp),%eax
  401133:       03 45 f4                add    -0xc(%rbp),%eax      ;a + b
  401136:       83 c0 03                add    $0x3,%eax            ;a + b + 3
  401139:       89 45 f0                mov    %eax,-0x10(%rbp)
  40113c:       8b 45 f8                mov    -0x8(%rbp),%eax
  40113f:       8b 4d f4                mov    -0xc(%rbp),%ecx
  401142:       83 c1 01                add    $0x1,%ecx            ;++b
  401145:       89 4d f4                mov    %ecx,-0xc(%rbp)
  401148:       29 c8                   sub    %ecx,%eax            ;a - ++b
  40114a:       89 45 f8                mov    %eax,-0x8(%rbp)
  40114d:       8b 45 f8                mov    -0x8(%rbp),%eax
  401150:       8b 4d f4                mov    -0xc(%rbp),%ecx
  401153:       89 ca                   mov    %ecx,%edx
  401155:       83 c2 01                add    $0x1,%edx            ;b++
  401158:       89 55 f4                mov    %edx,-0xc(%rbp)
  40115b:       0f af c1                imul   %ecx,%eax            ;a * b
  40115e:       c1 e0 02                shl    $0x2,%eax            ;a * b * 4
  401161:       89 45 f4                mov    %eax,-0xc(%rbp)
  401164:       31 c0                   xor    %eax,%eax
  401166:       5d                      pop    %rbp
  401167:       c3                      retq
```

**除法**
&emsp;&emsp;除法也有对应的汇编指令```div```和```idiv```实现，但是除法相对于其他操作指令周期更长，效率更低，因此编译器会尽可能尝试用其他指令优化当前的除法操作。
&emsp;&emsp;C++中的整数除法的结果依然是整数，其结果是向0求整（比如3.5求整为3，-3.5求整为-3）。编译器会对不同的除数进行不同的优化：
1. 除数为2的幂次。编译器会对除数是2的幂次的数利用右移指令优化；
2. 除数为无符号非2次幂。编译器会通过下面的公式将除法转换为乘法，因为```n```是固定的（32或者64）因此可以在与编译期计算出来；
$$
\frac{x}{c}=x\frac{2^n}{c}\frac{1}{2^n}
$$
3. 有符号整数相比无符号多了一个位表示富奥因此，其n的取值为```n-1```。

==**实验二**==
```cpp
#include <cstdio>

int main(){
    int a = 4096;
    scanf("%d", &a);
    int b = (unsigned int)a / (unsigned int)3;
    int c = 1;
    printf("%d %d", b, c);
    b = (int)a / (int)3;
    c = 2;
    printf("%d %d", b, c);
    return 0;
}
```
&emsp;&emsp;上面的scanf语句是为了避免编译器过度优化。
&emsp;&emsp;无符号除法，我们可以从上面的公式中推导出具体的值，$M=\frac{2^n}{c}\rightarrow c=\frac{2^n}{M}=\frac{2^32}{0xaaaaaaab}=2.9999999996507540345598531381041\approx 3$
&emsp;&emsp;有符号整数的触发的魔数刚好为无符号的一半。

```
0000000000401138 <main>:
  401138:       53                      push   %rbx
  401139:       48 83 ec 10             sub    $0x10,%rsp
  40113d:       48 8d 5c 24 0c          lea    0xc(%rsp),%rbx
  401142:       c7 03 00 10 00 00       movl   $0x1000,(%rbx)
  401148:       bf 07 20 40 00          mov    $0x402007,%edi
  40114d:       48 89 de                mov    %rbx,%rsi
  401150:       31 c0                   xor    %eax,%eax
  401152:       e8 e9 fe ff ff          callq  401040 <__isoc99_scanf@plt>
  401157:       8b 03                   mov    (%rbx),%eax
  401159:       be ab aa aa aa          mov    $0xaaaaaaab,%esi
  40115e:       48 0f af f0             imul   %rax,%rsi
  401162:       48 c1 ee 21             shr    $0x21,%rsi         ;(unsigned)a / (unsigned)3; 
  401166:       bf 04 20 40 00          mov    $0x402004,%edi
  40116b:       ba 01 00 00 00          mov    $0x1,%edx
  401170:       31 c0                   xor    %eax,%eax
  401172:       e8 b9 fe ff ff          callq  401030 <printf@plt>
  401177:       48 63 03                movslq (%rbx),%rax
  40117a:       48 69 f0 56 55 55 55    imul   $0x55555556,%rax,%rsi
  401181:       48 89 f0                mov    %rsi,%rax
  401184:       48 c1 e8 3f             shr    $0x3f,%rax
  401188:       48 c1 ee 20             shr    $0x20,%rsi         ;(int)a / (int)3
  40118c:       01 c6                   add    %eax,%esi
  40118e:       bf 04 20 40 00          mov    $0x402004,%edi
  401193:       ba 02 00 00 00          mov    $0x2,%edx
  401198:       31 c0                   xor    %eax,%eax
  40119a:       e8 91 fe ff ff          callq  401030 <printf@plt>
  40119f:       31 c0                   xor    %eax,%eax
  4011a1:       48 83 c4 10             add    $0x10,%rsp
  4011a5:       5b                      pop    %rbx
  4011a6:       c3                      retq
```

**取余**
&emsp;&emsp;对2的k次方取余的方式有很多：
1. 直接通过```k```位的亦或运算计算；
2. 通过公式$x\% 2^k=(x+(2^k - 1) \& (2^k - 1)) - (2^k - 1)$
3. 方案三：
   1. 正数：$x\% 2^k = x - (x\&!(2^k-1))$
   2. 负数：$x\% 2^k = x - (x + (2^k - 1) \& !(2^k - 1))$

&emsp;&emsp;非2次幂，则$x\% a = x - \frac{a}{b} * b$。

**==实验三==**
```cpp
#include <cstdio>
int main(){
    int a = 3;
    scanf("%d", &a);
    int b = a % 4;
    scanf("%d", &a);
    int d = a % 3;
    printf("%d %d %d %d", a, b, c, d);
}
```

```
0000000000401138 <main>:
  401138:       53                      push   %rbx
  401139:       48 83 ec 10             sub    $0x10,%rsp
  40113d:       48 8d 5c 24 0c          lea    0xc(%rsp),%rbx
  401142:       c7 03 03 00 00 00       movl   $0x3,(%rbx)
  401148:       bf 0d 20 40 00          mov    $0x40200d,%edi
  40114d:       48 89 de                mov    %rbx,%rsi
  401150:       31 c0                   xor    %eax,%eax
  401152:       e8 e9 fe ff ff          callq  401040 <__isoc99_scanf@plt>
  401157:       4c 63 03                movslq (%rbx),%r8
  40115a:       41 8d 40 03             lea    0x3(%r8),%eax
  40115e:       45 85 c0                test   %r8d,%r8d
  401161:       41 0f 49 c0             cmovns %r8d,%eax
  401165:       83 e0 fc                and    $0xfffffffc,%eax
  401168:       44 89 c2                mov    %r8d,%edx
  40116b:       29 c2                   sub    %eax,%edx
  40116d:       49 69 c0 56 55 55 55    imul   $0x55555556,%r8,%rax
  401174:       48 89 c1                mov    %rax,%rcx
  401177:       48 c1 e9 3f             shr    $0x3f,%rcx
  40117b:       48 c1 e8 20             shr    $0x20,%rax       ;这里明显为一个除法 a / 3
  40117f:       01 c8                   add    %ecx,%eax
  401181:       8d 04 40                lea    (%rax,%rax,2),%eax
  401184:       41 29 c0                sub    %eax,%r8d
  401187:       c7 03 01 00 00 00       movl   $0x1,(%rbx)
  40118d:       bf 04 20 40 00          mov    $0x402004,%edi
  401192:       be 01 00 00 00          mov    $0x1,%esi
  401197:       b9 02 00 00 00          mov    $0x2,%ecx
  40119c:       31 c0                   xor    %eax,%eax
  40119e:       e8 8d fe ff ff          callq  401030 <printf@plt>
  4011a3:       31 c0                   xor    %eax,%eax
  4011a5:       48 83 c4 10             add    $0x10,%rsp
  4011a9:       5b                      pop    %rbx
  4011aa:       c3                      retq
```

### 4.2 关系运算、位运算和逻辑运算
&emsp;&emsp;C++中关系运算符是计算一个表达式的结果然后配合```cmp```和```test```检查表达式结果，最终利用jmp相关指令实现。关系运算中的表达式短路也是类似的逻辑，如果前半部分的表达式满足就不会再计算后半部分，即便后半部分会引发严重问题因为不运行所以没有任何问题。
&emsp;&emsp;条件表达式的实现和```if```语句类似都是通过跳转实现，下面是一个简单的例子。位运算比较简单，每个位运算都有对应的汇编指令，具体不详细说了。

**==实验==**
&emsp;&emsp;
```cpp
int func(int c){
    c && (c += func(c - 1));
    return c;
}

int func2(int c){
    return c && func2(c - 1);
}
```

&emsp;&emsp;上面短路表达式和逻辑表达式都是通过```je```等跳转指令实现的，都是先计算前半部分的值然后判断，其代码等价于：
```cpp
  //num && func(num - 1)
  if(num){ func(num - 1); }
```

```
0000000000000000 <_Z4funci>:
   0:   55                      push   %rbp
   1:   48 89 e5                mov    %rsp,%rbp
   4:   48 83 ec 10             sub    $0x10,%rsp
   8:   89 7d fc                mov    %edi,-0x4(%rbp)      ;c
   b:   83 7d fc 00             cmpl   $0x0,-0x4(%rbp)      ;if(c == 0)
   f:   0f 84 17 00 00 00       je     2c <_Z4funci+0x2c>   ;jump to func + 2c
  15:   8b 7d fc                mov    -0x4(%rbp),%edi      ;
  18:   83 ef 01                sub    $0x1,%edi            ;c - 1
  1b:   e8 00 00 00 00          callq  20 <_Z4funci+0x20>   ;func(c - 1)
  20:   03 45 fc                add    -0x4(%rbp),%eax      ;c + 返回值
  23:   89 45 fc                mov    %eax,-0x4(%rbp)      ;c += 返回值
  26:   83 f8 00                cmp    $0x0,%eax
  29:   0f 95 c0                setne  %al
  2c:   8b 45 fc                mov    -0x4(%rbp),%eax
  2f:   48 83 c4 10             add    $0x10,%rsp
  33:   5d                      pop    %rbp
  34:   c3                      retq
  35:   66 2e 0f 1f 84 00 00    nopw   %cs:0x0(%rax,%rax,1)
  3c:   00 00 00
  3f:   90                      nop

0000000000000040 <_Z5func2i>:
  40:   55                      push   %rbp
  41:   48 89 e5                mov    %rsp,%rbp
  44:   48 83 ec 10             sub    $0x10,%rsp
  48:   89 7d fc                mov    %edi,-0x4(%rbp)      ;c
  4b:   31 c0                   xor    %eax,%eax
  4d:   83 7d fc 00             cmpl   $0x0,-0x4(%rbp)      ;if(c == 0)
  51:   88 45 fb                mov    %al,-0x5(%rbp)
  54:   0f 84 14 00 00 00       je     6e <_Z5func2i+0x2e>  ; jump to func2 + 2e
  5a:   8b 7d fc                mov    -0x4(%rbp),%edi
  5d:   83 ef 01                sub    $0x1,%edi            ;c - 1
  60:   e8 00 00 00 00          callq  65 <_Z5func2i+0x25>  ;func2(c - 1)
  65:   83 f8 00                cmp    $0x0,%eax
  68:   0f 95 c0                setne  %al
  6b:   88 45 fb                mov    %al,-0x5(%rbp)
  6e:   8a 45 fb                mov    -0x5(%rbp),%al
  71:   24 01                   and    $0x1,%al            ; c && func2(c - 1) 
  73:   0f b6 c0                movzbl %al,%eax
  76:   48 83 c4 10             add    $0x10,%rsp
  7a:   5d                      pop    %rbp
  7b:   c3                      retq
```

&emsp;&emsp;
**==实验二==**
```cpp
int func1(int c){
    return c == 3 ? 2 : 1;
}

int func2(int c){
    return c == 3 ? 2 : 5;
}

int func3(int c){
    return c == 3 ? 2 : c + 1;
}

int func4(int c){
    return c / 2 == 1 ? func4(c * 2): func4(c - 1);
}
```

&emsp;&emsp;书上将条件表达式分为了四种情况，但是从下面的反汇编中能够看到只有两种情况：
1. 如果条件表达式的两边都是常数，则将对应的两边的常数计算出来后放入寄存器，然后根据比较的变量的值判断是否需要将返回值的寄存器的值覆盖。这样的反汇编代码无分支的；
2. 如果两边的表达式中有一个是有变量的表达式，则是利用jmp指令来实现。

&emsp;&emsp;下面的反汇编注释的比较清楚，唯一需要注意的是有变量情况下创建的临时变量。
```
0000000000000000 <_Z5func1i>:
   0:   55                      push   %rbp
   1:   48 89 e5                mov    %rsp,%rbp
   4:   89 7d fc                mov    %edi,-0x4(%rbp)
   7:   8b 55 fc                mov    -0x4(%rbp),%edx  ;edx = c
   a:   b8 01 00 00 00          mov    $0x1,%eax        ;eax = 1
   f:   b9 02 00 00 00          mov    $0x2,%ecx        ;ecx = 2
  14:   83 fa 03                cmp    $0x3,%edx        ;c =? 3 
  17:   0f 44 c1                cmove  %ecx,%eax        ;eax = ecx
  1a:   5d                      pop    %rbp
  1b:   c3                      retq
  1c:   0f 1f 40 00             nopl   0x0(%rax)

0000000000000020 <_Z5func2i>:
  20:   55                      push   %rbp
  21:   48 89 e5                mov    %rsp,%rbp
  24:   89 7d fc                mov    %edi,-0x4(%rbp)
  27:   8b 55 fc                mov    -0x4(%rbp),%edx  ;edx = c
  2a:   b8 05 00 00 00          mov    $0x5,%eax        ;eax = 5
  2f:   b9 02 00 00 00          mov    $0x2,%ecx        ;ecx = 2
  34:   83 fa 03                cmp    $0x3,%edx        ;c =? 3
  37:   0f 44 c1                cmove  %ecx,%eax        ;eax = ecx
  3a:   5d                      pop    %rbp
  3b:   c3                      retq
  3c:   0f 1f 40 00             nopl   0x0(%rax)

0000000000000040 <_Z5func3i>:
  40:   55                      push   %rbp
  41:   48 89 e5                mov    %rsp,%rbp
  44:   89 7d fc                mov    %edi,-0x4(%rbp)
  47:   83 7d fc 03             cmpl   $0x3,-0x4(%rbp)        ;c =? 3
  4b:   0f 85 0d 00 00 00       jne    5e <_Z5func3i+0x1e>    ;jump to 5e
  51:   b8 02 00 00 00          mov    $0x2,%eax              ;eax = 2
  56:   89 45 f8                mov    %eax,-0x8(%rbp)        ;tmp = eax
  59:   e9 09 00 00 00          jmpq   67 <_Z5func3i+0x27>
  5e:   8b 45 fc                mov    -0x4(%rbp),%eax        ;eax = c
  61:   83 c0 01                add    $0x1,%eax
  64:   89 45 f8                mov    %eax,-0x8(%rbp)        ;tmp = c + 1
  67:   8b 45 f8                mov    -0x8(%rbp),%eax        ;eax = tmp
  6a:   5d                      pop    %rbp
  6b:   c3                      retq
  6c:   0f 1f 40 00             nopl   0x0(%rax)

0000000000000070 <_Z5func4i>:
  70:   55                      push   %rbp
  71:   48 89 e5                mov    %rsp,%rbp
  74:   89 7d fc                mov    %edi,-0x4(%rbp)
  77:   83 7d fc 03             cmpl   $0x3,-0x4(%rbp)      ;c =? 3
  7b:   0f 85 0e 00 00 00       jne    8f <_Z5func4i+0x1f>  ;jump to 8f
  81:   8b 45 fc                mov    -0x4(%rbp),%eax      ;eax = c
  84:   83 c0 02                add    $0x2,%eax  
  87:   89 45 f8                mov    %eax,-0x8(%rbp)      ;tmp = c + 2
  8a:   e9 09 00 00 00          jmpq   98 <_Z5func4i+0x28>
  8f:   8b 45 fc                mov    -0x4(%rbp),%eax      ;eax = c
  92:   c1 e0 01                shl    $0x1,%eax            ;c * 2
  95:   89 45 f8                mov    %eax,-0x8(%rbp)      ;tmp = c * 2
  98:   8b 45 f8                mov    -0x8(%rbp),%eax      ;eax = tmp
  9b:   5d                      pop    %rbp
  9c:   c3                      retq
```

&emsp;&emsp;另外上面是无优化的代码，当我们开启优化选项为```-Os```时就会出现下面的情况。可以看到代码大大简化了：
1. 当两边都是常量且差为1时，编译期会利用++指令来优化代码；
2. 当两边差大于1时，会利用add指令来替代cmov指令；
3. 当两边有变量且表达式简单时，编译器不使用```jmp```指令而是尝试cmov指令优化；
4. 但是当两边的表达式比较复杂时，编译器就会使用jmp指令实现。

```
0000000000000000 <_Z5func1i>:
   0:   31 c0                   xor    %eax,%eax
   2:   83 ff 03                cmp    $0x3,%edi
   5:   0f 94 c0                sete   %al
   8:   ff c0                   inc    %eax
   a:   c3                      retq

000000000000000b <_Z5func2i>:
   b:   31 c0                   xor    %eax,%eax
   d:   83 ff 03                cmp    $0x3,%edi
  10:   0f 95 c0                setne  %al
  13:   8d 04 40                lea    (%rax,%rax,2),%eax
  16:   83 c0 02                add    $0x2,%eax
  19:   c3                      retq

000000000000001a <_Z5func3i>:
  1a:   8d 4f 01                lea    0x1(%rdi),%ecx
  1d:   83 ff 03                cmp    $0x3,%edi
  20:   b8 02 00 00 00          mov    $0x2,%eax
  25:   0f 45 c1                cmovne %ecx,%eax
  28:   c3                      retq

Disassembly of section .comment:

0000000000000000 <.comment>:
   0:   00 55 62                add    %dl,0x62(%rbp)
   3:   75 6e                   jne    73 <_Z5func4i+0x4a>
   5:   74 75                   je     7c <_Z5func4i+0x53>
   7:   20 63 6c                and    %ah,0x6c(%rbx)
   a:   61                      (bad)
   b:   6e                      outsb  %ds:(%rsi),(%dx)
   c:   67 20 76 65             and    %dh,0x65(%esi)
  10:   72 73                   jb     85 <_Z5func4i+0x5c>
  12:   69 6f 6e 20 31 33 2e    imul   $0x2e333120,0x6e(%rdi),%ebp
  19:   30 2e                   xor    %ch,(%rsi)
  1b:   31 2d 32 75 62 75       xor    %ebp,0x75627532(%rip)        ## 75627553 <_Z5func4i+0x7562752a>
  21:   6e                      outsb  %ds:(%rsi),(%dx)
  22:   74 75                   je     99 <_Z5func4i+0x70>
  24:   32 7e 32                xor    0x32(%rsi),%bh
  27:   30 2e                   xor    %ch,(%rsi)
  29:   30 34 2e                xor    %dh,(%rsi,%rbp,1)
  2c:   31 00                   xor    %eax,(%rax)
```
### 4.3 编译器优化 
>&emsp;&emsp;代码优化：利用现有的资源对已有的程序进行实现上的改变以达到优化内存/执行速度等上的目的。代码优化前后的程序行为是等价的，即优化行为只影响执行速度等指标，不影响实际结果。

&emsp;&emsp;编译器将一个程序编译成二进制文件的过程是：预处理->词法分析->语法分析->语义分析->中间代码生成->目标代码生成（后端实现中省略了部分节点，比如指令重拍，寄存器选择等）。这里只会记录书中记录的部分执行速度优化的一部分内容(Intel处理器为参考)。

#### 4.3.1 常见的优化
- **常量折叠**：比如```x = 2 * 3;```这种代码，编译器可以在编译期完成计算直接生成```x = 6```相关代码；
- **常量传播**:依赖其他变量可以在编译期完成计算的代码都会生成直接的代码，比如```y=x + 3;```会生成```y = 9```;
- **减少变量**:对于一些临时变量有没有并不影响程序的执行，比如```x=2*i,y=2*j;if(x > y){}```和```if(i>j)```等价；
- **公共表达式**：多个变量依赖相同的表达式，比如```x=2 * i, y= 2 * i;```等价于```x=2*i,y=x```
- **复写传播**：类似常量传播，比如```x=a,y=x+c```等价于```y=a+c```；
- **剪枝**：比如```if(1 < 2)```这种永远不会执行的代码会被删除；
- **顺序语句替代分支**：比如上面的三元表达式编译器会尽量使用```cmov```来优化；
- **强度削弱**：用加法或移位替代乘法，用乘法替代除法；
- **数学变换**：将一些简单的数学表达式替换为对应相同结果的表达式，比如```x=a * y + b * y```等价于```x=(x + b) * y```；
- **代码外提**：比如```while(func())```如果```func()```的结果是固定值，编译器就会尝试```t=func();while(t)```，而不用每次循环都执行```func()```；
- 
#### 4.3.2 流水线优化
&emsp;&emsp;一条指令被CPU执行会经历：取指、译码、访存（访存分为读地址和取值两部分）、执行、写回（将计算的结果写回到对应地址或者寄存器）。其中取指、译码、执行是每个指令都需要的，比如nop指令。流水线就是将上述的多个步骤错开执行提升CPU的指令运行吞吐率。比如在执行第一条的译码工作时就可以尝试读取下一条指令准备译码。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/41cbfd18bf778be84444e5e3db2799b8.png)

&emsp;&emsp;指令流水线的实现有两种方式：
1. 长指令：比如Intel的CSIC架构，每个指令划分更多的阶段单个指令更长，每个步骤的工作简单更容易设计，但是在出现跳转指令且分支预测失败是失败的成本也高；
2. 精简指令：比如Arm的RSIC架构，每个指令更短，流水线数量更多，分支预测带来的错误成本更低，但是电路更复杂，流水线的管理成本更高。

&emsp;&emsp;影响流水线并行度的几个因素：
1. 指令相关。即数据依赖，后一条指令的运行依赖前一条指令的结果，就会形成指令依赖，影响并行效率（现如今编译器都会进行指令重排来提升并行度）；
2. 地址相关。前一条指令需要读取一块儿内存而后一条指令需要写入这块儿内存，就会形成地址依赖，影响并行度；
3. 分支预测。CPU中会有分支预测器对分支指令进行预测，预测下一步将会执行哪个指令以提高指令并行度。因此写代码时尽量避免分支代码；
4. 高速缓存优化。现代CPU都有多级高速缓存（L1，L2，L3）将需要处理的数据集中存放可以有效利用CPU的缓存策略提升性能。
  >数据局对也会影响性能，对齐后的数据能够一次定位到，而没有对齐的数据可能需要多次访问。

## 5 流程控制
### 5.1 if语句
&emsp;&emsp;流程控制是程序的重要组成部分，而流程控制是通过```if```语句实现的。汇编中是通过```jmp``来实现```if```语句的。但是针对不同的控制语句编译器会采用不同的优化策略来加速程序的执行。下面就从简到难逐步看不同控制结构的实际实现。
**if**
&emsp;&emsp;```if```是使用jmp实现，那为了符合```if```的语义，应该是满足条件则继续执行不满足则跳转。所以从汇编中看到的是和```if```相反的判断。对于一个```if```语句中一个表达式```expr```是否为真，实际反汇编中应该是判断```!expr```满足则跳转。

**==实验一==**
```cpp
int func(int a){
    if(a == 3){
        printf("the value is 3");   //printf是为了防止代码被优化，如果是无用代码会被编译器删除
    }

    return 0;
}
```

```
0000000000000000 <_Z4funci>:
   0:   83 ff 03                cmp    $0x3,%edi
   3:   75 11                   jne    16 <_Z4funci+0x16>
   5:   50                      push   %rax
   6:   bf 00 00 00 00          mov    $0x0,%edi
   b:   31 c0                   xor    %eax,%eax
   d:   e8 00 00 00 00          callq  12 <_Z4funci+0x12>
  12:   48 83 c4 08             add    $0x8,%rsp
  16:   31 c0                   xor    %eax,%eax
  18:   c3                      retq
```
**if...else**
&emsp;&emsp;```if...else```相比于```if```多了要执行的场景。因为代码是顺序的，为了实现```else``在原来```if```的基础上需要两条```jmp```指令。从下面的实验中能够看到```if...else...```是通过两条```jmp```指令实现：当输入符合条件时顺序执行对应的代码块，之后调用```jmp```到结尾；当输入不符合条件时直接跳转到```else```对应的代码块执行。当开始代码优化时，编译器会对条件语句进行优化，这里能够看到优化的结果和条件跳转类似。

**==实验二==**
```cpp
int ifelse(int a){
    if(a == 3){
        printf("the value is 3");
    }else{
        printf("the value is not 3");
    }
    return 0;
}
```

```
;;clang -O0
0000000000000030 <_Z6ifelsei>:
  30:   55                      push   %rbp
  31:   48 89 e5                mov    %rsp,%rbp
  34:   48 83 ec 10             sub    $0x10,%rsp
  38:   89 7d fc                mov    %edi,-0x4(%rbp)
  3b:   83 7d fc 03             cmpl   $0x3,-0x4(%rbp)
  3f:   0f 85 16 00 00 00       jne    5b <_Z6ifelsei+0x2b>   ;;if(argc != 3) goto else
  45:   48 bf 00 00 00 00 00    movabs $0x0,%rdi
  4c:   00 00 00
  4f:   b0 00                   mov    $0x0,%al
  51:   e8 00 00 00 00          callq  56 <_Z6ifelsei+0x26>   ;; call printf
  56:   e9 11 00 00 00          jmpq   6c <_Z6ifelsei+0x3c>   ;; goto end
  5b:   48 bf 00 00 00 00 00    movabs $0x0,%rdi              ;;else
  62:   00 00 00
  65:   b0 00                   mov    $0x0,%al
  67:   e8 00 00 00 00          callq  6c <_Z6ifelsei+0x3c>   ;; call printf
  6c:   31 c0                   xor    %eax,%eax
  6e:   48 83 c4 10             add    $0x10,%rsp
  72:   5d                      pop    %rbp
  73:   c3                      retq

;;clang -Os
0000000000000019 <_Z6ifelsei>:
  19:   50                      push   %rax
  1a:   83 ff 03                cmp    $0x3,%edi
  1d:   b8 00 00 00 00          mov    $0x0,%eax
  22:   bf 00 00 00 00          mov    $0x0,%edi
  27:   48 0f 44 f8             cmove  %rax,%rdi          ;条件移动
  2b:   31 c0                   xor    %eax,%eax
  2d:   e8 00 00 00 00          callq  32 <_Z6ifelsei+0x19>
  32:   31 c0                   xor    %eax,%eax
  34:   59                      pop    %rcx
  35:   c3                      retq
```
**if...else if...else**

&emsp;&emsp;```if...else if...else```和```if...else```的情况类似，只不过编译器会针对不同的情况优化。比如下面的代码```jmp```只是用来设置```printf```的输入参数，而不是有多个```printf```的call指令。

```cpp
int ifelseif(int a){
    if(a == 3){
        printf("the value is 3");
    }
    else if(a == 10){
        printf("the value is 10");
    }
    else if(a == 100){
        printf("the value is 100");
    }else{
        printf("the value is not a number");
    }

    return 0;
}
```

```
0000000000000000 <_Z8ifelseifi>:
   0:   83 ff 03                cmp    $0x3,%edi
   3:   74 11                   je     16 <_Z8ifelseifi+0x16>         ;a == 3 ? goto 16
   5:   83 ff 64                cmp    $0x64,%edi
   8:   74 13                   je     1d <_Z8ifelseifi+0x1d>         ;a == 100? goto 1d
   a:   83 ff 0a                cmp    $0xa,%edi
   d:   75 15                   jne    24 <_Z8ifelseifi+0x24>         ;a != 10 ? goto 24
   f:   bf 00 00 00 00          mov    $0x0,%edi
  14:   eb 13                   jmp    29 <_Z8ifelseifi+0x29>         ;goto printf
  16:   bf 00 00 00 00          mov    $0x0,%edi
  1b:   eb 0c                   jmp    29 <_Z8ifelseifi+0x29>         ;goto printf
  1d:   bf 00 00 00 00          mov    $0x0,%edi
  22:   eb 05                   jmp    29 <_Z8ifelseifi+0x29>         ;goto printf
  24:   bf 00 00 00 00          mov    $0x0,%edi
  29:   50                      push   %rax
  2a:   31 c0                   xor    %eax,%eax
  2c:   e8 00 00 00 00          callq  31 <_Z8ifelseifi+0x31>         ;call printf
  31:   31 c0                   xor    %eax,%eax
  33:   59                      pop    %rcx
  34:   c3                      retq
```

### 5.2 switch语句
&emsp;&emsp;```switch```语句是一种多分支结构，其结构相比于```if```更加复杂。编译器针对不同类型的```switch```语句进行针对性的优化：
1. 当```switch```语句中的选项数量小于等于3时，会生成类似```if...elseif```的代码；
2. 当```switch```语句中的选项数量大于3且选项之间有明显的的线性关系时，编译器会生成一个跳转表来表示```switch```；
3. 当```switch```语句中的选项数量大于3小于256且无法构成明显的线性关系时，编译器会生成两个表格，第一个表格存储跳转表的索引，第二个表格存储跳转表。程序执行时先通过当前值索引第一个表格得到跳转表索引，再通过该索引找到跳转地址（实测发现```clang```的优化策略和```cl```优化策略不同）；
4. 当```switch```语句中的选项数量大于255时，编译器会使用判定树来优化。即以每个选项的值作为树的节点来生成判定分支代码。

**==实验一：switch语句跳转少于等于3==**
```cpp
int switch1(int type){
    switch(type){
    case 1:
        printf("the value is 1");break;
    case 2:
        printf("the value is 2");break;
    case 3:
        printf("the value is 3");break;
    default:
        printf("unknown value");
    }
    return 0;
}
```

&emsp;&emsp;从下面的反汇编能够简单的看出，当条件小于3时生成的汇编的代码和```if...else if```有些类似（只是类似，还是有区别的）。上半部分，即4011b3以上的部分汇编是计算当前的值和```switch...case```的选项的比较，然后跳转到对应的代码块，下半部分就是具体的代码块儿。

```
//-O0
0000000000401130 <_Z7switch1i>:
  401130:       55                      push   %rbp
  401131:       48 89 e5                mov    %rsp,%rbp
  401134:       48 83 ec 10             sub    $0x10,%rsp
  401138:       89 7d fc                mov    %edi,-0x4(%rbp)
  40113b:       8b 45 fc                mov    -0x4(%rbp),%eax
  40113e:       89 45 f8                mov    %eax,-0x8(%rbp)
  401141:       83 e8 01                sub    $0x1,%eax                  
  401144:       0f 84 27 00 00 00       je     401171 <_Z7switch1i+0x41>    ;eax ?= 1 jump
  40114a:       e9 00 00 00 00          jmpq   40114f <_Z7switch1i+0x1f>
  40114f:       8b 45 f8                mov    -0x8(%rbp),%eax
  401152:       83 e8 02                sub    $0x2,%eax
  401155:       0f 84 2c 00 00 00       je     401187 <_Z7switch1i+0x57>    ;eax ?= 2 jump
  40115b:       e9 00 00 00 00          jmpq   401160 <_Z7switch1i+0x30>
  401160:       8b 45 f8                mov    -0x8(%rbp),%eax
  401163:       83 e8 03                sub    $0x3,%eax
  401166:       0f 84 31 00 00 00       je     40119d <_Z7switch1i+0x6d>    ;eax ?= 3 jump
  40116c:       e9 42 00 00 00          jmpq   4011b3 <_Z7switch1i+0x83>
  401171:       48 bf 04 20 40 00 00    movabs $0x402004,%rdi
  401178:       00 00 00
  40117b:       b0 00                   mov    $0x0,%al
  40117d:       e8 ae fe ff ff          callq  401030 <printf@plt>          ;printf the value is 1
  401182:       e9 3d 00 00 00          jmpq   4011c4 <_Z7switch1i+0x94>
  401187:       48 bf 13 20 40 00 00    movabs $0x402013,%rdi
  40118e:       00 00 00
  401191:       b0 00                   mov    $0x0,%al
  401193:       e8 98 fe ff ff          callq  401030 <printf@plt>          ;printf the value is 2
  401198:       e9 27 00 00 00          jmpq   4011c4 <_Z7switch1i+0x94>
  40119d:       48 bf 22 20 40 00 00    movabs $0x402022,%rdi
  4011a4:       00 00 00
  4011a7:       b0 00                   mov    $0x0,%al
  4011a9:       e8 82 fe ff ff          callq  401030 <printf@plt>          ;printf the value is 3
  4011ae:       e9 11 00 00 00          jmpq   4011c4 <_Z7switch1i+0x94>
  4011b3:       48 bf 31 20 40 00 00    movabs $0x402031,%rdi
  4011ba:       00 00 00
  4011bd:       b0 00                   mov    $0x0,%al 
  4011bf:       e8 6c fe ff ff          callq  401030 <printf@plt>          ;printf the value is unknwon
  4011c4:       31 c0                   xor    %eax,%eax                    ;函数准备返回
  4011c6:       48 83 c4 10             add    $0x10,%rsp
  4011ca:       5d                      pop    %rbp
  4011cb:       c3                      retq
  4011cc:       0f 1f 40 00             nopl   0x0(%rax)
```

**==实验二：switch语句大于3且选项有限性关系==**
```cpp
int switch2(int type){
    switch(type){
    case 1:
        printf("the value is 1");break;
    case 2:
        printf("the value is 2");break;
    case 3:
        printf("the value is 3");break;
    case 4:
        printf("the value is 3");break;
    default:
        printf("unknown value");
    }

    return 0;
}
```

&emsp;&emsp;我们一步一步分析下面的反汇编：
1. 首先```type```减去选项的最小值并将这个值存到栈上一个临时变量中，然后和选项最大值和最小值只差作差；
2. 当差值大于0时则跳转到```default```的代码块，否则继续执行；
3. 之后利用第1步存储的临时变量作为索引在跳转表格中找打实际的跳转地址，直接跳转到对应地址执行；
4. 跳转到对应的地址执行完对应的代码之后，跳转到```switch```的结尾处。

&emsp;&emsp;跳转表存储在```_IO_stdin_used```地址，从对应的地址中我们能够看到```0x004011fe,0x00401214,0x0040122a,0x00401240```四个地址，刚好对应每个选项的代码块的起始地址。

```
;-O0
00000000004011d0 <_Z7switch2i>:
  4011d0:       55                      push   %rbp
  4011d1:       48 89 e5                mov    %rsp,%rbp
  4011d4:       48 83 ec 10             sub    $0x10,%rsp
  4011d8:       89 7d fc                mov    %edi,-0x4(%rbp)
  4011db:       8b 45 fc                mov    -0x4(%rbp),%eax
  4011de:       83 c0 ff                add    $0xffffffff,%eax           ;type - 1
  4011e1:       89 c1                   mov    %eax,%ecx                
  4011e3:       48 89 4d f0             mov    %rcx,-0x10(%rbp)           ;tmp = type - 1
  4011e7:       83 e8 03                sub    $0x3,%eax                  
  4011ea:       0f 87 66 00 00 00       ja     401256 <_Z7switch2i+0x86>  ;if type > 4 then jump
  4011f0:       48 8b 45 f0             mov    -0x10(%rbp),%rax
  4011f4:       48 8b 04 c5 08 20 40    mov    0x402008(,%rax,8),%rax     ;从表格找中根据索引获取需要跳转的地址
  4011fb:       00
  4011fc:       ff e0                   jmpq   *%rax
  4011fe:       48 bf 28 20 40 00 00    movabs $0x402028,%rdi             ;当前代码块首地址
  401205:       00 00 00
  401208:       b0 00                   mov    $0x0,%al
  40120a:       e8 21 fe ff ff          callq  401030 <printf@plt>          ;printf
  40120f:       e9 53 00 00 00          jmpq   401267 <_Z7switch2i+0x97>    ;break
  401214:       48 bf 37 20 40 00 00    movabs $0x402037,%rdi             ;当前代码块首地址
  40121b:       00 00 00
  40121e:       b0 00                   mov    $0x0,%al
  401220:       e8 0b fe ff ff          callq  401030 <printf@plt>          ;printf
  401225:       e9 3d 00 00 00          jmpq   401267 <_Z7switch2i+0x97>    ;break
  40122a:       48 bf 46 20 40 00 00    movabs $0x402046,%rdi             ;当前代码块首地址
  401231:       00 00 00
  401234:       b0 00                   mov    $0x0,%al
  401236:       e8 f5 fd ff ff          callq  401030 <printf@plt>
  40123b:       e9 27 00 00 00          jmpq   401267 <_Z7switch2i+0x97>    ;break
  401240:       48 bf 46 20 40 00 00    movabs $0x402046,%rdi             ;当前代码块首地址
  401247:       00 00 00
  40124a:       b0 00                   mov    $0x0,%al
  40124c:       e8 df fd ff ff          callq  401030 <printf@plt>          ;call printf
  401251:       e9 11 00 00 00          jmpq   401267 <_Z7switch2i+0x97>    ;break;
  401256:       48 bf 55 20 40 00 00    movabs $0x402055,%rdi
  40125d:       00 00 00
  401260:       b0 00                   mov    $0x0,%al
  401262:       e8 c9 fd ff ff          callq  401030 <printf@plt>          ;call printf("unknown value")
  401267:       31 c0                   xor    %eax,%eax                    ;switch处理完的部分代码
  401269:       48 83 c4 10             add    $0x10,%rsp
  40126d:       5d                      pop    %rbp
  40126e:       c3                      retq
  40126f:       90                      nop

0000000000402000 <_IO_stdin_used>:
  402000:       01 00                   add    %eax,(%rax)
  402002:       02 00                   add    (%rax),%al
  402004:       00 00                   add    %al,(%rax)
  402006:       00 00                   add    %al,(%rax)
  402008:       fe                      (bad)
  402009:       11 40 00                adc    %eax,0x0(%rax)
  40200c:       00 00                   add    %al,(%rax)
  40200e:       00 00                   add    %al,(%rax)
  402010:       14 12                   adc    $0x12,%al
  402012:       40 00 00                add    %al,(%rax)
  402015:       00 00                   add    %al,(%rax)
  402017:       00 2a                   add    %ch,(%rdx)
  402019:       12 40 00                adc    0x0(%rax),%al
  40201c:       00 00                   add    %al,(%rax)
  40201e:       00 00                   add    %al,(%rax)
  402020:       40 12 40 00             adc    0x0(%rax),%al
  402024:       00 00                   add    %al,(%rax)
  402026:       00 00                   add    %al,(%rax)
```

&emsp;&emsp;```-Os```的代码相比于```-O0```更简单，相比而言```sub```指令被替换成了```dec```和```cmp```。```0x0040115d```这条指令存储的不是代码块的跳转表，而是字符串的表格，因为我们这里的代码比较简单全是```printf```函数唯一不同的地方是参数。因此编译器对这种场景优化，只存储字符串的跳转表。
```
;-Os
000000000040114c <_Z7switch2i>:
  40114c:       50                      push   %rax
  40114d:       89 f8                   mov    %edi,%eax
  40114f:       ff c8                   dec    %eax
  401151:       bf 31 20 40 00          mov    $0x402031,%edi
  401156:       83 f8 03                cmp    $0x3,%eax
  401159:       77 0a                   ja     401165 <_Z7switch2i+0x19>
  40115b:       48 98                   cltq
  40115d:       48 8b 3c c5 58 20 40    mov    0x402058(,%rax,8),%rdi
  401164:       00
  401165:       31 c0                   xor    %eax,%eax
  401167:       e8 c4 fe ff ff          callq  401030 <printf@plt>
  40116c:       31 c0                   xor    %eax,%eax
  40116e:       59                      pop    %rcx
  40116f:       c3                      retq
```

**==实验三：选项数大于3小于256且不具备任何线性关系==**

```cpp
int switch3(int type, int a) {
	switch (type) {
	case 3:
		type = sqrt(a); break;
	case 6:
		type = pow(a, 10); break;
	case 34:
		type = log(a); break;
	case 60:
		type = abs(a); break;
	default:
		printf("unknown value");
	}

  printf("the value is %d", type);
	return type;
}
```
&emsp;&emsp;测试发现```clang```一直是通过```jmp```来实现，连跳转表都没有生成。这里用```visual studio```来做反汇编测试。
1. 第一步是```type - 3```，和之前的流程相同，之后将```计算后的值和```57(0x39)```比较，大于则跳转调用```default```的代码；
2. 如果不跳转则先从索引表中获取跳转表的索引，然后根据索引找到需要跳转的地址，跳转到对应地址执行。

&emsp;&emsp;代码块我们就不看了，比较简单。我们直接看跳转表。跳转表索引中有58个选项，刚好对应3-60。而跳转表里面存储了5个地址（```0x00a21079, 0x00a2108c, 0x00a210a7, 0x00a210ba, 0x00a210c5```）分别对应具体的执行代码块。
```
;;跳转表索引
0x00A210FC  00 04 04 01 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04 02 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04 04  
0x00A2112D  04 04 04 04 04 04 04 04 03 3b 0d 04 30 a2 00 f2 75 02 f2 c3 f2 e9 79 02 00 00 56 6a 01 e8 44 0b 00 00 e8 81 06 00 00 50 e8 6f 0b 00 00 e8 9a 0b 00

;;跳转表
0x00A210E8  79 10 a2 00 8c 10 a2 00 a7 10 a2 00 ba 10 a2 00 c5 10 a2 00

;;实际测试发现clang好像不会生成跳转表，下面的反汇编来自于visual studio cl.exe
int main(int argc, char **argv) {       //这里switch直接被内联了
00A21050  push        ebp  
00A21051  mov         ebp,esp  
00A21053  and         esp,0FFFFFFF8h  
00A21056  push        ecx  
	switch3(argc, argv[0][0]);
00A21057  mov         eax,dword ptr [argv]  
00A2105A  push        esi  
00A2105B  mov         esi,dword ptr [argc]  
00A2105E  mov         eax,dword ptr [eax]  
00A21060  movsx       ecx,byte ptr [eax]  
00A21063  lea         eax,[esi-3]  
00A21066  cmp         eax,39h                       ;type >? 60
00A21069  ja          main+75h (0A210C5h)  
00A2106B  movzx       eax,byte ptr [eax+0A210FCh]   ;跳转表索引
00A21072  jmp         dword ptr [eax*4+0A210E8h]    ;跳转表
00A21079  movd        xmm0,ecx  
00A2107D  cvtdq2pd    xmm0,xmm0  
	switch3(argc, argv[0][0]);
00A21081  call        __libm_sse2_sqrt_precise (0A21D2Fh)  
00A21086  cvttsd2si   esi,xmm0  
00A2108A  jmp         main+82h (0A210D2h)         ;break
00A2108C  movsd       xmm1,mmword ptr [__real@4024000000000000 (0A22128h)]  
00A21094  movd        xmm0,ecx  
00A21098  cvtdq2pd    xmm0,xmm0  
00A2109C  call        __libm_sse2_pow_precise (0A21D29h)  
00A210A1  cvttsd2si   esi,xmm0  
00A210A5  jmp         main+82h (0A210D2h)         ;break  
00A210A7  movd        xmm0,ecx  
00A210AB  cvtdq2pd    xmm0,xmm0  
00A210AF  call        __libm_sse2_log_precise (0A21D23h)  
00A210B4  cvttsd2si   esi,xmm0  
00A210B8  jmp         main+82h (0A210D2h)         ;break  
00A210BA  mov         eax,ecx  
00A210BC  cdq  
00A210BD  mov         esi,eax  
00A210BF  xor         esi,edx  
00A210C1  sub         esi,edx  
00A210C3  jmp         main+82h (0A210D2h)         ;break  
00A210C5  push        offset string "unknown value" (0A22108h)        ;default 代码块
00A210CA  call        printf (0A21010h)  
00A210CF  add         esp,4  
00A210D2  push        esi                 
00A210D3  push        offset string "the vaue is %d" (0A22118h)  
00A210D8  call        printf (0A21010h)  
00A210DD  add         esp,8  
	return 0;
00A210E0  xor         eax,eax  
}
```
**==实验四：选项数量大于3且数值大于255==**
```cpp
int switch3(int type, int a) {
	switch (type) {
	case 3:
		type = sqrt(a); break;
	case 6:
		type = pow(a, 10); break;
	case 34:
		type = log(a); break;
	case 606:
		type = abs(a); break;
	default:
		printf("unknown value");
	}

	printf("the vaue is %d", type);
	return type;
}
```
&emsp;&emsp;如下面的反汇编所示，编译器会根据当前已经有的值生成判定树，不断判断跳转，直至跳转到对应的代码块。下面是对应代码判定树的结构：

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/465578f0c3b72aef0d7c41f8333a1471.png)

```
int main(int argc, char **argv) {
00961050  push        ebp  
00961051  mov         ebp,esp  
00961053  and         esp,0FFFFFFF8h               
00961056  push        ecx  
	switch3(argc, argv[0][0]);
00961057  mov         eax,dword ptr [argv]  
0096105A  push        esi  
0096105B  mov         esi,dword ptr [argc]  
0096105E  mov         eax,dword ptr [eax]  
00961060  movsx       eax,byte ptr [eax]  
00961063  cmp         esi,22h                       
00961066  jg          main+65h (09610B5h)           ;;if type > 34
00961068  je          main+52h (09610A2h)           ;;if type == 34
0096106A  cmp         esi,3  
0096106D  je          main+3Fh (096108Fh)           ;;if type == 3
0096106F  cmp         esi,6  
00961072  jne         main+6Dh (09610BDh)           ;;if type != 6
	switch3(argc, argv[0][0]);                        ;case 6 代码块
00961074  movsd       xmm1,mmword ptr [__real@4024000000000000 (0962128h)]  
0096107C  movd        xmm0,eax  
00961080  cvtdq2pd    xmm0,xmm0  
00961084  call        __libm_sse2_pow_precise (0961CD9h)  
00961089  cvttsd2si   esi,xmm0  
0096108D  jmp         main+83h (09610D3h)  
0096108F  movd        xmm0,eax                      ;;case 3 代码块
00961093  cvtdq2pd    xmm0,xmm0  
00961097  call        __libm_sse2_sqrt_precise (0961CDFh)  
0096109C  cvttsd2si   esi,xmm0  
009610A0  jmp         main+83h (09610D3h)  
009610A2  movd        xmm0,eax                      ;;case 34 代码块
009610A6  cvtdq2pd    xmm0,xmm0  
009610AA  call        __libm_sse2_log_precise (0961CD3h)  
009610AF  cvttsd2si   esi,xmm0  
009610B3  jmp         main+83h (09610D3h)  
009610B5  cmp         esi,25Eh  
009610BB  je          main+7Ch (09610CCh)           ;if type == 606
009610BD  push        offset string "unknown value" (0962108h)  
009610C2  call        printf (0961010h)  
009610C7  add         esp,4  
009610CA  jmp         main+83h (09610D3h)  
009610CC  cdq                                      ;;case 606 代码块
009610CD  mov         esi,eax  
009610CF  xor         esi,edx  
009610D1  sub         esi,edx  
009610D3  push        esi                           ;switch结束块
009610D4  push        offset string "the vaue is %d" (0962118h)  
009610D9  call        printf (0961010h)  
009610DE  add         esp,8  
	return 0;
009610E1  xor         eax,eax  
}
```

>&emsp;&emsp;下面的代都会用vs去分析，不再使用clang-linux，太麻烦了。
### 5.3 循环语句
#### 5.3.1 ```do...while```
&emsp;&emsp;```do...while```循环的代码比较简单就是，循环的判断在代码块的下面。

**==实验==**
```cpp
int dowhile(int a) {
	do {
		a = a + 30;
	} while (a < 60);
	return a;
}

int main(int argc, char **argv) {
	int a = dowhile(argc);
	printf("%d", a);
	return 0;
}
```

```
int main(int argc, char **argv) {
00B81040  push        ebp  
00B81041  mov         ebp,esp  
	int a = dowhile(argc);
00B81043  mov         eax,dword ptr [argc]  
00B81046  add         eax,1Eh           ;循环开头  a = a + 30
00B81049  cmp         eax,3Ch           
00B8104C  jl          main+6h (0B81046h); if a < 60 then jmp  
	printf("%d", a);
00B8104E  push        eax  
00B8104F  push        offset string "%d" (0B820F8h)  
00B81054  call        printf (0B81010h)  
00B81059  add         esp,8  
	return 0;
00B8105C  xor         eax,eax  
}
```
#### 5.3.2 ```while```
&emsp;&emsp;```while```相比于```do...while```是先判断再循环。

**==实验==**
```cpp
int dowhile(int a) {
	while(a<60) {
		a = a + 2;
	} ;
	return a;
}

int main(int argc, char **argv) {
	int a = dowhile(argc);
	printf("%d", a);
	return 0;
}
```

```
int main(int argc, char **argv) {
00A81040  push        ebp  
00A81041  mov         ebp,esp  
	int a = dowhile(argc);
00A81043  mov         ecx,dword ptr [argc]  
00A81046  cmp         ecx,3Ch  
00A81049  jge         main+1Ah (0A8105Ah)       ;;循环条件判断
00A8104B  mov         eax,3Bh  
00A81050  sub         eax,ecx  
00A81052  shr         eax,1  
00A81054  lea         ecx,[ecx+eax*2]  
00A81057  add         ecx,2  
	printf("%d", a);
00A8105A  push        ecx               ;;循环结束
00A8105B  push        offset string "%d" (0A820F8h)  
00A81060  call        printf (0A81010h)  
00A81065  add         esp,8  
	return 0;
00A81068  xor         eax,eax  
}
```
#### 5.3.3 ```for```
&emsp;&emsp;```for```相比于```while```循环在结尾处多了条件的处理。

**==实验==**
```cpp
int dowhile(int a) {
	for(int i = 0;i < a;i *= 37){
		printf("the value is %d", a);
	} 

	return a;
}

int main(int argc, char **argv) {
	int a = dowhile(argc);
	printf("%d", a);
	return 0;
}
```

```
int main(int argc, char **argv) {
00581040  push        ebp  
00581041  mov         ebp,esp  
00581043  push        esi  
00581044  push        edi  
	int a = dowhile(argc);
00581045  mov         edi,dword ptr [argc]  
00581048  xor         esi,esi  
0058104A  test        edi,edi  
0058104C  jle         main+25h (0581065h)  
0058104E  xchg        ax,ax  
00581050  push        edi                       ;循环体开头
00581051  push        offset string "the value is %d" (05820F8h)  
00581056  call        printf (0581010h)  
0058105B  imul        esi,esi,25h               ;a *= 37
0058105E  add         esp,8                 
00581061  cmp         esi,edi  
00581063  jl          main+10h (0581050h)       ;i < a jump
	printf("%d", a);
00581065  push        edi  
	printf("%d", a);
00581066  push        offset string "%d" (0582108h)  
0058106B  call        printf (0581010h)  
00581070  add         esp,8  
	return 0;
00581073  xor         eax,eax  
00581075  pop         edi  
00581076  pop         esi  
}
```

## 6 函数
### 6.1 函数调用
&emsp;&emsp;函数调用是通过栈实现的，每个函数都有自己的栈帧。栈帧中保存了当前函数调用过程中的函数参数，局部变量，函数的返回地址等内容。每一次函数被调用都会生成当前函数的栈帧，函数返回后会进行栈平衡，即返栈空间。因为栈空间是有限的的，所以我们无法无限递归调用某个函数这会导致爆栈。
&emsp;&emsp;调用函数时需要申请栈，函数调用结束时需要平衡栈。具体由哪一方负责做这个事情？根据不同的调用约定，函数入参的顺序和栈的维护方不同：
- ```_cdecl```：C\C++默认的调用方式，调用方平衡栈，不定参数的函数可以使用这种方式；
- ```_stdcall```：被调方平衡栈，不定参数的函数无法使用这种方式；
- ```_fastcall```：寄存器方式传参，被调方平衡栈，不定参数的函数无法使用这种方式。

>详情可见[函数的调用约定](https://blog.csdn.net/GrayOnDream/article/details/108330787)
>需要注意x64和x86的调用约定不同，x64只有一种调用约定。函数调用的前4个参数用寄存器传参即```rcx,rdx,r8,r9```，由右向左传参，任何大于8字节或者不是1字节，2字节，4字节，8字节的参数使用引用传参。浮点都是通过```xmm```寄存器传参（如果同时有浮点和整数则按原来的顺序传参，比如参数为```float,int,float,int```则使用的寄存器分别为```xmm0,rdx,xmm2,r9```）。虽然前四个参数用寄存器传参但是栈底也有对应的预留空间避免寄存器不够用。

**==实验一==**
```cpp
void _stdcall stdcall(int v) {
	printf("stdcall %d", v);
}

void _cdecl cdecalll(int v) {
	printf("cdcel %d", v);
	printf("cdcel %d", v);
	printf("cdcel %d", v);
	printf("cdcel %d", v);
}

int main(int argc, char **argv) {
	stdcall(argc);
	cdecalll(argc);
	return 0;
}
```
&emsp;&emsp;根据下面的反汇编这里描述下一个完整的函数调用过程：
1. 首先```push ebp```，当前```ebp```的值为被调用方的栈底，用于平衡栈；
2. 然后提升栈底为```esp```；
3. 调整栈顶指针，开辟栈帧（下面的反汇编因为代码中没有用到局部变量因此没有开辟额外的占空间）；
4. 执行函数相关代码；
5. 执行完成后，跳帧栈底指针，弹出```ebp```；
6. 栈平衡

>&emsp;&emsp;cdcel复写传播优化，当调用多个相同函数时并不会每次都在函数结尾平衡栈，而是统一平衡栈。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/917e770a6ff1df4cd58e339ad8645df8.png)


```
void _stdcall stdcall(int v) {
007E1080  push        ebp  
007E1081  mov         ebp,esp  
	printf("stdcall %d", v);
007E1083  mov         eax,dword ptr [v]  
007E1086  push        eax  
007E1087  push        7E21B8h  
007E108C  call        printf (07E1040h)  
007E1091  add         esp,8  
}
007E1094  pop         ebp  
007E1095  ret         4  


void _cdecl cdecalll(int v) {
00E31050  push        esi  
00E31051  push        edi  
00E31052  mov         esi,ecx  
	printf("cdcel %d", v);
00E31054  mov         edi,offset string "cdcel %d" (0E32104h)  
00E31059  push        esi  
00E3105A  push        edi  
00E3105B  call        printf (0E31025h)  
	printf("cdcel %d", v);
00E31060  push        esi  
00E31061  push        edi  
00E31062  call        printf (0E31025h)  
	printf("cdcel %d", v);
00E31067  push        esi  
00E31068  push        edi  
00E31069  call        printf (0E31025h)  
	printf("cdcel %d", v);
00E3106E  push        esi  
00E3106F  push        edi  
00E31070  call        printf (0E31025h)  
00E31075  add         esp,20h           ;;复写传播优化
00E31078  pop         edi  
00E31079  pop         esi  
}
00E3107A  ret
```

### 6.2 函数寻址
&emsp;&emsp;函数调用过程中参数传递、局部变量的创建都是通过栈或者寄存器传递。
&emsp;&emsp;局部变量都是栈上的一块内存，通常可以通过```ebp - n```的方式访问，因为对于一个栈帧，```ebp```是固定的，只需要根据```ebp```的偏移访问即可。对于某些栈况可以通过```esp```访问避免```ebp```的操作节省指令。
&emsp;&emsp;函数中的参数传递是通过```push```执行将对应的参数值复制到当前栈顶，因此可以通过```ebp + n```的方式访问参数，对于某些优化的情况下可以直接利用寄存器访问函数参数。
&emsp;&emsp;函数调用是使用```call```指令，该指令会将当前指令地址的下一个地址压入栈中，等函数返回就可以找到返回的地址，```ret```时就是利用该地址返回的。而函数返回时通常会使用```eax```传递返回值，但是当返回值大于机器地址长度个字节时会使用其他寄存器，如果太大就会通过基址寻址拷贝。
```cpp
a getV() {
	return a{};
}

int main(int argc, char **argv) {
	a a1 = getV();
	return 0;
}
```

```
a getV() {
00511010  push        ebp  
00511011  mov         ebp,esp  
00511013  sub         esp,20h  
00511016  push        esi  
00511017  push        edi  
	return a{};
00511018  xor         eax,eax  
0051101A  mov         dword ptr [ebp-20h],eax  
0051101D  mov         dword ptr [ebp-1Ch],eax  
00511020  mov         dword ptr [ebp-18h],eax  
00511023  mov         dword ptr [ebp-14h],eax  
00511026  mov         dword ptr [ebp-10h],eax  
00511029  mov         dword ptr [ebp-0Ch],eax  
0051102C  mov         dword ptr [ebp-8],eax  
0051102F  mov         dword ptr [ebp-4],eax  
00511032  mov         ecx,8  
00511037  lea         esi,[ebp-20h]  
0051103A  mov         edi,dword ptr [ebp+8]  
0051103D  rep movs    dword ptr es:[edi],dword ptr [esi]  
005
```

## 7 变量寻址
### 7.1 全局变量和静态变量
**全局变量和静态变量**
&emsp;&emsp;在了解全局变量和静态变量的存储方式之前，应该了解可执行文件的基本格式。无论是Windows上还是Linux上的可执行性文件都是以类COFF格式存储，Linux上是ELF文件，Windows上是PE文件。可执行文件中根据不同数据的类型将代码和数据以段的方式分开存储，比如代码存储在代码段，数据存储在数据段，数据段又分只读数据段。
&emsp;&emsp;而全局变量和静态变量就存储在数据段上，根据是否初始化又分为bss和data段。因此在实际程序运行时可以从装载到对应内存的地址访问到全局变量和静态变量的数据。在程序中使用静态变量和全局变量的方式是相同的，区别是编译器从语义上对不同符号进行了不用的签名，符号表上可见性也不同来保证语法正确。

**==实验一==**
```cpp
int a = 1;
static int b = 1;
int main(int argc, char **argv) {
	printf("%d %d", a, b);
	return 0;
}
```
&emsp;&emsp;从下面的反汇编中可以看出无论是静态变量还是全局变量都是通过绝对地址访问，其数据存储于数据区。
```
int a = 1;
static int b = 1;
int main(int argc, char **argv) {
00341080  push        ebp  
00341081  mov         ebp,esp  
	printf("%d %d", a, b);
00341083  mov         eax,dword ptr [b (034301Ch)]  
00341088  push        eax  
00341089  mov         ecx,dword ptr [a (0343018h)]  
0034108F  push        ecx  
00341090  push        3421F4h  
00341095  call        printf (0341040h)  
0034109A  add         esp,0Ch  
	return 0;
0034109D  xor         eax,eax  
}
```

**局部静态变量**
&emsp;&emsp;局部静态变量只能被初始化一次。编译器通过一个标志位来判断静态变量是否被初始化过，如果已经初始化就不再初始化。

**==实验二==**

```cpp
void func() {
	static int i = atoi("1");
	if (i != 1) {
		scanf("%d", i);
	}

	printf("%d", i);
}
```

&emsp;&emsp;从下面的反汇编可以看到初始化静态变量前会检查```0x8a43b0h```这个地址，初始化完后就会改写该标志位。
```
void func() {
008A1100  push        ebp  
008A1101  mov         ebp,esp  
	static int i = atoi("1");
008A1103  mov         eax,dword ptr fs:[0000002Ch]  
008A1109  mov         ecx,dword ptr [eax]  
008A110B  mov         edx,dword ptr ds:[8A43B0h]  
008A1111  cmp         edx,dword ptr [ecx+4]  
008A1117  jle         func+4Fh (08A114Fh)  
008A1119  push        8A43B0h  
008A111E  call        _Init_thread_header (08A1346h)      ;创建标志位
008A1123  add         esp,4  
008A1126  cmp         dword ptr ds:[8A43B0h],0FFFFFFFFh  ;检查静态变量是否被初始化
008A112D  jne         func+4Fh (08A114Fh)  
	static int i = atoi("1");
008A112F  push        8A32E8h  
008A1134  call        dword ptr [__imp__atoi (08A306Ch)]  
008A113A  add         esp,4  
008A113D  mov         dword ptr [i (08A4028h)],eax  
008A1142  push        8A43B0h  
008A1147  call        _Init_thread_footer (08A12FCh)      ；写标志位
008A114C  add         esp,4  
	if (i != 1) {
008A114F  cmp         dword ptr [i (08A4028h)],1  
008A1156  je          func+6Bh (08A116Bh)  
		scanf("%d", i);
008A1158  mov         eax,dword ptr [i (08A4028h)]  
008A115D  push        eax  
008A115E  push        8A32ECh  
008A1163  call        scanf (08A10C0h)  
008A1168  add         esp,8  
	}

	printf("%d", i);
008A116B  mov         ecx,dword ptr [i (08A4028h)]  
008A1171  push        ecx  
008A1172  push        8A32F0h  
008A1177  call        printf (08A1050h)  
008A117C  add         esp,8  
}
```

### 7.2 堆内存
&emsp;&emsp;堆内存时通过```new```或者```malloc```申请的，区别是```new```会构造对象，```malloc```不会。销毁的调用```free```和```delete```同理。
&emsp;&emsp;CRT的堆内存管理是通过下面的双向链表管理，每个节点存储了当前内存的大小和一些其他信息。
```cpp
struct _CrtMemBlockHeader{
    _CrtMemBlockHeader* _block_header_next;
    _CrtMemBlockHeader* _block_header_prev;
    char const*         _file_name;
    int                 _line_number;

    int                 _block_use;
    size_t              _data_size;

    long                _request_number;
    unsigned char       _gap[no_mans_land_size];

    // Followed by:
    // unsigned char    _data[_data_size];
    // unsigned char    _another_gap[no_mans_land_size];
};
```
&emsp;&emsp;下面是一段```new int```的内存，该内存的值为2。```0x12a7628```是具体的内存地址，地址前后的```fd```是越界检查。
```cpp
0x012A75F7  00 00 00 00 00 00 00 00 00 30 15 43 7f 00 13 00 88 58 74 2a 01 00 00 00 00 00 00 00 00 00 00 00 00 01 00 00 00 04 00 00 00 4f 00 00 00 fd fd fd fd
0x012A7628  02 00 00 00 fd fd fd fd
```

### 7.3 数组和指针寻址
&emsp;&emsp;数组就是一些列相同大小的元素的集合，通常通过首地址+偏移计算需要访问的具体元素。在反汇编中数组的访问就是通过地址偏移进行访问。当数组作为函数参数时，数组会退化为指针，因此传入的参数实际就是数组的首地址。另外需要注意的是反汇编中```type arr[]={}```和```type *arr={}```的区别，前者会将初始化列表中的值拷贝给对应的局部变量地址，而后置数据存储在data中，这里只是通过指针访问而已。其他类型的指针访问，比如函数指针，返回值为指针等都和普通寻址方式相同，只是具体对象不同而已。

**==实验==**
```cpp
void func() {
	static int i = atoi("1");
	if (i != 1) {
		scanf("%d", i);
	}

	printf("%d", i);
}

int main(int argc, char **argv) {
	const char *a = "12345";
	char b[] = "12345";
	auto t = func;
	printf("%s %s", a, b);
	t();
	return 0;
}
```

```
int main(int argc, char **argv) {
00571190  push        ebp  
00571191  mov         ebp,esp  
00571193  sub         esp,14h  
00571196  mov         eax,dword ptr [__security_cookie (0574008h)]  
0057119B  xor         eax,ebp  
0057119D  mov         dword ptr [ebp-4],eax  
	const char *a = "12345";
005711A0  mov         dword ptr [a],5732F4h           ;这里只是将局部变量给a
	char b[] = "12345";
005711A7  mov         eax,dword ptr ds:[005732FCh]  
005711AC  mov         dword ptr [b],eax               ;这里对整块内存进行了拷贝  
005711AF  mov         cx,word ptr ds:[573300h]  
005711B6  mov         word ptr [ebp-8],cx  
	auto t = func;
005711BA  mov         dword ptr [t],offset func (0571100h)  
	printf("%s %s", a, b);
005711C1  lea         edx,[b]  
005711C4  push        edx  
005711C5  mov         eax,dword ptr [a]  
005711C8  push        eax  
005711C9  push        573304h  
005711CE  call        printf (0571050h)  
005711D3  add         esp,0Ch  
	t();
005711D6  call        dword ptr [t]  
	return 0;
005711D9  xor         eax,eax  
}
```
## 9 结构体和类
### 9.1 类的内存布局
&emsp;&emsp;C++中类和结构体除了访问控制上的区别没有任何区别。一个类或者结构体就是一些列对象的集合，所以类中的成员布局是按照实际的类定义的顺序布局的。类的首地址一般和类的第一个成员的地址相同（有虚函数的类例外）。一般类的大小大于等于所有成员大小之和，一方面是因为虚函数表的存在导致有虚函数的类多一个虚函数表指针，另一方面编译器一般会做内存对齐，所以实际的大小根据编译器的不同而带下不同。下面的类大小计算公式中，成员不包含静态成员，另外空类的大小为1,1个字节大小是一个占位符，来表示一个类。
>类的静态成员本身不属于某个类实例，而是属于类。因此其和普通静态变量一样存储在静态区。

$$
sizoef(类对象)=sizoef(虚函数表指针)（此项对于有虚函数的类生效）+sizeof(成员1)+sizeof(成员2)+...+sizeof(成员n)+对齐的内存大小
$$

&emsp;&emsp;其中比较复杂的就是内存对齐，一般对齐的大小以期望对齐的大小M和成员的大小中的最小值作为对齐宽度。
```cpp
class myclass {
public:
	int a;
	char b;
	static int c;
};

int myclass::c = 0;

int main(int argc, char **argv) {
	myclass cls = {};
	printf("%d %d", cls.a, cls.b);
	return 0;
}
```
&emsp;&emsp;从下面的反汇编中可以看到```myclass:c```并不占用类空间，myclass也是4字节对齐的。
```
int main(int argc, char **argv) {
003E1080  push        ebp  
003E1081  mov         ebp,esp  
003E1083  sub         esp,0Ch  
003E1086  mov         eax,dword ptr ds:[003E3004h]  
003E108B  xor         eax,ebp  
003E108D  mov         dword ptr [ebp-4],eax  
	myclass cls = {};
003E1090  xor         eax,eax  
003E1092  mov         dword ptr [ebp-0Ch],eax       ;cls.a = 0; 
003E1095  mov         dword ptr [ebp-8],eax         ;cls.b = 0;
	printf("%d %d", cls.a, cls.b);
003E1098  movsx       ecx,byte ptr [ebp-8]  
003E109C  push        ecx  
003E109D  mov         edx,dword ptr [ebp-0Ch]  
003E10A0  push        edx  
003E10A1  push        3E21F4h  
003E10A6  call        003E1040  
003E10AB  add         esp,0Ch  
	return 0;
003E10AE  xor         eax,eax  
}
```
### 9.2 this指针
&emsp;&emsp;C++的类中，类成员变量时存储在对应的地址上，而类成员函数是存储在只读代码段的。当调用一个成员函数时，起始其会额外传入一个参数```this```，该函数内所有针对成员变量的访问都是通过该```this```指针访问的。对于一个成员函数```cls.func()```其调用等价于```func(&cls)```。
&emsp;&emsp;另外，类的成员函数的默认调用约定为[__thiscall](https://learn.microsoft.com/zh-cn/cpp/cpp/thiscall?view=msvc-170)。函数的第一个参数```this```指针都会用ecx传递（而对于x64程序第一个参数本身就是ecx，所有需要判断是否符合this指针的条件否则不一定是this指针）。

**==实验==**
```cpp
class myclass {
public:
	int a;
	int func() {
		return a;
	};
};

int main(int argc, char **argv) {
	myclass cls = {};
	printf("%d %d", cls.func());
	return 0;
}
```
&emsp;&emsp;上面声明的函数没有参数但是下面函数调用时却给函数传递了个参数ecx
```
int main(int argc, char **argv) {
002C1090  push        ebp  
002C1091  mov         ebp,esp  
002C1093  sub         esp,8  
002C1096  mov         eax,dword ptr ds:[002C3004h]  
002C109B  xor         eax,ebp  
002C109D  mov         dword ptr [ebp-4],eax 
	myclass cls = {};
002C10A0  xor         eax,eax  
	myclass cls = {};
002C10A2  mov         dword ptr [ebp-8],eax  ;cls.a = 0
	printf("%d %d", cls.func());
002C10A5  lea         ecx,[ebp-8]  ;this -> ecx
002C10A8  call        002C1080  
002C10AD  push        eax  
002C10AE  push        2C21F4h  
002C10B3  call        002C1040  
002C10B8  add         esp,8  
	return 0;
002C10BB  xor         eax,eax  
}
```

### 9.3 类对象的传参与返回
&emsp;&emsp;类对象的传参与返回相比于普通的变量传参要复杂一点，主要是类对象一般都比较大，很难用几个寄存器传值。
```cpp
class myclass {
public:
	int64_t a;
	int64_t b;
};

myclass func(myclass cls) {
	printf("%lld", cls.a);
	return myclass{1, 2};
}

int main(int argc, char **argv) {
	myclass cls = {};
	cls = func(cls);
	return 0;
}
```
&emsp;&emsp;下面的反汇编中发生了很多次的内存对象拷贝，但是实际代码中不会这样，编译器会优化避免多次拷贝。
```
myclass func(myclass cls) {
00971080  push        ebp  
00971081  mov         ebp,esp  
00971083  sub         esp,10h  
	printf("%lld", cls.a);
00971086  mov         eax,dword ptr [ebp+10h]  ;取出a和b的值访问
00971089  push        eax  
0097108A  mov         ecx,dword ptr [ebp+0Ch]  
0097108D  push        ecx  
0097108E  push        9721F4h  
00971093  call        00971040  
00971098  add         esp,0Ch  
	return myclass{1, 2};
0097109B  mov         dword ptr [ebp-10h],1       ;初始化一个myclass临时对象
009710A2  mov         dword ptr [ebp-0Ch],0  
009710A9  mov         dword ptr [ebp-8],2  
009710B0  mov         dword ptr [ebp-4],0  
009710B7  mov         edx,dword ptr [ebp+8]  
009710BA  mov         eax,dword ptr [ebp-10h]     ;将该临时对象的内存拷贝给返回的预留栈上
009710BD  mov         dword ptr [edx],eax  
009710BF  mov         ecx,dword ptr [ebp-0Ch]  
009710C2  mov         dword ptr [edx+4],ecx  
009710C5  mov         eax,dword ptr [ebp-8]  
009710C8  mov         dword ptr [edx+8],eax  
009710CB  mov         ecx,dword ptr [ebp-4]  
009710CE  mov         dword ptr [edx+0Ch],ecx  
009710D1  mov         eax,dword ptr [ebp+8]  
}
009710D4  mov         esp,ebp  
009710D6  pop         ebp  
009710D7  ret  


int main(int argc, char **argv) {
009710E0  push        ebp  
009710E1  mov         ebp,esp  
009710E3  sub         esp,34h  
009710E6  mov         eax,dword ptr ds:[00973004h]  
009710EB  xor         eax,ebp  
009710ED  mov         dword ptr [ebp-4],eax  
	myclass cls = {};
009710F0  xor         eax,eax  
009710F2  mov         dword ptr [ebp-14h],eax  ;初始化cls
009710F5  mov         dword ptr [ebp-10h],eax  
009710F8  mov         dword ptr [ebp-0Ch],eax  
009710FB  mov         dword ptr [ebp-8],eax  
	cls = func(cls);
009710FE  sub         esp,10h                  ;栈上预留10个字节的空间
00971101  mov         ecx,esp  
00971103  mov         edx,dword ptr [ebp-14h]  ;下面是将cls的内存拷贝到预留栈空间上
00971106  mov         dword ptr [ecx],edx  
00971108  mov         eax,dword ptr [ebp-10h]  
0097110B  mov         dword ptr [ecx+4],eax  
0097110E  mov         edx,dword ptr [ebp-0Ch]  
00971111  mov         dword ptr [ecx+8],edx  
00971114  mov         eax,dword ptr [ebp-8]  
00971117  mov         dword ptr [ecx+0Ch],eax  
0097111A  lea         ecx,[ebp-34h]  
0097111D  push        ecx                       ;传入类的首地址
	cls = func(cls);
0097111E  call        00971080  
00971123  add         esp,14h  
00971126  mov         edx,dword ptr [eax]       ;将返回值取出给临时变量
00971128  mov         dword ptr [ebp-24h],edx  
0097112B  mov         ecx,dword ptr [eax+4]  
0097112E  mov         dword ptr [ebp-20h],ecx  
00971131  mov         edx,dword ptr [eax+8]  
00971134  mov         dword ptr [ebp-1Ch],edx  
00971137  mov         eax,dword ptr [eax+0Ch]  
0097113A  mov         dword ptr [ebp-18h],eax  
0097113D  mov         ecx,dword ptr [ebp-24h]   ;将临时变量的内存拷贝给cls  
00971140  mov         dword ptr [ebp-14h],ecx  
00971143  mov         edx,dword ptr [ebp-20h]  
00971146  mov         dword ptr [ebp-10h],edx  
00971149  mov         eax,dword ptr [ebp-1Ch]  
0097114C  mov         dword ptr [ebp-0Ch],eax  
0097114F  mov         ecx,dword ptr [ebp-18h]  
00971152  mov         dword ptr [ebp-8],ecx  
	return 0;
00971155  xor         eax,eax  
}
```
### 9.4 构造函数和析构函数
&emsp;&emsp;C++中的类的初始化是在构造函数中完成的，当类被实例化就会调用构造函数。而析构函数是对象被销毁时销毁类对象的。C++中调用构造函数和析构函数的时机有：
1. 局部对象。通常局部变量的创建就会调用构造函数；
2. 堆对象。通过```new```创建堆对象时就会调用构造函数，反之用```delete```析构对象时就会调用析构函数；
3. 参数对象。当将一个对象传递给函数，而该函数的类又是值传递时就会触发拷贝构造函数，此对象出作用域被销毁时就会调用析构函数；
4. 全局对象。全局对象在进入```main```之前被初始化，退出之前被析构;
5. 静态对象。静态对象的生命周期和全局对象类似，唯一不同的是局部静态对象的初始化时lazy的。

&emsp;&emsp;另外需要注意几个点，一，C++中存在很多优化，比如RVO优化，就不会创建临时对象产生对象的拷贝等工作；二，并不是所有的类都会生成默认构造函数，必须是非trival的类才行，非trival的类并不会生成任何构造函数，也就不存在构造和析构。如果是trival的类自己显式声明了构造函数或者析构函数也会调用。

**==实验==**
```cpp
class myclass {
public:
	myclass() {}
	myclass(const myclass &cls) {}
	int a;
};

myclass func(myclass cls) {
	printf("%lld", cls.a);
	return myclass{};
}

int main(int argc, char **argv) {
	myclass *cls = new myclass;
	myclass ret = func(*cls);
	delete cls;
	return 0;
}
```
&emsp;&emsp;下面的调用关系比较清晰，不详细描述了。

```
myclass func(myclass cls) {
007310A0  push        ebp  
007310A1  mov         ebp,esp  
	printf("%lld", cls.a);
007310A3  mov         eax,dword ptr [cls]  
007310A6  push        eax  
	printf("%lld", cls.a);
007310A7  push        73227Ch  
007310AC  call        printf (0731040h)  
007310B1  add         esp,8  
	return myclass{};
007310B4  mov         ecx,dword ptr [ebp+8]  
007310B7  call        myclass::myclass (0731080h)  ;返回时调用构造函数构造一个类
007310BC  mov         eax,dword ptr [ebp+8]  
}
007310BF  pop         ebp  
007310C0  ret 


int main(int argc, char **argv) {
007310D0  push        ebp  
007310D1  mov         ebp,esp  
007310D3  push        0FFFFFFFFh  
007310D5  push        731F3Fh  
007310DA  mov         eax,dword ptr fs:[00000000h]  
007310E0  push        eax  
007310E1  sub         esp,24h  
007310E4  mov         eax,dword ptr [__security_cookie (0733004h)]  
007310E9  xor         eax,ebp  
007310EB  mov         dword ptr [ebp-10h],eax  
007310EE  push        eax  
007310EF  lea         eax,[ebp-0Ch]  
007310F2  mov         dword ptr fs:[00000000h],eax  
	myclass *cls = new myclass;
007310F8  push        4  
007310FA  call        operator new (07311B0h)         ;new申请内存
007310FF  add         esp,4  
00731102  mov         dword ptr [ebp-1Ch],eax  
00731105  mov         dword ptr [ebp-4],0  
0073110C  cmp         dword ptr [ebp-1Ch],0  
00731110  je          main+4Fh (073111Fh)               ;检查内存是否申请成功
00731112  mov         ecx,dword ptr [ebp-1Ch]  
00731115  call        myclass::myclass (0731080h)       ;调用构造函数  
0073111A  mov         dword ptr [ebp-20h],eax  
0073111D  jmp         main+56h (0731126h)  
0073111F  mov         dword ptr [ebp-20h],0  
00731126  mov         eax,dword ptr [ebp-20h]  
00731129  mov         dword ptr [ebp-28h],eax  
0073112C  mov         dword ptr [ebp-4],0FFFFFFFFh  
00731133  mov         ecx,dword ptr [ebp-28h]  
00731136  mov         dword ptr [cls],ecx  
	myclass ret = func(*cls);
00731139  push        ecx  
0073113A  mov         ecx,esp  
0073113C  mov         dword ptr [ebp-30h],esp  
0073113F  mov         edx,dword ptr [cls]  
00731142  push        edx  
00731143  call        myclass::myclass (0731090h)  ;调用拷贝构造函数拷贝对象
00731148  lea         eax,[ret]  
0073114B  push        eax  
0073114C  call        func (07310A0h)  
00731151  add         esp,8  
	delete cls;
00731154  mov         ecx,dword ptr [cls]  
00731157  mov         dword ptr [ebp-24h],ecx  
0073115A  push        4  
0073115C  mov         edx,dword ptr [ebp-24h]  
0073115F  push        edx  
00731160  call        operator delete (07311E0h)        ;因为类是一个trival的类，因此只调用了delete没有掉调用析构函数
00731165  add         esp,8  
00731168  cmp         dword ptr [ebp-24h],0  
0073116C  jne         main+0A7h (0731177h)  
0073116E  mov         dword ptr [ebp-2Ch],0  
00731175  jmp         main+0B4h (0731184h)  
00731177  mov         dword ptr [cls],8123h  
0073117E  mov         eax,dword ptr [cls]  
00731181  mov         dword ptr [ebp-2Ch],eax  
	return 0;
00731184  xor         eax,eax  
}
```

### 9.5 虚函数
&emsp;&emsp;C++中为了实现多态，每个虚类中都一个默认的成员，虚函数表指针。该指针位于类的首地址，表中存储了当前类对应的虚函数，因此调用虚函数时需要先寻址到虚函数表，再索引到具体第几个虚函数指针。另外，虚函数表的-1位置存储的是typeinfo的信息，可以通过该信息类识别类的RTTI信息。
&emsp;&emsp;C++中虚类的继承关系比较复杂，这里不会深究，建议深入了解下类的对象模型，了解下类对象是如何布局的。另外，需要注意的是clang和msvc的虚类布局不同。这里只简单看下普通虚类的结构。

**==实验==**
```
int main(int argc, char **argv) {
00F71040  push        ebp  
00F71041  mov         ebp,esp  
00F71043  sub         esp,0Ch  
	myclass *cls = new myclass;
00F71046  push        8  
00F71048  call        operator new (0F71096h)  
00F7104D  add         esp,4  
00F71050  mov         dword ptr [ebp-4],eax  
00F71053  cmp         dword ptr [ebp-4],0  
00F71057  je          main+26h (0F71066h)  
00F71059  mov         ecx,dword ptr [ebp-4]  
00F7105C  call        myclass::myclass (0F71020h)  
00F71061  mov         dword ptr [ebp-8],eax  
00F71064  jmp         main+2Dh (0F7106Dh)  
00F71066  mov         dword ptr [ebp-8],0  
00F7106D  mov         eax,dword ptr [ebp-8]  
00F71070  mov         dword ptr [cls],eax  
	cls->func();
00F71073  mov         ecx,dword ptr [cls]   ;类首地址，即虚函数表的地址
00F71076  mov         edx,dword ptr [ecx]   ;通过虚函数表地址找到虚函数表指针
00F71078  mov         ecx,dword ptr [cls]   ;this入参
00F7107B  mov         eax,dword ptr [edx]   ;找打虚函数表的第一项，如果有两项，第二项就是[edx + 4]
00F7107D  call        eax  
	return 0;
00F7107F  xor         eax,eax  
}
00F71081  mov         esp,ebp  
00F71083  pop         ebp  
00F71084  ret 
```

```cpp
class myclass {
public:
	virtual void func() {}
	int a;
};

int main(int argc, char **argv) {
	myclass *cls = new myclass;
	cls->func();
	return 0;
}

```
### 9.6 多继承和多重继承
&emsp;&emsp;建议直接看C++对象模型更直观。

## 10 异常处理
&emsp;&emsp;识别异常处理：
1. 在函数入口处设置异常回调函数，回调函数先将```eax```设置为```FuncInfo```数据的地址，然后跳往```___CxxFrameHandler```。
2. 异常的抛出由```__CxxThrowException```函数完成，该函数使用了两个参数，一个是抛出异常的关键字```throw```的参数指针，另一个是抛出信息类型的指针（```ThrowInfo*```）。
3. 在异常回调函数中 ，可以得到异常对象的地址和对应```ThrowInfo```数据的地址以及```FunInfo```表结构的地址。根据记录的异常类型，进行```try```块的匹配工作。
4. 如果没有找到```try```块，则析构异常对象，返回```ExceptionContinueSearch```，继续下一个异常回调函数的处理。
5. 当找到对应的```try```块时，通过```TryBlockMapEntry```表结构中的```pCatch```指向```catch```信息表，用```ThrowInfo```表结构中的异常类型遍历查找与之匹配的```catch```块，比较关键字名称（如整型为```.h```，单精度浮点为```.m```)，找到有效的```catch```块。
6. 执行栈展开操作，产生```catch```块中使用的异常对象（有4种不同的产生方法）。
7. 正确析构所有生命周期已结束的对象。
8. 跳转到```catch```块，执行```catch```块代码。
9. 调用```_JumpToContinuation```函数，返回所有catch语句块的结束地址

## 参考文献
- [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754)
- [x86-instructions](https://shell-storm.org/x86doc/)
- [Fraction Converter](https://www.toolhelper.cn/Digit/FractionConvert)
- [MMX](https://zh.wikipedia.org/wiki/MMX)
- [精简指令集](https://zh.wikipedia.org/wiki/%E7%B2%BE%E7%AE%80%E6%8C%87%E4%BB%A4%E9%9B%86%E8%AE%A1%E7%AE%97%E6%9C%BA)
- [分支预测](https://zh.wikipedia.org/zh-hans/%E5%88%86%E6%94%AF%E9%A0%90%E6%B8%AC%E5%99%A8)