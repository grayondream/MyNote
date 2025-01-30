# 理解PLT表和GOT表

## 1 简介
&emsp;&emsp;现代操作系统都是通过库来进行代码复用，降低开发成本提升系统整体效率。而库主要分为两种，一种是静态库，比如windows的```.lib```文件，macos的```.a```，linux的```.a```，另一种是动态库，比如windows的```dll```文件，macos的```.dylib```，linux的```so```。静态库本身就是中间产物的ar打包link阶段会参与直接的产物生成，而动态库本身已经是完整的二进制文件，link阶段只会进行符号定位。
&emsp;&emsp;传统意义上认为静态链接的函数加载和执行效率要高于动态链接，这是由于静态链接在编译-链接阶段就能够确定函数的库入口地址。而动态链接并不是所有场景下都能够提前知道入口地址，可能只有需要加载的时候才需要确定。为了实现这一点，和提升加载效率，便诞生了PLT和GOT表。

## 2 PLT表和GOT表
### 2.1 GOT表
&emsp;&emsp;GOT（Global Offset Table，全局偏移表）是为了实现地址无关代码而引入的一个偏移表格。函数调用或者数据访问时先访问GOT表，再通过该表中对应表项的偏移在动态库映射内存中找到具体的函数地址和数据地址。

&emsp;&emsp;**为什么需要使用GOT表进行重定位？**
1. 动态库需要生成地址无关代码方便动态库加载时定位函数地址或者数据地址，否则动态库的动态共享的优势不再存在，因此需要生成地址无关代码；
2. GOT表格存储在数据区而不属于代码段，这样可以保证各个进程各自持有一份各自的GOT表根据自己的内存映射地址进行调整。

&esmp;&esmp;**GOT表需要考虑哪些内容？**
1. 模块间数据和函数地址访问。模块内不需要考虑，模块内使用模块内相对偏移即可。
2. 全局数据，比如```extern```表示的数据。

### 2.2 PLT表
&emsp;&emsp;PLT(Procedure Link Table，过程绑定表)是为了实现延迟绑定的地址表格。由于动态链接是在运行期链接并且进行重定位，本来直接访问的内存可能变成间接访问，会导致性能降低。ELF采用延迟绑定来缓解性能问题，其假设就是动态库中并不是所有的函数与数据都会用到，类似copy-on-write，仅仅在第一次符号被使用时才进行相关的重定位工作，避免对一些不必要的符号的重定位。
&emsp;&emsp;ELF使用PLT（Procedure Linkage Table）实现延迟绑定。在进行重定位时每个符号需要了解符号在那个模块（模块ID）以及符号。当调用外部模块中的函数时，PLT为每个外部函数符号添加了PLT项，然后通过PLT项跳转到GOT表再到最终的函数地址。也就是说第一次调用会间接调用，之后可以直接通过PLT确认调用地址调用。

&emsp;&emsp;PLT解决了哪些性能问题？
1. 符号解析。动态库加载时不需要加载所有符号，只需要加载部分能够大幅度降低加载耗时；
2. 避免重复解析。当外部调用动态库内函数或者访问数据地址时需要搜索符号表访问找到对应的项，对于比较大的动态库这个过程比较耗时。通过延迟加载只会在第一次比较耗时，之后不会重复解析；

## 3 深入理解GOT和PLT
&emsp;&emsp;我们简单做个试验研究下PLT和GOT。下面是一段简单的代码，我们将其编译成动态库```libadd.so```

```cpp
#include <cstdio>
#include <cmath>

static int myadd(const int a, const int b){
    return a + b;
}

int myabs(const int a){
    return std::abs(a);
}

void test(const int a, const int b){
    printf("%d %d", myadd(a, b), myabs(a));
}

```

&emsp;&emsp;下面是生成的动态库的反汇编：
```
0000000000000630 <_Z5myabsi>:
 630:	55                   	push   %rbp
 631:	48 89 e5             	mov    %rsp,%rbp
 634:	89 7d fc             	mov    %edi,-0x4(%rbp)
 637:	8b 45 fc             	mov    -0x4(%rbp),%eax
 63a:	89 c1                	mov    %eax,%ecx
 63c:	f7 d9                	neg    %ecx
 63e:	0f 49 c1             	cmovns %ecx,%eax
 641:	5d                   	pop    %rbp
 642:	c3                   	retq   
 643:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
 64a:	84 00 00 00 00 00 

0000000000000650 <_Z4testii>:
 650:	55                   	push   %rbp
 651:	48 89 e5             	mov    %rsp,%rbp
 654:	48 83 ec 10          	sub    $0x10,%rsp
 658:	89 7d fc             	mov    %edi,-0x4(%rbp)
 65b:	89 75 f8             	mov    %esi,-0x8(%rbp)
 65e:	8b 7d fc             	mov    -0x4(%rbp),%edi
 661:	8b 75 f8             	mov    -0x8(%rbp),%esi
 664:	e8 27 00 00 00       	callq  690 <_ZL5myaddii>
 669:	89 45 f4             	mov    %eax,-0xc(%rbp)
 66c:	8b 7d fc             	mov    -0x4(%rbp),%edi
 66f:	e8 bc fe ff ff       	callq  530 <_Z5myabsi@plt>
 674:	8b 75 f4             	mov    -0xc(%rbp),%esi
 677:	89 c2                	mov    %eax,%edx
 679:	48 8d 3d 2d 00 00 00 	lea    0x2d(%rip),%rdi        # 6ad <_fini+0x9>
 680:	b0 00                	mov    $0x0,%al
 682:	e8 99 fe ff ff       	callq  520 <printf@plt>
 687:	48 83 c4 10          	add    $0x10,%rsp
 68b:	5d                   	pop    %rbp
 68c:	c3                   	retq   
 68d:	0f 1f 00             	nopl   (%rax)

0000000000000690 <_ZL5myaddii>:
 690:	55                   	push   %rbp
 691:	48 89 e5             	mov    %rsp,%rbp
 694:	89 7d fc             	mov    %edi,-0x4(%rbp)
 697:	89 75 f8             	mov    %esi,-0x8(%rbp)
 69a:	8b 45 fc             	mov    -0x4(%rbp),%eax
 69d:	03 45 f8             	add    -0x8(%rbp),%eax
 6a0:	5d                   	pop    %rbp
 6a1:	c3                   	retq   
```

&emsp;&emsp;从上面能够看到对于内部函数的调用直接使用的内部偏移，比如```myadd2```中调用```myadd```就是```callq  690 <_ZL5myaddii>```。而调用```printf```和```myabs```就是```callq  520 <printf@plt>```和```callq  530 <_Z5myabsi@plt>```。
&emsp;&emsp;下来我们分析下这个跳转指令。e8表示偏移跳转，后面跟的就是跳转地址偏移，即```0xfffffebc```，实际跳转地址便是```off + rip=0xfffffebc + 674=0x530```。（需要注意的是执行 callq 指令之前，RIP 指向 callq 指令的下一条指令，因此RIP是674）。
```
66f:	e8 bc fe ff ff       	callq  530 <_Z5myabsi@plt>
```
&emsp;&emsp;接下来我们找到```0x530```的地址能够看到该地址又跳转到了510即plt表项，最终跳转到```0x200af2(%rip)```从注释中能够看到是GOT的表项。
```
0000000000000510 <.plt>:
 510:	ff 35 f2 0a 20 00    	pushq  0x200af2(%rip)        # 201008 <_GLOBAL_OFFSET_TABLE_+0x8>
 516:	ff 25 f4 0a 20 00    	jmpq   *0x200af4(%rip)        # 201010 <_GLOBAL_OFFSET_TABLE_+0x10>
 51c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000000520 <printf@plt>:
 520:	ff 25 f2 0a 20 00    	jmpq   *0x200af2(%rip)        # 201018 <printf@GLIBC_2.2.5>
 526:	68 00 00 00 00       	pushq  $0x0
 52b:	e9 e0 ff ff ff       	jmpq   510 <.plt>

0000000000000530 <_Z5myabsi@plt>:
 530:	ff 25 ea 0a 20 00    	jmpq   *0x200aea(%rip)        # 201020 <_Z5myabsi@@Base+0x2009f0>
 536:	68 01 00 00 00       	pushq  $0x1
 53b:	e9 d0 ff ff ff       	jmpq   510 <.plt>
```

&emsp;&emsp;接下来要查看GOT需要运行时查看，我们用GDB调试即可。首先在调用```myabs```的地方断点，单步进入，可以看到当前的代码：
```
(gdb) x /10i $pc
=> 0x7fffff1f0530 <_Z5myabsi@plt>:      jmpq   *0x200aea(%rip)        # 0x7fffff3f1020
   0x7fffff1f0536 <_Z5myabsi@plt+6>:    pushq  $0x1
   0x7fffff1f053b <_Z5myabsi@plt+11>:   jmpq   0x7fffff1f0510
   0x7fffff1f0540 <__cxa_finalize@plt>: jmpq   *0x200a9a(%rip)        # 0x7fffff3f0fe0
   0x7fffff1f0546 <__cxa_finalize@plt+6>:       xchg   %ax,%ax
```
&emsp;&emsp;从上面的代码中能够看到需要跳转的地址是```RIP+off=0x7fffff1f0536+0x200aea=0x7fffff3f1020```。从下面的内容可以看到这个地址存储的是当前指令下一条执行的地址，即```0x7fffff1f0536```，也就是说这不是真正的函数地址还没有重定位。而上面的```push $0x1```就是预期这个符号在plt中的槽位编号。
```
(gdb) x /gx 0x7fffff3f1020
0x7fffff3f1020: 0x00007fffff1f0536
(gdb) x /gx 0x00007fffff1f0536
0x7fffff1f0536 <_Z5myabsi@plt+6>:       0xffd0e90000000168
```

&emsp;&emsp;我们再单步几次就能看到基本能够确认这个过程是在进行符号解析：
```
(gdb) si
_dl_runtime_resolve_xsavec () at ../sysdeps/x86_64/dl-trampoline.h:71
71      ../sysdeps/x86_64/dl-trampoline.h: No such file or directory.
(gdb) x /3i $pc
=> 0x7fffff4178f0 <_dl_runtime_resolve_xsavec>: push   %rbx
   0x7fffff4178f1 <_dl_runtime_resolve_xsavec+1>:       mov    %rsp,%rbx
   0x7fffff4178f4 <_dl_runtime_resolve_xsavec+4>:       and    $0xffffffffffffffc0,%rsp
```
&emsp;&emsp;退出当前函数，我们再看PLT表中的表项，可以看到已经被修改为```_Z5myabsi```的函数地址了。
```
(gdb) disass '_Z5myabsi@plt'
Dump of assembler code for function _Z5myabsi@plt:
   0x00007fffff1f0530 <+0>:     jmpq   *0x200aea(%rip)        # 0x7fffff3f1020
   0x00007fffff1f0536 <+6>:     pushq  $0x1
   0x00007fffff1f053b <+11>:    jmpq   0x7fffff1f0510
End of assembler dump.
(gdb) x /gx 0x7fffff3f1020
0x7fffff3f1020: 0x00007fffff1f0630
(gdb) x /gx 0x00007fffff1f0630
0x7fffff1f0630 <_Z5myabsi>:     0x8bfc7d89e5894855
```

## 4 总结
&emsp;&emsp;PLT 和 GOT 是现代动态链接的核心机制，通过延迟绑定和地址无关性，提升了动态库的加载效率和灵活性。这些机制确保了代码复用及共享的优势，同时优化了性能。