# Windows DLL共享库构建和管理
>&emsp;&emsp;Linux下的共享库为so文件为ELF格式，Windows下的共享库为dll文件PE格式。
## 1 dll简介
&emsp;&emsp;windows下采用dll共享对象让程序更加模块化方便升级，大多数情况为```.dll```文件，也可以是``.ocx,.CPL```文件。

### 1.1 进程地址空间和内存管理
&emsp;&emsp;早期的windows中的进程并没有独立的地址空间，32bit的windows开始进程才有独立的地址空间，一个dll在不同进程中拥有不同的私有数据副本。但是和elf不同dll中的代码并不是地址无关的，只是在某些情况下被过个进程共享。

### 1.2 基地址和RVA。
&emsp;&emsp;基地址：就是一个PE文件被装载时，其进程空间的起始地址。
&emsp;&emsp;RVA（相对虚拟地址）：也就是相对于基地址的偏移地址。

### 1.3 DLL共享数据段
&emsp;&emsp;dll允许进程将一部分数据设置为共享的，也就是一个进程包含两个数据段，一个进程成共享的数据段，另一个私有。可以利用dll共享数据段进行进程间通信。

### 1.4 dll的简单例子
&emsp;&emsp;elf文件默认是导出所有符号，而dll不同，默认不导出符号，需要显式指定某个需要导出的符号。而相对的在程序中使用dll的导出符号的过程为导入。mscv编译期可以通过```__declspec(dllexport)```和```__declspec(dllimport)```指定导出和导入的符号。
&emsp;&emsp;另外，也可以使用```.def```文件声明导入和导出的符号，改文件类似一个链接器的脚本。

### 1.5 创建dll
&emsp;&emsp;对于下面的```math.c```文件使用命令```cl /LDd math.c```便可以得到四个```math.dll,math.obj,math.exp,math.lib```文件。可以通过```dumpbin```查看dll的导出符号。

```cpp
__declspec(dllexport) int add(int a, int b){
    return a + b;
}

__declspec(dllexport) int sub(int a, int b){
    return a - b;
}
```

```bash
E:\code\tmp>dumpbin /EXPORTS math.dll
Microsoft (R) COFF/PE Dumper Version 14.16.27045.0
Copyright (C) Microsoft Corporation.  All rights reserved.


Dump of file math.dll

File Type: DLL

  Section contains the following exports for math.dll

    00000000 characteristics
    FFFFFFFF time date stamp
        0.00 version
           1 ordinal base
           2 number of functions
           2 number of names

    ordinal hint RVA      name

          1    0 00001000 add
          2    1 00001020 sub

  Summary

        3000 .data
        3000 .pdata
       12000 .rdata
        1000 .reloc
       37000 .text
```

### 1.6 使用dll
&emsp;&emsp;使用dll的过程就是导入的过程，只需要将相关符号显示生命为导入符号，然后在编译时链接即可。
```cpp
#include <stdio.h>

__declspec(dllimport) int add(int a, int b);

int main(){
    int result = add(1, 2);
    printf("value = %d\n", result);
    return 0;
}
```

&emsp;&emsp;能够看到这里链接的是```math.lib```而不是```math.dll```。只是因为链接时仅仅需要符号相关信息，这里的```math.lib```仅仅包含链接时需要的信息（通过查看文件大小也能看出区别，lib文件仅仅2kb）。

```bash
E:\code\tmp>cl -c main.c
用于 x64 的 Microsoft (R) C/C++ 优化编译器 19.16.27045 版
版权所有(C) Microsoft Corporation。保留所有权利。

main.c

E:\code\tmp>link main.obj math.lib
Microsoft (R) Incremental Linker Version 14.16.27045.0
Copyright (C) Microsoft Corporation.  All rights reserved.

E:\code\tmp>main.exe
value = 3
```

### 1.7 使用模块定义文件
&emsp;&emsp;dll可以通过```__declspec(dllexport)```定义导出符号，也可以通过```.def```文件定义。```.def```文件类似linux的链接脚本，用于控制链接过程，为链接器提供有关链接程序的导出符号、属性以及其他信息。
&emsp;&emsp;下面分别是```math.c```和```math.def```的内容。
```cpp
int add(int a, int b){
    return a + b;
}

int sub(int a, int b){
    return a - b;
}
```

```bash
LIBRARY math
EXPORTS
add
sub
```

&emsp;&emsp;然后通过下面的命令生成动态库。
```bash
E:\code\tmp>cl math.c -LD /DEF math.def
用于 x64 的 Microsoft (R) C/C++ 优化编译器 19.16.27045 版
版权所有(C) Microsoft Corporation。保留所有权利。

math.c
Microsoft (R) Incremental Linker Version 14.16.27045.0
Copyright (C) Microsoft Corporation.  All rights reserved.

/out:math.dll
/dll
/implib:math.lib
/def:math.def
math.obj
  正在创建库 math.lib 和对象 math.exp
```
&emsp;&emsp;使用```.def```可以很方便的控制链接过程，自定义链接中的一些规则。比如指定导出符号修饰方式、堆大小、文件名、段属性等等。

### 1.8 dll显示运行时链接
&emsp;&emsp;类似于elf，dll提供了类似的api可以在运行时链接动态库。
- ```LoadLibrary```：加载动态库；
- ```GetProcAddress```：获取某个符号的地址
- ```FreeLibrary```：卸载已经加载的模块。

```cpp
#include <stdio.h>
#include <windows.h>

typedef int (*myfunction)(int, int);

int main(){
    HINSTANCE id = LoadLibrary("math.dll");
    if(id == NULL){
        printf("无法加载动态库\n");
        exit(1);
    }

    myfunction function = (myfunction)GetProcAddress(id, "add");
    if(function == NULL){
        printf("无法找到库函数\n");
        exit(1);
    }

    int result = function(11, 2);
    printf("handle is %p, address is %p, value = %d\n", id, function, result);
    FreeLibrary(id);
    return 0;
}
```
&emsp;&emsp;下面是运行的结果。
```bash
E:\code\tmp>cl main.c
用于 x64 的 Microsoft (R) C/C++ 优化编译器 19.16.27045 版
版权所有(C) Microsoft Corporation。保留所有权利。

main.c
Microsoft (R) Incremental Linker Version 14.16.27045.0
Copyright (C) Microsoft Corporation.  All rights reserved.

/out:main.exe
main.obj

E:\code\tmp>main.exe
handle is 00007FFAA3D80000, address is 00007FFAA3D81000, value = 13
```

## 2 符号导出导入表
### 2.1 导出表
&emsp;&emsp;windows上存在和linux类似的导出表，导出表用来收集导出的符号，存储符号名和符号地址的映射关系。PE文件Header中又一个```DataDirectory```的结构数组，该数组的第一个元素就是导出表的结构地址和长度。导出表是一个定义在```Winnt.h```中的```_IMAGE_EXPORT_DIRECTORY```结构体。该结构的最后三个成员是三个指针，分别表示三个数组，分别是导出地址表EAT（即对应的符号地址表）、符号名表和名字序号对照表。其中名字序号对照表有一定历史原因，不做详述。
```cpp
    typedef struct _IMAGE_EXPORT_DIRECTORY {
      DWORD Characteristics;
      DWORD TimeDateStamp;
      WORD MajorVersion;
      WORD MinorVersion;
      DWORD Name;
      DWORD Base;
      DWORD NumberOfFunctions;
      DWORD NumberOfNames;
      DWORD AddressOfFunctions;
      DWORD AddressOfNames;
      DWORD AddressOfNameOrdinals;
    } IMAGE_EXPORT_DIRECTORY,*PIMAGE_EXPORT_DIRECTORY;
```
> 可以给链接器使用```/EXPORTS```传递参数表明希望导出的符号。

### 2.2 exp文件
&emsp;&emsp;在创建dll的同时也会得到一个EXP文件,这个文件实际上是链接器在创建dll时的临时文件。链接器在创建dll时与静态链接时一样采用两遍扫描过程，dll一般都有导出符号，链接器在第一遍时会遍历所有的目标文件并且收集所有导出符号信息并且创建dll的导处表。为了方便起见，链接器把这个导出表放到一个临时的目标文件叫做```.edata```的段中，这个目标文件就是EXP文件,EXP文件实际上是一个标准的PE/COFF目标文件，只不过它的扩展名不是.obj而是.exp。在第二遍时，链接器就把这个EXP文件当作普通目标文件一样，与其他输入的目标文件链接在一起并且输出DLL。这时候EXP文件中的```.edata```段也就会被输出到DLL文件中并且成为导出表。不过一般现在链接器很少会在DLL中单独保留```.edata```段，而是把它合并到只读数据段```rdata```中。

### 2.3 导出重定向
&emsp;&emsp;将某个导出符号重定向到另外一个dll。导出表的地址数组中包含的是函数的RVA，但是如果这个RVA指向的位置位于导出表中，那么表示这个符号被重定向了。

### 2.4 导入表
&emsp;&emsp;windows上有导入表保存模块所需要导入的符号等信息，Windows加载器会将所需要导入的函数地址确定并且将导入表中的元素调整到正确的地址，以实现动态链接。我们可以使用```dumpbin /IMPORTS main.dll```查看导入的符号，下面只是一部分输出，可以看出一些是程序运行必须的系统库，比如kernel32.dll,ntdll.dll等。
```bash
E:\code\tmp>dumpbin /IMPORTS math.dll
Microsoft (R) COFF/PE Dumper Version 14.16.27045.0
Copyright (C) Microsoft Corporation.  All rights reserved.


Dump of file math.dll

File Type: DLL

  Section contains the following imports:

    KERNEL32.dll
             18000C000 Import Address Table
             180014148 Import Name Table
                     0 time date stamp
                     0 Index of first forwarder reference

                         450 QueryPerformanceCounter
                         21E GetCurrentProcessId
                         222 GetCurrentThreadId
                         2F0 GetSystemTimeAsFileTime
```

&emsp;&emsp;在PE文件中，导入表是一个```_IMAGE_IMPORT_DESCRIPTOR```的结构体数组（定义在```winnt.h```中），每个```_IMAGE_IMPORT_DESCRIPTOR```对应一个被导入的dll。其中```FirstThunk```是一个导入地址数组的首地址，即导入地址表（IAT），该数组的每个元素就是一个被导入的符号，该元素的含义在不同情况下不同。当链接器刚开始重定位和符号解析时，该表中元素值表示对应的符号名或者序号；当解析完成时，则表示符号的真正地址。对于32bitPE，当元素值的高1bit被置为1时，低31bit表示序号；当为0时指向一个```_IMAGE_IMPORT_BY_NAME```的结构体表示符号名。另外```OriginalFirstThunk```是一个导入名称表（INT）保存的内容和导入符号表相同。

```cpp
typedef struct _IMAGE_IMPORT_DESCRIPTOR {
union {
DWORD Characteristics; // 0 for terminating null import descriptor
DWORD OriginalFirstThunk; // RVA to original unbound IAT (PIMAGE_THUNK_DATA)
//导入名称表，简称INT。和IAT一样
} DUMMYUNIONNAME;
DWORD TimeDateStamp; // 0 if not bound,
// -1 if bound, and real date\time stamp
// in IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT (new BIND)
// O.W. date/time stamp of DLL bound to (Old BIND)

DWORD ForwarderChain; // -1 if no forwarders
DWORD Name;
DWORD FirstThunk; // RVA to IAT (if bound this IAT has actual addresses)
//指向一个导入地址数组，IAT是导入表中最重要的结构，IAT中每个元素对应一个被导入的符号，元素的值在不同情况下有不同的含义。
} IMAGE_IMPORT_DESCRIPTOR;

typedef struct _IMAGE_IMPORT_BY_NAME {
    WORD Hint;
    BYTE Name[1];
} IMAGE_IMPORT_BY_NAME,*PIMAGE_IMPORT_BY_NAME;
```
> &emsp;&emsp;windows通过提供延迟载入，仅仅在符号第一次被使用时载入。

### 2.5 导入函数的调用
&emsp;&emsp;PE采用类似ELF GOT的方式间接访问一个函数，但是链接器无法判断一个符号的可见性。msvc使用```__desclspec(import)```来判断函数是否为外部导入，以便于生成相应的代码。另一种方式是不区分导入和导出函数，统一产生调用的指令，链接器在链接时会将导入函数的目标地址导向一小段桩代码，由桩代码将控制权移交给IAT真正的地址。然而实际上链接器不会生成代码，只负责链接，这些代码来自于lib文件，即导入库。当编译器生成导入库时，同一个导出符号会长生两个符号的定义，即```func```和```__imp__func```，前者指向桩代码，后者指向包含的IAT中的位置。但是明显的使用```__desclspec(import)```的方式少一条跳转指令，性能上更友好。

## 3 dll优化
&emsp;&emsp;DLL的代码段和数据段本身并不是地址无关的，它默认需要被装载到由ImageBase指定的目标地址中。如果目标地址被占用，那么就需要装载到其他地址，便会引起整个DLL的Rebase。符号和字符串的比较和查找过程也会影响DLL性能。

### 3.1 重定基地址（Rebasing）
&emsp;&emsp;PE的DLL中的代码段并不是地址无关的，它在被装载时有一个固定的目标地址，这个地址也就是PE里面所谓的基地址。对于DLL来说，一个进程中，多个DLL不可以被装载到同一个虚拟地址，每个DLL所占用的虚拟地址区域之间都不可以重叠。
&emsp;&emsp;Windows PE采用装载时重定位：在DLL模块装载时，如果目标地址被占用，那么操作系统就会为它分配一块新的空间，并且将DLL装载到该地址。对于DLL每个绝对地址引用都进程重定位。
由于DLL内部地址都是基于基地址的，或者是相对于基地址的RVA。那么所有需要重定位的地址都只需要加上一个固定差值。PE里面把这种特殊的重定位过程叫做重定基地址。EXE是不可以重定位的，不过这也没问题，因为EXE文件是进程运行时第一个装入虚拟空间的，所以它的地址不会被人抢占。
>&emsp;&emsp;```link```可以通过参数指定装载的地址，```editbin```可以改变已有dll的基地址。
>&emsp;&emsp;Windows系统在进程空间中专门划出一块0x70000000~0x80000000区域，用于映射这些常用的系统DLL。

### 3.2 序号
&emsp;&emsp;一个DLL中每一个导出的函数都有一个对应的序号。序号标示被导出函数地址在DLL导出表中位置。一个导出函数甚至没有函数名，但它必须有唯一的序号。当从一个dll导入函数时可以使用函数名也可以使用序号。不同windows版本之间api的函数名可能没有变化，但是序号是在不停的变化。另外我们可以在```def```文件中指定导出符号的序号。

### 3.3 导入符号绑定
&emsp;&emsp;当一个程序运行时，所有被依赖的DLL都会被装载，并且一系列导入导出符号依赖关系都会被重新解析。这些DLL都会以同样的顺序被装载到同样的内存地址，所以它们的导出符号的地址都是不变的。将这些导出函数的地址保存到模块的导入表中的过程就叫做dll绑定。dll绑定能够省去启动时符号解析的过程提升性能。
>&emsp;&emsp;可以利用```editbin```完成dll绑定。
&emsp;&emsp;dll的绑定实现：editbin对被绑定的程序的导入符号进行遍历查找，找到以后就把符号的运行时的目标地址写入到绑定程序的导入表内。
&emsp;&emsp;dll绑定地址失效的情况：
1. 被依赖的dll更新导致dll的导出函数地址发生变化。解决方式：windows将dll的时间戳和md5校验和保存到绑定的pe文件的导入表中，运行时如果校验失败则走正常流程导入；
2. 被依赖的dll在装载时发生重定基址导致dll的装载地址与绑定时不一致。

## 4 C++与动态链接
&emsp;&emsp;C++编写动态链接库在Windows平台下最好遵循以下指导：
- 所有的接口都应该抽象；
- 所有的全局函数都应该使用”extern C”来防止名字修饰的不兼容；
- 不要使用C++标准库STL；
- 不要使用异常；
- 不要使用虚析构函数；
- 不要在DLL里面申请内存；
- 不要在接口中使用重载方法。

## 5 dll hell
&emsp;&emsp;由于早期Windows缺乏一种很有效的DLL版本控制机制，DLL不兼容文件在Windows非常严重，被人们称为DLL噩梦(DLL hell)。DLL HELL发生的三种可能原因：
- 由使用旧版本的DLL替代原来一个新版本的DLL引起
- 由新版DLL中的函数无意发生改变而引起
- 由新版DLL按照引入一个新BUG
&emsp;&emsp;解决DLL Hell的方法：
- 静态链接；
- 防止DLL覆盖：使用windows保护技术；
- 避免DLL冲突：每个应用程序拥有一份自己依赖的DLL。

&emsp;&emsp;.NET 下DLL Hell的解决方案：在.NET框架中，一个程序集有两种类型：应用程序集以及库程序。一个程序集包块一个或多个文件，所以需要一个清单文件来描述程序集。这个清单文件叫做Manifest文件。Manifest文件描述了程序集的名字，版本号以及程序集的各种资源，同时也描述了该程序集的运行所依赖的资源，包括DLL以及其他资源文件等。操作系统会根据DLL的manifest文件去寻找对应的DLL并调用。
