# Windows和Linux共享库的入口

> 主要关注程序的启动过程。

## 1 入口函数和程序初始化
### 1.1 程序真正的入口
&emsp;&emsp;通常写代码时，我们认为程序的入口是```main```函数，但是实际上有一些现象值得我们怀疑该结论是不是正确的。比如全局变量的初始化，C++全局变量构造函数的调用，C++静态对象的构造函数调用。
&emsp;&emsp;程序开始运行的入口不是```main```，在进入```main```之前程序会先准备好环境运行一些必要的代码才进入```main```，运行这些代码的函数称为入口函数。入口函数根据平台的不同而不同，其实实际上程序初始化和结束的地方，一个典型的程序的运行步骤如下：
- 操作系统创建进程后，将控制权移交给程序入口，这个入口一般为运行库的某个入口函数；
- 入口函数对运行库和程序运行环境进行初始化，比如堆栈、IO、线程、全局变量构造等；
- 入口函数在完成初始化之后调用main函数，开氏运行程序主体部分；
- main函数执行结束后，返回到入口函数，入口函数进行清理工作，比如全局变量的析构，堆销毁，关闭IO等，然后结束进程。

### 1.2 入口函数实现
&emsp;&emsp;主要关注glibc静态库用于可执行文件的情况。
#### 1.2.1 glibc入口函数
&emsp;&emsp;程序的启动代码在glibc源代码的```glibc-2.33\sysdeps\i386\start.S```中，下面是i386的实现的简化代码，从代码中能够看到最终调用了```__libc_start_main```。
```cpp
ENTRY (_start)
	xorl %ebp, %ebp		!寄存器清零
	popl %esi			!此时esi就是argc的值
	movl %esp, %ecx		!此时栈顶的一部分就是argv，ecx指向第一个参数的栈地址
	andl $0xfffffff0, %esp
	pushl %eax		/* Push garbage because we allocate 28 more bytes.  */
	pushl %esp
	pushl %edx		/* Push address of the shared library termination function.  */

#ifdef PIC
	/* Load PIC register.  */
	call 1f
	addl $_GLOBAL_OFFSET_TABLE_, %ebx
	/* Push address of our own entry points to .fini and .init.  */
	leal __libc_csu_fini@GOTOFF(%ebx), %eax
	pushl %eax
	leal __libc_csu_init@GOTOFF(%ebx), %eax
	pushl %eax

	pushl %ecx		/* Push second argument: argv.  */
	pushl %esi		/* Push first argument: argc.  */

## ifdef SHARED
	pushl main@GOT(%ebx)
## else
	/* Avoid relocation in static PIE since _start is called before
	   it is relocated.  Don't use "leal main@GOTOFF(%ebx), %eax"
	   since main may be in a shared object.  Linker will convert
	   "movl main@GOT(%ebx), %eax" to "leal main@GOTOFF(%ebx), %eax"
	   if main is defined locally.  */
	movl main@GOT(%ebx), %eax
	pushl %eax
## endif
	call __libc_start_main@PLT
#else
	/* Push address of our own entry points to .fini and .init.  */
	!下面是传入__libc_start_main的参数
	pushl $__libc_csu_fini
	pushl $__libc_csu_init
	pushl %ecx		/* Push second argument: argv.  */
	pushl %esi		/* Push first argument: argc.  */
	pushl $main
	call __libc_start_main //	/* Call the user's main function, and exit with its value. But let the libc call main.    */
#endif

	hlt			/* Crash if somehow `exit' does return.  */
```
&emsp;&emsp;```__libc_start_main```实际的定义如下：
- ```main```：就是main函数的指针；
- ```argc```：参数个数；
- ```argv```：参数数组；
- ```init```：调用前初始化工作；
- ```fini```：调用结束的收尾工作；
- ```rtld_fini```：和动态加载相关的收尾工作；
- ```stack_end```：表明栈底的地址。

```cpp
STATIC int
LIBC_START_MAIN (int (*main) (int, char **, char ** MAIN_AUXVEC_DECL),
		 int argc, char **argv,
#ifdef LIBC_START_MAIN_AUXVEC_ARG
		 ElfW(auxv_t) *auxvec,
#endif
		 __typeof (main) init,
		 void (*fini) (void),
		 void (*rtld_fini) (void), void *stack_end)
```

&emsp;&emsp;下面是省略了部分代码的```__libc_start_main```的实现，其中省略了大部分初始化的代码，我们只关注上面传入的参数的执行情况。
- 首先是设置环境变量的指针```__environ```，环境变量在栈中位于argv后面；
- 之后是设置退出时执行的相关函数指针；
- 之后运行```init```函数进行初始化；
- 最后执行```main```，退出程序。

```cpp
/* Note: the fini parameter is ignored here for shared library.  It
   is registered with __cxa_atexit.  This had the disadvantage that
   finalizers were called in more than one place.  */
STATIC int
LIBC_START_MAIN (int (*main) (int, char **, char ** MAIN_AUXVEC_DECL),
		 int argc, char **argv,
#ifdef LIBC_START_MAIN_AUXVEC_ARG
		 ElfW(auxv_t) *auxvec,
#endif
		 __typeof (main) init,
		 void (*fini) (void),
		 void (*rtld_fini) (void), void *stack_end)
{
  /* Result of the 'main' function.  */
  int result;
  char **ev = &argv[argc + 1];
  __environ = ev;
  /* Store the lowest stack address.  This is done in ld.so if this is the code for the DSO.  */
  __libc_stack_end = stack_end;
  //省略部分初始化代码

## ifdef DL_SYSDEP_OSCHECK
  {
    /* This needs to run to initiliaze _dl_osversion before TLS
       setup might check it.  */
    DL_SYSDEP_OSCHECK (__libc_fatal);
  }
## endif
  /* Initialize libpthread if linked in.  */
  if (__pthread_initialize_minimal != NULL)
    __pthread_initialize_minimal ();
  /* Register the destructor of the dynamic linker if there is any.  */
  if (__glibc_likely (rtld_fini != NULL))
    __cxa_atexit ((void (*) (void *)) rtld_fini, NULL, NULL);

#ifndef SHARED
  /* Perform early initialization.  In the shared case, this function
     is called from the dynamic loader as early as possible.  */
  __libc_early_init (true);

  /* Call the initializer of the libc.  This is only needed here if we
     are compiling for the static library in which case we haven't
     run the constructors in `_dl_start_user'.  */
  __libc_init_first (argc, argv, __environ);

  /* Register the destructor of the program, if any.  */
  if (fini)
    __cxa_atexit ((void (*) (void *)) fini, NULL, NULL);
#endif
  if (init)
    (*init) (argc, argv, __environ MAIN_AUXVEC_PARAM);

  /* Nothing fancy, just call the function.  */
  result = main (argc, argv, __environ MAIN_AUXVEC_PARAM);

  exit (result);
}
```
&emsp;&emsp;我们可以简单看下```exit```的实现，exit实际的函数实体是```__run_exit_handlers```，其中又一个参数是```exit_funcs```我们可以怀疑是一个函数指针列表。从其结构体看```__exit_funcs```是一个存储注册的函数的链表：
```cpp
void exit (int status)
{
  __run_exit_handlers (status, &__exit_funcs, true, true);
}

struct exit_function_list{
    struct exit_function_list *next;
    size_t idx;
    struct exit_function fns[32];
  };
struct exit_function_list *__exit_funcs = &initial;
```

&emsp;&emsp;能够从代码中```exit```代码中会遍历注册的函数，一个一个执行，最终调用```_exit```退出程序。
```cpp
void
attribute_hidden
__run_exit_handlers (int status, struct exit_function_list **listp,
		     bool run_list_atexit, bool run_dtors)
{
  /* We do it this way to handle recursive calls to exit () made by
     the functions registered with `atexit' and `on_exit'. We call
     everyone on the list and use the status value in the last
     exit (). */
  while (true)
    {
      struct exit_function_list *cur;

      __libc_lock_lock (__exit_funcs_lock);

    restart:
      cur = *listp;
	//执行函数调用相关的代码省略
      *listp = cur->next;
      if (*listp != NULL)
	/* Don't free the last element in the chain, this is the statically
	   allocate element.  */
	free (cur);

      __libc_lock_unlock (__exit_funcs_lock);
    }

  if (run_list_atexit)
    RUN_HOOK (__libc_atexit, ());

  _exit (status);
}
```

#### 1.2.2 MSVC CRT入口函数
&emsp;&emsp;作者书中提到的系统版本比较老，我现在使用的是windows10，下面是vs2017的实现。
&emsp;&emsp;```mainCRTStartup```在```Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\crt\src\vcruntime\exe_main.cpp```中，其最终调用的是```__scrt_common_main()```，之后在完成一些```cookie```的初始化之后，调用```__scrt_common_main_seh```。下面是微软官方文档对```__security_init_cookie```的描述：
>&emsp;&emsp;[__security_init_cookie](https://docs.microsoft.com/zh-CN/cpp/c-runtime-library/reference/security-init-cookie?view=msvc-170)
>&emsp;&emsp;全局安全 Cookie 用于在使用 /GS（缓冲区安全检查）编译的代码中和使用异常处理的代码中提供缓冲区溢出保护。 进入受到溢出保护的函数时，Cookie 被置于堆栈之上；退出时，会将堆栈上的值与全局 Cookie 进行比较。 它们之间存在任何差异则表示已经发生缓冲区溢出，并导致该程序的立即终止。
>&emsp;&emsp;通常 ，__security_init_cookie 初始化时由 CRT 调用。 如果绕过 CRT 初始化（例如，如果使用/ENTRY指定入口点）则必须自己__security_init_cookie。 如果未 __security_init_cookie， 则全局安全 Cookie 将设置为默认值，缓冲区溢出保护会遭到入侵。 由于攻击者可以利用此默认 Cookie 值来阻止缓冲区溢出检查，因此，建议在定义自己的 入口点__security_init_cookie 始终调用命令。

```cpp
extern "C" int mainCRTStartup(){
    return __scrt_common_main();
}

// This is the common main implementation to which all of the CRT main functions
// delegate (for executables; DLLs are handled separately).
static __forceinline int __cdecl __scrt_common_main(){
    // The /GS security cookie must be initialized before any exception handling
    // targeting the current image is registered.  No function using exception
    // handling can be called in the current image until after this call:
    __security_init_cookie();

    return __scrt_common_main_seh();
}
```

&emsp;&emsp;下面是```__scrt_common_main_seh```的完整实现，其基本流程和glibc类似，都是先初始化环境，注册相关的函数调用，然后遍历函数指针列表并调用，之后调用```invoke_main```函数（```invoke_main```实际上就是对```main```的一个包装），最后调用```eixt```退出。
```cpp
_ACRTIMP int*       __cdecl __p___argc (void);
_ACRTIMP char***    __cdecl __p___argv (void);
_ACRTIMP wchar_t*** __cdecl __p___wargv(void);

#ifdef _CRT_DECLARE_GLOBAL_VARIABLES_DIRECTLY
    extern int       __argc;
    extern char**    __argv;
    extern wchar_t** __wargv;
#else
    #define __argc  (*__p___argc())  // Pointer to number of command line arguments
    #define __argv  (*__p___argv())  // Pointer to table of narrow command line arguments
    #define __wargv (*__p___wargv()) // Pointer to table of wide command line arguments
#endif

static int __cdecl invoke_main(){
        return main(__argc, __argv, _get_initial_narrow_environment());
}

static __declspec(noinline) int __cdecl __scrt_common_main_seh(){
    if (!__scrt_initialize_crt(__scrt_module_type::exe))
        __scrt_fastfail(FAST_FAIL_FATAL_APP_EXIT);

    bool has_cctor = false;
    __try
    {
        bool const is_nested = __scrt_acquire_startup_lock();

        if (__scrt_current_native_startup_state == __scrt_native_startup_state::initializing)
        {
            __scrt_fastfail(FAST_FAIL_FATAL_APP_EXIT);
        }
        else if (__scrt_current_native_startup_state == __scrt_native_startup_state::uninitialized)
        {
            __scrt_current_native_startup_state = __scrt_native_startup_state::initializing;
			//依次调用c的函数，__xi_a, __xi_z分别为c函数的首地址和尾地址
            if (_initterm_e(__xi_a, __xi_z) != 0)
                return 255;
			//依次调用c++的函数，__xc_a, __xc_z分别为c++函数的首地址和尾地址
            _initterm(__xc_a, __xc_z);

            __scrt_current_native_startup_state = __scrt_native_startup_state::initialized;
        }
        else
        {
            has_cctor = true;
        }

        __scrt_release_startup_lock(is_nested);

        // If this module has any dynamically initialized __declspec(thread)
        // variables, then we invoke their initialization for the primary thread
        // used to start the process:
        _tls_callback_type const* const tls_init_callback = __scrt_get_dyn_tls_init_callback();
        if (*tls_init_callback != nullptr && __scrt_is_nonwritable_in_current_image(tls_init_callback))
        {
            (*tls_init_callback)(nullptr, DLL_THREAD_ATTACH, nullptr);
        }

        // If this module has any thread-local destructors, register the
        // callback function with the Unified CRT to run on exit.
        _tls_callback_type const * const tls_dtor_callback = __scrt_get_dyn_tls_dtor_callback();
        if (*tls_dtor_callback != nullptr && __scrt_is_nonwritable_in_current_image(tls_dtor_callback))
        {
            _register_thread_local_exe_atexit_callback(*tls_dtor_callback);
        }

        //main函数的入口
        int const main_result = invoke_main();
		//退出程序
        if (!__scrt_is_managed_app())
            exit(main_result);

        if (!has_cctor)
            _cexit();

        // Finally, we terminate the CRT:
        __scrt_uninitialize_crt(true, false);
        return main_result;
    }
    __except (_seh_filter_exe(GetExceptionCode(), GetExceptionInformation()))
    {
        // Note:  We should never reach this except clause.
        int const main_result = GetExceptionCode();

        if (!__scrt_is_managed_app())
            _exit(main_result);

        if (!has_cctor)
            _c_exit();

        return main_result;
    }
}
```

### 1.3 运行库和IO
&emsp;&emsp;操作系统需要给用户提供读取相关上硬件上的文件或者设备的能力，但是又不能完全向用户开放对硬件的操作能力。在操作系统层面，对于文件的操作Linux提供了文件描述符，而windows则使用句柄来控制某个文件。
&emsp;&emsp;在Linux中，一个文件会有一个fd，每个进程维护一份私有的文件打开表，这个表格是一个指针数组，每个元素指向一个打开的文件对象，而用户拿到的fd就是该表的索引。这个表格由内核维护，用户是无法直接访问的。
&emsp;&emsp;在windows上类似，只不过windows上的句柄并不是索引号，而是通过特定的算法变换得到的一个数值。
&emsp;&emsp;另外，C中使用到的```FILE```并不是文件的真实指针，而是关于fd或者句柄相关联的一个指针。

>&emsp;&emsp;文中提到了FILE的实现，但是新版的实现已经改变，因此不在赘述。
```cpp
typedef struct _iobuf
    {
        void* _Placeholder;
    } FILE;
```

## 2 C/C++运行库
### 2.1 运行库
&emsp;&emsp;CRT(C Runtime Library，C运行时库)包含了程序能够正常运行的代码，以及相关的标准库实现等基本的内容。
&emsp;&emsp;Windows下CRT的源码目录为```Windows Kits\10\Source\10.0.17134.0\ucrt```。
&emsp;&emsp;一个C语言运行库的基本功能大致为：
- 启动和退出：包括入口函数及入口函数所依赖的其它函数；
- 标准函数：由C语言标准规定的C语言标准库所拥有的函数实现；
- IO：IO功能的封装与实现；
- 堆：堆的封装与实现；
- 语言实现：语言相关的特性实现；
- 调试：实现调试功能的代码。

### 2.2 标准库
&emsp;&emsp;不赘述，请参考[open-std-c99](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1256.pdf)。

### 2.3 glibc和MSVC CRT
&emsp;&emsp;运行库是和操作系统紧密相关的，C语言仅仅是针对不同操作系统平台的一个抽象层。glibc和MSVCCRT分别是Linux和windows平台下的C实现。
#### 2.3.1 glibc
&emsp;&emsp;glibc是Linux平台的C标准库实现，其包含了标准库的头文件和相关的二进制文件。二进制文件提供了静态和动态库，静态库位于```/usr/lib32/```下，动态库位于```/lib32/```。除了标准库外，还提供了一个运行库，比如```/usr/lib32/crt1.o  /usr/lib32/crti.o  /usr/lib32/crtn.o```。通过查看符号我们能够发现，```crt1.o```包含了入口启动函数相关的实现，而```crti.o```包含了初始化和结束的后处理相关的实现，```crti.o,crtn.o```共同组成```_init,_fini```的实现。编程语言本身编译器相关的，当然也需要包含一些gcc相关的库，具体目录在```/usr/lib/gcc/x86_64-linux-gnu/7/```下，其中```x86-64-linux-gnu/7```可以换成自己的系统版本，不再赘述。

```bash
➜  tmp nm /usr/lib32/crt1.o
00000000 D __data_start
00000000 W data_start
00000040 T _dl_relocate_static_pie
00000000 R _fp_hw
         U _GLOBAL_OFFSET_TABLE_
00000000 R _IO_stdin_used
         U __libc_csu_fini
         U __libc_csu_init
         U __libc_start_main
         U main
00000000 T _start
➜  tmp nm /usr/lib32/crti.o
00000000 T _fini
         U _GLOBAL_OFFSET_TABLE_
         w __gmon_start__
00000000 T _init
00000000 T __x86.get_pc_thunk.bx
➜  tmp nm /usr/lib32/crtn.o
nm: /usr/lib32/crtn.o: no symbols
➜  tmp objdump -dr /usr/lib32/crti.o

/usr/lib32/crti.o:     file format elf32-i386


Disassembly of section .init:

00000000 <_init>:
   0:   53                      push   %ebx
   1:   83 ec 08                sub    $0x8,%esp
   4:   e8 fc ff ff ff          call   5 <_init+0x5>
                        5: R_386_PC32   __x86.get_pc_thunk.bx
   9:   81 c3 02 00 00 00       add    $0x2,%ebx
                        b: R_386_GOTPC  _GLOBAL_OFFSET_TABLE_
   f:   8b 83 00 00 00 00       mov    0x0(%ebx),%eax
                        11: R_386_GOT32X        __gmon_start__
  15:   85 c0                   test   %eax,%eax
  17:   74 05                   je     1e <_init+0x1e>
  19:   e8 fc ff ff ff          call   1a <_init+0x1a>
                        1a: R_386_PLT32 __gmon_start__

Disassembly of section .gnu.linkonce.t.__x86.get_pc_thunk.bx:

00000000 <__x86.get_pc_thunk.bx>:
   0:   8b 1c 24                mov    (%esp),%ebx
   3:   c3                      ret

Disassembly of section .fini:

00000000 <_fini>:
   0:   53                      push   %ebx
   1:   83 ec 08                sub    $0x8,%esp
   4:   e8 fc ff ff ff          call   5 <_fini+0x5>
                        5: R_386_PC32   __x86.get_pc_thunk.bx
   9:   81 c3 02 00 00 00       add    $0x2,%ebx
                        b: R_386_GOTPC  _GLOBAL_OFFSET_TABLE_
```
#### 2.3.2 MSVC CRT
&emsp;&emsp;MSVC CRT库存储于```Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\lib\x64```，其中的```14.16.27023```可以替换为自己的版本，对应的库的名称的命名规则为```libc[p][mt][d].lib```：
- p表示C++标准库；
- mt表示支持多线程；
- d表示调试版本。

&emsp;&emsp;在编译是可以通过vs的选项选择编译的库版本，默认是```libcmt.lib```。我们随便写一个C++文件使用，cl编译，使用dumpbin查看依赖的库。从下面的输出能够看到当前的```main.obj```还依赖```libcmt,oldnames,libcpmt```三个库。
```cpp
E:\code\tmp>cl /c main.cpp
用于 x64 的 Microsoft (R) C/C++ 优化编译器 19.16.27045 版
版权所有(C) Microsoft Corporation。保留所有权利。

main.cpp
D:\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\include\xlocale(319): warning C4530: 使用了 C++ 异常处理程序，但未启用展开语义。请指定 /EHsc

E:\code\tmp>dumpbin /DIRECTIVES main.obj
Microsoft (R) COFF/PE Dumper Version 14.16.27045.0
Copyright (C) Microsoft Corporation.  All rights reserved.


Dump of file main.obj

File Type: COFF OBJECT

   Linker Directives
   -----------------
   /FAILIFMISMATCH:_MSC_VER=1900
   /FAILIFMISMATCH:_ITERATOR_DEBUG_LEVEL=0
   /FAILIFMISMATCH:RuntimeLibrary=MT_StaticRelease
   /DEFAULTLIB:libcpmt
   /FAILIFMISMATCH:_CRT_STDIO_ISO_WIDE_SPECIFIERS=0
   /DEFAULTLIB:LIBCMT
   /DEFAULTLIB:OLDNAMES
```

## 3 运行库和多线程
### 3.1 线程的问题
&emsp;&emsp;线程的访问能力非常自由,它可以访问进程内存里的所有数据,甚至包括其他线程的堆栈(如果它知道其他线程的堆栈地址,然而这是很少见的情况),但实际运用中线程也拥有自己的私有存储空间,包括:
1. 栈(尽管并非完全无法被其他线程访问,但一般情况下仍然可以认为是私有的数据)
2. 线程局部存储(Thread Local Storage,TLS)线程局部存储是某些操作系统为线程单独提供的私有空间,但通常只具有很有限的尺寸。
3. ·寄存器(包括PC寄存器),寄存器是执行流的基本数据,因此为线程私有。

>&emsp;&emsp;虽然C++11提供了thread的实现，但是其使用非常有限，无法对线程进行更加精确的控制。

&emsp;&emsp;C/C++标准库早期是不提供线程支持的，那么使用相关的库函数就无法做到线程安全。

### 3.2 CRT改进
&emsp;&emsp;为了支持多线程CRT针对多线程环境进行了一些改进。
**使用TLS**
&emsp;&emsp;多线程环境下有些变量的地址存放在线程的TLS中，比如errno。
**加锁**
&emsp;&emsp;多线程环境中，一些库函数会在函数内部加锁保证线程安全。
**改进函数调用方式**
&emsp;&emsp;改变一些库函数保证其线程安全比如```strtok```msvc的改进版本为```strtok_s```，glibc版本为```strtok_r```。但是无法做到向后兼容。

### 3.3 线程局部存储的实现
&emsp;&emsp;如果希望某个变量线程私有，就需要将变量存放到TLS上。gcc可以使用```__thread```修饰，msvc可以使用```__declspec(thread)```修饰，这样每个变量在各自的线程上都有一个副本。

**windows TLS实现**
&emsp;&emsp;使用```__declspec(thread)```的变量不会被放到数据段，而是放到```.tls```段中，当系统启动一个新线程时，系统会从堆中分配一块内存，将tls的内容拷贝到这块空间供线程使用。对于存放在TLS的全局变量，PE文件中的数据目录结构有一项标记为```IMAGE_DIRET_ENTRY_TLS```的项保存有TLS表，该表中存储了TLS所有TLS变量的构造函数和析构函数的地址，系统可以根据这些地址调用对应的函数完成构造和析构。TLS表本身存储在```.rdata```中。
&emsp;&emsp;现在有了TLS空间和表格，线程如何访问？对于windows线程，系统会构建一个线程环境快（TEB），该结构中存储了线程的堆栈、线程ID等信息，其中一项就是TLS数组，课题通过该数组访问。

**显式TLS**
&emsp;&emsp;使用```__thread,__declspec(thread)```修饰的变量程序员只需要直到他们是线程私有的变量即可，不需要管理，成为隐式TLS。相对的需要陈谷许愿管理的TLS叫做显式TLS。这部分了解就好。

## 4 C++全局构造和析构
### 4.1 glibc全局构造与析构
&emsp;&emsp;glibc中存在两个段```.init```和```.finit```组成```_init()_finit()```两个函数，分别执行初始化和善后的工作。本节就了解下他们如何完成对象的构造和析构工作。
&emsp;&emsp;我们使用下面的代码反汇编查看初始化过程。
```cpp
//gcc main.cpp && objdump -D a.out
#include <cstdio>

class myclass{
public:
    myclass(){ printf("constructor");}
    ~myclass(){ printf("destructor");}
};

myclass cls;

int main(){
    return 0;
}
```
&emsp;&emsp;我们找到```_start```能够看到初始化调用了```__libc_csu_init```。
```asm
00000000000004f0 <_start>:
 4f0:   31 ed                   xor    %ebp,%ebp
 4f2:   49 89 d1                mov    %rdx,%r9
 4f5:   5e                      pop    %rsi
 4f6:   48 89 e2                mov    %rsp,%rdx
 4f9:   48 83 e4 f0             and    $0xfffffffffffffff0,%rsp
 4fd:   50                      push   %rax
 4fe:   54                      push   %rsp
 4ff:   4c 8d 05 7a 01 00 00    lea    0x17a(%rip),%r8        ## 680 <__libc_csu_fini>
 506:   48 8d 0d 03 01 00 00    lea    0x103(%rip),%rcx        ## 610 <__libc_csu_init>
 50d:   48 8d 3d e6 00 00 00    lea    0xe6(%rip),%rdi        ## 5fa <main>
 514:   ff 15 c6 0a 20 00       callq  *0x200ac6(%rip)        ## 200fe0 <__libc_start_main@GLIBC_2.2.5>
 51a:   f4                      hlt
 51b:   0f 1f 44 00 00          nopl   0x0(%rax,%rax,1)
```
&emsp;&emsp;glibc在执行main之前会调用init相关函数，其调用的是```__libc_csu_init```，在这个函数中调用了```_init```即```.init```中的代码。
```cpp
void __libc_csu_init (int argc, char **argv, char **envp){
  /* For dynamically linked executables the preinit array is executed by
     the dynamic linker (before initializing any shared object).  */

#ifndef LIBC_NONSHARED
  /* For static executables, preinit happens right before init.  */
  {
    const size_t size = __preinit_array_end - __preinit_array_start;
    size_t i;
    for (i = 0; i < size; i++)
      (*__preinit_array_start [i]) (argc, argv, envp);
  }
#endif

#if ELF_INITFINI
  _init ();
#endif

  const size_t size = __init_array_end - __init_array_start;
  for (size_t i = 0; i < size; i++)
      (*__init_array_start [i]) (argc, argv, envp);
}

//下面是__libc_csu_init的反汇编
0000000000000610 <__libc_csu_init>:
!省略部分代码
 623:   48 8d 2d ce 07 20 00    lea    0x2007ce(%rip),%rbp        ## 200df8 <__init_array_end>
 62a:   53                      push   %rbx
 62b:   41 89 fd                mov    %edi,%r13d
 62e:   49 89 f6                mov    %rsi,%r14
 631:   4c 29 e5                sub    %r12,%rbp
 634:   48 83 ec 08             sub    $0x8,%rsp
 638:   48 c1 fd 03             sar    $0x3,%rbp
 63c:   e8 77 fe ff ff          callq  4b8 <_init>
!省略部分代码
```

&emsp;&emsp;```_init```的反汇编代码和书上不一样，这里调用了```__gmon_start__```。但是```__gmon_start```并不是初始化程序，实际上应该是```call *%rax```这一句，但是我们不知道具体的地址。
```asm
00000000000004b8 <_init>:
 4b8:   48 83 ec 08             sub    $0x8,%rsp
 4bc:   48 8b 05 25 0b 20 00    mov    0x200b25(%rip),%rax        ## 200fe8 <__gmon_start__>
 4c3:   48 85 c0                test   %rax,%rax
 4c6:   74 02                   je     4ca <_init+0x12>
 4c8:   ff d0                   callq  *%rax
 4ca:   48 83 c4 08             add    $0x8,%rsp
 4ce:   c3                      retq
```
&emsp;&emsp;我们可以尝试在汇编代码中查找```printf```寻找析构和构造数反向推导具体的调用位置。最后反推的路径为```printf@plt->_ZN7myclassC1Ev->_Z41__static_initialization_and_destruction_0ii->_GLOBAL__sub_I_cls```。我们可以看下中间的那个函数的内容，在这个函数中调用了构造函数```_ZN7myclassC1Ev```，并且使用```__cxa_atexit```注册析构函数```_ZN7myclassD1Ev```。
```
0000000000000735 <_Z41__static_initialization_and_destruction_0ii>:
 735:	55                   	push   %rbp
 736:	48 89 e5             	mov    %rsp,%rbp
 739:	48 83 ec 10          	sub    $0x10,%rsp
 73d:	89 7d fc             	mov    %edi,-0x4(%rbp)
 740:	89 75 f8             	mov    %esi,-0x8(%rbp)
 743:	83 7d fc 01          	cmpl   $0x1,-0x4(%rbp)
 747:	75 2f                	jne    778 <_Z41__static_initialization_and_destruction_0ii+0x43>
 749:	81 7d f8 ff ff 00 00 	cmpl   $0xffff,-0x8(%rbp)
 750:	75 26                	jne    778 <_Z41__static_initialization_and_destruction_0ii+0x43>
 752:	48 8d 3d c0 08 20 00 	lea    0x2008c0(%rip),%rdi        ## 201019 <cls>
 759:	e8 32 00 00 00       	callq  790 <_ZN7myclassC1Ev>
 75e:	48 8d 15 a3 08 20 00 	lea    0x2008a3(%rip),%rdx        ## 201008 <__dso_handle>
 765:	48 8d 35 ad 08 20 00 	lea    0x2008ad(%rip),%rsi        ## 201019 <cls>
 76c:	48 8d 3d 3d 00 00 00 	lea    0x3d(%rip),%rdi        ## 7b0 <_ZN7myclassD1Ev>
 773:	e8 88 fe ff ff       	callq  600 <__cxa_atexit@plt>
 778:	90                   	nop
 779:	c9                   	leaveq 
 77a:	c3                   	retq  
```
&emsp;&emsp;这个函数的大概的函数原型可能为：
```cpp
__static_initialization_and_destruction_0(int, int){
	myclass::myclass();
	atexit(myclass::~myclass());
}
```
&emsp;&emsp;根据书上的描述这个函数```_GLOBAL__sub_I_cls```是由编译器生成的，负责初始化全局静态对象并且注册析构函数，gcc会在编译单元的目标文件中生成```.ctors```存放该函数的地址，而析构会生成```.dtor```段存放析构函数的地址。

>&emsp;&emsp;书上的实现已经和现在的一些实现不同，感觉后续需要单独了解下glibc的全局构造和析构的具体情况。可参考[How to find global static initializations](https://stackoverflow.com/questions/28101243/how-to-find-global-static-initializations)

### 4.2 MSVC CRT全局构造与析构
&emsp;&emsp;前面```mainCRTStartup```源码中有调用```_initterm_e```来编译一个表格中所有的函数指针并执行相关内容，完成一些初始化。
```cpp
typedef void (__cdecl* _PVFV)(void);
typedef int  (__cdecl* _PIFV)(void);
typedef void (__cdecl* _PVFI)(int);

#ifndef _M_CEE
    _ACRTIMP void __cdecl _initterm(
        _In_reads_(_Last - _First) _In_ _PVFV*  _First,
        _In_                            _PVFV*  _Last
        );

    _ACRTIMP int  __cdecl _initterm_e(
        _In_reads_(_Last - _First)      _PIFV*  _First,
        _In_                            _PIFV*  _Last
        );
#endif
```
&emsp;&emsp;其中```__xi_a```和```__xc_z```等都是一个全局变量，且对应的变量是被分配在对应的段中的，比如```__xi_a```就在```.CRT$XIA```中。这些段是只读的，在链接时就会被合并到一起，形成了全局初始化函数数组。析构也是类似的都是使用```atexit```注册析构函数。
```cpp
#pragma section(".CRT$XCA",    long, read) // First C++ Initializer
#pragma section(".CRT$XCAA",   long, read) // Startup C++ Initializer
#pragma section(".CRT$XCZ",    long, read) // Last C++ Initializer
#define _CRTALLOC(x) __declspec(allocate(x))
#ifndef _M_CEE
    typedef void (__cdecl* _PVFV)(void);
    typedef int  (__cdecl* _PIFV)(void);

    extern _CRTALLOC(".CRT$XIA") _PIFV __xi_a[]; // First C Initializer
    extern _CRTALLOC(".CRT$XIZ") _PIFV __xi_z[]; // Last C Initializer
    extern _CRTALLOC(".CRT$XCA") _PVFV __xc_a[]; // First C++ Initializer
    extern _CRTALLOC(".CRT$XCZ") _PVFV __xc_z[]; // Last C++ Initializer
    extern _CRTALLOC(".CRT$XPA") _PVFV __xp_a[]; // First Pre-Terminator
    extern _CRTALLOC(".CRT$XPZ") _PVFV __xp_z[]; // Last Pre-Terminator
    extern _CRTALLOC(".CRT$XTA") _PVFV __xt_a[]; // First Terminator
    extern _CRTALLOC(".CRT$XTZ") _PVFV __xt_z[]; // Last Terminator
#endif
```

&emsp;&emsp;从上面大概能够看出linux和windows都是在运行前遍历全局表，逐个调用对应全局/静态对象的构造函数，并使用```atexit```注册析构函数。