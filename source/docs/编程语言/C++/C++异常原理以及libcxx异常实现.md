# C++异常原理以及libcxx异常实现

**摘要**：为了更好的理解C++中异常处理的实现，本文简单描述了Itanium ABI中异常处理的流程和llvm/libsdc++简要实现。
**关键字**：C++,exception,llvm,clang

&emsp;&emsp;C++他提供了异常处理机制来对程序中的错误进行处理，避免在一些异常情况下无法恢复现场而导致额外的损失。虽然在很多Google C++ Guide中禁止使用异常，但是作为C++的一部分我们仍然有足够的理由去了解其技术细节。

&emsp;&emsp;由于C++的ABI并不统一，主流的ABI有两种Itanium ABI和MSVC的ABI，也就是常用的clang/gcc、msvc编译器采用的两种ABI。因此具体的异常实现也不同。比如clang/gcc使用SJLJ(SetJmp/LongJmp)实现，而Windows上虽然大体流程相似但是细节上存在一些差异。本为主要以llvm/Itanium ABI为参考描述，对Windows实现有兴趣的可以参考[Exception Handling using the Windows Runtime](https://llvm.org/docs/ExceptionHandling.html#wineh)。

&emsp;&emsp;再次在在此之前需要了解几个概念：
>&emsp;&emsp;landing pad：A section of user code intended to catch, or otherwise clean up after, an exception. It gains control from the exception runtime via the personality routine, and after doing the appropriate processing either merges into the normal user code or returns to the runtime by resuming or raising a new exception.

>&emsp;&emsp;Itanium ABI中将实现相关和实现无关的内容拆分开，来保证ABI的灵活度。因此，Itanium C++ ABI中Exception Handling分成Level 1 Base ABI and Level 2 C++ ABI两部分。Base ABI描述了语言无关的stack unwinding部分，定义了_Unwind_* API。Level 2 则和C++实现相关，定义了__cxa_* API(__cxa_allocate_exception, __cxa_throw, __cxa_begin_catch等)。

&emsp;&emsp;首先简单说一下，C++ 异常处理的基本流程：
1. 调用```__cxa_allocate_exception```创建异常需要的一些对象，比如exception object等；
2. 调用```cxa_throw```抛出异常；
3. 异常抛出后开始栈展开并搜索匹配的异常类型；
   1. 阶段一，不断展开栈直到搜索到匹配的异常类型，并执行personality routine；
   2. 阶段二，从抛出异常的位置开始执行每个栈帧的cleanup内容；
4. 如果没有匹配到则调用std::terminate终止程序，否则执行根据表格指引执行landing pad；
5. 最后执行清理操作销毁异常对象，还原现场。

## 1 数据结构
### 1.1 Exception Objects
&emsp;&emsp;一个完整的 C++ 异常对象由一个头部组成，这个头部是围绕着一个带有额外的 C++ 特定信息的 unwind 对象头部的包装器，然后是抛出的 C++ 异常对象本身。头部的结构如下：

```cpp
struct __cxa_exception { 
	std::type_info *	exceptionType;
	void (*exceptionDestructor) (void *); 
	unexpected_handler	unexpectedHandler;
	terminate_handler	terminateHandler;
	__cxa_exception *	nextException;
	int			handlerCount;
	int			handlerSwitchValue;
	const char *		actionRecord;
	const char *		languageSpecificData;
	void *			catchTemp;
	void *			adjustedPtr;
	_Unwind_Exception	unwindHeader;
};
```

- ```exceptionType```字段编码了抛出的异常的类型。```exceptionDestructor```字段包含了一个指向被抛出类型的析构函数的函数指针，可能为空。这些指针必须存储在异常对象中，因为非多态和内置类型可以被抛出。
- ```unexpectedHandler```和```terminateHandler```字段包含了指向异常抛出点处的未预期和终止处理程序的指针。
- ```nextException```字段用于创建异常的链表（每个线程）。
- ```handlerCount```字段包含捕获此异常对象的处理程序数量。它还用于确定异常的生存时间。
- ```handlerSwitchValue```、```actionRecord```、```languageSpecificData```、```catchTemp```和```adjustedPtr```字段缓存在第一遍计算时得到的信息，但在第二遍时也很有用。通过将这些信息存储在异常对象中，清理阶段可以避免重新检查操作记录。这些字段保留供包含要调用的处理程序的栈帧的自定义函数使用。
- ```unwindHeader```结构用于在多种语言或相同语言的多个运行时存在的情况下正确操作异常。
&emsp;&emsp;按照惯例，一个 __cxa_exception 指针指向抛出的 C++ 异常对象的表示，紧随其后是头部。头部结构可以通过从 __cxa_exception 指针的负偏移访问。这种布局允许对来自不同语言（或同一语言的不同实现）的异常对象进行一致的处理，并允许在保持二进制兼容性的同时对头部结构进行将来的扩展。

&emsp;&emsp;下面是llvm/libstdc++中的定义，具体实现在```llvm-project/libcxxabi/src/cxa_exception.h```中。可以看到大体上的layout和ABI中定义的相同，但是在头部多了一些字段就是上面提及的为了兼容性实际使用时需要通过负偏移来读取。
```cpp
struct _LIBCXXABI_HIDDEN __cxa_exception {
#if defined(__LP64__) || defined(_WIN64) || defined(_LIBCXXABI_ARM_EHABI)
    // Now _Unwind_Exception is marked with __attribute__((aligned)),
    // which implies __cxa_exception is also aligned. Insert padding
    // in the beginning of the struct, rather than before unwindHeader.
    void *reserve;

    // This is a new field to support C++11 exception_ptr.
    // For binary compatibility it is at the start of this
    // struct which is prepended to the object thrown in
    // __cxa_allocate_exception.
    size_t referenceCount;
#endif

    //  Manage the exception object itself.
    std::type_info *exceptionType;
#ifdef __USING_WASM_EXCEPTIONS__
    // In Wasm, a destructor returns its argument
    void *(_LIBCXXABI_DTOR_FUNC *exceptionDestructor)(void *);
#else
    void (_LIBCXXABI_DTOR_FUNC *exceptionDestructor)(void *);
#endif
    std::unexpected_handler unexpectedHandler;
    std::terminate_handler  terminateHandler;

    __cxa_exception *nextException;

    int handlerCount;

#if defined(_LIBCXXABI_ARM_EHABI)
    __cxa_exception* nextPropagatingException;
    int propagationCount;
#else
    int handlerSwitchValue;
    const unsigned char *actionRecord;
    const unsigned char *languageSpecificData;
    void *catchTemp;
    void *adjustedPtr;
#endif

#if !defined(__LP64__) && !defined(_WIN64) && !defined(_LIBCXXABI_ARM_EHABI)
    // This is a new field to support C++11 exception_ptr.
    // For binary compatibility it is placed where the compiler
    // previously added padding to 64-bit align unwindHeader.
    size_t referenceCount;
#endif
    _Unwind_Exception unwindHeader;
};
```
### 1.2 Caught Exception Stack
&emsp;&emsp;c++-rt中每个线程都包含一个全局的对象来描述当前线程异常的状况。
```c
struct __cxa_eh_globals {
	__cxa_exception *	caughtExceptions;
	unsigned int		uncaughtExceptions;
};
```
- ```caughtExceptions```字段是一个活动异常列表，按照最近的异常排在前面，通过异常头部的```nextException```字段链接成stack。
- ```uncaughtExceptions```字段是未捕获异常的计数，供 C++ 库的```uncaught_exceptions```使用。
&emsp;&emsp;这些信息是基于每个线程维护的。因此，```caughtExceptions```是当前线程抛出并捕获的异常列表，```uncaughtExceptions```是当前线程抛出但尚未捕获的异常计数。（这包括重新抛出的异常，它们可能仍然具有活动的处理程序，但不被视为已捕获。）
&emsp;&emsp;下面是llvm/libstdc++中的定义，具体实现在```llvm-project/libcxxabi/src/cxa_exception.h```中。
```cpp
struct _LIBCXXABI_HIDDEN __cxa_eh_globals {
    __cxa_exception *   caughtExceptions;
    unsigned int        uncaughtExceptions;
#if defined(_LIBCXXABI_ARM_EHABI)
    __cxa_exception* propagatingExceptions;
#endif
};
```

## 2 抛异常
&emsp;&emsp;实现抛出异常所需的处理可能包括以下步骤：
1. 调用```__cxa_allocate_exception```来创建一个异常对象。
2. 评估被抛出的表达式，并将其复制到由```__cxa_allocate_exception```返回的缓冲区中，可能使用复制构造函数。如果评估被抛出的表达式通过抛出异常退出，那么异常将传播而不是表达式本身。清理代码必须确保在刚刚分配的异常对象上调用```__cxa_free_exception```。（如果复制构造函数本身通过抛出异常退出，将调用 terminate()。）
3. 调用```__cxa_throw```将异常传递给运行时库。

### 2.1 创建异常对象
&emsp;&emsp;抛出异常时需要存储对象，而对象必须存储在具体的内存空间中。这个存储空间必须在堆栈unwind时必须保证其生命周期，并且必须是线程安全的。因此，异常对象的存储空间通常将在堆中分配，尽管实现可能提供紧急缓冲区以支持在低内存条件下抛出```bad_alloc```异常。
&emsp;&emsp;内存由```__cxa_allocate_exception```分配，传递了要抛出的异常对象的大小（不包括```__cxa_exception```头部的大小），并返回指向异常对象临时空间的指针。如果可能的话，它将在堆上分配异常内存。如果堆分配失败，实现可以使用其他备份机制。
&emsp;&emsp;C++ 运行时库应为每个潜在任务分配至少4K字节的静态紧急缓冲区，最多64KB。该缓冲区仅在异常对象动态分配失败的情况下使用。它应以 1KB 块的形式分配。任何时候最多有 16 个任务可以使用紧急缓冲区，最多4个嵌套异常，每个异常对象（包括Header）的大小最多为1KB。其他线程将被阻塞，直到16个线程之一取消分配其紧急缓冲存储。紧急缓冲区的接口是实现定义的，并且仅由异常库使用。
&emsp;&emsp;如果在这些约束条件下```__cxa_allocate_exception```无法分配异常对象，它将调用```terminate()```终止程序。

```cpp
void * __cxa_allocate_exception(size_t thrown_size);
```
&emsp;&emsp;一旦空间被分配，throw 表达式必须根据 C++ 标准指定的抛出值初始化异常对象。临时空间将由```__cxa_free_exception```释放，该函数传递了前一个```__cxa_allocate_exception``` 返回的地址。

```cpp
void __cxa_free_exception(void *thrown_exception);
```
&emsp;&emsp;这些函数是线程安全的（在多线程环境中），并且在达到允许使用紧急缓冲区的最大线程数后，可能会阻塞线程。

### 2.2 抛异常
**cxa_throw**
在使用```throw```参数值构造异常对象后，生成的代码调用```__cxa_throw```运行时库函数。这个例程永远不会返回。
```cpp
void __cxa_throw(void *thrown_exception, std::type_info *tinfo, void (*dest)(void *));
```
参数包括：
- 抛出异常对象的地址（指向头部之后的抛出值，如上所述）。
- 一个```std::type_info```指针，给出抛出参数的静态类型作为一个```std::type_info```指针，用于将潜在的捕获点与抛出的异常匹配。
- 一个最终用于销毁对象的析构函数指针。
```__cxa_throw```例程将执行以下操作：
- 从抛出异常对象地址获取```__cxa_exception```头部，可以如下计算：
  ```cpp
  __cxa_exception *header = ((__cxa_exception *) thrown_exception - 1);
  ```
- 将当前的```unexpected_handler```和```terminate_handler```保存在 ```__cxa_exception```头部中。
- 将```tinfo```和```dest```参数保存在```__cxa_exception```头部中。
- 在unwind头部中设置```exception_class```字段。这是一个64位值，表示 ASCII 字符串 "XXXXC++\0"，其中 "XXXX"是一个供应商相关的字符串。C++中64 位值的低 4 字节将是 "C++\0"。
- 增加未捕获异常标志。
- 在系统unwind库中调用```_Unwind_RaiseException```。它的参数是```__cxa_throw```自身作为参数接收的指向抛出异常的指针。
- ```__Unwind_RaiseException```开始堆栈unwind过程。在特殊情况下，例如无法找到处理程序，```_Unwind_RaiseException```可能会返回。在这种情况下，假设没有处理程序处理异常，```__cxa_throw```将调用```terminate```。

**__Unwind_RaiseException**
&emsp;&emsp;抛出一个异常，传递给定的异常对象，该对象应该已经设置了其```exception_class```和```exception_cleanup```字段。异常对象已由特定于语言的运行时分配，并具有特定于语言的格式，除非它必须包含一个```_Unwind_Exception```结构体。```_Unwind_RaiseException``` 不返回，除非发现错误条件（例如异常没有处理程序、堆栈格式不正确等）。在这种情况下，将返回一个```_Unwind_Reason_Code```值。可能的情况包括：
- ```_URC_END_OF_STACK```：unwind在阶段1中遇到了堆栈的末尾，没有找到处理程序。unwind运行时不会修改堆栈。在这种情况下，C++运行时通常会调用```uncaught_exception```。
- ```_URC_FATAL_PHASE1_ERROR```：unwind在阶段1中遇到了意外错误，例如堆栈损坏。unwind运行时不会修改堆栈。在这种情况下，C++运行时通常会调用```terminate```。

&emsp;&emsp;如果unwind时在阶段2中遇到意外错误，它应该向其调用者返回```_URC_FATAL_PHASE2_ERROR```。在 C++ 中，这通常是```__cxa_throw```，后者将调用```terminate()```。

**注意**：unwind运行时很可能已经修改了堆栈（例如，从中弹出了帧）或者寄存器上下文，或者落脚点代码可能已经破坏了它们。因此，```_Unwind_RaiseException```的调用者无法对其堆栈或寄存器的状态做出任何假设。

## 3 捕获异常
### 3.1 两阶段
**UnWind**
&emsp;&emsp;为了捕获异常，抛出异常后需要对当前函数的调用栈进行展开。堆栈展开有两个主要原因：
1. 异常，由支持异常的语言定义（例如C++）；
2. “强制”展开（例如由longjmp或线程终止引起）。

>&emsp;&emsp;异常机制是语言无关的，为了和其他语言交互，将C++语言相关的部分抽象出来即```personality routine```。

&emsp;&emsp;在抛出异常的情况下，堆栈在异常传播时被展开，但是每个堆栈帧的```personality routine```都知道它是否想要捕获异常或将其传递是预期内的。因此，这个选择被委托给了```personality routine```，```personality routine```被期望对任何类型的异常（“本地”或“外部”）都能正确地做出反应。
&emsp;&emsp;在“强制展开”期间，外部代理驱动展开。例如，这可以是```longjmp```。这个外部代理，而不是每个```personality routine```决定何时停止展开。```personality routine```没有关于展开是否会继续的，而是否进行展开是由的_UA_FORCE_UNWIND标志表示。
&emsp;&emsp;为了适应这些差异，提出了两种不同的处理程序。```_Unwind_RaiseException```执行异常展开，受```personality routine```控制。另一方面，```_Unwind_ForcedUnwind```执行展开，但给外部代理拦截调用```personality routine```的时机。这是通过使用代理```personality routine```完成的，该代理拦截对```personality routine```的调用，让外部代理覆盖堆栈帧```personality routine```的默认设置。
&emsp;&emsp;因此，不需要每个```personality routine```了解可能导致展开的任何可能的外部代理。例如，C++```personality routine```只需要处理C++异常（可能伪装外部异常），但不需要了解关于代表longjmp或pthread取消执行的展开的任何特定信息。

**UnWind流程**
&emsp;&emsp;异常时的Unwind处理分为两阶段过程：
- 在搜索阶段，框架重复调用```personality routine```，使用_UA_SEARCH_PHASE标志，首先针对当前PC和寄存器状态，然后在每一步展开帧到新PC，直到```personality routine```报告在所有帧中成功（在查询的帧中找到处理程序）或失败（所有帧中都没有处理程序）。它实际上不会恢复展开的状态，```personality routine```必须通过API访问状态；
- 如果搜索阶段报告失败，例如因为未找到处理程序，它将调用terminate()而不是开始第二阶段。
- 如果搜索阶段报告成功，框架将重新启动清理阶段。同样，它重复调用```personality routine```，使用_UA_CLEANUP_PHASE标志，首先针对当前PC和寄存器状态，然后在每一步展开帧到新PC，直到到达具有已识别处理程序的帧为止。在那一点上，它恢复寄存器状态，并将控制传递给用户landing pad。
&emsp;&emsp;这两个阶段都使用展开库和```personality routine```，因为给定处理程序的有效性和将控制传递给它的机制是语言相关的，但定位和恢复先前堆栈帧的方法是语言无关的。
>两阶段异常处理模型并不是严格必要的来实现C++语言语义，但它确实提供了一些好处。例如，第一阶段允许异常处理机制在堆栈展开开始之前解除异常，这允许恢复式异常处理（纠正异常条件并在引发点恢复执行）。虽然C++不支持恢复式异常处理，但其他语言支持，而两阶段模型允许C++与这些语言在堆栈上共存。

&emsp;&emsp;请注意，即使采用了两阶段模型，对于单个异常，可能会多次执行每个阶段，就好像异常被抛出了多次一样。例如，由于无法确定给定的catch子句是否会重新抛出异常，而不执行它，异常传播实际上在每个catch子句处停止，并且如果需要重新开始，则重新开始第1阶段。对于析构函数（清理代码）不需要此过程，因此阶段1可以安全地一次性处理所有仅包含析构函数的帧，并在下一个封闭的catch子句处停止。
&emsp;&emsp;例如，如果展开的前两个帧仅包含清理代码，而第三个帧包含一个C++的catch子句，则第1阶段中的```personality routine```不会指示它在前两个帧中找到了处理程序。它必须在第三个帧中这样做，因为无法确定异常将如何传播出第三个帧，例如通过在C++中重新抛出异常或引发新异常。
&emsp;&emsp;堆栈展开库在堆栈上运行两次遍历，如下所示：
1. 恢复当前堆栈帧中的程序计数器（PC）。
2. 使用展开表，在该PC处查找关于如何处理在该PC处发生的异常的信息，特别是获取该地址范围内```personality routine```的地址。
3. 调用```personality routine```（见第2.5.2节）。```personality routine```将确定在堆栈的该级别是否找到了适当的处理程序（在第一次遍历中），并确定从landing pad调用哪个特定处理程序（在第二次遍历中），以及传递给landing pad的参数（见第3.5.2节）。```personality routine```将此信息传递回展开库。

&emsp;&emsp;在第二阶段，展开库跳转到与调用对应的landing pad以展开的堆栈级别。由```personality routine```指示设置landing pad参数。landing pad执行补偿代码（由后端生成），以恢复适当的寄存器和堆栈状态。
&emsp;&emsp;然后，前端生成的一些清理代码可能会执行，对应于try块的退出。例如，try块中的局部自动变量将在此处被销毁。
&emsp;&emsp;异常处理程序可能会选择并执行与C++ catch子句和其他处理程序对应的用户定义代码。生成的代码可能类似于switch语句，其中switch值由运行时基于异常类型确定，并作为landing pad参数传递。
&emsp;&emsp;一旦运行时确定执行将转到处理程序，展开库就认为展开过程对于其完成了。处理程序仍然可以重新抛出当前异常或不同的异常，但在两种情况下都将发生新的展开过程。否则，在处理程序中的代码执行完毕后，执行将在定义此处理程序的try块的末尾恢复。
&emsp;&emsp;如果可能的处理程序都不匹配正在抛出的异常，则运行时会选择一个不匹配任何switch语句的switch值。在这种情况下，控制流将通过所有switch语句，并转到其他清理代码，该清理代码将调用需要为当前堆栈帧调用的所有析构函数。例如，在函数的外部块的自动变量将在此处被销毁。这意味着处理程序必须在清理帧的过程中循环通过任何包围原始块的try块，尝试每一个switch值。
&emsp;&emsp;在当前清理代码的末尾，控制被转移回展开库，以展开另一个堆栈帧。

### 3.2 异常处理表格
**异常处理帧（Exception Handling Frame）**
&emsp;&emsp;异常处理帧```eh_frame```与DWARF调试信息中使用的展开帧非常相似。该帧包含了撤销当前帧并恢复先前帧状态所需的所有信息。每个编译单元中的函数都有一个异常处理帧，另外还有一个通用异常处理帧，其中定义了单元中所有函数共享的信息。
&emsp;&emsp;然而，这种调用帧信息（CFI）的格式通常与平台相关。例如，ARM 定义了自己的格式。苹果有自己的紧凑展开信息格式。在 Windows 上，自 32 位 x86 以来，所有架构都使用另一种格式。LLVM 将生成目标所需的任何信息。
&emsp;&emsp;eh_frame信息可参考[Linux Standard Base Core Specification 3.0RC1](https://refspecs.linuxfoundation.org/LSB_3.0.0/LSB-Core-generic/LSB-Core-generic/ehframechpt.html)月也可以使用命令```objdump -sj .eh_frame <your_exe_file>```查看具体的内容。

**异常表（Exception Tables）**
&emsp;&emsp;异常表包含有关在函数代码的特定部分抛出异常时应采取的操作的信息。这通常被称为特定于语言的数据区域（LSDA）。LSDA 表的格式特定于自定义函数，但其中的大部分使用的是 ```__gxx_personality_v0```所需的变体表。每个函数都有一个异常表，除了叶子函数和仅调用非抛出函数的函数。它们不需要异常表。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/exception_table.png)

### 3.3 Personality Routine
Personality Routine是C++（或其他语言）运行时库中的函数，它充当系统展开库与特定语言异常处理语义之间的接口。它是针对由展开信息块描述的代码片段而言的，并且始终通过展开信息块中的指针引用。当程序抛出异常时，该函数会被调用来处理异常。这个函数的确切实现取决于编译器和操作系统，但它的名称通常是固定的，并且在链接时被引入到程序中。Personality Routine需要在合适的时机将控制权交给LandingPad。
```cpp
_Unwind_Reason_Code (*__personality_routine)
	    (int version,
	     _Unwind_Action actions,
	     uint64 exceptionClass,
	     struct _Unwind_Exception *exceptionObject,
	     struct _Unwind_Context *context);
```
&emsp;&emsp;gcc和clang中该符号为```__gxx_personality_v0```。

## 4 LLVM/libstdc++ 实现
### 4.1 ```__cxa_allocate_exception```
&emsp;&emsp;```__cxa_allocate_exception```实现比较简单，主要是通过rt内置的allocate函数申请内存，然后初始化内存。唯一需要注意的返回的对象指针并不是allocate的内存开头，而是经过偏移的，所以后面读取对象的时候都要偏移回去。
```cpp
void *__cxa_allocate_exception(size_t thrown_size) throw() {
    size_t actual_size = cxa_exception_size_from_exception_thrown_size(thrown_size);

    // Allocate extra space before the __cxa_exception header to ensure the
    // start of the thrown object is sufficiently aligned.
    size_t header_offset = get_cxa_exception_offset();
    char *raw_buffer =
        (char *)__aligned_malloc_with_fallback(header_offset + actual_size);
    if (NULL == raw_buffer)
        std::terminate();
    __cxa_exception *exception_header =
        static_cast<__cxa_exception *>((void *)(raw_buffer + header_offset));
    ::memset(exception_header, 0, actual_size);
    return thrown_object_from_cxa_exception(exception_header);
}
```
### 4.2 ```__cxa_throw```
&emsp;&emsp;下面是llvm/libstdc++的实现，具体代码在```llvm-project/libcxxabi/src/cxa_exception.cpp```中。
```cpp
void
#ifdef __USING_WASM_EXCEPTIONS__
// In Wasm, a destructor returns its argument
__cxa_throw(void *thrown_object, std::type_info *tinfo, void *(_LIBCXXABI_DTOR_FUNC *dest)(void *)) {
#else
__cxa_throw(void *thrown_object, std::type_info *tinfo, void (_LIBCXXABI_DTOR_FUNC *dest)(void *)) {
#endif
  __cxa_eh_globals* globals = __cxa_get_globals();
  globals->uncaughtExceptions += 1; // Not atomically, since globals are thread-local

  __cxa_exception* exception_header = __cxa_init_primary_exception(thrown_object, tinfo, dest);
  exception_header->referenceCount = 1; // This is a newly allocated exception, no need for thread safety.

#if __has_feature(address_sanitizer)
  // Inform the ASan runtime that now might be a good time to clean stuff up.
  __asan_handle_no_return();
#endif

#ifdef __USING_SJLJ_EXCEPTIONS__
    _Unwind_SjLj_RaiseException(&exception_header->unwindHeader);
#else
    _Unwind_RaiseException(&exception_header->unwindHeader);
#endif
    //  This only happens when there is no handler, or some unexpected unwinding
    //     error happens.
    failed_throw(exception_header);
}
```
&emsp;&emsp;该函数接受的参数中描述了需要抛出的异常的对象地址和```typeinfo```信息，以及cleanup的函数指针。从上面的流程中能够看到执行步骤分别为：
1. 通过```cxa_get_globals```接口获取TLS全局EH对象。在第一次调用时会创建```__cxa_eh_globals```表格并创建一个对喜爱嗯，该接口中使用平台相关的Thread实现；
2. 然后调整```__cxa_eh_globals```和异常对象的引用技术，由于前者存储在TLS上，因此不需要atomic；
3. 调用具体的抛异常函数```_Unwind_RaiseException```，如果返回了说明失败，否则不会进行下一步；
4. 处理抛出异常失败的情况，调用一些处理函数后直接```__terminate```。

&emsp;&emsp;具体调用的```_Unwind_SjLj_RaiseException```还是```_Unwind_RaiseException```根据具体的配置而言。```_Unwind_RaiseException```的实现比较简单，首先就是调用```__unw_getcontext```获取unwind的context，然后分别调用两阶段的unwind函数。
```cpp
_LIBUNWIND_EXPORT _Unwind_Reason_Code
_Unwind_RaiseException(_Unwind_Exception *exception_object) {
  _LIBUNWIND_TRACE_API("_Unwind_RaiseException(ex_obj=%p)",
                       (void *)exception_object);
  unw_context_t uc;
  unw_cursor_t cursor;
  __unw_getcontext(&uc);

  // Mark that this is a non-forced unwind, so _Unwind_Resume()
  // can do the right thing.
  exception_object->private_1 = 0;
  exception_object->private_2 = 0;

  // phase 1: the search phase
  _Unwind_Reason_Code phase1 = unwind_phase1(&uc, &cursor, exception_object);
  if (phase1 != _URC_NO_REASON)
    return phase1;

  // phase 2: the clean up phase
  return unwind_phase2(&uc, &cursor, exception_object);
}
```
### 4.3 捕获异常
#### 4.3.1 两阶段
&emsp;&emsp;从上面能够看到捕获异常时栈展开的两阶段是由```unwind_phase1```和```unwind_phase2```完成的。
**阶段1：搜索匹配的栈**
&emsp;&emsp;原来的代码中有大量_LIBUNWIND_TRACE_UNWINDING相关的内容，下面的代码是删除_LIBUNWIND_TRACE_UNWINDING相关内容的代码。
```cpp
static _Unwind_Reason_Code
unwind_phase1(unw_context_t *uc, unw_cursor_t *cursor, _Unwind_Exception *exception_object) {
  __unw_init_local(cursor, uc);

  // Walk each frame looking for a place to stop.
  while (true) {
    // Ask libunwind to get next frame (skip over first which is
    // _Unwind_RaiseException).
    int stepResult = __unw_step(cursor);
    if (stepResult == 0) {
      _LIBUNWIND_TRACE_UNWINDING(
          "unwind_phase1(ex_obj=%p): __unw_step() reached "
          "bottom => _URC_END_OF_STACK",
          (void *)exception_object);
      return _URC_END_OF_STACK;
    } else if (stepResult < 0) {
      return _URC_FATAL_PHASE1_ERROR;
    }

    // See if frame has code to run (has personality routine).
    unw_proc_info_t frameInfo;
    unw_word_t sp;
    if (__unw_get_proc_info(cursor, &frameInfo) != UNW_ESUCCESS) {
      return _URC_FATAL_PHASE1_ERROR;
    }

    // If there is a personality routine, ask it if it will want to stop at
    // this frame.
    if (frameInfo.handler != 0) {
      _Unwind_Personality_Fn p =
          (_Unwind_Personality_Fn)(uintptr_t)(frameInfo.handler);
      _Unwind_Reason_Code personalityResult =
          (*p)(1, _UA_SEARCH_PHASE, exception_object->exception_class,
               exception_object, (struct _Unwind_Context *)(cursor));
      switch (personalityResult) {
      case _URC_HANDLER_FOUND:
        // found a catch clause or locals that need destructing in this frame
        // stop search and remember stack pointer at the frame
        __unw_get_reg(cursor, UNW_REG_SP, &sp);
        exception_object->private_2 = (uintptr_t)sp;
        return _URC_NO_REASON;

      case _URC_CONTINUE_UNWIND:
        // continue unwinding
        break;

      default:
        return _URC_FATAL_PHASE1_ERROR;
      }
    }
  }
  return _URC_NO_REASON;
}
```
&emsp;&emsp;阶段1搜索匹配的栈帧：
1. 首先```cursor```step到表格中的下一项。```cursor```是libunwind中实现的一个游标，即```AbstractUnwindCursor```；
   >UnwindCursor contains all state (including all register values) during an unwind.  This is normally stack-allocated inside a unw_cursor_t.
2. 然后根据游标中存储的信息获取栈帧信息，如果获取失败就会退出；
3. 最后检查当前栈帧是否包含```personality routine```处理程序，没有就会继续搜索，有的花将话就会调用并根据该程序的返回值举鼎下一步的行为。

**阶段2：CleanUp**

```cpp
static _Unwind_Reason_Code
unwind_phase2(unw_context_t *uc, unw_cursor_t *cursor, _Unwind_Exception *exception_object) {
  __unw_init_local(cursor, uc);
  // uc is initialized by __unw_getcontext in the parent frame. The first stack
  // frame walked is unwind_phase2.
  unsigned framesWalked = 1;
#ifdef _LIBUNWIND_USE_CET
  unsigned long shadowStackTop = _get_ssp();
#endif
  // Walk each frame until we reach where search phase said to stop.
  while (true) {
    // Ask libunwind to get next frame (skip over first which is
    // _Unwind_RaiseException).
    int stepResult = __unw_step_stage2(cursor);
    if (stepResult == 0) {
      return _URC_END_OF_STACK;
    } else if (stepResult < 0) {
      return _URC_FATAL_PHASE2_ERROR;
    }

    // Get info about this frame.
    unw_word_t sp;
    unw_proc_info_t frameInfo;
    __unw_get_reg(cursor, UNW_REG_SP, &sp);
    if (__unw_get_proc_info(cursor, &frameInfo) != UNW_ESUCCESS) {
      return _URC_FATAL_PHASE2_ERROR;
    }
// In CET enabled environment, we check return address stored in normal stack
// against return address stored in CET shadow stack, if the 2 addresses don't
// match, it means return address in normal stack has been corrupted, we return
// _URC_FATAL_PHASE2_ERROR.
#ifdef _LIBUNWIND_USE_CET
    if (shadowStackTop != 0) {
      unw_word_t retInNormalStack;
      __unw_get_reg(cursor, UNW_REG_IP, &retInNormalStack);
      unsigned long retInShadowStack = *(
          unsigned long *)(shadowStackTop + __cet_ss_step_size * framesWalked);
      if (retInNormalStack != retInShadowStack)
        return _URC_FATAL_PHASE2_ERROR;
    }
#endif
    ++framesWalked;
    // If there is a personality routine, tell it we are unwinding.
    if (frameInfo.handler != 0) {
      _Unwind_Personality_Fn p =
          (_Unwind_Personality_Fn)(uintptr_t)(frameInfo.handler);
      _Unwind_Action action = _UA_CLEANUP_PHASE;
      if (sp == exception_object->private_2) {
        // Tell personality this was the frame it marked in phase 1.
        action = (_Unwind_Action)(_UA_CLEANUP_PHASE | _UA_HANDLER_FRAME);
      }
       _Unwind_Reason_Code personalityResult =
          (*p)(1, action, exception_object->exception_class, exception_object,
               (struct _Unwind_Context *)(cursor));
      switch (personalityResult) {
      case _URC_CONTINUE_UNWIND:
        // Continue unwinding
        if (sp == exception_object->private_2) {
          // Phase 1 said we would stop at this frame, but we did not...
          _LIBUNWIND_ABORT("during phase1 personality function said it would "
                           "stop here, but now in phase2 it did not stop here");
        }
        break;
      case _URC_INSTALL_CONTEXT:
        // Personality routine says to transfer control to landing pad.
        // We may get control back if landing pad calls _Unwind_Resume().
        if (_LIBUNWIND_TRACING_UNWINDING) {
          unw_word_t pc;
          __unw_get_reg(cursor, UNW_REG_IP, &pc);
          __unw_get_reg(cursor, UNW_REG_SP, &sp);
        }

        __unw_phase2_resume(cursor, framesWalked);
        // __unw_phase2_resume() only returns if there was an error.
        return _URC_FATAL_PHASE2_ERROR;
      default:
        // Personality routine returned an unknown result code.
        return _URC_FATAL_PHASE2_ERROR;
      }
    }
  }
  // Clean up phase did not resume at the frame that the search phase
  // said it would...
  return _URC_FATAL_PHASE2_ERROR;
}
```
&emsp;&emsp;阶段2逐个栈帧调用cleanup：
1. 调用```__unw_step_stage2```调整当前的游标；
2. 获取当前栈帧的基本信息，比如PC，寄存器等；
3. 检查当前栈帧是否包含处理程序有的话调用，然后根据该程序的返回值执行下一步。

&emsp;&emsp;需要注意的是阶段2和阶段1的起点相同，都是从调用```__Unwind_RaiseException```处开遍历。

#### 4.3.2 Landing Pad
> landing pad：A section of user code intended to catch, or otherwise clean up after, an exception. It gains control from the exception runtime via the personality routine, and after doing the appropriate processing either merges into the normal user code or returns to the runtime by resuming or raising a new exception.

&emsp;&emsp;Landing pad是text section中的一段和exception相关的代码，它有三种：
- cleanup clause：通常调用局部对象的析构函数或__attribute__((cleanup(...)))注册的callbacks，然后用_Unwind_Resume跳转回cleanup phase；
- 捕获异常的catch clause：调用局部对象的析构函数，然后调用```__cxa_begin_catch```，执行catch代码，最后调用```__cxa_end_catch```；
- rethrow：调用catch clause中的局部对象的析构函数，然后调用```__cxa_end_catch```，接着用```_Unwind_Resume```跳转回cleanup phase。

## 5 异常编译后是什么样的
### 5.1 异常编译后是什么样的
&emsp;&emsp;我们用下面的代码作为例子：
```cpp
#include <cstdio>
#include <exception>

class UserDefine{
public:
  int a;
  UserDefine(){
    printf("run A cons\n");
  }
  ~UserDefine(){
    printf("run A des\n");
  }
};
class MyException : public std::exception{
public:
  int a;
  MyException(){
    printf("run MyException cons\n");
  }
  ~MyException(){
    printf("run MyException des\n");
  }
};

__attribute__((noinline)) void exceptionHanldeFunc1(){
  printf("the function %s is calling\n", __FUNCTION__);
}

__attribute__((noinline)) void exceptionHanldeFunc2(){
  printf("the function %s is calling\n", __FUNCTION__);
}

int main(int argc, char **argv){
    try{
      UserDefine a;
      throw MyException();
      printf("a %d", static_cast<int>(a.a));
    } catch(MyException &e){
      exceptionHanldeFunc2();
    } catch(...){
      exceptionHanldeFunc2();
    }

    return 0;
}
```
&emsp;&emsp;下面是main函数反汇编的结果，代码含义看下面的注释：

```
00000000000011e0 <main>:
    11e0:	53                   	push   %rbx
    11e1:	48 8d 3d 62 0e 00 00 	lea    0xe62(%rip),%rdi        # 204a <_IO_stdin_used+0x4a>
    11e8:	e8 83 fe ff ff       	call   1070 <puts@plt>
    11ed:	bf 08 00 00 00       	mov    $0x8,%edi
    11f2:	e8 59 fe ff ff       	call   1050 <__cxa_allocate_exception@plt>      ;;调用cxa_allocate_exceptionallocate异常
    11f7:	48 8d 0d 92 2b 00 00 	lea    0x2b92(%rip),%rcx        # 3d90 <_ZNSt9exceptionD2Ev@GLIBCXX_3.4>  ;;由于自定义的MyException是空类因此这里直接调用的std::exception的构造函数
    11fe:	48 89 08             	mov    %rcx,(%rax)              ;;将异常对象的地址、typeinf等放到寄存器上
    1201:	48 8d 35 60 2b 00 00 	lea    0x2b60(%rip),%rsi        # 3d68 <_ZTVN10__cxxabiv120__si_class_type_infoE@CXXABI_1.3>
    1208:	48 8b 15 c1 2d 00 00 	mov    0x2dc1(%rip),%rdx        # 3fd0 <_ZNSt9exceptionD2Ev@GLIBCXX_3.4>
    120f:	48 89 c7             	mov    %rax,%rdi
    1212:	e8 79 fe ff ff       	call   1090 <__cxa_throw@plt>   ;;调用cxa_throw
    1217:	48 89 c3             	mov    %rax,%rbx
    121a:	48 8d 3d 34 0e 00 00 	lea    0xe34(%rip),%rdi        # 2055 <_IO_stdin_used+0x55>
    1221:	e8 4a fe ff ff       	call   1070 <puts@plt>
    1226:	48 89 df             	mov    %rbx,%rdi
    1229:	e8 12 fe ff ff       	call   1040 <__cxa_begin_catch@plt> ;;调用__cxa_begin_catch
    122e:	e8 8d ff ff ff       	call   11c0 <_Z20exceptionHanldeFunc2v> 
    1233:	e8 48 fe ff ff       	call   1080 <__cxa_end_catch@plt>   ;;调用__cxa_end_catch
    1238:	31 c0                	xor    %eax,%eax
    123a:	5b                   	pop    %rbx
    123b:	c3                   	ret
    123c:	0f 1f 40 00          	nopl   0x0(%rax)
```

### 5.2 LLVM-IR中的landing pad
```
21:                                               ; preds = %12
  %22 = landingpad { ptr, i32 }
          catch ptr @_ZTI11MyException
          catch ptr null
  %23 = extractvalue { ptr, i32 } %22, 0
  store ptr %23, ptr %7, align 8
  %24 = extractvalue { ptr, i32 } %22, 1
  store i32 %24, ptr %8, align 4
  br label %25

25:                                               ; preds = %21, %17
  call void @_ZN10UserDefineD2Ev(ptr noundef nonnull align 4 dereferenceable(4) %6) #8
  br label %26
```

## 6 参考文献
- [Exception Handling Tables](https://itanium-cxx-abi.github.io/cxx-abi/exceptions.pdf)
- [Linux Standard Base Core Specification 3.0RC1](https://refspecs.linuxfoundation.org/LSB_3.0.0/LSB-Core-generic/LSB-Core-generic/ehframechpt.html)
- [Itanium C++ ABI: Exception Handling](https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html)
- [Exception Handling in LLVM](https://llvm.org/docs/ExceptionHandling.html)
- [C++ exception handling ABI](https://maskray.me/blog/2020-12-12-c++-exception-handling-abi#%E4%B8%AD%E6%96%87%E7%89%88)
- [linux 栈回溯(x86_64 )](https://zhuanlan.zhihu.com/p/302726082)
- [C++ 异常是如何实现的](https://zhuanlan.zhihu.com/p/406894769)
- [Recon-2012-Skochinsky-Compiler-Internals](http://www.hexblog.com/wp-content/uploads/2012/06/Recon-2012-Skochinsky-Compiler-Internals.pdf)