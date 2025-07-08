# 【crash】 cxa_pure_virtual 崩溃分析与原理
&emsp;&emsp;**摘要**：工作过程中处理线上的崩溃时发现了一例```cxa_pure_virtual```相关的crash，直接看堆栈基本山很容易确认是有异步调用导致出发了ABI的异常。但是对于为什么会触发```cxa_pure_virtual```虽然有大致的猜测但是没有直接的证据，因此本文主要描述触发该类型崩溃的原理。
&emsp;&emsp;**关键字**：cxxabi,llvm,cxa_pure_virtual,vptr

&emsp;&emsp;首先我们看一下崩溃的现象，线上的崩溃堆栈大概类似于下面形式：
```
0x********* abort()
0x********* std::terminate()
0x********* cxxabi::__cxa_pure_virtual()
0x********* ******::*******
```

&emsp;&emsp;上面的崩溃我们看实际的代码基本上能够判断出当前类已经被析构的情况下当前类却尝试访问虚函数导致了```cxa_pure_virtual```，要修复该问题直接排查哪里导致的异步调用即可。

&emsp;&emsp;```__cxa_pure_virtual```的描述如下：
```c
The __cxa_pure_virtual function is an error handler that is invoked when a pure virtual function is called.
If you are writing a C++ application that has pure virtual functions you must supply your own __cxa_pure_virtual error handler function.
```
&emsp;&emsp;当调用一个纯虚函数时被调用，看llvm中cxxabi的实现可以看到该函数被调用时会直接abort。那就比较奇怪，如果我们调用的是一个纯虚函数按理说编译都无法通过，但是查看代码发现对应的函数是被重写的。那我们此时可能怀疑的一个点便是，虚基类的虚函数表构造和销毁问题。可能是因为子类被销毁是基类的虚函数表被改回基类的虚函数表，而基类中对应虚函数指针就是编译器指定的```cxa_pure_virtual```。
```c
_LIBCXXABI_FUNC_VIS _LIBCXXABI_NORETURN void __cxa_pure_virtual(void) {
  abort_message("Pure virtual function called!");
}
```
&emsp;&emsp;怀疑到这一点，我这边开始找资料（类似的问题印象中标准中是不管的，那大概率在ABI中定义的，那我们去看ABI的定义）。从ABI的定义中找到如下的描述：
```c
An implementation shall provide a standard entry point that a compiler may reference in virtual tables to indicate a pure virtual function. Its interface is:
    extern "C" void __cxa_pure_virtual ();
This routine will only be called if the user calls a non-overridden pure virtual function, which has undefined behavior according to the C++ Standard. Therefore, this ABI does not specify its behavior, but it is expected that it will terminate the program, possibly with an error message.

 if C::f is a pure virtual function, no specific requirement is made for the corresponding virtual table entry. It may point to __cxa_pure_virtual (see 3.2.6 Pure Virtual Function API) or to a wrapper function for __cxa_pure_virtual (e.g., to adapt the calling convention). It may also simply be null in such cases.
```
&emsp;&emsp;上面这一段描述了```cxa_pure_virtual```实际的意义。下面再看一下CXXABI中关于对象以及虚函数表构造的过程的描述：
```c
     // Sub-VTT for D (embedded in VTT for its derived class X):
     static vtable *__VTT__1D [1+n+m] =
	{ D primary vtable,
	  // The sub-VTT for B-in-D in X may have further structure:
	  B-in-D sub-VTT (n elements),
	  // The secondary virtual pointers for D's bases have elements
	  // corresponding to those in the B-in-D sub-VTT,
	  // and possibly others for virtual bases of D:
	  D secondary virtual pointer for B and bases (m elements) }; 

     D ( D *this, vtable **ctorvtbls )
     {
	// (The following will be unwound, not a real loop):
	for ( each base A of D ) {

	   // A "boring" base is one that does not need a ctorvtbl:
	   if ( ! boring(A) ) {
	     // Call subobject constructors with sub-VTT index
	     // if the base needs it -- only B in our example:
	      A ( (A*)this, ctorvtbls + sub-VTT-index(A) ); 

	   } else {
	     // Otherwise, just invoke the complete-object constructor:
	      A ( (A*)this );
	   }
	}

        // Initialize virtual pointer with primary ctorvtbls address
	// (first element):
        this->vptr = ctorvtbls+0;	// primary virtual pointer

	// (The following will be unwound, not a real loop):
	for ( each subobject A of D ) {
	
	   // Initialize virtual pointers of subobjects with ctorvtbls
	   // addresses for the bases 
	   if ( ! boring(A) ) {
	      ((A*)this)->vptr = ctorvtbls + 1+n + secondary-vptr-index(A);
		   // where n is the number of elements in the sub-VTTs
	    
	   } else {
	     // Otherwise, just use the complete-object vtable:
	      ((A *)this)->vptr = &(A-in-D vtable);
	   }
	}

        // Code for D constructor.
	...
      }
```

&emsp;&emsp;从上面的描述中我们能够看到：
1. 当前类的虚函数表指针的确定是在执行具体的构造函数代码之前的；
2. 构建当前类之前会搜索当前类的继承图，找到基类按照继承图的先序序列构造基类；
3. 基类构造完成后开始调用当前类的构造函数的代码。

&emsp;&emsp;析构函数的顺序相反。对于一个具有直接继承关系的虚基类A和B（B继承自A）的构造顺序为：
```c
class A{
public:
    virtual void func() = 0;
};

class B: public A{
public:
    virtual void func(){}
};
```
1. B构造函数B::B被调用；
2. 遍历B的基类构造调用基类的构造函数，这里就是A::A();
3. 调用A的时候先将vfptr指向A的虚函数表，此表项中有基类偏移，typeinfo，```__cxa_pure_virtual```（因为func是纯虚函数因此该处的虚函数表指针以此填充）；
4. 调用A::A的用户代码，这里没有就不调用；
5. A构造函数执行完后开始设置B的虚函数指针为B的虚函数表。
6. 调用B构造函数的用户代码

&emsp;&emsp;析构顺序：
1. 调用B::~B析构函数；
2. 设置虚函数表指针为B的虚函数表；
3. 执行B析构的用户代码；
4. 调用基类A::~A()，该过程中先设置虚函数表指针为A的虚函数表再调用A的用户代码。

&emsp;&emsp;从上面的过程中大概也能看出```cxa_pure_virtual```可能被调用的时机。当类被析构时，基类的析构稍微比较耗时时，第二个线程尝试访问当前类的一个被重写的纯虚函数，由于此时的虚函数表中的纯虚函数已经被修改为```cxa_pure_virtual```就会直接abort。那我们复现下：
```c
class ClassA {
public:
    ClassA() {
        printf("Class A \n");
    }
    virtual ~ClassA() {
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    virtual void func() = 0;
};

class ClassB : public ClassA {
public:
    virtual ~ClassB() {
        printf("Class B \n");
    };

    virtual void func() override {
        printf("Class B func\n");
    }
};

void func(ClassA *p) {
    while (1) {
        p->func();
    }
}

int main(){
    std::cout << "Hello World!\n";
    ClassA* p = new ClassB;
    auto t = std::thread(func, p);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    delete p;
    t.join();
}
```

&emsp;&emsp;上面的代码中在析构函数中加了sleep函数来保证对象被析构过程中卡在基类的析构函数，第二个线程尝试访问该纯虚函数。
&emsp;&emsp;再clang/gcc系列编译器上触发的是```cxa_purer_virtual```，而msvc触发的是```_purecall```。

```c

extern "C" int __cdecl _purecall()
{
    _purecall_handler const purecall_handler = _get_purecall_handler();
    if (purecall_handler)
    {
        purecall_handler();

        // The user-registered purecall handler should not return, but if it does,
        // continue with the default termination behavior.
    }

    abort();
}
```

