# C++对象模型实验（clang虚函数表结构）

&emsp;&emsp;**摘要**：本科期间有对比过msvc，gcc，clang的内存布局，距今已经6-7年了，当时还是使用的c++11。时间过得比较久了，这部分内容特别是内存对齐似乎C++17发生了一些变化，因此再实践下C++类模型。本文描述了C++不同类型的实际内存模型实现，主要关注虚函数表的具体内存布局。虽然clang，msvc都提供了对应的命令让我们直接查看类对象的内存布局，但是我们自己解析一下理解更深一点儿。
&emsp;&emsp;**关键字**：c++，对象布局，clang++，msvc，g++
>&emsp;&emsp;测试环境：
- Target: x86_64-unknown-linux-gnu
- clang version 16.0.6 (https://github.com/llvm/llvm-project.git 7cbf1a2591520c2491aa35339f227775f4d3adf6)

>&emsp;&emsp;不同的编译器提供了不同的方式提取内存布局:
- ```clang++ -cc1 -emit-llvm -fdump-record-layouts -fdump-vtable-layouts```;
- ```cl /d1 reportSingleClassLayout[class_name] [filename]```;
- ```g++ -fdump-class-hierarchy```。
&emsp;&emsp;测试代码[CppObjectModelExperiment](https://github.com/grayondream/CppObjectModelExperiment)
## 1 无虚函数类
### 1.1 简单对象
&emsp;&emsp;即POD对象，对象中不存在虚函数，对象的内存仅仅有对象成员构成。C++的内存实现时表驱动，数据和函数分别存放在不同的位置，这里的```SimpleClass```同时包含了非静态成员，静态成员，非静态函数，静态函数。包含函数是为了更加直观的看到C++表驱动的实现方式，后续类只会包含虚函数，其他的函数就不会再列举。而类中给三个分别为不同类型的变量是为了查看内存对齐的策略，确认不同编译器对齐的策略是否不同。
```c
class SimpleClass{
    friend std::ostream& operator<<(std::ostream &os, const SimpleClass &cls);
public:
    SimpleClass(){}
    ~SimpleClass(){}

    static void staticFunc(){}
    void nonStaticFunc(){}

    int _nonStaticIntMember{44};
    bool _nonStaticBoolMember{false};
    short _nonStaticShortMember{3};
    static int staticIntMember;
};
```
&emsp;&emsp;下面是类中所有成员的地址，因为构造函数和析构函数比较特殊我们无法直接获取其函数指针。只能间接的访问栈当前函数ESP上一帧存储的指针即能获得当前调用函数的EIP，再根据固定的偏移获取当前函数的具体地址。下面的代码只能再clang++或者g++编译器上成功，msvc要另写，但是道理一样。
```c
void* fetchRunningFuncAddress(){
    //!TODO:此处的代码不可移植
    uint64_t rbp{};
    asm("\t movq 8(%%rbp),%0" : "=r"(rbp));
    return (void*)(rbp - 0x11);
}
```

```c
Simple Class:
size of class(bytes):                             8
align of class(bytes):                            4
address of SimpleClass::SimpleClass:              0x555555555934
address of class:                                 0x7fffffffd8b8
member address _nonStaticIntMember:               0x7fffffffd8b8
member address _nonStaticBoolMember:              0x7fffffffd8bc
member address _nonStaticShortMember:             0x7fffffffd8be
static function address staticIntMember:          0x555555558080
function addres staticFunc:                       0x5555555559d0
function addres nonStaticFunc:                    0x555555555a10
address of SimpleClass::~SimpleClass:             0x555555555a90
```
&emsp;&emsp;根据上面的地址我们大概能够画出下面的类结构图，可以看到非静态数据都存储在栈空间而且是连续的，除了成员本身占用的内存，其中还有内存对齐需要的内存。而函数都存储在代码段中，该部分代码在内存中的映射位置和堆很近（好像是废话，堆本身就很大）。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/85ea87ef302bccf681f25cba1f93359c.png)


```c
555555554000-555555555000 r--p 00000000 103:02 13918339                  /home/are/workspace/code/projs/CppObjectModelExperiment/build/src/cpp_object_model_experiment
555555555000-555555556000 r-xp 00001000 103:02 13918339                  /home/are/workspace/code/projs/CppObjectModelExperiment/build/src/cpp_object_model_experiment
555555556000-555555557000 r--p 00002000 103:02 13918339                  /home/are/workspace/code/projs/CppObjectModelExperiment/build/src/cpp_object_model_experiment
555555557000-555555558000 r--p 00002000 103:02 13918339                  /home/are/workspace/code/projs/CppObjectModelExperiment/build/src/cpp_object_model_experiment
555555558000-555555559000 rw-p 00003000 103:02 13918339                  /home/are/workspace/code/projs/CppObjectModelExperiment/build/src/cpp_object_model_experiment
7ffffffdd000-7ffffffff000 rw-p 00000000 00:00 0                          [stack]
```

### 1.2 单继承简单对象
```c++
class SimpleSingleInherit : public SimpleClass{
public:
    SimpleSingleInherit(){
    }

    ~SimpleSingleInherit(){
    }

    bool _ssiNonStaticBoolMem{false};
};
```
&emsp;&emsp;单继承比较简单直接继承上面的```SimpleClass```即可。上面说了函数地址就是固定存储在代码区的，后续除了虚函数我们就不关注这些内容了。
```c
address of SimpleClass::SimpleClass:                        0x555555555444
address of SimpleSingleInherit::SimpleSingleInherit:        0x555555555a85
address of class SimpleClass:                               0x55555556b2c0
size of class(bytes):                                       8
align of class(bytes):                                      4
member address _nonStaticIntMember:                         0x55555556b2c0
member address _nonStaticBoolMember:                        0x55555556b2c4
member address _nonStaticShortMember:                       0x55555556b2c6
static function address staticIntMember:                    0x555555558090
address of class SimpleSingleInherit:                       0x55555556b2c0
size of class(bytes):                                       12
align of class(bytes):                                      4
member address _ssiNonStaticBoolMem:                        0x55555556b2c8
address of SimpleSingleInherit::~SimpleSingleInherit:       0x555555555ce8
address of SimpleClass::~SimpleClass:                       0x555555555770
```
&emsp;&emsp;从上面的输出中能够看出来类和基类都有独立的空间，子类的padding依然会保留。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/3b073caf13ba837e5dd6506b936b60bf.png)

&emsp;&emsp;但是实际上这个padding空间似乎只是指当前类本身拥有的成员的需要padding的空间，比如下面的例子```_ssiNonStaticBoolMem```本身只占用一个字节空间不需要padding，因此后续```ssiNonStaticBoolMem2```紧跟前者的内存是合理的，最后的padding是编译器根据基类```SimpleClass```的对齐填充的padding。
```c
class SimpleSingleInherit2 : public SimpleSingleInherit{
public:
    bool _ssiNonStaticBoolMem2{false};
};
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/eae43b7953fe16afa2c7457e02dda7c5.png)

&emsp;&emsp;对于此类的内存空间我们应该只能有个概念，没有必要过多苛求，cppreference中对对齐的描述只是要求编译器对齐而已，具体怎么对齐似乎并没有规定。实际使用过程中应该以生产环境中的编译器实现为准。

### 1.3 多继承简单对象
```c
class SimpleClass2{
public:
    bool _nonStaticBoolMember{};
};

class SimpleMultiInherit : public SimpleClass, public SimpleClass2{
public:
    bool _ssiNonStaticBoolMem{false};
};
```

```c
address of SimpleClass::SimpleClass:                        0x5555555553f4
address of SimpleClass2::SimpleClass2:                      0x5555555566f7
address of SimpleMultiInherit::SimpleMultiInherit:          0x555555556397
address of class SimpleClass:                               0x55555556c2c0
size of class(bytes):                                       8
align of class(bytes):                                      4
member address _nonStaticIntMember:                         0x55555556c2c0
member address _nonStaticBoolMember:                        0x55555556c2c4
member address _nonStaticShortMember:                       0x55555556c2c6
static function address staticIntMember:                    0x555555559090
address of class SimpleClass2:                              0x55555556c2c8
size of class(bytes):                                       1
align of class(bytes):                                      1
member address _nonStaticBoolMember:                        0x55555556c2c8
address of class SimpleMultiInherit:                        0x55555556c2c0
size of class(bytes):                                       12
align of class(bytes):                                      4
member address _ssiNonStaticBoolMem:                        0x55555556c2c9
address of SimpleMultiInherit::~SimpleMultiInherit:         0x555555556628
address of SimpleClass2::~SimpleClass:                      0x555555556770
address of SimpleClass::~SimpleClass:                       0x555555555720
```
&emsp;&emsp;多重继承的类结构比较简单。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/db84ccefeedaf297b231d4ecffb3d8ef.png)


### 1.4 菱形继承对象
```cpp
class SimpleClassLeft : public SimpleClass{
public:
    bool _simpleLeftBool;
};

class SimpleClassRight : public SimpleClass{
public:
    bool _simpleRightBool;
};

class SimpleDiamandInherit : public SimpleClassLeft, public SimpleClassRight{
public:
    bool _simpleDiamandBool;
};
```

```c
address of SimpleClass::SimpleClass:                        0x5555555553f4
address of SimpleClassLeft::SimpleClassLeft:                0x555555556e8d
address of SimpleClass::SimpleClass:                        0x5555555553f4
address of SimpleClassRight::SimpleClassRight:              0x555555556f6d
address of SimpleDiamandInherit::SimpleDiamandInherit:      0x555555556b1f
address of class SimpleClass:                               0x55555556e2c0
size of class(bytes):                                       8
align of class(bytes):                                      4
member address _nonStaticIntMember:                         0x55555556e2c0
member address _nonStaticBoolMember:                        0x55555556e2c4
member address _nonStaticShortMember:                       0x55555556e2c6
static function address staticIntMember:                    0x55555555b090
address of class SimpleClassLeft:                           0x55555556e2c0
size of class(bytes):                                       12
align of class(bytes):                                      4
member address _simpleLeftBool:                             0x55555556e2c8
address of class SimpleClass:                               0x55555556e2cc
size of class(bytes):                                       8
align of class(bytes):                                      4
member address _nonStaticIntMember:                         0x55555556e2cc
member address _nonStaticBoolMember:                        0x55555556e2d0
member address _nonStaticShortMember:                       0x55555556e2d2
static function address staticIntMember:                    0x55555555b090
address of class SimpleClassRight:                          0x55555556e2cc
size of class(bytes):                                       12
align of class(bytes):                                      4
member address _simpleRightBool:                            0x55555556e2d4
address of class SimpleDiamandInherit:                      0x55555556e2c0
size of class(bytes):                                       24
align of class(bytes):                                      4
member address _simpleDiamandBool:                          0x55555556e2d5
address of SimpleDiamandInherit::~SimpleDiamandInherit:     0x555555556db8
address of SimpleClassRight::~SimpleClassRight:             0x555555557048
address of SimpleClass::~SimpleClass:                       0x555555555720
address of SimpleClassLeft::~SimpleClassLeft:               0x555555557118
address of SimpleClass::~SimpleClass:                       0x555555555720
```
&emsp;&emsp;菱形继承只是语义层面上时菱形的，但是实际上相同类型的基类完全是不同的对象，各自在内存中都有独立的副本。只是比较反直觉的是类```SimpleDiamandInherit```的成员复用了```SimpleClassRight```最后padding的内存。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/f5fef14d039f6f67e4a250aa1ec5eeac.png)


## 2 带虚函数类
&emsp;&emsp;涉及虚函数的类结构我们会重点关注虚函数表，其他成员的内存不再重点关注。
### 2.1 简单对象
```cpp
class VirtualClass{
public:
    VirtualClass(){ }
    virtual ~VirtualClass(){ }
    virtual void vfunc1(){
        std::cout<<"Virtual Class virtual function 1"<<std::endl;
    }

    virtual void vfunc2(){
        std::cout<<"Virtual Class virtual function 2"<<std::endl;
    }
    bool _vcBoolVal{};
};
```
&emsp;&emsp;类结构比较简单，我们不会主动调用类中的虚函数，而是通过虚函数表主动解析虚函数表拿到函数指针调用。具体如下：
```
address of VirtualClass::VirtualClass:                      0x5555555575e0
address of class VirtualClass:                              0x55555556e2c0
size of class(bytes):                                       16
align of class(bytes):                                      8
member address _vcBoolVal:                                  0x55555556e2c8
pares vtptr from this function
current class's typeinfo name:12VirtualClass
call virtual function from virtual pointer table
Virtual Class virtual function 1
Virtual Class virtual function 2
address of VirtualClass::~VirtualClass:                     0x555555557840
```
&emsp;&emsp;需要注意的是虚函数表的实现跟编译器有关，不同编译器实现不同，msvc虽然和clang等的实现大体一致，但是表项中的内容依然有区别。因此上面的实现仅供参考。
&emsp;&emsp;当类中有虚函数时，类起始位置会存储一个虚函数表指针，该指针指向一个表格即虚函数表。虚函数表中存储了不仅仅是虚函数还有RTTI的信息（这个和编译器有关，如果编译器实现本身不再虚函数表中存储RTTI信息也是正常的）。上面的例子中类起始位置就是一个8byte的指针，然后是具体的类成员。
&emsp;&emsp;虚函数表中的表项的数量并不是虚函数的数量，还有一些额外的表项。```VirtualClass```中直观能看到的有三个虚函数，但是表项中有5个成员，除了-1位置的typeinfo信息，多出一个表项。索引为1处也是一个函数指针，该函数指针时编译器生成的虚析构函数的包装器，大概的实现如下：
```
VirtualClass::~VirtualClass(VirtualClass *cls){
    cls->~VirtualClass();
}
```
&emsp;&emsp;简单总结下就是，虚函数表中存储的内容为：
- -1项存储RTTI；
- 0项存储虚析构函数；
- 1项存储虚析构函数的wrapper；
- 2项开始便是用户声明的析构函数的指针。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/6bda08a16d8c378b0c4dbbe1194c5a88.png)


### 2.2 单继承对象
```c
class VirtualSingleInherit : public VirtualClass{
public:
    virtual void vfunc1() override{
        std::cout<<"VirtualSingleInhert Class virtual function 1"<<std::endl;
    }

    virtual void vfunc3(){
        std::cout<<"VirtualSingleInhert Class virtual function 3"<<std::endl;
    }

    bool _vsiBoolValue{};
};
```
&emsp;&emsp;
```
address of VirtualClass::VirtualClass:                      0x555555557650
address of VirtualSingleInherit::VirtualSingleInherit:      0x555555557b19
address of class VirtualClass:                              0x55555556f2c0
size of class(bytes):                                       16
align of class(bytes):                                      8
member address _vcBoolVal:                                  0x55555556f2c8
pares vtptr from this function
current class's typeinfo name:20VirtualSingleInherit
call virtual function from virtual pointer table
VirtualSingleInhert Class virtual function 1
Virtual Class virtual function 2
address of class VirtualSingleInherit:                      0x55555556f2c0
size of class(bytes):                                       16
align of class(bytes):                                      8
member address _vsiBoolValue:                               0x55555556f2c9
pares vtptr from this function
current class's typeinfo name:20VirtualSingleInherit
call virtual function from virtual pointer table
VirtualSingleInhert Class virtual function 1
Virtual Class virtual function 2
VirtualSingleInhert Class virtual function 3
address of VirtualSingleInherit::~VirtualSingleInherit:     0x555555557d94
address of VirtualClass::~VirtualClass:                     0x5555555578b0
```
&emsp;&emsp;单继承类类本身的内存结构和不带虚函数相同，虚函数表有些不同，子类中重写的虚函数虚函数表项中会替换为重写的函数指针，而新增的虚函数会添加到虚函数的表的末尾。这也是为什么我们能够通过RTTI调用子类实际实现的虚函数的原因。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/dc702c4e8e0567ff1a454df45af24855.png)


### 2.3 多继承对象
&emsp;&emsp;这里对```VirtualClass,VirtualClass2```进行了简单的改造增加了各自单独的虚函数，方便观察虚函数结构的不同。
```c
class VirtualClass{
public:
    virtual void vfunc1(){
        std::cout<<"[**] Virtual Class virtual function 1"<<std::endl;
    }

    virtual void vfunc2(){
        std::cout<<"[**] Virtual Class virtual function 2"<<std::endl;
    }

    virtual void vc1func(){
        std::cout<<"[**] Virtual Class virtual function vc1func"<<std::endl;
    }
    bool _vcBoolVal{};
};


class VirtualClass2{
public:
    virtual void vfunc1(){
        std::cout<<"[**] Virtual Class virtual function 1"<<std::endl;
    }

    virtual void vfunc2(){
        std::cout<<"[**] Virtual Class virtual function 2"<<std::endl;
    }

    virtual void vc2func(){
        std::cout<<"[**] Virtual Class virtual function vc2func"<<std::endl;
    }

    bool _vcBoolVal2{};
};

class VirtualMultiInherit : public VirtualClass, public VirtualClass2{
public:
    virtual void vfunc1() override{
        std::cout<<"[**] VirtualMultiInherit Class virtual function 1"<<std::endl;
    }
    
    virtual void vc1func() override{
        std::cout<<"[**] VirtualMultiInherit virtual function vc1func"<<std::endl;
    }

    virtual void vc2func() override{
        std::cout<<"[**] VirtualMultiInherit Class virtual function vc2func"<<std::endl;
    }

    virtual void vmifunc(){
        std::cout<<"[**] VirtualMultiInherit Class virtual function vmifunc"<<std::endl;
    }
    bool _vmiBoolValue{};
};
```

```
address of VirtualClass::VirtualClass:                      0x555555558600
address of VirtualClass2::VirtualClass2:                    0x5555555592c0
address of class VirtualClass:                              0x5555555712e0
size of class(bytes):                                       16
align of class(bytes):                                      8
member address _vcBoolVal:                                  0x5555555712e8
pares vtptr from this function
current class's typeinfo name:19VirtualMultiInherit
call virtual function from virtual pointer table
[**] VirtualMultiInherit Class virtual function 1
[**] Virtual Class virtual function 2
[**] VirtualMultiInherit virtual function vc1func
[**] VirtualMultiInherit Class virtual function vc2func
[**] VirtualMultiInherit Class virtual function vmifunc
address of class VirtualClass2:                             0x5555555712f0
size of class(bytes):                                       16
align of class(bytes):                                      8
member address _vcBoolVal2:                                 0x5555555712f8
pares vtptr from this function
current class's typeinfo name:19VirtualMultiInherit
call virtual function from virtual pointer table
[**] VirtualMultiInherit Class virtual function 1
[**] VirtualClass2 Class virtual function 2
[**] VirtualMultiInherit Class virtual function vc2func
address of class VirtualMultiInherit:                       0x5555555712e0
size of class(bytes):                                       32
align of class(bytes):                                      8
member address _vmiBoolValue:                               0x5555555712f9
pares vtptr from this function
current class's typeinfo name:19VirtualMultiInherit
call virtual function from virtual pointer table
[**] VirtualMultiInherit Class virtual function 1
[**] Virtual Class virtual function 2
[**] VirtualMultiInherit virtual function vc1func
[**] VirtualMultiInherit Class virtual function vc2func
[**] VirtualMultiInherit Class virtual function vmifunc
address of VirtualClass2::~VirtualClass2:                   0x555555559570
address of VirtualClass::~VirtualClass:                     0x555555558860
```

&emsp;&emsp;多继承的情况要稍微复杂一点儿，多继承的情况下一个类有多少个基类就有多少个虚函数指针，如示例所示有两个虚基类的话就有两个虚函数指针。而对应虚函数表的地址就是基类的首地址，但是二者的内容大不相同。对于基类的同名虚函数，二则的虚函数表中的表项都会被替换成子类实现的虚函数指针，而如果子类中没有的虚函数则会被添加到第一个虚函数末尾，但也只是添加到了虚函数表的末尾，语法上基类访问子类的函数就不允许。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/b7dd6ba1349af25ab224e8222afa2537.png)


### 2.4 菱形虚继承对象
&emsp;&emsp;菱形继承如果不用虚继承的话就是和多继承相同，只不过多个类在不同基类中有多个副本而已。
```c
class VirtualDiamandLeft : virtual public VirtualClass{
public:
    virtual void vfunc1() override{
        std::cout<<"[**] VirtualDiamandLeft Class virtual function 1"<<std::endl;
    }

    bool _vdlBoolValue{};
};

class VirtualDiamandRight : virtual public VirtualClass{
public:
    virtual void vfunc2() override{
        std::cout<<"[**] VirtualDiamandRight Class virtual function 2"<<std::endl;
    }

    bool _vdrBoolValue{};
};

class VirtualMultiDiamand : public VirtualDiamandLeft, public VirtualDiamandRight{
public:
    virtual void vc1func() override{
        std::cout<<"[**] VirtualMultiDiamand Class virtual function vc1func"<<std::endl;
    }
    bool vmdBoolValue{};
};
```

&emsp;&emsp;菱形虚继承，相比于直接继承会复用同一个对象，不存在多个基类对象的副本。因此菱形继承中会有多个虚函数指针，首先是基类有一个虚函数表指针，该表格中存放的虚函数指针都是thunk，即改变this偏移再跳转到对应的处理函数：
```c
0000000000006300 <_ZTv0_n48_N19VirtualMultiDiamand7vc1funcEv>:
    6300:	55                   	push   %rbp
    6301:	48 89 e5             	mov    %rsp,%rbp
    6304:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    6308:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
    630c:	48 8b 07             	mov    (%rdi),%rax
    630f:	48 8b 40 d0          	mov    -0x30(%rax),%rax
    6313:	48 01 c7             	add    %rax,%rdi
    6316:	5d                   	pop    %rbp
    6317:	e9 24 ff ff ff       	jmp    6240 <_ZN19VirtualMultiDiamand7vc1funcEv>
    631c:	0f 1f 40 00          	nopl   0x0(%rax)
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/ddc0b57aedf925b0d52f8ffdd4667c6b.png)

&emsp;&emsp;另外除了基类的虚函数表中不仅仅存储了对应的虚函数指针还存储了当前虚函数表相对于基类虚函数表的偏移，该数据存储在```vptr[-3]```的位置。

## 参考文献
- [cppreference-align](https://zh.cppreference.com/w/c/language/object)  