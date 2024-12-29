# 《Effect C++》阅读笔记
## 一 然自己习惯C++
### 1 视C++为一个语言联邦
&emsp;&emsp;C++语言本身的出身和目标和其名称表达的意思相近，作为C语言的超集。C++的最初的目标是在保证对C的完全兼容的前提下扩充面向对象的能力，提升研发效率。典型的就是早期的C with Class版本，但是当C++继续发展出现重载、虚函数、模板之后这一目标已经无法完全保证了。因此无法简单的将C++看作C的超集，只能将其看作部分兼容C的面向对象语言。
&emsp;&emsp;为了更加清晰的认识C++可以将C++看作一个由多个次语言组成的语言的集合而不是某个单一语言，而每个次语言的规则简单、通俗易懂。可以划分的次语言分为以下四种：
- C。C++中的作用域、基本语法、预处理器、指针等都来自于C，当不使用C++特性时，写C++和写C相差不大；
- 面向对象的C++。面向对象的C++相比于C语言添加了封装、继承、多态等特性，也是C++相对于C的优势所在；
- 模板C++。模板编程功能强大，也不可捉摸，模板的强大带来了新的编程泛型——模板元编程，虽然大多数情况可能并不是很需要这部分功能，但是学会使用能让你在C++工程中游刃有余；
- STL。STL本身是一个模板程序库，不太像是语言的一部分，但是缺少它使用C++编程将变得很艰难。STL有自己的一套对容器、迭代器、算法和函数对象进行操作的规定，这套规定是独立于C++语言本身之外的。

### 2 尽量以const、enum、inline替换#define
> - 对于单纯常量，使用const或者enum替换#define；
> - 对于类似函数调用的宏，改用inline函数替换#define。

&emsp;&emsp;宏定义来自于C语言，其预编译阶段进行替换的规则导致一些缺陷：
- 无类型检查。当多层宏定义嵌套时可能发生隐式类型转换，导致不可见的错误；
- 影响可读性。宏定义本身有自己的标记符号可以帮助阅读代码但是当大量宏定义嵌套使用时其阅读难度难以想象；
- 难以debug。现代IDE可以插入断点进行调试，但是断点不会像函数那样进入宏定义，只会在使用宏定义处断点，无法直观的查看宏定义内部使用的变量等内容；
- 代码膨胀。

&emsp;&emsp;相比于宏定义本身const、enum、inline分别对应使用宏定义的预设值常量定义、预设标志位定义、代码复用三种场景：
- 预设值常量定义。这种方式就是定义一个默认的值供程序调用，也是一般宏定义使用最广的用法之一。使用const相比于使用宏定义而言，const定义的是一个变量有自己的作用域、类型检查等，能够保证变量不规范使用在编译期警告，并且可以将相关的变量的定义和具体的类或者文件相关联（当然编译器对const的实现本身的缺陷后面再谈）；
- 预设标志位。宏定义另一种常用的场景就是设置标志位，标志位相比于一些预设值而言不应该是一个变量，用户不应该获取到对应的指针，因此可以使用enum代替，enum代替的行为和宏定义很想但是提供了类型检查；
- 代码复用。代码复用一般是使用函数对具体功能进行封装，而使用宏定义相比于使用函数封装更加高效，因为不存在函数的寻址、堆栈操作、调用等内容。inline同样可以保证效率的同时给予类型检查并且方便调试。虽然inline并不保证对应的函数调用一定能inline，但是对于大多数简单的函数而言基本能够保证，当函数过大不适合inline时，宏定义也不应该被考虑。

### 3 尽可能使用const
> - 将某些东西声明为const可以帮助编译器检查出错误用法。const可被施加于任何作用域的对象、函数参数、函数返回类型、成员函数本体；
> - 编译器强制实施bitise constness，但是编程时应该使用conceptual constness；
> - 当const和non-const成员函数有着实质等价的实现时，令non-const本本调用const版本可避免代码重复。

&emsp;&emsp;const常用的使用场景为定义一个变量和修饰类的成员函数。
&emsp;&emsp;const在修饰变量时，对于普通变量其含义很简单就是常量。而对于指针的话根据const修饰的位置分为顶层const和底层const，具体使用哪种可以根据实际需求而定。
&emsp;&emsp;const修饰成员函数时，其表示该成员函数可作用于const对象上并且返回的对象是const，这对有些场景很重要（有些场景用户需要明确调用函数之后不会改变对象本身的状态）。但同时带来一个问题，如何定义对象的状态不变，可能的概念有：
- bitwise constness（physical constness）：不修改对象本身的每一块内存，即不修改对象的每个成员变量即可。但是有时也会出现符合bitwise constness但是对象的行为确实非const的情况。比如对象拥有一个指向上下文环境对象的指针，
- logical constness（conceptual constness）：可以修改对象的内容，但是只有使用类的端检测不出该情况时才如此。

&emsp;&emsp;一般比较好的做法是实现const和non-const的成员函数，但是这样又会造成代码重复，可以在non-const中调用const的实现减少代码重复，但是需要对传入的对象进行const化，并将const实现的返回值const化。

### 4 确定对象被使用前已经被初始化
>- 为内置类型对象进行手工初始化，因为C++不保证初始化它们；
>- 构造函数最好使用成员初始化列表，而不要在构造函数中赋值。初始化列表的成员变量其次序应该和类中声明次序相同，方便检查；
>- 为避免跨单元的初始化顺序问题，以local static对象替代non-locl static对象。

&emsp;&emsp;未初始化的变量或者对象的内存结构是不确定的，可能会导致预期外的bug，因此对于任何变量在声明处就要初始化，而自定义对象要在构造函数中设置默认值。设置默认值的方式有两种：使用初始化列表或者在构造函数中赋值，建议使用第一种，该种方式效率相对于第二种比较高。当然只是建议，因为也可能存在对象的成员变量比较多的同时也有多个构造函数，此时可以将初始化操作封装为一个特定的api供构造函数调用减少不必要的工作。
&emsp;&emsp;C++对象的初始化顺序是基类首先初始化然后是子类，初始化列表中的成员的初始化顺序和类中声明的顺序相关。
&emsp;&emsp;还有一类是函数外的static对象（函数内的static成为local static，函数外的成为non-local statis），一般在函数使用该对象之前基本能够保证对象的初始化，但是如果一个non-local static对象的初始化依赖于另一个non-local statis对象，此时无法保证二者能够按照预定的顺序初始化。因为C++保证，函数内local static对象会在该函数被调用期间首次遇到该对象的定义时被初始化，因此可能的做法是将non-local static转换成local static，即将static对象定义在函数内，然后函数返回对应对象的引用或者指针进行访问，也就是一般的单例类。

## 二 构造、析构、赋值运算
### 5 了解C++默默编写了哪些函数
>- 编译器可以暗自为class创建default构造函数、copy函数、copy assignment操作符和析构函数。
```cpp
class myclass{
public:
    myclass(){}
    myclass(const myclass &cls)
        :mMem1(cls.mMem1){
    }

    myclass& operator=(const myclass &cls){
        mMem1 = cls.mMem1;
        return *this;
    }
    ~myclass(){}
public:
    int mMem1;
};
```

&emsp;&emsp;如果自定义类并未声明构造函数或者析构函数，编译器会为类创建相关的函数，创建的时机是当使用到相应的构造函数时。默认生成的析构函数时非virtual，拷贝构造只能保证对象内的非静态成员之间1对1的赋值。当用户定义的类内创建了相关构造函数或者析构函数时，编译器便不再创建对应的函数。
&emsp;&emsp;当类的拷贝赋值函数被声明为private时，编译器将不会为其子类创建对应的拷贝赋值函数。另外需要注意的时因为编译器生成的拷贝构造函数或者拷贝赋值函数只是将成员简单赋值，因此会涉及到浅拷贝和深拷贝的问题。

### 6 如果不想使用编译器自动生成的函数，就该明确拒绝
>- 为驳回编译器自动提供的机制，可将相应的成员函数声明为private并不予实现或者将对应的构造函数声明为delete。
&emsp;&emsp;对于某些特定的类我们可能不希望该对象被拷贝或者被在外部构造，可以将对应的构造函数声明为private或者delete，防止编译器自动生成默认的构造函数产生预期之外的情况。

### 7 为多态基类声明virtual析构函数
>- 带多态性质的基类应该声明一个虚析构函数，如果类带有任何virtual函数，更应该拥有一个虚析构函数；
>- 类的设计目的如果不是作为基类使用，或者不是为了具备多态性，就不该声明virtual析构函数。
&emsp;&emsp;类声明虚析构函数能够保证在多态特性下的类能够有效析构，但是对于普通类也会带来额外的空间开销（虚函数表），因此并不是所有类都要声明虚析构函数，只有对于需要多态性质的类声明析构函数。

### 8 别让一场逃离析构函数
>- 析构函数绝对不要throw异常。如果一个被析构函数调用的函数可能抛出异常，析构函数应该捕捉任何异常，然后处理结束程序；
>- 如果用户需要对某个操作函数运行期间抛出异常作出反应，那么类应该提供一个普通函数中（而不是析构函数）执行该操作。

&emsp;&emsp;C++并不禁止析构函数throw异常，但是在析构函数中抛出异常会导致未定义的行为。析构函数应该保证要么销毁完成，要么销毁失败原内存依然有效的语义，但是对于抛出异常的析构函数该语义是不可保证的。
&emsp;&emsp;因此如果某些销毁动作可能失败，要么在析构函数中处理错误，要么提供额外的接口处理不要再析构函数中进行。
&emsp;&emsp;当然异常一直饱受诟病，甚至不使用异常就不用考虑该问题。[对使用 C++ 异常处理应具有怎样的态度？](https://www.zhihu.com/question/22889420)

### 9 绝对不要在构造函数或者析构函数中使用virtual函数
>- 在构造和析构期间不要调用virtual函数，因为这类调用会违背虚函数的语义不会下降至子类。

&emsp;&emsp;在析构函数或者构造函数中调用虚函数会使得虚函数失去多态语义，因为此时构成多态语义的虚函数表尚未被完整的创建。换言之，无论对构造还是析构而言，当前类已经不是完整的类。下列代码的结果为调用了base中的init。
```cpp
class base {
public:
    base() {
        init();         //期望调用的时derived::init
    }
    virtual void init() {
        printf("i am base constructor\n");
    }
};

class derived : public base {
public:
    derived() {}
    virtual void init() {
        printf("i am derived constructor\n");
    }
};
```

### 10 令operator=返回一个*this的引用
>- 令赋值操作符返回一个*this的引用。

### 11 在operator=中处理自我赋值
>- 确保当前对象自我赋值时operator=有良好的行为。其中技术包括比较来源对象和目标对象的地址、精心周到的语言顺序、以及copy-and-swap；
>- 确定任何函数如果操作一个以上的对象，而其中多个对象是同一个对象时，仍然运行正确。
&emsp;&emsp;可能出现对象自我赋值的情况，有效的处理该种情况可以通过：
- 拷贝前进行判断，如果为相同的对象则不进行拷贝
- 最后销毁被拷贝的对象避免数据的丢失；
- 通过拷贝构造函数和swap进行安全性保证。
```cpp
//拷贝前判断
base& base::operator=(const base& b){
    if(*this == b){return *this}
    .........
}
//调换语言的顺序，假设mdata是base的数据域
base &base::operator=(const base &rst){
    data *ptr = mdata;
    mdata = new data(*rst.mdata);
    delete ptr;
    return *this;
}
//使用swap，假设拷贝构造中进行了深拷贝
base &base::operator=(base b){
    swap(b.mdata, this->mdata);
    return *this;
}
```

### 12 复制对象时勿忘其每一个成分
>- copying函数应该确保赋值对象内的所有成员变量以及所有基类成分；
>- 不要尝试以某个拷贝函数实现另一个拷贝函数，应该将共同机制放入第三个函数中，并由两个拷贝函数调用。

## 三 资源管理
### 13 以对象管理资源
>- 为了防止资源泄露，请使用RAII对象，它们在构造函数中获得资源并在析构函数中释放资源；
>- 常被使用的RAII classes分别为shared_ptr。

&emsp;&emsp;智能指针能够帮助管理对象，自动销毁堆中的内存，而不是等待手动释放。

### 14 在资源管理类中小心copying行为
>- 复制RAII对象必须一并复制它所管理的资源，所以资源的copying行为决定ARAII对象的copying行为；
>- 普遍而常见的RAII class copying行为是：抑制copying、施行引用计数法。

&emsp;&emsp;对类资源的管理要使用到RAII的情况下面对RAII对象的复制可能的选择有：
- 复制其所掌握的资源，多份拷贝；
- 使用引用计数，所有共享引用计数的对象指向相同的资源类。比如shared_ptr；
- 禁止复制；
- 资源所有权转移，即原管理对象不再拥有资源。

&emsp;&emsp;无论是哪一种，需要确认的时你需要明确你面对的问题，以及你清楚你为什么这样做。

### 15 在资源管理类中提供对原始资源的访问
>- 每一个RAII class都应该提供取得其所管理的资源的办法；
>- 对原始资源的访问可能经由显示转换或者隐式转换。采用哪种方式可根据场景判断，一般而言采用显示转换像是在说明：“我很清楚我在做什么”。

&emsp;&emsp;资源管理类能够高效简洁的进行资源的创建释放的管理，但是有些场景需要直接访问管理类管理的资源，因此资源管理需要提供相应的访问资源的接口。

### 16 成对使用new和delete时要采用相同的形式
>- 如果你在new表达式中使用new ```[]```，必须在相应的delete表达式中也是用```delete[]```，对于不适用```[]```的new和delet同理。
&emsp;&emsp;当使用```new []```申请数组内存时，如果使用```delete```释放对应的内存的行为是未定义的，其行为和编译器的实现有关。一个可能的结果时只是放了数组的第一个元素的内存。

### 17 以独立一句将newed对象置入智能指针
>- 以独立语句将newed对象存储于智能指针内，否则如果有异常可能难以察觉资源泄露。
&emsp;&emsp;对于参数为智能指针的函数调用，没必要图方便直接在参数中new创建指针，将该语句拆分为独立的语句更方便调试和可能发生的问题的追踪。

## 四 设计与声明
### 18 让接口容易被正确使用，不易被误用
>- 好的接口更容易被正确使用，不容易被误用。
>- 促进正确使用的办法时保证接口的一致性，以及与内置类型的行为兼容；
>- 阻止误用的办法包括建立新类型、限制类型上的操作，束缚对象值，以及消除客户的资源管理的责任；
>- shared_ptr支持定制删除器。

&emsp;&emsp;好用的接口应该预想使用接口的人可能如何使用接口，接口设计时应该遵守一定的规范和使用习惯保证接口的一致性，当明确不该如何使用时需要在代码中体现出来提前暴漏问题，而不是在运行时报bug。

### 19 设计class如同设计type
>- class设计就是type设计。
&emsp;&emsp;当设计一个新的class时其实就是设计了一种类型，设计时需要考虑：
- 新的type的对象应该如何被创建和销毁？
- 对象的初始化和赋值之间应该有什么样的差别？
- 新的type的对象如果被passed by vale，意味着什么？
- 什么时性的type的合法值？
- 新的type需要配合某个继承体系吗？
- 新的type需要什么样的类型转换？
- 什么样的操作符对此新type而言是合理的？
- 什么样的标准函数应该驳回？
- 谁该取用type的新成员？
- 什么是新type的未声明接口？
- 新的type的泛化如何？
- 真的需要一个新的type吗？

### 20 宁以pass-by-reference-to-const替换pass-by-value
>- 尽量以pass-by-reference-to-const替换pass-by-value，能够提升效率和避免对象切片；
>- 该规则不适用于内置类型以及STL的迭代器和函数对象，对它们而言pass-by-value往往比较合适。

&emsp;&emsp;pass-by-reference-to-const相对于pass-by-value效率要高，因为减少了一次对象的拷贝和析构工作。同时reference能够使用多态使用使得函数的功能更加泛化。但是对于内置类型这条规则并不适用，因为编译器对待内置类型和自定义类型的处理方式不一样。

### 21 必须返回对象时，别妄想返回其reference
>- 绝对不要返回pointer或者reference只想一个local stack的对象，或者reference指向一个heap的对象，或者返回pointer或者reference指向一个local static对象而有可能同时需要多个这样的对象。

&emsp;&emsp;和传参类似，返回值也会发生值拷贝，但是一般不建议返回引用，因为大多数情况返回值都是local构造的，local的对象会在函数返回后被销毁。除非明确返回的reference或者指针的生命周期。

### 22 将成员变量声明为private
>- 切记将成员变量声明为private，这可赋予客户访问数据的一致性、可细微划分访问控制、允诺约束条件获得保证，并提供class作者以充分的实现弹性；
>- protected并不比public更有封装性。

&emsp;&emsp;将成员变量声明为private，并通过相应的API访问读写可以有效地追踪成员变量的使用情况，并能够给予一定重新设计的弹性，我们无法保证我们当前的设计就是最好的，可能是符合当前场景的。如果以后需要进行扩展可能需要调整，相应的API仍然可以复用，但是内部的成员的修改并不影响用户的使用。

### 23 宁以non-member、non-friend替换member函数
>- 宁可拿non-member non-friend函数替换member函数，这样做能够增加封装性、包裹弹性和机能扩充性。

&emsp;&emsp;封装的目的是减少使用相关API或者类的用户尽可能少的了解到内部的实现，以及类可能的构造简化使用的方式。因此描述封装性的一个维度就是——有多少代码可以看到某一块数据，如果访问的函数越多则封装性越低。
&emsp;&emsp;在有些时候我们可以使用成员函数和non-member函数实现同样的功能，区别是成员函数适合类绑定的。这种时候可能需要考虑是否需要使用non-member函数，因为non-member函数只是暴露了当前函数和使用的参数而已。

### 24 若所有参数皆需类型转换，请为此采用non-member函数
>- 如果某个函数的所有参数都需要进行类型转换，那么这个函数必须是一个non-member function。

&emsp;&emsp;当某个函数的所有参数都需要类型转换时，也就意味着可能某个类型会被隐式类型转换成某个自定义类型，将该函数声明为non-member能够支持混合类型函数调用，如果声明为类成员函数，则可能导致某些场景无法支持。比如自定义参数类型的操作符重载。第一种实现无法支持```1 + myclass(2)```的调用，第二种都支持。
```cpp
class myclas{
    myclass(int a){}
public:
    myclass operator+(const myclass &rst){}
};

myclass operator+(const myclass &rst, const myclass &snd){}
```

### 25 考虑写一个不抛异常的swap函数
>- 当std::swap对你而言效率不高时，提供一个swap成员函数，并确定这个函数不抛出异常；
>- 如果你提供了一个member swap，也该提供一个non-member swap用来调用前者。对于class应该特化std::swap；
>- 调用swap时应针对std::swap使用using声明，然后调用swap并不带任何命名空间的修饰；
>- 为用户类型继续宁std templates全特化是好的，但是不要尝试在std内添加某些对std而言全新的东西。

## 五 实现
### 26 尽可能延后变量定义式的出现时间
>- 尽可能延后变量定义式的出现，可增加程序的清晰度并改善程序的效率。

&emsp;&emsp;对象的创建必定伴随着对象构造和析构的开销，在使用变量的附近创建对象能够使得程序的结构更加清晰。另外，程序大多伴随着很懂的if分支等，并不是所有分支都会到达，延迟对象的创建可以避免在运行时有额外的开销。同时合理安排对象创建的位置，比如一个对象可能for循环中和循环外创建都可以，此时创建在循环外效率更高。

### 27 尽量少做转型动作
>- 如果可以尽量避免转型，特别是在注重效率的代码中避免dynamic_cast，如果有个设计需要进行类型转换，尝试发展无需转型的设计；
>- 如果转型是必须的，试着将他隐藏在某个函数背后，用户调用该函数而不需要自己进行转型；
>- 宁可使用C++风格的类型转换，不要使用显示类型转换和隐式类型转换。

&emsp;&emsp;类型转换可能隐藏bug和不可预见的行为，使用C++风格的类型转换表明自己的意图能够提高排查错误的效率：
- const_cast：常量移除；
- static_cast：一般的类型转换，比如int和double等转换，non-const转const等；
- dynamic_cast：类型安全向下转型，因为实现的原因可能有巨大的性能损耗；
- reinterpret_cast：对给定的内存进行重新解释，不可移植，具体的动作取决于编译器。


### 28 避免返回handles指向对象内部成分
>- 避免返回handle（reference、指针、迭代器）指向对象内部。

&emsp;&emsp;返回指向对象内部对象的handle破坏了封装性的同时获取的handle本身也不是安全的，因为其对自己掌握的handle所对应的内存一无所知，该块内存可能在不确定的时刻被释放。

### 29 为异常安全而努力是值得的
>- 异常安全函数：即使发生异常也不会泄露资源或者允许任何数据结构被破坏，这样的函数区分为三种可能的保证：基本型、强烈型、不抛出异常型；
>- 强烈保证往往能够以copy-and-swap实现出来，但强烈保证并非对所有函数都可实现或者具备现实意义；
>- 函数提供的异常安全保证通常最高只等于其所调用的各个函数的异常安全中最弱者。

&emsp;&emsp;异常安全函数的基本要求：
- 不泄露任何资源；
- 不允许数据损坏。

&emsp;&emsp;异常安全函数的基本保证：
- 基本承诺：如果异常被抛出，程序内的对象任然保持在有效状态；
- 强烈保证：如果异常抛出，程序状态不改变，即要么完全成功，要么完全失败，有些像数据库中的事务；
- 不抛出异常保证：承诺绝不抛出异常。

&emsp;&emsp;需要根据实际的场景判断该使用哪种保证。强烈保证一般很难达到，最起码要提供最低限度的保证。另外，可以使用copy-and-swap方式提供有限度的异常安全。copy-on-swap即先操作拷贝的临时类，如果此时异常触发则损坏的时copy的对象原来的对象不受影响，成功后将临时类和当前对象交换。

### 30 透彻了解inline的里里外外
>- 将大多数inline限制在小型、被频繁使用的函数上，可使得日后调试过程和二进制升级更容易，可使得潜在的代码膨胀问题最小化，使得程序的运行速度最大化；
>- 不要只因为function templates出现在头文件中就将它们声明为inline。

&emsp;&emsp;C++中inline的行为很像宏，也是进行代码替换，但是多了类型检查。也就意味着inline和宏一样会造成代码体积增大的问题，因此在使用inline时应该尽可能针对短小的函数使用。inline本身是建议，而不是强制，因此编译器可能拒绝你的inline建议，比如包含virtual function调用和函数指针调用简单函数的情况。inline也有其他缺陷，虽然其声明和函数一样，但是行为上和函数不同，导致被inline的函数无法随着库的升级而升级的情况是存在的。最后模板inline时应该明确模板的每一个类都需要inline。
&emsp;&emsp;inline适用于短小的函数，有时需要准确判断函数是否针对短小，因为多层函数调用可能隐含着不可直接观察的长调用。

### 31 将文件间的编译依存关系降至最低
>- 支持编译依存性最小化的一般构想是：相依于声明式，不要相依于定义式，基于此构想的两个手段是handle classes和interface classes；
>- 程序库头文件应该以完全且仅有声明式的形式存在，这种做法不论是否设计template都适用。

&emsp;&emsp;当两个类拥有包含关系时，一个类的修改会导致另一个类的定义也需要被重新编译。一个可能办法是使用指针替换类内的某个其他类的对象，将声明的依存性替换定义的依存性，对应类的修改不会牵扯到当前类。
```cpp
class myclass{
public:
    string to_string();     //接口部分
private:
    string name;            //实现部分
}
//最小编译实现
class myclass{
private:
    shared_ptr<string> pstring;
}
```
&emsp;&emsp;另一中做法是抽象出接口，即虚基类，这种类仅仅描述接口不包含数据，当外界使用是具体使用的是指向基类的指针或者引用，使用多态的机制调用自定义的部分。

## 六 继承与面向对象实现
### 32 确定你的public继承表达出is-a的关系
>- public继承意味着is-a，适用于base class身上的每一件事情也应该适用于derived class上。

&emsp;&emsp;在面向对象中，我们常常使用is-a表示继承关系，has-a表示组合关系，后者相对比较好处理因为交互的两个关系不是那么密切。但是对于is-a，其含义往往依赖于我们希望表达的语义的自然语言，但是很多情况下自然语言表达事物的范畴是很模糊的，并且是动态变化的。而我们的程序需要时确定的，不能含糊的，因此在进行类设计时应该明确每个类体系中可能面对的情况并为可能到来的变化做准备。另外，概念是动态变化的，没有永恒的答案，只能不断适应场景，适应变化。

### 33 避免遮掩继承而来的名称
>- derived class内的名称会遮掩base class内的名称；
>- 为了让被遮掩的名称再见天日，可使用using 声明式或者转交函数。

&emsp;&emsp;C++每个作用域都有自己的可见范围，当局部作用域和外部作用域的定义式冲突时局部作用域的命名会遮掩外部作用域的定义式，使其不可见。一般不建议命名遮掩，因为这可能带来不可预见的错误。另外类继承体系中，成员函数也会发生命名遮掩，我们一般称之为重写，对于重写的函数如果仍然需要调用基类的函数可以显示声明或者显示调用。

### 34 区分接口继承和实现继承
>- 接口继承和实现继承不同，在public继承下子类总是继承基类的接口；
>- pure virtual函数只具体指定接口继承；
>- 简朴的impure virtaul函数具体指定接口继承和缺省实现继承；
>- non-virtaul函数具体指定接口继承以及强制性实现继承。

### 35 考虑virtual函数以外的其他选择
>- virtual函数的替代方案包括NV1手法以及Strategy设计模式的多种形式；
>- 将机能从成员函数移到class外部函数，带来的一个缺点，非成员函数无法访问class的non-public成员；
>- function对象的行为就像一般的函数指针。

**Non-Virtual Interface**：
&emsp;&emsp;使用NVI方式实现template method模式。NVI方式如下，即使用非virtual函数调用virtual版本的实现，该种方式能够给予virtual方法实现一定的灵活性，可以在具体事件发生前进行准备工作，发生后进行清理工作。其中protected也可以是private。
```cpp
class myclass{
public:
    int get() const{
        ...
        doGet();
        ...
    }
protected:
    virtual int doGet() const{}
};
```

**Function Pointers实现Strategy**：
&emsp;&emsp;Strategy模式实现方式如下，该方式可以将具体的函数实现方式和类解绑，同一类的不同实体，不同时期可以有不同的实现方式。Strategy的实现也不仅仅局限于函数指针，函数对象也能够完成相应的工作。相比于经典的Strategy，经典的Strategy是将具体的func的实现抽象为类并使用虚函数抽象具体的实现，基本思想相同。
```cpp
class myclas{
public:
    typedef int (*func)(const myclass &);
    myclass(func f):mfunc(f){}
    int get() const{
        return f(*this);
    }
private:
    func mfunc;
};
```

&emsp;&emsp;采用virtual函数固然能够有效的对对象的实现进行抽象，但是其是和类深度绑定的，可以考虑使用以下几种方式替代：
- NVI，给予virtual更多的灵活性；
- virtual函数替换城函数指针成员变量；
- 以function对象替换virtual函数；
- 将继承体系内的virtual函数替换为另一个继承体系内的virtual函数，即Strategy实现。

### 36 绝不重新定义继承而来的non-virtual函数
>- 绝对不要重新定义继承而来的non-virtual函数。

&emsp;&emsp;重新定义继承而来的non-virtual函数可能导致行为的不一致性。

### 37 绝不重新定义继承而来的缺省参数值
>-  绝不重新定义继承而来的缺省参数值，因为缺省参数是静态绑定。

&emsp;&emsp;修改virtual函数的缺省参数可能出现和预期不一致的效果。virtual函数是动态绑定，但是函数参数是静态绑定，即默认参数不会像我们预期的那样工作。
```cpp
class base {
public:
    virtual void func(int a = 3) {
        printf("%d\n", a);
       }
};

class derived : public base {
public:
    virtual void func(int a = 4) {
        printf("%d\n", a);
    }
};

int main() {
    base *b = new derived;
    //运行结果为3，期望4
    b->func();
    return 0;
}
```
### 38 通过组合构造出has-a或者根据某物实现出
>- 组合的意义和public继承不同；
>- 应用域，组合意味着has-a；实现域，组合意味着is-implemented-in-terms-of。

&emsp;&emsp;组合即当前对象中包含其他对象，即has-a的关系。而应用域即所面对的场景而言，实现域即具体实现方式而言。当设计的类不符合is-a时往往has-a都能满足需求。

### 39 明智而审慎地使用private继承
>- private继承意味着is-implemented-in-terms-of，通常比组合的级别低；
>- 和组合不同，private继承可以造成empty base最优化。

&emsp;&emsp;private继承的所有基类的成员都会变成private，这使得当前类不具备基类的属性，因此无法构成is-a，只能是is-implemented-in-terms-of。相比于组合，private继承只有在涉及到virtual function或者protected成员时使用比较合理，同时private继承能够带来编译依存性最小化的额外优点，以及empty base最优化的特点。

### 40 明智而审慎地使用多重继承
>- 多重继承比单一继承复杂，可能导致新的歧义性；
>- virtual继承会增大类的大小、速度、初始化复杂度等成本，如果virtual base class不附带任何数据，将是最有实用价值的情况；
>- 多重继承的确有正当用途，比如接口继承等。

&emsp;&emsp;多重继承中基类可能有多个重复的声明，导致在使用具体function时导致歧义，可能的解决方案是使用virtual继承或者指明调用的对象。多重继承中菱形继承会导致单个基类的多个副本的存在，虽然virtual可以消除，但是virtual的虚函数表初始化等工作也会有额外的开销。
&emsp;&emsp;多重继承用于接口继承或者类之间的协助时其行为和继承有所区别。

## 七 模板与泛型编程
### 41 了解隐式接口和编译期多态
>- 类和模板都支持接口和多态；
>- class支持显式接口，以函数签名为中心，多态通过virtual函数发生于运行期；
>- template擦拭农户而言，接口是隐式的，基于有效表达式，而多态则通过template具现化和函数重载解析发生于编译期。

&emsp;&emsp;普通的virtual多态机制时显式接口，运行时多态，而模板是隐式接口，编译期多态。显式接口指在类定义中以一些列的成员函数显式的声明当前类支持的接口，而隐式接口至在具体的模板定义中模板的参数应该支持哪些接口。

### 42 了解typename的双重含义
>- 声明template参数时，前缀关键字class和typename可以互换；
>- 请使用关键字typename标识嵌套从属类型名称；但不在base class list或者member initialization list内以它作为base class的修饰。

&emsp;&emsp;独立类型名称：不依赖于模板类型的类型；
&emsp;&emsp;从属类型名称：依赖于模板类型的类型。

### 43 学习处理模板基类内的名称
>- 可在继承类模板内通过```this->```指涉基类模板内的成员名称，或者明确指定调用基类的函数的指示符。

&emsp;&emsp;模板中因为模板无法推断出类型可能继承自哪里，因此调用子类的方式时会报错，比如如下代码会报错无法找到func标识符。
```cpp
template<class T>
class number {
public:
    void func(T t) {
        cout << t << endl;
    }
};

template<class T>
class digit : public number<T> {
public:
    void myfunc() {
        func();
    }
};
```
&emsp;&emsp;解决方式有三种：
- 在基类调用之前使用```this->```，```this->func()```；
- 使用using声明，```using digit<T>::func```；
- 显式调用，```number<T>::func()```。

&emsp;&emsp;三种方式都是明确告诉编译器我要调用哪个函数。

### 44 将与参数无关的代码抽离模板
>- 模板生成多个class和多个函数，所以任何模板代码都不应该与某个造成膨胀的模板参数产生相互依存的关系；
>- 因非模板参数而造成代码膨胀，往往可以以函数参数或者成员变量替换模板参数消除；
>- 因模板参数造成的代码膨胀问题，往往可以降低，做法是让带有完全相同二进制表述的实例共享实现码。

&emsp;&emsp;因为模板存在展开的过程，如果模板代码与某个模板参数依赖过深，泽在展开的同时会生成多个相似的副本，增大代码体积。为了更好的避免代码膨胀，需要明确的功能和模板参数的依赖，并从中抽出相似的部分进行重新封装减小副本的体积。

### 45 运用成员函数模板接受所有兼容类型
>- 使用成员函数模板生成可接受所有兼容类型的函数；
>- 如果声明成员模板用于繁华copy构造或者泛化copy赋值操作，

&emsp;&emsp;为了让模板能够根据继承体系构造可以利用类型转换进行，比如智能指针基类指针指向子类的情况：
```cpp
template<typename T>
class myshared_ptr{
    T *ptr;
public:
    template<typename U>
    myshared_ptr(const myshared_ptr<U>& other) : ptr(other.get()){}
    T* get() const{ return ptr; }
};
```
### 46 需要类型转换时请为模板定义非成员函数
>- 当编写一个class template，而它所提供的“与此template相关的”函数支持“所有参数隐式类型转换”时，将那些函数定义为“class template内部的friend函数”。

&emsp;&emsp;面对以下场景时，我们希望的是在函数调用时编译器能够根据调用的方式自动推导模板的参数类型，然后准确调用相关函数。但是事实上模板函数调用无法触发隐式类型转换。因为隐式类型转换是先有函数，再有根据参数的类型进行转换。但是模板函数的第一步需要根据函数的传参推断可能的函数，即该如何特化函数实体，此时并没有具体的函数。隐式类型转换和模板函数类型推断的过程矛盾，因此无法完成转换。
```cpp
template<class T>
class number {
public:
    T val;
    number(T v) {
        val = v;
    }
};

template<typename T>
const number<T> operator+(number<T> &rst, number<T> &snd) {
    return rst.val + snd.val;
}

int main() {
    number<int> int1 = 1;
    number<int> int2 = 2;
    number<int> ret = int1 + int2;          //编译正常
    ret = int2 + 2;                         //编译错误
    return 0;
}
```
&emsp;&emsp;解决方式是将模板函数和类推断绑定，如声明为类的friend函数，当类被推断时附带friend函数的实体也被推断。
```cpp
template<class T>
class number {
public:
    T val;
    number(T v) {
        val = v;
    }

    friend const number operator+(const number &rst,const number &snd) {
        return number(rst.val + snd.val);
    }
};
```

### 47 使用traits classes表现类型信息
>- trait class使得类型信息在编译期可用；
>- 整合重载技术后，traits class有可能在编译期对类型进行if elss测试。

&emsp;&emsp;当希望通过类的类型信息针对不同场景进行有选择的操作时，可以设计trits class完成。其中traits class附带类的类型信息。
&emsp;&emsp;设计实现一个traits class：
- 确认若干你希望将来可取得的类型信息；
- 为该信息选择一个名称；
- 提供一个template和一组特化版本，内涵你希望支持的类型相关信息。

&emsp;&emsp;在设计完trait class之后就不应该使用```if(typeid() == typeid())```之类的代码进行运行期类型检查，应该利用重载技术在编译期根据不同类型调用不同的函数版本实现。

### 48 认识模板元编程
>- TMP（模板元编程）可将工作由运行期转移至编译期，因而得以实现早期错误侦测和更高的执行效率；
>- TMP可被用来生成基于政策组合的客户定制代码，也可用来避免生成对某些特殊类型并不适合的代码。

&emsp;&emsp;TMP的基本理念是利用模板编译期决定实体的能力，将部分本应该在运行期执行的代码在编译期就被完成。TMP已经被证明是图灵完全机器，足以表达所有的计算机功能，可以使用TMP声明变量、执行循环、编写及调用函数等。但是需要注意的是TMP的循环不是真正的循环而是利用模板递归实现的。TMP有几个优点：
- 扩展C++的实现能力，让原来无法实现的部分能力得以实现；
- 将工作从运行期转移至编译期，提前预知可能的错误；
- 降低运行期的运行时间，更小的可执行文件。
&emsp;&emsp;缺陷也很明显，编译时间边长。
&emsp;&emsp;假设现在有两个类taga和tagb表示两个类别，现在需要根据不同的tag调用不同版本的执行程序```execa```和```execb```。下面是两个版本的实现：
```cpp
//普通版本
void func(tag t){
    if(typeid(t) == typeid(taga)){
        execa();
    }else if(typeid(t) == typeid(tagb)){
        execb();
    }
}
//TMP
void funct(taga a){
    return execa();
}

void funct(tagb b){
    return execb();
}

void func(tag t){
    return funct(t); 
}
```
## 8 定制new和delete
### 49 了解new-handler的行为
>- set_new_handler允许客户制定一个函数，在内存分配无法获得满足时被调用；
>- newthrow new是一个颇为局限的工具，因为它只适用于内存分配；后续的构造函数调用还是可能抛出异常。

&emsp;&emsp;当new无法有效申请到内存时会首先调用用户自定义的handle程序，然后抛出异常，该handle程序就是new_handler，用户需要使用```set_new_handler```指定处理程序。
```cpp
void alloc_error(){
    std::cout<<"allocate memroy error\n";
}

int main(){
    std::set_new_handler(alloc_error);
}
```
### 50 了解new和delete的合理替换时机
>- 自定义new_handler有很多好处。

&emsp;&emsp;设计良好的new_handler需要：
- 让更多的内存可用；
- 安装另一个new_handler；
- 卸载new_handler抛出bad_alloc相关异常；
- 不返回。


&emsp;&emsp;尝试替换掉new和delete的理由：
- 监测运行时错误；
- 增强性能，比如内存池之类的；
- 收集使用上的统计数据；
- 降低分配和释放的速度；
- 降低缺省内存管理器带来的空间额外开销；
- 弥补缺省分配器中的非最佳对齐；
- 将相关对象收集在特定地址簇中；
- 自定义行为。

&emsp;&emsp;替换new或者delete可能的方式，具体的实现方式需要根据具体的场景进行判断。
```cpp
//自定义operator new并修改尺寸，在首4个字节写入版本信息
void *operator new(std::size_t size) throw(std::bad_alloc){
    size_t sz = size + 2 * sizeof(int);
    void *ptr = malloc(sz);
    if(!ptr){
        throw bad_alloc();
    }

    *(static_cast<int*>(ptr)) = 0x01;
    return ptr + sizeof(int);
}
```

### 51 编写new和delete时需要固守常规
>- operator new 应该内含一个无穷循环，并在其中尝试分配内存，如果它无法满足内存要求，就应该调用new_handler，也应该能够处理0byte申请，class版本也应该处理比正确大小更大的错误申请；
>- operator delete应该收到nullptr时什么也不做，operator delete也应该处理class大小不一致的问题。

&emsp;&emsp;这里建议使用while循环保证分配正确，但是之前在知乎上看到过相关的争论，结论是当第一次无法正确申请时应该exit程序，没有任何理由继续申请，因为此时大概率还是会继续报错。

### 52 写了placement new也要写placement delete
>- 当你写一个placement operator new，请确定也写出了对应的place operator delete；
>- 当你声明placement new和placement delete，请不要无意识地遮掩正常版本。

&emsp;&emsp;当new申请内存时，一般是先申请内存再调用类的构造函数。new、placement new和operator new的区别：
- new不能被冲在，行为总是一致，先分配内存再调用构造函数返回指针；
- operator new作为一个operator可以被重载，如果类没有重载operator new则调用的是全局的new；
- placement new只是operator new重载的一个版本，它并不分配内存，它允许用户将对象创建在指定的内存上，可以使堆也可以是栈。


## 9 杂项
### 53 不要忽略编译期的警告
>- 严肃对待编译期的警告信息；
>- 不要过度依赖编译期的报警能力，不同的编译器对特定的实现的态度不同。

### 54 让自己书序包括TR1在内的标准库
>- C++标准库的主要机能由STL、iostream是、locals组成，并包括C99标准程序库；
>- TR1添加了智能指针、函数对象、hash容器、正则表达式等；
>- TR1本身只是一份规范，可以选择更好的实现比如Boost。

### 55 让自己熟悉Boost
>- Boost提供了很多额外的工具库支持，当然facebook的folly也可以。