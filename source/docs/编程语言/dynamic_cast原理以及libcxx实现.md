# dynamic_cast原理以及libcxx实现

&emsp;&emsp;**摘要**：最近在看一个崩溃的过程中详细看了一遍cxxabi的定义，就想着看一些llvm中cxxabi的一些实现。本文描述了cxxabi中```dynamic_cast```的实现以及原理。
&emsp;&emsp;**关键字**：```cxxabi```,```dynamic_cast```

## 1 简介
&emsp;&emsp;C++中，```dynamic_cast```用于有虚函数的继承链中父类型到子类型的安全转换。比较常见的用法如下：
```c
class A{
public:
    virtual ~A() = default;
};

class B{
};

A *p = new B;
B* pp = dynamic_cast<B*>(p);
```

&emsp;&emsp;```dynamic_cast```如何识别当前类的类型，这依赖于RTTI。C++中包含虚函数的对象都有一个虚函数表，一般情况下都在首地址（多继承和虚继承会有多个）有一个指向该虚函数表的虚函数表指针。虚函数表中有以下内容：
- 基类偏移；
- typeinfo；
- 如果有虚函数的话会有虚析构函数指针，一般情况下有两个；
> The entries for virtual destructors are actually pairs of entries. The first destructor, called the complete object destructor, performs the destruction without calling delete() on the object. The second destructor, called the deleting destructor, calls delete() after destroying the object. Both destroy any virtual bases; a separate, non-virtual function, called the base object destructor, performs destruction of the object but not its virtual base subobjects, and does not call delete().
- 虚函数指针，如果是虚继承对应的虚函数指针可能是一个thunk function。
> A segment of code associated (in this ABI) with a target function, which is called instead of the target function for the purpose of modifying parameters (e.g. this) or other parts of the environment before transferring control to the target function, and possibly making further modifications after its return. A thunk may contain as little as an instruction to be executed prior to falling through to an immediately following target function, or it may be a full function with its own stack frame that does a full call to the target function.

&emsp;&emsp;C++中就是通过虚函数表携带的typeinfo信息来确认当前类的类型，如果是该类型就就可以转换成功，否则的话就会转换失败。cxxabi的基本实现思路也是如此。

## 2 cxxabi实现
&emsp;&emsp;先来看一下cxxabi中对应实现函数的声明。
- ```static_ptr```：期望将进行转换的类地址；
- ```static_type```：当前类的类型信息；
- ```dst_type```：目标转换类的类型信息；
- ```src2dst_offset```：由 Itanium ABI 所规定的 hint 值，辅助优化用：
    - 是一个非负整数值，说明From是To的唯一一个公开非虚基类，且From基类子对象在一个To对象中的偏移为src2dst_offset；
    - -1表示无hint；
    - -2表示From不是To的公开基类；
    - -3表示To存在多个公开From基类，但是这些公开From基类都不是To的虚基类。

```c
extern "C" _LIBCXXABI_FUNC_VIS void *
__dynamic_cast(const void *static_ptr, const __class_type_info *static_type, const __class_type_info *dst_type, std::ptrdiff_t src2dst_offset) 
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/dynamic_cast.drawio.svg)

&emsp;&emsp;首先是从虚函数表中提取出当前类对象相对于子类首地址的offset和typeinfo，这两项都是固定存储在虚函数表指针所指向位置的-1和-2位置处。

> The offset to top holds the displacement to the top of the object from the location within the object of the virtual table pointer that addresses this virtual table, as a  ptrdiff_t. It is always present. The offset provides a way to find the top of the object from any base subobject with a virtual table pointer. This is necessary for dynamic_cast<void*> in particular.

```c
    void **vtable = *static_cast<void ** const *>(static_ptr);
    ptrdiff_t offset_to_derived = reinterpret_cast<ptrdiff_t>(vtable[-2]);
    const void* dynamic_ptr = static_cast<const char*>(static_ptr) + offset_to_derived;
    const __class_type_info* dynamic_type = static_cast<const __class_type_info*>(vtable[-1]);
```
&emsp;&emsp;接下来便是根据期望转换的目标对象的typeinfo和当前获取的typeinfo作对比进而选择是否将进行更进一步的转换。需要注意的是比较两个对象是否相同的方式：
```c
static inline bool is_equal(const std::type_info* x, const std::type_info* y, bool use_strcmp){
    // Use std::type_info's default comparison unless we've explicitly asked
    // for strcmp.
    if (!use_strcmp)
        return *x == *y;
    // Still allow pointer equality to short circut.
    return x == y || strcmp(x->name(), y->name()) == 0;
}
```
**相同类型**
&emsp;&emsp;typeinfo是编译器生成的，是静态对象用户只能读写，这里提供了使用typeinfo的名字和指针来比较的方式，是为了避免一些场景下typeinfo不一致导致转换失败（虽然ABI中默认是关闭的）。
&emsp;&emsp;针对typeinfo相同的情况下则需要根据```src2dst_offset```来辅助优化：
- src2dst_offset为非负整数时，fromt是to的唯一公开非虚基类。但是to类型是有可能有其他非公开基类的，因此需要比较vtptr中的偏移和src2dst_offset，能够匹配则直接返回偏移后的地址，否则返回空指针；
- src2dst_offset为-2表示，from不是to的公开基类，直接返回空指针；
```c
        if (src2dst_offset >= 0){
            // The static type is a unique public non-virtual base type of
            //   dst_type at offset `src2dst_offset` from the origin of dst.
            // Note that there might be other non-public static_type bases. The
            //   hint only guarantees that the public base is non-virtual and
            //   unique. So we have to check whether static_ptr points to that
            //   unique public base sub-object.
            if (offset_to_derived == -src2dst_offset)
                dst_ptr = dynamic_ptr;
        }else if (src2dst_offset == -2){
            // static_type is not a public base of dst_type.
            dst_ptr = nullptr;
        }
```
- src2dst_offset无法提供帮助时，只能所搜继承图来确认是否存在继承关系。此时只能搜索整张继承图，搜索出从最派生对象所有的from公开基类子对象。如果from所指向的对象是这些from基类子对象中的某一个，那么转换成功，转换结果是最派生对象指针；否则转换失败。在搜索继承图的过程中可以应用一些剪枝方法降低搜索开销。例如一旦搜索到from指向的对象就停止搜索、不搜索包含私有继承的路径等。
```c
info.number_of_dst_type = 1;
// Do the  search
dynamic_type->search_above_dst(&info, dynamic_ptr, dynamic_ptr, public_path, false);
```

**不相同类型**
&emsp;&emsp;typeinfo不相同时，表示源和目标只是在相同的继承图上，但是并不存在直接的公开继承关系，因此为了确认是否真的存在该关系只能搜索当前源和目标的继承图来确认。

> libcxxabi的dynamic_cast实现似乎有性能问题[https://reviews.llvm.org/D137315#3910662](https://reviews.llvm.org/D137315#3910662)这个patch修复了该问题。修复完的benchmark[https://gist.github.com/Lancern/212a26a3144343f459428dffe202cde0](https://gist.github.com/Lancern/212a26a3144343f459428dffe202cde0)

**搜索策略**
&emsp;&emsp;对于不同继承类型的类的 type info 有不同的搜索策略。例如对于有虚多继承的类的 type info（__vmi_class_type_info）、单继承的类的 type info（__si_class_info）等。搜索的方式看起来就是广度优先搜索再加上一些剪枝的优化。
