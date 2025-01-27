# 正确对你的C++代码进行性能测试——DoNotOptimize实现原理


**摘要**:在对某些底层特性进行性能测试的时候，通常由于编译器优化的原因导致无法准确测试某些基础能力的具体性能。的类似于Google-benchmark等一些性能测试库提供了基本的DotOptimaze之类的模板函数来避免编译器的优化，以保证准确的性能测试。本文描述了该是实现的具体细节，具体细节不是C++标准范围内，主要有编译器提供，不同编译器的实现细节不同，本文以clang为基准进行描述。
**关键字**:clang++,g++,asm

## 1 面对的问题
&emsp;&emsp;当我们需要测试一些基础能力的性能时往往由于编译器优化导致无法准确的进行测试。比如下面的代码中，我们期望测试```testFunc```的性能，但是编译器判断```testFunc```对程序没有任何新的影响而删除了或者对该语句进行排序。
>需要注意的是下面这种性能测试通常是可行的，所以一般不用担心这么写有任何副作用。上面提到的是一种可能性。
```cpp
int main(int argc, char **argv){
    std::cout<<"Hello World\n";
    auto st = std::chrono::high_resolution_clock::now();
    testFunc();
    auto et = std::chrono::high_resolution_clock::now();
    auto str = std::format("the function run in {} ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count());
    std::cout<<str;
    return 0;
}
```
&emsp;&emsp;编译器修改后的语句可能变成:
```cpp
int main(int argc, char **argv){
    std::cout<<"Hello World\n";
    auto st = std::chrono::high_resolution_clock::now();
    auto et = std::chrono::high_resolution_clock::now();
    auto str = std::format("the function run in {} ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count());
    std::cout<<str;
    return 0;
}
```
&emsp;&emsp;和或者
```cpp
int main(int argc, char **argv){
    std::cout<<"Hello World\n";
    testFunc();
    auto st = std::chrono::high_resolution_clock::now();
    auto et = std::chrono::high_resolution_clock::now();
    auto str = std::format("the function run in {} ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count());
    std::cout<<str;
    return 0;
}
```

&emsp;&emsp;产生这样的原因是现代处理器将都支持多级流水线执行，编译器为了性能最大化，会分析代码中的数据依赖，根据数据依赖关系和从对程序的影响来对代码进行删除、替换、排序等优化，来最大化CPU的吞吐率（比如常量传播等）。
&emsp;&emsp;比如下面的代码:
```cpp
int testFunc(){
    int a = 1;
    return a;
}

int main(int argc, char **argv){
    std::cout<<"Hello World\n";
    auto st = std::chrono::high_resolution_clock::now();
    testFunc();
    
    auto et = std::chrono::high_resolution_clock::now();
    auto str = std::format("the function run in {} ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count());
    std::cout<<str;
    return 0;
}
```
&emsp;&emsp;使用clang++-18编译器O3编译的返汇编如下，从下面的代码中看不到testFunc的任何调用就是因为编译器认为这部分代码没有任何意义就删除了。
```asm
00000000000023e0 <main>:
    23e0:	41 56                	push   %r14
    23e2:	53                   	push   %rbx
    23e3:	48 83 ec 38          	sub    $0x38,%rsp
    23e7:	48 8b 1d da cb 00 00 	mov    0xcbda(%rip),%rbx        ## efc8 <_ZSt4cout@GLIBCXX_3.4>
    23ee:	48 8d 35 eb 9f 00 00 	lea    0x9feb(%rip),%rsi        ## c3e0 <_IO_stdin_used+0x3e0>
    23f5:	ba 0c 00 00 00       	mov    $0xc,%edx
    23fa:	48 89 df             	mov    %rbx,%rdi
    23fd:	e8 ce fd ff ff       	call   21d0 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    2402:	e8 29 fc ff ff       	call   2030 <_ZNSt6chrono3_V212system_clock3nowEv@plt>
    2407:	49 89 c6             	mov    %rax,%r14
    240a:	e8 21 fc ff ff       	call   2030 <_ZNSt6chrono3_V212system_clock3nowEv@plt>
    240f:	4c 29 f0             	sub    %r14,%rax
    2412:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
    2419:	de 1b 43 
    241c:	48 f7 e9             	imul   %rcx
    241f:	48 89 d0             	mov    %rdx,%rax
    2422:	48 c1 e8 3f          	shr    $0x3f,%rax
    2426:	48 c1 fa 12          	sar    $0x12,%rdx
    242a:	48 01 c2             	add    %rax,%rdx
    242d:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
    2432:	48 8d 15 b4 9f 00 00 	lea    0x9fb4(%rip),%rdx        ## c3ed <_IO_stdin_used+0x3ed>
    2439:	48 89 e7             	mov    %rsp,%rdi
    243c:	4c 8d 44 24 20       	lea    0x20(%rsp),%r8
    2441:	be 1a 00 00 00       	mov    $0x1a,%esi
    2446:	b9 51 00 00 00       	mov    $0x51,%ecx
    244b:	e8 50 00 00 00       	call   24a0 <_ZSt7vformatB5cxx11St17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE>
    2450:	48 8b 34 24          	mov    (%rsp),%rsi
    2454:	48 8b 54 24 08       	mov    0x8(%rsp),%rdx
    2459:	48 89 df             	mov    %rbx,%rdi
    245c:	e8 6f fd ff ff       	call   21d0 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    2461:	48 8b 3c 24          	mov    (%rsp),%rdi
    2465:	48 8d 44 24 10       	lea    0x10(%rsp),%rax
    246a:	48 39 c7             	cmp    %rax,%rdi
    246d:	74 05                	je     2474 <main+0x94>
    246f:	e8 fc fc ff ff       	call   2170 <_ZdlPv@plt>
    2474:	31 c0                	xor    %eax,%eax
    2476:	48 83 c4 38          	add    $0x38,%rsp
    247a:	5b                   	pop    %rbx
    247b:	41 5e                	pop    %r14
    247d:	c3                   	ret
    247e:	48 89 c3             	mov    %rax,%rbx
    2481:	48 8b 3c 24          	mov    (%rsp),%rdi
    2485:	48 8d 44 24 10       	lea    0x10(%rsp),%rax
    248a:	48 39 c7             	cmp    %rax,%rdi
    248d:	74 05                	je     2494 <main+0xb4>
    248f:	e8 dc fc ff ff       	call   2170 <_ZdlPv@plt>
    2494:	48 89 df             	mov    %rbx,%rdi
    2497:	e8 e4 fd ff ff       	call   2280 <_Unwind_Resume@plt>
    249c:	0f 1f 40 00          	nopl   0x0(%rax)
```

&emsp;&emsp;可能有些时候我们就是为了测试一些和硬件相关的场景的性能，比如lock-free队列的性能之类，如何避免编译器优化来保证性能测试结果正常？首先肯定不能通过编译器选项来实现，我们只是为了减少编译器优化对测试时间计算的影响而不是希望编译器不做任何优化。Google-Benchmark中提供了一种实现思路，即使用内联汇编组织编译指令重排详情见[Google-Benchmark-DoNotOptimize](https://github.com/google/benchmark/blob/e451e50e9b8af453f076dec10bd6890847f1624e/include/benchmark/benchmark.h#L339-L368)

## 2 DoNotOptimize
### 2.1 简介
&emsp;&emsp;```DoNotOptimize```是google-benchmark提供的一个函数，强制编译器不要对制定的变量进行任何优化。需要注意的是DoNotOptimize强制编译器不要对制定的不指定的编译器月变量优化而不是不对表达式进行优化，也就是说你的表达式可以被优化的化依然会被优化只是结果一定写内存。
>The DoNotOptimize(...) function can be used to prevent a value or expression from being optimized away by the compiler. This function is intended to add little to no overhead.

> DoNotOptimize(<expr>) works by forcing the result of <expr> to be stored to memory, which in turn forces the compiler to actually evaluate <expr>. It does not prevent the compiler from optimizing the evaluation of <expr> but it does prevent the expression from being discarded completely.

&emsp;&emsp;比如下面的例子:
```cpp
int testFunc(int a){
    return a;
}

int main(int argc, char **argv){
    int a = 1;
    DoNotOptimize(a);
    auto t = testFunc(a);
    DoNotOptimize(t);
    return 0;
}
```
&emsp;&emsp;不加DoNotOptimize的化testFunc、a和t都会被删除，而编译后的结果如下。可以看到两个变量都是读取的栈内存。
```asm
0000000000001140 <main>:
    1140:	c7 44 24 f8 01 00 00 	movl   $0x1,-0x8(%rsp)
    1147:	00 
    1148:	8b 44 24 f8          	mov    -0x8(%rsp),%eax
    114c:	89 44 24 fc          	mov    %eax,-0x4(%rsp)
    1150:	31 c0                	xor    %eax,%eax
    1152:	c3                   	ret
```

### 2.2 实现原理
&emsp;&emsp;具体实现是使用编译器的asm扩展实现，不同编译器实现原理不同，本文主要聚焦clang/gcc实现，msvc有兴趣的同学自己看。
```cpp
template <class Tp>
inline BENCHMARK_ALWAYS_INLINE void DoNotOptimize(Tp& value) {
#if defined(__clang__)
  asm volatile("" : "+r,m"(value) : : "memory");
#else
  asm volatile("" : "+m,r"(value) : : "memory");
#endif
}
```
&emsp;&emsp;由于clang和gcc的实现类似，只存在细节上的差异，这里只描述gcc的。
- https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html
- https://releases.llvm.org/14.0.0/tools/clang/docs/LanguageExtensions.html

&emsp;&emsp;DoNotOptimize的实现使用了编译内链汇编的扩展,具体语法如下：
```asm
asm asm-qualifiers ( AssemblerTemplate 
                 : OutputOperands 
                 [ : InputOperands
                 [ : Clobbers ] ])
```
- ```volatile```：asm-qualifiers标记为```volatile```表示禁止编译器优化；
- ```""```：一个空语句；
- ```"+r,m"(value)```：约束value读写内存的行为，+表示这是一个读/写操作数，r表示可以使用通用寄存器，m表示可以使用内存；
- ```memory```：是一个内存屏障，告诉编译器以此语句为分界线，上面的语句不能排序到下面，下面的语句不能排序到上面，来保证执行顺序。

&emsp;&emsp;```volatile```的唯一作用就是告诉编译器这个变量不能被优化必须经过寄存器读写到内存，而不是直接操作内存。而```"+r,m"(value)```限制了具体读写的行为。```memory```为了保证执行语义而存在，比如程序：
```cpp
int testFunc(int a){
    return a;
}

int main(int argc, char **argv){
    int a = 1;
    auto t = clock();
    DoNotOptimize(a);
    auto c = testFunc(a);
    DoNotOptimize(t);
    auto t2 = clock();
    return 0;
}
```
&emsp;&emsp;如果没有```memory```的存在，```a```和```t```便来仍然会读写内存不会被优化，但是不同语句的执行顺序不能严格保证的。```auto c=testFunc(a)```和```auto t=clock()```由于两个语句质量没有任何的关联性，在编译器看来对该语句进行排序不会有任何副作用，但是实际上的你的意图是计算耗时排序反而会导致计算不准确。

## 3 总结
&emsp;&emsp;为了精确的测试程序的耗时，尽量在测试区间添加```DonotOptimize```，避免编译器优化而导致测试错误。

## 4 参考文献
- https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html
- https://github.com/google/benchmark/blob/0baacde3618ca617da95375e0af13ce1baadea47/include/benchmark/benchmark.h#L331-L337
- https://github.com/google/benchmark/blob/e451e50e9b8af453f076dec10bd6890847f1624e/include/benchmark/benchmark.h#L339-L368
- https://releases.llvm.org/14.0.0/tools/clang/docs/LanguageExtensions.html
- https://github.com/google/benchmark/issues/242
- https://theunixzoo.co.uk/blog/2021-10-14-preventing-optimisations.html