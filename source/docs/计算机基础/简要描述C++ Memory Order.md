# 简要描述C++ Memory Order
&emsp;&emsp;现代CPU基本都是多核CPU，基本都具备多线程能力。而涉及到多线程一定会涉及到多线程共享资源数据竞争的问题。如果对竞争资源不加以保护或者针对多线程访问的管理就会出现不同线程读取数据不一致或者更加严重的问题。C++标准库提供了互斥锁（```std::mutex```）和原子变量（```std::atomic```）来保证数据在多线程场景下安全的读写。
&emsp;&emsp;互斥锁本身就是让当前线程独占当前代码块来保证线程安全，而原子变量仅仅保护用户定义的原子变量的数据。虽然原子变量只保护原子变量本身，但是我们可以利用原子变量的memory_order来实现相比于互斥锁更加高效的多线程同步。

## 1 缓存一致性
### 1.1 缓存一致性
&emsp;&emsp;现代处理器都是多处理器系统，且每个处理器都采用多级缓存策略。每个处理器都有自己的缓存（如L1缓存、L2缓存等），这些缓存是为了提高处理器访问内存数据的速度。然而，当一个处理器对某个内存位置的数据进行了修改，而另一个处理器的缓存中仍然保存着该位置的旧数据时，就会产生数据不一致的问题。缓存一致性就是为了解决这个问题而出现的协议。
&emsp;&emsp;简单的理解就是CPU0将内存中的a（值为x0）加载到自己的L2缓存中读取并修改为x1不会立即写回（频繁写回会有严重的性能问题），此时CPU1从内存中加载x的值并且修改为x2。在两个CPU写回内存时应该以哪个为准，同时CPU0看到的x值的变化序列为x0,x1,而CPU1看到的是x0,x2，不一致。而缓存一致性协议就是为了保障这个顺序的，以防止多线程程序读取错误的值。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/4d29b2ea8b604287af855bf5570766e4.png)

&emsp;&emsp;CPU缓存一致性（Cache Coherence）是指在多处理器系统中，多个处理器共享内存时，各个处理器的缓存中的数据保持一致的特性。这是为了避免一个处理器在其缓存中读取到的数据与另一个处理器写入的主存数据不一致的情况，从而保证程序执行的正确性。假如没有缓存一致性协议保障数据的读取顺序则可能不同线程对竞争数据的修改在不同时刻观测到的结果不同。
&emsp;&emsp;缓存一致性协议种类比较多，比较常见的为MESI协议，该协议保障不同线程读取相同数据的顺序可见性。虽然通常可以通过互斥锁来解决多县策划功能问题，但是互斥锁本身会强制对应代码块的内存序，可能会降低性能。因此通过MESI协议，C++标准支持了原子变量，让开发人员可以控制读取内存的顺序，来尽可能提升内存读写的性能。

### 1.2 编译器优化
&emsp;&emsp;编译器的优化都是基于运行的程序是单线程的前提，因此编译器和处理器为了优化性能，会对指令进行重排序。这种重排序可能导致多线程程序中的数据竞争和不可预测的行为。不仅仅编译器优化会影响，本身CPU内部也会将进行乱序执行，这也可能影响。

&emsp;&emsp;&emsp;&emsp;memory_order简单的描述就是为了解决多核CPU多线程场景下，单线程内指令执行顺序对于多线程的影响。不同的memory_order规定了不同的内存序，可以让我们根据具体的场景进行选择来优化性能。

## 2 修改顺序
&emsp;&emsp;在具体了解原子变量的memory_order之前先正确的理解下修改顺序。

### 2.1 修改顺序
&emsp;&emsp;在 C++ 内存模型中，每个变量都有一个单独的修改顺序。修改顺序指的是对该变量的所有修改操作（写操作）按某种全序（total order）排列。这意味着对于每个变量，所有线程都会一致地看到修改的顺序。修改顺序是按以下原则定义的：
- 全局一致性: 对于每个变量，所有线程看到的修改顺序是一致的。
- 原子性: 原子操作在修改顺序上是原子的，即对某个变量的原子修改要么完成，要么没有开始，不能处于中间状态。

&emsp;&emsp;不同的内存序对上述语义的支持不同，但是底线是都要保障原子性，也就是说使用原子变量时无论哪种内存序，原子变量一旦被修改，另一个线程读写时其值是可见的。也就是说一个原子变量其修改顺序是对其他线程可见的，如果x从a修改为b再到c，其另一个线程看到的也是该顺序，不可能看到cba。

### 2.2 Sequenced-Before
&emsp;&emsp;Sequenced-Before用来描述在同一个线程中两个操作之间的顺序关系。具体来说，如果操作 A sequenced-before 操作 B，则表示在程序执行过程中，A 操作的所有副作用在 B 操作开始之前完成。该顺序是由C++的evaluation order决定的，具体可参考[Order Of evaluation](https://en.cppreference.com/w/cpp/language/eval_order)，简单的理解就是我看看到的代码顺序。
&emsp;&emsp;Sequenced-Before具备传递性，也就是说如果A Sequenced-Before B，B Sequenced-Before C则A Sequenced-Before C。

### 2.3 Synchronizes-With
&emsp;&emsp;“Synchronizes with”用来描述两个线程之间操作的顺序关系，确保多线程环境下数据的一致性和线程间的正确协调。具体来说，如果一个操作 "synchronizes with" 另一个操作，那么第一个操作的副作用在第二个操作开始之前可见。

&emsp;&emsp;一个线程对原子变量的存储操作可以与另一个线程对同一原子变量的加载操作同步。比如：
```c
std::atomic<int> data(0);
void thread1() {
    data.store(42, std::memory_order_release);
}

void thread2() {
    int x = data.load(std::memory_order_acquire);
    std::cout << x;  // 确保看到 x = 42
}
```
&emsp;&emsp;上面描述的情况是一旦thread1中对data的store操作完成，那么对thread2中的data的读取一定生效，即x一定是42,同时下一行的输出一定也是输出42。如果用relaxed就不会有这种保证，可能出现对指令进行重排序导致data的load操作发生在输出之后。

### 2.4 Happens-Before
&emsp;&emsp;Happens-Before 是一种用于确定多线程程序中事件顺序的关系。这种关系帮助确定哪些操作在时间上先于其他操作，从而确保程序的正确性和线程间的协调。
&emsp;&emsp;单线程场景下Sequenced-Before可以构成Happens-Before。
&emsp;&emsp;多线程场景下为了明确两个操作之间的Happens-Before关系必须加入一些同步操作，也就是引入Synchronizes-With。通过同步操作可以建立跨线程的Happens-Before关系。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/9ff7cc174c06476584984dd9c8111dc7.png)


&emsp;&emsp;比如下面的操作中A Sequenced-Before B，C Sequenced-Before D。而B Sync With C。那么A Happens-Before D。

&emsp;&emsp;但是需要注意的是Happens-Before不代表实际上的执行顺序，CPU和编译器只需要保证在对应的顺序定义上，操作的效果能被后续指令可见即可。比如下面的程序a和b不形成依赖关系，也就是a的求值不依赖b，二者交换顺序不会对程序有额外的副作用，CPU和编译器完全可以针对指令进行排序和合并来提升访存的效率。
```cpp
void func(){
    a++;
    b++;
    std::cout<<a + 2<<std::endl;
}
```

## 3 memory order
### 3.1 memory_order_seq_cst
&emsp;&emsp;```memory_order_seq_cst```代表 "sequentially consistent" (顺序一致性)。使用这种内存顺序的原子操作确保了对内存访问的强顺序保证。
&emsp;&emsp;```memory_order_seq_cst```具备全局顺序一致性 (Sequential Consistency)。所有线程都以相同的顺序观察到所有的原子操作。这意味着如果一个线程观察到一个原子操作的结果，那么其他线程也会以相同的顺序观察到该操作的结果。例如，如果线程 A 执行了一个原子存储操作，然后线程 B 执行了一个原子加载操作，所有其他线程都将以相同的顺序观察到这些操作。
&emsp;&emsp;```memory_order_seq_cst```获取和释放操作 (Acquire and Release Operations)。加载操作会执行获取操作 (acquire)，这意味着加载操作之前的所有操作在当前线程中是可见的。存储操作会执行释放操作 (release)，这意味着当前线程的所有操作在存储操作之后对其他线程是可见的。读-改-写操作 (如 fetch_add) 同时执行获取和释放操作，这确保了线程之间的同步。
&emsp;&emsp;单一全局修改顺序 (Single Total Modification Order)。标记为 memory_order_seq_cst 的所有原子操作将被排列在一个全局的修改顺序中。每个线程都会按照这个全局顺序观察到这些原子操作。这意味着在所有线程中，原子操作的顺序是一致的，不会有线程看到不同的操作顺序。

```cpp
std::atomic<bool> x{false}, y{false};

void thread1() {
    x.store(true, std::memory_order_seq_cst); // (1)
}

void thread2() {
    y.store(true, std::memory_order_seq_cst); // (2)
}

std::atomic<int> z{0};

void read_x_then_y() {
    while (!x.load(std::memory_order_seq_cst)); // (3)
    if (y.load(std::memory_order_seq_cst)) ++z; // (4)
}

void read_y_then_x() {
    while (!y.load(std::memory_order_seq_cst)); // (5)
    if (x.load(std::memory_order_seq_cst)) ++z; // (6)
}

int main() {
    std::thread a(thread1), b(thread2), c(read_x_then_y), d(read_y_then_x);
    a.join(), b.join(), c.join(), d.join();
    assert(z.load() != 0); // (7)
}
```
&emsp;&emsp;上面的程序有2种情况：
1. 先执行thrad1再执行thread2，即修改顺序为x->y；
2. 先执行thrad2再执行thread1，即修改顺序为y->x。

&emsp;&emsp;memory_order_seq_cst能够保证全局唯一的修改顺序，无论哪种顺序至少执行一次z++，即assert不会命中。由于需要向其他核心同步修改顺序，导致这一顺序是原子变量中开销最大的一种。

### 3.2 acquire-release
&emsp;&emsp;acquire 和 release 是一种现比喻人比于releaxd更强的内存顺序，用于确保数据的同步和可见性。
- acquire 内存顺序用于加载操作（例如 load 或 fetch），确保该操作不会在它之后的任何操作之前执行。换句话说，它确保当前线程看到的所有加载操作的结果，包括 acquire 操作本身，都不会发生在之前的存储操作之后。
- release 内存顺序用于存储操作（例如 store 或 exchange），确保该操作不会将任何之前的操作重排序到它之后执行。换句话说，它确保当前线程的所有存储操作都不会发生在之前的加载操作之后。

&emsp;&emsp;简单的说就是A操作在其他线程不能看到发生于releae之后的情况，相对的B操作在其他线程不能看到发生与acquire之前的情况。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/cf358a6682f74cd5a1f7fe4e6fbc4aaf.png)


```cpp
std::atomic<bool> x{false}, y{false};

void thread1() {
    x.store(true, std::memory_order_relaxed); // (1)
    y.store(true, std::memory_order_release); // (2)
}

void thread2() {
    while (!y.load(std::memory_order_acquire)); // (3)
    assert(x.load(std::memory_order_relaxed)); // (4)
}
```
&emsp;&emsp;上面的代码2和3形成sync-with，1 sequenced-before 2,3 sequenced-before 4，即1 三happens-before 4。
&emsp;&emsp;
```cpp
std::atomic<bool> x{false}, y{false};

void thread1() {
    x.store(true, std::memory_order_seq_cst); // (1)
}

void thread2() {
    y.store(true, std::memory_order_seq_cst); // (2)
}

std::atomic<int> z{0};

void read_x_then_y() {
    while (!x.load(std::memory_order_seq_cst)); // (3)
    if (y.load(std::memory_order_seq_cst)) ++z; // (4)
}

void read_y_then_x() {
    while (!y.load(std::memory_order_seq_cst)); // (5)
    if (x.load(std::memory_order_seq_cst)) ++z; // (6)
}

int main() {
    std::thread a(thread1), b(thread2), c(read_x_then_y), d(read_y_then_x);
    a.join(), b.join(), c.join(), d.join();
    assert(z.load() != 0); // (7)
}
```

### 3.3 memory_order_consume
&emsp;&emsp;memory_order_consume旨在比 memory_order_acquire 更具性能优势，特别是对于一些特定的处理器架构。然而，由于目前大多数编译器对 memory_order_consume 的优化支持不足，因此实际上它通常被提升为 memory_order_acquire。
&emsp;&emsp;数据依赖。memory_order_consume 保证的是数据依赖关系。即，如果一个线程执行了一个 memory_order_consume 加载操作，那么所有依赖于这个加载的操作（也就是使用了加载结果的操作）在逻辑上都不能被重排序到这个加载操作之前。更好的性能。memory_order_consume 允许处理器和编译器对不依赖于加载结果的操作进行更多的优化和重排序，从而在某些架构上实现更好的性能。
&emsp;&emsp;consume和acquire的区别是，acquire限制了一整块代码不能重排序，但是阿而consume只是限制了和当前数据有数据依赖的代码不能排序。

### 3.4 memory_order_relaxed
&emsp;&emsp;memory_order_relaxed与其他内存顺序相比，它提供了最弱的同步保证，但在某些情况下可以显著提高性能。memory_order_relaxed不提供顺序保证，memory_order_relaxed 不会对内存访问进行任何顺序限制。这意味着使用 memory_order_relaxed 的原子操作只保证操作本身的原子性，不保证与其他内存操作的顺序关系。适用于无数据依赖的情况，当多个线程之间的操作不需要顺序保证时，可以使用 memory_order_relaxed 来提高性能。适用于统计计数、日志记录等场景，这些场景中操作的顺序不重要。
&emsp;&emsp;但是x86 架构具有强内存一致性模型，这会影响到使用 memory_order_relaxed 时的实际行为和性能特性。虽然如此releaxd模型也会影响编译器优化，可用于在不需要严格顺序保证的情况下优化并发操作的性能。

```cpp
std::atomic<bool> x{false}, y{false};

void thread1() {
    x.store(true, std::memory_order_relaxed); // (1)
    y.store(true, std::memory_order_relaxed); // (2)
}

void thread2() {
    while (!y.load(std::memory_order_relaxed)); // (3)
    assert(x.load()); // (4)
}
```
&emsp;&emsp;由于不保证顺序，x和y的存储可能是y->x，因此assert有可能命中。由于1和2没有内存顺序保证因此不构成sequence-before的关系，3和4同理，即1和4不构成happen-before。

## 4 应用
### 4.1 自旋锁
&emsp;&emsp;代码不能跨过所以进行重排序，因此使用acquire-release。
```cpp
class SpinLock {
public:
    SpinLock() : flag{ATOMIC_FLAG_INIT} {}

    void lock() {
        while (flag.test_and_set(std::memory_order_acquire)) {
            // busy-wait
        }
    }

    void unlock() {
        flag.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag;
};
```
### 4.2 引用计数
&emsp;&emsp;比如libcxx中的——shared_ptr的引用技术：
```cpp
  _LIBCPP_HIDE_FROM_ABI void __add_shared() _NOEXCEPT { __libcpp_atomic_refcount_increment(__shared_owners_); }
  _LIBCPP_HIDE_FROM_ABI bool __release_shared() _NOEXCEPT {
    if (__libcpp_atomic_refcount_decrement(__shared_owners_) == -1) {
      __on_zero_shared();
      return true;
    }
    return false;
  }
#endif
  _LIBCPP_HIDE_FROM_ABI long use_count() const _NOEXCEPT { return __libcpp_relaxed_load(&__shared_owners_) + 1; }
```
## 5 总结
&emsp;&emsp;memory_order保证原子读写顺序的可见性，C++11的atomic提供了多种内存顺序：
- memory_order_relaxed: 最宽松的内存顺序, 只保证操作的原子性和修改顺序 (modification order).
- memory_order_acquire, memory_order_release 和 memory_order_acq_rel: 实现 acquire 操作和 release 操作, 如果 acquire 操作读到了 release 操作写入的值, 或其 release sequence 写入的值, 则构成 synchronizes-with 关系, 进而可以推导出 happens-before 的关系.
- memory_order_consume: 实现 consume 操作, 能实现数据依赖相关的同步关系. 如果 consume 操作读到了 release 操作写入的值, 或其 release sequence 写入的值, 则构成 dependency-ordered before 的关系, 对于有数据依赖的操作可以进而推导出 happens-before 的关系.
- memory_order_seq_cst: 加强版的 acquire-release 模型, 除了可以实现 synchronizes-with 关系, 还保证全局顺序一致.


## 6 参考文献

- [Wiki-Happens-Before](https://en.wikipedia.org/wiki/Happened-before)
- [谈谈 C++ 中的内存顺序 (Memory Order)](https://luyuhuang.tech/2022/06/25/cpp-memory-order.html#%E6%80%BB%E7%BB%93)
- [cppreference-memory-order](https://en.cppreference.com/w/cpp/atomic/memory_order)
- [GCC-AtomicSync](https://gcc.gnu.org/wiki/Atomic/GCCMM/AtomicSync)