# 《Memory Barriers a Hardware View for Software Hackers》读书笔记
&emsp;&emsp;CPU 设计者引入内存屏障（memory barriers）是为了应对在多处理器系统（SMP）中，内存引用重排序可能导致的同步问题。尽管重排序可以提高性能，但在某些情况下（如同步原语），正确的操作依赖于有序的内存引用，因此需要使用内存屏障来强制执行顺序。

&emsp;&emsp;要深入理解这个问题，需要了解 CPU 缓存的工作原理，尤其是如何使缓存有效工作。以下是相关内容的概述：
1. 缓存结构：介绍缓存的基本结构和工作机制。
2. 缓存一致性协议：描述如何通过缓存一致性协议确保各个 CPU 对内存中每个位置的值达成一致。
3. 存储缓冲区与失效队列：概述存储缓冲区和失效队列如何帮助缓存和缓存一致性协议实现高性能。

## 1 缓存结构

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/4d29b2ea8b604287af855bf5570766e4.png)
&emsp;&emsp;现代 CPU 的速度远远超过现代内存系统的速度。例如，一款 2006 年的 CPU 可能每纳秒能够执行十条指令，而从主内存中获取数据则需要数十纳秒。这种速度差异导致了现代 CPU 中存在多兆字节的缓存，这些缓存与 CPU 关联，通常在几个周期内可以访问。

&emsp;&emsp;缓存的基本概念
- 缓存行：数据在 CPU 的缓存和内存之间以固定长度的块（称为“缓存行”）流动，通常大小为 16 到 256 字节。
- 缓存未命中：当 CPU 首次访问某个数据项时，如果该数据不在缓存中，则会发生缓存未命中（cache miss），CPU 需要等待数百个周期才能从内存中获取数据。获取后，该数据会被加载到 CPU 的缓存中，以便后续访问。

&emsp;&emsp;缓存的填充与替换
- 容量未命中：当缓存填满后，新的未命中请求会导致旧数据从缓存中驱逐，以释放空间。
- 关联未命中：由于缓存实现为硬件哈希表，缓存行的替换可能导致现有的行被驱逐，这种情况称为关联未命中（associativity miss）。

&emsp;&emsp;**写操作的处理**
&emsp;&emsp;在进行写操作之前，CPU 必须先使其他 CPU 的缓存中的数据失效，以确保所有 CPU 对数据项的值达成一致。这一过程称为“失效”。如果数据项在 CPU 的缓存中是只读的，写入时会发生“写未命中”（write miss）。因为不同 CPU 使用数据项进行通信（如互斥锁），当其他 CPU 尝试访问被写入的数据项时，可能会出现缓存未命中，这种情况称为“通信未命中”（communication miss）。

&emsp;&emsp;由于 CPU 之间的数据一致性至关重要，必须小心管理缓存中的数据，以避免数据丢失或不同 CPU 之间的值冲突。这些问题通过缓存一致性协议来防止，确保所有 CPU 维护一致的数据视图。

## 2 缓存一致性协议
&emsp;&emsp;缓存一致性协议用于管理缓存行的状态，以防止数据不一致或丢失。这些协议可能相当复杂，但在这里我们只关注四状态的 MESI 缓存一致性协议。

### 2.1 MESI 状态
&emsp;&emsp;MESI 代表“修改”（Modified）、“独占”（Exclusive）、“共享”（Shared）和“无效”（Invalid），每个缓存行在该协议中可以处于这四个状态。使用 MESI 协议的缓存会在每个缓存行上维护一个两位的状态标签，以及该行的物理地址和数据。
- 修改状态（Modified）：该行已被对应 CPU 最近存储修改，且该数据在其他任何 CPU 的缓存中都不存在。这意味着该缓存行是这个 CPU 所“拥有”的，只有它拥有最新的数据副本。此时，缓存必须在重用该行存储其他数据之前，将数据写回内存或交给其他缓存。
- 独占状态（Exclusive）：该状态与修改状态相似，唯一的区别是缓存行尚未被对应 CPU 修改，因此内存中的数据副本是最新的。尽管如此，CPU 仍然可以在不咨询其他 CPU 的情况下对该行进行存储，因此该行依然被视为对应 CPU 所拥有。此时，缓存可以在不写回内存的情况下丢弃该数据。
- 共享状态（Shared）：该行可能在至少一个其他 CPU 的缓存中被复制，因此该 CPU 不能在未咨询其他 CPU 的情况下对该行进行存储。与独占状态一样，内存中的数据副本是最新的，缓存可以丢弃该数据，而无需写回内存或交给其他 CPU。
- 无效状态（Invalid）：该行为空，不持有任何数据。当新数据进入缓存时，优先放入状态为“无效”的缓存行。这种做法是首选，因为替换其他状态的行可能导致在未来引用被替换行时出现昂贵的缓存未命中。
&emsp;&emsp;由于所有 CPU 必须保持对缓存行中数据的统一视图，因此缓存一致性协议提供了消息机制，以协调缓存行在系统中的移动。

### 2.2 MESI 协议消息
&emsp;&emsp;MESI 协议的许多状态转换需要 CPU 之间的通信。如果 CPU 位于单一共享总线上，以下消息就足够了：
- 读取（Read）：包含要读取的缓存行的物理地址。
- 读取响应（Read Response）：包含之前“读取”消息请求的数据，可能由内存或其他缓存提供。如果某个缓存中的数据处于“修改”状态，该缓存必须提供“读取响应”消息。
- 失效（Invalidate）：包含要失效的缓存行的物理地址。所有其他缓存必须从其缓存中移除相应的数据并作出响应。
- 失效确认（Invalidate Acknowledge）：接收到“失效”消息的 CPU 必须在从缓存中移除指定数据后，发送“失效确认”消息。
- 读取失效（Read Invalidate）：包含要读取的缓存行的物理地址，同时指示其他缓存移除该数据。这是“读取”和“失效”的组合消息，要求同时返回“读取响应”和一组“失效确认”消息。
- 写回（Writeback）：包含要写回内存的地址和数据（并可能“窥探”其他 CPU 的缓存）。该消息允许缓存根据需要驱逐“修改”状态的行，以腾出空间存放其他数据。

&emsp;&emsp;共享内存的多处理器系统在底层实际上是一个消息传递计算机。这意味着使用分布式共享内存的 SMP 机器集群在系统架构的两个不同层次上都使用消息传递来实现共享内存。


>1. 如果两个 CPU 同时尝试使同一缓存行失效，会发生什么？
结论：可能会导致数据不一致性，具体处理依赖于缓存一致性协议。
2. 当“失效”消息在大型多处理器中出现时，每个 CPU 都必须给予“失效确认”响应。这不会导致“失效确认”响应的风暴完全饱和系统总线吗？
结论：是的，这种情况可能会导致总线拥塞，因此在设计中需要优化以减少这种情况的发生。
3. 如果 SMP 机器已经在使用消息传递，为什么还要使用 SMP？
结论：SMP 提供了更简单的编程模型和更高效的共享内存访问，相比于纯消息传递的系统，SMP 能够更好地利用缓存和处理器之间的直接数据共享。

### 2.3 MESI 状态图

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/mesi.png)

&emsp;&emsp;MESI 协议中，缓存行的状态会随着协议消息的发送和接收而变化。下面是每个状态转移的简要说明：
- 转移 (a)：缓存行被写回内存，但 CPU 保留该行在缓存中的副本，并且仍然可以修改。此转移需要一个“写回”消息。
- 转移 (b)：CPU 对已独占访问的缓存行进行写操作。这一转移不需要发送或接收任何消息。
- 转移 (c)：CPU 收到针对已修改缓存行的“读取失效”消息。CPU 必须使其本地副本失效，并同时发送“读取响应”和“失效确认”消息。
- 转移 (d)：CPU 对不在其缓存中的数据项执行原子读取-修改-写操作。它发送“读取失效”消息，并通过“读取响应”接收数据，必须在收到所有“失效确认”响应后才能完成转移。
- 转移 (e)：CPU 对之前在缓存中为只读的数据项执行原子读取-修改-写操作，必须发送“失效”消息，并等待所有“失效确认”响应。
- 转移 (f)：其他 CPU 读取该缓存行，数据由该 CPU 的缓存提供，CPU 发送“读取响应”消息。
- 转移 (g)：其他 CPU 读取缓存行的数据项，数据可能来自该 CPU 的缓存或内存，该 CPU 保留只读副本，并发送“读取响应”消息。
- 转移 (h)：CPU 意识到将需要写入缓存行中的数据，因此发送“失效”消息，直到收到所有“失效确认”响应后才能完成转移。
- 转移 (i)：其他 CPU 对仅在该 CPU 的缓存中持有的数据项执行原子读取-修改-写操作，CPU 接收到“读取失效”消息后，将其从缓存中失效，并发送“读取响应”和“失效确认”消息。
- 转移 (j)：CPU 对不在其缓存中的数据项执行存储操作，发送“读取失效”消息，直到收到“读取响应”和所有“失效确认”消息后才能完成转移。
- 转移 (k)：CPU 加载不在其缓存中的数据项，发送“读取”消息，等待相应的“读取响应”后完成转移。
- 转移 (l)：其他 CPU 对此缓存行中的数据项进行存储操作，但由于其他 CPU 持有该行，当前 CPU 只能保持只读状态。接收到“失效”消息后，CPU 发送“失效确认”消息。

>硬件如何处理上述延迟的状态转移？
结论：硬件通过使用缓冲机制（如存储缓冲区和失效队列）来处理延迟的状态转移。它允许 CPU 在等待回复消息时继续执行其他操作，从而提高性能。此外，硬件还可以在状态转移过程中保持状态的一致性，确保所有相关的缓存行在完成操作时处于正确的状态。

### 2.4 MESI 协议示例
&emsp;&emsp;在一个四 CPU 系统中，我们将从一个缓存行的数据的角度，观察其如何通过各个单行直接映射的缓存。以下是该数据流的描述：
- 初始状态：所有 CPU 的缓存行处于“无效”（Invalid）状态，内存中的数据有效。
- 操作序列：
  - **1**: CPU 0 加载地址 0 的数据，状态变为“共享”（Shared），内存中的数据仍有效。
  - **2**: CPU 3 也加载地址 0 的数据，状态在 CPU 0 和 CPU 3 的缓存中均为“共享”，内存中的数据仍有效。
  - **3**: CPU 0 加载地址 8 的数据，迫使地址 0 的数据通过写回（Writeback）被驱逐。
  - **4**: CPU 2 从地址 0 加载数据，发送“读取失效”消息以获得独占副本，失效 CPU 3 的缓存中的数据（内存中的副本仍有效）。
  - **5**: CPU 2 进行存储操作，状态变为“修改”（Modified），此时内存中的副本失效。
  - **6**: CPU 1 进行原子增量操作，使用“读取失效”从 CPU 2 的缓存中获取数据并使其失效，CPU 1 的缓存状态变为“修改”，内存中的副本仍失效。
  - **7**: CPU 1 读取地址 8 的缓存行，使用“写回”消息将地址 0 的数据写回内存。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/cachecoherence.png)

## 3 存储操作导致不必要的停滞
### 3.1  Store Buffers
&emsp;&emsp;尽管所示的缓存结构在多个重复读取和写入操作中表现良好，但对于某个缓存行的首次写入，性能却相对较差。以下是相关内容的概述：
1. 写操作的性能问题
&emsp;&emsp;在 CPU 0 对存储在 CPU 1 缓存中的缓存行进行写操作时，CPU 0 必须等待该缓存行到达后才能进行写入。这导致 CPU 0 出现较长时间的停滞。尽管 CPU 0 实际上会无条件覆盖 CPU 1 缓存中的任何数据，但仍然需要等待，这种设计显得不够高效。
2. 存储缓冲区的引入
&emsp;&emsp;为了防止不必要的写入停滞，可以在每个 CPU 和其缓存之间添加“存储缓冲区”。添加存储缓冲区后，CPU 0 可以将写入记录在其存储缓冲区中，并继续执行其他操作，而无需等待缓存行的到达。

&emsp;&emsp;操作流程：CPU 0 将写入操作记录在存储缓冲区中。一旦 CPU 1 的缓存行传输到 CPU 0，数据将从存储缓冲区移动到缓存行。
&emsp;&emsp;需要解决的复杂问题:尽管存储缓冲区能够提高写入性能，但引入它们会带来一些复杂性。存储缓冲区的有效管理和同步机制是确保数据一致性和避免潜在错误的关键。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/4d29b2ea8b604287af855bf5570766e4.png)

### 3.2 存储转发
&emsp;&emsp;存储转发是一种解决自一致性违例的机制。考虑以下代码，其中变量“a”和“b”最初均为零，且包含变量“a”的缓存行最初由 CPU 1 拥有，而包含“b”的缓存行则由 CPU 0 拥有：
```
1 a = 1;
2 b = a + 1;
3 assert(b == 2);
```

&emsp;&emsp;在正常情况下，断言不会失败。然而，如果使用简单架构（如图 5 所示），则可能会出现意外的结果。以下是可能的事件序列：
- CPU 0 开始执行 a = 1。
- CPU 0 查找“a”在缓存中，发现缺失。
- CPU 0 发送“读取失效”消息以获取包含“a”的缓存行的独占权。
- CPU 0 在其存储缓冲区中记录对“a”的写入。
- CPU 1 接收到“读取失效”消息，响应并传输缓存行，同时从其缓存中移除该缓存行。
- CPU 0 开始执行 b = a + 1。
- CPU 0 收到来自 CPU 1 的缓存行，此时“a”的值仍为零。
- CPU 0 从缓存中加载“a”，发现其值为零。
- CPU 0 将其存储队列中的条目应用于新到达的缓存行，将缓存中“a”的值设置为一。
- CPU 0 将零的值加一，并存储到包含“b”的缓存行中（假设该缓存行已由 CPU 0 拥有）。
- CPU 0 执行 assert(b == 2)，该断言失败。
问题分析
&emsp;&emsp;问题在于存在两个“a”的副本，一个在缓存中，另一个在存储缓冲区中。这种情况违背了一个重要的保证，即每个 CPU 总是会看到自己的操作仿佛按照程序顺序发生。这种保证对软件开发者来说极具反直觉，因此硬件设计者引入了“存储转发”机制。
&emsp;&emsp;**存储转发机制**:存储转发允许每个 CPU 在执行加载时同时参考其存储缓冲区和缓存。具体来说：每个 CPU 的存储直接转发到其后续的加载操作，而无需通过缓存。通过引入存储转发，事件序列中的第 8 步会找到存储缓冲区中“a”的正确值 1，从而确保最终的“b”的值为 2，如预期的那样。这种机制有效解决了自一致性问题，确保了 CPU 在操作顺序上的一致性。

### 3.3 存储缓冲区与内存屏障
&emsp;&emsp;在多处理器系统中，存储缓冲区和内存屏障的引入是为了处理全局内存排序的违例。让我们考虑以下代码序列，其中变量“a”和“b”最初均为零：
```cpp
void foo(void) {
    a = 1;
    b = 1;
}

void bar(void) {
    while (b == 0) continue;
    assert(a == 1);
}
```

&emsp;&emsp;假设 CPU 0 执行 foo()，而 CPU 1 执行 bar()，并且缓存行“a”只存在于 CPU 1 的缓存中，而“b”的缓存行由 CPU 0 拥有。可能的操作序列如下：

1. CPU 0 执行 a = 1。由于 CPU 0 的缓存中没有该缓存行，它将新值放入存储缓冲区并发送“读取失效”消息。
2. CPU 1 执行 while(b == 0) continue，但它的缓存中没有包含“b”的缓存行，因此发送“读取”消息。
3. CPU 0 执行 b = 1，并将新值存储在缓存行中。
4. CPU 0 接收到“读取”消息，并将包含更新后“b”值的缓存行传输给 CPU 1。
5. CPU 1 接收到缓存行并将其安装到自己的缓存中。
6. CPU 1 继续执行 while(b == 0) continue，发现“b”的值为 1，继续到下一行。
7. CPU 1 执行 assert(a == 1)，但由于其使用的是“a”的旧值，这个断言失败。
8. CPU 1 收到“读取失效”消息，并将包含“a”的缓存行传输给 CPU 0，同时使其自己缓存中的缓存行失效。
9. CPU 0 接收到包含“a”的缓存行并应用缓冲的存储，导致失败的断言。

&emsp;&emsp;在这个例子中，CPU 1 看到的是旧的“a”的值，这违反了全局内存排序的原则。硬件设计者无法直接解决这个问题，因为 CPU 并不知道变量之间的相关性。因此，硬件设计者提供了内存屏障指令，以允许软件告知 CPU 这些关系。

**引入内存屏障**
&emsp;&emsp;通过在 foo() 中引入内存屏障 smp_mb()，可以确保 CPU 在执行后续存储之前清空存储缓冲区。更新后的代码如下：
```cpp
void foo(void) {
    a = 1;
    smp_mb();
    b = 1;
}

void bar(void) {
    while (b == 0) continue;
    assert(a == 1);
}
```

&emsp;&emsp;内存屏障的作用是：
- CPU 会在执行后续存储之前清空其存储缓冲区。
- CPU 可以暂停，直到存储缓冲区为空，或者使用存储缓冲区来保持后续存储，直到所有先前的条目都已应用。

&emsp;&emsp;更新后的操作序列，引入内存屏障后的操作序列如下：
1. CPU 0 执行 a = 1，缓存行不在 CPU 0 的缓存中，因此将新值放入存储缓冲区并发送“读取失效”消息。
2. CPU 1 执行 while(b == 0) continue，发送“读取”消息。
3. CPU 0 执行 smp_mb()，标记所有当前存储缓冲区条目（即 a = 1）。
4. CPU 0 执行 b = 1，由于存在标记条目，新的“b”的值被放入未标记的存储缓冲区中。
5. CPU 0 接收“读取”消息，传输原始“b”值的缓存行给 CPU 1。
6. CPU 1 安装缓存行并继续执行 while(b == 0) continue。此时“b”仍为 0，因此继续循环。
7. CPU 1 接收到“读取失效”消息，传输“a”的缓存行给 CPU 0。
8. CPU 0 接收到“a”的缓存行并应用缓冲的存储。
9. CPU 0 也可以存储新的“b”值，但由于缓存行现在处于“共享”状态，必须发送“失效”消息。
10. CPU 1 收到“失效”消息并失效“b”的缓存行。
11. CPU 1 再次执行 while(b == 0) continue，并发送“读取”消息。
12. CPU 0 将“b”缓存行置于“独占”状态并存储新值。
13. CPU 0 接收到“读取”消息并传输原始“b”值的缓存行给 CPU 1。
14. CPU 1 安装缓存行并继续执行 assert(a == 1)，此时“a”的值已更新，因此断言通过。

&emsp;&emsp;引入存储缓冲区和内存屏障可以有效解决全局内存排序的问题。尽管这一过程涉及大量的管理和步骤，但它确保了程序的正确性和一致性。

## 4 Invalidate Queue
### 4.1 存储序列导致不必要的停滞
&emsp;&emsp;在多处理器系统中，存储缓冲区的大小通常较小，这意味着当 CPU 执行一系列存储操作时，如果这些操作都导致缓存缺失，存储缓冲区可能会迅速填满。一旦存储缓冲区满了，CPU 就必须等待失效确认消息（invalidate acknowledge messages）完成，以便清空存储缓冲区，才能继续执行。这种情况在内存屏障（memory barriers）之后尤其明显，因为所有后续存储指令都必须等待失效确认消息，无论这些存储是否导致缓存缺失。

&emsp;&emsp;为了改善这一情况，可以采取一些措施来加快失效确认消息的到达速度。一个有效的方法是使用每个 CPU 的失效消息队列（invalidate queues）。这种方法可以减少等待时间，提高系统的整体性能。

&emsp;&emsp;失效消息队列的工作原理：
- 每个 CPU 分配独立的失效消息队列：每个 CPU 可以独立处理其失效消息，从而减少了消息在总线上传递的延迟。
- 异步处理：失效消息可以异步处理，允许 CPU 在发送失效消息的同时继续执行其他操作，而不必等待确认消息。
- 减少总线拥塞：通过将失效消息分散到各个 CPU 的队列中，可以减少总线的拥塞，从而提高数据传输的效率。

&emsp;通过优化失效确认消息的处理，尤其是引入每个 CPU 的失效消息队列，可以有效减少因存储缓冲区满而导致的停滞。这种改进不仅提升了 CPU 的执行效率，还增强了系统对并发存储操作的处理能力，从而实现更高的性能。

### 4.2 Invalidate Queue
&emsp;&emsp;失效队列（invalidate queue）可以通过以下方式改善存储操作的性能：
- 快速确认：失效队列可以在失效消息放入队列后立即确认，而不必等待相应的缓存行实际被失效。这减少了因等待确认而导致的停滞时间。
- 延迟传输：CPU 在准备传输失效消息时，需检查失效队列。如果对应的缓存行在队列中，CPU 必须等待该队列中的条目被处理后，才能发送失效消息。这种机制确保了消息的有序处理。
- 承诺处理：将条目放入失效队列意味着 CPU 承诺在发送任何 MESI 协议消息之前处理该条目。这种承诺通常不会给 CPU 带来太大负担，前提是相关数据结构的争用不严重。

&emsp;&emsp;尽管失效消息的缓冲提供了性能提升的机会，但它也可能引入内存乱序（memory misordering）的问题。由于失效消息可以在队列中被缓冲，可能导致不同 CPU 看到不一致的数据状态或操作顺序。这种情况在并发访问和修改共享数据时尤为重要。

### 4.3 内存屏障
&emsp;&emsp;失效消息队列通过允许 CPU 在不阻塞的情况下处理失效消息，提高了存储操作的效率。然而，这种缓冲机制也可能导致内存乱序，影响数据一致性和程序的正确性。需要谨慎设计和实现，以确保系统在性能和一致性之间取得平衡。
&emsp;&emsp;在多处理器系统中，失效队列和内存屏障的设计旨在解决数据一致性和内存顺序的问题。考虑以下代码片段，其中变量“a”和“b”最初均为零，且“a”处于只读（MESI “共享”状态），而“b”由 CPU 0 拥有（MESI “独占”或“修改”状态）:
```cpp
void foo(void) {
    a = 1;
    smp_mb();
    b = 1;
}

void bar(void) {
    while (b == 0) continue;
    assert(a == 1);
}
```
&emsp;&emsp;以下是可能的操作序列：
1. CPU 0 执行 a = 1。由于缓存行在 CPU 0 的缓存中是只读的，CPU 0 将新值放入存储缓冲区，并发送失效消息以从 CPU 1 的缓存中刷新对应的缓存行。
2. CPU 1 执行 while(b == 0) continue，但它的缓存中没有包含“b”的缓存行，因此发送“读取”消息。
3. CPU 0 执行 b = 1，并将新值存储在缓存行中。
4. CPU 0 接收“读取”消息，并将更新后的“b”值的缓存行传输给 CPU 1，同时将该行标记为“共享”。
5. CPU 1 收到针对“a”的失效消息，将其放入失效队列，并向 CPU 0 发送失效确认消息。此时，旧值的“a”仍然保留在 CPU 1 的缓存中。
6. CPU 1 收到包含“b”的缓存行并安装到自己的缓存中。
7. CPU 1 继续执行循环，发现“b”的值为 1，进入下一行。
8. CPU 1 执行 assert(a == 1)，由于“a”的旧值仍在 CPU 1 的缓存中，因此断言失败。
9. CPU 1 处理排队的失效消息，将“a”的缓存行失效。但此时已为时已晚。
10. CPU 0 接收到来自 CPU 1 的失效确认消息，并应用缓冲的存储，导致 CPU 1 的断言失败。

&emsp;&emsp;为了防止上述情况，可以在 bar() 函数中添加内存屏障：
```cpp
void bar(void) {
    while (b == 0) continue;
    smp_mb();
    assert(a == 1);
}
```
&emsp;&emsp;这种改变后，操作序列如下：
1. CPU 0 执行 a = 1。操作与之前一样。
2. CPU 1 执行 while(b == 0) continue，依旧发送“读取”消息。
3. CPU 0 执行 b = 1，并将新值存储在缓存行中。
4. CPU 0 接收到“读取”消息，并将更新后的“b”值的缓存行传输给 CPU 1。
5. CPU 1 接收到失效消息并将其放入失效队列。
6. CPU 1 收到缓存行并安装。
7. CPU 1 继续执行循环，发现“b”为 1，进入下一行。
8. CPU 1 执行 smp_mb()，标记失效队列中的条目。
9. CPU 1 执行 assert(a == 1)，由于失效队列中存在对应的标记条目，CPU 1 必须等待该条目处理完再进行加载。
10. CPU 1 处理失效消息，将“a”缓存行失效。
11. CPU 1 现在可以加载“a”，但由于这导致缓存缺失，它必须发送“读取”消息。
12. CPU 0 接收到失效确认消息并应用缓冲的存储，将“a”的 MESI 状态更改为“修改”。
13. CPU 0 接收到针对“a”的“读取”消息，并将对应的缓存行状态更改为“共享”，然后将缓存行传输给 CPU 1。
14. CPU 1 收到包含“a”的缓存行并执行加载。此时加载返回“a”的新值，因此断言通过。

&emsp;&emsp;通过引入失效队列和内存屏障，系统能够有效地处理数据一致性和内存顺序问题。在这个过程中，CPU 之间的 MESI 消息传递确保了最终的正确性，尽管这一过程涉及复杂的操作和管理。这种设计增强了多处理器系统在并发环境下的稳定性和可靠性。

## 5 读和写内存屏障
&emsp;&emsp;在多处理器系统中，内存屏障用于确保操作的顺序性，以满足并发执行时的数据一致性要求。在先前的内容中，内存屏障被用来标记存储缓冲区和失效队列中的条目。然而，在实际代码中，foo() 和 bar() 函数并没有必要与失效队列或存储队列进行交互。

&emsp;&emsp;为了应对这一情况，许多 CPU 架构提供了较弱的内存屏障指令，这些指令仅处理存储缓冲区或失效队列中的一个。大致而言：
- 读内存屏障（Read Memory Barrier）：仅标记失效队列。
- 写内存屏障（Write Memory Barrier）：仅标记存储缓冲区。
- 全功能内存屏障（Full Memory Barrier）：同时标记存储缓冲区和失效队列。

&emsp;&emsp;读和写内存屏障的效果
- 读内存屏障：确保在该屏障之前的所有加载操作在该屏障之后的加载操作之前完成。这意味着，所有在读内存屏障之前的加载将被视为在后续加载之前完成。
- 写内存屏障：确保在该屏障之前的所有存储操作在该屏障之后的存储操作之前完成。这意味着，所有在写内存屏障之前的存储将被视为在后续存储之前完成。
- 全功能内存屏障：同时确保加载和存储的顺序性，确保所有操作在执行该屏障之前完成。

```cpp
void foo(void) {
    a = 1;
    smp_wmb();  // 写内存屏障
    b = 1;
}

void bar(void) {
    while (b == 0) continue;
    smp_rmb();  // 读内存屏障
    assert(a == 1);
}
```

&emsp;&emsp;在 foo() 函数中，写内存屏障 smp_wmb() 确保 b = 1 之前的所有存储操作（如 a = 1）在执行该屏障之后被视为完成。
&emsp;在 bar() 函数中，读内存屏障 smp_rmb() 确保在该屏障之前的所有加载（如 b 的值）在执行 assert(a == 1) 之前完成。
结论
&emsp;&emsp;通过使用读和写内存屏障，开发者可以更灵活地控制内存操作的顺序，从而提高多处理器环境中的数据一致性。理解这三种内存屏障的基本概念有助于深入理解并发编程中的内存管理策略。

## 6 内存屏障序列示例
&emsp;&emsp;本节展示了一些看似有效但实际上存在潜在问题的内存屏障使用示例。虽然这些示例在大多数情况下可能有效，但为了确保代码在所有 CPU 上可靠工作，应该避免这些用法。首先，我们需要关注一种对顺序友好性有敌意的架构。

### 6.1 对顺序友好性有敌意的架构
&emsp;&emsp;为了探讨这一问题，我们设想一个虚构的、极端友好性敌对的计算机架构，其硬件需遵循以下顺序约束：

1. 每个 CPU 始终认为自己的内存访问是按程序顺序发生的。
2. 仅当两个操作引用不同位置时，CPU 才会重排与存储相关的操作。
3. 在某 CPU 执行读内存屏障（smp_rmb()）后，该 CPU 之前的所有加载操作将被认为在任何后续加载操作之前完成。
4. 在某 CPU 执行写内存屏障（smp_wmb()）后，该 CPU 之前的所有存储操作将被认为在任何后续存储操作之前完成。
5. 在某 CPU 执行全功能内存屏障（smp_mb()）后，该 CPU 之前的所有访问（加载和存储）将被认为在任何后续访问之前完成。

**例子1**
&emsp;&emsp;假设 CPU 0 最近经历了许多缓存缺失，其消息队列已满，而 CPU 1 在缓存中独占运行，其消息队列为空。以下是可能的代码片段：
```cpp
// CPU 0
a = 1;
smp_wmb(); 
b = 1;

//CPU 1
while (b == 0);
c = 1

// CPU 2
z = c;
smp_rmb(); 
x = a; 
assert(z == 0 || x == 1);
```

&emsp;&emsp;在这种情况下，CPU 2 可能会在看到 CPU 0 的 a 赋值之前先看到 CPU 1 的 c 赋值，从而导致断言失败，尽管使用了内存屏障。

**例子2**
&emsp;&emsp;在以下代码片段中，两个 CPU 的操作可能因缓存和消息队列的状态而导致不一致：
```cpp
// CPU 0
a = 1; 

//CPU 1
while (a == 0);
smp_mb(); 
b = 1;

// CPU 2
y=b;
smp_rmb();
x=a;
assert(y==0||x==1);
```

&emsp;&emsp;如上所示，CPU 2 可能在看到 CPU 0 的 a 赋值之前看到 CPU 1 的 b 赋值，从而导致不一致。

**例子3**
&emsp;&emsp;在这个代码片段中，所有操作都通过内存屏障进行了适当的同步：
```cpp
// CPU 0
a = 1; 
smp_wmb(); 
b = 1; 

while(c==0);
while(d==0);
smp_mb();
e=1;

// CPU 1
while (b == 0); 
smp_mb(); 
c = 1; 

// CPU 2
while (b == 0); 
smp_mb(); 
d = 1; 
assert(e == 0 || a == 1);
```
&emsp;&emsp;在此示例中，CPU 1 和 CPU 2 在执行其内存屏障之前必须首先看到 CPU 0 的 b 赋值。由于内存屏障的使用，CPU 2 的断言不会失败。
&emsp;&emsp;通过这些示例，我们可以看到内存屏障的使用虽然在某些情况下可能看起来有效，但由于系统架构中潜在的复杂性和顺序约束，可能导致程序的不确定性。为了编写跨平台和跨 CPU 可靠的代码，开发者必须谨慎使用内存屏障。

## 7 特定 CPU 的内存屏障指令
### 7.1 简介
&emsp;&emsp;每个 CPU 都有其特有的内存屏障指令，这使得跨平台的代码移植性面临挑战。许多软件环境（包括 pthreads 和 Java）限制程序员直接使用内存屏障，转而使用包含必要内存屏障的互斥原语。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20250124170140.png)
&emsp;&emsp;表 5 列出了不同 CPU 允许的加载和存储重排序的组合。具体而言，前四列表示 CPU 允许的四种加载和存储重排序的组合，接下来的两列表示 CPU 是否允许与原子指令的重排序。以下是一些关键点：
&emsp;&emsp;在六个 CPU 中，有五种不同的加载-存储重排序组合和四种原子指令重排序的三种可能。
&emsp;&emsp;Alpha CPU 需要对读取的依赖关系进行内存屏障，这意味着 Alpha 可以在获取数据指针之前先获取指向的数据。
&emsp;&emsp;最后一列指示 CPU 是否具有不一致的指令缓存和流水线，这种 CPU 需要执行特殊指令来处理自修改代码。

**Linux 内存屏障原语**
&emsp;&emsp;在 Linux 内核中，提供了一组经过仔细选择的内存屏障原语，包括：
- smp_mb()：内存屏障，确保在其之前的所有加载和存储在之后的加载和存储之前完成。
- smp_rmb()：读内存屏障，仅确保加载操作的顺序。
- smp_wmb()：写内存屏障，仅确保存储操作的顺序。
- smp_read_barrier_depends()：强制后续操作依赖于先前操作的顺序。在所有平台上，此原语在 Alpha 上才有效。
- mmiowb()：强制 MMIO 写操作的顺序，主要用于被全局自旋锁保护的情况。
这些原语确保编译器不会对内存优化进行重排，从而避免潜在的错误。对于 SMP 内核，这些原语生成代码，但在 UP 内核中也会生成相应的内存屏障。

&emsp;&emsp;大多数内核程序员不必担心每个 CPU 的内存屏障细节，只需使用 Linux 提供的接口。如果在特定 CPU 的架构特定代码中工作，则需要更深入的了解。

&emsp;&emsp;Linux 的所有锁原语（自旋锁、读写锁、信号量、RCU 等）都包含必要的内存屏障原语，因此在使用这些原语的代码中，开发者无需担心内存顺序的复杂性。

&emsp;&emsp;深入了解每个 CPU 的内存一致性模型在调试和编写架构特定代码时非常有帮助。虽然阅读特定 CPU 的文档是不可替代的，但对内存一致性模型的概述可以帮助开发者更好地理解和应用内存屏障。

### 7.2 Alpha
&emsp;&emsp;尽管 Alpha CPU 的生命周期已结束，但它因其最弱的内存排序模型而变得非常有趣。Alpha 具有极其激进的内存操作重排序能力，因此，其内存屏障原语在 Linux 内核中具有重要意义，了解 Alpha 对于 Linux 内核开发者来说尤为重要。

在 Alpha 架构中，可能会出现以下问题。在图 9 中，smp_wmb() 确保第 6-8 行的元素初始化在第 10 行将元素添加到列表之前执行，但在 Alpha 上，这一保证并不可靠。代码的第 20 行可能会看到初始化前的旧值。
```cpp
struct el *insert(long key, long data) {
    struct el *p;
    p = kmalloc(sizeof(*p), GPF_ATOMIC);
    spin_lock(&mutex);
    p->next = head.next;        //line6
    p->key = key;               //line7
    p->data = data;             //line8
    smp_wmb(); // 写内存屏障     //line9
    head.next = p;              //line10
    spin_unlock(&mutex);
}

struct el *search(long key) {
    struct el *p;
    p = head.next;
    while (p != &head) {
        if (p->key == key) { // 这里可能出现问题 //line20
            return (p);
        }
        p = p->next;
    }
    return (NULL);
}
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/alpha.png)

在图 10 中，假设头指针 head 在缓存bank 0 中处理，而新元素在缓存bank 1 中处理。smp_wmb() 确保第 6-8 行的缓存失效在第 10 行之前到达互连，但不保证新值到达读取 CPU 的顺序。可能的情况是，读取 CPU 的缓存bank 1 正忙，而缓存bank 0 空闲，导致读取 CPU 先获取指向新元素的指针，但看到的是旧的缓存值。
&emsp;&emsp;为了确保读取操作的安全性，可以在指针取值和解引用之间插入 smp_rmb()。然而，这在遵循数据依赖性的系统中（如 i386、IA64、PPC 和 SPARC）会引入不必要的开销。

&emsp;&emsp;为了解决这个问题，Linux 2.6 内核引入了 smp_read_barrier_depends() 原语，避免了这些系统的额外开销。也可以实现一个软件屏障来替代 smp_wmb()，强制所有读取 CPU 按顺序看到写入 CPU 的写入。然而，这种方法在 Linux 社区被认为在极其弱排序的 CPU（如 Alpha）上会引入过高的开销。此软件屏障可以通过向所有其他 CPU 发送处理器间中断（IPI）来实现。收到 IPI 时，CPU 将执行内存屏障指令，从而实现内存屏障的清理。
```c
smp_read_barrier_depends(); // 确保依赖顺序
```

&emsp;&emsp;Linux 的内存屏障原语命名来自于 Alpha 指令：
- smp_mb()：对应于 Alpha 的 mb
- smp_rmb()：对应于 Alpha 的 rmb
- smp_wmb()：对应于 Alpha 的 wmb

### 7.3 IA64
&emsp;&emsp;IA64 提供了一种弱一致性模型，这意味着在没有显式内存屏障指令的情况下，IA64 有权任意重排序内存引用。IA64 有一个名为 mf 的内存栅栏指令，同时也提供了对加载、存储及部分原子指令的“半内存栅栏”修饰符。
&emsp;&emsp;半内存栅栏指令
- acq 修饰符：防止后续的内存引用指令在 acq 之前被重排序，但允许之前的内存引用指令在 acq 之后被重排序。
- rel 修饰符：防止之前的内存引用指令在 rel 之后被重排序，但允许后续的内存引用指令在 rel 之前被重排序。

&emsp;&emsp;这些半内存栅栏在关键区段中非常有用，因为它们允许将操作推入关键区段，但如果允许它们泄漏到关键区段之外，则可能导致致命错误。

&emsp;&emsp;作为仅有的具有此属性的 CPU 之一，IA64 定义了与锁获取和释放相关的 Linux 内存排序语义。IA64 的 mf 指令用于 Linux 内核中的 smp_rmb()、smp_mb() 和 smp_wmb() 原语。尽管有相关传言，mf 确实代表“内存栅栏”。

&emsp;&emsp;IA64 提供了“释放”操作的全局总顺序，包括 mf 指令。这种顺序概念提供了传递性，即如果某段代码看到某个访问已发生，那么任何后续代码段也将看到该访问已发生。这一前提是所有相关代码段正确使用内存屏障。

### 7.4 PA-RISC 的内存屏障指令
&emsp;&emsp;PA-RISC 架构虽然允许完全重排序加载和存储操作，但实际的 CPU 执行时是完全有序的。这意味着在 PA-RISC 上，Linux 内核的内存排序原语不会生成任何实际的代码。然而，它们会利用 GCC 的内存属性来禁用编译器的优化，从而防止代码在内存屏障跨越时被重排序。

### 7.5 POWER
&emsp;&emsp;POWER 和 PowerPC CPU 系列提供了多种内存屏障指令，以支持不同的内存一致性模型。以下是一些主要的内存屏障指令及其功能：
&emsp;&emsp;内存屏障指令
1. sync：
   1. 确保所有先前的操作在任何后续操作开始之前都已完成。
   2. 该指令开销较高。
2. lwsync (轻量级同步)：
   1. 对后续的加载和存储进行排序，并确保所有存储操作的顺序。
   2. 不会对后续加载与存储之间的存储操作进行排序。
   3. 其排序行为与 zSeries 和 SPARC TSO 一致。
3. eieio (强制 I/O 的顺序执行)：
   1. 确保所有先前可缓存存储在所有后续存储之前完成。
   2. 可缓存内存的存储与非可缓存内存的存储分别排序，意味着 eieio 不会强制 MMIO 存储在自旋锁释放之前完成。
4. isync：
   1. 确保所有先前指令在任何后续指令开始执行之前完成。
   2. 这意味着先前的指令必须足够进展，以保证它们可能产生的任何陷阱要么发生，要么不发生。

&emsp;&emsp;虽然没有任何 POWER 指令完全对应于 Linux 的 wmb() 原语（该原语要求所有存储操作都被排序，而不需要 sync 指令的其他高开销操作），在实际使用中，ppc64 版本的 wmb() 和 mb() 被定义为开销较大的 sync 指令。

- smp_wmb() 指令被定义为较轻的 eieio 指令，而不是用于 MMIO 的 smp_mb() 指令（因为即使在 UP 和 SMP 内核中，驱动程序也必须仔细排序 MMIO）。
- smp_mb() 被定义为 sync 指令。
- smp_rmb() 和 rmb() 被定义为较轻的 lwsync 指令。

&emsp;&emsp;POWER 架构具有“累积性”特性，适当使用时，可以实现访问的传递性。也就是说，任何看到早期代码片段结果的代码，也将看到该早期代码片段所看到的访问。

&emsp;&emsp;POWER 架构的许多成员具有不一致的指令缓存，内存的存储可能不会反映在指令缓存中。尽管近年来编写自修改代码的人不多，但 JIT 编译器和编译器经常会遇到这种情况。为了解决这个问题，可以使用 icbi 指令（指令缓存块失效）来使指定缓存行失效。

### 7.6 SPARC 架构的内存模型：RMO、PSO 和 TSO

&emsp;&emsp;在 SPARC 架构上，Solaris 和 Linux 的实现有所不同：

- **TSO（Total Store Order）**：在 32 位的 “sparc” 架构下，Linux 使用 TSO。
- **RMO（Relaxed Memory Order）**：在 64 位的 “sparc64” 架构下，Linux 运行于 RMO 模式。
- **PSO（Partial Store Order）**：SPARC 还提供了一种中间模式 PSO。

RMO 模式下运行的程序也可以在 PSO 或 TSO 中运行，而在 PSO 中运行的程序同样可以在 TSO 中运行。将共享内存并行程序从 RMO 转换到 TSO 或 PSO 可能需要仔细插入内存屏障，但如前所述，标准同步原语的使用通常不需要担心内存屏障。

**内存屏障指令**

&emsp;&emsp;SPARC 架构提供了灵活的内存屏障指令，允许对内存操作进行细粒度的控制：

1. **StoreStore**：确保所有先前的存储操作在任何后续存储操作之前完成（由 `smp_wmb()` 使用）。
2. **LoadStore**：确保所有先前的加载操作在任何后续存储操作之前完成。
3. **StoreLoad**：确保所有先前的存储操作在任何后续加载操作之前完成。
4. **LoadLoad**：确保所有先前的加载操作在任何后续加载操作之前完成（由 `smp_rmb()` 使用）。
5. **Sync**：在开始任何后续操作之前，确保所有先前的操作均已完成。
6. **MemIssue**：确保所有先前的内存操作在任何后续内存操作之前完成，这对于某些内存映射 I/O 实例很重要。
7. **Lookaside**：与 MemIssue 相同，但仅适用于先前的存储和后续的加载，且仅当它们访问相同的内存位置时有效。

**Linux 内核中的内存屏障**

&emsp;&emsp;Linux 内核的 `smp_mb()` 原语结合了前四种选项，使用如下形式：

```assembly
membar #LoadLoad | #LoadStore | #StoreStore | #StoreLoad
```

&emsp;&emsp;这确保了内存操作的完全排序。

&emsp;&emsp;membar #MemIssue 是必要的，因为 membar #StoreLoad 可能允许后续加载从写缓冲区获取值，这在写入 MMIO 寄存器时可能会导致副作用。相反，membar #MemIssue 会等待写缓冲区被刷新后再允许加载操作执行，确保加载操作从 MMIO 寄存器获取值。

&emsp;&emsp;驱动程序可以使用 membar #Sync，但在不需要 membar #Sync 的附加功能时，建议使用较轻量的 membar #MemIssue。membar #Lookaside 是 membar #MemIssue 的轻量级版本，适用于写入特定 MMIO 寄存器会影响下一个读取值的情况。然而，当写入某个 MMIO 寄存器会影响下一个从其他 MMIO 寄存器读取的值时，必须使用较重的 membar #MemIssue。

&emsp;&emsp;SPARC 架构没有将 wmb() 定义为 membar #MemIssue，而将 smb wmb() 定义为 membar #StoreStore，这可能会使某些驱动程序易受错误影响。可能所有运行 Linux 的 SPARC CPU 实现的内存排序模型比架构允许的更为保守。

&emsp;&emsp;在 SPARC 中，必须在存储指令和执行指令之间使用刷新指令，以确保刷新 SPARC 的指令缓存中的任何先前值。刷新指令接收地址，并且仅从指令缓存中刷新该地址。在 SMP 系统中，所有 CPU 的缓存都被刷新，但没有方便的方法来确定离线 CPU 刷新完成的时间。

### 7.7 x86和AMD64 架构的内存屏障指令
**AMD64**

&emsp;&emsp;AMD64 兼容 x86，并且最近更新了其内存模型，以强制执行实际实现已提供的一些更严格的排序。以下是 AMD64 对 Linux 内核中内存屏障原语的实现：

- **`smp_mb()`**：实现为 `mfence`
- **`smp_rmb()`**：实现为 `lfence`
- **`smp_wmb()`**：实现为 `sfence`

&emsp;&emsp;理论上，这些指令可能会被放松，但任何这种放松必须考虑到 SSE 和 3DNOW 指令。这意味着在设计和优化多线程代码时，开发者需要谨慎处理与这些指令相关的内存排序行为。 

&emsp;&emsp;x86 CPU 提供“进程排序”，确保所有 CPU 对某个 CPU 的写入顺序达成一致，因此 `smp_wmb()` 原语在 CPU 上实际上是一个无操作（noop）。然而，需要使用编译器指令来防止编译器执行可能导致在 `smp_wmb()` 原语之间重排序的优化。

**内存排序特性**
- **加载的排序保证**：x86 CPU 传统上没有对加载操作提供排序保证，因此 `smp_mb()` 和 `smp_rmb()` 原语被扩展为 `lock; addl`。这个原子指令对加载和存储操作都起到屏障作用。

**Intel 的内存模型**
&emsp;&emsp;Intel 最近发布了 x86 的内存模型。实际上，Intel 的 CPU 强制执行的排序比以前的规范中声称的要紧密，因此这个模型实际上只是强制执行早期的事实行为。更近期，Intel 发布了更新的内存模型，强制要求存储操作的全局总顺序，尽管单个 CPU 仍然可以认为自己的存储操作发生在该全局顺序之前。这种对总排序的例外允许涉及存储缓冲区的重要硬件优化。

&emsp;&emsp;软件可以使用原子操作来覆盖这些硬件优化，这也是原子操作通常比非原子操作更昂贵的原因之一。需要注意的是，这种总存储顺序在较旧的处理器上并不保证。

**SSE 指令的特殊性**
&emsp;&emsp;值得注意的是，一些 SSE 指令是弱排序的（如 `clflush` 和非时间移动指令）。支持 SSE 的 CPU 可以使用以下指令：

- `mfence` 用于 `smp_mb()` 
- `lfence` 用于 `smp_rmb()` 
- `sfence` 用于 `smp_wmb()` 

&emsp;&emsp;某些版本的 x86 CPU 具有启用乱序存储的模式位，因此对于这些 CPU，`smp_wmb()` 也必须被定义为 `lock; addl`。

&emsp;&emsp;虽然许多较旧的 x86 实现可以在没有特殊指令的情况下处理自修改代码，但较新版本的 x86 架构不再要求 x86 CPU 具备这种宽容性。值得注意的是，这种放宽正好让 JIT 实现者感到不便。

### 7.8 zSeries 架构

&emsp;&emsp;zSeries 机器构成了 IBM 主机家族，以前被称为 360、370 和 390。尽管并行性在 zSeries 中出现较晚，但考虑到这些主机在 1960 年代中期首次发货，这并不算什么。

- **指令**：Linux 的 `smp_mb()`、`smp_rmb()` 和 `smp_wmb()` 原语使用 `bcr 15,0` 指令。
- **内存排序语义**：zSeries 具有相对较强的内存排序语义，允许 `smp_wmb()` 原语成为无操作（nop），并且在你阅读此内容时，这一变化可能已经发生。zSeries 的内存模型实际上是顺序一致的，这意味着所有 CPU 将一致地同意来自不同 CPU 的无关存储操作的顺序。

&emsp;&emsp;与大多数 CPU 一样，zSeries 架构不保证缓存一致的指令流，因此，自修改代码在更新指令和执行之间必须执行序列化指令。尽管如此，许多实际的 zSeries 机器确实可以在没有序列化指令的情况下支持自修改代码。

&emsp;&emsp;zSeries 指令集提供了大量的序列化指令，包括比较并交换、某些类型的分支（例如，上述的 `bcr 15,0` 指令）和测试并设置等。

### 8 Are Memory Barriers Forever?

&emsp;&emsp;最近出现了一些系统，在一般的乱序执行和特别是重排序内存引用方面显著不那么激进。这种趋势会持续到内存屏障成为过去式的地步吗？

**支持内存屏障消失的论点**
- **大规模多线程硬件架构**：提出的架构中，每个线程在内存准备好之前会等待，期间有数十、数百甚至数千个其他线程在进行。这种架构中，不再需要内存屏障，因为某个线程会等待所有未完成的操作完成后再执行下一条指令。由于可能有成千上万的其他线程，CPU 将被完全利用，因此不会浪费 CPU 时间。

- **延迟隐藏技术**：越来越复杂的延迟隐藏硬件实现技术可能允许 CPU 提供完全顺序一致执行的错觉，同时仍然能够提供几乎所有乱序执行的性能优势。

**反对内存屏障消失的论点**

- **应用程序的限制**：能够扩展到千线程的应用程序数量极其有限。

- **实时要求**：对于某些应用，实时响应要求在几十微秒，这种需求本身已经很难满足，而在大规模多线程场景下，单线程的吞吐量极低，这将使得这些要求更加难以实现。

- **能效要求**：电池供电设备及环境责任所带来的日益严苛的能效要求。

&emsp;&emsp;究竟谁是正确的？我们无法确定，因此准备迎接这两种情景的到来。

### 8 硬件设计者对软件开发者的挑战

&emsp;&emsp;硬件设计者可以采取多种措施让软件开发变得困难。以下是一些我们在过去遇到的问题，希望能帮助避免未来出现类似问题：

1. **忽略缓存一致性的 I/O 设备**  
   这种迷人的缺陷可能导致从内存进行的 DMA 操作错过输出缓冲区的最近更改，或者同样糟糕的是，导致输入缓冲区在 DMA 完成后被 CPU 缓存的内容覆盖。为了使系统在面对这种错误行为时正常工作，必须在将任何 DMA 缓冲区呈现给 I/O 设备之前仔细刷新 CPU 缓存。此外，还需要非常小心以避免指针错误，因为即使是对输入缓冲区的错误读取也可能导致数据输入损坏！

2. **忽略缓存一致性的设备中断**  
   这听起来似乎没什么问题——毕竟，中断不是内存引用，对吧？但想象一下，一个具有分裂缓存的 CPU，其中一个缓存银行非常繁忙，因此保持了输入缓冲区的最后一个缓存行。如果相应的 I/O 完成中断到达这个 CPU，那么该 CPU 对缓冲区最后缓存行的内存引用可能返回旧数据，从而导致数据损坏，而这种损坏在后续的崩溃转储中将是不可见的。当系统开始转储出问题的输入缓冲区时，DMA 很可能已经完成。

3. **忽略缓存一致性的处理器间中断 (IPI)**  
   如果 IPI 在相应消息缓冲区的所有缓存行尚未写入内存之前就到达目标处理器，这可能会导致问题。

4. **缓存一致性被超越的上下文切换**  
   如果内存访问的完成顺序过于混乱，则上下文切换可能会相当麻烦。如果任务在源 CPU 的内存访问可见之前从一个 CPU 切换到另一个 CPU，那么该任务可能会看到相应变量恢复到先前的值，这可能会严重混淆大多数算法。

5. **过于宽容的模拟器和仿真器**  
   编写强制内存重排序的模拟器或仿真器是困难的，因此在这些环境中运行良好的软件在首次在真实硬件上运行时可能会遭遇令人不快的惊喜。不幸的是，硬件的狡猾程度通常超过模拟器和仿真器，但我们希望这种情况能够改变。
