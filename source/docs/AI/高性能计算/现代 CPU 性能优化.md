# 现代 CPU 性能优化

**——以 AMD Ryzen 5 5600X 为实验平台的 PMU、Top-Down 与优化技术实践手册**


## 1. 为什么优化经验经常是错的

&emsp;&emsp;优化经验之所以经常失效，不是因为它全错，而是因为它默认结论可以跨场景迁移。下面两个实验分别演示两种失效方式：把混杂变量误读成优化效果，以及忽略微架构自身的并行能力。

### 1.1 分配对齐

&emsp;&emsp;**实验 A**：写一个求和循环，把结果累加到 `long` 局部变量后打印。代码完全不变，只改数组起始地址：一份由 `malloc` 默认分配（通常 16 字节对齐），另一份由 `posix_memalign` 强制 64 字节对齐。

```c
// 两个版本调用完全相同的求和函数
__attribute__((noinline))
long sum_loop(const int *a, size_t n) {
    long s = 0;
    for (size_t i = 0; i < n; i++) s += a[i];
    return s;
}
// v1: malloc(n * sizeof(int))              → base % 64 = 16
// v2: posix_memalign(&p, 64, n * sizeof(int)) → base % 64 = 0
```

&emsp;&emsp;直觉上，64 字节对齐应该更快：向量化后的 32B load 不会跨缓存行。用 `clang++-22 -O2 -march=znver3` 在 5600X 上编译，三种工作集各跑一轮，结果却没那么简单：

**表 1-1 · 单次测量（首次运行）**

| 场景 | 工作集 | v1 median | v2 median | 表面差异 |
|---|---|---|---|---|
| L2 驻留 | 512 KB × 10000 扫描 | 55.14 ms | 64.66 ms | **v2 慢 17%** |
| L3 驻留 | 2 MB × 2000 扫描 | 48.00 ms | 48.17 ms | 几乎相同 |
| DRAM | 256 MB × 1 扫描 | 10.75 ms | 10.33 ms | v2 快 4% |

&emsp;&emsp;仅看这一轮，L2 场景 17% 的差异足以支撑“对齐有害”或“对齐有益”任何一种叙事。但把 v1/v2 交替跑三轮，结论就变了：

**表 1-2 · 交替测量（每轮先 v1 后 v2，共三轮）**

| 轮次 | v1 median | v2 median | 差异 |
|---|---|---|---|
| Round 1 | 55.62 ms | 56.54 ms | v1 快 1.6% |
| Round 2 | 55.74 ms | 56.55 ms | v1 快 1.5% |
| Round 3 | 55.07 ms | 49.72 ms | v2 快 9.7% |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261030_b944d9.svg)

&emsp;&emsp;**交替测量之后，前两轮的差异缩到 1.5%，第三轮 v2 反而快 9.7%，没有任何方向一致的系统性差异。** 单次测量看到的 17% 并不是对齐带来的：v2 首次运行时 CPU 尚未进入热稳态，Precision Boost 频率还在爬升；同时 `posix_memalign` 分到的那组物理页，恰好对 L2 set 冲突更不友好。第三轮 9.7% 的反转，是物理页抽奖的另一次结果。

&emsp;&emsp;这个实验的价值不在回答“对齐到底有没有用”，而在展示**优化经验失效的两种典型方式**：

- **把混杂变量误读为优化效果**。单次测量的 17% 来自“v2 先跑 + 频率未稳 + 物理页随机”三者叠加，却被归因于对齐；据此得出的“对齐有害”或“对齐有益”，方向相反，却同样是错的。
- **忽略微架构的并行化能力**。Zen 3 的 L1D 会把跨 64B 行的 32B load 拆成两个 32B 操作**并行发出**，16B 偏移完全在硬件能吸收的范围内；只有跨 4K 页边界时，split load 才会产生真正昂贵的串行惩罚。**对齐不是一个通用优化，而是一个需要测量验证的假设**——它在某些微架构、某些访问模式下有效，在另一些条件下则被 L2/L3/DRAM 的供给延迟完全掩盖。

### 1.2 缓存行填充

&emsp;&emsp;**实验 B**：两个线程各递增一个 `long` 计数器，各跑 1 亿次。v1 不加填充，`a`、`b` 位于同一条 64B 缓存行内；v2 用 `alignas(64)` 把两者分隔到各自独占的缓存行。

```c
struct counters_bad  { long a; long b; };                 // 同一 64B 行
struct counters_good { alignas(64) long a; alignas(64) long b; };

// 两个线程各自对 a、b 做 n 次递增，每次递增后插入 asm volatile 屏障，
// 阻止编译器把连续的 ++ 合并成 += n——保持真实的"读-改-写"节奏。
static void worker(long *p, long n, std::atomic<int> &ready, std::atomic<int> &go) {
    ready.fetch_add(1, std::memory_order_release);
    while (go.load(std::memory_order_acquire) == 0) {}
    for (long i = 0; i < n; i++) {
        (*p)++;
        __asm__ __volatile__("" : "+m"(*p) : : "memory");
    }
}
```

&emsp;&emsp;用 `taskset -c 0,1` 绑定到两个不同的物理核，以 `clang++-22 -O2 -march=znver3` 编译。这一次的数据与实验 A 完全相反：**方向一致、幅度巨大、机制可解释**。

**表 1-3 · 裸跑 1 亿次/线程（每线程 1 亿，总计 2 亿次递增）**

| 指标 | v1（伪共享） | v2（填充） | 比值 |
|---|---|---|---|
| median | 653.108 ms | 23.894 ms | **27.3×** |
| min | 580.816 ms | 21.964 ms | 26.4× |
| 每次递增耗时 | 3.27 ns | 0.12 ns | — |
| 每秒递增数 | 306 M/s | 8.37 G/s | — |

&emsp;&emsp;`0.12 ns/inc` 意味着 5600X 在 boost 频率下每周期约完成 1.8 次递增——两个核心各自推进私有缓存行，每核每周期约 0.9 次 `inc [mem]`。**v2 已经触及 Zen 3 的 L1D 吞吐上限**，再优化布局也压不下去。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_a70396.svg)

&emsp;&emsp;v1 的五次运行还揭示了一个细节：696.782 → 683.002 → 653.108 → 580.816 → 596.956，先降 16.7% 再反弹。这不是噪声，而是 **MESI 状态在收敛**：首次运行时缓存行在 Invalid 与 Modified 之间反复迁移，后续运行中两个核心的访问节奏逐渐稳定，一致性协议的开销落入了更规则的模式。**这种“越跑越快”本身就是伪共享的指纹**——真实的计算瓶颈不会因为重复运行而改变，只有协议状态机才会。

&emsp;&emsp;时间差已经足够说明问题，但要看清**机制**，用 `perf stat` 把规模缩到每线程 1 千万次（避免 perf 的采样开销扭曲结果）：

**表 1-4 · perf stat，每线程 1 千万次**

| 指标 | v1（伪共享） | v2（填充） | 比值 |
|---|---|---|---|
| cache-references | **2,940,541** | **373,345** | **7.9×** |
| cache-misses | 2,542,736 | 115,114 | 22.1× |
| miss rate | **86.5%** | 30.8% | — |
| cycles | 476,428,214 | 71,192,368 | 6.7× |
| instructions | 196,487,676 | 198,807,570 | 1.01× |
| **CPI** | **2.42** | **0.358** | **6.8×** |

&emsp;&emsp;两版的 `instructions` 几乎相同（差 1.2%），说明编译器生成的代码没有本质差异；`cycles` 却相差 6.7 倍——**性能差异全部来自“等待”**。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_913b18.svg)

&emsp;&emsp;**真正解释性能差异的是 `cache-references` 的绝对数，而不是 miss rate。** v1 每次递增平均产生 **0.147 次 L2/LLC 引用**，这正是 MESI 协议的行为：核心 0 写 `a` 时向 L2/LLC 发出 RFO，把包含 `a`、`b` 的整条缓存行从核心 1 抢过来；核心 1 随后写 `b` 时再反向抢一次。**每一次写操作都触发一次跨核一致性事务。** v2 每次递增平均只产生 **0.019 次引用**——`inc` 在本核 L1D 命中已拥有的缓存行，不产生跨核流量。

&emsp;&emsp;**v2 的 miss rate 仍有 30.8%，看起来“也不健康”，但它的引用基数只有 373K，每次 miss 又落在 L2/L3 命中，实际代价几乎为零。** 这是本章最需要记住的反直觉点：**miss rate 本身不能用于诊断，要看 miss 的绝对数、代价，以及它发生在哪一层缓存。** 只盯着 miss rate，会得出“v2 也不健康”的错误结论，而真正的瓶颈是 v1 引用次数的膨胀。CPI 从 0.358 升到 2.42，全部增量都来自流水线槽位等待缓存行所有权迁移——**退休指令数没变，槽位被一致性协议的等待消耗了**。这与 Top-Down 的预测吻合：v1 的瓶颈落在 Backend Bound → Memory Bound，而不是 Frontend Bound 或 Bad Speculation。

### 1.3 结论

&emsp;&emsp;实验 A 与实验 B 合起来，构成了“优化经验为何失效”的两个典型。

- **实验 A**：看起来应该有效的对齐优化，在 Zen 3 上被 L1D 的并行化能力与 L2/DRAM 的供给延迟完全掩盖，交替测量后没有系统性差异。
- **实验 B**：看起来只是“数据结构布局”的改动（`alignas(64)`），却把每次递增从 3.27 ns 压到 0.12 ns，加速 27 倍，方向可复现、机制可解释。两者的差别不在改动大小，而在于**改动是否触及了当前瓶颈所在的微架构机制**：实验 A 的 16B 偏移落在 L1D 能并行吸收的范围内；实验 B 直接改变了 MESI 协议的行为，缓存行所有权迁移带来了真实的、可测量的 7.9 倍引用膨胀。

&emsp;&emsp;**“优化经验”之所以经常错，是因为它默认“对齐有用”“数据布局有用”这类结论可以跨场景迁移，而测量一次次证明它们不能。** 判断一个改动是否有效，必须看它是否触及当前瓶颈所在的微架构机制；而判断“触及了哪个机制”的唯一可靠方式，是带对照、交替、重复的测量加 PMU 事件计数，而不是一次运行的时间差。

&emsp;&emsp;两个实验最终指向同一个结论：**现代 CPU 的性能不由“执行了多少条指令”决定，而由“每个周期提供的流水线槽位如何被消耗”决定**。Bakhvalov 在 *Performance Analysis and Tuning on Modern CPUs*（下称 perf-book）中把这一视角系统化为 Top-Down Microarchitecture Analysis Methodology（TMA）：开发者不必预先掌握所有 PMU 事件的语义，也能把瓶颈路由到前端、后端、错误推测或退休四个互斥类别之一。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261030_376e7f.svg)

&emsp;&emsp;本文的目标不是罗列优化技巧，而是建立一条可复现的闭环：**先做实验，再看 PMU 输出，解释微架构机制，最后用重新测量验证假设**。所有命令、代码与事件名都以 AMD Ryzen 5 5600X 为实验环境——Zen 3 架构、6 核 12 线程、单 CCD、32MB 统一 L3，Linux perf 对它有较完整的 vendor event 支持，适合作为“把方法论落到具体机器上”的样本。


## 2. 测量的第一课：基准在测量微架构状态，而不是“计算量”

### 2.1 频率不是常量，但漂移幅度取决于负载

&emsp;&emsp;连续运行同一个 CPU 密集型循环五次，中间不插入其他负载。

```bash
for i in 1 2 3 4 5; do
    perf stat -e cycles,instructions,aperf,mperf ./workload 2>&1 | tail -4
done
```

&emsp;&emsp;5600X 的基础频率 3.7 GHz，最大 boost 4.6 GHz，TDP 65W，默认 PPT 约 76W。直觉上，“第一次运行快、后续运行慢”应该来自缓存预热或频率漂移；但在 `amd-pstate-epp` 驱动下，**频率是否漂移、漂移多少，取决于 governor 的 EPP 设置，以及负载功耗是否逼近 PPT 或温度墙**。用 `freq_drift` 逐次读取 `scaling_cur_freq`，四种条件的数据如下：

**表 2-1 · 四种条件下的频率与耗时**

| 条件 | 频率 | ns/iter | 单次耗时 | 相对 baseline |
|---|---|---|---|---|
| `powersave` EPP，单核 | 4099 MHz | 1.47 | 293.7 ms | **+13.4%** |
| `performance` EPP，单核 | 4649 MHz | 1.29 | 258.9 ms | baseline |
| `performance` + 5 核 stress-ng，61.8°C | 4600 MHz | 1.31 | 261.3 ms | +0.9% |
| `performance` + 更重负载 | 4550 MHz | 1.32 | 263.9 ms | +1.9% |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261030_391bcc.svg)

&emsp;&emsp;**时间比与频率比严格一致**：powersave 与 performance 之间是 1.134 倍时间、1.134 倍频率；5 核负载与 baseline 之间是 1.009 倍时间、1.010 倍频率；重负载与 baseline 之间是 1.019 倍时间、1.021 倍频率。三者都落在 0.5% 的误差内。这说明 `freq_drift` 在四种条件下执行的是**完全相同的指令流**，**时间差异 100% 由频率解释**，没有缓存、分支或代码生成的混杂因素。`scaling_cur_freq` 的读数与执行速度严格成比例，可以作为微基准频率条件的可靠锚点。

&emsp;&emsp;但四种条件也说明了另一件事：**频率漂移并不是一个固定幅度的噪声源**。

- **EPP 切换带来 13.4% 的差异**——这是四种条件中最大的单一变量。`powersave` 下五次运行波动小于 0.02%，`performance` 下小于 0.06%，两者各自都“非常稳定”，但一个稳定在 4099 MHz，另一个稳定在 4649 MHz。**稳定不等于正确**：`powersave` 下测出的 293.7 ms 高度可重复，却不代表代码在不受 EPP 压制时的真实性能。
- **5 核 stress-ng 只带来 0.9% 的差异**——温度 61.8°C，远低于 5600X 的 Tjmax（95°C）。Precision Boost 通常在 75–80°C 才开始明显退档，61.8°C 意味着散热系统还有 15–20°C 余量，CPU 没有理由大幅降频。频率在 4599–4643 MHz 之间小幅抖动而非单调下降，反映的是功耗预算内的微调。
- **更重的负载带来 1.9% 的差异**——频率在 4549–4574 MHz 之间抖动（0.5% 的幅度），仍无单调下降趋势。即便功耗更接近 PPT，在良好散热下，频率也只是在 boost 上限附近小幅调整，而不是断崖式退档。

> 实际上，用手机芯片做这类实验更容易撞上温度墙；PC 处理器通常需要长时间高负载才会触发。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261030_c063d1.svg)

&emsp;&emsp;这意味着：**“频率漂移”不是一个可以笼统假设的噪声源，而是一个需要被测量的条件变量**。漂移幅度取决于**负载功耗是否逼近 PPT 或温度墙**：单核 30–40W 时温度稳定在 60°C 以下，boost 不降；5 核 55–65W 时温度到 61.8°C，boost 只微降 1%；只有把功耗推到 76W、或把温度推到 80°C 以上，漂移才会进入 5–10% 的量级。

&emsp;&emsp;**`cycles` 本身会“吸收”频率变化**，让 CPI 看起来正常，而实际吞吐已经下降——`cycles` 按频率计数，频率降了，计数也按比例变少。稳健的做法是同时采集 `cycles`、`instructions` 和 `aperf/mperf`，用 `mperf/aperf` 判断频率是否偏离 TSC 基准；同时在记录中标注 **EPP 设置、governor、后台负载和温度**——这四项是解释任何跨运行时间差异的前提。在 5600X 上做微基准比较之前，先做两件事：**把 EPP 固定为 `performance`，并记下当前温度**。前者消除了 13.4% 量级的配置差异，后者界定了频率漂移的幅度上限。


### 2.2 被编译器消除的基准比“测不准”更危险

&emsp;&emsp;朴素微基准会被编译器整个消除。perf-book 的检测方法是：用 PMU 确认 `instructions` 退休数是否与预期相符——如果一条本应执行 1000 次的循环只退休了 3 条指令，基准已经失效。

&emsp;&emsp;在 5600X 上写四个版本的求和基准，每个版本用不同方式试图阻止编译器消除。v1 朴素，循环内联到调用方，结果只用于 `printf`；v2 每次迭代写一个 volatile 全局变量；v3 把循环放进 `noinline` 函数，在函数**外部**加 asm 屏障；v4 把 asm 屏障放进循环**内部**。用 `perf stat` 采集 Zen 3 的退休指令数 `ex_ret_instr` 与 Op Cache 事件：

```bash
PERF_EVENTS="ex_ret_instr,macro_ops_retired,op_cache_hit_miss.all_op_cache_accesses,op_cache_hit_miss.op_cache_miss,de_dis_cops_from_decoder.disp_op_type.any_integer_dispatch"

for v in 1 2 3 4; do
    echo "=== version $v ==="
    taskset -c 2 perf stat -e "$PERF_EVENTS" \
        ./elim_bench -v $v -n 100000000 2>&1 | \
        grep -E 'version|time|ns/iter|ex_ret_instr|macro_ops_retired|op_cache|de_dis'
done
```

**表 2-2 · 四个版本的退休指令与耗时（n = 1 亿次迭代）**

| 版本 | 写法 | time | `ex_ret_instr` | `ex_ret_instr / n` |
|---|---|---|---|---|
| v1 | 朴素（内联 + printf） | **0.000090 ms** | 3,234,521 | **0.032** |
| v2 | volatile 全局 sink | 24.512 ms | 341,231,541 | **3.41** |
| v3 | `noinline` + 函数外屏障 | **0.000110 ms** | 3,207,347 | **0.032** |
| v4 | 循环内 asm 屏障 | 28.828 ms | 703,811,029 | **7.04** |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_4df583.svg)

&emsp;&emsp;**v1 和 v3 的 `ex_ret_instr / n` 都是 0.032——1 亿次迭代只退休了约 320 万条指令。** 如果循环真的执行了 1 亿次，哪怕每次只有一条加法，也应该退休 1 亿条指令。实测只有 320 万，差了两个数量级；这 320 万条不是循环的指令，而是 `perf stat` 启动、动态链接和 `printf` 本身的开销。**循环根本没有执行**——编译器把 `s += i` 归纳成了 `n*(n-1)/2 = 4999999950000000`，直接算出结果。

&emsp;&emsp;**v3 是本章最重要的教学点。** 它写了 `noinline`、写了 `asm volatile` 屏障、结果也被 `printf` 使用——三个看起来“应该有效”的措施全部到位，但 `ex_ret_instr` 与 v1 一样低，`time` 一样是 0.000 ms。原因很具体：`noinline` 只阻止**调用方内联 `simple_sum_noinline`**，函数**内部**的循环仍然被归纳成闭式解——函数调用保留了，循环没了。`asm` 屏障在函数**外部**，保护的是“`s` 的返回值不被优化掉”，而不是“循环体内部的累加不被归纳”。**屏障的位置决定了它保护什么**：循环外的屏障保护返回值，循环内的屏障保护循环体。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_5e3c78.svg)

&emsp;&emsp;**v4 是唯一真正阻止消除的写法。** `asm volatile("" : "+r"(s) : : "memory")` 放在循环体内，每次迭代都告诉编译器“`s` 可能被外部修改”，编译器无法把累加折叠成 `n*(n-1)/2`，循环必须逐次执行。`ex_ret_instr / n = 7.04`，与源码预期一致（加法、循环比较、跳转，以及屏障对代码生成的影响），`time = 28.8 ms`。

&emsp;&emsp;v2 用 volatile 全局变量也阻止了消除，代价是每次迭代多一次 store：`ex_ret_instr / n = 3.41`，`time = 24.5 ms`。**v2 与 v4 的时间差（24.5 vs 28.8 ms）远小于指令数差（3.41 vs 7.04）**——v4 的指令数是 v2 的两倍，时间只多 17%。原因在依赖链：`s += i` 的加法延迟很短，多出的指令填充了原本空闲的流水线槽位，CPI 从 v2 的约 1 降到 v4 的约 0.5。**指令数不能单独用于判断性能**——退休指令数告诉你有多少工作被完成，却不告诉你有多少工作在等待。

&emsp;&emsp;5600X 的 Op Cache 事件提供了第二道验证：

| 版本 | Op Cache 访问 | Op Cache 缺失 | 缺失率 |
|---|---|---|---|
| v1 | 878,563 | 348,266 | **39.6%** |
| v2 | 51,055,620 | 450,570 | 0.88% |
| v3 | 863,583 | 341,952 | **39.6%** |
| v4 | 101,077,489 | 462,762 | 0.46% |

&emsp;&emsp;v1 和 v3 的 Op Cache 访问只有 86 万次，缺失率却高达 39.6%——因为程序只执行了 `perf stat` 启动和 `printf` 的少量冷代码，它们不在 Op Cache 里。v2 和 v4 的 Op Cache 访问在 5000 万到 1 亿次量级，缺失率低于 1%——**循环代码完全驻留在 Op Cache 中，前端供给不是瓶颈**。把 `ex_ret_instr` 与 `op_cache` 放在一起看，比单看任何一个都可靠：前者告诉你“退休了多少指令”，后者告诉你“前端是否真的在供给这些指令”。

&emsp;&emsp;因此后续所有实验在跑计时之前，都先用 `perf stat -e ex_ret_instr,macro_ops_retired` 采集一次，计算 `ex_ret_instr / n`。判据不是 `time`，而是这个比值——**v1 的 `time = 0.000090 ms` 看起来像一个极快的微基准，`ex_ret_instr / n = 0.032` 却直接揭穿了它**。只看 `time`，无法区分“循环被消除”和“微基准本来就在微秒量级”；`ex_ret_instr` 可以一锤定音：与源码数出的每迭代指令数一致到 2 倍以内，基准有效；差一个数量级以上，基准失效。这个检查的成本是每次实验多一次 `perf stat`，收益是排除“测的根本不是你想测的东西”这个最危险的陷阱——**它比“测不准”更危险，因为输出看起来完全正常：`time` 很小、`result` 正确（v1/v3 的 4999999950000000 与 v4 一致）、程序也没有崩溃，只有退休指令数能揭穿它**。


## 3. 用 PMU 建立指标语言：CPI 之后用什么

### 3.1 CPI 是聚合指标，不是诊断指标

&emsp;&emsp;四个 workload 的设计意图与实际行为如下：

- **v1 compute**：三条独立的 64 位乘法链，意图制造 Core Bound（执行端口/依赖链）。如果三条链真正并行，CPI 应接近乘法吞吐上限；如果编译器把链合并、或把依赖并行了，CPI 会异常低——实际发生的正是后者。
- **v2 chase**：随机排列的指针追逐，意图制造 Memory Bound（L3 延迟）。8MB 工作集超出 512KB L2、仍落在 32MB L3 内，每次追逐都会触发一次 L3 访问。
- **v3 branch**：数据驱动的 `if (v >= 128)`，意图制造 Bad Speculation（约 50% 误预测率）。分支方向由随机数据决定，预测器无法学习。
- **v4 frontend**：64 次宏展开的巨型循环体，意图制造 Frontend Bound（Op Cache 溢出）。展开后约 900 条 macro-op，预期超出 Zen 3 的 4K uop Op Cache。

&emsp;&emsp;`cycles / ex_ret_instr` 给出 CPI。直觉上 CPI 越低越快——每周期退休 2 条指令，比每周期退休 0.2 条高效 10 倍。四个 workload 的数据（n = 1e7，EPP = performance）却推翻了这个直觉：

**表 3-1 · 四个 workload 的 CPI 与绝对量**

| 版本 | 设计意图 | cycles | `ex_ret_instr` | CPI | 每迭代指令数 | 每迭代时间 |
|---|---|---|---|---|---|---|
| v1 compute | Core Bound | 33.6M | 82.1M | **0.41** | 8.2 | 3.4 ns |
| v2 chase | Memory Bound | 593.0M | 115.7M | **5.12** | 11.6 | 59.3 ns |
| v3 branch | Bad Speculation | 198.5M | 508.1M | **0.39** | 50.8 | 19.8 ns |
| v4 frontend | Frontend Bound | 3217.8M | 8990.2M | **0.358** | 899.0 | 321.8 ns |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_9e39b0.svg)

&emsp;&emsp;**v4 的 CPI 是 0.358，比 v1 的 0.41 还低，却花了 3.2G 个周期，是 v1 的 96 倍。** 原因在“每迭代指令数”一列：v1 每迭代 8.2 条，v4 每迭代 899 条。CPI 低只说明“每条指令平均占用的周期少”，完全不告诉你有多少条指令。**每迭代时间其实是三个量的乘积**：

$$\text{每迭代时间} = \text{每迭代指令数} \times \text{CPI} \times \text{每周期时间}$$

&emsp;&emsp;v1 是 8.2 × 0.41 × 0.21 ns ≈ 0.71 ns，v4 是 899 × 0.358 × 0.21 ns ≈ 67.6 ns。表中每迭代时间（3.4 ns vs 321.8 ns）比这个估算大一个量级，因为 `n` 是外层迭代数，而 v4 每次外层迭代执行了 64 次内层 STEP——899 条指令是 64 次 STEP 的展开结果，不是单次累加。**具体数值不重要，重要的是三者的关系**：CPI 只是其中一个因子，单独看它无法判断快慢。

&emsp;&emsp;更关键的是，**CPI 也不告诉你“慢在哪里”**。v2 的 CPI 是 5.12，远高于其他三个，看起来最慢；但如果只看 CPI，你会以为 v2 需要 12.8 倍的指令级优化——实际上 v2 的时间是 v1 的 17.6 倍，CPI 只差 12.5 倍，差额来自每迭代指令数也不同（11.6 vs 8.2）。**CPI 把“指令数”和“每指令周期”两个维度揉成了一个数**；要拆分它们，必须同时采集 `cycles` 和 `ex_ret_instr` 两个原始量，而不是只看比值。

### 3.2 低 CPI 的三种不同成因

&emsp;&emsp;原本预期实验 A/B 会演示“同一个 CPI，两种瓶颈”，但四个 workload 的数据显示：**没有两个版本的 CPI 相同**。v1（0.41）、v3（0.39）、v4（0.358）都很低，成因却完全不同——这不是“同一个 CPI”的案例，而是“**低 CPI 背后藏着三种完全不同的状态**”的案例。

&emsp;&emsp;要判断每个版本受限于什么，需要看**三个比值**，而不是 CPI：

**表 3-2 · 三个比值把“低 CPI”拆成三个方向**

| 版本 | CPI | `cache-misses / cache-references` | `ex_ret_brn_misp / ex_ret_brn` | `op_cache_miss / all_accesses` | `ls_mab_alloc` | 判读 |
|---|---|---|---|---|---|---|
| v1 | 0.41 | 11.4% | 0.58% | **1.87%** | **51.8K** | Retiring（无瓶颈） |
| v2 | 5.12 | **48.4%** | 3.12% | 20.7% | **13.2M** | **Memory Bound** |
| v3 | 0.39 | 6.27% | 2.20% | 11.2% | 2.29M | **基准失效（cmov）** |
| v4 | 0.358 | 4.07% | 2.51% | **0.41%** | 193K | **基准失效（Op Cache 未溢出）** |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_e30fbc.svg)

&emsp;&emsp;**v2 是唯一被数据证实的瓶颈**。`cache-misses / cache-references = 48.4%`——近一半的缓存访问都 miss 到 L2/L3；`ls_mab_alloc = 13.2M`，即每迭代 1.32 次 MAB 分配。每次随机指针追逐都要在 Miss Address Buffer 里排队等待 L3 返回，这是 Memory Bound 的直接证据。`op_cache_miss / all_accesses = 20.7%` 看起来也高，但那是随机访问让代码预取失效所致，并不说明前端供给成了瓶颈。

&emsp;&emsp;**v1 是唯一“健康”的版本**。`cache-misses / cache-references = 11.4%`，但 `cache-references` 只有 258.8K——绝对量极小，11.4% 的 miss 只对应 29.4K 次缺失，对 82M 条指令而言是噪声。`ls_mab_alloc = 51.8K`，`op_cache_miss / all_accesses = 1.87%`，两个与内存、前端相关的关键事件都极低。**v1 是真正的 Retiring**：槽位被有效退休指令填满，没有哪个子系统拖后腿。CPI 0.41 意味着每周期退休 2.44 条指令，约为 Zen 3 六宽 dispatch 上限的 40%。

&emsp;&emsp;**v3 是基准失效**。它的 CPI 只有 0.39，`ex_ret_instr` 却有 508M（每迭代 50.8 条）。原因在于编译器把 `if (v >= 128) sum += v; else sum -= v;` 优化成了**条件移动 `cmov`**——根本没有分支。`ex_ret_brn = 63.8M / 10M = 6.4` 分支/迭代，但其中绝大多数是**循环控制分支**（10M 次 taken，只有最后一次不 taken），不是数据分支。`brn_misp / brn = 2.2%` 恰好等于循环控制分支的误预测率加上少量残留，**没有测到任何数据分支**。

&emsp;&emsp;**v4 也是基准失效**。它的 CPI 只有 0.358，`ex_ret_instr` 高达 8990M（每迭代 899 条）。设计目标是让巨型循环体超出 Op Cache 容量，但 `op_cache_miss / all_accesses = 0.41%`——**Op Cache 完全接住了循环体**。Zen 3 的 Op Cache 容量是 4096 个 uop，64 次 STEP 展开只产生约 900 条 macro-op，远未触及上限。`ls_mab_alloc = 193K`（每迭代 0.019 次）也极低，说明内存子系统不是瓶颈。**v4 同样是 Retiring**——只是它每迭代退休 899 条指令，CPI 天然就低。

### 3.3 从“测量了什么”到“基准是否有效”

&emsp;&emsp;表 3-2 的最后一列是本章最重要的发现：**四个精心设计的 workload，三个没有测到预期瓶颈**。v1 设计为 Core Bound，但三条独立乘法链让编译器把延迟完全并行化了，CPI 0.41 说明没有瓶颈；v3 设计为 Bad Speculation，但 clang 把数据依赖分支优化成了 `cmov`，根本没有分支可误预测；v4 设计为 Frontend Bound，但 899 条指令的循环体对 4096 个 uop 的 Op Cache 来说远远不够。

&emsp;&emsp;**这不是实验失败，而是第 2 章“基准在测你想测的东西吗”的延续**：第 2.2 节讲的是“循环被完全消除”，本节讲的是“循环被完整执行了，但没有走到你想测的那条路径”。两者都要过同一道检查——**PMU 数据是否与源码预期一致**。v3 的 `ex_ret_brn / n = 6.4` 看起来像“每次迭代 6.4 个分支”，但源码里只有一个数据分支；差额来自循环展开后的控制分支，数据分支已被 `cmov` 吃掉。**只有看 `ex_ret_brn` 的绝对量，才能发现“分支数不对”**。

&emsp;&emsp;**修复三个失效的基准**，需要回到源码：

- **v1 改成真正的 Core Bound**：去掉循环末尾的 asm 屏障，把三条独立链改成一条串行链：
  ```c
  long a = 1;
  for (long i = 0; i < n; i++) {
      a = a * 3 + 1;              // 乘法延迟 ~3 cycles
      __asm__ __volatile__("" : "+r"(a) : : "memory");  // 阻止合并成 a*15+7
      a = a * 5 + 2;
  }
  ```
  这样每次乘法必须等上一次，CPI 会升到 3 左右，`cache-misses` 和 `brn-misp` 仍然极低——真正的 Core Bound。

- **v3 改成真正的 Bad Speculation**：用 `__attribute__((noinline))` 把分支体拆成两个函数，强制保留分支：
  ```c
  __attribute__((noinline)) long add_slow(long s, int v) { return s + v; }
  __attribute__((noinline)) long sub_slow(long s, int v) { return s - v; }
  // 循环体内：
  if (__builtin_expect(v >= 128, 0)) s = add_slow(s, v);
  else                                s = sub_slow(s, v);
  ```
  `cmov` 无法跨函数调用，分支必须保留，`brn_misp / brn` 会升到 30–50%。

- **v4 改成真正的 Frontend Bound**：把 STEP 展开从 64 次增加到 **512 次**，让循环体超过 4096 个 uop：
  ```c
  #define STEP(a,b,c) do { a = a*3+1; b = b*5+2; c = c*7+3; \
                            a ^= a>>5; b ^= b>>7; c ^= c>>9; } while (0)
  for (long i = 0; i < n; i++) {
      STEP(a,b,c); STEP(a,b,c); ...  /* 共 512 次 */ ... STEP(a,b,c);
  }
  ```
  循环体宏展开后约 8000 条指令，远超 Op Cache 容量，`op_cache_miss / all_accesses` 会升到 20% 以上，真正的 Frontend Bound。

&emsp;&emsp;因此后续所有实验在看 CPI 之前，先看 `cache-misses / cache-references`、`ex_ret_brn_misp / ex_ret_brn` 和 `op_cache_miss / all_accesses` 三个比值。**三个比值都低、CPI 也低**——要么是 Retiring，要么是基准失效，需要进一步区分；**某个比值高**——信号明确，可以下钻到对应的 Zen 3 事件。**v2 是唯一被数据证实的瓶颈**：48.4% 的 miss 率加上 13.2M 次 MAB 分配，构成了 Memory Bound 的直接证据。**v3 和 v4 的失效不是坏事**——它们演示了第 2 章的教训在更大规模上的延续：**“执行了循环”和“执行了你想测的那条路径”是两件不同的事**，只有 PMU 事件计数能区分。


## 4. Top-Down 方法论与 5600X 上的采集实现

&emsp;&emsp;本章沿用第 2、3 章的测量纪律：EPP 固定为 `performance`，`taskset -c 2` 绑单核，n = 1e7，每个事件独立一个 `-e`。四个 workload 分别对应四个桶的设计意图：

```c
// topdown_bench.cpp 的核心函数（完整代码见附录）
__attribute__((noinline)) long core_bound(long n) {         // v1: Core Bound
    long a = 1;
    for (long i = 0; i < n; i++) {
        __asm__ __volatile__("" : "+r"(a) : : "memory");
        a = a / g_divisor + (i & 7) + 1;                     // volatile 除数，强制 idiv
        __asm__ __volatile__("" : "+r"(a) : : "memory");
    }
    return a;
}

__attribute__((noinline)) long memory_bound(const long *next, long n, long s) {  // v2: Memory Bound
    long idx = s, sum = 0;
    for (long i = 0; i < n; i++) {
        idx = next[idx];                                     // 8MB 随机指针追逐
        sum += idx;
        __asm__ __volatile__("" : "+r"(idx), "+r"(sum) : : "memory");
    }
    return sum;
}

__attribute__((noinline)) long branch_bound(const int *data, long n) {          // v3: Bad Speculation
    long sum = 0;
    for (long i = 0; i < n; i++) {
        int v = data[i];
        if (v >= 128) { sum += v; __asm__ __volatile__("" : "+r"(sum) : : "memory"); }
        else          { sum -= v; __asm__ __volatile__("" : "+r"(sum) : : "memory"); }
    }
    return sum;
}

__attribute__((noinline)) long frontend_bound(long n) {     // v4: Frontend Bound
    long a = 1, b = 2, c = 3;
    for (long i = 0; i < n; i++) {
        /* 512 次 STEP 宏展开，循环体约 7176 条 macro-op，超 Op Cache 容量 */
        __asm__ __volatile__("" : "+r"(a), "+r"(b), "+r"(c) : : "memory");
    }
    return a ^ b ^ c;
}
```

&emsp;&emsp;采集分两层：**Level 1** 用 `cycles`、`ex_ret_instr`、`ex_ret_brn`、`ex_ret_brn_misp`、`cache-references`、`cache-misses` 六个事件计算 CPI 和三个比值；**Level 2** 用 `op_cache_hit_miss.*`、`de_dis_cops_from_decoder.*`、`ls_mab_alloc.*` 三组 Zen 3 事件下钻到前端和内存子系统。实测中 `perf list metric | grep -i pipeline` 返回空——**这一版 perf 在 5600X 上不提供 `PipelineL1` metric**，所以 Level 1 的四个桶只能从原始事件近似，不能像 Intel 那样直接跑 `--topdown`。

```bash
# Level 1：六个事件
taskset -c 2 sudo perf stat -e cycles -e ex_ret_instr -e ex_ret_brn \
    -e ex_ret_brn_misp -e cache-references -e cache-misses \
    ./topdown_bench -v <1|2|3|4> -n 10000000

# Level 2 前端
taskset -c 2 sudo perf stat -e cycles -e op_cache_hit_miss.all_op_cache_accesses \
    -e op_cache_hit_miss.op_cache_miss \
    -e de_dis_cops_from_decoder.disp_op_type.any_integer_dispatch \
    ./topdown_bench -v <1|2|3|4> -n 10000000

# Level 2 内存
taskset -c 2 sudo perf stat -e cycles -e ls_mab_alloc.load_store_allocations \
    -e cache-references -e cache-misses \
    ./topdown_bench -v <1|2|3|4> -n 10000000
```

&emsp;&emsp;四个精心设计的 workload 中，只有一个明确落入了设计意图的桶：v2 的 `cache-misses / cache-references = 47.3%` 与 `ls_mab_alloc / n = 1.32` 确认了 Memory Bound。v4 的 `op_cache_miss / all_accesses = 84.1%` 与 `de_dis_cops / cycles = 3.19` 同时偏高，本章据此把它暂列 Frontend Bound，但它与第 5 章五个版本处在同一比值区间，严格判定留到 5.4 节。v1 的几轮修复（串行乘法链 → 常数除法 → runtime 除法 → volatile 序列化）全部失效——Zen 3 约 320 条 ROB 和六宽 dispatch 足以把 `idiv` 20 多个周期的延迟藏在后续独立指令后面；v3 的 `ex_ret_brn_misp / ex_ret_brn = 7.9%` 说明 asm 屏障仍然没能阻止编译器把数据分支优化掉。**这两个失效不是实验失败，而是 5600X 上做 Top-Down 诊断的两个真实边界**——它们与 v2 的成功、v4 的临界状态一起，构成了本章的核心叙事：**Top-Down 的路由结果必须用 PMU 数据验证，验证失败时的信息量往往比成功时更大**。

### 4.1 Level 1 的四个类别

&emsp;&emsp;Top-Down Level 1 输出四个互斥类别，总和约 100%：**Frontend Bound**（取指/解码供给不足）、**Backend Bound**（执行或内存资源成为瓶颈）、**Bad Speculation**（错误推测浪费槽位）、**Retiring**（槽位被有效退休指令填充）。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_9ede2b.svg)

### 4.2 在 5600X 上采集 Level 1

&emsp;&emsp;`perf stat --topdown` 在 Zen 3 上会回退到手动公式和原始 PMU 事件。5600X 没有与 Intel TMA 完全对应的硬件级槽位分类器，`PipelineL1` metric 在实测中不可用——因此 Level 1 的四个桶必须从原始事件**近似**：

| 桶 | 近似判据 | 主要事件 |
|---|---|---|
| **Frontend Bound** | `op_cache_miss / all_accesses > 10%` 且 `de_dis_cops / cycles > 3` | `op_cache_hit_miss.*`、`de_dis_cops_from_decoder.*` |
| **Bad Speculation** | `ex_ret_brn_misp / ex_ret_brn > 30%` | `ex_ret_brn_misp`、`ex_ret_brn` |
| **Backend Bound / Memory** | `cache-misses / cache-references > 30%` 且 `ls_mab_alloc / n > 0.5` | `cache-misses`、`ls_mab_alloc` |
| **Backend Bound / Core** | CPI 高（> 2）且以上三个比值都低 | 排除法 |
| **Retiring** | CPI 低（< 1）且以上三个比值都低 | 排除法 |

&emsp;&emsp;四个 workload 的实际数据如下（n = 1e7，EPP = performance，`taskset -c 2`）：

**表 4-1 · 四个版本的 Level 1 与 Level 2 关键比值**

| 版本 | 设计意图 | CPI | `cache-miss%` | `brn-misp%` | `op_cache-miss%` | `ls_mab_alloc/n` | 实际落桶 |
|---|---|---|---|---|---|---|---|
| v1 core | Core Bound | **0.96** | — | — | — | — | **失效（Retiring）** |
| v2 chase | Memory Bound | **4.96** | **47.3%** | 3.4% | 19.8% | **1.32** | **✅ Memory Bound** |
| v3 branch | Bad Speculation | **0.62** | 7.0% | **7.9%** | 7.8% | 0.23 | **失效（分支被消除）** |
| v4 frontend | Frontend Bound | **0.358** | 2.5% | 8.0% | **84.1%** | 0.12 | **Frontend 临界（见 5.4）** |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_0e5298.svg)

&emsp;&emsp;**v2 是唯一被数据证实的 Memory Bound**。`cache-misses / cache-references = 11,722,664 / 24,774,509 = 47.3%`——近一半的缓存访问 miss 到 L2/L3；`ls_mab_alloc / n = 13,217,809 / 10,000,000 = 1.32`——每次迭代平均分配 1.32 个 MAB 条目，排队等待 L3 返回。两个内存事件同时高，CPI 5.0 正是这种等待的结果。`op_cache_miss / all_accesses = 19.8%` 看起来也高，但那是随机指针追逐让代码预取失效所致，不是前端供给瓶颈——`de_dis_cops / cycles = 117.9M / 588.9M = 0.20`，远低于 3，说明解码器没有成为瓶颈。

&emsp;&emsp;**v4 是当时判为 Frontend Bound 的版本，第 5 章会把它修正为临界案例**。`op_cache_miss / all_op_cache_accesses = 3,730,052,076 / 4,432,747,576 = 84.1%`——512 次 STEP 展开（约 7176 条 macro-op）远超 Zen 3 的 4096 uop Op Cache 容量，绝大多数取指回落到四宽解码器。`de_dis_cops / cycles = 82,021,949,578 / 25,693,920,601 = 3.19`——每周期从解码器分发 3.19 个 cop，约为四宽解码器上限的 80%。**两项签名同时偏高**，所以在第 4 章的数据粒度下被归入 Frontend Bound；但 3.19 与第 5 章五个版本的 3.18 几乎相同，说明仅凭这组比值无法区分“前端成为瓶颈”和“后端仍在消费”。第 5 章会用完整的展开梯度重新检验这个判定。

&emsp;&emsp;**v1 和 v3 都是基准失效**。v1 的 CPI 0.96 看起来“健康”，但几轮修复（串行乘法链 → 常数除法 → runtime 除法 → volatile 序列化）全部失败——`result = 11` 说明除法确实执行了，只是 Zen 3 约 320 条 ROB 把 `idiv` 20 多个周期的延迟藏在了后续独立指令后面，CPI 被稀释到 1 以下。v3 的 `ex_ret_brn_misp / ex_ret_brn = 6,271,330 / 79,042,222 = 7.9%`，远低于 Bad Speculation 的 30% 阈值——asm 屏障仍然没能阻止编译器把数据分支优化成 `cmov` 或条件选择，真正的分支只剩循环控制分支（10M 次 taken，最后一次不 taken，误预测率极低）。

### 4.3 下钻到 Level 2：Backend Bound 的 Memory/Core 分裂

&emsp;&emsp;Backend Bound 被划分为 Memory Bound 和 Core Bound。**Memory Bound 是一个桶，不是一种原因**——它同时装着 L1D 缺失、L2/L3 缺失、DTLB 缺失、预取效率不足和带宽饱和。在 5600X 上，尤其要关注 **MAB（Miss Address Buffer）分配压力**：`ls_mab_alloc` 直接反映 load/store 请求在 miss 队列中的排队程度。六个核心共享 32MB L3，L3 未命中会经 Infinity Fabric 走向内存控制器，而 FCLK/UCLK/MCLK 的同步状态决定了未命中的实际延迟。

&emsp;&emsp;v2 的数据给出了这种分裂的直接证据：`cache-misses / cache-references = 47.3%` 说明缓存缺失率高，`ls_mab_alloc / n = 1.32` 说明每次缺失都在 MAB 里排队。**两个事件一起看，才能确认“慢在等 L3”，而不是“慢在执行端口”**——v2 的 `op_cache_miss = 19.8%` 虽然也不低，但 `de_dis_cops / cycles = 0.20` 说明前端供给不是瓶颈，真正的排队点在 `ls_mab_alloc`。

&emsp;&emsp;**v1 的失效恰好是 Core Bound 的反面证据**。设计意图是让串行 `idiv` 依赖链把后端执行资源变成瓶颈，但 Zen 3 的宽乱序窗口把 `idiv` 的延迟藏了起来，CPI 降到 0.96——三个比值和 `ls_mab_alloc` 都低，看起来像 Retiring，实际上是基准没有制造出 Core Bound。**在 5600X 上制造 Core Bound，需要比 Intel 苛刻得多的条件**（端口冲突、序列化指令或极长的依赖链），这不是代码问题，而是 Zen 3 的微架构特性。

### 4.4 最小闭环：预取在什么条件下无效

&emsp;&emsp;v5 是流式 triad 基线（`a[i] = b[i] + c[i]`，256MB 工作集超出 32MB L3），v6 用 `__builtin_prefetch` 提前 16 个迭代预取 `b` 和 `c`。数据如下：

**表 4-2 · 预取实验（n = 67108864，256MB 工作集）**

| 指标 | v5 baseline | v6 预取 | 变化 |
|---|---|---|---|
| cycles | 2,782,432,894 | 2,806,034,566 | **+0.85%** |
| `ex_ret_instr` | 5,676,460,733 | 5,994,411,209 | +5.6% |
| `cache-references` | 124,651,733 | 125,110,436 | +0.37% |
| `cache-misses` | 6,354,277 | 6,278,120 | **-1.2%** |
| `ls_mab_alloc` | 53,545,665 | **47,897,248** | **-10.5%** |

&emsp;&emsp;**预取没有降低 Backend Bound**。`cache-misses` 只降了 1.2%，`ls_mab_alloc` 降了 10.5%——软件预取确实减轻了 MAB 排队压力，`cycles` 反而增加了 0.85%。原因在 `ex_ret_instr`：v6 多了 5.6% 的指令（预取指令本身），它们填充了原本空闲的槽位，却**没有改变内存控制器的可持续吞吐**。这组实验的内存配置是 **DDR4-3600 双通道**（与第 10、11 章的 DDR4-2667 不同），256MB 工作集下的可持续带宽约 25 GB/s。预取只是让请求更早进入 MAB 队列，队列的**平均长度**并没有显著缩短——`ls_mab_alloc` 降 10.5%，是因为预取把部分 demand miss 转成了 prefetch hit，而 `cycles` 的瓶颈已经从“延迟等待”切换到了“带宽饱和”。

&emsp;&emsp;**“改了—重测—解释为什么变了或没变”，才是 TMA 的使用方式**。v5 → v6 的改动让 `ls_mab_alloc` 降了 10.5%，`cycles` 却没降，说明预取解决的是“MAB 排队”，不是“内存带宽”。如果已经处于带宽上限，预取**不会降低 Backend Bound**——它只是把“延迟问题”换成了“带宽问题”。**判据不是“预取有没有让 cache-misses 下降”，而是“预取有没有让 cycles 下降”**——后者才是 Backend Bound 是否真正缓解的唯一标准。

### 4.5 小结

&emsp;&emsp;四个精心设计的 workload，只有 v2（Memory Bound）明确落入了设计意图的桶；v4 的 Op Cache 确实溢出，但只能算“前端临界”，第 5 章会重新判定。v1 与 v3 的失效不是实验失败——它们恰好演示了第 1 章主题在 Top-Down 层面的延续：

- **v1 的多次修复**（串行乘法链 → 常数除法 → runtime 除法 → volatile 序列化）全部被编译器或 Zen 3 的乱序窗口化解。**在 5600X 上制造 Core Bound，需要比 Intel 苛刻得多的条件**——这是 Zen 3 的微架构特性，不是代码问题。
- **v3 的 asm 屏障**（两个分支体各自带 `asm volatile`）仍然没能阻止编译器优化掉数据分支。`ex_ret_brn_misp / ex_ret_brn = 7.9%` 说明真正的数据分支几乎不存在，只有循环控制分支被保留。**在 `clang -O2 -march=znver3` 下制造 Bad Speculation，需要比 asm 屏障更激进的序列化手段**（函数调用边界、`noinline` 分支体，或 `__builtin_expect` 配合 `noinline`）。

&emsp;&emsp;**这两个失效是本章最有价值的教学内容**。它们说明：**Top-Down 的路由结果必须用 PMU 数据验证，而不能假设“设计了哪个桶就会落到哪个桶”**。v2 证明了“用原始事件映射到四个桶”在 5600X 上可行；v4 的临界状态说明，单个比值只能用来排除、不足以完成最终分类；v1 和 v3 则证明，“桶的设计意图”与“实际测到的桶”之间隔着编译器优化和微架构特性两层，每一层都需要 PMU 数据验证。**验证失败时的信息量，往往比成功时更大**——它指出了基准设计中的隐藏假设，只有修正这些假设，下一轮测量才能落到真正的瓶颈上。


## 5. 前端双路径：解码器、Op Cache 与 dispatch

&emsp;&emsp;第 4 章的 v4 已经展示了 512 次 STEP 展开时 Op Cache 的溢出（84.1% miss），并在那一章被临时归入 Frontend Bound。但这个判定只建立在一个数据点上，回答不了两个问题：Op Cache 的边界在哪里，溢出之后前端是否真的成为瓶颈。本章用五个 STEP 展开版本，画出从完全命中到完全溢出的完整梯度。

&emsp;&emsp;**实验设计**。五个版本共用同一个 STEP 宏（`a*3+1`、`b*5+2`、`c*7+3`，三次异或、三次移位），每个 STEP 约 14 条 macro-op，三条独立链（`a`、`b`、`c`）各自的依赖链约 6 cycles。唯一的变量是外层循环体内 STEP 的展开次数：

```c
// 一次 STEP 约 14 条 macro-op，三条独立链，无跨 STEP 代数合并空间
#define STEP(a,b,c) do { \
    a = a * 3 + 1; b = b * 5 + 2; c = c * 7 + 3; \
    a ^= (a >> 5); b ^= (b >> 7); c ^= (c >> 9); \
} while (0)

// v1: S4    → 每迭代 ~58 条 macro-op
// v2: S16   → ~227 条
// v3: S128  → ~1799 条    ← 预期接近 Op Cache 边界
// v4: S512  → ~7187 条    ← 预期溢出
// v5: S1024 → ~14385 条   ← 彻底溢出
```

&emsp;&emsp;选择 STEP 而不是任意指令填充，是因为它**天然阻止编译器优化**：三条链各自有乘法→加法→异或→移位的非线性依赖，跨 STEP 合并需要模拟非线性递推，任何生产级编译器都不会做。于是“每迭代 macro-op 数”严格等于 STEP 次数 × 14 加循环开销，不需要额外的 asm 屏障去防消除——**在基准设计层面取胜，比事后用屏障堵编译器的洞更可靠**。

&emsp;&emsp;五个版本的 `n` 按 STEP 次数反比缩放，保证总指令量在 1e9 量级、单次运行 0.1–0.5 秒：v1 用 `n=1e8`，v2 用 `n=1e7`，v3 用 `n=1e6`，v4 用 `n=3e5`，v5 用 `n=1e5`。测量条件与第 2–4 章一致：EPP 固定为 `performance`，`taskset -c 2` 绑单核，每个事件独立一个 `-e`（避免 `-e "a,b,c"` 在 Zen 3 上解析失败）。

&emsp;&emsp;采集分两组：**Op Cache 组**用 `op_cache_hit_miss.all_op_cache_accesses` 和 `.op_cache_miss` 计算命中率，这是“前端走了哪条路径”的唯一直接证据；**dispatch 组**用 `de_dis_cops_from_decoder.disp_op_type.any_integer_dispatch` 除以 `cycles` 计算分发速率，这是“前端供给是否跟得上后端消费”的唯一直接证据。两组一起看，才能区分“Op Cache 溢出”与“Frontend Bound”——这正是本章要证明的核心区分。

```bash
# 有效性检查：五个版本的 ex_ret_instr / n 应严格按 STEP 次数线性增长
for v in 1 2 3 4 5; do
    taskset -c 2 sudo perf stat -e cycles -e ex_ret_instr \
        ./opcache_bench -v $v
done

# Op Cache 组
for v in 1 2 3 4 5; do
    taskset -c 2 sudo perf stat -e cycles -e ex_ret_instr \
        -e op_cache_hit_miss.all_op_cache_accesses \
        -e op_cache_hit_miss.op_cache_miss \
        ./opcache_bench -v $v
done

# dispatch 组
for v in 1 2 3 4 5; do
    taskset -c 2 sudo perf stat -e cycles -e ex_ret_instr \
        -e de_dis_cops_from_decoder.disp_op_type.any_integer_dispatch \
        ./opcache_bench -v $v
done
```

&emsp;&emsp;五个版本的 Op Cache miss 率从 0.28% 跳到 99.2%，断层落在 v3（1799 条）与 v4（7187 条）之间，与 Zen 3 的 4096 uop 规格容量吻合。但 `de_dis_cops / cycles` 在五个版本上全部稳定在 **3.18–3.24**：Op Cache miss 率变化了 350 倍，分发速率纹丝不动。**这说明 Op Cache 溢出只改变了前端的取指路径，没有改变前端的供给能力**——后端的消费速率（3.18 cop/cycle）始终低于解码器的供给上限（四宽），溢出没有转化为 Frontend Bound。本章的核心结论由此展开：**“Op Cache 溢出”和“Frontend Bound”之间不能画等号，判据是 `de_dis_cops / cycles` 与 4 的关系**。

### 5.1 有效性检查：五个版本的指令数都符合宏展开预期

&emsp;&emsp;先确认五个版本都真的在跑对应数量的 STEP，没有被编译器合并。`ex_ret_instr / n` 是唯一可靠的判据：

```bash
for v in 1 2 3 4 5; do
    echo "=== v$v 有效性检查 ==="
    taskset -c 2 sudo perf stat \
        -e cycles \
        -e ex_ret_instr \
        ./opcache_bench -v $v
done
```

**表 5-1 · 五个版本的指令数与源码预期对比**

| 版本 | STEP 次数 | n | `ex_ret_instr` | `ex_ret_instr / n` | 每 STEP 指令数 |
|---|---|---|---|---|---|
| v1 | 4 | 100,000,000 | 5,807,900,441 | 58.1 | **14.5** |
| v2 | 16 | 10,000,000 | 2,265,344,218 | 226.5 | **14.2** |
| v3 | 128 | 1,000,000 | 1,799,037,542 | 1799 | **14.1** |
| v4 | 512 | 300,000 | 2,156,276,274 | 7187 | **14.0** |
| v5 | 1024 | 100,000 | 1,438,520,224 | 14385 | **14.1** |

&emsp;&emsp;**每个 STEP 稳定在 14.0–14.5 条 macro-op**——三次乘法、三次加法、三次异或、三次移位，加上两次比较和跳转，以及 STEP 之间的寄存器搬运，合计 14 条。五个版本的 `ex_ret_instr / n` 严格按 STEP 次数线性增长，说明编译器没有做任何跨 STEP 合并——STEP 内的 `a ^= (a >> 5)` 是非线性运算，无法折叠成闭式解。

&emsp;&emsp;这次有效性检查比第 3、4 章更顺利，原因同样在 **STEP 宏的设计天然阻止了优化**：三条独立链（`a`、`b`、`c`）各有乘法→加法→异或→移位的依赖，每次 STEP 的输出又是下一次 STEP 的输入，跨 STEP 合并需要模拟非线性递推，生产级编译器不会这么做。**这是“基准设计”层面的胜利——与其事后用屏障阻止优化，不如从一开始就让代码结构本身不可优化**。

### 5.2 Op Cache 命中率的阶梯：容量边界在 1799 到 7187 条之间

&emsp;&emsp;Op Cache 是否接住循环体，唯一的直接判据是 `op_cache_miss / all_op_cache_accesses`：

```bash
for v in 1 2 3 4 5; do
    echo ""
    echo "=== v$v : op cache ==="
    taskset -c 2 sudo perf stat \
        -e cycles \
        -e ex_ret_instr \
        -e op_cache_hit_miss.all_op_cache_accesses \
        -e op_cache_hit_miss.op_cache_miss \
        ./opcache_bench -v $v
done
```

**表 5-2 · 五个版本的 Op Cache 命中率**

| 版本 | STEP 次数 | 每迭代 macro-op | `all_op_cache_accesses` | `op_cache_miss` | **miss 率** |
|---|---|---|---|---|---|
| v1 | 4 | 58 | 502,684,371 | 1,422,363 | **0.28%** |
| v2 | 16 | 227 | 151,666,276 | 823,348 | **0.54%** |
| v3 | 128 | 1799 | 112,529,370 | 831,026 | **0.74%** |
| v4 | 512 | 7187 | 133,964,124 | 112,858,536 | **84.2%** |
| v5 | 1024 | 14385 | 89,482,495 | 88,739,762 | **99.2%** |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_abb49e.svg)

&emsp;&emsp;**断层在 v3（1799 条）与 v4（7187 条）之间**——miss 率从 0.74% 跳到 84.2%。这个断层与 Zen 3 的 Op Cache 规格容量 **4096 个 uop** 吻合：1799 条 macro-op 展开后约 2000–2500 个 uop，落在 4096 以内；7187 条展开后约 8000–10000 个 uop，远超 4096。**规格数字是理论上限，实测给出的可用边界在 2000–7000 个 uop 之间**——这个区间包含分支密度、代码布局、替换策略等实际因素，是“在 5600X 上判断循环体是否过大”的实用判据。

&emsp;&emsp;**v5 的 99.2% 是“彻底溢出”的极端**：每迭代 14385 条 macro-op，Op Cache 几乎完全接不住，每次外层迭代都要从四宽解码器重新取指。但下面会看到，**“彻底溢出”并不等于“前端成为瓶颈”**——这是本章最重要的反直觉发现。

### 5.3 分发速率：Op Cache 溢出没有降低前端供给

&emsp;&emsp;如果 Op Cache 溢出导致前端供给不足，`de_dis_cops / cycles` 应该显著下降。数据：

```bash
for v in 1 2 3 4 5; do
    echo ""
    echo "=== v$v : dispatch ==="
    taskset -c 2 sudo perf stat \
        -e cycles \
        -e ex_ret_instr \
        -e de_dis_cops_from_decoder.disp_op_type.any_integer_dispatch \
        ./opcache_bench -v $v
done
```

**表 5-3 · 五个版本的分发速率**

| 版本 | STEP | 每迭代 macro-op | cycles | `de_dis_cops` | **`de_dis_cops / cycles`** |
|---|---|---|---|---|---|
| v1 | 4 | 58 | 2,009,901,499 | 6,511,779,420 | **3.24** |
| v2 | 16 | 227 | 806,246,904 | 2,577,781,807 | **3.20** |
| v3 | 128 | 1799 | 645,784,023 | 2,056,130,819 | **3.18** |
| v4 | 512 | 7187 | 774,346,986 | 2,465,611,852 | **3.18** |
| v5 | 1024 | 14385 | 516,986,790 | 1,644,997,293 | **3.18** |

&emsp;&emsp;**五个版本的 `de_dis_cops / cycles` 全部落在 3.18–3.24 之间**——Op Cache miss 率从 0.28% 跳到 99.2%，分发速率**没有任何变化**。每周期稳定分发约 3.2 个 cop，远低于六宽 dispatch 的上限，也低于四宽解码器的上限。**Op Cache 溢出没有导致前端供给不足**——因为前端的供给能力（Op Cache 的八宽或解码器的四宽）始终高于后端的消费能力（3.18 cop/cycle）。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_4d8963.svg)

&emsp;&emsp;**后端才是真正的瓶颈**。STEP 内三条独立链（`a`、`b`、`c`）各自的依赖链是乘法（3 cycles）→ 加法（1 cycle）→ 异或（1 cycle）→ 移位（1 cycle），约 6 cycles。三条链并行，每 6 cycles 可以退休约 18 条指令，即 **3.0 cop/cycle**——与实测的 3.18 吻合。**无论前端走 Op Cache 还是解码器，都能轻松跟上这个速率**。

### 5.4 为什么 Op Cache 溢出不等于 Frontend Bound

&emsp;&emsp;五个版本的数据合起来，给出了一个比“Op Cache 溢出就慢”更精确的判据：**Op Cache 溢出只有在“后端消费速率超过四宽解码器供给能力”时，才转化为 Frontend Bound**。

&emsp;&emsp;**反例（本章数据）**：STEP 宏的三条独立链让后端消费速率稳定在 3.18 cop/cycle，低于四宽解码器的供给上限（4 cop/cycle）。此时 Op Cache 溢出到 99.2%，解码器仍然跟得上后端的节奏——**Op Cache miss 率高，但前端不是瓶颈**。

&emsp;&emsp;**边界案例（第 4 章 v4）**：512 次 STEP 展开的循环体，每迭代约 7176 条 macro-op，Op Cache miss 84.1%。它的 `de_dis_cops / cycles = 3.19`，与本章的 3.18 只差 0.01；`ex_ret_instr / cycles = 1 / 0.358 ≈ 2.79`，也落在本章五个版本（2.78–2.89）的同一区间。**把两组数据放在一起，没有任何一个比值能把“前端瓶颈”和“后端受限”区分开**。第 4 章当初依据“Op Cache miss 84.1% + 分发率 3.19”两项同时偏高，把它归入前端桶；严格说，这只是**临界状态**：前端路径已经切到解码器，但分发速率尚未贴住四宽上限（3.19 < 4），后端仍有消费余量。第 4 章的表格现在把它标为“Frontend 临界”，理由就在这里。

&emsp;&emsp;这也解释了为什么需要一个阈值，而不是直接拿 3.18 和 3.19 比较：**阈值只负责“排除”，不负责“确认”**。低于 3.5 时，前端不可能成为首要瓶颈；只有分发速率继续向 4.0 逼近，才能确认 Op Cache 溢出已经转化为 Frontend Bound。

&emsp;&emsp;两类案例的区别不在 Op Cache miss 率（84% vs 99%），而在**后端消费速率是否已贴住解码器供给能力**。判据是 `de_dis_cops / cycles` 与 4 的关系：

**表 5-4 · 用 `de_dis_cops / cycles` 判断溢出是否转化为瓶颈**

| `de_dis_cops / cycles` | Op Cache 溢出时 | 判读 |
|---|---|---|
| **< 3.5** | Op Cache miss 高，但分发速率低于解码器上限 | **后端受限**——前端路径无关，优化应该转向后端 |
| **3.5–4.0** | Op Cache miss 高，分发速率接近解码器上限 | **前端接近瓶颈**——减少代码足迹可能有效 |
| **≈ 4.0** | Op Cache miss 高，分发速率被四宽解码器锁死 | **真正的 Frontend Bound**——必须减少代码足迹或增加每指令工作量 |

&emsp;&emsp;本章五个版本全部落在第一行（3.18–3.24）；第 4 章 v4 的 3.19 也在这一区间，只是当时基于单点数据，判得过于乐观。**Op Cache 溢出是“前端路径的切换”，不是“前端瓶颈的出现”**。这比“Op Cache 是 4096 uop 容量”更精细——**规格数字告诉你容量上限，`de_dis_cops / cycles` 告诉你溢出之后前端是否真的成为瓶颈**。

### 5.5 总结

&emsp;&emsp;综合第 4 章的临界案例与本章五个版本的数据，5600X 上 Frontend Bound 的下钻顺序可以精确化为三步。

**第一步：看 `op_cache_miss / all_op_cache_accesses`**。低于 5%，说明 Op Cache 接住了循环体，前端不是瓶颈；高于 20%，说明 Op Cache 部分或完全溢出，进入第二步。

**第二步：看 `de_dis_cops / cycles`**。低于 3.5，说明 Op Cache 虽已溢出，但分发速率低于解码器上限——**后端受限，不是 Frontend Bound**；达到 3.5 以上，说明解码器接近或成为瓶颈，进入第三步。

**第三步：看 `ex_ret_instr / n` 与循环体结构**。循环体很大（超过 2000 条 macro-op）且后端消费速率高时，前端优化应转向**减少代码足迹**（拆分热点函数、避免巨型循环展开）；循环体不大、`de_dis_cops / cycles` 却仍接近 4 时，瓶颈在**指令密度**（每条指令做的工作太少），优化应转向**用更宽的 SIMD 或 FMA 提高每指令工作量**。

&emsp;&emsp;**前两步判据合起来，可以避免一个常见误判**：看到 `op_cache_miss = 84%` 就认为“需要减少代码足迹”，但 `de_dis_cops / cycles = 3.18 < 3.5` 说明后端才是瓶颈——减少代码足迹不会改善性能，优化应转向后端执行资源。**Op Cache miss 高只是“前端走了慢路径”，不是“前端拖慢了整体”**；只有分发速率接近解码器上限时，前端的慢路径才会变成整体瓶颈。

&emsp;&emsp;**“Op Cache 溢出”是一个测量事实，“Frontend Bound”是一个瓶颈判断**，两者之间不能画等号；只有当后端消费速率超过解码器供给能力时，前者才转化为后者。**判断这个转化是否发生，只需要一个比值：`de_dis_cops / cycles` 与 4 的关系**——这是五个版本给出 3.18–3.24 的稳定数据之后，最有诊断价值的结论。


## 6. 分层缓存：32MB 统一 L3 与内存时钟域

&emsp;&emsp;**实验设计**。本章要回答的问题是：**5600X 的随机访存延迟在多大的工作集上，从“L3 供给”切换为“DRAM 延迟”**。规格书上的 32MB 统一 L3 是理论上限，实际可用边界受 set 冲突、Infinity Fabric 时钟域和物理页布局影响，只有测量能给出可用判据。

&emsp;&emsp;workload 用**随机指针追逐**：构建 `next` 数组，`next[perm[i]] = perm[(i+1) % m]`，其中 `perm` 是 `[0, m)` 的随机排列。从 `perm[0]` 开始，每次 `idx = next[idx]`，访问序列在物理地址空间里随机分布，硬件预取器完全无法预测。`m = sz_mb × 1024 × 1024 / 8` 个 `long`。迭代数默认等于 `m`，保证完整遍历一遍工作集。每个工作集先 warmup 遍历一遍，把数据从 DRAM 拉到缓存层级，再正式测量——测的是“热工作集”的访问延迟，不是冷缓存第一次访问的延迟。

```c
// random_access.cpp 核心函数
__attribute__((noinline))
long random_chase(const long *next, long n, long start) {
    long idx = start, sum = 0;
    for (long i = 0; i < n; i++) {
        idx = next[idx];          // 随机地址的依赖链 load
        sum += idx;
        __asm__ __volatile__("" : "+r"(idx), "+r"(sum) : : "memory");
    }
    return sum;
}
```

&emsp;&emsp;**测量条件**：EPP = performance，`taskset -c 2` 绑单核，每个事件独立一个 `-e`。四个工作集（16/24/32/40 MB）分别跑，采集 `cycles`、`ex_ret_instr`、`cache-references`、`cache-misses`、`ls_mab_alloc.load_store_allocations`。判据不是 `cache-misses` 的绝对值，而是 **`ns/access` 的台阶**——它直接反映每次随机访问的平均延迟，跨过 L3/DRAM 边界时会跳变。

```bash
echo "sz_mb,ns_per_access,cycles,cache_refs,cache_miss,cache_miss_pct,ls_mab"

for sz in $(seq 16 40); do
    OUT=$(taskset -c 2 sudo perf stat \
        -e cycles -e ex_ret_instr -e cache-references -e cache-misses \
        -e ls_mab_alloc.load_store_allocations \
        ./random_access -s $sz 2>&1)

    NS=$(echo "$OUT" | grep -oP '[\d.]+(?= ns/access)')
    CREF=$(echo "$OUT" | grep -oP '[\d,]+(?= +cache-references)' | tr -d ',')
    CMISS=$(echo "$OUT" | grep -oP '[\d,]+(?= +cache-misses)' | tr -d ',')
    MAB=$(echo "$OUT" | grep -oP '[\d,]+(?= +ls_mab_alloc)' | tr -d ',')

    PCT=$(awk "BEGIN {printf \"%.2f\", $CMISS * 100.0 / $CREF}")
    echo "$sz,$NS,$CREF,$CMISS,$PCT,$MAB"
done
```

**表 6-1 · 16–40 MB 逐 1 MB 扫描**

| 工作集 | ns/access | `cache-miss%` | `ls_mab / m` | 区间 |
|---|---|---|---|---|
| 16 MB | 14.32 | 40.91% | 6.24 | **L3 驻留** |
| 17 MB | 14.19 | 40.51% | 6.74 | L3 驻留 |
| 18 MB | 14.39 | 40.37% | 6.24 | L3 驻留 |
| 19 MB | 14.79 | 39.32% | 6.02 | L3 驻留 |
| 20 MB | 15.30 | 39.98% | 6.13 | L3 驻留 |
| 21 MB | 15.55 | 39.75% | 5.94 | L3 驻留 |
| 22 MB | 15.97 | 39.70% | 5.69 | L3 驻留 |
| **23 MB** | **18.16** | 39.54% | 5.70 | **过渡起点** |
| 24 MB | 19.50 | 39.46% | 5.62 | 过渡 |
| 25 MB | 18.80 | 39.49% | 5.82 | 过渡 |
| 26 MB | 23.84 | 39.29% | 5.59 | 过渡 |
| 27 MB | 21.36 | 39.23% | 5.44 | 过渡 |
| 28 MB | 21.54 | 39.02% | 5.33 | 过渡 |
| 29 MB | 22.58 | 39.01% | 5.22 | 过渡 |
| 30 MB | 23.54 | 38.74% | 5.19 | 过渡 |
| 31 MB | 27.05 | 38.98% | 5.18 | 过渡 |
| 32 MB | 26.22 | 38.77% | 5.12 | 过渡 |
| **33 MB** | **34.37** | 38.78% | 5.00 | **DRAM 起点** |
| 34 MB | 36.22 | 38.85% | 4.94 | DRAM |
| 35 MB | 37.28 | 38.95% | 4.93 | DRAM |
| 36 MB | 33.61 | 38.67% | 4.92 | DRAM（噪声） |
| 37 MB | 38.29 | 38.66% | 4.87 | DRAM |
| 38 MB | 37.26 | 39.01% | 4.84 | DRAM |
| 39 MB | 37.25 | 38.59% | 4.84 | DRAM |
| 40 MB | 40.84 | 38.52% | 4.72 | DRAM |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_da2231.svg)

&emsp;&emsp;**数据呈现清晰的三阶段**，但两个边界都与规格数字不同：

&emsp;&emsp;**第一阶段（16–22 MB）：L3 驻留**。`ns/access` 从 14.3 缓慢升到 16.0，涨幅 12%。这段上升并非“工作集变大导致更多 miss”，而是 **L3 set 冲突随工作集增长缓慢积累**——工作集已覆盖 L3 的物理容量，但每个 set 的占用还没到冲突阈值，多数访问仍在 L3 命中。

&emsp;&emsp;**第二阶段（23–32 MB）：过渡带**。`ns/access` 从 18.2 爬升到 26.2，涨幅 44%，中间还有抖动（25 MB 的 18.8 低于 24 MB 的 19.5，26 MB 的 23.8 又高于 27 MB 的 21.4）。**过渡带宽达 10 MB**——这是随机访问模式最反直觉的特征：工作集跨过 L3 容量后，不是“一刀切”地全部落到 DRAM，而是**一部分地址命中 L3、一部分 miss 到 DRAM，比例随工作集连续变化**。23 MB 时约 15% 的访问 miss 到 DRAM，32 MB 时约 50%，`ns/access` 是两种延迟的加权平均。

&emsp;&emsp;**第三阶段（33–40 MB）：DRAM 主导**。32→33 MB 出现整个扫描中最明显的单步跳变：26.22 → 34.37 ns，涨幅 **31%**。33 MB 之后，`ns/access` 在 34–41 ns 之间波动，涨幅收窄到 4% 以内。**33 MB 就是“工作集几乎完全溢出 L3”的实测边界**——比规格的 32 MB 晚 1 MB：规格容量是六核共享的 32 MB，但单核跑随机访问时其他核的 L3 占用可以忽略，单核可用容量接近 32 MB；再加上栈、代码、页表占用的内存，实际边界略高于 32 MB。

&emsp;&emsp;**两个反直觉的“平坦”，比台阶本身更有信息量**：

- **`cache-miss%` 从 40.91% 降到 38.52%，几乎不变**。工作集从 16 MB 涨到 40 MB，`ns/access` 翻了近 3 倍，`cache-misses / cache-references` 却只降了 2.4 个百分点。原因：Zen 3 的 `cache-misses` 主要统计 **L2 miss**，并不区分“L2 miss 但 L3 命中”和“L3 miss 到 DRAM”。16 MB 时所有访问都 miss L2（工作集远超 512 KB L2），但多数在 L3 命中；40 MB 时同样全部 miss L2，只是多数 miss 到 DRAM。**`cache-misses` 在这两个状态下几乎一样**——它只能告诉你“L2 没接住”，不能告诉你“L3 接住没有”。
- **`ls_mab_alloc / m` 从 6.24 降到 4.72，也几乎不变**。每次访问分配的 MAB 条目数不随工作集变化，变化的是条目在队列里待的时间：16 MB 时等 L3 返回（约 50 cycles），33 MB 时等 DRAM 返回（约 200 cycles）。**MAB 计数本身无法区分“等 L3”和“等 DRAM”，只有 `ns/access` 能区分**。

&emsp;&emsp;**同是 32 MB，两次测量差 15 ns**。正式扫描之前先做过一轮 16/24/32/40 MB 的四点预采样，32 MB 读到 41.32 ns/access；逐 1 MB 扫描中，同一工作集只有 26.22。两者相差 15 ns（58%），远大于扫描内部的抖动。原因有三：四点采样每次间隔几秒，温度与频率状态略有不同；`posix_memalign` 或 `std::vector` 每次进程启动分配不同的物理页，L3 set 映射不同；四点采样时 32 MB 是第三个点，前面 16/24 MB 已经预热了缓存，而 32 MB 的物理页恰好映射到冲突更严重的 set 上。**这个差异本身就证明了单点测量的不可靠**——只看 32 MB 一个点，你可能得出“32 MB 已全是 DRAM 延迟”或“32 MB 仍在 L3 过渡带”两个相反的结论，取决于那次运行落在物理页抽奖的哪一侧。**扫描整个 16–40 MB 范围，才能同时测出“边界在 22–33 MB 之间”这个宽区间和“33 MB 处 31% 跳变”这个精确位置**。

&emsp;&emsp;**对 Roofline 的衔接**：第 10 章的 Roofline 需要用“可持续内存带宽”作为屋顶，而扫描数据给出了这个带宽的适用边界——**工作集小于 22 MB 时，屋顶应该是 L3 供给带宽（不是 DRAM 带宽）；22–33 MB 是过渡带，Roofline 的“内存屋顶”在数学上不适用（延迟不是常量，而是工作集的函数）；超过 33 MB，DRAM 带宽才是真正的屋顶**。对一个 16 MB 工作集的内核画 Roofline 并用 DRAM 带宽做屋顶，会高估内存受限程度，把实际上仍受 L3 供给限制的内核误判为 DRAM 带宽受限。**“内存带宽”不是常量，而是工作集大小与缓存层级的函数**——这是第 1 章“对齐不是通用优化”在缓存层次上的延续。

## 7. 功耗与频率耦合：Curve Optimizer 改变测量基线

### 7.1 实验设置

&emsp;&emsp;**做**：在默认 BIOS 下跑 6 核 AVX2 FMA 负载（1e9 迭代 × 8 轮，约 8 秒），采集 `gflops` 和 `temp_c`。然后进 BIOS 启用 PBO + Curve Optimizer 全核 -20，重跑同样的命令。

```bash
clang++-22 -O2 -march=znver3 -o freq_co freq_co.cpp -pthread

# 默认 BIOS 基线
taskset -c 0-5 ./freq_co -t 6 -n 1000000000 -r 8 > baseline.csv

# 进 BIOS 开 PBO + CO -20，重启后重跑
taskset -c 0-5 ./freq_co -t 6 -n 1000000000 -r 8 > co20.csv
```

**表 7-1 · 默认 vs. CO -20（6 核 AVX2 FMA，1e9 迭代 × 8 轮）**

| 轮次 | 默认 gflops | 默认 temp | CO20 gflops | CO20 temp |
|---|---|---|---|---|
| 1 | 266.9 | 50.0°C | 267.5 | 39.2°C |
| 2 | 270.6 | 52.8°C | 271.5 | 42.1°C |
| 3 | 275.3 | 55.0°C | 275.7 | 44.6°C |
| 4 | 276.4 | 56.0°C | 288.8 | 45.5°C |
| 5 | 294.5 | 56.8°C | 304.3 | 46.5°C |
| 6 | 314.2 | 57.2°C | 328.3 | 47.2°C |
| 7 | 325.6 | 57.6°C | 333.5 | 47.9°C |
| 8 | **327.6** | **57.8°C** | **333.7** | **48.2°C** |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_b8063c.svg)

### 7.2 数据解读：温度降了 9.6°C，吞吐只涨 1.87%

&emsp;&emsp;**看数据**：CO 降压让温度从 57.8°C 降到 48.2°C（-9.6°C，-16.6%），稳态吞吐却只从 327.55 涨到 333.69 GFLOP/s（+1.87%），稳态时间从 879.256 ms 降到 863.070 ms（-1.84%）。两组 `gflops` 都在第 6–8 轮稳定在同一水平，对应频率都是 4649 MHz——**5600X 的 boost 上限**。温度降了 9.6°C，频率却已在天花板，省下的功耗预算无处可去。

&emsp;&emsp;两个容易误读的现象先说清楚：`freq_after` 里那些 1738 MHz 的读数是 `scaling_cur_freq` 在进程结束时的采样抖动，不是稳态频率；`gflops` 从 run 1 到 run 8 的爬升（267 → 333）也不是退档解除，而是 **Precision Boost 从空闲状态爬坡到满血 4.6 GHz 的过程**。两组数据都在 run 6–8 稳定在 4649 MHz / 333 GFLOP/s 附近，那才是真实的稳态。

&emsp;&emsp;**解释机制**：5600X 的 65W TDP / 约 76W PPT / 60A TDC / 88A EDC 限制，使功耗墙只在“高功耗 + 高温度”时生效。6 核 AVX2 FMA 的功耗约 55–65W，低于 76W PPT；温度 48–58°C，远低于 75–80°C 的退档阈值——**这个负载根本没有触及功耗墙**。CO 降压的收益被 boost 上限完全吸收，只留下温度的下降。

### 7.3 适用边界：CO 什么时候值得开

&emsp;&emsp;实验最初的设计意图，是展示“CO 降压 → 退档减少 → 吞吐提升 10%”的典型叙事。但数据给出了更精确的边界：**在 5600X 上，6 核 AVX2 FMA 负载在良好散热下根本触及不到功耗墙**。57.8°C 离 95°C 的 Tjmax 还有 37°C，离 75–80°C 的退档阈值还有 17–22°C；功耗约 55–65W，低于 76W PPT。**CO 的收益被 boost 上限完全吸收。**

&emsp;&emsp;要看到 CO 的 10% 收益，需要让负载真正触及功耗墙：

```bash
# 12 线程（SMT 兄弟也满载），功耗从 ~60W 推到 ~75-80W
taskset -c 0-11 ./freq_co -t 12 -n 1000000000 -r 10 > baseline_smt.csv
```

&emsp;&emsp;**预期**：12 线程时 PPT 触及 76W，温度升到 75–85°C，Precision Boost 退档到 4.0–4.2 GHz，`gflops` 从 333 降到 250 左右。此时 CO -20 的收益才会显现——温度降 8–10°C，频率回 0.2–0.3 GHz，吞吐提升 8–12%。

&emsp;&emsp;即便不跑 12 线程，当前数据也足以给出一个比“CO 有用”更精细的结论：

> **CO 的收益取决于散热条件和负载功耗**。在 5600X 上，6 核 AVX2 FMA（约 60W）在良好散热下温度只有 48–58°C，频率稳定在 boost 上限 4649 MHz，CO 降压 9.6°C 但吞吐只涨 1.87%——**收益被 boost 上限吸收**。只有当负载功耗触及 76W PPT、温度超过 75°C 时，CO 的降压才会转化为可观测的频率和吞吐提升。**判断要不要动 BIOS，先看温度：低于 60°C 时 CO 的收益在噪声边缘，高于 75°C 时才值得做**。

&emsp;&emsp;**“代码更快”和“系统更快”之间隔着一个功耗预算；预算没触及上限时，降压的收益会被频率上限吸收**——这是第 1 章“优化经验经常是错的”在功耗层面的延续。


## 8. SMT 与共享资源争抢

&emsp;&emsp;**实验设计**。5600X 是 6 核 12 线程，物理核 0–5 的主线程对应 cpu0–5，SMT 兄弟是 cpu6–11（cpu0↔cpu6、cpu1↔cpu7……）。本章要回答：**同一物理核上的两个 SMT 线程，在什么条件下能接近 2× 吞吐，在什么条件下会因共享资源互相拖慢**。关键变量是 workload 的**指令足迹是否驻留 Op Cache**：`small` 模式循环体约 60 条 macro-op，完全驻留；`large` 模式约 7000 条，彻底溢出。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_8c6398.svg)

&emsp;&emsp;workload 用三条独立的非线性链（`a*3+1`、`b*5+2`、`c*7+3`，三次异或、三次移位），STEP 宏天然阻止编译器做跨 STEP 代数合并。程序自己用 `pthread_setaffinity_np` 把每个线程绑到 `-c` 指定的 CPU，不需要 `taskset`。五种配置：单核 `0`、SMT 对 `0,6`、双物理核 `0,1`、六物理核 `0-5`、十二逻辑核 `0-11`。

```bash
clang++-22 -O2 -march=znver3 -o smt_bench smt_bench.cpp -pthread

for mode in small large; do
    for cfg in "0" "0,6" "0,1" "0-5" "0-11"; do
        echo "=== $mode / cpus=$cfg ==="
        ./smt_bench -c $cfg -m $mode
    done
done
```

**表 8-1 · 五个配置 × 两种模式的吞吐（GIPS = 十亿指令/秒）**

| 配置 | 线程 | small 时间 | small GIPS | small 效率 | large 时间 | large GIPS | large 效率 |
|---|---|---|---|---|---|---|---|
| cpu0 | 1 | 1004.4 ms | 11.95 | 1.00× | 1132.6 ms | 12.36 | 1.00× |
| **cpu0,6（SMT 对）** | 2 | 1396.1 ms | 17.19 | **1.44×** | 1657.6 ms | 16.89 | **1.37×** |
| **cpu0,1（双物理核）** | 2 | 1060.4 ms | 22.63 | **1.89×** | 1383.1 ms | 20.24 | **1.64×** |
| cpu0-5（6 物理核） | 6 | 1091.4 ms | 65.97 | 5.52× | 1472.5 ms | 57.05 | 4.62× |
| **cpu0-11（12 逻辑核）** | 12 | 1488.0 ms | 96.78 | **8.10×** | 1905.5 ms | 88.17 | **7.13×** |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_50e040.svg)

&emsp;&emsp;**SMT 对的效率差异只有 0.07**——small 1.44× vs large 1.37×。这个数字**推翻了“SMT 前端争抢是主瓶颈”的假设**：如果 Op Cache 溢出让两个 SMT 线程争抢同一个四宽解码器，large 的效率应该比 small 低 20–40%（1.44 → 0.9–1.15），实测只低 5%。原因很具体：**SMT 对的主要瓶颈是共享执行单元**，不是前端供给。small 模式下 Op Cache 供给八宽，large 模式下解码器供给四宽，但两个线程叠加后，执行单元（每核每周期约 3–4 条 macro-op 的退休能力）始终是共同上限。无论走 Op Cache 还是解码器，前端都能跟上——**前端不是瓶颈，执行单元才是**。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_97f5a0.svg)

&emsp;&emsp;**双物理核的效率差异是 0.25**——small 1.89× vs large 1.64×。这是整份数据中最反直觉的结果：**同一个 workload 跑在两个物理核上，large 的效率比 small 低 13%**，而两个核各有独立的 Op Cache 和解码器，本不该有前端争抢。差异来自**共享的 L3 代码供给带宽**：

- `small` 的循环体约 60 条 macro-op ≈ 240 字节，完全在 L1 I-Cache（32KB）内，两个核各取各的，不争抢。
- `large` 的循环体约 7000 条 macro-op ≈ 28KB，已接近 L1 I-Cache 的 32KB；加上函数体、栈和库代码，两个核同时跑时各自的 L1 I-Cache 可能装不下，需要从**共享的 L3** 补充取指。L3 的代码供给带宽被两个核平分，取指成了跨核瓶颈。

&emsp;&emsp;**12 线程的效率差异是 0.97**——small 8.10× vs large 7.13×。这是双物理核差异在满载时的放大：6 个物理核各跑 2 个 SMT 线程，large 模式下 12 个线程都要从 L3 补充取指，L3 带宽被 12 个线程平分，代码供给瓶颈更严重。**8.10× 接近 6 核 12 线程的实际上限（每核 SMT 效率 1.35× × 6 ≈ 8.1×），7.13× 低了 12%**——这 12% 就是 L3 代码供给带宽的代价。


&emsp;&emsp;**解释机制**：SMT 对的数据推翻了一个常见假设——**“SMT 效率低是因为两个线程争抢前端”**。实测的 1.44× 和 1.37× 接近“执行单元共享”的理论上限：每核每周期能退休约 3–4 条 macro-op，两个 SMT 线程平分后，总吞吐约为单线程的 1.4×。Op Cache 驻留（small）把前端供给从四宽提升到八宽，但**执行单元没有变**——所以效率只从 1.37× 提升到 1.44%，改善 5%。

&emsp;&emsp;真正被 Op Cache 驻留影响的是**跨核场景**：large 的循环体约 28KB，接近单核 L1 I-Cache 的 32KB；两个物理核同时跑时，各自的 L1 I-Cache 装不下完整循环体，需要从共享 L3 补充取指。L3 带宽被两核平分，这就是 1.89× → 1.64× 的来源。**在 5600X 上优化多线程负载时，减少代码足迹的收益不在 SMT 内部，而在跨核的 L3 代码供给**——这是本章与第 5 章“Op Cache 溢出只在分发速率接近解码器上限时才转化为瓶颈”的衔接：单线程时溢出可能被后端吸收，多线程时会转化为共享 L3 带宽的争抢。

&emsp;&emsp;因此，判断 5600X 上多线程负载的瓶颈，不能只看单一的 SMT 效率数字，要看**不同并行粒度下的效率差异**：

| 并行粒度 | small 效率 | large 效率 | 差异 | 判读 |
|---|---|---|---|---|
| SMT 对（同核 2 线程） | 1.44× | 1.37× | 0.07 | **执行单元共享**，前端不是主瓶颈 |
| 双物理核（跨核 2 线程） | 1.89× | 1.64× | **0.25** | **L3 代码供给争抢**，代码足迹是关键 |
| 12 线程（满载） | 8.10× | 7.13× | **0.97** | 跨核争抢在满载时放大 |

&emsp;&emsp;**优化优先级**：如果负载主要跑在 SMT 对上，减少代码足迹的收益只有 5%；如果负载跑在多个物理核上，收益是 15%（1.89 vs 1.64）。**代码足迹优化应该在跨核场景下做，而不是在 SMT 场景下做**。


## 9. 持续感知：eBPF 作为定向诊断的前置层

&emsp;&emsp;本章要回答的问题是：**eBPF 持续 profiling 的开销是否低到可以 7×24 运行，perf 定向采样的开销又是什么量级**。判据是同一个服务在三种条件下的吞吐差异：无 profiler（baseline）、eBPF `profile:hz:99` 持续采集、perf record 99Hz 采样。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_b9b710.svg)

&emsp;&emsp;测量条件严格固定：频率锁定在 3.7 GHz（`scaling_min_freq = scaling_max_freq = 3700000`），避开 Precision Boost 在 4.0–4.65 GHz 之间的动态调频；EPP = performance，温度控制在 44–46°C。service 零 IO，主线程只在结束时输出一行 `RESULT`，测量窗口内不写任何日志——每秒 `printf` + `fflush` 会让采样间隔在 0.9–1.5 秒之间漂移，产生伪波动。每线程使用独立计数器和独立 sink（各自 `alignas(64)` 独占 cacheline），消除跨核 atomic 争抢和 false sharing，因为这两者会被采样中断“缓解”，从而扭曲 profiler 的真实开销。

&emsp;&emsp;5 轮交错：`baseline → eBPF → baseline → perf → baseline`，共 15 次 baseline、5 次 eBPF、5 次 perf；每轮之间等温度降到 62°C 以下再开始。吞吐用跨运行变异衡量，不用单次运行内部的每秒采样——后者的采样间隔本身受 IO 和调度抖动影响。

&emsp;&emsp;service 用 `handler_light`（3 条指令）模拟高吞吐、短执行的请求处理，吞吐约 1.2 G req/s，每次请求约 0.83 ns。

```bash
clang++-22 -O2 -g -fno-omit-frame-pointer -march=znver3 -o service service.cpp -pthread

# 5 轮交错，窗口 8s，频率锁 3.7 GHz
for r in 1 2 3 4 5; do
    for cond in baseline ebpf baseline perf baseline; do
        case $cond in
        baseline) ./service -t 2 -w 8 | grep -oP 'throughput=\K\d+' >> /tmp/tp_baseline.txt ;;
        ebpf)     ./service -t 2 -w 8 > /tmp/svc_ebpf_$r.txt 2>/dev/null &
                  SVC=$!
                  sudo bpftrace -e 'profile:hz:99 /comm=="service"/ { @[ustack] = count(); }' \
                      -o /tmp/ebpf_$r.out 2>/dev/null &
                  BPID=$!
                  wait $SVC
                  sudo kill -INT $BPID 2>/dev/null; sleep 2
                  sudo kill -TERM $BPID 2>/dev/null; wait $BPID 2>/dev/null
                  grep -oP 'throughput=\K\d+' /tmp/svc_ebpf_$r.txt >> /tmp/tp_ebpf.txt ;;
        perf)     ./service -t 2 -w 8 > /tmp/svc_perf_$r.txt 2>/dev/null &
                  SVC=$!
                  sudo perf record -F 99 -g -C 0,1 -o /tmp/perf_$r.data -- sleep 8 > /dev/null 2>&1
                  wait $SVC
                  grep -oP 'throughput=\K\d+' /tmp/svc_perf_$r.txt >> /tmp/tp_perf.txt ;;
        esac
    done
done
```

**表 9-1 · 三种条件的吞吐（M req/s，5 轮交错）**

| 条件 | n | 中位数 | 均值 | 标准差 | 变异系数 | 温度 |
|---|---|---|---|---|---|---|
| baseline | 15 | 1200.78 | 1197.07 | 8.54 | **0.71%** | 45.5–46.1°C |
| eBPF | 5 | 1193.31 | 1189.09 | 12.33 | **1.04%** | 44.1–44.6°C |
| perf record | 5 | 1200.07 | 1189.31 | 17.73 | **1.49%** | 44.6–45.5°C |

**表 9-2 · 相对 baseline 的差异**

| 条件 | 中位数差异 | 均值差异 | 变异系数增量 |
|---|---|---|---|
| eBPF | **-0.62%** | -0.67% | +0.33 pp |
| perf record | **-0.06%** | -0.65% | +0.78 pp |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_5b6b30.svg)

&emsp;&emsp;**均值差异不可检测**。eBPF 与 baseline 的中位数差异是 -0.62%，perf 与 baseline 是 -0.06%，都落在 baseline 自身噪声带（cv 0.71%）与 profiler 自身噪声（1.04% / 1.49%）的合并区间内。按 99Hz × 2 CPU 的采样频率估算，每次中断约 3 μs 内核处理时间，每秒总暂停约 0.6 ms——**理论上限就是 0.06%**。perf 的 -0.06% 与这个算术下限精确吻合；eBPF 的 -0.62% 略大于理论下限，但 5 次测量的标准差是 1.04%，-0.62% 落在 1 个标准差内，不能判定为系统性开销。

&emsp;&emsp;**方差差异有信息**。三种条件的变异系数依次是 0.71% → 1.04% → 1.49%。均值相同而方差递增，说明 **profiler 的开销不仅体现在“平均暂停了多少时间”，还体现在“暂停发生在什么时候”**——每次采样中断都打断执行连续性，中断的随机性转化为吞吐的随机性。perf 的 cv 最高（1.49%），因为 `perf record` 每次中断都要写 `perf.data` 缓冲区、展开调用栈，持续时间比 eBPF 长；eBPF 的 cv 1.04% 接近 baseline，因为它在内核中聚合、只上报摘要，每次中断更短。**这个阶梯不是“eBPF 比 perf 好”，而是“两种 profiler 采样频率相同，但每次中断成本不同，成本差异体现在方差而不是均值上”**。在延迟敏感的场景下，1.49% 的 cv 比 0.06% 的均值损失更值得关注——**均值掩盖了 profiler 对吞吐稳定性的影响，方差暴露了它**。

&emsp;&emsp;**温度数据排除了频率混杂**。baseline 45.5–46.1°C，eBPF 44.1–44.6°C，perf 44.6–45.5°C，三种条件之间的温差在 1.5°C 以内，且 eBPF 和 perf 反而略低于 baseline——这排除了“profiler 让 CPU 更忙、频率随之上升”的混杂因素。频率锁定在 3.7 GHz 后，Precision Boost 的动态调频不再参与，吞吐差异只能归因于 profiler 本身。

&emsp;&emsp;**结论：持续感知优先用 eBPF，定向诊断才升级到 perf**。两者的均值开销都在噪声内，但方差开销不同（+0.33 pp vs +0.78 pp）。在 7×24 持续采集的场景下，方差开销意味着吞吐波动更大——eBPF“内核中聚合、只上报摘要”的设计让每次中断更短，方差也更接近 baseline。

&emsp;&emsp;**一个诚实的限制**：当前测量在 1.2 G req/s 的高吞吐下完成，`handler_light` 每次请求只有 3 条指令（约 0.83 ns），**调用一次 `process_request` 函数的开销（5–10 cycles）已占请求总时间的 50% 以上**。在这种负载下，profiler 的开销比例天然偏低——如果换到每次请求执行上千条指令的负载，profiler 的开销比例会更低，但这不代表“profiler 更便宜”，而是“被测对象更重，分母更大”。**任何 profiler 开销的百分比，都必须和被测负载的绝对执行时间一起报告**，否则“0.62%”和“0.06%”会被误读为 profiler 的固有属性，而不是“profiler 与负载的比值”。这也意味着：**在短请求、高吞吐的服务上，profiler 的开销更容易被检测到；在长请求、低吞吐的服务上，同样的开销会被更大的执行时间稀释**——选择工具时，被测负载的形态和工具的开销同样重要。


## 10. Roofline：把“该优化计算还是访存”变成可计算的判断

&emsp;&emsp;Roofline 模型把内核的算术强度（FLOP/Byte）作为横轴，可达到的浮点性能作为纵轴。算术强度低于拐点的内核落在斜坡上，是内存受限；高于拐点的内核落在平顶上，是计算受限。Roofline 的价值不在精确预测，而在**把优化方向的选择变成算术**——perf-book 把它放在“Performance Analysis Approaches”一章，正是因为它是一种应该在动手写代码之前完成的静态判断。

&emsp;&emsp;实验内核是一个向量化的 `a[i] = fma_chain(b[i], K)`：每个 float 元素读一次、写一次，做 K 次 8-wide AVX2 FMA。算术强度 `AI = K × 2 × 8 / (4 + 4) = 2K FLOP/Byte`，通过 `-k` 参数控制 K 从 1 到 128，覆盖从内存受限到计算受限的完整范围。

```c
// 内联汇编阻止编译器把 K 次 FMA 折叠成闭式解
__attribute__((noinline))
__m256 fma_chain(__m256 x, __m256 c, int K) {
    for (int k = 0; k < K; k++) {
        __asm__ __volatile__(
            "vfmadd231ps %[c], %[c], %[x]\n"
            : [x] "+x"(x) : [c] "x"(c)
        );
    }
    return x;
}

__attribute__((noinline))
void kernel(float* __restrict__ a, const float* __restrict__ b, long n, int K) {
    const __m256 c = _mm256_set1_ps(1.0000001f);
    for (long i = 0; i < n; i += 8) {
        __m256 x = _mm256_loadu_ps(&b[i]);
        x = fma_chain(x, c, K);
        _mm256_storeu_ps(&a[i], x);
    }
}
```

&emsp;&emsp;内联汇编是编译器的黑盒——它既不能把 K 条 FMA 折叠成 `x × c^K`，也不能把循环展开成固定次数。这一步是必须的：**不阻止折叠，Roofline 平顶的高度就由编译器决定，而不是由 CPU 决定**。测量条件：单核 `taskset -c 2`，工作集 64MB（超出 32MB L3，走 DRAM），频率锁定 3.7 GHz，内存为 DDR4-2667 双通道（`dmidecode` 确认 `Configured Memory Speed: 2667 MT/s`），FCLK/MCLK 1:1。

```bash
clang++-22 -O2 -march=znver3 -ffast-math -o roofline_bench roofline_bench.cpp -pthread
./roofline_sweep.sh | tee roofline.csv
```

**表 10-1 · 单核 Roofline 扫描（DDR4-2667，64MB 工作集）**

| K | AI（FLOP/B） | BW（GB/s） | GFLOP/s | 区域 |
|---|---|---|---|---|
| 1 | 0.25 | **17.74** | 4.43 | 斜坡 |
| 2 | 0.50 | **18.45** | 9.23 | 斜坡 |
| 4 | 1.00 | **18.44** | 18.44 | 斜坡结束 |
| 8 | 2.00 | 14.73 | 29.47 | 过渡带 |
| **16** | **4.00** | 9.35 | **37.40** | **峰值** |
| 32 | 8.00 | 3.96 | 31.69 | 平顶（下降） |
| 64 | 16.00 | 1.55 | 24.84 | 平顶（下降） |
| 128 | 32.00 | 0.64 | 20.32 | 平顶（下降） |

**表 10-2 · 多核对比（6 核，`taskset -c 0-5`）**

| 配置 | K | BW（GB/s） | GFLOP/s | 相对单核 |
|---|---|---|---|---|
| 单核 | 1 | 17.74 | 4.43 | — |
| **6 核** | 1 | **20.95** | 5.24 | **BW 1.18×** |
| 单核 | 64 | 1.55 | 24.84 | — |
| **6 核** | 64 | 9.13 | **146.08** | **GFLOP/s 5.88×** |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_98d96d.svg)

&emsp;&emsp;**斜坡在 AI ≈ 1 结束，峰值在 AI = 4**。K=4（AI=1）时 GFLOP/s = 18.44，恰好等于 18.45 × 1，完全落在斜坡上——这是内存受限的最后一个点。K=8（AI=2）时 GFLOP/s = 29.47，如果仍在斜坡应该等于 18.45 × 2 = 36.90，实测只有 29.47——**已经偏离斜坡 20%，但还没到峰值**，这就是 1–4 的过渡带：内存不再是唯一瓶颈，计算也还没饱和。K=16（AI=4）时 GFLOP/s = 37.40，是整条曲线的最高点；此时 BW 降到 9.35（斜坡上限 18.45 的一半），说明内存完全不是瓶颈，GFLOP/s 由单链 FMA 的并行度决定。

&emsp;&emsp;**Roofline 的理论 ridge point = 峰值 GFLOP/s ÷ 斜坡带宽 = 37.40 ÷ 18.45 ≈ 2.03 FLOP/Byte**——但这个数字只对**水平平顶**有意义。对这个单链内核，平顶是**拱形**的：K=32 降到 31.69，K=64 降到 24.84，K=128 降到 20.32。原因在单链 FMA 的固有限制——K 次 FMA 串行依赖，每次延迟 4 cycles，理论纯串行 GFLOP/s 是 18.4（4.6 GHz × 16 FLOP / 4 cycles）。K=16 时软件流水线（编译器自动把不同元素的 FMA 链重叠执行）把有效吞吐推到 37.40，是理论串行的 2 倍；K 越大，流水线窗口覆盖越不全，平均并行度下降，GFLOP/s 逐渐向 18.4 收敛。K=128 的 20.32 已接近纯串行的 18.4。

&emsp;&emsp;**这意味着 Roofline 的“平顶”是单链内核的真实上限，不是 Zen 3 的 AVX2 峰值 147 GFLOP/s**。要让平顶接近理论峰值，需要 8 条独立 FMA 链填满 2 个 FMA 端口 × 4 cycle 延迟——那是另一种内核设计，属于“计算优化”的范畴，不是 Roofline 要回答的问题。Roofline 要回答的是“**这个具体内核**该优化计算还是访存”——对这个单链内核，AI > 4 之后优化访存无效，但优化计算（增加并行链）还有接近 2 倍的空间。

&emsp;&emsp;**多核数据给出本章最有诊断价值的一对数字**：带宽几乎不扩展（K=1 时 6 核 20.95 GB/s，单核 17.74 GB/s，**1.18×**），计算几乎线性扩展（K=64 时 6 核 146.08 GFLOP/s，单核 24.84，**5.88×**）。原因很具体：DDR4-2667 双通道的理论峰值是 42.6 GB/s，单核 17.74 GB/s 已达 42% 效率；6 核同时跑时，内存子系统的实际供给上限是 20.95 GB/s——**单核时已接近饱和，多核无法线性扩展**。计算吞吐则受每核独立的 FMA 端口限制，6 核各有自己的执行单元，平顶高度可以近似 6 倍扩展（实测 5.88×，差额来自共享 L3 和调度开销）。

&emsp;&emsp;**这个扩展性差异直接决定优化优先级**：在 5600X 上优化内核的访存（预取、布局、压缩），单核和多核的收益都卡在 18–21 GB/s 的斜坡天花板下，无法靠增加核心数突破；优化内核的计算（SIMD 宽度、多链并行、FMA 融合），多核可以带来接近线性的收益。**“该优化计算还是访存”不是内核的固有属性，而是（内核 × 并行粒度 × 内存配置）的函数**——同一份代码，单核时可能落在斜坡上，6 核时如果工作集被分摊到各核的 L2、带宽需求下降，也可能滑向平顶。

&emsp;&emsp;**Roofline 在 5600X 上的三个关键数字，全部由实测给出，不引用规格表**：

| 参数 | 实测值 | 含义 |
|---|---|---|
| **单核斜坡带宽** | **18.45 GB/s** | DDR4-2667 双通道的 43% 效率，优化访存的天花板 |
| **单核峰值 GFLOP/s** | **37.40**（AI=4） | 单链 + 2 路软件流水线的实际上限，不是 AVX2 的 147 |
| **斜坡结束点** | **AI ≈ 1** | 低于此值严格内存受限；1–4 是过渡带；4 是峰值 |

&emsp;&emsp;**Roofline 的硬件上限必须标注三个条件**：内存配置（DDR4-2667 vs 3600 会改变斜坡高度）、工作集大小（超过 32MB L3 才走 DRAM，否则用 L3 带宽）、并行粒度（单核带宽 18 GB/s，6 核只有 21 GB/s，几乎不扩展）。**不标注这三个条件的 Roofline 图，只是一张示意图，不是判断工具**——这与第 6 章“32MB L3 改变内存受限起点”和第 1 章“对齐不是通用优化”是同一种纪律：**性能的边界由具体的硬件配置和工作负载形态共同决定，不由抽象的规格决定**。

&emsp;&emsp;**实践判断要区分斜坡、过渡带、峰值、缓降四个区间，而不是简单一句“低于拐点优化访存、高于拐点优化计算”**：

| AI | 判读 | 优化方向 |
|---|---|---|
| < 1 | 严格内存受限 | 减少内存流量（数据布局、压缩、计算复用），不堆 SIMD |
| 1–4 | 过渡带 | 两者都有边际收益，优先选改动小的那个 |
| 4–16 | 计算受限，**但单链并行度不足** | **增加独立链数**——这是“优化计算”在这一区间的真正含义 |
| > 16 | 计算受限，单链并行度已饱和 | 增加 SIMD 宽度或降低每元素 FMA 次数 |

&emsp;&emsp;**“AI > 4 就优化计算”这句话不够精确**——准确地说，AI 在 4–16 区间时，优化计算的方向是**增加并行链**（从 1 条到 8 条还有接近 2 倍的收益），而不是单纯堆 SIMD 宽度。单链内核在 AI=16 时只有 24.84 GFLOP/s，理论 8 链可以推到 70–90——**这就是为什么 Roofline 必须和内核结构一起看，不能只看 AI 一个数**。Roofline 是“静态判断”，它的价值在于把“该优化什么”从直觉变成算术；但这个算术的边界条件（内存配置、工作集、并行粒度、内核结构）必须和数字一起报告，否则“AI=4 是峰值”会变成另一个被误用的经验法则。

## 11. 四个在 5600X 上可直接复现的闭环实验

&emsp;&emsp;每个实验的结构：**基线测量 → 预测 → 改一行/几行 → 重新测量 → 用微架构机制解释差异**。

### 11.1 False Sharing：同一缓存行上的两个写者

**表 11-1 · 裸跑（每线程 1 亿次递增，cpu0+cpu1）**

| 版本 | 布局 | 时间 | ns/inc | result |
|---|---|---|---|---|
| **v1** | a、b 同一条 64B 行 | **675.8 ms** | **3.38** | a=100000000, b=100000000 |
| **v2** | a、b 各占一条 64B 行 | **40.6 ms** | **0.20** | a=100000000, b=100000000 |

**表 11-2 · PMU 事件（单次运行）**

| 指标 | v1（同缓存行） | v2（独占行） | 比值 |
|---|---|---|---|
| `cache-references` | **39,094,352** | **374,314** | **104×** |
| `cache-misses` | **37,496,977** | **97,861** | **383×** |
| miss rate | **95.9%** | 26.1% | — |
| `cycles` | **5,678,147,536** | **226,365,549** | **25.1×** |
| `ex_ret_instr` | 681,974,281 | 664,598,866 | **1.03×** |
| **CPI** | **8.33** | **0.341** | **24.4×** |
| **每递增引用数** | **0.195** | **0.0019** | **104×** |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_1a1f56.svg)

&emsp;&emsp;**时间差 16.7 倍**——v1 是 675.8 ms，v2 是 40.6 ms，`ns/inc` 从 3.38 降到 0.20。同样的循环、同样的指令、同样的线程数，只因两个 `long` 是否落在同一条缓存行上，吞吐差了 16.7 倍。`ex_ret_instr` 几乎相同（681M vs 664M），但 `cache-references` 差 104 倍（3909 万 vs 37.4 万），`cache-misses` 差 383 倍（3750 万 vs 9.8 万），CPI 从 8.33 降到 0.341——**所有差异都在“等待”，不在“计算”**。miss rate 不能直接用于诊断：v1 的 95.9% 和 v2 的 26.1% 看起来都“不健康”，但 v2 的 9.8 万次 miss 落在 L2/L3，代价几乎为零；v1 的 3750 万次全部是跨核 RFO，每次都要迁移整条缓存行——**同一个百分比，绝对代价差 383 倍**。

&emsp;&emsp;缓存一致性协议以缓存行为单位。核心 0 写 `a` 时，核心 1 中该行被置为 Invalid；核心 1 写 `b` 时，必须重新获取整行所有权——**即使 `a`、`b` 只差 8 字节、语义上完全无关，一致性流量仍以 64 字节为粒度发生**。`alignas(64)` 把两者分到独立的缓存行，两个核心各持有自己的行，不再触发 RFO。代价是每个计数器浪费 56 字节，换来 16.7 倍的时间差——**用 56 字节换 635 毫秒**。

&emsp;&emsp;两个核心在 MESI 状态机上反复拉锯，每次运行的碰撞节奏都不同。v2 的时间序列方差极小（40.6 ms 附近波动小于 0.1%），因为两个核心独立推进，没有共享状态机。**方差本身就是 false sharing 的指纹**。

### 11.2 Op Cache 驻留率：代码足迹对 Frontend Bound 的影响

&emsp;&emsp;用同一套 STEP 宏，只改变循环体内的展开次数：A 版本展开 4 次（每迭代约 58 条指令），B 版本展开 512 次（约 7180 条），其余条件完全相同。实测数据：

**表 11-3 · 两个代码足迹的指令数与 Op Cache 行为**

| 版本 | STEP | n | `ex_ret_instr` | instr/iter | `all_op_cache_accesses` | `op_cache_miss` | **miss rate** |
|---|---|---|---|---|---|---|---|
| A | 4 | 1e8 | 5,808,200,039 | **58.08** | 403,077,498 | 1,761,517 | **0.44%** |
| B | 512 | 1e6 | 7,180,503,251 | **7180.5** | 444,652,982 | 376,375,938 | **84.6%** |

**表 11-4 · 两个代码足迹的分发速率**

| 版本 | cycles | `de_dis_cops` | **`de_dis_cops / cycles`** |
|---|---|---|---|
| A | 2,016,084,632 | 6,513,038,117 | **3.23** |
| B | 2,645,103,862 | 8,208,863,823 | **3.10** |

&emsp;&emsp;**`ex_ret_instr / n` 严格按 STEP 次数成比例**——A 是 58.08，B 是 7180.5，比值 123.6，与 STEP 次数比（512/4 = 128）差 3.4%。差异来自循环控制指令的固定开销（每次迭代 3–4 条）在 A 中占比更高。**两个版本的指令数都符合源码预期，没有跨 STEP 折叠**。

&emsp;&emsp;**Op Cache miss 从 0.44% 跳到 84.6%**——A 的 58 条 macro-op 完全在 Zen 3 的 4096 uop Op Cache 内，403M 次访问中只有 1.76M 次 miss；B 的 7180 条远超 4096 uop，444M 次访问中有 376M 次 miss。**这就是 Op Cache 容量边界的直接读数：58 条完全接住，7180 条彻底溢出**。

&emsp;&emsp;**但 `de_dis_cops / cycles` 几乎不变：3.23 vs 3.10**——A 走八宽 Op Cache 路径，B 走四宽解码器路径，两条路径的供给能力差一倍，分发速率却几乎相同。原因在后端：STEP 宏内三条独立链各自的依赖链约 6 cycles，每 6 cycles 退休约 18 条指令，即 **3.0 cop/cycle**——后端的消费速率始终低于解码器的四宽上限，**Op Cache 溢出没有转化为前端供给不足**。

&emsp;&emsp;**这就是 11.2 与第 5 章共同的核心判据：Op Cache 溢出只是“前端走了慢路径”，不等于“前端成为瓶颈”**。判据是 `de_dis_cops / cycles` 与 4 的关系：低于 3.5 说明后端受限（前端路径切换无关）；3.5–4.0 说明前端接近瓶颈；接近 4.0 说明分发被四宽解码器锁死，是真正的 Frontend Bound。**A 和 B 都落在第一行（3.23 / 3.10）——看到 `op_cache_miss = 84.6%` 就认为“需要减少代码足迹”是错的**，因为后端才是瓶颈，减少代码足迹不会改善性能。

&emsp;&emsp;**“前端慢”在 5600X 上不等于“分支预测差”或“取指带宽不够”，它首先等于“Op Cache 没接住”**——但“没接住”只是路径切换，要转化为真正的瓶颈，还需要后端的消费速率超过解码器的供给能力。**这就是为什么 11.2 的判据不是单一的 `op_cache_miss%`，而是两个比值的组合：`op_cache_miss / all_accesses` 告诉你“走了哪条路径”，`de_dis_cops / cycles` 告诉你“这条路径是否成为瓶颈”**。

### 11.3 分支排序：Bad Speculation 是输入顺序的函数

&emsp;&emsp;通过随机排列改变访问顺序，模拟分支预测频繁失败与频繁成功的对比：

**表 11-5 · 随机顺序 vs 排序后（n = 16M，perf 只统计 `branch_sum`）**

| 版本 | `ex_ret_brn` | `ex_ret_brn_misp` | **misp rate** | `cycles` | `ex_ret_instr` | CPI |
|---|---|---|---|---|---|---|
| 随机 | 65,087,869 | 10,758,908 | **16.5%** | 457,047,346 | 238,700,679 | **1.91** |
| 排序 | 64,594,606 | 2,349,536 | **3.64%** | 265,873,630 | 236,455,558 | **1.12** |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_118afc.svg)

&emsp;&emsp;**`ex_ret_brn` 两版本几乎相同**（65.09M vs 64.59M，差 0.8%）——排序不改变分支数量，只改变分支方向序列。每元素约 3.9 个分支（一个数据分支加循环控制），与源码预期一致。**这是比较有效的前提：两个版本的分支数相同，差异只能来自误预测**。

&emsp;&emsp;**误预测率从 16.5% 降到 3.64%，差 4.5 倍**。随机顺序下 `v >= 128` 的方向由均匀随机数决定，50/50，预测器无法从历史学习；排序后所有 `<128` 在前、`>=128` 在后，方向形成两个长序列，预测器只在切换点附近错几次。**16.5% 而不是 50%**，是因为 Zen 3 的分支预测器（TAGE 类）对随机序列仍能捕捉一部分短程模式，加上循环控制分支高度可预测，拉低了整体误预测率——**但数据分支本身的误预测率仍远高于排序后的 3.64%**。

&emsp;&emsp;**CPI 从 1.91 降到 1.12，cycles 从 457M 降到 266M**。`ex_ret_instr` 两版本几乎相同（238.7M vs 236.5M，差 0.9%）——指令流没有变化，**42% 的 cycles 差异全部来自 Bad Speculation**：随机顺序下每次误预测冲刷流水线约 15–20 cycles，10.76M 次 × 17.5 ≈ 188M cycles，与实测的 191M cycles 差值（457M − 266M）精确吻合。

&emsp;&emsp;**判据：`ex_ret_brn_misp / ex_ret_brn`**。这个比值是 Bad Speculation 的直接读数——高于 10% 说明分支预测失败是主要瓶颈，低于 5% 说明分支预测没有拖后腿。**同一个内核、同一组数据分布、同样的分支数，只因为输入顺序不同，误预测率差 4.5 倍、cycles 差 1.7 倍**——这是 Bad Speculation 作为“输入顺序的函数”的直接证据。

&emsp;&emsp;**排序不是优化，是把“不可预测”变成“可预测”的输入预处理**。排序本身有成本（16M 元素约 0.5–1 秒），但如果内核要反复执行（数据库扫描、图像遍历），成本可以被多次执行摊销。**判断“排序是否值得”的算术是：排序成本 ÷ 单次内核节省的 cycles**——如果内核执行 100 次，排序的 0.5 秒摊销到每次 5 ms，而单次节省约 100 ms（457M − 266M cycles / 4.6 GHz），排序就是值得的。**这个判断不需要 profiler，只需要一个除法**——与第 10 章 Roofline 的“静态判断”是同一种纪律。

### 11.4 预取 + FCLK 同步：Memory Bound 的两个独立杠杆

&emsp;&emsp;基线是流式 triad（192MB 工作集，10 次重复），实验组用 `__builtin_prefetch` 提前 16 个迭代预取 `b`、`c`，其余条件相同。下表取 5 轮中位数：

**表 11-6 · baseline vs prefetch（16M float × 3 数组 = 192MB，10 次重复，单核）**

| 指标 | baseline（5 轮中位数） | prefetch（5 轮中位数） | 变化 |
|---|---|---|---|
| `cycles` | **740,529,264** | **868,044,046** | **+17.22%** |
| `cache-references` | 97,955,206 | 108,989,009 | +11.3% |
| `cache-misses` | 4,216,334 | 3,670,370 | **-13.0%** |
| `ls_mab_alloc` | **42,042,937** | **31,829,241** | **-24.3%** |


![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_0fc859.svg)

&emsp;&emsp;**预取降了 MAB 排队 24%、cache-misses 13%，cycles 反而升了 17%**——这与“预取改善性能”的直觉完全相反。原因很具体：**triad 是顺序访问，硬件预取器已经完美预测了访问模式，软件预取是冗余的**。

&emsp;&emsp;**`ls_mab_alloc` 从 42M 降到 32M（-24%）**——软件预取把 `b[i+16]` 和 `c[i+16]` 提前拉进 L1/L2，让后续的 demand load 在 L1D 命中，MAB 里排队的 demand miss 条目减少。**`cache-misses` 从 4.2M 降到 3.7M（-13%）**——同样的原因，部分 demand miss 被预取转成了 hit。

&emsp;&emsp;**但 `cache-references` 从 97.9M 升到 109.0M（+11.3%）**——**预取指令本身被计为 cache-reference**。每元素 2 条 `__builtin_prefetch`，16M 元素 × 10 次重复 = 3.2 亿次预取指令，实际只增加 1100 万次引用——因为预取命中已在缓存的数据时可能被合并或丢弃，不产生新引用。这部分额外引用没有带来新数据，只消耗了前端分发带宽和 MAB 条目。

&emsp;&emsp;**`cycles` 上升 17% 的直接原因**：预取指令本身（每元素 2 条）增加了指令数，而这些指令在**已经带宽饱和**的内核上不产生收益——DRAM 带宽上限约 18 GB/s，192MB × 10 次 = 1.92 GB，理论最短时间 107 ms，对应 cycles ≈ 490M。实测 baseline 740M cycles（约 161 ms），已经是带宽饱和状态（有效带宽 11.9 GB/s，低于上限，但受 MAB 排队和预取效率限制）。**预取无法突破带宽上限，只是把请求更早推进队列**——队列长度没有缩短（`ls_mab_alloc` 降了，但每个条目的平均等待时间没变），而预取指令本身消耗了额外的前端带宽和执行槽位。

&emsp;&emsp;**判据：预取有没有降低 `cycles`**。如果 `cycles` 降了 > 5%，说明预取解决了延迟问题（内核尚未带宽饱和）；如果 `cycles` 不变或上升，说明预取没有帮助——**`ls_mab_alloc` 降低只是“请求更早进入队列”，不是“队列更短”**。这个实验里 `cycles` 上升 17%，是预取的负面效果：**多余的指令开销没有被延迟节省抵消**。

&emsp;&emsp;**软件预取的适用边界**：只在**硬件预取器失效**的场景有效——随机指针追逐（第 4 章 v2）、不规则访问模式、跨页边界的大步长。**对顺序流式访问（triad、STREAM、memcpy），硬件预取器已经做到最优，软件预取只会增加开销**。这与第 4.4 节“预取在什么条件下无效”的结论一致，但在 5600X 上更具体：**单核 DDR4-2667 的带宽上限 18 GB/s 在 192MB 工作集下已经饱和，任何“加快字节传输”的优化都无效，只有“减少字节数”有效**。

&emsp;&emsp;**FCLK 同步的杠杆在这个实验里无法单独测试**——当前配置是 DDR4-2667 + FCLK 1333（1:1），要测分频的影响，需要在 BIOS 里把内存频率调到 3600 或 4000，让 FCLK 被迫分频（UCLK 2:1），再重跑同样的 triad。**如果分频后 `cycles` 上升超过 10%，说明 FCLK 同步是比预取更强的杠杆**；如果 `cycles` 不变，说明 DDR4-2667 的带宽已经低于分频后的有效带宽，FCLK 不是瓶颈。这个对比需要两次 BIOS 重启，成本较高，但信息量比预取实验更大——**预取是代码层面的微调，FCLK 是内存子系统的根本配置，两者的杠杆量级不同**。

&emsp;&emsp;**当前的 baseline vs prefetch 数据已经给出一个有效结论**：在 5600X 上，流式 triad 在 192MB 工作集下已经带宽饱和，软件预取降低了 MAB 排队却不降低 cycles，反而因额外指令增加 17% 的 cycles。**优化访存的优先级应该从“加预取”转向“减少内存流量”**（数据压缩、计算复用、分块让工作集落进 L3），因为带宽已经饱和，预取只是在饱和的管道里重新排列请求。这与第 10 章 Roofline 的结论一致：AI < 1 的内核严格内存受限，优化方向是**减少字节数**，不是**加快字节传输**。


## 12. 指令集扩展：从 AVX2 到 AVX-512 与 VNNI 的测量边界

&emsp;&emsp;第 10 章的 Roofline 给出了一个基本判断：向量化的收益取决于内核是否处于计算受限区域。但在具体硬件上，“向量化”不是二元状态——它是**指令集宽度、数据通路宽度和降频行为**的联合函数。5600X 的 AVX2 是 256-bit 原生数据通路，不存在拆分成本，可以作为效率基线；Zen 4 的 AVX-512 则通过 **double-pumping 256-bit 数据通路**实现：512-bit FMA 被拆成两个 256-bit μop，每周期退休一个 512-bit FMA，峰值 FLOP 上限与 AVX2 相同。

&emsp;&emsp;内核是一个 FMA 密集的逐元素变换：`a[i] = fma_chain(b[i], K)`，每个元素做 K 次 FMA。三个版本分别用标量、AVX2（8-wide）、AVX-512（16-wide）实现，循环体内加 `asm` 屏障阻止编译器把 K 次 FMA 折叠成闭式解——第 10 章已经证明，不加屏障，`x = fma(x, c, c)` 会被化简成 `x × c^K`，GFLOP/s 虚高 5–10 倍。

```c
// 三个版本的 chain 函数，结构完全相同，只有向量宽度不同
__attribute__((noinline)) float scalar_chain(float x, float c, int K) {
    for (int k = 0; k < K; k++) {
        x = fmaf(x, c, c);
        __asm__ __volatile__("" : "+x"(x) : : "memory");
    }
    return x;
}

__attribute__((noinline)) __m256 avx2_chain(__m256 x, __m256 c, int K) {
    for (int k = 0; k < K; k++) {
        x = _mm256_fmadd_ps(x, c, c);
        __asm__ __volatile__("" : "+x"(x) : : "memory");
    }
    return x;
}

#ifdef __AVX512F__
__attribute__((noinline)) __m512 avx512_chain(__m512 x, __m512 c, int K) {
    for (int k = 0; k < K; k++) {
        x = _mm512_fmadd_ps(x, c, c);
        __asm__ __volatile__("" : "+v"(x) : : "memory");
    }
    return x;
}
#else
// Zen 3 上回退到 AVX2——条件编译，5600X 的 v3 实际测的是 AVX2
void avx512_kernel(float* a, const float* b, long n, int K) { avx2_kernel(a, b, n, K); }
#endif
```

&emsp;&emsp;测量条件：`taskset -c 2` 绑单核，EPP = performance，K = 16，n = 1M 个 float（4MB，驻留 L2/L3，排除内存带宽干扰），每个版本跑一次 warmup + 一次正式测量。

```bash
clang++-22 -O2 -march=znver3 -o vec_bench vec_bench.cpp

for v in 1 2 3; do
    taskset -c 2 sudo perf stat -e cycles -e ex_ret_instr \
        ./vec_bench -v $v -k 16 -n 1048576
done
```

**表 12-1 · 三个版本的吞吐与指令数（5600X，K=16，n=1M）**

| 版本 | 实际指令集 | 时间 | **GFLOP/s** | `ex_ret_instr` | 相对标量 |
|---|---|---|---|---|---|
| v1 scalar | 标量 | 6.899 ms | **4.86** | 78,885,651 | 1.00× |
| v2 AVX2 | AVX2 8-wide | 1.191 ms | **28.17** | 23,708,209 | **GFLOP/s 5.79×，指令 0.30×** |
| v3 “AVX-512” | **AVX2**（5600X 回退） | 1.165 ms | **28.79** | 23,680,022 | 与 v2 相同 |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_6fbed7.svg)

&emsp;&emsp;**v3 与 v2 的时间、指令数、GFLOP/s 几乎完全相同**（1.165 vs 1.191 ms，23.68M vs 23.71M，28.79 vs 28.17）——这是条件编译正确工作的证据：`-march=znver3` 下 `__AVX512F__` 不定义，`avx512_kernel` 回退到 `avx2_kernel`。**在 5600X 上，“更宽的指令集是否转化为计算吞吐”这个问题无法回答——CPU 不支持 AVX-512**。要测 AVX-512 的 double-pumped 效应，必须在 Zen 4 上跑 v3。

&emsp;&emsp;**AVX2 的指令数是标量的 0.30×，GFLOP/s 是 5.79×**。指令数只降 3.3 倍而不是 8 倍，因为固定开销（屏障、加载、存储、循环控制）在标量版本中每元素重复一次，在 AVX2 中每 8 个元素重复一次，摊薄了 8-wide 的宽度收益。GFLOP/s 提升 5.79 倍，大于指令数降幅的倒数（1/0.30 = 3.3），因为 AVX2 不仅减少指令数，还减少了每指令的依赖等待：标量版本 16 条 FMA 串行、每条延迟 4 cycles，AVX2 把 8 条标量 FMA 合并成 1 条向量 FMA，同样的 64 cycles 关键路径上退休了 8 倍的工作。

> 5600X 不支持 AVX-512，因此本章只能给出 AVX2 的效率基线，无法测得更宽指令集的收益。

&emsp;&emsp;**5600X 的基线意义**：AVX2 每周期有效 FLOP = `28.17 GFLOP/s / 4.65 GHz ≈ 6.06 FLOP/cycle`，只有 8-wide × 2 FLOP = 16 FLOP/cycle 理论峰值的 38%。剩余 62% 的差距来自单链 FMA 的串行延迟（16 次 × 4 cycles = 64 cycles/向量）以及内存与循环开销。


## 13. 安全缓解措施：当“安全”成为微架构实验的隐藏变量

&emsp;&emsp;第 2 章讨论了频率和热状态作为测量噪声源。在现代 CPU 上，还有一个同等量级的变量：**操作系统和微码为抵御瞬态执行攻击而启用的缓解措施**。这些措施直接改变分支预测和推测执行的行为，从而改变 Top-Down 的分布。

&emsp;&emsp;在 5600X 上，`spectre_v2` 的缓解措施是 **Retpolines**（外加 IBPB conditional、IBRS_FW、STIBP always-on、RSB filling），`spec_store_bypass` 显示为空（未启用 SSB 缓解），`meltdown` 为 Not affected（AMD 不受影响，PTI 默认关闭）。**这意味着本章能测到的是 retpoline 对间接分支的影响，测不到 PTI 的代价**——PTI 的差异只在 Intel 上显现。

```bash
# 检查缓解状态
for f in /sys/devices/system/cpu/vulnerabilities/*; do
    printf "%-25s : %s\n" "$(basename $f)" "$(cat $f)"
done
```


&emsp;&emsp;workload 用 16 个不同目标函数的函数指针数组，由 LCG 生成随机索引并调用 `targets[idx](sum)`。三个版本：v1 纯间接调用；v2 每 4096 次加一个系统调用；v3 每 16 次加一个系统调用。

```c
__attribute__((noinline))
long indirect_loop(long n, unsigned seed, int syscall_every) {
    long sum = 0;
    unsigned x = seed;
    for (long i = 0; i < n; i++) {
        x = x * 1103515245u + 12345u;
        int idx = (x >> 16) & 15;
        sum += targets[idx](sum);
        __asm__ __volatile__("" : "+r"(sum), "+r"(x) : : "memory");
        if (syscall_every > 0 && (i % syscall_every) == 0)
            syscall(SYS_getpid);
    }
    return sum;
}
```

**表 13-1 · 三个版本的间接分支行为（n = 10M，缓解措施开启）**

| 版本 | `cycles` | `ex_ret_instr` | `ex_ret_brn` | `ex_ret_brn_misp` | `ex_ret_brn_ind_misp` | **ind_misp / 迭代** |
|---|---|---|---|---|---|---|
| v1 纯间接 | 300.4M | 202.8M | 40.8M | 9.45M | **9.38M** | **93.8%** |
| v2 + 1 sc/4096 | 314.0M | 283.8M | 61.0M | 9.48M | **9.38M** | **93.8%** |
| v3 + 1 sc/16 | 1112.7M | 518.6M | 120.6M | 16.85M | **9.22M** | **92.2%** |

<svg width="760" height="280" viewBox="0 0 760 280" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica,Arial,sans-serif">
  <rect x="8" y="8" width="744" height="264" fill="#fafafa" stroke="#ccc" rx="6"/>
  <text x="30" y="34" font-size="13" font-weight="bold" fill="#333">Retpoline：93.8% 的间接分支误预测，系统调用不改变这个数字</text>
  <line x1="80" y1="225" x2="720" y2="225" stroke="#333"/>
  <line x1="80" y1="225" x2="80" y2="55" stroke="#333"/>
  <text x="380" y="262" font-size="10" fill="#555">版本 →</text>
  <text x="22" y="140" font-size="10" fill="#555" transform="rotate(-90 22,140)">比值</text>

  <line x1="80" y1="150" x2="720" y2="150" stroke="#999" stroke-dasharray="3,3"/>
  <text x="685" y="146" font-size="9" fill="#999">100%</text>

  <g transform="translate(150,0)">
    <rect x="0" y="155" width="80" height="70" fill="#c62828"/>
    <text x="12" y="149" font-size="11" fill="#c62828" font-weight="bold">93.8%</text>
    <text x="20" y="248" font-size="10" fill="#333" font-weight="bold">v1 纯间接</text>
    <text x="15" y="263" font-size="9" fill="#666">300M cycles</text>
  </g>
  <g transform="translate(320,0)">
    <rect x="0" y="155" width="80" height="70" fill="#c62828"/>
    <text x="12" y="149" font-size="11" fill="#c62828" font-weight="bold">93.8%</text>
    <text x="0" y="248" font-size="10" fill="#333" font-weight="bold">v2 +sc/4096</text>
    <text x="15" y="263" font-size="9" fill="#666">314M cycles</text>
  </g>
  <g transform="translate(490,0)">
    <rect x="0" y="156" width="80" height="69" fill="#c62828"/>
    <text x="12" y="150" font-size="11" fill="#c62828" font-weight="bold">92.2%</text>
    <text x="5" y="248" font-size="10" fill="#333" font-weight="bold">v3 +sc/16</text>
    <text x="10" y="263" font-size="9" fill="#666">1113M cycles</text>
  </g>
</svg>

&emsp;&emsp;**v1 的 9.38M 次间接分支误预测 / 10M 次迭代 = 93.8%**。16 个目标的随机选择让间接分支预测器无法学习——每次间接调用有约 1/3 的概率猜错是合理的，但 93.8% 不是预测器的问题：**retpoline 把间接 `call`/`jmp` 换成了 `lfence; jmp *%rax` 的等价序列，强制推测执行停在 `lfence` 之后**。在 Top-Down 视角下，每次间接分支都被当作“总是误预测”处理——`ex_ret_brn_ind_misp` 统计的是 retpoline thunk 的推测失败，不是预测器猜错了目标。

&emsp;&emsp;**v2 和 v3 的 `ex_ret_brn_ind_misp` 与 v1 几乎相同（9.38M / 9.22M）**——系统调用增加了 `ex_ret_brn`（v3 的 120.6M 是 v1 的 3 倍，系统调用路径里的分支更多），也增加了 `cycles`（v3 的 1113M 是 v1 的 3.7 倍），但**间接分支的误预测数不变**。retpoline 的成本是“每次间接调用固定付一次”，与系统调用次数无关——**这是缓解措施施加的固定成本，不是“系统调用导致的”**。

&emsp;&emsp;**v3 的系统调用代价可以单独算出来**：10M / 16 = 625K 次系统调用，cycles 从 300M 涨到 1113M，增量 813M，**每次系统调用约 1300 cycles**。这个数字在 5600X 上是 `SYS_getpid` 的纯内核进入/退出成本（AMD 不受 Meltdown 影响，PTI 默认关闭），不包含 PTI 的 CR3 切换和 TLB 失效。**Intel 上开启 PTI 时，同样的系统调用会额外增加 1000–2000 cycles**——这是本章在 5600X 上测不到的部分。

&emsp;&emsp;**关闭 retpoline 后，v1 的 `ex_ret_brn_ind_misp` 应降到约 6M（20% 左右）**——预测器可以重新学习 16 个目标的模式。这个下降不是“优化”，而是把“推测被禁用”恢复成“推测可以犯错”。**两者在 Top-Down 中都归入 Bad Speculation，来源却不同**：retpoline 施加的是**固定成本**（每次间接分支必付），预测器犯错是**可变成本**（取决于目标数和输入序列）。前者只能靠**减少间接分支数量**消除（虚函数改模板、回调改 switch、解释器改 computed goto），后者可以靠**改变输入顺序**缓解（第 11.3 节的分支排序）。**在 5600X 上做分支优化之前，先检查 `spectre_v2` 的状态**——如果默认启用了 retpoline，任何包含间接分支的 workload，其 Bad Speculation 基线都会被系统性抬高 93% 量级；这个数字不是程序能优化的，而是缓解措施施加的。


## 14. 优化技术选型：把诊断结果接到正确的技术上

&emsp;&emsp;第 4 章到第 13 章建立了一条诊断链路：从 PMU 事件到 Top-Down 路由，再到 Roofline 边界、指令集扩展与安全缓解等变量。但诊断本身不产生性能——**从“知道瓶颈在哪里”到“让瓶颈消失”之间，还隔着一层优化技术选型**。选型的依据只有一条：当前测量数据指向的微架构原因，与这项技术要解决的问题是否匹配。

### 14.1 前端优化：让 Op Cache 接住，让代码布局不浪费槽位

&emsp;&emsp;5600X 的前端是“四宽解码器 + 八宽 Op Cache → 六宽 dispatch”的双路径。前端优化的第一原则因此非常具体：**尽量让热点代码驻留在 Op Cache 中；做不到时，至少不要让代码布局本身制造额外的前端停顿**。

&emsp;&emsp;**做**：对同一个循环，分别用 `-O2` 和 `-O2 -march=znver3` 编译，采集 `op_cache_hit_miss`、`de_dis_cops_from_decoder` 和 `ic_tag_hit_miss`。如果 `op_cache_hit` 比例高、`de_dis_cops_from_decoder` 不是主要来源，前端优化应转向**减少代码足迹**（避免巨型函数内联、控制热点函数的指令数量）。如果 `op_cache_miss` 高、`ic_tag_hit_miss` 也高，问题可能不在 Op Cache 容量，而在**代码布局导致 I-cache 行利用率下降**——这时基本块对齐、把循环体移进单条 I-cache 行、函数分组才开始有意义。

&emsp;&emsp;**5600X 上的边界**：热点代码足迹超过 Op Cache 容量时，前端瓶颈的形态会从“分支预测差”切换为“供给带宽不足”。此时更有效的手段可能是**函数拆分**——把冷路径从热点函数中剥离，让热路径独占 Op Cache。

### 14.2 后端优化：Core Bound 与 Memory Bound 的分野

&emsp;&emsp;Core Bound 的优化对象是执行端口、依赖链和指令延迟；Memory Bound 的优化对象是缓存层次、预取和带宽。

&emsp;&emsp;**做（Core Bound）**：对一个受依赖链限制的循环，采集 `ex_ret_ops` 和 `cycles`，然后手动展开循环、打断累加器依赖链，或把除法换成乘法倒数。如果展开后 `instructions` 基本不变而 `cycles` 下降，说明原循环的槽位被依赖链延迟阻塞；如果 `cycles` 不变，瓶颈可能在某个执行端口的吞吐上限。

&emsp;&emsp;**做（Memory Bound）**：`ls_mab_alloc` 是区分“延迟问题”和“带宽问题”的关键事件。它持续高位而实际 DRAM 带宽未饱和，瓶颈在 **MAB 排队**，此时加大预取深度可能反而加剧排队；它高位且带宽计数器接近上限，瓶颈在**带宽饱和**，预取和 Non-Temporal Store 的收益都会被压缩。

&emsp;&emsp;**5600X 上的边界**：32MB L3 把“内存受限”的起点推后了。工作集仍能装进 L3 时，过早使用 DRAM 级优化（如 Non-Temporal Store）是无效的；只有工作集明确超过 32MB，DRAM 带宽导向的优化才进入有效区间。

### 14.3 错误投机优化：降低 Bad Speculation 的“可避免部分”

&emsp;&emsp;**做**：用 `perf stat -e branches,branch-misses` 定位误预测率最高的区域，然后尝试：(a) 用查表替换分支；(b) 用算术或条件传送替换分支；(c) 用 PGO/BOLT 改善预测器的学习条件。如果 `branch-misses / branches` 显著下降、`cycles` 也同步下降，说明误预测确实是槽位消耗的来源；如果 `branch-misses` 下降而 `cycles` 不变，说明 Bad Speculation 在 Top-Down 中的占比本来就不高。

&emsp;&emsp;**5600X 上的边界**：Zen 3 的分支预测器对规则模式的学习能力较强。用查表替换分支的收益，取决于**查表本身是否引入新的缓存压力**——查表数组一旦超出 L1D 工作集，错误投机优化就可能把 Bad Speculation 的槽位节省换成 Memory Bound 的增加。同时，第 13 章提醒：先确认 Bad Speculation 中有多少是**缓解措施施加的固定成本**，再决定优化动作。

### 14.4 内存访问优化：数据布局是 Memory Bound 的“源头杠杆”

&emsp;&emsp;数据布局优化是唯一一类能在不改算法、不加指令的前提下，直接改变“每条缓存行承载多少有用字节”的技术。perf-book 第二版新增的**结构体字段重排**（field reordering）逻辑是：按访问频率和访问模式重排字段，让热字段紧凑地落在同一条缓存行内。

&emsp;&emsp;**做**：取一个遍历结构体数组、只使用少数字段的循环，先采集 `l1d_cache_misses` 和 `ls_mab_alloc`，再按访问频率把热字段前置并连续排列、冷字段后置。如果热字段与冷字段原本跨缓存行边界，重排后应观察到 `l1d_cache_misses` 和 `ls_mab_alloc` 同时下降。

&emsp;&emsp;**5600X 上的边界**：Zen 3 的 L1D 组相联结构和预取器行为会与字段排列方式交互。某些访问模式下，过于紧凑的字段排列可能让多个高频访问流映射到同一 L1D set，产生冲突缺失。重排之后，除了看 miss 率，还应检查冲突相关事件，确认优化没有把容量缺失换成冲突缺失。

### 14.5 多线程优化：伪共享只是起点，同步与扩展性是主体

&emsp;&emsp;第 11.1 节的 False Sharing 展示的只是缓存一致性协议在写共享粒度上的代价。多线程优化的完整版图还包括**同步开销、锁争用、线程数扩展性、任务调度**。

&emsp;&emsp;**做**：对一个包含共享计数器的多线程程序，分三步改动并测量：(a) 用 `alignas(64)` 消除伪共享；(b) 把全局原子计数器改为**每线程本地计数 + 最终归约**；(c) 用 `perf c2c` 确认是否还存在跨核缓存行弹跳。

```bash
perf c2c record -a ./multithread_bench
perf c2c report --stdio
```

&emsp;&emsp;**看数据**：伪共享消除后，`cache-misses` 会下降，但 `cycles` 可能仍然高——**原子操作本身的序列化开销**和**跨核缓存行所有权迁移**是伪共享之外的另一层成本。每线程本地计数把写操作转移到私有缓存行，归约阶段只发生一次跨核交互，应进一步降低 `cycles`。如果 `perf c2c` 仍报告高比例的 HITM 事件，说明共享写入的粒度问题没有完全解决。

&emsp;&emsp;**5600X 上的边界**：单 CCD、六核共享 32MB L3，跨核通信的延迟基线相对均匀。线程亲和性（`taskset` 绑定到同一 CCX 内的核心）对同步密集型负载的收益较为可预测。

### 14.6 编译器与构建配置：最便宜的优化层，也是最容易被忽略的测量变量

&emsp;&emsp;编译器优化不改变算法，却改变机器码的指令选择、内联决策、循环结构和分支布局。在 5600X 上，至少有三个变量值得单独测：目标架构标志（`-march=znver3`）、优化级别（`-O2`/`-O3`）和**基于剖析的优化（PGO）**。

```bash
gcc -O2 -march=znver3 workload.c -o w_O2_znver3
gcc -O3 -march=znver3 workload.c -o w_O3_znver3
gcc -O2 -march=znver3 -fprofile-generate workload.c && ./a.out && \
gcc -O2 -march=znver3 -fprofile-use workload.c -o w_pgo
```

&emsp;&emsp;**看数据**：`-march=znver3` 启用 AVX2/FMA 和 Zen 3 特定的调度模型。PGO 的收益通常体现在 **Bad Speculation 和 Frontend Bound 的下降**——它重新排列基本块，把热路径放在一起，减少 taken branch 和 I-cache 行的浪费。如果 PGO 后 `branch-misses` 下降、`cycles` 下降幅度却有限，说明分支误预测在原负载中本来就不是主导瓶颈。

&emsp;&emsp;**5600X 上的边界**：`-march=znver3` 要求 GCC 10.3+ 或相应版本的 Clang。另一个容易忽略的变量是**内联策略**：过度的 `always_inline` 会迅速推高热点代码的 Op Cache 足迹，把 Frontend Bound 从“解码器受限”推向“Op Cache 缺失”。

### 14.7 计算优化：SIMD 与算术强度，在 Roofline 的斜坡上做正确的事

&emsp;&emsp;计算优化（SIMD 向量化、FMA、减少慢速浮点）只应在 Roofline 判断为计算受限时进入视野。

&emsp;&emsp;**做**：对标量浮点循环，先测 `-O3 -march=znver3 -ffast-math` 的自动向量化版本，再测手写 AVX2 intrinsics 版本，采集 `instructions` 和 `cycles`，计算 IPC 与实际 FLOP/cycle。如果手写 AVX2 版本的 `instructions` 显著下降，`cycles` 的降幅却不成比例，说明循环已处在内存受限的斜坡上；如果 `cycles` 与 `instructions` 同步下降，且 IPC 接近执行宽度上限，说明计算优化命中了 Core Bound。

&emsp;&emsp;**5600X 上的边界**：Zen 3 的 AVX2 是 256-bit 原生执行，不存在 AVX 降频问题。但 **六核共享 32MB L3，意味着多线程 SIMD 负载很容易把带宽推向饱和**：在单核上有效的向量化内核，六核同时运行时可能从 Core Bound 切换为 Memory Bound，此时继续增加 SIMD 宽度或展开因子，收益会迅速递减。计算优化的前提，是**在目标线程数下重新做一次 Roofline 判断**。同时，第 12 章提醒：更宽的指令集（AVX-512/VNNI）只抬高计算屋顶，不改变内存屋顶；内核落在内存斜坡上时，指令集扩展的收益不可见。


## 15. 性能优化是“假设—测量—修正”的循环，不是技术清单

&emsp;&emsp;回到开篇的对齐实验，用 5600X 的完整工具链重走一遍：

1. **先锁定测量条件**：确认 FCLK/UCLK/MCLK 的同步状态，记录 PBO 与 Curve Optimizer 配置，让机器进入热稳态（第 2、7 章）。
2. **记录缓解措施状态**：检查 Spectre 缓解措施的启用情况，作为解释 Bad Speculation 的上下文（第 13 章）。
3. **采集原始事件组**：`op_cache_hit_miss`、`de_dis_cops_from_decoder`、`ls_mab_alloc`、`dispatch_stalls`，而不是直接依赖 Intel 式的 `--topdown` 自动输出（第 3、4 章）。
4. **前端嫌疑高**：先看 Op Cache 命中率，再看四宽解码器压力，最后考虑分支预测（第 5、14.1 节）。
5. **Memory Bound 嫌疑高**：先用 32MB L3 容量判断工作集位置，再用 FCLK 同步状态判断延迟还是带宽；预取和内存控制器配置是两个独立杠杆（第 6、10、11.4 节）。
6. **工作负载是 LLM 推理**：decode 阶段通常落在内存斜坡上；先用低精度量化（INT8/INT4）抬高内存屋顶，再考虑权重布局与 L3 驻留，最后才看 SIMD 宽度（第 10、12 章）。
7. **做最小改动，先预测哪个事件会变、变成什么方向，再测量**。预测失败时的信息量通常更大——它暴露了你对微架构模型的某个假设需要修正。

&emsp;&emsp;perf-book 的方法论收敛于一句话：**优化不是在代码里堆技术，而是在测量数据和微架构模型之间建立可验证的对应关系**。Top-Down 提供槽位级别的路由，Roofline 提供算术边界，PMU 提供可重复的观测；5600X 的具体实现——Op Cache/解码器双路径、32MB 统一 L3、Infinity Fabric 时钟域、Precision Boost 功耗行为——决定了这条闭环在这台机器上的锚点和判据。指令集扩展、安全缓解措施和工作负载形态，则是三个必须在测量设计中显式控制的变量：它们不改变方法论，却改变数据的解读方式。

&emsp;&emsp;每一次改动都应先预期“哪个指标会变、变成什么方向”，再用下一次测量证实或否定。修正假设、重新测量、更新模型——这就是性能优化工作本身的形态。