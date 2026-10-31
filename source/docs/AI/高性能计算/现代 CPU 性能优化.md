# 现代 CPU 性能优化

**——以 AMD Ryzen 5 5600X 为实验平台的 PMU、Top-Down 与优化技术实践手册**


## 1. 为什么优化经验经常是错的

&emsp;&emsp;先做两个实验。

### 1.1 分配对齐

&emsp;&emsp;**实验 A**：写一个求和循环，累加到一个 `long` 局部变量中并打印。保持代码完全相同，只改变数组起始地址的对齐方式——一个用 `malloc` 默认分配（通常 16 字节对齐），另一个用 `posix_memalign` 强制 64 字节对齐。

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

&emsp;&emsp;直觉上，64 字节对齐应该更快——向量化后的 32B load 不会跨缓存行。在 5600X 上用 `clang++-22 -O2 -march=znver3` 编译，三种工作集大小各跑一轮，得到的数据却比直觉复杂得多：

**表 1 · 单次测量（首次运行）**

| 场景 | 工作集 | v1 median | v2 median | 表面差异 |
|---|---|---|---|---|
| L2 驻留 | 512 KB × 10000 扫描 | 55.14 ms | 64.66 ms | **v2 慢 17%** |
| L3 驻留 | 2 MB × 2000 扫描 | 48.00 ms | 48.17 ms | 几乎相同 |
| DRAM | 256 MB × 1 扫描 | 10.75 ms | 10.33 ms | v2 快 4% |

&emsp;&emsp;L2 场景的 17% 差异看起来足以支撑“对齐有害”或“对齐有益”的任何叙事。但把同样的 v1/v2 交替跑三轮，结果变了：

**表 2 · 交替测量（每轮 v1 后 v2，共三轮）**

| 轮次 | v1 median | v2 median | 差异 |
|---|---|---|---|
| Round 1 | 55.62 ms | 56.54 ms | v1 快 1.6% |
| Round 2 | 55.74 ms | 56.55 ms | v1 快 1.5% |
| Round 3 | 55.07 ms | 49.72 ms | v2 快 9.7% |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261030_b944d9.svg)

&emsp;&emsp;**交替测量后，前两轮 v1/v2 的差异缩到 1.5%，第三轮 v2 反而快 9.7%。没有出现任何方向一致的系统性差异。** 单次测量看到的 17% 在交替测量里消失了——它不是对齐的效果，而是 **v2 首次运行时 CPU 尚未进入热稳态、Precision Boost 频率还在爬升**，以及 `posix_memalign` 分配到的那一组物理页恰好对 L2 set 冲突更不友好。第三轮的 9.7% 方向反转，是物理页抽奖的另一次结果。

&emsp;&emsp;这个实验的真正价值不在“对齐到底有没有用”，而在于它展示了**优化经验失效的两种典型方式**：
- **把混杂变量误读为优化效果**。单次测量的 17% 差异不是对齐造成的，而是“v2 先跑 + 频率未稳 + 物理页随机”三者叠加的结果。如果把这次单次测量当成结论，就会得出“对齐有害”或“对齐有益”的相反叙事——两者都是错的。
- **忽略微架构的并行化能力**。Zen 3 的 L1D 把跨 64B 行的 32B load 拆成两个 32B 操作后**并行发出**，16B 偏移完全在硬件并行化能吸收的范围内。只有跨 4K 页边界时 split load 才会产生真正昂贵的串行惩罚。**对齐不是“通用优化”，而是一个需要被测量验证的假设**——它在某些微架构上、某些访问模式下有效，在另一些条件下完全被 L2/L3/DRAM 的供给延迟掩盖。

### 1.2 缓存行填充

&emsp;&emsp;**实验 B**：两个线程各递增一个 `long` 计数器，不加填充，运行 1 亿次。v1 的 `a` 和 `b` 位于同一条 64B 缓存行内，v2 用 `alignas(64)` 把两者分隔到各自独占的缓存行。

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

&emsp;&emsp;用 `taskset -c 0,1` 绑定到两个不同物理核，`clang++-22 -O2 -march=znver3` 编译。这次的数据与实验 A 完全相反：**方向一致、幅度巨大、机制可解释**。

**表 3 · 裸跑 1 亿次/线程（每线程 1 亿，总计 2 亿次递增）**

| 指标 | v1（伪共享） | v2（填充） | 比值 |
|---|---|---|---|
| median | 653.108 ms | 23.894 ms | **27.3×** |
| min | 580.816 ms | 21.964 ms | 26.4× |
| 每次递增耗时 | 3.27 ns | 0.12 ns | — |
| 每秒递增数 | 306 M/s | 8.37 G/s | — |

&emsp;&emsp;`0.12 ns/inc` 对应 5600X 在 boost 频率下每周期约 1.8 次递增——两个核心并行推进各自的私有缓存行，每个核心每周期约 0.9 次 `inc [mem]`。**v2 的时间已经触及 Zen 3 的 L1D 吞吐上限**，再优化布局也压不下去。

<svg width="760" height="270" viewBox="0 0 760 270" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica,Arial,sans-serif">
  <rect x="8" y="8" width="744" height="254" fill="#fafafa" stroke="#ccc" rx="6"/>
  <text x="30" y="34" font-size="13" font-weight="bold" fill="#333">实验 B：伪共享把每次递增拖到 27 倍</text>
  <line x1="70" y1="225" x2="720" y2="225" stroke="#333"/>
  <line x1="70" y1="225" x2="70" y2="70" stroke="#333"/>
  <rect x="160" y="80" width="120" height="145" fill="#c62828"/>
  <rect x="420" y="219" width="120" height="6" fill="#2e7d32"/>
  <text x="180" y="72" font-size="13" font-weight="bold" fill="#c62828">653 ms</text>
  <text x="185" y="212" font-size="11" fill="#fff" font-weight="bold">3.27 ns/inc</text>
  <text x="440" y="212" font-size="11" fill="#2e7d32" font-weight="bold">23.9 ms</text>
  <text x="440" y="72" font-size="13" font-weight="bold" fill="#2e7d32">0.12 ns/inc</text>
  <text x="190" y="245" font-size="11" fill="#333" font-weight="bold">v1 伪共享</text>
  <text x="450" y="245" font-size="11" fill="#333" font-weight="bold">v2 填充</text>
  <line x1="290" y1="150" x2="410" y2="150" stroke="#888" stroke-width="2"/>
  <polygon points="410,150 400,144 400,156" fill="#888"/>
  <text x="310" y="140" font-size="12" font-weight="bold" fill="#888">27.3×</text>
  <text x="20" y="150" font-size="10" fill="#555" transform="rotate(-90 20,150)">时间</text>
</svg>

&emsp;&emsp;v1 的 5 次运行时间序列还揭示了一个细节：696.782 → 683.002 → 653.108 → 580.816 → 596.956，下降 16.7% 后又反弹。这不是噪声，是 **MESI 状态在收敛**。首次运行时缓存行在 Invalid 与 Modified 之间反复迁移；后续运行中两个核心的访问节奏逐渐稳定，一致性协议的开销落入更规则的模式。**这种"越跑越快"的模式本身就是伪共享的一个指纹**——真实的计算瓶颈不会因为重复运行而改变，只有协议状态机才会。

&emsp;&emsp;时间差已经足够说服人，但要看**机制**，用 `perf stat` 把 1 亿次缩到 1 千万次（保证 perf 的采样开销不扭曲结果）：

**表 4 · perf stat，每线程 1 千万次**

| 指标 | v1（伪共享） | v2（填充） | 比值 |
|---|---|---|---|
| cache-references | **2,940,541** | **373,345** | **7.9×** |
| cache-misses | 2,542,736 | 115,114 | 22.1× |
| miss rate | **86.5%** | 30.8% | — |
| cycles | 476,428,214 | 71,192,368 | 6.7× |
| instructions | 196,487,676 | 198,807,570 | 1.01× |
| **CPI** | **2.42** | **0.358** | **6.8×** |

&emsp;&emsp;两版本的 `instructions` 几乎相同（差 1.2%）——编译器生成的代码没有本质差异。`cycles` 相差 6.7 倍，**所有性能差异都来自"等待"**。

<svg width="760" height="290" viewBox="0 0 760 290" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica,Arial,sans-serif">
  <rect x="8" y="8" width="744" height="274" fill="#fafafa" stroke="#ccc" rx="6"/>
  <text x="30" y="34" font-size="13" font-weight="bold" fill="#333">PMU 指纹：v1 的缓存引用次数膨胀 7.9 倍，CPI 升到 2.42</text>
  <text x="30" y="62" font-size="11" font-weight="bold" fill="#333">cache-references（相对 v2 归一化）</text>
  <rect x="220" y="70" width="440" height="30" fill="#c62828"/>
  <text x="228" y="90" font-size="11" fill="#fff" font-weight="bold">v1: 2.94M  （每次 inc 触发 0.147 次 RFO）</text>
  <rect x="220" y="110" width="56" height="30" fill="#2e7d32"/>
  <text x="286" y="130" font-size="11" fill="#2e7d32" font-weight="bold">v2: 373K</text>
  <text x="30" y="168" font-size="11" font-weight="bold" fill="#333">CPI（cycles / instructions）</text>
  <rect x="220" y="176" width="440" height="30" fill="#c62828"/>
  <text x="228" y="196" font-size="11" fill="#fff" font-weight="bold">v1: 2.42</text>
  <rect x="220" y="216" width="65" height="30" fill="#2e7d32"/>
  <text x="295" y="236" font-size="11" fill="#2e7d32" font-weight="bold">v2: 0.358</text>
  <text x="30" y="270" font-size="10" fill="#666">instructions 几乎相同（196M vs 199M），差异全在等待缓存行所有权迁移</text>
</svg>

&emsp;&emsp;**真正解释性能差异的是 `cache-references` 的绝对数，而不是 miss rate**。v1 每次递增平均产生 **0.147 次 L2/LLC 引用**——这正是 MESI 协议的行为：核心 0 写 `a` 时，向 L2/LLC 发出 RFO 请求把包含 `a`、`b` 的整条缓存行从核心 1 抢过来；核心 1 随后写 `b` 时再反向抢一次。**每一次写操作触发一次跨核一致性事务。** v2 每次递增平均只产生 **0.019 次引用**——`inc` 在本核 L1D 命中已拥有的缓存行，不产生跨核流量。

&emsp;&emsp;**v2 的 miss rate 仍有 30.8%，看起来"也不健康"——但它的引用基数只有 373K，每次 miss 又落在 L2/L3 命中，实际代价几乎为零。** 这是本章最需要记住的反直觉点：**miss rate 本身不能用于诊断，要看 miss 的绝对数、代价以及它发生在哪个缓存层级。** 只看 miss rate 会得出"v2 也不健康"的错误结论，而真正的瓶颈是 v1 的引用次数膨胀。CPI 从 0.358 升到 2.42 的全部增量，来自流水线槽位在等待缓存行所有权迁移——**退休指令数没变，槽位被一致性协议的等待消耗了**。这与 Top-Down 的预测完全吻合：v1 的瓶颈会落在 Backend Bound → Memory Bound，而不是 Frontend Bound 或 Bad Speculation。

### 1.3 结论
&emsp;&emsp;实验 A 和实验 B 合起来，构成了"优化经验为何失效"的两个典型。
- **实验 A**：看起来该有效的对齐优化（求和数组起始地址），在 Zen 3 上被 L1D 并行化能力 + L2/DRAM 供给延迟完全掩盖，交替测量后无系统性差异。
- **实验 B**：看起来只是"数据结构布局"的改动（`alignas(64)`），把每次递增从 3.27 ns 降到 0.12 ns，27 倍加速，方向可复现、机制可解释。两者的差别不在改动大小，而在于**改动是否触及了当前瓶颈所在的微架构机制**——实验 A 的 16B 偏移落在 L1D 能并行吸收的范围内；实验 B 的布局改动直接改变了 MESI 协议的行为，缓存行所有权迁移是真实的、可测量的、7.9 倍的引用膨胀。

&emsp;&emsp;**"优化经验"之所以经常错，就是因为它默认"对齐有用""数据布局有用"这些结论可跨场景迁移，而测量一次次在告诉你它们不可迁移。** 判断一个改动是否有效，必须看它是否触及了当前瓶颈所在的微架构机制；而判断"触及了哪个机制"的唯一可靠方式，是带对照、交替、重复的测量加 PMU 事件计数，而不是一次运行的时间差。
&emsp;&emsp;这两个实验指向同一个结论：**现代 CPU 的性能不由“执行了多少条指令”决定，而由“每个周期提供的流水线槽位（pipeline slot）如何被消耗”决定**。Bakhvalov 在 *Performance Analysis and Tuning on Modern CPUs*（下称 perf-book）中将这一视角系统化为 Top-Down Microarchitecture Analysis Methodology（TMA）：它不要求开发者预先掌握所有 PMU 事件的语义，就能把瓶颈路由到前端、后端、错误推测或退休四个互斥类别之一。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261030_376e7f.svg)

&emsp;&emsp;本文的目标不是罗列优化技巧，而是建立一条可复现的闭环：**先做实验，再看 PMU 输出，然后解释微架构机制，最后用重新测量来验证假设**。所有命令、代码、事件名都以 AMD Ryzen 5 5600X 为实验环境给出——它是 Zen 3 架构、6 核 12 线程、单 CCD、32MB 统一 L3，Linux perf 对其有较完整的 vendor event 支持，适合作为“把方法论落到具体机器上”的样本。


## 2. 测量的第一课：基准在测量微架构状态，而不是“计算量”

### 2.1 频率不是常量，但漂移幅度取决于负载

&emsp;&emsp;连续运行同一个 CPU 密集型循环 5 次，中间不插入其他负载。

```bash
for i in 1 2 3 4 5; do
    perf stat -e cycles,instructions,aperf,mperf ./workload 2>&1 | tail -4
done
```

&emsp;&emsp;5600X 的基础频率 3.7 GHz，最大 boost 4.6 GHz，TDP 65W，默认 PPT 约 76W。直觉上，“第一次运行快、后续运行慢”应该来自缓存预热或频率漂移。但在 `amd-pstate-epp` 驱动下，**频率是否漂移、漂移多少，取决于 governor 的 EPP 设置和负载功耗是否逼近 PPT/温度墙**。用 `freq_drift` 逐次读取 `scaling_cur_freq`，四种条件下的数据如下：

**表 2-1 · 四种条件下的频率与耗时**

| 条件 | 频率 | ns/iter | 单次耗时 | 相对 baseline |
|---|---|---|---|---|
| `powersave` EPP，单核 | 4099 MHz | 1.47 | 293.7 ms | **+13.4%** |
| `performance` EPP，单核 | 4649 MHz | 1.29 | 258.9 ms | baseline |
| `performance` + 5 核 stress-ng，61.8°C | 4600 MHz | 1.31 | 261.3 ms | +0.9% |
| `performance` + 更重负载 | 4550 MHz | 1.32 | 263.9 ms | +1.9% |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261030_391bcc.svg)

&emsp;&emsp;**时间比与频率比严格一致**：powersave vs performance 是 1.134 倍时间、1.134 倍频率；5 核负载 vs baseline 是 1.009 倍时间、1.010 倍频率；重负载 vs baseline 是 1.019 倍时间、1.021 倍频率。三者都落在 0.5% 的误差内。这说明 `freq_drift` 的循环在四种条件下执行的是**完全相同的指令流**，**时间差异 100% 由频率解释**，没有任何缓存、分支或代码生成的混杂因素。`scaling_cur_freq` 读数与执行速度严格成比例，可以作为微基准频率条件的可靠锚点。

&emsp;&emsp;但四种条件揭示了**频率漂移并不是一个固定幅度的噪音源**：

- **EPP 切换造成 13.4% 的差异**——这是四种条件中最大的单一变量。`powersave` 下 5 次运行波动 < 0.02%，`performance` 下 5 次运行波动 < 0.06%——两者各自都“非常稳定”，但前者稳定在 4099 MHz，后者稳定在 4649 MHz。**稳定性不等于正确性**：`powersave` 下测出的 293.7 ms 是一个高度可重复的数字，但它不代表代码在不受 EPP 压制时的真实性能。
- **5 核 stress-ng 只造成 0.9% 的差异**——温度 61.8°C，远低于 5600X 的 Tjmax（95°C）。Precision Boost 的频率退档通常从 75–80°C 才开始明显，61.8°C 意味着散热系统还有 15–20°C 的余量，CPU 完全没有理由大幅降频。频率在 4599–4643 MHz 之间小幅抖动（不是单调下降），反映的是 Precision Boost 在功耗预算内做的微调，不是热降频。
- **更重负载造成 1.9% 的差异**——频率在 4549–4574 MHz 之间抖动（0.5% 的范围），仍然没有出现单调下降的趋势。这说明即便功耗更接近 PPT 限制，在良好散热下，频率也只是在更高 boost 频率附近小幅调整，而不是断崖式退档。

>实际上用手机芯片测试更容易遇到温度墙，PC机器的CPU需要长时间高负载任务才容易触发温度墙。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261030_c063d1.svg)

&emsp;&emsp;这意味着：**“频率漂移”不是一个可以笼统假设的噪音源，而是一个需要被测量的条件变量**。漂移的幅度取决于**负载功耗是否逼近 PPT/温度墙**：单核 30–40W 时温度稳定在 60°C 以下，boost 不降；5 核 55–65W 时温度到 61.8°C，boost 只微降 1%；只有把功耗推到 76W 或温度推到 80°C+，漂移才会进入 5–10% 量级。

&emsp;&emsp;**`cycles` 本身会“吸收”频率变化**，使得 CPI 看起来正常而实际吞吐已经下降——powersave 下 `cycles / instructions` 可能和 performance 下几乎相同，因为 `cycles` 是按频率计数的，频率降了，`cycles` 计数也按比例变少了。稳健的做法是同时采集 `cycles`、`instructions` 和 `aperf/mperf`，用 `mperf/aperf` 的比值判断频率是否偏离了 TSC 基准；同时在实验记录中明确标注 **EPP 设置、governor、后台负载和温度**——这四项是解释任何跨运行时间差异的前提。在 5600X 上做微基准比较之前，先做两件事：**把 EPP 固定为 `performance`，并在实验记录中记下当前温度**。前者消除了 13.4% 量级的配置差异，后者界定了当前条件下频率漂移的幅度上限。


### 2.2 被编译器消除的基准比“测不准”更危险

&emsp;&emsp;朴素微基准会被编译器整个消除。perf-book 的检测方法是：用 PMU 确认 `instructions` 退休数是否与预期相符。如果一条本应执行 1000 次的循环只退休了 3 条指令，基准已经失效。

&emsp;&emsp;在 5600X 上，写四个版本的求和基准，每个版本用不同的方式试图“阻止编译器消除”。v1 朴素——循环内联到调用方，结果只用于 `printf`；v2 每次迭代写 volatile 全局变量；v3 把循环放进 `noinline` 函数，在函数**外部**加 asm 屏障；v4 把 asm 屏障放进循环**内部**。用 `perf stat` 采集 Zen 3 的退休指令数 `ex_ret_instr` 和 Op Cache 事件：

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

&emsp;&emsp;**v1 和 v3 的 `ex_ret_instr / n` 都是 0.032——1 亿次迭代只退休了约 320 万条指令**。如果循环真的执行了 1 亿次，即使每次只有 1 条加法，也应该退休 1 亿条指令。实测只有 320 万，差了两个数量级。这 320 万条不是循环的指令，是 `perf stat` 启动、动态链接和 `printf` 本身的开销。**循环根本没有执行**——编译器把 `s += i` 归纳成了 `n*(n-1)/2 = 4999999950000000`，直接算出结果。

&emsp;&emsp;**v3 是本章最重要的教学点。** 它写了 `noinline`、写了 `asm volatile` 屏障、结果被 `printf` 使用——三个看起来“应该有效”的措施全部到位，但 `ex_ret_instr` 和 v1 一样低，`time` 一样是 0.000 ms。原因很具体：`noinline` 只阻止**调用方内联 `simple_sum_noinline`**，但 `simple_sum_noinline` **内部**的循环仍然被编译器归纳成闭式解——函数调用保留了，循环没了。`asm` 屏障在函数**外部**，保护的是“`s` 的返回值不被优化掉”，而不是“循环体内部的累加不被归纳”。**屏障的位置决定了它保护的是什么**：循环外的屏障保护返回值，循环内的屏障保护循环体。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_5e3c78.svg)

&emsp;&emsp;**v4 是唯一真正阻止消除的写法。** `asm volatile("" : "+r"(s) : : "memory")` 放在循环体内，每次迭代都告诉编译器“`s` 可能被外部修改”——编译器无法把累加折叠成 `n*(n-1)/2`，循环必须逐次执行。`ex_ret_instr / n = 7.04`，与源码预期一致（加法、循环比较、跳转、屏障对代码生成的影响），`time = 28.8 ms`。

&emsp;&emsp;v2 用 volatile 全局变量也阻止了消除，但代价是每次迭代多一次 store：`ex_ret_instr / n = 3.41`，`time = 24.5 ms`。**v2 和 v4 的时间差（24.5 vs 28.8 ms）比指令数差（3.41 vs 7.04）小得多**——v4 的指令数是 v2 的两倍，时间只多 17%。原因在依赖链：`s += i` 的加法延迟很短，多出的指令填充了原本空闲的流水线槽位，CPI 从 v2 的约 1 降到 v4 的约 0.5。**指令数不能单独用于判断性能**——退休指令数告诉你有多少工作被完成，但不告诉你有多少工作被等待。

&emsp;&emsp;5600X 的 Op Cache 事件提供了第二道验证：

| 版本 | Op Cache 访问 | Op Cache 缺失 | 缺失率 |
|---|---|---|---|
| v1 | 878,563 | 348,266 | **39.6%** |
| v2 | 51,055,620 | 450,570 | 0.88% |
| v3 | 863,583 | 341,952 | **39.6%** |
| v4 | 101,077,489 | 462,762 | 0.46% |

&emsp;&emsp;v1 和 v3 的 Op Cache 访问只有 86 万次，缺失率却高达 39.6%——因为程序只执行了 `perf stat` 启动和 `printf` 的少量代码，这些冷代码不在 Op Cache 里。v2 和 v4 的 Op Cache 访问在 5000 万到 1 亿次量级，缺失率低于 1%——**循环代码完全驻留在 Op Cache 中，前端供给不是瓶颈**。`ex_ret_instr` 和 `op_cache` 的对比比单看任何一个都可靠：前者告诉你“退休了多少指令”，后者告诉你“前端是否真的在供给这些指令”。

&emsp;&emsp;后续所有实验在跑计时之前，先用 `perf stat -e ex_ret_instr,macro_ops_retired` 采集一次，计算 `ex_ret_instr / n`。判据不是 `time`，而是这个比值——**v1 的 `time = 0.000090 ms` 看起来像一个极快的微基准，但 `ex_ret_instr / n = 0.032` 揭穿了它**。只看 `time` 无法区分“循环被消除”和“微基准本来就在微秒量级”；`ex_ret_instr` 一锤定音。如果它与源码里数出的每迭代指令数一致到 2 倍以内，基准有效；如果差一个数量级以上，基准失效。这个检查的成本是每次实验多一次 `perf stat` 运行，收益是排除“测的根本不是你想测的东西”这个最危险的陷阱——**它比“测不准”更危险，因为它的输出看起来完全正常：`time` 很小、`result` 正确（v1/v3 的 4999999950000000 与 v4 一致）、程序没有崩溃，只有退休指令数能揭穿它**。


## 3. 用 PMU 建立指标语言：CPI 之后用什么

### 3.1 CPI 是聚合指标，不是诊断指标

&emsp;&emsp;四个 workload 的设计意图和实际行为如下：
- **v1 compute**：三条独立的 64 位乘法链，设计意图是 Core Bound（执行端口/依赖链）。如果三条链真的并行，CPI 应该接近乘法吞吐上限；如果编译器把链合并或把依赖并行了，CPI 会异常低——后者正是实际发生的情况。
- **v2 chase**：随机排列的指针追逐，设计意图是 Memory Bound（L3 延迟）。8MB 工作集超出 512KB L2、落在 32MB L3 内，每次追逐都触发一次 L3 访问。
- **v3 branch**：数据驱动的 if (v >= 128) 分支，设计意图是 Bad Speculation（约 50% 误预测率）。分支方向由随机数据决定，预测器无法学习。
- **v4 frontend**：64 次宏展开的巨型循环体，设计意图是 Frontend Bound（Op Cache 溢出）。循环体展开后约 900 条 macro-op，预期超出 Zen 3 的 4K uop Op Cache。

&emsp;&emsp;`cycles / ex_ret_instr` 给出 CPI。直觉上，CPI 越低越快——1 个周期退休 2 条指令，比 1 个周期退休 0.2 条指令高效 10 倍。四个 workload 的数据（n = 1e7，EPP = performance）却推翻了这个直觉：

**表 3-1 · 四个 workload 的 CPI 与绝对量**

| 版本 | 设计意图 | cycles | `ex_ret_instr` | CPI | 每迭代指令数 | 每迭代时间 |
|---|---|---|---|---|---|---|
| v1 compute | Core Bound | 33.6M | 82.1M | **0.41** | 8.2 | 3.4 ns |
| v2 chase | Memory Bound | 593.0M | 115.7M | **5.12** | 11.6 | 59.3 ns |
| v3 branch | Bad Speculation | 198.5M | 508.1M | **0.39** | 50.8 | 19.8 ns |
| v4 frontend | Frontend Bound | 3217.8M | 8990.2M | **0.358** | 899.0 | 321.8 ns |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_9e39b0.svg)

&emsp;&emsp;**v4 的 CPI 是 0.358，比 v1 的 0.41 还低——但它花了 3.2G 个周期，是 v1 的 96 倍。** 原因在 `每迭代指令数` 这一列：v1 每迭代 8.2 条指令，v4 每迭代 899 条指令。CPI 低说明“每条指令平均占用的周期少”，但它完全不告诉你有多少条指令。**真正的每迭代时间是三个量的乘积**：

$$\text{每迭代时间} = \text{每迭代指令数} \times \text{CPI} \times \text{每周期时间}$$

&emsp;&emsp;v1 是 8.2 × 0.41 × 0.21 ns ≈ 0.71 ns，v4 是 899 × 0.358 × 0.21 ns ≈ 67.6 ns。表中显示的每迭代时间（3.4 ns vs 321.8 ns）比这个计算大一个量级，因为 `n` 是外层迭代数，而 v4 每次外层迭代执行了 64 次内层 STEP——899 条指令是 64 次 STEP 的展开结果，不是单次累加。**具体数值不重要，重要的是三个量的关系**：CPI 只是三个量中的一个，单独看它无法判断“快慢”。

&emsp;&emsp;更关键的是，**CPI 也完全不告诉你“慢在哪里”**。v2 的 CPI 是 5.12，远高于其他三个，看起来“最慢”。但如果只看 CPI，你会以为 v2 需要 12.8 倍的指令级优化——实际上 v2 的时间是 v1 的 17.6 倍，而 CPI 只差 12.5 倍，差额来自“每迭代指令数”也不同（11.6 vs 8.2）。**CPI 是聚合指标，它把“指令数”和“每指令周期”两个维度揉成了一个数**。要拆分这两个维度，必须采集 `cycles` 和 `ex_ret_instr` 两个原始量，不能只看比值。

### 3.2 同一个 CPI，两个完全不同的瓶颈——在数据里没出现

&emsp;&emsp;第 1 章预期实验 A/B 会演示“同一个 CPI，两种瓶颈”。但四个 workload 的实际数据显示：**没有两个版本的 CPI 相同**。v1（0.41）、v3（0.39）、v4（0.358）三者都低，但它们的瓶颈完全不同——这不是“同一个 CPI”的案例，而是“**低 CPI 掩盖了三个完全不同的瓶颈类别**”的案例。

&emsp;&emsp;要判断哪个版本受限于什么，需要看**两个比值**，而不是 CPI：

**表 3-2 · 两个比值把“低 CPI”拆成三个方向**

| 版本 | CPI | `cache-misses / cache-references` | `ex_ret_brn_misp / ex_ret_brn` | `op_cache_miss / all_accesses` | `ls_mab_alloc` | 判读 |
|---|---|---|---|---|---|---|
| v1 | 0.41 | 11.4% | 0.58% | **1.87%** | **51.8K** | Retiring（无瓶颈） |
| v2 | 5.12 | **48.4%** | 3.12% | 20.7% | **13.2M** | **Memory Bound** |
| v3 | 0.39 | 6.27% | 2.20% | 11.2% | 2.29M | **基准失效（cmov）** |
| v4 | 0.358 | 4.07% | 2.51% | **0.41%** | 193K | **基准失效（Op Cache 未溢出）** |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_e30fbc.svg)

&emsp;&emsp;**v2 是唯一被数据证实的瓶颈**。`cache-misses / cache-references = 48.4%`——近一半的缓存访问都 miss 到 L2/L3，且 `ls_mab_alloc = 13.2M`（每迭代 1.32 次 MAB 分配）。这是 Memory Bound 的直接证据：每次随机指针追逐都要在 Miss Address Buffer 里排队等 L3 返回。`op_cache_miss / all_accesses = 20.7%` 看起来也高，但这是因为指针追逐的随机性导致代码预取失效，不是前端供给成为瓶颈。

&emsp;&emsp;**v1 是唯一“健康”的版本**。`cache-misses / cache-references = 11.4%` 但 `cache-references` 只有 258.8K——绝对量极小，11.4% 的 miss 只对应 29.4K 次缺失，对 82M 条指令来说是噪声。`ls_mab_alloc = 51.8K`、`op_cache_miss / all_accesses = 1.87%`——两个与内存/前端相关的关键事件都极低。**v1 是真正的 Retiring**：槽位被有效退休指令填充，没有哪个子系统在拖后腿。CPI 0.41 意味着每周期退休 2.44 条指令，接近 Zen 3 的 6-wide dispatch 上限的 40%。

&emsp;&emsp;**v3 是基准失效**。它的 CPI 只有 0.39，`ex_ret_instr` 却有 508M（每迭代 50.8 条指令）。这是编译器把 `if (v >= 128) sum += v; else sum -= v;` 优化成了**条件移动 `cmov`**——根本没有分支。`ex_ret_brn = 63.8M / 10M = 6.4` 分支/迭代，但其中绝大多数是**循环控制分支**（10M 次 taken，只有最后一次不 taken），不是数据分支。`brn_misp / brn = 2.2%` 恰好等于“循环控制分支的误预测率”+“少量数据分支残留”，**没有测到任何 `cmov` 优化的分支**。

&emsp;&emsp;**v4 也是基准失效**。它的 CPI 只有 0.358，`ex_ret_instr` 高达 8990M（每迭代 899 条指令）。设计目标是“巨型循环体超出 Op Cache 容量”，但 `op_cache_miss / all_accesses = 0.41%`——**Op Cache 完全接住了循环体**。Zen 3 的 Op Cache 是 4096 个 uop 的容量，64 次 STEP 展开只产生约 900 条 macro-op，远未触及 4096 的上限。`ls_mab_alloc = 193K`（每迭代 0.019 次）也极低，说明内存子系统不是瓶颈。**v4 也是 Retiring**——只是它退休了 899 条指令/迭代，CPI 天然就低。

### 3.3 从“测量了什么”到“基准是否有效”

&emsp;&emsp;表 3-2 的最后一列是本章最重要的发现：**四个精心设计的 workload，三个没有测到预期瓶颈**。v1 设计为 Core Bound，但三条独立乘法链让编译器把延迟完全并行了，CPI 0.41 说明没有瓶颈；v3 设计为 Bad Speculation，但 clang 把数据依赖分支优化成了 `cmov`，根本没有分支可误预测；v4 设计为 Frontend Bound，但 899 条指令的循环体对 4096 个 uop 的 Op Cache 来说远远不够，前端供给顺畅。

&emsp;&emsp;**这不是实验失败，这是第 2 章“基准在测你想测的东西吗”的延续**。第 2.2 节讲的是“循环被完全消除”，本节讲的是“循环被完整执行了，但没有执行到你想测的那条路径”。两者都需要同一道检查：**看 PMU 数据是否与源码预期一致**。v3 的 `ex_ret_brn / n = 6.4` 看起来像“每次迭代 6.4 个分支”，但源码里只有 1 个数据分支——差额来自循环展开后的循环控制分支，数据分支被 `cmov` 吃掉了。**只有看 `ex_ret_brn` 的绝对量，才能发现“分支数不对”**。

&emsp;&emsp;**修复三个失效的基准**需要回到源码：

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

&emsp;&emsp;后续所有实验在看 CPI 之前，先看 `cache-misses / cache-references`、`ex_ret_brn_misp / ex_ret_brn` 和 `op_cache_miss / all_accesses` 三个比值。**三个比值都低、CPI 也低**——基准失效或 Retiring（需要进一步区分）；**某个比值高**——信号明确，下钻到对应的 Zen 3 事件。**v2 是唯一被数据证实的瓶颈**，它的 `cache-misses / cache-references = 48.4%` 和 `ls_mab_alloc = 13.2M` 一起，给出了 Memory Bound 的直接证据。**v3 和 v4 的失效不是坏事**——它们恰好演示了第 2 章的教训在更大规模上的延续：**“执行了循环”和“执行了你想测的那条路径”是两件不同的事**，只有 PMU 事件计数能区分这两种状态。


## 4. Top-Down 方法论与 5600X 上的采集实现

&emsp;&emsp;本章实验沿用第 2、3 章的测量纪律：EPP 固定为 `performance`，`taskset -c 2` 绑单核，n = 1e7，每个事件独立一个 `-e`。四个 workload 分别对应四个桶的设计意图：

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

&emsp;&emsp;采集分两层：**Level 1** 用 `cycles`、`ex_ret_instr`、`ex_ret_brn`、`ex_ret_brn_misp`、`cache-references`、`cache-misses` 六个事件算 CPI 和三个比值；**Layer 2** 用 `op_cache_hit_miss.*`、`de_dis_cops_from_decoder.*`、`ls_mab_alloc.*` 三个 Zen 3 组事件下钻到前端和内存子系统。`perf list metric | grep -i pipeline` 在实测中返回空——**`PipelineL1` metric 在 5600X 这版 perf 上不可用**，所以 Level 1 的四个桶必须从原始事件近似，不能像 Intel 那样直接跑 `--topdown`。

```bash
# Level 1：六个事件
taskset -c 2 sudo perf stat -e cycles -e ex_ret_instr -e ex_ret_brn \
    -e ex_ret_brn_misp -e cache-references -e cache-misses \
    ./topdown_bench -v <1|2|3|4> -n 10000000

# Layer 2 前端
taskset -c 2 sudo perf stat -e cycles -e op_cache_hit_miss.all_op_cache_accesses \
    -e op_cache_hit_miss.op_cache_miss \
    -e de_dis_cops_from_decoder.disp_op_type.any_integer_dispatch \
    ./topdown_bench -v <1|2|3|4> -n 10000000

# Layer 2 内存
taskset -c 2 sudo perf stat -e cycles -e ls_mab_alloc.load_store_allocations \
    -e cache-references -e cache-misses \
    ./topdown_bench -v <1|2|3|4> -n 10000000
```

&emsp;&emsp;四个精心设计的 workload 中只有两个落入了设计意图的桶——v2 的 `cache-misses / cache-references = 47.3%` 和 `ls_mab_alloc / n = 1.32` 确认了 Memory Bound；v4 的 `op_cache_miss / all_accesses = 84.1%` 和 `de_dis_cops / cycles = 3.19` 确认了 Frontend Bound。v1 的三次修复（乘法链 → 常数除法 → volatile 除数）全部失效——Zen 3 的约 320 条 ROB 和 6-wide dispatch 足以把 idiv 的 20+ cycle 延迟藏在后续独立指令后面；v3 的 `ex_ret_brn_misp / ex_ret_brn = 7.9%` 说明 asm 屏障仍然没能阻止编译器把数据分支优化掉。**这两个失效不是实验失败，而是 5600X 上做 Top-Down 诊断的两个真实边界**——它们和 v2/v4 的成功一起，构成了本章的核心叙事：**Top-Down 的路由结果必须用 PMU 数据验证，验证失败时的信息量往往比成功时更大**。

### 4.1 Level 1 的四个类别

&emsp;&emsp;Top-Down Level 1 输出四个互斥类别，总和约 100%：**Frontend Bound**（取指/解码供给不足）、**Backend Bound**（执行或内存资源成为瓶颈）、**Bad Speculation**（错误推测浪费槽位）、**Retiring**（槽位被有效退休指令填充）。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_9ede2b.svg)

### 4.2 在 5600X 上采集 Level 1

&emsp;&emsp;`perf stat --topdown` 在 Zen 3 上回退到手动公式和原始 PMU 事件。5600X 没有与 Intel TMA 完全对应的硬件级槽位分类器，`PipelineL1` metric 在实测中不可用——所以 Level 1 的四个桶必须从原始事件**近似**：

| 桶 | 近似判据 | 主要事件 |
|---|---|---|
| **Frontend Bound** | `op_cache_miss / all_accesses > 10%` 且 `de_dis_cops / cycles > 3` | `op_cache_hit_miss.*`、`de_dis_cops_from_decoder.*` |
| **Bad Speculation** | `ex_ret_brn_misp / ex_ret_brn > 30%` | `ex_ret_brn_misp`、`ex_ret_brn` |
| **Backend Bound / Memory** | `cache-misses / cache-references > 30%` 且 `ls_mab_alloc / n > 0.5` | `cache-misses`、`ls_mab_alloc` |
| **Backend Bound / Core** | CPI 高（> 2）且以上三个比值都低 | 排除法 |
| **Retiring** | CPI 低（< 1）且以上三个比值都低 | 排除法 |

&emsp;&emsp;四个 workload 的实际数据如下（n = 1e7，EPP = performance，`taskset -c 2`）：

**表 4-1 · 四个版本的 Level 1 与 Layer 2 关键比值**

| 版本 | 设计意图 | CPI | `cache-miss%` | `brn-misp%` | `op_cache-miss%` | `ls_mab_alloc/n` | 实际落桶 |
|---|---|---|---|---|---|---|---|
| v1 core | Core Bound | **0.96** | — | — | — | — | **失效（Retiring）** |
| v2 chase | Memory Bound | **4.96** | **47.3%** | 3.4% | 19.8% | **1.32** | **✅ Memory Bound** |
| v3 branch | Bad Speculation | **0.62** | 7.0% | **7.9%** | 7.8% | 0.23 | **失效（分支被消除）** |
| v4 frontend | Frontend Bound | **0.358** | 2.5% | 8.0% | **84.1%** | 0.12 | **✅ Frontend Bound** |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_0e5298.svg)

&emsp;&emsp;**v2 是唯一被数据证实的 Memory Bound**。`cache-misses / cache-references = 11,722,664 / 24,774,509 = 47.3%`——近一半的缓存访问都 miss 到 L2/L3。`ls_mab_alloc / n = 13,217,809 / 10,000,000 = 1.32`——每次迭代平均分配 1.32 个 MAB 条目，排队等 L3 返回。两个内存事件同时高，CPI 5.0 是这个等待的直接结果。`op_cache_miss / all_accesses = 19.8%` 看起来也高，但这是因为指针追逐的随机性导致代码预取失效，不是前端供给成为瓶颈——`de_dis_cops / cycles = 117.9M / 588.9M = 0.20`，远低于 3，说明解码器没有成为瓶颈。

&emsp;&emsp;**v4 是唯一被数据证实的 Frontend Bound**。`op_cache_miss / all_op_cache_accesses = 3,730,052,076 / 4,432,747,576 = 84.1%`——512 次 STEP 展开（约 7176 条 macro-op）远超 Zen 3 的 4096 uop Op Cache 容量，绝大多数取指回落到 4-wide 解码器。`de_dis_cops / cycles = 82,021,949,578 / 25,693,920,601 = 3.19`——每周期从解码器分发 3.19 个 cop，接近 4-wide 解码器的上限。**两项签名同时命中**，CPI 0.358 只是“指令多但每条都不等”的表面现象，真正的瓶颈是前端供给带宽。

&emsp;&emsp;**v1 和 v3 都是基准失效**。v1 的 CPI 0.96 看起来“健康”，但三次修复（乘法链 → 常数除法 → volatile 除数）全部失败——`result = 11` 说明除法确实执行了，但 Zen 3 的约 320 条 ROB 把 `idiv` 的 20+ cycle 延迟藏在了后续独立指令后面，CPI 被稀释到 1 以下。v3 的 `ex_ret_brn_misp / ex_ret_brn = 6,271,330 / 79,042,222 = 7.9%`——远低于 Bad Speculation 的 30% 阈值，说明 `asm` 屏障仍然没能阻止编译器把数据分支优化成 `cmov` 或条件选择，真正的分支只有循环控制分支（10M 次 taken，最后一次不 taken，误预测率极低）。

### 4.3 下钻到 Level 2：Backend Bound 的 Memory/Core 分裂

&emsp;&emsp;Backend Bound 被划分为 Memory Bound 和 Core Bound。**Memory Bound 是一个桶，不是一种原因**。里面同时装着 L1D 缺失、L2/L3 缺失、DTLB 缺失、预取效率不足、以及带宽饱和。在 5600X 上，Memory Bound 内部需要特别关注 **MAB（Miss Address Buffer）分配压力**：`ls_mab_alloc` 直接反映 load/store 请求在 miss 队列中的排队程度。所有 6 个核心共享 32MB L3，L3 未命中会经过 Infinity Fabric 走向内存控制器，而 FCLK/UCLK/MCLK 的同步状态决定了未命中的实际延迟。

&emsp;&emsp;v2 的数据给出了这个分裂的直接证据：`cache-misses / cache-references = 47.3%` 说明缓存缺失率高，`ls_mab_alloc / n = 1.32` 说明每次缺失都在 MAB 里排队。**两个事件一起看，才能确认“慢在等 L3”，而不是“慢在执行端口”**——v2 的 `op_cache_miss = 19.8%` 虽然也不低，但 `de_dis_cops / cycles = 0.20` 说明前端供给不是瓶颈，`ls_mab_alloc` 才是真正的排队点。

&emsp;&emsp;**v1 的失效恰好是 Core Bound 的反面证据**。设计意图是“串行 idiv 依赖链让后端执行资源成为瓶颈”，但 Zen 3 的宽乱序窗口把 idiv 的延迟藏了起来，CPI 降到 0.96——四个比值全部低，看起来像 Retiring，实际上是“基准没有制造出 Core Bound”。**在 5600X 上制造 Core Bound 需要比 Intel 苛刻得多的条件**（端口冲突、序列化指令、或极长的依赖链），这不是代码问题，是 Zen 3 的微架构特性。

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

&emsp;&emsp;**预取没有降低 Backend Bound**。`cache-misses` 只降了 1.2%，`ls_mab_alloc` 降了 10.5%——预取确实减少了 MAB 排队压力，但 `cycles` 反而增加了 0.85%。原因在 `ex_ret_instr`：v6 多了 5.6% 的指令（预取指令本身），这些指令填充了原本空闲的槽位，但**没有改变内存控制器的可持续吞吐**。256MB 工作集在 DDR4-3600 双通道上的可持续带宽约 25 GB/s，预取只是让请求更早进入 MAB 队列，队列的**平均长度**没有显著缩短——`ls_mab_alloc` 降 10.5% 是因为预取把部分 demand miss 转成了 prefetch hit，但 `cycles` 的瓶颈已经从“延迟等待”切换到了“带宽饱和”。

&emsp;&emsp;**“改了—重测—解释为什么变了或没变”** 才是 TMA 的使用方式。v5 → v6 的改动让 `ls_mab_alloc` 降了 10.5%，但 `cycles` 没降——这说明预取解决的是“MAB 排队”，不是“内存带宽”。如果已经处于带宽上限，预取**不会降低 Backend Bound**——它只是把“延迟问题”换成了“带宽问题”。**判据不是“预取有没有让 cache-misses 下降”，而是“预取有没有让 cycles 下降”**——后者才是 Backend Bound 是否被真正缓解的唯一标准。

### 4.5 小结

&emsp;&emsp;四个精心设计的 workload，只有 v2（Memory Bound）和 v4（Frontend Bound）落入了设计意图的桶。v1 和 v3 的失效不是实验失败——它们恰好演示了第 1 章的主题在 Top-Down 层面的延续：

- **v1 的多次修复**（乘法链 → 常数除法 → runtime 除法 → volatile 序列化）全部被编译器或 Zen 3 的乱序窗口化解。**在 5600X 上制造 Core Bound 需要比 Intel 苛刻得多的条件**——这是 Zen 3 的微架构特性，不是代码问题。
- **v3 的 `asm` 屏障**（两个分支体各自带 `asm volatile`）仍然没能阻止编译器把数据分支优化掉。`ex_ret_brn_misp / ex_ret_brn = 7.9%` 说明真正的数据分支几乎不存在，只有循环控制分支被保留。**在 clang -O2 -march=znver3 下制造 Bad Speculation，需要比 `asm` 屏障更激进的序列化手段**（函数调用边界、`noinline` 分支体、或 `__builtin_expect` 配合 `noinline`）。

&emsp;&emsp;**这两个失效是本章最有价值的教学内容**。它们说明：**Top-Down 的路由结果必须用 PMU 数据验证，而不是假设“设计了哪个桶就会落到哪个桶”**。v2 和 v4 的成功证明了“用原始事件映射到四个桶”在 5600X 上是可行的；v1 和 v3 的失效证明了“桶的设计意图”和“实际测到的桶”之间隔着编译器优化和微架构特性两层，每一层都需要用 PMU 数据去验证。**验证失败时的信息量，往往比成功时更大**——它指出了基准设计中的隐藏假设，只有修正这些假设，下一轮测量才能落到真正的瓶颈上。


## 5. 前端双路径：解码器、Op Cache 与 dispatch

&emsp;&emsp;第 4 章的 v4 已经展示了 512 STEP 展开时 Op Cache 溢出（84.1% miss）。但 84.1% 只是一个数据点——它不能回答“Op Cache 的边界在哪里”，也不能回答“溢出之后，前端是否真的成为瓶颈”。本章用五个 STEP 展开版本画出从完全命中到完全溢出的完整梯度。

&emsp;&emsp;**实验设计**。五个版本共用同一个 STEP 宏（`a*3+1`、`b*5+2`、`c*7+3`，三次异或、三次移位），每个 STEP 约 14 条 macro-op，三条独立链（`a`、`b`、`c`）各自的依赖链约 6 cycles 延迟。唯一的变量是外层循环体内 STEP 的展开次数：

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

&emsp;&emsp;选择 STEP 而不是任意指令填充，是因为它**天然阻止编译器优化**：三条链各自有乘法→加法→异或→移位的非线性依赖，跨 STEP 合并需要模拟非线性递推，任何生产级编译器都不会做。这样“每迭代 macro-op 数”严格等于 STEP 次数 × 14 + 循环开销，不需要额外的 asm 屏障去防消除——**基准设计层面的胜利，比事后用屏障去堵编译器的洞更可靠**。

&emsp;&emsp;五个版本的 `n` 按 STEP 次数反比缩放，保证总指令量在 1e9 量级、单次跑 0.1–0.5 秒：v1 用 `n=1e8`、v2 用 `n=1e7`、v3 用 `n=1e6`、v4 用 `n=3e5`、v5 用 `n=1e5`。测量条件与第 2–4 章一致：EPP 固定为 `performance`，`taskset -c 2` 绑单核，每个事件独立一个 `-e`（避免 `-e "a,b,c"` 在 Zen 3 上的解析失败）。

&emsp;&emsp;采集分两组：**Op Cache 组**用 `op_cache_hit_miss.all_op_cache_accesses` 和 `.op_cache_miss` 计算命中率，这是“前端走了哪条路径”的唯一直接证据；**dispatch 组**用 `de_dis_cops_from_decoder.disp_op_type.any_integer_dispatch` 除以 `cycles` 计算分发速率，这是“前端供给是否跟得上后端消费”的唯一直接证据。两组一起看，才能区分“Op Cache 溢出”和“Frontend Bound”——这是本章要证明的核心区分。

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

&emsp;&emsp;五个版本的 Op Cache miss 率从 0.28% 跳到 99.2%，断层落在 v3（1799 条）和 v4（7187 条）之间——与 Zen 3 的 4096 uop 规格容量吻合。但 `de_dis_cops / cycles` 在五个版本上全部稳定在 **3.18–3.24**，Op Cache miss 率变化 350 倍，分发速率纹丝不动。**这说明 Op Cache 溢出只改变了前端的取指路径，没有改变前端的供给能力**——后端的消费速率（3.18 cop/cycle）始终低于解码器的供给上限（4-wide），Op Cache 溢出没有转化为 Frontend Bound。本章的核心结论由此展开：**“Op Cache 溢出”和“Frontend Bound”之间不画等号，判据是 `de_dis_cops / cycles` 与 4 的关系**。

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

&emsp;&emsp;**每个 STEP 稳定在 14.0–14.5 条 macro-op**——`a*3+1`、`b*5+2`、`c*7+3` 三次乘法加三次加法、三次异或、三次移位、两次比较和跳转，再加上 STEP 之间的寄存器搬运，总共 14 条。五个版本的 `ex_ret_instr / n` 严格按 STEP 次数线性增长，说明编译器没有做任何跨 STEP 的合并——STEP 宏内的 `a ^= (a >> 5)` 是非线性运算，编译器无法把它折叠成闭式解。

&emsp;&emsp;这个有效性检查比第 3、4 章更顺利，原因是 **STEP 宏的设计天然阻止了优化**：三条独立的链（`a`、`b`、`c`）各自有乘法→加法→异或→移位的依赖，每次 STEP 的输出是下一次 STEP 的输入，跨 STEP 合并需要模拟非线性递推，任何生产级编译器都不会做。**这是“基准设计”层面的胜利——与其事后用屏障去阻止优化，不如从一开始就让代码结构本身不可优化**。

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

&emsp;&emsp;**断层在 v3（1799 条）和 v4（7187 条）之间**——miss 率从 0.74% 跳到 84.2%。这个断层与 Zen 3 的 Op Cache 规格容量 **4096 个 uop** 完全吻合：1799 条 macro-op 展开后约 2000–2500 个 uop，落在 4096 以内；7187 条 macro-op 展开后约 8000–10000 个 uop，远超 4096。**规格数字是理论上限，实测给出的可用边界在 2000–7000 个 uop 之间**——这个区间包含了分支密度、代码布局、Op Cache 替换策略等实际因素，是“能在 5600X 上用来判断循环体是否过大”的实用判据。

&emsp;&emsp;**v5 的 99.2% 是“彻底溢出”的极端**：每迭代 14385 条 macro-op，Op Cache 几乎完全接不住，每次外层迭代都要从 4-wide 解码器重新取指。但接下来你会看到，**“彻底溢出”并不等于“前端成为瓶颈”**——这是本章最重要的反直觉发现。

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

&emsp;&emsp;**五个版本的 `de_dis_cops / cycles` 全部在 3.18–3.24 之间**——Op Cache miss 率从 0.28% 跳到 99.2%，分发速率**没有任何变化**。`de_dis_cops` 每周期稳定分发约 3.2 个 cop，远低于 6-wide dispatch 的上限，也低于 4-wide 解码器的上限。**Op Cache 溢出没有导致前端供给不足**——因为前端的供给能力（Op Cache 的 8-wide 或解码器的 4-wide）始终高于后端的消费能力（3.18 cop/cycle）。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261031_4d8963.svg)

&emsp;&emsp;**后端才是真正的瓶颈**。STEP 宏内三条独立链（`a`、`b`、`c`）各自的依赖链是乘法（3 cycles）→ 加法（1 cycle）→ 异或（1 cycle）→ 移位（1 cycle），约 6 cycles 延迟。三条链并行，每 6 cycles 可以退休约 18 条指令，即 **3.0 cop/cycle**——与实测的 3.18 完全吻合。**前端的 Op Cache 或解码器，无论哪条路径，都能轻松跟上这个速率**。



### 5.4 为什么 Op Cache 溢出不等于 Frontend Bound

&emsp;&emsp;五个版本的数据合起来，给出一个比“Op Cache 溢出就慢了”更精确的判据：**Op Cache 溢出只在“后端消费速率超过 4-wide 解码器供给能力”时，才转化为 Frontend Bound**。

&emsp;&emsp;**反例（本章数据）**：STEP 宏的三条独立链让后端消费速率稳定在 3.18 cop/cycle，低于 4-wide 解码器的供给上限（4 cop/cycle）。此时 Op Cache 溢出到 99.2%，解码器仍然能跟上后端的节奏——**Op Cache miss 率高，但前端不是瓶颈**。

&emsp;&emsp;**正例（第 4 章 v4）**：512 STEP 展开的循环体，每迭代约 7176 条 macro-op，后端消费速率被展开的独立运算推高。第 4 章 v4 的 `de_dis_cops / cycles = 3.19`——和本章的 3.18 几乎相同。但第 4 章 v4 的 `ex_ret_instr / n = 7176`、CPI 0.358，说明后端每周期退休 2.79 条指令，接近 4-wide 解码器的供给上限。**此时解码器成为瓶颈，Op Cache 溢出转化为 Frontend Bound**。

&emsp;&emsp;两者的区别不在 Op Cache miss 率（84% vs 99%），而在**后端消费速率是否超过解码器供给能力**。判据是 `de_dis_cops / cycles` 与 4 的关系：

| `de_dis_cops / cycles` | Op Cache 溢出时 | 判读 |
|---|---|---|
| **< 3.5** | Op Cache miss 高，但分发速率低于解码器上限 | **后端受限**——前端路径无关，优化应该转向后端 |
| **3.5–4.0** | Op Cache miss 高，分发速率接近解码器上限 | **前端接近瓶颈**——减少代码足迹可能有效 |
| **≈ 4.0** | Op Cache miss 高，分发速率被 4-wide 解码器锁死 | **真正的 Frontend Bound**——必须减少代码足迹或增加每指令工作量 |

&emsp;&emsp;本章五个版本全部落在第一行（3.18–3.24），**Op Cache 溢出是“前端路径的切换”，不是“前端瓶颈的出现”**。这是一个比“Op Cache 是 4096 uop 容量”更精细的判据——**规格数字告诉你容量上限，`de_dis_cops / cycles` 告诉你溢出之后前端是否真的成为瓶颈**。

### 5.5 总结

&emsp;&emsp;综合第 4 章 v4 和本章五个版本的数据，5600X 上 Frontend Bound 的下钻顺序可以精确化为三步：

**第一步：看 `op_cache_miss / all_op_cache_accesses`**。如果 < 5%，Op Cache 接住了循环体，前端不是瓶颈；如果 > 20%，Op Cache 部分或完全溢出，需要进入第二步。

**第二步：看 `de_dis_cops / cycles`**。如果 < 3.5，Op Cache 溢出但分发速率低于解码器上限——**后端受限，不是 Frontend Bound**；如果 ≥ 3.5，解码器接近或成为瓶颈，进入第三步。

**第三步：看 `ex_ret_instr / n` 和循环体结构**。如果循环体很大（> 2000 条 macro-op）且后端消费速率高，前端优化应转向**减少代码足迹**（拆分热点函数、避免巨型循环展开）；如果循环体不大但 `de_dis_cops / cycles` 仍然接近 4，说明瓶颈在**指令密度**（每条指令做的工作太少），优化应转向**用更宽的 SIMD 或 FMA 提高每指令工作量**。

&emsp;&emsp;**两步判据合起来，避免了一个常见误判**：看到 `op_cache_miss = 84%` 就认为“需要减少代码足迹”，但 `de_dis_cops / cycles = 3.18 < 3.5` 说明后端才是瓶颈——减少代码足迹不会改善性能，优化应该转向后端执行资源。**Op Cache miss 高只是“前端走了慢路径”，不是“前端拖慢了整体”**；只有分发速率接近解码器上限时，前端的慢路径才转化为整体的瓶颈。

&emsp;&emsp;**“Op Cache 溢出”是一个测量事实，“Frontend Bound”是一个瓶颈判断**。两者之间的等号不成立，只有当后端消费速率超过解码器供给能力时，前者才转化为后者。**判断这个转化是否发生，只需要一个比值：`de_dis_cops / cycles` 与 4 的关系**——这是本章五个版本给出 3.18–3.24 的稳定数据之后，最有诊断价值的结论。


## 6. 分层缓存：32MB 统一 L3 与内存时钟域

&emsp;&emsp;**实验设计**。本章要回答的具体问题是：**5600X 的随机访存延迟在什么工作集大小上从“L3 供给”切换为“DRAM 延迟”**。规格书上 32MB 统一 L3 是理论上限，但实际可用边界受 set 冲突、Infinity Fabric 时钟域和物理页布局影响，只有测量能给出可用判据。

&emsp;&emsp;workload 用**随机指针追逐**：构建 `next` 数组，`next[perm[i]] = perm[(i+1) % m]`，其中 `perm` 是 `[0, m)` 的随机排列。从 `perm[0]` 开始，每次 `idx = next[idx]`，访问序列在物理地址空间里随机分布，硬件预取器完全无法预测。`m = sz_mb × 1024 × 1024 / 8` 个 `long`。迭代数默认等于 `m`，保证完整遍历一遍工作集。每个工作集先 warmup 遍历一遍，把数据从 DRAM 拉到缓存层级，然后正式测量——测的是“热工作集”的访问延迟，不是冷缓存第一次访问的延迟。

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

&emsp;&emsp;**测量条件**：EPP = performance，`taskset -c 2` 绑单核，每个事件独立一个 `-e`。四个工作集（16 / 24 / 32 / 40 MB）分别跑，采集 `cycles`、`ex_ret_instr`、`cache-references`、`cache-misses`、`ls_mab_alloc.load_store_allocations`。判据不是 `cache-misses` 的绝对值，而是 **`ns/access` 的台阶**——它直接反映每次随机访问的平均延迟，跨过 L3/DRAM 边界时会跳变。

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





<svg width="760" height="330" viewBox="0 0 760 330" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica,Arial,sans-serif">
  <rect x="8" y="8" width="744" height="314" fill="#fafafa" stroke="#ccc" rx="6"/>
  <text x="30" y="34" font-size="13" font-weight="bold" fill="#333">逐 1 MB 扫描：L3 有效边界不是 32 MB，而是 22–33 MB 的过渡带</text>
  <line x1="80" y1="265" x2="720" y2="265" stroke="#333"/>
  <line x1="80" y1="265" x2="80" y2="55" stroke="#333"/>
  <text x="380" y="302" font-size="10" fill="#555">工作集（MB）→</text>
  <text x="20" y="165" font-size="10" fill="#555" transform="rotate(-90 20,165)">ns / access</text>

  <rect x="80" y="55" width="264" height="210" fill="#c8e6c9" opacity="0.25"/>
  <rect x="344" y="55" width="264" height="210" fill="#fff3e0" opacity="0.5"/>
  <rect x="608" y="55" width="112" height="210" fill="#ffcdd2" opacity="0.3"/>

  <text x="150" y="78" font-size="11" fill="#2e7d32" font-weight="bold">L3 驻留</text>
  <text x="150" y="94" font-size="9" fill="#2e7d32">14–16 ns</text>
  <text x="420" y="78" font-size="11" fill="#ef6c00" font-weight="bold">过渡带</text>
  <text x="420" y="94" font-size="9" fill="#ef6c00">18–27 ns</text>
  <text x="628" y="78" font-size="11" fill="#c62828" font-weight="bold">DRAM</text>
  <text x="628" y="94" font-size="9" fill="#c62828">34–41 ns</text>

  <line x1="344" y1="55" x2="344" y2="265" stroke="#ef6c00" stroke-dasharray="4,3"/>
  <text x="315" y="115" font-size="9" fill="#ef6c00">22 MB</text>
  <line x1="608" y1="55" x2="608" y2="265" stroke="#c62828" stroke-dasharray="4,3"/>
  <text x="580" y="115" font-size="9" fill="#c62828">33 MB</text>

  <polyline points="80,255 122,258 164,255 206,250 248,244 290,239 332,234 374,212 416,194 458,205 500,164 542,186 584,183 626,170 668,152 710,138"
            fill="none" stroke="#1565c0" stroke-width="2" opacity="0.4"/>
  <polyline points="80,255 122,258 164,255 206,250 248,244 290,239 332,234 374,212 416,194 458,205 500,164 542,186 584,183 626,170 668,152 710,138"
            fill="none" stroke="#1565c0" stroke-width="3" stroke-dasharray="0 8" stroke-linecap="round"/>

  <polyline points="608,144 650,123 692,111 734,160"
            fill="none" stroke="#c62828" stroke-width="3"/>

  <circle cx="80" cy="255" r="3" fill="#1565c0"/>
  <circle cx="344" cy="234" r="5" fill="#ef6c00"/>
  <circle cx="608" cy="152" r="5" fill="#c62828"/>

  <text x="60" y="272" font-size="9" fill="#333">16</text>
  <text x="330" y="280" font-size="9" fill="#333">22</text>
  <text x="595" y="280" font-size="9" fill="#333">33</text>
  <text x="700" y="280" font-size="9" fill="#333">40</text>
</svg>

&emsp;&emsp;**数据呈现出清晰的三阶段**，但两个边界与规格数字都不同：

&emsp;&emsp;**第一阶段（16–22 MB）：L3 驻留**。`ns/access` 从 14.3 缓慢升到 16.0，涨幅 12%。这段的上升不是“工作集变大导致更多 miss”，而是 **L3 set 冲突随工作集增长缓慢积累**——工作集已经覆盖了 L3 的物理容量，但每个 set 的占用还没到冲突阈值，多数访问仍在 L3 命中。

&emsp;&emsp;**第二阶段（23–32 MB）：过渡带**。`ns/access` 从 18.2 爬升到 26.2，涨幅 44%，且中间有抖动（25 MB 的 18.8 比 24 MB 的 19.5 还低，26 MB 的 23.8 比 27 MB 的 21.4 高）。**过渡带的宽度有 10 MB**——这是随机访问模式下最反直觉的特征：工作集跨过 L3 容量后，不是“一刀切”地全部落到 DRAM，而是**一部分地址 L3 命中、一部分 miss 到 DRAM，比例随工作集增长而连续变化**。23 MB 时约 15% 的访问 miss 到 DRAM，32 MB 时约 50%，`ns/access` 是这两种延迟的加权平均。

&emsp;&emsp;**第三阶段（33–40 MB）：DRAM 主导**。32→33 MB 出现最明显的单步跳变：26.22 → 34.37 ns，涨幅 **31%**——这是整个扫描中相邻两点的最大增幅。33 MB 之后 `ns/access` 在 34–41 ns 之间波动，涨幅收窄到 4% 以内。**33 MB 是“工作集几乎完全溢出 L3”的实测边界**——比规格的 32 MB 晚 1 MB，原因是：规格容量是 6 核共享的 32 MB，但单核跑随机访问时，其他核的 L3 占用可以忽略，单核可用容量接近 32 MB；加上栈、代码、页表占用的内存后，实际边界略高于 32 MB。

&emsp;&emsp;**两个反直觉的“平坦”比台阶本身更有信息量**：

- **`cache-miss%` 从 40.91% 降到 38.52%，几乎不变**。工作集从 16 MB 涨到 40 MB，`ns/access` 翻了近 3 倍，但 perf 的 `cache-misses / cache-references` 只降了 2.4 个百分点。原因：Zen 3 的 `cache-misses` 事件主要统计 **L2 miss**，不区分“L2 miss 但 L3 命中”和“L3 miss 到 DRAM”。16 MB 时所有访问都 miss L2（工作集远超 512 KB L2），但多数在 L3 命中；40 MB 时同样全部 miss L2，只是多数 miss 到 DRAM。**`cache-misses` 在这两个状态下几乎一样**——它只能告诉你“L2 没接住”，不能告诉你“L3 接住了没有”。
- **`ls_mab_alloc / m` 从 6.24 降到 4.72，也几乎不变**。每次访问分配的 MAB 条目数不随工作集变化，变化的是条目在队列里待的时间。16 MB 时 MAB 条目等 L3 返回（约 50 cycles），33 MB 时等 DRAM 返回（约 200 cycles）。**MAB 计数本身无法区分“等 L3”和“等 DRAM”，只有 `ns/access` 能区分**。

&emsp;&emsp;**同是 32 MB，两次测量差 15 ns**。第 6 章开头的 4 点采样中，32 MB 的 `ns/access` 是 41.32；逐 1 MB 扫描中，32 MB 只有 26.22。两个数字相差 15 ns（58%），远大于扫描内部的抖动。原因：4 点采样每次间隔几秒，温度/频率状态略有不同；`posix_memalign` 或 `std::vector` 每次进程启动分配不同的物理页，L3 set 映射不同；4 点采样时 32 MB 是第三个点，前面 16/24 MB 的测量已经预热了缓存，但 32 MB 的物理页恰好映射到冲突更严重的 set 上。**这个差异本身证明了单点测量的不可靠性**——只看 32 MB 一个点，你可能得出“32 MB 已经全是 DRAM 延迟”或“32 MB 还在 L3 过渡带”两个相反的结论，取决于那次运行落在物理页抽奖的哪一侧。**扫描整个 16–40 MB 范围，才能把“边界在 22–33 MB 之间”这个宽区间和“33 MB 处 31% 跳变”这个精确位置同时测出来**。

&emsp;&emsp;**对 Roofline 的衔接**：第 10 章的 Roofline 模型需要用“可持续内存带宽”作为屋顶。扫描数据给出了这个带宽的适用边界——**工作集 < 22 MB 时，屋顶应该是 L3 供给带宽（不是 DRAM 带宽）；22–33 MB 是过渡带，Roofline 的“内存屋顶”在数学上不适用（因为延迟不是一个常量，是工作集的函数）；> 33 MB 时，DRAM 带宽才是真正的屋顶**。对一个 16 MB 工作集的内核画 Roofline 并用 DRAM 带宽做屋顶，会高估内存受限程度，把实际上仍然是 L3 供给受限的内核误判为 DRAM 带宽受限。**“内存带宽”不是一个常量，它是工作集大小和缓存层级的函数**——这是第 1 章“对齐不是通用优化”在缓存层次上的延续。
## 7. 功耗与频率耦合：Curve Optimizer 改变测量基线

### 7.1 做：默认 vs. PBO + Curve Optimizer

&emsp;&emsp;**做**：在默认 BIOS 下采集多线程 workload 的 `cycles`、`instructions` 和有效频率。然后在 BIOS 中启用 PBO 并设置 Curve Optimizer 全核 -20，重测。

```bash
perf stat -e cycles,instructions,aperf,mperf \
          taskset -c 0-11 ./multithread_workload
```

&emsp;&emsp;**看数据**：Curve Optimizer 的负向偏移本质上是**降压**。5600X 的 65W TDP / 约 76W PPT / 60A TDC / 88A EDC 限制使这个效应特别明显。降压后温度下降，Precision Boost 有更多余量把频率推高。多线程负载下，CO -20 到 -30 配合 PBO boost override，可能获得 10–15% 的吞吐提升。

<svg width="760" height="235" viewBox="0 0 760 235" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica,Arial,sans-serif">
  <rect x="8" y="8" width="744" height="219" fill="#fafafa" stroke="#ccc" rx="6"/>
  <text x="30" y="32" font-size="12" font-weight="bold" fill="#333">Curve Optimizer 降压如何把功耗预算转化为频率余量</text>
  <line x1="70" y1="185" x2="720" y2="185" stroke="#333"/><line x1="70" y1="185" x2="70" y2="45" stroke="#333"/>
  <polyline points="110,165 230,128 350,98 470,76 590,62 700,56" fill="none" stroke="#1565c0" stroke-width="3.5"/>
  <polyline points="110,165 230,118 350,82 470,56 590,45 700,40" fill="none" stroke="#2e7d32" stroke-width="3.5" stroke-dasharray="7,4"/>
  <circle cx="590" cy="62" r="6" fill="#1565c0"/><circle cx="590" cy="45" r="6" fill="#2e7d32"/>
  <line x1="70" y1="98" x2="720" y2="98" stroke="#c62828" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="626" y="94" font-size="10" fill="#c62828" font-weight="bold">PPT / 温度限制</text>
  <text x="630" y="56" font-size="10" fill="#1565c0" font-weight="bold">默认</text><text x="630" y="40" font-size="10" fill="#2e7d32" font-weight="bold">CO -20</text>
  <line x1="590" y1="45" x2="590" y2="62" stroke="#333" stroke-width="2"/>
  <text x="20" y="115" font-size="10" fill="#555" transform="rotate(-90 20,115)">频率</text><text x="380" y="210" font-size="10" fill="#555">电压 →</text>
</svg>

&emsp;&emsp;**解释机制**：任何“优化前后对比”，**BIOS 的功耗/电压配置是实验设计的一部分**。如果在默认设置下采集基线，然后只改代码，得到的差异可能被功耗限制掩盖——代码确实减少了 stall，但 CPU 因为已经触及 PPT 限制而无法把节省的功耗预算转化为更高的频率。

&emsp;&emsp;**实践要点**：在 5600X 上做微基准，必须明确记录 **PBO 状态、Curve Optimizer 偏移值、PPT/TDC/EDC 限制**。如果目标是诊断代码本身的微架构瓶颈，建议固定功耗限制以减少变量。


## 8. SMT 与共享资源争抢

### 8.1 做：单核、SMT 对、全核的对比

&emsp;&emsp;**做**：用 `taskset` 将同一个 workload 分别绑定到单个物理核心、两个 SMT 线程和全部 12 个逻辑核。

```bash
taskset -c 0    perf stat -e cycles,instructions,op_cache_hit_miss ./workload
taskset -c 0,1  perf stat -e cycles,instructions,op_cache_hit_miss ./workload
taskset -c 0-11 perf stat -e cycles,instructions,op_cache_hit_miss ./workload
```

&emsp;&emsp;**看数据**：SMT 的两个线程共享物理核心的执行资源，但 Zen 3 的前端设计使争抢形态取决于 Op Cache 命中率。如果指令足迹小、能驻留 Op Cache，两个线程可以各自从 Op Cache 取指；如果 Op Cache 频繁缺失，两个线程会在 4-wide 解码器处产生更明显的竞争。

&emsp;&emsp;**解释机制**：**在 5600X 上优化 SMT 负载时，提高 Op Cache 驻留率（减少代码足迹、避免巨型函数内联）是缓解前端争抢的直接手段**。


## 9. 持续感知：eBPF 作为定向诊断的前置层

### 9.1 做：eBPF 持续采集 vs. perf 定向采样

&emsp;&emsp;**做**：在一个持续运行的服务旁路进程中，用 eBPF profiler 采集函数级 CPU 时间分布，同时用 `perf record` 做定向采样，对比两者的开销和发现的问题。

&emsp;&emsp;**看数据**：perf 的精度到函数级加内核态区分，侵入性约 2–3%；eBPF 的精度到微秒级，侵入性可低于 0.5%；uProf/VTune 的精度到指令级，侵入性 10–15%。

<svg width="760" height="250" viewBox="0 0 760 250" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica,Arial,sans-serif">
  <rect x="8" y="8" width="744" height="234" fill="#fafafa" stroke="#ccc" rx="6"/>
  <text x="30" y="32" font-size="12" font-weight="bold" fill="#333">工具三角：精度 / 侵入性 / 持续可用性</text>
  <polygon points="380,45 120,215 640,215" fill="#e3f2fd" stroke="#1565c0" stroke-width="2"/>
  <text x="368" y="37" font-size="11" font-weight="bold" fill="#1565c0">精度</text>
  <text x="90" y="235" font-size="11" font-weight="bold" fill="#1565c0">低侵入性</text><text x="600" y="235" font-size="11" font-weight="bold" fill="#1565c0">持续可用性</text>
  <circle cx="380" cy="95" r="14" fill="#c62828"/><text x="402" y="100" font-size="11" fill="#c62828" font-weight="bold">uProf / VTune</text>
  <circle cx="285" cy="175" r="14" fill="#ef6c00"/><text x="205" y="165" font-size="11" fill="#ef6c00" font-weight="bold">perf 原始事件</text>
  <circle cx="480" cy="180" r="14" fill="#2e7d32"/><text x="500" y="185" font-size="11" fill="#2e7d32" font-weight="bold">eBPF</text>
  <path d="M460,170 Q440,150 400,110" fill="none" stroke="#999" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#a3)"/>
  <path d="M310,165 Q340,140 370,105" fill="none" stroke="#999" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#a3)"/>
  <defs><marker id="a3" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#999"/></marker></defs>
</svg>

&emsp;&emsp;**解释机制**：perf 的定向采样适合“诊断已知热点”；eBPF 的持续 profiling 适合“在热点变成事故之前发现它”。合理的工作流是：eBPF 持续采集函数级分布，发现回归窗口后触发 `perf record` 和 Zen 3 原始事件采集。


## 10. Roofline：把“该优化计算还是访存”变成可计算的判断

### 10.1 屋顶线的算术

&emsp;&emsp;Roofline 模型将内核的算术强度（FLOP/Byte）作为横轴，可达到的浮点性能作为纵轴。算术强度低于拐点的内核落在斜坡上，是内存受限；高于拐点的内核落在平顶上，是计算受限。Roofline 的价值不在于精确预测，而在于**把优化方向的选择变成算术**。perf-book 将 Roofline 放在“Performance Analysis Approaches”一章，正因为它是一种**应该在动手写代码之前完成的静态判断**。

### 10.2 在 5600X 上画屋顶线：先锁定内存配置

&emsp;&emsp;在 5600X 上，**可持续带宽必须在 FCLK/UCLK/MCLK 1:1 配置下测量**，否则得到的“带宽”会被分频惩罚压低，把实际上仍然是计算受限的内核误判为内存受限。

<svg width="760" height="255" viewBox="0 0 760 255" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica,Arial,sans-serif">
  <rect x="8" y="8" width="744" height="239" fill="#fafafa" stroke="#ccc" rx="6"/>
  <text x="30" y="32" font-size="12" font-weight="bold" fill="#333">5600X 的 Roofline：内存屋顶随 FCLK/UCLK/MCLK 同步状态移动</text>
  <line x1="80" y1="210" x2="720" y2="210" stroke="#333"/><line x1="80" y1="210" x2="80" y2="50" stroke="#333"/>
  <line x1="80" y1="78" x2="290" y2="78" stroke="#1565c0" stroke-width="3.5"/><line x1="290" y1="78" x2="710" y2="78" stroke="#1565c0" stroke-width="3.5"/>
  <line x1="80" y1="118" x2="215" y2="118" stroke="#ef6c00" stroke-width="3.5"/><line x1="215" y1="118" x2="710" y2="118" stroke="#ef6c00" stroke-width="3.5"/>
  <line x1="80" y1="158" x2="155" y2="158" stroke="#c62828" stroke-width="3.5"/><line x1="155" y1="158" x2="710" y2="158" stroke="#c62828" stroke-width="3.5"/>
  <circle cx="340" cy="78" r="6" fill="#1565c0"/><circle cx="340" cy="118" r="6" fill="#ef6c00"/><circle cx="340" cy="158" r="6" fill="#c62828"/>
  <line x1="340" y1="78" x2="340" y2="158" stroke="#999" stroke-dasharray="3,3"/>
  <text x="352" y="73" font-size="10" fill="#1565c0" font-weight="bold">DDR4-3600 1:1</text>
  <text x="352" y="112" font-size="10" fill="#ef6c00" font-weight="bold">DDR4-4000 2:1</text>
  <text x="352" y="172" font-size="10" fill="#c62828" font-weight="bold">JEDEC DDR4-3200</text>
  <text x="20" y="130" font-size="10" fill="#555" transform="rotate(-90 20,130)">GFLOP/s</text><text x="370" y="235" font-size="10" fill="#555">算术强度 (FLOP/Byte) →</text>
</svg>

### 10.3 32MB L3 改变了“内存受限”的起点

&emsp;&emsp;以 LLM decode/prefill 为例：prefill 阶段算术强度高，通常计算受限；decode 阶段 batch=1、算术强度约 1 FLOP/Byte，通常内存受限。但 5600X 的 32MB L3 意味着 **decode 阶段的工作集如果小于 L3，瓶颈可能不在 DRAM 带宽，而在 L3 到核心的供给带宽**。此时 Roofline 的“内存屋顶”应该使用 L3 带宽而非 DRAM 带宽。

&emsp;&emsp;**实践要点**：“这个内核该用什么优化”不是内核的固有属性，而是（内核 × 工作负载形态 × 缓存层次 × 内存控制器配置）的函数。


## 11. 四个在 5600X 上可直接复现的闭环实验

&emsp;&emsp;每个实验的结构：**基线测量 → 预测 → 改一行/几行 → 重新测量 → 用微架构机制解释差异**。

### 11.1 False Sharing：同一缓存行上的两个写者

```c
#include <pthread.h>
#include <stdio.h>

struct counters_bad  { long a; long b; };
struct counters_good { alignas(64) long a; alignas(64) long b; };

static void *worker(void *arg) {
    long *p = (long *)arg;
    for (long i = 0; i < 100000000; i++) (*p)++;
    return NULL;
}

int main(void) {
    struct counters_bad c;
    pthread_t t1, t2;
    pthread_create(&t1, NULL, worker, &c.a);
    pthread_create(&t2, NULL, worker, &c.b);
    pthread_join(t1, NULL); pthread_join(t2, NULL);
    printf("%ld %ld\n", c.a, c.b);
    return 0;
}
```

```bash
perf stat -e cache-misses,cache-references,cycles,instructions ./false_sharing
```

&emsp;&emsp;**预测**：对齐后 `cache-misses` 应显著下降，`cycles` 应下降，`instructions` 基本不变。

<svg width="760" height="200" viewBox="0 0 760 200" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica,Arial,sans-serif">
  <rect x="8" y="8" width="744" height="184" fill="#fafafa" stroke="#ccc" rx="6"/>
  <text x="30" y="32" font-size="12" font-weight="bold" fill="#333">False sharing：不同核心写同一缓存行的不同字节</text>
  <defs><marker id="red" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#c62828"/></marker></defs>
  <text x="30" y="65" font-size="11" fill="#333" font-weight="bold">未填充</text>
  <rect x="130" y="50" width="480" height="40" fill="#ffcdd2" stroke="#c62828" stroke-width="2"/><rect x="130" y="50" width="240" height="40" fill="#ef9a9a" opacity="0.7"/>
  <text x="220" y="75" font-size="11" fill="#7f0000" font-weight="bold">core0 写</text><text x="420" y="75" font-size="11" fill="#7f0000" font-weight="bold">core1 写</text>
  <path d="M370,50 Q400,30 430,50" fill="none" stroke="#c62828" stroke-width="2" marker-end="url(#red)"/>
  <path d="M430,90 Q400,110 370,90" fill="none" stroke="#c62828" stroke-width="2" marker-end="url(#red)"/>
  <text x="630" y="75" font-size="10" fill="#c62828" font-weight="bold">同一行</text>
  <text x="30" y="145" font-size="11" fill="#333" font-weight="bold">对齐后</text>
  <rect x="130" y="130" width="230" height="40" fill="#c8e6c9" stroke="#2e7d32" stroke-width="2"/><text x="200" y="155" font-size="11" fill="#1b5e20" font-weight="bold">core0 写</text>
  <rect x="380" y="130" width="230" height="40" fill="#c8e6c9" stroke="#2e7d32" stroke-width="2"/><text x="450" y="155" font-size="11" fill="#1b5e20" font-weight="bold">core1 写</text>
  <text x="640" y="155" font-size="10" fill="#2e7d32" font-weight="bold">各行独占</text>
</svg>

&emsp;&emsp;**解释机制**：缓存一致性协议以缓存行为单位。核心 0 写入字节 0 时，核心 1 缓存中的该行被置为 Invalid；核心 1 写入字节 8 时，需要重新获取所有权。即使访问的字节没有重叠，一致性流量仍然以整行为粒度发生。

### 11.2 Op Cache 驻留率：代码足迹对 Frontend Bound 的影响

```c
// A：循环体小，能驻留 Op Cache
for (size_t i = 0; i < n; i++) out[i] = in[i] * 2 + 1;

// B：手工展开 + 巨型内联，超出 Op Cache 容量
for (size_t i = 0; i < n; i += 8) {
    out[i+0] = in[i+0] * 2 + 1; out[i+1] = in[i+1] * 2 + 1;
    /* ... 反复展开 ... */
}
```

```bash
perf stat -e cycles,instructions,op_cache_hit_miss,\
          de_dis_cops_from_decoder ./opcache_bench
```

&emsp;&emsp;**预测**：A 的 `op_cache_hit` 比例高、Frontend Bound 低；B 的 `op_cache_miss` 上升，`de_dis_cops_from_decoder` 成为前端供给的主要来源。

&emsp;&emsp;**解释机制**：在 5600X 上，“前端慢”不等于“分支预测差”或“取指带宽不够”，它首先等于“Op Cache 没接住”。

### 11.3 分支排序：Bad Speculation 是输入顺序的函数

```c
for (size_t i = 0; i < n; i++)
    if (data[i] >= 128) sum += data[i];
```

```bash
perf stat -e branches,branch-misses ./branch_bench_unsorted
perf stat -e branches,branch-misses ./branch_bench_sorted
```

&emsp;&emsp;**预测**：随机输入的 `branch-misses / branches` 应显著高于排序输入。不可预测输入下误预测率可达 25% 量级；排序后相同方向的长序列使预测器容易学习，误预测率骤降。

### 11.4 预取 + FCLK 同步：Memory Bound 的两个独立杠杆

```bash
# 场景 1：JEDEC DDR4-3200，FCLK 1600（1:1）
perf stat -e cycles,instructions,ls_mab_alloc,l3_misses_from_l2_misses ./stream_triad

# 场景 2：DDR4-3600，FCLK 1800（1:1）
# 场景 3：DDR4-4000，UCLK 2:1（分频）
# 每个场景分别跑 baseline 和软件预取版本
```

&emsp;&emsp;**预测**：如果瓶颈在内存延迟而非带宽，FCLK 1:1 同步的收益可能大于软件预取；如果瓶颈在带宽饱和，预取可能只改变 L1/L2 命中率而不降低 `ls_mab_alloc` 的排队压力。


## 12. 指令集扩展：从 AVX2 到 AVX-512 与 VNNI 的测量边界

&emsp;&emsp;第 10 章的 Roofline 给出了一个基本判断：向量化的收益取决于内核是否处于计算受限区域。但在具体硬件上，“向量化”不是二元状态——它是**指令集宽度、数据通路宽度和降频行为**的联合函数。5600X 的 AVX2 是 256-bit 原生数据通路，不存在拆分成本，可以作为效率基线；而 Zen 4 的 AVX-512 通过 **double-pumping 256-bit 数据通路**实现：512-bit FMA 被拆分为两个 256-bit μop，每周期退休一个 512-bit FMA，峰值 FLOP 上限与 AVX2 相同。

### 12.1 做：AVX2 基线与 AVX-512 的“指令数下降 vs. 周期数下降”对比

&emsp;&emsp;**做**：取一个浮点乘加密集型内核，分别编译标量、AVX2（256-bit）和 AVX-512（在 Zen 4 上走 double-pumped 路径）三个版本。在 5600X 上跑前两个，在 Zen 4 平台上补充第三个。采集：

```bash
perf stat -e cycles,instructions,ex_ret_ops ./vec_bench
```

&emsp;&emsp;**看数据**：如果 AVX-512 版本的 `instructions` 相对 AVX2 显著下降，但 `cycles` 下降幅度不成比例，说明收益被 **256-bit 数据通路的双周期执行**抵消。如果 `cycles` 同步下降，说明收益来自**前端供给压力的降低**——更少的指令让 Op Cache 和分发级更轻松。

&emsp;&emsp;**5600X 的基线意义**：在 5600X 上测得的 AVX2 每周期有效 FLOP，可以作为判断“更宽指令集是否真正转化为计算吞吐”的算术锚点。

### 12.2 做：VNNI 的收益边界——当 INT8 算力被内存带宽吃掉

&emsp;&emsp;Zen 4 的 AVX-512 VNNI 将 INT8 点积吞吐提升到每周期 64 次操作（4 INT8 ops/lane × 8 lanes × 2 FMA）。AMD 官方宣称 INT8 多线程性能提升可达 2.5 倍。但这是**计算受限**条件下的结论。

&emsp;&emsp;**做**：在一个 INT8 矩阵乘内核上，分别在 AVX2 和 AVX-512 VNNI 路径下测量。工作集设置在两个量级：(a) L2 驻留；(b) 超过 32MB L3，走向 DRAM。采集 `instructions`、`cycles` 和 `ls_mab_alloc`。

&emsp;&emsp;**看数据**：在 L2 驻留的工作集上，VNNI 版本的 `cycles` 应显著低于 AVX2。一旦工作集超过 L3，VNNI 版本的 `cycles` 可能不再下降，甚至与 AVX2 持平——此时 `ls_mab_alloc` 高位运行，**INT8 算力的提升被内存供给瓶颈淹没**。VNNI 只改变了计算屋顶的高度，没有改变内存屋顶的高度。


## 13. 安全缓解措施：当“安全”成为微架构实验的隐藏变量

&emsp;&emsp;第 2 章讨论了频率和热状态作为测量噪音源。在现代 CPU 上，还有一个同等量级的变量：**操作系统和微码为抵御瞬态执行攻击而启用的缓解措施**。这些措施直接改变分支预测和推测执行的行为，从而改变 Top-Down 的分布。

### 13.1 做：缓解措施开启与关闭时的 Top-Down 对比

&emsp;&emsp;**做**：在一个包含间接分支调用和密集系统调用的 workload 上，分别采集默认内核配置（缓解措施开启）和受控环境中关闭 `pti`、`ibrs`/`retpoline`、`ssbd` 后的数据（仅用于实验诊断）。采集 Top-Down Level 1 和 `branch-misses`。

&emsp;&emsp;**看数据**：关闭缓解措施后，如果 **Bad Speculation 的比例显著下降**，说明原负载的一部分“错误投机”实际上是由 **retpoline 或 IBRS 引入的间接分支序列化**造成的。retpoline 将间接分支替换为 thunk，实质上**使间接分支预测失效**，而 IBRS 则限制间接分支的推测范围。这部分代价在 Top-Down 中被归入 Bad Speculation，但它与输入顺序不可预测导致的 Bad Speculation 有本质区别：后者可以通过改变数据顺序缓解，前者只能通过**减少间接分支的数量**来降低。

### 13.2 在 5600X 上建立“缓解措施开启”的基线快照

```bash
grep -r . /sys/devices/system/cpu/vulnerabilities/ 2>/dev/null | sort
```

&emsp;&emsp;**实践要点**：如果 `ibrs_enabled` 或 retpoline 处于开启状态，且 workload 包含密集的间接调用（C++ 虚函数、回调、解释器 dispatch），Top-Down 中 Bad Speculation 的基线可能被系统性抬高。在做任何“分支优化”之前，先确认这部分 Bad Speculation 是**程序可优化的**，还是**缓解措施施加的固定成本**。后者的正确策略是在算法层面**减少间接调用的频率**，而不是试图“消除分支”。


## 14. 工作负载形态：LLM 推理如何改变 Roofline 的判断前提

&emsp;&emsp;LLM 推理引入了一类新的工作负载形态：**权重矩阵的体量远超缓存，而每次推理的有效计算密度极低**。Decode 阶段的算术强度约为 1 FLOP/Byte，是典型的内存带宽受限工作负载。

### 14.1 做：INT8 量化对“内存屋顶”的抬升

&emsp;&emsp;如果权重是 FP16（2 字节/参数），INT8（1 字节/参数）在同等 DRAM 带宽下将有效参数供给速率翻倍，把 Roofline 的内存斜坡整体上移。Decode 的算术强度从 1 FLOP/Byte 提升到 2 FLOP/Byte。

&emsp;&emsp;**做**：在 5600X 上，取一个中小规模的语言模型或 Transformer 编码器，分别在 FP16 和 INT8 权重下运行单序列 decode，测量 tokens/s 和 `ls_mab_alloc`。工作集应设置到明显超过 32MB L3。

&emsp;&emsp;**看数据**：INT8 版本应表现出更高的 tokens/s，但如果提升幅度**显著低于 2 倍**，说明 `ls_mab_alloc` 或 DRAM 带宽尚未被完全利用，瓶颈可能转移到了**反量化计算**或**低精度矩阵乘的指令开销**上。INT8 的收益上限是“内存屋顶翻倍”，但实际收益取决于计算路径是否能够在不增加瓶颈的情况下消费这翻倍的供给。

### 14.2 做：权重布局与访问粒度

&emsp;&emsp;LLM 推理每次 decode 步骤读取一整套权重矩阵，如果权重按不利于缓存行利用的方式布局，有效带宽会被浪费在“加载了但只用了部分字节”上——这与 11.2 节的 AoS 问题同构，但发生在更大的数据规模上。

&emsp;&emsp;**做**：在 5600X 上，对比标准行优先布局和按输出通道分组的布局，用 `perf stat -e cache-misses,cache-references,ls_mab_alloc` 采集。如果分组布局下 `cache-misses / cache-references` 显著下降，说明原始布局的“连续读”在 decode 阶段的细粒度访问模式下并没有转化为高有效载荷。

&emsp;&emsp;**5600X 上的定位**：32MB L3 对于小型模型或量化后的权重可能仍然容纳一部分，因此 LLM 推理的 Memory Bound 在 5600X 上可能首先表现为 **L3 到核心的供给瓶颈**，而非 DRAM 带宽饱和。权重布局和 L3 驻留策略的优先级可能高于 DRAM 级优化。


## 15. 优化技术选型：把诊断结果接到正确的技术上

&emsp;&emsp;第 4 章到第 14 章建立了从 PMU 事件到 Top-Down 路由、Roofline 边界判断、再到指令集/安全缓解/工作负载形态等变量的诊断链路。但诊断本身不产生性能。**从“知道瓶颈在哪里”到“让瓶颈消失”之间，还隔着一层优化技术选型**——选型的依据，是当前测量数据指向的微架构原因，与这项技术要解决的问题是否匹配。

### 15.1 前端优化：让 Op Cache 接住，让代码布局不浪费槽位

&emsp;&emsp;5600X 的前端是“4-wide 解码器 + 8-wide Op Cache → 6-wide dispatch”的双路径。前端优化的第一原则因此非常具体：**尽量让热点代码驻留在 Op Cache 中；如果做不到，至少不要让代码布局本身制造额外的前端停顿**。

&emsp;&emsp;**做**：对同一个循环，分别用 `-O2` 和 `-O2 -march=znver3` 编译，采集 `op_cache_hit_miss`、`de_dis_cops_from_decoder` 和 `ic_tag_hit_miss`。如果 `op_cache_hit` 比例高、`de_dis_cops_from_decoder` 不是主要来源，前端优化应转向**减少代码足迹**（避免巨型函数内联、控制热点函数的指令数量）。如果 `op_cache_miss` 高、`ic_tag_hit_miss` 也高，问题可能不在 Op Cache 容量，而在**代码布局导致 I-cache 行利用率下降**——此时基本块对齐、循环体移位到单条 I-cache 行内、函数分组才开始有意义。

&emsp;&emsp;**5600X 上的边界**：热点代码足迹超过 Op Cache 容量时，前端瓶颈的形态会从“分支预测差”切换为“供给带宽不足”。此时更有效的手段可能是**函数拆分**：把冷路径从热点函数中剥离，让热路径独占 Op Cache。

### 15.2 后端优化：Core Bound 与 Memory Bound 的分野

&emsp;&emsp;Core Bound 的优化对象是执行端口、依赖链和指令延迟；Memory Bound 的优化对象是缓存层次、预取和带宽。

&emsp;&emsp;**做（Core Bound）**：对一个受依赖链限制的循环，采集 `ex_ret_ops` 和 `cycles`。然后手动展开循环、打断累加器依赖链、或把除法替换为乘法倒数。如果展开后 `instructions` 基本不变而 `cycles` 下降，说明原循环的槽位被依赖链延迟阻塞。如果 `cycles` 不变，瓶颈可能在某个执行端口的吞吐上限。

&emsp;&emsp;**做（Memory Bound）**：`ls_mab_alloc` 是区分“延迟问题”和“带宽问题”的关键事件。如果 `ls_mab_alloc` 持续高位而实际 DRAM 带宽未饱和，瓶颈在 **MAB 排队**，此时增加预取深度可能反而加剧排队；如果 `ls_mab_alloc` 高且带宽计数器接近上限，瓶颈在**带宽饱和**，预取和 Non-Temporal Store 的收益会被压缩。

&emsp;&emsp;**5600X 上的边界**：32MB L3 把“内存受限”的起点推后了。在 L3 仍能容纳工作集时过早使用 DRAM 级优化（如 Non-Temporal Store）是无效的。只有工作集明确超过 32MB 时，DRAM 带宽导向的优化才进入有效区间。

### 15.3 错误投机优化：降低 Bad Speculation 的“可避免部分”

&emsp;&emsp;**做**：用 `perf stat -e branches,branch-misses` 定位误预测率最高的区域。然后尝试：(a) 用查表替换分支；(b) 用算术或条件传送替换分支；(c) 用 PGO/BOLT 改善预测器的学习条件。如果 `branch-misses / branches` 显著下降而 `cycles` 也下降，说明误预测是实际的槽位消耗来源。如果 `branch-misses` 下降但 `cycles` 不变，说明 Bad Speculation 在 Top-Down 中的占比本来就不高。

&emsp;&emsp;**5600X 上的边界**：Zen 3 的分支预测器对规则模式的学习能力较强。用查表替换分支的收益取决于**查表本身是否引入新的缓存压力**——如果查表数组超出 L1D 工作集，错误投机优化可能把 Bad Speculation 的槽位节省换成了 Memory Bound 的增加。同时，第 13 章提醒：先确认 Bad Speculation 中有多少是**缓解措施施加的固定成本**，再决定优化动作。

### 15.4 内存访问优化：数据布局是 Memory Bound 的“源头杠杆”

&emsp;&emsp;数据布局优化是唯一一类能在不改算法、不加指令的前提下，直接改变“每条缓存行承载多少有用字节”的技术。perf-book 第二版新增的**结构体字段重排**（field reordering）逻辑是：按访问频率和访问模式重新排列字段，让热字段紧凑地落在同一条缓存行内。

&emsp;&emsp;**做**：取一个遍历结构体数组、只使用少数字段的循环。先采集 `l1d_cache_misses` 和 `ls_mab_alloc`。然后按访问频率把热字段前置并连续排列，冷字段后置。如果热字段和冷字段原本跨越缓存行边界，重排后应观察到 `l1d_cache_misses` 和 `ls_mab_alloc` 同时下降。

&emsp;&emsp;**5600X 上的边界**：Zen 3 的 L1D 组相联结构和预取器行为会与字段排列方式交互。在某些访问模式下，过于紧凑的字段排列可能让多个高频访问流映射到同一 L1D set，产生冲突缺失。做字段重排后，除了看 miss 率，还应检查冲突相关的事件，确认优化没有把容量缺失换成了冲突缺失。

### 15.5 多线程优化：伪共享只是起点，同步与扩展性是主体

&emsp;&emsp;第 11.1 节的 False Sharing 展示的是缓存一致性协议在写共享粒度上的代价。多线程优化的完整版图还包括**同步开销、锁争用、线程数扩展性、任务调度**。

&emsp;&emsp;**做**：对一个包含共享计数器的多线程程序，分三步改动并测量：(a) 用 `alignas(64)` 消除伪共享；(b) 把全局原子计数器改为**每线程本地计数 + 最终归约**；(c) 用 `perf c2c` 确认是否还存在跨核缓存行弹跳。

```bash
perf c2c record -a ./multithread_bench
perf c2c report --stdio
```

&emsp;&emsp;**看数据**：伪共享消除后，`cache-misses` 下降但 `cycles` 可能仍然高，因为**原子操作本身的序列化开销**和**跨核缓存行所有权迁移**是伪共享之外的另一层成本。每线程本地计数把写操作转移到私有缓存行，归约阶段只发生一次跨核交互，应进一步降低 `cycles`。如果 `perf c2c` 仍然报告高比例的 HITM 事件，说明共享写入的粒度问题没有完全解决。

&emsp;&emsp;**5600X 上的边界**：单 CCD、6 核共享 32MB L3，跨核通信延迟基线相对均匀。线程亲和性（`taskset` 绑定到同一 CCX 内的核心）对同步密集型负载的收益较为可预测。

### 15.6 编译器与构建配置：最便宜的优化层，也是最容易被忽略的测量变量

&emsp;&emsp;编译器优化不改变算法，但改变机器码的指令选择、内联决策、循环结构和分支布局。在 5600X 上，至少有三个变量：目标架构标志（`-march=znver3`）、优化级别（`-O2`/`-O3`）和**基于剖析的优化（PGO）**。

```bash
gcc -O2 -march=znver3 workload.c -o w_O2_znver3
gcc -O3 -march=znver3 workload.c -o w_O3_znver3
gcc -O2 -march=znver3 -fprofile-generate workload.c && ./a.out && \
gcc -O2 -march=znver3 -fprofile-use workload.c -o w_pgo
```

&emsp;&emsp;**看数据**：`-march=znver3` 启用 AVX2/FMA 和 Zen 3 特定的调度模型。PGO 的收益通常体现在 **Bad Speculation 和 Frontend Bound 的下降**——PGO 重新排列基本块，把热路径放在一起，减少了 taken branch 和 I-cache 行的浪费。如果 PGO 后 `branch-misses` 下降而 `cycles` 下降幅度有限，说明分支误预测在原负载中本来就不是主导瓶颈。

&emsp;&emsp;**5600X 上的边界**：`-march=znver3` 要求 GCC 10.3+ 或相应版本的 Clang。另一个容易忽略的变量是**内联策略**：过度的 `always_inline` 可能迅速推高热点代码的 Op Cache 足迹，把 Frontend Bound 从“解码器受限”推向“Op Cache 缺失”。

### 15.7 计算优化：SIMD 与算术强度，在 Roofline 的斜坡上做正确的事

&emsp;&emsp;计算优化（SIMD 向量化、FMA、减少慢速浮点）只应该在 Roofline 判断为计算受限时进入视野。

&emsp;&emsp;**做**：对标量浮点循环，先用 `-O3 -march=znver3 -ffast-math` 自动向量化版本测量，再用手写 AVX2 intrinsics 版本测量，采集 `instructions` 和 `cycles`，计算 IPC 和实际 FLOP/cycle。如果手写 AVX2 版本的 `instructions` 显著下降但 `cycles` 下降幅度不成比例，说明循环已处于内存受限斜坡。如果 `cycles` 与 `instructions` 同步下降，且 IPC 接近执行宽度上限，说明计算优化命中了 Core Bound。

&emsp;&emsp;**5600X 上的边界**：Zen 3 的 AVX2 是 256-bit 原生执行，不存在 AVX 降频问题。但 **6 核共享 32MB L3 意味着多线程 SIMD 负载很容易把带宽推向饱和**。在单核上有效的向量化内核，在 6 核同时运行时可能从 Core Bound 切换为 Memory Bound；此时继续增加 SIMD 宽度或展开因子，收益会迅速递减。计算优化的前提是**在目标线程数下重新做一次 Roofline 判断**。同时，第 12 章提醒：更宽的指令集（AVX-512/VNNI）只抬高计算屋顶，不改变内存屋顶；当内核落在内存斜坡上时，指令集扩展的收益不可见。


## 16. 性能优化是“假设—测量—修正”的循环，不是技术清单

&emsp;&emsp;回到开篇的对齐实验，用 5600X 的完整工具链重走一遍：

1. **先锁定测量条件**：确认 FCLK/UCLK/MCLK 同步状态，记录 PBO 和 Curve Optimizer 配置，让机器进入热稳态（第 2、7 章）。
2. **记录缓解措施状态**：检查 Spectre 缓解措施的启用情况，作为解释 Bad Speculation 的上下文（第 13 章）。
3. **采集原始事件组**：`op_cache_hit_miss`、`de_dis_cops_from_decoder`、`ls_mab_alloc`、`dispatch_stalls`，而不是直接依赖 Intel 式的 `--topdown` 自动输出（第 3、4 章）。
4. **前端嫌疑高**：先看 Op Cache 命中率，再看 4-wide 解码器压力，最后考虑分支预测（第 5、15.1 节）。
5. **Memory Bound 嫌疑高**：先用 32MB L3 容量判断工作集位置，再用 FCLK 同步状态判断延迟/带宽。预取和内存控制器配置是两个独立杠杆（第 6、10、11.4 节）。
6. **工作负载是 LLM 推理**：decode 阶段默认在内存斜坡上；先用量化（INT8/INT4）抬高内存屋顶，再考虑权重布局和 L3 驻留，最后才看 SIMD 宽度（第 12、14 章）。
7. **做最小改动，先预测哪个事件会变、变成什么方向，再测量**。预测失败时的信息量通常更大——它暴露了你对微架构模型的某个假设需要修正。

&emsp;&emsp;perf-book 的方法论收敛于一句话：**优化不是在代码里堆技术，而是在测量数据和微架构模型之间建立可验证的对应关系**。Top-Down 提供槽位级别的路由，Roofline 提供算术边界，PMU 提供可重复的观测。5600X 的具体实现——Op Cache/解码器双路径、32MB 统一 L3、Infinity Fabric 时钟域、Precision Boost 功耗行为——决定了这条闭环在这台机器上的具体锚点和判据。而指令集扩展、安全缓解措施和工作负载形态，是三个必须在测量设计中显式控制的变量：它们不改变方法论，但改变数据的解读方式。

&emsp;&emsp;每一次改动都应预期“哪个指标会变、变成什么方向”，然后用下一次测量来证实或否定。修正假设、重新测量、更新模型，就是性能优化工作本身的形态。