# 现代 CPU 性能优化

**——以 AMD Ryzen 5 5600X 为实验平台的 PMU、Top-Down 与优化技术实践手册**


## 1. 为什么优化经验经常是错的

&emsp;&emsp;先做两个实验。

&emsp;&emsp;**实验 A**：写一个求和循环，累加到一个 `long` 局部变量中并打印。保持逻辑不变，仅将数组起始地址对齐到 64 字节边界，重新编译。

```c
long sum_v1(const int *a, size_t n) {
    long s = 0;
    for (size_t i = 0; i < n; i++) s += a[i];
    return s;
}
// 版本 2 完全相同，仅调用侧 posix_memalign((void**)&buf, 64, n*sizeof(int));
```

&emsp;&emsp;在 5600X 上，两个版本的差异可能远小于你的预期——对齐本身不是“优化”，只是一个需要被测量验证的假设。

&emsp;&emsp;**实验 B**：两个线程各递增一个 `long` 计数器，不加填充，运行 1 亿次。

```c
struct counters_bad  { long a; long b; };          // 同一 64B 行
struct counters_good { alignas(64) long a; alignas(64) long b; };
```

&emsp;&emsp;`perf stat -e cache-misses,cache-references` 显示缓存未命中率超过 15%。加一行 `alignas(64)`，吞吐量翻倍。改动不触及任何算法或数据结构语义。

&emsp;&emsp;这两个实验指向同一个结论：**现代 CPU 的性能不由“执行了多少条指令”决定，而由“每个周期提供的流水线槽位（pipeline slot）如何被消耗”决定**。Bakhvalov 在 *Performance Analysis and Tuning on Modern CPUs*（下称 perf-book）中将这一视角系统化为 Top-Down Microarchitecture Analysis Methodology（TMA）：它不要求开发者预先掌握所有 PMU 事件的语义，就能把瓶颈路由到前端、后端、错误推测或退休四个互斥类别之一。

<svg width="760" height="230" viewBox="0 0 760 230" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica,Arial,sans-serif">
  <rect x="8" y="8" width="744" height="214" fill="#fafafa" stroke="#ccc" rx="6"/>
  <text x="30" y="35" font-size="13" font-weight="bold" fill="#333">一个周期内 6 个流水线槽位的四种去向</text>
  <g transform="translate(30,55)">
    <rect x="0" y="0" width="68" height="42" fill="#4caf50" stroke="#2e7d32" stroke-width="1.5"/>
    <rect x="78" y="0" width="68" height="42" fill="#4caf50" stroke="#2e7d32" stroke-width="1.5"/>
    <rect x="156" y="0" width="68" height="42" fill="#ff9800" stroke="#e65100" stroke-width="1.5"/>
    <rect x="234" y="0" width="68" height="42" fill="#f44336" stroke="#b71c1c" stroke-width="1.5"/>
    <rect x="312" y="0" width="68" height="42" fill="#f44336" stroke="#b71c1c" stroke-width="1.5"/>
    <rect x="390" y="0" width="68" height="42" fill="#9e9e9e" stroke="#616161" stroke-width="1.5"/>
    <text x="14" y="27" font-size="10" fill="#fff" font-weight="bold">Retire</text>
    <text x="92" y="27" font-size="10" fill="#fff" font-weight="bold">Retire</text>
    <text x="164" y="27" font-size="10" fill="#fff" font-weight="bold">Bad Spec</text>
    <text x="242" y="27" font-size="10" fill="#fff" font-weight="bold">FE Bound</text>
    <text x="320" y="27" font-size="10" fill="#fff" font-weight="bold">BE Bound</text>
    <text x="405" y="27" font-size="10" fill="#fff" font-weight="bold">空闲</text>
  </g>
  <g transform="translate(30,130)">
    <rect x="0" y="0" width="12" height="12" fill="#4caf50"/><text x="18" y="11" font-size="10" fill="#333">Retiring</text>
    <rect x="90" y="0" width="12" height="12" fill="#ff9800"/><text x="108" y="11" font-size="10" fill="#333">Bad Speculation</text>
    <rect x="220" y="0" width="12" height="12" fill="#f44336"/><text x="238" y="11" font-size="10" fill="#333">FE / BE Bound</text>
    <rect x="350" y="0" width="12" height="12" fill="#9e9e9e"/><text x="368" y="11" font-size="10" fill="#333">核心提供的槽位</text>
  </g>
  <text x="30" y="200" font-size="10" fill="#666">TMA 把这 6 个槽位的去向互斥地归入四类；总和即 Level 1 百分比。</text>
</svg>

&emsp;&emsp;本文的目标不是罗列优化技巧，而是建立一条可复现的闭环：**先做实验，再看 PMU 输出，然后解释微架构机制，最后用重新测量来验证假设**。所有命令、代码、事件名都以 AMD Ryzen 5 5600X 为实验环境给出——它是 Zen 3 架构、6 核 12 线程、单 CCD、32MB 统一 L3，Linux perf 对其有较完整的 vendor event 支持，适合作为“把方法论落到具体机器上”的样本。


## 2. 测量的第一课：基准在测量微架构状态，而不是“计算量”

### 2.1 频率不是常量

&emsp;&emsp;连续运行同一个 CPU 密集型循环 5 次，中间不插入其他负载。

```bash
for i in 1 2 3 4 5; do
    perf stat -e cycles,instructions,aperf,mperf ./workload 2>&1 | tail -4
done
```

&emsp;&emsp;5600X 的基础频率 3.7 GHz，最大 boost 4.6 GHz，TDP 65W，默认 PPT 约 76W。Precision Boost 在温度较低时会推高短时频率，温度上升后回落。第一次运行可能快于后续，不是“缓存冷了”，而是频率在漂移。

<svg width="760" height="235" viewBox="0 0 760 235" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica,Arial,sans-serif">
  <rect x="8" y="8" width="744" height="219" fill="#fafafa" stroke="#ccc" rx="6"/>
  <text x="30" y="32" font-size="12" font-weight="bold" fill="#333">持续负载下的频率—功耗—温度耦合</text>
  <line x1="55" y1="190" x2="715" y2="190" stroke="#333"/>
  <line x1="55" y1="190" x2="55" y2="45" stroke="#333"/>
  <polyline points="55,68 135,71 215,84 295,102 375,115 455,122 535,125 615,126 695,126" fill="none" stroke="#1565c0" stroke-width="3.5"/>
  <polyline points="55,152 135,142 215,126 295,112 375,104 455,100 535,99 615,99 695,99" fill="none" stroke="#ef6c00" stroke-width="3" stroke-dasharray="7,4"/>
  <line x1="55" y1="108" x2="715" y2="108" stroke="#c62828" stroke-width="1.5" stroke-dasharray="4,4"/>
  <text x="60" y="103" font-size="10" fill="#c62828">▲ 温度墙 / PPT</text>
  <text x="590" y="118" font-size="10" fill="#1565c0" font-weight="bold">频率</text>
  <text x="590" y="92" font-size="10" fill="#ef6c00" font-weight="bold">功耗</text>
  <circle cx="135" cy="71" r="4" fill="#1565c0"/><circle cx="135" cy="142" r="4" fill="#ef6c00"/>
  <text x="105" y="215" font-size="10" fill="#555">冷启动</text>
  <text x="560" y="215" font-size="10" fill="#555">热稳态</text>
  <line x1="135" y1="190" x2="135" y2="195" stroke="#999"/><line x1="600" y1="190" x2="600" y2="195" stroke="#999"/>
  <text x="20" y="125" font-size="10" fill="#555" transform="rotate(-90 20,125)">GHz / W</text>
</svg>

&emsp;&emsp;这意味着：**任何不控制热状态的微基准比较，都是在比较两个不同频率下的运行结果**。`cycles` 本身会“吸收”频率变化，使得 CPI 看起来正常而实际吞吐已经下降。稳健的做法是同时采集 `cycles`、`instructions` 和 `aperf/mperf`，用 `mperf/aperf` 的比值判断频率是否偏离了 TSC 基准。

### 2.2 被编译器消除的基准比“测不准”更危险

&emsp;&emsp;朴素微基准会被编译器整个消除。perf-book 的检测方法是：用 PMU 确认 `instructions` 退休数是否与预期相符。如果一条本应执行 1000 次的循环只退休了 3 条指令，基准已经失效。5600X 的 perf 支持一组额外的前端事件，提供第二道信号：

```bash
perf stat -e instructions,uops_retired,op_cache_hit_miss,\
          de_dis_cops_from_decoder ./workload
```

&emsp;&emsp;`op_cache_hit_miss` 和 `de_dis_cops_from_decoder` 是 Zen 3 新增的 vendor events，从 Linux 5.13 起可用。如果编译器消除了一个本应重复执行的循环，`instructions` 与 `op_cache_hit` 的比例会异常——把几组计数放在一起看，比单看 `instructions` 更可靠地发现“基准已经失效”。


## 3. 用 PMU 建立指标语言：CPI 之后用什么

### 3.1 CPI 是聚合指标，不是诊断指标

```bash
perf stat -e cycles,instructions,branches,branch-misses,\
          cache-references,cache-misses ./workload
```

&emsp;&emsp;`cycles / instructions` 给出 CPI。CPI > 1 是常态，但 **CPI 本身不告诉任何关于“慢在哪里”的信息**。一个 CPI 为 1.5 的程序可能受限于 L1D 缺失，也可能受限于分支预测失败。5600X 每个周期提供 **6 个 dispatch slot**，槽位被退休 macro-op 填充的比例才是更有信息量的度量。

&emsp;&emsp;5600X 上对诊断特别有用的一组原始事件是：**`op_cache_hit_miss`**（Op Cache 命中/缺失）、**`de_dis_cops_from_decoder`**（解码器分发）、**`ls_mab_alloc`**（MAB 分配，反映内存请求排队压力）、**`dispatch_stalls`**（分发级停顿）。它们暴露了 5600X 前端的一个关键特征：**4-wide 解码器 + 8-wide Op Cache 的双路径，最终由 6-wide dispatch 供给后端**。

<svg width="760" height="260" viewBox="0 0 760 260" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica,Arial,sans-serif">
  <rect x="8" y="8" width="744" height="244" fill="#fafafa" stroke="#ccc" rx="6"/>
  <defs><marker id="a1" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#444"/></marker></defs>
  <rect x="40" y="80" width="130" height="60" fill="#e3f2fd" stroke="#1565c0" stroke-width="2" rx="6"/>
  <text x="72" y="115" font-size="12" fill="#1565c0" font-weight="bold">L1 I-Cache</text>
  <rect x="230" y="50" width="160" height="52" fill="#fff3e0" stroke="#ef6c00" stroke-width="2" rx="6"/>
  <text x="255" y="80" font-size="11" fill="#ef6c00">4-wide 解码器</text>
  <rect x="230" y="120" width="160" height="52" fill="#c8e6c9" stroke="#2e7d32" stroke-width="2" rx="6"/>
  <text x="255" y="150" font-size="11" fill="#2e7d32">Op Cache 8-wide</text>
  <rect x="460" y="80" width="130" height="60" fill="#ede7f6" stroke="#4527a0" stroke-width="2" rx="6"/>
  <text x="478" y="106" font-size="11" fill="#4527a0">6-wide</text><text x="478" y="124" font-size="11" fill="#4527a0">Dispatch</text>
  <rect x="640" y="85" width="80" height="50" fill="#ffcdd2" stroke="#c62828" stroke-width="2" rx="6"/>
  <text x="664" y="115" font-size="11" fill="#c62828" font-weight="bold">后端</text>
  <line x1="170" y1="100" x2="230" y2="78" stroke="#444" stroke-width="2" marker-end="url(#a1)"/>
  <line x1="170" y1="120" x2="230" y2="144" stroke="#444" stroke-width="2" marker-end="url(#a1)"/>
  <line x1="390" y1="78" x2="460" y2="100" stroke="#444" stroke-width="2" marker-end="url(#a1)"/>
  <line x1="390" y1="144" x2="460" y2="120" stroke="#444" stroke-width="2" marker-end="url(#a1)"/>
  <line x1="590" y1="110" x2="640" y2="110" stroke="#444" stroke-width="2" marker-end="url(#a1)"/>
  <circle cx="195" cy="78" r="10" fill="#2e7d32"/><text x="190" y="82" font-size="10" fill="#fff" font-weight="bold">H</text>
  <circle cx="195" cy="144" r="10" fill="#ef6c00"/><text x="190" y="148" font-size="10" fill="#fff" font-weight="bold">M</text>
  <text x="40" y="215" font-size="10" fill="#2e7d32">● 命中：前端带宽充沛，瓶颈后移至 dispatch</text>
  <text x="40" y="235" font-size="10" fill="#ef6c00">● 缺失：回落 4-wide 解码器，前端带宽骤降</text>
</svg>

### 3.2 同一个 CPI，两种截然不同的槽位消耗形态

&emsp;&emsp;构造两个 workload：A 为计算密集的整数运算链，B 为依赖指针追逐的内存访问。调整使 CPI 都接近 1.5。Top-Down Level 1 会显示：

- **A**：Backend Bound 中的 **Core Bound** 占主导——槽位在等待执行端口或依赖链延迟。
- **B**：Backend Bound 中的 **Memory Bound** 占主导——槽位在等待未完成的 demand load。

&emsp;&emsp;CPI 相同只说明“平均每条指令花了同样多周期”，但槽位被阻塞的微架构原因完全不同。


## 4. Top-Down 方法论与 5600X 上的采集实现

### 4.1 Level 1 的四个类别

&emsp;&emsp;Top-Down Level 1 输出四个互斥类别，总和约 100%：**Frontend Bound**（取指/解码供给不足）、**Backend Bound**（执行或内存资源成为瓶颈）、**Bad Speculation**（错误推测浪费槽位）、**Retiring**（槽位被有效退休指令填充）。

<svg width="760" height="290" viewBox="0 0 760 290" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica,Arial,sans-serif">
  <rect x="8" y="8" width="744" height="274" fill="#fafafa" stroke="#ccc" rx="6"/>
  <defs><marker id="a2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#888"/></marker></defs>
  <rect x="290" y="35" width="180" height="36" fill="#e3f2fd" stroke="#1565c0" stroke-width="2" rx="5"/>
  <text x="316" y="58" font-size="12" font-weight="bold" fill="#1565c0">Dispatch Slots</text>
  <rect x="30" y="105" width="150" height="36" fill="#fff3e0" stroke="#ef6c00" stroke-width="2" rx="5"/><text x="62" y="128" font-size="11" fill="#ef6c00" font-weight="bold">Frontend</text>
  <rect x="210" y="105" width="150" height="36" fill="#ffebee" stroke="#c62828" stroke-width="2" rx="5"/><text x="228" y="128" font-size="11" fill="#c62828" font-weight="bold">Bad Spec.</text>
  <rect x="390" y="105" width="150" height="36" fill="#e8f5e9" stroke="#2e7d32" stroke-width="2" rx="5"/><text x="434" y="128" font-size="11" fill="#2e7d32" font-weight="bold">Retiring</text>
  <rect x="570" y="105" width="150" height="36" fill="#ede7f6" stroke="#4527a0" stroke-width="2" rx="5"/><text x="602" y="128" font-size="11" fill="#4527a0" font-weight="bold">Backend</text>
  <line x1="380" y1="71" x2="105" y2="105" stroke="#888" marker-end="url(#a2)"/><line x1="380" y1="71" x2="285" y2="105" stroke="#888" marker-end="url(#a2)"/>
  <line x1="380" y1="71" x2="465" y2="105" stroke="#888" marker-end="url(#a2)"/><line x1="380" y1="71" x2="645" y2="105" stroke="#888" marker-end="url(#a2)"/>
  <rect x="20" y="175" width="170" height="55" fill="#fff" stroke="#ef6c00" rx="4"/>
  <text x="38" y="197" font-size="10" fill="#ef6c00">op_cache_hit_miss</text><text x="38" y="215" font-size="10" fill="#ef6c00">de_dis_cops…</text>
  <rect x="210" y="175" width="150" height="55" fill="#fff" stroke="#c62828" rx="4"/>
  <text x="228" y="197" font-size="10" fill="#c62828">branch-misses</text><text x="228" y="215" font-size="10" fill="#c62828">L1 BTB</text>
  <rect x="390" y="175" width="150" height="55" fill="#fff" stroke="#2e7d32" rx="4"/><text x="430" y="207" font-size="10" fill="#2e7d32">instructions</text>
  <rect x="570" y="175" width="170" height="55" fill="#fff" stroke="#4527a0" rx="4"/>
  <text x="588" y="197" font-size="10" fill="#4527a0">ls_mab_alloc</text><text x="588" y="215" font-size="10" fill="#4527a0">l2/l3 miss</text>
  <line x1="105" y1="141" x2="105" y2="175" stroke="#ccc" stroke-dasharray="3,2"/><line x1="285" y1="141" x2="285" y2="175" stroke="#ccc" stroke-dasharray="3,2"/>
  <line x1="465" y1="141" x2="465" y2="175" stroke="#ccc" stroke-dasharray="3,2"/><line x1="645" y1="141" x2="645" y2="175" stroke="#ccc" stroke-dasharray="3,2"/>
  <text x="30" y="263" font-size="10" fill="#666">5600X 上：用原始事件映射到四桶，比 Intel 式自动四桶更可靠</text>
</svg>

### 4.2 在 5600X 上采集 Level 1

&emsp;&emsp;`perf stat --topdown` 在 Zen 3 上回退到手动公式和原始 PMU 事件。5600X 没有与 Intel TMA 完全对应的硬件级槽位分类器，所以需要理解 Level 1 的来源是 **dispatch slot 的 stall 分解**。

```bash
# 方式一：使用 perf 的 PipelineL1 指标
perf stat -M PipelineL1 ./workload

# 方式二：直接用原始事件组，手动映射
perf stat -e cycles,instructions,\
          op_cache_hit_miss,de_dis_cops_from_decoder,\
          ls_mab_alloc,dispatch_stalls ./workload
```

### 4.3 下钻到 Level 2：Backend Bound 的 Memory/Core 分裂

&emsp;&emsp;Backend Bound 被划分为 Memory Bound 和 Core Bound。**Memory Bound 是一个桶，不是一种原因**。里面同时装着 L1D 缺失、L2/L3 缺失、DTLB 缺失、预取效率不足、以及带宽饱和。在 5600X 上，Memory Bound 内部需要特别关注 **MAB（Miss Address Buffer）分配压力**：`ls_mab_alloc` 直接反映 load/store 请求在 miss 队列中的排队程度。所有 6 个核心共享 32MB L3，L3 未命中会经过 Infinity Fabric 走向内存控制器，而 FCLK/UCLK/MCLK 的同步状态决定了未命中的实际延迟。

### 4.4 最小闭环：预取在什么条件下无效

```c
// 基线
for (size_t i = 0; i < n; i++) a[i] = b[i] + c[i];

// 软件预取
for (size_t i = 0; i < n; i++) {
    __builtin_prefetch(&b[i + 16], 0, 0);
    __builtin_prefetch(&c[i + 16], 0, 0);
    a[i] = b[i] + c[i];
}
```

&emsp;&emsp;选一个工作集超过 32MB L3 的内存带宽敏感 kernel。如果已经处于内存带宽上限，预取**不会降低 Backend Bound**——它只是把“延迟问题”换成了“带宽问题”。**“改了—重测—解释为什么变了或没变”** 才是 TMA 的使用方式，而不是“Backend Bound 高就加预取”。


## 5. 前端双路径：解码器、Op Cache 与 dispatch

### 5.1 做：Op Cache 驻留率与代码足迹

&emsp;&emsp;**做**：取一个前端瓶颈微基准（密集小循环、大量 taken branch），采集 `op_cache_hit_miss` 和 `de_dis_cops_from_decoder`。

```bash
perf stat -e cycles,instructions,branches,branch-misses,\
          op_cache_hit_miss,de_dis_cops_from_decoder ./branch_bench
```

&emsp;&emsp;**看数据**：如果代码能驻留在 Op Cache 中，前端带宽远超 4-wide 解码器；如果 Op Cache 频繁缺失，前端才回落到 4-wide 路径。

&emsp;&emsp;**解释机制**：Zen 3 的前端设计使“前端受限”需要被拆成两个独立问题：**Op Cache 是否接住**，以及**解码器是否成为瓶颈**。两者的优化手段不同——前者靠减少代码足迹、避免巨型函数内联；后者靠减少指令数量、提高每指令工作量。5600X 的 L1 BTB 有 1024 条目，分支预测带宽翻倍，这意味着对于代码规模适中、分支目标数在 BTB 覆盖范围内的负载，Bad Speculation 的基线可能已经较低，不应不加区分地堆分支消除。

&emsp;&emsp;**实践要点**：在 5600X 上，Frontend Bound 的下钻顺序是：先看 `op_cache_hit_miss`，再看 `de_dis_cops_from_decoder`，最后才回到分支预测指标。


## 6. 分层缓存：32MB 统一 L3 与内存时钟域

### 6.1 做：工作集—延迟台阶

&emsp;&emsp;**做**：构造一个工作集在 16MB 到 40MB 之间的随机访存 workload，采集 L2/L3 miss 和 `ls_mab_alloc`。

```bash
for sz in 16 24 32 40; do
  perf stat -e cycles,instructions,ls_mab_alloc,\
            l2_cache_misses_from_dc_misses,l3_misses_from_l2_misses \
            ./random_access $((sz*1024*1024))
done
```

&emsp;&emsp;**看数据**：5600X 是单 CCD，6 个核心共享 32MB 统一 L3。工作集在 32MB 以内时，Memory Bound 的触发点不是 DRAM 带宽，而是 **L3 到核心的供给带宽**。超过 32MB 后，未命中路径经过 Infinity Fabric 到达内存控制器。

<svg width="760" height="250" viewBox="0 0 760 250" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica,Arial,sans-serif">
  <rect x="8" y="8" width="744" height="234" fill="#fafafa" stroke="#ccc" rx="6"/>
  <text x="30" y="32" font-size="12" font-weight="bold" fill="#333">工作集—延迟台阶（32MB 统一 L3）</text>
  <line x1="70" y1="195" x2="720" y2="195" stroke="#333"/><line x1="70" y1="195" x2="70" y2="50" stroke="#333"/>
  <polyline points="70,180 190,176 310,172 430,120 550,88 690,76" fill="none" stroke="#c62828" stroke-width="3.5"/>
  <line x1="430" y1="55" x2="430" y2="195" stroke="#4527a0" stroke-width="1.5" stroke-dasharray="5,3"/>
  <text x="396" y="48" font-size="10" fill="#4527a0">32MB</text>
  <rect x="70" y="180" width="120" height="15" fill="#2e7d32" opacity="0.25"/><rect x="190" y="176" width="240" height="19" fill="#ef6c00" opacity="0.25"/><rect x="430" y="120" width="290" height="75" fill="#c62828" opacity="0.15"/>
  <text x="95" y="214" font-size="10" fill="#2e7d32" font-weight="bold">L1/L2</text><text x="270" y="214" font-size="10" fill="#ef6c00" font-weight="bold">L3 命中</text><text x="530" y="214" font-size="10" fill="#c62828" font-weight="bold">DRAM（经 IF）</text>
  <text x="20" y="125" font-size="10" fill="#555" transform="rotate(-90 20,125)">延迟</text><text x="330" y="232" font-size="10" fill="#555">工作集 →</text>
</svg>

&emsp;&emsp;**解释机制**：一旦超过 32MB，FCLK/UCLK/MCLK 的同步状态成为延迟的主导因素。**在 5600X 上，“内存受限”首先是一个缓存层次问题，然后才是一个 DRAM 带宽问题**——工作集落在哪个层级，决定了应该用 L3 带宽还是 DRAM 带宽作为 Roofline 的屋顶。


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