# 从计数器到证据链：台式机上的 perf 分层诊断工作流

&emsp;&emsp;perf 是一套覆盖硬件计数器、调用栈采样、缓存行一致性、指令级归因与数据源标注的分层诊断系统。下文给出一条递进式工作流，面向台式机 Linux 环境下的计算密集型程序。整条工作流分为七个步骤，每一步只回答一个问题，并为下一步缩小问题空间：

1. **建立心智模型**：理解 PMU 的计数模式与采样模式，明确后续所有命令的共同基础。
2. **快速画像**：用 `perf stat` 判定程序处于计算受限还是访存受限状态。
3. **热点定位**：用 `perf record` / `perf report` 把范围收敛到具体函数。
4. **一致性流量诊断**：用 `perf c2c` 识别伪共享与缓存行迁移，并确认这条路径在本平台的边界。
5. **函数到指令**：用火焰图与 `perf annotate` 下钻到循环体。
6. **精确采样**：用 PEBS / IBS 消除 skid，检验数据源层级归因是否真的可得。
7. **调优闭环**：实施单变量改动，并回到同一测量起点验证证据链。

## 0. 测量基线

&emsp;&emsp;本文所有数字来自同一次完整运行，不是"典型值"或估计值。复现环境如下：

| 项目 | 值 |
|---|---|
| CPU | AMD Ryzen 5 5600X（Zen 3，6 核 12 线程，单 CCD，L3 32 MB） |
| 内核 | Linux 7.0.0-42-generic |
| perf | perf version 7.0.14 |
| `perf_event_paranoid` | 1 |
| CPU governor | `powersave` |
| 被测程序 | `perf_bench`，`g++ -O2 -g -march=native -std=c++17 -pthread` |

---

## 1. 建立心智模型——perf 的观测对象与两种基本模式

&emsp;&emsp;perf 的所有输出都源自同一个硬件模块，理解这一点是理解后续每条命令的前提。

&emsp;&emsp;CPU 内部包含一个硬件模块 **PMU**（Performance Monitoring Unit，性能监控单元），其可编程计数器能够统计 cycles、instructions、cache-misses、branch-misses 等事件。perf 对 PMU 的使用分为两种根本不同的模式：

&emsp;&emsp;**计数模式**（counting）：计数器在程序运行期间持续累加，运行结束后读出总量。该模式回答"事件发生了多少次"，不提供事件发生位置。`perf stat` 属于此类。

&emsp;&emsp;**采样模式**（sampling）：计数器溢出时触发 PMI 中断，内核在中断上下文中记录指令指针与调用栈。该模式回答"时间花在哪些代码位置"，但样本归属存在近似性（详见第 6 章）。`perf record` / `perf report` 属于此类。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261102_8be4bc.svg)

&emsp;&emsp;首先确认当前机器暴露的事件集合。这台机器上的 `perf list` 默认输出并不按 `[Hardware event]` / `[Software event]` 分节，而是先列出内核 PMU 事件：

```bash
perf list | head -6
```

```
  amd_iommu/cmd_processed/                           [Kernel PMU event]
  amd_iommu/cmd_processed_inv/                       [Kernel PMU event]
  amd_iommu/ign_rd_wr_mmio_1ff8h/                    [Kernel PMU event]
  amd_iommu/int_dte_hit/                             [Kernel PMU event]
  amd_iommu/int_dte_mis/                             [Kernel PMU event]
  amd_iommu/mem_dte_hit/                             [Kernel PMU event]
```

&emsp;&emsp;默认列表里有 554 个事件行，去重后 495 个事件——部分内核 PMU 事件会在不同小节重复出现。想看分类，应显式指定类别：

```bash
perf list hw
perf list sw
perf list tracepoint
```

| 类别 | 去重事件数 | 本机说明 |
|---|---|---|
| 全部 | 495 | 含内核 PMU、raw 描述符、SDT 探针 |
| `hw`（legacy hardware） | 10 | `cycles`、`instructions`、`branches`、`cache-misses` 等 |
| `sw`（software） | 28 | `task-clock`、`context-switches`、`page-faults` 等 |
| `tracepoint` | 0 | tracefs 未挂载，当前用户看不到任何 tracepoint |
| IBS raw | 8 | `ibs_op`、`ibs_fetch` 及其别名 |

&emsp;&emsp;`hw` 事件直接映射到 PMU 可编程计数器；`sw` 事件由内核在软件路径上计数，不占用 PMU 计数器；`tracepoint` 是内核预埋的静态探针，属于事件追踪而非 PMU 计数。计算密集型场景下硬件事件是主要观测对象。本机 `tracepoint` 为 0 是一个环境事实：没有挂载 tracefs 时，tracepoint 这一整类观测都不可用，诊断只能建立在硬件计数与采样之上。

&emsp;&emsp;perf 无需重新编译程序，但需要打开 `perf_event_open(2)`。多数发行版默认的 `perf_event_paranoid` 会限制普通用户进行采样。检查当前值：

```bash
cat /proc/sys/kernel/perf_event_paranoid
```

&emsp;&emsp;本机输出为 `1`：per-process 的用户态与内核态采样可用，系统级（`-a`）计数与采样被拒绝。在个人台式机上，可以直接降低该值：

```bash
sudo sysctl kernel.perf_event_paranoid=1
```

&emsp;&emsp;若需要进行系统级采样或 uncore 事件观测，需设为 `-1` 或 `0`。该设置仅影响本机，重启后失效；如需持久化，写入 `/etc/sysctl.d/`。另一种不降低全局安全设置的方式是 `CAP_PERFMON`（Linux ≥ 5.9）：为 `perf` 二进制授予该 capability，普通用户即可执行采样。

&emsp;&emsp;最后确认 LLC 与精确采样事件的可用性，这两项会直接影响第 2、4、6 章：

```bash
perf stat -e LLC-loads -- true        # 本机: The LLC-loads event is not supported.
perf stat -e cache-references,cache-misses -- true   # 可用
perf record -e ibs_op// -- true       # record 可用；perf stat 打开同一事件会被拒绝
```

&emsp;&emsp;这份能力清单是后面所有解读的前提：`LLC-loads` 不可用时，只能用 `cache-references` / `cache-misses` 近似 LLC 行为；`ibs_op` 只能通过 `perf record` 使用，`perf stat -e ibs_op//` 会报错。

> &emsp;&emsp;这一阶段的局限在于：`perf list` 只反映硬件与内核暴露了什么，不指示当前程序应关注哪些事件。事件选择取决于待诊断的问题，也取决于哪些事件在这台机器上真的能打开。

## 2. 快速画像——以 `perf stat` 判定程序的计算效率状态

&emsp;&emsp;程序跑完一次，最先要回答的不是"哪里慢"，而是"整体处于计算受限还是访存受限"。这一节用单线程 `gemm --n 1024` 作为起点，因为单线程排除了并行调度和负载均衡的干扰，任何 IPC 异常都只能来自微架构本身。

&emsp;&emsp;`perf stat` 在一次运行中并行收集多个 PMU 计数器，并派生关键比率。计算密集型程序应优先关注 **IPC**（Instructions per Cycle，每周期退休指令数）与 cache miss rate，而非绝对 cycles 数值。

```bash
perf stat -d ./perf_bench --workload gemm --n 1024 --reps 3 --threads 1
```

&emsp;&emsp;输出（AMD Zen 3，单线程，节选）：

```
                78      context-switches                 #      7.6 cs/sec  cs_per_second
                 4      cpu-migrations                   #      0.4 migrations/sec  migrations_per_second
             6,293      page-faults                      #    609.8 faults/sec  page_faults_per_second
         10,320.19 msec task-clock                       #      1.0 CPUs  CPUs_utilized
     3,459,201,578      L1-dcache-load-misses            #     52.0 %  l1d_miss_rate          (42.84%)
         7,419,366      branch-misses                    #      0.2 %  branch_miss_rate       (42.85%)
     3,297,983,050      branches                         #    319.6 M/sec  branch_frequency   (42.86%)
    46,816,493,067      cpu-cycles                       #      4.5 GHz  cycles_frequency     (42.87%)
    22,973,155,749      instructions                     #      0.5 instructions  insn_per_cycle  (42.87%)
       222,396,952      stalled-cycles-frontend          #     0.00 frontend_cycles_idle    (42.86%)
```

&emsp;&emsp;`IPC` = `22.97e9 / 46.82e9` ≈ `0.49`。这意味着**平均每条指令要等 2 个 cycle**。在 Zen 3 核心上，分发与退休宽度都是 6，理论 IPC 上限接近 6；实测 `0.49` 不到峰值的十分之一。核心长期处于空转状态。

&emsp;&emsp;与此同时，`L1-dcache-load-misses` 占 L1 访问的 `52.0%`——**超过一半的 load 在 L1 就没命中**。作为对照，`branch-misses` 只有 `0.2%`；`stalled-cycles-frontend` 的派生列虽然显示 `0.00`，原始计数是 `2.22 亿`（占 cycles 的 `0.5%`），量级很小，但该别名在 AMD 上的语义不完全对应 Intel，只作旁证。把三项放在一起看：

| 指标 | 观测值 | 排除的假设 |
|---|---|---|
| `branch-misses = 0.2%` | 控制流极稳定 | **不是分支误预测** |
| `stalled-cycles-frontend = 2.22e8`（0.5%） | 前端停顿占比低 | **前端/取指不是主因**（旁证） |
| `IPC = 0.49` + `L1 miss = 52.0%` | 核心等数据 | **是后端数据供应问题** |

&emsp;&emsp;所以结论是：**程序不是算得慢，是等得久**。核心大部分时间挂在 load 的 miss 上。这解释了为什么 `GFLOP/s` 只有 `0.63`——单线程 naive ijk 的每一次内层 `k` 迭代，`B[k*n+j]` 都是一次跨步访问（`n` 个元素跨度、8 KB），几乎必然 L1 miss，而 `A[i*n+k]` 是顺序访问可以复用。`52.0%` 的 L1 miss 就是 `B` 的跨步访问贡献的。

&emsp;&emsp;一个容易被跳过、但会改变后续优化方向的细节是**工作集大小**。`n = 1024`，三个 `1024×1024` 的 `double` 矩阵共 `24 MB`，而 5600X 的 L3 是 `32 MB`。也就是说，`B` 的跨步访问在 L1 失手后，大概率在 L2 或 L3 就被接住，并不会一路打到 DRAM。后面第 7 章会看到：以"让 DRAM 访问更友好"为目标的分块只能带来 1.3 倍改善，而直接消除 L1 跨步的循环重排带来 9.3 倍改善。**先判断 miss 打在哪一层，再决定改哪一层**，这是第 2 章真正要留下的判据。

&emsp;&emsp;为了把"并行因素"也排除掉，用同一段代码跑 8 线程，只改线程数：

```bash
perf stat -e cycles,instructions,cache-references,cache-misses \
    ./perf_bench --workload gemm --n 1024 --reps 3 --threads 8
```

&emsp;&emsp;结果是 `IPC = 0.48`，gemm 内核耗时 `1.705s`（单线程 `10.32s`，加速 `6.1×`，6 个物理核接近线性），`cache-misses / cache-references = 21.6%`。两个事实同时成立：**多线程扩展得很好，但每个核的 IPC 没有变**。这说明瓶颈不在负载均衡，而在单核访存路径；加线程只是把同样的等待复制到更多核上。

> &emsp;&emsp;有两个数字需要额外解释，否则容易被误读。
> &emsp;&emsp;**第一，`CPUs utilized = 1.0`。** 单线程运行的预期值，它的意义是把并行因素排除掉：即便单线程，IPC 也只有 `0.49`。
> &emsp;&emsp;**第二，`-d` 在 AMD 上展开的事件和 Intel 不同。** 这次 `-d` 只给出了 `L1-dcache-load-misses`，没有自动带出 LLC 层级；`LLC-loads` 与 `LLC-load-misses` 在本机直接报 `not supported`。要近似 LLC 行为，用 `cache-references` / `cache-misses`：
> ```bash
> perf stat -e cache-references,cache-misses ./perf_bench --workload gemm --n 1024 --reps 3
> ```
> &emsp;&emsp;`L2-requests` 可用，`L2-misses` 不可用；`l3_read_miss_latency` 在 `perf list` 里存在，但 `perf stat` 打不开。这些差异不是 perf 的缺陷，而是 AMD 平台事件命名与内核映射的现状。输出里每行末的 `(42.84%)` 是计数器复用率，低于 `50%` 说明多个事件在分时复用一个计数器，数值精度会受影响；要更高精度就减少同时采集的事件数，或分多次运行采集不同事件组。

&emsp;&emsp;结论：下一步不是优化分支、不是调优前端、不是加线程，而是**沿着内存层次下钻**。第 3 步用 `perf record -g` 定位到具体函数（应该是 `gemm_naive_ijk` 的内层循环），第 5 步用 `perf annotate` 定位到具体是 `A` 还是 `B` 的那条 load 在吃 cycle，第 7 步再用单变量实验验证这个判断。

## 3. 热点定位——以 `perf record` / `perf report` 收敛至函数级

&emsp;&emsp;第 2 步只确认了"核心在等数据"，但 `perf stat` 不告诉你等在哪。这一步要把 `52.0%` 的 L1 miss 和 `0.49` 的 IPC 落到具体函数上，并且建立"从这份 `perf.data` 出发，后续所有下钻都靠它"的工作方式。

&emsp;&emsp;`perf record` 将 PMU 计数器配置为采样模式：每次溢出记录一个样本，包含指令指针与（启用 `-g` 时）调用栈。采样频率由 `-F` 控制。**应避免 100 Hz 或 1000 Hz**，二者易与调度器 tick 锁相，导致样本向特定代码路径系统性偏移。`99` 或 `997` 是常用的非锁相取值。

```bash
perf record -F 997 -g -o perf.data -- \
    ./perf_bench --workload gemm --n 1024 --reps 3 --threads 1
perf report --stdio
```

&emsp;&emsp;单线程约 10.3 秒，在 997 Hz 下得到 `10K` 个样本。这个量级足以让符号排名稳定，又不至于让 `perf.data` 膨胀到难以分析。`perf report` 的关键输出（节选）：

```
# Samples: 10K  of event 'cpu/cycles/P'
# Event count (approx.): 45835659429
#
# Children      Self  Command     Shared Object         Symbol
# ........  ........  ..........  ....................  ...................................
#
    99.83%     0.00%  perf_bench  perf_bench            [.] _start
            |
            ---_start
               __libc_start_main@@GLIBC_2.34
               call_init (inlined)
               __libc_start_call_main
                --99.72%--main
                           --99.69%--gemm_naive_ijk(double const*, double const*, double*, int)
                                     parallel_for_threads<...> (inlined)
                                     |--80.51%--gemm_naive_ijk(...)
                                      --19.36%--operator() (inlined)
```

&emsp;&emsp;换用 `--no-children --sort=symbol` 看自身占比：`gemm_naive_ijk` 的 `Self` 是 `99.48%`，其余所有符号加起来不到 `0.5%`。第一眼能读出三件事：

&emsp;&emsp;**热点是干净的。** 样本全部落在计算内核的指令上，没有被线程同步、libgomp 屏障或别的库稀释。这正是我们想要的起点——`perf annotate` 接下来会在这个函数内部下钻；如果这里有 30% 的样本落在 `barrier_wait`，下钻就无从谈起。

&emsp;&emsp;**符号化是完整的。** `--sort=dso,symbol` 的报告里有 33 行内核符号，`[unknown]` 为 0 行；`libm.so.6` 里的 `__cos_fma`、`__sin_fma` 也能解析。这份 `perf.data` 可以直接复用给第 5 章的 `perf annotate` 与火焰图。如果看到大量 `[unknown] [k] 0xffffffff...`，通常是 `kptr_restrict` 没关或 `DEBUGINFOD_URLS` 没配；这次没有出现。

&emsp;&emsp;**线程化没有带来额外开销。** `80.51%` 走直接调用路径、`19.36%` 走 `operator()` 内联路径，是编译器对内联展开做的两种决策，两者的样本都属于 `gemm_naive_ijk` 本身。纯 `std::thread` 版本在单线程时不会引入线程创建成本，多线程时才会出现，届时再单独分析。

&emsp;&emsp;**IPC 列是 `-`，而且加 `-e cycles,instructions` 也不会变成数字。** 旧版 perf 在同时采集 `cycles` 和 `instructions` 时会按符号给出 IPC；在本机的 perf 7.0.14 上，重跑：

```bash
perf record -e cycles,instructions -F 997 -g -o perf.data.ipc -- \
    ./perf_bench --workload gemm --n 1024 --reps 3
perf report -i perf.data.ipc --stdio --no-children --sort=symbol
```

&emsp;&emsp;`IPC` 与 `IPC Coverage` 两列仍然都是 `-`（样本数同样是 `10K`）。逐样本 IPC 需要硬件在样本里附带指令计数或延迟权重，本机 `ibs_op` 的 `Local Weight` 全为 0（第 6 章会看到），因此算不出来。函数级 IPC 请回到 `perf stat` 的聚合值 `0.49`，不要指望 `perf report` 的 IPC 列。

&emsp;&emsp;**再看调用链。** `--no-children --sort=dso,symbol` 只留 `Overhead / Shared Object / Symbol`，适合快速扫一眼；保留调用链时，`80.51%` 与 `19.36%` 的差异来自内联决策：`tid` 在编译期不可知时走完整 lambda 对象，`tid` 在 `parallel_for_threads` 内部展开后可能走 `operator()`。**这不影响 Self 归属**，因为两个分支最终都在 `gemm_naive_ijk` 的指令上执行。调用链的真正价值在于确认没有任何一层"中间人"在消耗 cycle。

&emsp;&emsp;如果后续想在 `perf report` 里按线程区分，可以给 `perf record` 加 `-t <TID>`，但通常没必要——`perf report` 默认跨线程聚合，按符号排序的结果更有用。

&emsp;&emsp;所以第 3 步的产出不是"哪个函数慢"，而是**一份可以复用的 `perf.data`**，以及一个明确的下一步方向：进第 4 章看缓存行，或进第 5 章看指令。两条路径都从这份 `perf.data` 出发（`annotate` 直接读它），或者重新跑一次 `perf c2c record`（因为它需要 HITM 或 IBS op 事件，`cycles/P` 采样不包含）。

&emsp;&emsp;如果输出里 `gemm_naive_ijk` 的 Self 不足 `90%`，先检查三件事：`--reps` 是否够大（把初始化开销稀释掉）、`-g` 是否加在 `perf record` 上、符号化是否成功。**样本量、符号化、调用链**是这一章成立的三根支柱。

&emsp;&emsp;还有一个容易踩的坑：`--workload gemm` 不加 `--blocksize` 时走 naive 路径，热点是 `gemm_naive_ijk`；加 `--blocksize 64` 会走 `gemm_blocked_ijk`，热点符号随之改变。分块后总时间确实变短（`10.34s → 7.91s`），但代码路径也换了，Self 占比、指令数、缓存行为全都变了。**`perf stat` 的 runtime 才是判断优化是否生效的依据，`perf report` 的占比只用来定位热点**；不要拿"热点占比下降"当成功指标。

> &emsp;&emsp;`perf report` 给出的是**统计归属**，不是因果时序。`gemm_naive_ijk` 占 `99.48%` 只表示"采样时 CPU 在这个函数的指令上"，不区分这 `99.48%` 里有多少是"在算"、多少是"在等 load"。区分它们要靠第 5 章的 `perf annotate` 看热点落在哪条指令上，以及第 6 章确认数据源字段是否可用。单看第 3 步，能得出的结论只是"范围收敛到了一个函数"。

## 4. 一致性流量诊断——`perf c2c` 的实际输出与解读

&emsp;&emsp;`perf c2c` 的硬件基础是 **HITM**（Hit Modified）事件：当本核请求的缓存行正被另一核以 Modified 状态持有，硬件会记录一次跨核转发。perf 捕获这些样本，在 `perf c2c report` 中按缓存行聚合，给出争用排名。这一节先用 stencil 检验这条路径在本平台是否可行；它的结论会决定第 7 章用什么方式验证伪共享——如果 `perf c2c` 给不出 HITM，就只能改用单变量对照实验。

&emsp;&emsp;被测对象是 stencil：`n = 2048`、`threads = 8`、`reps = 1000`。它的写回发生在每处理完一行之后，总写回次数约 `8 × 256 × 1000 ≈ 205 万` 次。写回的绝对次数足够多，如果伪共享真的存在，采样窗口是充分的。

```bash
perf c2c record -o c2c.data -- \
    ./perf_bench --workload stencil --n 2048 --reps 1000 --threads 8
perf c2c report -i c2c.data --stdio
```

&emsp;&emsp;`c2c details` 显示 `Events: ibs_op//`，说明在 Zen 3 上走的是 IBS op PMU 路径，而非 Intel 的 HITM 硬件计数器。`Trace Event Information` 的关键字段：

```
  Total records                     :      65453
  Load Operations                   :      35198
  Load L1D hit                      :      32147
  Load L2D hit                      :       1804
  Load LLC hit                      :         76
  Load Local HITM                   :          0
  Load Remote HITM                  :          0
  Load Local DRAM                   :       1092
  Load Remote DRAM                  :          0
  Load MESI State Exclusive         :       1092
  Load MESI State Shared            :          0
  LLC Misses to Local DRAM          :      100.0%
  LLC Misses to Remote cache (HITM) :        0.0%
  Store Operations                  :       5711
  Store L1D Hit                     :       4121
  Store L1D Miss                    :          0
  Unable to parse data source       :      24544
```

&emsp;&emsp;`Load Operations: 35198` 中 `Load L1D hit: 32147`，占 `91.3%`；`Load L2D hit: 1804`、`Load LLC hit: 76`，L2 和 LLC 拦截的请求量很小；`Load Local DRAM: 1092`，占 `3.1%`，是 `2048×2048` 双精度网格跨页访存时的冷 miss。`Store L1D Hit: 4121`，写回大量命中 L1，`Store L1D Miss: 0`。这条 stencil 的主访存路径在缓存层级上其实很健康。

&emsp;&emsp;最关键的仍然是那三行：**`Load Local HITM: 0`、`Load Remote HITM: 0`、`LLC Misses to Remote cache (HITM): 0.0%`**——没有任何一次 load 撞上了被另一核修改过的缓存行。`Global Shared Cache Line Event Information` 同样为空：

```
  Total Shared Cache Lines          :          0
  Load HITs on shared lines         :          0
  L1D hits on shared lines          :          0
  Store HITs on shared lines        :          0
```

&emsp;&emsp;`Total Shared Cache Lines: 0` 意味着 `Shared Data Cache Line Table` 没有内容可排。写回次数很多的 stencil 在 205 万次写回的窗口下仍然没有产生任何 HITM 样本，这个零值需要认真解读，而不能直接当作"伪共享不存在"。

&emsp;&emsp;解读的关键在最后一行：`Unable to parse data source: 24544`，占 `Total records` 的 `37.5%`。这个比例反映的是 perf 对 Zen 3 上 IBS op 数据源编码的解析覆盖率不足。三分之一样本无法被归因到 L1/L2/LLC/DRAM 层级，这意味着即使真有 HITM 发生，也有三分之一的概率没被识别出来。更麻烦的是这个比例本身不稳定：在三次完整运行中，`Total records` 在 `6.5 万 ~ 8.0 万` 之间波动，不可解析比例在 `37% ~ 45%` 之间波动。**单个零值 + 高比例不可解析，只能说明"这条路径看不到伪共享"，不能说明"伪共享不存在"。**

&emsp;&emsp;为了进一步缩小可能性，再做一次 `--pad` 对照：把 `acc` 数组从 `stride = 1` 改成 `stride = 8`，让每个线程的累加器落在不同缓存行：

```bash
perf c2c record -o c2c_pad.data -- \
    ./perf_bench --workload stencil --n 2048 --reps 1000 --threads 8 --pad
perf c2c report -i c2c_pad.data --stdio
```

&emsp;&emsp;结果同样是 `Load Local HITM = 0`、`Total Shared Cache Lines = 0`，不可解析比例反而升到 `44.1%`。两组样本都没有抓到 HITM，也就无法用 `perf c2c` 区分"真没有伪共享"和"没采到"。

&emsp;&emsp;这条路径的结论因此分两层。第一，`perf c2c` 在 Zen 3 + IBS op 上的定性判断（有没有 HITM 被采到）在本机不可用——采样窗口充分的 workload 得到零值，无法与"伪共享确实不存在"区分。第二，定量判断（HITM 占比、具体缓存行地址）同样不可用——统计基础只有约三分之二的可解析样本，代表性有限，任何百分比都不能作为跨平台对比的依据。

&emsp;&emsp;要获得可用的 `perf c2c` 输出，必须换到支持 HITM 硬件计数器的平台——Intel 或非 Zen 3 的 AMD。在那些平台上，同一条命令会填充 `Shared Data Cache Line Table`：每条争用缓存行的地址、`LclHitm` / `RmtHitm` 计数、涉及的核、对应的指令地址，`Unable to parse data source` 的比例会降到个位数。

&emsp;&emsp;在无法换平台的前提下，验证伪共享必须绕开 `perf c2c`，改用"同一份程序只改内存布局"的对照实验。这个实验放在第 7 章：`--pad` 版本与基线都跑 `perf stat -r 3`，比较 `cycles` 与 `runtime`。第 7 章会看到，`--pad` 没有带来任何收益（cycles `+3.1%`、runtime `+0.7%`），结合 `perf c2c` 的零 HITM，可以更有把握地说：**在这个 workload 上，`acc` 的伪共享不是可测瓶颈**。原因不在"伪共享不存在"，而在"写回太稀疏"——8 个线程合计每秒约 45 万次写，平均每 2.2 微秒才有一次写，而缓存行迁移的代价在百纳秒量级，被行内计算掩盖了。

> &emsp;&emsp;`perf c2c` 给出的是缓存行级的争用证据，前提是样本能被完整归因，且硬件能正确报告 HITM。在 Zen 3 + IBS op 路径上，`Unable to parse data source` 占 37%–45%，使定量结论不可用；`Total Shared Cache Lines: 0` 出现在写回频率充分的 workload 上，则说明定性结论的区分力也不足以支撑判断。这条路径的价值在于划出边界——它告诉你在什么平台上 `perf c2c` 能用，在什么平台上必须换成 `perf stat` 对照实验或换硬件。

## 5. 从函数到指令——火焰图与 `perf annotate`

&emsp;&emsp;第 3 步把热点收敛到了 `gemm_naive_ijk`，第 4 步在 `perf c2c` 的边界上停住了。这一步要把"函数级热点"压缩到"循环体级热点"，回答一个更具体的问题：`gemm_naive_ijk` 内部，是哪条 load、哪条算术指令在吃 cycle。火焰图和 `perf annotate` 共用同一份 `perf.data`，不需要重新采样。

```bash
perf script -i perf.data | stackcollapse-perf.pl | flamegraph.pl > gemm.svg
```

&emsp;&emsp;火焰图的规则很简单：**宽度是样本占比，纵向是调用深度**。最宽的叶子帧对应采样最集中的代码路径。打开本次生成的 `flamegraph.svg`（`perf_out/run_20261103_231030/ch05/`，上面命令示例里叫 `gemm.svg`，内容相同），形态是**极窄且极深**：整张图只有一条主干，底部调用栈从下往上依次是 `perf_bench → _start → __libc_start_main → call_init → __libc_start_call_main → main → gemm_naive_ijk → parallel_for_threads`，在 `gemm_naive_ijk` 这一层分成两支：左支占 `80.96%`，右支占 `19.04%`，标注为 `operator()`。两支的上层都汇入同一个 `gemm_naive_ijk`，没有更深的子帧。折叠数据里两支的权重分别是 `36.9e9` 和 `8.68e9`（这里记的是事件计数，不是样本条数）。

&emsp;&emsp;两条分支的宽度差异来自编译器对内联的不同决策：一条走直接调用，一条走 lambda 的 `operator()`。**两支的样本都落在 `gemm_naive_ijk` 的指令上**，语义上不是两个不同的热点，只是同一个内层循环被编译器展开成了两条入口。如果误读成"`operator()` 也占了 19%，要单独优化"，方向就偏了。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261102_8be4bcflamegraph.svg)

&emsp;&emsp;火焰图能回答"哪个函数热"，但回答不了"为什么热"。`gemm_naive_ijk` 占据 `99.48%` 的宽度，只说明采样时 CPU 在这个函数的指令上，不区分这 `99.48%` 里有多少是"在算"、多少是"在等 load"。区分它们要靠 `perf annotate`：

```bash
perf annotate -i perf.data --symbol='gemm_naive_ijk(double const*, double const*, double*, int)' --stdio
```

&emsp;&emsp;这里有一个与旧版经验相反的细节：本机的 perf 7.0.14 **只接受完整 demangle 符号**，`--symbol=gemm_naive_ijk` 这种前缀写法会静默返回空结果。要么传完整签名，要么先 `perf report --no-children --sort=symbol` 拿到完整符号名再复制。如果 `annotate` 输出为空，先确认 `--symbol` 匹配的是函数符号而不是 `perf_bench`（后者是 Shared Object 名）。

&emsp;&emsp;`annotate` 的每一行左侧是采样百分比，右侧是对应的汇编指令。本次内层 `k` 循环的结果：

| 占比 | 地址 | 指令 | 作用 |
|---|---|---|---|
| 16.98% | `a7b3` | `cmp %r8,%rdx` | 循环条件 |
| 16.66% | `a7ac` | `add %rdi,%rsi` | `B` 指针 += 8 KB（跨步） |
| 16.62% | `a7a4` | `add $0x8,%rdx` | `A` 指针 += 8 |
| 16.61% | `a7a0` | `vmovsd (%rdx),%xmm0` | 读 `A[i*n+k]`（顺序） |
| 16.60% | `a7a8` | `vmulsd (%rsi),%xmm0,%xmm0` | 乘 `B[k*n+j]`（跨步内存操作数） |
| 16.27% | `a7af` | `vaddsd %xmm0,%xmm1,%xmm1` | 累加 |

&emsp;&emsp;六条指令加起来约 `99.7%`，每条 `16%–17%`，几乎是均匀分布。这与"热点会集中在两条 load 上"的预期不同，原因有两个：内层循环体只有六条指令，任何一次停顿都会摊到整个循环体；采样 skid 又让样本落在触发事件的指令之后几条，进一步抹平了差异。**在这种情况下，`annotate` 能给出的正确结论不是"优化 `vaddsd`"，而是"整个循环体都在等，瓶颈是循环里的跨步内存操作数 `B`"。**

&emsp;&emsp;注意 `vmulsd (%rsi)`：它把 `B[k*n+j]` 作为内存操作数直接乘进来，而 `rsi` 每次加 `rdi = 8192`。这条指令的地址计算与内存读取是绑在一起的，正是第 2 章 `52.0%` L1 miss 的来源。`vmovsd (%rdx)` 则是对 `A` 的顺序读取，`rdx` 每次加 8，可以复用缓存行。两者的采样占比几乎相同，说明"顺序 load"和"跨步 load"在等待时间上已经被混在一起；`annotate` 无法再把它们分开。

&emsp;&emsp;第 5 步的产出是**一个循环体 + 一条跨步内存操作数**，而不是一个函数名。火焰图确认了 `gemm_naive_ijk` 内部没有隐藏的分支，`perf annotate` 把范围压缩到六条指令。带着这个结论进入第 6 章，`ibs_op` 才能在正确的位置上附加数据源信息——前提是数据源字段真的可用。

> &emsp;&emsp;火焰图回答"谁在吃 cycle"，`perf annotate` 回答"哪条指令在吃 cycle"，但两者都不回答"这条指令为什么慢"。`gemm_naive_ijk` 占 `99.48%` 是统计归属，不是因果判定；annotate 的热点指令同样是统计归属，受 skid 影响可能偏移 1–2 条。第 5 步的边界是：它能告诉你下一步该看哪条指令或哪个循环体，不能告诉你这条指令为什么慢。

## 6. 精确采样与偏差控制——当指令级归因不够精确时

&emsp;&emsp;第 5 步的 `perf annotate` 已经把热点压缩到 `gemm_naive_ijk` 的六条指令，但常规 `cycles` 采样存在机制性偏差：计数器溢出到 PMI 中断响应之间，乱序流水线已经执行了若干条后续指令，样本可能被归到触发事件的 load 之后几条指令上，这就是 skid。Intel 的 PEBS 和 AMD 的 IBS 由硬件在指令退休时直接记录样本，把记录工作从中断处理器移出，从而消除 skid；PEBS 还可以在样本中携带数据源信息，说明访问由 L1、L2、LLC、本地 DRAM 或远程 NUMA 节点服务。AMD 平台对应 IBS op 采样。

&emsp;&emsp;本机是 Zen 3，走 IBS op 路径：

```bash
perf record -e ibs_op// -F 997 -g -o perf.data.ibs -- \
    ./perf_bench --workload gemm --n 1024 --reps 3 --threads 1
perf report -i perf.data.ibs --stdio --no-children --sort=local_weight,symbol,dso
```

&emsp;&emsp;先看能拿到的部分。报告头部是 `Samples: 10K of event 'ibs_op//'`、`Event count (approx.): 47733153209`，热点归属非常干净：

```
# Overhead  Local Weight  Symbol                                                          Shared Object      IPC   [IPC Coverage]
# ........  ............  ..............................................................  .................  ....................
#
    99.42%  0             [.] gemm_naive_ijk(double const*, double const*, double*, int)  perf_bench         -      -
            |
            |--81.47%--gemm_naive_ijk(double const*, double const*, double*, int)
            |          parallel_for_threads<...> (inlined)
            |          gemm_naive_ijk(...)
            |          main
            |          ...
             --17.89%--operator() (inlined)
```

&emsp;&emsp;IBS op 把 `99.42%` 的样本精确归到 `gemm_naive_ijk`，调用链分成 `81.47%` 直接调用与 `17.89%` 的 `operator()` 内联路径，和第 3、5 章的 `80.5% / 19.4%` 一致。函数级精确归属是成立的：如果第 5 章的 annotate 热点在 load 和 FMA 之间摇摆，IBS op 至少可以排除"热点不在这个函数"的怀疑。`Local Weight` 一列却全是 `0`，说明 IBS 的权重没有提供额外信息。

&emsp;&emsp;再看拿不到的部分。要回答"`gemm_naive_ijk` 的 load 由哪一层服务"，需要 `Memory access`、`Data Symbol`、`Snoop`、`TLB`、`Local INSTR Latency` 这些列。本机的实际尝试如下：

| 尝试 | 结果 |
|---|---|
| `perf report --sort=memory` | **SIGSEGV**，退出码 `139`，无输出 |
| `--sort=local_weight,memory,symbol,dso,addr,dso_daddr,snoop,tlb,locked,blocked,...` | `Error: Invalid --sort key: 'dso_daddr'`，无数据行 |
| `perf report -F overhead,sample,memory,symbol` | `Error: Unknown --fields key: 'memory'` |
| `perf report --header-only` 查 `Total weight` | 没有该字段 |

&emsp;&emsp;也就是说，`memory` 这个排序键在本机 perf 7.0.14 + `ibs_op` 数据上会直接崩溃，`-F` 字段列表也不认它，`dso_daddr` 虽然出现在帮助里却无法与其他键组合。最终能稳定输出的只有 `local_weight,symbol,dso` 这一组，它只回答"哪个符号"，不回答"哪一层内存"。

&emsp;&emsp;配置层面同样受限。`/sys/bus/event_source/devices/ibs_op/format/` 下只有两个可配置项：

```
cnt_ctl=config:19
swfilt=config2:0
```

&emsp;&emsp;常见教程里的 `ibs_op/cnt_ctl=1,13missonly=1/` 在本机不可用——`13missonly` 不是合法字段，`perf record` 会报 `valid terms: cnt_ctl,swfilt,...`。`cnt_ctl=1` 与 `ibs_op//` 都能打开，但它们只改变采样条件，不改变数据源列的可用性。

&emsp;&emsp;本文以当前可复现的输出为准：**IBS op 在这台机器上给出了函数级精确归属，但没有给出数据源层级、数据符号、延迟和阻塞信息。**

&emsp;&emsp;所以第 6 章的产出不是一条可用的数据源结论，而是一组边界。能成立的只有"函数级精确归属"：`99.42%` 的样本落在 `gemm_naive_ijk`，调用链没有中间层消耗显著比例。不能成立的是"数据源层级归因"：`memory` 排序键直接段错误，`-F memory` 不识别，`Local Weight` 为 `0`，`Total weight` 缺失。下一步要么换到 Intel PEBS 平台（`mem_load_retired.l3_miss:pp` 一类事件能带数据源），要么接受这条路径在当前平台上的上限，回到 `perf stat` 的 IPC / cache-miss 率、`perf annotate` 的循环热点、以及第 7 章的对照实验来验证"在等数据"。

> &emsp;&emsp;精确采样的价值在于消除一部分 skid，但它不保证给出数据源。本机 IBS op 能确认热点函数，不能落到数组和缓存层级；继续在当前 `ibs_op//` 输出上调整排序字段不会改变这一结论。把"精确采样"当成万能归因，是这一章最需要避免的误读。

## 7. 调优闭环——从证据到改动，再以 perf 验证

&emsp;&emsp;前六章已经攒出一条完整的证据链：第 2 章 `IPC = 0.49` + `L1 miss = 52.0%`，第 3 章热点收敛到 `gemm_naive_ijk`，第 5 章定位到内层循环里的跨步内存操作数 `vmulsd (%rsi)`，第 4、6 章则划清了两条"看起来更高级"的路径在本机的能力边界。这一章要做的，是把"跨步 load 导致等待"这个假设变成一次可验证的改动，并确认改善来自预期机制，而不是噪声。

&emsp;&emsp;假设很具体：`B[k*n+j]` 每次跨 8 KB，几乎必然 L1 miss；如果改成按行遍历 `B`，L1 miss 应该大幅下降，IPC 应该上升，runtime 应该下降。三个指标同向，才算闭环。

### 7.1 单变量实验 A：循环序 ijk → ikj

&emsp;&emsp;只改循环顺序：外层 `i`、中层 `k`、内层 `j`，让 `B` 与 `C` 都按行连续访问，`A[i*n+k]` 只读一次。

```bash
perf stat -d ./perf_bench --workload gemm --n 1024 --reps 3 --threads 1
perf stat -d ./perf_bench --workload gemm --n 1024 --reps 3 --threads 1 --loop-order ikj
```

| 指标 | ijk（基线） | ikj（改动后） | 变化 |
|---|---|---|---|
| IPC | `0.49` | `3.82` | `7.80×` |
| L1-dcache-load-misses | `52.0%` | `4.3%` | `-47.7` 个百分点 |
| cpu-cycles | `46,655,764,618` | `5,128,464,066` | `-89.0%` |
| runtime | `10.34 s` | `1.11 s` | `9.31×` |
| GFLOP/s | `0.63` | `5.97` | `9.48×` |

&emsp;&emsp;三个指标同向：IPC 上升、L1 miss 下降、runtime 下降。`4.3%` 的残余 miss 已经回到顺序流水的量级，`3.82` 的 IPC 说明核心不再长期等数据。这条证据链闭合了——第 2 章的怀疑、第 5 章的指令、第 7 章的改动，指向同一个机制。

&emsp;&emsp;需要注意，ikj 与 ijk 的浮点累加顺序不同，`checksum` 不会逐位相同。性能对比成立，但正确性验证要单独做：用容差比较结果，而不是比较哈希。

### 7.2 单变量实验 A2：分块为什么只带来 1.3 倍

&emsp;&emsp;第二个候选改动是分块：把 `jj` 方向的工作集压进 L2/L3，期望减少 `B` 的跨步 miss。

```bash
perf stat -d ./perf_bench --workload gemm --n 1024 --reps 3 --threads 1 --blocksize 64
```

| 指标 | ijk（基线） | blocked（64） | 变化 |
|---|---|---|---|
| IPC | `0.49` | `1.07` | `2.18×` |
| L1-dcache-load-misses | `52.0%` | `19.0%` | `-33.0` 个百分点 |
| instructions | `22.97e9` | `39.07e9` | `+70%` |
| runtime | `10.34 s` | `7.91 s` | `1.31×` |
| GFLOP/s | `0.63` | `0.82` | `1.30×` |

&emsp;&emsp;分块确实把 L1 miss 从 `52.0%` 降到 `19.0%`，runtime 也确实变快了，但只有 `1.31×`，远不及 ikj 的 `9.31×`。原因有两个：第一，分块没有消除跨步——块内 `B[k*n+j]` 仍然按 8 KB 跨步，只是把工作集压小，让更多访问落在 L2/L3；第二，分块把 `C[i][j] = s` 改成了 `C[i][j] += s`，每次内层迭代多一次读改写，指令数从 `22.97e9` 涨到 `39.07e9`（`+70%`）。**"L1 miss 降低"本身不是目标；它只有在不引入额外工作时才等于变快。**

&emsp;&emsp;这个对照也纠正了一个常见误读：分块不是"更高级的优化"。它把跨步访问的落点从 L3 挪到 L2（块内 `64 × 1024 × 8 B ≈ 512 KB` 的 `B` 子块刚好装进 512 KB 的 L2），所以有 `1.31×` 的收益；但它没有消除 8 KB 跨步本身，还引入了 `C` 的读改写，因此远不及 ikj。**优化要对着"哪一层 miss"下手，而不是对着"更复杂的技术"下手。**

&emsp;&emsp;反过来说，如果工作集真的装不进 L3，分块的收益会显著放大——那时 DRAM 带宽才是主要矛盾。本机 `24 MB` 工作集小于 `32 MB` L3，DRAM 从来不是主角，这解释了为什么分块的提升有限。

### 7.3 单变量实验 B：`--pad` 与伪共享

&emsp;&emsp;第 4 章没能在 `perf c2c` 上拿到可用的 HITM 证据，于是改用同一份程序、只改内存布局的对照实验：`acc` 数组从 `stride = 1`（8 个线程的累加器共享一条缓存行）改成 `stride = 8`（各占一条缓存行）。

```bash
perf stat -e cycles,instructions -r 3 ./perf_bench --workload stencil --n 2048 --reps 2000 --threads 8
perf stat -e cycles,instructions -r 3 ./perf_bench --workload stencil --n 2048 --reps 2000 --threads 8 --pad
```

| 指标 | 无 `--pad` | `--pad` | 变化 |
|---|---|---|---|
| cycles（3 轮均值） | `141,165,030,591` | `145,471,559,958` | `+3.1%` |
| runtime | `4.601 s` | `4.634 s` | `+0.7%` |
| IPC | `0.61` | `0.59` | `-` |

&emsp;&emsp;`--pad` 没有带来收益，cycles 甚至略高，runtime 差异在噪声范围内。结合第 4 章的零 HITM，结论是：**在这个 workload 上，`acc` 的伪共享不是可测瓶颈**。原因不是"伪共享不存在"，而是写回相对计算太稀疏——8 个线程合计每秒约 45 万次写，平均每 2.2 微秒才有一次写（单个线程两次写之间约 18 微秒），而缓存行迁移的代价在百纳秒量级，被行内计算掩盖了。如果 `acc` 在每行、甚至每个元素上更新，`--pad` 的收益才会显现。

### 7.4 单变量实验 C：绑核与"跨 CCD"假设

&emsp;&emsp;第三个候选改动是绑核：把 8 个线程固定到 `0-7` 号逻辑 CPU，减少迁移。

```bash
perf stat -e cycles,instructions -r 3 taskset -c 0-7 \
    ./perf_bench --workload stencil --n 2048 --reps 2000 --threads 8
```

| 指标 | 不绑核 | `taskset -c 0-7` | 变化 |
|---|---|---|---|
| cycles（3 轮均值） | `141,165,030,591` | `139,107,085,878` | `-1.5%` |
| runtime | `4.601 s` | `4.830 s` | `+5.0%` |
| IPC | `0.61` | `0.62` | `-` |

&emsp;&emsp;绑核同样没有收益，runtime 反而慢了 `5.0%`。一个必须纠正的前提是：**5600X 是单 CCD**，6 个核心共享一块 `32 MB` L3，不存在"跨 CCD 迁移"这回事。12 个逻辑 CPU 对 8 个线程本来就有余量，绑核没有消除任何可测的迁移代价；`0-7` 里还包含两个 SMT 兄弟，反而可能制造核心争用。**硬件拓扑不同，同一个"最佳实践"的结论就不同。**

### 7.5 闭环

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261104_53557d.svg)

&emsp;&emsp;回到**同一测量起点**，把 A、A2 的结果放在一起看：

| 指标 | ijk（基线） | blocked（64） | ikj | 结论 |
|---|---|---|---|---|
| IPC | `0.49` | `1.07` | `3.82` | 循环重排最有效 |
| L1 miss | `52.0%` | `19.0%` | `4.3%` | 只有 ikj 消除了跨步 |
| instructions | `22.97e9` | `39.07e9` | `19.60e9` | 分块引入 +70% 工作量 |
| runtime | `10.34 s` | `7.91 s` | `1.11 s` | 最终判据 |

&emsp;&emsp;IPC 由 `0.49` 升至 `3.82`，表明核心等待数据的周期显著减少；L1 miss 由 `52.0%` 降至 `4.3%`，验证了"跨步 load 是主因"的假设；runtime 由 `10.34 s` 降至 `1.11 s`，构成最终验证。三者同向，闭环成立。

&emsp;&emsp;反过来，B、C 两个实验是**阴性结果**，同样有价值：它们说明"教科书上的优化"不一定适用于当前硬件。`--pad` 没有收益，因为写回相对每行的计算量太稀疏；绑核没有收益，因为单 CCD、12 个逻辑 CPU 对 8 个线程本来就有余量。如果没有做这两个对照，很容易把"改了但没测"当成"改好了"。

&emsp;&emsp;perf 的核心价值不在于产出单一数字，而在于**以不同层级的证据逐步收敛问题空间**：计数器指出内存层次异常；采样定位至函数；c2c 划定缓存行路径的边界；annotate 定位至循环体；精确采样确认函数级归属，并诚实报告数据源字段不可用。每次改动后应回到同一组测量，确认证据链各环节均向预期方向移动。

> &emsp;&emsp;若改动同时涉及多个因素（循环序、分块、padding、绑核），对比结果无法单独归因至某一项。严格归因要求每次仅改变一个变量。本章的 A、A2、B、C 四个实验各自只动一个开关，这也是它们能互相解释的原因：A 说明根因，A2 说明"指标降了不等于变快"，B、C 说明"移植过来的优化必须重新测量"。
