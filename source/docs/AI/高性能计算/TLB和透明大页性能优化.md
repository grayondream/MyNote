# TLB 与透明大页：一个可观测的尾延迟悬崖

**摘要**：&emsp;&emsp;TLB 的容量以条目计量，而应用工作集以字节增长；4 KB 页下，L2 TLB 的 1536 个条目仅能覆盖约 6 MB 地址空间。2 MB 大页将单条目覆盖面积扩大 512 倍，使同一 TLB 可覆盖约 3 GB，这是透明大页（Transparent Huge Pages, THP）的理论收益来源。然而 THP 的自动晋升、内存规整（compaction）、TLB shootdown、写时复制放大与碎片化，会在延迟敏感路径上制造均值不可见的孤立尖峰。本文以一次被完整记录的 400 µs 停顿为线索，给出 TLB 覆盖的算术模型、THP 运行时机制的拆解、可复现的 C++ 测量代码、可观测性信号体系，以及 THP 与 HugeTLB 的工程选型契约；并在 AMD Zen 3 / Linux 7.0 上用一套 C++20 基准实测了 TLB 台阶、晋升/折叠延迟、fork/COW 与 shootdown 缩放。本文的核心论点是：THP 的收益与代价都是结构性的，前者体现在平均 TLB 命中率上，后者体现在尾延迟分布上；只有将观测粒度下沉到逐请求与内核事件，这道悬崖才可归因、可管理。

**关键词**：TLB；透明大页；尾延迟；TLB shootdown；内存规整；可观测性；基准测试

---

## 1 引言

&emsp;&emsp;地址转换是每次内存访问的前置步骤。CPU 发出虚拟地址，MMU 先查 TLB；命中则直接得到物理页框号（PFN），未命中则走页表漫游并回填 TLB。TLB 命中与 miss 的延迟相差一到两个数量级，因此 TLB 的容量与覆盖面积直接决定工作集能否被"免费"转换。本节用四张图刻画这一机制，再引出 THP 的收益与风险。

### 1.1 地址转换路径

&emsp;&emsp;图 1 给出 TLB 在地址转换中的位置。绿色路径为命中：MMU 查 TLB 得到 PFN，与页内偏移拼接后访问物理内存。红色路径为 miss：硬件（或内核）从页表根 PML4 开始逐级漫游，得到 PFN 后回填 TLB。TLB 的层级（L1 dTLB、L2 STLB）决定命中延迟，页表的层级决定 miss 延迟。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261109_391328.svg)

### 1.2 地址切分与 TLB 条目

&emsp;&emsp;图 2 把虚拟地址按 4 级页表切成 9+9+9+9+12 位。TLB 条目不保存完整虚拟地址，只保存 VPN tag、PFN 与权限/脏位。查 TLB 即匹配 tag；命中后取出 PFN，与低 12 位 offset 拼接成物理地址。条目数固定，单条目覆盖的字节数等于页大小，因此页大小直接决定 TLB 的总覆盖面积。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261109_a618f6.svg)

### 1.3 覆盖面积：页大小的乘数效应

&emsp;&emsp;图 3 用对数轴对比 4 KB 页与 2 MB 页的 TLB 总覆盖。1536 个条目 × 4 KB = 6 MB；同样条目数 × 2 MB = 3 GB。示例工作集（2 MB 订单簿 + 64 MB DMA 缓冲 + 8 MB 策略状态，合计 74 MB）落在 4 KB 覆盖线的右侧、2 MB 覆盖线的左侧，这就是大页在纸面上"应该赢"的算术来源。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261109_f9acc4.svg)

### 1.4 漫游层级：大页在哪里截断

&emsp;&emsp;图 4 对比 4 KB 页与 2 MB 页的漫游深度。4 KB 页需走完 PML4 → PDPT → PD → PT 四级；2 MB 页在 PD 层即可得到页框号并终止，省去 PT 一级。漫游层级减少，既降低了单次 miss 的代价，也减少了对 TLB 条目数的需求。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261109_c78bcb.svg)

### 1.5 问题定义与贡献

&emsp;&emsp;图 1–4 说明大页的收益是结构性的。但 THP 的"自动"引入另一条成本曲线：晋升、合并、拆分与失效涉及物理页迁移、页表修改与跨核 TLB shootdown，这些操作在回收、compaction 与 `madvise` 路径上会被重复触发。更关键的是，其代价不体现在平均延迟，而体现在 P999 与最大值上的孤立尖峰。

&emsp;&emsp;本文以一次 400 µs 的生产停顿为案例，回答三个问题：第一，工作集在什么条件下越过 TLB 覆盖线，使大页收益成立；第二，THP 的哪些运行时机制会把页表操作放大为跨核事件；第三，如何用可观测信号建立从尾延迟尖峰到内核机制的归因证据链。贡献包括：（1）TLB 覆盖与工作集需求的算术模型及可复现测量；（2）THP 晋升、compaction 与 shootdown 的成本拆解与 400 µs 阶段分解；（3）面向尾延迟的 THP 可观测性信号体系与工程选型矩阵。

---

## 2 背景：HugeTLB 与 THP

&emsp;&emsp;Linux 提供两种大页机制，二者解决同一个 TLB 覆盖问题，但设计契约不同：HugeTLB 是显式的静态预留，THP 是机会性的自动晋升。理解二者的差异，是判断尾延迟风险归属的前提。

### 2.1 HugeTLB：显式预留的静态契约

&emsp;&emsp;HugeTLB 在系统启动或运行时通过 `nr_hugepages` 预留指定数量的物理连续大页，形成独立的大页池；应用通过 `mmap(MAP_HUGETLB)` 或挂载 `hugetlbfs` 并映射文件来显式使用 [7]。大页池的页在分配后即从伙伴系统中隔离，不参与常规匿名页的回收与迁移，因此具备三条对延迟敏感负载至关重要的性质：不会被 compaction 搬移，不会被 swap 换出，不会被内核拆分回 4 KB 页。这三条性质使 HugeTLB 的地址转换行为在运行期保持稳定：一次映射建立后，TLB 条目的失效只来自应用自身的 `munmap` 或权限变更，而不来自内核的后台整理。

&emsp;&emsp;静态契约的代价同样明确。第一，预留必须在容量规划阶段完成，`nr_hugepages` 设置过大浪费内存，设置过小则在需求增长时无法即时扩容；运行时调整需要先回收再重新分配，可能失败。第二，物理连续内存的供给受碎片化约束，长时间运行的系统可能无法预留出足够的大页。第三，应用必须显式适配：`malloc` 默认不会落入大页池，需要使用 `mmap(MAP_HUGETLB)`、`hugetlbfs` 文件映射，或依赖 `libhugetlbfs` 的 `mallopt` 包装。第四，HugeTLB 页不可 swap，内存超售场景下需要额外容量保障。这些代价换来的是可预测性：延迟敏感路径上的 TLB 行为不随系统内存压力漂移。

### 2.2 THP：机会性的自动晋升

&emsp;&emsp;THP 的设计目标是把大页收益从显式管理中解放出来：内核自动将满足条件的匿名映射晋升为 2 MB 大页，内存紧张时再降级回 4 KB 页，应用无需改动 [2]。其全局模式由 `/sys/kernel/mm/transparent_hugepage/enabled` 控制，取值为 `always`、`madvise`、`never`；规整策略由同目录下的 `defrag` 控制，取值为 `always`、`defer`、`madvise`、`never`；`shmem_enabled` 另行控制 tmpfs 的 THP 行为 [2][3]。`always` 模式下内核对所有符合条件的匿名映射尝试晋升，`madvise` 模式只对应用显式标记 `MADV_HUGEPAGE` 的区域晋升，`never` 模式关闭 THP。

&emsp;&emsp;THP 的晋升有两条路径。第一条是直接缺页路径：匿名页缺页时，内核在慢路径尝试直接分配 2 MB 大页，成功则记录 `thp_fault_alloc`，失败则回落 4 KB 页并记录 `thp_fault_fallback`。第二条是后台折叠路径：`khugepaged` 内核线程周期性扫描匿名映射，寻找可折叠的连续 4 KB 页，成功则记录 `thp_collapse_alloc`，失败则记录 `thp_collapse_alloc_failed`。两条路径的共同前提是找到 512 个物理连续的 4 KB 页；内存运行一段时间后出现外部碎片，内核必须执行内存规整（compaction），搬移物理页以拼凑连续空间 [4][5]。

### 2.3 参数、失效路径与成本来源

&emsp;&emsp;`khugepaged` 的扫描节奏由 `/sys/kernel/mm/transparent_hugepage/khugepaged/` 下的参数控制：`scan_sleep_millisecs` 与 `alloc_sleep_millisecs` 决定扫描与分配的间隔，`pages_to_scan` 决定单轮扫描的页数，`max_ptes_none`、`max_ptes_swap`、`max_ptes_shared` 决定折叠时允许的空页、换出页与共享页比例 [3]。这些参数决定了晋升的时机与成功率，也决定了 compaction 与 TLB shootdown 在时间轴上的分布。THP 的“透明”并不意味着无运行时决策，而是把决策交给了这些启发式参数；参数与负载分配模式的匹配程度，直接决定 THP 是收益还是噪声来源。

&emsp;&emsp;与晋升相对的是失效路径。THP 大页在内存压力、`MADV_NOHUGEPAGE`、COW 或显式拆分请求下会被拆回 4 KB 页，记录为 `thp_split`。拆分需要修改页表并对曾映射该区域的 CPU 发送 TLB shootdown；晋升与折叠同样涉及 unmap、页表修改与 shootdown。因此 THP 的成本来源可归为四类：compaction 的物理页迁移、页表修改的 unmap/remap、跨核 TLB shootdown 的 IPI 同步、以及 COW 路径上按 2 MB 粒度复制造成的写放大。这四类成本单次都不大，但在回收、compaction 与逐 PMD 遍历的路径上会被重复触发，累积为尾延迟上的孤立尖峰。这一机制差异是后文 400 µs 停顿归因与第 7 节选型矩阵的基础。

---

## 3 TLB 覆盖模型与工作集算术

### 3.1 工作集需求计算

&emsp;&emsp;以一个交易系统为例：2 MB 订单簿、64 MB DMA 缓冲区、8 MB 策略状态。4 KB 页下，三者分别需要 512、16384、2048 个 TLB 条目，合计 18944 个条目，是可用 L2 TLB 容量的 12.3 倍，TLB 处于持续过载状态。2 MB 页下，同样三个区域仅需 37 个条目，L2 TLB 利用率约 2.4%，稳态 TLB miss 基本消失。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261109_4c49a7.svg)

&emsp;&emsp;图 5 给出容量与需求的对比。该计算解释了“大页在纸面上应该赢”的原因，但不能推出“THP 总是赢”；THP 的运行时整理与失效成本是另一条独立曲线，其风险不体现在平均 TLB 命中率上。

### 3.2 可复现的测量：随机指针追逐

&emsp;&emsp;要验证工作集是否越过 TLB 覆盖线，可采用随机排列的指针追逐：每次解引用依赖上一次结果，CPU 无法用乱序执行或硬件预取掩盖，测出的延迟接近纯 TLB + 内存访问成本。工作集按 2 的幂递增扫描，在 6 MB 附近应出现台阶。代码清单 1 给出完整实现。

```cpp
#include <x86intrin.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

// 构造一个跨越 working_set 字节的环形链表：节点按随机顺序排列，
// 每次 next 跳转都落在不同的 cache line 和页上。
// 依赖链 p = next[p] 让乱序和预取器都帮不上忙，测出来的是纯 TLB + 内存延迟。
static std::vector<size_t> build_ring(size_t working_set, size_t stride) {
    const size_t n = working_set / stride;
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    std::mt19937_64 rng(0x9E3779B97F4A7C15ull);
    std::shuffle(idx.begin(), idx.end(), rng);

    std::vector<size_t> next(n);
    for (size_t i = 0; i < n; ++i) next[idx[i]] = idx[(i + 1) % n];
    return next;
}

static double chase_once(const std::vector<size_t>& next, size_t iters = 1000) {
    volatile size_t sink = 0;
    size_t p = 0;
    unsigned int aux;
    unsigned long long t0 = __rdtscp(&aux);
    for (size_t i = 0; i < iters; ++i) {
        p = next[p];   // 关键：真实依赖链
        sink ^= p;
    }
    unsigned long long t1 = __rdtscp(&aux);
    (void)sink;
    return double(t1 - t0) / double(iters);   // cycles per dereference
}

int main() {
    // stride = 128 字节：每个节点独占 2 条 cache line，排除 false sharing
    const size_t stride = 128;
    for (size_t ws = 64u << 10; ws <= (256u << 20); ws <<= 1) {
        auto next = build_ring(ws, stride);
        for (int w = 0; w < 3; ++w) chase_once(next);   // 预热页表与缓存
        double cyc = chase_once(next);
        std::printf("working_set=%7zu KB  cycles/deref=%7.1f\n", ws >> 10, cyc);
    }
    return 0;
}
```

&emsp;&emsp;代码清单 1 以 `g++ -O2 -std=c++20 tlb_chase.cpp -o tlb_chase` 编译；第 8 节的完整版本由 CMake 管理。在 L2 TLB 为 1536 条目的机器上，输出通常在 4 MB 到 8 MB 之间从个位数周期跳至上百周期。将 `/proc/meminfo` 中的 `AnonHugePages` 与输出对照，可判断 THP 是否抹平该台阶。第 8.2 节用 stride = 4 KB 的隔离版本在 Zen 3 上实测，台阶落在 8–10 MB；stride = 128 B 的版本因叠加 LLC 容量效应而更早抬升、形态更缓。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261109_a88219.svg)

&emsp;&emsp;图 6 中红色曲线为 4 KB 页下的示意命中率，在 6 MB 附近出现台阶；蓝色曲线为 2 MB 页下的示意命中率，台阶被抹平。该图用于说明测量形态，不代表某一具体 CPU 的精确数值。

### 3.3 测量注意事项

&emsp;&emsp;该实验需绑定 CPU 核心、禁用 swap 或对工作集执行 `mlock`，否则缺页与换页会污染延迟；需区分 `dTLB-load-misses` 与页表漫游计数，避免把冷 miss 误判为容量 miss；需在 THP 的 `always`、`madvise`、`never` 三种模式下分别运行，才能建立 THP 配置与台阶形态的因果关系。若工作集未越过 6 MB 线，THP 对本负载的 TLB 收益本就不成立，尾延迟优化应优先转向其他路径。

---

## 4 THP 运行时机制与停顿归因

### 4.1 晋升与 compaction 的成本结构

&emsp;&emsp;合并的前提是找到 512 个物理连续的 4 KB 页；内存运行一段时间后出现外部碎片，内核必须执行内存规整（compaction），搬移物理页以拼凑连续空间。compaction 不是零成本的后台整理：它需要 unmap 目标区域、迁移物理页、重新映射，并对曾映射该区域的所有 CPU 发送 TLB shootdown。在高吞吐存储与数据库场景中，该过程可能持有锁并导致内存分配系统调用突发阻塞 [5]。

&emsp;&emsp;直接 compaction 位于 `alloc_pages` 慢路径（`__alloc_pages_direct_compact` → `try_to_compact_pages` → `compact_zone` → `migrate_pages`），其触发受 `extfrag_threshold`、`compaction_proactiveness` 等参数影响 [3]。THP 的晋升成功率与碎片化程度强相关，失败会回落到 4 KB 页并记录 `thp_fault_fallback` 或 `thp_collapse_alloc_failed`。这意味着“透明”晋升在碎片化条件下会退化为反复尝试与反复失败，其成本由分配路径承担。

### 4.2 TLB shootdown 的放大机制

&emsp;&emsp;Linux 的 TLB 不是硬件一致的。修改页表后，内核必须向所有可能缓存了旧映射的 CPU 发送 IPI，使其失效对应 TLB 条目。x86 上该过程由软件驱动：发起者读取 `mm_cpumask`，过滤 lazy TLB 模式的核，逐核获取 CSD 结构、标记 busy、入队，发送 IPI 并轮询等待目标清除 busy 位。相关路径包括 `flush_tlb_mm_range`、`native_flush_tlb_multi`、`smp_call_function_many` 与 `csd_lock_wait` [6][9]。

&emsp;&emsp;系统刻画显示，bookkeeping（获取与入队 CSD）的延迟随目标核数线性增长，可占总 shootdown 时长的显著比例，发起者本身成为瓶颈 [6]。IPI 传播延迟还受拓扑影响：跨 NUMA 节点的 IPI 延迟显著高于节点内；在 chiplet 架构上，跨 chiplet 延迟高于同 chiplet。THP 的特殊之处在于一次页表操作影响的地址范围：拆分或合并一个 2 MB PMD 条目，需要失效整个 2 MB 范围；当回收、compaction 或 `madvise(MADV_COLD)` 逐 PMD 遍历大区域时，每次 `pmdp_invalidate()` 都可能触发同步 shootdown。单次 shootdown 可能是微秒级，但 THP 把触发粒度从 4 KB 抬到 2 MB，并在多条路径上重复触发。

### 4.3 400 µs 停顿的阶段分解

&emsp;&emsp;回到引言中的案例。系统稳态单周期耗时约 22 µs；某日 14:23 UTC，其中一个周期耗时 400 µs，随后立即恢复。订单流、网络、CPU 利用率、GC、锁竞争均无异常；CPU 利用率仅 14%，交易循环内无内存分配。唯一同步跳变的信号来自 `perf stat`：同一周期内 `minor_page_faults` 记录了 342 次 [11]。根因是 THP 的 compaction 路径：内核在该时间窗口决定将订单簿对应的 512 个 4 KB 页折叠为一个 2 MB 大页，该操作需要短暂 unmap 目标区域、搬移物理页、重新映射，并向其他 CPU 发送 TLB flush IPI。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261109_a900e8.svg)

&emsp;&emsp;图 7 将 400 µs 停顿按阶段拆开：第①段为 unmap 与 PMD 修改；第②段为 342 次 minor fault 的簇发；第③段为 IPI 发送与等待 ack；第④段为重映射。第③段是 shootdown 的真正成本；第②段单次不贵，但簇发叠加后更贵。这解释了“均值 22 µs、max 400 µs”的成因：每一小段单独看都不致命，累积后超过阈值。该案例的关键不在于“THP 有开销”，而在于悬崖的形态：平均延迟几乎不变，P99 可能正常，但 P999 或最大值出现孤立尖峰。第 8.3 节把这一阶段分解回逐事件分布：在本机实测中，2 MB THP 缺页的 p50 约为 600 µs、p99 约为 4.8 ms、尾部可达 18 ms；400 µs 落在分布的偏快一侧，而不是一个普适常数。

---

## 5 二次放大：COW 与碎片化

### 5.1 COW 写放大

&emsp;&emsp;4 KB 页下，fork 后子进程写入 4 字节，只复制一个 4 KB 页。THP 下，同一写操作可能强制复制整个 2 MB 大页，写放大约 50 万倍；较新内核也可能先拆分大页、再按 4 KB 复制（第 8.4 节在本机 Linux 7.0 上实测到后者）。对于使用 fork 做快照、日志或进程派生的服务，该成本直接反映在写延迟与内存带宽上。内核 COW 路径（`do_wp_page`、`copy_present_pte`、folio 引用冻结）对大页的处理需要拆分或整体复制，具体行为取决于内核版本与 THP 拆分策略 [9]。因此定位 COW 成本时，应区分“整页复制”的带宽放大与“拆分 + 页表修改 + shootdown”的延迟成本。对延迟敏感负载，可通过 `MADV_DONTFORK` 或避免在 THP 区域执行高频 fork 来规避。

### 5.2 对齐启发式与碎片化

&emsp;&emsp;较新内核曾在匿名内存分配 ≥2 MB 时将其对齐到 PMD 边界，以期提高 THP 命中率。但在模型推理等负载中，分配块经常略小于 2 MB，对齐后反而产生无法被 `khugepaged` 合并的碎片，导致 TLB 局部性下降与缺页处理开销上升。这说明 THP 的“透明”高度依赖分配模式与内核启发式的匹配程度；不匹配时，它制造碎片的速度可能快于合并碎片的速度 [4][5]。碎片化的可观测信号是 `thp_collapse_alloc_failed` 高企与 `thp_split` 上升并存。

---

## 6 面向尾延迟的可观测性方法

&emsp;&emsp;THP 的问题不是平均性能差，而是尾延迟分布上存在不可从均值预测的孤立尖峰。因此观测必须瞄准分布与事件，而不仅是吞吐。本节给出三组信号及其采集方法。

### 6.1 THP 事件计数

&emsp;&emsp;`/proc/vmstat` 中的 `thp_fault_alloc`、`thp_fault_fallback`、`thp_collapse_alloc`、`thp_collapse_alloc_failed`、`thp_split` 直接反映 THP 的成功、失败与拆分活动。`thp_split` 上升意味着大页正在被拆回小页，通常与内存压力或 `MADV_NOHUGEPAGE` 相关；`thp_collapse_alloc_failed` 高企则指向碎片化阻碍合并。`/proc/meminfo` 的 `AnonHugePages` 给出当前 THP 驻留规模。代码清单 2 给出可用于差分采集的读取器。

```cpp
#include <cstdint>
#include <cstdio>
#include <cstring>

struct ThpStats {
    uint64_t fault_alloc;
    uint64_t fault_fallback;
    uint64_t collapse_alloc;
    uint64_t collapse_alloc_failed;
    uint64_t split;
};

static uint64_t read_vmstat(const char* key) {
    FILE* f = std::fopen("/proc/vmstat", "r");
    if (!f) return 0;
    char name[64];
    unsigned long long val;
    uint64_t result = 0;
    while (std::fscanf(f, "%63s %llu", name, &val) == 2) {
        if (std::strcmp(name, key) == 0) { result = val; break; }
    }
    std::fclose(f);
    return result;
}

static ThpStats thp_snapshot() {
    return ThpStats{
        read_vmstat("thp_fault_alloc"),
        read_vmstat("thp_fault_fallback"),
        read_vmstat("thp_collapse_alloc"),
        read_vmstat("thp_collapse_alloc_failed"),
        read_vmstat("thp_split"),
    };
}

// 使用：每隔 100 ms 采一次，差分后与延迟尖峰时间对齐。
// split 上升 → 大页正在被拆回；collapse_alloc_failed 高企 → 碎片化阻碍合并。
```

### 6.2 TLB miss 与 shootdown 追踪

&emsp;&emsp;`perf stat -e dTLB-load-misses,dTLB-store-misses,page-faults` 可建立 TLB miss 与 THP 配置之间的关联 [11]。但 TLB miss 计数本身不区分正常冷 miss 与 shootdown 引发的失效。要定位 shootdown，需要 trace `smp_call_function_*`、`flush_tlb_mm_range` 或 IPI 路径，观察 shootdown 的目标核数、频率以及与延迟尖峰的时间对齐。若尖峰时刻伴随 `minor_page_faults` 簇发与 IPI 突发，THP compaction/shootdown 的嫌疑显著上升。

&emsp;&emsp;在具备 eBPF 的环境中，可对 `try_to_compact_pages`、`__alloc_pages_direct_compact`、`split_huge_page`、`__split_huge_pmd`、`flush_tlb_mm_range` 设置 kprobe，记录调用频次、持续时间与目标核数；结合 `tracepoint:exceptions:page_fault_user` 与 `sched:sched_switch`，可建立“compaction → minor fault 簇发 → IPI → 调度延迟”的时间线。perf 侧可用 `perf record -e cycles:k -g` 采样内核栈，确认尖峰期间是否落在 compaction 或 shootdown 路径。需注意探针本身的开销与采样偏差，避免观测改变被观测系统。

### 6.3 逐请求延迟分布

&emsp;&emsp;400 µs 案例的教训是：若只采集 1 分钟粒度的 P99，单周期 400 µs 很可能被平滑掉。需要对关键路径做逐请求或高分辨率环形缓冲的延迟记录，至少保留 P999 与 max，并将延迟尖峰时间戳与 `/proc/vmstat` 差分、IPI trace 对齐。代码清单 3 给出保留原始样本并导出分位数的记录器。

```cpp
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

class LatencyRecorder {
public:
    explicit LatencyRecorder(size_t reserve = 1u << 20) { samples_.reserve(reserve); }

    static uint64_t now_ns() {
        using namespace std::chrono;
        return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    }

    void record(uint64_t ns) { samples_.push_back(ns); }
    size_t size() const      { return samples_.size(); }
    void clear()             { samples_.clear(); }

    void dump(const char* label) const {
        if (samples_.empty()) return;
        auto s = samples_;
        std::sort(s.begin(), s.end());
        auto q = [&](double p) {
            return s[static_cast<size_t>(p * (s.size() - 1))];
        };
        std::printf("%-10s n=%zu p50=%llu p99=%llu p999=%llu max=%llu (ns)\n",
                    label, s.size(),
                    (unsigned long long)q(0.50),
                    (unsigned long long)q(0.99),
                    (unsigned long long)q(0.999),
                    (unsigned long long)s.back());
    }

private:
    std::vector<uint64_t> samples_;
};

// 策略循环用法：
// LatencyRecorder rec;
// for (;;) {
//     uint64_t t0 = LatencyRecorder::now_ns();
//     do_one_cycle();
//     rec.record(LatencyRecorder::now_ns() - t0);
//     if (rec.size() >= 100000) { rec.dump("cycle"); rec.clear(); }
// }
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261109_68124f.svg)

&emsp;&emsp;图 8 左面板为均值/P99，平坦在 22 µs；右面板为 max，出现孤立的 400 µs 尖峰。若监控只保留左面板，悬崖不可见。

---

## 7 讨论与工程选型

### 7.1 两种契约

&emsp;&emsp;THP 与 static huge pages（HugeTLB）解决同一个 TLB 覆盖问题，但契约不同。HugeTLB 在启动时预留物理连续内存，通过 `mmap(MAP_HUGETLB)` 或 `hugetlbfs` 显式使用；它们不会被 compaction 搬移，不会被 swap，不会被内核拆分 [7]。代价是必须预分配、应用显式适配，灵活性低。THP 的卖点是零应用改动，内核自动将 ≥2 MB 的分配机会性晋升为大页，内存紧张时再降级。晋升与降级过程本身就是成本来源：compaction、unmap/remap、TLB shootdown IPI、COW 放大，全部被封装在“透明”之下。表 1 给出基于负载特征的选型矩阵。

**表 1  THP 与 HugeTLB 的选型矩阵**

| 负载特征 | 建议策略 | 理由 |
|---|---|---|
| 延迟敏感、工作集稳定、访问模式确定（交易系统、数据库 buffer pool、确定性推理） | 禁用 THP `always`；使用 HugeTLB 或 `MAP_HUGETLB` 预分配 | 获得可预测 TLB 收益，避免运行时 compaction 与 shootdown 尾风险 |
| 吞吐优先、延迟容忍度高、分配模式与内核启发式匹配（批处理、离线分析） | THP `madvise`，对明确区域 `MADV_HUGEPAGE` | 保留大页收益，避免全局 `always` 的不可预测性 |
| 高频 fork、COW 密集（快照、日志、进程派生） | 禁用 THP 或对相关区域 `MADV_NOHUGEPAGE` | 避免 2 MB COW 写放大 |
| 工作集远小于 6 MB | 无需大页优化 | TLB 未过载，THP 收益不成立，风险不必要 |
| 内存碎片化严重、`thp_collapse_alloc_failed` 高企 | 评估 `madvise` 或 HugeTLB；调整 `khugepaged` 参数 | 避免无效 compaction 与碎片制造 |

### 7.2 参数与内核版本注意事项

&emsp;&emsp;THP 的全局模式由 `/sys/kernel/mm/transparent_hugepage/enabled` 控制（`always` / `madvise` / `never`），规整策略由 `defrag` 控制（`always` / `defer` / `madvise` / `never`），`shmem_enabled` 控制 tmpfs 的 THP 行为 [2][3]。较新内核提供 `MADV_COLLAPSE`（6.1+），允许应用同步触发折叠，避免 `khugepaged` 的延迟不可控；但同步折叠本身可能触发 compaction 与 shootdown，需在延迟预算内使用 [8]。内核版本差异还体现在 THP 拆分策略、folio 化、COW 路径与 `khugepaged` 启发式上，生产环境应以目标内核版本的文档与实测为准。

### 7.3 局限性

&emsp;&emsp;本文的 400 µs 阶段分解来自单一生产案例的 tracing，各阶段比例随 CPU 拓扑、内核版本与内存碎片状态变化，不应视为普适常数。指针追逐实验测量的是单线程 TLB 覆盖形态，多线程共享地址空间下的 shootdown 放大需结合 `mm_cpumask` 与目标核数另行评估。可观测性方法依赖内核探针与高分辨率采样，探针开销与采样偏差可能影响结论，需在离线与在线环境交叉验证。第 8 节的实测同样来自单机（6 物理核、单 NUMA、Linux 7.0），绝对数值不应外推；它提供的是可重跑的相对趋势与分布形态。

---

## 8 实测：可复现的 TLB 与 THP 基准

&emsp;&emsp;前七节给出机制、模型与可观测性方法，但关键参数必须落回具体机器。本节给出一套独立的 C++20 基准，在单台桌面级 AMD 机器上回答四个问题：TLB 台阶在哪里、THP 晋升一次要多久、fork/COW 的代价如何转移、TLB shootdown 随目标核数如何增长。实验代码、原始 CSV 与 SVG 绘图脚本均放在独立目录 `tests/tlb`，可由一条命令重跑。

### 8.1 实验环境与设计

&emsp;&emsp;表 2 列出实验平台。选择 THP 的 `madvise` 模式而非修改全局 `always`，有两个原因：第一，避免依赖 root 与全局系统状态，实验可在普通账户下复现；第二，逐映射的 `MADV_HUGEPAGE` / `MADV_NOHUGEPAGE` 能把策略差异精确绑定到被测区域，避免其他进程的 THP 行为污染对照。

**表 2  实验平台与关键参数**

| 项目 | 配置 |
|---|---|
| CPU | AMD Ryzen 5 5600X（Zen 3，6 物理核 / 12 线程，单 NUMA 节点） |
| 内存 | 30 GiB，无 swap |
| 内核 | Linux 7.0.0-42-generic |
| THP | `enabled = always [madvise] never`；`defrag = always defer defer+madvise [madvise] never`；`hpage_pmd_size = 2 MiB` |
| 编译器 | g++ 15.2.0，C++20，CMake Release（`-O3 -DNDEBUG`） |
| 计数器 | `perf_event_paranoid = 1`；`PERF_COUNT_HW_CACHE_DTLB` 可用，与 `l2_dtlb_misses` 交叉验证一致 |
| 绑定 | 单线程实验固定 CPU 0；shootdown 实验主线程固定 CPU 11 |

&emsp;&emsp;四个实验的矩阵如表 3。所有延迟均使用 `CLOCK_MONOTONIC_RAW` 或经校准的 invariant TSC 计时；指针追逐每次扫描取 7 次试验的中位数，再做 3 次独立扫描并取中位数；故障/折叠实验保留全部原始样本，因此可以同时报告 p50、p99、p999 与 max。

**表 3  实验矩阵**

| 编号 | 问题 | 方法 | 主要指标 |
|---|---|---|---|
| A | 4 KB 工作集何时越过 L2 TLB 覆盖线 | 随机 Hamiltonian 环指针追逐，stride = 4 KB，256 KB–256 MB | ns/deref、L2 TLB miss 率 |
| B | 一次 2 MB 晋升/折叠的逐事件成本 | 1 GiB 区域首次触碰；逐 2 MB 同步 `MADV_COLLAPSE` | 单事件延迟分布、vmstat 差分 |
| C | fork 与 COW 的代价如何随页大小转移 | 512 MiB 区域 fork 后，子进程逐 2 MB 写 1 字节 | fork 耗时、COW 写延迟、Private_Dirty 增量 |
| D | shootdown 是否随目标核数放大 | 多线程读共享页，主线程反复切换 `mprotect` 权限 | mprotect 延迟 vs worker 数 |

&emsp;&emsp;实验 A 的 stride 取 4 KB，而不是代码清单 1 的 128 B：4 KB stride 让每个节点独占一个 4 KB 页，把 TLB 覆盖从 cache 容量中分离出来；128 B stride 的敏感性结果仍保留在 `results/tlb_chase_stride128.csv`，用于说明 cache 效应叠加后台阶会变缓。实验 B 的 `MADV_COLLAPSE` 需要 Linux 6.1+；本机内核满足该条件。实验 D 的 `mprotect` 是 THP 拆分/折叠中失效步骤的受控代理：它触发同一条 `flush_tlb_mm_range` → IPI → ack 路径，但不包含真实 THP 路径上的 unmap/remap 与页迁移，因此得到的是 shootdown 成本的下界。

&emsp;&emsp;全套实验由一条脚本驱动，构建与运行命令如下：

```bash
cmake -S tests/tlb -B tests/tlb/build -DCMAKE_BUILD_TYPE=Release
cmake --build tests/tlb/build -j"$(nproc)"
tests/tlb/scripts/run_all.sh
```

&emsp;&emsp;脚本先构建四个可执行文件，再依次运行 `tlb_chase`、`fault_latency`、`cow` 与 `shootdown`，最后调用 `tools/svgplot.py` 生成图 9–12。原始 CSV 保留在 `results/`，便于复核分位数与重绘；`QUICK=1` 可缩短扫描与采样，用于冒烟测试。

&emsp;&emsp;对应的 CMake 配置如下（节选），C++20 与 `-O3` 由 Release 构建提供：

```cmake
cmake_minimum_required(VERSION 3.20)
project(tlb_bench LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)
foreach(bench tlb_chase fault_latency cow shootdown)
    add_executable(${bench} src/bench_${bench}.cpp)
    target_include_directories(${bench} PRIVATE include)
    target_link_libraries(${bench} PRIVATE Threads::Threads)
endforeach()
```


&emsp;&emsp;公共设施中的两个关键点是“在首次触碰前绑定 THP 策略”和“用硬件计数器区分 TLB 行为”。代码清单 4 给出 `include/bench_common.hpp` 的节选；完整实现还包含 `smaps` 汇总与逐映射的 `AnonHugePages` 统计。

```cpp
// include/bench_common.hpp（节选）
constexpr size_t kPage2M = 2u << 20;

// 2 MB 对齐 + 逐映射 THP 策略；必须在首次触碰前调用。
void* map_region(size_t len, int policy, bool align_2m = true) {
    size_t total = len + (align_2m ? kPage2M : 0);
    char* raw = static_cast<char*>(mmap(nullptr, total, PROT_READ | PROT_WRITE,
                                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    uintptr_t a = (reinterpret_cast<uintptr_t>(raw) + kPage2M - 1) & ~(kPage2M - 1);
    size_t head = align_2m ? a - reinterpret_cast<uintptr_t>(raw) : 0;
    char* base = raw + head;
    if (head) munmap(raw, head);
    if (size_t tail = total - head - len) munmap(base + len, tail);
    madvise(base, len, policy ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    return base;
}

// L2 TLB miss（触发 page walk）的硬件计数器。
class DtlbMissCounter {
    int fd_ = -1;
public:
    DtlbMissCounter() {
        perf_event_attr pe{};
        pe.type = PERF_TYPE_HW_CACHE;
        pe.size = sizeof(pe);
        pe.config = PERF_COUNT_HW_CACHE_DTLB |
                    (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                    (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        fd_ = static_cast<int>(syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0));
    }
    void start() { ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
                   ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0); }
    void stop()  { ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0); }
    uint64_t value() const {
        uint64_t v = 0; ::read(fd_, &v, sizeof(v)); return v;
    }
};
```


### 8.2 实验 A：TLB 覆盖台阶

&emsp;&emsp;图 9 给出实测结果。左面板是每次依赖解引用的延迟，右面板是 L2 TLB 未命中率（该计数器与 `l2_dtlb_misses` 一致，未命中会触发 page walk）。4 KB 曲线在 8–10 MB 之间出现明显台阶：L2 TLB 未命中率从 8 MB 的 2.9% 跳到 10 MB 的 62.0%，12 MB 时达到 98.8%；延迟从 8 MB 的 10.6 ns 升到 12 MB 的 14.6 ns，随后在 16–64 MB 稳定在 14–19 ns。2 MB THP 曲线在同一区间保持在 9.6–10.4 ns，未命中率始终低于 0.16%。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261109_ca9591.svg)

**表 4  指针追逐的关键数据点（stride = 4 KB，三次独立运行的中位数）**

| 工作集 | 4 KB 延迟 | 4 KB L2 miss | 2 MB THP 延迟 | 2 MB L2 miss |
|---:|---:|---:|---:|---:|
| 4 MB | 9.40 ns | 0.13% | 9.55 ns | 0.001% |
| 8 MB | 10.64 ns | 2.90% | 9.87 ns | <0.001% |
| 10 MB | 12.71 ns | 62.04% | 9.63 ns | <0.001% |
| 12 MB | 14.55 ns | 98.79% | 9.71 ns | 0.001% |
| 16 MB | 15.65 ns | 98.75% | 9.73 ns | 0.001% |
| 32 MB | 15.93 ns | 98.89% | 10.08 ns | 0.001% |
| 64 MB | 18.64 ns | 99.00% | 10.44 ns | 0.001% |
| 128 MB* | 53.42 ns | 99.21% | 28.31 ns | 0.002% |
| 256 MB | 111.76 ns | ≈100% | 94.44 ns | 0.153% |

\*128 MB 位于 LLC 过渡区，单次运行方差较大；表中为三次独立运行的中位数。128 MB 处 4 KB 与 THP 的单次中位数分别在 45.7–106.2 ns 与 27.2–38.2 ns 之间波动，因此不把该点作为强结论依据。

&emsp;&emsp;表 4 支持三个结论。第一，台阶位置与“1536 条目 × 4 KB = 6 MB”的算术模型同量级，但在本机落在 8–10 MB，对应约 2048–2560 个 4 KB 页；这说明覆盖线是机器相关的，不能把 6 MB 当作普适常数。第二，THP 的收益集中在工作集刚越过覆盖线、但仍能放进 LLC 的区间：10–64 MB 上 4 KB 为 12.7–18.6 ns，THP 为 9.6–10.4 ns，降幅约 23%–44%。第三，工作集达到 256 MB 后，两者都进入 DRAM 延迟主导区，THP 的收益收窄到约 15%；128 MB 位于 LLC 过渡区，run-to-run 方差较大，因此不作为强结论的依据。TLB 不再是主要瓶颈时，继续围绕大页做优化收益有限。

&emsp;&emsp;工作集只有 2–4 MB 时，THP 没有收益，甚至略慢：2 MB 时约慢 4 ns，4 MB 时基本持平。这不矛盾：此时 4 KB 页的 512–1024 个条目仍能装进 L2 TLB，大页没有消除任何 page walk，却改变了 TLB 组相联与页对齐行为；收益必须等工作集真正越过覆盖线才出现。这也解释了为什么“先测工作集、再谈大页”是必要的。

&emsp;&emsp;敏感性实验（stride = 128 B）显示，当每个节点只占 128 B 时，cache line 足迹是 stride = 4 KB 时的 32 倍，曲线在 6–10 MB 就开始受 LLC 容量影响，台阶更缓且噪声更大。这从反面确认：stride = 4 KB 的干净台阶确实来自 TLB，而不是 cache。

&emsp;&emsp;代码清单 5 给出环构造与依赖链的核心。两种策略使用同一随机置换，差异只来自页大小；`stride` 由命令行传入，4 KB 用于隔离 TLB，128 B 用于敏感性对照。

```cpp
// src/bench_tlb_chase.cpp（节选）
struct Ring { char* base; size_t len; size_t nodes; };

Ring build_ring(size_t len, size_t stride, int policy) {
    Ring r{static_cast<char*>(tlb::map_region(len, policy)),
           len, len / stride};
    std::vector<uint32_t> idx(r.nodes);
    std::iota(idx.begin(), idx.end(), 0u);
    std::mt19937_64 rng(0x9E3779B97F4A7C15ull ^ (len << 1) ^ (stride << 17));
    std::shuffle(idx.begin(), idx.end(), rng);      // 两种策略使用同一置换
    for (size_t i = 0; i < r.nodes; ++i) {
        size_t self = size_t(idx[i]) * stride;
        size_t next = size_t(idx[(i + 1) % r.nodes]) * stride;
        *reinterpret_cast<uint64_t*>(r.base + self) = next;  // 存下一节点偏移
    }
    return r;
}

double chase(char* base, size_t iters) {
    uint64_t p = 0, sink = 0; unsigned aux;
    uint64_t t0 = __rdtscp(&aux);
    for (size_t i = 0; i < iters; ++i) {
        p = *reinterpret_cast<const uint64_t*>(base + p);  // 真实依赖链
        sink ^= p;
    }
    uint64_t t1 = __rdtscp(&aux);
    asm volatile("" : : "r"(sink) : "memory");
    return static_cast<double>(t1 - t0) / static_cast<double>(iters);
}
```


### 8.3 实验 B：晋升与折叠的逐事件成本

&emsp;&emsp;实验 B 在 1 GiB 匿名区域上分别测量三类事件：4 KB 页首次触碰（262 144 次）、2 MB THP 首次触碰（512 次）、以及先铺满 4 KB 页后对每个 2 MB 块同步执行 `MADV_COLLAPSE`（512 次）。图 10 是逐事件延迟的 CDF，横轴为对数尺度。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261109_edf34a.svg)

**表 5  首次触碰与同步折叠的延迟分布**

| 事件 | 次数 | p50 | p99 | p999 | max |
|---|---:|---:|---:|---:|---:|
| 4 KB 缺页 | 262 144 | 1.44 µs | 4.71 µs | 40.5 µs | 879 µs |
| 2 MB THP 缺页 | 512 | 600 µs | 4.83 ms | 11.3 ms | 18.2 ms |
| `MADV_COLLAPSE` | 456 成功 / 56 失败 | 663 µs | 2.63 ms | 3.73 ms | 7.17 ms |

&emsp;&emsp;图 10 和表 5 给出本文最重要的一组实测事实。4 KB 缺页的 p50 只有 1.44 µs，p99 为 4.71 µs；THP 缺页的 p50 是 600 µs，p99 是 4.83 ms，尾部到 18.2 ms；同步折叠的 p50 是 663 µs，p99 是 2.63 ms。也就是说，单次 2 MB 晋升比单次 4 KB 缺页慢约 400 倍，尾部慢三个数量级。按字节摊薄，两者其实接近：4 KB 缺页约 0.35 ns/字节，THP 缺页约 0.29 ns/字节；两种方式铺满 1 GiB 的总时间也几乎相同（471.6 ms 对 466.8 ms）。差别不在总吞吐，而在事件粒度与尾部分布：延迟敏感系统关心的是“一次事件阻塞多久”，而不是摊薄后的吞吐。这正是“均值不可见、P999 可见”的结构性来源。

&emsp;&emsp;本次 THP 缺页中，`/proc/vmstat` 记录 `thp_fault_alloc` 增加 440、`thp_fault_fallback` 增加 72，最终 `AnonHugePages` 为 880 MiB（区域 1024 MiB 的 85.9%）；折叠实验中 `thp_collapse_alloc` 增加 458、`thp_collapse_alloc_failed` 增加 54，`AnonHugePages` 为 912 MiB（89.1%）。这两组计数器与逐事件延迟对齐，正好构成第 6.1 节所说的归因证据链：成功晋升对应几百微秒的分配与清零，失败回退对应更长的 compaction 尝试；折叠失败则直接指向碎片化。

&emsp;&emsp;与 400 µs 案例对照，600 µs / 663 µs 的 p50 表明该案例落在分布的偏快一侧，而不是常数。不同内核、不同碎片状态与不同 CPU 拓扑会把这条 CDF 整体平移；可迁移的结论是形态，而不是具体数值。

&emsp;&emsp;代码清单 6 给出逐事件计时与同步折叠的核心。`samples` 保留全部原始值，因此能计算 p999 与 max；折叠循环先对每个 2 MB 块设置 `MADV_HUGEPAGE`，再计时 `MADV_COLLAPSE`，把“晋升”从后台线程拉到可归因的同步路径。

```cpp
// src/bench_fault_latency.cpp（节选）
std::vector<uint64_t> samples;   // 逐事件延迟
std::vector<int> rc;             // 折叠返回值（0 或 errno）

// 首次触碰：4 KB 或 2 MB 步长，逐事件计时
for (size_t off = 0; off < len; off += page) {
    uint64_t t0 = tlb::now_ns();
    *reinterpret_cast<volatile uint64_t*>(base + off) = 1;
    uint64_t t1 = tlb::now_ns();
    samples.push_back(t1 - t0);
}

// 同步折叠：先铺 4 KB 页，再逐 2 MB 调 MADV_COLLAPSE
for (size_t off = 0; off < len; off += tlb::kPage2M) {
    char* p = base + off;
    madvise(p, tlb::kPage2M, MADV_HUGEPAGE);   // VMA 原本是 NOHUGEPAGE
    uint64_t t0 = tlb::now_ns();
    int r = madvise(p, tlb::kPage2M, MADV_COLLAPSE);
    uint64_t t1 = tlb::now_ns();
    samples.push_back(t1 - t0);
    rc.push_back(r == 0 ? 0 : errno);
}
```


### 8.4 实验 C：fork 与 COW 的代价转移

&emsp;&emsp;实验 C 在 512 MiB 区域上先逐 4 KB 触碰，再 fork；子进程在每个 2 MB 边界写 1 字节（共 256 次）。图 11 左面板是 fork 耗时，右面板是 COW 首次写延迟。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261109_3ef4f3.svg)

**表 6  fork 与 COW 的实测对比（512 MiB，256 次写）**

| 指标 | 4 KB 页 | 2 MB THP | 对比 |
|---|---:|---:|---|
| `fork()` 耗时 | 10.86 ms | 0.30 ms | THP 快 36× |
| COW 首次写 p50 | 2.58 µs | 9.15 µs | THP 慢 3.5× |
| COW 首次写 p99 | 6.02 µs | 16.07 µs | THP 慢 2.7× |
| Private_Dirty 增量 | 1.0 MiB | 1.0 MiB | 均为 4 KB/次 |

&emsp;&emsp;fork 的差异来自页表复制：4 KB 页需要复制 131 072 个 PTE，2 MB 页只需复制 256 个 PMD，THP 因此快 36 倍。COW 的方向相反：子进程第一次写某个 2 MB 区域时，内核需要先把 PMD 拆分、建立 4 KB 页表，再执行 4 KB 粒度的 COW；每次写的延迟从 2.58 µs 升到 9.15 µs。Private_Dirty 增量显示两种配置都只复制了约 4 KB/次，总复制量 1.0 MiB，而不是 512 MiB。

&emsp;&emsp;这与第 5.1 节“THP 会复制整个 2 MB 页、写放大约 50 万倍”的表述需要区分内核版本：本机 Linux 7.0 在 COW 路径上选择先拆分、再按 4 KB 复制，因此没有出现带宽型写放大；代价转移到了拆分、页表修改与 TLB 失效上。若目标内核选择整页 COW，Private_Dirty 会跳到 2 MB/次，写放大才会出现。工程结论不变——高频 fork/COW 的负载应避免 THP 或使用 `MADV_DONTFORK` / `MADV_NOHUGEPAGE`——但定位成本时不能只看复制字节数，还要看拆分与 shootdown 的延迟。

&emsp;&emsp;代码清单 7 是 fork/COW 的核心：父进程记录 `fork()` 耗时（包含页表复制），子进程在每个 2 MB 边界写 1 字节，并用 `Private_Dirty` 增量反推实际复制粒度。样本通过独立管道回传，避免测量进程自身的 `smaps` 开销污染子进程的写循环。

```cpp
// src/bench_cow.cpp（节选）
uint64_t t0 = tlb::now_ns();
pid_t pid = fork();
uint64_t fork_ns = tlb::now_ns() - t0;          // 父进程计时，含页表复制

if (pid == 0) {
    volatile char* v = base;
    uint64_t dirty0 = tlb::smaps_rollup_kb("Private_Dirty:");
    std::vector<uint64_t> samples;
    for (size_t i = 0; i < len / tlb::kPage2M; ++i) {
        uint64_t s = tlb::now_ns();
        v[i * tlb::kPage2M] ^= 1;               // 每个 2 MB 区域只写 1 字节
        samples.push_back(tlb::now_ns() - s);
    }
    uint64_t dirty1 = tlb::smaps_rollup_kb("Private_Dirty:");
    // 将 samples 与 (dirty1 - dirty0) 打包后经 result_fd 回传
    _exit(0);
}
```


### 8.5 实验 D：TLB shootdown 的核数缩放

&emsp;&emsp;实验 D 让 N 个 worker 线程持续读共享页，主线程反复把范围从可写切到只读，再切回；每次切换都必须向 `mm_cpumask` 中的 CPU 发送 IPI 并等待 ack。图 12 给出 4 KB 与 2 MB 范围的 p50/p99。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261109_5de47c.svg)

**表 7  mprotect 延迟随 worker 数的变化（µs）**

| worker 数 | 4 KB p50 | 4 KB p99 | 2 MB p50 | 2 MB p99 |
|---:|---:|---:|---:|---:|
| 0 | 1.35 | 1.98 | 1.25 | 1.90 |
| 1 | 2.89 | 4.54 | 2.00 | 3.00 |
| 6 | 5.19 | 11.08 | 3.70 | 5.88 |
| 11 | 5.92 | 13.52 | 4.40 | 7.34 |

&emsp;&emsp;从 0 到 11 个 worker，4 KB 范围的 p50 从 1.35 µs 升到 5.92 µs（4.4×），2 MB 范围从 1.25 µs 升到 4.40 µs（3.5×）；p99 分别升到 13.5 µs 与 7.3 µs，max 可达 48 µs 与 101 µs。前 6 个 worker 占满物理核，之后加入 SMT 兄弟；SMT 阶段仍有增长，但斜率小于物理核阶段。单 NUMA 节点内没有跨节点 IPI，因此这个绝对量级不大；在多路系统上，bookkeeping 随目标核数线性增长，尾延迟会被进一步放大。

&emsp;&emsp;一个反直觉的结果是：2 MB 范围的延迟并不比 4 KB 范围高，甚至略低。这说明单次 shootdown 的成本主要由目标核数与 IPI 等待决定，而不是失效范围的大小；THP 的放大来自“每个 PMD 都触发一次”这一频率因素，以及拆分/折叠前后的 unmap、页表修改与迁移，而不是单次 2 MB 失效本身贵 512 倍。第 4.2 节“THP 把触发粒度从 4 KB 抬到 2 MB，并在多条路径上重复触发”的判断，在实测中应更准确地表述为：范围不是主要成本，重复次数与目标核数才是。

&emsp;&emsp;代码清单 8 是 shootdown 的核心。worker 持续读共享页，确保其 CPU 在 `mm_cpumask` 中；主线程只对权限切换本身计时，恢复可写的那次切换放在计时区间之外。这样得到的是单次失效等待 ack 的成本，而不是两次切换的总和。

```cpp
// src/bench_shootdown.cpp（节选）
std::atomic<bool> stop{false};
std::atomic<uint64_t> sink{0};
std::thread worker([&] {
    tlb::pin_cpu(cpu);
    volatile char* p = base + (cpu * 64) % range;
    uint64_t s = 0;
    while (!stop.load(std::memory_order_relaxed)) s += *p;
    sink.fetch_xor(s, std::memory_order_relaxed);
});

std::vector<uint64_t> samples;
for (int i = 0; i < iters; ++i) {
    uint64_t t0 = tlb::now_ns();
    mprotect(base, range, PROT_READ);            // 触发 flush_tlb_mm_range
    uint64_t t1 = tlb::now_ns();                 // 等所有目标 CPU ack
    samples.push_back(t1 - t0);
    mprotect(base, range, PROT_READ | PROT_WRITE);
}
stop.store(true);
worker.join();
```


### 8.6 与可观测性信号的对应

&emsp;&emsp;表 8 把实验 B 的延迟与 `/proc/vmstat` 差分放在一起。它说明第 6 节的信号体系不是理论清单：`thp_fault_alloc` 与 `thp_fault_fallback` 的比例直接解释 THP 缺页分布的双峰形态；`thp_collapse_alloc` 与 `thp_collapse_alloc_failed` 解释折叠实验的长尾；`AnonHugePages` 则给出策略是否真正生效的最终证据。

**表 8  实验 B 的 vmstat 差分与 THP 驻留**

| 计数 / 场景 | 4 KB 缺页 | 2 MB THP 缺页 | 同步折叠 |
|---|---:|---:|---:|
| `thp_fault_alloc` | 0 | 440 | 0 |
| `thp_fault_fallback` | 0 | 72 | 0 |
| `thp_collapse_alloc` | 0 | 0 | 458 |
| `thp_collapse_alloc_failed` | 0 | 0 | 54 |
| `AnonHugePages` | 0 | 880 MiB | 912 MiB |

&emsp;&emsp;生产环境应按第 6.3 节保留逐请求 P999 与 max，并把尖峰时间戳与表 8 的差分对齐：若尖峰伴随 `thp_fault_fallback` 或 `thp_collapse_alloc_failed` 上升，优先怀疑 compaction；若伴随 `thp_split` 上升，优先怀疑回收、COW 或显式拆分；若伴随 IPI 路径的 `smp_call_function_*` 延迟，则指向 shootdown。只保留分钟级 P99 会把 600 µs–18 ms 的事件平均掉，从而看不到悬崖。

### 8.7 对选型矩阵的实测修正

&emsp;&emsp;把四个实验放回表 1 的选型矩阵，可以得到几条更具体的工程判断。第一，工作集小于 8 MB 时，4 KB 页的 L2 miss 仍低于 10%，THP 收益不成立，不必为“可能的大页收益”承担晋升尾风险。第二，工作集在 10–64 MB 且访问模式稳定时，THP 的 TLB 收益是实测的 23%–44%，`madvise` + 区域级 `MADV_HUGEPAGE` 是合理选择；但晋升必须在启动或预热阶段完成，不能把 0.6–18 ms 的首次触碰留在延迟敏感路径上。第三，工作集达到 256 MB 后，DRAM 延迟占主导，THP 的 TLB 收益收窄到约 15%；128 MB 处于 LLC 过渡区、方差较大，此时更值得优化的是访问局部性、预取与 NUMA，而不是继续追求大页。第四，延迟敏感且工作集稳定的系统仍应优先 HugeTLB 或启动期 `MADV_COLLAPSE` 的静态契约；THP 的 `always` 模式把晋升时机交给内核，尾延迟不可控。第五，高频 fork/COW 的负载应实测目标内核的 COW 策略：本机是拆分后 4 KB 复制，代价在拆分与 shootdown；另一些内核可能整页复制，代价在带宽。两者的规避手段相同，但监控指标不同。

&emsp;&emsp;本节结果的适用边界需要明确：单机、单 NUMA、6 物理核、Linux 7.0、THP `madvise` 模式、1 GiB 区域、`mprotect` 代理 shootdown。绝对数值不应外推为其他机器的常数；可迁移的是三条相对结论：TLB 台阶由条目数 × 页大小决定；THP 的收益在覆盖线右侧、LLC 左侧最大；THP 的代价在逐事件分布的长尾，而不在摊薄后的均值。

## 9 结论

&emsp;&emsp;TLB 的账以条目计算，大页的收益是结构性的；THP 的代价也是结构性的，只是它藏在均值以下。悬崖不是因为 THP 损坏，而是因为“自动”与“可预测”在延迟敏感系统中通常不是同一件事。

&emsp;&emsp;第 8 节的实测给出这套账在 Zen 3 上的刻度：4 KB 覆盖线约 8–10 MB；THP 在 10–64 MB 区间把每次解引用延迟降低 23%–44%；但单次晋升/折叠的 p50 为 0.6–0.7 ms，尾部可达 7–18 ms。收益与代价确实分别落在均值与尾部分布上，这也解释了为什么只保留平均延迟的监控会让悬崖消失。

&emsp;&emsp;工程上应先回答三个问题：工作集是否越过 TLB 覆盖线；THP 的晋升与拆分事件是否与尾延迟尖峰时间对齐；延迟分布是否保留了 P999 与 max。若答案指向 THP 运行时整理，延迟敏感路径应切向 HugeTLB 的静态契约；吞吐优先的弹性负载可保留 `madvise` 模式，但必须把 `thp_split`、`thp_collapse_alloc_failed` 与 P999 一起纳入监控。TLB 的容量是条目，大页的收益是结构，THP 的风险是分布；只有分布可观测，悬崖才可管理。

