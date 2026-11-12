# 基于性能计数器的 GPU Occupancy 诊断与优化

&emsp;&emsp;**摘要：** nvidia-smi 里的高利用率并不等于高效执行，Occupancy 也不是“越高越好”的 KPI。这篇博客围绕 NVIDIA Nsight Compute（ncu）的硬件性能计数器，梳理一套自己常用的 Occupancy 排查思路：把 Theoretical Occupancy、Achieved Occupancy、限流因子、Eligible Warps 与 Stall Reasons 放在一起读，从而判断一个 kernel 到底是“座位不够”“warp 都卡住了”还是“grid 根本没铺开”。文中给出四类常见读数组合、从 nsys 到 ncu 再到代码改动的排查流程、若干调优手段，以及实践中的三个 profiling 案例。实验部分在 RTX 3050 上给出可运行的代理实测：用 CUDA occupancy API、软件驻留采样与 CUDA event 验证四类读数组合；本机 ncu 因权限无法读取 Eligible Warps/Stall Reasons，相关列标为 N/A。核心结论只有一句：Occupancy 是隐藏延迟的手段，不是优化的目标本身。

&emsp;&emsp;**关键词：** GPU 性能优化；Occupancy；性能计数器；Nsight Compute；延迟隐藏

## 1 引言

&emsp;&emsp;GPU 性能调优里，Occupancy 大概是被误用最多的指标之一。看到 `achieved_occupancy` 只有 40%，第一反应往往是“调 block size 把它拉上去”；但真拉上去之后，性能纹丝不动的场景并不少见。

&emsp;&emsp;问题出在 nvidia-smi 那个“利用率”上：它只说明 GPU 在忙，不说明忙得有效率。一个配置糟糕的 kernel 可以让大部分 SM 完全闲置，监控工具依然显示有任务在跑。所以 Occupancy 排查真正要回答的不是“这个百分比是多少”，而是三个问题：

1. 实际驻留了多少 warp？
2. 这些 warp 为什么停住？
3. 是什么限制了驻留量——寄存器、共享内存，还是 block 数？

&emsp;&emsp;这篇博客记录的就是用 ncu 的性能计数器回答这三个问题的过程。内容偏实操：先讲两个 Occupancy 计数器的区别，再讲怎么和 Stall Reasons 联读，然后给出四类常见读数组合对应的排查方向、一套可复用的流程、几个调优手段，最后是三个实践案例和一个代理实验章节。

&emsp;&emsp;需要先摆明的观点：**Occupancy 是隐藏延迟的手段，不是优化的目标**。刻意降低 Occupancy 换取更高 ILP 或更少寄存器溢出的 kernel 并不少见，看到低 Occupancy 就往上调，经常会调错方向。

## 2 背景：两个 Occupancy 计数器

### 2.1 Theoretical 与 Achieved

&emsp;&emsp;ncu 的 Occupancy 区段会给出两个核心计数器：

- **Theoretical Occupancy**：受寄存器、共享内存、block size 约束的理论上限，回答硬件“允许”多少 warp 同时驻留。
- **Achieved Occupancy**：kernel 运行期间实际活跃 warp 占理论上限的加权平均，回答硬件“实际”驻留了多少 warp。

&emsp;&emsp;这两个值的差距本身就是信号。举一个实际遇到的 GEMM kernel：Theoretical Occupancy 83.33%，Achieved 只有 32.28%，同时 Active warps per scheduler 是 3.81（上限 12），Eligible warps 低到 0.36。翻译一下就是：硬件留足了座位，但实际坐进来的 warp 远不够，而且坐进来的那批大多在打盹。根因既不是寄存器（48/thread 不算高），也不是共享内存，而是 grid 规模不足加上归约依赖链造成的严重停顿。

&emsp;&emsp;另一个极端例子是个 2×2 矩阵的 kernel：Achieved Occupancy 2.09%，grid 只有 1 个 block、9 个线程，多个 SM 完全空转。ncu 会直接给出预期加速比，并建议增大 grid、block size 用 32 的倍数。这类情况下计数器不只是“诊断”，还顺带告诉了你修好之后大概能赚多少。

### 2.2 限流因子计数器

&emsp;&emsp;ncu 为 Occupancy 准备了细粒度的限流因子计数器，直接回答“是谁卡住了驻留量”：

| 计数器 | 含义 |
|---|---|
| `launch__occupancy_limit_registers` | 寄存器用量限制了每 SM 的驻留 warp 数 |
| `launch__occupancy_limit_shared_mem` | 共享内存用量限制了并发 block 数 |
| `launch__occupancy_limit_blocks` | 每 SM 最大 block 数成为瓶颈 |

&emsp;&emsp;所以调优的第一步应该是 `ncu --section Occupancy`，把 Theoretical、Achieved 和 Limiting Factor 三项读出来，而不是凭直觉改 block size 或加 shared memory。

### 2.3 为什么不能只看 Occupancy

&emsp;&emsp;单独一个 Occupancy 数值几乎没有判读价值。Achieved 是 40% 时，如果这 40% 的 warp 已经足够把内存延迟盖住，40% 就是合适的；反过来，Occupancy 很高但大量 warp 卡在同一条依赖链上，性能照样上不去。Occupancy 必须和停顿原因、可调度 warp 数、Roofline 约束放在一起看，才有意义。

## 3 计数器联读：四类读数组合

### 3.1 联读框架

&emsp;&emsp;我习惯把四类信息摆在一起读，如表 1。先用 SpeedOfLight 判断 bound 类型：SM Throughput 和 Memory Throughput 都低（比如都低于 40%），一般是 latency-bound，这时候 Occupancy 和 Stall Reasons 才是主要方向；Memory Throughput 已经很高，Occupancy 再高也救不了带宽瓶颈；SM Throughput 很高，问题多半在计算密度而不是驻留 warp 数。

**表 1 四类计数器信息的联读框架**

| 维度 | 主要计数器/区段 | 回答的问题 | 怎么用 |
|---|---|---|---|
| 理论上限 | Theoretical Occupancy | 硬件允许多少 warp | 判断资源约束是否成立 |
| 实际驻留 | Achieved Occupancy | 实际驻留多少 warp | 判断填充效率 |
| 限流因子 | `occupancy_limit_*` | 什么卡住了驻留量 | 定位寄存器/共享内存/block 数 |
| 停顿与调度 | Eligible Warps、Stall Reasons | warp 为何停住 | 区分延迟隐藏不足与管线饱和 |

### 3.2 Stall Reasons 对照

&emsp;&emsp;ncu 的 WarpStateStats / SchedulerStats 区段会给出各 stall reason 的占比，常见几项和 Occupancy 的关系大致是：

| Stall Reason | 含义 | 和 Occupancy 的关系 |
|---|---|---|
| Long Scoreboard | 等 Global/Local Memory 返回 | 提高 Occupancy 可增加可切换 warp，有助于隐藏延迟 |
| LG Throttle | Load/Store 单元饱和 | 该优化访存模式（合并、减少冗余读写），不是盲目加 Occupancy |
| MIO Throttle | Shared Memory / Constant 管线饱和 | 减少 shared memory 访问或 bank conflict |
| Not Selected | warp 就绪但没被调度器选中 | 通常说明 Occupancy 已经够了，不是瓶颈 |
| Barrier | 等 `__syncthreads()` | 减少同步点，或用 warp-level 原语 |

&emsp;&emsp;一个 memory-bound kernel 的 Long Scoreboard 占到 70%，如果 Achieved Occupancy 还很低，加 Occupancy 是合理的；但如果 Occupancy 已经贴着上限，Long Scoreboard 还是 70%，那就说明瓶颈在单次访存的延迟或带宽，而不是“并发 warp 不够”。ncu 的 Scheduler Statistics 页面把 Active Warps、Eligible Warps 和 stall 分布放在一起，三者联读才不会把手段当目标。

### 3.3 Roofline 给出的天花板

&emsp;&emsp;Roofline 告诉你 kernel 离算力屋顶和带宽屋顶还有多远。如果 kernel 落在左下角的 latency-bound 区域，调 Occupancy 是合理方向；如果已经贴着 Memory Roof，那该做的是提高算术强度（tiling、fusion、少搬数据），而不是继续推 Occupancy。

### 3.4 四类典型读数组合

&emsp;&emsp;把上面的信息组合起来，实际排查中常见四类情况（表 2）。

**表 2 四类常见读数组合**

| 情况 | 计数器特征 | 典型根因 | 处理方向 |
|---|---|---|---|
| 限流因子受限 | Achieved ≪ Theoretical，限流因子指向寄存器或共享内存 | 每线程寄存器过高，或 per-block 共享内存过大 | `__launch_bounds__`、调 block size、减共享内存 |
| 依赖链停顿 | Achieved 不低但 Eligible Warps 很低，stall 集中 | warp 驻留了，但都卡在同一条依赖链上 | 增加独立累加器、提高 ILP/MLP |
| grid 规模不足 | Theoretical 和 Achieved 都低 | block 数铺不满所有 SM | 增大问题规模、合并小 kernel、persistent kernel |
| 访存管线饱和 | Occupancy 较高，LG/MIO Throttle 占主导 | 访存或 shared memory 管线饱和 | 合并访问、消除 bank conflict、减少共享内存流量 |

## 4 排查流程与调优手段

### 4.1 从全局到局部的排查顺序

&emsp;&emsp;ncu 是重放 kernel 来采计数器的，开销不小，不适合全量扫描。所以实际流程是两级：

1. **先用 nsys 或 PyTorch Profiler 看全局时间轴**，确认 GPU 时间花在哪些 kernel 上，有没有 CPU-GPU 同步导致的空闲间隙。这一步的目的是缩小 ncu 的打击范围。
2. **对排名靠前的 1–3 个热点 kernel 跑 ncu**：`--section SpeedOfLight` 定 bound 类型，`--section Occupancy` 读 Theoretical/Achieved/限流因子，`SchedulerStats`/`WarpStateStats` 看 Eligible Warps 和 stall 分布。
3. **结合 Roofline 判断天花板**，决定 Occupancy 是不是主要杠杆。
4. **每次只改一个变量，重新采计数器验证**。ncu 报告里的 “Est. Speedup” 可以当优先级参考，但最终收益以端到端 profiling 为准。

&emsp;&emsp;要提醒的是，PyTorch Profiler 本身给不了 per-SM 的 achieved occupancy 或 stall 细分，它的价值在于“找到该深挖哪个 kernel”。训练性能问题也不一定在 kernel Occupancy 上——TorchMetrics 收集引入的 CPU-GPU 同步、DataLoader 的搬运瓶颈，都可能在 nsys 时间轴上先暴露出来。

### 4.2 block size 与 `__launch_bounds__`

&emsp;&emsp;block size 用 32 的倍数，否则会浪费 warp 槽位。`__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` 可以给编译器一个“至少 N 个 block 必须能驻留”的约束，编译器据此控制寄存器分配，在 Occupancy 和寄存器溢出之间找平衡。注意这是编译器提示，不是硬保证，最终有没有达到目标 Occupancy，还是要用 `achieved_occupancy` 计数器确认。

### 4.3 寄存器压力：有时降低 Occupancy 更快

&emsp;&emsp;当每线程寄存器用量成为限流因子时，有两个方向：

- **减少寄存器用量**，让更多 warp 驻留，提高 Occupancy；
- **有意降低 Occupancy 换 ILP**：如果 kernel 本身指令级并行度高，低 Occupancy 下每线程拥有更多寄存器，能减少 spill 到 local memory，反而更快。

&emsp;&emsp;社区经验（比如 Volkov 在 GTC 2010 的演讲）反复验证过：从 60% 推到 100% Occupancy，性能收益可能很有限；而降低 Occupancy 消除 spill，有时收益更大。`achieved_occupancy` 和 `branch_efficiency` 可以联读，判断低 Occupancy 是在放大分支停顿，还是已经被 ILP 掩盖了。

### 4.4 共享内存：限流因子和 bank conflict 是两件事

&emsp;&emsp;如果 `launch__occupancy_limit_shared_mem` 显示共享内存限制了并发 block 数，减少 per-block 共享内存能直接抬高 Theoretical Occupancy。但别把这两件事混为一谈：如果瓶颈是 bank conflict（MIO Throttle 高），减少用量解决不了冲突，得调访问 pattern 或用 padding。

## 5 实验

### 5.1 目的与假设

&emsp;&emsp;这一节原本计划直接用 ncu 的硬件计数器验证 Occupancy 的判读框架。本机的 `ncu` 2024.1.1.0 可以启动，但读取计数器时返回 `ERR_NVGPUCTRPERM`，`/proc/driver/nvidia/params` 中 `RmProfilingAdminOnly` 为 1，当前用户也没有 passwordless sudo。因此这里做的是一套**可运行的代理实验**：能测 Theoretical Occupancy、限流因子、CUDA event 时长、带宽、GFLOP/s 和 CPU 正确性；Achieved Occupancy 用 block 级 `%globaltimer`/`%smid` 驻留采样估算；Eligible Warps、Stall Reasons、Est. Speedup 和端到端影响全部标 `N/A`，不拿代理指标冒充硬件计数器。

&emsp;&emsp;实验仍围绕三个假设展开，但把“可验证”和“只能留待有 ncu 权限的机器”分开：

- **H1**：Theoretical 与 Achieved 的差距能稳定指示尾效应或 grid 规模不足。代理实验可以用 device 与 active-SM 两个视角检查这一点，但不能给出 ncu 的逐周期计数。
- **H2**：Eligible Warps 与 Stall Reasons 的联读能区分“并行度不足”和“依赖链停顿”。本机没有这两个计数器；依赖链部分用一个共享内存指针追逐微内核做方向性代理，只验证“同 occupancy 下增加独立链是否有收益”。
- **H3**：按限流因子和实际驻留做针对性调优，比只盯 Occupancy 数值更有效。代理实验可以比较资源限流、grid 覆盖和访存饱和三条路径，但没有真实应用负载，不能报告端到端收益。

### 5.2 实验环境

&emsp;&emsp;测试平台是一台单卡 RTX 3050。GPU 动态调频不可锁定，正式测量前运行 15 s FMA burn-in；实验开始与结束时 `nvidia-smi` 分别记录到 1935 MHz / 1942 MHz，温度 60 °C / 56 °C，功率 84.35 W / 75.77 W。host compiler 为 `g++-13.4.0`，CMake 固定 `CMAKE_CUDA_ARCHITECTURES=86`。

**表 3 实验环境配置**

| 项目 | 配置 |
|---|---|
| GPU 型号 | NVIDIA GeForce RTX 3050（GA106，`sm_86`） |
| SM 数量 | 20 |
| 显存容量/带宽 | 8 GB GDDR6，128-bit，7001 MHz；理论 224.0 GB/s；K1 实测 173.6 GB/s |
| CUDA 驱动/Toolkit | 595.99.02 / CUDA 12.4，`nvcc` 12.4.131 |
| Nsight Compute 版本 | 2024.1.1.0（可启动，但无计数器权限） |
| 框架版本 | 无（纯 CUDA runtime，不涉及 PyTorch） |
| 操作系统/容器 | Ubuntu 26.04.1 LTS，Linux 7.0.0-42-generic；非容器 |
| profiling 权限配置 | `RmProfilingAdminOnly: 1`；无 sudo；`ncu` 返回 `ERR_NVGPUCTRPERM` |

### 5.3 测试用例与基线

&emsp;&emsp;四个用例分别对应第 3 章的四类读数组合。A 拆成寄存器和共享内存两个受控子实验（A-smem 另测 24 KB 中间点，用来观察限流因子从 registers 切到 shared memory 的过渡）；B 的主对照放在 16.7% occupancy，另保留 Theoretical 100% 作为“高占用率是否已经掩盖依赖”的对照；C 用固定总工作量的计算内核扫 grid；D 用逐元素流式内核扫 occupancy。所有计时变体共用同一份经 CPU 参考校验的代码路径（B 校验 chain 数、C 校验 grid=4/480 的代表性配置，其余配置复用同一内核），且计时内核与采样内核的 `blocks_per_sm` 必须一致。

**表 4 测试用例与基线配置**

| 用例 | 类型 | 基线配置 | 优化配置 | 对应读数组合 |
|---|---|---|---|---|
| A-reg | K3 FP32 GEMM | 8×8 tile，128 regs，2 block/SM，33.3% | 4×4 tile，64 regs，4 block/SM，66.7% | 资源限流（registers） |
| A-smem | K3 4×4 + 动态 smem | 48 KB dyn，1 block/SM，16.7% | 同内核 0 KB，4 block/SM，66.7% | 资源限流（shared memory） |
| B | 共享内存指针追逐 | 1 条依赖链，16.7% occupancy | 4 条独立链，16.7% occupancy | 同 occupancy 的依赖链/MLP |
| C | 固定工作量 FMA | grid=4，device achieved 3.33% | grid=480，device achieved 87.24% | Theoretical 高、device Achieved 低 |
| D | K1 逐元素流式 | 80 KB dyn，16.7% occupancy | 0 KB dyn，100% occupancy | 访存管线饱和 |

### 5.4 评价指标

&emsp;&emsp;实测项包括：`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 给出的 Theoretical Occupancy；寄存器、共享内存、线程数、block 数四类静态限流估算；CUDA event 的 median/p99 kernel 时间；K1 的带宽；K3 与 C 的 GFLOP/s；CPU 参考校验结果。

&emsp;&emsp;软件 Achieved Occupancy 的定义是：

```text
achieved_device = 100 × Σ(active_warps × Δt) / (SM 数 × 最大 warp 数 × 时间窗)
```

&emsp;&emsp;每个 block 的 thread 0 在开始/结束读取 `%globaltimer`，在结束时读取 `%smid`；分析时按 SM 积分驻留 warp-time。另报 `achieved_active_sm`（只对有活动的 SM 求平均，接近 ncu 的 per-SM 视角）、各 SM 的 p10/p50/p90 和 busy 时间占比。`%smid` 不保证连续，因此先排序去重再映射到 0..N-1，不用 raw smid 直接索引。采样内核与计时内核分开编译；采样记录会增加寄存器（例如 K1 从 12 增到 18），所以每个变体都检查二者 `blocks_per_sm` 相同，并分别记录 `local_bytes`。

&emsp;&emsp;`N/A` 项：Eligible Warps、Long Scoreboard / LG Throttle / MIO Throttle / Barrier / Not Selected 的硬件占比、ncu 的 Est. Speedup、端到端耗时或吞吐。没有真实应用负载，端到端表不填数值。

### 5.5 实验流程

1. 运行 15 s burn-in，记录实验起止的 SM/显存时钟、温度、功率和 ncu 权限状态。
2. 用 `cudaFuncGetAttributes` 与 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 读取资源、Theoretical Occupancy 和限流因子。
3. 对每个变体运行一次独立的采样内核，按 block 驻留时间估算 Achieved、per-SM 分布和 grid fill。
4. 用 CUDA event 计时：每变体预热 10 次、测量 50 次、重复 3 轮，变体交错执行；报告 median/p99。
5. 对代表性配置做 CPU 正确性校验；CSV 中任一 `correct=false` 即停止，不把错误数据写入文档。B/C 的其余配置复用同一内核，共享校验结论。
6. 按第 3 章的读数组合选择动作；在有 ncu 权限的机器上重跑附录 A 的命令，把本节的 `N/A` 列补成硬件计数器。

&emsp;&emsp;本机实际执行的是步骤 1–5；步骤 6 没有执行，因为 `ncu` 在 `bench_limits` 上直接返回 `ERR_NVGPUCTRPERM`，且 `--launch-count 1` 也没有采到任何 kernel。复现脚本是 `scripts/run_all.sh`，结果 CSV 在 `results/`。

### 5.6 实验结果

&emsp;&emsp;表 5 给出四个用例的 Occupancy 与 kernel 时间。A-reg 的 4×4 tile 用 64 regs 换到 4 block/SM，时间从 8.931 ms 降到 7.434 ms（1.20×）；A-smem 把 48 KB 动态 smem 降到 0，时间从 26.462 ms 降到 7.434 ms（3.56×）；中间的 24 KB 点是 50.0% theoretical / 45.14% achieved、10.106 ms，差距比两端大，保留为回填/尾效应的中间观察点。B 在 16.7% occupancy 下，4 条独立链把 0.209 ms 降到 0.140 ms（1.49×）；同样的 1 链/4 链在 Theoretical 100% 对照下（实测 device 76.99% / 62.60%）几乎相同（0.822 vs 0.820 ms），说明高占用率已经把这段依赖隐藏掉。C 从 grid=4 到 grid=480，device Achieved 从 3.33% 升到 87.24%，时间从 5.877 ms 降到 0.529 ms（11.12×）；D 的带宽从 100% 的 173.6 GB/s 只降到 66.7% 的 169.7 GB/s，33.3% 时 150.0 GB/s，16.7% 时 98.5 GB/s。

**表 5 Occupancy 与 kernel 性能对比**

| 用例 | 基线 Theoretical | 基线 Achieved(device/active-SM) | 优化后 Theoretical | 优化后 Achieved(device/active-SM) | 基线时间(ms) | 优化后时间(ms) | 加速比 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A-reg | 33.3% | 31.02% / 31.02% | 66.7% | 65.72% / 65.72% | 8.931 | 7.434 | 1.20× | 寄存器 128→64；0 spill |
| A-smem | 16.7% | 16.43% / 16.43% | 66.7% | 65.72% / 65.72% | 26.462 | 7.434 | 3.56× | 48 KB 动态 smem→0；限流因子 shared→registers |
| B | 16.7% | 16.67% / 16.67% | 16.7% | 16.67% / 16.67% | 0.209 | 0.140 | 1.49× | 16.7% occupancy 下 1→4 链；Theoretical 100% 对照（device 76.99/62.60）无差异 |
| C | 100.0% | 3.33% / 16.66% | 100.0% | 87.24% / 87.24% | 5.877 | 0.529 | 11.12× | grid 4→480；grid=20 首次铺满 SM，480 为回填点 |
| D | 16.7% | 16.54% / 16.54% | 100.0% | 99.11% / 99.11% | 5.449 | 3.093 | 1.76× | 80 KB→0；带宽 98.5→173.6 GB/s |

&emsp;&emsp;表 6 保留 ncu 的 Stall Reasons 列，但本机无法取得任何一项，全部标 `N/A`。表 6-1 把可测的代理信号单独列出，避免把“代理观察”误读成硬件归因。

**表 6 主要 Stall Reasons 分布（本机不可测）**

| 用例 | 版本 | Long Scoreboard | LG Throttle | MIO Throttle | Barrier | Not Selected | 其他 |
|---|---|---|---|---|---|---|---|
| A | 基线/优化 | N/A | N/A | N/A | N/A | N/A | N/A |
| B | 基线/优化 | N/A | N/A | N/A | N/A | N/A | N/A |
| C | 基线/优化 | N/A | N/A | N/A | N/A | N/A | N/A |
| D | 基线/优化 | N/A | N/A | N/A | N/A | N/A | N/A |

&emsp;&emsp;表 6 的空缺不是“没有发生停顿”，而是 `ERR_NVGPUCTRPERM` 导致这些计数器不可读。任何基于 Long Scoreboard 或 LG/MIO Throttle 的百分比结论都不应出自本机数据。

**表 6-1 停顿诊断的代理信号**

| 用例 | 版本 | 代理信号 | 支持的判读 | 不能支持的判读 |
|---|---|---|---|---|
| A-reg 基线 | 实测 | K3 8×8，128 regs，2 block/SM，限流 registers；achieved 31.02% | 降低寄存器到 64 后 4 block/SM、achieved 65.72%，时间 8.931→7.434 ms（1.20×） | 无法给出 Long Scoreboard 等 stall 占比 |
| A-reg 优化 | 实测 | 64 regs，4 block/SM；0 spill；achieved 65.72% | 资源限流诊断与实测方向一致：同算法、不同 tile/寄存器配置换来 2 block/SM | 不能区分收益中有多少来自 ILP 变化 |
| A-smem 基线 | 实测 | 48 KB 动态 smem，1 block/SM，限流 shared memory；achieved 16.43% | 限流因子从 registers 切到 shared memory，achieved 随块数下降 | 无法证明真实 kernel 中这块 smem 是“无用”的；这是受控注入 |
| A-smem 优化 | 实测 | 0 KB 动态 smem，4 block/SM、achieved 65.72%；26.462→7.434 ms（3.56×） | 减少无用 smem 可以同时抬高 Theoretical/Achieved 并缩短时间 | 这是资源注入实验，不是真实 kernel 的调优收益 |
| B | 实测 | 16.7% occupancy：1 链 0.209 ms / 402 Gload/s；4 链 0.140 ms / 598 Gload/s（1.49×）；Theoretical 100% 对照（device 76.99% / 62.60%）1 链 0.822 vs 4 链 0.820 ms | 低 occupancy 下 4 条独立链提高 MLP，支持依赖链方向；高 occupancy 下差异消失 | 没有 Eligible Warps / Long Scoreboard，不能确认硬件 stall 归属 |
| C | 实测 | grid 4→480：device achieved 3.33%→87.24%，时间 5.877→0.529 ms（11.12×）；grid=20 已铺满 SM（1.060 ms），grid=120 100% fill 但 device 仅 72.33% | Theoretical 固定 100% 时，device Achieved 低可直接指示 grid 覆盖/回填不足 | grid=480 仍未到 100%；无法给出尾效应硬件计数 |
| D | 实测 | occupancy 100/66.7/33.3/16.7%；带宽 173.6/169.7/150.0/98.5 GB/s | 带宽在 2/3 occupancy 已接近平台，16.7% 才明显下降；说明 occupancy 不是越高越好 | 没有 LG/MIO Throttle 计数，不能把平台归因到具体管线 |

**表 7 端到端影响**

| 场景 | 基线端到端耗时/吞吐 | 优化后端到端耗时/吞吐 | 端到端加速比 | 说明 |
|---|---|---|---|---|
| 本实验（单 kernel 合成负载） | N/A | N/A | N/A | 无真实应用负载；只报 kernel 级数据 |

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261112_8dbc87.svg)


![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261112_1be0a1.svg)


![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261112_c33510.svg)


![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261112_892ac8.svg)

### 5.7 结果分析

&emsp;&emsp;**H1 的判读要加条件。** C 把 Theoretical 固定在 100%，grid 从 1 增到 480 时 device Achieved 从 0.83% 升到 87.24%，时间从 23.383 ms 降到 0.529 ms，Theoretical–Achieved 的差距确实指示了 grid 覆盖不足。但 A 的两个资源限流变体差距很小：A-reg 的 33.3% 对 31.02%，66.7% 对 65.72%；A-smem 的 48 KB 16.7% 对 16.43%，24 KB 中间点 50.0% 对 45.14%（差距 4.86 pp，明显大于两端）。所以“差距大 = grid 不足”只在 grid 是约束时成立，不能把任何 Achieved 低于 Theoretical 都归因到尾效应。C 还有一个细节：grid=120 正好是 6 block/SM 的容量，`grid_fill=100%`，但 device Achieved 只有 72.33%；grid=240/480 通过新 block 回填才升到 83.05%/87.24%。如果只按“grid fill 到 100%”判断，会漏掉回填不足。

&emsp;&emsp;**H2 只得到方向性支持。** 16.7% occupancy 下，1 条共享内存指针追逐链是 0.209 ms、402 Gload/s，4 条独立链是 0.140 ms、598 Gload/s，快 1.49×；同样对比在 Theoretical 100% 对照下（实测 device 76.99% / 62.60%）是 0.822 对 0.820 ms，差异消失。这个结果符合“warp 足够多时依赖链被切换隐藏，warp 不足时增加独立链（MLP）有效”的解释。但本机没有 Eligible Warps，也没有 Long Scoreboard 占比，不能证明慢版本一定停在 Long Scoreboard；它只能说明依赖链/MLP 是一个可操作的调优方向。

&emsp;&emsp;**H3 的方向成立，但不能外推成端到端收益。** A-reg 的 128→64 regs 换来 2→4 block/SM，时间快 1.20×；A-smem 的 48 KB→0 动态 smem 换来 1→4 block/SM，时间快 3.56×。D 则说明 occupancy 不是越高越好：100%→66.7% 带宽只降约 2%，16.7% 才降 43%。C 的“只盯 occupancy”对照也有意思：grid=120 已经 100% fill，但 grid=480 的 device Achieved 更高、时间再快 6.8%（0.565→0.529 ms）；真正的大头是把 grid 从 4 铺到 20（5.877→1.060 ms，5.54×）。也就是说，先判断“grid 是否铺开、限流因子是什么”，比盲目把某个百分比推高更有效；但本实验没有应用层负载，不能给出端到端加速比。

&emsp;&emsp;两个需要明确的偏差。第一，A-reg 的高寄存器版本实际是 128 regs，而不是计划中的 112：`__launch_bounds__(256,2)` 覆盖了 `--maxrregcount=112`，ptxas 仍分配 128 regs、0 spill、2 block/SM；文档按实测 128 记录。第二，软件采样是 block 级估算，不是 ncu 的逐周期 active warp 计数；采样内核可能比计时内核多用寄存器（K1 12→18，K3 不变），所以每个变体都检查了 `blocks_per_sm` 一致，但不能保证采样窗口与计时窗口完全同构。

&emsp;&emsp;关键代码只有两段：采样器和依赖链内核。采样器用 `%globaltimer`/`%smid` 记录 block 起止，分析时先把 raw smid 去重映射，再按时间积分；依赖链内核在 16.7% occupancy 下用 1 条或 4 条 `idx[p]` 链，保持总 load 数相同。完整代码在 `/media/nvm1/workspace/tests/cuda-occ`。

```cuda
// include/occ_sampler.cuh（节选）
__device__ __forceinline__ unsigned long long read_globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned int read_smid() {
  unsigned int s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}
// 分析：先对 smid 排序去重，再对每个 SM 积分 active_warps * dt
```

```cuda
// src/bench_b_dep.cu（节选，CHAINS = 1 或 4）
int p[CHAINS];
for (int j = 0; j < CHAINS; ++j)
  p[j] = (threadIdx.x + j * (blockDim.x / CHAINS)) % blockDim.x;
for (int it = 0; it < iters; ++it) {
  for (int j = 0; j < CHAINS; ++j) {
    p[j] = idx[p[j]];          // 依赖 load；4 条链提供 MLP
    acc += static_cast<float>(p[j]);
  }
}
```

&emsp;&emsp;**局限与复现。** 本机 ncu 不可用；软件采样是 block 级；没有真实应用负载；只测了一张 RTX 3050；时钟不可锁定，依赖 burn-in 与交错测量降低漂移。复现命令：

```bash
cd /media/nvm1/workspace/tests/cuda-occ
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
./scripts/run_all.sh
```

&emsp;&emsp;`run_all.sh` 会重新跑 15 s burn-in、四个用例、`validate_results.py` 和图 13–16。CSV 在 `results/`；`results/ncu_attempt.txt` 保留了本机的 `ERR_NVGPUCTRPERM` 记录。若在允许 profiling 的机器上复现，按附录 A 的 `ncu --section Occupancy --section SchedulerStats --section WarpStateStats` 采集后，把表 6 的 `N/A` 替换成硬件读数，并重新核对表 6-1 的代理判读是否仍成立。

## 6 实践案例

&emsp;&emsp;以下三个案例来自日常 profiling 记录，用来说明计数器读数和优化动作之间的对应关系，不作为第 5 章受控实验的结果。

### 6.1 尾效应：Achieved 明显低于 Theoretical

&emsp;&emsp;某 kernel 的 Theoretical Occupancy 50%，Achieved 只有 41.52%，Nsight 显示 SM 之间负载明显不均，是典型的尾效应。优化后 Theoretical 到 62.50%，Achieved 到 61.31%，运行时间从 4.535 ms 降到 3.825 ms，约 1.19×。这个案例有意思的点在于，收益不是“把 Occupancy 推高”带来的，而是 Achieved 与 Theoretical 的差距先暴露了尾效应——grid 的 block 数不是 SM 数的整数倍，最后几 wave 的 SM 闲置拉低了加权平均。

### 6.2 SDPA Attention 的 Occupancy 提升

&emsp;&emsp;一次硬件归因的 PyTorch 算子 profiling 中，SDPA Attention 的 occupancy 计数器从 16.6% 提到 73.9%，对应 kernel 级加速约 2.24×；同一项工作里 GPT-2 的 `tensor_core_active_pct` 从 0% 提到 40.5%，kernel time 加速约 1.76×。收益的来源不是盲目调 block size，而是 profiling 把问题定位到了具体算子的 tile 配置或 GEMM 的 M 维过小，然后做针对性改动。

### 6.3 低 Eligible Warps 指向依赖链，而不是“座位不够”

&emsp;&emsp;某 Triton GEMM 的读数：Theoretical Occupancy 83.33%、Achieved 32.28%、Eligible Warps 0.36。这组数字指向 reduction 依赖链和 grid 并行度不足。修复方向不是把 Achieved 推到 83%，而是增加独立累加器、提高 `num_stages` 增加 memory-level parallelism、扩大 grid。这就是前面说的：计数器给的是诊断坐标，动作要按 stall 根因选，不能按 Occupancy 数值选。

**表 8 案例汇总**

| 案例 | 关键计数器读数 | 优化动作 | 观察收益 |
|---|---|---|---|
| 尾效应 kernel | Theoretical 50%，Achieved 41.52% | 改善负载均衡与 grid 配置 | 约 1.19× |
| SDPA Attention | occupancy 16.6% → 73.9% | 针对 tile / 编译配置调整 | kernel 级约 2.24× |
| Triton GEMM | Achieved 32.28%，Eligible 0.36 | 打破归约依赖链、提高 MLP | 诊断导向 |

## 7 一些坑和局限

&emsp;&emsp;最大的坑前面已经反复说了：把 `achieved_occupancy` 当成“越高越好”的 KPI。正确用法是把它和 Theoretical Occupancy、限流因子、Eligible Warps、Stall Reasons 摆在一起读——Theoretical 说明硬件允许什么，Achieved 说明实际发生了什么，限流因子说明哪里被卡住，stall reason 说明卡住之后在等什么。四个信息对齐了，才能判断“加 Occupancy”是不是下一步该做的事。很多真实 kernel 里答案恰恰相反：降低 Occupancy、换更多寄存器或更少的资源竞争，才是计数器数据指的方向。

&emsp;&emsp;工具层面的坑：ncu 采计数器需要权限，容器环境通常要 `--cap-add SYS_ADMIN` 或调整驱动的 profiling 权限；WSL2 下可能碰到 ncu 路径或库加载问题。建议先用一个最小可复现 kernel 验证工具链能用，再对真实 workload 做 profiling。另外，第 6 章案例里的数字来自特定软硬件配置，收益不一定能平移；ncu 的 Est. Speedup 是局部建议，端到端优化还得结合全局时间轴判断。

## 8 小结

&emsp;&emsp;Occupancy 优化的起点不是把某个百分比推上去，而是用性能计数器把“实际驻留多少 warp、为什么停住、谁限制了驻留量”这三件事搞清楚。这篇博客整理的就是这套读法：Theoretical 和 Achieved 一起看，限流因子定位约束，Eligible Warps 和 Stall Reasons 判断停顿性质，再用 Roofline 卡一下天花板。只有当读数指向“并行度不足”时，加 Occupancy 才是合理动作；如果瓶颈在依赖链、访存管线或 grid 规模，该改的就是 ILP/MLP、访存模式或问题规模和 launch 配置。

## 9 参考文献

- [NVIDIA. Nsight Compute User Guide[EB/OL]. NVIDIA Developer Documentation](https://docs.nvidia.com/nsight-compute/).
- [NVIDIA. CUDA C++ Programming Guide[EB/OL]. NVIDIA Developer Documentation](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)
- [NVIDIA. CUDA C++ Best Practices Guide[EB/OL]. NVIDIA Developer Documentation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Roofline: An Insightful Visual Performance Model for Multicore Architectures[J]. Communications of the ACM, 2009, 52(4): 65-76](https://www.osti.gov/biblio/1407073)
- [Better Performance at Lower Occupancy[C]//GPU Technology Conference](https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf)
- [NVIDIA. Nsight Systems User Guide[EB/OL]. NVIDIA Developer Documentation](https://docs.nvidia.com/nsight-compute/)

## 附录 A 常用命令

```bash
# 只看 Occupancy 区段
ncu --section Occupancy ./your_kernel

# SpeedOfLight + SchedulerStats + WarpStateStats
ncu --section SpeedOfLight --section SchedulerStats --section WarpStateStats ./your_kernel

# 直接读关键计数器
ncu --metrics achieved_occupancy,sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem ./your_kernel
```