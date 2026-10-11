# 深入理解 GPU 分支发散：从 SIMT 模型到硬件实现

## 1. 引言：SIMT 与 Warp

&emsp;&emsp;现代 GPU 采用 **SIMT（Single Instruction, Multiple Threads）** 执行模型。在 NVIDIA 的术语中，一个 **warp** 包含 32 个线程（lane）；在 AMD 中称为 **wavefront**，通常为 64 个线程。为简化讨论，本文以 32-lane warp 为例。SIMT 的核心思想是：
- 一个 warp 内的 32 个 lane 共享 **取指、译码、发射** 单元；
- 每个 lane 拥有自己独立的 **寄存器、条件码、数据**；
- 同一时刻，warp 调度器发射 **一条指令**，所有 **活跃 lane** 一起执行。

&emsp;&emsp;如果所有 lane 的“下一条指令地址”（PC）一致，那么一条指令就能服务 32 个线程，效率极高。但现实世界中的控制流往往因线程而异——这就是 **分支发散（Branch Divergence）** 的根源。

## 2. 分支发散

&emsp;&emsp;考虑下面这段 CUDA 代码：

```cuda
if (threadIdx.x < 16) {
    A();
} else {
    B();
}
```

&emsp;&emsp;假设一个 warp 内 `threadIdx.x` 从 0 到 31。那么：
- 线程 0–15 满足条件，希望执行 `A()`；
- 线程 16–31 不满足条件，希望执行 `B()`。

此时，warp 内出现了 **两个不同的 PC 需求**：一部分 lane 想跳到 `A` 的入口，另一部分想跳到 `B` 的入口。但硬件每个周期只能发射一条指令，对应一个 PC。于是，硬件无法在同一周期内同时执行 `A` 和 `B`。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261011_accb95.svg)

&emsp;&emsp;当 warp 执行到 `if (threadIdx.x < 16) { A(); } else { B(); }` 时，硬件按时刻管理 lane 的活跃状态：T0 全部 32 个 lane 活跃，PC 指向分支指令；T1 先走 `if` 路径，只有 lane 0–15 保持活跃执行 `A`，lane 16–31 被 mask；T2 再走 `else` 路径，lane 16–31 活跃执行 `B`，lane 0–15 被 mask；T3 到达汇合点后恢复全部 lane 活跃，继续执行后续指令。被 mask 的 lane 不执行指令、不写寄存器、不访问内存、不产生副作用。因此，从 warp 整体看确实先后经过了 `if` 和 `else` 两段代码，但从每个线程看只执行了自己那条路径，所谓“摒弃错误分支结果”更准确的说法是：非活跃 lane 的写回和副作用被屏蔽。

---

**发散的真实原因：多 PC 需求 vs 单 PC 发射**

&emsp;&emsp;很多人以为“`if` 导致发散”，但更本质的原因是：

> **同一 warp 内线程需要不同的下一条指令地址，而 SIMT 硬件以 warp 为单位取指和发射。**

&emsp;&emsp;也就是说，**不是 `if` 导致发散，而是 warp 内线程的 PC 不一致导致发散**。`if` 只是让 PC 不一致的一种常见写法。因此，任何导致 warp 内控制流不一致的因素都会引发发散，包括但不限于：

| 触发源 | 说明 |
|---|---|
| `if / else / switch` | 条件按 lane 不同时发散；若整个 warp 条件一致，则不发散 |
| 循环 | 每个 lane 循环次数不同，最后一次迭代只有部分 lane 活跃 |
| `break / continue` | 部分 lane 跳出，部分继续，PC 不一致 |
| `return` / 线程退出 | 部分线程提前退出，warp 内剩余线程继续执行 |
| 函数指针 / 虚函数 | 不同 lane 跳到不同函数，产生调用发散 |
| 异常 / trap | 部分线程触发异常，其他线程正常继续 |
| 短路逻辑 | `a && b` 中若 `a` 按 lane 不同，可能引入分支 |

&emsp;&emsp;所以更准确的说法是 **warp 内控制流发散**，而不是简单的“分支发散”。

&emsp;&emsp;需要注意谓词不是分支发散。有时编译器会采用 **谓词执行（Predication）** 来避免真正的分支发散。例如：

```cuda
x = cond ? a : b;
```

&emsp;&emsp;编译器可能生成：

```
计算 a
计算 b
根据 cond 选择结果
```

&emsp;&emsp;所有 lane 都执行 `a` 和 `b` 的计算，但没有控制流分叉。这叫做 **predication / select**。它没有“分支发散”，但可能浪费指令。你之前问的“所有线程都执行 if 和 else 再选结果”，更接近这种编译器优化，而不是一般分支发散的语义。

---

**性能影响和建议**
&emsp;&emsp;分支发散会导致 **串行化**：原本可以并行执行的路径，现在被拆成多个时刻顺序执行。如果两条路径的代码量都很大，warp 的执行时间会接近两者之和，而不是取最大值。优化建议：
1. **尽量让同一 warp 内的线程走相同路径**：例如，将条件判断放在 warp 级别，而不是线程级别。
2. **避免在 warp 内使用 `threadIdx.x` 做分支条件**：如果必须分支，尝试重新组织数据，使同一 warp 的线程处理相似的数据。
3. **利用谓词执行**：对于短小的条件赋值，编译器通常会自动使用 predication，无需手动干预。
4. **注意循环边界**：让同一 warp 内的线程循环次数尽量一致，避免最后一次迭代只有部分 lane 活跃。
5. **使用 `__syncwarp()` 或 `__shfl_sync()` 等原语**：在需要 warp 内同步或通信时，确保活跃掩码正确。
6. **考虑 Volta 及之后的 ITS**：虽然 ITS 提供了更大的灵活性，但不要依赖它来消除发散成本。

## 3. 硬件如何管理发散：Divergence Stack 与掩码

&emsp;&emsp;传统 NVIDIA 架构使用 **divergence stack**（或称 reconvergence stack）来管理发散：

1. 遇到分支时，硬件将当前活跃掩码和 **汇合点 PC** 压入栈；
2. 选择一条路径执行，其他 lane 被 mask；
3. 执行到路径末尾时，从栈中弹出信息，恢复之前被 mask 的 lane；
4. 继续执行另一条路径，直到到达汇合点；
5. 最终在汇合点恢复全 warp 活跃。

&emsp;&emsp;AMD GPU 则使用 **exec mask** 来控制哪些 lane 活跃，分支指令直接修改 exec mask。无论哪种方式，核心都是：**用掩码表示活跃 lane，用栈记录汇合点，将不同路径串行化执行。**

---

**Volta 之后的 Independent Thread Scheduling**

&emsp;&emsp;从 NVIDIA Volta 架构开始，GPU 引入了 **Independent Thread Scheduling（ITS）**：
- 每个线程拥有自己的 PC 和调用栈；
- warp 调度器可以更灵活地组合活跃线程；
- reconvergence 不再完全由编译器静态强制；
- 但取指/发射仍以 warp 为粒度，所以发散仍有性能代价。

&emsp;&emsp;ITS 缓解了死锁和过度串行化，但没有消除 SIMT 的发散成本。程序员仍然需要关注 warp 内的控制流一致性。

## 4 发散实验

&emsp;&emsp;为了确保 if/else 产生真实分支而不是被编译器谓词化，发散内核中的两条路径被写成 __noinline__ 设备函数。函数调用无法被谓词化，必须发出真实分支指令，从而保证 warp 内两路并存时产生发散。非发散内核把两条路径都算一遍，最后用 select 选择结果。它没有真实分支，所有 lane 始终活跃，但计算量更大。发散内核则根据数据条件走不同路径，warp 内两路都有人走时，两段代码串行执行。

```cpp
// 路径 A / 路径 B 做成 __noinline__ device 函数:
// nvcc 对短的、可展开的 if/else 会用 predication(谓词指令)替换分支,那样
// 没有真实 branch、不会有发散。函数调用无法谓词化,必须发真实 branch,
// 从而保证 warp 内两路并存时产生发散。
__device__ __noinline__ float div_path_a(float x) {
  float y = x;
  for (int k = 0; k < 8; ++k) y = y * 1.1f + 0.01f;  // 路径 A:8 次迭代
  return y;
}

__device__ __noinline__ float div_path_b(float x) {
  float y = x;
  for (int k = 0; k < 2; ++k) y = y * 2.5f + 0.05f;  // 路径 B:2 次迭代
  return y;
}

// ---------------------------------------------------------------------------
// Kernel 3:case_divergent,分支发散。
// if/else 按数据值 x>0.5 分叉,两条路径循环次数不同(8 vs 2),且走的是
// __noinline__ 函数 -> 真实 branch;warp 内两路都有人走时两段代码串行执行,
// 后走的那路线程停顿 -> 发散。
// ---------------------------------------------------------------------------
__global__ void case_divergent(const float* __restrict__ in,
                               float* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float x = in[i];
    float y = (x > 0.5f) ? div_path_a(x) : div_path_b(x);  // 真实分支
    out[i] = y;
  }
}

// ---------------------------------------------------------------------------
// Kernel 4:case_nondivergent,无发散(branchless)。
// 同样的计算,两条路径都算,最后用 select 选结果,没有 branch 指令 -> 无发散。
// ---------------------------------------------------------------------------
__global__ void case_nondivergent(const float* __restrict__ in,
                                  float* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float x = in[i];
    float ya = x;
    for (int k = 0; k < 8; ++k) ya = ya * 1.1f + 0.01f;
    float yb = x;
    for (int k = 0; k < 2; ++k) yb = yb * 2.5f + 0.05f;
    out[i] = (x > 0.5f) ? ya : yb;  // 编译为 FSEL/SEL,无 branch
  }
}
```

&emsp;&emsp;两份 Nsight Compute 报告中提取核心数据，整理如下：

```cpp
    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        36.23
    Warp Cycles Per Executed Instruction           cycle        40.29
    Avg. Active Threads Per Warp                                22.70
    Avg. Not Predicated Off Threads Per Warp                    21.58
    ---------------------------------------- ----------- ------------

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle        1,328
    Total DRAM Elapsed Cycles        cycle       63,488
    Average L1 Active Cycles         cycle     1,728.50
    Total L1 Elapsed Cycles          cycle       70,160
    Average L2 Active Cycles         cycle     1,114.33
    Total L2 Elapsed Cycles          cycle       40,992
    Average SM Active Cycles         cycle     1,728.50
    Total SM Elapsed Cycles          cycle       70,160
    Average SMSP Active Cycles       cycle     1,677.96
    Total SMSP Elapsed Cycles        cycle      280,640
    -------------------------- ----------- ------------

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.23
    Branch Instructions              inst        2,560
    Branch Efficiency                   %        83.33
    Avg. Divergent Branches                       3.20
    ------------------------- ----------- ------------
```

&emsp;&emsp;发散内核执行了 11,008 条指令，而非发散内核只有 6,656 条，增加了 65%。这正是 SIMT 分支发散的典型代价：同一个 warp 内，原本可以并行执行的 if 和 else 路径，被迫串行执行。每条路径都要单独发射指令，导致总指令数接近两条路径之和。分支指令从 512 增加到 2,560，Avg. Divergent Branches 从 0 变为 3.20，说明发散内核中平均每个 warp 约有 3.2 个发散分支。Branch Efficiency 为 83.33%，意味着约 16.7% 的分支是发散的。

```cpp
    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        47.47
    Warp Cycles Per Executed Instruction           cycle        56.26
    Avg. Active Threads Per Warp                                   32
    Avg. Not Predicated Off Threads Per Warp                    30.15
    ---------------------------------------- ----------- ------------
        Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle        1,324
    Total DRAM Elapsed Cycles        cycle       58,368
    Average L1 Active Cycles         cycle     1,594.25
    Total L1 Elapsed Cycles          cycle       64,890
    Average L2 Active Cycles         cycle       980.58
    Total L2 Elapsed Cycles          cycle       37,920
    Average SM Active Cycles         cycle     1,594.25
    Total SM Elapsed Cycles          cycle       64,890
    Average SMSP Active Cycles       cycle     1,478.22
    Total SMSP Elapsed Cycles        cycle      259,560
    -------------------------- ----------- ------------

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.08
    Branch Instructions              inst          512
    Branch Efficiency                   %            0
    Avg. Divergent Branches                          0
    ------------------------- ----------- ------------
```

&emsp;&emsp;非发散内核的 Avg. Active Threads Per Warp 是 32，即每个周期全部 lane 都活跃，SIMT 效率 100%。发散内核只有 22.70，下降了 29%。如果考虑谓词执行，Avg. Not Predicated Off Threads Per Warp 进一步降到 21.58，说明实际参与有效计算的线程更少。这直接验证了：分支发散时，被 mask 的 lane 不执行指令，warp 内并行度被浪费。硬件通过掩码和 divergence stack 将不同路径串行化，非活跃 lane 的写回和副作用被屏蔽。

## 5. 总结

&emsp;&emsp;理解分支发散，是编写高性能 GPU 程序的基础。希望这篇博客能帮你建立清晰的 SIMT 执行模型，并在实际开发中避开发散陷阱。
- **分支发散的本质**：同一 warp 内线程的 PC 不一致，而 SIMT 硬件以 warp 为单位取指和发射。
- **`if` 只是触发器**：任何导致控制流不一致的因素都会引发发散。
- **硬件处理方式**：通过掩码和 divergence stack，将不同路径串行化执行，非活跃 lane 被 mask。
- **不是所有线程都执行 if 和 else 再选结果**：一般分支发散是路径串行化；谓词执行才是“都执行再选”。
- **发散单位是 warp**：不同 warp 之间不受影响；条件一致就不发散。
- **性能优化核心**：尽量保持 warp 内控制流一致，减少路径串行化。



