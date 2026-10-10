# Swizzle：Conflict-Free

## 1 理解 Bank Conflict

### 1.1 Memory Bank

&emsp;&emsp;在 HPC 场景中，无论是编写 CPU 程序还是 GPU kernel，性能优化的核心目标之一，都是让计算单元尽可能少地等待数据。为此，我们会尽可能并行地访问内存，以最大化内存带宽、降低有效内存延迟。并行度越高，越能掩盖单次访问的长延迟；而要维持高并行度，就必须尽量减少数据依赖和数据冲突，避免访问请求在存储层次中排队甚至串行化。

&emsp;&emsp;无论是 CPU 内存、CPU 缓存，还是 GPU 显存与片上共享内存，为了支持快速访问，硬件通常不会把整个存储阵列视为一个单一的大队列，而是将其划分为多个可并行访问的存储单元或分区。这些单元就是 memory bank（存储体）。每个 bank 可以在同一时刻独立服务一个访问请求，多个 bank 并行工作，就能提供远高于单个 bank 的聚合带宽。不同硬件的 bank 划分粒度、数量、地址映射与冲突规则各不相同：CPU DRAM 中有 channel、rank、bank，cache 中有 slice、bank，GPU 显存中有 partition、channel，CUDA shared memory 则有经典的 32 个 bank。但它们的核心思想一致：通过地址交错，把访问分散到多个并行单元上，让连续或规则访问尽可能同时得到服务。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261008_c4e7e3.svg)

&emsp;&emsp;反之，如果多个访问在时间上集中落到同一个 bank，尤其是访问不同地址时，硬件只能将它们串行化，从而形成 bank conflict。并行度随之下降，带宽利用率降低，延迟被放大。因此，理解 memory bank 的划分方式与地址映射，是理解内存层次性能，进而理解 CUDA shared memory bank conflict 与 swizzle 技术的基础。在 CUDA 中，最常被讨论的正是 shared memory 的 32 个 bank，它也是 Bank Conflict 与 Swizzle 技术真正登场的舞台。

### 1.2 Bank Conflict

&emsp;&emsp;在 CUDA 中，shared memory 的 bank 划分是理解冲突的起点。在大多数架构中，shared memory 被划分为 32 个 bank，每个 bank 每个时钟周期可服务一个 32-bit 字。一个 warp 包含 32 个线程，因此理想情况下，一个 warp 的 32 个 shared memory 访问可以恰好分布到 32 个 bank 上，并在一次事务中完成。地址到 bank 的映射可以抽象为：

$$
\text{bank} = \left(\frac{\text{address}}{4}\right) \bmod 32
$$

&emsp;&emsp;其中 `address` 为字节地址，除以 4 后得到 32-bit word 索引。也就是说，连续的 4 字节字依次落入 bank 0、1、2、…、31，然后循环。

&emsp;&emsp;当 warp 执行一条 shared memory 指令时，硬件会将 32 个线程的地址按 bank 分组。若每个 bank 至多收到一个不同地址的请求，这些请求便可并行服务；若多个线程访问同一地址，硬件可以广播该值，仍然只需一次服务；若某个 bank 同时收到多个不同地址的请求，则发生 **bank conflict**。冲突请求只能被串行化，通常需要多个 transaction 才能完成。若最坏情况下 32 个线程都访问同一 bank 的不同地址，就会形成 32-way bank conflict，吞吐率降至 1/32。

&emsp;&emsp;用几个例子说明：

```cuda
__shared__ float tile[32][32];

// 无冲突：bank = tid % 32
float a = tile[0][threadIdx.x];

// 32-way conflict：所有线程访问 bank 0 的不同地址
float b = tile[threadIdx.x][0];

// 无冲突：bank = (tid * 33) % 32 = tid
float c = tile[threadIdx.x][threadIdx.x];
```

&emsp;&emsp;对于 `tile[threadIdx.x][0]`，其 word 索引为 `tid * 32`，因此 bank 为 `(tid * 32) % 32 = 0`。一个 warp 内的 32 个线程都落在 bank 0，但访问的是不同地址，所以硬件只能串行服务，形成 32-way bank conflict。

&emsp;&emsp;经典的矩阵转置就是 bank conflict 的典型场景：

```cuda
__shared__ float tile[32][32];

tile[ty][tx] = in[ty * 32 + tx];

// 读 tile[tx][ty] 时，bank = (tx * 32 + ty) % 32 = ty
// warp 内 tx 变化，所有线程落在同一个 bank，但地址不同
out[tx * 32 + ty] = tile[tx][ty];
```

![](https://leimao.github.io/images/blog/2022-06-22-CUDA-Shared-Memory-Bank/examples-of-strided-shared-memory-accesses.png)

&emsp;&emsp;如果为 shared memory 增加 padding：

```cuda
__shared__ float tile[32][33];
```

&emsp;&emsp;此时 `tile[tx][ty]` 的 word 索引为 `tx * 33 + ty`，bank 为：

$$
(tx \times 33 + ty) \bmod 32 = (tx + ty) \bmod 32
$$

&emsp;&emsp;随着 `tx` 变化，线程会落到不同的 bank 上，从而消除冲突。这就是 padding 方法的基本原理。需要特别注意以下几点：

- **广播不算冲突**：多个线程访问同一bank的同一地址时，硬件可以广播，仍然只需一次服务。
- **冲突按 warp 内单条指令判断**：不同 warp 之间不会直接构成 bank conflict，但它们会竞争 shared memory 的 bank 带宽。
- **访问宽度会影响冲突形式**：访问 64-bit 或 128-bit 类型时，一条指令会被拆成多个 32-bit 请求，bank 冲突按拆分后的请求判断。
- **冲突程度用 n-way 表示**：n 是同一 bank 上不同地址请求的最大数量，也大致对应需要串行化的事务数。

&emsp;&emsp;消除 bank conflict 的常见思路包括：

1. **改变访问模式**：让相邻线程访问相邻 bank，例如尽量使 `bank = tid`。
2. **Padding**：破坏规则步长，例如把 `[32][32]` 改成 `[32][33]`。
3. **Swizzle**：通过异或等映射重排逻辑索引，例如 `col ^= row` 或 `index ^= (index >> 5)`，在不增加空间开销的情况下打散 bank。
4. **利用广播**：多个线程读取同一地址时，主动合并为广播访问。
5. **向量化与对齐**：在避免冲突的前提下使用 `float4` 等向量访问，减少指令数量。

&emsp;&emsp;bank conflict 的本质，是多个不同地址的访问在同一个 bank 上发生竞争。它不影响正确性，但会降低 shared memory 的有效带宽并放大延迟。只要从地址映射出发，预测每个线程会落到哪个 bank，就可以通过 padding、访问重排和 swizzle 等手段消除冲突。Swizzle 相比 padding 更通用，也更节省空间。

## 2 Swizzle

### 2.1 Swizzle

&emsp;&emsp;在内存访问优化中，Swizzle通过异或、移位、加法等轻量运算，对逻辑地址到物理地址的映射进行重排。与 padding 不同，swizzle 不增加存储空间，也不简单地插入空隙，而是把同一行或同一列中原本会落到同一 bank 的元素，重新映射到不同的 bank 上。其核心目标是：在保持数据布局规则、便于索引和向量化的同时，消除 bank conflict。

&emsp;&emsp;最经典的 swizzle 是 XOR swizzle。对于二维数组 `tile[row][col]`，如果按行主序存储，那么 word 索引为 `row * stride + col`，bank 为 `(row * stride + col) % 32`。当 `stride` 是 32 的倍数时，同一列 `col` 在不同行上会落到同一个 bank，导致按列访问时发生冲突。XOR swizzle 把列索引做变换：

$$
col' = col \oplus row
$$

或更一般地：

$$
col' = col \oplus (row \bmod 32)
$$

然后按 `tile[row][col']` 存储和访问。这样，当按列访问时，实际 bank 会随 `row` 变化而打散。

&emsp;&emsp;举例：`__shared__ float tile[32][32]`，原始布局中 `tile[tx][ty]` 的 bank 为 `(tx * 32 + ty) % 32 = ty`。当 warp 内 `tx` 变化而 `ty` 固定时，所有线程都落在 bank `ty`，形成 32-way conflict。若采用 XOR swizzle，存储时使用 `tile[row][col ^ row]`，读取时也使用相同映射。此时 `tile[tx][ty]` 实际访问的列是 `ty ^ tx`，其 word 索引为 `tx * 32 + (ty ^ tx)`，bank 为：

$$
(tx * 32 + (ty \oplus tx)) \bmod 32 = (ty \oplus tx) \bmod 32
$$

当 `tx` 从 0 到 31 变化时，`ty ^ tx` 会遍历 0 到 31，因此每个线程落到不同 bank，冲突被消除。

&emsp;&emsp;注意，swizzle 的映射必须可逆，且写入和读取使用相同的映射。通常把逻辑索引 `(row, col)` 映射为物理索引 `(row, col ^ f(row))`，其中 `f(row)` 是行相关的掩码。常见形式包括：

- `col ^ row`
- `col ^ (row % 32)`
- `col ^ ((row % 32) << 1)` 等，具体取决于 bank 宽度和访问粒度。
- 更一般地，`physical_index = logical_index ^ ((logical_index >> shift) & mask)`。

&emsp;&emsp;与 padding 相比，swizzle 的优势是不浪费 shared memory 空间，且可以保持数组维度是 2 的幂，便于向量化和地址计算。缺点是映射不再是简单的线性地址，调试和手工索引稍微复杂，但在 CUDA 中通常只增加一次异或运算，成本极低。

![](http://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/db6e5553-3cd0-49a2-93af-7cd4d1a9c054.png)

### 2.2 CUDA swizzle

### 2.2 CUDA swizzle

&emsp;&emsp;为了避免 bank conflict，swizzle 通常需要自己实现。比较通用的实现是：**把行索引的低位异或到列索引的若干位上，使不同行的同一列访问落在不同的 bank 上**。这一操作的本质是地址位的 XOR 置换，写入端和读取端使用完全相同的置换函数即可保证物理地址一致。

&emsp;&emsp;以 128B swizzle 为例，假设数据为 fp16，一行 128 字节（64 个 half 元素）。地址的低 7 位是行内偏移，其中 `bit[4:6]` 是 16 字节 chunk 索引（共 8 个 chunk），`bit[7:9]` 是行索引的低 3 位。Swizzle 规则可以写成：

```text
物理 chunk 索引 = 逻辑 chunk 索引 XOR (行索引低 3 位)
```

&emsp;&emsp;对应的 CUDA 实现通常是一个内联函数：

```cuda
__device__ __forceinline__
int swizzle_col(int row, int col) {
    // col 以 16 字节（8 个 half）为一个 chunk
    int chunk = col >> 3;          // 当前 chunk 编号 (0~7)
    int inner = col & 7;           // chunk 内偏移
    int swizzled = (chunk ^ (row & 7)) << 3 | inner;
    return swizzled;
}
```

&emsp;&emsp;写入端（如 `cp.async` 从全局内存写入共享内存）和读取端（如 `ldmatrix` 从共享内存读入寄存器）都调用同一个 `swizzle_col`，就能保证物理地址一致。XOR 的自逆性质 `a XOR b XOR b = a` 使得同一个函数既可以用于写入，也可以用于读取，无需维护两套地址映射。

&emsp;&emsp;这里有一个容易被忽视的**相位对齐**问题。硬件在执行 swizzle 时，是根据共享内存的**绝对地址**来计算 XOR 源位域的；而软件函数通常计算的是**相对于共享内存基址的偏移**。只有当基址对齐到 swizzle 周期的整数倍时，两者计算出的 XOR mask 才一致。对于 128B swizzle，周期为 `8 行 × 128 字节 = 1024 字节`，因此共享内存基址通常要求 **1024 字节对齐**。如果基址不对齐，写入端和读取端的相位就会错位，swizzle 失效，甚至产生静默的数据错误。

&emsp;&emsp;同时，Hopper 架构中支持硬件层面的 swizzle。TMA（Tensor Memory Accelerator）引擎可以在将数据从全局内存搬运到共享内存时，自动按照 tensor map 中指定的 swizzle 模式进行地址置换；从共享内存写回全局内存时，再自动执行反置换。TMA 支持三种标准模式：`CU_TENSOR_MAP_SWIZZLE_32B`、`CU_TENSOR_MAP_SWIZZLE_64B` 和 `CU_TENSOR_MAP_SWIZZLE_128B`，分别对应 32 字节、64 字节和 128 字节的 swizzle 粒度。使用 TMA swizzle 时，程序员不再需要手写地址置换函数，只需在创建 tensor map 时指定模式，硬件就会自动完成 swizzle 和反 swizzle。

&emsp;&emsp;但 TMA swizzle 也有其约束。`CU_TENSOR_MAP_SWIZZLE_128B` 要求 bounding box 的内维度不超过 128 字节，这意味着单个 tile 的行宽不能超过 128 字节。如果矩阵的逻辑行宽超过这个限制，就需要在 TMA 描述符中做额外的切分处理。此外，TMA swizzle 的原子粒度较粗，适合规则 tile；对于更细粒度、非均匀的访问模式，手写 XOR swizzle 仍然具有更高的灵活性。

&emsp;&emsp;因此，在实际 kernel 中，两种 swizzle 常常配合使用：TMA 负责全局内存到共享内存的大块搬运，利用硬件 swizzle 避免写入冲突；寄存器到共享内存的 `ldmatrix` 路径则继续使用手写 XOR swizzle，保证读取端的 bank conflict 也被消除。




