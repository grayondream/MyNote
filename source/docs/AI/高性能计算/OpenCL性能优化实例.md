# OpenCL性能优化示例

&emsp;&emsp;GPU 优化的核心逻辑高度一致：尽量让计算单元（ALU）保持忙碌，同时避免数据传输成为瓶颈（Memory Wall）。对应的优化套路相对固定，例如：

- 利用共享内存（Shared Memory）实现内存压缩；
- 保证访存合并（Memory Coalescing）；
- 消除分支分歧（Warp Divergence）；
- 流水线并行（Streams & Graphs）；
- 寄存器压力控制（Register Pressure）；
- 利用专用硬件单元（Tensor Cores / RT Cores）。

&emsp;&emsp;下面通过典型场景逐步拆解 OpenCL GPU 优化的基本方法。

## 1 直方图

**Base 实现**

&emsp;&emsp;直方图最直接的 OpenCL 实现如下，性能为 4.63ms。

```cpp
__kernel void histogramv1( __global const uchar* input, __global uint* output, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x >= width || y >= height){
        return;
    }

    int pixelId = width * y + x;
    uchar grayVal = input[pixelId];
    atomic_add(&output[grayVal], 1);
}
```

**Local Memory + cl_image**

&emsp;&emsp;上述实现存在明显问题：不同 work-group 都要访问 `output`，导致严重的竞争。最直接的优化方式是使用 local memory 避免 work-group 之间的竞争。另外，上述 kernel 使用 `cl_buffer`，但图像类任务通常更适合 `cl_image`，不过这也取决于具体硬件架构，需要针对实际硬件测试选择（笔者在 RTX 3050 上实测 `cl_image` 性能更优）。下面是 local memory + cl_image 的优化 kernel，性能提升至 0.69ms。

```cpp
__kernel void histogramv3_climage_local_histogram(__read_only image2d_t input, __global uint* output, int width, int height) {
    __local uint localHist[256];
    int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    // 1. 所有 work-item 都必须初始化自己的槽位（16x16=256，正好覆盖全部 bin）
    localHist[lid] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. 仅有效像素范围内的 work-item 参与累加
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        uint4 pixel = read_imageui(input, histogram_sampler, (int2)(x, y));
        uchar grayVal = (uchar)(pixel.x);
        atomic_add(&localHist[grayVal], 1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. 所有 work-item 都必须把自己槽位的结果合并到全局输出
    atomic_add(&output[lid], localHist[lid]);
}
```

**Patch 优化**

&emsp;&emsp;使用 local memory 一定程度上降低了竞争，但 work-group 内部的竞争仍然存在。可以一次处理多个像素来提升单个 work-item 的计算量，例如一次处理 4 个像素。优化后性能来到 0.680ms，可以看到优化幅度不大。

```cpp
__kernel void histogramv3_climage_local_histogram_patch(__read_only image2d_t input, __global uint* output, int width, int height) {
    __local uint localHist[256];
    int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    // 1. 所有 work-item 都必须初始化自己的槽位（16x16=256，正好覆盖全部 bin）
    localHist[lid] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. 仅有效像素范围内的 work-item 参与累加
    int gx = get_global_id(0) * 4;
    int y = get_global_id(1);
    for(int i = 0;i < 4;i ++){
        int x = gx + i;
        if(x >= width || y >= height){
            continue;
        }

        uint4 pixel = read_imageui(input, histogram_sampler, (int2)(x, y));
        uchar grayVal = (uchar)(pixel.x);
        atomic_add(&localHist[grayVal], 1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. 所有 work-item 都必须把自己槽位的结果合并到全局输出
    atomic_add(&output[lid], localHist[lid]);
}
```

**Partial 优化**

&emsp;&emsp;上述优化无论如何都避免不了竞争——使用单一 histogram 结构，这种竞争本质上是不可避免的。一种思路是为每个 work-item 开辟独立的内存空间暂存 histogram 结果，然后利用另一个 kernel 合并结果。这种优化降低了内存竞争，但引入了一个额外的 kernel，因此只有当竞争开销足够大时才有实际收益。笔者自测该方案反而劣化至 5.29ms。

```cpp
__kernel void histogram_partial(
    __read_only image2d_t input,
    __global uint *partial,
    int width,
    int height)
{
    __local uint localHist[256];

    int lid = get_local_id(1) * get_local_size(0)
            + get_local_id(0);

    if (lid < 256)
        localHist[lid] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height)
    {
        uint4 pixel =
            read_imageui(input, histogram_sampler, (int2)(x, y));

        uchar gray = (uchar)(pixel.x);

        atomic_add(&localHist[gray], 1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int group =
        get_group_id(1) * get_num_groups(0)
      + get_group_id(0);

    if (lid < 256)
    {
        partial[group * 256 + lid] = localHist[lid];
    }
}

__kernel void histogram_reduce(
    __global uint *partial,
    __global uint *output,
    int groupCount)
{
    int bin = get_global_id(0);

    if (bin >= 256)
        return;

    uint sum = 0;

    for (int i = 0; i < groupCount; i++)
    {
        sum += partial[i * 256 + bin];
    }

    output[bin] = sum;
}
```

**Tile 优化**

&emsp;&emsp;前面提到每次可以计算多个像素，patch 优化中每次处理 4 个像素。可以进一步基于 tile 进行处理，即每次处理 `tile x tile` 个像素，这样能充分利用 `cl_image` 针对图像的局部读写特性，同时增加每个 work-item 的并行度。优化后性能提升至 0.102ms。

```cpp
__kernel void histogram_tile4x4(
    __read_only image2d_t input,
    __global uint *output,
    int width,
    int height)
{
    __local uint localHist[256];

    int lid = get_local_id(1) * get_local_size(0)
            + get_local_id(0);

    if (lid < 256)
        localHist[lid] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    int baseX = get_global_id(0) * 4;
    int baseY = get_global_id(1) * 4;

    for (int dy = 0; dy < 4; dy++)
    {
        for (int dx = 0; dx < 4; dx++)
        {
            int x = baseX + dx;
            int y = baseY + dy;

            if (x >= width || y >= height)
                continue;

            uint4 pixel =
                read_imageui(input, histogram_sampler, (int2)(x, y));

            uchar gray = (uchar)(pixel.x);

            atomic_add(&localHist[gray], 1);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < 256)
    {
        atomic_add(&output[lid], localHist[lid]);
    }
}
```

**总结**

```bash
BM_Histogram_CPU/iterations:100                                57.4 ms         57.2 ms          100
BM_Histogram_OpenCL/iterations:100                             5.15 ms         5.14 ms          100
BM_Histogram_OpenCLV2_CLImage/iterations:100                   3.18 ms         3.18 ms          100
BM_Histogram_OpenCLV3_LocalHistogram/iterations:100           0.644 ms        0.642 ms          100
BM_Histogram_OpenCLV4_LocalHistogramPatch/iterations:100      0.642 ms        0.640 ms          100
BM_Histogram_OpenCL_Partial/iterations:100                     4.73 ms         4.71 ms          100
BM_Histogram_OpenCL_Tile4x4/iterations:100                    0.118 ms        0.118 ms          100
BM_Histogram_OpenCL_TilePartialStriped/iterations:100         0.258 ms        0.257 ms          100
BM_Histogram_OpenCL_TilePartialStripedBuf/iterations:100      0.265 ms        0.264 ms          100
```

&emsp;&emsp;上述优化的核心思路即开头所述：尽可能增加每个 work-item 的计算量，降低 work-item 之间的竞争。不同硬件架构的内存频率、显存结构等各不相同，因此需要针对具体场景选择具体的优化方案。例如 partial 方案在笔者的显卡上性能较差，但对于 atomic 竞争开销较大的硬件，反而可能提升性能。此外，local size 等参数也需要根据具体硬件场景进行调整，以最大化性能。

## 2 卷积

&emsp;&emsp;卷积是图像处理中最常见的计算，不同计算之间的具体逻辑和模式有所区别，但基本构型都是通过一个 n×n 的窗口对原图进行滤波，因此优化思路基本一致。

**Base**

&emsp;&emsp;首先实现一个基本的 5×5 高斯滤波 kernel，基础耗时为 3.02ms。

```cpp
__kernel void convolution_v1_naive(
    __global const float* input,
    __global float* output,
    const int width,
    const int height,
    __constant float* kernel_weights,
    const int kernel_radius)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int ksize = 2 * kernel_radius + 1;
    for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
        for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
            int sx = clamp(x + kx, 0, width - 1);
            int sy = clamp(y + ky, 0, height - 1);
            sum += input[sy * width + sx] * kernel_weights[(ky + kernel_radius) * ksize + (kx + kernel_radius)];
        }
    }
    output[y * width + x] = sum;
}
```

**Tile 优化**

&emsp;&emsp;频繁访问 global memory 会导致较大延迟，因此可以将 input 数据临时加载到 local memory 上。通常 local memory 的读取效率高于 global memory。优化后 kernel 性能来到 1.67ms。需要注意的是，local memory 优化效果因硬件架构而异：对独立显卡优化幅度较大，但对移动端显卡效果不一定理想。

```cpp
__kernel void convolution_v2_local_memory(
    __global const float* input,
    __global float* output,
    const int width,
    const int height,
    __constant float* kernel_weights,
    const int kernel_radius)
{
    __local float local_tile[20 * 20];
    int gx = get_global_id(0), gy = get_global_id(1);
    int lx = get_local_id(0), ly = get_local_id(1);
    int group_x = get_group_id(0) * 16;
    int group_y = get_group_id(1) * 16;

    for (int idx = ly * 16 + lx; idx < 400; idx += 256) {
        int ty = idx / 20, tx = idx % 20;
        int sx = clamp(group_x + tx - 2, 0, width - 1);
        int sy = clamp(group_y + ty - 2, 0, height - 1);
        local_tile[idx] = input[sy * width + sx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx >= width || gy >= height) return;

    float sum = 0.0f;
    for (int ky = 0; ky < 5; ky++)
        for (int kx = 0; kx < 5; kx++)
            sum += local_tile[(ly + ky) * 20 + (lx + kx)] * kernel_weights[ky * 5 + kx];
    output[gy * width + gx] = sum;
}
```

**Unroll 优化**

&emsp;&emsp;将循环内的部分计算外移以降低重复计算。优化后 1.46ms，提升幅度不大，若性能波动反而可能劣化。

```cpp
__kernel void convolution_v3_optimized(
    __global const float* input,
    __global float* output,
    const int width,
    const int height,
    __constant float* kernel_weights,
    const int kernel_radius)
{
    __local float local_tile[20 * 20];
    int gx = get_global_id(0), gy = get_global_id(1);
    int lx = get_local_id(0), ly = get_local_id(1);
    int group_x = get_group_id(0) * 16;
    int group_y = get_group_id(1) * 16;

    for (int idx = ly * 16 + lx; idx < 400; idx += 256) {
        int ty = idx / 20, tx = idx % 20;
        int sx = clamp(group_x + tx - 2, 0, width - 1);
        int sy = clamp(group_y + ty - 2, 0, height - 1);
        local_tile[idx] = input[sy * width + sx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx >= width || gy >= height) return;

    float sum = 0.0f;
    __local float* base = &local_tile[ly * 20 + lx];
    #pragma unroll
    for (int ky = 0; ky < 5; ky++) {
        #pragma unroll
        for (int kx = 0; kx < 5; kx++) {
            sum += base[ky * 20 + kx] * kernel_weights[ky * 5 + kx];
        }
    }
    output[gy * width + gx] = sum;
}
```

**cl_image 优化**

&emsp;&emsp;图像处理中使用 `cl_image` 通常更有优势。将输入改为 `read_only image2d_t` 后，读取效率大幅提升，性能来到 1.30ms，可见 `cl_image` 在图像处理中的确高效。

```cpp
__kernel void convolution_v4_image(
    __read_only image2d_t input_image,
    __global float* output,
    const int width,
    const int height,
    __constant float* kernel_weights,
    const int kernel_radius)
{
    __local float local_tile[20 * 20];
    int gx = get_global_id(0), gy = get_global_id(1);
    int lx = get_local_id(0), ly = get_local_id(1);
    int group_x = get_group_id(0) * 16;
    int group_y = get_group_id(1) * 16;
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    for (int idx = ly * 16 + lx; idx < 400; idx += 256) {
        int ty = idx / 20, tx = idx % 20;
        int sx = clamp(group_x + tx - 2, 0, width - 1);
        int sy = clamp(group_y + ty - 2, 0, height - 1);
        local_tile[idx] = read_imagef(input_image, sampler, (int2)(sx, sy)).x;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx >= width || gy >= height) return;

    float sum = 0.0f;
    #pragma unroll
    for (int ky = 0; ky < 5; ky++)
        #pragma unroll
        for (int kx = 0; kx < 5; kx++)
            sum += local_tile[(ly + ky) * 20 + (lx + kx)] * kernel_weights[ky * 5 + kx];
    output[gy * width + gx] = sum;
}
```

**Vectorization 优化**

&emsp;&emsp;输入图像为 4K，所需线程较多，线程调度本身存在开销。尝试将每个 kernel 处理的像素数量提升 4 倍，增加每个线程的计算量，降低线程调度开销。但优化后性能反而劣化至 2.06ms，说明计算压力增大后无法有效掩盖调度开销。

```cpp
__kernel void convolution_v5_vectorized(
    __global const float* input,
    __global float* output,
    const int width,
    const int height,
    __constant float* kernel_weights,
    const int kernel_radius)
{
    const int VEC = 4;
    const int LW = 16 * VEC + 4;
    const int LH = 16 + 4;
    __local float local_tile[LH * LW];

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    int group_x = get_group_id(0) * 16 * VEC;
    int group_y = get_group_id(1) * 16;

    int tid = ly * 16 + lx;
    int total_elems = LH * LW;

    for (int idx = tid; idx < total_elems; idx += 16 * 16) {
        int ty = idx / LW;
        int tx = idx % LW;
        int sx = clamp(group_x + tx - 2, 0, width - 1);
        int sy = clamp(group_y + ty - 2, 0, height - 1);
        local_tile[ty * LW + tx] = input[sy * width + sx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int out_x = gx * VEC;
    int out_y = gy;
    if (out_y >= height) return;

    for (int vi = 0; vi < VEC; vi++) {
        if (out_x + vi >= width) return;
        float sum = 0.0f;
        __local float* b = &local_tile[(ly + 2) * LW + (lx * VEC + vi + 2)];
        #pragma unroll
        for (int ky = 0; ky < 5; ky++) {
            #pragma unroll
            for (int kx = 0; kx < 5; kx++) {
                sum += b[ky * LW + kx] * kernel_weights[ky * 5 + kx];
            }
        }
        output[out_y * width + out_x + vi] = sum;
    }
}
```

**Wide Tile 优化**

&emsp;&emsp;为进一步降低每个 work-group 的读写压力，可以增大 local memory 的 tile 尺寸，使得同一 work-group 中各 work-item 能够减少重复读取 global memory 的次数，更多地命中 local memory。这里将 local tile 设为 38×12：预留 halo 区域避免边界像素的 cache miss，同时适配图像宽/高不对称的特点。local size 设为 32×8 且小于 local memory 容量，有利于掩盖 memory bound，使计算线程保持忙碌。优化后性能为 1.40ms。

```cpp
__kernel void convolution_v6_wide_tile(
    __global const float* input,
    __global float* output,
    const int width,
    const int height,
    __constant float* kernel_weights,
    const int kernel_radius)
{
    const int TW = 32;
    const int TH = 8;
    const int LW = TW + 4;
    const int LH = TH + 4;
    __local float local_tile[LH * LW];

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int group_x = get_group_id(0) * TW;
    int group_y = get_group_id(1) * TH;

    int tid = ly * TW + lx;
    int total_elems = LH * LW;
    for (int idx = tid; idx < total_elems; idx += TW * TH) {
        int ty = idx / LW, tx = idx % LW;
        int sx = clamp(group_x + tx - 2, 0, width - 1);
        int sy = clamp(group_y + ty - 2, 0, height - 1);
        local_tile[idx] = input[sy * width + sx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx >= width || gy >= height) return;

    float sum = 0.0f;
    #pragma unroll
    for (int ky = 0; ky < 5; ky++)
        #pragma unroll
        for (int kx = 0; kx < 5; kx++)
            sum += local_tile[(ly + ky) * LW + (lx + kx)] * kernel_weights[ky * 5 + kx];
    output[gy * width + gx] = sum;
}
```

**Register Tiling 优化**

&emsp;&emsp;在 v6 local memory 缓存的基础上引入**寄存器分块（Register Tiling）**技术（即 Thread Coarsening / 线程粗粒化），让每个线程利用寄存器同时计算并输出相邻的 2×2（共 4 个）像素。这种策略大幅摊薄了 local memory 的索引计算与数据读取开销，在不增加线程总数的前提下成倍提升了数据复用率与计算密度。优化后性能达到 0.88ms。

```cpp
__kernel void convolution_v7_register_tile_8x8(
    __global const float* input,
    __global float* output,
    const int width,
    const int height,
    __constant float* kernel_weights,
    const int kernel_radius)
{
    const int TILE = 8;
    const int LW = TILE * 2 + 4;
    const int LH = TILE * 2 + 4;
    __local float local_tile[LH * LW];

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int group_x = get_group_id(0) * TILE * 2;
    int group_y = get_group_id(1) * TILE * 2;

    int tid = ly * TILE + lx;
    int total_elems = LH * LW;
    for (int idx = tid; idx < total_elems; idx += TILE * TILE) {
        int ty = idx / LW, tx = idx % LW;
        int sx = clamp(group_x + tx - 2, 0, width - 1);
        int sy = clamp(group_y + ty - 2, 0, height - 1);
        local_tile[idx] = input[sy * width + sx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int ox = gx * 2, oy = gy * 2;
    if (ox + 1 < width && oy + 1 < height) {
        float s00 = 0, s10 = 0, s01 = 0, s11 = 0;
        #pragma unroll
        for (int ky = 0; ky < 5; ky++) {
            #pragma unroll
            for (int kx = 0; kx < 5; kx++) {
                float w = kernel_weights[ky * 5 + kx];
                int b = (ly * 2 + ky) * LW + (lx * 2 + kx);
                s00 += local_tile[b] * w;
                s10 += local_tile[b + 1] * w;
                s01 += local_tile[b + LW] * w;
                s11 += local_tile[b + LW + 1] * w;
            }
        }
        output[oy * width + ox] = s00;
        output[oy * width + ox + 1] = s10;
        output[(oy + 1) * width + ox] = s01;
        output[(oy + 1) * width + ox + 1] = s11;
    } else if (ox < width && oy < height) {
        float sum = 0.0f;
        for (int ky = 0; ky < 5; ky++)
            for (int kx = 0; kx < 5; kx++)
                sum += local_tile[(ly * 2 + ky) * LW + (lx * 2 + kx)] * kernel_weights[ky * 5 + kx];
        output[oy * width + ox] = sum;
    }
}
```

**总结**

```bash
BM_Convolution_CPU/iterations:50                               695 ms          693 ms           50
BM_Convolution_OpenCL_V1/iterations:50                        3.02 ms         3.01 ms           50
BM_Convolution_OpenCL_V2/iterations:50                        1.68 ms         1.67 ms           50
BM_Convolution_OpenCL_V3/iterations:50                        1.46 ms         1.46 ms           50
BM_Convolution_OpenCL_V4/iterations:50                        1.30 ms         1.30 ms           50
BM_Convolution_OpenCL_V5/iterations:50                        2.06 ms         2.06 ms           50
BM_Convolution_OpenCL_V6/iterations:50                        1.40 ms         1.39 ms           50
BM_Convolution_OpenCL_V7/iterations:50                       0.802 ms        0.800 ms           50
```

&emsp;&emsp;卷积与直方图的优化思路有相似之处，但侧重不同。直方图侧重于降低线程间的竞争，而卷积主要在于提升并行度、避免计算被访存瓶颈限制、提升计算密集性。这些优化思路需根据具体场景灵活选择。

## Pipeline

&emsp;&emsp;通常我们独立实现多个 kernel，然后通过 event 同步调度。这样虽然 kernel 内部是并行执行的，但 kernel 之间仍然同步——必须等第一个 kernel 完全执行结束，第二个 kernel 才能开始。OpenCL 提供了 Pipe 机制，可以实现 work-item 之间的流水线并行。Pipe Kernel 相比传统的连续 Kernel Enqueue 方式，其核心优势在于实现了高度并发的硬件级流水线并行化：数据通过硬件内部的 FIFO 直接流式传输，消除了对外部内存的读写依赖，使 Producer 和 Consumer Kernel 能够在时间上完全重叠运行，从而大幅提升吞吐量并降低数据传延迟。

&emsp;&emsp;但该特性需要设备支持，笔者使用的 RTX 3050 不支持 Pipe，因此无法测试。

```cpp
__kernel void convolution_to_pipe(
    __global const float* input,
    __write_only pipe float output_pipe,
    const int width,
    const int height,
    __constant float* kernel_weights,
    const int kernel_radius)
{
    __local float local_tile[20 * 20];
    int gx = get_global_id(0), gy = get_global_id(1);
    int lx = get_local_id(0), ly = get_local_id(1);
    int group_x = get_group_id(0) * 16;
    int group_y = get_group_id(1) * 16;

    for (int idx = ly * 16 + lx; idx < 400; idx += 256) {
        int ty = idx / 20, tx = idx % 20;
        int sx = clamp(group_x + tx - 2, 0, width - 1);
        int sy = clamp(group_y + ty - 2, 0, height - 1);
        local_tile[idx] = input[sy * width + sx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx >= width || gy >= height) return;

    float sum = 0.0f;
    for (int ky = 0; ky < 5; ky++)
        for (int kx = 0; kx < 5; kx++)
            sum += local_tile[(ly + ky) * 20 + (lx + kx)] * kernel_weights[ky * 5 + kx];

    write_pipe(&output_pipe, &sum);
}

// Pipe version: Histogram reads float from pipe
__kernel void histogram_from_pipe(
    __read_only pipe float input_pipe,
    __global uint* output,
    const int width,
    const int height)
{
    float val;
    read_pipe(&input_pipe, &val);

    int bin = clamp((int)(val + 0.5f), 0, 255);
    atomic_add(&output[bin], 1);
}
```

## 3 整体层面的优化策略

&emsp;&emsp;前面聊的都是在单个 kernel 里面折腾。但在实际项目中，往往是一堆 kernel 串成一个流水线。这时候你会发现——**单个 kernel 再快，整体也可能慢得离谱**。我自己的体会是，整体层面的优化往往比扣一个 kernel 的细节收益更大，但坑也更多。下面聊几个我踩过的坑。

### 3.1 CPU-GPU 异步调度——别让 CPU 傻等

&emsp;&emsp;先说一个最常见的错误：Enqueue 完 kernel 之后马上调 `clFinish` 等结果。这种做法相当于让 CPU 啥也不干蹲在那里等 GPU 干完活。其实 OpenCL 的 Enqueue 默认是非阻塞的，你大可以在 Enqueue 之后让 CPU 去干别的事。

&emsp;&emsp;举个例子，之前做一个视频处理 pipeline，要经过"直方图统计 → 亮度调整 → 输出编码"三个步骤。最开始的写法是每步都 `clFinish` 等结果，整条流水线跑下来 CPU 大量时间在空转。后来改成用 `cl_event` 链起来——直方图 kernel 的 `event` 传给亮度调整 kernel 的 `event_wait_list`，亮度调整的 `event` 传给输出 kernel。这样 GPU 端自己就能维护执行顺序，CPU 完全不需要插手。同样的逻辑，单纯改掉 `clFinish`，整个 pipeline 吞吐量就提升了 20% 多。

&emsp;&emsp;至于 `clSetEventCallback`，我实际用的并不多，因为大部分场景用 event_wait_list 就够了。但在一些需要 CPU 在 GPU 完成后立即处理结果的场景（比如实时预览），直接wait event就可以了。

### 3.2 数据传输与计算重叠——别让 PCIe 卡脖子

&emsp;&emsp;我之前有个认知误区：总觉得计算是瓶颈，花了很多精力优化 kernel 的指令级并行。结果一上 profiler 发现 GPU 计算单元空闲率很高，真正卡的地方是 PCIe 传输。说白了就是数据还没到，GPU 只能干等着。

&emsp;&emsp;这是个比较典型的场景——图像处理中需要把视频帧从 CPU 传到 GPU，处理完再传回来。最笨的做法是：传一帧 → 处理一帧 → 读回一帧。这样传输和计算是串行的。

&emsp;&emsp;**双缓冲（Ping-Pong Buffer）** 是我用得最多的解法。大概思路是这样的：
- 分配两个 buffer A 和 B
- 帧 1 传入 buffer A 的同时，GPU 开始处理 buffer B 中的上一帧
- 帧 2 传入 buffer B，GPU 处理 buffer A

&emsp;&emsp;看起来很简单，但实现上有个容易翻车的地方——event 依赖没设对的话，要么数据覆盖了，要么算到一半数据还没到。我的习惯是用 `clEnqueueBarrierWithWaitList` 来保证传输完成之后才开始计算，而不是靠 `clFinish`。

&emsp;&emsp;另外，对于 `clEnqueueMapBuffer` 和 `clEnqueueWriteBuffer` 的选择，我自己的测试结果是：如果数据量比较大（几百 MB 以上），Map 模式配合 `CL_MEM_ALLOC_HOST_PTR` 有时反而比 Write 慢，因为 Map 涉及 page fault 的开销。所以还是要实测，不要迷信某个方案。

### 3.3 Kernel Launch 开销——小 kernel 多了也疼

&emsp;&emsp;以前觉得 launch kernel 能有多贵，反正就是提交个命令。直到有一次做简单的逐像素操作，我拆了三个 kernel（阈值化 + 减均值 + 归一化），每个 kernel 就几行代码。结果 profiler 一看，每个 kernel 跑得飞快（十几微秒），但 launch overhead 占了大头——三个 kernel 加起来 launch 时间比执行时间还长。

&emsp;&emsp;这种情况我一般会考虑几种做法：

**Kernel Fusion（合并）**：把三个小 kernel 捏成一个。但要注意寄存器压力——三个 kernel 合并后临时变量多了，寄存器不够用的话会被 spill 到 local memory，反而变慢。我之前就在这翻过车，合并后 kernel 从十几微秒变成了四十多微秒，还不如分开跑。后来裁剪了中间变量的数量才解决。

**Batching**：举个例子，需要对一批 1000 张图做相同的前处理。不优化的话就是 for 循环 enqueue 1000 次 kernel。其实可以一次 enqueue 把 global size 设成 `1000 * width * height`，然后每个 work-item 通过 global_id 判断处理哪张图的哪个像素。这样 enqueue 次数从 1000 降到了 1，launch 开销直接省了。

**Persistent Kernel**：这个用的场景比较特定——比如在做实时渲染的时候，需要反复对同一帧做多次滤波。可以让 kernel 起来之后不退出，自己循环从队列里取任务。好处是没有反复 launch 的开销，坏处是调试困难，而且占着 GPU 资源不让别人用。我自己只在需要极致延迟的场景用过一次，一般来说 fusion 就够用了。

### 3.4 内存管理——频繁分配显存是个坑

&emsp;&emsp;`clCreateBuffer` 看着就是一个函数调用，但实际开销比想象中大得多——它涉及驱动层的内存分配和映射。我之前在实时处理系统里，每帧都创建释放临时 buffer，结果 profiler 显示 `clCreateBuffer` 和 `clReleaseMemObject` 占了不少时间。

&emsp;&emsp;解决的思路也很直接：**用内存池**。在初始化阶段就把可能用到的 buffer 全部分配好，运行时只复用不释放。比如做多帧的流水线处理时，我习惯预先分配 3~4 个 buffer 做循环队列，远比每帧动态分配靠谱。

&emsp;&emsp;还有一个容易被忽视的点是 **Sub-buffer**。比如你有两个 kernel，一个处理图像上半部分，一个处理下半部分。如果各自分配 buffer，就需要两次分配和传输。但如果先分配一个整图大小的 buffer，然后用 `clCreateSubBuffer` 切分成两个逻辑区域，两个 kernel 分别操作各自的子区域——这样分配次数减半、传输也只需要一次。不过要注意 sub-buffer 的对齐要求，偏移必须是 `CL_DEVICE_MEM_BASE_ADDR_ALIGN` 的整数倍，否则会报错。

&emsp;&emsp;至于 Zero-copy（`CL_MEM_USE_HOST_PTR`），我在移动端（Mali GPU）上试过确实有效，因为 UMA 架构下 CPU 和 GPU 共享物理内存，不需要真的拷贝。但在独立显卡上就别想了——`CL_MEM_USE_HOST_PTR` 实际上还是会把数据拷到显存，并不会真的 zero-copy。所以这个优化也要看平台。

### 3.5 多队列并行——让 GPU 同时干多件事

&emsp;&emsp;GPU 其实可以同时干不同的事，前提是你用多个 command queue。但这事我没有成功过几次，因为大部分 OpenCL 实现默认队列是 in-order 的，即使创建多个队列，驱动也不一定会真正并行调度。

&emsp;&emsp;不过我见过一个比较成功的案例：在一个实时视频处理系统里，把计算密集的滤波 kernel 放一个队列，把轻量的直方图统计放另一个队列。由于滤波 kernel 占满了大部分计算单元，直方图 kernel 可以在滤波的间隙"见缝插针"地跑。整体吞吐量比单队列高了 10% 左右。

&emsp;&emsp;需要注意的是，多队列调试起来非常痛苦——事件依赖很容易搞乱，跑出来的结果时对时错。我的建议是先确保单队列调通再考虑多队列，而且一定要用 `clEnqueueBarrierWithWaitList` 或 event 依赖做同步，别指望 GPU 自动做对。

&emsp;&emsp;总的来说，整体优化最核心的一点就是：**别让任何一方闲着**。CPU 不等 GPU，GPU 不等数据，传输和计算重叠。说起来简单，做起来全是细节。每个方案都有前提条件，都得根据自己实际的硬件和场景来试。


## 4 Profiling——怎么找到那个"真正的瓶颈"

&emsp;&emsp;说实话，优化思路网上到处都是，翻来覆去就那么几条。真正的难点从来不是"知不知道 local memory 比 global memory 快"，而是**面对几千行代码，你根本不知道瓶颈在哪**。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/perfetto_cuda.png)

&emsp;&emsp;我自己就经常这样：凭直觉觉得某个地方"应该是瓶颈"，花了一整天优化，结果 profiler 一看——根本就不是它。这种事情发生几次之后，我就学乖了：先 profiling，再动手。优化最忌讳的就是"我觉得"。

### 4.1 先想清楚方向，再动手

&emsp;&emsp;做 profiling 之前先问自己一个问题：**这个 kernel 是被什么限制的？计算还是访存？**

&emsp;&emsp;这个问题很重要，因为方向错了的话，优化就是南辕北辙。举两个例子：

- **如果瓶颈在访存**（memory-bound）：你花精力去抠指令并行度、展开循环、改用 FMA——这些优化几乎不会有收益，因为计算单元本来就在等数据。这时候应该做的是：提高 data reuse、用 local memory 做 tiling、保证访存合并。比如前面直方图从 naive 到 local memory 再到 tile，每一步都在提升数据复用率，核心逻辑就是"少访问 global memory"。

- **如果瓶颈在计算**（compute-bound）：你优化 local memory 也提升不大，因为计算单元已经满载了。这时候应该关注的是：减少冗余计算、提高 ILP（指令级并行）、使用 native 函数代替昂贵数学函数、利用 Tensor Cores 等专用硬件。

&emsp;&emsp;怎么判断？可以粗略地算一下计算强度（Arithmetic Intensity = 总计算量 / 总访存量），然后和硬件的 OI（Operational Intensity）曲线对比。但更直接的方法是上 profiler 看 stall reason——如果大部分 stall 是 memory 相关的，那就是 memory-bound；如果是 math pipe 相关的，那就是 compute-bound。Nsight Compute 里直接就有这个分类，一目了然。

&emsp;&emsp;另外，有时候可以考虑换个方向——**计算等价变换**。比如大卷积核（7×7 以上）用滑窗实现计算量很大，但如果把它转成 im2col + GEMM，或者用 FFT 加速，性能可能直接翻倍。不过这种较大幅度的重构需要验证正确性，我一般在 profiling 确认了确实是 compute-bound 之后才考虑。

### 4.2 各平台 Profiling 工具——挑顺手的用

&emsp;&emsp;这块没什么好说的，每个平台都有自己的工具。我把自己用过的列出来，各有优劣，没必要全学，挑自己平台能用的就行。

**NVIDIA GPU**

- **Nsight Compute**：我最常用的。它的 GUI 很直观，打开一个 kernel 就能看到 occupancy 是多少、被什么限制了、memory stall 占比多少。对于 OpenCL 来说，它会把 kernel 自动转成 CUDA 的 SASS 指令来分析。有一栏叫 **Speed of Light**，能一眼看到当前 kernel 距离硬件极限还有多远——是没跑满计算吞吐还是没跑满显存带宽。
- **Nsight Systems**：适合看全局 timeline。我曾经用它发现一个"怪现象"：某个 kernel 在 timeline 上明明很早就 enqueue 了，但实际执行时间比预期晚了很多。仔细一看发现是前一个 kernel 的输出 buffer 还在被 CPU 读，导致 GPU 没法写入——这就是没做对双缓冲的后果。
- **关键 counter**：`gld_efficiency`，这个看 global load 的效率，如果低于 100% 说明访存没有完全合并；`stall_memory_threshold` 看有多少 stall 在等内存。这两个是我平时最关注的。

**AMD GPU**

- **rocprof**：命令行工具，输出 CSV 格式。虽然没有 Nsight Compute 那么好看的 GUI，但该有的 counter 都有。我主要在 ROCm 上用，关注 `SQ_WAVES`（wavefront 数量）和 `VGPR` 使用情况——这两个直接关系到 occupancy。
- **OmniTrace**：基于 rocprof 的 trace 工具，有个 Web UI 可以看 timeline，比原始 CSV 友好一些。

**Intel / 移动端 GPU**

&emsp;&emsp;Intel 的集成显卡我用得不多，VTune 功能很全但是有点重。移动端的话，Qualcomm 有 Adreno Profiler，Arm 有 Streamline——两者都能拿到 GPU 频率、带宽、shader 核心利用率。我个人觉得移动端 profiling 的难点不是工具不够，而是功耗和温控带来的降频问题——经常 bench 到一半 GPU 降频了，数据变得没法看。所以移动端测试要特别注意控制温度，测试时间不要太长。

### 4.3 拿到数据之后怎么看

&emsp;&emsp;profiler 吐出来的数据很多，但实际有用的就那几个指标。我一般按这个顺序看：

**① Occupancy——线程够不够多？**

&emsp;&emsp;Occupancy 低的原因通常就几个：寄存器用太多、local memory 占太多、work-group size 太小。但要注意一个反直觉的点：occupancy 不是越高越好。我自己就遇到过，调低了 occupancy 反而性能更好——因为线程少了，cache 竞争也少了。所以 occupancy 是一个"参考，但不迷信"的指标。

**② 带宽利用率——显存放在瓶颈？**

&emsp;&emsp;如果实际使用带宽远低于硬件峰值，说明访存模式有问题。最常见的原因：**访存没有合并**——相邻的线程没有访问连续的内存地址。比如前面的 histogram v1 用 global memory 做 atomic，每个线程访问的 output 地址完全随机，根本无法合并。后来改 local memory 之后，虽然 local memory 内部还是随机访问，但至少 global 侧的带宽占用大幅下降了。

**③ 计算 vs 访存 stall——谁在拖后腿？**

&emsp;&emsp;这个是最关键的。Nsight 里直接会告诉你：这个 kernel 有 x% 的 stall 是 memory 相关，y% 是 math 相关。如果 memory stall 超过 50%，那就别折腾计算优化了，先去修访存问题。

**④ 分支分歧——if-else 多了也疼**

&emsp;&emsp;GPU 里 SIMT 执行模式下，一个 warp 里的线程走不同分支会导致所有线程都执行一遍所有分支。之前做图像分类时有个 kernel 里写了 `if (class_id == 0) ... else if (class_id == 1) ... else ...`，结果一共 10 个分支，每个线程只用到 1 个，但每次都要跑 10 个分支——性能直接掉了 80%。后来把分类拆成 10 个独立的 kernel，每个只处理一类，速度就正常了。

**⑤ 指令 mix——有没有用到坑爹指令？**

&emsp;&emsp;`sin`、`cos`、`exp`、`pow`、`sqrt` 在 GPU 上都很贵。能换成 lookup table 或者 `native_sin`、`native_exp`（精度要求不高的情况下）就换。还有就是 FMA（融合乘加）——很多 GPU 的 FMA 吞吐是普通乘+加的 2 倍，所以能用 `fma(a, b, c)` 就尽量用。

### 4.4 我的优化流程——先测再改、一次只改一个

&emsp;&emsp;最后分享一下我自己用的优化流程，谈不上什么方法论，就是吃了很多次亏之后总结的：

1. **记录基线**：先跑一次未经优化的代码，记下耗时和 profiler 数据。这一步很容易被跳过去——"哎呀反正后面会优化，基线后面再补"。但后面就忘了补了，最后也没法对比优化了多少。所以我现在的习惯是：打开 profiler 之后，先导出一份 baseline 报告再开始改。
2. **提出假设**：根据 profiling 数据说一句完整的话——"我觉得这个 kernel 慢是因为内存访问不合并导致 L1 cache miss 率高，进而使大部分 time 花在了 memory stall 上"。
3. **做一项改动**：只改一个变量。比如只改访存模式、或者只改 local size、或者只加 unroll，但不要同时改三个。我承认这个很难忍住——改都改了，不如把能想起来的优化都加上。但这样跑出来之后，你根本不知道是哪个改动起了作用，可能是 A 改好了但 B 改坏了，最后抵消了。
4. **验证**：重新跑一次，看假设对不对。如果问题真的缓解了，那就继续；如果没变化或者更差了，那说明假设错了，回退改动重新分析。
5. **重复**：找到下一个瓶颈，再来一轮。当优化收益降到 10% 以下、或者性能已经达到需求时，就停手。过度优化也是浪费时间的。

&emsp;&emsp;说起来，"一次只改一个变量"这个道理谁都懂，但我做了这么多年还是经常会犯"改完顺手把 local size 也调了"这种毛病。可能这就是优化的常态吧——理论是一回事，实操的时候老老实实改一个测一个，反而才是最快的路。但是现如今有了AI,这方面反而简单了，可以给AI指定优化方向将所有流程自动化让AI完成优化。
