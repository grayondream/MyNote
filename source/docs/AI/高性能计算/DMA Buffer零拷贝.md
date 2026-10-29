# DMA Buffer 的使用与创建：从内核映射到 GPU API

&emsp;&emsp;DMA（Direct Memory Access）让外设与内存之间的数据搬运绕开 CPU 直接进行，是高性能驱动开发的基石。但"绕过 CPU"恰恰是复杂性的来源：CPU 和 device 看到的内存可能不一致，物理内存可能不连续，映射的建立和拆除需要严格的配对。DMA Buffer 的创建和使用，本质上就是在正确性、性能和可移植性之间做出系统性的选择。

---

## 1. DMA
### 1.1 简介

&emsp;&emsp;驱动调用 `dma_alloc_coherent()` 或 `dma_map_single()` 时，内核完成的不只是内存分配。`dma_alloc_coherent()` 同时做两件事：分配一块 CPU 可访问的内存，并建立 device 侧的 DMA 地址映射；`dma_map_single()` 则是对一块**已经存在**的缓冲区（通常是 `kmalloc()` 得到的）临时建立映射。前者返回两个地址——CPU 虚拟地址和 `dma_addr_t` 类型的 DMA 地址；后者在映射成功时返回 DMA 地址，缓冲区本身由驱动自行管理。

&emsp;&emsp;这两条路径背后是两套不同的正确性模型，选错路径导致的 bug 往往表现为静默的数据损坏，而非明确的错误返回。

&emsp;&emsp;**图 1：两种映射模型——coherent 双方可同时访问；streaming 所有权在 CPU 与 device 之间转移**

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261029_0baea1.svg)


---

### 1.2. DMA Buffer 的创建、使用与连续内存

&emsp;&emsp;**一致性映射的核心契约。** `dma_alloc_coherent()` 分配的缓冲区在 CPU 和 device 之间是**硬件或架构层面保证一致**的。驱动不需要在传输前后调用 `dma_sync_single_for_*()` 做 cache 冲刷或失效。在支持硬件 cache 一致性的总线上，它返回的是普通可缓存内存；在不支持的平台上，内核会退化为 uncached 内存，或从特殊窗口分配。如果平台连 uncached 都无法提供，分配会直接失败。

&emsp;&emsp;**一致性映射的适用场景与代价。** 它的不可替代性在于**CPU 和 device 需要同时访问同一块内存**。典型场景是 DMA 描述符环：device 在处理描述符的同时，CPU 可能正在检查已完成项、填充新描述符。如果这块内存是流式映射的，CPU 对描述符的写入可能还停留在 cache 中，device 读到的就是旧描述符——而一致性映射从根源上消除了这种窗口。网络驱动的环形缓冲区、存储控制器的命令队列，都属于这一类；此时 buffer 的生命周期通常与设备打开/关闭同步，在整个数据传输期间持续存在。代价则因架构而异：在 x86 等平台上，coherent 内存通常是非可缓存的，CPU 读写会承受额外的延迟；在 ARM 上，常见实现是关闭对应区域的 cache。此外，`dma_alloc_coherent()` 的分配粒度是页的整数倍，并不适合"小块临时数据"的场景。还有一个容易被忽视的约束：**一致性映射不应与流式同步 API 混用**。`dma_sync_single_for_*()` 属于流式 API，对 coherent buffer 调用它们没有意义，在某些实现上甚至可能触发警告。它的创建与释放接口简单，但配对的纪律同样重要：

```c
dma_addr_t dma_handle;
void *cpu_addr = dma_alloc_coherent(dev, size, &dma_handle, GFP_KERNEL);
if (!cpu_addr)
    return -ENOMEM;

/* cpu_addr 用于 CPU 访问，dma_handle 用于编程 device */
/* 期间不需要任何 dma_sync_* 调用 */

dma_free_coherent(dev, size, cpu_addr, dma_handle);
```

&emsp;&emsp;`dma_alloc_coherent()` 的第四个参数是分配标志。在原子上下文或不能睡眠的路径中，需要把 `GFP_KERNEL` 换成 `GFP_ATOMIC`，否则可能触发睡眠告警或死锁。

&emsp;&emsp;**流式映射的所有权模型。** 流式映射不分配内存，它把驱动自己持有的缓冲区临时"交给" device 使用。Russell King 在 LKML 上对这套 API 的概括是最清晰的：`dma_map_single()` 和 `dma_sync_single_for_device()` 把 buffer 的**所有权从 CPU 转移给 device**；`dma_unmap_single()` 和 `dma_sync_single_for_cpu()` 把所有权从 device **交还 CPU**。由此可以推出两条铁律：其一，**在 `map` 和 `unmap` 之间，CPU 原则上不应触碰缓冲区**，因为此时 device 是 owner；如果确实需要在传输中途让 CPU 读一下进度或修改数据，必须用 `dma_sync_single_for_cpu()` 正式"收回"访问权，处理后再用 `dma_sync_single_for_device()` 交还。其二，**如果从 `map` 到 `unmap` 之间 CPU 完全不碰数据**，则中间的 `dma_sync_*` 调用可以完全省略，只需要保证 `map` 与 `unmap` 严格配对。

&emsp;&emsp;**图 2：所有权时间线——map/unmap 是交接点，中途访问需 sync 临时收回**

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261029_9b9e17.svg)

&emsp;&emsp;**方向参数不是装饰。** `dma_map_single(dev, addr, size, direction)` 的 `direction` 参数直接决定内核在映射和同步时执行哪组 cache 操作：`DMA_TO_DEVICE` 表示 CPU 已经写好数据，映射时执行 **clean**（写回）操作，把 dirty cache line 刷到内存，确保 device 读到的是最新值；`DMA_FROM_DEVICE` 表示 device 将写入数据，unmap 或 `sync_for_cpu` 时执行 **invalidate** 操作，丢弃 CPU cache 中可能存在的旧行，避免 CPU 读到"看似有效、实际是 stale 数据"的 cache 行；`DMA_BIDIRECTIONAL` 两个方向都做，开销最大，只在确实需要时使用。方向填错是流式 DMA 最隐蔽的 bug 之一：数据不会立刻损坏，而是表现为偶发的、与负载和时序相关的数值异常。以一次典型的"发送"为例，流式映射的完整生命周期是：

```c
/* 驱动已经持有 buf，大小 len，方向为 TO_DEVICE */
dma_addr_t dma = dma_map_single(dev, buf, len, DMA_TO_DEVICE);
if (dma_mapping_error(dev, dma)) {
    /* 处理映射失败，可能因为 DMA mask 不覆盖该缓冲区 */
    return -EIO;
}

/* 将 dma 写入 device 的地址寄存器，启动传输 */
writel(dma, dev_base + TX_ADDR);
writel(len, dev_base + TX_LEN);
writel(START, dev_base + TX_CTRL);

/* 等待传输完成中断…… */

/* 中断到来，传输结束，解除映射，CPU 重新获得所有权 */
dma_unmap_single(dev, dma, len, DMA_TO_DEVICE);
```

&emsp;&emsp;如果需要在传输中途由 CPU 检查缓冲区（例如接收路径上先读 header 再决定是否整包上送），则在读取前插入 `dma_sync_single_for_cpu()`，读完后视情况用 `dma_sync_single_for_device()` 交还 device。注意 `dma_rmb()` 之类的内存屏障解决的是 CPU 与 device 之间的**指令重排和可见性顺序**问题，它不替代 cache 失效操作；两者解决的是不同层面的问题，不能互相替代。

&emsp;&emsp;**分散/聚集：`dma_map_sg()` 解决"不连续"问题。** 当缓冲区由多个物理页拼成（`skb`、`bio` 的 bvec、SG 表等）时，`dma_map_single()` 不再适用。`dma_map_sg()` 接收一个 `scatterlist` 数组，返回实际映射的 DMA 段数，device 侧可以通过 `sg_dma_address()` / `sg_dma_len()` 获取每段的 DMA 地址和长度。如果平台配有 IOMMU，多个物理不连续的段还可以被映射为 device 视角下的一段连续 IOVA，进一步简化 device 侧的寻址逻辑。

&emsp;&emsp;**如何在这两种模型之间选择。** 判断标准很直接：**用一致性映射，当且仅当 CPU 和 device 需要在时间上重叠地访问同一块内存**——描述符环、命令队列、硬件与 CPU 共同维护的状态结构，都属于这一范畴。如果缓冲区很大且 CPU 并不频繁访问，一致性映射在 uncached 平台上的访问代价可能超过流式映射的 cache 维护开销，此时应重新评估。**用流式映射处理"交给 device 就不再回头碰"的数据缓冲区**——网络数据包、块设备的 I/O 数据、音视频采样的单次传输，是流式映射的主场；映射的生命周期应当尽可能短，传输结束立即 unmap，把 DMA 地址资源和控制结构占用的窗口降到最小。

&emsp;&emsp;**如果使用 DMA Engine 框架。** 以上是"驱动自己编程 DMA 控制器"的路径。如果 SoC 提供了共享的 DMA 控制器，外设驱动应当通过 DMA Engine 子系统申请通道、准备传输描述符，而不是直接操作控制器寄存器。基本步骤是：`dma_request_chan()` 获取通道，`dmaengine_slave_config()` 配置方向和总线参数，用 `dmaengine_prep_slave_sg()` 或 `dmaengine_prep_dma_cyclic()` 准备描述符，`dmaengine_submit()` 提交，`dma_async_issue_pending()` 启动。DMA Engine 的价值在于把"谁需要搬运数据"和"谁负责搬运数据"解耦，使外设驱动不绑定特定 DMA 控制器。不过，**DMA Engine 管理的是传输的编排，缓冲区本身仍然需要按前述两种方式创建和映射**——描述符环通常用 coherent 内存，数据缓冲区通常用流式映射。

&emsp;&emsp;**连续内存从哪里来。** `dma_alloc_coherent()` 以及底层 `dma_map_single()` 的成功，最终依赖于 device 能够访问到的物理连续内存（或在 IOMMU 辅助下的 IOVA 连续）。系统运行一段时间后，物理内存碎片化是常态，直接分配大块连续内存很容易失败。**CMA（Contiguous Memory Allocator）** 是内核提供的机制：启动时预留一段区域，平时可供可移动页面使用，当 DMA 分配需要连续内存时，内核把该区域中的可移动页面迁移出去，腾出连续空间。`dma_alloc_coherent()` 在支持 CMA 的平台上会自动使用它，驱动代码不需要修改。CMA 的大小通过内核命令行参数 `cma=` 配置，但预留过大会浪费内存，预留过小则大块分配会失败。

&emsp;&emsp;**图 3：CMA 把预留区中的可移动页迁出，腾出物理连续的 DMA 块**

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261029_16cd07.svg)

&emsp;&emsp;**IOMMU** 从另一个方向解决问题：它允许 DMA 使用虚拟地址（IOVA），由 IOMMU 把多个物理上不连续的页面映射为 device 视角下连续的一段地址。对于支持 SG 的设备，`dma_map_sg()` 配合 IOMMU 是处理大缓冲区的首选方式。在没有 IOMMU 且设备只支持 32 位地址的 64 位系统上，CMA 还可以通过 `cma=64M@0-4G` 的形式把内存限制在低 4GB 范围内。

&emsp;&emsp;**图 4：IOMMU 把物理不连续的页，映射为 device 视角下连续的 IOVA**

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261029_44c5ec.svg)

&emsp;&emsp;**大缓冲区优先考虑 SG + IOMMU，而不是硬扛 CMA。** 当 `dma_alloc_coherent()` 因连续内存不足而失败时，加 CMA 往往只能暂时缓解碎片化；如果设备支持分散/聚集，`dma_map_sg()` 配合 IOMMU 是更具扩展性的方案。

---

## 2. GPU API 如何使用 DMA

&emsp;&emsp;将视线从内核的 `dma_map_single()` 抬升到用户态 GPU API，会发现一个根本性的断层：Vulkan、OpenCL、CUDA、OpenGL **都不直接暴露内核的 `dma_addr_t` 或 `dma_map_*` 接口**。它们各自封装了一套内存对象模型，DMA 的使用方式被隐藏在这些对象和标志位之后。下面逐一看每个 API 通过什么机制把"用户态缓冲区"变成"DMA 引擎可访问的内存"，以及它们之间如何通过 **dma-buf** 这个 Linux 内核载体实现跨 API 的零拷贝共享。

&emsp;&emsp;**两种内核映射模型在上层 API 中的对应关系。** 第 2 节讨论的一致性映射与流式映射，与上层 API 的使用方式存在清晰的对应：**一致性映射 ↔ 单次映射、长期驻留的控制结构**——Vulkan 的 `VkDeviceMemory`、CUDA 的 device 端本地内存，底层通常由 GPU 驱动以 coherent 或 device-local 内存池管理，生命周期长，CPU 只在提交命令时短暂访问；**流式映射 ↔ 短生命周期、单方向搬运的数据缓冲区**——CUDA 的 `cudaMemcpyAsync`、OpenGL 的 PBO 上传、OpenCL 的 enqueue read/write 操作，在驱动内部往往以流式映射实现，传输开始前 map，传输完成后立即 unmap，CPU 只在两端处理数据。理解这一点，就能在遇到上层 API 的性能问题时，判断它究竟是"不该有 CPU 拷贝的地方产生了拷贝"（例如 OpenCL 没加 `CL_MEM_ALLOC_HOST_PTR`、CUDA 用了可分页内存），还是"不该有的同步 flush 发生了"（例如映射窗口内被驱动迫做了 cache 维护）。

&emsp;&emsp;**图 5：四个上层 API 各自封装 DMA 内存对象，dma-buf 是它们之间共享的底层 fd 载体**

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/deepseek_svg_20261029_28c63c.svg)

### 2.1 Vulkan：external memory fd 是入口

&emsp;&emsp;Vulkan 的内存模型是显式的 `VkDeviceMemory`。要让这块内存由 DMA 引擎或外部设备访问，入口是 `VK_EXT_external_memory_dma_buf`（配合 `VK_KHR_external_memory_fd` 使用）。导入一个 dma-buf fd 到 Vulkan 资源，需要经过"查属性 → 声明外部句柄类型 → 导入内存 → 绑定资源"四步：

```c
/* 1. 拿到 dma-buf fd（来自解码器、摄像头或其他生产者） */
int dma_fd = producer_get_dmabuf_fd();

/* 2. 查询该 fd 允许的 memoryTypeBits */
VkMemoryFdPropertiesKHR fd_props = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
};
vkGetMemoryFdPropertiesKHR(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                           dma_fd, &fd_props);

/* 3. 创建 image 时声明外部内存句柄类型 */
VkExternalMemoryImageCreateInfo ext_info = {
    .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
};
VkImageCreateInfo img_info = {
    .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    .pNext       = &ext_info,
    .imageType   = VK_IMAGE_TYPE_2D,
    .format      = VK_FORMAT_B8G8R8A8_UNORM,
    .extent      = { WIDTH, HEIGHT, 1 },
    .usage       = VK_IMAGE_USAGE_SAMPLED_BIT,
};
vkCreateImage(device, &img_info, NULL, &image);

/* 4. 用 fd 导入内存，并绑定到 image */
VkImportMemoryFdInfoKHR import_info = {
    .sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
    .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    .fd         = dma_fd,
};
VkMemoryAllocateInfo alloc_info = {
    .sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .pNext          = &import_info,
    .allocationSize = size,
    .memoryTypeIndex = pick_memory_type(fd_props.memoryTypeBits),
};
vkAllocateMemory(device, &alloc_info, NULL, &memory);
vkBindImageMemory(device, image, memory, 0);

/* dma_fd 在导入后即可 close，Vulkan 内部已持有引用 */
close(dma_fd);
```

&emsp;&emsp;反向导出也很对称：分配内存时声明导出句柄类型（`VkExportMemoryAllocateInfo`），分配完成后用 `vkGetMemoryFdKHR()` 拿 fd。关键约束是：**导出/导入时必须在创建信息中提前声明外部内存标志和句柄类型，事后无法追加**。

### 2.2 OpenCL：固定内存与外部内存两条路

&emsp;&emsp;OpenCL 使用 DMA 有两条路径，对应两种不同的目标。

&emsp;&emsp;**路径一：`CL_MEM_ALLOC_HOST_PTR`——让 host 侧缓冲区进入固定内存。** 默认的 `clCreateBuffer` 可能分配可分页内存，GPU DMA 引擎无法直接访问，传输前需要驱动做一次 staging copy。加上 `CL_MEM_ALLOC_HOST_PTR` 后，驱动会把缓冲区放在**页锁定（pinned）内存**中，DMA 引擎可以直接从这块 host 内存读写，配合 `clEnqueueMapBuffer` / `clEnqueueUnmapMemObject` 使用，消除 staging 开销：

```c
/* 分配 pinned 缓冲区 */
cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            size, NULL, &err);

/* map 到用户空间——底层就是一段固定的 host 内存 */
void *p = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_WRITE,
                             0, size, 0, NULL, NULL, &err);
memcpy(p, src, size);
clEnqueueUnmapMemObject(queue, buf, p, 0, NULL, NULL);

/* 后续 kernel 读写 cl_mem 时，DMA 引擎直接访问这块固定内存 */
clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
```

&emsp;&emsp;**路径二：`cl_khr_external_memory_dma_buf`——导入 dma-buf 实现跨 API 零拷贝。** 如果缓冲区来自 Vulkan、VAAPI 解码器或其他 dma-buf 生产者，可以用 `clCreateBufferWithProperties()`，属性列表中传入 `CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR` 和对应的 `dma_fd`。这样 `cl_mem` 直接引用同一块底层内存，不产生任何拷贝：

```c
cl_mem_properties props[] = {
    CL_MEM_DEVICE_HANDLE_LIST_KHR,  /* 指定设备 */
    (cl_mem_properties)device,
    CL_MEM_DEVICE_HANDLE_LIST_END_KHR,
    CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR, (cl_mem_properties)dma_fd,
    0
};

cl_mem buf = clCreateBufferWithProperties(ctx, props, CL_MEM_READ_WRITE,
                                          size, NULL, &err);
/* buf 与 dma_fd 指向同一块内存，无需再 memcpy */
```

### 2.3 CUDA：pinned memory 是 DMA 的前提

&emsp;&emsp;CUDA 的 DMA 使用逻辑最简洁，也最严格：**只有页锁定（pinned / page-locked）的 host 内存才能被 GPU DMA 引擎直接访问**。普通 `malloc()` 得到的可分页内存，`cudaMemcpy` 在底层必须先拷贝到一个临时的 pinned staging buffer，再由 DMA 搬运，这使异步传输名存实亡。正确做法是全程使用 `cudaHostAlloc()` 分配 pinned 内存，配合 `cudaMemcpyAsync()` 和 CUDA Stream 发起真正的异步 DMA：

```c
float *h_data;
cudaHostAlloc(&h_data, bytes, cudaHostAllocDefault); /* pinned */

float *d_data;
cudaMalloc(&d_data, bytes);

/* H2D：真正的异步 DMA，不经过 staging buffer */
cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream);

kernel<<<grid, block, 0, stream>>>(d_data);
cudaMemcpyAsync(h_data, d_data, bytes, cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);

cudaFree(d_data);
cudaFreeHost(h_data);
```

&emsp;&emsp;`cudaHostAlloc` 的三个常用标志：`cudaHostAllocDefault` 是默认；`cudaHostAllocWriteCombined` 在 x86 上绕过 cache 嗅探，适合"CPU 只写一次、GPU 只读"的 H2D 数据；`cudaHostAllocMapped` 允许 GPU 通过 `cudaHostGetDevicePointer()` 拿到 device 侧指针，在 kernel 中直接访问 host 内存（零拷贝），但受 PCIe 带宽限制。CUDA 的世界里没有像 dma-buf 那样公开的跨 API fd 句柄，它的 pinned memory 是 CUDA 运行时管理的私有资源，跨 API 共享需要借助 `cudaExternalMemory` 等显式互操作接口，通常比 Linux 原生 dma-buf 路径更繁琐。

### 2.4 OpenGL：PBO 与 EGLImage 两条路径

&emsp;&emsp;OpenGL 的传统 `glTexImage2D` 上传路径是同步的，且可能经历"CPU 内存 → 驱动临时缓冲 → GPU 显存"的多段拷贝。使用 DMA 有两种方式。

&emsp;&emsp;**PBO（Pixel Buffer Object）路径**是 OpenGL 内部的 DMA 优化，把数据先写入 PBO（通常是 pinned 内存），再让 GPU 通过 DMA 从 PBO 拉取：

```c
/* 一次性分配 PBO，长期复用 */
GLuint pbo;
glGenBuffers(1, &pbo);
glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
glBufferData(GL_PIXEL_UNPACK_BUFFER, bytes, NULL, GL_STREAM_DRAW);

/* 每帧：map PBO → CPU 写入 → unmap → 从 PBO 上传 */
void *p = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, bytes,
                           GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
memcpy(p, src, bytes);
glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

/* 最后一个参数传 0，表示"从 PBO 读"，GPU 通过 DMA 拉取 */
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0,
             GL_RGBA, GL_UNSIGNED_BYTE, 0);
```

&emsp;&emsp;`GL_MAP_INVALIDATE_BUFFER_BIT` 告知驱动丢弃 PBO 旧内容，避免隐式同步 flush。**EGLImage + dma-buf 路径**用于跨 API 零拷贝：拿到 dma-buf fd 后，用 `eglCreateImageKHR()` 配合 `EGL_LINUX_DMA_BUF_EXT` 创建 EGLImage，再用 `glEGLImageTargetTexture2DOES()` 把它绑定为 GL 纹理：

```c
/* 1. 定义 dma-buf 的平面属性 */
EGLint attrs[] = {
    EGL_WIDTH, w,
    EGL_HEIGHT, h,
    EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_ARGB8888,
    EGL_DMA_BUF_PLANE0_FD_EXT, dma_fd,
    EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
    EGL_DMA_BUF_PLANE0_PITCH_EXT, pitch,
    EGL_NONE
};

/* 2. 从 dma-buf fd 创建 EGLImage */
EGLImageKHR img = eglCreateImageKHR(egl_dpy, EGL_NO_CONTEXT,
                                    EGL_LINUX_DMA_BUF_EXT, NULL, attrs);

/* 3. 绑定为 GL 纹理，后续直接渲染，无 CPU 拷贝 */
glGenTextures(1, &tex);
glBindTexture(GL_TEXTURE_2D, tex);
glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, img);
```

&emsp;&emsp;这是 Linux 上视频解码器、摄像头输出直接进入 OpenGL 渲染管线的标准方式，全程不经过 CPU 拷贝。

### 2.5 横向对比与选择准则

&emsp;&emsp;把四个 API 放在一起，DMA 使用的核心差异不在"有没有 DMA"，而在"谁负责把缓冲区变成 DMA 可访问的形态"：

| API | DMA 使能机制 | 跨 API 共享载体 | 显式同步需求 |
|---|---|---|---|
| **Vulkan** | `VkDeviceMemory` + external memory fd | dma-buf fd | Vulkan semaphore / fence |
| **OpenCL** | `CL_MEM_ALLOC_HOST_PTR`（固定内存）或 `cl_khr_external_memory_dma_buf` | dma-buf fd | cl_event / 命令队列 |
| **CUDA** | `cudaHostAlloc` 分配 pinned memory | 无原生 dma-buf fd（需专用互操作接口） | Stream / event |
| **OpenGL** | PBO 内部 DMA，或 `EGL_EXT_image_dma_buf_import` | dma-buf fd（经 EGLImage） | `glFenceSync` / EGL 同步对象 |

&emsp;&emsp;**如果数据只在单一 GPU API 内部流转**，优先用该 API 原生机制：CUDA 用 `cudaHostAlloc` + `cudaMemcpyAsync`，OpenCL 用 `CL_MEM_ALLOC_HOST_PTR`，OpenGL 用 PBO，Vulkan 则按设备本地内存 + staging buffer 的常规模式处理。此时内核的 DMA 映射由驱动在幕后完成，开发者不需要触碰 `dma_map_*`。

&emsp;&emsp;**如果数据需要跨 API 零拷贝**（解码器 → Vulkan 渲染、Vulkan 计算 → OpenCL 后处理、摄像头 → OpenGL），统一走 **dma-buf fd**——Vulkan 的 `VK_EXT_external_memory_dma_buf`、OpenCL 的 `cl_khr_external_memory_dma_buf`、OpenGL 的 `EGL_EXT_image_dma_buf_import` 都是同一个 dma-buf 的不同消费者接口；CUDA 在这个生态里是相对封闭的一环，跨 API 共享往往需要额外的互操作层。

---

## 3. 实测性能分析

&emsp;&emsp;为了验证上述理论，在一个典型的桌面环境下进行了微基准测试。测试平台搭载 NVIDIA GeForce RTX 3050（CUDA 12.4 环境），运行在 PCIe 4.0 x8 总线上，宿主机 CPU 为 AMD Ryzen 5 5600X，系统内核版本 7.0.0。测试对比了各 API 的 DMA 路径与传统内存拷贝（copy/upload）路径在 4MB 和 16MB 数据量下的传输耗时与有效带宽。数据无可辩驳地显示了 DMA 路径在传输带宽与延迟上的绝对优势。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/202610291219_dma_fig_cuda_sync.svg)

&emsp;&emsp;**CUDA：pinned memory 与异步流缺一不可。** 首先看 CUDA 的测试结果。在 H2D（Host to Device）方向，4MB 和 16MB 下的 Pinned（直接 DMA）和 Pageable（可分页内存）差距似乎不大，甚至在 4MB 的同步传输中 Pageable 略占优势（15.7 GB/s vs 13.0 GB/s），这是因为小数据量下 CPU 的 memcpy 缓存效应掩盖了 staging buffer 的开销。但在 **D2H（Device to Host）方向**，Pinned 直接 DMA 的优势彻底显现：4MB 时 Pinned 保持了 12.9 GB/s，而 Pageable 跌至 9.7 GB/s。这在 Roundtrip（往返）场景中更为明显——Pinned 内存搭配异步 Stream，其往返耗时（0.68ms）远低于可分页内存的 0.89ms。更关键的是，异步（Async）传输的样本方差（图表中的误差棒）极小，证明了 DMA 配合 Stream 能够提供极其稳定的流水线延迟。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/202610291219_dma_fig_cuda.svg)

&emsp;&emsp;**OpenGL 与 Vulkan：EGLImage/dma-buf 带来 2~3 倍加速。** 在图形 API 侧，我们通过 `gbm` 分配 dma-buf，并对比了导入 vs 传统上传。结果非常震撼：在 OpenGL 中，4MB 纹理的 `transfer+process` 耗时，DMA 路径仅需 0.30ms（等效带宽 13.86 GB/s），而传统 copy 路径需要 0.95ms（带宽 4.42 GB/s）；当数据增大到 16MB 时，DMA 路径耗时 0.99ms（带宽 16.85 GB/s），copy 路径则暴涨到 3.19ms（带宽 5.25 GB/s）。在延迟表现上，DMA 路径几乎等同于纯 GPU 处理时间，说明导入的 dma-buf 完全消除了 CPU 介入的中间缓冲。Vulkan 也呈现了完全一致的趋势，16MB 数据下，DMA 路径带宽 15.83 GB/s，copy 路径仅 7.09 GB/s。从加速比图表（speedup）可以看出，Vulkan 通过 dma-buf 导入获得了约 2.2 倍的加速，而 OpenGL 在 EGLImage 的加持下甚至达到了 3 倍以上的加速。

```cpp
EGLImage image = imp.glEGLImageTargetTexture2DOES(display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, nullptr, attribs.data());
  if (image == EGL_NO_IMAGE_KHR) throw std::runtime_error("eglCreateImage(dma_buf): " + egl_error_text());

  GLuint tex = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  imp.target_texture(GL_TEXTURE_2D, image);
  GLenum err = glGetError();
  if (err != GL_NO_ERROR) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "glEGLImageTargetTexture2DOES failed: 0x%04x", err);
    throw std::runtime_error(buf);
  }
  return tex;
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/202610291219_dma_fig_gl.svg)

```cpp
Image import_dmabuf_image(const Ctx& ctx, const GbmBuffer& gbm, uint32_t w, uint32_t h) {
  VkPhysicalDeviceImageDrmFormatModifierInfoEXT mod_info{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT};
  mod_info.drmFormatModifier = gbm.modifier;
  mod_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkPhysicalDeviceExternalImageFormatInfo ext_info{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
  ext_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
  ext_info.pNext = &mod_info;

  VkPhysicalDeviceImageFormatInfo2 fmt_info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
  fmt_info.format = VK_FORMAT_R8G8B8A8_UNORM;
  fmt_info.type = VK_IMAGE_TYPE_2D;
  fmt_info.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
  fmt_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  fmt_info.pNext = &ext_info;

  VkExternalImageFormatProperties ext_props{VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
  VkImageFormatProperties2 props{VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
  props.pNext = &ext_props;
  VkResult q = vkGetPhysicalDeviceImageFormatProperties2(ctx.phys, &fmt_info, &props);
  if (q != VK_SUCCESS) {
    std::fprintf(stderr, "vk: modifier=0x%llx stride=%d offset=%d query failed (%s)\n",
                 static_cast<unsigned long long>(gbm.modifier), gbm.stride, gbm.offset,
                 vk_result_name(q));
    throw std::runtime_error(std::string("drm modifier not queryable: ") + vk_result_name(q));
  }
  if ((ext_props.externalMemoryProperties.compatibleHandleTypes &
       VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) == 0) {
    throw std::runtime_error("dma-buf handle type not compatible with this modifier");
  }

  Image img;
  img.width = w;
  img.height = h;

  VkExternalMemoryImageCreateInfo ext_ci{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
  ext_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

  uint64_t modifier = gbm.modifier;
  VkImageDrmFormatModifierListCreateInfoEXT mod_ci{
      VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT};
  mod_ci.drmFormatModifierCount = 1;
  mod_ci.pDrmFormatModifiers = &modifier;
  ext_ci.pNext = &mod_ci;

  VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ci.pNext = &ext_ci;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = img.format;
  ci.extent = {w, h, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
  ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  check(vkCreateImage(ctx.device, &ci, nullptr, &img.image), "vkCreateImage(external)");

  VkMemoryRequirements req{};
  vkGetImageMemoryRequirements(ctx.device, img.image, &req);

  VkMemoryFdPropertiesKHR fd_props{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
  check(ctx.pfn_get_memory_fd_props(ctx.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, gbm.fd,
                                    &fd_props),
        "vkGetMemoryFdPropertiesKHR");

  const uint32_t compatible = req.memoryTypeBits & fd_props.memoryTypeBits;
  if (compatible == 0) throw std::runtime_error("no memory type accepts the dma-buf fd");

  VkImportMemoryFdInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
  import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
  import.fd = gbm.fd;

  VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
  dedicated.image = img.image;
  import.pNext = &dedicated;

  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.pNext = &import;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = ctx.memory_type(compatible, 0);
  check(vkAllocateMemory(ctx.device, &ai, nullptr, &img.memory), "vkAllocateMemory(import)");
  check(vkBindImageMemory(ctx.device, img.image, img.memory, 0), "vkBindImageMemory(import)");

  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = img.image;
  vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vi.format = img.format;
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  check(vkCreateImageView(ctx.device, &vi, nullptr, &img.view), "vkCreateImageView");
  return img;
}
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/202610291219_dma_fig_vulkan.svg)

&emsp;&emsp;这组数据完美印证了第 2 节的核心结论：**DMA Buffer 的本质是消除中间拷贝（staging buffer）与冗余的 Cache 维护**。一旦跨越了页锁定或 dma-buf 的边界，让硬件 DMA 引擎直接访存，无论上层封装如何复杂，底层的性能回报都是数量级的。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/202610291219_dma_fig_summary.svg)

---

## 4. 总结

&emsp;&emsp;DMA Buffer 的正确性取决于是否严格遵守所选映射模型的配对和所有权规则；性能取决于是否把两种模型放在各自擅长的场景中使用。把这两件事分开对待，DMA Buffer 的创建和使用就不再是一团"玄学 cache 问题"，而是一组可以逐项验证的工程决策。

&emsp;&emsp;无论上层 API 如何封装——Vulkan 的 external memory fd、OpenCL 的 pinned memory、CUDA 的 `cudaHostAlloc`、OpenGL 的 PBO/EGLImage——底层 DMA 的正确性仍然取决于**映射配对、方向正确、cache 同步时机**。实测数据证明，上层 API 不会替你修复违反所有权模型的用法，但只要你正确触发了 DMA 路径，硬件就会用极低的延迟和接近 PCIe 线速的带宽给予回报。它只是把内核的 `dma_addr_t` 换了一个名字交到你手里。