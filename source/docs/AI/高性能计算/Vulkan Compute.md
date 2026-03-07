# Vulkan Compute 完全指南：GPU通用计算入门

## 前言

&emsp;&emsp;在现代GPU编程领域，Vulkan不仅仅是一款图形API，更是一把打开GPU通用计算大门的钥匙。作为Compute领域的初学者，你可能会问：为什么要学习Vulkan Compute？它与其他GPU计算方案（如CUDA、OpenCL）有何不同？

&emsp;&emsp;答案是：Vulkan Compute具有**跨平台**、**低开销**、**现代API设计**等优势。更重要的是，Vulkan将计算支持作为**强制特性**——这意味着任何支持Vulkan的设备都能运行Compute Shader，这一点是CUDA和OpenCL都无法比拟的。

&emsp;&emsp;本文将带你系统性地学习Vulkan Compute，从基础概念到实践代码，帮助你建立起完整的GPU计算知识体系。

---

## 第一章：计算基本概念

### 1.1 什么是GPGPU？

&emsp;&emsp;GPGPU（General-Purpose Graphics Processing Unit，通用图形处理器）是指利用GPU进行图形渲染以外的一般性计算任务的技术。

&emsp;&emsp;传统上，GPU专门用于处理图形任务（如顶点变换、像素着色），但现代GPU拥有数千个小型计算核心，非常适合处理**数据并行**（Data Parallel）的任务——即对大量数据执行相同操作。

&emsp;&emsp;**典型应用场景**：
- 粒子系统模拟
- 图像/视频处理
- 机器学习推理
- 物理模拟
- 加密货币计算
- 路径追踪渲染

### 1.2 Vulkan Compute的独特优势

```
&emsp;&emsp;┌─────────────────────────────────────────────────────────────┐
&emsp;&emsp;│                      Vulkan vs 其他方案                      │
&emsp;&emsp;├──────────────────┬──────────────┬──────────────┬───────────┤
&emsp;&emsp;│     特性          │   Vulkan     │    CUDA      │  OpenCL   │
&emsp;&emsp;├──────────────────┼──────────────┼──────────────┼───────────┤
&emsp;&emsp;│  跨平台支持        │ ✓ (桌面/移动/  │ ✗ (NVIDIA)   │    ✓      │
&emsp;&emsp;│                  │   嵌入式)      │              │           │
&emsp;&emsp;├──────────────────┼──────────────┼──────────────┼───────────┤
&emsp;&emsp;│  计算强制支持       │ ✓            │    ✓         │    ✓      │
&emsp;&emsp;├──────────────────┼──────────────┼──────────────┼───────────┤
&emsp;&emsp;│  API设计现代性     │ ✓ (显式API)  │   中等        │   中等     │
&emsp;&emsp;├──────────────────┼──────────────┼──────────────┼───────────┤
&emsp;&emsp;│  与图形API统一     │ ✓            │    ✗        │    ✗      │
&emsp;&emsp;└──────────────────┴──────────────┴──────────────┴───────────┘
```

&emsp;&emsp;**Vulkan Compute的核心优势**：

&emsp;&emsp;1. **强制性支持**：任何实现Vulkan的设备都必须支持计算功能
&emsp;&emsp;2. **统一性**：计算与图形使用同一套API体系，可以无缝结合
&emsp;&emsp;3. **显式控制**：低级别API给予开发者充分的控制权
&emsp;&emsp;4. **跨平台**：从桌面到移动设备，从Windows到Linux，甚至是macOS（通过MoltenVK）

---

## 第二章：Vulkan Compute架构解析

### 2.1 Vulkan Pipeline整体架构

&emsp;&emsp;在深入了解Compute之前，我们需要理解Vulkan的Pipeline概念。

```
&emsp;&emsp;┌──────────────────────────────────────────────────────────────────────┐
&emsp;&emsp;│                           Vulkan Pipeline                             │
&emsp;&emsp;├────────────────────────────────────┬───────────────────────────────────┤
&emsp;&emsp;│        图形管线 (Graphics)          │         计算管线 (Compute)         │
&emsp;&emsp;├────────────────────────────────────┼───────────────────────────────────┤
&emsp;&emsp;│  ┌─────────────┐                   │                                   │
&emsp;&emsp;│  │  顶点着色器   │ ─► 顶点处理       │                                   │
&emsp;&emsp;│  └─────────────┘                   │                                   │
&emsp;&emsp;│  ┌─────────────┐                   │       ┌─────────────┐            │
&emsp;&emsp;│  │  几何着色器   │ ─► 图元处理       │       │  计算着色器   │ ◄── 唯一   │
&emsp;&emsp;│  └─────────────┘                   │       │ Compute Shader│    编程阶段  │
&emsp;&emsp;│  ┌─────────────┐                   │       └─────────────┘            │
&emsp;&emsp;│  │  片段着色器   │ ─► 像素处理       │                                   │
&emsp;&emsp;│  └─────────────┘                   │                                   │
&emsp;&emsp;├────────────────────────────────────┴───────────────────────────────────┤
&emsp;&emsp;│                    共享资源：Descriptor Sets                           │
&emsp;&emsp;│                    共享资源：Pipeline Layout                          │
&emsp;&emsp;└──────────────────────────────────────────────────────────────────────┘
```

&emsp;&emsp;**关键观察**：
- **计算管线极度简化**：相比图形管线，计算管线只有一个可编程阶段——Compute Shader
- **资源共享**：Descriptor Sets和Pipeline Layout在两种管线间共享
- **完全解耦**：Compute Shader可以独立于图形管线运行（Headless Compute）

### 2.2 Compute Shader的工作原理

&emsp;&emsp;Compute Shader（计算着色器）是GPU上运行的特殊程序，它：

&emsp;&emsp;1. **无输入装配**：不像图形管线有顶点缓冲区，Compute Shader直接通过API参数获取工作范围
&emsp;&emsp;2. **几何化执行模型**：通过`dispatch`命令启动，以3D网格形式组织
&emsp;&emsp;3. **灵活的数据访问**：可以读写任意内存位置

```
&emsp;&emsp;                    Dispatch (3, 2, 1)
&emsp;&emsp;                         │
&emsp;&emsp;                         ▼
&emsp;&emsp;    ┌─────────────────────────────────────────┐
&emsp;&emsp;    │         WorkGroup Grid (3 × 2 × 1)       │
&emsp;&emsp;    │  ┌─────────┐ ┌─────────┐ ┌─────────┐    │
&emsp;&emsp;    │  │ WG(0,0,0)│ │WG(1,0,0)│ │WG(2,0,0)│    │
&emsp;&emsp;    │  ├─────────┤ ├─────────┤ ├─────────┤    │
&emsp;&emsp;    │  │ WG(0,1,0)│ │WG(1,1,0)│ │WG(2,1,0)│    │
&emsp;&emsp;    │  └─────────┘ └─────────┘ └─────────┘    │
&emsp;&emsp;    └─────────────────────────────────────────┘
&emsp;&emsp;                         │
&emsp;&emsp;                         ▼
&emsp;&emsp;    ┌─────────────────────────────────────────┐
&emsp;&emsp;    │      WorkGroup (8 × 8 × 1) = 64 Threads  │
&emsp;&emsp;    │  ┌──┬──┬──┬──┬──┬──┬──┬──┐               │
&emsp;&emsp;    │  │  │  │  │  │  │  │  │  │  │               │
&emsp;&emsp;    │  ├──┼──┼──┼──┼──┼──┼──┼──┤               │
&emsp;&emsp;    │  │  │  │  │  │  │  │  │  │  │               │
&emsp;&emsp;    │  ├──┼──┼──┼──┼──┼──┼──┼──┤               │
&emsp;&emsp;    │  │  │  │  │  │  │  │  │  │  │               │
&emsp;&emsp;    │  ├──┼──┼──┼──┼──┼──┼──┼──┤               │
&emsp;&emsp;    │  │  │  │  │  │  │  │  │  │  │               │
&emsp;&emsp;    │  ├──┼──┼──┼──┼──┼──┼──┼──┤               │
&emsp;&emsp;    │  │  │  │  │  │  │  │  │  │  │               │
&emsp;&emsp;    │  └──┴──┴──┴──┴──┴──┴──┴──┘               │
&emsp;&emsp;    └─────────────────────────────────────────┘
```

### 2.3 线程标识系统

&emsp;&emsp;Compute Shader使用一套完整的内置变量来标识当前执行的线程：

| 内置变量 | 类型 | 描述 |
|---------|------|------|
| `gl_WorkGroupID` | `uvec3` | 当前WorkGroup在dispatch网格中的索引 |
| `gl_LocalInvocationID` | `uvec3` | 当前线程在WorkGroup内的局部索引 |
| `gl_GlobalInvocationID` | `uvec3` | 全局唯一索引 = WorkGroupID × WorkGroupSize + LocalInvocationID |
| `gl_LocalInvocationIndex` | `uint` | 一维化的LocalInvocationID |
| `gl_NumWorkGroups` | `uvec3` | dispatch命令指定的总WorkGroup数量 |

&emsp;&emsp;**重要公式**：

```
&emsp;&emsp;gl_GlobalInvocationID.x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x
&emsp;&emsp;gl_LocalInvocationIndex = gl_LocalInvocationID.z * gl_WorkGroupSize.x * gl_WorkGroupSize.y 
&emsp;&emsp;                        + gl_LocalInvocationID.y * gl_WorkGroupSize.x 
&emsp;&emsp;                        + gl_LocalInvocationID.x
```

### 2.4 GPU硬件架构基础

&emsp;&emsp;理解Compute Shader需要了解GPU的基本架构：

```
&emsp;&emsp;                    GPU 架构示意图
&emsp;&emsp;    ┌────────────────────────────────────────────┐
&emsp;&emsp;    │                   GPU                        │
&emsp;&emsp;    │  ┌──────────────────────────────────────┐   │
&emsp;&emsp;    │  │         Compute Unit (CU)            │   │
&emsp;&emsp;    │  │  ┌────────────────────────────────┐  │   │
&emsp;&emsp;    │  │  │     Shader Processor (SP)      │  │   │
&emsp;&emsp;    │  │  │   ┌─────────────────────────┐  │  │   │
&emsp;&emsp;    │  │  │   │  ALU  ALU  ALU  ALU     │  │  │   │
&emsp;&emsp;    │  │  │   │  (多个算术逻辑单元并行)   │  │  │   │
&emsp;&emsp;    │  │  │   └─────────────────────────┘  │  │   │
&emsp;&emsp;    │  │  │   ┌─────────────────────────┐  │  │   │
&emsp;&emsp;    │  │  │   │   Register File        │  │  │   │
&emsp;&emsp;    │  │  │   │   (高速寄存器)           │  │  │   │
&emsp;&emsp;    │  │  │   └─────────────────────────┘  │  │   │
&emsp;&emsp;    │  │  │   ┌─────────────────────────┐  │  │   │
&emsp;&emsp;    │  │  │   │   Shared Memory        │  │  │   │
&emsp;&emsp;    │  │  │   │   (WorkGroup内共享)     │  │  │   │
&emsp;&emsp;    │  │  │   └─────────────────────────┘  │  │   │
&emsp;&emsp;    │  │  └────────────────────────────────┘  │   │
&emsp;&emsp;    │  └──────────────────────────────────────┘   │
&emsp;&emsp;    │              ... 更多CU ...                  │
&emsp;&emsp;    └────────────────────────────────────────────┘
```

&emsp;&emsp;**SIMT vs SIMD**：
- **SIMT (Single Instruction Multiple Threads)**：单指令多线程，NVIDIA和AMD GPU使用的模型
- 每个线程独立执行，但同一Warp（32个线程）内的线程同步执行同一指令
- 线程通过分支会引发分支分化，导致性能下降

---

## 第三章：Vulkan Compute核心组件

### 3.1 队列与队列族

&emsp;&emsp;Vulkan使用**队列族**（Queue Family）来组织不同类型的操作：

```cpp
&emsp;&emsp;// 队列族类型
&emsp;&emsp;enum VkQueueFlagBits {
&emsp;&emsp;    VK_QUEUE_GRAPHICS_BIT,      // 图形操作
&emsp;&emsp;    VK_QUEUE_COMPUTE_BIT,       // 计算操作
&emsp;&emsp;    VK_QUEUE_TRANSFER_BIT,      // 传输操作
&emsp;&emsp;    VK_QUEUE_SPARSE_BINDING_BIT,// 稀疏内存绑定
&emsp;&emsp;    VK_QUEUE_PROTECTED_BIT,     // 受保护内存
&emsp;&emsp;};
```

&emsp;&emsp;**查找支持Compute的队列**：

```cpp
&emsp;&emsp;uint32_t findComputeQueueFamily(const VkPhysicalDevice& physicalDevice) {
&emsp;&emsp;    uint32_t queueFamilyCount = 0;
&emsp;&emsp;    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    
&emsp;&emsp;    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
&emsp;&emsp;    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    
&emsp;&emsp;    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
&emsp;&emsp;        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
&emsp;&emsp;            return i;  // 找到支持Compute的队列族
&emsp;&emsp;        }
&emsp;&emsp;    }
    
&emsp;&emsp;    return -1;  // 未找到
&emsp;&emsp;}
```

&emsp;&emsp;**重要说明**：
- Vulkan要求支持图形的设备**必须**至少有一个支持Compute的队列族
- 可以使用单独的Compute队列，也可以使用支持Graphics+Compute的共享队列

### 3.2 描述符与描述符集

&emsp;&emsp;**描述符**是Vulkan中连接CPU端资源（缓冲区、图像）与GPU端着色器的桥梁。

#### 描述符类型

| 类型 | 用途 | 典型使用场景 |
|------|------|-------------|
| `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` | 均匀缓冲区（只读，小数据） | 变换矩阵、配置参数 |
| `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` | 存储缓冲区（读写，大数据） | 粒子数据、计算结果 |
| `VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER` | 均匀纹理缓冲区 | 只读数据数组 |
| `VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER` | 存储纹理缓冲区 | 读写数据数组 |
| `VK_DESCRIPTOR_TYPE_STORAGE_IMAGE` | 存储图像 | 图像处理输出 |
| `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` | 组合图像采样器 | 纹理采样 |

#### Compute Shader常用的描述符

&emsp;&emsp;对于Compute Shader，我们主要关注两类描述符：

&emsp;&emsp;1. **Storage Buffer (存储缓冲区)**：读写任意数据
   ```glsl
&emsp;&emsp;   layout(binding = 0, std430) buffer InputBuffer {
&emsp;&emsp;       float data[];
&emsp;&emsp;   } input;
   
&emsp;&emsp;   layout(binding = 1, std430) buffer OutputBuffer {
&emsp;&emsp;       float results[];
&emsp;&emsp;   } output;
   ```

&emsp;&emsp;2. **Storage Image (存储图像)**：读写图像数据
   ```glsl
&emsp;&emsp;   layout(binding = 0, rgba8) uniform readonly image2D inputImage;
&emsp;&emsp;   layout(binding = 1, rgba8) uniform writeonly image2D outputImage;
   ```

### 3.3 描述符集布局

```cpp
&emsp;&emsp;// 创建描述符集布局 - 用于Compute Shader
&emsp;&emsp;std::array<VkDescriptorSetLayoutBinding, 2> layoutBindings = {};

&emsp;&emsp;// 绑定0：输入存储缓冲区
&emsp;&emsp;layoutBindings[0].binding = 0;
&emsp;&emsp;layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
&emsp;&emsp;layoutBindings[0].descriptorCount = 1;
&emsp;&emsp;layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

&emsp;&emsp;// 绑定1：输出存储缓冲区
&emsp;&emsp;layoutBindings[1].binding = 1;
&emsp;&emsp;layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
&emsp;&emsp;layoutBindings[1].descriptorCount = 1;
&emsp;&emsp;layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

&emsp;&emsp;VkDescriptorSetLayoutCreateInfo layoutInfo = {};
&emsp;&emsp;layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
&emsp;&emsp;layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
&emsp;&emsp;layoutInfo.pBindings = layoutBindings.data();

&emsp;&emsp;vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
```

### 3.4 描述符池与描述符集分配

```cpp
&emsp;&emsp;// 创建描述符池
&emsp;&emsp;VkDescriptorPoolSize poolSize = {};
&emsp;&emsp;poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
&emsp;&emsp;poolSize.descriptorCount = 2;  // 2个缓冲区描述符

&emsp;&emsp;VkDescriptorPoolCreateInfo poolInfo = {};
&emsp;&emsp;poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
&emsp;&emsp;poolInfo.poolSizeCount = 1;
&emsp;&emsp;poolInfo.pPoolSizes = &poolSize;
&emsp;&emsp;poolInfo.maxSets = 1;  // 分配1个描述符集

&emsp;&emsp;vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);

&emsp;&emsp;// 分配描述符集
&emsp;&emsp;VkDescriptorSetAllocateInfo allocInfo = {};
&emsp;&emsp;allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
&emsp;&emsp;allocInfo.descriptorPool = descriptorPool;
&emsp;&emsp;allocInfo.descriptorSetCount = 1;
&emsp;&emsp;allocInfo.pSetLayouts = &descriptorSetLayout;

&emsp;&emsp;vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);
```

### 3.5 更新描述符集

```cpp
&emsp;&emsp;// 准备缓冲区描述符信息
&emsp;&emsp;VkDescriptorBufferInfo bufferInfo = {};
&emsp;&emsp;bufferInfo.buffer = storageBuffer;
&emsp;&emsp;bufferInfo.offset = 0;
&emsp;&emsp;bufferInfo.range = VK_WHOLE_SIZE;

&emsp;&emsp;// 写入描述符集
&emsp;&emsp;VkWriteDescriptorSet descriptorWrite = {};
&emsp;&emsp;descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
&emsp;&emsp;descriptorWrite.dstSet = descriptorSet;
&emsp;&emsp;descriptorWrite.dstBinding = 0;  // 绑定点
&emsp;&emsp;descriptorWrite.dstArrayElement = 0;
&emsp;&emsp;descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
&emsp;&emsp;descriptorWrite.descriptorCount = 1;
&emsp;&emsp;descriptorWrite.pBufferInfo = &bufferInfo;

&emsp;&emsp;vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
```

---

## 第四章：Compute Pipeline详解

### 4.1 Pipeline基本概念

&emsp;&emsp;**Pipeline**（管线）是Vulkan中定义一系列处理阶段的抽象。与图形管线相比，Compute管线极为简洁：

```
&emsp;&emsp;Graphics Pipeline:                    Compute Pipeline:
&emsp;&emsp;┌─────────────────┐                  ┌─────────────────┐
&emsp;&emsp;│ Input Assembler │                  │                 │
&emsp;&emsp;│    (顶点输入)     │                  │                 │
&emsp;&emsp;└────────┬────────┘                  │                 │
&emsp;&emsp;         ▼                           │                 │
&emsp;&emsp;┌─────────────────┐                  │                 │
&emsp;&emsp;│  Vertex Shader  │ ◄── 可编程        │                 │
&emsp;&emsp;│    (顶点着色器)   │                  │                 │
&emsp;&emsp;└────────┬────────┘                  │                 │
&emsp;&emsp;         ▼                           │                 │
&emsp;&emsp;┌─────────────────┐                  │                 │
&emsp;&emsp;│Geometry Shader  │ ◄── 可编程        │                 │
&emsp;&emsp;│   (几何着色器)   │                  │                 │
&emsp;&emsp;└────────┬────────┘                  │                 │
&emsp;&emsp;         ▼                           │                 │
&emsp;&emsp;┌─────────────────┐                  │                 │
&emsp;&emsp;│  Rasterizer     │ ◄── 固定功能      │                 │
&emsp;&emsp;│    (光栅化)      │                  │                 │
&emsp;&emsp;└────────┬────────┘                  │                 │
&emsp;&emsp;         ▼                           │                 │
&emsp;&emsp;┌─────────────────┐                  │                 │
&emsp;&emsp;│Fragment Shader  │ ◄── 可编程        │                 │
&emsp;&emsp;│   (片段着色器)   │                  └────────┬────────┘
&emsp;&emsp;└─────────────────┘                           │
&emsp;&emsp;         │                           ┌────────▼────────┐
&emsp;&emsp;         ▼                           │ Compute Shader  │ ◄── 唯一可编程阶段
&emsp;&emsp;┌─────────────────┐                  │   (计算着色器)   │
&emsp;&emsp;│  Color Blend    │                  └─────────────────┘
&emsp;&emsp;│    (颜色混合)   │
&emsp;&emsp;└─────────────────┘
```

### 4.2 创建Compute Pipeline

```cpp
&emsp;&emsp;// 1. 创建Shader Module（从SPIR-V加载）
&emsp;&emsp;std::vector<char> computeShaderCode = readFile("compute.spv");

&emsp;&emsp;VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
&emsp;&emsp;shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
&emsp;&emsp;shaderModuleCreateInfo.codeSize = computeShaderCode.size();
&emsp;&emsp;shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(computeShaderCode.data());

&emsp;&emsp;VkShaderModule computeShaderModule;
&emsp;&emsp;vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &computeShaderModule);

&emsp;&emsp;// 2. 创建Pipeline Shader Stage
&emsp;&emsp;VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
&emsp;&emsp;shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
&emsp;&emsp;shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;  // 关键：Compute阶段
&emsp;&emsp;shaderStageCreateInfo.module = computeShaderModule;
&emsp;&emsp;shaderStageCreateInfo.pName = "main";  // 入口函数名

&emsp;&emsp;// 3. 创建Pipeline Layout
&emsp;&emsp;VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
&emsp;&emsp;pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
&emsp;&emsp;pipelineLayoutCreateInfo.setLayoutCount = 1;
&emsp;&emsp;pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;

&emsp;&emsp;VkPipelineLayout pipelineLayout;
&emsp;&emsp;vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);

&emsp;&emsp;// 4. 创建Compute Pipeline
&emsp;&emsp;VkComputePipelineCreateInfo computePipelineCreateInfo = {};
&emsp;&emsp;computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
&emsp;&emsp;computePipelineCreateInfo.stage = shaderStageCreateInfo;
&emsp;&emsp;computePipelineCreateInfo.layout = pipelineLayout;

&emsp;&emsp;VkPipeline computePipeline;
&emsp;&emsp;vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, 
&emsp;&emsp;                        &computePipelineCreateInfo, nullptr, 
&emsp;&emsp;                        &computePipeline);
```

---

## 第五章：Compute Shader编程

### 5.1 GLSL Compute Shader基础

&emsp;&emsp;Compute Shader使用GLSL编写，需要指定版本声明：

```glsl
#version 460  // GLSL版本，建议使用460或更高
```

#### 工作组大小声明

```glsl
&emsp;&emsp;// 方式1：硬编码
&emsp;&emsp;layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

&emsp;&emsp;// 方式2：使用Specialization Constant
&emsp;&emsp;layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
```

#### 完整示例：向量平方计算

```glsl
#version 460

&emsp;&emsp;// 定义输入输出存储缓冲区
&emsp;&emsp;layout(set = 0, binding = 0) readonly buffer InputBuffer {
&emsp;&emsp;    float inputData[];
&emsp;&emsp;};

&emsp;&emsp;layout(set = 0, binding = 1) buffer OutputBuffer {
&emsp;&emsp;    float outputData[];
&emsp;&emsp;};

&emsp;&emsp;// Push Constants
&emsp;&emsp;layout(push_constant) uniform PushConstants {
&emsp;&emsp;    uint count;      // 要处理的元素数量
&emsp;&emsp;    float scale;     // 缩放因子
&emsp;&emsp;} pc;

&emsp;&emsp;// 设置工作组大小
&emsp;&emsp;layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

&emsp;&emsp;void main() {
&emsp;&emsp;    // 获取全局线程ID
&emsp;&emsp;    uint index = gl_GlobalInvocationID.x;
    
&emsp;&emsp;    // 边界检查
&emsp;&emsp;    if (index >= pc.count) {
&emsp;&emsp;        return;
&emsp;&emsp;    }
    
&emsp;&emsp;    // 执行计算：输出 = 输入² × 缩放因子
&emsp;&emsp;    outputData[index] = inputData[index] * inputData[index] * pc.scale;
&emsp;&emsp;}
```

### 5.2 图像处理示例

```glsl
#version 460

&emsp;&emsp;// 输入输出图像（只读输入，只写输出）
&emsp;&emsp;layout(set = 0, binding = 0, rgba8) uniform readonly image2D inputImage;
&emsp;&emsp;layout(set = 0, binding = 1, rgba8) uniform writeonly image2D outputImage;

&emsp;&emsp;// 图像尺寸uniform
&emsp;&emsp;layout(set = 0, binding = 2) uniform ImageInfo {
&emsp;&emsp;    ivec2 size;
&emsp;&emsp;} imageInfo;

&emsp;&emsp;layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

&emsp;&emsp;void main() {
&emsp;&emsp;    // 获取当前线程对应的像素坐标
&emsp;&emsp;    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
    
&emsp;&emsp;    // 边界检查
&emsp;&emsp;    if (pixelCoord.x >= imageInfo.size.x || 
&emsp;&emsp;        pixelCoord.y >= imageInfo.size.y) {
&emsp;&emsp;        return;
&emsp;&emsp;    }
    
&emsp;&emsp;    // 读取像素
&emsp;&emsp;    vec4 color = imageLoad(inputImage, pixelCoord);
    
&emsp;&emsp;    // 简单的亮度调整
&emsp;&emsp;    float luminance = dot(color.rgb, vec3(0.299, 0.587, 0.114));
&emsp;&emsp;    vec4 result = vec4(vec3(luminance), color.a);
    
&emsp;&emsp;    // 写入输出
&emsp;&emsp;    imageStore(outputImage, pixelCoord, result);
&emsp;&emsp;}
```

### 5.3 粒子系统模拟示例

```glsl
#version 460

&emsp;&emsp;// 粒子结构体
&emsp;&emsp;struct Particle {
&emsp;&emsp;    vec2 position;
&emsp;&emsp;    vec2 velocity;
&emsp;&emsp;    float life;  // 生命值，0表示死亡
&emsp;&emsp;};

&emsp;&emsp;// 输入粒子（只读）
&emsp;&emsp;layout(set = 0, binding = 0) readonly buffer ParticleBuffer {
&emsp;&emsp;    Particle particles[];
&emsp;&emsp;};

&emsp;&emsp;// 输出粒子（可写）
&emsp;&emsp;layout(set = 0, binding = 1) buffer ParticleOutputBuffer {
&emsp;&emsp;    Particle outParticles[];
&emsp;&emsp;};

&emsp;&emsp;layout(push_constant) uniform SimParams {
&emsp;&emsp;    float deltaTime;
&emsp;&emsp;    vec2 gravity;
&emsp;&emsp;    vec2 boundsMin;
&emsp;&emsp;    vec2 boundsMax;
&emsp;&emsp;} params;

&emsp;&emsp;layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

&emsp;&emsp;void main() {
&emsp;&emsp;    uint index = gl_GlobalInvocationID.x;
&emsp;&emsp;    Particle p = particles[index];
    
&emsp;&emsp;    if (p.life <= 0.0) {
&emsp;&emsp;        outParticles[index] = p;
&emsp;&emsp;        return;
&emsp;&emsp;    }
    
&emsp;&emsp;    // 更新速度（应用重力）
&emsp;&emsp;    p.velocity += params.gravity * params.deltaTime;
    
&emsp;&emsp;    // 更新位置
&emsp;&emsp;    p.position += p.velocity * params.deltaTime;
    
&emsp;&emsp;    // 边界检查与反弹
&emsp;&emsp;    if (p.position.x < params.boundsMin.x) {
&emsp;&emsp;        p.position.x = params.boundsMin.x;
&emsp;&emsp;        p.velocity.x = -p.velocity.x * 0.8;
&emsp;&emsp;    }
&emsp;&emsp;    if (p.position.x > params.boundsMax.x) {
&emsp;&emsp;        p.position.x = params.boundsMax.x;
&emsp;&emsp;        p.velocity.x = -p.velocity.x * 0.8;
&emsp;&emsp;    }
&emsp;&emsp;    if (p.position.y < params.boundsMin.y) {
&emsp;&emsp;        p.position.y = params.boundsMin.y;
&emsp;&emsp;        p.velocity.y = -p.velocity.y * 0.8;
&emsp;&emsp;    }
&emsp;&emsp;    if (p.position.y > params.boundsMax.y) {
&emsp;&emsp;        p.position.y = params.boundsMax.y;
&emsp;&emsp;        p.velocity.y = -p.velocity.y * 0.8;
&emsp;&emsp;    }
    
&emsp;&emsp;    // 减少生命值
&emsp;&emsp;    p.life -= params.deltaTime;
    
&emsp;&emsp;    // 写入输出
&emsp;&emsp;    outParticles[index] = p;
&emsp;&emsp;}
```

### 5.4 Shared Memory与线程同步

```glsl
#version 460

&emsp;&emsp;// 共享内存 - 在WorkGroup内线程间共享
&emsp;&emsp;layout(local_size_x = 256) in;

&emsp;&emsp;// 共享内存声明（必须在函数外部）
&emsp;&emsp;shared vec4 sharedData[256];

&emsp;&emsp;layout(set = 0, binding = 0) buffer InputBuffer {
&emsp;&emsp;    float data[];
&emsp;&emsp;};

&emsp;&emsp;layout(set = 0, binding = 1) buffer OutputBuffer {
&emsp;&emsp;    float results[];
&emsp;&emsp;};

&emsp;&emsp;// Push Constants
&emsp;&emsp;layout(push_constant) uniform PC {
&emsp;&emsp;    uint count;
&emsp;&emsp;} pc;

&emsp;&emsp;void main() {
&emsp;&emsp;    uint localId = gl_LocalInvocationID.x;
    
&emsp;&emsp;    // 每个线程加载一个数据到共享内存
&emsp;&emsp;    uint globalId = gl_GlobalInvocationID.x;
&emsp;&emsp;    if (globalId < pc.count) {
&emsp;&emsp;        sharedData[localId] = vec4(data[globalId], 0.0, 0.0, 0.0);
&emsp;&emsp;    }
    
&emsp;&emsp;    // 同步！确保所有线程都完成数据加载
&emsp;&emsp;    // barrier() - 所有线程必须到达此点
&emsp;&emsp;    // memoryBarrierShared() - 确保共享内存访问同步
&emsp;&emsp;    barrier();
&emsp;&emsp;    memoryBarrierShared();
    
&emsp;&emsp;    // 现在可以使用共享内存进行计算
&emsp;&emsp;    // 例如：并行归约
&emsp;&emsp;    // ... (复杂的并行算法)
    
&emsp;&emsp;    // 将结果写回
&emsp;&emsp;    if (globalId < pc.count) {
&emsp;&emsp;        results[globalId] = sharedData[localId].x;
&emsp;&emsp;    }
&emsp;&emsp;}
```

---

## 第六章：执行Compute Shader

### 6.1 命令记录与Dispatch

```cpp
&emsp;&emsp;// 1. 创建命令池和命令缓冲区
&emsp;&emsp;VkCommandPool commandPool;
&emsp;&emsp;VkCommandPoolCreateInfo poolInfo = {};
&emsp;&emsp;poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
&emsp;&emsp;poolInfo.queueFamilyIndex = computeQueueFamilyIndex;
&emsp;&emsp;poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

&emsp;&emsp;vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);

&emsp;&emsp;VkCommandBuffer commandBuffer;
&emsp;&emsp;VkCommandBufferAllocateInfo allocInfo = {};
&emsp;&emsp;allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
&emsp;&emsp;allocInfo.commandPool = commandPool;
&emsp;&emsp;allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
&emsp;&emsp;allocInfo.commandBufferCount = 1;

&emsp;&emsp;vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

&emsp;&emsp;// 2. 开始记录命令
&emsp;&emsp;VkCommandBufferBeginInfo beginInfo = {};
&emsp;&emsp;beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

&emsp;&emsp;vkBeginCommandBuffer(commandBuffer, &beginInfo);

&emsp;&emsp;// 3. 绑定Pipeline和Descriptor Sets
&emsp;&emsp;vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
&emsp;&emsp;vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, 
&emsp;&emsp;                        pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

&emsp;&emsp;// 4. 设置Push Constants（如果需要）
&emsp;&emsp;struct PushConstants {
&emsp;&emsp;    uint32_t count;
&emsp;&emsp;    float scale;
&emsp;&emsp;} pc = {1024, 2.0f};

&emsp;&emsp;vkCmdPushConstants(commandBuffer, pipelineLayout, 
&emsp;&emsp;                  VK_SHADER_STAGE_COMPUTE_BIT, 
&emsp;&emsp;                  0, sizeof(PushConstants), &pc);

&emsp;&emsp;// 5. Dispatch执行
&emsp;&emsp;// 格式：vkCmdDispatch(commandBuffer, x, y, z)
&emsp;&emsp;// - x × y × z = 总WorkGroup数量
&emsp;&emsp;uint32_t workGroupCountX = (1024 + 255) / 256;  // 4
&emsp;&emsp;uint32_t workGroupCountY = 1;
&emsp;&emsp;uint32_t workGroupCountZ = 1;

&emsp;&emsp;vkCmdDispatch(commandBuffer, workGroupCountX, workGroupCountY, workGroupCountZ);

&emsp;&emsp;// 6. 结束记录
&emsp;&emsp;vkEndCommandBuffer(commandBuffer);
```

### 6.2 队列提交与同步

```cpp
&emsp;&emsp;// 1. 获取Compute队列
&emsp;&emsp;VkQueue computeQueue;
&emsp;&emsp;vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);

&emsp;&emsp;// 2. 提交命令到队列
&emsp;&emsp;VkSubmitInfo submitInfo = {};
&emsp;&emsp;submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
&emsp;&emsp;submitInfo.commandBufferCount = 1;
&emsp;&emsp;submitInfo.pCommandBuffers = &commandBuffer;

&emsp;&emsp;vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);

&emsp;&emsp;// 3. 等待完成
&emsp;&emsp;vkQueueWaitIdle(computeQueue);

&emsp;&emsp;// 或者使用Fence进行更精细的控制
&emsp;&emsp;VkFence fence;
&emsp;&emsp;VkFenceCreateInfo fenceInfo = {};
&emsp;&emsp;fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
&emsp;&emsp;vkCreateFence(device, &fenceInfo, nullptr, &fence);

&emsp;&emsp;vkQueueSubmit(computeQueue, 1, &submitInfo, fence);

&emsp;&emsp;// ... 做其他事情 ...

&emsp;&emsp;// 等待完成
&emsp;&emsp;vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
```

### 6.3 完整的工作组数量计算

```cpp
&emsp;&emsp;// 计算Dispatch参数
&emsp;&emsp;struct ComputeDispatchParams {
&emsp;&emsp;    uint32_t totalItemCount;
&emsp;&emsp;    uint32_t workGroupSizeX;
&emsp;&emsp;    uint32_t workGroupSizeY;
&emsp;&emsp;    uint32_t workGroupSizeZ;
&emsp;&emsp;};

&emsp;&emsp;ComputeDispatchParams calculateDispatchParams(uint32_t totalItems) {
&emsp;&emsp;    ComputeDispatchParams params = {};
    
&emsp;&emsp;    // 假设Shader中声明 local_size_x = 256
&emsp;&emsp;    params.workGroupSizeX = 256;
&emsp;&emsp;    params.workGroupSizeY = 1;
&emsp;&emsp;    params.workGroupSizeZ = 1;
    
&emsp;&emsp;    // 计算需要的WorkGroup数量（向上取整）
&emsp;&emsp;    uint32_t workGroupCountX = (totalItems + params.workGroupSizeX - 1) 
&emsp;&emsp;                               / params.workGroupSizeX;
    
&emsp;&emsp;    params.totalItemCount = totalItems;
    
&emsp;&emsp;    return params;
&emsp;&emsp;}

&emsp;&emsp;// 使用示例
&emsp;&emsp;auto params = calculateDispatchParams(1024);
&emsp;&emsp;vkCmdDispatch(commandBuffer, 
&emsp;&emsp;              (params.totalItemCount + 255) / 256,  // 4
&emsp;&emsp;              1, 
&emsp;&emsp;              1);
```

---

## 第七章：内存管理与数据传输

### 7.1 Vulkan内存模型

```
&emsp;&emsp;┌─────────────────────────────────────────────────────────────┐
&emsp;&emsp;│                      Vulkan 内存模型                         │
&emsp;&emsp;├─────────────────────────────────────────────────────────────┤
&emsp;&emsp;│                                                             │
&emsp;&emsp;│   CPU端                              GPU端                  │
&emsp;&emsp;│  ┌─────────────┐                  ┌─────────────┐          │
&emsp;&emsp;│  │ Host Memory │                  │Device Memory│          │
&emsp;&emsp;│  │  (系统内存)  │◄── Pinned Mem   │  (VRAM)     │          │
&emsp;&emsp;│  │             │    CPU访问        │             │          │
&emsp;&emsp;│  │             │◄── Mapped Mem    │             │          │
&emsp;&emsp;│  └──────┬──────┘                  └──────┬──────┘          │
&emsp;&emsp;│         │                                │                  │
&emsp;&emsp;│         │ vkMapMemory                    │                  │
&emsp;&emsp;│         ├────────────────────────────────┤                  │
&emsp;&emsp;│         │     Buffer/Image 对象          │                  │
&emsp;&emsp;│         │  (虚拟地址，GPU可见)            │                  │
&emsp;&emsp;│         │                                │                  │
&emsp;&emsp;│  ┌──────▼──────┐                  ┌──────▼──────┐          │
&emsp;&emsp;│  │  Host       │                  │  Device     │          │
&emsp;&emsp;│  │  Visible   │                  │  Local      │          │
&emsp;&emsp;│  │  + Host    │                  │  (最快)     │          │
&emsp;&emsp;│  │  Coherent  │                  │             │          │
&emsp;&emsp;│  └─────────────┘                  └─────────────┘          │
&emsp;&emsp;│                                                             │
&emsp;&emsp;│  内存类型：                                                  │
&emsp;&emsp;│  - VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT    CPU可见          │
&emsp;&emsp;│  - VK_MEMORY_PROPERTY_HOST_COHERENT_BIT   CPU一致性         │
&emsp;&emsp;│  - VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT    GPU本地(最快)      │
&emsp;&emsp;│  - VK_MEMORY_PROPERTY_HOST_CACHED_BIT     CPU缓存           │
&emsp;&emsp;│                                                             │
&emsp;&emsp;└─────────────────────────────────────────────────────────────┘
```

### 7.2 创建存储缓冲区

```cpp
&emsp;&emsp;// 缓冲区大小（以字节为单位）
&emsp;&emsp;VkDeviceSize bufferSize = sizeof(float) * 1024;

&emsp;&emsp;// 创建缓冲区
&emsp;&emsp;VkBufferCreateInfo bufferInfo = {};
&emsp;&emsp;bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
&emsp;&emsp;bufferInfo.size = bufferSize;
&emsp;&emsp;bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;  // 关键：存储缓冲区用途
&emsp;&emsp;bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

&emsp;&emsp;vkCreateBuffer(device, &bufferInfo, nullptr, &storageBuffer);

&emsp;&emsp;// 查询内存需求
&emsp;&emsp;VkMemoryRequirements memRequirements;
&emsp;&emsp;vkGetBufferMemoryRequirements(device, storageBuffer, &memRequirements);

&emsp;&emsp;// 查找合适的内存类型
&emsp;&emsp;uint32_t memoryTypeIndex = findMemoryType(
&emsp;&emsp;    physicalDevice,
&emsp;&emsp;    memRequirements.memoryTypeBits,
&emsp;&emsp;    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
&emsp;&emsp;);

&emsp;&emsp;// 分配内存
&emsp;&emsp;VkMemoryAllocateInfo allocInfo = {};
&emsp;&emsp;allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
&emsp;&emsp;allocInfo.allocationSize = memRequirements.size;
&emsp;&emsp;allocInfo.memoryTypeIndex = memoryTypeIndex;

&emsp;&emsp;vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);

&emsp;&emsp;// 绑定内存
&emsp;&emsp;vkBindBufferMemory(device, storageBuffer, bufferMemory, 0);

&emsp;&emsp;// 映射内存（如果是Host Visible）
&emsp;&emsp;void* mappedMemory;
&emsp;&emsp;vkMapMemory(device, bufferMemory, 0, bufferSize, 0, &mappedMemory);

&emsp;&emsp;// 写入数据
&emsp;&emsp;float* data = static_cast<float*>(mappedMemory);
&emsp;&emsp;for (int i = 0; i < 1024; i++) {
&emsp;&emsp;    data[i] = static_cast<float>(i);
&emsp;&emsp;}

&emsp;&emsp;// 取消映射（可选，取决于内存类型）
&emsp;&emsp;// vkUnmapMemory(device, bufferMemory);
```

### 7.3 数据同步与屏障

&emsp;&emsp;Compute Shader中的数据同步是至关重要的：

```cpp
&emsp;&emsp;// 在Compute Shader执行后添加内存屏障
&emsp;&emsp;// 确保Compute Shader的写操作对后续操作可见
&emsp;&emsp;VkMemoryBarrier memoryBarrier = {};
&emsp;&emsp;memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
&emsp;&emsp;memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;  // 之前的写操作
&emsp;&emsp;memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;     // 后续的Host读操作

&emsp;&emsp;vkCmdPipelineBarrier(
&emsp;&emsp;    commandBuffer,
&emsp;&emsp;    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,    // 生产者阶段
&emsp;&emsp;    VK_PIPELINE_STAGE_HOST_BIT,               // 消费者阶段
&emsp;&emsp;    0,
&emsp;&emsp;    1, &memoryBarrier,
&emsp;&emsp;    0, nullptr,
&emsp;&emsp;    0, nullptr
&emsp;&emsp;);

&emsp;&emsp;// 如果Compute后面还要进行Graphics操作
&emsp;&emsp;VkMemoryBarrier memoryBarrier2 = {};
&emsp;&emsp;memoryBarrier2.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
&emsp;&emsp;memoryBarrier2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
&emsp;&emsp;memoryBarrier2.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT 
                              | VK_ACCESS_INDEX_READ_BIT;

&emsp;&emsp;vkCmdPipelineBarrier(
&emsp;&emsp;    commandBuffer,
&emsp;&emsp;    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
&emsp;&emsp;    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
&emsp;&emsp;    0,
&emsp;&emsp;    1, &memoryBarrier2,
&emsp;&emsp;    0, nullptr,
&emsp;&emsp;    0, nullptr
&emsp;&emsp;);
```

&emsp;&emsp;**常见的Pipeline Stage和Access Mask组合**：

| 场景 | srcStage | dstStage | srcAccess | dstAccess |
|------|----------|----------|-----------|-----------|
| Compute → Host Read | COMPUTE | HOST | SHADER_WRITE | HOST_READ |
| Compute → Graphics | COMPUTE | VERTEX_INPUT | SHADER_WRITE | VERTEX_ATTRIBUTE_READ |
| Host Write → Compute | HOST | COMPUTE | HOST_WRITE | SHADER_READ |

---

## 第八章：完整的Vulkan Compute示例

### 8.1 示例概述

&emsp;&emsp;让我们创建一个完整的示例：将输入数组的每个元素平方后输出。

&emsp;&emsp;**功能**：对1024个浮点数进行平方计算

### 8.2 C++代码实现

```cpp
#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include <iostream>
#include <cmath>

&emsp;&emsp;class VulkanComputeApp {
&emsp;&emsp;public:
&emsp;&emsp;    void run() {
&emsp;&emsp;        createInstance();
&emsp;&emsp;        createPhysicalDevice();
&emsp;&emsp;        createDevice();
&emsp;&emsp;        createCommandPool();
&emsp;&emsp;        createBuffers();
&emsp;&emsp;        createDescriptorSetLayout();
&emsp;&emsp;        createComputePipeline();
&emsp;&emsp;        createDescriptorPoolAndSet();
&emsp;&emsp;        createCommandBuffer();
&emsp;&emsp;        recordCommandBuffer();
&emsp;&emsp;        execute();
&emsp;&emsp;        verifyResults();
        
&emsp;&emsp;        std::cout << "Vulkan Compute completed successfully!" << std::endl;
&emsp;&emsp;    }

&emsp;&emsp;private:
&emsp;&emsp;    vk::raii::Instance instance{nullptr};
&emsp;&emsp;    vk::raii::PhysicalDevice physicalDevice{nullptr};
&emsp;&emsp;    vk::raii::Device device{nullptr};
&emsp;&emsp;    vk::raii::Queue computeQueue{nullptr};
&emsp;&emsp;    vk::raii::CommandPool commandPool{nullptr};
&emsp;&emsp;    vk::raii::CommandBuffer commandBuffer{nullptr};
&emsp;&emsp;    vk::raii::PipelineLayout pipelineLayout{nullptr};
&emsp;&emsp;    vk::raii::DescriptorSetLayout descriptorSetLayout{nullptr};
&emsp;&emsp;    vk::raii::DescriptorPool descriptorPool{nullptr};
&emsp;&emsp;    vk::raii::DescriptorSet descriptorSet{nullptr};
&emsp;&emsp;    vk::raii::Pipeline computePipeline{nullptr};
    
&emsp;&emsp;    vk::Buffer inputBuffer{nullptr};
&emsp;&emsp;    vk::Buffer outputBuffer{nullptr};
&emsp;&emsp;    vk::DeviceMemory inputMemory{nullptr};
&emsp;&emsp;    vk::DeviceMemory outputMemory{nullptr};
    
&emsp;&emsp;    const uint32_t BUFFER_SIZE = 1024;
    
&emsp;&emsp;    void createInstance() {
&emsp;&emsp;        vk::ApplicationInfo appInfo{
&emsp;&emsp;            "VulkanCompute",
&emsp;&emsp;            VK_MAKE_VERSION(1, 0, 0),
&emsp;&emsp;            "VulkanCompute",
&emsp;&emsp;            VK_MAKE_VERSION(1, 0, 0),
&emsp;&emsp;            VK_API_VERSION_1_3
&emsp;&emsp;        };
        
&emsp;&emsp;        instance = vk::raii::Instance(
&emsp;&emsp;            vk::InstanceCreateInfo{
&emsp;&emsp;                vk::InstanceCreateFlags(),
&emsp;&emsp;                &appInfo
&emsp;&emsp;            }
&emsp;&emsp;        );
&emsp;&emsp;    }
    
&emsp;&emsp;    void createPhysicalDevice() {
&emsp;&emsp;        auto physicalDevices = instance.enumeratePhysicalDevices();
&emsp;&emsp;        physicalDevice = physicalDevices[0];
        
&emsp;&emsp;        auto properties = physicalDevice.getProperties();
&emsp;&emsp;        std::cout << "Using device: " << properties.deviceName << std::endl;
&emsp;&emsp;    }
    
&emsp;&emsp;    void createDevice() {
&emsp;&emsp;        auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
&emsp;&emsp;        uint32_t computeQueueFamilyIndex = 0;
&emsp;&emsp;        for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
&emsp;&emsp;            if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute) {
&emsp;&emsp;                computeQueueFamilyIndex = i;
&emsp;&emsp;                break;
&emsp;&emsp;            }
&emsp;&emsp;        }
        
&emsp;&emsp;        float queuePriority = 1.0f;
&emsp;&emsp;        vk::DeviceQueueCreateInfo queueCreateInfo{
&emsp;&emsp;            vk::DeviceQueueCreateFlags(),
&emsp;&emsp;            computeQueueFamilyIndex,
&emsp;&emsp;            1,
&emsp;&emsp;            &queuePriority
&emsp;&emsp;        };
        
&emsp;&emsp;        vk::PhysicalDeviceVulkan13Features vulkan13Features;
&emsp;&emsp;        vulkan13Features.dynamicRendering = true;
        
&emsp;&emsp;        device = vk::raii::Device(
&emsp;&emsp;            physicalDevice,
&emsp;&emsp;            vk::DeviceCreateInfo{
&emsp;&emsp;                vk::DeviceCreateFlags(),
&emsp;&emsp;                1,
&emsp;&emsp;                &queueCreateInfo,
&emsp;&emsp;                0, nullptr,
&emsp;&emsp;                0, nullptr,
&emsp;&emsp;                &queueCreateInfo,
&emsp;&emsp;                nullptr,
&emsp;&emsp;                &vulkan13Features
&emsp;&emsp;            }
&emsp;&emsp;        );
        
&emsp;&emsp;        computeQueue = device.getQueue(computeQueueFamilyIndex, 0);
&emsp;&emsp;    }
    
&emsp;&emsp;    void createCommandPool() {
&emsp;&emsp;        auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
&emsp;&emsp;        uint32_t computeQueueFamilyIndex = 0;
&emsp;&emsp;        for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
&emsp;&emsp;            if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute) {
&emsp;&emsp;                computeQueueFamilyIndex = i;
&emsp;&emsp;                break;
&emsp;&emsp;            }
&emsp;&emsp;        }
        
&emsp;&emsp;        commandPool = vk::raii::CommandPool(
&emsp;&emsp;            device,
&emsp;&emsp;            vk::CommandPoolCreateInfo{
&emsp;&emsp;                vk::CommandPoolCreateFlags(),
&emsp;&emsp;                computeQueueFamilyIndex
&emsp;&emsp;            }
&emsp;&emsp;        );
&emsp;&emsp;    }
    
&emsp;&emsp;    void createBuffers() {
&emsp;&emsp;        vk::DeviceSize size = sizeof(float) * BUFFER_SIZE;
        
&emsp;&emsp;        inputBuffer = vk::raii::Buffer(
&emsp;&emsp;            device,
&emsp;&emsp;            vk::BufferCreateInfo{
&emsp;&emsp;                vk::BufferCreateFlags(),
&emsp;&emsp;                size,
&emsp;&emsp;                vk::BufferUsageFlagBits::eStorageBuffer
&emsp;&emsp;            }
&emsp;&emsp;        );
        
&emsp;&emsp;        auto memRequirements = inputBuffer.getMemoryRequirements();
&emsp;&emsp;        vk::MemoryAllocateInfo allocInfo{
&emsp;&emsp;            memRequirements.size,
&emsp;&emsp;            findMemoryType(
&emsp;&emsp;                memRequirements.memoryTypeBits,
&emsp;&emsp;                vk::MemoryPropertyFlagBits::eHostVisible | 
&emsp;&emsp;                vk::MemoryPropertyFlagBits::eHostCoherent
&emsp;&emsp;            )
&emsp;&emsp;        };
        
&emsp;&emsp;        inputMemory = vk::raii::DeviceMemory(device, allocInfo);
&emsp;&emsp;        device.bindBufferMemory(*inputBuffer, *inputMemory, 0);
        
&emsp;&emsp;        float* data = static_cast<float*>(device.mapMemory(*inputMemory, 0, size));
&emsp;&emsp;        for (uint32_t i = 0; i < BUFFER_SIZE; i++) {
&emsp;&emsp;            data[i] = static_cast<float>(i);
&emsp;&emsp;        }
&emsp;&emsp;        device.unmapMemory(*inputMemory);
        
&emsp;&emsp;        outputBuffer = vk::raii::Buffer(
&emsp;&emsp;            device,
&emsp;&emsp;            vk::BufferCreateInfo{
&emsp;&emsp;                vk::BufferCreateFlags(),
&emsp;&emsp;                size,
&emsp;&emsp;                vk::BufferUsageFlagBits::eStorageBuffer
&emsp;&emsp;            }
&emsp;&emsp;        );
        
&emsp;&emsp;        memRequirements = outputBuffer.getMemoryRequirements();
&emsp;&emsp;        allocInfo.allocationSize = memRequirements.size;
&emsp;&emsp;        allocInfo.memoryTypeIndex = findMemoryType(
&emsp;&emsp;            memRequirements.memoryTypeBits,
&emsp;&emsp;            vk::MemoryPropertyFlagBits::eHostVisible | 
&emsp;&emsp;            vk::MemoryPropertyFlagBits::eHostCoherent
&emsp;&emsp;        );
        
&emsp;&emsp;        outputMemory = vk::raii::DeviceMemory(device, allocInfo);
&emsp;&emsp;        device.bindBufferMemory(*outputBuffer, *outputMemory, 0);
&emsp;&emsp;    }
    
&emsp;&emsp;    uint32_t findMemoryType(uint32_t typeFilter, 
&emsp;&emsp;                           vk::MemoryPropertyFlags properties) {
&emsp;&emsp;        auto memProperties = physicalDevice.getMemoryProperties();
        
&emsp;&emsp;        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
&emsp;&emsp;            if ((typeFilter & (1 << i)) && 
&emsp;&emsp;                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
&emsp;&emsp;                return i;
&emsp;&emsp;            }
&emsp;&emsp;        }
        
&emsp;&emsp;        throw std::runtime_error("Failed to find suitable memory type!");
&emsp;&emsp;    }
    
&emsp;&emsp;    void createDescriptorSetLayout() {
&emsp;&emsp;        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {};
        
&emsp;&emsp;        bindings[0].binding = 0;
&emsp;&emsp;        bindings[0].descriptorType = vk::DescriptorType::eStorageBuffer;
&emsp;&emsp;        bindings[0].descriptorCount = 1;
&emsp;&emsp;        bindings[0].stageFlags = vk::ShaderStageFlagBits::eCompute;
        
&emsp;&emsp;        bindings[1].binding = 1;
&emsp;&emsp;        bindings[1].descriptorType = vk::DescriptorType::eStorageBuffer;
&emsp;&emsp;        bindings[1].descriptorCount = 1;
&emsp;&emsp;        bindings[1].stageFlags = vk::ShaderStageFlagBits::eCompute;
        
&emsp;&emsp;        descriptorSetLayout = vk::raii::DescriptorSetLayout(
&emsp;&emsp;            device,
&emsp;&emsp;            vk::DescriptorSetLayoutCreateInfo{
&emsp;&emsp;                vk::DescriptorSetLayoutCreateFlags(),
&emsp;&emsp;                static_cast<uint32_t>(bindings.size()),
&emsp;&emsp;                bindings.data()
&emsp;&emsp;            }
&emsp;&emsp;        );
&emsp;&emsp;    }
    
&emsp;&emsp;    void createComputePipeline() {
&emsp;&emsp;        // 实际使用时需要加载SPIR-V文件
&emsp;&emsp;        // 这里需要包含shader加载代码
&emsp;&emsp;    }
    
&emsp;&emsp;    void createDescriptorPoolAndSet() {
&emsp;&emsp;        std::array<vk::DescriptorPoolSize, 1> poolSizes = {};
&emsp;&emsp;        poolSizes[0].type = vk::DescriptorType::eStorageBuffer;
&emsp;&emsp;        poolSizes[0].descriptorCount = 2;
        
&emsp;&emsp;        descriptorPool = vk::raii::DescriptorPool(
&emsp;&emsp;            device,
&emsp;&emsp;            vk::DescriptorPoolCreateInfo{
&emsp;&emsp;                vk::DescriptorPoolCreateFlags(),
&emsp;&emsp;                1,
&emsp;&emsp;                static_cast<uint32_t>(poolSizes.size()),
&emsp;&emsp;                poolSizes.data()
&emsp;&emsp;            }
&emsp;&emsp;        );
        
&emsp;&emsp;        descriptorSet = vk::raii::DescriptorSet(
&emsp;&emsp;            device,
&emsp;&emsp;            vk::DescriptorSetAllocateInfo{
&emsp;&emsp;                *descriptorPool,
&emsp;&emsp;                1,
&emsp;&emsp;                &(*descriptorSetLayout)
&emsp;&emsp;            }
&emsp;&emsp;        );
        
&emsp;&emsp;        vk::DescriptorBufferInfo inputBufferInfo{
&emsp;&emsp;            *inputBuffer, 0, VK_WHOLE_SIZE
&emsp;&emsp;        };
&emsp;&emsp;        vk::DescriptorBufferInfo outputBufferInfo{
&emsp;&emsp;            *outputBuffer, 0, VK_WHOLE_SIZE
&emsp;&emsp;        };
        
&emsp;&emsp;        std::array<vk::WriteDescriptorSet, 2> writes = {};
        
&emsp;&emsp;        writes[0].dstSet = *descriptorSet;
&emsp;&emsp;        writes[0].dstBinding = 0;
&emsp;&emsp;        writes[0].dstArrayElement = 0;
&emsp;&emsp;        writes[0].descriptorType = vk::DescriptorType::eStorageBuffer;
&emsp;&emsp;        writes[0].descriptorCount = 1;
&emsp;&emsp;        writes[0].pBufferInfo = &inputBufferInfo;
        
&emsp;&emsp;        writes[1].dstSet = *descriptorSet;
&emsp;&emsp;        writes[1].dstBinding = 1;
&emsp;&emsp;        writes[1].dstArrayElement = 0;
&emsp;&emsp;        writes[1].descriptorType = vk::DescriptorType::eStorageBuffer;
&emsp;&emsp;        writes[1].descriptorCount = 1;
&emsp;&emsp;        writes[1].pBufferInfo = &outputBufferInfo;
        
&emsp;&emsp;        device.updateDescriptorSets(writes, {});
&emsp;&emsp;    }
    
&emsp;&emsp;    void createCommandBuffer() {
&emsp;&emsp;        commandBuffer = vk::raii::CommandBuffer(
&emsp;&emsp;            device,
&emsp;&emsp;            vk::CommandBufferAllocateInfo{
&emsp;&emsp;                *commandPool,
&emsp;&emsp;                vk::CommandBufferLevel::ePrimary,
&emsp;&emsp;                1
&emsp;&emsp;            }
&emsp;&emsp;        );
&emsp;&emsp;    }
    
&emsp;&emsp;    void recordCommandBuffer() {
&emsp;&emsp;        commandBuffer.begin(vk::CommandBufferBeginInfo{
&emsp;&emsp;            vk::CommandBufferBeginFlags()
&emsp;&emsp;        });
        
&emsp;&emsp;        commandBuffer.bindPipeline(
&emsp;&emsp;            vk::PipelineBindPoint::eCompute,
&emsp;&emsp;            *computePipeline
&emsp;&emsp;        );
        
&emsp;&emsp;        commandBuffer.bindDescriptorSets(
&emsp;&emsp;            vk::PipelineBindPoint::eCompute,
&emsp;&emsp;            *pipelineLayout,
&emsp;&emsp;            0,
&emsp;&emsp;            { *descriptorSet },
&emsp;&emsp;            {}
&emsp;&emsp;        );
        
&emsp;&emsp;        uint32_t groupCount = (BUFFER_SIZE + 255) / 256;
&emsp;&emsp;        commandBuffer.dispatch(groupCount, 1, 1);
        
&emsp;&emsp;        commandBuffer.pipelineBarrier(
&emsp;&emsp;            vk::PipelineStageFlagBits::eComputeShader,
&emsp;&emsp;            vk::PipelineStageFlagBits::eHost,
&emsp;&emsp;            vk::DependencyFlags(),
&emsp;&emsp;            vk::MemoryBarrier{
&emsp;&emsp;                vk::AccessFlagBits::eShaderWrite,
&emsp;&emsp;                vk::AccessFlagBits::eHostRead
&emsp;&emsp;            },
&emsp;&emsp;            {},
&emsp;&emsp;            {}
&emsp;&emsp;        );
        
&emsp;&emsp;        commandBuffer.end();
&emsp;&emsp;    }
    
&emsp;&emsp;    void execute() {
&emsp;&emsp;        computeQueue.submit(
&emsp;&emsp;            vk::SubmitInfo{
&emsp;&emsp;                {},
&emsp;&emsp;                nullptr,
&emsp;&emsp;                nullptr,
&emsp;&emsp;                1,
&emsp;&emsp;                &(*commandBuffer),
&emsp;&emsp;                {},
&emsp;&emsp;                nullptr
&emsp;&emsp;            }
&emsp;&emsp;        );
        
&emsp;&emsp;        computeQueue.waitIdle();
&emsp;&emsp;    }
    
&emsp;&emsp;    void verifyResults() {
&emsp;&emsp;        vk::DeviceSize size = sizeof(float) * BUFFER_SIZE;
&emsp;&emsp;        float* outputData = static_cast<float*>(
&emsp;&emsp;            device.mapMemory(*outputMemory, 0, size)
&emsp;&emsp;        );
        
&emsp;&emsp;        bool success = true;
&emsp;&emsp;        for (uint32_t i = 0; i < BUFFER_SIZE; i++) {
&emsp;&emsp;            float expected = static_cast<float>(i * i);
&emsp;&emsp;            if (std::abs(outputData[i] - expected) > 0.001f) {
&emsp;&emsp;                std::cout << "Error at index " << i 
&emsp;&emsp;                          << ": expected " << expected 
&emsp;&emsp;                          << ", got " << outputData[i] << std::endl;
&emsp;&emsp;                success = false;
&emsp;&emsp;                break;
&emsp;&emsp;            }
&emsp;&emsp;        }
        
&emsp;&emsp;        device.unmapMemory(*outputMemory);
        
&emsp;&emsp;        if (success) {
&emsp;&emsp;            std::cout << "All results verified! Sample: " 
&emsp;&emsp;                      << outputData[10] << " = " << 10 << "²" << std::endl;
&emsp;&emsp;        }
&emsp;&emsp;    }
&emsp;&emsp;};

&emsp;&emsp;int main() {
&emsp;&emsp;    VulkanComputeApp app;
&emsp;&emsp;    app.run();
&emsp;&emsp;    return 0;
&emsp;&emsp;}
```

### 8.3 Compute Shader代码

```glsl
#version 460

&emsp;&emsp;// 输入缓冲区 - 只读
&emsp;&emsp;layout(set = 0, binding = 0) readonly buffer InputBuffer {
&emsp;&emsp;    float inputData[];
&emsp;&emsp;};

&emsp;&emsp;// 输出缓冲区 - 可写
&emsp;&emsp;layout(set = 0, binding = 1) buffer OutputBuffer {
&emsp;&emsp;    float outputData[];
&emsp;&emsp;};

&emsp;&emsp;// 工作组大小：256个线程
&emsp;&emsp;layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

&emsp;&emsp;void main() {
&emsp;&emsp;    // 获取全局线程ID
&emsp;&emsp;    uint index = gl_GlobalInvocationID.x;
    
&emsp;&emsp;    // 获取元素数量
&emsp;&emsp;    uint count = inputData.length();
    
&emsp;&emsp;    // 边界检查
&emsp;&emsp;    if (index >= count) {
&emsp;&emsp;        return;
&emsp;&emsp;    }
    
&emsp;&emsp;    // 执行平方计算
&emsp;&emsp;    outputData[index] = inputData[index] * inputData[index];
&emsp;&emsp;}
```

---

## 第九章：高级主题

### 9.1 Compute与Graphics的结合

```cpp
&emsp;&emsp;commandBuffer.begin(vk::CommandBufferBeginInfo{});

&emsp;&emsp;// 1. 执行Compute更新粒子
&emsp;&emsp;commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);
&emsp;&emsp;commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, 
&emsp;&emsp;                                  computePipelineLayout, 0, 1, &computeDescriptorSet);
&emsp;&emsp;commandBuffer.dispatch(particleCount / 256, 1, 1);

&emsp;&emsp;// 2. 内存屏障
&emsp;&emsp;commandBuffer.pipelineBarrier(
&emsp;&emsp;    vk::PipelineStageFlagBits::eComputeShader,
&emsp;&emsp;    vk::PipelineStageFlagBits::eVertexInput,
&emsp;&emsp;    vk::DependencyFlags(),
&emsp;&emsp;    vk::MemoryBarrier{
&emsp;&emsp;        vk::AccessFlagBits::eShaderWrite,
&emsp;&emsp;        vk::AccessFlagBits::eVertexAttributeRead | vk::AccessFlagBits::eIndexRead
&emsp;&emsp;    },
&emsp;&emsp;    {}, {}
&emsp;&emsp;);

&emsp;&emsp;// 3. 渲染粒子
&emsp;&emsp;commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
&emsp;&emsp;commandBuffer.bindVertexBuffers(0, 1, &particleBuffer, {0});
&emsp;&emsp;commandBuffer.draw(particleCount, 1, 0, 0);

&emsp;&emsp;commandBuffer.end();
```

### 9.2 性能优化建议

&emsp;&emsp;1. **合理设置工作组大小**
   ```cpp
&emsp;&emsp;   VkPhysicalDeviceLimits limits = properties.limits;
&emsp;&emsp;   uint32_t maxWorkGroupSize = limits.maxComputeWorkGroupSize[0];
   ```

&emsp;&emsp;2. **使用Coherent Memory减少同步开销**

&emsp;&emsp;3. **批量提交减少Queue Submit开销**

&emsp;&emsp;4. **使用Fence进行CPU-GPU同步**

---

## 第十章：实践项目建议

### 10.1 推荐学习路线

&emsp;&emsp;1. **入门阶段**：实现简单数组平方计算，理解Descriptor Set和Pipeline
&emsp;&emsp;2. **进阶阶段**：图像处理、粒子系统、Shared Memory
&emsp;&emsp;3. **高级阶段**：矩阵乘法、并行归约、结合Graphics后处理

### 10.2 调试工具

- Vulkan Validation Layers
- RenderDoc
- vulkaninfo

### 10.3 学习资源

- [Vulkan Tutorial - Compute Shader](https://vulkan-tutorial.com/Compute_Shader)
- [Khronos Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples)
- [Vulkan Guide](https://vkguide.dev/)

---

## 总结

&emsp;&emsp;Vulkan Compute为开发者提供了强大而灵活的GPU通用计算能力。通过本文，我们涵盖了：

&emsp;&emsp;1. **基础概念**：GPGPU、Vulkan Compute优势
&emsp;&emsp;2. **架构理解**：Compute Pipeline、Thread ID系统
&emsp;&emsp;3. **核心组件**：队列、描述符、Pipeline
&emsp;&emsp;4. **编程实践**：Compute Shader编写、执行流程
&emsp;&emsp;5. **内存管理**：数据传输、同步机制
&emsp;&emsp;6. **完整示例**：向量平方计算
&emsp;&emsp;7. **高级主题**：Compute与Graphics结合、性能优化

&emsp;&emsp;作为初学者，建议从简单例子开始，逐步深入理解。Vulkan学习曲线较陡，但掌握后将拥有对GPU编程的完整控制能力。

&emsp;&emsp;祝你在Vulkan Compute的学习旅程中有所收获！

---

&emsp;&emsp;*本文会持续更新，如有问题欢迎指正。*
