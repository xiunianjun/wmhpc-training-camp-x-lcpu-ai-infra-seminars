# Assignment 01 书面作答 / 填表题答题稿

> 整理需要书面作答或填表记录的题，方便自己填写答案。
> 包含 `CONCEPT`，以及带表格/问答的 `EXPERIMENT`、`MODIFY`、`FILL-IN` 题。
> 不包含只需要改代码、没有额外书面回答的纯代码题，也不包含纯 `HANDS-ON`、`DEBUG`、`FROM-SCRATCH`、`Bonus`。

## Module 0 环境准备

### prob 0.2 FILL-IN

文件：`cuda/m0_env/02_device_query.cu`

`02_device_query.cu` 的五个空都是 `cudaDeviceProp` 的字段名。对照 CUDA Runtime API 文档补全，然后编译运行：

```bash
cd assignment01/cuda
make run/m0_env/02_device_query
```

把这块卡的参数填进下表，并对照 Guide 附录 Compute Capabilities 中对应架构的表，核对 warp 大小和每 SM 最大线程数。

| 项目 | 你的卡 |
| --- | --- |
| 型号 / compute capability | NVIDIA H200 NVL / 9.0 |
| SM 数量 | 132 |
| warp 大小 | 32 |
| shared memory / block | 49152 B = 48 KiB |
| 最大常驻线程 / SM | 2048 |
| 显存总量 | 150109880320 B ≈ 139.8 GiB |

---

## Module 1 为什么要用 GPU

### prob 1.1 CONCEPT

判断对错，可以顺带补一句理由。

| 小题 | 判断 | 理由 |
| --- | --- | --- |
| 一块标称 100 TFLOPS 的 GPU，执行单条指令的延迟一定低于 5 GHz 的 CPU。 | × | TFLOPS 衡量的是吞吐量，GPU 靠大量并行 lane/warp 同时发射来堆吞吐，单个线程上一条指令的延迟未必比 CPU 低，CPU 反而为低延迟做了很多设计。|
| HBM 的“高带宽”指大块连续访问时的吞吐，零散的随机访问达不到标称值。 | √ | “标称值”指厂商给出的峰值内存带宽，比如几 TB/s。这个数通常假设大规模、连续、合并良好的访问，能让很多 memory transaction 同时在路上。随机访问会导致 coalescing 差、cache 命中差、事务利用率低，所以实际 GB/s 远低于标称峰值。 |
| 严格串行的迭代算法，每步依赖上一步的结果，即使换一块算力更强的 GPU 也快不了多少。 | √ | 发挥不了并行性 |
| “算力 1000 TFLOPS”意味着每次运算的延迟是 `10^-15` 秒。 | × | 1000 TFLOPS 表示每秒总共能完成约 10^15 次浮点操作，顶多只能算出平均延迟 |

### prob 1.2 CONCEPT

Session 1 讲座里提过“N 方过百万”这个例子。总计算量 `10^12` FLOP 在当代 GPU 上的运算时间大概是毫秒级，那为什么一个严格在线的串行算法仍然做不到几秒内跑完？从“延迟”和“吞吐”的角度考虑。

答案：
GPU 的高 TFLOPS 只能提高大量独立操作的总吞吐，串行算法无法并行计算，时延低不了；但可以同时执行多个串行算法提高吞吐量。

### prob 1.3 CONCEPT

补全下表。

| 执行层次 | 软件含义 | 对应硬件 | 直接可用的存储 | 同步与通信手段 |
| --- | --- | --- | --- | --- |
| thread | kernel 的最小执行单位 | 计算单元上的一个 lane | 自己的寄存器 | 自身天然有序 |
| warp | 共同执行的一组 thread，通常 32 个 | SM 上的调度单位 | 各线程寄存器 | `__syncwarp`、shuffle / vote |
| block / CTA | 可协作的一组 thread | 调度到某个 SM 的线程块 | shared memory | `__syncthreads`、shared memory |
| grid | 一次 kernel launch 的全部 blocks | 分布到多个 SM 上执行 | global / constant memory | 普通 kernel 内无全局同步，常用多次 kernel launch |

### prob 1.4 CONCEPT

SIMD 与 SIMT 的区别？

答案：
Single Instruction, Multiple Data  和Single Instruction, Multiple Threads。主要是编程模型不同：前者程序员显式写“向量”，后者程序员写“线程”，硬件把线程组织成类似向量的方式执行。比如说，SIMD 没有 thread divergence 的抽象。

判断正误：NVIDIA GPU 在 Volta 之后每个线程有独立的 program counter，所以 branch divergence 不再有性能代价。

判断与理由：
错误。SIMT在branch divergence时需要按分支分批执行，部分lane会空转。

### prob 1.5 EXPERIMENT

文件：`cuda/m1_why_gpu/01_scaling.cu`

同一个向量加法，四种配置各跑一遍。

```bash
cd assignment01/cuda
make run/m1_why_gpu/01_scaling
```

将数据填入下表。

CPU 单线程      :      9.844 ms  (  2.35 ns/元素)
GPU <<<1, 1>>>  :    241.838 ms  ( 57.66 ns/元素)
GPU <<<1, 256>>>:      5.593 ms  (  1.33 ns/元素)
GPU 铺满 grid   :      0.023 ms  (  0.01 ns/元素, 16384 blocks x 256 threads)

回答：

1. GPU 单线程为什么比 CPU 慢这么多？

   答案：包含 cuda launch
   device 端调度、显存访问等开销，且无高频、乱序执行、更大 cache 等CPU具有的优势。

2. 从单 block 到铺满 grid 的提速，说明 GPU 加速计算靠的是什么？

   答案：多个 block 铺满 grid 后，调度器可以把不同 block 分发到多个 SM 上，于是每个 SM 上也能驻留多个 block / warp，遇到访存延迟时，SM 可以切换去执行别的 warp，计算和内存带宽利用率都更高。

---

## Module 2 第一个 CUDA 程序

### prob 2.2 CONCEPT

为下列五个场景选择正确的修饰符，例如 `__global__` 等。

| 场景 | 修饰符 |
| --- | --- |
| 在 GPU 上执行、由 CPU 侧启动的 kernel 函数。 | `__global__` |
| 只会被 kernel 调用的辅助函数。 | `__device__` |
| host 和 device 代码都要调用的小工具函数。 | `__host__ __device__` |
| 整个 kernel 运行期间不变、所有线程都要读的系数表。 | `__constant__` |
| block 内线程共享的暂存数组。 | `__shared__` |

### prob 2.3 MODIFY

文件：`cuda/m2_first_kernel/02_vector_add_um.cu`

先按原样编译运行一次，记下显式内存管理版本耗时。再按文件头说明改成 unified memory 版，并保持计时窗口不变。

```bash
cd assignment01/cuda
make run/m2_first_kernel/02_vector_add_um
```

记录：

| 版本 | 耗时 |
| --- | --- |
| 显式内存管理 | 80.2 ms |
| unified memory | 56.0 ms |

回答：

1. kernel 启动之后、CPU 读结果之前，为什么必须有一次同步？在原先的版本里这次同步发生在哪个调用里？

   答案：需要确保所有 kernel 执行完成，才能得到正确的结果。cuda memcpy。

2. 对比两版“搬运 + kernel + 读回”的耗时，分析差距的原因。

   答案：
   虽然 UM 可能有同步的页迁移和 page fault 开销，但在本机上 UM 版本更快，是因为显式版在计时窗口内包含两次 H2D 和一次 D2H 的整块 cudaMemcpy，而 UM 版本把数据迁移交给驱动，可以按需、按页处理，存在和计算 overlap 的空间。

### prob 2.4 CONCEPT

判断对错，可以顺带补一句理由。

| 小题 | 判断 | 理由 |
| --- | --- | --- |
| `vectorAdd<<<...>>>(...)` 这条语句返回时，kernel 一定已经执行完毕。 | × |  |
| 同一个 stream 里，`cudaMemcpy`，device 到 host，会等它前面的 kernel 全部完成后才开始拷贝。 | √ |  |
| kernel 内部的非法访存，会在启动语句处同步地报出来。 | × | 只会在cudaGetLastError类似的API表现出来 |

### prob 2.7 MODIFY

文件：`cuda/m2_first_kernel/05_grid_stride.cu`

回答：

1. 这种写法的价值在哪里？

   答案：编程更方便。

2. launch 只有 16384 个线程时，性能上要付出什么代价？

   答案：减少并行度，很多 SM 空闲。

### prob 2.8 EXPERIMENT

文件：`cuda/m2_first_kernel/06_whoami.cu`

运行两三次，观察 16 个 block 打印输出的先后。

```bash
cd assignment01/cuda
make run/m2_first_kernel/06_whoami
```

回答：

1. block 输出顺序由谁决定？

   答案：由 GPU 的 block 调度和运行时输出缓冲决定，不保证按 `blockIdx.x` 顺序。

2. 程序的正确性可以依赖 block 的执行顺序吗？这条限制和 Guide 1.1 说的 scalable programming model 有什么关系？

   答案：不能依赖。CUDA 把 grid 拆成许多相互独立的 block，硬件可以按任意顺序把 block 分配到任意数量的 SM 上执行；这样同一个程序才能在 SM 数不同的 GPU 上扩展运行，这就是 scalable programming model 的核心。

---

## Module 3 SIMT 执行

### prob 3.1 CONCEPT

设 `blockDim = (8, 8, 1)`。

1. `threadIdx = (3, 5, 0)` 的线性编号是多少？它在第几个 warp、warp 内第几个 lane？

   答案：

2. 这个 block 一共占多少个 warp？

   答案：

3. 若 `blockDim = (33, 1, 1)`，占几个 warp？这样配置浪费在哪里？

   答案：

### prob 3.2 EXPERIMENT

文件：`cuda/m3_simt/01_divergence.cu`

两个 kernel 每线程计算量相同，分支划分不同：一个按 thread 编号奇偶分，一个按 warp 边界对齐分。

先预测，再运行验证：

```bash
cd assignment01/cuda
make run/m3_simt/01_divergence
```

| 版本 | 预测耗时 / 相对速度 | 实测耗时 / 相对速度 |
| --- | --- | --- |
| 按 thread 奇偶分 |  |  |
| 按 warp 边界对齐分 |  |  |

回答：

1. 请解释实测比值。

   答案：

2. 若两个分支的计算量一大一小，按 thread 编号奇偶分的 kernel 和按 warp 边界对齐分的 kernel 的运行时间分别由什么决定？

   答案：

### prob 3.3 EXPERIMENT

文件：`cuda/m3_simt/02_sync_matters.cu`

让每个 block 用 shared memory 把自己的 256 个元素倒序。按文件开头注释内容进行实验。

```bash
cd assignment01/cuda
make run/m3_simt/02_sync_matters
```

回答：

1. 为什么注释掉 sync 后代码不能正确地运行？

   答案：

2. Optional：注释掉 sync 后，翻转后的数组错的位置比较随机，但是有些位置一直是对的，试解释原因。

   答案：

### prob 3.4 CONCEPT

`__syncthreads` 只能同步本 block 内的 threads，那需要全 grid 同步时，标准做法是什么？

答案：


---

## Module 4 存储空间

### prob 4.1 CONCEPT

补全下表。

| 空间 | 谁可见 | 生命周期 | 片上 / 片外 | 谁管理 |
| --- | --- | --- | --- | --- |
| register | 单个线程 | 线程 | 片上 | 编译器 |
| local |  |  |  |  |
| shared |  |  |  |  |
| global |  |  |  |  |
| constant |  |  |  |  |
| L1 / L2 cache |  |  |  |  |

### prob 4.3 MODIFY

文件：`cuda/m4_memory/02_constant_coeff.cu`

按文件头说明修改。测试命令：

```bash
cd assignment01/cuda
make run/m4_memory/02_constant_coeff
```

记录：

| 版本 | 耗时 |
| --- | --- |
| 原版本 |  |
| constant memory 版本 |  |

回答：constant cache 真正的优势在哪种访问模式？

答案：


### prob 4.4 CONCEPT

判断对错，可以顺带补一句理由。

| 小题 | 判断 | 理由 |
| --- | --- | --- |
| local memory 的“local”指作用域私有，它实际上在片外显存里。 |  |  |
| 对数组用运行期才知道的下标做索引，可能迫使它被放进 local memory。 |  |  |

### prob 4.6 MODIFY

文件：`cuda/m4_memory/04_histogram_priv.cu`

按要求修改代码。测试命令：

```bash
cd assignment01/cuda
make run/m4_memory/04_histogram_priv
```

记录：

| 版本 | 耗时 | 吞吐 |
| --- | --- | --- |
| naive |  |  |
| privatized |  |  |

回答：试解释提速来自哪里。

答案：


### prob 4.7 EXPERIMENT

文件：`cuda/m4_memory/05_bandwidth.cu`

运行实验，并根据实验数据填表。

```bash
cd assignment01/cuda
make run/m4_memory/05_bandwidth
```

| stride | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| GB/s |  |  |  |  |  |  |

回答：观察数据变化趋势，并简析趋势的成因。

答案：


### prob 4.8 EXPERIMENT

文件：`cuda/m4_memory/06_occupancy.cu`

运行实验并记录数据。

```bash
cd assignment01/cuda
make run/m4_memory/06_occupancy
```

| shared memory / block (KB) |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 理论驻留 block / SM |  |  |  |  |  |  |
| occupancy |  |  |  |  |  |  |
| 实测带宽 (GB/s) |  |  |  |  |  |  |

回答：

1. 用程序开头打印的“shared memory / SM”和“最大常驻线程 / SM”，手算其中一个的驻留 block 数和 occupancy，和 API 的结果对照。

   答案：

2. 带宽为什么随 occupancy 下降？用“延迟隐藏需要足够多的常驻 warp”组织你的解释。

   答案：

3. 表中带宽随 occupancy 单调下降，但明显不成正比：从 100% 到 75% 带宽掉了多少？从 37.5% 到 12.5% 又掉了多少？试解释这个差别。

   答案：

---

## Module 5 计时与异步初步

### prob 5.1 EXPERIMENT

文件：`cuda/m5_async/01_timing_trap.cu`

运行实验：

```bash
cd assignment01/cuda
make run/m5_async/01_timing_trap
```

记录：

| 计时方式 | 数值 | 具体测的是什么 |
| --- | --- | --- |
|  |  |  |
|  |  |  |
|  |  |  |

回答：

1. 哪个数值可以当作 kernel 耗时写进报告？

   答案：

2. 另外两个各具体测的是什么？

   答案：

### prob 5.2 CONCEPT

判断对错，可以顺带补一句理由。

| 小题 | 判断 | 理由 |
| --- | --- | --- |
| 同一个 stream 里的操作按提交顺序执行。 |  |  |
| kernel 启动后，host 代码立刻继续往下执行。 |  |  |
| unified memory 下，CPU 访问一页正被 GPU 占用的内存，会触发缺页与页迁移。 |  |  |

---

## Module 6 Tile 视角

### prob 6.1 CONCEPT

判断对错，可以顺带补一句理由。

| 小题 | 判断 | 理由 |
| --- | --- | --- |
| tile 是显存里的一块可变区域，kernel 通过指针直接改写它。 |  |  |
| 对 tile 的一次运算，如两个 tile 相加，由编译器映射到 block 内的多个线程上执行。 |  |  |
| tile 模型与 SIMT 模型互斥，一个 CUDA 程序只能选一种。 |  |  |

### prob 6.2 CONCEPT

下面是 Guide 2.4.6 的 cuTile Python 向量加法：

```python
import cuda.tile as ct

@ct.kernel
def vec_add(a, b, c, TILE: ct.Constant[int]):
    a_view = a.tiled_view((TILE,))
    b_view = b.tiled_view((TILE,))
    c_view = c.tiled_view((TILE,))

    bid = ct.bid(0)
    a_tile = a_view.load((bid,))
    b_tile = b_view.load((bid,))
    c_view.store((bid,), a_tile + b_tile)
```

据此补全下表。Triton 一列可以做完 Module 7 再回来填。

|  | CUDA SIMT | cuTile | Triton |
| --- | --- | --- | --- |
| 并行单位 | block 里的 thread | block |  |
| 编号 | `blockIdx` / `threadIdx` |  |  |
| 数据分工 | 线程用全局下标来划分数据 |  |  |
| 边界处理 | `if` 判断 |  |  |

### prob 6.3 CONCEPT

仍看 prob 6.2 的代码。

1. “每个线程对应哪个/些元素”由谁决定？

   答案：

2. 列出一些在 CUDA SIMT 版向量加法里一定会出现、这里完全没体现出的概念。

   答案：

---

## Module 7 TileLang 与 Triton

### prob 7.2 MODIFY

文件：`kernels/fused_op.py`

按文件开头注释要求修改。

```bash
cd assignment01
uv run pytest tests/test_fused_op.py
```

回答：

1. 与 Module 2 里改 CUDA kernel 相比，这次的改动主要集中在 kernel 的什么部分？

   答案：

2. 主体代码为什么一行都不用动？

   答案：

### prob 7.4 FILL-IN

文件：`kernels/tilelang_copy2d.py`

请根据代码注释填空。测试命令：

```bash
cd assignment01
uv run pytest tests/test_tilelang.py -k copy2d
```

填完想一想：2.6 的四个空，行号、列号、边界保护、grid 尺寸，哪些在这里还有对应？没有对应的那个去哪了？

答案：


### prob 7.5 CONCEPT

补全下表。每个空填“用户”或“编译器”；二者都涉及的要写清楚各自的范围。

| 谁负责 | CUDA SIMT | cuTile | Triton | TileLang |
| --- | --- | --- | --- | --- |
| 线程到数据的映射 | 用户 |  |  |  |
| 边界处理 | 用户 |  |  |  |
| tile / block 尺寸的选择 | 用户 |  |  |  |
| block 内同步 | 用户 |  |  |  |

### prob 7.6 FILL-IN

文件：`kernels/tilelang_matmul.py`

请根据代码注释填空。测试命令：

```bash
cd assignment01
uv run pytest tests/test_tilelang.py -k matmul
```

填完回答：这五个空涉及到了 shared memory、寄存器 tile、流水，而 Triton 版 matmul，Bonus 的 `kernels/matmul_triton.py`，并未被显式指定，为什么？

答案：


---

## Module 8 平台与编译

### prob 8.1 CONCEPT

判断对错，可以顺带补一句理由。

| 小题 | 判断 | 理由 |
| --- | --- | --- |
| PTX 是 GPU 直接执行的机器码。 |  |  |
| 只嵌入了 `sm_70` SASS 的可执行文件，能在 compute capability 9.0 的卡上运行。 |  |  |
| 一个 fatbin 可以同时携带多个架构的 SASS 和 PTX。 |  |  |
| JIT 编译由驱动在运行时完成。 |  |  |

### prob 8.2 EXPERIMENT Optional

文件：`cuda/m0_env/01_hello.cu`

两个编译实验，对象是 Module 0 的 `01_hello.cu`。

1. 生成只含 `sm_90` SASS 的可执行文件并运行。如果你的卡本身就是 compute capability 9.0，先把 `ARCH_HIGH` 调成 100 或更高再 make。

   ```bash
   cd assignment01/cuda
   make sassonly/m0_env/01_hello
   ./bin/m0_env/01_hello_sassonly
   ```

   记录报错信息：

2. 生成只含 `compute_75` PTX 的版本并运行。

   ```bash
   cd assignment01/cuda
   make ptxonly/m0_env/01_hello
   ./bin/m0_env/01_hello_ptxonly
   ```

   能正常运行吗？PTX 是在什么时候、由谁编译成这块卡的机器码的？

   答案：

### prob 8.3 CONCEPT Optional

请简单说明 Runtime API 与 Driver API 各自的定位。`cudaMalloc` 属于哪个？

答案：
