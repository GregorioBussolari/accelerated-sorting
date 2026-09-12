# Parallel Bubble Sort

Project developed for the **Accelerated Computing Systems M** course.

**Authors:** Gregorio Bussolari, Andrea Pivetti

---

## Table of Contents

- [Introduction](#introduction)
- [Part I — CUDA Parallelization](#part-i--cuda-parallelization)
  - [V1 — Naïve Implementation (two kernels)](./parallel_bubble_sort_Base)
  - [V2 — Optimization with Cooperative Groups](./parallel_bubble_sort_CG_Shared)
  - [V3 — Optimization with Shared Memory](./parallel_bubble_sort_CoopGroups)
- [Part II — SIMD Implementation](#part-ii--simd-implementation)
  - [SIMD Algorithm Strategy](./bubble_sort_SIMD)
- [Docs](./docs/)


---

## Introduction

This project tackles the parallelization of the **Bubble Sort** sorting algorithm, exploring two parallel computing paradigms on different architectures:

- **SIMT (Single Instruction, Multiple Thread)** via **CUDA**, on an NVIDIA GPU;
- **SIMD (Single Instruction, Multiple Data)** via **SSE2** vector instructions, on a CPU.

The goal is to show how an inherently sequential algorithm — one that is not "embarrassingly parallel" — can be decomposed, parallelized, and progressively optimized, analyzing its bottlenecks in detail using profiling tools (**Nsight Systems**, **Nsight Compute**, **perf**).

## The Problem: Bubble Sort

Bubble Sort is a well known sorting algorithm with **O(N²)** complexity. It is known as one of the basic and least efficient algorithms.
In the classic implementation, each iteration depends on the previous one (**strong data dependency**). This characteristic makes parallelization delicate under both the SIMT and SIMD paradigms, where **synchronization** and/or **divergence** issues can arise.

## Part I — CUDA Parallelization

### Problem Decomposition

**Domain level:** given an array of `N` elements, it is logically organized into `R` rows of `K` columns.

**Logical level:** each row is assigned to a different thread, which performs `K-1` comparisons to sort the elements of its own row (an operation that can be carried out in parallel by `N` threads).

- **Problem:** the boundary between rows must be "broken" moving larger elements into subsequent rows of the array.
- **Solution:** at the end of each row-sorting phase, a **boundary-check** phase is executed, in which `N-1` threads compare the last element of each row with the first element of the following row.
- **Consequence:** the row-sorting phase and the boundary-check phase must necessarily be **sequentialized**.

### Thread Hierarchy and Data Access

Definitions:
- `N` = number of array elements
- `K` = number of elements per row (columns)
- `R` = number of rows, computed as `R = ceil(N / K)`

CUDA grid configuration:

```
dim3 block(blockSizeX, blockSizeY);          // parametrizable dimensions: 64, 256, 512
Number_of_blocks = ceil(R / blockSize)
gridDimX = ceil(sqrt(Number_of_blocks))
dim3 grid(gridDimX, (Number_of_blocks + gridDimX - 1) / gridDimX);
```

Data access:
- **Row sorting:** each thread computes the starting index of its own row: `Start_Index = Global_Index * K`
- **Boundary check:** each thread accesses the last element of its own row and the first element of the next row.

### V1 — Naïve Implementation (two kernels)

Two distinct kernels were implemented:

1. **`sort_row`** — a thread performs at most `K-1` comparisons and `K-1` swaps to move the largest element to the end of the row. Complexity `O(K)`.
2. **`check_boundaries`** — a thread compares the last element of a row with the first element of the following row. Complexity `O(1)`.

These two operations must be serialized: after each kernel finishes, control returns to the CPU, all threads are synchronized (`cudaDeviceSynchronize`), and the next kernel is launched. Complexity per iteration: `O(K) + O(1) = O(K)`. Both phases must be repeated `N` times, yielding a total complexity of `O(N * K)`.

**Problems and inefficiencies:**
- The continuous CPU↔GPU handoff when launching successive kernels significantly degrades performance;
- Complexity depends on the number of columns the array is split into;
- Launching `N` distinct kernels makes analysis with Nsight Compute impractical;
- Repeatedly launching kernels is very costly in terms of latency.

**Nsight Systems profiling** (test on an array of 131k integer elements, 500 KB, GTX 1660 S GPU):
- The total time the kernels spend executing on the GPU (763 ms) is lower than the time the CPU spends just *launching* the kernels (817 ms) → kernel launching is the real bottleneck.
- 52.8% of the time spent in `cudaDeviceSynchronize()` is due to system overhead.

| Array Size | Tgpu (ms) | Tcpu (ms) | Speedup | Block Size |
|---|---|---|---|---|
| 8k (32KB) | 112.551 | 203.187 | 1.805 | 256 |
| 16K (64KB) | 862.332 | 12343.165 | 14.314 | 256 |
| 131K (500KB) | 1899.040 | 49045.695 | 25.827 | 256 |

### V2 — Optimization with Cooperative Groups

To reduce the overhead from continuous CPU↔GPU handoffs, the second version leverages CUDA's **Cooperative Groups** API, which allows defining groups of threads that can be synchronized with each other — including synchronizing **all threads of a grid** (`grid.sync()`), which is ideal for this problem.

**Advantages:** the bottleneck introduced by repeated launches is eliminated, since thread synchronization now happens directly on the GPU, without repeatedly returning to the CPU.

**Limitations:** all launched blocks must be able to be **simultaneously resident** on the GPU's SMs. If the number of blocks exceeds this limit, the workload per thread must be increased (more columns per row), which reduces parallelism and raises complexity to `O(K*N)`.

**Nsight Compute analysis:**
- The kernel performs no floating-point operations and has negligible arithmetic intensity: the Roofline analysis is not meaningful; the kernel is clearly **Memory Bound**.
- Theoretical occupancy is 100%, achieved occupancy is ~73%, due to unbalanced workload across blocks and an insufficient number of launched blocks (an intrinsic limitation of Cooperative Groups).
- Low intra-warp divergence (Branch Efficiency 97.89%).
- Very low IPC (0.40): the kernel is heavily **synchronization bound** — two `grid.sync()` calls are executed at every iteration.
- Main stall causes: **Long Scoreboard** (waits for high-latency memory accesses) and **Barrier/Membar** (waits due to synchronization).

### V3 — Optimization with Shared Memory

To reduce stalls caused by global memory accesses, **shared memory** is introduced.

**Goal:** minimize the number of global memory transactions needed to serve the maximum number of memory requests.

**Basic idea:** shared memory is dynamically allocated to hold one row of the logical matrix; all comparisons are performed in SMEM during the kernel, with a single final write back to global memory.

```
SMEM_DIM = block.x * block.y * K * sizeof(int)
```

**Problem:** shared memory is allocated per block, with no way to synchronize SMEM across different blocks — this complicates sorting the final array.

**Solution:** synchronization between blocks happens through global memory, and is only needed during the boundary-check phase between two adjacent rows belonging to different blocks (threads within the same block don't need synchronization, since rows within the same block are independent of each other).

**Bank conflict:** managing shared memory with 4-column rows results in a **4-way bank conflict**. This layout was adopted anyway, trading some efficiency for implementation clarity and simplicity.

**Nsight Compute analysis:**
- Occupancy remains essentially unchanged (~73%): the register limit per block decreases slightly due to new management variables, while shared memory becomes an active hardware constraint.
- Memory throughput drops from 45% to 30%, as most of the traffic shifts to shared memory; compute throughput increases slightly due to fewer stalls from slow memory access.
- Stalls from high-latency memory access drop sharply (from 25 to 9 cycles), while barrier stalls increase (from 19 to 32), due to the global memory write during the boundary-check phase.
- The **Memory Chart** confirms the pattern shift: in V2, continuous L1↔L2 traffic for every comparison creates an internal traffic bottleneck; in V3, swap operations happen directly in shared memory, and DRAM access only occurs during synchronization.
- Note: disabling the L1 cache does **not** improve performance — it actually makes it worse, despite its low utilization.

**Pros:** fewer global memory transactions, faster execution time.
**Cons:** non-coalesced accesses (though less frequent than in V1); SMEM size per block grows with array size, reducing occupancy beyond a certain range.

### Results and Speedup Comparison (CUDA)

Executed on an **NVIDIA GTX 1660 Super** GPU (Turing architecture, 6 GB GDDR6, 1408 CUDA cores, 192-bit memory bus, 336 GB/s bandwidth).

**Cooperative Groups — base version (int 32-bit):**

| Array Size | Tgpu (ms) | Tcpu (ms) | Speedup | Block Size |
|---|---|---|---|---|
| 2k (8KB) | 4.730 | 8.486 | 1.794 | 256 |
| 16K (64KB) | 35.180 | 670.935 | 19.071 | 256 |
| 64K (256KB) | 281.895 | 12749.733 | 45.229 | 256 |
| 128K (500KB) | 1216.623 | 51289.597 | 42.157 | 256 |
| 256K (1MB) | 6298.442 | 202178.513 | 32.100 | 256 |
| 128K (500KB) | 1627.017 | 51289.597 | 31.466 | 512 |

**Cooperative Groups + Shared Memory (int 32-bit):**

| Array Size | Tgpu (ms) | Tcpu (ms) | Speedup | Block Size |
|---|---|---|---|---|
| 2k (8KB) | 4.281 | 8.486 | 1.982 | 256 |
| 16K (64KB) | 30.180 | 670.935 | 22.231 | 256 |
| 64K (256KB) | 194.136 | 12749.733 | 60.472 | 256 |
| 128K (500KB) | 691.203 | 51289.597 | 72.138 | 256 |
| 256K (1MB) | 3630.719 | 202178.513 | 53.894 | 256 |
| 128K (500KB) | 793.413 | 51289.597 | 59.962 | 512 |

**Analysis:**
- There is a minimum threshold (around 2k elements) below which exploiting parallelism is not effective.
- The optimal number of threads per block is **256**.
- Speedup over the serial version peaks around **500 KB**, the optimal condition for managing shared memory per block.
- In the best case, introducing shared memory yields an additional **1.71×** speedup compared to the version without it.

## Part II — SIMD Implementation

### SIMD Algorithm Strategy

The SIMD implementation follows a phased approach designed to find the minimum (or maximum) across multiple vector lanes in parallel:

1. **Phase 1:** using the SIMD paradigm, find the 4 smallest elements, one per column of the vector register (parallel processing across different lanes).
2. **Phase 2:** sequentially iterate (SISD) over the previously found column-wise minimums, identifying the **global** minimum and its index in the array.
3. **Phase 3:** the found minimum value is stored in a temporary register (`TEMP`), at position `i % 4`, where `i` is the current iteration index (from 0 to N).
4. **Phase 4:** the minimum element found in the array is swapped with the element at position `i`, and the original position of the minimum is marked (set to the maximum value) to exclude it from subsequent iterations.
5. **Phase 5:** every time `i % 4 == 3` (the `TEMP` register is full), its contents are stored back into array `A`, starting at the correct index `floor(i/4)`; from this point on, a growing block of already-sorted elements — of size `floor(i/4)` — can also be skipped.

### Performance Analysis (SIMD)

Tests were run on an **Intel Core i7-1255U (12th Gen)** CPU, using **SSE2** instructions, on 32-bit integer elements and 8-bit (char) elements, at different compiler optimization levels.

**Speedup — 4-byte elements (int):**

| N elements | -O0 | -O1 | -O2 | -O3 |
|---|---|---|---|---|
| 64K | x2 | x7.74 | x15.5 | x11 |
| 128K | x2.36 | x7.13 | x13.7 | x15 |
| 256K | x2.15 | x8.65 | x13.7 | x13.4 |

**Speedup — 1-byte elements (char):**

| N elements | -O0 | -O1 | -O2 | -O3 |
|---|---|---|---|---|
| 64K | x2.83 | x8.12 | x8.69 | x10.21 |
| 128K | x2.52 | x9.17 | x10.53 | x10.68 |
| 256K | x3.28 | x9.81 | x11.52 | x11 |

Compiler optimizations achieve speedups of up to **15×**: the compiler has more "room for improvement" with SIMD code than with sequential code.

**`perf` analysis:** an IPC of 3.82 indicates a high number of instructions executed per cycle, but a high **backend cycle idle** rate (58.64%) is also observed, indicative of stalls due to data or resource dependencies — consistent with the strong data dependency inherent to Bubble Sort.

## Conclusions and Future Work

**CUDA:**
- Using **Cooperative Groups** produced excellent results, but this API can compromise code scalability as the number of elements grows, offering limited room for improving occupancy and latency hiding.
- A possible alternative to overcome this limitation is the use of **CUDA Streams**.
- The code could also be revised to avoid shared memory bank conflicts, further improving performance.

**SIMD:**
- The `perf` analysis suggests that using wider vector registers (e.g., **AVX-512**) could yield better performance.
- The strong data dependency between instructions was confirmed as a limiting factor; processing more data at once (using more registers) could help mitigate this issue.

**Overall conclusions:**
Although the problem is not "embarrassingly parallel," this project demonstrates how proper problem decomposition and targeted use of available hardware resources can still deliver significant performance gains. In general, programs with a high need for synchronization that extends beyond the boundaries of a single block are more critical to optimize and require specific adaptations to run correctly under the SIMT paradigm.

## Reference Hardware/Software Setup

**GPU (CUDA):**
- NVIDIA GTX 1660 S / Super — Turing architecture, Compute Capability 7.5
- 6 GB GDDR6 memory, 192-bit bus, 336 GB/s bandwidth, 1408 CUDA cores

| Parameter | Value |
|---|---|
| Max Threads per Block | 1024 |
| Max Block Dimensions | 1024, 1024, 64 |
| Max Grid Dimensions | 2³¹-1, 65535, 65535 |
| Warp Size | 32 |
| Max Registers per Block | 65536 |
| Max Registers per Thread | 255 |
| Shared Memory per Block | 64 KB |
| Shared Memory per SM | 96 KB |
| Max Blocks per SM | 16 |
| Max Threads per SM | 1024 |

**CPU (SIMD):**
- Intel Core i7-1255U (12th Gen), SSE2 SIMD instructions

**Profiling tools used:**
- NVIDIA Nsight Systems (system-level profiling)
- NVIDIA Nsight Compute (kernel analysis, occupancy, memory chart, stall chart)
- `perf` (SIMD CPU code performance analysis)
