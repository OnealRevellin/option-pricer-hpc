# option_pricer_hpc

This project implements a **vectorized and parallelized option pricer** using the Generalized Black-Scholes-Merton (GBSM) model. Designed for speed, this tool benchmarks different configurations in C++ and Python (NumPy) across millions of options.


## Benchmark Results (50M Options x 100 runs)

| Method                    | Avg Exec Time (s) | Avg Time per Option (ns)  |
|---------------------------|-------------------|---------------------------|
| Python (NumPy)            | 8.65659           | 173.1319                  |
| C++ (no OpenMP, no opt)   | 6.96975           | 139.395                   |
| C++ (no OpenMP, optimized)| 0.456861          | 9.13723                   |
| C++ (OpenMP, optimized)   | 0.173291          | 3.46583                   |


---

**Note:**  
- "Optimized" versions include compiler optimizations such as `-O3`, `-march=native`, loop unrolling, fast math, and use OpenMP for parallelism.  
- "Not optimized" runs use default compilation without these flags or parallelization "g++ -std=c++20", resulting in significantly slower performance.
- OpenMP versions use multi-threaded execution over CPU cores for parallel vectorized computation.

---


## Build Instructions (CMake)

### Dependencies
- C++20 compatible compiler (e.g., `g++ >= 11`)
- CMake ≥ 3.16

### Build with CMake
'bash

git clone https://github.com/OnealRevellin/option-pricer-hpc.git

cd option-pricer-hpc

mkdir build

cd build

cmake ..

make -j$(nproc)

#Run the exe file.

./option_pricer_hpc


## System Configuration for GBSM Option Pricer

This section documents the system used for benchmarking and development of the option pricing engine.

---------------------------------------------------------------------------------------------------------

### OS & Environment

Operating System: Ubuntu 24.04.2 LTS

Kernel: Microsoft WSL2

---------------------------------------------------------------------------------------------------------

### CPU

Model: 13th Gen Intel(R) Core(TM) i5-13600KF

Cores/Threads: 10 Performance Cores, 20 Threads

Base Frequency: 3.5 GHz (P-cores), 2.6 GHz (E-cores)

L1 Cache: 480 KiB

L2 Cache: 20 MiB

L3 Cache: 24 MiB

Instruction Sets: AVX2, FMA, SHA-NI, VAES, BMI1/2, ADX, etc.

---------------------------------------------------------------------------------------------------------

### Memory

Installed RAM: 16 GiB

Swap: 4.0 GiB

Memory Usage (Idle): ~1.8 GiB used

---------------------------------------------------------------------------------------------------------

### GPU (not currently used)

Model: NVIDIA GeForce RTX 4080

Driver Version: 560.94

CUDA Version: 12.6

Memory: 16 GB GDDR6X

Current Usage: ~3.2 GB

---------------------------------------------------------------------------------------------------------

### Compiler & Build Options

Compiler: g++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0

Flags: -O3 -march=native -ffast-math -funroll-loops -fopenmp -std=c++20

Parallelism: OpenMP-enabled

Instruction Tuning: Auto-detected with -march=native
