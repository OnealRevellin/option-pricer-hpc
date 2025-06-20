# option_pricer_hpc

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