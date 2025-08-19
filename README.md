# GPU-Accelerated 2D Poisson Equation Solver (Axial Symmetry, CUDA)
**Run on Google Colab**
[https://colab.research.google.com/drive/1aOTjWGov35s7MMDJo5lAFTy2tH48Blhs?usp=sharing]

This project is a GPU-optimized version of a Python program that numerically solves the IYPT 2024 Problem #13 ("Charge meter"). It calculates the potential distribution and charge on a ball using the Poisson equation.

## Details
- **Shared memory** (150x vs c++).  
- **Numerical method**: Finite Difference Method (FDM) + Jacobi iterations
- **Optimizations**:  
  - Shared memory tiling (32x32 blocks with ghost cells)  
  - Double buffering to avoid race conditions
    
## GPU Performance (NVIDIA T4)
- The CUDA version achieves a significant speedup (see `OPJ` file for benchmarks).
- ~ **1 second** for **3*10^6** cells in grid on NVIDEA T4
- **Nsight Compute Analysis:**
- Memory throughput : **46.61%** (DRAM — bottleneck, **12.14%).  
- Compute (SM) Throughput : **8.53%**.
- L1/TEX Cache Throughput: **93.22%**.

## Documentation
A full description of the method and results is available in the presentation file.

## Optimal performance parameters  
- **Block size**: 16×16 (256 threads) → 50% faster 32×32.  
- **Problem**: Less grid size avialable.

## Benchmark
A full speed comparasing of the method and results is in .OPJ files.
<img width="978" height="754" alt="Full time comparasing" src="https://github.com/user-attachments/assets/dcb549fe-a8ed-4f25-95a6-90c0537b6861" />
