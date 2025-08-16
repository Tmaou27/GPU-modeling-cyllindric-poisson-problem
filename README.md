# GPU-Accelerated 2D Poisson Equation Solver (Axial Symmetry, CUDA)

This project is a GPU-optimized version of a Python program that numerically solves the IYPT 2024 Problem #13 ("Charge meter"). It calculates the potential distribution and charge on a ball using the Poisson equation.

## Key Features
- **CUDA parallelization** for massive performance gains (vs. CPU implementation).  
- **2D axial symmetry** to simplify calculations.
- **Shared memory** to increase speed(150x vs c++).  

## Performance
- The CUDA version achieves a significant speedup (see `OPJ` file for benchmarks).
- ~ 1 second for 3*10^6 cells in grid on NVIDEA T4

## Documentation
A full description of the method and results is available in the presentation file.

## Benchmark
A full speed comparasing of the method and results is in .OPJ files.
<img width="978" height="754" alt="Full time comparasing" src="https://github.com/user-attachments/assets/dcb549fe-a8ed-4f25-95a6-90c0537b6861" />
