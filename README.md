# GPU-Accelerated 2D Poisson Equation Solver (Axial Symmetry, CUDA)

This project is a GPU-optimized version of a Python program that numerically solves the IYPT 2023 Problem #4 ("Electrometer"). It calculates the potential distribution and charge on a ball using the Poisson equation.

## Key Features
- **CUDA parallelization** for massive performance gains (vs. CPU implementation).  
- **2D axial symmetry** to simplify calculations.
- **Shared memory** to increase speed.  

## Performance
- The original Python program (CPU) takes several hours to compute.  
- The CUDA version achieves a significant speedup (see `OPJ` file for benchmarks).  

## Documentation
A full description of the method and results is available in the presentation file.
