# CSD-DLRA
The CSD-DLRA framework offers open source solvers for time and memory efficient deterministic proton and electron dose calculations in 2D/3D. 
We solve the continuous slowing down approximation (CSD) to transport equations using the dynamical low rank approximation. See [1] for a detailed description of the methodology and some initial results for 2D electron transport.


# How to use the code
1. Install julia and open REPL 
2. Add packages: Open package manager by typing 
```bash
]
```
and 
```bash
pkg> add  ProgressMeter LinearAlgebra LegendrePolynomials QuadGK FastGaussQuadrature SparseArrays SphericalHarmonicExpansions SphericalHarmonics TypedPolynomials GSL MultivariatePolynomials Einsum CUDA Base Distributions PyCall PyPlot DelimitedFiles WriteVTK Interpolations Images FileIO
``` 
3. Set dimensions, rank, solver and particle type of your problem in main.jl and choose test case (defined in settings.jl)
4. run main.jl, in REPL:
```bash
include("main.jl")
```
or from command line
```bash
julia main.jl
```
5. Results are written to the output folder

#Branches
There are many branches containing subprojects and updates we are working on, the names should be mostly self-explanatory. master contains the code used for [1] and 3DProtons, e.g. has the latest extensions to 3D proton transport, which are however still in work.

#Cite
To cite our associated paper use
@article{csd-dlra2021,
  title={A robust collision source method for rank adaptive dynamical low-rank approximation in radiation therapy},
  author={Kusch, Jonas and Stammer, Pia},
  journal={arXiv preprint arXiv:2111.07160},
  year={2021}
}
