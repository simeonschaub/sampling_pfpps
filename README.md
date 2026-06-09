# Code for "Sampling Pfaffian point processes and the symplectic Arnoldi method"


See [our preprint](https://arxiv.org/abs/2605.01202) for details.

## Setup

`corner_growth_pfpp.jl`, `goe_pfpp.jl`, and `gse_pfpp.jl` require [KrylovKit.jl#151](https://github.com/https://github.com/Jutho/KrylovKit.jl/pull/151) and were tested with commit `362331172110bffbf2befb27d7b39cc14f709d06`. Checkout that commit in your Julia dev folder:

```sh
cd ~/.julia/dev # or whatever `JULIA_PKG_DEVDIR` is set to
git clone https://github.com/simeonschaub/KrylovKit.jl KrylovKit
cd KrylovKit
git checkout 362331172110bffbf2befb27d7b39cc14f709d06
```

## Index

### [Pluto.jl](https://plutojl.org/) notebooks

- [Corner growth model](corner_growth_pfpp.jl)
- [Gaussian orthogonal ensemble](goe_pfpp.jl)
- [Gaussian symplectic ensemble](gse_pfpp.jl)
- [Plots for the Airy point processes](airy_plots.jl)
- [Benchmarks of our symplectic Arnoldi method for the paper](arnoldi_benchmarks.jl)

### Jupyter notebook

- [Calculations for the Airy point processes](Airy_PfPP.ipynb) (produces the data used in [`airy_plots.jl`](airy_plots.jl)

### Data for plotting the Airy point processes

See the [`data` folder](data/)
