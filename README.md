# Sparse Hamiltonian Flows

This repository contains code used to generate experiment results in [Bayesian inference via sparse Hamiltonian flows](https://arxiv.org/abs/2203.05723).

The Julia package that contains the implementation of `sparse Hamiltonian flows` can be found [here](https://github.com/NaitongChen/SparseHamiltonianFlows.jl).

* To generate plots for Synthetic Gaussian, execute 
    * `examples/synthetic_gaussian/run_all.sh` to generate all output results, and
    * `examples/synthetic_gaussian/run_plot.sh` to generate all plots.
* To generate plots for Bayesian Linear Regression, execute 
    * `examples/linear_regression/run_all.sh` to generate all output results, and
    * `examples/linear_regression/run_plot.sh` to generate all plots.
* To generate plots for Bayesian Logistic Regression, execute 
    * `examples/logistic_regression/run_all.sh` to generate all output results, and 
    * `examples/logistic_regression/run_plot.sh` to generate all plots.
