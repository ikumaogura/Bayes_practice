This repository contains R & C++ scripts useful for Bayesian analysis. 
C++ codes can be sourced into R using `Rcpp` package. Specifically, following code files are included;

## Random Number Generation
* `rmvnorm.cpp`: contains a function to draw random numbers from a multivariate distribution. The function is adopted from <a href = "http://gallery.rcpp.org/articles/simulate-multivariate-normal/" target = "_blank">this website</a>.

* `rtruncnorm.cpp`: contains a function to draw random numbers from a truncated normal distribution. `truncnorm.R` tests whether the function works fine.

## Popular Models for Social Scientists
* `linear_regression.cpp`: Gibbs sampler for linear regression models with conditionally conjugate priors.

* `count_regression.cpp`: contains functions to sample from posterior distributions of Poisson and Negative Binomial regression models using Hamiltonian Monte Carlo algorithm.

* `binary_DV.cpp`: contains a function to sample from the posterior of a probit regression model using data augmentation algorithm.


