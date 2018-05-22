This repository contains useful R & C++ scripts useful for Bayesian analysis. 
C++ codes can be sourced into R using `Rcpp` package. Specifically,

## Random Number Generation
* `rmvnorm.cpp`: contains two functions to draw random numbers from a multivariate distribution. The first one uses spectral decomposition, and the second one employs Cholesky decomposition.

* `rtruncnorm.cpp`: contains a function to draw random numbers from a truncated normal distribution.
