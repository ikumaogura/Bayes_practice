
# Load packages & functions
require(Rcpp)
require(RcppArmadillo)
sourceCpp("rtruncnorm.cpp")

# Example: random numbers from N(0, 9) and the upper tail is truncated at 0
set.seed(1234)
tmp <- matrix(NA, nrow = 4, ncol = 10000)
for (i in 1:10000){
  tmp[, i] <- rtruncnormC(rep(0, 4), 3, -Inf, 0)
}
rowMeans(tmp) # means
apply(tmp, 1, var) # variances

# Theoretical values
0 - 3*(dnorm(0)/pnorm(0)) # mean
9 * (1 - 0 * (dnorm(0)/pnorm(0)) - (dnorm(0)/pnorm(0))^2) # variance
