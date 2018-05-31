// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

/* Drawing random numbers from a multivariate normal distribution.
   See the Wikipedia entry for multivariate normal distribution for detail.
   The code is adopted from 
   http://gallery.rcpp.org/articles/simulate-multivariate-normal/ 
   Returns a n-by-ncols matrix of random numbers */
// [[Rcpp::export]]
mat mvrnormArma(int n, vec mu, mat Sigma) {
  int ncols = Sigma.n_cols;
  mat Y = randn(n, ncols);
  return repmat(mu, 1, n).t() + Y * arma::chol(Sigma);
}

