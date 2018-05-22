// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

/* Drawing random numbers from a multivariate normal distribution.
   See the Wikipedia entry for multivariate normal distribution for detail.
   Also see http://gallery.rcpp.org/articles/simulate-multivariate-normal/ 
   for another implementation */
// [[Rcpp::export]]
vec rmvnormC (vec mu, mat Sigma){
  vec eigval;
  mat eigvec;
  eig_sym(eigval, eigvec, Sigma);
  mat lambda = sqrtmat_sympd(diagmat(eigval));
  mat A = eigvec * lambda;
  int n = mu.size();
  vec z = rnorm(n, 0, 1);
  vec x = mu + A * z;
  return x;
}
vec rmvnormC1 (vec mu, mat Sigma){
  mat A = chol(Sigma);
  int n = mu.size();
  vec z = rnorm(n, 0, 1);
  vec x = mu + A * z;
  return x;
}
