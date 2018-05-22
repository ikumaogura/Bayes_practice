// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

/* Sampling from a truncated normal distribution using inverse CDF method. 
 For the detail on the script below, see Jackman(2009) Example 3.6 (pp. 154-6), 
 Lynch (2007) Section 8.1.3 (pp. 200-6), and 
 the Wikipedia entry for truncate normal distribution */
// [[Rcpp::export]]
NumericVector rtruncnormC (int n, double mu, double sigma, 
                           double lower, double upper){
  double alpha = R::pnorm((lower - mu)/sigma, 0, 1, true, false);
  double beta = R::pnorm((upper - mu)/sigma, 0, 1, true, false);
  NumericVector out(n);
  for (int i = 0; i < n; i++){
    double u = R::runif(0, 1);
    double p = alpha + u * (beta - alpha);
    out(i) = R::qnorm(p, 0, 1, true, false) * sigma + mu;
  }
  return out;
}

