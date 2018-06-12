// [[Rcpp::depends(RcppArmadillo)]]
#include <iostream>
#include <limits>
#include <math.h>
#include <RcppArmadillo.h>
using namespace std;
using namespace Rcpp;
using namespace arma;

/* Prepare */

/* Random number generator from a multivariate normal distribution */
mat mvrnormArma(int n, vec mu, mat Sigma) {
  int ncols = Sigma.n_cols;
  mat Y = randn(n, ncols);
  return repmat(mu, 1, n).t() + Y * arma::chol(Sigma);
}

/* Random number generator from a truncated normal distribution */
vec rtruncnormC (vec mu, double sigma, double lower, double upper){
  int n = mu.size();
  vec out(n);
  for (int i = 0; i < n; i ++ ){
    double alpha = R::pnorm((lower - mu(i))/sigma, 0, 1, true, false);
    double beta = R::pnorm((upper - mu(i))/sigma, 0, 1, true, false);
    double u = R::runif(0, 1);
    double p = alpha + u * (beta - alpha);
    out(i) = R::qnorm(p, 0, 1, true, false) * sigma + mu(i);
  }
  return out;
}

/* Gibbs sampler for probit regression model with data augmentation */
// [[Rcpp::export]]
List probit_gibbs (vec y, mat X, vec b0, mat B0, int burn, int iter, int thin = 1){
  /* Prepare */
  int n = X.n_rows;
  int k = X.n_cols;
  vec y1 = conv_to<vec>::from(find(y == 1));
  vec y0 = conv_to<vec>::from(find(y == 0));
  int n1 = y1.n_elem;
  int n2 = y0.n_elem;
  vec ystar(n);
  ystar.elem(find(y == 1)) = rtruncnormC(zeros<vec>(n1), 1, 0, std::numeric_limits<double>::infinity());
  ystar.elem(find(y == 0)) = rtruncnormC(zeros<vec>(n2), 1, -1 * std::numeric_limits<double>::infinity(), 0);
  mat beta_store((iter/thin), k);
  mat ystar_store((iter/thin), n);
  
  /* Burn-in & main iteration loop */
  for (int i = 1; i <= (burn + iter); i ++ ){
    /* Update beta */
    mat B1 = inv(B0.i() + X.t() * X);
    vec b1 = B1 * (B0.i() * b0 + X.t() * ystar);
    vec b = mvrnormArma(1, b1, B1).t();
    /* Sample ystar */
    vec xb = X * b;
    ystar.elem(find(y == 1)) = rtruncnormC(xb.elem(find(y == 1)), 1, 0, numeric_limits<double>::infinity());
    ystar.elem(find(y == 0)) = rtruncnormC(xb.elem(find(y == 0)), 1, -1 * numeric_limits<double>::infinity(), 0);
    /* Store sampled values */
    if (i > burn && (i - burn)%thin == 0){
      beta_store.row((i - burn)/thin - 1) = b.t(); 
      ystar_store.row((i - burn)/thin - 1) = ystar.t();
    }
    /* show process */
    if (i == burn){
      cout << "Burn-in step finished." << "\n";
    }
    if (i == (burn + iter)){
      cout << "Main iteration loop finished." << "\n";
    }
  }
  
  /* Return results */
  return List::create(Named("beta") = beta_store, Named("ystar") = ystar_store);
}

