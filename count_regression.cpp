// [[Rcpp::depends(RcppArmadillo)]]
#include <iostream>
#include <math.h>
#include <RcppArmadillo.h>
using namespace std;
using namespace Rcpp;
using namespace arma;

/* Prepare */

/* Random number generation from a multivariate normal */
mat mvrnormArma(int n, vec mu, mat Sigma) {
  int ncols = Sigma.n_cols;
  mat Y = randn(n, ncols);
  return repmat(mu, 1, n).t() + Y * arma::chol(Sigma);
}

/* Density function for a multivariate normal distribution */
double dmvnorm (vec x, vec mu, mat Sigma){
  int k = x.size();
  double density = ((double)1/sqrt(pow(2 * M_PI, k) * det(Sigma))) * 
    as_scalar(exp((-1/2) * trans(x - mu) * inv(Sigma) * (x - mu)));
  return density;
}

/* vectorized power function */
vec vec_pow (vec a, vec b){
  int l = a.size();
  vec out = zeros<vec>(l);
  for (int i = 0; i < l; i ++){
    out(i) = pow(a(i), b(i));
  }
  return out;
}

/* Function to compute log-posterior of Poisson regression model */
double pois_logposterior (vec y, mat X, vec beta, vec b0, mat B0){
  vec mu = exp(X * beta);
  double posterior = accu(log(vec_pow(mu, y)) - lgamma(y + 1) - mu) + 
    log(dmvnorm(beta, b0, B0));
  return posterior;
}

/* Function to compute log-posterior of Negative binomial regression model */
double negbin_logposterior (vec y, mat X, vec beta, double alpha, 
                            vec b0, mat B0, double a, double b){
  int n = y.size();
  vec mu = exp(X * beta);
  vec first = lgamma(y + alpha * ones<vec>(n)) - lgamma(y + 1) 
    - lgamma(alpha * ones<vec>(n));
  vec second = alpha * (log(alpha * ones<vec>(n)) 
                          - log(alpha * ones<vec>(n) + mu));
  vec third = y % (log(mu) - log(alpha * ones<vec>(n) + mu));
  double posterior = accu(first + second + third) + dmvnorm(beta, b0, B0) + 
    R::dgamma(alpha, a, (double)1/b, false);
  return posterior;
}

/* Numerical gradient for Poisson regression model */
vec deriv_pois (vec y, mat X, vec beta, vec b0, mat B0, double e = 0.0001){
  int k = beta.size();
  vec res = zeros<vec>(k);
  for (int i = 0; i < k; i++){
    vec beta_hi = beta, beta_lo = beta;
    beta_hi[i] += e;
    beta_lo[i] -= e;
    res[i] = (pois_logposterior(y, X, beta_hi, b0, B0) - 
      pois_logposterior(y, X, beta_lo, b0, B0))/(2 * e);
  }
  return res;
}

/* Numerical gradient for Negative binomial regression model */
vec deriv_negbin_coef (vec y, mat X, vec beta, double alpha, 
                       vec b0, mat B0, double a, double b, double e = 0.0001){
  int k = beta.size();
  vec res = zeros<vec>(k);
  for (int i = 0; i < k; i++){
    vec beta_hi = beta, beta_lo = beta;
    beta_hi[i] += e;
    beta_lo[i] -= e;
    res[i] = (negbin_logposterior(y, X, beta_hi, alpha, b0, B0, a, b) - 
      negbin_logposterior(y, X, beta_lo, alpha, b0, B0, a, b))/(2 * e);
  }
  return res;
}
double deriv_negbin_alpha (vec y, mat X, vec beta, double alpha, 
                           vec b0, mat B0, double a, double b, double e = 0.0001){
  double alpha_hi = alpha + e;
  double alpha_lo = alpha - e;
  double res = (negbin_logposterior(y, X, beta, alpha_hi, b0, B0, a, b) - 
                negbin_logposterior(y, X, beta, alpha_lo, b0, B0, a, b))/(2 * e);
  return res;
}

/* Hamiltoniam Monte Carlo for Poisson regression model */
// [[Rcpp::export]]
List pois_HMC (vec y, mat X, vec b0, mat B0, vec beta_start, int L, double epsilon, 
               int burn, int iter, int thin = 1){
  /* Prepare */
  int k = X.n_cols;
  vec beta = beta_start;
  int r = 0;
  mat beta_store = zeros<mat>((iter/thin), k);

  /* Burn-in */
  for (int i = 1; i <= burn; i ++){
    /* Draw phi ('momentum') */
    mat phi = mvrnormArma(1, zeros<vec>(k), eye<mat>(k, k));
    /* find proposal point using leap-frog */
    vec current_beta = beta;
    vec current_phi = phi.t() + (epsilon/(double)2) * deriv_pois(y, X, current_beta, b0, B0);
    for (int l = 1; l <= (L - 1); l ++){
      current_beta = current_beta + epsilon * inv(eye<mat>(k, k)) * current_phi;
      current_phi = current_phi + epsilon * deriv_pois(y, X, current_beta, b0, B0);
    }
    current_beta = current_beta + epsilon * inv(eye<mat>(k, k)) * current_phi;
    current_phi = current_phi + ((double)1/(double)2) * epsilon * 
      deriv_pois(y, X, current_beta, b0, B0);
    /* judge accept/reject */
    double H_old = pois_logposterior(y, X, beta, b0, B0) 
      + accu(pow(phi, 2))/(double)2;
    double H_new = pois_logposterior(y, X, current_beta, b0, B0) 
      + accu(pow(current_phi, 2))/(double)2;
    double u = log(randu<double>());
    if (u <= H_new - H_old){
      beta = current_beta;
      r += 1;
    }
  }
  
  cout << "Burn-in step finished." << "\n";

  /* Main iteration loop */
  for (int i = 1; i <= iter; i ++){
    /* Draw phi ('momentum') */
    mat phi = mvrnormArma(1, zeros<vec>(k), eye<mat>(k, k));
    /* find proposal point using leap-frog */
    vec current_beta = beta;
    vec current_phi = phi.t() + ((double)1/(double)2) * epsilon * 
      deriv_pois(y, X, current_beta, b0, B0);
    for (int l = 1; l <= (L - 1); l ++){
      current_beta = current_beta + epsilon * inv(eye<mat>(k, k)) * current_phi;
      current_phi = current_phi + epsilon * deriv_pois(y, X, current_beta, b0, B0);
    }
    current_beta = current_beta + epsilon * inv(eye<mat>(k, k)) * current_phi;
    current_phi = current_phi + ((double)1/(double)2) * epsilon * 
      deriv_pois(y, X, current_beta, b0, B0);
    /* judge accept/reject */
    double H_old = pois_logposterior(y, X, beta, b0, B0) 
      + accu(pow(phi, 2))/(double)2;
    double H_new = pois_logposterior(y, X, current_beta, b0, B0) 
      + accu(pow(current_phi, 2))/(double)2;
    double u = log(randu<double>());
    if (u <= H_new - H_old){
      beta = current_beta;
      r += 1;
    }
    /* store result */
    if (i % thin == 0){
      beta_store.row((i/thin) - 1) = beta.t();
    }
  }
  
  cout << "Main iteration loop finished." << "\n";
  
  /* Return results */
  double ratio = (double)r/((double)iter + (double)burn);
  return List::create(Named("beta") = beta_store, Named("ratio") = ratio);
}

/* Hamiltoniam Monte Carlo for Negative binomial regression model */
// [[Rcpp::export]]
List negbin_HMC (vec y, mat X, vec b0, mat B0, double a, double b, 
                 vec beta_start, double alpha_start, int L, double epsilon,
                 int burn, int iter, int thin = 1){
  /* Prepare */
  int k = X.n_cols;
  vec beta = beta_start;
  double alpha = alpha_start;
  int r_alpha = 0;
  int r_beta = 0;
  mat beta_store = zeros<mat>((iter/thin), k);
  vec alpha_store = zeros<vec>(iter/thin);
  
  /* Burn-in + Main iteration loop */
  for (int i = 1; i <= (burn + iter); i ++ ){
    /* Update alpha */
    double phi = R::rnorm(0, 1); // draw phi ('momentum')
    double current_alpha = alpha; // leap-frog
    double current_phi = phi + (epsilon/(double)2) * 
      deriv_negbin_alpha(y, X, beta, current_alpha, b0, B0, a, b);
    for (int l = 1; l <= (L - 1); l ++ ){
      current_alpha = current_alpha + epsilon * current_phi;
      current_phi = current_phi + epsilon * deriv_negbin_alpha(y, X, beta, current_alpha, 
                                                               b0, B0, a, b);
    }
    current_alpha = current_alpha + epsilon * current_phi;
    current_phi = current_phi + (epsilon/(double)2) * 
      deriv_negbin_alpha(y, X, beta, current_alpha, b0, B0, a, b);
    double H_old = negbin_logposterior(y, X, beta, alpha, b0, B0, a, b) + 
      pow(phi, 2)/(double)2; // judge accept/reject
    double H_new = negbin_logposterior(y, X, beta, current_alpha, b0, B0, a, b) + 
      pow(current_phi, 2)/(double)2;
    double u = log(randu<double>());
    if (u <= H_new - H_old){
      alpha = current_alpha;
      r_alpha += 1;
    }
    /* Update beta */
    mat phi_vec = mvrnormArma(1, zeros<vec>(k), eye<mat>(k, k)); //draw phi
    vec current_beta = beta; //leap-frog
    vec current_phi_vec = phi_vec.t() + (epsilon/(double)2) * 
      deriv_negbin_coef(y, X, current_beta, alpha, b0, B0, a, b);
    for (int l = 1; l <= (L - 1); l ++){
      current_beta = current_beta + epsilon * inv(eye<mat>(k, k)) * current_phi_vec;
      current_phi_vec = current_phi_vec + epsilon * 
        deriv_negbin_coef(y, X, current_beta, alpha, b0, B0, a, b);
    }
    current_beta = current_beta + epsilon * inv(eye<mat>(k, k)) * current_phi_vec;
    current_phi_vec = current_phi_vec + (epsilon/(double)2) * 
      deriv_negbin_coef(y, X, current_beta, alpha, b0, B0, a, b);
    H_old = negbin_logposterior(y, X, beta, alpha, b0, B0, a, b) + 
      accu(pow(phi_vec, 2))/(double)2; // judge accept/reject
    H_new = negbin_logposterior(y, X, current_beta, alpha, b0, B0, a, b) + 
      accu(pow(current_phi_vec, 2))/(double)2;
    u = log(randu<double>());
    if (u <= H_new - H_old){
      beta = current_beta;
      r_beta += 1;
    }
    /* store results */
    if (i > burn && (i - burn)%thin == 0){
      beta_store.row((i - burn)/thin - 1) = beta.t();
      alpha_store((i - burn)/thin - 1) = alpha;
    }
    /* show process */
    if (i == burn){
      cout << "Burn-in step finished." << "\n";
    }
    if (i == (burn + iter)){
      cout << "Main iteration loop finished." << "\n";
    }
  }
  
  /* return results */
  double ratio_alpha = (double)r_alpha/((double)burn + (double)iter);
  double ratio_beta = (double)r_beta/((double)burn + (double)iter);
  return List::create(Named("beta") = beta_store, 
                      Named("alpha") = alpha_store,
                      Named("ratio_beta") = ratio_beta, 
                      Named("ratio_alpha") = ratio_alpha);
}


  