#include<iostream>
#include<stdio.h>
#include<RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]

using namespace std;
using namespace Rcpp;
using namespace arma;

/*function to generate random numbers from multivariate normal distribution*/
mat r_mvnorm (mat mu, mat Sigma){
  int l = mu.n_rows;
  mat A = chol(Sigma);
  mat z = randn<mat>(l,1);
  mat x = mu + A*z;
  return(x);
}

/*function to generate random numbers from truncate normal distribution*/
mat r_truncnorm (int n, double mu, double sd, double upper, double lower){
  double alpha = (lower - mu)/sd;
  double beta = (upper - mu)/sd;
  double a = R::pnorm(alpha, 0, 1, 1, 0);
  double b = R::pnorm(beta, 0, 1, 1, 0);
  mat x = zeros<mat>(n, 1);
  for (int i = 0; i < n; i++){
    double d = R::runif(0, 1);
    double q = d*(b- a);
    double randn = R::qnorm(a + q, 0, 1, 1, 0)*sd + mu;
    x.row(i) = randn;
  }
  return x;
}

/*Gibbs sampler homoskedastic linear regression model
using conditional conjugate normal and inverse-gamma prior*/
//[[Rcpp::export]]
List lm_gibbs (mat y, mat X, mat b0, mat B0, double a0, double s0, mat beta_start,
               int burn, int iter, int thin)
{
  /*prepare*/
  int n = X.n_rows;
  int k = X.n_cols;
  mat beta_hat = inv(trans(X)*X)*trans(X)*y;
  double a1 = a0 + n;
  mat beta_store = mat(k,(iter/thin));
  mat sigma_store = mat(1,(iter/thin));

  /*start values*/
  mat beta = beta_start;
  
  /*Burn-in*/
  for (int i = 1; i <= burn; i++){
    /*update sigma^2*/
    double s1 = s0 + as_scalar(trans(y - X*beta)*(y - X*beta));
    double sigma2 = 1/R::rgamma(a1/(double)2, (double)2/s1);
    /*update beta*/
    mat B1 = inv(pow(sigma2, -1.0)*trans(X)*X + inv(B0));
    mat b1 = inv(B1)*(inv(B0)*b0 + pow(sigma2, -1.0)*trans(X)*y);
    mat beta = r_mvnorm(b1, B1);
  }
  
  /*show process*/
  cout << "Burn-in iterations have finished \n";
  
  /*Main iteration loop*/
  for (int i = 1; i <= iter; i++){
    /*update sigma^2*/
    double s1 = s0 + as_scalar(trans(y - X*beta)*(y - X*beta));
    double sigma2 = 1/R::rgamma(a1/(double)2, (double)2/s1);
    /*update beta*/
    mat B1 = inv(pow(sigma2, -1.0)*trans(X)*X + inv(B0));
    mat b1 = inv(B1)*(inv(B0)*b0 + pow(sigma2, -1.0)*trans(X)*y);
    mat beta = r_mvnorm(b1, B1);
    
    /*store sampled values*/
    if(i%thin == 0){
      beta_store.col((i/thin)-1) = beta;
      sigma_store((i/thin)-1) = sigma2;
    }
  }
  
  /*show process*/
    cout << "Main iteration step has finished. \n";
  
  /*return results*/
    return List::create(Named("beta") = beta_store, Named("Sigma") = sigma_store);
}


