// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "mirtjml_omp.h"

// [[Rcpp::export]]
arma::vec prox_func_cpp(const arma::vec &y, double C){
  double y_norm2 = arma::accu(square(y));
  if(y_norm2 <= C*C){
    return y;
  }
  else{
    return sqrt(C*C / y_norm2) * y;
  }
}
arma::vec prox_func_theta_cpp(arma::vec y, double C){
  double y_norm2 = arma::accu(square(y)) - 1;
  if(y_norm2 <= C*C-1){
    return y;
  }
  else{
    y = sqrt((C*C-1) / y_norm2) * y;
    y(0) = 1;
    return y;
  }
}
// [[Rcpp::export]]
double neg_loglik(const arma::mat &theta, const arma::mat &At, const arma::mat &response, const arma::mat &nonmis_ind, double lambda1 = 0, double lambda2 = 0){
  arma::mat thetaA = theta * At;
  double ll = arma::accu( nonmis_ind % (thetaA % response - arma::log(1.0+arma::exp(thetaA))));
  return -ll + arma::accu(lambda1 * abs(theta) + lambda2 * pow(theta,2)) + arma::accu(lambda1 * abs(At) + lambda2 * pow(At,2));
}
// [[Rcpp::export]]
double neg_loglik_i_cpp(const arma::vec &response_i, const arma::vec &nonmis_ind_i,
                        const arma::mat &A, const arma::vec &theta_i, double lambda1 = 0, double lambda2 = 0){
  arma::vec tmp = A * theta_i;
  return -arma::accu(nonmis_ind_i % (tmp % response_i - arma::log(1.0 + arma::exp(tmp)))) + arma::accu(lambda1 * abs(A) + lambda2 * pow(A,2)) + arma::accu(lambda1 * abs(theta_i) + lambda2 * pow(theta_i,2));
}

// [[Rcpp::export]]
arma::vec grad_neg_loglik_thetai_cpp(const arma::vec &response_i, const arma::vec &nonmis_ind_i,
                                     const arma::mat &A, const arma::vec &theta_i, double lambda1 = 0, double lambda2 = 0){
  arma::vec tmp = response_i - 1.0 / (1.0 + arma::exp(- A * theta_i));
  arma::mat grad = -A.t() * (nonmis_ind_i % tmp);
  arma::mat l1 = lambda1 * theta_i/abs(theta_i);
  arma::mat l2 = lambda2 * 2 * theta_i;
  return grad + l1 + l2;
}

// [[Rcpp::plugins(openmp)]]
arma::mat Update_theta_cpp(const arma::mat &theta0, const arma::mat &response,
                           const arma::mat &nonmis_ind, const arma::mat &A0, double C, double step_theta=200, double lambda1 = 0, double lambda2 = 0){
  arma::mat theta1 = theta0.t();
  int N = response.n_rows;
#pragma omp parallel for num_threads(getmirtjml_threads())
  for(int i=0;i<N;++i){
    double step = step_theta;
    arma::vec h = grad_neg_loglik_thetai_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0.row(i).t(), lambda1, lambda2);
    h(0) = 0;
    theta1.col(i) = theta0.row(i).t() - step * h;
    theta1.col(i) = prox_func_theta_cpp(theta1.col(i), C);
    while(neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta1.col(i), lambda1, lambda2) >
            neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0.row(i).t(), lambda1, lambda2) &&
            step > 1e-7){
      step *= 0.5;
      theta1.col(i) = theta0.row(i).t() - step * h;
      theta1.col(i) = prox_func_theta_cpp(theta1.col(i), C);
    }
    //Rcpp::Rcout << "\n final step loop when updating theta = "<< -log(step/step_theta)/log(2)<< "\n";
  }
  return(theta1.t());
}
// [[Rcpp::plugins(openmp)]]
Rcpp::List Update_theta_init_cpp(const arma::mat &theta0, const arma::mat &response,
                           const arma::mat &nonmis_ind, const arma::mat &A0, double C, double step_theta=1000, 
                           double lambda1 = 0, double lambda2 = 0){
  arma::mat theta1 = theta0.t();
  int N = response.n_rows;
  arma::vec final_step(N);
#pragma omp parallel for num_threads(getmirtjml_threads())
  for(int i=0;i<N;++i){
    double step = step_theta;
    arma::vec h = grad_neg_loglik_thetai_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0.row(i).t(), lambda1, lambda2);
    h(0) = 0;
    theta1.col(i) = theta0.row(i).t() - step * h;
    theta1.col(i) = prox_func_theta_cpp(theta1.col(i), C);
    while(neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta1.col(i), lambda1, lambda2) >
            neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0.row(i).t(), lambda1, lambda2) &&
            step > 1e-7){
      step *= 0.5;
      theta1.col(i) = theta0.row(i).t() - step * h;
      theta1.col(i) = prox_func_theta_cpp(theta1.col(i), C);
    }
    final_step(i) = step;
  }
  return(Rcpp::List::create(Rcpp::Named("theta") = theta1.t(),
                            Rcpp::Named("step_theta") = final_step));
}
// [[Rcpp::export]]
double neg_loglik_j_cpp(const arma::vec &response_j, const arma::vec &nonmis_ind_j,
                        const arma::vec &A_j, const arma::mat &theta, double lambda1 = 0, double lambda2 = 0){
  arma::vec tmp = theta * A_j;
  return -arma::accu(nonmis_ind_j % (tmp % response_j - log(1+exp(tmp)))) + arma::accu(lambda1 * abs(A_j +  lambda2 * pow(A_j,2))) + arma::accu(lambda1 * abs(theta)  + lambda2 * pow(theta,2));
}
