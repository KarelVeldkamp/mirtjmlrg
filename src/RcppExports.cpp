// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// grad_neg_loglik_A_j_conf_cpp
arma::vec grad_neg_loglik_A_j_conf_cpp(const arma::vec& response_j, const arma::vec& nonmis_ind_j, const arma::vec& A_j, const arma::vec& Q_j, const arma::mat& theta, double lambda1, double lambda2);
RcppExport SEXP _mirtjml_grad_neg_loglik_A_j_conf_cpp(SEXP response_jSEXP, SEXP nonmis_ind_jSEXP, SEXP A_jSEXP, SEXP Q_jSEXP, SEXP thetaSEXP, SEXP lambda1SEXP, SEXP lambda2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type response_j(response_jSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type nonmis_ind_j(nonmis_ind_jSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type A_j(A_jSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Q_j(Q_jSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type lambda1(lambda1SEXP);
    Rcpp::traits::input_parameter< double >::type lambda2(lambda2SEXP);
    rcpp_result_gen = Rcpp::wrap(grad_neg_loglik_A_j_conf_cpp(response_j, nonmis_ind_j, A_j, Q_j, theta, lambda1, lambda2));
    return rcpp_result_gen;
END_RCPP
}
// cjmle_conf_cpp
Rcpp::List cjmle_conf_cpp(const arma::mat& response, const arma::mat& nonmis_ind, arma::mat theta0, arma::mat A0, arma::mat Q, double cc, double tol, bool print_proc, double lambda1, double lambda2);
RcppExport SEXP _mirtjml_cjmle_conf_cpp(SEXP responseSEXP, SEXP nonmis_indSEXP, SEXP theta0SEXP, SEXP A0SEXP, SEXP QSEXP, SEXP ccSEXP, SEXP tolSEXP, SEXP print_procSEXP, SEXP lambda1SEXP, SEXP lambda2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type response(responseSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type nonmis_ind(nonmis_indSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type theta0(theta0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A0(A0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Q(QSEXP);
    Rcpp::traits::input_parameter< double >::type cc(ccSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< bool >::type print_proc(print_procSEXP);
    Rcpp::traits::input_parameter< double >::type lambda1(lambda1SEXP);
    Rcpp::traits::input_parameter< double >::type lambda2(lambda2SEXP);
    rcpp_result_gen = Rcpp::wrap(cjmle_conf_cpp(response, nonmis_ind, theta0, A0, Q, cc, tol, print_proc, lambda1, lambda2));
    return rcpp_result_gen;
END_RCPP
}
// grad_neg_loglik_A_j_cpp
arma::vec grad_neg_loglik_A_j_cpp(const arma::vec& response_j, const arma::vec& nonmis_ind_j, const arma::vec& A_j, const arma::mat& theta, double lambda1, double lambda2);
RcppExport SEXP _mirtjml_grad_neg_loglik_A_j_cpp(SEXP response_jSEXP, SEXP nonmis_ind_jSEXP, SEXP A_jSEXP, SEXP thetaSEXP, SEXP lambda1SEXP, SEXP lambda2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type response_j(response_jSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type nonmis_ind_j(nonmis_ind_jSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type A_j(A_jSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type lambda1(lambda1SEXP);
    Rcpp::traits::input_parameter< double >::type lambda2(lambda2SEXP);
    rcpp_result_gen = Rcpp::wrap(grad_neg_loglik_A_j_cpp(response_j, nonmis_ind_j, A_j, theta, lambda1, lambda2));
    return rcpp_result_gen;
END_RCPP
}
// cjmle_expr_cpp
Rcpp::List cjmle_expr_cpp(const arma::mat& response, const arma::mat& nonmis_ind, arma::mat theta0, arma::mat A0, double cc, double tol, bool print_proc, double lambda1, double lambda2);
RcppExport SEXP _mirtjml_cjmle_expr_cpp(SEXP responseSEXP, SEXP nonmis_indSEXP, SEXP theta0SEXP, SEXP A0SEXP, SEXP ccSEXP, SEXP tolSEXP, SEXP print_procSEXP, SEXP lambda1SEXP, SEXP lambda2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type response(responseSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type nonmis_ind(nonmis_indSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type theta0(theta0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A0(A0SEXP);
    Rcpp::traits::input_parameter< double >::type cc(ccSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< bool >::type print_proc(print_procSEXP);
    Rcpp::traits::input_parameter< double >::type lambda1(lambda1SEXP);
    Rcpp::traits::input_parameter< double >::type lambda2(lambda2SEXP);
    rcpp_result_gen = Rcpp::wrap(cjmle_expr_cpp(response, nonmis_ind, theta0, A0, cc, tol, print_proc, lambda1, lambda2));
    return rcpp_result_gen;
END_RCPP
}
// prox_func_cpp
arma::vec prox_func_cpp(const arma::vec& y, double C);
RcppExport SEXP _mirtjml_prox_func_cpp(SEXP ySEXP, SEXP CSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type C(CSEXP);
    rcpp_result_gen = Rcpp::wrap(prox_func_cpp(y, C));
    return rcpp_result_gen;
END_RCPP
}
// neg_loglik
double neg_loglik(const arma::mat& theta, const arma::mat& At, const arma::mat& response, const arma::mat& nonmis_ind, double lambda1, double lambda2);
RcppExport SEXP _mirtjml_neg_loglik(SEXP thetaSEXP, SEXP AtSEXP, SEXP responseSEXP, SEXP nonmis_indSEXP, SEXP lambda1SEXP, SEXP lambda2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type At(AtSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type response(responseSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type nonmis_ind(nonmis_indSEXP);
    Rcpp::traits::input_parameter< double >::type lambda1(lambda1SEXP);
    Rcpp::traits::input_parameter< double >::type lambda2(lambda2SEXP);
    rcpp_result_gen = Rcpp::wrap(neg_loglik(theta, At, response, nonmis_ind, lambda1, lambda2));
    return rcpp_result_gen;
END_RCPP
}
// neg_loglik_i_cpp
double neg_loglik_i_cpp(const arma::vec& response_i, const arma::vec& nonmis_ind_i, const arma::mat& A, const arma::vec& theta_i, double lambda1, double lambda2);
RcppExport SEXP _mirtjml_neg_loglik_i_cpp(SEXP response_iSEXP, SEXP nonmis_ind_iSEXP, SEXP ASEXP, SEXP theta_iSEXP, SEXP lambda1SEXP, SEXP lambda2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type response_i(response_iSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type nonmis_ind_i(nonmis_ind_iSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta_i(theta_iSEXP);
    Rcpp::traits::input_parameter< double >::type lambda1(lambda1SEXP);
    Rcpp::traits::input_parameter< double >::type lambda2(lambda2SEXP);
    rcpp_result_gen = Rcpp::wrap(neg_loglik_i_cpp(response_i, nonmis_ind_i, A, theta_i, lambda1, lambda2));
    return rcpp_result_gen;
END_RCPP
}
// grad_neg_loglik_thetai_cpp
arma::vec grad_neg_loglik_thetai_cpp(const arma::vec& response_i, const arma::vec& nonmis_ind_i, const arma::mat& A, const arma::vec& theta_i, double lambda1, double lambda2);
RcppExport SEXP _mirtjml_grad_neg_loglik_thetai_cpp(SEXP response_iSEXP, SEXP nonmis_ind_iSEXP, SEXP ASEXP, SEXP theta_iSEXP, SEXP lambda1SEXP, SEXP lambda2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type response_i(response_iSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type nonmis_ind_i(nonmis_ind_iSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta_i(theta_iSEXP);
    Rcpp::traits::input_parameter< double >::type lambda1(lambda1SEXP);
    Rcpp::traits::input_parameter< double >::type lambda2(lambda2SEXP);
    rcpp_result_gen = Rcpp::wrap(grad_neg_loglik_thetai_cpp(response_i, nonmis_ind_i, A, theta_i, lambda1, lambda2));
    return rcpp_result_gen;
END_RCPP
}
// neg_loglik_j_cpp
double neg_loglik_j_cpp(const arma::vec& response_j, const arma::vec& nonmis_ind_j, const arma::vec& A_j, const arma::mat& theta, double lambda1, double lambda2);
RcppExport SEXP _mirtjml_neg_loglik_j_cpp(SEXP response_jSEXP, SEXP nonmis_ind_jSEXP, SEXP A_jSEXP, SEXP thetaSEXP, SEXP lambda1SEXP, SEXP lambda2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type response_j(response_jSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type nonmis_ind_j(nonmis_ind_jSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type A_j(A_jSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type lambda1(lambda1SEXP);
    Rcpp::traits::input_parameter< double >::type lambda2(lambda2SEXP);
    rcpp_result_gen = Rcpp::wrap(neg_loglik_j_cpp(response_j, nonmis_ind_j, A_j, theta, lambda1, lambda2));
    return rcpp_result_gen;
END_RCPP
}
// getmirtjml_threads
int getmirtjml_threads();
RcppExport SEXP _mirtjml_getmirtjml_threads() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(getmirtjml_threads());
    return rcpp_result_gen;
END_RCPP
}
// hasOpenMP
bool hasOpenMP();
RcppExport SEXP _mirtjml_hasOpenMP() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(hasOpenMP());
    return rcpp_result_gen;
END_RCPP
}
// setmirtjml_threads
int setmirtjml_threads(int threads);
RcppExport SEXP _mirtjml_setmirtjml_threads(SEXP threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type threads(threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(setmirtjml_threads(threads));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_mirtjml_grad_neg_loglik_A_j_conf_cpp", (DL_FUNC) &_mirtjml_grad_neg_loglik_A_j_conf_cpp, 7},
    {"_mirtjml_cjmle_conf_cpp", (DL_FUNC) &_mirtjml_cjmle_conf_cpp, 10},
    {"_mirtjml_grad_neg_loglik_A_j_cpp", (DL_FUNC) &_mirtjml_grad_neg_loglik_A_j_cpp, 6},
    {"_mirtjml_cjmle_expr_cpp", (DL_FUNC) &_mirtjml_cjmle_expr_cpp, 9},
    {"_mirtjml_prox_func_cpp", (DL_FUNC) &_mirtjml_prox_func_cpp, 2},
    {"_mirtjml_neg_loglik", (DL_FUNC) &_mirtjml_neg_loglik, 6},
    {"_mirtjml_neg_loglik_i_cpp", (DL_FUNC) &_mirtjml_neg_loglik_i_cpp, 6},
    {"_mirtjml_grad_neg_loglik_thetai_cpp", (DL_FUNC) &_mirtjml_grad_neg_loglik_thetai_cpp, 6},
    {"_mirtjml_neg_loglik_j_cpp", (DL_FUNC) &_mirtjml_neg_loglik_j_cpp, 6},
    {"_mirtjml_getmirtjml_threads", (DL_FUNC) &_mirtjml_getmirtjml_threads, 0},
    {"_mirtjml_hasOpenMP", (DL_FUNC) &_mirtjml_hasOpenMP, 0},
    {"_mirtjml_setmirtjml_threads", (DL_FUNC) &_mirtjml_setmirtjml_threads, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_mirtjml(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
