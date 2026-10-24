// mgcca_solve_guard.cpp -- the rank guard for method = "solve" on the
// covariance route, run before the fit starts.
//
// The ordinary inverse of X_j'K_jX_j exists only if the block has full column
// rank on its measured rows, which needs p_j <= m_j (one fewer once the columns
// are centred). A block with m_j <= p_j < n is still sent to the covariance
// route, where X'KX is singular: the Cholesky inverse is then either refused
// deep inside BigDataStatMeth with a message that names nothing, or computed
// from rounding error and returned as if it were a result. mgcca() calls this
// guard after the import and before mgcca_rcpp(), so both are stopped before
// any inverse, with a message that says which block and what to do.
//
// The guard reads only the imported tables. K_j leaves the measured rows of
// the padded block untouched and its zero rows add nothing to the
// cross-product, so X_j'K_jX_j equals Z_j'Z_j, where Z_j is the imported table
// (its measured rows only), centred and scaled over those rows when
// scale = TRUE, exactly as the fit standardises it.
//
// The rank is decided exactly as the Gram route decides it: eigenvalues of the
// cross-product above 1e-12 times the largest count as numerically positive.
// On the Gram route `solve` keeps only those directions, which is the
// generalized-inverse result; here that result is what method = "geninv"
// computes, and the message says so rather than switching method behind the
// caller's back.
//
// A block with more variables than measured rows is singular by construction,
// so it is refused from the counts alone, without reading the block. Otherwise
// p_j <= m_j <= n, and the block and its p x p cross-product are no larger than
// the n x n operator the eigen stage holds in memory anyway.
//
// This file is its own translation unit and includes none of the fit's stage
// headers, so the stages compile exactly as they did without the guard.
//
// [[Rcpp::depends(BH, RcppEigen, Rhdf5lib, BigDataStatMeth)]]
#include <BigDataStatMeth.hpp>
#include "mgcca_io.h"

#include <Eigen/Dense>
#include <cmath>
#include <set>
#include <string>
#include <vector>

namespace {

// Refuse block `ds` if its cross-product has no ordinary inverse.
void solve_guard_one(BigDataStatMeth::hdf5File* file, const std::string& group,
                     const std::string& ds, bool scale) {
    std::unique_ptr<BigDataStatMeth::hdf5Dataset> d =
        mgcca::open_on(file, group, ds, "mgcca_solve_rank_guard_rcpp");
    const long m = (long)d->nrows_r();
    const long p = (long)d->ncols_r();
    d.reset();

    long rank = -1;
    if (p <= m) {
        Rcpp::NumericMatrix X = mgcca::read_matrix(file, group, ds);   // m x p
        Eigen::Map<Eigen::MatrixXd> Xm(&X[0], (Eigen::Index)m, (Eigen::Index)p);
        Eigen::MatrixXd Z = Xm;
        if (scale) {
            // base R scale(): centre by the column mean, divide by the column
            // SD with the (m - 1) divisor, as the fit's in-file standardisation.
            for (long c = 0; c < p; ++c) {
                const double mu = Z.col(c).mean();
                Z.col(c).array() -= mu;
                const double sd = std::sqrt(Z.col(c).squaredNorm() / (double)(m - 1));
                Z.col(c) /= sd;
            }
        }
        const Eigen::MatrixXd XtX = Z.transpose() * Z;                    // p x p
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
            0.5 * (XtX + XtX.transpose()), Eigen::EigenvaluesOnly);
        if (es.info() != Eigen::Success)
            throw std::runtime_error("block '" + ds + "': the eigenvalues of its "
                                     "cross-product could not be computed");
        const Eigen::VectorXd ev = es.eigenvalues();
        const double lmax = ev.maxCoeff();
        const double tol = (lmax > 0 ? lmax : 1.0) * 1e-12;
        rank = 0;
        for (long i = 0; i < ev.size(); ++i) if (ev(i) > tol) rank++;
        if (rank == p) return;
    }
    const std::string why = (rank < 0)
        ? ("it has " + std::to_string((long long)p) + " variables but only " +
           std::to_string((long long)m) + " measured individuals")
        : ("its " + std::to_string((long long)p) + " variables have numerical rank " +
           std::to_string((long long)rank) + " on its " + std::to_string((long long)m) +
           " measured individuals (too few individuals, a constant or a repeated "
           "variable)");
    throw std::runtime_error(
        "method = \"solve\" needs an ordinary inverse of X'X for every table, and "
        "table '" + ds + "' does not have one: " + why + ". Use method = "
        "\"penalized\" with a lambda (recommended), or method = \"geninv\" for the "
        "generalized-inverse result, or remove variables from that table.");
}

} // namespace

//' Refuse method = "solve" on a table whose cross-product has no inverse
//'
//' Checks every table that the covariance route would take (the route
//' \code{\link{mgcca_rcpp}} picks from \code{route}) and stops, naming the
//' table, when its cross-product \eqn{X_j^\top K_j X_j} has no ordinary
//' inverse. Tables on the Gram route are not checked. Called by
//' \code{\link{mgcca}} after the import and before the fit, for
//' \code{method = "solve"} only.
//'
//' @param filename Path to the HDF5 file.
//' @param group Group holding the imported tables.
//' @param datasets Names of the tables, in fit order.
//' @param scale Whether the fit centres and scales the tables.
//' @param route Per-table algebra route, as given to \code{mgcca_rcpp}.
//' @return \code{NULL} when every checked table has an inverse;
//'   otherwise an error naming the first table that has none.
//' @keywords internal
// [[Rcpp::export]]
SEXP mgcca_solve_rank_guard_rcpp(std::string filename, std::string group,
                                 std::vector<std::string> datasets,
                                 bool scale = true, std::string route = "auto") {
    H5::Exception::dontPrint();
    // An unknown route is mgcca_rcpp's to refuse, with its own message.
    if (route != "auto" && route != "cov" && route != "dual") return R_NilValue;
    if (route == "dual") return R_NilValue;
    try {
        // One open for every table. Read-write, although nothing here writes:
        // HDF5 refuses a second open of a file this process already holds in
        // the other mode, and BigDataStatMeth's own reader opens read-write
        // even to read.
        std::unique_ptr<BigDataStatMeth::hdf5File> file(
            new BigDataStatMeth::hdf5File(filename, false));
        file->openFile("rw");
        mgcca::check_readable(file.get(), group, "", "mgcca_solve_rank_guard_rcpp");

        // n = the union of individual IDs, the row count of every padded block.
        std::set<std::string> ids;
        for (const std::string& ds : datasets) {
            Rcpp::CharacterVector rn = mgcca::read_rownames(file.get(), group, ds);
            for (R_xlen_t i = 0; i < rn.size(); ++i)
                ids.insert(Rcpp::as<std::string>(rn[i]));
        }
        const std::size_t n = ids.size();

        for (const std::string& ds : datasets) {
            Rcpp::checkUserInterrupt();
            std::unique_ptr<BigDataStatMeth::hdf5Dataset> d =
                mgcca::open_on(file.get(), group, ds, "mgcca_solve_rank_guard_rcpp");
            const std::size_t p = d->ncols_r();
            d.reset();
            const bool dual = (route == "auto" && p >= n);
            if (!dual) solve_guard_one(file.get(), group, ds, scale);
        }
        file.reset();
    } catch (H5::Exception& e) {
        Rf_error("mgcca_rcpp HDF5 error: %s", e.getDetailMsg().c_str());
    } catch (std::exception& e) {
        Rf_error("mgcca_rcpp error: %s", e.what());
    }
    return R_NilValue;
}
