// =============================================================================
// reliabilityPhases.h -- the participant-space phases of the reliability layer,
// as a C++ orchestrator over the BigDataStatMeth API.
//
// ARCHITECTURE. This mirrors `mgcca_phases.h` deliberately, because that is the
// architecture the estimator already uses and the one DESCRIPTION states: the R
// functions are thin wrappers around a C++ core. The split is the same one
// `run_svd` makes:
//
//   * anything whose size involves the number of VARIABLES is done out-of-core
//     through BigDataStatMeth (tcrossprod streams the block; it is never loaded);
//   * anything that is n x n in PARTICIPANTS is done in RAM with Eigen, because
//     n is small by the premise of the method (mgcca exists for p >> n).
//
// That is not a compromise: forcing an n x n dense solve through HDF5 would be
// slower and no more scalable, and `run_svd` already established the convention.
//
// CONVENTIONS FOLLOWED (the number one source of bugs in this port):
//   * always work in R dims via nrows_r()/ncols_r();
//   * crossprod(A,B) = t(A) %*% B and tcrossprod(A,B) = A %*% t(B), R semantics;
//   * std::unique_ptr for every working object;
//   * throw std::runtime_error inside try, Rf_error in catch, NEVER Rcpp::stop();
//   * H5::Exception::dontPrint() as the first line of the entry point;
//   * open with overwrite=false, create with overwrite=true and inherit compression.
//
// BLOCKING. Every out-of-core step takes an explicit block size. `block_size = 0`
// means "let BigDataStatMeth choose"; any positive value FORCES that block size,
// including a value far smaller than the data. That exists so the blocked path can
// be exercised on small fixtures in the test suite rather than only on inputs too
// large to test routinely -- separating "does the blocking work" from "does it work
// at scale". Without it the blocked path is, in practice, never tested.
// =============================================================================
#ifndef MGCCA_RELIABILITY_PHASES_HPP
#define MGCCA_RELIABILITY_PHASES_HPP

#include <BigDataStatMeth.hpp>
#include <RcppEigen.h>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "mgcca_io.h"
#include "mgcca_phases.h"   // run_normalize: the package's in-HDF5 standardisation

namespace mgcca {
namespace reliability {

// ---------------------------------------------------------------------------
// Phase R1 -- the present-only participant Gram of one block, out-of-core.
//
// The stored input block holds exactly its own individuals (the padded copies
// live in the temporary group), so standardising it IS the present-only
// convention: a block-absent individual is not present as an observed zero row.
// Reading the padded copy instead would silently reintroduce the zero-padding
// defect this layer exists to prevent.
//
// Writes an n_j x n_j Gram to out_group/ds and returns its participant ids in the
// block's own order.
// ---------------------------------------------------------------------------
inline Rcpp::CharacterVector gram_block(const std::string& filename,
                                        const std::string& in_group,
                                        const std::string& ds,
                                        const std::string& out_group,
                                        int block_size,
                                        Rcpp::Nullable<int> threads,
                                        bool standardize = true) {
    // STANDARDISE FIRST, and do it here rather than assuming the caller did.
    // The Gram of the RAW block and the Gram of the standardised block are
    // different objects, and the layer's convention is the standardised one. An
    // earlier version of this function documented the standardisation and did not
    // perform it; the parity test against the in-memory path is what caught it.
    // The block is standardised into a scratch dataset which is what tcrossprod
    // then reads, so the stored input is never modified.
    std::string src_group = in_group, src_ds = ds;
    std::string tmp_group = out_group + "_Z";
    if (standardize) {
        std::vector<std::string> one(1, ds);
        // center + scale, exactly the package's own in-HDF5 standardisation.
        // The scale factors go to a scratch group; they are not needed downstream
        // because the Gram is invariant to them once applied.
        mgcca::run_normalize(filename, in_group, tmp_group, tmp_group + "_sc",
                             one, /*center*/ true, /*scale*/ true, threads);
        src_group = tmp_group;
    }
    std::unique_ptr<BigDataStatMeth::hdf5Dataset> dX(
        new BigDataStatMeth::hdf5Dataset(filename, src_group, src_ds, false));
    dX->openDataset();
    if (dX->getDatasetptr() == nullptr)
        throw std::runtime_error("cannot open input block " + in_group + "/" + ds);
    std::unique_ptr<BigDataStatMeth::hdf5Dataset> dX2(
        new BigDataStatMeth::hdf5Dataset(filename, src_group, src_ds, false));
    dX2->openDataset();

    std::unique_ptr<BigDataStatMeth::hdf5Dataset> dG(
        new BigDataStatMeth::hdf5Dataset(filename, out_group, ds, true));
    dG->setCompressionLevel(dX->getCompressionLevel());

    // block_size > 0 forces the block size (used by the parity tests on small
    // fixtures); 0 asks BigDataStatMeth for its own choice.
    Rcpp::Nullable<int> bs = R_NilValue;
    if (block_size > 0) bs = Rcpp::Nullable<int>(Rcpp::wrap(block_size));
    int blk = BigDataStatMeth::getMaxBlockSize(dX->nrows(), dX->ncols(),
                                               dX2->nrows(), dX2->ncols(), 2, bs);
    // tcrossprod(A, A) = A %*% t(A) in R semantics -> n x n over individuals.
    BigDataStatMeth::tcrossprod(dX.get(), dX2.get(), dG.get(),
                                /*bisSym*/ true, blk, 0,
                                /*bparal*/ false, /*browmajor*/ true, threads);

    return read_rownames(filename, src_group, src_ds);
}

// ---------------------------------------------------------------------------
// Phase R2 -- the weighted operator and its eigenbasis, in RAM.
//
//   R_j = (G_j + lambda_j I)^-1        (n x n, per block)
//   M   = sum_j G_j R_j
//   S   = D M D, symmetrised, with D = 1/sqrt(K) and K the per-participant block count
//
// D is why a participant present in no block is a hard error rather than an Inf:
// the availability weight is not defined for them, and letting 1/sqrt(0) through
// produces a fit that fails much later and far from the cause.
//
// TOP-L BALANCING (balance = true). Each summand is then weighted before the sum:
//
//   A_j     = D (G_j R_j) D, symmetrised      (block j's own summand of S)
//   c_{j,L} = sum of the L largest eigenvalues of A_j
//   w_j     = J c_{j,L}^-1 / sum_h c_{h,L}^-1 (so the weights average one)
//   S       = D (sum_j w_j G_j R_j) D, symmetrised
//
// c_{j,L} is the most block j could contribute to any L-dimensional subspace,
// and the weights make w_j c_{j,L} the same for every block, so no block steers
// the retained subspace through its scale alone. They depend on the blocks and
// the availability pattern only, never on a query. The ridges and the resolvents
// R_j are the same as without balancing; only the weights in the sum change.
// A block is admitted only if A_j has numerical rank >= L (eigenvalues above
// rank_tol times the largest) and c_{j,L} > eps; otherwise its weight would not
// be defined, and that is an error rather than a capped number. The algebra and
// its tolerances are those of the participant-space subspace refit in
// reliabilitySubspace.h, which the tests compare this against.
//
// Without balancing every weight is one and the operator is computed exactly as
// it always was.
// ---------------------------------------------------------------------------
struct RefFit {
    Eigen::MatrixXd S;          // n x n weighted operator
    Eigen::MatrixXd V;          // n x n eigenvectors, columns in DECREASING eigenvalue order
    Eigen::VectorXd mu;         // n eigenvalues, decreasing
    Eigen::VectorXd D;          // n availability weights
    std::vector<Eigen::MatrixXd> Rlist;   // per-block resolvents (n x n)
    double gap;                 // mu[L-1] - mu[L], the separation the expansion needs
    Eigen::VectorXd w;          // J balancing weights (all one without balancing)
    Eigen::VectorXd cL;         // J top-L ceilings c_{j,L} (NaN without balancing)
    Eigen::VectorXi rank_num;   // J numerical rank of A_j (-1 without balancing)
};

inline RefFit reference_fit(const std::vector<Eigen::MatrixXd>& Glist,
                            const std::vector<std::vector<char> >& present,
                            const std::vector<double>& lambda,
                            long L,
                            bool balance = false,
                            const std::vector<std::string>& names = std::vector<std::string>(),
                            double eps = 1e-8, double rank_tol = 1e-8) {
    const std::size_t J = Glist.size();
    if (J < 1) throw std::runtime_error("no blocks supplied");
    if (present.size() != J || lambda.size() != J)
        throw std::runtime_error("Glist, present and lambda must have the same length");
    const long n = Glist[0].rows();
    for (std::size_t j = 0; j < J; ++j)
        if (Glist[j].rows() != n || Glist[j].cols() != n)
            throw std::runtime_error("every block Gram must be n x n over the universe");
    if (L < 1 || L >= n) throw std::runtime_error("L must satisfy 1 <= L < n");

    Eigen::VectorXd Kc = Eigen::VectorXd::Zero(n);
    for (std::size_t j = 0; j < J; ++j)
        for (long i = 0; i < n; ++i) if (present[j][(std::size_t)i]) Kc(i) += 1.0;
    for (long i = 0; i < n; ++i)
        if (Kc(i) < 1.0)
            throw std::runtime_error("participant present in no block: the availability "
                                     "weight 1/sqrt(K) is not defined");
    Eigen::VectorXd D = Kc.array().rsqrt();

    RefFit F;
    F.Rlist.resize(J);
    F.w = Eigen::VectorXd::Ones((Eigen::Index)J);
    F.cL = Eigen::VectorXd::Constant((Eigen::Index)J,
                                     std::numeric_limits<double>::quiet_NaN());
    F.rank_num = Eigen::VectorXi::Constant((Eigen::Index)J, -1);
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, n);
    std::vector<Eigen::MatrixXd> Bj;              // G_j R_j, kept only to balance
    if (balance) Bj.resize(J);
    for (std::size_t j = 0; j < J; ++j) {
        Rcpp::checkUserInterrupt();
        Eigen::MatrixXd A = Glist[j];
        A.diagonal().array() += lambda[j];
        // LDLT: the ridge makes A symmetric positive definite, and a decomposition
        // that assumes it will fail loudly if the caller's lambda does not.
        Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
        if (ldlt.info() != Eigen::Success)
            throw std::runtime_error("block resolvent: the ridged Gram is not decomposable");
        F.Rlist[j] = ldlt.solve(Eigen::MatrixXd::Identity(n, n));
        if (balance) Bj[j] = Glist[j] * F.Rlist[j];
        else M.noalias() += Glist[j] * F.Rlist[j];
    }

    if (balance) {
        for (std::size_t j = 0; j < J; ++j) {
            Rcpp::checkUserInterrupt();
            const std::string nm = (j < names.size() && !names[j].empty())
                ? names[j] : std::to_string((long long)(j + 1));
            Eigen::MatrixXd Aj = D.asDiagonal() * Bj[j] * D.asDiagonal();
            Aj = 0.5 * (Aj + Aj.transpose().eval());
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ea(Aj, Eigen::EigenvaluesOnly);
            if (ea.info() != Eigen::Success)
                throw std::runtime_error("block '" + nm + "': the eigenvalues of its "
                                         "summand could not be computed");
            const Eigen::VectorXd ev = ea.eigenvalues().reverse();   // decreasing
            const double mx = ev(0);
            long rk = 0;
            for (long r = 0; r < ev.size(); ++r) if (ev(r) > rank_tol * mx) rk++;
            double c = 0.0;
            for (long l = 0; l < L; ++l) c += ev(l);
            F.rank_num((Eigen::Index)j) = (int)rk;
            F.cL((Eigen::Index)j) = c;
            if (rk < L)
                throw std::runtime_error("block '" + nm + "' cannot be balanced: its "
                    "summand has numerical rank " + std::to_string((long long)rk) +
                    ", below L = " + std::to_string((long long)L) + ", so its top-L "
                    "ceiling is not defined. Lower L, or leave this block out.");
            if (!(std::isfinite(c) && c > eps))
                throw std::runtime_error("block '" + nm + "' cannot be balanced: the "
                    "sum of the L largest eigenvalues of its summand is zero or not "
                    "finite, so its balancing weight is not defined.");
        }
        const Eigen::VectorXd inv = F.cL.cwiseInverse();
        F.w = (double)J * inv / inv.sum();
        if (!F.w.allFinite() || (F.w.array() <= 0.0).any())
            throw std::runtime_error("the balancing weights are not finite and positive");
        for (std::size_t j = 0; j < J; ++j) M += F.w((Eigen::Index)j) * Bj[j];
    }
    F.S = D.asDiagonal() * M * D.asDiagonal();
    F.S = 0.5 * (F.S + F.S.transpose().eval());        // M is symmetric only to rounding

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(F.S);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("eigendecomposition of the weighted operator failed");
    // Eigen returns ASCENDING; the layer works in decreasing order throughout, and
    // reversing here means no other step has to remember which convention it is in.
    F.mu = es.eigenvalues().reverse();
    F.V  = es.eigenvectors().rowwise().reverse();
    F.D  = D;
    F.gap = F.mu(L - 1) - F.mu(L);
    return F;
}

// ---------------------------------------------------------------------------
// Phase R3 -- the query in the fit metric, its alignment, and the first-order
// perturbation coefficients.
//
//   z*   = z / D                      (the query in the availability metric)
//   T    = ||V_L' z*||^2 / ||z*||^2   (bounded in [0,1] by construction)
//   C    = 2 (V' A V)[L, R] / (mu_L - mu_R),  A = z* z*' / ||z*||^2
//
// The eigenvalue gaps in the denominator are the reason an ill-separated subspace
// gives large coefficients: the quantity is genuinely undetermined there, and that
// is reported as a validity flag rather than returned as a confident number.
// ---------------------------------------------------------------------------
inline double alignment_T(const RefFit& F, const Eigen::VectorXd& zstar, long L) {
    const double den = zstar.squaredNorm();
    if (!(den > 0)) throw std::runtime_error("the query is zero on every participant");
    Eigen::VectorXd w = F.V.leftCols(L).transpose() * zstar;
    return w.squaredNorm() / den;
}

inline Eigen::MatrixXd perturbation_C(const RefFit& F, const Eigen::VectorXd& zstar,
                                      long L) {
    const long n = F.V.rows();
    const double den = zstar.squaredNorm();
    if (!(den > 0)) throw std::runtime_error("the query is zero on every participant");
    // A = z* z*' / ||z*||^2 is rank one, so V'AV = (V'z*)(V'z*)' / ||z*||^2 and the
    // n x n matrix A is never formed -- the same result at a fraction of the cost.
    Eigen::VectorXd u = F.V.transpose() * zstar;
    Eigen::MatrixXd C(L, n - L);
    for (long a = 0; a < L; ++a)
        for (long b = 0; b < n - L; ++b) {
            const double gap = F.mu(a) - F.mu(L + b);
            C(a, b) = 2.0 * (u(a) * u(L + b) / den) / gap;
        }
    return C;
}

// Per-block inputs for the sealed streaming kernel: Rr_j = R_j D V_L, Sm_j = R_j D V_R.
inline void block_inputs(const RefFit& F, std::size_t j, long L,
                         Eigen::MatrixXd& Rr, Eigen::MatrixXd& Sm, double& alpha,
                         const std::vector<double>& lambda) {
    const long n = F.V.rows();
    Eigen::MatrixXd DV = F.D.asDiagonal() * F.V;
    Rr = F.Rlist[j] * DV.leftCols(L);
    Sm = F.Rlist[j] * DV.rightCols(n - L);
    alpha = lambda[j];
}

}  // namespace reliability
}  // namespace mgcca
#endif  // MGCCA_RELIABILITY_PHASES_HPP
