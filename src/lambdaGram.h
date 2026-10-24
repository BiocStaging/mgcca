// lambdaGram.h -- the participant Gram of a SCALED block, accumulated over
// COLUMN CHUNKS read straight from HDF5, so no dense block and no dense scaled
// block ever exists in RAM.
//
// WHAT IT IS FOR. The scale-matched penalty rule needs, per block,
//   s_j = sum(ev) / rank(ev),  ev = eigenvalues of tcrossprod(scale(X_j)),
// and the only step of it that wants the whole block dense is the Gram. With
// Z = scale(X_j) (n x p) split by COLUMNS into slabs Z = [Z_1 | ... | Z_K],
//       Z Z' = sum_k Z_k Z_k'
// is an exact sum over column chunks. Two consequences carry the design:
//   (a) every statistic the rule computes is PER COLUMN, and a column chunk
//       holds whole columns, so each column's centre and scale come from
//       exactly the doubles the dense rule works with;
//   (b) only the SUMMATION ORDER of the Gram changes (K accumulated chunk
//       products instead of one dsyrk over all p columns). That is a
//       floating-point difference, not an algebraic one, so the contract is a
//       declared tolerance and never bitwise identity.
//
// THE RULE REPRODUCED IS THE PACKAGE'S OWN, R/mgcca_select_lambda.R
// (.mgcca_scale_block + .mgcca_lambda_scale):
//   centre   colMeans(X)                        -- long double sum / n
//   scale    apply(X, 2, stats::sd)
//   keep     is.finite(centre) & is.finite(scale) & scale > sd_tol
//   scaled   (x - centre) / scale  over the KEPT columns only
// A column with any non-finite entry has a non-finite centre and is therefore
// dropped, so the accumulated Gram is finite whenever the dense rule's is.
//
// The centre is reproduced bit-exactly. stats::sd is NOT reproducible bitwise
// from outside var()'s own accumulation loop; the closest attainable form is
// used (R's refined two-pass mean, then the sum of double squares accumulated
// in long double) and lands within ~2.5e-16 relative. Because `scale > sd_tol`
// is a THRESHOLD on that value, the kept/dropped margin is MEASURED and
// reported (sd_min_kept, sd_max_dropped) rather than assumed safe.
//
// The n x n Gram and its eigendecomposition stay in RAM: n is the sample count,
// small by the p >> n assumption. The rank/sum tail is deliberately left to R,
// so both routes run the same tail code on their respective Grams.
//
// Include AFTER BigDataStatMeth.hpp.
#ifndef MGCCA_LAMBDA_GRAM_HPP
#define MGCCA_LAMBDA_GRAM_HPP

#include <BigDataStatMeth.hpp>
#include <Eigen/Dense>
#include "mgcca_io.h"
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace mgcca {
namespace lambda {

// One block's accumulated Gram plus the diagnostics of the accumulation.
struct BlockGram {
    Eigen::MatrixXd G;            // n x n, symmetrized
    long n;                       // individuals (R rows)
    long p;                       // variables (R columns) in the dataset
    long n_kept;                  // columns that entered the Gram
    long n_dropped;               // columns the rule dropped
    long chunk_cols;              // column-chunk width used
    int  n_chunks;
    long last_chunk;              // width of the final (possibly partial) chunk
    double sym_err;               // max |G - G'| BEFORE symmetrizing
    double sd_min_kept;           // smallest sd among kept columns (Inf if none)
    double sd_max_dropped;        // largest finite sd among dropped columns (0 if none)
    std::vector<double> center;   // kept columns, in order; empty unless asked for
    std::vector<double> scale;    // kept columns, in order; empty unless asked for
};

// The centre the rule uses: colMeans(), a long double accumulation over the
// column divided by n. `long double` tracks R's own LDOUBLE on each platform.
inline double column_center(const double* x, long n) {
    long double s = 0.0L;
    for (long i = 0; i < n; ++i) s += (long double) x[i];
    return (double) (s / (long double) n);
}

// The scale the rule uses: stats::sd(). Not reproducible bitwise from outside
// var()'s own loop; this is the closest attainable form -- R's refined two-pass
// mean (mean.default's own correction), then sum((x - m)^2) accumulated in long
// double over DOUBLE squares, as colSums() would.
inline double column_sd(const double* x, long n) {
    long double m = 0.0L;
    for (long i = 0; i < n; ++i) m += (long double) x[i];
    m /= (long double) n;
    if (std::isfinite((double) m)) {
        long double t = 0.0L;
        for (long i = 0; i < n; ++i) t += ((long double) x[i] - m);
        m += t / (long double) n;
    }
    const double mref = (double) m;
    long double q = 0.0L;
    for (long i = 0; i < n; ++i) {
        const double d = x[i] - mref;
        q += (long double) (d * d);
    }
    return std::sqrt((double) q / (double) (n - 1));
}

// Accumulate one block's Gram over column chunks, on a file SOMEBODY ELSE holds
// open: the dataset object binds to that handle and releases only itself, so a
// caller with J blocks opens the file once.
inline BlockGram scaled_block_gram(BigDataStatMeth::hdf5File* file,
                                   const std::string& group,
                                   const std::string& ds,
                                   long chunk_cols,
                                   bool do_scale,
                                   double sd_tol,
                                   bool return_stats) {
    if (chunk_cols < 1)
        throw std::runtime_error("scaled_block_gram: chunk_cols must be >= 1");

    std::unique_ptr<BigDataStatMeth::hdf5Dataset> d =
        mgcca::open_on(file, group, ds, "scaled_block_gram");

    BlockGram R;
    R.n = (long) d->nrows_r();              // individuals
    R.p = (long) d->ncols_r();              // variables
    if (R.n < 2)
        throw std::runtime_error("scaled_block_gram: '" + group + "/" + ds +
                                 "' has fewer than two individuals; a block "
                                 "scale is undefined");
    if (R.p < 1)
        throw std::runtime_error("scaled_block_gram: '" + group + "/" + ds +
                                 "' has no variables");

    const long n = R.n, p = R.p;
    R.chunk_cols = std::min<long>(chunk_cols, p);
    R.n_kept = 0; R.n_dropped = 0; R.n_chunks = 0; R.last_chunk = 0;
    R.sd_min_kept = std::numeric_limits<double>::infinity();
    R.sd_max_dropped = 0.0;
    R.G = Eigen::MatrixXd::Zero(n, n);

    Eigen::MatrixXd slab(n, R.chunk_cols);  // the only p-sized allocation
    Eigen::MatrixXd Z;
    if (do_scale) Z.resize(n, R.chunk_cols);

    for (long c0 = 0; c0 < p; c0 += R.chunk_cols) {
        Rcpp::checkUserInterrupt();
        const long cb = std::min<long>(R.chunk_cols, p - c0);

        // A column range in R view is a ROW range of the HDF5 layout, hence a
        // contiguous slab. off/cnt are {column, row}, as in mgcca_io.h.
        std::vector<hsize_t> off = {(hsize_t) c0, 0},
                             cnt = {(hsize_t) cb, (hsize_t) n},
                             st  = {1, 1}, bl = {1, 1};
        d->readDatasetBlock(off, cnt, st, bl, slab.data());

        if (!do_scale) {
            // The rule's do_scale = FALSE branch is as.matrix(X): the raw block,
            // no column dropped.
            R.G.noalias() += slab.leftCols(cb) * slab.leftCols(cb).transpose();
            R.n_kept += cb;
        } else {
            long kept = 0;
            for (long c = 0; c < cb; ++c) {
                const double* x = slab.data() + (std::size_t) c * (std::size_t) n;
                const double ctr = column_center(x, n);
                const double sdv = column_sd(x, n);
                // The keep rule, with NaN handled by the negated comparison.
                const bool keep = std::isfinite(ctr) && std::isfinite(sdv) &&
                                  sdv > sd_tol;
                if (!keep) {
                    ++R.n_dropped;
                    if (std::isfinite(sdv) && sdv > R.sd_max_dropped)
                        R.sd_max_dropped = sdv;
                    continue;
                }
                if (sdv < R.sd_min_kept) R.sd_min_kept = sdv;
                double* z = Z.data() + (std::size_t) kept * (std::size_t) n;
                for (long i = 0; i < n; ++i) z[i] = (x[i] - ctr) / sdv;
                if (return_stats) { R.center.push_back(ctr); R.scale.push_back(sdv); }
                ++kept;
            }
            // Kept columns keep their relative order, so the chunk product sees
            // them in the order the dense rule's single dsyrk would.
            if (kept > 0)
                R.G.noalias() += Z.leftCols(kept) * Z.leftCols(kept).transpose();
            R.n_kept += kept;
        }
        ++R.n_chunks; R.last_chunk = cb;
    }

    if (R.n_kept == 0)
        throw std::runtime_error("scaled_block_gram: '" + group + "/" + ds +
                                 "' has no column with non-zero variance");

    // Asymmetry of the accumulation, recorded BEFORE the deterministic
    // symmetrization (the pattern of reliabilityGram.h).
    R.sym_err = (R.G - R.G.transpose()).cwiseAbs().maxCoeff();
    R.G = 0.5 * (R.G + R.G.transpose());
    return R;
}

}  // namespace lambda
}  // namespace mgcca
#endif  // MGCCA_LAMBDA_GRAM_HPP
