// mgcca_lambda_gram.cpp -- thin [[Rcpp::export]] wrapper over
// mgcca::lambda::scaled_block_gram (src/lambdaGram.h).
//
// It answers for EVERY dataset named, in ONE file open: the penalty rule is
// asked for J blocks at once, and a loop that opened and shut the same file J
// times is J chances for the file to be refused on Windows. Same discipline as
// src/mgcca_read_datasets.cpp.
//
// NOTHING HERE DECIDES THE PENALTY. It returns each block's n x n Gram; the
// rank/sum tail that turns a Gram into a block scale stays in R, where the
// dense rule's own tail lives, so both routes run the same code.
//
// [[Rcpp::depends(BH, RcppEigen, Rhdf5lib, BigDataStatMeth)]]
#include <BigDataStatMeth.hpp>
#include "mgcca_io.h"
#include "lambdaGram.h"

using namespace Rcpp;

//' Participant Gram of each scaled block, accumulated over HDF5 column chunks
//'
//' Reads every dataset named in \code{datasets} in column chunks and returns
//' the n by n Gram of the scaled block, so neither the block nor its scaled
//' copy is ever held dense. The scaling is the one the package's penalty rule
//' applies: centre by \code{colMeans}, scale by \code{sd}, and drop any column
//' whose centre or scale is non-finite or whose scale is at or below
//' \code{sd_tol}.
//'
//' @param filename Path to the HDF5 file.
//' @param group Group holding the datasets.
//' @param datasets Names of the datasets to measure, in the order wanted.
//' @param chunk_cols Number of variables per chunk; capped at the block width.
//' @param do_scale Whether to centre and scale the columns before accumulating.
//' @param sd_tol Columns with a scale at or below this are dropped.
//' @param return_stats Whether to return the per-column centre and scale of the
//'   kept columns.
//' @return A list with one entry per dataset, each a list holding the Gram
//'   \code{G}, the sizes \code{n}, \code{p}, \code{n_kept}, \code{n_dropped},
//'   the chunking \code{chunk_cols}, \code{n_chunks}, \code{last_chunk}, and
//'   the diagnostics \code{sym_err}, \code{sd_min_kept}, \code{sd_max_dropped}
//'   (plus \code{center} and \code{scale} when \code{return_stats} is TRUE).
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List mgcca_lambda_gram_rcpp(std::string filename, std::string group,
                                  std::vector<std::string> datasets,
                                  int chunk_cols, bool do_scale = true,
                                  double sd_tol = 1e-8,
                                  bool return_stats = false) {
    H5::Exception::dontPrint();
    Rcpp::List out(datasets.size());
    try {
        if (datasets.empty())
            throw std::runtime_error("mgcca_lambda_gram_rcpp: no dataset named");

        // One open for the whole batch. Read-write, although nothing here
        // writes: HDF5 refuses a second open of a file this process already
        // holds in the other mode, and BigDataStatMeth's own reader opens
        // read-write even to read.
        std::unique_ptr<BigDataStatMeth::hdf5File> file(
            new BigDataStatMeth::hdf5File(filename, false));
        file->openFile("rw");
        mgcca::check_readable(file.get(), group, "", "mgcca_lambda_gram_rcpp");

        for (std::size_t k = 0; k < datasets.size(); ++k) {
            mgcca::lambda::BlockGram B = mgcca::lambda::scaled_block_gram(
                file.get(), group, datasets[k], (long) chunk_cols, do_scale,
                sd_tol, return_stats);

            const int n = (int) B.n;
            Rcpp::NumericMatrix G(n, n);
            std::copy(B.G.data(), B.G.data() + (std::size_t) n * n, G.begin());

            Rcpp::List one = Rcpp::List::create(
                Rcpp::Named("G") = G,
                Rcpp::Named("n") = (double) B.n,
                Rcpp::Named("p") = (double) B.p,
                Rcpp::Named("n_kept") = (double) B.n_kept,
                Rcpp::Named("n_dropped") = (double) B.n_dropped,
                Rcpp::Named("chunk_cols") = (double) B.chunk_cols,
                Rcpp::Named("n_chunks") = B.n_chunks,
                Rcpp::Named("last_chunk") = (double) B.last_chunk,
                Rcpp::Named("sym_err") = B.sym_err,
                Rcpp::Named("sd_min_kept") = B.sd_min_kept,
                Rcpp::Named("sd_max_dropped") = B.sd_max_dropped);
            if (return_stats) {
                one["center"] = Rcpp::NumericVector(B.center.begin(), B.center.end());
                one["scale"]  = Rcpp::NumericVector(B.scale.begin(), B.scale.end());
            }
            out[k] = one;
        }
        file.reset();
    } catch (H5::Exception& e) {
        Rf_error("mgcca_lambda_gram_rcpp HDF5 error: %s", e.getDetailMsg().c_str());
    } catch (std::exception& e) {
        Rf_error("mgcca_lambda_gram_rcpp: %s", e.what());
    } catch (...) {
        Rf_error("mgcca_lambda_gram_rcpp: unknown error");
    }
    out.names() = Rcpp::CharacterVector(datasets.begin(), datasets.end());
    return out;
}
