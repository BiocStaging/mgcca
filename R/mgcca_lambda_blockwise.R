## =============================================================================
## The scale-matched block scale, computed over HDF5 column chunks.
##
## WHY IT EXISTS, IN ONE PARAGRAPH. The scale-matched rule needs, per block,
## s_j = sum(ev)/rank(ev) of the Gram of the SCALED block, and computing it the
## direct way wants the whole block scaled in RAM at once -- on a methylation
## block (877 x 299,221) that is a multi-gigabyte spike, and it is the one step
## of an otherwise file-backed pipeline that has to ask whether the machine is
## big enough. It does not have to: with Z = scale(X_j) split by COLUMNS into
## slabs, Z Z' = sum_k Z_k Z_k' is an exact sum over column chunks, and every
## statistic the rule computes is PER COLUMN, so a chunk holding whole columns
## has everything it needs for its own columns. The Gram is accumulated chunk by
## chunk in C++ straight from the file; the block never exists dense.
##
## WHAT CHANGES NUMERICALLY, AND WHAT DOES NOT. Only the summation order of the
## Gram (K accumulated chunk products instead of one BLAS call over all p
## columns). That is a floating-point difference, not an algebraic one, so this
## route is held to a declared TOLERANCE against the dense rule and never to
## bitwise identity -- see tests/testthat/test-lambda-blockwise.R, and the
## certification of the same algebra on the real HELIX blocks (worst relative
## difference 1.4e-15 against a declared 1e-10; invariance across chunk sizes
## 6.3e-16 against 1e-13).
##
## The rank/sum tail is NOT reimplemented here: .mgcca_lambda_from_gram() is the
## tail of .mgcca_lambda_scale() itself, so the two routes differ in how the
## Gram is accumulated and in nothing else.
## =============================================================================


#' Scale-matched block scale from HDF5, over column chunks
#'
#' @description Computes the per-block scale \eqn{s_j} of the scale-matched
#'   penalty parameterisation for tables held in an HDF5 file, reading each
#'   block in column chunks so that neither the block nor its scaled copy is
#'   ever held in memory. The penalty is \eqn{\lambda_j = \gamma s_j}, with
#'   \eqn{s_j = \mathrm{tr}(G_j)/\mathrm{rank}(G_j)} the mean positive
#'   eigenvalue of the Gram \eqn{G_j} of the scaled block: the scale of the
#'   eigenvalues the penalty actually shrinks. A fixed absolute \eqn{\lambda}
#'   does not hold the amount of regularisation constant as \eqn{p_j} grows,
#'   which is why the dimensionless \eqn{\gamma} is the comparable quantity.
#'
#' @details The rule applied is the package's own, as used by
#'   \code{\link{mgcca_select_lambda}}: each column is centred by its mean and
#'   scaled by its standard deviation, and a column whose centre or scale is
#'   non-finite, or whose scale is at or below \code{sd_tol}, is dropped. A
#'   column holding any non-finite value therefore drops out, and the result is
#'   finite.
#'
#'   Memory is one column chunk plus one \eqn{n \times n} Gram per block, so
#'   \code{chunk_cols} sets the footprint directly: at \code{n} individuals a
#'   chunk costs \code{n * chunk_cols * 8} bytes. The result is invariant to
#'   \code{chunk_cols} up to floating-point summation order.
#'
#'   Because \code{sd_tol} is a threshold, the function reports how far the
#'   kept/dropped decision ran from it (\code{sd_min_kept},
#'   \code{sd_max_dropped} in \code{details}) and warns when either sits within
#'   a factor of \code{1e-6} of \code{sd_tol}, where the decision would be at
#'   the mercy of the last bits of a standard deviation.
#'
#' @param filename Path to the HDF5 file holding the tables.
#' @param group HDF5 group holding the tables. Default \code{"MGCCA_IN"}, the
#'   group \code{\link{mgcca_import_hdf5}} writes.
#' @param datasets Optional character vector naming the tables to use, in
#'   order. Defaults to every dataset in \code{group}.
#' @param gamma Numeric, the dimensionless ridge: finite positive values, length
#'   1 (recycled) or one value per table. Default 1.
#' @param scale Logical, whether to centre and scale the columns before forming
#'   the Gram. Default \code{TRUE}, the scale-matched rule.
#' @param chunk_cols Number of variables read per chunk, capped at the block
#'   width. Default 5000.
#' @param rank_tol Relative tolerance for the numerical rank of \eqn{G_j}: an
#'   eigenvalue counts when it exceeds \code{rank_tol} times the largest.
#'   Default \code{1e-8}.
#' @param sd_tol Columns whose standard deviation is at or below this are
#'   dropped. Default \code{1e-8}.
#'
#' @return A list with
#'   \describe{
#'     \item{\code{lambda}}{named numeric, \code{gamma * s}, ready for
#'       \code{\link{mgcca}}'s \code{lambda} argument.}
#'     \item{\code{s}}{named numeric, the per-block scale \eqn{s_j}.}
#'     \item{\code{gamma}}{the gamma used, one value per table.}
#'     \item{\code{details}}{a data frame with one row per table: the sizes
#'       \code{n}, \code{p}, \code{n_kept}, \code{n_dropped}, the chunking
#'       \code{chunk_cols}, \code{n_chunks}, and the diagnostics
#'       \code{sym_err}, \code{sd_min_kept}, \code{sd_max_dropped}.}
#'   }
#'
#' @seealso \code{\link{mgcca_select_lambda}} for choosing \code{gamma} by
#'   held-out cross-block agreement, and \code{\link{mgcca_import_hdf5}} for
#'   writing tables into the file this function reads.
#'
#' @examples
#' data(cardiovascular)
#' ids <- Reduce(union, list(rownames(X1), rownames(X2), rownames(X3)))[1:100]
#' num <- function(d, cols = seq_len(ncol(d))) {
#'     m <- as.matrix(d[rownames(d) %in% ids, cols, drop = FALSE])
#'     storage.mode(m) <- "double"
#'     m
#' }
#' X <- list(methylation = num(X1, 1:10), clinical = num(X2), other = num(X3))
#'
#' h5 <- tempfile(fileext = ".h5")
#' desc <- mgcca_import_hdf5(X, filename = h5, overwriteFile = TRUE)
#'
#' sm <- mgcca_lambda_scale_hdf5(desc$filename, desc$group, desc$datasets)
#' sm$lambda
#' sm$details
#'
#' ## the same scales, read in narrower chunks
#' sm2 <- mgcca_lambda_scale_hdf5(desc$filename, desc$group, desc$datasets,
#'                                chunk_cols = 3)
#' max(abs(sm2$s - sm$s) / abs(sm$s))
#' unlink(h5)
#' @export
mgcca_lambda_scale_hdf5 <- function(filename, group = "MGCCA_IN",
                                    datasets = NULL, gamma = 1, scale = TRUE,
                                    chunk_cols = 5000, rank_tol = 1e-8,
                                    sd_tol = 1e-8) {

    if (!is.character(filename) || length(filename) != 1L || is.na(filename))
        stop("'filename' must be a single path to an HDF5 file", call. = FALSE)
    if (!file.exists(filename))
        stop("the HDF5 file does not exist: ", filename, call. = FALSE)
    chunk_cols <- as.integer(chunk_cols)
    if (length(chunk_cols) != 1L || is.na(chunk_cols) || chunk_cols < 1L)
        stop("'chunk_cols' must be a single positive integer", call. = FALSE)
    if (!is.numeric(rank_tol) || length(rank_tol) != 1L || !is.finite(rank_tol) ||
        rank_tol < 0)
        stop("'rank_tol' must be a single non-negative number", call. = FALSE)
    if (!is.numeric(sd_tol) || length(sd_tol) != 1L || !is.finite(sd_tol) ||
        sd_tol < 0)
        stop("'sd_tol' must be a single non-negative number", call. = FALSE)

    if (is.null(datasets)) {
        # mgcca's own listing, which opens and closes the file in one call. It
        # also returns the hidden dimnames entries (".<table>_dimnames"), which
        # are not tables.
        datasets <- grep("^\\.", mgcca_list_group_rcpp(filename, group),
                         value = TRUE, invert = TRUE)
    }
    datasets <- as.character(datasets)
    if (!length(datasets))
        stop("no table found in group '", group, "' of ", filename, call. = FALSE)

    J <- length(datasets)
    if (!is.numeric(gamma) || !length(gamma) || any(!is.finite(gamma)) ||
        any(gamma <= 0))
        stop("'gamma' must be finite positive numbers", call. = FALSE)
    if (length(gamma) == 1L) gamma <- rep(gamma, J)
    if (length(gamma) != J)
        stop("'gamma' must be length 1 or one value per table (", J, ")",
             call. = FALSE)

    # No .mgcca_refuse_held_file() here: that guard is for R-level
    # BigDataStatMeth opens, and the listing above and the reader below are
    # mgcca's own HDF5 instance, which a held mgcca handle lets through (it is
    # the open the single-handle discipline exists to allow).
    blocks <- mgcca_lambda_gram_rcpp(filename, group, datasets,
                                     chunk_cols = chunk_cols,
                                     do_scale = isTRUE(scale),
                                     sd_tol = sd_tol, return_stats = FALSE)

    # The tail is .mgcca_lambda_scale()'s own, so the two routes differ only in
    # how the Gram was accumulated.
    s <- vapply(seq_len(J), function(j)
        .mgcca_lambda_from_gram(blocks[[j]]$G, rank_tol,
                                sprintf("block %d ('%s')", j, datasets[j])),
        numeric(1))
    names(s) <- datasets

    num <- function(field) vapply(blocks, function(b) as.numeric(b[[field]]),
                                  numeric(1))
    details <- data.frame(
        dataset        = datasets,
        n              = num("n"),
        p              = num("p"),
        n_kept         = num("n_kept"),
        n_dropped      = num("n_dropped"),
        chunk_cols     = num("chunk_cols"),
        n_chunks       = num("n_chunks"),
        sym_err        = num("sym_err"),
        sd_min_kept    = num("sd_min_kept"),
        sd_max_dropped = num("sd_max_dropped"),
        stringsAsFactors = FALSE, row.names = NULL)

    .mgcca_warn_sd_margin(details, sd_tol)

    lambda <- gamma * s
    names(lambda) <- datasets
    list(lambda = lambda, s = s, gamma = gamma, details = details)
}


## ---- the lambda mgcca() computes when the caller leaves it unset -----------
## lambda_j = gamma * s_j. That is the same number mgcca_select_lambda() hands
## back for the same gamma (it returns gamma * s over the same s_j), so a user
## who tunes gamma by cross-validation afterwards is moving along the same
## parameterisation and not switching rule.
##
## Two routes, one rule: the in-memory fit has its tables in RAM and uses the
## dense rule directly; the file-backed fit reads its blocks in column chunks.
## Both end in .mgcca_lambda_from_gram(), so the semantics are the same and only
## the Gram's summation order differs. Both return the FINAL penalty, gamma
## already applied, which is what the fit runs with and what the manifest saves.
.mgcca_default_lambda_tables <- function(tables, do_scale = TRUE, gamma = 1) {
    lam <- gamma * .mgcca_lambda_scale(tables, do_scale = isTRUE(do_scale))
    names(lam) <- names(tables)
    .mgcca_report_default_lambda(lam, gamma, "in memory")
    lam
}

.mgcca_default_lambda_hdf5 <- function(desc, do_scale = TRUE, gamma = 1) {
    sm <- mgcca_lambda_scale_hdf5(desc$filename, group = desc$group,
                                  datasets = desc$datasets, gamma = gamma,
                                  scale = isTRUE(do_scale))
    .mgcca_report_default_lambda(sm$lambda, gamma, "over HDF5 column chunks")
    sm$lambda
}

## Say what was computed and on what rule, so a fit that chose its own penalty
## never looks like a fit that was given one. The values reported are the ones
## the fit uses, with gamma already in them.
.mgcca_report_default_lambda <- function(lam, gamma, how) {
    nms <- names(lam)
    if (is.null(nms) || !all(nzchar(nms)))
        nms <- paste("table", seq_along(lam))
    message("method 'penalized' without 'lambda': using the scale-matched ",
            "penalty computed from the data (gamma = ",
            format(gamma, digits = 6), ", ", how, "): ",
            paste(sprintf("%s = %.6g", nms, as.numeric(lam)), collapse = ", "),
            ".")
    invisible(NULL)
}


# The kept/dropped decision is a threshold on a standard deviation, and a
# standard deviation accumulated over column chunks agrees with one accumulated
# over the whole column to the last bits, not beyond them. So a column sitting
# within those last bits of sd_tol could be kept here and dropped by the
# in-memory rule, and THAT would move the block scale by far more than the
# summation-order difference this route is held to. It cannot be settled by
# arithmetic, so it is measured: a tight margin is reported where the caller can
# act on it instead of being assumed away.
.mgcca_warn_sd_margin <- function(details, sd_tol, rel = 1e-6) {
    if (sd_tol <= 0) return(invisible(NULL))
    near <- function(v) is.finite(v) & v > 0 &
                        abs(v / sd_tol - 1) < rel
    tight <- near(details$sd_min_kept) | near(details$sd_max_dropped)
    if (!any(tight)) return(invisible(NULL))
    warning("in ", sum(tight), " table(s) (",
            paste(details$dataset[tight], collapse = ", "),
            ") a column's standard deviation sits within a factor of ", rel,
            " of sd_tol = ", format(sd_tol), ", so whether that column is ",
            "dropped is decided by the last bits of the standard deviation ",
            "and the block scale may differ from the in-memory rule's by more ",
            "than summation order. Move sd_tol clear of that column, or drop ",
            "it upstream.", call. = FALSE)
    invisible(NULL)
}
