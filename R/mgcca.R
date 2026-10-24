#' Generalized Canonical Correlation Analysis with missing individuals
#'
#' @description High-level, one-call entry point for the C++/HDF5 MGCCA pipeline:
#'   import any supported input into HDF5, then run the single-call orchestrator
#'   \code{\link{mgcca_rcpp}}. Results are written to the HDF5 file under
#'   \code{FINAL_RESULTS} (\code{Y}, \code{corsY}, \code{pval}, \code{AVE}, and
#'   \code{scores} if requested); with \code{collect = TRUE} they are also
#'   returned in memory as an object of class \code{"mgcca"} ready for
#'   \code{\link{plotIndividuals}} / \code{\link{plotVariables}}.
#'
#' @param x Input data: an HDF5 file path (data already under \code{group}), a
#'   \code{MultiAssayExperiment}, an \code{ExpressionSet}/
#'   \code{SummarizedExperiment}, or a named \code{list} of matrices
#'   (individuals x variables, rownames = individual IDs).
#' @param filename Target HDF5 file. When given (the default behaviour up to and
#'   including 1.3.0), the whole pipeline runs out-of-core on that file. When
#'   \code{NULL}, mgcca fits \emph{in memory} instead: the tables are held in RAM
#'   and the estimator runs in pure R, with no HDF5 file written. The in-memory
#'   path is meant for tables that fit in RAM; it accepts a list of matrices, a
#'   \code{MultiAssayExperiment}, a \code{SummarizedExperiment} or an
#'   \code{ExpressionSet} (all materialised in RAM), but not an HDF5 path as
#'   \code{x} (that requires a \code{filename}). Ignored when \code{x} is itself
#'   an HDF5 path. The presence of \code{filename} \emph{is} the choice of
#'   backend -- there is no automatic fallback by size.
#' @param group Input HDF5 group. Default \code{"MGCCA_IN"}.
#' @param datasets Optional dataset names (subset/order).
#' @param nfac Number of shared components. Default 2.
#' @param scale If TRUE (default), column-center+scale each table before
#'   analysis, with base-R \code{scale()} semantics (sd with the n-1 divisor),
#'   over the individuals present in each table. On the file-backed path
#'   (\code{filename} given) the scaling is performed \emph{inside} the HDF5 file
#'   (out-of-core) after a raw, untransformed import, so it applies uniformly to
#'   every input type -- including inputs that are already an HDF5 file -- and
#'   never loads a full table into RAM. On the in-memory path
#'   (\code{filename = NULL}) it is done in RAM on the materialised tables.
#' @param method Inversion method: \code{"solve"} (SPD Cholesky),
#'   \code{"penalized"} (penalty from \code{lambda}, computed when it is left
#'   unset), or \code{"geninv"}/\code{"ginv"} (Moore-Penrose pseudoinverse).
#'   On the covariance route (see \code{route}), \code{"solve"} needs an
#'   ordinary inverse of the table's cross-product \eqn{X_j^\top K_j X_j}. A
#'   table whose variables are not linearly independent on its measured
#'   individuals (at least as many variables as measured individuals when the
#'   columns are centred, or a constant or repeated variable) has none, and is
#'   refused with an error naming it rather than inverted from rounding error.
#' @param lambda The penalty for \code{method = "penalized"}, either given or
#'   computed. Pass a numeric vector (one value per table, or one value recycled
#'   to all of them) to set it yourself; the fit then uses exactly those values.
#'   Leave it \code{NULL} (the default) and mgcca computes the scale-matched
#'   penalty from the data: \eqn{\lambda_j = s_j}, where \eqn{s_j} is the mean
#'   positive eigenvalue of table \emph{j}'s Gram, which is the scale of the
#'   eigenvalues the penalty shrinks. A fixed absolute \eqn{\lambda} does not
#'   hold the amount of regularisation constant as a table gets wider, so the
#'   scale-matched value is the one that is comparable across tables of very
#'   different widths. The strength of that computed penalty is set by
#'   \code{gamma}. The values actually used are reported with a message and
#'   saved in the file's manifest, and they are the same numbers
#'   \code{\link{mgcca_select_lambda}} reports for the same \code{gamma}. On the
#'   file-backed path the computation reads each table in column chunks, so it
#'   never holds a table in RAM; see \code{\link{mgcca_lambda_scale_hdf5}}.
#' @param gamma How strong the \emph{computed} penalty should be, as a multiple
#'   of each table's own scale: the penalty used is \eqn{\lambda_j = \gamma s_j}.
#'   Default 1. Raise it to regularise harder, lower it to regularise less;
#'   because it multiplies each table's own \eqn{s_j}, one \code{gamma} means the
#'   same amount of regularisation in every table however wide they are, which a
#'   single absolute \eqn{\lambda} cannot do. It applies only when the penalty is
#'   computed: with an explicit \code{lambda}, or with any method other than
#'   \code{"penalized"}, there is nothing for it to scale and it is ignored with
#'   a warning. Use \code{\link{mgcca_select_lambda}} to choose it by held-out
#'   cross-block agreement instead of by hand.
#' @param scores If TRUE, also compute per-table weights and scores. A table's
#'   scores are standardised over the individuals \emph{present in that table},
#'   and individuals absent from it get \code{NA} (\code{NaN} in the HDF5 file)
#'   rather than a score of zero. The returned weights carry that rescaling
#'   (see \code{\link{mgcca_results}}). Only the shared components \code{Y}
#'   are defined for every individual of the union.
#' @param route Per-table algebra route: \code{"auto"} (default) picks per table
#'   by \code{min(n, p)} (covariance \code{X'X} when \code{p < n}, Gram
#'   \code{XX'} when \code{p >= n}); \code{"cov"} or \code{"dual"} force one route
#'   for all tables. The Gram/dual route is the one that scales to \code{p >> n}
#'   (e.g. full methylation) without ever forming a \code{p x p} matrix.
#' @param threads Optional thread count.
#' @param overwrite If TRUE (default), overwrite the file/datasets on import.
#' @param collect If TRUE (default), read the results back from HDF5 and return an
#'   in-memory object of class \code{"mgcca"} (via \code{\link{mgcca_results}})
#'   ready for \code{\link{plotIndividuals}} / \code{\link{plotVariables}} /
#'   \code{\link{getSignif}}; the HDF5 descriptor is kept on its \code{"desc"}
#'   attribute. With \code{collect = FALSE} the lightweight descriptor list is
#'   returned instead (results still live in the HDF5 file). On the in-memory
#'   path (\code{filename = NULL}) \code{collect = FALSE} has no meaning -- there
#'   is no file to leave the results in -- and is refused with a clear error.
#' @param outputs Which components to collect when \code{collect = TRUE}.
#'   \code{NULL} (default) collects all of them -- exactly the behaviour of every
#'   previous version. A character subset of \code{"Y"},
#'   \code{"blocks_present"}, \code{"corsY"}, \code{"scores"}, \code{"pval"},
#'   \code{"weights"}, \code{"scaling"}, \code{"AVE"}, \code{"eigen"},
#'   \code{"overlap"} skips the HDF5 reads behind
#'   the rest; what you do ask for is bit-for-bit identical to a full
#'   collection, because the choice only changes what is read back, never what
#'   was computed. Ignored when \code{collect = FALSE}. See
#'   \code{\link{mgcca_results}}.
#'
#' @return With \code{collect = TRUE} (default), an object of class
#'   \code{"mgcca"} (see \code{\link{mgcca_results}}); its descriptor is on
#'   \code{attr(., "desc")}. With \code{collect = FALSE} (file-backed only), the
#'   descriptor list (\code{filename}, \code{datasets}, \code{nfac}, \code{m},
#'   \code{eig_values}, \code{route}, ...); results still live in the file under
#'   \code{FINAL_RESULTS}. An in-memory fit (\code{filename = NULL}) returns the
#'   same-shaped \code{"mgcca"} object, with a descriptor that reports
#'   \code{backend = "memory"} and \code{filename = NA}; such a fit is not written
#'   to disk and is therefore not reloadable with \code{\link{mgcca_load}}, and
#'   the reliability analyses (\code{\link{mgcca_sensitivity}},
#'   \code{\link{mgcca_stability}}) require a file-backed fit.
#'
#'   Tables need not share the same individuals: they are aligned on the union of
#'   the row names. Per-table quantities are computed on the individuals a table
#'   actually holds -- the p-values in \code{pval} carry that table's
#'   \eqn{n_{present} - 2} degrees of freedom, and its \code{scores} are
#'   standardised over the same individuals.
#'
#' @section Numerical identifiability:
#'   The fit carries the external spectral gap \eqn{\mu_L - \mu_{L+1}} of the
#'   mgcca operator (\code{$eigen}, and \code{attr(., "desc")$eigen}), which is
#'   what separates "the solver converged" from "the leading subspace is
#'   identified". With \code{method = "solve"} the two can come apart: in the
#'   regime where a table has about as many variables as it has observed rows,
#'   the top eigenvalues coincide, the gap falls to machine zero and the
#'   directions the solver returns are arbitrary -- while the solver reports
#'   success. When that happens \code{mgcca} emits a warning of class
#'   \code{"mgcca_degenerate_subspace"} recommending \code{method = "penalized"},
#'   which restores the separation. The warning never changes or aborts the fit.
#' @seealso \code{\link{mgcca_results}}, \code{\link{plotIndividuals}},
#'   \code{\link{mgcca_import_hdf5}}, \code{\link{mgcca_rcpp}}
#' @examples
#' data(cardiovascular)
#' ids <- Reduce(union, list(rownames(X1), rownames(X2), rownames(X3)))[1:150]
#' num <- function(d, cols = seq_len(ncol(d))) {
#'     m <- as.matrix(d[rownames(d) %in% ids, cols, drop = FALSE])
#'     storage.mode(m) <- "double"
#'     m
#' }
#' X <- list(methylation = num(X1, 1:20), clinical = num(X2), other = num(X3))
#'
#' ## In-memory fit: no 'filename', so nothing is written to disk.
#' fit <- mgcca(X, nfac = 2, method = "penalized", lambda = rep(0.1, 3),
#'              scores = TRUE)
#' fit
#' head(fit$Y)
#' fit$AVE$AVE_outer_model
#'
#' ## The same call with a 'filename' runs out-of-core and leaves a
#' ## self-describing HDF5 results file behind (see mgcca_load()).
#' \donttest{
#' h5 <- tempfile(fileext = ".h5")
#' fit2 <- mgcca(X, filename = h5, nfac = 2, method = "penalized",
#'               lambda = rep(0.1, 3))
#' attr(fit2, "desc")$eig_values
#' unlink(h5)
#' }
#' @export
mgcca <- function(x, filename = NULL, group = "MGCCA_IN", datasets = NULL, nfac = 2,
                  scale = TRUE, method = "penalized", lambda = NULL, gamma = 1,
                  scores = FALSE, route = c("auto", "cov", "dual"),
                  threads = NULL, overwrite = TRUE, collect = TRUE,
                  outputs = NULL) {

    route <- match.arg(route)
    inv <- switch(method, solve = 1L, penalized = 2L, geninv = 3L, ginv = 3L,
                  stop("method must be 'solve', 'penalized', 'geninv' or 'ginv'"))
    if (!is.numeric(gamma) || length(gamma) != 1L || !is.finite(gamma) ||
        gamma <= 0)
        stop("'gamma' must be a single finite positive number", call. = FALSE)

    # A penalized fit needs a lambda, and it may either be given or computed.
    # Given, it is used verbatim. Absent, the scale-matched penalty is computed
    # from the data further down -- on the materialised tables for an in-memory
    # fit (.mgcca_memory), and in column chunks off the file for a file-backed
    # one, so a wide table never has to be held in RAM just to be penalised.
    if (inv == 2L && !is.null(lambda)) {
        if (!is.numeric(lambda) || !length(lambda) || anyNA(lambda))
            stop("'lambda' must be numeric and free of NA; leave it NULL to ",
                 "have the scale-matched penalty computed from the data",
                 call. = FALSE)
    }
    # gamma scales a penalty mgcca computes; it has nothing to scale otherwise.
    # Say so rather than drop it silently: a caller who set it meant something.
    if (!(inv == 2L && is.null(lambda)) && !isTRUE(all.equal(gamma, 1))) {
        why <- if (inv != 2L)
            paste0("method '", method, "' is not penalized")
        else "'lambda' was given, and an explicit penalty wins"
        warning("'gamma' is ignored here because ", why, ". It scales the ",
                "penalty only when the penalty is computed, that is with ",
                "method = 'penalized' and 'lambda' left unset.", call. = FALSE)
    }

    # No filename => fit in memory (pure R), for tables that fit in RAM. The
    # file-backed path below is left exactly as it was: it runs only when a
    # filename is supplied, so nothing about it changes bit-for-bit.
    if (is.null(filename))
        return(.mgcca_memory(x, group = group, datasets = datasets, nfac = nfac,
                             scale = scale, method = method, lambda = lambda,
                             gamma = gamma,
                             inv = inv, scores = scores, route = route,
                             collect = collect, outputs = outputs))

    # Import RAW, untransformed: any transformation belongs in HDF5, not in R
    # memory. Scaling (if requested) is done out-of-core by mgcca_rcpp before
    # getK, so it applies uniformly whatever the input type -- including inputs
    # that are already an HDF5 file.
    desc <- mgcca_import_hdf5(x, filename = filename, group = group,
                              datasets = datasets, overwriteFile = overwrite,
                              overwriteDataset = overwrite)

    # The computed penalty, once the tables are in the file and before anything
    # consumes lambda: `lambda` below is what the fit uses AND what the manifest
    # saves, so a reloaded fit reports the penalty it was actually run with.
    if (inv == 2L && is.null(lambda))
        lambda <- .mgcca_default_lambda_hdf5(desc, do_scale = scale,
                                             gamma = gamma)
    # One value given for all tables: recycle it here, as the in-memory path
    # does, so the fit, the manifest and the descriptor all carry one value per
    # table.
    if (inv == 2L && length(lambda) == 1L)
        lambda <- rep(lambda, length(desc$datasets))

    # method = "solve" needs an ordinary inverse of each covariance-route
    # table's cross-product. Refuse a table without one here, naming it, before
    # the fit computes any inverse.
    if (inv == 1L)
        mgcca_solve_rank_guard_rcpp(desc$filename, desc$group, desc$datasets,
                                    scale = isTRUE(scale), route = route)

    res <- mgcca_rcpp(desc$filename, desc$group, desc$datasets,
                      nfac = as.integer(nfac), inv_method = inv, lambda = lambda,
                      scores = scores, scale = isTRUE(scale), route = route,
                      threads = threads)

    # Validity diagnostic of the eigen stage, read back from the file run_eigen
    # wrote it to. Carried on the descriptor so a collect = FALSE caller has it
    # too. It is a REPORT: nothing below depends on it.
    res$eigen <- .mgcca_eigen_report(res$filename)

    # The breakdown warning. With method = "solve" the leading subspace can stop
    # being identified altogether -- the external gap goes to machine zero and
    # the top eigenvectors become arbitrary -- in the regime where a table has
    # about as many variables as observed rows. The solver converges all the
    # same, so nothing else in the pipeline notices. Warn, and only warn: the fit
    # is returned exactly as computed. The other methods reach the user through a
    # different route (`penalized` is the remedy this warning recommends, and
    # `geninv` is already a pseudo-inverse), so the screen is deliberately
    # scoped to `solve`.
    if (identical(method, "solve") && .mgcca_subspace_degenerate(res$eigen))
        .mgcca_warn_degenerate(res$eigen, as.integer(nfac))

    # Persist the provenance manifest as native HDF5 attributes so the results
    # file is self-describing and reloadable via mgcca_load() without this
    # session. Done whether or not we collect, so collect = FALSE files carry it.
    .mgcca_write_manifest(res, method = method, lambda = lambda,
                          input_group = desc$group, scale = scale)

    # The descriptor carried by the returned object must say the same things the
    # manifest says, or the commonest workflow -- fit, then analyse -- would fail
    # while the same fit reloaded from disk succeeded. `mgcca_rcpp` reports where
    # the RESULTS went; the INPUT group and the scaling flag are the caller's and
    # are added here, so the reliability functions work on a fresh fit too.
    res$input_group <- desc$group
    res$scale       <- isTRUE(scale)
    # The penalty actually used, given or computed. The manifest saves it and
    # mgcca_load() rebuilds `desc$lambda` from there, so a fresh fit that did
    # not carry it was the one descriptor shape that disagreed with a reloaded
    # one -- and a caller refitting in a loop (mgcca_permtest) had no way to
    # reuse a computed penalty instead of recomputing it every time.
    res$lambda      <- if (inv == 2L) lambda else NULL

    if (!isTRUE(collect))
        return(res)

    obj <- mgcca_results(res, outputs = outputs)
    attr(obj, "desc") <- res
    obj
}
