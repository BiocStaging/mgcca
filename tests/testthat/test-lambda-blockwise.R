# =============================================================================
# mgcca_lambda_scale_hdf5() -- the scale-matched block scale accumulated over
# HDF5 column chunks, against the package's own dense rule.
#
# WHAT IS BEING ASSERTED, AND WHY IT IS A TOLERANCE AND NOT AN IDENTITY. The
# chunked route and the dense rule differ in the SUMMATION ORDER of the Gram (K
# accumulated chunk products instead of one BLAS call over all p columns) and in
# nothing else algebraic, so they agree to floating point and not bitwise. The
# tolerances below are the ones pre-declared for the same algebra before it was
# certified on the real HELIX blocks (up to 299,221 columns), where the worst
# relative difference against the dense values came out at 1.4e-15 and
# invariance across chunk sizes at 6.3e-16 -- four to five orders of magnitude
# inside the declared tolerances, so a pass here is not a near-miss.
#
# Fixtures are built in tempdir() through the package's own importer; nothing
# is shipped.
# =============================================================================

TOL_DENSE <- 1e-10        # chunked vs the dense rule
TOL_INVAR <- 1e-13        # across chunk sizes

h5_cleanup_on_exit()

## ---- fixture: write a named list of matrices to a fresh HDF5 file ----------
lb_write <- function(tabs) {
    h5 <- h5_tmp()
    desc <- mgcca_import_hdf5(tabs, filename = h5, overwriteFile = TRUE)
    desc
}

## The dense rule, reached through the package's own internals so that the
## comparison is against the code that ships and not a copy of it.
lb_dense <- function(tabs, do_scale = TRUE, rank_tol = 1e-8, sd_tol = 1e-8) {
    f <- getFromNamespace(".mgcca_lambda_scale", "mgcca")
    stats::setNames(f(tabs, do_scale = do_scale, rank_tol = rank_tol,
                      sd_tol = sd_tol), names(tabs))
}

lb_rel <- function(a, b) max(abs(a - b) / pmax(abs(b), 1e-2))

## Blocks of assorted shape and content. `ids` are shared so the importer is
## happy; the rule itself is per block.
lb_tables <- function(seed = 20261009L, n = 60L) {
    set.seed(seed)
    ids <- sprintf("ind%04d", seq_len(n))
    mk <- function(p, f = identity) {
        M <- matrix(stats::rnorm(n * p), n, p)
        dimnames(M) <- list(ids, sprintf("v%d", seq_len(p)))
        f(M)
    }
    list(
        narrow = mk(7L),
        wide   = mk(5000L),                       # the p >> n block that matters
        rankdef = mk(40L, function(M) {           # exact dependence, exercises rank
            M[, 40L] <- M[, 1L]; M[, 39L] <- M[, 2L] + M[, 3L]; M }))
}


test_that("the chunked scale matches the dense rule, wide block included", {
    tabs <- lb_tables()
    desc <- lb_write(tabs)

    den <- lb_dense(tabs)
    got <- mgcca_lambda_scale_hdf5(desc$filename, desc$group, desc$datasets,
                                   chunk_cols = 512L)

    expect_named(got$s, desc$datasets)
    expect_true(all(is.finite(got$s)))
    expect_true(all(got$s > 0))
    # the wide block is the one the dense rule cannot afford at scale
    expect_equal(got$details$p[got$details$dataset == "wide"], 5000)
    expect_gt(got$details$n_chunks[got$details$dataset == "wide"], 1)

    rel <- lb_rel(got$s[desc$datasets], den[desc$datasets])
    expect_lt(rel, TOL_DENSE)
    cat(sprintf("\n  chunked vs dense: worst relative difference %.3e (tol %.1e)\n",
                rel, TOL_DENSE))

    # lambda is gamma * s, and gamma may be per block
    expect_equal(unname(got$lambda), unname(got$s))
    g <- c(0.5, 2, 3)
    got2 <- mgcca_lambda_scale_hdf5(desc$filename, desc$group, desc$datasets,
                                    gamma = g, chunk_cols = 512L)
    expect_equal(unname(got2$lambda), unname(g * got2$s))
})


test_that("the scale is invariant to the chunk size", {
    tabs <- lb_tables()
    desc <- lb_write(tabs)

    widths <- c(1L, 13L, 512L, 4999L, 5000L, 10000L)   # includes >= p
    S <- vapply(widths, function(w)
        mgcca_lambda_scale_hdf5(desc$filename, desc$group, desc$datasets,
                                chunk_cols = w)$s,
        numeric(length(desc$datasets)))

    # global over the declared set: largest spread across widths, per block,
    # against a common reference
    inv <- max(apply(S, 1L, function(v)
        max(abs(outer(v, v, "-"))) / max(abs(v))))
    expect_lt(inv, TOL_INVAR)
    cat(sprintf("  chunk-size invariance over widths {%s}: worst %.3e (tol %.1e)\n",
                paste(widths, collapse = ", "), inv, TOL_INVAR))

    # a chunk at least as wide as the block is a single chunk
    one <- mgcca_lambda_scale_hdf5(desc$filename, desc$group, desc$datasets,
                                   chunk_cols = 10000L)
    expect_true(all(one$details$n_chunks == 1))
    expect_equal(one$details$chunk_cols, one$details$p)
    # and still within tolerance of the dense rule
    expect_lt(lb_rel(one$s, lb_dense(tabs)[desc$datasets]), TOL_DENSE)
})


test_that("constant and non-finite columns are dropped exactly as the dense rule drops them", {
    n <- 60L
    set.seed(4L)
    ids <- sprintf("ind%04d", seq_len(n))
    mk <- function(p) {
        M <- matrix(stats::rnorm(n * p), n, p)
        dimnames(M) <- list(ids, sprintf("v%d", seq_len(p)))
        M
    }
    a <- mk(30L); a[, 1L] <- 1                  # exactly constant -> sd 0
    b <- mk(30L); b[, 2L] <- 0.1                # constant only in intent
    d <- mk(30L); d[3L, 4L] <- NA_real_         # non-finite entry
    e <- mk(30L); e[, 5L] <- 1 + stats::rnorm(n) * 1e-14   # near-constant
    tabs <- list(const_exact = a, const_fp = b, has_na = d, nearconst = e)
    desc <- lb_write(tabs)

    got <- mgcca_lambda_scale_hdf5(desc$filename, desc$group, desc$datasets,
                                   chunk_cols = 7L)
    den <- lb_dense(tabs)

    # every scale is finite: the rule drops the offending column rather than
    # letting a NaN reach the Gram
    expect_true(all(is.finite(got$s)))
    expect_lt(lb_rel(got$s[desc$datasets], den[desc$datasets]), TOL_DENSE)

    # and it drops exactly the columns the dense rule's own keep rule drops
    dense_dropped <- vapply(tabs, function(M) {
        ctr <- colMeans(M); sdv <- apply(M, 2L, stats::sd)
        sum(!(is.finite(ctr) & is.finite(sdv) & sdv > 1e-8))
    }, numeric(1))
    got_dropped <- stats::setNames(got$details$n_dropped, got$details$dataset)
    expect_equal(got_dropped[names(dense_dropped)], dense_dropped,
                 ignore_attr = TRUE)
    # One column each: the exactly constant one, the one holding an NA, and --
    # because sd_tol is 1e-8 and not machine epsilon -- the column that is
    # constant only in floating point (sd 0 here) and the near-constant one
    # (sd ~9e-15). The driver-level rule of script 129, which has no sd_tol,
    # keeps the last two; the package's rule drops them, and this route follows
    # the package.
    expect_equal(unname(got_dropped[c("const_exact", "const_fp", "has_na",
                                      "nearconst")]), c(1, 1, 1, 1))
    expect_equal(unname(got$details$n_kept), rep(29, 4))
})


test_that("the kept/dropped margin against sd_tol is reported", {
    tabs <- lb_tables(n = 40L)
    desc <- lb_write(tabs)
    got <- mgcca_lambda_scale_hdf5(desc$filename, desc$group, desc$datasets)

    # Nothing here is anywhere near the threshold, which is the point: the
    # diagnostic says so per run instead of it being assumed.
    expect_true(all(got$details$sd_min_kept > 1e-8))
    expect_true(all(got$details$sd_max_dropped == 0))
    expect_gt(min(got$details$sd_min_kept / 1e-8), 1e6)

    # the accumulation's own asymmetry, recorded before symmetrising
    expect_true(all(is.finite(got$details$sym_err)))
})


test_that("scale = FALSE reproduces the dense rule's unscaled branch", {
    tabs <- lb_tables(seed = 99L, n = 50L)[c("narrow", "rankdef")]
    desc <- lb_write(tabs)

    den <- lb_dense(tabs, do_scale = FALSE)
    got <- mgcca_lambda_scale_hdf5(desc$filename, desc$group, desc$datasets,
                                   scale = FALSE, chunk_cols = 2L)
    expect_lt(lb_rel(got$s, den[desc$datasets]), TOL_DENSE)
    # no column is dropped on the unscaled branch
    expect_true(all(got$details$n_dropped == 0))
    expect_equal(got$details$n_kept, got$details$p)
})


test_that("rank_tol reaches the rank the way the dense rule does", {
    tabs <- lb_tables()[c("rankdef", "narrow")]
    desc <- lb_write(tabs)
    for (rt in c(1e-8, 1e-2)) {
        got <- mgcca_lambda_scale_hdf5(desc$filename, desc$group,
                                       desc$datasets, rank_tol = rt,
                                       chunk_cols = 9L)
        den <- lb_dense(tabs, rank_tol = rt)
        expect_lt(lb_rel(got$s, den[desc$datasets]), TOL_DENSE)
    }
})


test_that("bad arguments are refused", {
    tabs <- lb_tables(n = 30L)[c("narrow", "rankdef")]
    desc <- lb_write(tabs)
    f <- desc$filename; g <- desc$group; d <- desc$datasets

    expect_error(mgcca_lambda_scale_hdf5(file.path(tempdir(), "no-such.h5")),
                 "does not exist")
    expect_error(mgcca_lambda_scale_hdf5(f, g, d, chunk_cols = 0),
                 "positive integer")
    expect_error(mgcca_lambda_scale_hdf5(f, g, d, chunk_cols = NA),
                 "positive integer")
    expect_error(mgcca_lambda_scale_hdf5(f, g, d, sd_tol = -1),
                 "non-negative")
    expect_error(mgcca_lambda_scale_hdf5(f, g, d, rank_tol = -1),
                 "non-negative")
    expect_error(mgcca_lambda_scale_hdf5(f, g, d, gamma = c(1, 2, 3)),
                 "one value per table")
    for (bad in list(0, -1, Inf, NA_real_, c(1, 0), "1", numeric(0)))
        expect_error(mgcca_lambda_scale_hdf5(f, g, d, gamma = bad),
                     "finite positive")
    expect_error(mgcca_lambda_scale_hdf5(f, g, "not_a_table"),
                 "not in the file")
    expect_error(mgcca_lambda_scale_hdf5(f, "NO_SUCH_GROUP", d),
                 "not in the file")
})


test_that("a block with no usable column is refused, not silently scaled", {
    n <- 40L
    ids <- sprintf("ind%04d", seq_len(n))
    flat <- matrix(1, n, 5L, dimnames = list(ids, sprintf("v%d", 1:5)))
    set.seed(5L)
    ok <- matrix(stats::rnorm(n * 4L), n, 4L,
                 dimnames = list(ids, sprintf("w%d", 1:4)))
    desc <- lb_write(list(flat = flat, ok = ok))
    expect_error(
        mgcca_lambda_scale_hdf5(desc$filename, desc$group, desc$datasets),
        "no column with non-zero variance")
})


test_that("it reads a file mgcca itself holds open", {
    # The reader is mgcca's own HDF5 instance, so a held mgcca handle lets it
    # through -- that open is what the single-handle discipline exists to allow.
    tabs <- lb_tables(n = 30L)[c("narrow", "rankdef")]
    desc <- lb_write(tabs)

    ctx <- new.env(parent = emptyenv()); ctx$file <- desc$filename
    withr::defer(mgcca:::.mgcca_rel_release(ctx))
    ctx$handle <- mgcca:::.mgcca_rel_hold(ctx$file)

    held <- mgcca_lambda_scale_hdf5(desc$filename, desc$group, desc$datasets,
                                    chunk_cols = 5L)
    expect_true(all(is.finite(held$s)))
    expect_lt(lb_rel(held$s, lb_dense(tabs)[desc$datasets]), TOL_DENSE)

    # The listing (datasets = NULL) is mgcca's own too, so it also goes through
    # while the handle is held, and it finds the same tables.
    listed <- mgcca_lambda_scale_hdf5(desc$filename, desc$group,
                                      chunk_cols = 5L)
    expect_setequal(names(listed$s), desc$datasets)
    expect_lt(lb_rel(listed$s[desc$datasets], held$s[desc$datasets]), TOL_INVAR)

    mgcca:::.mgcca_rel_release(ctx)
    # once released, the listing route agrees with the named-datasets route
    auto <- mgcca_lambda_scale_hdf5(desc$filename, desc$group, chunk_cols = 5L)
    expect_lt(lb_rel(auto$s[desc$datasets], held$s[desc$datasets]), TOL_INVAR)
})


test_that("datasets = NULL lists the tables, not the hidden dimnames", {
    tabs <- lb_tables(n = 30L)[c("narrow", "rankdef")]
    desc <- lb_write(tabs)

    # the group really holds hidden entries next to the tables
    raw <- mgcca:::mgcca_list_group_rcpp(desc$filename, desc$group)
    expect_true(any(startsWith(raw, ".")))

    auto  <- mgcca_lambda_scale_hdf5(desc$filename, desc$group, chunk_cols = 5L)
    named <- mgcca_lambda_scale_hdf5(desc$filename, desc$group, desc$datasets,
                                     chunk_cols = 5L)
    expect_setequal(names(auto$s), desc$datasets)
    expect_false(any(startsWith(names(auto$s), ".")))
    expect_equal(auto$s[desc$datasets], named$s[desc$datasets], tolerance = 0)
})


# =============================================================================
# mgcca()'s two ways to a penalty: given, or computed.
# =============================================================================

## A small fixture both fit paths can take: three tables, shared individuals.
lb_fit_tables <- function(n = 40L, seed = 31L) {
    set.seed(seed)
    ids <- sprintf("ind%03d", seq_len(n))
    mk <- function(p) {
        M <- matrix(stats::rnorm(n * p), n, p)
        dimnames(M) <- list(ids, sprintf("v%d", seq_len(p)))
        M
    }
    list(first = mk(6L), second = mk(5L), third = mk(4L))
}


test_that("an explicit lambda is used verbatim on both fit paths", {
    tabs <- lb_fit_tables()
    lam  <- c(0.3, 0.7, 1.1)

    # in memory
    mem <- mgcca(tabs, nfac = 2, method = "penalized", lambda = lam)
    expect_s3_class(mem, "mgcca")

    # file-backed: same lambda, and the descriptor reports it
    h5 <- h5_tmp()
    fb <- mgcca(tabs, filename = h5, nfac = 2, method = "penalized",
                lambda = lam, collect = FALSE)
    expect_equal(unname(fb$lambda), lam)

    # neither path emitted the computed-penalty message
    expect_silent(mgcca(tabs, filename = h5_tmp(), nfac = 2,
                        method = "penalized", lambda = lam, collect = FALSE))
})


test_that("one lambda is recycled to every table on the file-backed path", {
    tabs <- lb_fit_tables()
    J <- length(tabs)

    one <- mgcca(tabs, filename = h5_tmp(), nfac = 2, method = "penalized",
                 lambda = 0.5, collect = FALSE)
    vec <- mgcca(tabs, filename = h5_tmp(), nfac = 2, method = "penalized",
                 lambda = rep(0.5, J), collect = FALSE)

    expect_length(one$lambda, J)
    expect_equal(unname(one$lambda), rep(0.5, J))
    expect_identical(one$eig_values, vec$eig_values)
    expect_identical(mgcca_results(one)$Y, mgcca_results(vec)$Y)
})


test_that("an absent lambda is computed from the tables for an in-memory fit", {
    tabs <- lb_fit_tables()
    dense <- getFromNamespace(".mgcca_lambda_scale", "mgcca")(tabs)

    expect_message(fit <- mgcca(tabs, nfac = 2, method = "penalized"),
                   "scale-matched")
    expect_s3_class(fit, "mgcca")

    # The in-memory route calls the dense rule itself, so the penalty the fit
    # ran with is that rule's value exactly -- no tolerance needed here.
    expect_identical(as.numeric(attr(fit, "desc")$lambda), as.numeric(dense))
})


test_that("an absent lambda is computed in column chunks for a file-backed fit", {
    tabs <- lb_fit_tables()
    dense <- getFromNamespace(".mgcca_lambda_scale", "mgcca")(tabs)

    h5 <- h5_tmp()
    expect_message(fb <- mgcca(tabs, filename = h5, nfac = 2,
                               method = "penalized", collect = FALSE),
                   "scale-matched")
    expect_length(fb$lambda, length(tabs))
    rel <- max(abs(fb$lambda - dense) / abs(dense))
    expect_lt(rel, TOL_DENSE)
    cat(sprintf("  mgcca() computed lambda vs dense rule: %.3e (tol %.1e)\n",
                rel, TOL_DENSE))

    # and the manifest saved what the fit ran with, so a reload agrees
    back <- mgcca_load(h5)
    expect_equal(unname(attr(back, "desc")$lambda[names(fb$lambda)]),
                 unname(fb$lambda), tolerance = 1e-12)
})


test_that("both ways produce a working fit, end to end", {
    tabs <- lb_fit_tables()

    given    <- mgcca(tabs, nfac = 2, method = "penalized",
                      lambda = rep(1, 3))
    computed <- suppressMessages(mgcca(tabs, nfac = 2, method = "penalized"))

    for (fit in list(given, computed)) {
        expect_s3_class(fit, "mgcca")
        Y <- fit$Y
        expect_equal(nrow(Y), nrow(tabs$first))
        expect_equal(ncol(Y), 2L)
        expect_true(all(is.finite(Y)))
        # the eigenvalues of a collected fit live on its descriptor
        ev <- attr(fit, "desc")$eig_values
        expect_length(ev, 2L)
        expect_true(all(is.finite(ev)))
    }
    # a different penalty is a different fit: the computed one is not the
    # lambda = 1 one in disguise
    expect_false(isTRUE(all.equal(attr(given, "desc")$eig_values,
                                  attr(computed, "desc")$eig_values)))
    expect_false(isTRUE(all.equal(given$Y, computed$Y)))

    # file-backed, computed, collected: the full object comes back
    h5 <- h5_tmp()
    obj <- suppressMessages(mgcca(tabs, filename = h5, nfac = 2,
                                  method = "penalized"))
    expect_s3_class(obj, "mgcca")
    expect_true(all(is.finite(obj$Y)))
})


test_that("gamma scales the computed penalty on both routes", {
    tabs <- lb_fit_tables()
    s <- getFromNamespace(".mgcca_lambda_scale", "mgcca")(tabs)

    # in memory: exactly gamma * s, and exactly 2x the gamma = 1 values
    one <- suppressMessages(mgcca(tabs, nfac = 2, method = "penalized"))
    two <- suppressMessages(mgcca(tabs, nfac = 2, method = "penalized",
                                  gamma = 2))
    lam1 <- attr(one, "desc")$lambda
    lam2 <- attr(two, "desc")$lambda
    expect_identical(as.numeric(lam2), as.numeric(2 * s))
    expect_identical(as.numeric(lam2), as.numeric(2 * lam1))

    # file-backed: the same, to the Gram's summation order
    h5a <- h5_tmp(); h5b <- h5_tmp()
    fb1 <- suppressMessages(mgcca(tabs, filename = h5a, nfac = 2,
                                  method = "penalized", collect = FALSE))
    fb2 <- suppressMessages(mgcca(tabs, filename = h5b, nfac = 2,
                                  method = "penalized", gamma = 2,
                                  collect = FALSE))
    expect_lt(lb_rel(fb2$lambda, 2 * s), TOL_DENSE)
    expect_lt(lb_rel(fb2$lambda, 2 * fb1$lambda), TOL_INVAR)

    # the message reports the gamma used, and the final values with it applied
    expect_message(mgcca(tabs, nfac = 2, method = "penalized", gamma = 2),
                   "gamma = 2")

    # and gamma changes the fit, it is not cosmetic
    expect_false(isTRUE(all.equal(attr(one, "desc")$eig_values,
                                  attr(two, "desc")$eig_values)))
})


test_that("the computed penalty is what mgcca_select_lambda() reports for that gamma", {
    tabs <- lb_fit_tables(n = 40L)
    grid <- c(0.5, 1, 2)
    sel <- suppressWarnings(suppressMessages(
        mgcca_select_lambda(tabs, nfac = 2, gamma = grid, K = 3, seed = 7)))

    # lambda_by_candidate is the selector's own outer(grid, s_full): row g is
    # the penalty it would report at that gamma. mgcca() must land on it.
    expect_equal(sort(sel$grid$gamma), sort(grid))
    for (g in grid) {
        fit <- suppressMessages(mgcca(tabs, nfac = 2, method = "penalized",
                                      gamma = g))
        want <- sel$lambda_by_candidate[match(g, sort(grid)), ]
        expect_identical(as.numeric(attr(fit, "desc")$lambda),
                         as.numeric(want))
    }
})


test_that("an explicit lambda wins over gamma, with a warning", {
    tabs <- lb_fit_tables()
    lam  <- c(0.3, 0.7, 1.1)

    expect_warning(fit <- mgcca(tabs, nfac = 2, method = "penalized",
                                lambda = lam, gamma = 2),
                   "'gamma' is ignored")
    # used verbatim: gamma did NOT multiply it
    expect_equal(as.numeric(attr(fit, "desc")$lambda), lam)

    # gamma = 1 alongside an explicit lambda stays silent
    expect_silent(mgcca(tabs, filename = h5_tmp(), nfac = 2,
                        method = "penalized", lambda = lam, gamma = 1,
                        collect = FALSE))

    # a method that has no penalty to scale says so too
    expect_warning(mgcca(tabs, nfac = 2, method = "solve", gamma = 2),
                   "not penalized")
})


test_that("an invalid gamma is refused", {
    tabs <- lb_fit_tables(n = 30L)
    for (bad in list(0, -1, NA_real_, Inf, c(1, 2), "1", NULL))
        expect_error(mgcca(tabs, nfac = 2, method = "penalized", gamma = bad),
                     "single finite positive number")
})


test_that("mgcca_permtest() resolves the penalty once, not per permutation", {
    tabs <- lb_fit_tables(n = 30L)
    # One message, from the observed fit: the permutations reuse its penalty
    # instead of each recomputing an identical one.
    expect_message(
        pt <- mgcca_permtest(tabs, filename = h5_tmp(), nperm = 2, nfac = 2),
        "scale-matched")
    expect_length(pt$p.value, length(pt$eigenvalues))
    expect_true(all(pt$p.value > 0 & pt$p.value <= 1))
})


test_that("mgcca_permtest() at gamma != 1 builds its null at the observed penalty", {
    tabs <- lb_fit_tables(n = 30L)
    s <- getFromNamespace(".mgcca_lambda_scale", "mgcca")(tabs)

    # gamma reaches the observed fit through `...`, and the permutations are then
    # handed that fit's FINAL penalty -- so gamma must not be forwarded with it,
    # or every permutation would warn that gamma is ignored.
    h5 <- h5_tmp()
    expect_message(
        pt <- mgcca_permtest(tabs, filename = h5, nperm = 2, nfac = 2,
                             gamma = 3),
        "gamma = 3")
    expect_length(pt$p.value, length(pt$eigenvalues))

    # the observed fit really used gamma * s, and the manifest says so
    back <- mgcca_load(h5)
    expect_lt(lb_rel(attr(back, "desc")$lambda[names(tabs)], 3 * s), TOL_DENSE)

    # no warning escaped from the permutation refits
    expect_no_warning(
        mgcca_permtest(tabs, filename = h5_tmp(), nperm = 2, nfac = 2,
                       gamma = 3) |> suppressMessages())
})


test_that("the dense rule is unchanged by the shared tail", {
    # .mgcca_lambda_scale() now calls .mgcca_lambda_from_gram() for its tail.
    # Its values and its error message must be what they were.
    tabs <- lb_tables(seed = 7L, n = 40L)["narrow"]
    f  <- getFromNamespace(".mgcca_lambda_scale", "mgcca")
    sb <- getFromNamespace(".mgcca_scale_block", "mgcca")
    tl <- getFromNamespace(".mgcca_lambda_from_gram", "mgcca")

    # the refactor is bit-identical: the dense route IS scale-block + this tail
    expect_identical(f(tabs),
                     tl(tcrossprod(sb(tabs$narrow)$X), 1e-8, "block 1"))
    # and the rank-0 message is the one it always raised (reached on the
    # unscaled branch, where a zero block survives as far as the tail)
    expect_error(f(list(matrix(0, 10L, 3L)), do_scale = FALSE),
                 "block 1 has numerical rank 0; its scale is undefined",
                 fixed = TRUE)
})
