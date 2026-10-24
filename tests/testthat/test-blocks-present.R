# The consensus scores come back with each individual's block count K_ii: the
# number of tables in which the individual was measured. It is read from the
# same presence masks the fit used, so it must equal the count taken from the
# input tables, agree with the NA pattern of the per-table scores, be the same
# on both backends, and follow the selectable-outputs rule like every other
# component.

h5_cleanup_on_exit()

bp_tables <- function() {
    set.seed(5L)
    n <- 25L; ids <- sprintf("k%02d", seq_len(n))
    mk <- function(p, keep) {
        M <- matrix(stats::rnorm(n * p), n, p,
                    dimnames = list(ids, sprintf("v%02d", seq_len(p))))
        M[keep, , drop = FALSE]
    }
    list(t1 = mk(4L, seq_len(n)),
         t2 = mk(3L, -c(2, 7, 11, 19)),
         t3 = mk(5L, -c(2, 3, 20, 21, 22, 25)))
}

bp_cache <- new.env(parent = emptyenv())
bp_fit <- function() {
    if (is.null(bp_cache$f)) {
        bp_cache$f <- suppressMessages(mgcca(bp_tables(), filename = h5_tmp(),
                                             nfac = 2L, method = "penalized",
                                             lambda = rep(1, 3), scores = TRUE))
        BigDataStatMeth::hdf5_close_all()
    }
    bp_cache$f
}

test_that("blocks_present counts the tables each individual was measured in", {
    f <- bp_fit()
    tabs <- bp_tables()
    ids <- rownames(f$Y)
    expected <- base::rowSums(vapply(tabs, function(M) ids %in% rownames(M),
                                     logical(length(ids))))
    expect_type(f$blocks_present, "integer")
    expect_identical(names(f$blocks_present), ids)
    expect_identical(unname(f$blocks_present), as.integer(expected))
    expect_true(all(f$blocks_present >= 1L))
    expect_true(any(f$blocks_present < length(tabs)))   # the fixture has missing rows
})

test_that("blocks_present matches the NA pattern of the per-table scores", {
    f <- bp_fit()
    scored <- base::rowSums(vapply(f$scores, function(S) !is.na(S[, 1]),
                                   logical(nrow(f$Y))))
    expect_identical(unname(f$blocks_present), as.integer(scored))
})

test_that("blocks_present is the same in memory and from the file", {
    f <- bp_fit()
    m <- suppressMessages(mgcca(bp_tables(), nfac = 2L, method = "penalized",
                                lambda = rep(1, 3), scores = TRUE))
    expect_identical(m$blocks_present, f$blocks_present)
})

test_that("blocks_present follows the selectable-outputs rule", {
    f <- bp_fit()
    desc <- attr(f, "desc")
    on.exit(try(BigDataStatMeth::hdf5_close_all(), silent = TRUE), add = TRUE)
    alone <- mgcca_results(desc, outputs = "blocks_present")
    expect_identical(alone$blocks_present, f$blocks_present)
    expect_null(alone$Y)
    onlyY <- mgcca_results(desc, outputs = "Y")
    expect_null(onlyY$blocks_present)
})
