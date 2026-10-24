# method = "solve" on the covariance route needs an ordinary inverse of each
# table's cross-product X'KX. A table with m_j <= p_j < n is still sent to that
# route (p_j < n), and there X'KX is singular. Before this guard the Cholesky
# inverse was attempted anyway; now the fit stops before it, names the table
# and says what to do, on both backends and with the same words.
#
# The fixtures are tiny: 30 individuals, a healthy table measured on all of
# them, and a second table measured on 10.

h5_cleanup_on_exit()

sr_tables <- function(p_bad, m_bad = 10L, dup = FALSE, seed = 11L) {
    set.seed(seed)
    n <- 30L; ids <- sprintf("r%02d", seq_len(n))
    good <- matrix(stats::rnorm(n * 4L), n, 4L,
                   dimnames = list(ids, sprintf("g%d", 1:4)))
    bad <- matrix(stats::rnorm(m_bad * p_bad), m_bad, p_bad,
                  dimnames = list(ids[seq_len(m_bad)], sprintf("b%02d", seq_len(p_bad))))
    if (dup) bad[, p_bad] <- bad[, 1L]
    list(good = good, bad = bad)
}

sr_hdf5 <- function(tabs, ...) {
    on.exit(try(BigDataStatMeth::hdf5_close_all(), silent = TRUE), add = TRUE)
    suppressMessages(mgcca(tabs, filename = h5_tmp(), nfac = 2L, ...))
}
sr_memory <- function(tabs, ...) suppressMessages(mgcca(tabs, nfac = 2L, ...))

sr_message <- function(expr) {
    m <- tryCatch({ expr; NA_character_ }, error = conditionMessage)
    sub("^mgcca_rcpp error: ", "", m)
}

test_that("a table with more variables than measured rows is refused by name", {
    tabs <- sr_tables(p_bad = 12L)               # m = 10 <= p = 12 < n = 30
    for (run in list(sr_hdf5, sr_memory)) {
        msg <- sr_message(run(tabs, method = "solve"))
        expect_match(msg, "table 'bad' does not have one", fixed = TRUE)
        expect_match(msg, "12 variables but only 10 measured individuals", fixed = TRUE)
        expect_match(msg, "method = \"penalized\"", fixed = TRUE)
        expect_match(msg, "method = \"geninv\"", fixed = TRUE)
    }
    # the two backends say exactly the same thing
    expect_identical(sr_message(sr_hdf5(tabs, method = "solve")),
                     sr_message(sr_memory(tabs, method = "solve")))
})

test_that("a table that is singular once centred is refused by its rank", {
    tabs <- sr_tables(p_bad = 10L)               # p = m = 10: rank 9 after centring
    for (run in list(sr_hdf5, sr_memory)) {
        msg <- sr_message(run(tabs, method = "solve"))
        expect_match(msg, "table 'bad' does not have one", fixed = TRUE)
        expect_match(msg, "numerical rank 9 on its 10 measured individuals", fixed = TRUE)
    }
})

test_that("a repeated variable is refused even with enough measured rows", {
    tabs <- sr_tables(p_bad = 5L, m_bad = 20L, dup = TRUE)
    for (run in list(sr_hdf5, sr_memory)) {
        msg <- sr_message(run(tabs, method = "solve"))
        expect_match(msg, "numerical rank 4 on its 20 measured individuals", fixed = TRUE)
    }
})

test_that("the remedies the message names do run on the same tables", {
    tabs <- sr_tables(p_bad = 12L)
    pen <- sr_hdf5(tabs, method = "penalized", lambda = c(1, 1))
    expect_true(all(is.finite(pen$Y)))
    gi <- sr_hdf5(tabs, method = "geninv")
    expect_true(all(is.finite(gi$Y)))
    # and a full-rank table is not touched by the guard
    ok <- sr_hdf5(sr_tables(p_bad = 5L, m_bad = 20L), method = "solve")
    expect_true(all(is.finite(ok$Y)))
})
