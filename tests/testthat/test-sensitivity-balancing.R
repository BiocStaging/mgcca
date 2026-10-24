# The sensitivity layer reads a BALANCED operator: each block's summand is
# weighted by its top-L weight before the retained subspace is taken.
#
# The reference below is written from the formulas alone, in plain base R, and
# shares no code with the package: it standardises the tables itself, builds
# the participant Grams itself, and takes the ridge, the ceilings, the weights,
# the balanced sum, its top-L subspace and the first-order sensitivity from
# their definitions:
#
#   G_j     = Z_j Z_j'  (Z_j standardised over its measured rows, zero elsewhere)
#   lambda_j = gamma * mean positive eigenvalue of G_j
#   R_j     = (G_j + lambda_j I)^-1,  D = diag(K_ii^-1/2)
#   A_j     = D G_j R_j D,  c_jL = sum of its L largest eigenvalues
#   w_j     = J c_jL^-1 / sum_h c_hL^-1          (weights average one)
#   S       = D (sum_j w_j G_j R_j) D,  V_L = its top-L eigenvectors
#
# and the block-j first-order term carries w_j next to lambda_j, with both held
# fixed. Both backends of the layer must reproduce it at machine precision on a
# small fixture with missing rows, and so must the participant-space subspace
# refit kernel that computes the same balanced subspace from Grams in a file.
#
# The fixture is built so that balancing MATTERS: the blocks have very
# different widths, the weights are far from one, and the balanced subspace is
# measurably away from the unweighted one. A test that passed because every
# weight was one would prove nothing.

h5_cleanup_on_exit()

BAL_TOL <- 1e-12                       # subspace, weights, ridge, alignment

bal_cache <- new.env(parent = emptyenv())

bal_fixture <- function() {
    if (!is.null(bal_cache$fx)) return(bal_cache$fx)
    set.seed(20261022)
    n <- 36L; L <- 2L
    ids <- sprintf("p%02d", seq_len(n))
    p <- c(narrow = 3L, middle = 8L, wide = 20L)
    Z <- matrix(stats::rnorm(n * L), n, L)
    scale_k <- c(narrow = 1, middle = 3, wide = 0.5)
    X <- lapply(names(p), function(b) {
        M <- Z %*% matrix(stats::rnorm(L * p[[b]], sd = scale_k[[b]]), L, p[[b]]) +
            matrix(stats::rnorm(n * p[[b]]), n, p[[b]])
        dimnames(M) <- list(ids, sprintf("%s_%02d", b, seq_len(p[[b]])))
        M
    })
    names(X) <- names(p)
    keep <- list(narrow = rep(TRUE, n),
                 middle = !(ids %in% ids[c(3, 9, 10, 22, 30)]),
                 wide   = !(ids %in% ids[c(1, 2, 15, 16, 17, 33, 36)]))
    tabs <- Map(function(M, k) M[k, , drop = FALSE], X, keep)
    h5 <- h5_tmp()
    fit <- suppressMessages(mgcca(tabs, filename = h5, nfac = L,
                                  method = "penalized", lambda = rep(1, 3)))
    BigDataStatMeth::hdf5_close_all()
    fx <- list(tabs = tabs, fit = fit, h5 = h5, L = L, ids = rownames(fit$Y),
               query = stats::setNames(as.numeric(Z[, 1]) + stats::rnorm(n, 0, 0.5), ids),
               group = stats::setNames(rep(c("u", "v", "w"), length.out = n), ids))
    bal_cache$fx <- fx
    fx
}

# The reference, from the definitions. `lambda = NULL` uses the gamma rule.
bal_reference <- function(tabs, ids, L, lambda = NULL, gamma = 1) {
    n <- length(ids); J <- length(tabs)
    present <- lapply(tabs, function(M) ids %in% rownames(M))
    G <- lapply(tabs, function(M) {
        Zs <- base::scale(M, center = TRUE, scale = TRUE)
        g <- matrix(0, n, n)
        i <- match(rownames(M), ids)
        g[i, i] <- base::tcrossprod(Zs)
        g
    })
    lam <- if (is.null(lambda)) vapply(G, function(g) {
        ev <- eigen(g, symmetric = TRUE, only.values = TRUE)$values
        gamma * sum(pmax(ev, 0)) / sum(ev > 1e-8 * max(ev))
    }, numeric(1)) else rep_len(as.numeric(lambda), J)
    K <- Reduce(`+`, lapply(present, as.numeric))
    d <- 1 / sqrt(K)
    R <- lapply(seq_len(J), function(j) solve(G[[j]] + lam[j] * diag(n)))
    Mj <- lapply(seq_len(J), function(j) G[[j]] %*% R[[j]])
    cL <- vapply(seq_len(J), function(j) {
        A <- outer(d, d) * Mj[[j]]; A <- (A + t(A)) / 2
        sum(eigen(A, symmetric = TRUE, only.values = TRUE)$values[seq_len(L)])
    }, numeric(1))
    w <- J * (1 / cL) / sum(1 / cL)
    side <- function(wt) {
        S <- outer(d, d) * Reduce(`+`, Map(`*`, wt, Mj)); S <- (S + t(S)) / 2
        eigen(S, symmetric = TRUE)
    }
    e <- side(w)
    list(G = G, present = present, lambda = lam, d = d, R = R, cL = cL, w = w,
         V = e$vectors, mu = e$values, V0 = side(rep(1, J))$vectors[, seq_len(L)])
}

# 1 - mean squared canonical cosine between two column spaces.
bal_ds <- function(A, B) {
    qa <- qr.Q(qr(A)); qb <- qr.Q(qr(B))
    1 - mean(base::svd(base::crossprod(qa, qb))$d^2)
}

# The first-order sensitivity per block, from the definitions: the alignment
# T = ||V_L' z*||^2 / ||z*||^2, its eigenprojector derivative, and block j's
# term w_j lambda_j D R_j dG_j R_j D -- contracted over the grouped set exactly
# as the layer documents it (total and between-group parts).
bal_sensitivity <- function(ref, L, zs, grp) {
    n <- length(zs); V <- ref$V; mu <- ref$mu
    iL <- seq_len(L); iR <- (L + 1L):n
    u <- as.numeric(base::crossprod(V, zs)); den <- sum(zs^2)
    C <- 2 * outer(u[iL], u[iR]) / den / outer(mu[iL], mu[iR], `-`)
    DV <- ref$d * V
    g <- as.integer(factor(grp))
    t(vapply(seq_along(ref$G), function(j) {
        Rr <- ref$R[[j]] %*% DV[, iL, drop = FALSE]
        Sm <- ref$R[[j]] %*% DV[, iR, drop = FALSE]
        A0 <- Rr %*% C %*% t(Sm)
        H  <- ref$w[j] * ref$lambda[j] * (A0 + t(A0))
        Tm <- ref$G[[j]] %*% H
        Tm[, !ref$present[[j]]] <- 0
        cm <- colMeans(Tm)
        accX <- Tm - matrix(cm, n, n, byrow = TRUE)
        gm <- base::rowsum(Tm, g) / as.vector(table(g))
        accB <- gm[g, , drop = FALSE] - matrix(cm, n, n, byrow = TRUE)
        c(S_total = sum(accX^2), S_between = sum(accB^2))
    }, numeric(2)))
}

bal_check <- function(s, ref, fx, label) {
    L <- fx$L
    expect_lte(max(abs(s$settings$lambda - ref$lambda) / ref$lambda), BAL_TOL,
               label = paste(label, "ridge"))
    expect_lte(max(abs(s$balancing$weight - ref$w)), BAL_TOL,
               label = paste(label, "weights"))
    expect_lte(max(abs(s$balancing$ceiling - ref$cL) / ref$cL), BAL_TOL,
               label = paste(label, "ceilings"))
    expect_lte(abs(mean(s$balancing$weight) - 1), BAL_TOL,
               label = paste(label, "weights average one"))
    expect_lte(bal_ds(s$reference$Y, ref$d * ref$V[, seq_len(L)]), BAL_TOL,
               label = paste(label, "subspace"))
    zs <- fx$query[fx$ids] / ref$d
    VL <- ref$V[, seq_len(L)]
    T_ref <- sum(base::crossprod(VL, zs)^2) / sum(zs^2)
    expect_lte(abs(s$overall$T - T_ref), BAL_TOL, label = paste(label, "alignment"))
    sens <- bal_sensitivity(ref, L, zs, fx$group[fx$ids])
    bb <- s$by_block[match(names(fx$tabs), s$by_block$block), ]
    expect_lte(max(abs(bb$S_total - sens[, "S_total"]) / max(sens[, "S_total"])), 1e-10,
               label = paste(label, "per-block total sensitivity"))
    expect_lte(max(abs(bb$S_between - sens[, "S_between"]) / max(sens[, "S_between"])),
               1e-10, label = paste(label, "per-block between-group sensitivity"))
}

test_that("the fixture is one where balancing changes the answer", {
    fx <- bal_fixture()
    ref <- bal_reference(fx$tabs, fx$ids, fx$L)
    expect_gt(max(ref$w) / min(ref$w), 1.2)
    expect_gt(bal_ds(ref$V[, seq_len(fx$L)], ref$V0), 1e-6)
})

test_that("the memory backend reproduces the balanced reference", {
    fx <- bal_fixture()
    on.exit(BigDataStatMeth::hdf5_close_all(), add = TRUE)
    ref <- bal_reference(fx$tabs, fx$ids, fx$L)
    s <- suppressMessages(mgcca_sensitivity(fx$fit, fx$query, group = fx$group,
                                            backend = "memory"))
    bal_check(s, ref, fx, "memory")
    expect_identical(s$balancing$block, names(fx$tabs))
    expect_true(all(s$balancing$rank >= fx$L))
})

test_that("the HDF5 backend reproduces the balanced reference, blocked or not", {
    fx <- bal_fixture()
    on.exit(BigDataStatMeth::hdf5_close_all(), add = TRUE)
    ref <- bal_reference(fx$tabs, fx$ids, fx$L)
    h <- suppressMessages(mgcca_sensitivity(fx$fit, fx$query, group = fx$group,
                                            backend = "hdf5"))
    bal_check(h, ref, fx, "hdf5")
    b <- suppressMessages(mgcca_sensitivity(fx$fit, fx$query, group = fx$group,
                                            backend = "hdf5", block_size = 3L))
    bal_check(b, ref, fx, "hdf5 blocked")
})

test_that("a user-supplied ridge is balanced with that ridge", {
    fx <- bal_fixture()
    on.exit(BigDataStatMeth::hdf5_close_all(), add = TRUE)
    lam <- c(0.5, 2, 7)
    ref <- bal_reference(fx$tabs, fx$ids, fx$L, lambda = lam)
    s <- suppressMessages(mgcca_sensitivity(fx$fit, fx$query, group = fx$group,
                                            lambda = lam, backend = "memory"))
    bal_check(s, ref, fx, "user ridge")
    expect_identical(s$settings$lambda_source, "user")
})

test_that("the layer agrees with the subspace refit kernel run from a file", {
    fx <- bal_fixture()
    ref <- bal_reference(fx$tabs, fx$ids, fx$L)
    gfile <- h5_tmp()
    nm <- names(fx$tabs)
    for (j in seq_along(nm))
        BigDataStatMeth::bdCreate_hdf5_matrix(gfile, ref$G[[j]], group = "G",
                                              dataset = nm[j],
                                              overwriteFile = (j == 1L),
                                              overwriteDataset = TRUE)
    BigDataStatMeth::hdf5_close_all()
    masks <- vapply(ref$present, as.integer, integer(length(fx$ids)))
    k <- mgcca:::reliability_subspace_fit(gfile, "G", nm, masks, fx$L, 1)
    expect_true(k$eligible)
    s <- suppressMessages(mgcca_sensitivity(fx$fit, fx$query, backend = "memory"))
    BigDataStatMeth::hdf5_close_all()
    expect_lte(max(abs(s$settings$lambda - k$lambda) / k$lambda), BAL_TOL)
    expect_lte(max(abs(s$balancing$weight - k$w)), BAL_TOL)
    expect_lte(max(abs(s$balancing$ceiling - k$cL) / k$cL), BAL_TOL)
    expect_lte(bal_ds(s$reference$Y, k$D * k$V), BAL_TOL)
})

test_that("the result and the print report the block weights", {
    fx <- bal_fixture()
    on.exit(BigDataStatMeth::hdf5_close_all(), add = TRUE)
    s <- suppressMessages(mgcca_sensitivity(fx$fit, fx$query, backend = "memory"))
    expect_named(s$balancing, c("block", "ceiling", "weight", "rank"))
    expect_output(print(s), "balancing : top-L block weights")
    # the weights depend on the blocks only, never on the query
    s2 <- suppressMessages(mgcca_sensitivity(fx$fit, -3 * fx$query, backend = "memory"))
    expect_identical(s2$balancing, s$balancing)
})

test_that("a block whose summand cannot reach rank L is refused by name", {
    set.seed(7)
    n <- 30L; ids <- sprintf("q%02d", seq_len(n))
    tabs <- list(one = matrix(stats::rnorm(n), n, 1, dimnames = list(ids, "v1")),
                 two = matrix(stats::rnorm(n * 6), n, 6,
                              dimnames = list(ids, sprintf("w%d", 1:6))))
    fit <- suppressMessages(mgcca(tabs, filename = h5_tmp(), nfac = 2L,
                                  method = "penalized", lambda = c(1, 1)))
    BigDataStatMeth::hdf5_close_all()
    q <- stats::setNames(stats::rnorm(n), ids)
    expect_error(suppressMessages(mgcca_sensitivity(fit, q, lambda = c(1, 1),
                                                    backend = "memory")),
                 "block 'one' cannot be balanced")
    # the refusal gives the held file back
    expect_null(mgcca:::.mgcca_held_file$path)
})
