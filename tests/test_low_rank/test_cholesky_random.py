"""Test the randomly pivoted partial Cholesky decomposition (RPCholesky)."""

from matfree import low_rank, test_util
from matfree.backend import func, linalg, np, prng, testing


def _psd_low_rank(n, rank, key, *, dtype=float):
    eigvals = np.concatenate([1.0 + 1.0 * np.arange(rank), np.zeros((n - rank,))])
    return test_util.hermitian_matrix_from_eigenvalues(eigvals, key, dtype=dtype)


@testing.parametrize("seed", [0, 1, 2])
@testing.parametrize("block_size", [1, 2, 4])
@testing.parametrize("dtype", [float, complex])
def test_full_rank_reconstructs_matrix(seed, block_size, dtype):
    n = 8
    key_eig, key_mat, key_chol = prng.split(prng.prng_key(seed), num=3)
    cov_eig = 1.0 + prng.uniform(key_eig, shape=(n,), dtype=float)
    cov = test_util.hermitian_matrix_from_eigenvalues(cov_eig, key_mat, dtype=dtype)

    cholesky = low_rank.cholesky_partial_pivot_random(
        lambda i, j: cov[i, j], nrows=n, rank=n, block_size=block_size
    )
    L, info = cholesky(key_chol)

    assert L.shape == (n, n)
    assert L.dtype == cov.dtype
    assert info["success"]
    assert info["rank"] == n
    tol = 100 * np.finfo_eps(L.real.dtype)
    assert np.allclose(L @ L.T.conj(), cov, atol=tol, rtol=tol)


@testing.parametrize("seed", [0, 1, 2])
@testing.parametrize("block_size", [1, 3])
@testing.parametrize("dtype", [float, complex])
def test_low_rank_matrix_is_recovered_and_stops_early(seed, block_size, dtype):
    n, true_rank, rank = 12, 3, 9
    key_mat, key_chol = prng.split(prng.prng_key(seed))
    cov = _psd_low_rank(n, true_rank, key_mat, dtype=dtype)
    # After the true rank, only rounding is left.
    tol = 1e4 * np.finfo_eps(cov.real.dtype)

    cholesky = low_rank.cholesky_partial_pivot_random(
        lambda i, j: cov[i, j], nrows=n, rank=rank, block_size=block_size, rtol=tol
    )
    L, info = cholesky(key_chol)

    # Exact after the true rank (rounded up to a whole block), then stop.
    assert info["rank"] == block_size * (-(-true_rank // block_size))
    assert info["trace_residual"] <= tol * info["trace"]
    assert np.allclose(L @ L.T.conj(), cov, atol=tol)
    # Columns beyond the computed rank are zero; their pivots are -1.
    assert np.all(L[:, info["rank"] :] == 0.0)
    assert np.all(info["pivots"][info["rank"] :] == -1)


@testing.parametrize("seed", [0, 1])
@testing.parametrize("dtype", [float, complex])
def test_pivots_are_distinct_and_trace_residual_is_consistent(seed, dtype):
    n, rank, block_size = 10, 6, 2
    key_eig, key_mat, key_chol = prng.split(prng.prng_key(seed), num=3)
    cov_eig = 0.1 + prng.uniform(key_eig, shape=(n,))
    cov = test_util.hermitian_matrix_from_eigenvalues(cov_eig, key_mat, dtype=dtype)

    cholesky = low_rank.cholesky_partial_pivot_random(
        lambda i, j: cov[i, j], nrows=n, rank=rank, block_size=block_size
    )
    L, info = cholesky(key_chol)

    pivots = info["pivots"]
    assert len(set(int(p) for p in pivots)) == rank
    expected = linalg.trace(cov - L @ L.T.conj()).real
    assert np.allclose(info["trace_residual"], expected, rtol=1e-6)
    assert np.allclose(info["trace"], linalg.trace(cov).real)
    # The factor interpolates the matrix on the pivot rows.
    assert np.allclose((L @ L.T.conj())[pivots], cov[pivots], atol=1e-6)


def test_different_keys_give_different_pivots():
    n, rank = 20, 4
    key_eig, key_mat = prng.split(prng.prng_key(3))
    cov = test_util.hermitian_matrix_from_eigenvalues(
        1.0 + prng.uniform(key_eig, shape=(n,)), key_mat
    )
    cholesky = low_rank.cholesky_partial_pivot_random(
        lambda i, j: cov[i, j], nrows=n, rank=rank
    )
    _, info1 = cholesky(prng.prng_key(0))
    _, info2 = cholesky(prng.prng_key(1))
    assert not np.all(info1["pivots"] == info2["pivots"])


def test_parameters_are_passed_to_the_matrix_elements():
    n = 6
    key_mat, key_chol = prng.split(prng.prng_key(4))
    cov = test_util.hermitian_matrix_from_eigenvalues(1.0 + np.arange(n * 1.0), key_mat)

    def element(i, j, scale):
        return scale * cov[i, j]

    cholesky = low_rank.cholesky_partial_pivot_random(element, nrows=n, rank=n)
    L, _info = cholesky(key_chol, 2.0)
    assert np.allclose(L @ L.T, 2.0 * cov, atol=1e-6)


def test_preconditioner_solves_correctly():
    n = 10
    cov_eig = 1.5 ** np.arange(-n // 2, n // 2, step=1.0)
    cov = test_util.hermitian_matrix_from_eigenvalues(cov_eig, prng.prng_key(1))
    cholesky = low_rank.cholesky_partial_pivot_random(
        lambda i, j: cov[i, j], nrows=n, rank=n, block_size=2
    )

    small_value = 1e-1
    b = np.arange(1.0, 1 + n)
    b /= linalg.vector_norm(b)
    expected = linalg.solve(cov + small_value * np.eye(n), b)

    precondition = low_rank.preconditioner(cholesky)
    received, _info = precondition(b, small_value, prng.prng_key(5))
    assert np.allclose(received, expected, rtol=1e-2)


def test_jacobian_gramian_of_a_small_network():
    # Gauss-Newton Gramian J^T J of a tiny network: numerically low rank.
    key_w, key_x, key_chol = prng.split(prng.prng_key(6), num=3)
    n_in, width, n_points = 2, 6, 40
    n_params = width * n_in + width + width + 1
    theta = prng.normal(key_w, shape=(n_params,))
    x = prng.uniform(key_x, shape=(n_points, n_in))

    def net(theta, x):
        W1 = np.reshape(theta[: width * n_in], (width, n_in))
        b1 = theta[width * n_in : width * n_in + width]
        w2 = theta[width * n_in + width : width * n_in + 2 * width]
        return linalg.inner(w2, np.sin(W1 @ x + b1)) + theta[-1]

    J = func.vmap(lambda xi: func.jacfwd(net)(theta, xi))(x)  # (n_points, n_params)
    gram = J.T @ J / n_points

    tol = 1e4 * np.finfo_eps(gram.dtype)
    cholesky = low_rank.cholesky_partial_pivot_random(
        lambda i, j: gram[i, j], nrows=n_params, rank=n_params, block_size=1, rtol=tol
    )
    L, info = cholesky(key_chol)
    assert info["success"]
    assert info["rank"] < n_params  # numerically low rank: stops early
    assert info["trace_residual"] <= tol * info["trace"]
    assert np.allclose(L @ L.T, gram, atol=10 * tol * info["trace"])
