"""Test the partial Cholesky decompositions."""

from matfree import low_rank, test_util
from matfree.backend import linalg, np, prng, testing
from matfree.backend.typing import Callable


@testing.parametrize(
    "low_rank", [low_rank.cholesky_partial, low_rank.cholesky_partial_pivot]
)
def test_full_rank_cholesky_reconstructs_matrix(low_rank, n=5):
    key = prng.prng_key(2)

    cov_eig = 1.0 + prng.uniform(key, shape=(n,), dtype=float)
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation, _info = low_rank(rank=n)(lambda i, j: cov[i, j], n)

    tol = np.finfo_eps(approximation.dtype)
    assert np.allclose(approximation @ approximation.T, cov, atol=tol, rtol=tol)


def test_full_rank_nopivot_matches_cholesky(n=10):
    key = prng.prng_key(2)
    cov_eig = 0.01 + prng.uniform(key, shape=(n,), dtype=float)
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)
    cholesky = linalg.cholesky(cov)

    # Sanity check: pivoting should definitely not satisfy this:
    received, info = low_rank.cholesky_partial_pivot(rank=n)(lambda i, j: cov[i, j], n)
    assert not np.allclose(received, cholesky)

    # But without pivoting, we should get there!
    received, info = low_rank.cholesky_partial(rank=n)(lambda i, j: cov[i, j], n)
    assert np.allclose(received, cholesky, atol=1e-6)


@testing.parametrize(
    "low_rank", [low_rank.cholesky_partial, low_rank.cholesky_partial_pivot]
)
def test_output_the_right_shapes(low_rank: Callable, n=4, rank=4):
    key = prng.prng_key(1)

    cov_eig = 0.1 + prng.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation, _info = low_rank(rank=rank)(lambda i, j: cov[i, j], n)
    assert approximation.shape == (n, rank)


def test_pivoting_improves_the_estimate(n=10, rank=5):
    key = prng.prng_key(1)

    cov_eig = 0.1 + prng.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    def element(i, j):
        return cov[i, j]

    nopivot, _info = low_rank.cholesky_partial(rank=rank)(element, n)
    pivot, _info = low_rank.cholesky_partial_pivot(rank=rank)(element, n)

    error_nopivot = linalg.matrix_norm(cov - nopivot @ nopivot.T, which="fro")
    error_pivot = linalg.matrix_norm(cov - pivot @ pivot.T, which="fro")
    assert error_pivot < error_nopivot


def test_preconditioner_solves_correctly(n=10):
    # Create a relatively ill-conditioned matrix
    cov_eig = 1.5 ** np.arange(-n // 2, n // 2, step=1.0)
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    def element(i, j):
        return cov[i, j]

    # Assert that the Cholesky decomposition is full-rank.
    cholesky = low_rank.cholesky_partial(rank=n)
    matrix, _info = cholesky(element, n)
    assert np.allclose(matrix @ matrix.T, cov)

    # Solve the linear system
    b = np.arange(1.0, 1 + len(cov))
    b /= linalg.vector_norm(b)
    small_value = 1e-1
    cov_added = cov + small_value * np.eye(len(cov))

    expected = linalg.solve(cov_added, b)

    # Derive the preconditioner
    precondition = low_rank.preconditioner(cholesky)
    solve, info = precondition(element, n)

    # Test that the preconditioner solves correctly
    received = solve(b, small_value)
    assert np.allclose(received, expected, rtol=1e-2)
