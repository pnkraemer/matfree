"""Test the partial Cholesky decompositions."""

from matfree import low_rank, test_util
from matfree.backend import linalg, np, prng, testing
from matfree.backend.typing import Callable


def case_cholesky_partial():
    return low_rank.cholesky_partial


def case_cholesky_partial_pivot():
    return low_rank.cholesky_partial_pivot


@testing.parametrize_with_cases("cholesky", cases=".", prefix="case_cholesky")
def test_full_rank_cholesky_reconstructs_matrix(cholesky, n=5):
    key = prng.prng_key(2)

    cov_eig = 1.0 + prng.uniform(key, shape=(n,), dtype=float)
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation, _info = cholesky(rank=n)(lambda i, j: cov[i, j], n)

    tol = np.finfo_eps(approximation.dtype)
    assert np.allclose(approximation @ approximation.T, cov, atol=tol, rtol=tol)


@testing.parametrize_with_cases("cholesky", cases=".", prefix="case_cholesky")
def test_output_the_right_shapes(cholesky: Callable, n=4, rank=4):
    key = prng.prng_key(1)

    cov_eig = 0.1 + prng.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation, _info = cholesky(rank=rank)(lambda i, j: cov[i, j], n)
    assert approximation.shape == (n, rank)


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
