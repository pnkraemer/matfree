"""Tests for Lanczos functionality."""

from matfree import decomp, test_util
from matfree.backend import linalg, np, prng, testing


@testing.fixture()
def A(n, num_significant_eigvals):
    """Make a positive definite matrix with certain spectrum."""
    # 'Invent' a spectrum. Use the number of pre-defined eigenvalues.
    d = np.arange(n) + 10.0
    d = d.at[num_significant_eigvals:].set(0.001)

    return test_util.symmetric_matrix_from_eigenvalues(d)


@testing.parametrize("n", [6])
@testing.parametrize("num_significant_eigvals", [6])
def test_max_order(A):
    """If m == n, the matrix should be equal to the full tridiagonal."""
    n, _ = np.shape(A)
    order = n - 1
    key = prng.prng_key(1)
    v0 = prng.normal(key, shape=(n,))
    alg = decomp.lanczos_tridiag_full_reortho(order)
    Q, (d_m, e_m) = decomp.decompose_fori_loop(v0, lambda v: A @ v, algorithm=alg)

    # Lanczos is not stable.
    tols_decomp = {"atol": 1e-5, "rtol": 1e-5}

    # Since full-order mode: Q must be unitary
    assert np.shape(Q) == (order + 1, n)
    assert np.allclose(Q @ Q.T, np.eye(n), **tols_decomp), Q @ Q.T
    assert np.allclose(Q.T @ Q, np.eye(n), **tols_decomp), Q.T @ Q

    # T = Q A Qt
    T = _sym_tridiagonal_dense(d_m, e_m)
    QAQt = Q @ A @ Q.T
    assert np.shape(T) == (order + 1, order + 1)

    # Fail early if the (off)diagonals don't coincide
    assert np.allclose(linalg.diagonal(QAQt), d_m, **tols_decomp)
    assert np.allclose(linalg.diagonal(QAQt, 1), e_m, **tols_decomp)
    assert np.allclose(linalg.diagonal(QAQt, -1), e_m, **tols_decomp)

    # Test the full decomposition
    # (i.e. assert that the off-tridiagonal elements are actually small)
    # be loose with this test. off-diagonal elements accumulate quickly.
    tols_decomp = {"atol": 1e-5, "rtol": 1e-5}
    assert np.allclose(QAQt, T, **tols_decomp)

    # Since full-order mode: Qt T Q = A
    # Since Q is unitary and T = Q A Qt, this test
    # should always pass.
    assert np.allclose(Q.T @ T @ Q, A, **tols_decomp)


@testing.parametrize("n", [50])
@testing.parametrize("num_significant_eigvals", [4])
@testing.parametrize("order", [6])  # ~1.5 * num_significant_eigvals
def test_identity(A, order):
    """Test that Lanczos tridiagonalisation yields an orthogonal-tridiagonal decomp."""
    n, _ = np.shape(A)
    key = prng.prng_key(1)
    v0 = prng.normal(key, shape=(n,))
    alg = decomp.lanczos_tridiag_full_reortho(order)
    Q, tridiag = decomp.decompose_fori_loop(v0, lambda v: A @ v, algorithm=alg)
    (d_m, e_m) = tridiag

    # Lanczos is not stable.
    tols_decomp = {"atol": 1e-5, "rtol": 1e-5}

    assert np.shape(Q) == (order + 1, n)
    assert np.allclose(Q @ Q.T, np.eye(order + 1), **tols_decomp), Q @ Q.T

    # T = Q A Qt
    T = _sym_tridiagonal_dense(d_m, e_m)
    QAQt = Q @ A @ Q.T
    assert np.shape(T) == (order + 1, order + 1)

    # Fail early if the (off)diagonals don't coincide
    assert np.allclose(linalg.diagonal(QAQt), d_m, **tols_decomp)
    assert np.allclose(linalg.diagonal(QAQt, 1), e_m, **tols_decomp)
    assert np.allclose(linalg.diagonal(QAQt, -1), e_m, **tols_decomp)

    # Test the full decomposition
    assert np.allclose(QAQt, T, **tols_decomp)


def _sym_tridiagonal_dense(d, e):
    diag = linalg.diagonal_matrix(d)
    offdiag1 = linalg.diagonal_matrix(e, 1)
    offdiag2 = linalg.diagonal_matrix(e, -1)
    return diag + offdiag1 + offdiag2
