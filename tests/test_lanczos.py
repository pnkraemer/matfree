"""Tests for Lanczos functionality."""

from hutch import lanczos
from hutch.backend import linalg, np, prng, testing


@testing.fixture
# don't exceed this value of 'n' because lanczos is not very stable.
@testing.parametrize("n", [6])
def A(n):
    A = np.reshape(np.arange(1.0, n**2 + 1.0), (n, n))
    A = A / linalg.norm(A)
    return A.T @ A + np.eye(n)


def test_tridiagonal_error_for_too_high_order(A):
    n, _ = np.shape(A)
    order = n
    key = prng.PRNGKey(1)
    with testing.raises(ValueError):
        _ = lanczos.tridiagonal(lambda v: A @ v, order + 10, key=key, shape=(n,))
    with testing.raises(ValueError):
        _ = lanczos.tridiagonal(lambda v: A @ v, order, key=key, shape=(n,))


def test_tridiagonal_max_order(A):
    """If m == n, the matrix should be equal to the full tridiagonal."""
    n, _ = np.shape(A)
    order = n - 1
    key = prng.PRNGKey(1)
    Q, (d_m, e_m) = lanczos.tridiagonal(lambda v: A @ v, order, key=key, shape=(n,))

    # Lanczos is not stable.
    tols_lanczos = {"atol": 1e-5, "rtol": 1e-5}

    # Since full-order mode: Q must be unitary
    assert np.shape(Q) == (order + 1, n)
    assert np.allclose(Q @ Q.T, np.eye(n), **tols_lanczos), Q @ Q.T
    assert np.allclose(Q.T @ Q, np.eye(n), **tols_lanczos), Q.T @ Q

    # T = Q A Qt
    T = np.diag(d_m) + np.diag(e_m, -1) + np.diag(e_m, 1)
    QAQt = Q @ A @ Q.T
    assert np.shape(T) == (order + 1, order + 1)
    # Fail early if the (off)diagonals don't coincide
    assert np.allclose(np.diag(QAQt), d_m, **tols_lanczos)
    assert np.allclose(np.diag(QAQt, 1), e_m, **tols_lanczos)
    assert np.allclose(np.diag(QAQt, -1), e_m, **tols_lanczos)
    # Test the full decompoisition
    assert np.allclose(QAQt, T, **tols_lanczos)

    # Since full-order mode: Qt T Q = A
    # Since Q is unitary and T = Q A Qt, this test
    # should always pass.
    assert np.allclose(Q.T @ T @ Q, A)


def test_tridiagonal(A):
    n, _ = np.shape(A)
    order = n - 2
    key = prng.PRNGKey(1)
    Q, (d_m, e_m) = lanczos.tridiagonal(lambda v: A @ v, order, key=key, shape=(n,))

    # Lanczos is not stable.
    tols_lanczos = {"atol": 1e-5, "rtol": 1e-5}

    assert np.shape(Q) == (order + 1, n)
    assert np.allclose(Q @ Q.T, np.eye(order + 1), **tols_lanczos)

    # T = Q A Qt
    T = np.diag(d_m) + np.diag(e_m, -1) + np.diag(e_m, 1)
    QAQt = Q @ A @ Q.T
    assert np.shape(T) == (order + 1, order + 1)
    # Fail early if the (off)diagonals don't coincide
    assert np.allclose(np.diag(QAQt), d_m, **tols_lanczos)
    assert np.allclose(np.diag(QAQt, 1), e_m, **tols_lanczos)
    assert np.allclose(np.diag(QAQt, -1), e_m, **tols_lanczos)
    # Test the full decompoisition
    assert np.allclose(QAQt, T, **tols_lanczos)
