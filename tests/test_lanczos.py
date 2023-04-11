"""Tests for Lanczos functionality."""

from hutch import lanczos
from hutch.backend import linalg, np, prng, testing


@testing.fixture
@testing.parametrize("n", [5])
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


def test_tridiagonal(A):
    """If m == n, the matrix should be equal to the full tridiagonal."""
    n, _ = np.shape(A)
    order = n - 1
    key = prng.PRNGKey(1)
    Q, (d_m, e_m) = lanczos.tridiagonal(lambda v: A @ v, order, key=key, shape=(n,))

    # T = Q A Qt
    T = np.diag(d_m) + np.diag(e_m, -1) + np.diag(e_m, 1)
    assert np.shape(T) == (order + 1, order + 1)
    assert np.allclose(Q @ A @ Q.T, T)

    # Since full-order mode: Q must be unitary
    print(Q.T @ Q)
    assert np.shape(Q) == (order + 1, n)
    assert np.allclose(Q.T @ Q, np.eye(n))
    assert np.allclose(Q @ Q.T, np.eye(n))

    # Since full-order mode: Qt T Q = A
    assert np.allclose(Q.T @ T @ Q, A)
