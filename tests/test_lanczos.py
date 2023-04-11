"""Tests for Lanczos functionality."""

from hutch import lanczos
from hutch.backend import linalg, np, prng, testing


@testing.fixture
@testing.parametrize("n", [5])
def A(n):
    A = np.reshape(np.arange(1.0, n**2 + 1.0), (n, n))
    A = A / linalg.norm(A)
    return A.T @ A + np.eye(n)


#
#
# def test_trace_fun(B):
#
#     key = prng.PRNGKey(seed=1)
#     keys = prng.split(key, num=1)
#     log_det = lanczos.trace_fun(np.log, lambda v: B @ v, m, keys=keys, shape=(n,))
#
#     sign, logdet = linalg.slogdet(B)
#     assert log_det == logdet
#
#
# def test_tridiagonal_sym_reorthogonalise():
#     n = 10
#     m = 10
#     A = np.reshape(np.arange(1.0, n**2 + 1.0), (n, n))
#     B = A.T @ A  # symmetrise for good measure
#     B = B / linalg.norm(B) + np.eye(n)
#     # B = np.eye(n)
#
#     key = prng.PRNGKey(seed=1)
#     (d_m, e_m), W_m = lanczos.tridiagonal_sym_reorthogonalise(
#         lambda v: B @ v, m, key=key, shape=(n,)
#     )
#     assert np.shape(W_m) == (m, n)
#     assert np.shape(d_m) == (m,)
#     assert np.shape(e_m) == (m - 1,)
#
#     T_m = W_m @ B @ W_m.T
#     print(np.diag(T_m), d_m)
#     print()
#     print(np.diag(T_m, -1) - e_m)
#     assert np.allclose(np.diag(T_m), d_m, atol=1e-6)
#     assert np.allclose(np.diag(T_m, 1), e_m, atol=1e-5)


def test_tridiagonal_error_for_too_high_order(A):
    n, _ = np.shape(A)
    order = n
    key = prng.PRNGKey(1)
    with testing.raises(ValueError):
        _ = lanczos.tridiagonal(lambda v: A @ v, order + 10, key=key, shape=(n,))
    with testing.raises(ValueError):
        _ = lanczos.tridiagonal(lambda v: A @ v, order, key=key, shape=(n,))


def test_full_batch(A):
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
