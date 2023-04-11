"""Tests for Lanczos functionality."""

from hutch import lanczos
from hutch.backend import linalg, np, prng


def test_trace_fun():
    n = 100
    m = 10
    A = np.reshape(np.arange(1.0, n**2 + 1.0), (n, n))
    B = A.T @ A / linalg.norm(A.T @ A) + np.eye(n)  # symmetrise for good measure
    # B = np.eye(n)

    key = prng.PRNGKey(seed=1)
    keys = prng.split(key, num=1)
    log_det = lanczos.trace_fun(np.log, lambda v: B @ v, m, keys=keys, shape=(n,))

    sign, logdet = linalg.slogdet(B)
    assert log_det == logdet


def test_tridiagonal_sym_reorthogonalise():
    n = 100
    m = 5
    A = np.reshape(np.arange(1.0, n**2 + 1.0), (n, n))
    B = A.T @ A  # symmetrise for good measure
    B = B / linalg.norm(B) + np.eye(n)
    # B = np.eye(n)

    key = prng.PRNGKey(seed=1)
    (d_m, e_m), W_m = lanczos.tridiagonal_sym_reorthogonalise(
        lambda v: B @ v, m, key=key, shape=(n,)
    )
    assert np.shape(W_m) == (m, n)
    assert np.shape(d_m) == (m,)
    assert np.shape(e_m) == (m - 1,)

    T_m = W_m @ B @ W_m.T
    print(np.diag(T_m), d_m)
    print()
    print(np.diag(T_m, -1) - e_m)
    assert np.allclose(np.diag(T_m), d_m, atol=1e-6)
    assert np.allclose(np.diag(T_m, 1), e_m, atol=1e-4)
