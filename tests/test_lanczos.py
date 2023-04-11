"""Tests for Lanczos functionality."""

from hutch import lanczos
from hutch.backend import np, prng


def test_tridiagonal_sym():
    n = 100
    m = 5
    np.reshape(np.arange(1.0, n**2 + 1.0), (n, n))
    # B = A.T @ A + np.eye(n)  # symmetrise for good measure
    B = np.eye(n)

    key = prng.PRNGKey(seed=1)
    success, (d_m, e_m), W_m = lanczos.tridiagonal_sym(
        lambda v: v, m, key=key, shape=(n,)
    )
    assert np.shape(W_m) == (m, n)
    assert np.shape(d_m) == (m,)
    assert np.shape(e_m) == (m - 1,)

    T_m = W_m @ B @ W_m.T
    assert np.allclose(np.diag(T_m), d_m, atol=1e-6)
    assert np.allclose(np.diag(T_m, 1), e_m, atol=1e-6)
