"""Tests for Lanczos functionality."""

from hutch import lanczos
from hutch.backend import np, prng


def test_tridiagonal_sym():
    n = 12
    m = 4
    A = np.reshape(np.arange(1.0, n**2 + 1.0), (n, n))
    B = A.T @ A + np.eye(n)  # symmetrise for good measure

    key = prng.PRNGKey(seed=2)
    v0 = prng.normal(key, shape=(n,))

    success, (d_m, e_m), W_m = lanczos.tridiagonal_sym(lambda v: B @ v, m, init_vec=v0)
    # assert np.shape(W_m) == (n, m)
    # assert np.shape(d_m) == (m,)
    # assert np.shape(e_m) == (m-1,)
    print()
    print("Success:", success)
    print()
    print()
    T_m = W_m.T @ B @ W_m
    print(np.diag(T_m))
    print(d_m)

    print()
    print(np.diag(T_m, -1))
    print(e_m)

    assert np.allclose(np.diag(T_m), d_m)
    assert np.allclose(np.diag(T_m, -1), e_m)
