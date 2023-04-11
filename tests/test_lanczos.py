"""Tests for Lanczos functionality."""

from hutch import lanczos
from hutch.backend import np, prng


def test_tridiagonal_sym():
    n = 100
    m = 5
    np.reshape(np.arange(1.0, n**2 + 1.0), (n, n))
    # B = A.T @ A + np.eye(n)  # symmetrise for good measure
    B = np.eye(n)

    key = prng.PRNGKey(seed=2)
    success, (d_m, e_m), W_m = lanczos.tridiagonal_sym(
        lambda v: v, m, key=key, shape=(n,)
    )
    # assert np.shape(W_m) == (n, m)
    # assert np.shape(d_m) == (m,)
    # assert np.shape(e_m) == (m-1,)
    print()
    print("Success:", success)
    print()
    print()
    T_m = W_m @ B @ W_m.T
    print(W_m @ B @ W_m.T)
    print()
    print("Diag:", np.diag(T_m))
    print("DIag:", d_m)

    print()
    print("Offdiag (via W):", np.diag(T_m, 1))
    print("Offdiag (Lanczos):", e_m)

    assert np.allclose(np.diag(T_m), d_m)
    assert np.allclose(np.diag(T_m, 1), e_m)
