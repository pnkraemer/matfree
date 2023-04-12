"""Tests for Lanczos functionality."""

from hutch import lanczos
from hutch.backend import linalg, np, prng, testing


@testing.fixture
def A(n, num_significant_eigvals):
    """Make a positive definite matrix with certain spectrum."""
    # Need _some_ matrix to start with
    A = np.reshape(np.arange(1.0, n**2 + 1.0), (n, n))
    A = A / linalg.norm(A)
    X = A.T @ A + np.eye(n)

    # QR decompose. We need the orthogonal matrix.
    # Treat Q as a stack of eigenvectors.
    Q, R = linalg.qr(X)

    # 'Invent' a spectrum. Use the number of pre-defined eigenvalues.
    d = np.arange(n) + 1.0
    d = d.at[num_significant_eigvals:].set(0.001)

    # Treat Q as eigenvectors, and 'D' as eigenvalues.
    # return Q D Q.T.
    # This matrix will be dense, symmetric, and have a given spectrum.
    D = np.diag(d)
    return Q @ D @ Q.T


@testing.parametrize("n", [200])
@testing.parametrize("num_significant_eigvals", [4])
@testing.parametrize("order", [6])  # ~1.5 * num_significant_eigvals
def test_logdet(A, order):
    n, _ = np.shape(A)
    key = prng.PRNGKey(1)
    keys = prng.split(key, num=10_000)
    received, _is_nan_index = lanczos.trace_of_matfn(
        np.log,
        lambda v: A @ v,
        order,
        keys=keys,
        tangents_shape=(n,),
        tangents_dtype=np.dtype(A),
    )
    expected = linalg.slogdet(A)[1]
    print_if_assert_fails = ("error", np.abs(received - expected), "target:", expected)
    assert np.allclose(received, expected, atol=1e-2, rtol=1e-2), print_if_assert_fails


@testing.parametrize("n", [6])
@testing.parametrize("num_significant_eigvals", [4])
def test_tridiagonal_error_for_too_high_order(A):
    n, _ = np.shape(A)
    order = n
    key = prng.PRNGKey(1)
    v0 = prng.normal(key, shape=(n,))
    with testing.raises(ValueError):
        _ = lanczos.tridiagonal(lambda v: A @ v, order + 10, v0)
    with testing.raises(ValueError):
        _ = lanczos.tridiagonal(lambda v: A @ v, order, v0)


@testing.parametrize("n", [6])
@testing.parametrize("num_significant_eigvals", [6])
def test_tridiagonal_max_order(A):
    """If m == n, the matrix should be equal to the full tridiagonal."""
    n, _ = np.shape(A)
    order = n - 1
    key = prng.PRNGKey(1)
    v0 = prng.normal(key, shape=(n,))
    Q, (d_m, e_m) = lanczos.tridiagonal(lambda v: A @ v, order, v0)

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

    # Test the full decomposition
    # (i.e. assert that the off-tridiagonal elements are actually small)
    # be loose with this test. off-diagonal elements accumulate quickly.
    tols_lanczos = {"atol": 1e-5, "rtol": 1e-5}
    assert np.allclose(QAQt, T, **tols_lanczos)

    # Since full-order mode: Qt T Q = A
    # Since Q is unitary and T = Q A Qt, this test
    # should always pass.
    assert np.allclose(Q.T @ T @ Q, A, **tols_lanczos)


@testing.parametrize("n", [50])
@testing.parametrize("num_significant_eigvals", [4])
@testing.parametrize("order", [6])  # ~1.5 * num_significant_eigvals
def test_tridiagonal(A, order):
    n, _ = np.shape(A)
    key = prng.PRNGKey(1)
    init_vec = prng.normal(key, shape=(n,))
    Q, (d_m, e_m) = lanczos.tridiagonal(lambda v: A @ v, order, init_vec)

    # Lanczos is not stable.
    tols_lanczos = {"atol": 1e-5, "rtol": 1e-5}

    assert np.shape(Q) == (order + 1, n)
    assert np.allclose(Q @ Q.T, np.eye(order + 1), **tols_lanczos), Q @ Q.T

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
