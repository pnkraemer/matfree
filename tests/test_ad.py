"""Tests for autodiff functionality."""


from matfree import lanczos
from matfree.backend import linalg, np, prng, testing


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
    d = np.arange(n) + 10.0
    d = d.at[num_significant_eigvals:].set(0.001)

    # Treat Q as eigenvectors, and 'D' as eigenvalues.
    # return Q D Q.T.
    # This matrix will be dense, symmetric, and have a given spectrum.
    D = np.diag(d)
    return Q @ D @ Q.T


@testing.parametrize("n", [200])
@testing.parametrize("num_significant_eigvals", [30])
@testing.parametrize("order", [10])
# usually: ~1.5 * num_significant_eigvals.
# But logdet seems to converge sooo much faster.
def test_logdet(A, order):
    key = prng.PRNGKey(1)

    def fun(s):
        return _logdet(s, order, key)

    testing.check_grads(fun, (A,), order=1, atol=1e-1, rtol=1e-1)


def _logdet(A, order, key):
    n, _ = np.shape(A)
    received, num_nans = lanczos.trace_of_matfun(
        np.log,
        lambda v: A @ v,
        order,
        key=key,
        num_samples_per_batch=10,
        num_batches=1,
        tangents_shape=(n,),
        tangents_dtype=float,
    )
    return received
