"""Test matrix-polynomial-vector algorithms via Chebyshev's recursion."""

from matfree import funm, test_util
from matfree.backend import linalg, np, prng


def test_funm_chebyshev(n=12):
    """Test matrix-polynomial-vector algorithms via Chebyshev's recursion."""
    # Create a test-problem: matvec, matrix function,
    # vector, and parameters (a matrix).

    def matvec(x, p):
        return p @ x

    def fun(x):
        return np.sin(x)

    v = prng.normal(prng.prng_key(2), shape=(n,))

    eigvals = np.linspace(-1 + 0.01, 1 - 0.01, num=n)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    # Compute the solution
    eigvals, eigvecs = linalg.eigh(matrix)
    log_matrix = eigvecs @ linalg.diagonal(fun(eigvals)) @ eigvecs.T
    expected = log_matrix @ v

    # Create an implementation of the Chebyshev-algorithm
    num_matvecs = 6
    matfun_vec = funm.funm_chebyshev(fun, num_matvecs, matvec)

    # Compute the matrix-function vector product
    received = matfun_vec(v, matrix)
    assert np.allclose(expected, received, rtol=1e-4)
