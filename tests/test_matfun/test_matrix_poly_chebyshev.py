"""Test matrix-polynomial-vector algorithms via Chebyshev's recursion."""
from matfree import matfun, test_util
from matfree.backend import linalg, np, prng


def test_matrix_poly_chebyshev(n=12):
    """Test matrix-polynomial-vector algorithms via Chebyshev's recursion."""
    # Create a test-problem: matvec, matrix function,
    # vector, and parameters (a matrix).

    def matvec(x, p):
        return p @ x

    # todo: write a test for matfun=np.inv,
    #  because this application seems to be brittle
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
    order = 6
    algorithm = matfun.matrix_poly_chebyshev(fun, order, matvec)

    # Compute the matrix-function vector product
    matfun_vec = matfun.matrix_poly_vector_product(algorithm)
    received = matfun_vec(v, matrix)
    assert np.allclose(expected, received, rtol=1e-4)
