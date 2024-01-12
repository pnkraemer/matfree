"""Test matrix-polynomial-vector algorithms via Chebyshev's recursion."""
from matfree import matfun, test_util
from matfree.backend import linalg, np, prng, testing


@testing.parametrize("eigvals_range", [(-1, 1), (3, 4)])
def test_matrix_poly_chebyshev(eigvals_range, n=4):
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

    eigvals = np.linspace(0, 1, num=n)
    eigvals = eigvals_range[0] + eigvals * (eigvals_range[1] - eigvals_range[0])
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
    assert np.allclose(expected, received)
