"""Test matrix-function-vector products via Lanczos' algorithm."""
from matfree import lanczos, test_util
from matfree.backend import linalg, np, prng


def test_funm_vector_product(n=11):
    """Test matrix-function-vector products via Lanczos' algorithm."""
    # Create a test-problem: matvec, matrix function,
    # vector, and parameters (a matrix).

    def matvec(x, p):
        return p @ x

    # todo: write a test for matfun=np.inv,
    #  because this application seems to be brittle
    def fun(x):
        return np.sin(x)

    v = prng.normal(prng.prng_key(2), shape=(n,))

    eigvals = np.linspace(0.01, 0.99, num=n)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    # Compute the solution
    eigvals, eigvecs = linalg.eigh(matrix)
    log_matrix = eigvecs @ linalg.diagonal(fun(eigvals)) @ eigvecs.T
    expected = log_matrix @ v

    # Compute the matrix-function vector product
    order = 6
    matfun_vec = lanczos.funm_vector_product_spd(fun, order, matvec)
    received = matfun_vec(v, matrix)
    assert np.allclose(expected, received, atol=1e-6)
