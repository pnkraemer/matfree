from matfree import matfun, test_util
from matfree.backend import linalg, np, prng


def test_chebyshev_via_matfun_vector_product(n=4):
    eigvals = np.linspace(-0.99, 0.99, num=n)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    def matvec(x, p):
        return p @ x

    # todo: write a test for matfun=np.inv,
    #  because this application seems to be brittle
    def fun(x):
        return np.sin(x)

    eigvals, eigvecs = linalg.eigh(matrix)
    log_matrix = eigvecs @ linalg.diagonal(fun(eigvals)) @ eigvecs.T

    v = prng.normal(prng.prng_key(2), shape=(n,))
    expected = log_matrix @ v

    order = 6
    algorithm = matfun.chebyshev(fun, order, matvec)
    matfun_vec = matfun.matfun_vector_product(algorithm)
    received = matfun_vec(v, matrix)
    assert np.allclose(expected, received)
