"""Test matrix-function-vector products via Lanczos' algorithm."""

from matfree import decomp, funm, test_util
from matfree.backend import linalg, np, prng, testing


@testing.parametrize("dense_funm", [funm.dense_funm_sym_eigh, funm.dense_funm_schur])
@testing.parametrize("reortho", ["full", "none"])
def test_funm_lanczos_sym_matches_eigh_implementation(dense_funm, reortho, n=11):
    """Test matrix-function-vector products via Lanczos' algorithm."""
    # Create a test-problem: matvec, matrix function,
    # vector, and parameters (a matrix).

    def matvec(x, p):
        return p @ x

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
    dense_funm = dense_funm(fun)
    lanczos = decomp.tridiag_sym(6, materialize=True, reortho=reortho)
    matfun_vec = funm.funm_lanczos_sym(dense_funm, lanczos)
    received = matfun_vec(matvec, v, matrix)
    assert np.allclose(expected, received, atol=1e-6)
