"""Test matrix-function-vector products via the Arnoldi iteration."""

from matfree import decomp, funm
from matfree.backend import np, prng, testing


def case_expm():
    return funm.dense_funm_pade_exp()


@testing.parametrize("reortho", ["full", "none"])
@testing.parametrize_with_cases("dense_funm", cases=".", prefix="case_")
def test_funm_arnoldi_matches_schur_implementation(dense_funm, reortho, n=11):
    """Test matrix-function-vector products via the Arnoldi iteration."""
    # Create a test-problem: matvec, matrix function,
    # vector, and parameters (a matrix).

    matrix = prng.normal(prng.prng_key(1), shape=(n, n))
    v = prng.normal(prng.prng_key(2), shape=(n,))

    # Compute the solution
    expected = dense_funm(matrix) @ v

    # Compute the matrix-function vector product
    arnoldi = decomp.hessenberg((n * 3) // 4, reortho=reortho)
    matfun_vec = funm.funm_arnoldi(dense_funm, arnoldi)
    received = matfun_vec(lambda s, p: p @ s, v, matrix)

    assert np.allclose(expected, received, rtol=1e-1, atol=1e-1)
