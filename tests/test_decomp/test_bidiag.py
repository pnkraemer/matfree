"""Test the Golub-Kahan-Lanczos bi-diagonalisation with full re-orthogonalisation."""

from matfree import decomp, test_util
from matfree.backend import linalg, np, prng, testing


def make_A(nrows, ncols, num_significant_singular_vals):
    """Make a positive definite matrix with certain spectrum."""
    # 'Invent' a spectrum. Use the number of pre-defined eigenvalues.
    n = min(nrows, ncols)
    d = np.arange(n) + 10.0
    d = d.at[num_significant_singular_vals:].set(0.001)
    return test_util.asymmetric_matrix_from_singular_values(d, nrows=nrows, ncols=ncols)


@testing.parametrize("nrows", [50])
@testing.parametrize("ncols", [49])
@testing.parametrize("num_significant_singular_vals", [4])
@testing.parametrize("num_matvecs", [6])  # ~1.5 * num_significant_eigvals
def test_bidiag_decomposition_is_satisfied(
    nrows, ncols, num_significant_singular_vals, num_matvecs
):
    """Test that Lanczos tridiagonalisation yields an orthogonal-tridiagonal decomp."""
    A = make_A(nrows, ncols, num_significant_singular_vals)
    key = prng.prng_key(1)
    v0 = prng.normal(key, shape=(ncols,))

    algorithm = decomp.bidiag(num_matvecs, materialize=True)
    (U, V), B, res, ln = algorithm(lambda v: A @ v, v0)

    test_util.assert_columns_orthonormal(U)
    test_util.assert_columns_orthonormal(V)

    em = np.eye(num_matvecs)[:, -1]
    test_util.assert_allclose(A @ V, U @ B)
    test_util.assert_allclose(A.T @ U, V @ B.T + linalg.outer(res, em))
    test_util.assert_allclose(1.0 / linalg.vector_norm(v0), ln)
