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
@testing.parametrize("order", [6])  # ~1.5 * num_significant_eigvals
def test_bidiag_decomposition_is_satisfied(
    nrows, ncols, num_significant_singular_vals, order
):
    """Test that Lanczos tridiagonalisation yields an orthogonal-tridiagonal decomp."""
    A = make_A(nrows, ncols, num_significant_singular_vals)
    key = prng.prng_key(1)
    v0 = prng.normal(key, shape=(ncols,))

    def Av(v):
        return A @ v

    def vA(v):
        return v @ A

    algorithm = decomp.bidiag(order, matrix_shape=np.shape(A), materialize=True)
    (U, V), B, res, ln = algorithm(Av, vA, v0)

    test_util.assert_columns_orthonormal(U)
    test_util.assert_columns_orthonormal(V)

    em = np.eye(order + 1)[:, -1]
    test_util.assert_allclose(A @ V, U @ B)
    test_util.assert_allclose(A.T @ U, V @ B.T + linalg.outer(res, em))
    test_util.assert_allclose(1.0 / linalg.vector_norm(v0), ln)


@testing.parametrize("nrows", [5])
@testing.parametrize("ncols", [3])
@testing.parametrize("num_significant_singular_vals", [3])
def test_error_too_high_depth(nrows, ncols, num_significant_singular_vals):
    """Assert that a ValueError is raised when the depth exceeds the matrix size."""
    A = make_A(nrows, ncols, num_significant_singular_vals)
    max_depth = min(nrows, ncols) - 1

    with testing.raises(ValueError, match=""):
        _ = decomp.bidiag(max_depth + 1, matrix_shape=np.shape(A), materialize=False)


@testing.parametrize("nrows", [5])
@testing.parametrize("ncols", [3])
@testing.parametrize("num_significant_singular_vals", [3])
def test_error_too_low_depth(nrows, ncols, num_significant_singular_vals):
    """Assert that a ValueError is raised when the depth is negative."""
    A = make_A(nrows, ncols, num_significant_singular_vals)
    min_depth = 0
    with testing.raises(ValueError, match=""):
        _ = decomp.bidiag(min_depth - 1, matrix_shape=np.shape(A), materialize=False)


@testing.parametrize("nrows", [15])
@testing.parametrize("ncols", [3])
@testing.parametrize("num_significant_singular_vals", [3])
def test_no_error_zero_depth(nrows, ncols, num_significant_singular_vals):
    """Assert the corner case of zero-depth does not raise an error."""
    A = make_A(nrows, ncols, num_significant_singular_vals)
    key = prng.prng_key(1)
    v0 = prng.normal(key, shape=(ncols,))

    def Av(v):
        return A @ v

    def vA(v):
        return v @ A

    algorithm = decomp.bidiag(0, matrix_shape=np.shape(A), materialize=False)
    (U, V), (d_m, e_m), res, ln = algorithm(Av, vA, v0)

    assert np.shape(U) == (nrows, 1)
    assert np.shape(V) == (ncols, 1)
    assert np.shape(d_m) == (1,)
    assert np.shape(e_m) == (0,)
    assert np.shape(res) == (ncols,)
    assert np.shape(ln) == ()
