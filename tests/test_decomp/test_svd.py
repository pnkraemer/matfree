"""Tests for SVD functionality."""


from matfree import decomp, test_util
from matfree.backend import linalg, np, testing


@testing.fixture()
@testing.parametrize("nrows", [10])
@testing.parametrize("ncols", [3])
@testing.parametrize("num_significant_singular_vals", [3])
def A(nrows, ncols, num_significant_singular_vals):
    """Make a positive definite matrix with certain spectrum."""
    # 'Invent' a spectrum. Use the number of pre-defined eigenvalues.
    n = min(nrows, ncols)
    d = np.arange(n) + 10.0
    d = d.at[num_significant_singular_vals:].set(0.001)
    return test_util.asymmetric_matrix_from_singular_values(d, nrows=nrows, ncols=ncols)


@testing.parametrize("full_matrices", [False])
def test_svd_max_depth(A, full_matrices, **svd_kwargs):
    """The output of full-depth SVD should be equal (*) to linalg.svd().

    (*) Note: The orthogonal matrices should be equal up to orthogonal transformations.
    """
    nrows, ncols = np.shape(A)
    depth = min(nrows, ncols) - 1

    def Av(v):
        return A @ v

    def vA(v):
        return v @ A

    v0 = np.ones((ncols,))
    U, S, Vt = decomp.svd(v0, depth, Av, vA, matrix_shape=np.shape(A), **svd_kwargs)
    U_, S_, Vt_ = linalg.svd(A, full_matrices=full_matrices)

    assert np.allclose(S, S_)
    assert np.allclose(U @ U.T, U_ @ U_.T)
    assert np.allclose(U.T @ U, U_.T @ U_)
    assert np.allclose(Vt @ Vt.T, Vt_ @ Vt_.T)
    assert np.allclose(Vt.T @ Vt, Vt_.T @ Vt_)
