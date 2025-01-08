"""Tests for SVD functionality."""

from matfree import decomp, eig, test_util
from matfree.backend import linalg, np, testing


@testing.parametrize("nrows", [10])
@testing.parametrize("ncols", [3])
def test_equal_to_linalg_svd(nrows, ncols):
    """The output of full-depth SVD should be equal (*) to linalg.svd().

    (*) Note: The singular values should be identical,
    and the orthogonal matrices should be orthogonal. They are not unique.
    """
    d = np.arange(1.0, 1.0 + min(nrows, ncols))
    A = test_util.asymmetric_matrix_from_singular_values(d, nrows=nrows, ncols=ncols)
    v0 = np.ones((ncols,))
    num_matvecs = min(nrows, ncols)

    bidiag = decomp.bidiag(num_matvecs)
    svd = eig.svd_partial(bidiag)
    Ut, S, Vt = svd(lambda v, p: p @ v, v0, A)

    U_, S_, Vt_ = linalg.svd(A, full_matrices=False)

    tols_decomp = {"atol": 1e-5, "rtol": 1e-5}
    assert np.allclose(S, S_, **tols_decomp)
    assert np.allclose(Ut.T @ Ut, U_ @ U_.T, **tols_decomp)
    assert np.allclose(Vt @ Vt.T, Vt_ @ Vt_.T, **tols_decomp)


@testing.parametrize("nrows", [10])
@testing.parametrize("ncols", [3])
@testing.parametrize("num_matvecs", [0, 2, 3])
def test_shapes_as_expected(nrows, ncols, num_matvecs):
    """The output of full-depth SVD should be equal (*) to linalg.svd().

    (*) Note: The singular values should be identical,
    and the orthogonal matrices should be orthogonal. They are not unique.
    """
    d = np.arange(1.0, 1.0 + min(nrows, ncols))
    A = test_util.asymmetric_matrix_from_singular_values(d, nrows=nrows, ncols=ncols)
    v0 = np.ones((ncols,))

    bidiag = decomp.bidiag(num_matvecs)
    svd = eig.svd_partial(bidiag)
    Ut, S, Vt = svd(lambda v, p: p @ v, v0, A)

    assert Ut.shape == (num_matvecs, nrows)
    assert S.shape == (num_matvecs,)
    assert Vt.shape == (num_matvecs, ncols)
