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
def test_shapes_as_expected_vectors(nrows, ncols, num_matvecs):
    d = np.arange(1.0, 1.0 + min(nrows, ncols))
    A = test_util.asymmetric_matrix_from_singular_values(d, nrows=nrows, ncols=ncols)
    v0 = np.ones((ncols,))

    bidiag = decomp.bidiag(num_matvecs)
    svd = eig.svd_partial(bidiag)
    Ut, S, Vt = svd(lambda v, p: p @ v, v0, A)

    assert Ut.shape == (num_matvecs, nrows)
    assert S.shape == (num_matvecs,)
    assert Vt.shape == (num_matvecs, ncols)


@testing.parametrize("nrows", [10])
@testing.parametrize("num_matvecs", [0, 2, 3])
def test_shapes_as_expected_lists_tuples(nrows, num_matvecs):
    K = np.arange(1.0, 10.0).reshape((3, 3))
    v0 = np.ones((nrows, nrows))  # tensor-valued input

    # Map Pytrees to Pytrees
    def Av(v: tuple, stencil) -> list:
        (x,) = v
        return [np.convolve2d(stencil, x)]

    bidiag = decomp.bidiag(num_matvecs)
    svd = eig.svd_partial(bidiag)
    [Ut], S, (Vt,) = svd(Av, (v0,), K)

    # Ut inherits pytree-shape from the outputs (list),
    # and Vt inherits pytree-shape from the inputs (tuple)
    [u0] = Av((v0,), K)
    assert Ut.shape == (num_matvecs, *u0.shape)
    assert S.shape == (num_matvecs,)
    assert Vt.shape == (num_matvecs, *v0.shape)
