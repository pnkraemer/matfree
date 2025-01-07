"""Tests for eigenvalue functionality."""

from matfree import decomp, eig, test_util
from matfree.backend import linalg, np, testing


@testing.fixture()
@testing.parametrize("nrows", [10])
def A(nrows):
    """Make a positive definite matrix with certain spectrum."""
    # 'Invent' a spectrum. Use the number of pre-defined eigenvalues.
    d = np.arange(1.0, 1.0 + nrows)
    return test_util.asymmetric_matrix_from_singular_values(d, nrows=nrows, ncols=nrows)


def test_equal_to_linalg_eigh(A):
    """The output of full-depth decomposition should be equal (*) to linalg.svd().

    (*) Note: The singular values should be identical,
    and the orthogonal matrices should be orthogonal. They are not unique.
    """
    nrows, ncols = np.shape(A)
    num_matvecs = min(nrows, ncols)

    v0 = np.ones((ncols,))
    v0 /= linalg.vector_norm(v0)

    hessenberg = decomp.hessenberg(num_matvecs, reortho="full")
    alg = eig.eigh_partial(hessenberg)
    S, U = alg(lambda v, p: p @ v, v0, A)

    S_, U_ = linalg.eigh(A)

    assert np.allclose(S, S_)
    assert np.shape(U) == np.shape(U_)

    tols_decomp = {"atol": 1e-5, "rtol": 1e-5}
    assert np.allclose(U @ U.T, U_ @ U_.T, **tols_decomp)


@testing.parametrize("nrows", [8])
@testing.parametrize("num_matvecs", [8, 4, 0])
def test_shapes_as_expected(nrows, num_matvecs):
    A = np.arange(1.0, 1.0 + nrows**2).reshape((nrows, nrows))
    A = A + A.T

    v0 = np.ones((nrows,))
    v0 /= linalg.vector_norm(v0)

    tridiag_sym = decomp.tridiag_sym(num_matvecs, reortho="full")
    alg = eig.eigh_partial(tridiag_sym)
    S, U = alg(lambda v, p: p @ v, v0, A)
    assert S.shape == (num_matvecs,)
    assert U.shape == (nrows, num_matvecs)
