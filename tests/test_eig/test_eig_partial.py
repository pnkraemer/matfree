"""Tests for eigenvalue functionality."""

from matfree import decomp, eig
from matfree.backend import linalg, np, testing


def test_equal_to_linalg_eig(nrows=7):
    """The output of full-depth decomposition should be equal (*) to linalg.svd().

    (*) Note: The singular values should be identical,
    and the orthogonal matrices should be orthogonal. They are not unique.
    """
    A = np.triu(np.arange(1.0, 1.0 + nrows**2).reshape((nrows, nrows)))
    v0 = np.ones((nrows,))
    num_matvecs = nrows

    hessenberg = decomp.hessenberg(num_matvecs, reortho="full")
    alg = eig.eig_partial(hessenberg)
    vals, vecs = alg(lambda v, p: p @ v, v0, A)

    S, U = linalg.eig(A)
    assert np.allclose(vecs.T @ vecs, U @ U.T, atol=1e-5, rtol=1e-5)
    assert np.allclose(vals, S)


@testing.parametrize("nrows", [8])
@testing.parametrize("num_matvecs", [8, 4, 0])
def test_shapes_as_expected(nrows, num_matvecs):
    A = np.triu(np.arange(1.0, 1.0 + nrows**2).reshape((nrows, nrows)))

    v0 = np.ones((nrows,))
    v0 /= linalg.vector_norm(v0)

    hessenberg = decomp.hessenberg(num_matvecs, reortho="full")
    alg = eig.eig_partial(hessenberg)
    S, U = alg(lambda v, p: p @ v, v0, A)
    assert S.shape == (num_matvecs,)
    assert U.shape == (num_matvecs, nrows)
