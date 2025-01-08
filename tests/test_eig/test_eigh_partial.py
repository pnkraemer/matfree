"""Tests for eigenvalue functionality."""

from matfree import decomp, eig, test_util
from matfree.backend import linalg, np, testing


def test_equal_to_linalg_eigh(nrows=10):
    eigvals = np.arange(1.0, 1.0 + nrows)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    v0 = np.ones((nrows,))
    num_matvecs = nrows

    hessenberg = decomp.hessenberg(num_matvecs, reortho="full")
    alg = eig.eigh_partial(hessenberg)
    vals, vecs = alg(lambda v, p: p @ v, v0, A)

    S, U = linalg.eigh(A)
    assert np.allclose(vecs.T @ vecs, U @ U.T, atol=1e-5, rtol=1e-5)
    assert np.allclose(vals, S)


@testing.parametrize("nrows", [8])
@testing.parametrize("num_matvecs", [8, 4, 0])
def test_shapes_as_expected_vector(nrows, num_matvecs):
    eigvals = np.arange(1.0, 1.0 + nrows)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    v0 = np.ones((nrows,))

    tridiag_sym = decomp.tridiag_sym(num_matvecs, reortho="full")
    alg = eig.eigh_partial(tridiag_sym)
    S, U = alg(lambda v, p: p @ v, v0, A)
    assert S.shape == (num_matvecs,)
    assert U.shape == (num_matvecs, nrows)


@testing.parametrize("nrows", [10])
@testing.parametrize("num_matvecs", [0, 2, 3])
def test_shapes_as_expected_lists_tuples(nrows, num_matvecs):
    eigvals = np.arange(1.0, 1.0 + nrows)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    v0 = np.ones((nrows,))

    # Map Pytrees to Pytrees
    def Av(v: list, p) -> list:
        [(x,)] = v
        return [(p @ x,)]

    tridiag_sym = decomp.tridiag_sym(num_matvecs, reortho="none")
    eigh_partial = eig.eigh_partial(tridiag_sym)
    vals, [(vecs,)] = eigh_partial(Av, [(v0,)], A)

    # Ut inherits pytree-shape from the outputs (list),
    # and Vt inherits pytree-shape from the inputs (tuple)
    assert vecs.shape == (num_matvecs, *v0.shape)
    assert vals.shape == (num_matvecs,)
