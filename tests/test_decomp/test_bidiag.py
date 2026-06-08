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
    """Test that bidiagonalisation yields an orthogonal-bidiagonal decomp."""
    A = make_A(nrows, ncols, num_significant_singular_vals)
    key = prng.prng_key(1)
    v0 = prng.normal(key, shape=(ncols,))

    def Av(v):
        (x,) = v  # tuple input (right-space)
        return [(A @ x,)]  # list-of-tuple output (left-space)

    algorithm = decomp.bidiag(num_matvecs, materialize=True)
    (U_pytree, V_pytree), B, res_pytree, ln = algorithm(Av, (v0,))
    [(U,)] = U_pytree  # U shape (k, nrows) — rows are left Krylov vectors
    (V,) = V_pytree  # V shape (k, ncols) — rows are right Krylov vectors
    (res,) = res_pytree  # res shape (ncols,)

    test_util.assert_columns_orthonormal(U.T)
    test_util.assert_columns_orthonormal(V.T)

    em = np.eye(num_matvecs)[:, -1]
    test_util.assert_allclose(A @ V.T - U.T @ B, 0.0)
    test_util.assert_allclose(A.T @ U.T - V.T @ B.T - linalg.outer(res, em), 0.0)
    test_util.assert_allclose(1.0 / linalg.vector_norm(v0), ln)


@testing.parametrize("nrows", [15, 13])
@testing.parametrize("ncols", [15, 13])
@testing.parametrize("num_matvecs", [12])
def test_bidiag_decomposition_is_satisfied_hilbert(nrows, ncols, num_matvecs):
    """Test that bidiagonalisation implementation is numerically stable."""
    a = np.arange(0, max(ncols, nrows), step=1)
    A = 1.0 / (1.0 + a[:, None] + a[None, :])[:nrows, :ncols]

    key = prng.prng_key(1)
    v0 = prng.normal(key, shape=(ncols,))

    def Av(v):
        (x,) = v  # tuple input (right-space)
        return [(A @ x,)]  # list-of-tuple output (left-space)

    algorithm = decomp.bidiag(num_matvecs, materialize=True, reortho="full")
    (U_pytree, V_pytree), B, res_pytree, ln = algorithm(Av, (v0,))
    [(U,)] = U_pytree  # U shape (k, nrows)
    (V,) = V_pytree  # V shape (k, ncols)
    (res,) = res_pytree  # res shape (ncols,)

    test_util.assert_columns_orthonormal(U.T)
    test_util.assert_columns_orthonormal(V.T)

    em = np.eye(num_matvecs)[:, -1]
    test_util.assert_allclose(A @ V.T - U.T @ B, 0.0)
    test_util.assert_allclose(A.T @ U.T - V.T @ B.T - linalg.outer(res, em), 0.0)
    test_util.assert_allclose(1.0 / linalg.vector_norm(v0), ln)
