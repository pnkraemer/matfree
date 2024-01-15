"""Test the Golub-Kahan-Lanczos bi-diagonalisation with full re-orthogonalisation."""

from matfree import lanczos, test_util
from matfree.backend import linalg, np, prng, testing


@testing.fixture()
def A(nrows, ncols, num_significant_singular_vals):
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
def test_lanczos_bidiag_full_reortho(A, order):
    """Test that Lanczos tridiagonalisation yields an orthogonal-tridiagonal decomp."""
    nrows, ncols = np.shape(A)
    key = prng.prng_key(1)
    v0 = prng.normal(key, shape=(ncols,))

    def Av(v):
        return A @ v

    def vA(v):
        return v @ A

    algorithm = lanczos.alg_bidiag_full_reortho(Av, vA, order, matrix_shape=np.shape(A))
    v0 /= linalg.vector_norm(v0)
    Us, Bs, Vs, (b, v) = algorithm(v0)
    (d_m, e_m) = Bs

    tols_decomp = {"atol": 1e-5, "rtol": 1e-5}

    assert np.shape(Us) == (nrows, order + 1)
    assert np.allclose(Us.T @ Us, np.eye(order + 1), **tols_decomp), Us.T @ Us

    assert np.shape(Vs) == (order + 1, ncols)
    assert np.allclose(Vs @ Vs.T, np.eye(order + 1), **tols_decomp), Vs @ Vs.T

    UAVt = Us.T @ A @ Vs.T
    assert np.allclose(linalg.diagonal(UAVt), d_m, **tols_decomp)
    assert np.allclose(linalg.diagonal(UAVt, 1), e_m, **tols_decomp)

    B = _bidiagonal_dense(d_m, e_m)
    assert np.shape(B) == (order + 1, order + 1)
    assert np.allclose(UAVt, B, **tols_decomp)

    em = np.eye(order + 1)[:, -1]
    AVt = A @ Vs.T
    UtB = Us @ B
    AtUt = A.T @ Us
    VtBtb_plus_bve = Vs.T @ B.T + b * v[:, None] @ em[None, :]
    assert np.allclose(AVt, UtB, **tols_decomp)
    assert np.allclose(AtUt, VtBtb_plus_bve, **tols_decomp)


def _bidiagonal_dense(d, e):
    diag = linalg.diagonal_matrix(d)
    offdiag = linalg.diagonal_matrix(e, 1)
    return diag + offdiag


@testing.parametrize("nrows", [5])
@testing.parametrize("ncols", [3])
@testing.parametrize("num_significant_singular_vals", [3])
def test_error_too_high_depth(A):
    """Assert that a ValueError is raised when the depth exceeds the matrix size."""
    nrows, ncols = np.shape(A)
    max_depth = min(nrows, ncols) - 1

    with testing.raises(ValueError):

        def eye(v):
            return v

        _ = lanczos.alg_bidiag_full_reortho(
            eye, eye, max_depth + 1, matrix_shape=np.shape(A)
        )


@testing.parametrize("nrows", [5])
@testing.parametrize("ncols", [3])
@testing.parametrize("num_significant_singular_vals", [3])
def test_error_too_low_depth(A):
    """Assert that a ValueError is raised when the depth is negative."""
    min_depth = 0
    with testing.raises(ValueError):

        def eye(v):
            return v

        _ = lanczos.alg_bidiag_full_reortho(
            eye, eye, min_depth - 1, matrix_shape=np.shape(A)
        )


@testing.parametrize("nrows", [15])
@testing.parametrize("ncols", [3])
@testing.parametrize("num_significant_singular_vals", [3])
def test_no_error_zero_depth(A):
    """Assert the corner case of zero-depth does not raise an error."""
    nrows, ncols = np.shape(A)
    key = prng.prng_key(1)
    v0 = prng.normal(key, shape=(ncols,))

    def Av(v):
        return A @ v

    def vA(v):
        return v @ A

    algorithm = lanczos.alg_bidiag_full_reortho(Av, vA, 0, matrix_shape=np.shape(A))
    Us, Bs, Vs, (b, v) = algorithm(v0)
    (d_m, e_m) = Bs
    assert np.shape(Us) == (nrows, 1)
    assert np.shape(Vs) == (1, ncols)
    assert np.shape(d_m) == (1,)
    assert np.shape(e_m) == (0,)
    assert np.shape(b) == ()
    assert np.shape(v) == (ncols,)


@testing.parametrize("nrows", [15])
@testing.parametrize("ncols", [3])
@testing.parametrize("num_significant_singular_vals", [3])
@testing.parametrize("order", [2])
def test_validate_unit_norm(A, order):
    """Test that the outputs are NaN if the input is not normalized."""
    nrows, ncols = np.shape(A)
    key = prng.prng_key(1)

    # Not normalized!
    v0 = prng.normal(key, shape=(ncols,)) + 1.0

    def Av(v):
        return A @ v

    def vA(v):
        return v @ A

    algorithm = lanczos.alg_bidiag_full_reortho(
        Av, vA, order, matrix_shape=np.shape(A), validate_unit_2_norm=True
    )
    Us, (d_m, e_m), Vs, (b, v) = algorithm(v0)

    # Since v0 is not normalized, all inputs are NaN
    for x in (Us, d_m, e_m, Vs, b, v):
        assert np.all(np.isnan(x))
