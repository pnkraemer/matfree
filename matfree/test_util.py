"""Test utilities."""

from matfree.backend import linalg, np, prng, tree


def hermitian_matrix_from_eigenvalues(eigvals, /, key, *, dtype=None):
    """Generate a Hermitian matrix with prescribed real eigenvalues.

    For real dtype the result is symmetric; for complex dtype it is Hermitian.
    """
    (n,) = eigvals.shape
    if dtype is None:
        dtype = eigvals.dtype
    eigvals = eigvals.real
    Q, _ = linalg.qr_reduced(prng.normal(key, shape=(n, n), dtype=dtype))
    return (Q * eigvals) @ Q.T.conj()


def eigenvalues_fast_spectral_decay(n, /):
    """Eigenvalue array with rapid geometric decay."""
    return 0.7 ** np.arange(n)


def eigenvalues_large_spectral_drop(n, /, *, num_flat=50, drop_value=1e-3):
    """Eigenvalue array that is flat then drops sharply."""
    eigvals_flat = np.ones(num_flat)
    eigvals_drop = np.ones(n - num_flat) * drop_value
    return np.concatenate([eigvals_flat, eigvals_drop])


def asymmetric_matrix_from_singular_values(vals, /, nrows, ncols):
    """Generate an asymmetric matrix with specific singular values."""
    A = np.reshape(np.arange(1.0, nrows * ncols + 1.0), (nrows, ncols))
    A /= nrows * ncols
    U, _S, Vt = linalg.svd(A, full_matrices=False)
    return U @ linalg.diagonal(vals) @ Vt


def to_dense_bidiag(d, e, /, offset=1):
    """Materialize a bidiagonal matrix."""
    diag = linalg.diagonal_matrix(d)
    offdiag = linalg.diagonal_matrix(e, offset=offset)
    return diag + offdiag


def to_dense_tridiag_sym(d, e, /):
    """Materialize a symmetric tridiagonal matrix."""
    diag = linalg.diagonal_matrix(d)
    offdiag1 = linalg.diagonal_matrix(e, offset=1)
    offdiag2 = linalg.diagonal_matrix(e, offset=-1)
    return diag + offdiag1 + offdiag2


def tree_random_like(key, pytree, *, generate_func=prng.normal):
    """Fill a tree with random values."""
    flat, unflatten = tree.ravel_pytree(pytree)
    flat_like = generate_func(key, shape=flat.shape, dtype=flat.dtype)
    return unflatten(flat_like)


def assert_columns_orthonormal(Q, /):
    """Assert that the columns in a matrix are orthonormal."""
    eye_like = Q.T @ Q
    ref = np.eye(len(eye_like))
    assert_allclose(eye_like, ref)


def assert_allclose(a, b, /, atol=None, rtol=None):
    """Assert that two arrays are close.

    This function uses a different default tolerance to
    jax.numpy.allclose. Instead of fixing values, the tolerance
    depends on the floating-point precision of the input variables.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    tol = 10 * np.sqrt(np.finfo_eps(np.dtype(a)))

    # For double precision sqrt(eps) is very tight...
    if tol < 1e-6:
        tol *= 10

    rtol = rtol if rtol is not None else tol
    atol = atol if atol is not None else tol
    assert np.allclose(a, b, atol=atol, rtol=atol)
