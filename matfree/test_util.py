"""Test utilities."""

from matfree.backend import linalg, np, prng, tree


def symmetric_matrix_from_eigenvalues(eigvals, /):
    """Generate a symmetric matrix with prescribed eigenvalues."""
    (n,) = eigvals.shape

    # Need _some_ matrix to start with
    A = np.reshape(np.arange(1.0, n**2 + 1.0), (n, n))
    A = A / linalg.matrix_norm(A, which="fro")
    X = A.T @ A + np.eye(n)

    # QR decompose. We need the orthogonal matrix.
    # Treat Q as a stack of eigenvectors.
    Q, _R = linalg.qr_reduced(X)

    # Treat Q as eigenvectors, and 'D' as eigenvalues.
    # return Q D Q.T.
    # This matrix will be dense, symmetric, and have a given spectrum.
    return Q @ (eigvals[:, None] * Q.T)


def asymmetric_matrix_from_singular_values(vals, /, nrows, ncols):
    """Generate an asymmetric matrix with specific singular values."""
    A = np.reshape(np.arange(1.0, nrows * ncols + 1.0), (nrows, ncols))
    A /= nrows * ncols
    U, S, Vt = linalg.svd(A, full_matrices=False)
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


def assert_allclose(a, b, /):
    """Assert that two arrays are close.

    This function uses a different default tolerance to
    jax.numpy.allclose. Instead of fixing values, the tolerance
    depends on the floating-point precision of the input variables.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    tol = np.sqrt(np.finfo_eps(np.dtype(b)))

    # For double precision sqrt(eps) is very tight...
    if tol < 1e-7:
        tol *= 10
    assert np.allclose(a, b, atol=tol, rtol=tol)
