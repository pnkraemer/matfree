"""Test utilities."""

from matfree.backend import linalg, np


def symmetric_matrix_from_eigenvalues(eigvals, /):
    """Generate a symmetric matrix with prescribed eigenvalues."""
    assert np.array_min(eigvals) > 0
    (n,) = eigvals.shape

    # Need _some_ matrix to start with
    A = np.reshape(np.arange(1.0, n**2 + 1.0), (n, n))
    A = A / linalg.matrix_norm(A, which="fro")
    X = A.T @ A + np.eye(n)

    # QR decompose. We need the orthogonal matrix.
    # Treat Q as a stack of eigenvectors.
    Q, R = linalg.qr(X)

    # Treat Q as eigenvectors, and 'D' as eigenvalues.
    # return Q D Q.T.
    # This matrix will be dense, symmetric, and have a given spectrum.
    return Q @ (eigvals[:, None] * Q.T)


def asymmetric_matrix_from_singular_values(vals, /, nrows, ncols):
    """Generate an asymmetric matrix with specific singular values."""
    assert np.array_min(vals) > 0
    A = np.reshape(np.arange(1.0, nrows * ncols + 1.0), (nrows, ncols))
    A /= nrows * ncols
    U, S, Vt = linalg.svd(A, full_matrices=False)
    return U @ linalg.diagonal(vals) @ Vt
