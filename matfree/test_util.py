"""Test utilities."""

from matfree.backend import linalg, np


def generate_symmetric_matrix_from_eigvals(eigvals, /):
    """Generate a symmetric matrix with prescribed eigenvalues."""
    (n,) = eigvals.shape

    # Need _some_ matrix to start with
    A = np.reshape(np.arange(1.0, n**2 + 1.0), (n, n))
    A = A / linalg.norm(A)
    X = A.T @ A + np.eye(n)

    # QR decompose. We need the orthogonal matrix.
    # Treat Q as a stack of eigenvectors.
    Q, R = linalg.qr(X)

    # Treat Q as eigenvectors, and 'D' as eigenvalues.
    # return Q D Q.T.
    # This matrix will be dense, symmetric, and have a given spectrum.
    return Q @ (eigvals[:, None] * Q.T)
