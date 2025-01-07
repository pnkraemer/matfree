"""Matrix-free eigenvalue and singular-value analysis."""

from matfree import decomp
from matfree.backend import linalg
from matfree.backend.typing import Array, Callable


# todo: why does this function not return a callable?
def svd_partial(v0: Array, num_matvecs: int, Av: Callable):
    """Partial singular value decomposition.

    Combines bidiagonalisation with full reorthogonalisation
    and computes the full SVD of the (small) bidiagonal matrix.

    Parameters
    ----------
    v0:
        Initial vector for Golub-Kahan-Lanczos bidiagonalisation.
    num_matvecs:
        Number of matrix-vector products aka the depth of the Krylov space
        constructed by Golub-Kahan-Lanczos bidiagonalisation.
        Choosing `num_matvecs = min(nrows, ncols) - 1` would yield behaviour similar to
        e.g. `np.linalg.svd`.
    Av:
        Matrix-vector product function.
    """
    # Factorise the matrix
    algorithm = decomp.bidiag(num_matvecs, materialize=True)
    (u, v), B, *_ = algorithm(Av, v0)

    # Compute SVD of factorisation
    U, S, Vt = linalg.svd(B, full_matrices=False)

    # Combine orthogonal transformations
    return u @ U, S, Vt @ v.T


def _bidiagonal_dense(d, e):
    diag = linalg.diagonal_matrix(d)
    offdiag = linalg.diagonal_matrix(e, 1)
    return diag + offdiag
