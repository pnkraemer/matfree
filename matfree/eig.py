"""Matrix-free eigenvalue and singular-value analysis."""

from matfree import decomp
from matfree.backend import linalg
from matfree.backend.typing import Array, Callable


# todo: why does this function not return a callable?
def svd_partial(num_matvecs: int):
    """Partial singular value decomposition.

    Combines bidiagonalisation with a full SVD of the (small) bidiagonal matrix.

    Parameters
    ----------
    num_matvecs:
        Number of matrix-vector products aka the depth of the Krylov space
        constructed by Golub-Kahan-Lanczos bidiagonalisation.
        Choosing `num_matvecs = min(nrows, ncols) - 1` would yield behaviour similar to
        e.g. `np.linalg.svd`.
    """

    def svd(Av: Callable, v0: Array):
        # Factorise the matrix
        algorithm = decomp.bidiag(num_matvecs, materialize=True)
        (u, v), B, *_ = algorithm(Av, v0)

        # Compute SVD of factorisation
        U, S, Vt = linalg.svd(B, full_matrices=False)

        # Combine orthogonal transformations
        return u @ U, S, Vt @ v.T

    return svd
