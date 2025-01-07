"""Matrix-free eigenvalue and singular-value analysis."""

from matfree.backend import linalg
from matfree.backend.typing import Array, Callable


def svd_partial(bidiag: Callable):
    """Partial singular value decomposition.

    Combines bidiagonalisation with a full SVD of the (small) bidiagonal matrix.

    Parameters
    ----------
    bidiag:
        An implementation of bidiagonalisation.
        For example, the output of
        [decomp.bidiag][matfree.decomp.bidiag].
        Note how this function assumes that the bidiagonalisation
        materialises the bidiagonal matrix.

    """

    def svd(Av: Callable, v0: Array):
        # Factorise the matrix
        (u, v), B, *_ = bidiag(Av, v0)

        # Compute SVD of factorisation
        U, S, Vt = linalg.svd(B, full_matrices=False)

        # Combine orthogonal transformations
        return u @ U, S, Vt @ v.T

    return svd
