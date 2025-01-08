"""Matrix-free eigenvalue and singular-value analysis."""

from matfree.backend import linalg
from matfree.backend.typing import Array, Callable


def svd_partial(bidiag: Callable) -> Callable:
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

    def svd(Av: Callable, v0: Array, *parameters):
        # Factorise the matrix
        (u, v), B, *_ = bidiag(Av, v0, *parameters)

        # Compute SVD of factorisation
        U, S, Vt = linalg.svd(B, full_matrices=False)

        # Combine orthogonal transformations
        return (u @ U).T, S, Vt @ v.T

    return svd


def eigh_partial(tridiag_sym: Callable) -> Callable:
    """Partial symmetric/Hermitian eigenvalue decomposition.

    Combines tridiagonalization with a decomposition
    of the (small) tridiagonal matrix.

    Parameters
    ----------
    tridiag_sym:
        An implementation of tridiagonalization.
        For example, the output of
        [decomp.tridiag_sym][matfree.decomp.tridiag_sym].

    """

    def eigh(Av: Callable, v0: Array, *parameters):
        # Factorise the matrix
        Q, H, *_ = tridiag_sym(Av, v0, *parameters)

        # Compute SVD of factorisation
        vals, vecs = linalg.eigh(H)
        vecs = Q @ vecs
        return vals, vecs.T

    return eigh


def eig_partial(hessenberg: Callable) -> Callable:
    """Partial eigenvalue decomposition.

    Combines Hessenberg factorisation with a decomposition
    of the (small) Hessenberg matrix.

    Parameters
    ----------
    hessenberg:
        An implementation of Hessenberg factorisation.
        For example, the output of
        [decomp.hessenberg][matfree.decomp.hessenberg].

    """

    def eig(Av: Callable, v0: Array, *parameters):
        # Factorise the matrix
        Q, H, *_ = hessenberg(Av, v0, *parameters)

        # Compute SVD of factorisation
        vals, vecs = linalg.eig(H)
        vecs = Q @ vecs
        return vals, vecs.T

    return eig
