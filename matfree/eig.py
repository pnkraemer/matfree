"""Matrix-free eigenvalue and singular-value analysis."""

from matfree import decomp
from matfree.backend import linalg
from matfree.backend.typing import Array, Callable, Tuple


# todo: why does this function not return a callable?
def svd_partial(
    v0: Array, depth: int, Av: Callable, vA: Callable, matrix_shape: Tuple[int, ...]
):
    """Partial singular value decomposition.

    Combines bidiagonalisation with full reorthogonalisation
    and computes the full SVD of the (small) bidiagonal matrix.

    Parameters
    ----------
    v0:
        Initial vector for Golub-Kahan-Lanczos bidiagonalisation.
    depth:
        Depth of the Krylov space constructed by Golub-Kahan-Lanczos bidiagonalisation.
        Choosing `depth = min(nrows, ncols) - 1` would yield behaviour similar to
        e.g. `np.linalg.svd`.
    Av:
        Matrix-vector product function.
    vA:
        Vector-matrix product function.
    matrix_shape:
        Shape of the matrix involved in matrix-vector and vector-matrix products.
    """
    # Factorise the matrix
    algorithm = decomp.bidiag(depth, matrix_shape=matrix_shape, materialize=True)
    u, B, vt, *_ = algorithm(Av, vA, v0)

    # Compute SVD of factorisation
    U, S, Vt = linalg.svd(B, full_matrices=False)

    # Combine orthogonal transformations
    return u @ U, S, Vt @ vt


def _bidiagonal_dense(d, e):
    diag = linalg.diagonal_matrix(d)
    offdiag = linalg.diagonal_matrix(e, 1)
    return diag + offdiag
