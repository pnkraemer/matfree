"""Matrix decomposition algorithms."""

from matfree.backend import containers, control_flow, linalg, np
from matfree.backend.typing import Array, Callable, Tuple


def svd_approx(
    v0: Array, depth: int, Av: Callable, vA: Callable, matrix_shape: Tuple[int, ...]
):
    """Approximate singular value decomposition.

    Uses GKL with full reorthogonalisation to bi-diagonalise the target matrix
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
    algorithm = lanczos_bidiag_full_reortho(depth, matrix_shape=matrix_shape)
    u, (d, e), vt, _ = decompose_fori_loop(v0, Av, vA, algorithm=algorithm)

    # Compute SVD of factorisation
    B = _bidiagonal_dense(d, e)
    U, S, Vt = linalg.svd(B, full_matrices=False)

    # Combine orthogonal transformations
    return u @ U, S, Vt @ vt


def _bidiagonal_dense(d, e):
    diag = linalg.diagonal_matrix(d)
    offdiag = linalg.diagonal_matrix(e, 1)
    return diag + offdiag
