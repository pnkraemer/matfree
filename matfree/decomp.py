"""Matrix decomposition algorithms."""

from matfree import lanczos
from matfree.backend import containers, control_flow, linalg
from matfree.backend.typing import Array, Callable, Tuple


def svd(
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
    algorithm = lanczos.bidiagonal_full_reortho(depth, matrix_shape=matrix_shape)
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


class _DecompAlg(containers.NamedTuple):
    """Matrix decomposition algorithm."""

    init: Callable
    """Initialise the state of the algorithm. Usually, this involves pre-allocation."""

    step: Callable
    """Compute the next iteration."""

    extract: Callable
    """Extract the solution from the state of the algorithm."""

    lower_upper: Tuple[int, int]
    """Range of the for-loop used to decompose a matrix."""


AlgorithmType = Tuple[Callable, Callable, Callable, Tuple[int, int]]
"""Decomposition algorithm type.

For example, the output of
[matfree.lanczos.lanczos_full_reortho(...)][matfree.lanczos.lanczos_full_reortho].
"""


# all arguments are positional-only because we will rename arguments a lot
def decompose_fori_loop(v0, *matvec_funs, algorithm: AlgorithmType):
    r"""Decompose a matrix purely based on matvec-products with A.

    The behaviour of this function is equivalent to

    ```python
    def decompose(v0, *matvec_funs, algorithm):
        init, step, extract, (lower, upper) = algorithm
        state = init(v0)
        for _ in range(lower, upper):
            state = step(state, *matvec_funs)
        return extract(state)
    ```

    but the implementation uses JAX' fori_loop.
    """
    # todo: turn the "practically equivalent" bit above into a doctest.
    init, step, extract, (lower, upper) = algorithm
    init_val = init(v0)

    def body_fun(_, s):
        return step(s, *matvec_funs)

    result = control_flow.fori_loop(lower, upper, body_fun=body_fun, init_val=init_val)
    return extract(result)
