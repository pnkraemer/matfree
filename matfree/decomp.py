"""Matrix decomposition algorithms."""

from matfree.backend import containers, control_flow, linalg, np
from matfree.backend.typing import Array, Callable, Tuple


class _Alg(containers.NamedTuple):
    """Matrix decomposition algorithm."""

    init: Callable
    """Initialise the state of the algorithm. Usually, this involves pre-allocation."""

    step: Callable
    """Compute the next iteration."""

    extract: Callable
    """Extract the solution from the state of the algorithm."""

    lower_upper: Tuple[int, int]
    """Range of the for-loop used to decompose a matrix."""


def lanczos_tridiag_full_reortho(depth, /):
    """Construct an implementation of **tridiagonalisation**.

    Uses pre-allocation. Fully reorthogonalise vectors at every step.

    This algorithm assumes a **symmetric matrix**.

    Decompose a matrix into a product of orthogonal-**tridiagonal**-orthogonal matrices.
    Use this algorithm for approximate **eigenvalue** decompositions.

    """

    class State(containers.NamedTuple):
        i: int
        basis: Array
        tridiag: Tuple[Array, Array]
        q: Array

    def init(init_vec: Array) -> State:
        (ncols,) = np.shape(init_vec)
        if depth >= ncols or depth < 1:
            raise ValueError

        diag = np.zeros((depth + 1,))
        offdiag = np.zeros((depth,))
        basis = np.zeros((depth + 1, ncols))

        return State(0, basis, (diag, offdiag), init_vec)

    def apply(state: State, Av: Callable) -> State:
        i, basis, (diag, offdiag), vec = state

        # Re-orthogonalise against ALL basis elements before storing.
        # Note: we re-orthogonalise against ALL columns of Q, not just
        # the ones we have already computed. This increases the complexity
        # of the whole iteration from n(n+1)/2 to n^2, but has the advantage
        # that the whole computation has static bounds (thus we can JIT it all).
        # Since 'Q' is padded with zeros, the numerical values are identical
        # between both modes of computing.
        vec, length = _normalise(vec)
        vec, _ = _gram_schmidt_orthogonalise_set(vec, basis)

        # I don't know why, but this re-normalisation is soooo crucial
        vec, _ = _normalise(vec)
        basis = basis.at[i, :].set(vec)

        # When i==0, Q[i-1] is Q[-1] and again, we benefit from the fact
        #  that Q is initialised with zeros.
        vec = Av(vec)
        basis_vectors_previous = np.asarray([basis[i], basis[i - 1]])
        vec, (coeff, _) = _gram_schmidt_orthogonalise_set(vec, basis_vectors_previous)
        diag = diag.at[i].set(coeff)
        offdiag = offdiag.at[i - 1].set(length)

        return State(i + 1, basis, (diag, offdiag), vec)

    def extract(state: State, /):
        _, basis, (diag, offdiag), _ = state
        return basis, (diag, offdiag)

    return _Alg(init=init, step=apply, extract=extract, lower_upper=(0, depth + 1))


def lanczos_bidiag_full_reortho(depth, /, matrix_shape):
    """Construct an implementation of **bidiagonalisation**.

    Uses pre-allocation. Fully reorthogonalise vectors at every step.

    Works for **arbitrary matrices**. No symmetry required.

    Decompose a matrix into a product of orthogonal-**bidiagonal**-orthogonal matrices.
    Use this algorithm for approximate **singular value** decompositions.
    """
    nrows, ncols = matrix_shape
    max_depth = min(nrows, ncols) - 1
    if depth > max_depth or depth < 0:
        msg1 = f"Depth {depth} exceeds the matrix' dimensions. "
        msg2 = f"Expected: 0 <= depth <= min(nrows, ncols) - 1 = {max_depth} "
        msg3 = f"for a matrix with shape {matrix_shape}."
        raise ValueError(msg1 + msg2 + msg3)

    class State(containers.NamedTuple):
        i: int
        Us: Array
        Vs: Array
        alphas: Array
        betas: Array
        beta: Array
        vk: Array

    def init(init_vec: Array) -> State:
        nrows, ncols = matrix_shape
        alphas = np.zeros((depth + 1,))
        betas = np.zeros((depth + 1,))
        Us = np.zeros((depth + 1, nrows))
        Vs = np.zeros((depth + 1, ncols))
        v0, _ = _normalise(init_vec)
        return State(0, Us, Vs, alphas, betas, 0.0, v0)

    def apply(state: State, Av: Callable, vA: Callable) -> State:
        i, Us, Vs, alphas, betas, beta, vk = state
        Vs = Vs.at[i].set(vk)
        betas = betas.at[i].set(beta)

        uk = Av(vk) - beta * Us[i - 1]
        uk, alpha = _normalise(uk)
        uk, _ = _gram_schmidt_orthogonalise_set(uk, Us)  # full reorthogonalisation
        uk, _ = _normalise(uk)
        Us = Us.at[i].set(uk)
        alphas = alphas.at[i].set(alpha)

        vk = vA(uk) - alpha * vk
        vk, beta = _normalise(vk)
        vk, _ = _gram_schmidt_orthogonalise_set(vk, Vs)  # full reorthogonalisation
        vk, _ = _normalise(vk)

        return State(i + 1, Us, Vs, alphas, betas, beta, vk)

    def extract(state: State, /):
        _, uk_all, vk_all, alphas, betas, beta, vk = state
        return uk_all.T, (alphas, betas[1:]), vk_all, (beta, vk)

    return _Alg(init=init, step=apply, extract=extract, lower_upper=(0, depth + 1))


def _normalise(vec):
    length = linalg.vector_norm(vec)
    return vec / length, length


def _gram_schmidt_orthogonalise_set(vec, vectors):  # Gram-Schmidt
    vec, coeffs = control_flow.scan(_gram_schmidt_orthogonalise, vec, xs=vectors)
    return vec, coeffs


def _gram_schmidt_orthogonalise(vec1, vec2):
    coeff = linalg.vecdot(vec1, vec2)
    vec_ortho = vec1 - coeff * vec2
    return vec_ortho, coeff


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
[matfree.decomp.lanczos_tridiag_full_reortho(...)][matfree.decomp.lanczos_tridiag_full_reortho].
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
