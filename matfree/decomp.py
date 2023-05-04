"""Matrix decomposition algorithms."""

from matfree.backend import containers, control_flow, func, linalg, np
from matfree.backend.typing import Any, Array, Callable, Tuple


def svd(v0: Array, depth: int, Av: Callable, vA: Callable, matrix_shape: Tuple[int]):
    """Approximate singular value decomposition.

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
    alg = golub_kahan_lanczos_bidiagonal(depth, matrix_shape=matrix_shape)
    u, (d, e), vt, _ = decompose_fori_loop(0, depth + 1, v0, Av, vA, alg=alg)

    # Compute SVD of factorisation
    B = _bidiagonal_dense(d, e)
    U, S, Vt = linalg.svd(B, full_matrices=False)

    # Combine orthogonal transformations
    return u @ U, S, Vt @ vt


def _bidiagonal_dense(d, e):
    diag = linalg.diagonal_matrix(d)
    offdiag = linalg.diagonal_matrix(e, 1)
    return diag + offdiag


class DecompAlg(containers.NamedTuple):
    """Matrix decomposition algorithm."""

    init: Callable
    """Initialise the state of the algorithm. Usually, this involves pre-allocation."""

    step: Callable
    """Compute the next iteration."""

    extract: Callable
    """Extract the solution from the state of the algorithm."""


# all arguments are positional-only because we will rename arguments a lot
def decompose_fori_loop(lower, upper, v0, *matvec_funs, alg: DecompAlg):
    r"""Decompose a matrix purely based on matvec-products with A.

    This behaviour of this function is equivalent to

    ```python
    def decompose(lower, upper, v0, *matvec_funs, alg):
        state = alg.init(v0)
        for _ in range(lower, upper):
            state = alg.step(state, *matvec_funs)
        return alg.extract(state)
    ```

    but the implementation uses JAX' fori_loop.
    """
    # todo: turn the "practically equivalent" bit above into a doctest.
    init_val = alg.init(v0)

    def body_fun(_, s):
        return alg.step(s, *matvec_funs)

    result = control_flow.fori_loop(lower, upper, body_fun=body_fun, init_val=init_val)
    return alg.extract(result)


def lanczos_tridiagonal(depth, /) -> DecompAlg:
    """**Lanczos** algorithm with pre-allocation and re-orthogonalisation.

    Decompose a matrix into a product of orthogonal-**tridiagonal**-orthogonal matrices.
    Use this algorithm for approximate **eigenvalue** decompositions.

    Lanczos' algorithm assumes a symmetric matrix.
    """
    # this algorithm is massively unstable.
    # but despite this instability, quadrature might be stable?
    # https://www.sciencedirect.com/science/article/abs/pii/S0920563200918164
    return DecompAlg(
        init=func.partial(_lanczos_tridiagonal_init, depth),
        step=_lanczos_tridiagonal_apply,
        extract=_lanczos_tridiagonal_extract,
    )


class _LanczosState(containers.NamedTuple):
    i: int
    basis: Any
    tridiag: Any
    q: Any


def _lanczos_tridiagonal_init(depth: int, init_vec: Array) -> _LanczosState:
    (ncols,) = np.shape(init_vec)
    if depth >= ncols or depth < 1:
        raise ValueError

    diag = np.zeros((depth + 1,))
    offdiag = np.zeros((depth,))
    basis = np.zeros((depth + 1, ncols))

    return _LanczosState(0, basis, (diag, offdiag), init_vec)


def _lanczos_tridiagonal_apply(state: _LanczosState, Av: Callable) -> _LanczosState:
    i, basis, (diag, offdiag), vec = state

    # This one is a hack:
    #
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

    return _LanczosState(i + 1, basis, (diag, offdiag), vec)


def _lanczos_tridiagonal_extract(state: _LanczosState, /):
    _, basis, (diag, offdiag), _ = state
    return basis, (diag, offdiag)


def golub_kahan_lanczos_bidiagonal(depth, /, matrix_shape) -> DecompAlg:
    """**Golub-Kahan-Lanczos** algorithm with pre-allocation and re-orthogonalisation.

    Decompose a matrix into a product of orthogonal-**bidiagonal**-orthogonal matrices.
    Use this algorithm for approximate **singular value** decompositions.
    """
    return DecompAlg(
        init=func.partial(_gkl_bidiagonal_init, depth, matrix_shape),
        step=_gkl_bidiagonal_apply,
        extract=_gkl_bidiagonal_extract,
    )


class _GKLState(containers.NamedTuple):
    i: int
    Us: Any
    Vs: Any
    alphas: Any
    betas: Any
    beta: Any
    vk: Any


def _gkl_bidiagonal_init(depth: int, matrix_shape, init_vec: Array) -> _GKLState:
    nrows, ncols = matrix_shape
    alphas = np.zeros((depth + 1,))
    betas = np.zeros((depth + 1,))
    Us = np.zeros((depth + 1, nrows))
    Vs = np.zeros((depth + 1, ncols))
    v0, _ = _normalise(init_vec)
    return _GKLState(0, Us, Vs, alphas, betas, 0.0, v0)


def _gkl_bidiagonal_apply(state: _GKLState, Av: Callable, vA: Callable) -> _GKLState:
    i, Us, Vs, alphas, betas, beta, vk = state
    Vs = Vs.at[i].set(vk)
    betas = betas.at[i].set(beta)

    uk = Av(vk) - beta * Us[i - 1]
    uk, alpha = _normalise(uk)
    uk, _ = _gram_schmidt_orthogonalise_set(uk, Us)  # hack
    uk, _ = _normalise(uk)  # hack
    Us = Us.at[i].set(uk)
    alphas = alphas.at[i].set(alpha)

    vk = vA(uk) - alpha * vk
    vk, beta = _normalise(vk)
    vk, _ = _gram_schmidt_orthogonalise_set(vk, Vs)  # hack
    vk, _ = _normalise(vk)  # hack

    return _GKLState(i + 1, Us, Vs, alphas, betas, beta, vk)


def _gkl_bidiagonal_extract(state: _GKLState, /):
    _, uk_all, vk_all, alphas, betas, beta, vk = state
    return uk_all.T, (alphas, betas[1:]), vk_all, (beta, vk)


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
