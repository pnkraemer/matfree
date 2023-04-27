"""Matrix decomposition algorithms."""

from matfree.backend import containers, control_flow, func, linalg, np
from matfree.backend.typing import Any, Array, Callable


class DecompAlg(containers.NamedTuple):
    """Matrix decomposition algorithm."""

    init: Callable
    step: Callable
    extract: Callable


# all arguments are positional-only because we will rename arguments a lot
def decompose_fori_loop(lower, upper, Av, v0, /, alg: DecompAlg):
    r"""Decompose a matrix purely based on matvec-products with A.

    The semantics of this function are practically equivalent to

    ```python
    def decompose(lower, upper, Av, v0, alg):
        state = alg.init(v0)
        for _ in range(lower, upper):
            state = alg.step(state, Av)
        return alg.extract(state)
    ```

    but the implementation uses JAX' fori_loop.
    """
    init_val = alg.init(v0)

    def body_fun(_, s):
        return alg.step(s, Av=Av)

    result = control_flow.fori_loop(lower, upper, body_fun=body_fun, init_val=init_val)
    return alg.extract(result)


def lanczos(depth, /) -> DecompAlg:
    r"""Lanczos' algorithm with pre-allocation and re-orthogonalisation.

    Decompose a matrix into a product orthogonal-tridiagonal-orthogonal matrix.

    More specifically, Lanczos' algorithm orthogonally projects the original
    matrix onto the (n+1)-deep Krylov subspace

    \{ v, Av, A^2v, ..., A^n v \}

    using the Gram-Schmidt procedure.

    The spectrum of the resulting tridiagonal matrix
    approximates the spectrum of the original matrix.
    (More specifically, the spectrum tends to the 'most extreme' eigenvalues.)
    """
    # this algorithm is massively unstable.
    # but despite this instability, quadrature might be stable?
    # https://www.sciencedirect.com/science/article/abs/pii/S0920563200918164
    return DecompAlg(
        init=func.partial(_lanczos_init, depth),
        step=_lanczos_apply,
        extract=_lanczos_extract,
    )


class _LanczosState(containers.NamedTuple):
    i: int
    basis: Any
    tridiag: Any
    q: Any


def _lanczos_init(depth: int, init_vec: Array) -> _LanczosState:
    (ncols,) = np.shape(init_vec)
    if depth >= ncols or depth < 1:
        raise ValueError

    (ncols,) = init_vec.shape
    diag = np.zeros((depth + 1,))
    offdiag = np.zeros((depth,))
    basis = np.zeros((depth + 1, ncols))

    return _LanczosState(0, basis, (diag, offdiag), init_vec)


def _lanczos_apply(state: _LanczosState, Av: Callable) -> _LanczosState:
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


def _lanczos_extract(state: _LanczosState, /):
    _, basis, (diag, offdiag), _ = state
    return basis, (diag, offdiag)


def _normalise(vec):
    length = linalg.norm(vec)
    return vec / length, length


def _gram_schmidt_orthogonalise_set(vec, vectors):  # Gram-Schmidt
    vec, coeffs = control_flow.scan(_gram_schmidt_orthogonalise, init=vec, xs=vectors)
    return vec, coeffs


def _gram_schmidt_orthogonalise(vec1, vec2):
    coeff = np.dot(vec1, vec2)
    vec_ortho = vec1 - coeff * vec2
    return vec_ortho, coeff
