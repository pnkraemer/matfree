"""All things Lanczos' algorithm."""

from matfree import decomp
from matfree.backend import containers, control_flow, func, linalg, np
from matfree.backend.typing import Array, Callable, Tuple


def funm_vector_product_spd(matfun, order, matvec, /):
    """Implement a matrix-function-vector product via Lanczos' algorithm.

    This algorithm uses Lanczos' tridiagonalisation with full re-orthogonalisation
    and therefore applies only to symmetric, positive definite matrices.
    """
    # Lanczos' algorithm
    algorithm = alg_tridiag_full_reortho(order)

    def estimate(vec, *parameters):
        def matvec_p(v):
            return matvec(v, *parameters)

        length = linalg.vector_norm(vec)
        vec /= length
        basis, tridiag = decompose_fori_loop(vec, matvec_p, algorithm=algorithm)
        (diag, off_diag) = tridiag

        # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
        #  use it here. Until then: an eigen-decomposition of size (order + 1)
        #  does not hurt too much...
        diag = linalg.diagonal_matrix(diag)
        offdiag1 = linalg.diagonal_matrix(off_diag, -1)
        offdiag2 = linalg.diagonal_matrix(off_diag, 1)
        dense_matrix = diag + offdiag1 + offdiag2
        eigvals, eigvecs = linalg.eigh(dense_matrix)

        fx_eigvals = func.vmap(matfun)(eigvals)
        return length * (basis.T @ (eigvecs @ (fx_eigvals * eigvecs[0, :])))

    return estimate


AlgorithmType = Tuple[Callable, Callable, Callable, Tuple[int, int]]
"""Decomposition algorithm type.

For example, the output of
[matfree.decomp.lanczos_tridiag_full_reortho(...)][matfree.decomp.lanczos_tridiag_full_reortho].
"""


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


def alg_tridiag_full_reortho(depth, /, validate_unit_2_norm=False) -> AlgorithmType:
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
        length: Array

    def init(init_vec: Array) -> State:
        (ncols,) = np.shape(init_vec)
        if depth >= ncols or depth < 1:
            raise ValueError

        if validate_unit_2_norm:
            init_vec = _validate_unit_2_norm(init_vec)

        diag = np.zeros((depth + 1,))
        offdiag = np.zeros((depth,))
        basis = np.zeros((depth + 1, ncols))

        return State(0, basis, (diag, offdiag), init_vec, 1.0)

    def apply(state: State, Av: Callable) -> State:
        i, basis, (diag, offdiag), vec, length = state

        # Compute the next off-diagonal entry
        offdiag = offdiag.at[i - 1].set(length)

        # Re-orthogonalise against ALL basis elements before storing.
        # Note: we re-orthogonalise against ALL columns of Q, not just
        # the ones we have already computed. This increases the complexity
        # of the whole iteration from n(n+1)/2 to n^2, but has the advantage
        # that the whole computation has static bounds (thus we can JIT it all).
        # Since 'Q' is padded with zeros, the numerical values are identical
        # between both modes of computing.
        vec, *_ = _gram_schmidt_classical(vec, basis)

        # Store new basis element
        basis = basis.at[i, :].set(vec)

        # When i==0, Q[i-1] is Q[-1] and again, we benefit from the fact
        #  that Q is initialised with zeros.
        vec = Av(vec)
        basis_vectors_previous = np.asarray([basis[i], basis[i - 1]])
        vec, length, (coeff, _) = _gram_schmidt_classical(vec, basis_vectors_previous)
        diag = diag.at[i].set(coeff)

        return State(i + 1, basis, (diag, offdiag), vec, length)

    def extract(state: State, /):
        # todo: return final output "_ignored"
        _, basis, (diag, offdiag), *_ignored = state
        return basis, (diag, offdiag)

    return _Alg(init=init, step=apply, extract=extract, lower_upper=(0, depth + 1))


def _normalise(vec):
    length = linalg.vector_norm(vec)
    return vec / length, length


def _gram_schmidt_classical(vec, vectors):  # Gram-Schmidt
    vec, coeffs = control_flow.scan(_gram_schmidt_classical_step, vec, xs=vectors)
    vec, length = _normalise(vec)
    return vec, length, coeffs


def _gram_schmidt_classical_step(vec1, vec2):
    coeff = linalg.vecdot(vec1, vec2)
    vec_ortho = vec1 - coeff * vec2
    return vec_ortho, coeff


# all arguments are positional-only because we will rename arguments a lot
def decompose_fori_loop(v0, *matvec_funs, algorithm):
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

    # todo: parametrized matrix-vector products
    # todo: move matvec_funs into the algorithm,
    #  and parameters into the decompose
    def body_fun(_, s):
        return step(s, *matvec_funs)

    result = control_flow.fori_loop(lower, upper, body_fun=body_fun, init_val=init_val)
    return extract(result)


def _validate_unit_2_norm(v, /):
    # Lanczos assumes a unit-2-norm vector as an input
    # We cannot raise an error based on values of the init_vec,
    # but we can make it obvious that the result is unusable.
    is_not_normalized = np.abs(linalg.vector_norm(v) - 1.0) > 10 * np.finfo_eps(v.dtype)
    return control_flow.cond(
        is_not_normalized,
        lambda s: np.nan() * np.ones_like(s),
        lambda s: s,
        v,
    )
