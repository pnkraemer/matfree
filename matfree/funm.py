"""Matrix-free implementations of functions of matrices.

Examples
--------
>>> import jax.random
>>> import jax.numpy as jnp
>>> from matfree import decomp
>>>
>>> jnp.set_printoptions(1)
>>>
>>> M = jax.random.normal(jax.random.PRNGKey(1), shape=(10, 10))
>>> A = M.T @ M
>>> v = jax.random.normal(jax.random.PRNGKey(2), shape=(10,))
>>>
>>> # Compute a matrix-logarithm with Lanczos' algorithm
>>> tridiag = decomp.tridiag_sym(lambda s: A @ s, 4)
>>> matfun_vec = funm_lanczos_sym(jnp.log, tridiag)
>>> matfun_vec(v)
Array([-4.1, -1.3, -2.2, -2.1, -1.2, -3.3, -0.2,  0.3,  0.7,  0.9],      dtype=float32)
"""

from matfree.backend import containers, control_flow, func, linalg, np
from matfree.backend.typing import Array, Callable


def funm_chebyshev(matfun: Callable, order: int, matvec: Callable, /) -> Callable:
    """Compute a matrix-function-vector product via Chebyshev's algorithm.

    This function assumes that the **spectrum of the matrix-vector product
    is contained in the interval (-1, 1)**, and that the **matrix-function
    is analytic on this interval**. If this is not the case,
    transform the matrix-vector product and the matrix-function accordingly.
    """
    # Construct nodes
    nodes = _chebyshev_nodes(order)
    fx_nodes = matfun(nodes)

    class _ChebyshevState(containers.NamedTuple):
        interpolation: Array
        poly_coefficients: tuple[Array, Array]
        poly_values: tuple[Array, Array]

    def init_func(vec, *parameters):
        # Initialize the scalar recursion
        # (needed to compute the interpolation weights)
        t2_n, t1_n = nodes, np.ones_like(nodes)
        c1 = np.mean(fx_nodes * t1_n)
        c2 = 2 * np.mean(fx_nodes * t2_n)

        # Initialize the vector-valued recursion
        # (this is where the matvec happens)
        t2_x, t1_x = matvec(vec, *parameters), vec
        value = c1 * t1_x + c2 * t2_x
        return _ChebyshevState(value, (t2_n, t1_n), (t2_x, t1_x))

    def recursion_func(val: _ChebyshevState, *parameters) -> _ChebyshevState:
        value, (t2_n, t1_n), (t2_x, t1_x) = val

        # Apply the next scalar recursion and
        # compute the next coefficient
        t2_n, t1_n = 2 * nodes * t2_n - t1_n, t2_n
        c2 = 2 * np.mean(fx_nodes * t2_n)

        # Apply the next matrix-vector product recursion and
        # compute the next interpolation-value
        t2_x, t1_x = 2 * matvec(t2_x, *parameters) - t1_x, t2_x
        value += c2 * t2_x
        return _ChebyshevState(value, (t2_n, t1_n), (t2_x, t1_x))

    def extract_func(val: _ChebyshevState):
        return val.interpolation

    alg = (0, order - 1), init_func, recursion_func, extract_func
    return _funm_polyexpand(alg)


def _chebyshev_nodes(n, /):
    k = np.arange(n, step=1.0) + 1
    return np.cos((2 * k - 1) / (2 * n) * np.pi())


def _funm_polyexpand(matrix_poly_alg, /):
    """Compute a matrix-function-vector product via a polynomial expansion."""
    (lower, upper), init_func, step_func, extract_func = matrix_poly_alg

    def matvec(vec, *parameters):
        final_state = control_flow.fori_loop(
            lower=lower,
            upper=upper,
            body_fun=lambda _i, v: step_func(v, *parameters),
            init_val=init_func(vec, *parameters),
        )
        return extract_func(final_state)

    return matvec


def funm_lanczos_sym(matfun: Callable, tridiag_sym: Callable, /) -> Callable:
    """Implement a matrix-function-vector product via Lanczos' tridiagonalisation.

    This algorithm uses Lanczos' tridiagonalisation
    and therefore applies only to symmetric matrices.

    Parameters
    ----------
    matfun
        Matrix function.
    tridiag_sym
        Tridiagonalisation implementation.
        Output of [decomp.tridiag_sym][matfree.decomp.tridiag_sym].
    """

    def estimate(vec, *parameters):
        length = linalg.vector_norm(vec)
        vec /= length
        (basis, (diag, off_diag)), _ = tridiag_sym(vec, *parameters)
        eigvals, eigvecs = _eigh_tridiag_sym(diag, off_diag)

        fx_eigvals = func.vmap(matfun)(eigvals)
        return length * (basis.T @ (eigvecs @ (fx_eigvals * eigvecs[0, :])))

    return estimate


def _eigh_tridiag_sym(diag, off_diag):
    # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
    #  use it here. Until then: an eigen-decomposition of size (order + 1)
    #  does not hurt too much...
    diag = linalg.diagonal_matrix(diag)
    offdiag1 = linalg.diagonal_matrix(off_diag, -1)
    offdiag2 = linalg.diagonal_matrix(off_diag, 1)
    dense_matrix = diag + offdiag1 + offdiag2

    eigvals, eigvecs = linalg.eigh(dense_matrix)
    return eigvals, eigvecs
