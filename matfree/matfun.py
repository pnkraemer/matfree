"""Implement approximations of matrix-function-vector products.

This module is experimental.

Examples
--------
>>> import jax.random
>>> import jax.numpy as jnp
>>>
>>> jnp.set_printoptions(1)
>>>
>>> M = jax.random.normal(jax.random.PRNGKey(1), shape=(10, 10))
>>> A = M.T @ M
>>> v = jax.random.normal(jax.random.PRNGKey(2), shape=(10,))
>>>
>>> # Compute a matrix-logarithm with Lanczos' algorithm
>>> matfun_vec = lanczos_spd(jnp.log, 4, lambda s: A @ s)
>>> matfun_vec(v)
Array([-4. , -2.1, -2.7, -1.9, -1.3, -3.5, -0.5, -0.1,  0.3,  1.5],      dtype=float32)
"""

from matfree.backend import containers, control_flow, np
from matfree.backend.typing import Array


def matrix_poly_vector_product(matrix_poly_alg, /):
    """Implement a matrix-function-vector product via a polynomial expansion.

    Parameters
    ----------
    matrix_poly_alg
        Which algorithm to use.
        For example, the output of
        [matrix_poly_chebyshev][matfree.matfun.matrix_poly_chebyshev].
    """
    lower, upper, init_func, step_func, extract_func = matrix_poly_alg

    def matvec(vec, *parameters):
        final_state = control_flow.fori_loop(
            lower=lower,
            upper=upper,
            body_fun=lambda _i, v: step_func(v, *parameters),
            init_val=init_func(vec, *parameters),
        )
        return extract_func(final_state)

    return matvec


def _chebyshev_nodes(n, /):
    k = np.arange(n, step=1.0) + 1
    return np.cos((2 * k - 1) / (2 * n) * np.pi())


def matrix_poly_chebyshev(matfun, order, matvec, /):
    """Construct an implementation of matrix-Chebyshev-polynomial interpolation.

    This function assumes that the spectrum of the matrix-vector product
    is contained in the interval (-1, 1), and that the matrix-function
    is analytic on this interval.
    If this is not the case,
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

    return 0, order - 1, init_func, recursion_func, extract_func
