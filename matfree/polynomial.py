"""Approximate matrix-function-vector products with polynomial expansions.

This module does not include Lanczos-style approximations, which are
in [matfree.lanczos][matfree.lanczos].
"""

from matfree.backend import containers, control_flow, np
from matfree.backend.typing import Array


def funm_vector_product_chebyshev(matfun, order, matvec, /):
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
    return _funm_vector_product_polyexpand(alg)


def _chebyshev_nodes(n, /):
    k = np.arange(n, step=1.0) + 1
    return np.cos((2 * k - 1) / (2 * n) * np.pi())


def _funm_vector_product_polyexpand(matrix_poly_alg, /):
    """Implement a matrix-function-vector product via a polynomial expansion."""
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
