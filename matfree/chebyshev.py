"""Chebyshev approximation of matrix functions."""

from matfree.backend import containers, control_flow, np
from matfree.backend.typing import Array


def chebyshev(matfun, order, matvec, /):
    # Prepare Chebyshev approximation
    nodes = _chebyshev_nodes(order)
    fx_nodes = matfun(nodes)

    def estimate(vec, *parameters):
        # Apply Chebyshev recursion
        init_func, recursion_func = _chebyshev(nodes, fx_nodes, matvec)
        final_state = control_flow.fori_loop(
            lower=0,
            upper=len(nodes) - 1,
            body_fun=lambda _i, v: recursion_func(v, *parameters),
            init_val=init_func(vec, *parameters),
        )
        return final_state.interpolation

    return estimate


def _chebyshev_nodes(n, /):
    k = np.arange(n, step=1.0) + 1
    return np.cos((2 * k - 1) / (2 * n) * np.pi())


def _chebyshev_scalar_init(x):
    return x, np.ones_like(x)


def _chebyshev_matvec_init(matvec, vec, *parameters):
    return matvec(vec, *parameters), vec


def _chebyshev(nodes, fx_nodes, matvec):
    class _ChebyshevState(containers.NamedTuple):
        interpolation: Array
        poly_coefficients: tuple[Array, Array]
        poly_values: tuple[Array, Array]

    def init_func(vec, *parameters):
        # Initialize the scalar recursion
        # (needed to compute the interpolation weights)
        t2_n, t1_n = _chebyshev_scalar_init(nodes)
        c1 = np.mean(fx_nodes * t1_n)
        c2 = 2 * np.mean(fx_nodes * t2_n)

        # Initialize the vector-valued recursion
        # (this is where the matvec happens)
        t2_x, t1_x = _chebyshev_matvec_init(matvec, vec, *parameters)
        value = c1 * t1_x + c2 * t2_x
        return _ChebyshevState(value, (t2_n, t1_n), (t2_x, t1_x))

    def recursion_func(val: _ChebyshevState, *parameters) -> _ChebyshevState:
        value, (t2_n, t1_n), (t2_x, t1_x) = val

        t2_n, t1_n = _chebyshev_scalar_step(nodes, (t2_n, t1_n))
        c2 = 2 * np.mean(fx_nodes * t2_n)

        t2_x, t1_x = _chebyshev_matvec_step(matvec, *parameters, ts=(t2_x, t1_x))
        value += c2 * t2_x
        return _ChebyshevState(value, (t2_n, t1_n), (t2_x, t1_x))

    return init_func, recursion_func


def _chebyshev_scalar_step(x, ts):
    t2, t1 = ts
    return 2 * x * t2 - t1, t2


def _chebyshev_matvec_step(matvec, *parameters, ts):
    t2, t1 = ts
    return 2 * matvec(t2, *parameters) - t1, t2
