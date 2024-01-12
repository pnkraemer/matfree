"""Chebyshev approximation of matrix functions."""

from matfree.backend import control_flow, np


def chebyshev(matfun, order, matvec, /):
    nodes = chebyshev_nodes(order)
    fx_nodes = matfun(nodes)

    def estimate(vec, *parameters):
        t2_n, t1_n = chebyshev_scalar_init(nodes)
        c1 = np.mean(fx_nodes * t1_n)
        c2 = 2 * np.mean(fx_nodes * t2_n)

        t2_x, t1_x = chebyshev_matvec_init(matvec, vec, *parameters)
        value = c1 * t1_x + c2 * t2_x

        def body(_i, val):
            return _chebyshev_body_fun(val, nodes, fx_nodes, matvec, *parameters)

        init = value, (t2_n, t1_n), (t2_x, t1_x)
        value, *_ = control_flow.fori_loop(0, len(nodes) - 1, body, init)
        return value

    return estimate


def chebyshev_nodes(n, /):
    k = np.arange(n, step=1.0) + 1
    return np.cos((2 * k - 1) / (2 * n) * np.pi())


def chebyshev_scalar_init(x):
    return x, np.ones_like(x)


def chebyshev_matvec_init(matvec, vec, *parameters):
    return matvec(vec, *parameters), vec


def _chebyshev_body_fun(val, nodes, fx_nodes, matvec, *parameters):
    value, (t2_n, t1_n), (t2_x, t1_x) = val

    t2_n, t1_n = chebyshev_scalar_step(nodes, (t2_n, t1_n))
    c2 = 2 * np.mean(fx_nodes * t2_n)

    t2_x, t1_x = chebyshev_matvec_step(matvec, *parameters, ts=(t2_x, t1_x))
    value += c2 * t2_x
    return value, (t2_n, t1_n), (t2_x, t1_x)


def chebyshev_scalar_step(x, ts):
    t2, t1 = ts
    return 2 * x * t2 - t1, t2


def chebyshev_matvec_step(matvec, *parameters, ts):
    t2, t1 = ts
    return 2 * matvec(t2, *parameters) - t1, t2
