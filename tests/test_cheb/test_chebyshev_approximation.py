from matfree import chebyshev, test_util
from matfree.backend import control_flow, linalg, np, prng


def chebyshev_init(x):
    return x, np.ones_like(x)


def chebyshev_step(x, ts):
    t2, t1 = ts
    return 2 * x * t2 - t1, t2


def chebyshev_approximate(fun, order, x):
    nodes = chebyshev_nodes(order)
    fx_nodes = fun(nodes)

    t2_n, t1_n = chebyshev_init(nodes)
    c1 = np.mean(fx_nodes * t1_n)
    c2 = 2 * np.mean(fx_nodes * t2_n)

    t2_x, t1_x = chebyshev_init(x)
    value = c1 * t1_x + c2 * t2_x

    def body(_i, val):
        return _chebyshev_body_fun(val, nodes=nodes, fx_nodes=fx_nodes, x=x)

    init = value, (t2_n, t1_n), (t2_x, t1_x)
    value, *_ = control_flow.fori_loop(0, len(nodes) - 1, body, init)
    return value


def _chebyshev_body_fun(val, nodes, fx_nodes, x):
    value, (t2_n, t1_n), (t2_x, t1_x) = val
    t2_n, t1_n = chebyshev_step(nodes, (t2_n, t1_n))
    c2 = 2 * np.mean(fx_nodes * t2_n)
    t2_x, t1_x = chebyshev_step(x, (t2_x, t1_x))
    value += c2 * t2_x
    return value, (t2_n, t1_n), (t2_x, t1_x)


def chebyshev_nodes(n, /):
    k = np.arange(n, step=1.0) + 1
    return np.cos((2 * k - 1) / (2 * n) * np.pi())


def test_interpolate_function():
    def fun(x):
        return x**4

    x0 = np.linspace(-0.9, 0.9, num=5, endpoint=True)
    order = 5
    approximate = chebyshev_approximate(fun, order, x0)

    assert np.allclose(approximate, fun(x0), atol=1e-6)


def test_chebyshev_approximate(n=4):
    eigvals = np.linspace(-0.99, 0.99, num=n)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    def matvec(x, p):
        return p @ x

    def matfun(x):
        return np.sin(x)

    eigvals, eigvecs = linalg.eigh(matrix)
    log_matrix = eigvecs @ linalg.diagonal(matfun(eigvals)) @ eigvecs.T

    v = prng.normal(prng.prng_key(2), shape=(n,))
    expected = log_matrix @ v

    order = 6
    estimate = chebyshev.chebyshev(matfun, order, matvec)
    received = estimate(v, matrix)
    assert np.allclose(expected, received)
