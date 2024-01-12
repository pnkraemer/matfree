import matplotlib.pyplot as plt

from matfree.backend import np


def chebyshev_init(x):
    return x, np.ones_like(x)


def chebyshev_step(x, ts):
    t2, t1 = ts
    return 2 * x * t2 - t1, t2


def chebyshev_approximate(fun, nodes, x):
    # nodes = x
    fx_nodes = fun(nodes)

    t2_n, t1_n = chebyshev_init(nodes)
    c1 = np.mean(fx_nodes * t1_n)
    c2 = 2 * np.mean(fx_nodes * t2_n)
    print()
    print(0, c1)
    print(1, c2)
    t2_x, t1_x = chebyshev_init(x)
    value = c1 * t1_x + c2 * t2_x
    for i in range(len(nodes)):
        t2_n, t1_n = chebyshev_step(nodes, (t2_n, t1_n))
        c2 = 2 * np.mean(fx_nodes * t2_n)
        print(i + 2, c2)

        t2_x, t1_x = chebyshev_step(x, (t2_x, t1_x))
        value += c2 * t2_x
    print()
    return value


def chebyshev_nodes(n, /):
    k = np.arange(n, step=1.0) + 1
    return np.cos((2 * k - 1) / (2 * n) * np.pi())


def test_interpolate_function():
    def fun(x):
        return np.sin(x)

    x0 = np.linspace(-0.9, 0.9, num=5, endpoint=True)

    nodes = chebyshev_nodes(10)
    # print(nodes)
    # t1, t0 = chebyshev_init(nodes)
    # c0 = np.mean(fun(nodes) * t0)
    # c1 = 2*np.mean(fun(nodes) * t1)

    approximate = chebyshev_approximate(fun, nodes, x0)
    print("Approximation\t", approximate)
    print("Solution\t", fun(x0))
    # print("Ration\t", approximate / fun(x0))
    assert False
