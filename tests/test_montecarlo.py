"""Tests for Monte-Carlo machinery."""

from matfree import montecarlo
from matfree.backend import np, prng, testing


@testing.fixture(name="f_mc")
def fixture_f_mc():
    def f(x):
        return x**2

    return montecarlo.montecarlo(f, sample_fun=prng.normal)


_ALL_MEAN_FNS = [montecarlo.mean_vmap, montecarlo.mean_map, montecarlo.mean_loop]


@testing.parametrize("key", [prng.PRNGKey(1)])
@testing.parametrize("mean_fun", _ALL_MEAN_FNS)
def test_mean(f_mc, key, mean_fun):
    f_mc_mean = mean_fun(f_mc, 10_000)
    received, _num_nans = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)


@testing.parametrize("key", [prng.PRNGKey(1)])
@testing.parametrize("mean_fun1, mean_fun2", (_ALL_MEAN_FNS[1:], _ALL_MEAN_FNS[:-1]))
def test_mean_nested(f_mc, key, mean_fun1, mean_fun2):
    f_mc_mean = montecarlo.mean_loop(montecarlo.mean_vmap(f_mc, 5), 10_000)
    received, _num_nans = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)


@testing.parametrize("key", [prng.PRNGKey(1)])
@testing.parametrize("mean_fun", _ALL_MEAN_FNS)
def test_mean_many_nans(key, mean_fun):
    def f(x):
        return np.nan * np.ones_like(x)

    f_mc = montecarlo.montecarlo(f, sample_fun=prng.normal)
    f_mc_mean = mean_fun(f_mc, 10_000)
    _received, num_nans = f_mc_mean(key)
    assert num_nans == 10_000


@testing.parametrize("key", [prng.PRNGKey(1)])
@testing.parametrize(
    "mean_fun1, mean_fun2, mean_fun3",
    zip(_ALL_MEAN_FNS[1:], _ALL_MEAN_FNS[:-1], _ALL_MEAN_FNS[1:]),
)
def test_mean_many_nans_nested(key, mean_fun1, mean_fun2, mean_fun3):
    def f(x):
        return np.nan * np.ones_like(x)

    n1, n2, n3 = 3, 4, 5
    f_mc = montecarlo.montecarlo(f, sample_fun=prng.normal)
    f_mc_mean = mean_fun3(mean_fun2(mean_fun1(f_mc, n1), n2), n3)
    _received, num_nans = f_mc_mean(key)
    assert num_nans == n1 * n2 * n3
