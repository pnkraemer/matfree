"""Tests for Monte-Carlo machinery."""

from matfree import montecarlo
from matfree.backend import np, prng, testing


def test_van_der_corput():
    """Assert that the van-der-Corput sequence yields values as expected."""
    expected = np.asarray([0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625])
    received = np.asarray([montecarlo.van_der_corput(i) for i in range(9)])
    assert np.allclose(received, expected)

    expected = np.asarray([0.0, 1 / 3, 2 / 3, 1 / 9, 4 / 9, 7 / 9, 2 / 9, 5 / 9, 8 / 9])
    received = np.asarray([montecarlo.van_der_corput(i, base=3) for i in range(9)])
    assert np.allclose(received, expected)


@testing.fixture(name="f_mc")
def fixture_f_mc():
    """Fix a Monte-Carlo problem."""

    def f(x):
        return x**2

    return montecarlo.montecarlo(f, sample_fun=prng.normal)


_ALL_MEAN_FNS = [montecarlo.mean_vmap, montecarlo.mean_map, montecarlo.mean_loop]


@testing.parametrize("key", [prng.PRNGKey(1)])
@testing.parametrize("mean_fun", _ALL_MEAN_FNS)
def test_mean(f_mc, key, mean_fun):
    """Assert that the mean estimate is accurate."""
    f_mc_mean = mean_fun(f_mc, 10_000)
    received, _num_nans = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)


@testing.parametrize("key", [prng.PRNGKey(1)])
@testing.parametrize("mean_fun1, mean_fun2", (_ALL_MEAN_FNS[1:], _ALL_MEAN_FNS[:-1]))
def test_mean_nested(f_mc, key, mean_fun1, mean_fun2):
    """Assert that nested mean-computation remains accurate."""
    f_mc_mean = mean_fun1(mean_fun2(f_mc, 5), 10_000)
    received, _num_nans = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)


@testing.parametrize("key", [prng.PRNGKey(1)])
@testing.parametrize("mean_fun", _ALL_MEAN_FNS)
def test_mean_many_nans(key, mean_fun):
    """Assert that NaNs are captured (in an all-NaN situtation)."""

    def f(x):
        return np.nan * np.ones_like(x)

    f_mc = montecarlo.montecarlo(f, sample_fun=prng.normal)
    f_mc_mean = mean_fun(f_mc, 10_000)
    _received, num_nans = f_mc_mean(key)
    assert num_nans == 10_000


@testing.parametrize("key", [prng.PRNGKey(1)])
@testing.parametrize(
    "mean_fun1, mean_fun2, mean_fun3",
    zip(_ALL_MEAN_FNS[1:], _ALL_MEAN_FNS[:-1], _ALL_MEAN_FNS[1:], strict=True),
)
def test_mean_many_nans_nested(key, mean_fun1, mean_fun2, mean_fun3):
    """Assert that nested mean-computation captures NaNs (in an all-Nan situation)."""

    def f(x):
        return np.nan * np.ones_like(x)

    n1, n2, n3 = 3, 4, 5
    f_mc = montecarlo.montecarlo(f, sample_fun=prng.normal)
    f_mc_mean = mean_fun3(mean_fun2(mean_fun1(f_mc, n1), n2), n3)
    _received, num_nans = f_mc_mean(key)
    assert num_nans == n1 * n2 * n3
