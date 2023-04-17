"""Tests for Monte-Carlo machinery."""

from hutch import montecarlo
from hutch.backend import np, prng, testing


@testing.fixture(name="f_mc")
def fixture_f_mc():
    def f(x):
        return x**2

    return montecarlo.montecarlo(f, sample_fn=prng.normal)


_ALL_MEAN_FNS = [montecarlo.mean_vmap, montecarlo.mean_map, montecarlo.mean_loop]


@testing.parametrize("key", [prng.PRNGKey(1)])
@testing.parametrize("mean_fn", _ALL_MEAN_FNS)
def test_mean(f_mc, key, mean_fn):
    f_mc_mean = mean_fn(f_mc, 10_000)
    received, info = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)
    assert isinstance(info, dict)


@testing.parametrize("key", [prng.PRNGKey(1)])
@testing.parametrize("mean_fn", _ALL_MEAN_FNS)
def test_mean_parametrised(f_mc, key, mean_fn):
    f_mc_mean = mean_fn(f_mc, 10_000)
    received, info = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)
    assert isinstance(info, dict)


@testing.parametrize("key", [prng.PRNGKey(1)])
@testing.parametrize("mean_fn1, mean_fn2", (_ALL_MEAN_FNS[1:], _ALL_MEAN_FNS[:-1]))
def test_mean_nested(f_mc, key, mean_fn1, mean_fn2):
    f_mc_mean = montecarlo.mean_loop(montecarlo.mean_vmap(f_mc, 5), 10_000)
    received, info = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)
    assert isinstance(info, dict)
