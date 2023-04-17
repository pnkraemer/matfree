"""Tests for Monte-Carlo machinery."""

from hutch import montecarlo
from hutch.backend import np, prng, testing


@testing.fixture(name="f_mc")
def fixture_f_mc():
    def f(x):
        return x**2

    return montecarlo.montecarlo(f, sample_fn=prng.normal)


@testing.parametrize("key", [prng.PRNGKey(1)])
def test_mean_vmap(f_mc, key):
    f_mc_mean = montecarlo.mean_vmap(f_mc, 10_000)
    received = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)


@testing.parametrize("key", [prng.PRNGKey(1)])
def test_mean_map(f_mc, key):
    f_mc_mean = montecarlo.mean_map(f_mc, 10_000)
    received = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)


@testing.parametrize("key", [prng.PRNGKey(1)])
def test_mean_loop(f_mc, key):
    f_mc_mean = montecarlo.mean_loop(f_mc, 10_000)
    received = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)


@testing.parametrize("key", [prng.PRNGKey(1)])
def test_mean_nested_loop_map(f_mc, key):
    f_mc_mean = montecarlo.mean_loop(montecarlo.mean_vmap(f_mc, 5), 10_000)
    received = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)


@testing.parametrize("key", [prng.PRNGKey(1)])
def test_mean_nested_vmap_map(f_mc, key):
    f_mc_mean = montecarlo.mean_vmap(montecarlo.mean_vmap(f_mc, 5), 10_000)
    received = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)
