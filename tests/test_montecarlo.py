"""Tests for Monte-Carlo machinery."""

from hutch import montecarlo
from hutch.backend import np, prng, testing


@testing.parametrize("sample_fn", [prng.normal])
def test_montecarlo(sample_fn):
    def f(x):
        return x**2

    key = prng.PRNGKey(1)
    f_mc = montecarlo.montecarlo(f, sample_fn=sample_fn)

    # Vmap
    f_mc_mean = montecarlo.mean_vmap(f_mc, 10_000)
    received, _isnan = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)

    # Map
    f_mc_mean = montecarlo.mean_map(f_mc, 10_000)
    received, _isnan = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)

    # For-loop
    f_mc_mean = montecarlo.mean_loop(f_mc, 10_000)
    received, _isnan = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)

    # Nested
    f_mc_mean = montecarlo.mean_loop(montecarlo.mean_vmap(f_mc, 5), 10_000)
    received, _isnan = f_mc_mean(key)
    assert np.allclose(received, 1.0, rtol=1e-2)
