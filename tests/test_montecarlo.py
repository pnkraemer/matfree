"""Tests for Monte-Carlo machinery."""

from hutch import montecarlo
from hutch.backend import np, prng


def test_montecarlo():
    def f(x):
        return x**2

    f_mc = montecarlo.montecarlo(f, sample_fn=prng.normal)

    key = prng.PRNGKey(1)

    assert np.allclose(f_mc(key), f(prng.normal(key)))

    print(f_mc)
