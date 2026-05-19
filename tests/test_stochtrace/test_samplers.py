"""Test samplers for stochastic trace estimation."""

from matfree import stochtrace
from matfree.backend import np, prng, testing


@testing.parametrize("n", [5])
@testing.parametrize("dtype", [float, complex])
def test_sampler_sphere(n, dtype):
    """Assert that the sampler_sphere samples from a unit sphere scaled to have identity covariance."""

    num_samples = 100000
    key1, key2 = prng.split(prng.prng_key(1), 2)
    sampler = stochtrace.sampler_sphere(np.ones(n, dtype=dtype), num=num_samples)
    x = sampler(key1)
    if dtype is complex:
        # hack to check that the result is complex
        assert not np.allclose(x.imag, 0)
    x2 = sampler(key1)
    y = sampler(key2)
    assert np.allclose(x, x2)
    assert not np.allclose(x, y)
    assert np.allclose(np.mean(x, axis=0), 0, atol=1e-2)
    assert np.allclose(
        (x.T.conj() @ x) / num_samples, np.eye(n).astype(dtype), atol=1e-2
    )


def test_sampler_sphere_pytrees():
    """Assert that the sampler_sphere supports pytrees."""
    n1, n2, num_samples = 3, 4, 5
    x = {"a": np.ones(n1), "b": np.ones(n2)}
    sampler = stochtrace.sampler_sphere(x, num=num_samples)
    x = sampler(prng.prng_key(1))
    assert type(x) is dict
    assert set(x.keys()) == {"a", "b"}
    assert x["a"].shape == (num_samples, n1)
    assert x["b"].shape == (num_samples, n2)
