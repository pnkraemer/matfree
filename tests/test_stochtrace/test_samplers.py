"""Test samplers for stochastic trace estimation."""

from matfree import stochtrace
from matfree.backend import np, prng, testing


@testing.parametrize("n", [5])
@testing.parametrize("dtype", [float, complex])
def test_sampler_sphere(n, dtype):
    """Assert that the sampler_sphere samples from a unit sphere scaled to have identity covariance."""

    num_samples = 100_000
    key1, key2 = prng.split(prng.prng_key(1), 2)
    sampler = stochtrace.sampler_sphere(np.ones(n, dtype=dtype), num=num_samples)
    x = sampler(key1)
    if dtype is complex:
        # hack to check that the result is complex
        assert not np.allclose(x.imag, 0)
    # Verify moments
    assert np.allclose(np.mean(x, axis=0), 0, atol=1e-2)
    assert np.allclose(
        np.cov(x, rowvar=False, ddof=0), np.eye(n, dtype=dtype), atol=1e-2
    )

    # Verify determinism (same key)
    x_again = sampler(key1)
    assert np.allclose(x, x_again)

    # Verify that samples differ if the key changes
    y = sampler(key2)
    assert not np.allclose(x, y)


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


@testing.parametrize("n", [5])
@testing.parametrize("dtype", [float, complex])
def test_sampler_signs(n, dtype):
    """Assert that the sampler_signs samples from a Rademacher/Steinhaus distribution."""
    num_samples = 100_000
    key = prng.prng_key(1)
    x_like = np.ones(n, dtype=dtype)
    sampler = stochtrace.sampler_signs(x_like, num=num_samples)
    x = sampler(key)

    # Verify basic properties
    assert np.allclose(np.abs(x), 1)
    if dtype is complex:
        assert x.dtype.kind == "c"
        assert not np.allclose(x.imag, 0)
    else:
        assert x.dtype.kind != "c"
        assert np.allclose(x.imag, 0)

    # Verify moments
    assert np.allclose(np.mean(x), 0, atol=1e-2)
    assert np.allclose(
        np.cov(x, rowvar=False, ddof=0), np.eye(n, dtype=dtype), atol=1e-2
    )

    # Verify determinism (same key)
    x_again = sampler(key)
    assert np.allclose(x, x_again)

    # Verify that samples differ if the key changes
    y = sampler(prng.prng_key(2))
    assert not np.allclose(x, y)
