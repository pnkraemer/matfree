"""Tests for Monte-Carlo machinery."""

from matfree import montecarlo
from matfree.backend import np, prng, testing


@testing.parametrize("key", [prng.prng_key(1)])
@testing.parametrize("num_batches, num_samples", [[1, 10_000], [10_000, 1], [100, 100]])
def test_mean(key, num_batches, num_samples):
    """Assert that the mean estimate is accurate."""

    def fun(x):
        return x**2

    mean, amax = montecarlo.multiestimate(
        fun,
        num_batches=num_batches,
        num_samples_per_batch=num_samples,
        key=key,
        sample_fun=montecarlo.normal(shape=()),
        stats_per_batch=[np.mean, np.array_max],
        stats_combine_batches=[np.mean, np.array_max],
    )

    assert np.allclose(mean, 1.0, rtol=1e-1)
    assert mean < amax
