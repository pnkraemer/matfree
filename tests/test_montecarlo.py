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


@testing.parametrize("key", [prng.PRNGKey(1)])
@testing.parametrize("num_batches, num_samples", [[1, 10_000], [10_000, 1], [100, 100]])
def test_mean(key, num_batches, num_samples):
    """Assert that the mean estimate is accurate."""

    def fun(x):
        return x**2

    received = montecarlo.estimate(
        fun,
        num_batches=num_batches,
        num_samples_per_batch=num_samples,
        key=key,
        sample_fun=montecarlo.normal(shape=()),
    )
    assert np.allclose(received, 1.0, rtol=1e-1)
