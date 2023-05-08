"""Tests for estimating traces."""

from matfree import hutch, montecarlo
from matfree.backend import func, linalg, np, prng, testing


@testing.fixture(name="fun")
def fixture_fun():
    """Create a nonlinear, to-be-differentiated function."""

    def f(x):
        return np.sin(np.flip(np.cos(x)) + 1.0) * np.sin(x) + 1.0

    return f


@testing.fixture(name="key")
def fixture_key():
    """Fix a pseudo-random number generator."""
    return prng.prng_key(seed=1)


@testing.parametrize("num_batches", [1_000])
@testing.parametrize("num_samples_per_batch", [1_000])
@testing.parametrize("dim", [1, 10])
def test_normal(fun, key, num_batches, num_samples_per_batch, dim):
    """Assert that the estimated trace approximates the true trace accurately."""
    # Linearise function
    x0 = prng.uniform(key, shape=(dim,))  # random lin. point
    _, jvp = func.linearize(fun, x0)
    J = func.jacfwd(fun)(x0)

    # Estimate the trace
    fun = montecarlo.normal(shape=np.shape(x0), dtype=np.dtype(x0))
    trace, variance = hutch.trace_with_variance(
        jvp,
        key=key,
        num_batches=num_batches,
        num_samples_per_batch=num_samples_per_batch,
        sample_fun=fun,
    )

    # Assert the trace is correct
    truth = linalg.trace(J)
    assert np.allclose(trace, truth, rtol=1e-2)

    # Assert the variance is correct:
    norm = hutch.frobeniusnorm_squared(
        jvp,
        key=key,
        num_batches=num_batches,
        num_samples_per_batch=num_samples_per_batch,
        sample_fun=fun,
    )
    assert np.allclose(variance, norm * 2, rtol=1e-2)


@testing.parametrize("num_batches", [1_000])
@testing.parametrize("num_samples_per_batch", [1_000])
@testing.parametrize("dim", [1, 10])
def test_rademacher(fun, key, num_batches, num_samples_per_batch, dim):
    """Assert that the estimated trace approximates the true trace accurately."""
    # Linearise function
    x0 = prng.uniform(key, shape=(dim,))  # random lin. point
    _, jvp = func.linearize(fun, x0)
    J = func.jacfwd(fun)(x0)

    # Estimate the trace
    fun = montecarlo.rademacher(shape=np.shape(x0), dtype=np.dtype(x0))
    trace, variance = hutch.trace_with_variance(
        jvp,
        key=key,
        num_batches=num_batches,
        num_samples_per_batch=num_samples_per_batch,
        sample_fun=fun,
    )

    # Assert the trace is correct
    truth = linalg.trace(J)
    assert np.allclose(trace, truth, rtol=1e-2)

    # Assert the variance is correct:
    norm = hutch.frobeniusnorm_squared(
        jvp,
        key=key,
        num_batches=num_batches,
        num_samples_per_batch=num_samples_per_batch,
        sample_fun=fun,
    )
    truth = 2 * (norm - linalg.trace(J**2))
    assert np.allclose(variance, truth, atol=1e-2, rtol=1e-2)
