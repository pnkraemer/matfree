"""Tests for basic trace estimators."""

from matfree import hutchinson, montecarlo
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
@testing.parametrize("sample_fun", [montecarlo.normal, montecarlo.rademacher])
def test_trace(fun, key, num_batches, num_samples_per_batch, dim, sample_fun):
    """Assert that the estimated trace approximates the true trace accurately."""
    # Linearise function
    x0 = prng.uniform(key, shape=(dim,))  # random lin. point
    _, jvp = func.linearize(fun, x0)
    J = func.jacfwd(fun)(x0)

    # Estimate the trace
    fun = sample_fun(shape=np.shape(x0), dtype=np.dtype(x0))
    estimate = hutchinson.trace(
        jvp,
        num_batches=num_batches,
        key=key,
        num_samples_per_batch=num_samples_per_batch,
        sample_fun=fun,
    )
    truth = linalg.trace(J)
    assert np.allclose(estimate, truth, rtol=1e-2)
