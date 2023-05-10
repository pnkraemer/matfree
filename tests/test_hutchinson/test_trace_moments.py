"""Tests for estimating traces."""

from matfree import hutchinson, montecarlo
from matfree.backend import func, linalg, np, prng, testing


@testing.fixture(name="key")
def fixture_key():
    """Fix a pseudo-random number generator."""
    return prng.prng_key(seed=1)


@testing.fixture(name="J_and_jvp")
def fixture_J_and_jvp(key, dim):
    """Create a nonlinear, to-be-differentiated function."""

    def fun(x):
        return np.sin(np.flip(np.cos(x)) + 1.0) * np.sin(x) + 1.0

    # Linearise function
    x0 = prng.uniform(key, shape=(dim,))  # random lin. point
    _, jvp = func.linearize(fun, x0)
    J = func.jacfwd(fun)(x0)

    return J, jvp


@testing.parametrize("num_batches", [1_000])
@testing.parametrize("num_samples_per_batch", [1_000])
@testing.parametrize("dim", [1, 10])
def test_variance_normal(J_and_jvp, key, num_batches, num_samples_per_batch, dim):
    """Assert that the estimated trace approximates the true trace accurately."""
    # Estimate the trace
    J, jvp = J_and_jvp
    fun = montecarlo.normal(shape=(dim,), dtype=float)
    first, second = hutchinson.trace_moments(
        jvp,
        key=key,
        num_batches=num_batches,
        num_samples_per_batch=num_samples_per_batch,
        sample_fun=fun,
        moments=(1, 2),
    )

    # Assert the trace is correct
    truth = linalg.trace(J)
    assert np.allclose(first, truth, rtol=1e-2)

    # Assert the variance is correct:
    norm = linalg.matrix_norm(J, which="fro") ** 2
    assert np.allclose(second - first**2, norm * 2, rtol=1e-2)


@testing.parametrize("num_batches", [1_000])
@testing.parametrize("num_samples_per_batch", [1_000])
@testing.parametrize("dim", [1, 10])
def test_variance_rademacher(J_and_jvp, key, num_batches, num_samples_per_batch, dim):
    """Assert that the estimated trace approximates the true trace accurately."""
    # Estimate the trace
    J, jvp = J_and_jvp
    fun = montecarlo.rademacher(shape=(dim,), dtype=float)
    first, second = hutchinson.trace_moments(
        jvp,
        key=key,
        num_batches=num_batches,
        num_samples_per_batch=num_samples_per_batch,
        sample_fun=fun,
        moments=(1, 2),
    )

    # Assert the trace is correct
    truth = linalg.trace(J)
    assert np.allclose(first, truth, rtol=1e-2)

    # Assert the variance is correct:
    norm = linalg.matrix_norm(J, which="fro") ** 2
    truth = 2 * (norm - linalg.trace(J**2))
    assert np.allclose(second - first**2, truth, atol=1e-2, rtol=1e-2)
