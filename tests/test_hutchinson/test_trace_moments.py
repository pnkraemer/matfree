"""Tests for estimating higher moments of traces."""

from matfree import hutchinson
from matfree.backend import func, linalg, np, prng, testing


@testing.fixture(name="key")
def fixture_key():
    """Fix a pseudo-random number generator."""
    return prng.prng_key(seed=1)


@testing.fixture(name="J_and_jvp")
def fixture_J_and_jvp(key):
    """Create a nonlinear, to-be-differentiated function."""

    def fun(x):
        return np.sin(np.flip(np.cos(x)) + 1.0) * np.sin(x) + 1.0

    # Linearise function
    x0 = prng.uniform(key, shape=(2,))  # random lin. point
    _, jvp = func.linearize(fun, x0)
    J = func.jacfwd(fun)(x0)

    return J, jvp, x0


def test_variance_normal(J_and_jvp, key):
    """Assert that the estimated trace approximates the true trace accurately."""
    # Estimate the trace
    J, jvp, args_like = J_and_jvp
    problem = hutchinson.integrand_trace_moments(jvp, [1, 2])
    sampler = hutchinson.sampler_normal(args_like, num=1_000_000)
    estimate = hutchinson.hutchinson(problem, sample_fun=sampler, stats_fun=np.mean)
    first, second = estimate(key)

    # Assert the trace is correct
    truth = linalg.trace(J)
    assert np.allclose(first, truth, rtol=1e-2)

    # Assert the variance is correct:
    norm = linalg.matrix_norm(J, which="fro") ** 2
    assert np.allclose(second - first**2, norm * 2, rtol=1e-2)


def test_variance_rademacher(J_and_jvp, key):
    """Assert that the estimated trace approximates the true trace accurately."""
    # Estimate the trace
    J, jvp, args_like = J_and_jvp

    problem = hutchinson.integrand_trace_moments(jvp, [1, 2])
    sampler = hutchinson.sampler_rademacher(args_like, num=500)
    estimate = hutchinson.hutchinson(problem, sample_fun=sampler, stats_fun=np.mean)
    first, second = estimate(key)

    # Assert the trace is correct
    truth = linalg.trace(J)
    assert np.allclose(first, truth, rtol=1e-2)

    # Assert the variance is correct:
    norm = linalg.matrix_norm(J, which="fro") ** 2
    truth = norm - linalg.trace(J**2)
    assert np.allclose(second - first**2, truth, atol=1e-2, rtol=1e-2)
