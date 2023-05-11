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


@testing.parametrize("num_samples", [10_000])
@testing.parametrize("dim", [5])
@testing.parametrize("sample_fun", [montecarlo.normal, montecarlo.rademacher])
def test_trace_and_diagonal(fun, key, num_samples, dim, sample_fun):
    """Assert that the estimated trace and diagonal approximations are accurate."""
    # Linearise function
    x0 = prng.uniform(key, shape=(dim,))
    _, jvp = func.linearize(fun, x0)
    J = func.jacfwd(fun)(x0)

    # Estimate the trace
    fun = sample_fun(shape=np.shape(x0), dtype=np.dtype(x0))
    trace, diag = hutchinson.trace_and_diagonal(
        jvp, key=key, num_levels=num_samples, sample_fun=fun
    )

    # Print errors if test fails
    error_diag = linalg.vector_norm(diag - linalg.diagonal(J))
    error_trace = linalg.vector_norm(trace - linalg.trace(J))
    assert np.allclose(diag, linalg.diagonal(J), rtol=1e-2), error_diag
    assert np.allclose(trace, linalg.trace(J), rtol=1e-2), error_trace
