"""Tests for basic trace estimators."""

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
    return prng.PRNGKey(seed=1)


@testing.parametrize("num_batches", [1_000])
@testing.parametrize("num_samples_per_batch", [1_000])
@testing.parametrize("dim", [1, 10])
@testing.parametrize("sample_fun", [montecarlo.normal, montecarlo.rademacher])
def test_frobeniusnorm_squared(
    fun, key, num_batches, num_samples_per_batch, dim, sample_fun
):
    """Assert that the Frobenius norm estimate is accurate."""
    # Linearise function
    x0 = prng.uniform(key, shape=(dim,))  # random lin. point
    _, jvp = func.linearize(fun, x0)
    J = func.jacfwd(fun)(x0)

    # Estimate the trace
    fun = sample_fun(shape=np.shape(x0), dtype=np.dtype(x0))
    estimate = hutch.frobeniusnorm_squared(
        jvp,
        num_batches=num_batches,
        key=key,
        num_samples_per_batch=num_samples_per_batch,
        sample_fun=fun,
    )
    truth = np.trace(J.T @ J)
    assert np.allclose(estimate, truth, rtol=1e-2)


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
    estimate = hutch.trace(
        jvp,
        num_batches=num_batches,
        key=key,
        num_samples_per_batch=num_samples_per_batch,
        sample_fun=fun,
    )
    truth = np.trace(J)
    assert np.allclose(estimate, truth, rtol=1e-2)


@testing.parametrize("num_batches", [1_000])
@testing.parametrize("num_samples_per_batch", [1_000])
@testing.parametrize("dim", [1, 10])
@testing.parametrize("sample_fun", [montecarlo.normal, montecarlo.rademacher])
def test_diagonal(fun, key, num_batches, num_samples_per_batch, dim, sample_fun):
    """Assert that the estimated diagonal approximates the true diagonal accurately."""
    # Linearise function
    x0 = prng.uniform(key, shape=(dim,))  # random lin. point
    _, jvp = func.linearize(fun, x0)
    J = func.jacfwd(fun)(x0)

    # Estimate the trace
    fun = sample_fun(shape=np.shape(x0), dtype=np.dtype(x0))
    estimate = hutch.diagonal(
        jvp,
        num_batches=num_batches,
        key=key,
        num_samples_per_batch=num_samples_per_batch,
        sample_fun=fun,
    )
    truth = np.diagonal(J)
    assert np.allclose(estimate, truth, rtol=1e-2)


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
    trace, diag = hutch.trace_and_diagonal(
        jvp, key=key, num_levels=num_samples, sample_fun=fun
    )

    # Print errors if test fails
    error_diag = linalg.norm(diag - np.diagonal(J))
    error_trace = linalg.norm(trace - np.trace(J))
    assert np.allclose(diag, np.diagonal(J), rtol=1e-2), error_diag
    assert np.allclose(trace, np.trace(J), rtol=1e-2), error_trace
