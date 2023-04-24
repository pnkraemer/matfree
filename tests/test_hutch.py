"""Tests for basic trace estimators."""

from hutch import hutch
from hutch.backend import linalg, np, prng, testing, transform


@testing.fixture(name="fun")
def fixture_fun():
    def f(x):
        return np.sin(np.flip(np.cos(x)) + 1.0) * np.sin(x) + 1.0

    return f


@testing.fixture(name="key")
def fixture_key():
    return prng.PRNGKey(seed=1)


@testing.parametrize("num_batches", [1_000])
@testing.parametrize("num_samples_per_key", [1_000])
@testing.parametrize("dim", [1, 10])
@testing.parametrize("sample_fun", [prng.normal, prng.rademacher])
def test_trace(fun, key, num_batches, num_samples_per_key, dim, sample_fun):
    # Linearise function
    x0 = prng.uniform(key, shape=(dim,))  # random lin. point
    _, jvp = transform.linearize(fun, x0)
    J = transform.jacfwd(fun)(x0)

    # Estimate the trace
    estimate = hutch.trace(
        matvec_fun=jvp,
        tangents_shape=np.shape(x0),
        tangents_dtype=np.dtype(x0),
        num_batches=num_batches,
        key=key,
        num_samples_per_batch=num_samples_per_key,
        sample_fun=sample_fun,
    )
    truth = np.trace(J)
    assert np.allclose(estimate, truth, rtol=1e-2)


@testing.parametrize("num_batches", [1_000])
@testing.parametrize("num_samples_per_key", [1_000])
@testing.parametrize("dim", [1, 10])
@testing.parametrize("sample_fun", [prng.normal, prng.rademacher])
def test_diagonal(fun, key, num_batches, num_samples_per_key, dim, sample_fun):
    # Linearise function
    x0 = prng.uniform(key, shape=(dim,))  # random lin. point
    _, jvp = transform.linearize(fun, x0)
    J = transform.jacfwd(fun)(x0)

    # Estimate the trace
    estimate = hutch.diagonal(
        matvec_fun=jvp,
        tangents_shape=np.shape(x0),
        tangents_dtype=np.dtype(x0),
        num_batches=num_batches,
        key=key,
        num_samples_per_batch=num_samples_per_key,
        sample_fun=sample_fun,
    )
    truth = np.diag(J)
    assert np.allclose(estimate, truth, rtol=1e-2)


@testing.parametrize("num_samples", [10_000])
@testing.parametrize("dim", [5])
@testing.parametrize("sample_fun", [prng.normal, prng.rademacher])
def test_trace_and_diagonal(fun, key, num_samples, dim, sample_fun):
    # Linearise function
    x0 = prng.uniform(key, shape=(dim,))
    _, jvp = transform.linearize(fun, x0)
    J = transform.jacfwd(fun)(x0)

    # Sequential batches
    keys = prng.split(key, num=num_samples)

    # Estimate the trace
    trace, diag = hutch.trace_and_diagonal(
        matvec_fun=jvp,
        tangents_shape=np.shape(x0),
        tangents_dtype=np.dtype(x0),
        keys=keys,
        sample_fun=sample_fun,
    )
    assert np.allclose(diag, np.diag(J), rtol=1e-2), linalg.norm(diag - np.diag(J))
    assert np.allclose(trace, np.trace(J), rtol=1e-2), linalg.norm(trace - np.trace(J))
