"""Tests for basic trace estimators."""

from hutch import hutch
from hutch.backend import linalg, np, prng, testing, transform


@testing.fixture(name="fn")
def fixture_fn():
    def f(x):
        return np.sin(np.flip(np.cos(x)) + 1.0) * np.sin(x) + 1.0

    return f


@testing.fixture(name="key")
def fixture_key():
    return prng.PRNGKey(seed=1)


@testing.parametrize("num_batches", [1_000])
@testing.parametrize("num_samples_per_key", [1_000])
@testing.parametrize("dim", [1, 10])
@testing.parametrize("generate_samples_fn", [prng.normal, prng.rademacher])
def test_trace(fn, key, num_batches, num_samples_per_key, dim, generate_samples_fn):
    # Linearise function
    x0 = prng.uniform(key, shape=(dim,))  # random lin. point
    _, jvp = transform.linearize(fn, x0)
    J = transform.jacfwd(fn)(x0)

    # Estimate the trace
    estimate = hutch.trace(
        matvec_fn=jvp,
        tangents_shape=np.shape(x0),
        tangents_dtype=np.dtype(x0),
        num_batches=num_batches,
        key=key,
        num_samples_per_batch=num_samples_per_key,
        generate_samples_fn=generate_samples_fn,
    )
    truth = np.trace(J)
    assert np.allclose(estimate, truth, rtol=1e-2)


@testing.parametrize("num_samples", [10_000])
@testing.parametrize("dim", [5])
@testing.parametrize("generate_samples_fn", [prng.normal, prng.rademacher])
def test_trace_and_diagonal(fn, key, num_samples, dim, generate_samples_fn):
    # Linearise function
    x0 = prng.uniform(key, shape=(dim,))
    _, jvp = transform.linearize(fn, x0)
    J = transform.jacfwd(fn)(x0)

    # Sequential batches
    keys = prng.split(key, num=num_samples)

    # Estimate the trace
    trace, diag = hutch.trace_and_diagonal(
        matvec_fn=jvp,
        tangents_shape=np.shape(x0),
        tangents_dtype=np.dtype(x0),
        keys=keys,
        generate_samples_fn=generate_samples_fn,
    )
    assert np.allclose(diag, np.diag(J), rtol=1e-2), linalg.norm(diag - np.diag(J))
    assert np.allclose(trace, np.trace(J), rtol=1e-2), linalg.norm(trace - np.trace(J))
