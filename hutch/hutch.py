"""Hutchinson-style trace and diagonal estimation.

See e.g.

http://www.nowozin.net/sebastian/blog/thoughts-on-trace-estimation-in-deep-learning.html

"""

import functools

from hutch import montecarlo
from hutch.backend import flow, np, prng, transform


@functools.partial(
    transform.jit,
    static_argnames=(
        "matvec_fn",
        "tangents_shape",
        "tangents_dtype",
        "num_batches",
        "num_samples_per_batch",
        "generate_samples_fn",
    ),
)
def trace(matvec_fn, **kwargs):
    """Estimate the trace of a matrix stochastically."""

    def Q(x):
        return np.dot(x, matvec_fn(x))

    return _stochastic_estimate(Q, **kwargs)


def diagonal(matvec_fn, **kwargs):
    """Estimate the diagonal of a matrix stochastically."""

    def Q(x):
        return x * matvec_fn(x)

    return _stochastic_estimate(Q, **kwargs)


def _stochastic_estimate(
    Q,
    /,
    *,
    tangents_shape,
    tangents_dtype,
    key,
    num_batches=1,
    num_samples_per_batch=10_000,
    generate_samples_fn=prng.rademacher,
):
    """Hutchinson-style stochastic estimation."""

    def sample_fn(k):
        return generate_samples_fn(k, shape=tangents_shape, dtype=tangents_dtype)

    Q_mc = montecarlo.montecarlo(Q, sample_fn=sample_fn)
    Q_single_batch = montecarlo.mean_vmap(Q_mc, num_samples_per_batch)
    Q_batch = montecarlo.mean_map(Q_single_batch, num_batches)
    mean, _ = Q_batch(key)
    return mean


@functools.partial(
    transform.jit,
    static_argnames=(
        "matvec_fn",
        "tangents_shape",
        "tangents_dtype",
        "generate_samples_fn",
    ),
)
def trace_and_diagonal(
    matvec_fn,
    *,
    tangents_shape,
    tangents_dtype,
    keys,
    generate_samples_fn=prng.normal,
):
    """Jointly estimate the trace and the diagonal stochastically.

    The advantage of computing both quantities simultaneously is
    that the diagonal estimate
    may serve as a control variate for the trace estimate,
    thus reducing the variance of the estimator
    (and thereby accelerating convergence.)

    See:
    Adams et al., Estimating the Spectral Density of Large Implicit Matrices, 2018.
    """
    zeros = np.zeros(shape=tangents_shape, dtype=tangents_dtype)
    init = (0.0, zeros, 0)  # trace, diag, n

    def body_fn(carry, key):
        trace, diag, n = carry

        z = generate_samples_fn(key, shape=tangents_shape, dtype=tangents_dtype)
        y = z * (matvec_fn(z) - diag * z)

        # todo: allow batch-mode.
        trace_new = _increment(trace, n, np.sum(y) + sum(diag))
        diag_new = _increment(diag, n, y + diag)

        return (trace_new, diag_new, n + 1), ()

    (trace_final, diag_final, _), _ = flow.scan(body_fn, init=init, xs=keys)
    return trace_final, diag_final


def _increment(old, count, incoming):
    return (old * count + incoming) / (count + 1)
