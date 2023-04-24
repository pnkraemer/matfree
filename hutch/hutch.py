"""Hutchinson-style trace and diagonal estimation.

See e.g.

http://www.nowozin.net/sebastian/blog/thoughts-on-trace-estimation-in-deep-learning.html

"""

import functools

from hutch import montecarlo
from hutch.backend import containers, flow, np, prng, transform


@functools.partial(
    transform.jit,
    static_argnames=(
        "matvec_fun",
        "tangents_shape",
        "tangents_dtype",
        "num_batches",
        "num_samples_per_batch",
        "sample_fun",
    ),
)
def trace(matvec_fun, **kwargs):
    """Estimate the trace of a matrix stochastically."""

    def quadform(vec):
        return np.dot(vec, matvec_fun(vec))

    return _stochastic_estimate(quadform, **kwargs)


def diagonal(matvec_fun, **kwargs):
    """Estimate the diagonal of a matrix stochastically."""

    def quadform(vec):
        return vec * matvec_fun(vec)

    return _stochastic_estimate(quadform, **kwargs)


def _stochastic_estimate(
    quadform,
    /,
    *,
    tangents_shape,
    tangents_dtype,
    key,
    num_batches=1,
    num_samples_per_batch=10_000,
    sample_fun=prng.rademacher,
):
    """Hutchinson-style stochastic estimation."""

    def sample(k):
        return sample_fun(k, shape=tangents_shape, dtype=tangents_dtype)

    quadform_mc = montecarlo.montecarlo(quadform, sample_fun=sample)
    quadform_single_batch = montecarlo.mean_vmap(quadform_mc, num_samples_per_batch)
    quadform_batch = montecarlo.mean_map(quadform_single_batch, num_batches)
    mean, _ = quadform_batch(key)
    return mean


_EstState = containers.namedtuple("EstState", ["traceest", "diagest", "num"])


@functools.partial(
    transform.jit,
    static_argnames=(
        "matvec_fun",
        "tangents_shape",
        "tangents_dtype",
        "sample_fun",
    ),
)
def trace_and_diagonal(
    matvec_fun,
    *,
    tangents_shape,
    tangents_dtype,
    keys,
    sample_fun=prng.normal,
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
    state = _EstState(traceest=0.0, diagest=zeros, num=0)

    def sample(key):
        return sample_fun(key, shape=tangents_shape, dtype=tangents_dtype)

    body_fun = transform.partial(_update, sample_fun=sample, matvec_fun=matvec_fun)
    state, _ = flow.scan(body_fun, init=state, xs=keys)

    (trace_final, diag_final, _) = state
    return trace_final, diag_final


# todo: use fori_loop.
def _update(carry, key, *, sample_fun, matvec_fun):
    traceest, diagest, num = carry

    vec_sample = sample_fun(key)
    quadform_value = vec_sample * (matvec_fun(vec_sample) - diagest * vec_sample)

    # todo: allow batch-mode.
    traceest = _incr(traceest, num, np.sum(quadform_value) + sum(diagest))
    diagest = _incr(diagest, num, quadform_value + diagest)

    return _EstState(traceest=traceest, diagest=diagest, num=num + 1), ()


def _incr(old, count, incoming):
    return (old * count + incoming) / (count + 1)
