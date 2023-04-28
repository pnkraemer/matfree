"""Hutchinson-style trace and diagonal estimation.

See e.g.

http://www.nowozin.net/sebastian/blog/thoughts-on-trace-estimation-in-deep-learning.html

"""


from matfree import montecarlo
from matfree.backend import containers, control_flow, func, np
from matfree.backend.typing import Any


def trace(matvec_fun, /, **kwargs):
    """Estimate the trace of a matrix stochastically."""

    def quadform(vec):
        return np.vecdot(vec, matvec_fun(vec))

    return stochastic_estimate(quadform, **kwargs)


def frobeniusnorm_squared(matvec_fun, /, **kwargs):
    """Estimate the squared Frobenius norm of a matrix stochastically."""

    def quadform(vec):
        Av = matvec_fun(vec)
        return np.vecdot(Av, Av)

    return stochastic_estimate(quadform, **kwargs)


def diagonal(matvec_fun, /, **kwargs):
    """Estimate the diagonal of a matrix stochastically."""

    def quadform(vec):
        return vec * matvec_fun(vec)

    return stochastic_estimate(quadform, **kwargs)


def stochastic_estimate(
    fun, /, *, key, sample_fun, num_batches=1, num_samples_per_batch=10_000
):
    """Hutchinson-style stochastic estimation."""
    fun_mc = montecarlo.montecarlo(fun, sample_fun=sample_fun)
    fun_single_batch = montecarlo.mean_vmap(fun_mc, num_samples_per_batch)
    fun_batched = montecarlo.mean_loop(fun_single_batch, num_batches)
    mean, _ = fun_batched(key)
    return mean


class _EstState(containers.NamedTuple):
    traceest: float
    diagest: Any
    num: int


def trace_and_diagonal(matvec_fun, /, *, sample_fun, keys):
    """Jointly estimate the trace and the diagonal stochastically.

    The advantage of computing both quantities simultaneously is
    that the diagonal estimate
    may serve as a control variate for the trace estimate,
    thus reducing the variance of the estimator
    (and thereby accelerating convergence.)

    See:
    Adams et al., Estimating the Spectral Density of Large Implicit Matrices, 2018.
    """
    sample_dummy = func.eval_shape(sample_fun, keys[0])
    zeros = np.zeros(shape=sample_dummy.shape, dtype=sample_dummy.dtype)
    state = _EstState(traceest=0.0, diagest=zeros, num=0)

    body_fun = func.partial(_update, sample_fun=sample_fun, matvec_fun=matvec_fun)
    state, _ = control_flow.scan(body_fun, init=state, xs=keys)

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
