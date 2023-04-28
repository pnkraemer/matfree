"""Hutchinson-style trace and diagonal estimation.

See e.g.

http://www.nowozin.net/sebastian/blog/thoughts-on-trace-estimation-in-deep-learning.html

"""


from matfree import montecarlo
from matfree.backend import containers, control_flow, func, np, prng
from matfree.backend.typing import Any


def trace(matvec_fun, /, **kwargs):
    """Estimate the trace of a matrix stochastically."""

    def quadform(vec):
        return np.vecdot(vec, matvec_fun(vec))

    return stochastic_estimate(quadform, **kwargs)


def frobeniusnorm_squared(matvec_fun, /, **kwargs):
    r"""Estimate the squared Frobenius norm of a matrix stochastically.

    The Frobenius norm of a matrix $A$ is defined as

    $$
    \|A\|_F^2 = \text{trace}(A^\top A)
    $$

    so computing squared Frobenius norms amounts to trace estimation.
    """

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
    return fun_batched(key)


class _EstState(containers.NamedTuple):
    diagest: Any
    num: int


def trace_and_diagonal(*args, **kwargs):
    """Jointly estimate the trace and the diagonal stochastically.

    The advantage of computing both quantities simultaneously is
    that the diagonal estimate
    may serve as a control variate for the trace estimate,
    thus reducing the variance of the estimator
    (and thereby accelerating convergence.)

    See:
    Adams et al., Estimating the Spectral Density of Large Implicit Matrices, 2018.
    """
    diagonal_estimate = diagonal_multilevel(*args, **kwargs)
    return np.sum(diagonal_estimate), diagonal_estimate


def diagonal_multilevel(
    matvec_fun,
    /,
    *,
    num_levels,
    sample_fun,
    key,
    num_batches=1,
    num_samples_per_batch=1,
):
    """Estimate the diagonal in a multilevel framework.

    The general idea is that a diagonal estimate serves as a control variate
    for the next step's diagonal estimate.
    """
    keys = prng.split(key, num=num_levels)

    diagonal_fun = func.partial(
        diagonal,
        num_samples_per_batch=num_samples_per_batch,
        num_batches=num_batches,
        sample_fun=sample_fun,
    )

    # todo: use fori_loop and repeated split()s instead of a scan?
    def body(carry, k):
        diagest, num = carry

        def Av(v):
            return matvec_fun(v) - diagest * v

        update = diagonal_fun(Av, key=k)
        diagest = _incr(diagest, num, update + diagest)
        return _EstState(diagest=diagest, num=num + 1), ()

    sample_dummy = func.eval_shape(sample_fun, keys[0])
    zeros = np.zeros(shape=sample_dummy.shape, dtype=sample_dummy.dtype)
    state = _EstState(diagest=zeros, num=0)

    state, _ = control_flow.scan(body, init=state, xs=keys)

    (diag_final, _) = state
    return diag_final


def _incr(old, count, incoming):
    return (old * count + incoming) / (count + 1)
