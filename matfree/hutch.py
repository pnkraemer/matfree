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


def diagonal_with_control_variate(Av, control, /, **kwargs):
    """Estimate the diagonal of a matrix stochastically and with a control variate."""
    return diagonal(lambda v: Av(v) - control * v, **kwargs) + control


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
    diagonal_estimate: Any
    key: Any


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


# todo: implement this as multilevel(diagonal)(...)?
def diagonal_multilevel(
    Av,
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
    kwargs = {
        "num_batches": num_batches,
        "num_samples_per_batch": num_samples_per_batch,
        "sample_fun": sample_fun,
    }

    def update_fun(level, x):
        """Update the diagonal estimate."""
        diag, k = x

        _, subkey = prng.split(k, num=2)
        update = diagonal_with_control_variate(Av, diag, key=subkey, **kwargs)

        diag = _incr(diag, level, update)
        return _EstState(diag, subkey)

    # Todo: ask for this initial estimate?
    sample_dummy = func.eval_shape(sample_fun, key)
    diagonal_estimate = np.zeros(shape=sample_dummy.shape, dtype=sample_dummy.dtype)
    state = _EstState(diagonal_estimate=diagonal_estimate, key=key)

    state = control_flow.fori_loop(0, num_levels, body_fun=update_fun, init_val=state)
    (final, *_) = state
    return final


def _incr(old, count, incoming):
    return (old * count + incoming) / (count + 1)
