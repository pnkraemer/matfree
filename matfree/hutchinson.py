"""Hutchinson-style trace and diagonal estimation."""


from matfree import montecarlo
from matfree.backend import containers, control_flow, func, linalg, np, prng
from matfree.backend.typing import Any, Array, Callable, Sequence


def trace(Av: Callable, /, **kwargs) -> Array:
    """Estimate the trace of a matrix stochastically.

    Parameters
    ----------
    Av:
        Matrix-vector product function.
    **kwargs:
        Keyword-arguments to be passed to
        [montecarlo.estimate()][matfree.montecarlo.estimate].
    """

    def quadform(vec):
        return linalg.vecdot(vec, Av(vec))

    return montecarlo.estimate(quadform, **kwargs)


def trace_moments(Av: Callable, /, moments: Sequence[int] = (1, 2), **kwargs) -> Array:
    """Estimate the trace of a matrix and the variance of the estimator.

    Parameters
    ----------
    Av:
        Matrix-vector product function.
    moments:
        Which moments to compute. For example, selection `moments=(1,2)` computes
        the first and second moment.
    **kwargs:
        Keyword-arguments to be passed to
        [montecarlo.multiestimate(...)][matfree.montecarlo.multiestimate].
    """

    def quadform(vec):
        return linalg.vecdot(vec, Av(vec))

    def moment(x, axis, *, power):
        return np.mean(x**power, axis=axis)

    statistics_batch = [func.partial(moment, power=m) for m in moments]
    statistics_combine = [np.mean] * len(moments)
    return montecarlo.multiestimate(
        quadform,
        statistics_batch=statistics_batch,
        statistics_combine=statistics_combine,
        **kwargs,
    )


def frobeniusnorm_squared(Av: Callable, /, **kwargs) -> Array:
    r"""Estimate the squared Frobenius norm of a matrix stochastically.

    The Frobenius norm of a matrix $A$ is defined as

    $$
    \|A\|_F^2 = \text{trace}(A^\top A)
    $$

    so computing squared Frobenius norms amounts to trace estimation.

    Parameters
    ----------
    Av:
        Matrix-vector product function.
    **kwargs:
        Keyword-arguments to be passed to
        [montecarlo.estimate()][matfree.montecarlo.estimate].

    """

    def quadform(vec):
        x = Av(vec)
        return linalg.vecdot(x, x)

    return montecarlo.estimate(quadform, **kwargs)


def diagonal_with_control_variate(Av: Callable, control: Array, /, **kwargs) -> Array:
    """Estimate the diagonal of a matrix stochastically and with a control variate.

    Parameters
    ----------
    Av:
        Matrix-vector product function.
    control:
        Control variate.
        This should be the best-possible estimate of the diagonal of the matrix.
    **kwargs:
        Keyword-arguments to be passed to
        [montecarlo.estimate()][matfree.montecarlo.estimate].

    """
    return diagonal(lambda v: Av(v) - control * v, **kwargs) + control


def diagonal(Av: Callable, /, **kwargs) -> Array:
    """Estimate the diagonal of a matrix stochastically.

    Parameters
    ----------
    Av:
        Matrix-vector product function.
    **kwargs:
        Keyword-arguments to be passed to
        [montecarlo.estimate()][matfree.montecarlo.estimate].

    """

    def quadform(vec):
        return vec * Av(vec)

    return montecarlo.estimate(quadform, **kwargs)


def trace_and_diagonal(Av: Callable, /, *, sample_fun: Callable, key: Array, **kwargs):
    """Jointly estimate the trace and the diagonal stochastically.

    The advantage of computing both quantities simultaneously is
    that the diagonal estimate
    may serve as a control variate for the trace estimate,
    thus reducing the variance of the estimator
    (and thereby accelerating convergence.)

    Parameters
    ----------
    Av:
        Matrix-vector product function.
    sample_fun:
        Sampling function.
        Usually, either [montecarlo.normal][matfree.montecarlo.normal]
        or [montecarlo.rademacher][matfree.montecarlo.normal].
    key:
        Pseudo-random number generator key.
    **kwargs:
        Keyword-arguments to be passed to
        [diagonal_multilevel()][matfree.hutchinson.diagonal_multilevel].


    See:
    Adams et al., Estimating the Spectral Density of Large Implicit Matrices, 2018.
    """
    fx_value = func.eval_shape(sample_fun, key)
    init = np.zeros(shape=fx_value.shape, dtype=fx_value.dtype)
    final = diagonal_multilevel(Av, init, sample_fun=sample_fun, key=key, **kwargs)
    return np.sum(final), final


class _EstState(containers.NamedTuple):
    diagonal_estimate: Any
    key: Any


def diagonal_multilevel(
    Av: Callable,
    init: Array,
    /,
    *,
    key: Array,
    sample_fun: Callable,
    num_levels: int,
    num_batches_per_level: int = 1,
    num_samples_per_batch: int = 1,
) -> Array:
    """Estimate the diagonal in a multilevel framework.

    The general idea is that a diagonal estimate serves as a control variate
    for the next step's diagonal estimate.


    Parameters
    ----------
    Av:
        Matrix-vector product function.
    init:
        Initial guess.
    key:
        Pseudo-random number generator key.
    sample_fun:
        Sampling function.
        Usually, either [montecarlo.normal][matfree.montecarlo.normal]
        or [montecarlo.rademacher][matfree.montecarlo.normal].
    num_levels:
        Number of levels.
    num_batches_per_level:
        Number of batches per level.
    num_samples_per_batch:
        Number of samples per batch (per level).

    """
    kwargs = {
        "sample_fun": sample_fun,
        "num_batches": num_batches_per_level,
        "num_samples_per_batch": num_samples_per_batch,
    }

    def update_fun(level: int, x: _EstState) -> _EstState:
        """Update the diagonal estimate."""
        diag, k = x

        _, subkey = prng.split(k, num=2)
        update = diagonal_with_control_variate(Av, diag, key=subkey, **kwargs)

        diag = _incr(diag, level, update)
        return _EstState(diag, subkey)

    state = _EstState(diagonal_estimate=init, key=key)
    state = control_flow.fori_loop(0, num_levels, body_fun=update_fun, init_val=state)
    (final, *_) = state
    return final


def _incr(old, count, incoming):
    return (old * count + incoming) / (count + 1)
