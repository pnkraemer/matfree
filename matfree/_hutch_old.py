"""Temporary."""

from matfree.backend import containers, control_flow, func, linalg, np, prng
from matfree.backend.typing import Any, Array, Callable, Sequence


def mc_estimate(
    fun: Callable,
    /,
    *,
    key: Array,
    sample_fun: Callable,
    num_batches: int = 1,
    num_samples_per_batch: int = 10_000,
    statistic_batch: Callable = np.mean,
    statistic_combine: Callable = np.mean,
) -> Array:
    """Monte-Carlo estimation: Compute the expected value of a function.

    Parameters
    ----------
    fun:
        Function whose expected value shall be estimated.
    key:
        Pseudo-random number generator key.
    sample_fun:
        Sampling function.
        For trace-estimation, use
        either [normal(...)][matfree.hutchinson.sampler_normal]
        or [rademacher(...)][matfree.hutchinson.sampler_normal].
    num_batches:
        Number of batches when computing arithmetic means.
    num_samples_per_batch:
        Number of samples per batch.
    statistic_batch:
        The summary statistic to compute on batch-level.
        Usually, this is np.mean. But any other
        statistical function with a signature like
        [one of these functions](https://data-apis.org/array-api/2022.12/API_specification/statistical_functions.html)
        would work.
    statistic_combine:
        The summary statistic to combine batch-results.
        Usually, this is np.mean. But any other
        statistical function with a signature like
        [one of these functions](https://data-apis.org/array-api/2022.12/API_specification/statistical_functions.html)
        would work.
    """
    [result] = mc_multiestimate(
        fun,
        key=key,
        sample_fun=sample_fun,
        num_batches=num_batches,
        num_samples_per_batch=num_samples_per_batch,
        statistics_batch=[statistic_batch],
        statistics_combine=[statistic_combine],
    )
    return result


def mc_multiestimate(
    fun: Callable,
    /,
    *,
    key: Array,
    sample_fun: Callable,
    num_batches: int = 1,
    num_samples_per_batch: int = 10_000,
    statistics_batch: Sequence[Callable] = (np.mean,),
    statistics_combine: Sequence[Callable] = (np.mean,),
) -> Array:
    """Compute a Monte-Carlo estimate with multiple summary statistics.

    The signature of this function is almost identical to
    [mc_estimate(...)][matfree.hutchinson.mc_estimate].
    The only difference is that statistics_batch and statistics_combine are iterables
    of summary statistics (of equal lengths).

    The result of this function is an iterable of matching length.

    Parameters
    ----------
    fun:
        Same as in [mc_estimate(...)][matfree.hutchinson.mc_estimate].
    key:
        Same as in [mc_estimate(...)][matfree.hutchinson.mc_estimate].
    sample_fun:
        Same as in [mc_estimate(...)][matfree.hutchinson.mc_estimate].
    num_batches:
        Same as in [mc_estimate(...)][matfree.hutchinson.mc_estimate].
    num_samples_per_batch:
        Same as in [mc_estimate(...)][matfree.hutchinson.mc_estimate].
    statistics_batch:
        List or tuple of summary statistics to compute on batch-level.
    statistics_combine:
        List or tuple of summary statistics to combine batches.

    """
    assert len(statistics_batch) == len(statistics_combine)
    fun_mc = _montecarlo(fun, sample_fun=sample_fun, num_stats=len(statistics_batch))
    fun_single_batch = _stats_via_vmap(fun_mc, num_samples_per_batch, statistics_batch)
    fun_batched = _stats_via_map(fun_single_batch, num_batches, statistics_combine)
    return fun_batched(key)


def _montecarlo(f, /, sample_fun, num_stats):
    """Turn a function into a Monte-Carlo problem.

    More specifically, f(x) becomes g(key) = f(h(key)),
    using a sample function h: key -> x.
    This can then be evaluated and averaged in batches, loops, and compositions thereof.
    """
    # todo: what about randomised QMC? How do we best implement this?

    def f_mc(key, /):
        sample = sample_fun(key)
        return [f(sample)] * num_stats

    return f_mc


def _stats_via_vmap(f, num, /, statistics: Sequence[Callable]):
    """Compute summary statistics via jax.vmap."""

    def f_mean(key, /):
        subkeys = prng.split(key, num)
        fx_values = func.vmap(f)(subkeys)
        return [stat(fx, axis=0) for stat, fx in zip(statistics, fx_values)]

    return f_mean


def _stats_via_map(f, num, /, statistics: Sequence[Callable]):
    """Compute summary statistics via jax.lax.map."""

    def f_mean(key, /):
        subkeys = prng.split(key, num)
        fx_values = control_flow.array_map(f, subkeys)
        return [stat(fx, axis=0) for stat, fx in zip(statistics, fx_values)]

    return f_mean


def sampler_normal(*, shape, dtype=float):
    """Construct a function that samples from a standard normal distribution."""

    def fun(key):
        return prng.normal(key, shape=shape, dtype=dtype)

    return fun


def sampler_rademacher(*, shape, dtype=float):
    """Construct a function that samples from a Rademacher distribution."""

    def fun(key):
        return prng.rademacher(key, shape=shape, dtype=dtype)

    return fun


def trace(Av: Callable, /, **kwargs) -> Array:
    """Estimate the trace of a matrix stochastically.

    Parameters
    ----------
    Av:
        Matrix-vector product function.
    **kwargs:
        Keyword-arguments to be passed to
        [mc_estimate()][matfree.hutchinson.mc_estimate].
    """

    def quadform(vec):
        return linalg.vecdot(vec, Av(vec))

    return mc_estimate(quadform, **kwargs)


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
        [mc_multiestimate(...)][matfree.hutchinson.mc_multiestimate].
    """

    def quadform(vec):
        return linalg.vecdot(vec, Av(vec))

    def moment(x, axis, *, power):
        return np.mean(x**power, axis=axis)

    statistics_batch = [func.partial(moment, power=m) for m in moments]
    statistics_combine = [np.mean] * len(moments)
    return mc_multiestimate(
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
        [mc_estimate()][matfree.hutchinson.mc_estimate].

    """

    def quadform(vec):
        x = Av(vec)
        return linalg.vecdot(x, x)

    return mc_estimate(quadform, **kwargs)


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
        [mc_estimate()][matfree.hutchinson.mc_estimate].

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
        [mc_estimate()][matfree.hutchinson.mc_estimate].

    """

    def quadform(vec):
        return vec * Av(vec)

    return mc_estimate(quadform, **kwargs)


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
        Usually, either [normal][matfree.hutchinson.sampler_normal]
        or [rademacher][matfree.hutchinson.sampler_normal].
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
        Usually, either [normal][matfree.hutchinson.sampler_normal]
        or [rademacher][matfree.hutchinson.sampler_normal].
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
