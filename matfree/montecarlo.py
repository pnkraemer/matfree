"""Monte-Carlo estimation."""

from matfree.backend import containers, control_flow, func, np, prng
from matfree.backend.typing import Array, Callable, Sequence

# todo: allow a fun() that returns pytrees instead of arrays.
#  why? Because then we rival trace_and_variance as
#  trace_and_frobeniusnorm(): y=Ax; return (x@y, y@y)


def estimate(
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
        either [montecarlo.normal(...)][matfree.montecarlo.normal]
        or [montecarlo.rademacher(...)][matfree.montecarlo.normal].
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
    [result] = multiestimate(
        fun,
        key=key,
        sample_fun=sample_fun,
        num_batches=num_batches,
        num_samples_per_batch=num_samples_per_batch,
        statistics_batch=[statistic_batch],
        statistics_combine=[statistic_combine],
    )
    return result


def multiestimate(
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
    [montecarlo.estimate(...)][matfree.montecarlo.estimate].
    The only difference is that statistics_batch and statistics_combine are iterables
    of summary statistics (of equal lengths).

    The result of this function is an iterable of matching length.

    Parameters
    ----------
    fun:
        Same as in [montecarlo.estimate(...)][matfree.montecarlo.estimate].
    key:
        Same as in [montecarlo.estimate(...)][matfree.montecarlo.estimate].
    sample_fun:
        Same as in [montecarlo.estimate(...)][matfree.montecarlo.estimate].
    num_batches:
        Same as in [montecarlo.estimate(...)][matfree.montecarlo.estimate].
    num_samples_per_batch:
        Same as in [montecarlo.estimate(...)][matfree.montecarlo.estimate].
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


def normal(*, shape, dtype=float):
    """Construct a function that samples from a standard normal distribution."""

    def fun(key):
        return prng.normal(key, shape=shape, dtype=dtype)

    return fun


def rademacher(*, shape, dtype=float):
    """Construct a function that samples from a Rademacher distribution."""

    def fun(key):
        return prng.rademacher(key, shape=shape, dtype=dtype)

    return fun


class _VDCState(containers.NamedTuple):
    n: int
    vdc: float
    denom: int


def van_der_corput(n, /, base=2):
    """Compute the 'n'th element of the Van-der-Corput sequence."""
    state = _VDCState(n, vdc=0, denom=1)

    vdc_modify = func.partial(_van_der_corput_modify, base=base)
    state = control_flow.while_loop(_van_der_corput_cond, vdc_modify, state)
    return state.vdc


def _van_der_corput_cond(state: _VDCState):
    return state.n > 0


def _van_der_corput_modify(state: _VDCState, *, base):
    denom = state.denom * base
    num, remainder = divmod(state.n, base)
    vdc = state.vdc + remainder / denom
    return _VDCState(num, vdc, denom)
