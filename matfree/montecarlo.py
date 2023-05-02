"""Monte-Carlo estimation."""

from matfree.backend import containers, control_flow, func, np, prng
from matfree.backend.typing import Array, Callable


def estimate(
    fun: Callable,
    /,
    *,
    key: Array,
    sample_fun: Callable,
    num_batches: int = 1,
    num_samples_per_batch: int = 10_000,
    target_fun=np.nanmean,
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
    target_fun:
        The target function to compute. Usually, this is np.mean. But any other
        statistical function with a signature like
        [those](https://data-apis.org/array-api/2022.12/API_specification/statistical_functions.html)
        would work. The default is np.nanmean.
    """
    fun_mc = _montecarlo(fun, sample_fun=sample_fun)
    fun_single_batch = _stats_via_vmap(fun_mc, num_samples_per_batch, target_fun)
    fun_batched = _stats_via_map(fun_single_batch, num_batches, target_fun)
    return fun_batched(key)


def _montecarlo(f, /, *, sample_fun):
    """Turn a function into a Monte-Carlo problem.

    More specifically, f(x) becomes g(key) = f(h(key)),
    using a sample function h: key -> x.
    This can then be evaluated and averaged in batches, loops, and compositions thereof.
    """
    # todo: what about randomised QMC? How do we best implement this?
    # todo: vmapping over keys feels less efficient than computing N samples at once.

    def f_mc(key, /):
        sample = sample_fun(key)
        return f(sample)

    return f_mc


def _stats_via_vmap(f, num, /, target_fun):
    """Compute summary statistics via jax.vmap."""

    def f_mean(key, /):
        subkeys = prng.split(key, num)
        fx_values = func.vmap(f)(subkeys)
        return target_fun(fx_values, axis=0)

    return f_mean


def _stats_via_map(f, num, /, target_fun):
    """Compute summary statistics via jax.lax.map."""

    def f_mean(key, /):
        subkeys = prng.split(key, num)
        fx_values = control_flow.array_map(f, subkeys)
        return target_fun(fx_values, axis=0)

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
