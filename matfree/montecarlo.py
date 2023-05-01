"""Monte-Carlo estimation."""

from matfree.backend import containers, control_flow, func, np, prng
from matfree.backend.typing import Any, Array, Callable


def estimate(
    fun: Callable,
    /,
    *,
    key: Array,
    sample_fun: Callable,
    num_batches: int = 1,
    num_samples_per_batch: int = 10_000,
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
    """
    fun_mc = _montecarlo(fun, sample_fun=sample_fun)
    fun_single_batch = _mean_vmap(fun_mc, num_samples_per_batch)
    fun_batched = _mean_loop(fun_single_batch, num_batches)
    return fun_batched(key)


def _montecarlo(f, /, *, sample_fun):
    """Turn a function into a Monte-Carlo problem.

    More specifically, f(x) becomes g(key) = f(h(key)),
    using a sample function h: key -> x.
    This can then be evaluated and averaged in batches, loops, and compositions thereof.
    """
    # todo: what about randomised QMC? How do we best implement this?

    def f_mc(key, /):
        sample = sample_fun(key)
        return f(sample)

    return f_mc


def _mean_vmap(f, num, /):
    """Compute a batch-mean via jax.vmap."""

    def f_mean(key, /):
        subkeys = prng.split(key, num)
        fx_values = func.vmap(f)(subkeys)
        return np.nanmean(fx_values, axis=0)

    return f_mean


class _MState(containers.NamedTuple):
    mean: Any
    key: Any


def _mean_loop(f, num, /):
    """Compute an on-the-fly mean via a for-loop."""

    def f_mean(key, /):
        # Initialise
        fx_shape = func.eval_shape(f, key).shape
        mean = np.zeros(fx_shape, dtype=float)
        mstate = _MState(mean=mean, key=key)

        # Run for-loop
        increment = func.partial(_mean_increment, fun=f)
        mstate = control_flow.fori_loop(0, num, body_fun=increment, init_val=mstate)
        mean, _key = mstate  # todo: why not return key?

        # Return results
        return mean

    return f_mean


def _mean_increment(i, mstate: _MState, fun) -> _MState:
    """Increment the current mean-estimate."""
    # Read and split key
    mean, key = mstate
    _, subkey = prng.split(key)

    # Evaluate function
    fx_values = fun(subkey)

    # Update mean estimate
    mean_new = np.sum(np.asarray([mean * i, fx_values]), axis=0) / (i + 1)
    return _MState(mean=mean_new, key=subkey)


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
