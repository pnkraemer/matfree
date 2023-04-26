"""A million ways of computing arithmetic means."""

from matfree.backend import containers, flow, func, np, prng


def montecarlo(f, /, *, sample_fun):
    """Turn a function into a Monte-Carlo problem.

    More specifically, f(x) becomes g(key) = f(h(key)),
    using a sample function h: key -> x.
    This can then be evaluated and averaged in batches, loops, and compositions thereof.
    """
    # todo: what about randomised QMC? How do we best implement this?

    def f_mc(key, /):
        sample = sample_fun(key)
        return f(sample), 0

    return f_mc


def mean_vmap(f, num, /):
    """Compute a batch-mean via jax.vmap."""

    def f_mean(key, /):
        subkeys = prng.split(key, num)
        fx_values, how_many_previously = func.vmap(f)(subkeys)
        return _filter_nan_and_mean(fx_values, how_many_previously)

    return f_mean


def mean_map(f, num, /):
    """Compute a batch-mean via jax.lax.map."""

    def f_mean(key, /):
        subkeys = prng.split(key, num)
        fx_values, how_many_previously = flow.map(f, subkeys)
        return _filter_nan_and_mean(fx_values, how_many_previously)

    return f_mean


def _filter_nan_and_mean(fx_values, how_many_previously):
    is_nan = np.any(np.isnan(fx_values))
    how_many = np.sum(np.where(is_nan, np.maximum(1, how_many_previously), 0))
    mean = np.nanmean(fx_values, axis=0)
    return mean, how_many


_MState = containers.namedtuple("_MState", ["mean", "key", "num_nans"])


def mean_loop(f, num, /):
    """Compute an on-the-fly mean via a for-loop."""

    def f_mean(key, /):
        # Initialise
        fx_shape = func.eval_shape(f, key)[0].shape
        mean = np.zeros(fx_shape, dtype=float)
        mstate = _MState(mean=mean, key=key, num_nans=0)

        # Run for-loop
        increment = func.partial(_mean_increment, fun=f)
        mstate = flow.fori_loop(0, num, body_fun=increment, init_val=mstate)
        mean, _key, num_nans = mstate  # todo: why not return key?

        # Return results
        return mean, num_nans

    return f_mean


def _mean_increment(i, mstate: _MState, fun) -> _MState:
    """Increment the current mean-estimate."""
    # Read and split key
    mean, key, num_nans = mstate
    _, subkey = prng.split(key)

    # Evaluate function
    fx_values, how_many_previously = fun(subkey)

    # Update NaN-count
    how_many_max = np.maximum(1, np.sum(how_many_previously))
    fx_is_nan = np.any(np.isnan(fx_values))
    num_nans_new = num_nans + fx_is_nan * how_many_max

    # Update mean estimate
    mean_new = np.sum(np.asarray([mean * i, fx_values]), axis=0) / (i + 1)
    return _MState(mean=mean_new, key=subkey, num_nans=num_nans_new)
