"""A million ways of computing arithmetic means."""

from hutch.backend import flow, np, prng, transform


def montecarlo(f, /, *, sample_fn):
    def f_mc(key, /):
        x = sample_fn(key)
        return f(x), 0

    return f_mc


def mean_vmap(f, num, /):
    def g(key, /):
        subkeys = prng.split(key, num)
        fx, how_many_previously = transform.vmap(f)(subkeys)
        return _filter_nan_and_mean(fx, how_many_previously)

    return g


def mean_map(f, num, /):
    def g(key, /):
        subkeys = prng.split(key, num)
        fx, how_many_previously = flow.map(f, subkeys)
        return _filter_nan_and_mean(fx, how_many_previously)

    return g


def _filter_nan_and_mean(fx, how_many_previously):
    is_nan = np.isnan(fx)
    how_many = np.sum(np.where(is_nan, np.maximum(1, np.sum(how_many_previously)), 0))
    mean = np.nanmean(fx, axis=0)
    return mean, how_many


def mean_loop(f, num, /):
    def g(key, /):
        def body(i, mean_and_key):
            mean, k, n = mean_and_key
            _, subk = prng.split(k)

            fx, how_many_previously = f(subk)
            num_nans_new = n + np.any(np.isnan(fx)) * np.maximum(
                1, np.sum(how_many_previously)
            )

            mean_new = np.sum(np.asarray([mean * i, fx]), axis=0) / (i + 1)
            print(mean_new, subk, num_nans_new)
            return mean_new, subk, num_nans_new

        fx_shape = transform.eval_shape(f, key)[0].shape
        init_val = (np.zeros(fx_shape, dtype=float), key, 0)
        mean, _key, num_nans = flow.fori_loop(
            1, upper=num + 1, body_fun=body, init_val=init_val
        )

        return mean, num_nans

    return g
