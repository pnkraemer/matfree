"""A million ways of computing arithmetic means."""

from hutch.backend import flow, np, prng, transform


def montecarlo(f, /, *, sample_fn):
    def f_mc(key, /):
        x = sample_fn(key)
        return f(x), None

    return f_mc


def mean_vmap(f, num, /):
    def g(key, /):
        subkeys = prng.split(key, num)
        fx, _isnan = transform.vmap(f)(subkeys)

        isnan = None
        return np.mean(fx, axis=0), isnan

    return g


def mean_map(f, num, /):
    def g(key, /):
        subkeys = prng.split(key, num)
        fx, _isnan = flow.map(f, subkeys)

        isnan = None
        return np.mean(fx, axis=0), isnan

    return g


def mean_loop(f, num, /):
    def g(key, /):
        def body(i, mean_and_key):
            mean, k = mean_and_key
            _, subk = prng.split(k)

            fx, _isnan = f(subk)

            mean_new = (mean * i + fx) / (i + 1)
            return mean_new, subk

        init_val = (0.0, key)
        mean, _key = flow.fori_loop(1, upper=num + 1, body_fun=body, init_val=init_val)

        isnan = None
        return mean, isnan

    return g
