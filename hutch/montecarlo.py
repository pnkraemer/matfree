"""A million ways of computing arithmetic means."""

from hutch.backend import flow, np, prng, transform


def montecarlo(f, /, *, sample_fn):
    def f_mc(key, /):
        x = sample_fn(key)
        return f(x), {}

    return f_mc


def mean_vmap(f, num, /):
    def g(key, /):
        subkeys = prng.split(key, num)
        fx, info = transform.vmap(f)(subkeys)
        return np.mean(fx, axis=0), info

    return g


def mean_map(f, num, /):
    def g(key, /):
        subkeys = prng.split(key, num)
        fx, info = flow.map(f, subkeys)
        return np.mean(fx, axis=0), info

    return g


def mean_loop(f, num, /):
    def g(key, /):
        def body(i, mean_and_key):
            mean, k, _ = mean_and_key
            _, subk = prng.split(k)

            fx, info = f(subk)

            mean_new = (mean * i + fx) / (i + 1)
            return mean_new, subk, info

        init_val = (0.0, key, {})
        mean, _key, info = flow.fori_loop(
            1, upper=num + 1, body_fun=body, init_val=init_val
        )

        return mean, info

    return g
