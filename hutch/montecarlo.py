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


# r"""
# Example
# -------
#
# Make a Monte-Carlo problem:
#
# Q(x) becomes f(key)=Q(sample(key))
#
# And the rest becomes a matter of JAX' function transformations.
#
# # todo: control variates
# # todo: any way of implementing num_samples?
# # todo: quasi-monte carlo?
# # if we change "key" into "random state", and "split" into "advance" or whatever
# , random state can include a QMC generator state.
# # I think we might need to ask the user for a "split" implementation,
# # because otherwi
#
#
# # Trace estimation as a sampling problem
# >>> Q = trace(matvec_fn)
# >>> f = montecarlo(Q, sample_fn=prng.normal)
# >>> f = montecarlo(Q, sample_fn=prng.normal)
# >>> f = montecarlo(Q, sample_fn=prng.normal)
# >>> f = montecarlo(Q, sample_fn=prng.normal)
#
# # Random number generator
# >>> key = prng.PRNGKey(seed=4)
#
# # A single batch
# >>> _, key = prng.split(key)
# >>> fbar, _ = mean_via_vmap(f, 200)(key)
#
# >>> _, key = prng.split(key)
# >>> fbar, index = mean_via_map(f, 200, exclude=np.isnan)(key)
#
# # O(1) memory:
# >>> _, key = prng.split(key)
# >>> fbar, _ = mean_via_fori_loop(f, 200)(key)
#
# # Combination:
# >>> _, key = prng.split(key)
# >>> fbar, _ = mean_via_fori_loop(mean_via_vmap(f, 200), 1000)(key)
#
# >>> _, key = prng.split(key)
# >>> fbar, nan_indices = mean_via_map(mean_via_fori_loop(mean_via_vmap(f, 200), 1000), 4)(key)
# >>> nan_indices.range
# (4, 1000, 200)
#
# """
