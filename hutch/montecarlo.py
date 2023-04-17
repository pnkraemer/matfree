"""A million ways of computing arithmetic means."""

from hutch.backend import flow, np, prng, transform


def montecarlo(f, /, *, sample_fn):
    def f_mc(key, /):
        x = sample_fn(key)
        return f(x)

    return f_mc


def mean_vmap(f, num, /):
    def g(key, /):
        subkeys = prng.split(key, num)
        fx = transform.vmap(f)(subkeys)

        isnan = None
        return np.mean(fx, axis=0), isnan

    return g


def mean_map(f, num, /):
    def g(key, /):
        subkeys = prng.split(key, num)
        fx = flow.map(f, subkeys)

        isnan = None
        return np.mean(fx, axis=0), isnan

    return g


#
#
# from hutch.backend import np, flow, transform
#
#
# def montecarlo(f, sample_fn):
#     def g(key):
#         x = sample_fn(key)
#         return f(x), key
#
#     return g
#
#
# def mean_vmap(f, n, advance_fn=prng.split):
#     def g(key):
#         keys = advance_fn(key, n)
#         return np.mean(transform.vmap(f)(keys)), keys
#
#     return g
#
#
# def mean_fori_loop(f, n, advance_fn=prng.split):
#     def g(key):
#         def body_fn(i, val):
#             mean, k = val
#             _, k = advance_fn(k, 2)
#             fx, k = f(k)
#             mean_new = (val * (i + 1) + fx) / (i + 2)
#             return mean_new, k
#
#         return flow.fori_loop(
#             lower=1, upper=n + 1, body_fn=body_fn, init_val=(key, 0.0)
#         )
#
#     return g
#
#
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
