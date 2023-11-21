"""Hutchinson-style trace and diagonal estimation."""


from matfree.backend import func, linalg, np, tree_util

# todo: allow a fun() that returns pytrees instead of arrays.
#  why? Because then we rival trace_and_variance as
#  trace_and_frobeniusnorm(): y=Ax; return (x@y, y@y)


def integrand_diagonal(matvec):
    def integrand(v):
        Qv = matvec(v)
        v_flat, unflatten = tree_util.ravel_pytree(v)
        Qv_flat, _unflatten = tree_util.ravel_pytree(Qv)
        return unflatten(v_flat * Qv_flat)

    return integrand


def integrand_trace(matvec):
    def integrand(v):
        Qv = matvec(v)
        v_flat, unflatten = tree_util.ravel_pytree(v)
        Qv_flat, _unflatten = tree_util.ravel_pytree(Qv)
        return linalg.vecdot(v_flat, Qv_flat)

    return integrand


def sampler_from_prng(sample_func, *args_like, num=10_000):
    x_flat, unflatten = tree_util.ravel_pytree(*args_like)

    def sample(key):
        samples = sample_func(key, shape=(num, *x_flat.shape), dtype=x_flat.dtype)
        return func.vmap(unflatten)(samples)

    return sample


def stats_mean_and_std():
    def stats(arr, /, axis):
        return {"mean": np.mean(arr, axis=axis), "std": np.std(arr, axis=axis)}

    return stats


def montecarlo(integrand_fun, /, sample_fun, stats_fun):
    def sample(key):
        samples = sample_fun(key)
        Qs = func.vmap(integrand_fun)(samples)
        return tree_util.tree_map(lambda s: stats_fun(s, axis=0), Qs)

    return sample
