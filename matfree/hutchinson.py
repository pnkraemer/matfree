"""Hutchinson-style trace and diagonal estimation."""

from matfree.backend import func, linalg, np, prng, tree_util

# todo: allow a fun() that returns pytrees instead of arrays.
#  why? Because then we rival trace_and_variance as
#  trace_and_frobeniusnorm(): y=Ax; return (x@y, y@y)


def integrand_diagonal(matvec, /):
    """Construct the integrand for estimating the diagonal.

    When plugged into the Monte-Carlo estimator,
    the result will be an Array or PyTree of Arrays with the
    same tree-structure as
    ``
    matvec(*args_like)
    ``
    where ``*args_like`` is an argument of the sampler.
    """

    def integrand(v, /):
        Qv = matvec(v)
        v_flat, unflatten = tree_util.ravel_pytree(v)
        Qv_flat, _unflatten = tree_util.ravel_pytree(Qv)
        return unflatten(v_flat * Qv_flat)

    return integrand


def integrand_trace(matvec, /):
    def integrand(v, /):
        Qv = matvec(v)
        v_flat, unflatten = tree_util.ravel_pytree(v)
        Qv_flat, _unflatten = tree_util.ravel_pytree(Qv)
        return linalg.vecdot(v_flat, Qv_flat)

    return integrand


def integrand_trace_and_diagonal(matvec, /):
    def integrand(v, /):
        Qv = matvec(v)
        v_flat, unflatten = tree_util.ravel_pytree(v)
        Qv_flat, _unflatten = tree_util.ravel_pytree(Qv)
        trace_form = linalg.vecdot(v_flat, Qv_flat)
        diagonal_form = unflatten(v_flat * Qv_flat)
        return {"trace": trace_form, "diagonal": diagonal_form}

    return integrand


def integrand_frobeniusnorm_squared(matvec, /):
    def integrand(vec, /):
        x = matvec(vec)
        v_flat, unflatten = tree_util.ravel_pytree(x)
        return linalg.vecdot(v_flat, v_flat)

    return integrand


def integrand_trace_moments(matvec, moments, /):
    def moment_fun(x):
        return tree_util.tree_map(lambda m: x**m, moments)

    def integrand(vec, /):
        x = matvec(vec)
        v_flat, unflatten = tree_util.ravel_pytree(vec)
        x_flat, _unflatten = tree_util.ravel_pytree(x)
        fx = linalg.vecdot(x_flat, v_flat)
        return moment_fun(fx)

    return integrand


def sampler_normal(*args_like, num):
    return _sampler_from_jax_random(prng.normal, *args_like, num=num)


def sampler_rademacher(*args_like, num):
    return _sampler_from_jax_random(prng.rademacher, *args_like, num=num)


def _sampler_from_jax_random(sample_func, *args_like, num):
    x_flat, unflatten = tree_util.ravel_pytree(*args_like)

    def sample(key):
        samples = sample_func(key, shape=(num, *x_flat.shape), dtype=x_flat.dtype)
        return func.vmap(unflatten)(samples)

    return sample


def stats_mean_and_std():
    def stats(arr, /, axis):
        return {"mean": np.mean(arr, axis=axis), "std": np.std(arr, axis=axis)}

    return stats


def hutchinson(integrand_fun, /, sample_fun, stats_fun=np.mean):
    def sample(key):
        samples = sample_fun(key)
        Qs = func.vmap(integrand_fun)(samples)
        return tree_util.tree_map(lambda s: stats_fun(s, axis=0), Qs)

    return sample
