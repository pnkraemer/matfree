"""Hutchinson-style estimation."""

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
    """Construct the integrand for estimating the trace."""

    def integrand(v, /):
        Qv = matvec(v)
        v_flat, unflatten = tree_util.ravel_pytree(v)
        Qv_flat, _unflatten = tree_util.ravel_pytree(Qv)
        return linalg.vecdot(v_flat, Qv_flat)

    return integrand


def integrand_trace_and_diagonal(matvec, /):
    """Construct the integrand for estimating the trace and diagonal jointly."""

    def integrand(v, /):
        Qv = matvec(v)
        v_flat, unflatten = tree_util.ravel_pytree(v)
        Qv_flat, _unflatten = tree_util.ravel_pytree(Qv)
        trace_form = linalg.vecdot(v_flat, Qv_flat)
        diagonal_form = unflatten(v_flat * Qv_flat)
        return {"trace": trace_form, "diagonal": diagonal_form}

    return integrand


def integrand_frobeniusnorm_squared(matvec, /):
    """Construct the integrand for estimating the squared Frobenius norm."""

    def integrand(vec, /):
        x = matvec(vec)
        v_flat, unflatten = tree_util.ravel_pytree(x)
        return linalg.vecdot(v_flat, v_flat)

    return integrand


def integrand_trace_moments(matvec, moments, /):
    """Construct the integrand for estimating (higher) moments of the trace."""

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
    """Construct a function that samples from a standard-normal distribution."""
    return _sampler_from_jax_random(prng.normal, *args_like, num=num)


def sampler_rademacher(*args_like, num):
    """Construct a function that samples from a Rademacher distribution."""
    return _sampler_from_jax_random(prng.rademacher, *args_like, num=num)


def _sampler_from_jax_random(sample_func, *args_like, num):
    x_flat, unflatten = tree_util.ravel_pytree(*args_like)

    def sample(key):
        samples = sample_func(key, shape=(num, *x_flat.shape), dtype=x_flat.dtype)
        return func.vmap(unflatten)(samples)

    return sample


def stats_mean_and_std():
    """Evaluate mean and standard-deviation of the samples."""

    def stats(arr, /, axis):
        return {"mean": np.mean(arr, axis=axis), "std": np.std(arr, axis=axis)}

    return stats


def hutchinson(integrand_fun, /, sample_fun, stats_fun=np.mean):
    """Construct Hutchinson's estimator.

    Parameters
    ----------
    integrand_fun
        The integrand function. For example, the return-value of
        [integrand_trace][matfree.hutchinson.integrand_trace].
        But any other integrand works, too.
    sample_fun
        The sample function. Usually, either
        [sampler_normal][matfree.hutchinson.sampler_normal] or
        [sampler_rademacher][matfree.hutchinson.sampler_rademacher].
    stats_fun
        The statistics to evaluate.
        Usually, this is jnp.mean. But any other
        statistical function which expects arguments like
        [one of these functions](https://data-apis.org/array-api/2022.12/API_specification/statistical_functions.html)
        and returns a pytree of arrays works;
        for example,
        [stats_mean_and_std][matfree.hutchinson.stats_mean_and_std].

    Returns
    -------
    A function that maps a random key to an estimate.
    This function can be jitted, vmapped, or looped over as the user desires.

    """

    def sample(key):
        samples = sample_fun(key)
        Qs = func.vmap(integrand_fun)(samples)
        return tree_util.tree_map(lambda s: stats_fun(s, axis=0), Qs)

    return sample
