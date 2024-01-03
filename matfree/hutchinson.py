"""Hutchinson-style estimation."""

from matfree.backend import func, linalg, np, prng, tree_util


def hutchinson(integrand_fun, /, sample_fun):
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

    Returns
    -------
    integrand_fun
        A function that maps a random key to an estimate.
        This function can be jitted, vmapped, or looped over as the user desires.

    """

    def sample(key, *parameters):
        samples = sample_fun(key)
        Qs = func.vmap(lambda vec: integrand_fun(vec, *parameters))(samples)
        return tree_util.tree_map(lambda s: np.mean(s, axis=0), Qs)

    return sample


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

    def integrand(v, *parameters):
        Qv = matvec(v, *parameters)
        v_flat, unflatten = tree_util.ravel_pytree(v)
        Qv_flat, _unflatten = tree_util.ravel_pytree(Qv)
        return unflatten(v_flat * Qv_flat)

    return integrand


def integrand_trace(matvec, /):
    """Construct the integrand for estimating the trace."""

    def integrand(v, *parameters):
        Qv = matvec(v, *parameters)
        v_flat, unflatten = tree_util.ravel_pytree(v)
        Qv_flat, _unflatten = tree_util.ravel_pytree(Qv)
        return linalg.vecdot(v_flat, Qv_flat)

    return integrand


def integrand_trace_and_diagonal(matvec, /):
    """Construct the integrand for estimating the trace and diagonal jointly."""

    def integrand(v, *parameters):
        Qv = matvec(v, *parameters)
        v_flat, unflatten = tree_util.ravel_pytree(v)
        Qv_flat, _unflatten = tree_util.ravel_pytree(Qv)
        trace_form = linalg.vecdot(v_flat, Qv_flat)
        diagonal_form = unflatten(v_flat * Qv_flat)
        return {"trace": trace_form, "diagonal": diagonal_form}

    return integrand


def integrand_frobeniusnorm_squared(matvec, /):
    """Construct the integrand for estimating the squared Frobenius norm."""

    def integrand(vec, *parameters):
        x = matvec(vec, *parameters)
        v_flat, unflatten = tree_util.ravel_pytree(x)
        return linalg.vecdot(v_flat, v_flat)

    return integrand


def integrand_wrap_moments(integrand_fun, /, moments):
    """Wrap an integrand into another integrand that computes moments.

    Parameters
    ----------
    integrand_fun
        Any integrand function compatible with Hutchinson-style estimation.
    moments
        Any Pytree (tuples, lists, dictionaries) whose leafs that are
        valid inputs to ``lambda m: x**m`` for an array ``x``,
        usually, with data-type ``float``
        (but that depends on the wrapped integrand).
        For example, ``moments=4``, ``moments=(1,2)``,
        or ``moments={"a": 1, "b": 2}``.

    Returns
    -------
    integrand_fun
        An integrand function compatible with Hutchinson-style estimation whose
        output has a PyTree-structure that mirrors the structure of the ``moments``
        argument.

    """

    def integrand_wrapped(vec, *parameters):
        Qs = integrand_fun(vec, *parameters)
        return tree_util.tree_map(moment_fun, Qs)

    def moment_fun(x, /):
        return tree_util.tree_map(lambda m: x**m, moments)

    return integrand_wrapped


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
