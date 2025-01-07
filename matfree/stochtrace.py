"""Stochastic estimation of traces, diagonals, and more."""

from matfree.backend import func, linalg, np, prng, tree
from matfree.backend.typing import Callable


def estimator(integrand: Callable, /, sampler: Callable) -> Callable:
    """Construct a stochastic trace-/diagonal-estimator.

    Parameters
    ----------
    integrand
        The integrand function. For example, the return-value of
        [integrand_trace][matfree.stochtrace.integrand_trace].
        But any other integrand works, too.
    sampler
        The sample function. Usually, either
        [sampler_normal][matfree.stochtrace.sampler_normal] or
        [sampler_rademacher][matfree.stochtrace.sampler_rademacher].

    Returns
    -------
    estimate
        A function that maps a random key to an estimate.
        This function can be compiled, vectorised, differentiated,
        or looped over as the user desires.

    """

    def estimate(matvecs, key, *parameters):
        samples = sampler(key)
        Qs = func.vmap(lambda vec: integrand(matvecs, vec, *parameters))(samples)
        return tree.tree_map(lambda s: np.mean(s, axis=0), Qs)

    return estimate


def integrand_diagonal():
    """Construct the integrand for estimating the diagonal.

    When plugged into the Monte-Carlo estimator,
    the result will be an Array or PyTree of Arrays with the
    same tree-structure as
    ``
    matvec(*args_like)
    ``
    where ``*args_like`` is an argument of the sampler.
    """

    def integrand(matvec, v, *parameters):
        Qv = matvec(v, *parameters)
        v_flat, unflatten = tree.ravel_pytree(v)
        Qv_flat, _unflatten = tree.ravel_pytree(Qv)
        return unflatten(v_flat * Qv_flat)

    return integrand


def integrand_trace():
    """Construct the integrand for estimating the trace."""

    def integrand(matvec, v, *parameters):
        Qv = matvec(v, *parameters)
        v_flat, unflatten = tree.ravel_pytree(v)
        Qv_flat, _unflatten = tree.ravel_pytree(Qv)
        return linalg.inner(v_flat, Qv_flat)

    return integrand


def integrand_trace_and_diagonal():
    """Construct the integrand for estimating the trace and diagonal jointly."""

    def integrand(matvec, v, *parameters):
        Qv = matvec(v, *parameters)
        v_flat, unflatten = tree.ravel_pytree(v)
        Qv_flat, _unflatten = tree.ravel_pytree(Qv)
        trace_form = linalg.inner(v_flat, Qv_flat)
        diagonal_form = unflatten(v_flat * Qv_flat)
        return {"trace": trace_form, "diagonal": diagonal_form}

    return integrand


def integrand_frobeniusnorm_squared():
    """Construct the integrand for estimating the squared Frobenius norm."""

    def integrand(matvec, vec, *parameters):
        x = matvec(vec, *parameters)
        v_flat, unflatten = tree.ravel_pytree(x)
        return linalg.inner(v_flat, v_flat)

    return integrand


def integrand_wrap_moments(integrand, /, moments):
    """Wrap an integrand into another integrand that computes moments.

    Parameters
    ----------
    integrand
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
    integrand
        An integrand function compatible with Hutchinson-style estimation whose
        output has a PyTree-structure that mirrors the structure of the ``moments``
        argument.

    """

    def integrand_wrapped(vec, *parameters):
        Qs = integrand(vec, *parameters)
        return tree.tree_map(moment_fun, Qs)

    def moment_fun(x, /):
        return tree.tree_map(lambda m: x**m, moments)

    return integrand_wrapped


def sampler_normal(*args_like, num):
    """Construct a function that samples from a standard-normal distribution."""
    return _sampler_from_jax_random(prng.normal, *args_like, num=num)


def sampler_rademacher(*args_like, num):
    """Construct a function that samples from a Rademacher distribution."""
    return _sampler_from_jax_random(prng.rademacher, *args_like, num=num)


def _sampler_from_jax_random(sampler, *args_like, num):
    x_flat, unflatten = tree.ravel_pytree(*args_like)

    def sample(key):
        samples = sampler(key, shape=(num, *x_flat.shape), dtype=x_flat.dtype)
        return func.vmap(unflatten)(samples)

    return sample
