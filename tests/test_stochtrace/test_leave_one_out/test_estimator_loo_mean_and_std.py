"""Test estimator_leave_one_out_mean_and_sem."""

import jax.numpy as jnp

from matfree import stochtrace
from matfree.backend import np, prng, testing


@testing.parametrize("seed", [1, 2, 3])
def test_mean_matches_estimator_leave_one_out(seed):
    """Assert that the mean equals the result of estimator_leave_one_out."""
    A = jnp.reshape(jnp.arange(16.0), (4, 4)) / 16 + jnp.eye(4)

    def matvec(x):
        return A @ x

    x_like = jnp.ones((4,))
    key = prng.prng_key(seed)
    sampler = stochtrace.sampler_normal(x_like, num=4)
    integrand = stochtrace.leave_one_out_xtrace()

    estimate_plain = stochtrace.estimator_leave_one_out(integrand, sampler=sampler)
    mean_only = estimate_plain(matvec, key)
    estimate = stochtrace.estimator_leave_one_out_mean_and_sem(
        integrand, sampler=sampler
    )
    mean, _sem = estimate(matvec, key)
    assert np.allclose(mean, mean_only)


@testing.parametrize("seed", [1, 2, 3])
def test_sem_error_is_nonnegative(seed):
    """Assert that sem is non-negative."""
    A = jnp.reshape(jnp.arange(25.0), (5, 5)) / 25 + jnp.eye(5)

    def matvec(x):
        return A @ x

    x_like = jnp.ones((5,))
    key = prng.prng_key(seed)
    sampler = stochtrace.sampler_normal(x_like, num=5)
    integrand = stochtrace.leave_one_out_xtrace()

    estimate = stochtrace.estimator_leave_one_out_mean_and_sem(
        integrand, sampler=sampler
    )
    _mean, sem = estimate(matvec, key)
    assert sem >= 0.0
