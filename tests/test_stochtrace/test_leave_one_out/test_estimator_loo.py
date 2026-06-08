"""Test estimator_leave_one_out and estimator_leave_one_out_mean_and_sem."""

from matfree import stochtrace
from matfree.backend import np, prng


def test_estimator_leave_one_out_pytree_output():
    """Assert that estimator_leave_one_out handles pytree-returning integrands."""
    n, num_samples = 7, 5
    key = prng.prng_key(1)

    def pytree_integrand(_matvec, samples, *_params):
        # samples: array of shape (num_samples, n)
        return {"a": samples[:, :3], "b": samples[:, 3:]}

    sampler = stochtrace.sampler_normal(np.ones(n), num=num_samples)
    estimate = stochtrace.estimator_leave_one_out(pytree_integrand, sampler)
    result = estimate(lambda v: v, key)

    assert isinstance(result, dict)
    assert result["a"].shape == (3,)
    assert result["b"].shape == (4,)


def test_estimator_leave_one_out_mean_and_sem_pytree_output():
    """Assert that estimator_leave_one_out_mean_and_sem handles pytree-returning integrands."""
    n, num_samples = 7, 5
    key = prng.prng_key(1)

    def pytree_integrand(_matvec, samples, *_params):
        return {"a": samples[:, :3], "b": samples[:, 3:]}

    sampler = stochtrace.sampler_normal(np.ones(n), num=num_samples)
    estimate = stochtrace.estimator_leave_one_out_mean_and_sem(
        pytree_integrand, sampler
    )
    mean, sem = estimate(lambda v: v, key)

    assert isinstance(mean, dict)
    assert mean["a"].shape == (3,)
    assert mean["b"].shape == (4,)
    assert isinstance(sem, dict)
    assert sem["a"].shape == (3,)
    assert sem["b"].shape == (4,)
