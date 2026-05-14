"""Smoke test for estimator_loo infrastructure."""
import jax
import jax.numpy as jnp
from matfree import stochtrace


def test_estimator_loo_calls_integrand_with_full_matrix():
    """estimator_loo should pass the full Omega matrix to the integrand, not individual rows."""
    n = 5
    key = jax.random.PRNGKey(0)
    sampler = stochtrace.sampler_normal(jnp.ones(n), num=10)

    shapes_seen = []

    def recording_integrand(matvec, Omega, *params):
        shapes_seen.append(Omega.shape)
        return jnp.zeros(())

    estimate = stochtrace.estimator_loo(recording_integrand, sampler)
    estimate(lambda v: v, key)

    assert len(shapes_seen) == 1, "integrand should be called once with the full matrix"
    assert shapes_seen[0] == (10, n), f"expected (10, {n}), got {shapes_seen[0]}"
