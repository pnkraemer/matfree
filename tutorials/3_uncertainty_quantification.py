"""Implement uncertainty quantification for trace estimation.

The standard error returned by the mean-and-std estimator
directly gives an error bar on the estimate without any
manual bookkeeping of sample counts.
"""

import jax
import jax.numpy as jnp

from matfree import stochtrace

A = jnp.reshape(jnp.arange(36.0), (6, 6)) / 36


def matvec(x):
    """Evaluate a matrix-vector product."""
    return A.T @ (A @ x) + x


x_like = jnp.ones((6,))
num_samples = 10_000

# ## Uncertainty quantification
#
# Use `estimator_monte_carlo_mean_and_sem` to get both the estimate and its
# standard error in one call. The standard error equals
# std(samples) / sqrt(num_samples) and serves directly as an error bar —
# no need to track the sample count separately.

signs = stochtrace.sampler_signs(x_like, num=num_samples)
integrand = stochtrace.monte_carlo_trace()
estimator = stochtrace.estimator_monte_carlo_mean_and_sem(integrand, sampler=signs)
mean, sem = estimator(matvec, jax.random.PRNGKey(1))

print(mean)
print(sem)

# For sign-distributed (Rademacher) base-samples,
# the variance equals twice the sum of squared off-diagonal entries.

M = A.T @ A + jnp.eye(6)
variance_truth = jnp.linalg.norm(M, ord="fro") ** 2 - jnp.linalg.norm(jnp.diag(M)) ** 2
print(sem**2 * num_samples)  # should be close to 2 * variance_truth
print(2 * variance_truth)
