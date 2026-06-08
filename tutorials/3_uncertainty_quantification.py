"""Implement uncertainty quantification for trace estimation.

Computing higher moments of trace-estimates can easily
be turned into uncertainty quantification.
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

# ## Higher moments
#
# Trace estimation involves estimating expected values of random variables.
# Sometimes, second and higher moments of a random variable are interesting.

signs = stochtrace.sampler_signs(x_like, num=num_samples)
integrand = stochtrace.monte_carlo_trace()
integrand = stochtrace.monte_carlo_wrap_moments(integrand, [1, 2])
estimator = stochtrace.estimator_monte_carlo(integrand, sampler=signs)
first, second = estimator(matvec, jax.random.PRNGKey(1))


# For sign-distributed (Rademacher) base-samples,
# the variance equals twice the sum of squared off-diagonal entries.


M = A.T @ A + jnp.eye(6)
print(second - first**2)
print(2 * (jnp.linalg.norm(M, ord="fro") ** 2 - jnp.linalg.norm(jnp.diag(M)) ** 2))


# ## Uncertainty quantification
#
# Variance estimation leads to uncertainty quantification:
# The variance of the estimator is equal to the variance of the random variable
# divided by the number of samples.

variance = (second - first**2) / num_samples
print(variance)
