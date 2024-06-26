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

normal = stochtrace.sampler_normal(x_like, num=num_samples)
integrand = stochtrace.integrand_trace()
integrand = stochtrace.integrand_wrap_moments(integrand, [1, 2])
estimator = stochtrace.estimator(integrand, sampler=normal)
first, second = estimator(matvec, jax.random.PRNGKey(1))


# For normally-distributed base-samples,
# we know that the variance is twice the squared Frobenius norm.


print(second - first**2)
print(2 * jnp.linalg.norm(A.T @ A + jnp.eye(6), ord="fro") ** 2)


# ## Uncertainty quantification
#
# Variance estimation leads to uncertainty quantification:
# The variance of the estimator is equal to the variance of the random variable
# divided by the number of samples.

variance = (second - first**2) / num_samples
print(variance)
