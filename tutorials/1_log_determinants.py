"""Compute log-determinants with stochastic Lanczos quadrature.

Log-determinant estimation can be implemented with stochastic Lanczos quadrature,
which can be loosely interpreted as an extension of Hutchinson's trace estimator.
"""

import jax
import jax.numpy as jnp

from matfree import hutchinson, slq

# Set up a matrix.

nhidden, nrows = 6, 5
A = jnp.reshape(jnp.arange(1.0, 1.0 + nhidden * nrows), (nhidden, nrows))
A /= nhidden * nrows


def matvec(x):
    """Compute a matrix-vector product."""
    return A.T @ (A @ x) + x


x_like = jnp.ones((nrows,), dtype=float)  # use to determine shapes of vectors


# Estimate log-determinants with stochastic Lanczos quadrature.

order = 3
problem = slq.integrand_logdet_spd(order, matvec)
sampler = hutchinson.sampler_normal(x_like, num=1_000)
estimator = hutchinson.hutchinson(problem, sample_fun=sampler)
logdet = estimator(jax.random.PRNGKey(1))
print(logdet)

# For comparison:

print(jnp.linalg.slogdet(A.T @ A + jnp.eye(nrows)))


# We can compute the log determinant of a matrix
# of the form $M = B^\top B$, purely based
# on arithmetic with $B$; no need to assemble $M$:

A = jnp.reshape(jnp.arange(1.0, 1.0 + nrows**2), (nrows, nrows))
A += jnp.eye(nrows)
A /= nrows**2


def matvec_right(x):
    """Compute a matrix-vector product."""
    return A @ x


def vecmat_left(x):
    """Compute a vector-matrix product."""
    return x @ A


order = 3
problem = slq.integrand_logdet_product(order, matvec_right, vecmat_left)
sampler = hutchinson.sampler_normal(x_like, num=1_000)
estimator = hutchinson.hutchinson(problem, sample_fun=sampler)
logdet = estimator(jax.random.PRNGKey(1))
print(logdet)

# For comparison:

print(jnp.linalg.slogdet(A.T @ A))
