"""Compute log-determinants with stochastic Lanczos quadrature.

Log-determinant estimation can be implemented with stochastic Lanczos quadrature,
which can be loosely interpreted as an extension of Hutchinson's trace estimator.
"""

import jax
import jax.numpy as jnp

from matfree import decomp, funm, stochtrace

# Set up a matrix.

nhidden, nrows = 6, 5
A = jnp.reshape(jnp.arange(1.0, 1.0 + nhidden * nrows), (nhidden, nrows))
A /= nhidden * nrows


def matvec(x):
    """Compute a matrix-vector product."""
    return A.T @ (A @ x) + x


x_like = jnp.ones((nrows,), dtype=float)  # use to determine shapes of vectors


# Estimate log-determinants with stochastic Lanczos quadrature.

num_matvecs = 3
tridiag_sym = decomp.tridiag_sym(num_matvecs)
problem = funm.integrand_funm_sym_logdet(tridiag_sym)
sampler = stochtrace.sampler_normal(x_like, num=1_000)
estimator = stochtrace.estimator(problem, sampler=sampler)
logdet = estimator(matvec, jax.random.PRNGKey(1))
print(logdet)

# For comparison:

print(jnp.linalg.slogdet(A.T @ A + jnp.eye(nrows)))


# We can compute the log determinant of a matrix
# of the form $M = B^\top B$, purely based
# on arithmetic with $B$; no need to assemble $M$:

A = jnp.reshape(jnp.arange(1.0, 1.0 + nrows**2), (nrows, nrows))
A += jnp.eye(nrows)
A /= nrows**2


def matvec_half(x):
    """Compute a matrix-vector product."""
    return A @ x


num_matvecs = 3
bidiag = decomp.bidiag(num_matvecs)
problem = funm.integrand_funm_product_logdet(bidiag)
sampler = stochtrace.sampler_normal(x_like, num=1_000)
estimator = stochtrace.estimator(problem, sampler=sampler)
logdet = estimator(matvec_half, jax.random.PRNGKey(1))
print(logdet)

# Internally, Matfree uses JAX's vector-Jacobian products to
# turn the matrix-vector product into a vector-matrix product.
#
# For comparison:

print(jnp.linalg.slogdet(A.T @ A))
