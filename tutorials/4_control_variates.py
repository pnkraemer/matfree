"""Combine trace estimation with control variates.

Here is how to implement control variates.
"""

import jax
import jax.numpy as jnp

from matfree import stochtrace

# Create a matrix to whose trace/diagonal to approximate.


nrows, ncols = 4, 4
A = jnp.reshape(jnp.arange(1.0, 1.0 + nrows * ncols), (nrows, ncols))


# Set up the sampler.

x_like = jnp.ones((ncols,), dtype=float)
sample_fun = stochtrace.sampler_normal(x_like, num=10_000)


# First, compute the diagonal.


problem = stochtrace.integrand_diagonal()
estimate = stochtrace.estimator(problem, sample_fun)
diagonal_ctrl = estimate(lambda v: A @ v, jax.random.PRNGKey(1))


# Then, compute trace and diagonal jointly
# using the estimate of the diagonal as a control variate.


def matvec_ctrl(v):
    """Evaluate a matrix-vector product with a control variate."""
    return A @ v - diagonal_ctrl * v


problem = stochtrace.integrand_trace_and_diagonal()
estimate = stochtrace.estimator(problem, sample_fun)
trace_and_diagonal = estimate(matvec_ctrl, jax.random.PRNGKey(2))
trace, diagonal = trace_and_diagonal["trace"], trace_and_diagonal["diagonal"]


# We can, of course, compute it without a control variate as well.

problem = stochtrace.integrand_trace_and_diagonal()
estimate = stochtrace.estimator(problem, sample_fun)
trace_and_diagonal = estimate(lambda v: A @ v, jax.random.PRNGKey(2))
trace_ref, diagonal_ref = trace_and_diagonal["trace"], trace_and_diagonal["diagonal"]


# Compare the results.
# First, the diagonal.


print("True value:", jnp.diag(A))
print("Control variate:", diagonal_ctrl, jnp.linalg.norm(jnp.diag(A) - diagonal_ctrl))
print("Approximation:", diagonal_ref, jnp.linalg.norm(jnp.diag(A) - diagonal_ref))
print(
    "Control-variate approximation:",
    diagonal + diagonal_ctrl,
    jnp.linalg.norm(jnp.diag(A) - diagonal - diagonal_ctrl),
)


# Then, the trace.


print("True value:", jnp.trace(A))
print(
    "Control variate:",
    jnp.sum(diagonal_ctrl),
    jnp.abs(jnp.trace(A) - jnp.sum(diagonal_ctrl)),
)
print("Approximation:", trace_ref, jnp.abs(jnp.trace(A) - trace_ref))
print(
    "Control variate approximation:",
    trace + jnp.sum(diagonal_ctrl),
    jnp.abs(jnp.trace(A) - trace - jnp.sum(diagonal_ctrl)),
)
