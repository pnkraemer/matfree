"""Carry out stochastic trace estimation with minimal memory.

Matfree's implementation of stochastic trace estimation
via Hutchinson's method defaults to computing all
Monte-Carlo samples at once, because this is the fastest
implementation as long as all samples fit into memory.


Some matrix-vector products, however, are so large that
we can only store a single sample in memory at once.
Here is how we can wrap calls around the trace estimators
in such a scenario to save memory.
"""

import functools

import jax
import jax.numpy as jnp

from matfree import stochtrace

# ## Stochastic trace estimation
#
# The conventional setup for estimating the trace of a large matrix
# would look like this.

nrows = 100  # but imagine nrows=100,000,000,000 instead
nsamples = 1_000


def large_matvec(v):
    """Evaluate a (dummy for a) large matrix-vector product."""
    return 1.2345 * v


integrand = stochtrace.integrand_trace()
x0 = jnp.ones((nrows,))
sampler = stochtrace.sampler_rademacher(x0, num=nsamples)
estimate = stochtrace.estimator(integrand, sampler)

key = jax.random.PRNGKey(1)
trace = estimate(large_matvec, key)
print(trace)


# The above code requires nrows $\times$ nsamples storage, which
# is prohibitive for extremely large matrices.
# Instead, we can loop around estimate() to do the following:
# The below code requires nrows $\times$ 1 storage:

sampler = stochtrace.sampler_rademacher(x0, num=1)
estimate = stochtrace.estimator(integrand, sampler)
estimate = functools.partial(estimate, large_matvec)

key = jax.random.PRNGKey(2)
keys = jax.random.split(key, num=nsamples)
traces = jax.lax.map(estimate, keys)
trace = jnp.mean(traces)
print(trace)

# In practice, we often combine both approaches by choosing
# the largest nsamples (in the first implementation) so that
# nrows $\times$ nsamples fits into memory, and handle all samples beyond
# that via the split-and-map combination.
#
#
# If we reverse-mode differentiate through the sampler, we have to
# be careful because by default, reverse-mode differentiation
# stores all intermediate results (and the memory-efficiency of using
# jax.lax.map is void).
# To solve this problem, place a jax.checkpoint around the estimator:

traces = jax.lax.map(jax.checkpoint(estimate), keys)
trace = jnp.mean(traces)
print(trace)

# This implementation recomputes the forward pass for each key during the
# backward pass, but preserves the memory-efficiency on the backward pass.
#
#
# In summary, memory efficiency can be achieved by calling estimators
# inside jax.lax.map (with or without checkpoints).
