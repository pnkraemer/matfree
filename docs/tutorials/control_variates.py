# # Multilevel estimation and control variates
#
# Jointly estimating traces and diagonals improves performance.
# Here is how to use it:

import jax
import jax.numpy as jnp

from matfree import hutchinson, slq

a = jnp.reshape(jnp.arange(12.0), (6, 2))
key = jax.random.PRNGKey(1)
matvec = lambda x: a.T @ (a @ x)
sample_fun = hutchinson.sampler_normal(shape=(2,))


trace, diagonal = hutchinson.trace_and_diagonal(
    matvec, key=key, num_levels=10_000, sample_fun=sample_fun
)
print(jnp.round(trace))
print(jnp.round(diagonal))

# For comparison:

print(jnp.round(jnp.trace(a.T @ a)))
print(jnp.round(jnp.diagonal(a.T @ a)))


# Why is the argument called `num_levels`? Because under the hood,
# `trace_and_diagonal` implements a multilevel diagonal-estimation scheme:
# +
_, diagonal_1 = hutchinson.trace_and_diagonal(
    matvec, key=key, num_levels=10_000, sample_fun=sample_fun
)
init = jnp.zeros(shape=(2,), dtype=float)
diagonal_2 = hutchinson.diagonal_multilevel(
    matvec, init, key=key, num_levels=10_000, sample_fun=sample_fun
)

print(jnp.round(diagonal_1, 4))

print(jnp.round(diagonal_2, 4))

diagonal = hutchinson.diagonal_multilevel(
    matvec,
    init,
    key=key,
    num_levels=10,
    num_samples_per_batch=1000,
    num_batches_per_level=10,
    sample_fun=sample_fun,
)
print(jnp.round(diagonal))

# -

# Does the multilevel scheme help? That is not always clear; but [here](https://github.com/pnkraemer/matfree/blob/main/docs/benchmarks/control_variates.py) is a benchmark.
