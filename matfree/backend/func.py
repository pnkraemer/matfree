"""Function funcations (including algorithmic differentiation)."""

import functools

import jax

# todo: renamed to `func`?
linearize = jax.linearize
vjp = jax.vjp
jacfwd = jax.jacfwd
jit = jax.jit
vmap = jax.vmap
partial = functools.partial
disable_jit = jax.disable_jit
eval_shape = jax.eval_shape
