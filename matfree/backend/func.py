"""Function transformations (algorithmic differentiation, vmap, partial, and so on)."""

import functools

import jax

linearize = jax.linearize
vjp = jax.vjp
jacfwd = jax.jacfwd
jit = jax.jit
vmap = jax.vmap
partial = functools.partial
disable_jit = jax.disable_jit
eval_shape = jax.eval_shape
