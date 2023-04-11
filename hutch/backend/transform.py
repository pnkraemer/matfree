"""Function transformations (including algorithmic differentiation)."""

import functools

import jax

linearize = jax.linearize
jacfwd = jax.jacfwd
jit = jax.jit
vmap = jax.vmap
partial = functools.partial
