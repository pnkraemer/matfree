"""Function transformations (including algorithmic differentiation)."""

import jax

linearize = jax.linearize
jacfwd = jax.jacfwd
jit = jax.jit
vmap = jax.vmap
