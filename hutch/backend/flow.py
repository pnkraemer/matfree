"""Control flow."""

import jax

scan = jax.lax.scan
map = jax.lax.map
cond = jax.lax.cond
fori_loop = jax.lax.fori_loop
