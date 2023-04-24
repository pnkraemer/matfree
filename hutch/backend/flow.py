"""Control flow."""

import jax

scan = jax.lax.scan
map = jax.lax.map  # pylint: disable=redefined-builtin
cond = jax.lax.cond
fori_loop = jax.lax.fori_loop
