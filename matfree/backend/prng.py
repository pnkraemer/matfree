"""Pseudo-random-number utilities."""


import jax.random

PRNGKey = jax.random.PRNGKey
normal = jax.random.normal
rademacher = jax.random.rademacher
uniform = jax.random.uniform
split = jax.random.split
