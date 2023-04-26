"""Numerical linear algebra."""

# import scipy.linalg
import jax.lax
import jax.numpy as jnp

eigh = jnp.linalg.eigh
norm = jnp.linalg.norm
det = jnp.linalg.det
slogdet = jnp.linalg.slogdet
tridiagonal = jax.lax.linalg.tridiagonal
eigh_tridiagonal = jax.scipy.linalg.eigh_tridiagonal
qr = jnp.linalg.qr
