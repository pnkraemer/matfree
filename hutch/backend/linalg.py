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
#
# def eigh_tridiagonal(d, e, **kwargs):
#     d = onp.asarray(d)
#     e = onp.asarray(e)
#     x, y = scipy.linalg.eigh_tridiagonal(d, e, **kwargs)
#     x = jnp.asarray(x)
#     y = jnp.asarray(y)
#     return x, y
