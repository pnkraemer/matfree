"""Numerical linear algebra."""

import jax.numpy as jnp
import numpy as onp
import scipy.linalg

norm = jnp.linalg.norm
det = jnp.linalg.det
slogdet = jnp.linalg.slogdet


def eigh_tridiagonal(d, e, **kwargs):
    d = onp.asarray(d)
    e = onp.asarray(e)
    x, y = scipy.linalg.eigh_tridiagonal(d, e, **kwargs)
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return x, y
