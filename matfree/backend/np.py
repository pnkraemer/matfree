"""NumPy-style API."""

import jax.numpy as jnp

# Creation functions:


def arange(start, /, stop=None, step=1):
    return jnp.arange(start, stop, step)


def asarray(obj, /):
    return jnp.asarray(obj)


def eye(n_rows):
    return jnp.eye(n_rows)


def ones_like(x, /):
    return jnp.ones_like(x)


def zeros(shape, *, dtype=None):
    return jnp.zeros(shape, dtype=dtype)


# Element-wise functions


def abs(x, /):  # noqa: A001
    return jnp.abs(x)


def log(x, /):
    return jnp.log(x)


def isnan(x, /):
    return jnp.isnan(x)


def sin(x, /):
    return jnp.sin(x)


def cos(x, /):
    return jnp.cos(x)


# Linear algebra functions
# Todo: move to backend.linalg?


def vecdot(x1, x2, /):
    return jnp.dot(x1, x2)


def diagonal(x, /, offset=0):
    return jnp.diag(x, offset)


def trace(x, /):
    return jnp.trace(x)


# Utility functions


def any(x, /):  # noqa: A001
    return jnp.any(x)


def allclose(x1, x2, /, *, rtol=1e-5, atol=1e-8):
    return jnp.allclose(x1, x2, rtol=rtol, atol=atol)


# Statistical functions


def sum(x, /, axis=None):  # noqa: A001
    return jnp.sum(x, axis)


def maximum(x1, x2):  # todo: call max()
    return jnp.maximum(x1, x2)


def nanmean(x, /, axis=None):
    return jnp.nanmean(x, axis)


# Searching functions


def where(condition, x1, x2, /):
    return jnp.where(condition, x1, x2)


# Manipulation functions


def reshape(x, /, shape):
    return jnp.reshape(x, shape)


def flip(x, /):
    return jnp.flip(x)


# Functional implementation of methods


def shape(x, /):
    return jnp.shape(x)


def dtype(x, /):
    return jnp.dtype(x)


# Functional implementation of constants


def nan():
    return jnp.nan
