"""NumPy-style API.

In here, we loosely follow the Array API:

https://data-apis.org/array-api/2022.12/

But deviate in a few points:
* The functions here do not have all the arguments specified in the API
  (we only wrap the arguments we need)
* Our current version of diag/diagonal is slightly different
* We do not use methods on Array types, e.g. shape(), dtype(). Instead,
  these are functions. (Not all backends might always follow this method interface.)
* We do not implement any constants (e.g. NaN, Pi). Instead, these are functions.
* We call max/min/amax/amin array_max and elementwise_max.
  This is more verbose than what the array API suggests.

"""

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


def ones(shape, *, dtype=None):
    return jnp.ones(shape, dtype=dtype)


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


def sqrt(x, /):
    return jnp.sqrt(x)


def sign(x, /):
    return jnp.sign(x)


# Utility functions


def any(x, /):  # noqa: A001
    return jnp.any(x)


def allclose(x1, x2, /, *, rtol=1e-5, atol=1e-8):
    return jnp.allclose(x1, x2, rtol=rtol, atol=atol)


# Statistical functions


def mean(x, /, axis=None):
    return jnp.mean(x, axis)


def std(x, /, axis=None):
    return jnp.std(x, axis)


def sum(x, /, axis=None):  # noqa: A001
    return jnp.sum(x, axis)


def array_min(x, /):
    return jnp.amin(x)


def array_max(x, /, axis=None):
    return jnp.amax(x, axis=axis)


def elementwise_max(x1, x2, /):
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


def roll(x, /, shift):
    return jnp.roll(x, shift)


# Functional implementation of what are usually array-methods


def shape(x, /):
    return jnp.shape(x)


def dtype(x, /):
    return jnp.dtype(x)


# Functional implementation of constants


def nan():
    return jnp.nan


# Others


def convolve(a, b, /, mode="full"):
    return jnp.convolve(a, b, mode=mode)
