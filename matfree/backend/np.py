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
import jax.scipy.signal

# Creation functions:


def arange(start, /, stop=None, step=1):
    return jnp.arange(start, stop, step)


def linspace(start, stop, /, *, num, endpoint=True):
    return jnp.linspace(start, stop, num=num, endpoint=endpoint)


def asarray(obj, /):
    return jnp.asarray(obj)


def eye(n_rows):
    return jnp.eye(n_rows)


def ones_like(x, /):
    return jnp.ones_like(x)


def zeros_like(x, /):
    return jnp.zeros_like(x)


def zeros(shape, *, dtype=None):
    return jnp.zeros(shape, dtype=dtype)


def ones(shape, *, dtype=None):
    return jnp.ones(shape, dtype=dtype)


def concatenate(list_of_arrays, /):
    return jnp.concatenate(list_of_arrays)


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


def logical_and(a, b, /):
    return jnp.logical_and(a, b)


def logical_not(a, /):
    return jnp.logical_not(a)


# Utility functions


def any(x, /):  # noqa: A001
    return jnp.any(x)


def all(x, /):  # noqa: A001
    return jnp.all(x)


def allclose(x1, x2, /, *, rtol=1e-5, atol=1e-8):
    return jnp.allclose(x1, x2, rtol=rtol, atol=atol)


# Statistical functions


def mean(x, /, axis=None):
    return jnp.mean(x, axis)


def std(x, /, axis=None):
    return jnp.std(x, axis)


def sum(x, /, axis=None):  # noqa: A001
    return jnp.sum(x, axis)


def argmax(x, /, axis=None):
    return jnp.argmax(x, axis=axis)


def argsort(x, /):
    return jnp.argsort(x)


def nanmean(x, /, axis=None):
    return jnp.nanmean(x, axis)


def elementwise_max(a, b, /):
    return jnp.maximum(a, b)


def elementwise_min(a, b, /):
    return jnp.minimum(a, b)


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
    return x.shape


def dtype(x, /):
    return jnp.dtype(x)


# Functional implementation of constants


def nan():
    return jnp.nan


def pi():
    return jnp.pi


def finfo_eps(x, /):
    return jnp.finfo(x).eps


# Others


def convolve(a, b, /, mode="full"):
    return jnp.convolve(a, b, mode=mode)


def convolve2d(a, b, /, mode="full"):
    return jax.scipy.signal.convolve2d(a, b, mode=mode)


def tril(x, /, shift=0):
    return jnp.tril(x, shift)


def triu(x, /, shift=0):
    return jnp.triu(x, shift)
