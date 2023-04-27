"""NumPy-style API."""

import jax.numpy as jnp

abs = jnp.abs  # noqa: A001
any = jnp.any  # noqa: A001
asarray = jnp.asarray
allclose = jnp.allclose
arange = jnp.arange
cos = jnp.cos
dot = jnp.dot
diag = jnp.diag
dtype = jnp.dtype
empty = jnp.empty
eye = jnp.eye
finfo = jnp.finfo  # pylint: disable=invalid-name
flip = jnp.flip
isnan = jnp.isnan
isscalar = jnp.isscalar
log = jnp.log
log10 = jnp.log10
logical_not = jnp.logical_not
maximum = jnp.maximum
mean = jnp.mean
ndim = jnp.ndim
nan = jnp.nan
nanmean = jnp.nanmean
nansum = jnp.nansum
ones = jnp.ones
ones_like = jnp.ones_like
reshape = jnp.reshape
round = jnp.round  # noqa: A001
set_printoptions = jnp.set_printoptions
shape = jnp.shape
sin = jnp.sin
stack = jnp.stack
sum = jnp.sum  # noqa: A001
trace = jnp.trace
where = jnp.where
zeros = jnp.zeros
zeros_like = jnp.zeros_like
