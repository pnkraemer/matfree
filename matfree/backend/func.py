"""Function transformations (algorithmic differentiation, vmap, partial, and so on)."""

# API-wise, we tend to follow JAX and functools.
# But we only implement those function arguments that we need.
# For example, algorithmic differentiation functions do not offer a 'has_aux' argument
# at the moment, because we don't need it.

import functools

import jax

# Vectorisation


def vmap(fun, /, in_axes=0, out_axes=0):
    return jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)


# Partial and the like


def partial(func, /, *args, **kwargs):
    return functools.partial(func, *args, **kwargs)


def wraps(func, /):
    return functools.wraps(func)


# Algorithmic differentiation


def linearize(func, /, *primals):
    return jax.linearize(func, *primals)


def jacfwd(fun, /, argnums=0):
    return jax.jacfwd(fun, argnums)


def vjp(func, /, *primals):
    return jax.vjp(func, *primals)


# Inferring input and output shapes:


def eval_shape(func, /, *args, **kwargs):
    return jax.eval_shape(func, *args, **kwargs)


# Compilation (don't use in source!)


def jit(fun, /, *, static_argnums=None, static_argnames=None):
    return jax.jit(fun, static_argnames=static_argnames, static_argnums=static_argnums)
