"""Control flow."""

import jax

# API follows JAX, but we liberally work with positional- and keyword-only arguments
# We also rename some arguments for improved consistency:
# For example, we always use 'body_fun' and 'init_val',
#  even though jax.lax.scan uses 'f' and 'init'.


def scan(body_fun, init_val, /, xs, *, reverse=False):
    return jax.lax.scan(body_fun, init=init_val, xs=xs, reverse=reverse)


def cond(pred, /, true_fun, false_fun, *operands):
    return jax.lax.cond(pred, true_fun, false_fun, *operands)


def fori_loop(lower, upper, body_fun, init_val):
    return jax.lax.fori_loop(lower, upper, body_fun, init_val)


def while_loop(cond_fun, body_fun, init_val):
    return jax.lax.while_loop(cond_fun, body_fun, init_val)


def array_map(fun, /, xs):
    return jax.lax.map(fun, xs)
