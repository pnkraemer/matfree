"""Pseudo-random-number utilities."""


import jax.random


def prng_key(seed):
    return jax.random.PRNGKey(seed=seed)


def split(key, num=2):
    return jax.random.split(key, num=num)


def normal(key, *, shape, dtype=None):
    if dtype is None:
        return jax.random.normal(key, shape=shape)
    return jax.random.normal(key, shape=shape, dtype=dtype)


def uniform(key, *, shape, dtype=None):
    if dtype is None:
        return jax.random.uniform(key, shape=shape)
    return jax.random.uniform(key, shape=shape, dtype=dtype)


def rademacher(key, *, shape, dtype=None):
    if dtype is None:
        return jax.random.rademacher(key, shape=shape)
    return jax.random.rademacher(key, shape=shape, dtype=dtype)
