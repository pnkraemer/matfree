"""Sampling algorithms."""

from hutch.backend import prng


def normal(*, shape, dtype):
    """Standard normal distributions."""

    def fun(key):
        return prng.normal(key, shape=shape, dtype=dtype)

    return fun


def rademacher(*, shape, dtype):
    """Normalised Rademacher distributions."""

    def fun(key):
        return prng.rademacher(key, shape=shape, dtype=dtype)

    return fun
