"""Configuration."""

import jax


def update(what, how, /):
    jax.config.update(what, how)
