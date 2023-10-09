"""Numerical linear algebra."""

import jax
import jax.numpy as jnp


def vector_norm(x, /):
    return jnp.linalg.norm(x)


def matrix_norm(x, /, which):
    return jnp.linalg.norm(x, ord=which)


def qr(x, /, *, mode="reduced"):
    return jnp.linalg.qr(x, mode=mode)


def eigh(x, /):
    return jnp.linalg.eigh(x)


def slogdet(x, /):
    return jnp.linalg.slogdet(x)


def vecdot(x1, x2, /):
    return jnp.dot(x1, x2)


def diagonal(x, /, offset=0):
    """Extract the diagonal of a matrix."""
    return jnp.diag(x, offset)


def diagonal_matrix(x, /, offset=0):  # not part of array API
    """Construct a diagonal matrix."""
    return jnp.diag(x, offset)


def trace(x, /):
    return jnp.trace(x)


def svd(A, /, *, full_matrices=True):
    return jnp.linalg.svd(A, full_matrices=full_matrices)


def pinv(A, /):
    return jnp.linalg.pinv(A)


def solve(A, b, /):
    return jnp.linalg.solve(A, b)


def cg(Av, b, /):
    return jax.scipy.sparse.linalg.cg(Av, b)
