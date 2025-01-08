"""Numerical linear algebra."""

import jax
import jax.numpy as jnp


def vector_norm(x, /):
    return jnp.linalg.norm(x)


def matrix_norm(x, /, which):
    return jnp.linalg.norm(x, ord=which)


def qr_reduced(x, /):
    return jnp.linalg.qr(x, mode="reduced")


def eigh(x, /):
    return jnp.linalg.eigh(x)


def eig(x, /):
    vals, vecs = jnp.linalg.eig(x)

    # Order the values and vectors by magnitude
    # (jax.numpy.linalg.eig does not do this)
    ordered = jnp.argsort(vals)[::-1]
    vals = vals[ordered]
    vecs = vecs[:, ordered]
    return vals, vecs


def cholesky(x, /):
    return jnp.linalg.cholesky(x)


def cho_factor(matrix, /):
    return jax.scipy.linalg.cho_factor(matrix)


def cho_solve(factor, b, /):
    return jax.scipy.linalg.cho_solve(factor, b)


def slogdet(x, /):
    return jnp.linalg.slogdet(x)


def inner(x1, x2, /):
    # todo: distinguish vecdot, vdot, dot, and matmul?
    return jnp.inner(x1, x2)


def outer(a, b, /):
    return jnp.outer(a, b)


def hilbert(n, /):
    return jax.scipy.linalg.hilbert(n)


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


def funm_schur(A, f, /):
    return jax.scipy.linalg.funm(A, f)


def funm_pade_exp(A, /):
    return jax.scipy.linalg.expm(A)
