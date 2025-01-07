"""Compute matrix functions without materializing large matrices.

Sometimes, we need to compute matrix exponentials, log-determinants,
or similar functions of matrices, but our matrices are too big to
use functions from
[scipy.linalg](https://docs.scipy.org/doc/scipy/reference/linalg.html)
or
[jax.scipy.linalg](https://jax.readthedocs.io/en/latest/jax.scipy.html#module-jax.scipy.linalg).
However, matrix-free linear algebra scales to even the largest of matrices.
Here is how to use Matfree to compute functions of large matrices.
"""

import functools

import jax

from matfree import decomp, funm

n = 7  # imagine n = 10^5 or larger

key = jax.random.PRNGKey(1)
key, subkey = jax.random.split(key, num=2)
large_matrix = jax.random.normal(subkey, shape=(n, n))


# The expected value is computed with jax.scipy.linalg.

key, subkey = jax.random.split(key, num=2)
vector = jax.random.normal(subkey, shape=(n,))
expected = jax.scipy.linalg.expm(large_matrix) @ vector
print(expected)


# Instead of using jax.scipy.linalg, we can use matrix-vector products
# in combination with the Arnoldi iteration to approximate the
# matrix-function-vector product.


def large_matvec(v):
    """Evaluate a matrix-vector product."""
    return large_matrix @ v


num_matvecs = 5
arnoldi = decomp.hessenberg(num_matvecs, reortho="full")
dense_funm = funm.dense_funm_pade_exp()
matfun_vec = funm.funm_arnoldi(dense_funm, arnoldi)
received = matfun_vec(large_matvec, vector)
print(received)


# The matrix-function vector product can be combined with all usual
# JAX transformations. For example, after fixing the matvec-function
# as the first argument, we can vectorize the matrix function with jax.vmap
# and compile it with jax.jit.

matfun_vec = functools.partial(matfun_vec, large_matvec)
key, subkey = jax.random.split(key, num=2)
vector_batch = jax.random.normal(subkey, shape=(5, n))  # a batch of 5 vectors
received = jax.jit(jax.vmap(matfun_vec))(vector_batch)
print(received.shape)


# Talking about function transformations: we can also
# reverse-mode-differentiate the matrix functions efficiently.

jac = jax.jacrev(matfun_vec)(vector)
print(jac)

# Under the hood, reverse-mode derivatives of Arnoldi- and Lanczos-based
# matrix functions use the fast algorithm for gradients of the
# Lanczos and Arnoldi iterations from
# [this paper](https://arxiv.org/abs/2405.17277).
# Please consider citing it if you use reverse-mode derivatives
# functions of matrices
# (a BibTex is [here](https://pnkraemer.github.io/matfree/)).
