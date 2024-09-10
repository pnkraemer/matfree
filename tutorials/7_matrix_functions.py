"""Compute matrix-functions without materializing large matrices.

Sometimes, we need to compute matrix-exponentials, log-determinants,
or similar functions of matrices, but our matrices are too big to
use functions from
[scipy.linalg](https://docs.scipy.org/doc/scipy/reference/linalg.html)
or
[jax.scipy.linalg](https://jax.readthedocs.io/en/latest/jax.scipy.html#module-jax.scipy.linalg).
However, matrix-free linear algebra scales to even the largest of matrices.
Here is how to use Matfree to compute functions of large matrices.
"""

import jax

from matfree import decomp, funm

n = 7  # imagine n = 10^5 or larger

key = jax.random.PRNGKey(1)
key1, key2 = jax.random.split(key, num=2)
large_matrix = jax.random.normal(key1, shape=(n, n))
vector = jax.random.normal(key2, shape=(n,))


# Here is the expected value, computed with jax.scipy.linalg.

expected = jax.scipy.linalg.expm(large_matrix) @ vector
print(expected)


# Instead of using jax.scipy.linalg, we can use matrix-vector products
# in combination with the Arnoldi iteration to approximate the
# matrix-function-vector product


def large_matvec(v):
    """Evaluate a matrix-vector product."""
    return large_matrix @ v


krylov_depth = 5
arnoldi = decomp.hessenberg(krylov_depth, reortho="full")
dense_funm = funm.dense_funm_pade_exp()
matfun_vec = funm.funm_arnoldi(dense_funm, arnoldi)
received = matfun_vec(large_matvec, vector)
print(received)
