"""Estimate log-determinants of PyTree-valued functions.

Can we compute log-determinants if the matrix-vector
products are pytree-valued?
Yes, we can. Matfree natively supports PyTrees.
"""

import jax
import jax.numpy as jnp

from matfree import decomp, funm, stochtrace

# Create a test-problem: a function that maps a pytree (dict) to a pytree (tuple).
# Its (regularised) Gauss--Newton Hessian shall be the matrix-vector product
# whose log-determinant we estimate.


def testfunc(x):
    """Map a dictionary to a tuple with some arbitrary values."""
    return jnp.linalg.norm(x["weights"]), x["bias"]


# Create a test-input

b = jnp.arange(1.0, 40.0)
W = jnp.stack([b + 1.0, b + 2.0])
x0 = {"weights": W, "bias": b}

# Linearise the functions

f0, jvp = jax.linearize(testfunc, x0)
_f0, vjp = jax.vjp(testfunc, x0)
print(jax.tree.map(jnp.shape, f0))
print(jax.tree.map(jnp.shape, jvp(x0)))
print(jax.tree.map(jnp.shape, vjp(f0)))


# Use the same API as if the matrix-vector product were array-valued.
# Matfree flattens all trees internally.


def make_matvec(alpha):
    """Create a matrix-vector product function."""

    def fun(fx, /):
        r"""Matrix-vector product with $J J^\top + \alpha I$."""
        vjp_eval = vjp(fx)
        matvec_eval = jvp(*vjp_eval)
        return jax.tree.map(lambda x, y: x + alpha * y, matvec_eval, fx)

    return fun


matvec = make_matvec(alpha=0.1)
num_matvecs = 3
tridiag_sym = decomp.tridiag_sym(num_matvecs)
integrand = funm.integrand_funm_sym_logdet(tridiag_sym)
sample_fun = stochtrace.sampler_normal(f0, num=10)
estimator = stochtrace.estimator(integrand, sampler=sample_fun)
key = jax.random.PRNGKey(1)
logdet = estimator(matvec, key)
print(logdet)


# For reference: flatten all arguments
# and compute the dense log-determinant:

f0_flat, unravel_func_f = jax.flatten_util.ravel_pytree(f0)


def make_matvec_flat(alpha):
    """Create a flattened matrix-vector-product function."""

    def fun(f_flat):
        """Evaluate a flattened matrix-vector product."""
        f_unravelled = unravel_func_f(f_flat)
        vjp_eval = vjp(f_unravelled)
        matvec_eval = jvp(*vjp_eval)
        f_eval, _unravel_func = jax.flatten_util.ravel_pytree(matvec_eval)
        return f_eval + alpha * f_flat

    return fun


matvec_flat = make_matvec_flat(alpha=0.1)
M = jax.jacfwd(matvec_flat)(f0_flat)
print(jnp.linalg.slogdet(M))
