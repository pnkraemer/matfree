"""Implement vector calculus in linear complexity.

Implementing vector calculus with conventional
algorithmic differentiation can be inefficient.
For example, computing the divergence of a
vector field requires computing the trace of a Jacobian.
The divergence of a vector field is
important when evaluating Laplacians of scalar functions.

Here is how we can implement divergences and
Laplacians without forming full Jacobian matrices:
"""

import jax
import jax.numpy as jnp

from matfree import hutchinson

# ## Divergences and Laplacians
#
# The divergence of a vector field is the trace of its Jacobian.
# The conventional implementation would look like this:


def divergence_dense(vf):
    """Compute the divergence of a vector field."""

    def div_fn(x):
        J = jax.jacfwd(vf)
        return jnp.trace(J(x))

    return div_fn


# This implementation computes the divergence of a vector field:


def fun(x):
    """Evaluate a scalar valued function."""
    return jnp.dot(x, x) ** 2


x0 = jnp.arange(1.0, 4.0)
gradient = jax.grad(fun)
laplacian = divergence_dense(gradient)
print(jax.hessian(fun)(x0))
print(laplacian(x0))


# But the implementation above requires $O(d^2)$ storage
# because it evaluates the dense Jacobian.
# This is problematic for high-dimensional problems.
#
# ## Matrix-free implementation
#
# If we have access to Jacobian-vector products (which we usually do),
# we can use matrix-free trace estimation
# to approximate divergences and Laplacians without forming full Jacobians:
#


def divergence_matfree(vf, /, *, num):
    """Compute the divergence with Hutchinson's estimator."""

    def divergence(k, x):
        _fx, jvp = jax.linearize(vf, x)
        integrand_laplacian = hutchinson.integrand_trace(jvp)
        normal = hutchinson.sampler_normal(x, num=num)
        estimator = hutchinson.hutchinson(integrand_laplacian, sample_fun=normal)
        return estimator(k)

    return divergence


# -
# The difference to the "naive" implementation is that the implicit one
# does not form dense Jacobians. It requires $O(d)$ memory and
# $O(d N)$ operations (for $N$ Monte-Carlo samples).
# For large-scale problems, it may be the only way of computing Laplacians reliably.

laplacian_matfree = divergence_matfree(gradient, num=10_000)
print(laplacian(x0))
print(laplacian_matfree(jax.random.PRNGKey(1), x0))


# In summary, compute matrix-free linear algebra
# and algorithmic differentiation to implement vector calculus.
#
# ## Diagonals of Jacobians
#
# If we replace trace estimation with diagonal estimation,
# we can compute the diagonal of Jacobian matrices in
# $O(d)$ memory and $O(dN)$ operations.
