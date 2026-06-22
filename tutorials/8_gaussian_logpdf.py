"""Differentiate a Gaussian log-density for GP hyperparameter learning.

Compute the Gaussian log-density in a matrix-free way by combining stochastic
Lanczos quadrature for the log-determinant with JAX's CG solver for the linear
system, and differentiate with respect to kernel hyperparameters.
"""

import jax
import jax.numpy as jnp

from matfree import decomp, funm, stochtrace

# Set up a squared-exponential GP covariance parametrised by
# log-lengthscale and log-observation-noise.

n = 30
xs = jnp.linspace(0.0, 1.0, n)
params = {"log_ell": jnp.array(0.0), "log_noise": jnp.array(-2.0)}


def cov_matrix(params):
    """Assemble the covariance matrix."""
    ell = jnp.exp(params["log_ell"])
    noise = jnp.exp(params["log_noise"])
    sq_dists = (xs[:, None] - xs[None, :]) ** 2
    return jnp.exp(-sq_dists / (2 * ell**2)) + noise * jnp.eye(n)


def matvec(x, params):
    """Compute a matrix-vector product."""
    return cov_matrix(params) @ x


# Build a matrix-free logpdf that bundles the SLQ log-determinant estimator
# and the CG linear solver into a single callable.


def create_logpdf(matvec, vec_like, *, num_matvecs=6, num_samples=500):
    """Create a matrix-free Gaussian logpdf."""
    tridiag = decomp.tridiag_sym(num_matvecs)
    integrand = funm.monte_carlo_funm_sym_logdet(tridiag)
    sampler = stochtrace.sampler_signs(vec_like, num=num_samples)
    estimator = stochtrace.estimator_monte_carlo(integrand, sampler=sampler)

    def logpdf(y, key, *params):
        dim = len(y)
        logdet = estimator(matvec, key, *params)
        y_solve = jax.scipy.sparse.linalg.cg(lambda v: matvec(v, *params), y)[0]
        quad = y @ y_solve
        return -0.5 * (quad + logdet + dim * jnp.log(2 * jnp.pi))

    return logpdf


logpdf = create_logpdf(matvec, xs)


# Evaluate and differentiate the log-density with respect to both hyperparameters.

y_obs = jax.random.normal(jax.random.PRNGKey(1), shape=(n,))
key = jax.random.PRNGKey(2)

print("Matrix-free | Value:", logpdf(y_obs, key, params))

grad_fn = jax.grad(logpdf, argnums=2)
print("Matrix-free | Grad: ", grad_fn(y_obs, key, params))


# For comparison, use jax.scipy.stats as a dense reference.


def logpdf_ref(y, params):
    """Dense reference logpdf via jax.scipy.stats."""
    K = cov_matrix(params)
    return jax.scipy.stats.multivariate_normal.logpdf(y, mean=jnp.zeros(n), cov=K)


print("Reference   | Value:", logpdf_ref(y_obs, params))
print("Reference   | Grad: ", jax.grad(logpdf_ref, argnums=1)(y_obs, params))
