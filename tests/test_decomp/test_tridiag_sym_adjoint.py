"""Test the adjoint of tri-diagonalisation."""

import functools

import jax.flatten_util
import jax.numpy as jnp
import pytest_cases

from matfree import decomp, test_util


@pytest_cases.parametrize("reortho", ["full", "none"])
def test_adjoint_vjp_matches_jax_vjp(reortho, n=10, krylov_order=4):
    """Test that the custom VJP yields the same output as autodiff."""
    # Set up a test-matrix
    eigvals = jax.random.uniform(jax.random.PRNGKey(2), shape=(n,)) + 1.0
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    params = _sym(matrix)

    def matvec(s, p):
        return (p + p.T) @ s

    # Set up an initial vector
    vector = jax.random.normal(jax.random.PRNGKey(1), shape=(n,))

    # Flatten the inputs
    flat, unflatten = jax.flatten_util.ravel_pytree((vector, params))

    # Construct a vector-to-vector decomposition function
    def decompose(f, *, custom_vjp):
        kwargs = {"reortho": reortho, "custom_vjp": custom_vjp}
        algorithm = decomp.tridiag_sym(matvec, krylov_order, **kwargs)
        output = algorithm(*unflatten(f))
        return jax.flatten_util.ravel_pytree(output)[0]

    # Construct the two implementations
    reference = jax.jit(functools.partial(decompose, custom_vjp=False))
    implementation = jax.jit(functools.partial(decompose, custom_vjp=True))

    # Compute both VJPs
    fx_ref, vjp_ref = jax.vjp(reference, flat)
    fx_imp, vjp_imp = jax.vjp(implementation, flat)
    # Assert that the forward-passes are identical
    assert jnp.allclose(fx_ref, fx_imp)

    # Assert that the VJPs into a bunch of random directions are identical
    for seed in [4, 5, 6]:
        key = jax.random.PRNGKey(seed)
        dnu = jax.random.normal(key, shape=jnp.shape(reference(flat)))
        assert jnp.allclose(*vjp_ref(dnu), *vjp_imp(dnu), atol=1e-4, rtol=1e-4)


def _sym(m):
    return jnp.triu(m) - jnp.diag(0.5 * jnp.diag(m))
