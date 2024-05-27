import jax
import jax.flatten_util
import jax.numpy as jnp
import pytest
import pytest_cases

from matfree import decomp, test_util
from matfree.backend import linalg


@pytest_cases.parametrize("nrows", [3])
@pytest_cases.parametrize("krylov_depth", [2])
@pytest_cases.parametrize("reortho", ["none", "full"])
@pytest_cases.parametrize("dtype", [float])
def test_adjoint_matches_jax_dot_vjp(nrows, krylov_depth, reortho, dtype):
    # todo: see which components simplify for symmetric matrices

    # Create a matrix and a direction as a test-case
    A = jax.random.normal(jax.random.PRNGKey(1), shape=(nrows, nrows), dtype=dtype)
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(nrows,), dtype=dtype)

    # Set up the algorithms
    algorithm_autodiff = decomp.hessenberg(
        lambda s, p: p @ s, krylov_depth, reortho=reortho, custom_vjp=False
    )
    algorithm_adjoint = decomp.hessenberg(
        lambda s, p: p @ s, krylov_depth, reortho=reortho, custom_vjp=True
    )

    # Forward pass
    (Q, H, r, c), vjp_autodiff = jax.vjp(algorithm_autodiff, v, A)
    (_Q, _H, _r, _c), vjp_adjoint = jax.vjp(algorithm_adjoint, v, A)

    # Random input gradients (no sparsity at all)
    (dQ, dH, dr, dc) = test_util.tree_random_like(jax.random.PRNGKey(3), (Q, H, r, c))

    # Call the auto-diff VJP
    dv_autodiff, dp_autodiff = vjp_autodiff((dQ, dH, dr, dc))
    dv_adjoint, dp_adjoint = vjp_adjoint((dQ, dH, dr, dc))

    # Tie the tolerance to the floating-point accuracy
    small_value = 10 * jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    # Assert gradients match
    assert jnp.allclose(dv_adjoint, dv_autodiff, **tols)
    assert jnp.allclose(dp_adjoint, dp_autodiff, **tols)

    # Assert that the values are only similar, not identical
    assert not jnp.all(dv_adjoint == dv_autodiff)
    assert not jnp.all(dp_adjoint == dp_autodiff)


@pytest_cases.parametrize("nrows", [15])
@pytest_cases.parametrize("krylov_depth", [10])
@pytest_cases.parametrize("reortho", ["full"])
def test_adjoint_matches_jax_dot_vjp_hilbert_matrix_and_full_reortho(
    nrows, krylov_depth, reortho
):
    jax.config.update("jax_enable_x64", True)

    # Create a matrix and a direction as a test-case
    A = _lower(linalg.hilbert(nrows))
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(nrows,), dtype=A.dtype)

    def matvec(s, p):
        return (p + p.T) @ s

    # Set up the algorithms
    algorithm_autodiff = decomp.hessenberg(
        matvec, krylov_depth, reortho=reortho, custom_vjp=False
    )
    algorithm_adjoint = decomp.hessenberg(
        matvec, krylov_depth, reortho=reortho, custom_vjp=True
    )

    # Forward pass
    (Q, H, r, c), vjp_autodiff = jax.vjp(algorithm_autodiff, v, A)
    (_Q, _H, _r, _c), vjp_adjoint = jax.vjp(algorithm_adjoint, v, A)

    # Random input gradients (no sparsity at all)
    (dQ, dH, dr, dc) = test_util.tree_random_like(jax.random.PRNGKey(3), (Q, H, r, c))

    # Call the auto-diff VJP
    dv_autodiff, dp_autodiff = vjp_autodiff((dQ, dH, dr, dc))
    dv_adjoint, dp_adjoint = vjp_adjoint((dQ, dH, dr, dc))

    # Tie the tolerance to the floating-point accuracy
    small_value = 10 * jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    # Assert gradients match
    assert jnp.allclose(dv_adjoint, dv_autodiff, **tols)
    assert jnp.allclose(dp_adjoint, dp_autodiff, **tols)

    # Assert that the values are only similar, not identical
    assert not jnp.all(dv_adjoint == dv_autodiff)
    assert not jnp.all(dp_adjoint == dp_autodiff)

    jax.config.update("jax_enable_x64", False)


def _lower(m):
    m_tril = jnp.tril(m)
    return m_tril - 0.5 * linalg.diagonal_matrix(linalg.diagonal(m_tril))


@pytest_cases.parametrize("reortho_wrong", [True, "full_with_sparsity", "None"])
def test_raises_type_error_for_wrong_reorthogonalisation_flag(reortho_wrong):
    # Create a matrix and a direction as a test-case

    # Set up the algorithms
    with pytest.raises(TypeError, match="Unexpected input"):
        _ = decomp.hessenberg(lambda s: s, 1, reortho=reortho_wrong)
