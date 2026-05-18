"""Tests for integrand_trace_svd (XTrace)."""

import jax
import jax.numpy as jnp
import pytest
from matfree import stochtrace


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("resphere", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_trace_svd_fast_spectral_decay(resphere, dtype):
    """Assert that the trace of a matrix with fast spectral decay is estimated accurately."""
    rdtype = dtype(0).real.dtype
    n = 1000
    num_rep = 10
    key = jax.random.PRNGKey(1)
    key_mat, key = jax.random.split(key)
    U = jnp.linalg.qr(jax.random.normal(key_mat, (n, n), dtype=dtype))[0]
    d = 0.7 ** jnp.arange(n, dtype=rdtype)
    expected = jnp.sum(d).astype(dtype)

    sampler = stochtrace.sampler_normal(jnp.ones(n, dtype=dtype), num=35)
    integrand = stochtrace.integrand_trace_svd(resphere=resphere)
    estimate = stochtrace.estimator_loo(integrand, sampler)
    matvec = lambda v: U @ (d * (U.T.conj() @ v))

    key_ests = jax.random.split(key, num_rep)
    received = jax.vmap(lambda key: estimate(matvec, key))(key_ests)
    rel_err = jnp.abs(received - expected) / jnp.abs(expected)
    mean_rel_err = jnp.mean(rel_err)
    assert float(mean_rel_err) < 1e-5


@pytest.mark.parametrize("resphere", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_trace_svd_large_spectral_drop(resphere, dtype):
    """Assert that the trace of a matrix with some large eigenvalues and the rest small is estimated accurately."""
    rdtype = dtype(0).real.dtype
    n = 1000
    m = 50
    num_rep = 10
    key = jax.random.PRNGKey(4)
    key_mat, key = jax.random.split(key)
    U = jnp.linalg.qr(jax.random.normal(key_mat, (n, n), dtype=dtype))[0]
    d = jnp.concatenate([jnp.ones(m, dtype=rdtype), jnp.full(n-m, 1e-3, dtype=rdtype)])
    expected = jnp.sum(d).astype(dtype)

    sampler = stochtrace.sampler_normal(jnp.ones(n, dtype=dtype), num=m + 10)
    integrand = stochtrace.integrand_trace_svd(resphere=resphere)
    estimate = stochtrace.estimator_loo(integrand, sampler)
    matvec = lambda v: U @ (d * (U.T.conj() @ v))

    key_ests = jax.random.split(key, num_rep)
    received = jax.vmap(lambda key: estimate(matvec, key))(key_ests)
    rel_err = jnp.abs(received - expected) / jnp.abs(expected)
    mean_rel_err = jnp.mean(rel_err)
    assert float(mean_rel_err) < (1e-5 if resphere else 1e-4)


@pytest.mark.parametrize("n, rank", [(50, 10), (100, 30)])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_trace_svd_low_rank_operator(n, rank, dtype):
    """Assert that the trace of an already-low-rank operator is computed exactly."""
    key = jax.random.PRNGKey(5)
    key_mat1, key_mat2, key = jax.random.split(key, 3)
    A = jax.random.normal(key_mat1, (n, rank), dtype=dtype)
    B = jax.random.normal(key_mat2, (rank, n), dtype=dtype)
    expected = jnp.trace(A @ B)

    def matvec(v):
        return A @ (B @ v)

    sampler = stochtrace.sampler_normal(jnp.ones(n, dtype=dtype), num=rank + 1)
    integrand = stochtrace.integrand_trace_svd()
    estimate = stochtrace.estimator_loo(integrand, sampler)
    assert jnp.allclose(estimate(matvec, key), expected)
