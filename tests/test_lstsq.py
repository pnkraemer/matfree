"""Tests for least-squares functionality."""

from typing import Callable

import jax
import jax.numpy as jnp
import pytest_cases as ptc

from matfree import lstsq


@ptc.case()
def case_lstsq_lsmr() -> Callable:
    return lstsq.lsmr(atol=1e-5, btol=1e-5, ctol=1e-5)


def case_A_shape_wide() -> tuple:
    return 3, 6


def case_A_shape_tall() -> tuple:
    return 6, 3


def case_A_shape_square() -> tuple:
    return 3, 3


@ptc.parametrize_with_cases("lstsq_fun", cases=".", prefix="case_lstsq_")
@ptc.parametrize_with_cases("A_shape", cases=".", prefix="case_A_shape_")
def test_fwd_matches_numpy_lstsq(lstsq_fun: Callable, A_shape: tuple):
    key = jax.random.PRNGKey(1)

    key, subkey = jax.random.split(key, 2)
    matrix = jax.random.normal(subkey, shape=A_shape)
    key, subkey = jax.random.split(key, 2)
    rhs = jax.random.normal(subkey, shape=(A_shape[0],))

    def vecmat(vector, A):
        return A.T @ vector

    received, _stats = lstsq_fun(vecmat, rhs, vecmat_args=(matrix,))
    expected = jnp.linalg.lstsq(matrix, rhs)[0]
    tol = jnp.sqrt(jnp.finfo(received.dtype).eps)
    assert jnp.allclose(received, expected, atol=tol, rtol=tol)
