"""Tests for least-squares functionality."""

from matfree import lstsq, test_util
from matfree.backend import linalg, prng, testing
from matfree.backend.typing import Callable


@testing.case()
def case_lstsq_lsmr() -> Callable:
    return lstsq.lsmr(atol=1e-5, btol=1e-5, ctol=1e-5)


def case_A_shape_wide() -> tuple:
    return 3, 6


def case_A_shape_tall() -> tuple:
    return 6, 3


def case_A_shape_square() -> tuple:
    return 3, 3


@testing.parametrize_with_cases("lstsq_fun", cases=".", prefix="case_lstsq_")
@testing.parametrize_with_cases("A_shape", cases=".", prefix="case_A_shape_")
def test_fwd_matches_numpy_lstsq(lstsq_fun: Callable, A_shape: tuple):
    key = prng.prng_key(1)

    key, subkey = prng.split(key, 2)
    matrix = prng.normal(subkey, shape=A_shape)
    key, subkey = prng.split(key, 2)
    rhs = prng.normal(subkey, shape=(A_shape[0],))

    def vecmat(vector):
        return matrix.T @ vector

    received, _stats = lstsq_fun(vecmat, rhs)
    expected = linalg.lstsq(matrix, rhs)[0]
    test_util.assert_allclose(received, expected)


@testing.parametrize_with_cases("lstsq_fun", cases=".", prefix="case_lstsq_")
@testing.parametrize_with_cases("A_shape", cases=".", prefix="case_A_shape_")
def test_fwd_matches_numpy_lstsq_parametrized(lstsq_fun: Callable, A_shape: tuple):
    key = prng.prng_key(1)

    key, subkey = prng.split(key, 2)
    matrix = prng.normal(subkey, shape=A_shape)
    key, subkey = prng.split(key, 2)
    rhs = prng.normal(subkey, shape=(A_shape[0],))

    def vecmat(vector, A):
        return A.T @ vector

    received, _stats = lstsq_fun(vecmat, rhs, matrix)
    expected = linalg.lstsq(matrix, rhs)[0]
    test_util.assert_allclose(received, expected)


@testing.parametrize_with_cases("lstsq_fun", cases=".", prefix="case_lstsq_")
@testing.parametrize_with_cases("A_shape", cases=".", prefix="case_A_shape_")
def test_fwd_matches_numpy_lstsq_damped(lstsq_fun: Callable, A_shape: tuple):
    key = prng.prng_key(1)

    key, subkey = prng.split(key, 2)
    matrix = prng.normal(subkey, shape=A_shape)
    key, subkey = prng.split(key, 2)
    rhs = prng.normal(subkey, shape=(A_shape[0],))

    def vecmat(vector):
        return matrix.T @ vector

    received, _stats = lstsq_fun(vecmat, rhs, damp=0.0)
    expected = linalg.lstsq(matrix, rhs)[0]
    test_util.assert_allclose(received, expected)
