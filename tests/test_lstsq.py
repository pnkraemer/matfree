"""Tests for least-squares functionality."""

from matfree import lstsq, test_util
from matfree.backend import func, linalg, prng, testing
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
def test_value_and_grad_matches_numpy_lstsq(lstsq_fun: Callable, A_shape: tuple):
    key = prng.prng_key(1)

    key, subkey = prng.split(key, 2)
    matrix = prng.normal(subkey, shape=A_shape)
    key, subkey = prng.split(key, 2)
    rhs = prng.normal(subkey, shape=(A_shape[0],))
    key, subkey = prng.split(key, num=2)
    dsol = prng.normal(subkey, shape=(A_shape[1],))

    def lstsq_jnp(a, b):
        sol, *_ = linalg.lstsq(a, b)
        return sol

    expected, expected_vjp = func.vjp(lstsq_jnp, matrix, rhs)
    dmatrix1, drhs1 = expected_vjp(dsol)

    def vecmat(vector, p_as_list):
        [p] = p_as_list
        return p.T @ vector

    def lstsq_matfree(a, b):
        sol, _ = lstsq_fun(vecmat, a, b)
        return sol

    received, received_vjp = func.vjp(lstsq_matfree, rhs, [matrix])
    drhs2, [dmatrix2] = received_vjp(dsol)  # mind the order of rhs & matrix

    test_util.assert_allclose(received, expected)
    test_util.assert_allclose(dmatrix1, dmatrix2)
    test_util.assert_allclose(drhs1, drhs2)
