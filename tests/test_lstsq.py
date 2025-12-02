"""Tests for least-squares functionality."""

from matfree import lstsq, test_util
from matfree.backend import config, func, linalg, np, prng, testing


def case_A_shape_wide() -> tuple:
    return 3, 6


def case_A_shape_tall() -> tuple:
    return 6, 3


def case_A_shape_square() -> tuple:
    return 3, 3


@testing.parametrize_with_cases("A_shape", cases=".", prefix="case_A_shape_")
@testing.parametrize("provide_x0", [True, False])
@testing.parametrize("seed", [1])
def test_value_and_grad_matches_numpy_lstsq_no_damping(
    seed, A_shape: tuple, provide_x0: bool
):
    key = prng.prng_key(seed)

    key, subkey = prng.split(key, 2)
    matrix = prng.normal(subkey, shape=A_shape)
    key, subkey = prng.split(key, 2)
    rhs = prng.normal(subkey, shape=(A_shape[0],))
    key, subkey = prng.split(key, num=2)
    dsol = prng.normal(subkey, shape=(A_shape[1],))

    # If the matrix is wide, any nonzero initial guess affects the optimal solution
    # so the comparison to np.linalg.lstsq() is no longer valid. Thus, the caveat below.
    key, subkey = prng.split(key, num=2)
    is_wide = A_shape[1] > A_shape[0]
    x0_suggestion = prng.normal(subkey, shape=(A_shape[1],))
    x0 = x0_suggestion if provide_x0 and not is_wide else None

    def lstsq_jnp(a, b):
        sol, *_ = linalg.lstsq(a, b)
        return sol

    expected, expected_vjp = func.vjp(lstsq_jnp, matrix, rhs)
    dmatrix1, drhs1 = expected_vjp(dsol)

    def vecmat(vector, p_as_list):
        [p] = p_as_list
        return p.T @ vector

    @func.jit
    def lstsq_matfree(a, b):
        lsmr = lstsq.lsmr(atol=1e-5, btol=1e-5, ctol=1e-5)
        sol, _ = lsmr(vecmat, a, b, x0=x0)
        return sol

    received, received_vjp = func.vjp(lstsq_matfree, rhs, [matrix])
    drhs2, [dmatrix2] = received_vjp(dsol)  # mind the order of rhs & matrix

    test_util.assert_allclose(received, expected)
    test_util.assert_allclose(drhs1, drhs2)
    test_util.assert_allclose(dmatrix1, dmatrix2)


@testing.parametrize_with_cases("A_shape", cases=".", prefix="case_A_shape_")
@testing.parametrize("seed", [1])
@testing.parametrize("rank_deficient", [True, False])
def test_value_and_grad_matches_linalg_solve_with_damping(
    seed, A_shape: tuple, rank_deficient
):
    key = prng.prng_key(seed)

    key, subkey = prng.split(key, 2)
    matrix = prng.normal(subkey, shape=A_shape)
    if rank_deficient:
        matrix = matrix.at[:, 1].set(0.0)
        matrix = matrix.at[1, :].set(0.0)

    key, subkey = prng.split(key, 2)
    rhs = prng.normal(subkey, shape=(A_shape[0],))
    key, subkey = prng.split(key, 2)
    damp = prng.normal(subkey, shape=())
    key, subkey = prng.split(key, num=2)
    dsol = prng.normal(subkey, shape=(A_shape[1],))

    def lstsq_linalg_solve(a, b, dmp):
        eye = np.eye(len(a.T))
        return linalg.solve(dmp**2 * eye + a.T @ a, a.T @ b)

    expected, expected_vjp = func.vjp(lstsq_linalg_solve, matrix, rhs, damp)
    dmatrix1, drhs1, ddmp1 = expected_vjp(dsol)

    def vecmat(vector, p_as_list):
        [p] = p_as_list
        return p.T @ vector

    @func.jit
    def lstsq_matfree(a, b, dmp):
        lsmr = lstsq.lsmr(
            atol=1e-10, btol=1e-10, ctol=1e-10, is_full_rank=not rank_deficient
        )
        sol, _ = lsmr(vecmat, a, b, damp=dmp)
        return sol

    received, received_vjp = func.vjp(lstsq_matfree, rhs, [matrix], damp)
    drhs2, [dmatrix2], ddmp2 = received_vjp(dsol)  # mind the order of rhs & matrix

    test_util.assert_allclose(received, expected)
    test_util.assert_allclose(drhs1, drhs2)
    test_util.assert_allclose(ddmp1, ddmp2)
    test_util.assert_allclose(dmatrix1, dmatrix2)


@testing.parametrize_with_cases("A_shape", cases=".", prefix="case_A_shape_")
@testing.filterwarnings("ignore: overflow encountered in")  # SciPy LSMR warns...
def test_output_matches_original_scipy_lsmr(A_shape: tuple):
    """Assert that the implementation of scipy's LSMR is matched exactly."""
    import numpy as onp  # noqa: ICN001
    import scipy.sparse.linalg

    # Scipy uses double precision, so we emulate this behaviour
    config.update("jax_enable_x64", True)

    key = prng.prng_key(1)
    key, subkey = prng.split(key, 2)
    matrix = prng.normal(subkey, shape=A_shape)
    key, subkey = prng.split(key, 2)
    rhs = prng.normal(subkey, shape=(A_shape[0],))
    key, subkey = prng.split(key, num=2)
    x0 = prng.normal(subkey, shape=(A_shape[1],))
    key, subkey = prng.split(key, num=2)
    damp = (prng.uniform(subkey, shape=())) ** 2

    # Our code
    lsmr = lstsq.lsmr(atol=1e-5, btol=1e-5, ctol=1e-5)
    sol, _ = lsmr(lambda v: matrix.T @ v, rhs, damp=damp, x0=x0)

    # Original NumPy
    matrix = onp.asarray(matrix)
    rhs = onp.asarray(rhs)
    x0 = onp.asarray(x0)
    damp = onp.asarray(damp)
    sol2, *_ = scipy.sparse.linalg.lsmr(
        matrix, rhs, atol=1e-5, btol=1e-5, conlim=1e5, damp=damp, x0=x0
    )

    assert np.allclose(sol, np.asarray(sol2))

    # Scipy uses double precision, so we emulate this behaviour
    config.update("jax_enable_x64", False)
