"""Tests for leave_one_out_xdiag."""

from matfree import stochtrace, test_util
from matfree.backend import func, linalg, np, prng, testing


@testing.parametrize("n", [10, 20])
def test_xdiag_error_num_samples_more_than_dimension(n):
    """Assert that num_samples greater than the dimension raises a ValueError."""
    key = prng.prng_key(1)
    A = np.eye(n)

    def matvec(v, A):
        return A @ v

    integrand = stochtrace.leave_one_out_xdiag()
    sampler = stochtrace.sampler_normal(np.ones(n), num=n + 1)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    message = f"Number of samples num={n + 1} exceeds the acceptable range."
    message = f"{message} Expected: 1 <= num <= {n}."
    with testing.raises(ValueError, match=message):
        estimate(matvec, key, A)


@testing.parametrize("n, num_samples", [(10, 5), (21, 11)])
@testing.parametrize(
    "dtype_op, dtype_sample",
    [(float, float), (complex, complex), (complex, float), (float, complex)],
)
def cases_exact_num_samples_at_least_half_dimension(
    n, num_samples, dtype_op, dtype_sample
):
    """Exact diagonal when num_samples is at least half the operator's dimension.

    Uses two differently-sized pytree blocks, also exercising heterogeneous
    pytree support.
    """
    n1 = max(1, n // 3)
    n2 = n - n1
    A1 = np.tril(np.ones((n1, n1))).astype(dtype_op)
    A2 = np.tril(np.ones((n2, n2))).astype(dtype_op)
    expected = {"fx": linalg.diagonal(A1), "fy": linalg.diagonal(A2)}

    def matvec(v, A1, A2):
        return {"fx": A1 @ v["fx"], "fy": A2 @ v["fy"]}

    params = (A1, A2)
    x_like = {
        "fx": np.ones(n1, dtype=dtype_sample),
        "fy": np.ones(n2, dtype=dtype_sample),
    }
    sampler = stochtrace.sampler_normal(x_like, num=num_samples)
    return matvec, params, sampler, expected


@testing.parametrize("n, num_samples", [(50, 10), (100, 30)])
@testing.parametrize("dtype", [float, complex])
def cases_exact_num_samples_more_than_rank(n, num_samples, dtype):
    rank = num_samples - 1
    key_mat1, key_mat2 = prng.split(prng.prng_key(1), 2)
    A = prng.normal(key_mat1, shape=(n, rank), dtype=dtype)
    B = prng.normal(key_mat2, shape=(rank, n), dtype=dtype)
    expected = {"fx": linalg.diagonal(A @ B)}

    def matvec(v, A, B):
        return {"fx": A @ (B @ v["fx"])}

    params = (A, B)
    x_like = {"fx": np.ones(n, dtype=dtype)}
    sampler = stochtrace.sampler_normal(x_like, num=num_samples)
    return matvec, params, sampler, expected


@testing.parametrize_with_cases(
    "matvec, params, sampler, expected", cases=".", prefix="cases_exact_"
)
def test_xdiag_exact(matvec, params, sampler, expected):
    """Assert exact diagonal computation when num_samples is large enough."""
    key = prng.prng_key(5)
    integrand = stochtrace.leave_one_out_xdiag()
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    received = estimate(matvec, key, *params)
    for leaf, expected_leaf in expected.items():
        test_util.assert_allclose(received[leaf], expected_leaf)


def cases_experiments_exp():
    """Hermitian matrix with eigenvalues that decay rapidly."""
    return test_util.hermitian_matrix_eigvals_decaying, 35, 1e-2


def cases_experiments_step():
    """Hermitian matrix with eigenvalues that are flat with a sudden drop."""
    return test_util.hermitian_matrix_eigvals_step, 60, 5e-2


@testing.parametrize("dtype", [float, complex])
@testing.parametrize_with_cases(
    "make_A, num_samples, max_rel_err", cases=".", prefix="cases_experiments_"
)
def test_xdiag_reproduce_experiments(make_A, num_samples, max_rel_err, dtype):
    """Assert that the experiments from Fig 16.1 of Ethan Epperly's thesis are reproduced accurately."""
    n = 1_000
    num_rep = 10
    key = prng.prng_key(50)
    key_eigvecs, key = prng.split(key)
    A = make_A(n, key_eigvecs, dtype=dtype)
    expected = linalg.diagonal(A).real

    x_like = {"fx": np.ones(n, dtype=dtype)}
    sampler = stochtrace.sampler_signs(x_like, num=num_samples)
    integrand = stochtrace.leave_one_out_xdiag()
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    def matvec(v, A):
        return {"fx": A @ v["fx"]}

    key_ests = prng.split(key, num_rep)
    received = func.vmap(lambda key: estimate(matvec, key, A))(key_ests)["fx"]
    max_abs_err = np.array_max(np.abs(received - expected), axis=1)
    median_max_rel_err = np.median(max_abs_err / np.array_max(np.abs(expected)))
    assert float(median_max_rel_err) < max_rel_err
