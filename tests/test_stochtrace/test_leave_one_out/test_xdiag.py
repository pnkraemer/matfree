"""Tests for leave_one_out_xdiag."""

from matfree import stochtrace, test_util
from matfree.backend import func, linalg, np, prng, testing

from .conftest import exp_eigvals, step_eigvals


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
    A = np.tril(np.ones((n, n))).astype(dtype_op)
    expected = {"fx": linalg.diagonal(A)}

    def matvec(v, A):
        return {"fx": A @ v["fx"]}

    params = (A,)
    x_like = {"fx": np.ones(n, dtype=dtype_sample)}
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
    test_util.assert_allclose(received["fx"], expected["fx"])


def cases_experiments_exp():
    """Eigenvalues that decay rapidly."""
    eigvals = exp_eigvals(1_000)
    num_samples = 35
    max_rel_err = 1e-2
    return eigvals, num_samples, max_rel_err


def cases_experiments_step():
    """Eigenvalues that are flat with a sudden drop."""
    eigvals = step_eigvals(1_000)
    num_samples = 60
    max_rel_err = 5e-2
    return eigvals, num_samples, max_rel_err


@testing.parametrize("dtype", [float, complex])
@testing.parametrize_with_cases(
    "eigvals, num_samples, max_rel_err", cases=".", prefix="cases_experiments_"
)
def test_xdiag_reproduce_experiments(eigvals, num_samples, max_rel_err, dtype):
    """Assert that the experiments from Fig 16.1 of Ethan Epperly's thesis are reproduced accurately."""
    rdtype = np.abs(dtype(0)).dtype
    n = len(eigvals)
    num_rep = 10
    key = prng.prng_key(50)
    key_eigvecs, key = prng.split(key)
    eigvals = eigvals.astype(rdtype)
    A = test_util.hermitian_matrix_from_eigenvalues(eigvals, key_eigvecs, dtype=dtype)
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
