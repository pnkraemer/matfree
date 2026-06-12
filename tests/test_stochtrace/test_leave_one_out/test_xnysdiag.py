"""Tests for leave_one_out_xnysdiag."""

from matfree import stochtrace, test_util
from matfree.backend import func, linalg, np, prng, testing


def test_xnysdiag_kwargs_customizable():
    """Assert that the XNysDiag method supports customizable kwargs."""
    n = 10
    num_samples = 5
    rank = 3
    key_eigvals, key_eigvecs, key = prng.split(prng.prng_key(1), 3)
    # Low-rank + small regularisation: shifted Cholesky and eigh give different results
    nz_eigvals = linalg.abs2(prng.normal(key_eigvals, shape=(rank,)))
    z_eigvals = np.zeros(n - rank)
    d = np.concatenate([nz_eigvals, z_eigvals]) + 1e-3
    A = test_util.hermitian_matrix_from_eigenvalues(d, key_eigvecs)

    def matvec(v, A):
        return A @ v

    nystrom_eigh = stochtrace.nystrom_eigh()
    nystrom_shifted_cholesky = stochtrace.nystrom_shifted_cholesky(shift=1.0)

    make_integrand = stochtrace.leave_one_out_xnysdiag
    integrand_default = make_integrand()
    integrand_eigh = make_integrand(nystrom=nystrom_eigh)
    integrand_shifted_cholesky = make_integrand(nystrom=nystrom_shifted_cholesky)

    sampler = stochtrace.sampler_normal(np.ones(n), num=num_samples)
    make_estimator = stochtrace.estimator_leave_one_out

    est_default = make_estimator(integrand_default, sampler)
    est_eigh = make_estimator(integrand_eigh, sampler)
    est_shifted_cholesky = make_estimator(integrand_shifted_cholesky, sampler)

    result_default = est_default(matvec, key, A)
    assert np.allclose(est_eigh(matvec, key, A), result_default)
    assert not np.allclose(est_shifted_cholesky(matvec, key, A), result_default)


@testing.parametrize("n", [10, 20])
def test_xnysdiag_error_num_samples_more_than_dimension(n):
    """Assert that num_samples greater than the dimension raises a ValueError."""
    key = prng.prng_key(1)
    A = np.eye(n)

    def matvec(v, A):
        return A @ v

    integrand = stochtrace.leave_one_out_xnysdiag()
    sampler = stochtrace.sampler_normal(np.ones(n), num=n + 1)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    message = f"Number of samples num={n + 1} exceeds the acceptable range."
    message = f"{message} Expected: 1 <= num <= {n}."
    with testing.raises(ValueError, match=message):
        estimate(matvec, key, A)


@testing.parametrize("n", [10, 20])
@testing.parametrize(
    "dtype_op, dtype_sample",
    [(float, float), (complex, complex), (complex, float), (float, complex)],
)
def cases_exact_num_samples_equals_dimension(nystrom, n, dtype_op, dtype_sample):
    key_eigvals, key_eigvecs = prng.split(prng.prng_key(1), 2)
    d = linalg.abs2(prng.normal(key_eigvals, shape=(n,), dtype=dtype_op)) + 1e-6
    A = test_util.hermitian_matrix_from_eigenvalues(d, key_eigvecs, dtype=dtype_op)
    expected = {"fx": linalg.diagonal(A).real}

    def matvec(v, A):
        return {"fx": A @ v["fx"]}

    params = (A,)
    x_like = {"fx": np.ones(n, dtype=dtype_sample)}
    sampler = stochtrace.sampler_normal(x_like, num=n)
    integrand = stochtrace.leave_one_out_xnysdiag(nystrom=nystrom)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    return matvec, params, estimate, expected


@testing.parametrize("n, num_samples", [(50, 10), (100, 30)])
@testing.parametrize("dtype", [float, complex])
def cases_exact_nystrom_eigh_num_samples_more_than_rank(n, num_samples, dtype):
    rdtype = np.abs(dtype(0)).dtype
    rank = num_samples - 5
    key_eigvals, key_eigvecs = prng.split(prng.prng_key(1), 2)
    eigvals_nz = (
        linalg.abs2(prng.normal(key_eigvals, shape=(rank,), dtype=rdtype)) + 1e-6
    )
    d = np.concatenate([eigvals_nz, np.zeros(n - rank, dtype=rdtype)]).real
    A = test_util.hermitian_matrix_from_eigenvalues(d, key_eigvecs, dtype=dtype)
    expected = {"fx": linalg.diagonal(A).real}

    def matvec(v, A):
        return {"fx": A @ v["fx"]}

    params = (A,)
    x_like = {"fx": np.ones(n, dtype=dtype)}
    sampler = stochtrace.sampler_normal(x_like, num=num_samples)
    integrand = stochtrace.leave_one_out_xnysdiag(nystrom=stochtrace.nystrom_eigh())
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    return matvec, params, estimate, expected


@testing.parametrize_with_cases(
    "matvec, params, estimate, expected", cases=".", prefix="cases_exact_"
)
def test_xnysdiag_exact(matvec, params, estimate, expected):
    """Assert exact diagonal computation when num_samples is large enough."""
    key = prng.prng_key(7)
    received = estimate(matvec, key, *params)
    test_util.assert_allclose(received["fx"], expected["fx"])


def cases_experiments_exp():
    """Hermitian matrix with eigenvalues that decay rapidly."""
    return test_util.hermitian_matrix_eigvals_decaying, 50, 1e-3


def cases_experiments_step():
    """Hermitian matrix with eigenvalues that are flat with a sudden drop."""
    return test_util.hermitian_matrix_eigvals_step, 110, 5e-2


@testing.parametrize_with_cases(
    "make_A, num_samples, max_rel_err", cases=".", prefix="cases_experiments_"
)
@testing.parametrize("dtype", [float, complex])
def test_xnysdiag_reproduce_experiments(
    nystrom, make_A, num_samples, max_rel_err, dtype
):
    """Assert that the experiments from Fig 16.1 of Ethan Epperly's thesis are reproduced accurately."""
    n = 1_000
    num_rep = 10
    key = prng.prng_key(50)
    key_eigvecs, key = prng.split(key)
    A = make_A(n, key_eigvecs, dtype=dtype)
    expected = linalg.diagonal(A).real

    x_like = {"fx": np.ones(n, dtype=dtype)}
    sampler = stochtrace.sampler_normal(x_like, num=num_samples)
    integrand = stochtrace.leave_one_out_xnysdiag(nystrom=nystrom)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    def matvec(v, A):
        return {"fx": A @ v["fx"]}

    key_ests = prng.split(key, num_rep)
    received = func.vmap(lambda key: estimate(matvec, key, A))(key_ests)["fx"]
    max_abs_err = np.array_max(np.abs(received - expected), axis=1)
    norm_expected = np.array_max(np.abs(expected))
    median_max_rel_err = np.median(max_abs_err / norm_expected)
    assert float(median_max_rel_err) < max_rel_err


@testing.parametrize("dtype", [float])
def test_xnysdiag_sum_equals_xnystrace(nystrom, dtype):
    """Assert that summing XNysDiag estimates over entries gives the XNysTrace estimate.

    This tests the mathematical identity: sum_j diag_loo[j,i] = tr_loo[i] for every i.
    Both estimators use the same operator, key, and sampler so the results should be
    equal up to floating point.
    """
    n = 100
    num_samples = 20
    key_eigvals, key_eigvecs, key_est = prng.split(prng.prng_key(1), 3)
    d = linalg.abs2(prng.normal(key_eigvals, shape=(n,), dtype=dtype)) + 1e-6
    A = test_util.hermitian_matrix_from_eigenvalues(d, key_eigvecs, dtype=dtype)

    def matvec(v, A):
        return A @ v

    sampler = stochtrace.sampler_signs(np.ones(n), num=num_samples)

    integrand_diag = stochtrace.leave_one_out_xnysdiag(nystrom=nystrom)
    estimate_diag = stochtrace.estimator_leave_one_out(integrand_diag, sampler)
    diag_result = estimate_diag(matvec, key_est, A)

    integrand_trace = stochtrace.leave_one_out_xnystrace(
        nystrom=nystrom, apply_resphering=False
    )
    estimate_trace = stochtrace.estimator_leave_one_out(integrand_trace, sampler)
    trace_result = estimate_trace(matvec, key_est, A)

    trace_via_diag = np.sum(diag_result)
    test_util.assert_allclose(trace_via_diag, trace_result)
