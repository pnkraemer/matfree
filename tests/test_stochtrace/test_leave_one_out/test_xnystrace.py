"""Tests for leave_one_out_xnystrace."""

from matfree import stochtrace, test_util
from matfree.backend import func, linalg, np, prng, testing


def test_xnystrace_kwargs_customizable():
    """Assert that the XNysTrace method supports customizable kwargs."""
    n = 10
    num_samples = 5
    key_mat, key = prng.split(prng.prng_key(1), 2)
    A = prng.normal(key_mat, shape=(n, n))
    A = A @ A.T.conj() + np.eye(n) * 1e-6

    def matvec(v, A):
        return A @ v

    nystrom_eigh = stochtrace.nystrom_eigh()
    nystrom_shifted_cholesky = stochtrace.nystrom_shifted_cholesky()
    qr_r = linalg.qr_r

    def qr_r_cholesky(X):
        return linalg.cholesky(X @ X.T.conj()).T.conj()

    make_integrand = stochtrace.leave_one_out_xnystrace
    integrand = make_integrand()
    integrand_eigh = make_integrand(nystrom=nystrom_eigh)
    integrand_shifted_cholesky = make_integrand(nystrom=nystrom_shifted_cholesky)
    integrand_qr_r = make_integrand(qr_r=qr_r)
    integrand_qr_r_cholesky = make_integrand(qr_r=qr_r_cholesky)

    sampler = stochtrace.sampler_normal(np.ones(n), num=num_samples)

    make_estimator = stochtrace.estimator_leave_one_out
    est = make_estimator(integrand, sampler)
    est_eigh = make_estimator(integrand_eigh, sampler)
    est_shifted_cholesky = make_estimator(integrand_shifted_cholesky, sampler)
    est_qr_r = make_estimator(integrand_qr_r, sampler)
    est_qr_r_cholesky = make_estimator(integrand_qr_r_cholesky, sampler)

    est_default = est(matvec, key, A)
    assert est_eigh(matvec, key, A) == est_default
    assert est_qr_r(matvec, key, A) == est_default
    assert est_shifted_cholesky(matvec, key, A) != est_default
    assert est_qr_r_cholesky(matvec, key, A) != est_default


@testing.parametrize("n", [10, 20])
def test_xnystrace_error_num_samples_more_than_dimension(n):
    """Assert that num_samples greater than the dimension raises a ValueError."""
    key = prng.prng_key(1)
    A = np.eye(n)

    def matvec(v, A):
        return A @ v

    integrand = stochtrace.leave_one_out_xnystrace()

    sampler = stochtrace.sampler_normal(np.ones(n), num=n + 1)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    message = f"Number of samples num={n + 1} exceeds the acceptable range."
    message = f"{message} Expected: 1 <= num <= {n}."
    with testing.raises(ValueError, match=message):
        estimate(matvec, key, A)


@testing.parametrize(
    "nystrom", [stochtrace.nystrom_eigh(), stochtrace.nystrom_shifted_cholesky()]
)
@testing.parametrize("n", [10, 20])
@testing.parametrize(
    "dtype_op, dtype_sample",
    [(float, float), (complex, complex), (complex, float), (float, complex)],
)
def cases_exact_num_samples_equals_dimension(nystrom, n, dtype_op, dtype_sample):
    """Exact trace when num_samples equals the operator's dimension.

    Uses two differently-sized pytree blocks, also exercising heterogeneous
    pytree support.
    """
    n1 = max(1, n // 3)
    n2 = n - n1
    key1, key2 = prng.split(prng.prng_key(1), 2)
    A1 = prng.normal(key1, shape=(n1, n1), dtype=dtype_op)
    A1 = A1 @ A1.T.conj() + np.eye(n1, dtype=dtype_op) * 1e-6
    A2 = prng.normal(key2, shape=(n2, n2), dtype=dtype_op)
    A2 = A2 @ A2.T.conj() + np.eye(n2, dtype=dtype_op) * 1e-6
    expected = (linalg.trace(A1) + linalg.trace(A2)).real

    def matvec(v, A1, A2):
        return {"fx": A1 @ v["fx"], "fy": A2 @ v["fy"]}

    params = (A1, A2)
    x_like = {
        "fx": np.ones(n1, dtype=dtype_sample),
        "fy": np.ones(n2, dtype=dtype_sample),
    }
    sampler = stochtrace.sampler_normal(x_like, num=n)
    integrand = stochtrace.leave_one_out_xnystrace(nystrom=nystrom)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    return matvec, params, estimate, expected


@testing.parametrize("n, num_samples", [(50, 10), (100, 30)])
@testing.parametrize("dtype", [float, complex])
def cases_exact_nystrom_eigh_num_samples_more_than_rank(n, num_samples, dtype):
    """Hermitian low-rank matrix; only nystrom_eigh recovers the trace exactly here."""
    rdtype = np.abs(dtype(0)).dtype
    rank = num_samples - 5
    key_eigvals, key_eigvecs = prng.split(prng.prng_key(1), 2)
    eigvals_nz = (
        linalg.abs2(prng.normal(key_eigvals, shape=(rank,), dtype=rdtype)) + 1e-6
    )
    d = np.concatenate([eigvals_nz, np.zeros(n - rank, dtype=rdtype)]).real
    A = test_util.hermitian_matrix_from_eigenvalues(d, key_eigvecs, dtype=dtype)
    expected = linalg.trace(A).real

    def matvec(v, A):
        return {"fx": A @ v["fx"]}

    params = (A,)
    x_like = {"fx": np.ones(n, dtype=dtype)}
    sampler = stochtrace.sampler_normal(x_like, num=num_samples)
    integrand = stochtrace.leave_one_out_xnystrace(nystrom=stochtrace.nystrom_eigh())
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    return matvec, params, estimate, expected


@testing.parametrize_with_cases(
    "matvec, params, estimate, expected", cases=".", prefix="cases_exact_"
)
def test_xnystrace_exact(matvec, params, estimate, expected):
    """Assert exact trace computation when num_samples is large enough."""
    key = prng.prng_key(1)
    received = estimate(matvec, key, *params)
    test_util.assert_allclose(received, expected)


@testing.parametrize("apply_resphering", [True, False])
def cases_experiments_exp(apply_resphering):
    """Hermitian matrix with eigenvalues that decay rapidly."""
    sampler_factory = (
        stochtrace.sampler_normal if apply_resphering else stochtrace.sampler_signs
    )
    return (
        test_util.hermitian_matrix_eigvals_decaying,
        50,
        1e-4,
        apply_resphering,
        sampler_factory,
        prng.prng_key(1),
    )


@testing.parametrize("apply_resphering", [True, False])
def cases_experiments_step(apply_resphering):
    """Hermitian matrix with eigenvalues that are flat with a sudden drop."""
    sampler_factory = (
        stochtrace.sampler_sphere if apply_resphering else stochtrace.sampler_signs
    )
    return (
        test_util.hermitian_matrix_eigvals_step,
        110,
        1e-3,
        apply_resphering,
        sampler_factory,
        prng.prng_key(4),
    )


@testing.parametrize(
    "nystrom", [stochtrace.nystrom_eigh(), stochtrace.nystrom_shifted_cholesky()]
)
@testing.parametrize("dtype", [float, complex])
@testing.parametrize_with_cases(
    "make_A, num_samples, max_rel_err, apply_resphering, sampler_factory, key",
    cases=".",
    prefix="cases_experiments_",
)
def test_xnystrace_reproduce_experiments(
    make_A,
    num_samples,
    max_rel_err,
    apply_resphering,
    sampler_factory,
    key,
    dtype,
    nystrom,
):
    """Assert that the experiments from the XTrace paper are reproduced accurately."""
    n = 1_000
    num_rep = 10
    key_mat, key = prng.split(key)
    A = make_A(n, key_mat, dtype=dtype)
    expected = linalg.trace(A).real

    sampler = sampler_factory(np.ones(n, dtype=dtype), num=num_samples)
    integrand = stochtrace.leave_one_out_xnystrace(
        nystrom=nystrom, apply_resphering=apply_resphering
    )
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    def matvec(v, A):
        return A @ v

    key_ests = prng.split(key, num_rep)
    received = func.vmap(lambda key: estimate(matvec, key, A))(key_ests)
    rel_err = np.abs(received - expected) / np.abs(expected)
    mean_rel_err = np.mean(rel_err)
    assert float(mean_rel_err) < max_rel_err
