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
@testing.parametrize("apply_resphering", [True, False])
@testing.parametrize("dtype", [float, complex])
def test_xnystrace_fast_spectral_decay(nystrom, apply_resphering, dtype):
    """Assert that a trace with fast spectral decay is estimated accurately.

    Reproduces the results of the experiment 'exp' from the XTrace paper.
    """
    n = 1_000
    num_samples = 50
    num_rep = 10
    key = prng.prng_key(1)
    key_mat, key = prng.split(key)
    A = test_util.hermitian_matrix_eigvals_decaying(n, key_mat, dtype=dtype)
    expected = linalg.trace(A).real

    if apply_resphering:
        sampler = stochtrace.sampler_normal(np.ones(n, dtype=dtype), num=num_samples)
    else:
        sampler = stochtrace.sampler_signs(np.ones(n, dtype=dtype), num=num_samples)
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
    assert float(mean_rel_err) < 1e-4


@testing.parametrize(
    "nystrom", [stochtrace.nystrom_eigh(), stochtrace.nystrom_shifted_cholesky()]
)
@testing.parametrize("apply_resphering", [True, False])
@testing.parametrize("dtype", [float, complex])
def test_xnystrace_large_spectral_drop(nystrom, apply_resphering, dtype):
    """Assert that a trace with a large spectral drop is estimated accurately.

    Reproduces the results of the experiment 'step' from the XTrace paper.
    """
    n = 1_000
    m = 50
    num_rep = 10
    key = prng.prng_key(4)
    key_mat, key = prng.split(key)
    A = test_util.hermitian_matrix_eigvals_step(n, key_mat, dtype=dtype)
    expected = linalg.trace(A).real

    if apply_resphering:
        sampler = stochtrace.sampler_sphere(np.ones(n, dtype=dtype), num=2 * m + 10)
    else:
        sampler = stochtrace.sampler_signs(np.ones(n, dtype=dtype), num=2 * m + 10)
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
    assert float(mean_rel_err) < 1e-3


@testing.parametrize(
    "nystrom", [stochtrace.nystrom_eigh(), stochtrace.nystrom_shifted_cholesky()]
)
@testing.parametrize("n", [10, 20])
@testing.parametrize(
    "dtype_op, dtype_sample",
    [(float, float), (complex, complex), (complex, float), (float, complex)],
)
def test_xnystrace_exact_when_num_samples_equals_dimension(
    n, nystrom, dtype_op, dtype_sample
):
    """Assert exact trace computation when num_samples equals the dimension."""
    key = prng.prng_key(1)
    A = prng.normal(key, shape=(n, n), dtype=dtype_op)
    A = A @ A.T.conj() + np.eye(n, dtype=dtype_op) * 1e-6
    expected = linalg.trace(A).real

    def matvec(v, A):
        return A @ v

    sampler = stochtrace.sampler_normal(np.ones(n, dtype=dtype_sample), num=n)
    integrand = stochtrace.leave_one_out_xnystrace(nystrom=nystrom)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    test_util.assert_allclose(estimate(matvec, key, A), expected)


@testing.parametrize(
    "nystrom", [stochtrace.nystrom_eigh(), stochtrace.nystrom_shifted_cholesky()]
)
def test_xnystrace_pytrees_supported(nystrom):
    """Assert that the XNysTrace method supports pytrees."""
    n1 = 100
    n2 = 50
    key_mat1, key_mat2, key_est = prng.split(prng.prng_key(1), 3)
    A = prng.normal(key_mat1, shape=(n1, n1))
    A = A @ A.T.conj() + np.eye(n1, dtype=A.dtype) * 1e-6
    B = prng.normal(key_mat2, shape=(n2, n2))
    B = B @ B.T.conj() + np.eye(n2, dtype=B.dtype) * 1e-6

    def matvec(v, A, B):
        return {"fx": A @ v["fx"], "fy": B @ v["fy"]}

    integrand = stochtrace.leave_one_out_xnystrace(nystrom=nystrom)
    x_like = {"fx": np.ones(n1), "fy": np.ones(n2)}
    sampler = stochtrace.sampler_normal(x_like, num=n1 + n2 - 1)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    received = estimate(matvec, key_est, A, B)
    expected = linalg.trace(A) + linalg.trace(B)
    assert np.allclose(received, expected, rtol=1e-2)
