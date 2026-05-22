"""Tests for leave_one_out_xnystrace."""

from matfree import stochtrace, test_util
from matfree.backend import func, linalg, np, prng, testing


@testing.parametrize("n", [10, 20])
def test_xnystrace_error_num_samples_more_than_dimension(n):
    """Assert that a ValueError is raised when the number of samples is greater than the dimension of the matrix."""
    key = prng.prng_key(1)
    A = np.eye(n)

    def matvec(v, A):
        return A @ v

    integrand = stochtrace.leave_one_out_xnystrace()

    sampler = stochtrace.sampler_normal(np.ones(n), num=n + 1)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    with testing.raises(
        ValueError,
        match=f"Number of samples num={n + 1} exceeds the acceptable range. Expected: 1 <= num <= {n}.",
    ):
        estimate(matvec, key, A)


@testing.parametrize("apply_resphering", [True, False])
@testing.parametrize("dtype", [float, complex])
def test_xnystrace_fast_spectral_decay(apply_resphering, dtype):
    """Assert that the trace of a matrix with fast spectral decay is estimated accurately.

    Reproduces the results of the experiment 'exp' from the XTrace paper.
    """
    rdtype = np.abs(dtype(0)).dtype
    n = 1000
    num_samples = 50
    num_rep = 10
    key = prng.prng_key(1)
    key_mat, key = prng.split(key)
    U = linalg.qr_reduced(prng.normal(key_mat, shape=(n, n), dtype=dtype))[0]
    d = 0.7 ** np.arange(n).astype(rdtype)
    expected = np.sum(d).astype(dtype)

    if apply_resphering:
        sampler = stochtrace.sampler_normal(np.ones(n, dtype=dtype), num=num_samples)
    else:
        sampler = stochtrace.sampler_rademacher(
            np.ones(n, dtype=dtype), num=num_samples
        )
    integrand = stochtrace.leave_one_out_xnystrace(apply_resphering=apply_resphering)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    def matvec(v, d, U):
        return U @ (d * (U.T.conj() @ v))

    key_ests = prng.split(key, num_rep)
    received = func.vmap(lambda key: estimate(matvec, key, d, U))(key_ests)
    rel_err = np.abs(received - expected) / np.abs(expected)
    mean_rel_err = np.mean(rel_err)
    assert float(mean_rel_err) < 1e-4


@testing.parametrize("apply_resphering", [True, False])
@testing.parametrize("dtype", [float, complex])
def test_xnystrace_large_spectral_drop(apply_resphering, dtype):
    """Assert that the trace of a matrix with some large eigenvalues and the rest small is estimated accurately.

    Reproduces the results of the experiment 'step' from the XTrace paper.
    """
    rdtype = np.abs(dtype(0)).dtype
    n = 1000
    m = 50
    num_rep = 10
    key = prng.prng_key(4)
    key_mat, key = prng.split(key)
    U = linalg.qr_reduced(prng.normal(key_mat, shape=(n, n), dtype=dtype))[0]
    d = np.concatenate([np.ones(m, dtype=rdtype), np.ones(n - m, dtype=rdtype) * 1e-3])
    expected = np.sum(d).astype(dtype)

    if apply_resphering:
        sampler = stochtrace.sampler_sphere(np.ones(n, dtype=dtype), num=2 * m + 10)
    else:
        sampler = stochtrace.sampler_rademacher(np.ones(n, dtype=dtype), num=2 * m + 10)
    integrand = stochtrace.leave_one_out_xnystrace(apply_resphering=apply_resphering)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    def matvec(v, d, U):
        return U @ (d * (U.T.conj() @ v))

    key_ests = prng.split(key, num_rep)
    received = func.vmap(lambda key: estimate(matvec, key, d, U))(key_ests)
    rel_err = np.abs(received - expected) / np.abs(expected)
    mean_rel_err = np.mean(rel_err)
    assert float(mean_rel_err) < 1e-3


@testing.parametrize("n", [10, 20])
@testing.parametrize(
    "dtype_op, dtype_sample",
    [(float, float), (complex, complex), (complex, float), (float, complex)],
)
def test_xnystrace_exact_when_num_samples_equals_dimension(n, dtype_op, dtype_sample):
    """Assert that the method computes the trace exactly when the number of samples equals the dimension of the matrix."""
    key = prng.prng_key(1)
    A = prng.normal(key, shape=(n, n), dtype=dtype_op)
    A = A @ A.T.conj() + np.eye(n, dtype=dtype_op) * 1e-6
    expected = linalg.trace(A)

    def matvec(v, A):
        return A @ v

    sampler = stochtrace.sampler_normal(np.ones(n, dtype=dtype_sample), num=n)
    integrand = stochtrace.leave_one_out_xnystrace()
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    test_util.assert_allclose(estimate(matvec, key, A), expected)


def test_xnystrace_pytrees_supported():
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

    integrand = stochtrace.leave_one_out_xnystrace()
    sampler = stochtrace.sampler_normal(
        {"fx": np.ones(n1), "fy": np.ones(n2)}, num=n1 + n2 - 1
    )
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    received = estimate(matvec, key_est, A, B)
    expected = linalg.trace(A) + linalg.trace(B)
    assert np.allclose(received, expected, rtol=1e-2)
