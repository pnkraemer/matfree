"""Tests for leave_one_out_xtrace."""

from matfree import stochtrace, test_util
from matfree.backend import config, func, linalg, np, prng, testing

config.update("jax_enable_x64", True)


@testing.parametrize("n", [10, 20])
def test_error_num_samples_more_than_dimension(n):
    """Assert that a ValueError is raised when the number of samples is greater than the dimension of the matrix."""
    key = prng.prng_key(1)
    A = np.eye(n)

    def matvec(v, A):
        return A @ v

    integrand = stochtrace.leave_one_out_xtrace()

    sampler = stochtrace.sampler_normal(np.ones(n), num=n + 1)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    with testing.raises(
        ValueError,
        match=f"Number of samples num={n + 1} exceeds the acceptable range. Expected: 1 <= num <= {n}.",
    ):
        estimate(matvec, key, A)


@testing.parametrize("resphere", [True, False])
@testing.parametrize("dtype", [float, complex])
def test_trace_svd_fast_spectral_decay(resphere, dtype):
    """Assert that the trace of a matrix with fast spectral decay is estimated accurately."""
    rdtype = np.abs(dtype(0)).dtype
    n = 1000
    num_rep = 10
    key = prng.prng_key(1)
    key_mat, key = prng.split(key)
    U = linalg.qr_reduced(prng.normal(key_mat, shape=(n, n), dtype=dtype))[0]
    d = 0.7 ** np.arange(n).astype(rdtype)
    expected = np.sum(d).astype(dtype)

    sampler = stochtrace.sampler_normal(np.ones(n, dtype=dtype), num=35)
    integrand = stochtrace.leave_one_out_xtrace(resphere=resphere)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    def matvec(v, d, U):
        return U @ (d * (U.T.conj() @ v))

    key_ests = prng.split(key, num_rep)
    received = func.vmap(lambda key: estimate(matvec, key, d, U))(key_ests)
    rel_err = np.abs(received - expected) / np.abs(expected)
    mean_rel_err = np.mean(rel_err)
    assert float(mean_rel_err) < 1e-5


@testing.parametrize("resphere", [True, False])
@testing.parametrize("dtype", [float, complex])
def test_trace_svd_large_spectral_drop(resphere, dtype):
    """Assert that the trace of a matrix with some large eigenvalues and the rest small is estimated accurately."""
    rdtype = np.abs(dtype(0)).dtype
    n = 1000
    m = 50
    num_rep = 10
    key = prng.prng_key(4)
    key_mat, key = prng.split(key)
    U = linalg.qr_reduced(prng.normal(key_mat, shape=(n, n), dtype=dtype))[0]
    d = np.concatenate([np.ones(m, dtype=rdtype), np.ones(n - m, dtype=rdtype) * 1e-3])
    expected = np.sum(d).astype(dtype)

    sampler = stochtrace.sampler_normal(np.ones(n, dtype=dtype), num=m + 10)
    integrand = stochtrace.leave_one_out_xtrace(resphere=resphere)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    def matvec(v, d, U):
        return U @ (d * (U.T.conj() @ v))

    key_ests = prng.split(key, num_rep)
    received = func.vmap(lambda key: estimate(matvec, key, d, U))(key_ests)
    rel_err = np.abs(received - expected) / np.abs(expected)
    mean_rel_err = np.mean(rel_err)
    assert float(mean_rel_err) < (1e-5 if resphere else 1e-4)


@testing.parametrize("n, rank", [(50, 10), (100, 30)])
@testing.parametrize("dtype", [float, complex])
def test_trace_svd_low_rank_operator(n, rank, dtype):
    """Assert that the trace of an already-low-rank operator is computed exactly."""
    key = prng.prng_key(5)
    key_mat1, key_mat2, key = prng.split(key, 3)
    A = prng.normal(key_mat1, shape=(n, rank), dtype=dtype)
    B = prng.normal(key_mat2, shape=(rank, n), dtype=dtype)
    expected = linalg.trace(A @ B)

    def matvec(v, A, B):
        return A @ (B @ v)

    sampler = stochtrace.sampler_normal(np.ones(n, dtype=dtype), num=rank + 1)
    integrand = stochtrace.leave_one_out_xtrace()
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    test_util.assert_allclose(estimate(matvec, key, A, B), expected)


config.update("jax_enable_x64", False)
