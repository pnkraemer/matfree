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
def test_xdiag_exact_when_num_samples_more_than_half_dimension(
    n, num_samples, dtype_op, dtype_sample
):
    """Assert exact diagonal computation when num_samples is large."""
    key = prng.prng_key(1)
    A = np.tril(np.ones((n, n), dtype=dtype_op))
    expected = linalg.diagonal(A)

    def matvec(v, A):
        return A @ v

    sampler = stochtrace.sampler_normal(np.ones(n, dtype=dtype_sample), num=num_samples)
    integrand = stochtrace.leave_one_out_xdiag()
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    test_util.assert_allclose(estimate(matvec, key, A), expected)


@testing.parametrize("n, rank", [(50, 10), (100, 30)])
@testing.parametrize("dtype", [float, complex])
def test_xdiag_low_rank_operator(n, rank, dtype):
    """Assert that the diagonal of a low-rank operator is computed exactly."""
    key = prng.prng_key(5)
    key_mat1, key_mat2, key = prng.split(key, 3)
    A = prng.normal(key_mat1, shape=(n, rank), dtype=dtype)
    B = prng.normal(key_mat2, shape=(rank, n), dtype=dtype)
    expected = linalg.diagonal(A @ B)

    def matvec(v, A, B):
        return A @ (B @ v)

    sampler = stochtrace.sampler_normal(np.ones(n, dtype=dtype), num=rank + 1)
    integrand = stochtrace.leave_one_out_xdiag()
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    test_util.assert_allclose(estimate(matvec, key, A, B), expected)


@testing.parametrize("dtype", [float, complex])
def test_xdiag_fast_spectral_decay(dtype):
    """Assert that a diagonal with fast spectral decay is estimated accurately.

    Reproduces the setup of the experiment 'exp' from Fig 16.1 of Ethan Epperly's thesis.
    """
    rdtype = np.abs(dtype(0)).dtype
    n = 1000
    num_rep = 10
    key = prng.prng_key(1)
    key_eigvecs, key = prng.split(key)
    d = 0.7 ** np.arange(n).astype(rdtype)
    A = test_util.hermitian_matrix_from_eigenvalues(d, key_eigvecs, dtype=dtype)
    expected = linalg.diagonal(A).real

    sampler = stochtrace.sampler_signs(np.ones(n, dtype=dtype), num=35)
    integrand = stochtrace.leave_one_out_xdiag()
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    def matvec(v, A):
        return A @ v

    key_ests = prng.split(key, num_rep)
    received = func.vmap(lambda key: estimate(matvec, key, A))(key_ests)
    max_abs_err = np.array_max(np.abs(received - expected), axis=1)
    max_rel_err = max_abs_err / np.array_max(np.abs(expected))
    assert float(np.median(max_rel_err)) < 1e-2


@testing.parametrize("dtype", [float, complex])
def test_xdiag_large_spectral_drop(dtype):
    """Assert that a diagonal with a large spectral drop is estimated accurately.

    Reproduces the setup of the experiment 'step' from Fig 16.1 of Ethan Epperly's thesis.
    """
    rdtype = np.abs(dtype(0)).dtype
    n = 1000
    m = 50
    num_rep = 10
    key = prng.prng_key(4)
    key_eigvecs, key = prng.split(key)
    large_eigenvalues = np.ones(m, dtype=rdtype)
    small_eigenvalues = np.ones(n - m, dtype=rdtype) * 1e-3
    d = np.concatenate([large_eigenvalues, small_eigenvalues])
    A = test_util.hermitian_matrix_from_eigenvalues(d, key_eigvecs, dtype=dtype)
    expected = linalg.diagonal(A).real

    sampler = stochtrace.sampler_signs(np.ones(n, dtype=dtype), num=m + 10)
    integrand = stochtrace.leave_one_out_xdiag()
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    def matvec(v, A):
        return A @ v

    key_ests = prng.split(key, num_rep)
    received = func.vmap(lambda key: estimate(matvec, key, A))(key_ests)
    max_abs_err = np.array_max(np.abs(received - expected), axis=1)
    norm_expected = np.array_max(np.abs(expected))
    median_max_rel_err = np.median(max_abs_err / norm_expected)
    assert float(median_max_rel_err) < 5e-2


def test_xdiag_pytrees_supported():
    """Assert that the XDiag algorithm supports pytrees."""
    n1 = 100
    n2 = 50
    key_mat1, key_mat2, key_est = prng.split(prng.prng_key(1), 3)
    A = prng.normal(key_mat1, shape=(n1, n1))
    B = prng.normal(key_mat2, shape=(n2, n2))

    def matvec(v, A, B):
        return {"fx": A @ v["fx"], "fy": B @ v["fy"]}

    integrand = stochtrace.leave_one_out_xdiag()
    x_like = {"fx": np.ones(n1), "fy": np.ones(n2)}
    sampler = stochtrace.sampler_sphere(x_like, num=n1 + n2 - 1)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    received = estimate(matvec, key_est, A, B)
    expected = {"fx": linalg.diagonal(A), "fy": linalg.diagonal(B)}
    assert isinstance(received, dict)
    assert set(received.keys()) == {"fx", "fy"}
    assert np.allclose(received["fx"], expected["fx"], rtol=1e-4)
    assert np.allclose(received["fy"], expected["fy"], rtol=1e-4)
