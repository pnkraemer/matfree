"""Tests for leave_one_out_xtrace."""

from matfree import stochtrace, test_util
from matfree.backend import func, linalg, np, prng, testing


@testing.parametrize("dtype", [float, complex])
@testing.parametrize("n, num_samples", [(10, 4), (15, 6)])
def test_qr_leave_one_out_factor_downdate_identity(n, num_samples, dtype):
    """Assert Q_i Q_i^H = Q (I - s_i s_i^H) Q^H for each i."""
    key = prng.prng_key(10)
    Y = prng.normal(key, shape=(n, num_samples), dtype=dtype)
    Q, R = linalg.qr_reduced(Y)
    S = stochtrace._qr_leave_one_out_factor(R)

    for i in range(num_samples):
        Y_i = np.concatenate([Y.T[:i], Y.T[i + 1 :]]).T
        Q_i, _ = linalg.qr_reduced(Y_i)

        s_i = S[:, i]
        Id = np.eye(num_samples, dtype=dtype)
        P_i_downdate = Q @ (Id - linalg.outer(s_i, s_i.conj())) @ Q.T.conj()
        P_i_direct = Q_i @ Q_i.T.conj()

        test_util.assert_allclose(P_i_downdate, P_i_direct)


@testing.parametrize("n", [10, 20])
def test_xtrace_error_num_samples_more_than_dimension(n):
    """Assert that num_samples greater than the dimension raises a ValueError."""
    key = prng.prng_key(1)
    A = np.eye(n)

    def matvec(v, A):
        return A @ v

    integrand = stochtrace.leave_one_out_xtrace()

    sampler = stochtrace.sampler_normal(np.ones(n), num=n + 1)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    message = f"Number of samples num={n + 1} exceeds the acceptable range."
    message = f"{message} Expected: 1 <= num <= {n}."
    with testing.raises(ValueError, match=message):
        estimate(matvec, key, A)


@testing.parametrize("apply_resphering", [True, False])
@testing.parametrize("dtype", [float, complex])
def test_xtrace_fast_spectral_decay(apply_resphering, dtype):
    """Assert that a trace with fast spectral decay is estimated accurately.

    Reproduces the results of the experiment 'exp' from the XTrace paper.
    """
    n = 1_000
    num_rep = 10
    key = prng.prng_key(1)
    key_mat, key = prng.split(key)
    A = test_util.hermitian_matrix_eigvals_decaying(n, key_mat, dtype=dtype)
    expected = linalg.trace(A)

    sampler = stochtrace.sampler_normal(np.ones(n, dtype=dtype), num=35)
    integrand = stochtrace.leave_one_out_xtrace(apply_resphering=apply_resphering)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    def matvec(v, A):
        return A @ v

    key_ests = prng.split(key, num_rep)
    received = func.vmap(lambda key: estimate(matvec, key, A))(key_ests)
    rel_err = np.abs(received - expected) / np.abs(expected)
    mean_rel_err = np.mean(rel_err)
    assert float(mean_rel_err) < 1e-5


@testing.parametrize("apply_resphering", [True, False])
@testing.parametrize("dtype", [float, complex])
def test_xtrace_large_spectral_drop(apply_resphering, dtype):
    """Assert that a trace with a large spectral drop is estimated accurately.

    Reproduces the results of the experiment 'step' from the XTrace paper.
    """
    n = 1_000
    m = 50
    num_rep = 10
    key = prng.prng_key(4)
    key_mat, key = prng.split(key)
    A = test_util.hermitian_matrix_eigvals_step(n, key_mat, dtype=dtype)
    expected = linalg.trace(A)

    sampler = stochtrace.sampler_normal(np.ones(n, dtype=dtype), num=m + 10)
    integrand = stochtrace.leave_one_out_xtrace(apply_resphering=apply_resphering)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    def matvec(v, A):
        return A @ v

    key_ests = prng.split(key, num_rep)
    received = func.vmap(lambda key: estimate(matvec, key, A))(key_ests)
    rel_err = np.abs(received - expected) / np.abs(expected)
    mean_rel_err = np.mean(rel_err)
    assert float(mean_rel_err) < (1e-5 if apply_resphering else 1e-4)


@testing.parametrize("n, rank", [(50, 10), (100, 30)])
@testing.parametrize("dtype", [float, complex])
def test_xtrace_low_rank_operator(n, rank, dtype):
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


@testing.parametrize("n, num_samples", [(10, 5), (21, 11)])
@testing.parametrize(
    "dtype_op, dtype_sample",
    [(float, float), (complex, complex), (complex, float), (float, complex)],
)
def test_xtrace_exact_when_num_samples_more_than_half_dimension(
    n, num_samples, dtype_op, dtype_sample
):
    """Assert exact trace computation when num_samples is large."""
    key = prng.prng_key(1)
    A = np.tril(np.ones((n, n), dtype=dtype_op))
    expected = linalg.trace(A)

    def matvec(v, A):
        return A @ v

    sampler = stochtrace.sampler_normal(np.ones(n, dtype=dtype_sample), num=num_samples)
    integrand = stochtrace.leave_one_out_xtrace()
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    test_util.assert_allclose(estimate(matvec, key, A), expected)


def test_xtrace_pytrees_supported():
    """Assert that the XTrace algorithm supports pytrees."""
    n1 = 100
    n2 = 50
    key_mat1, key_mat2, key_est = prng.split(prng.prng_key(1), 3)
    A = prng.normal(key_mat1, shape=(n1, n1))
    B = prng.normal(key_mat2, shape=(n2, n2))

    def matvec(v, A, B):
        return {"fx": A @ v["fx"], "fy": B @ v["fy"]}

    integrand = stochtrace.leave_one_out_xtrace()
    x_like = {"fx": np.ones(n1), "fy": np.ones(n2)}
    sampler = stochtrace.sampler_normal(x_like, num=n1 + n2 - 1)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    received = estimate(matvec, key_est, A, B)
    expected = linalg.trace(A) + linalg.trace(B)
    assert np.allclose(received, expected, rtol=1e-2)
