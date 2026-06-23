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


@testing.parametrize("n, num_samples", [(10, 5), (21, 11)])
@testing.parametrize(
    "dtype_op, dtype_sample",
    [(float, float), (complex, complex), (complex, float), (float, complex)],
)
def cases_exact_num_samples_at_least_half_dimension(
    n, num_samples, dtype_op, dtype_sample
):
    """Exact trace when num_samples is at least half the operator's dimension.

    Uses two differently-sized pytree blocks, also exercising heterogeneous
    pytree support.
    """
    n1 = max(1, n // 3)
    n2 = n - n1
    A1 = np.tril(np.ones((n1, n1))).astype(dtype_op)
    A2 = np.tril(np.ones((n2, n2))).astype(dtype_op)
    expected = linalg.trace(A1) + linalg.trace(A2)

    def matvec(v, A1, A2):
        return {"fx": A1 @ v["fx"], "fy": A2 @ v["fy"]}

    params = (A1, A2)
    x_like = {
        "fx": np.ones(n1, dtype=dtype_sample),
        "fy": np.ones(n2, dtype=dtype_sample),
    }
    sampler = stochtrace.sampler_normal(x_like, num=num_samples)
    return matvec, params, sampler, expected


@testing.parametrize("n, rank", [(50, 10), (100, 30)])
@testing.parametrize("dtype", [float, complex])
def cases_exact_num_samples_more_than_rank(n, rank, dtype):
    key_mat1, key_mat2 = prng.split(prng.prng_key(1), 2)
    A = prng.normal(key_mat1, shape=(n, rank), dtype=dtype)
    B = prng.normal(key_mat2, shape=(rank, n), dtype=dtype)
    expected = linalg.trace(A @ B)

    def matvec(v, A, B):
        return {"fx": A @ (B @ v["fx"])}

    params = (A, B)
    x_like = {"fx": np.ones(n, dtype=dtype)}
    sampler = stochtrace.sampler_normal(x_like, num=rank + 1)
    return matvec, params, sampler, expected


@testing.parametrize_with_cases(
    "matvec, params, sampler, expected", cases=".", prefix="cases_exact_"
)
def test_xtrace_exact(matvec, params, sampler, expected):
    """Assert exact trace computation when num_samples is large enough."""
    key = prng.prng_key(1)
    integrand = stochtrace.leave_one_out_xtrace()
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    received = estimate(matvec, key, *params)
    test_util.assert_allclose(received, expected)


@testing.parametrize("apply_resphering", [True, False])
def cases_experiments_exp(apply_resphering):
    """Hermitian matrix with eigenvalues that decay rapidly."""
    return (
        test_util.hermitian_matrix_eigvals_decaying,
        35,
        1e-5,
        apply_resphering,
        prng.prng_key(1),
    )


@testing.parametrize("apply_resphering", [True, False])
def cases_experiments_step(apply_resphering):
    """Hermitian matrix with eigenvalues that are flat with a sudden drop."""
    max_rel_err = 1e-5 if apply_resphering else 1e-4
    return (
        test_util.hermitian_matrix_eigvals_step,
        60,
        max_rel_err,
        apply_resphering,
        prng.prng_key(4),
    )


@testing.parametrize("dtype", [float, complex])
@testing.parametrize_with_cases(
    "make_A, num_samples, max_rel_err, apply_resphering, key",
    cases=".",
    prefix="cases_experiments_",
)
def test_xtrace_reproduce_experiments(
    make_A, num_samples, max_rel_err, apply_resphering, key, dtype
):
    """Assert that the experiments from the XTrace paper are reproduced accurately."""
    n = 1_000
    num_rep = 10
    key_mat, key = prng.split(key)
    A = make_A(n, key_mat, dtype=dtype)
    expected = linalg.trace(A)

    sampler = stochtrace.sampler_normal(np.ones(n, dtype=dtype), num=num_samples)
    integrand = stochtrace.leave_one_out_xtrace(apply_resphering=apply_resphering)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    def matvec(v, A):
        return A @ v

    key_ests = prng.split(key, num_rep)
    received = func.vmap(lambda key: estimate(matvec, key, A))(key_ests)
    rel_err = np.abs(received - expected) / np.abs(expected)
    mean_rel_err = np.mean(rel_err)
    assert float(mean_rel_err) < max_rel_err
