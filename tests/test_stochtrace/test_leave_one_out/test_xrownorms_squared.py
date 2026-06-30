"""Tests for leave_one_out_xrownorms_squared and leave_one_out_xsymrownorms_squared."""

from matfree import stochtrace, test_util
from matfree.backend import config, func, linalg, np, prng, testing
from matfree.backend.typing import Array, Callable, NamedTuple


def _make_integrand(is_normal):
    if is_normal:
        return stochtrace.leave_one_out_xsymrownorms_squared()
    return stochtrace.leave_one_out_xrownorms_squared()


@testing.parametrize("is_normal", [False, True])
@testing.parametrize("n", [10, 20])
def test_xrownorms_squared_error_num_samples_more_than_dimension(n, is_normal):
    """Assert that num_samples greater than the dimension raises a ValueError."""
    key = prng.prng_key(1)
    A = np.eye(n)

    def matvec(v, A):
        return A @ v

    integrand = _make_integrand(is_normal)
    sampler = stochtrace.sampler_signs(np.ones(n), num=n + 1)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    message = f"Number of samples num={n + 1} exceeds the acceptable range."
    message = f"{message} Expected: 1 <= num <= {n}."
    with testing.raises(ValueError, match=message):
        estimate(matvec, key, A)


@testing.parametrize("is_normal", [False, True])
@testing.parametrize("n", [12, 30])
def cases_exact_num_samples_at_least_dimension(n, is_normal):
    """Exact squared row norms when num_samples is at least n/2 (normal) or n/3 (general).

    Uses two differently-sized pytree blocks, also exercising heterogeneous
    pytree support.
    """
    n1 = max(1, n // 3)
    n2 = n - n1
    num_samples = n // 2 if is_normal else n // 3
    key_eigvals1, key_eigvals2, key_eigvecs1, key_eigvecs2 = prng.split(
        prng.prng_key(1), 4
    )
    eigvals1 = linalg.abs2(prng.normal(key_eigvals1, shape=(n1,)))
    eigvals2 = linalg.abs2(prng.normal(key_eigvals2, shape=(n2,)))
    A1 = test_util.hermitian_matrix_from_eigenvalues(eigvals1, key_eigvecs1)
    A2 = test_util.hermitian_matrix_from_eigenvalues(eigvals2, key_eigvecs2)
    expected = {
        "fx": np.sum(linalg.abs2(A1), axis=1),
        "fy": np.sum(linalg.abs2(A2), axis=1),
    }

    def matvec(v, A1, A2):
        return {"fx": A1 @ v["x"], "fy": A2 @ v["y"]}

    params = (A1, A2)
    x_like = {"x": np.ones(n1), "y": np.ones(n2)}
    sampler = stochtrace.sampler_signs(x_like, num=num_samples)
    integrand = _make_integrand(is_normal)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    return matvec, params, estimate, expected


@testing.parametrize("is_normal", [False, True])
@testing.parametrize("dtype", [float, complex])
@testing.parametrize("n, rank", [(50, 10), (100, 30)])
def cases_exact_num_samples_more_than_rank(n, rank, dtype, is_normal):
    """Exact squared row norms when num_samples exceeds the operator's rank."""
    num_samples = rank + 1
    key_mat1, key_mat2 = prng.split(prng.prng_key(1), 2)
    if is_normal:
        nz_eigvals = prng.normal(key_mat1, shape=(rank,), dtype=dtype)
        eigvals = np.concatenate([nz_eigvals, np.zeros(n - rank, dtype=dtype)])
        A = test_util.hermitian_matrix_from_eigenvalues(eigvals, key_mat2, dtype=dtype)
    else:
        A1 = prng.normal(key_mat1, shape=(n + 1, rank), dtype=dtype)
        A2 = prng.normal(key_mat2, shape=(rank, n), dtype=dtype)
        A = A1 @ A2
    expected = {"fx": np.sum(linalg.abs2(A), axis=1)}

    def matvec(v, A):
        return {"fx": A @ v["x"]}

    params = (A,)
    x_like = {"x": np.ones(n, dtype=dtype)}
    sampler = stochtrace.sampler_signs(x_like, num=num_samples)
    integrand = _make_integrand(is_normal)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    return matvec, params, estimate, expected


@testing.parametrize_with_cases(
    "matvec, params, estimate, expected", cases=".", prefix="cases_exact_"
)
def test_xrownorms_squared_exact(matvec, params, estimate, expected):
    """Assert exact squared row norms when num_samples is large enough."""
    key = prng.prng_key(1)
    received = estimate(matvec, key, *params)
    for leaf, expected_leaf in expected.items():
        test_util.assert_allclose(received[leaf], expected_leaf)


class _ExperimentParams(NamedTuple):
    make_A: Callable[[int, Array, type], Array]
    num_samples: int
    max_rel_err: float
    is_normal: bool
    requires_x64: bool
    dtype: type


@testing.parametrize("dtype", [float, complex])
def cases_experiments_exp_normal(dtype):
    """Hermitian matrix with eigenvalues that decay rapidly, XSymRowNorm"""
    return _ExperimentParams(
        make_A=test_util.hermitian_matrix_eigvals_decaying,
        num_samples=50,
        max_rel_err=1e-12,
        is_normal=True,
        requires_x64=True,
        dtype=dtype,
    )


@testing.parametrize("dtype", [float, complex])
def cases_experiments_exp_general(dtype):
    """Hermitian matrix with eigenvalues that decay rapidly, XRowNorm"""
    return _ExperimentParams(
        make_A=test_util.hermitian_matrix_eigvals_decaying,
        num_samples=35,
        max_rel_err=1e-9,
        is_normal=False,
        requires_x64=True,
        dtype=dtype,
    )


@testing.parametrize("dtype", [float, complex])
def cases_experiments_step_normal(dtype):
    """Hermitian matrix with eigenvalues that are flat with a sudden drop, XSymRowNorm"""
    return _ExperimentParams(
        make_A=test_util.hermitian_matrix_eigvals_step,
        num_samples=60,
        max_rel_err=9e-3,
        is_normal=True,
        requires_x64=False,
        dtype=dtype,
    )


@testing.parametrize("dtype", [float, complex])
def cases_experiments_step_general(dtype):
    """Hermitian matrix with eigenvalues that are flat with a sudden drop, XRowNorm"""
    return _ExperimentParams(
        make_A=test_util.hermitian_matrix_eigvals_step,
        num_samples=60,
        max_rel_err=1e-5,
        is_normal=False,
        requires_x64=False,
        dtype=dtype,
    )


@testing.parametrize_with_cases(
    "make_A, num_samples, max_rel_err, is_normal, requires_x64, dtype",
    cases=".",
    prefix="cases_experiments_",
)
def test_xrownorms_squared_reproduce_experiments(
    make_A, num_samples, max_rel_err, is_normal, requires_x64, dtype
):
    """Assert that squared row norms are estimated accurately for structured spectra."""
    config.update("jax_enable_x64", requires_x64)
    n = 1_000
    num_rep = 10
    key = prng.prng_key(50)
    key_eigvecs, key = prng.split(key)
    A = make_A(n, key_eigvecs, dtype=dtype)
    expected = np.sum(linalg.abs2(A), axis=1)

    x_like = {"x": np.ones(n, dtype=dtype)}
    sampler = stochtrace.sampler_signs(x_like, num=num_samples)
    integrand = _make_integrand(is_normal)
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)

    def matvec(v, A):
        return {"fx": A @ v["x"]}

    key_ests = prng.split(key, num_rep)
    received = func.vmap(lambda key: estimate(matvec, key, A))(key_ests)["fx"]
    max_rel_errs = np.array_max(np.abs(received - expected) / np.abs(expected), axis=1)
    median_max_rel_err = np.median(max_rel_errs)
    assert float(median_max_rel_err) < max_rel_err
    config.update("jax_enable_x64", False)
