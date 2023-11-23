"""Tests for Lanczos functionality."""

from matfree import hutchinson, slq, test_util
from matfree.backend import linalg, np, prng, testing


@testing.fixture()
def A(n, num_significant_eigvals):
    """Make a positive definite matrix with certain spectrum."""
    # 'Invent' a spectrum. Use the number of pre-defined eigenvalues.
    d = np.arange(n) / n + 1.0
    d = d.at[num_significant_eigvals:].set(0.001)

    return test_util.symmetric_matrix_from_eigenvalues(d)


@testing.parametrize("n", [200])
@testing.parametrize("num_significant_eigvals", [30])
@testing.parametrize("order", [10])
# usually: ~1.5 * num_significant_eigvals.
# But logdet seems to converge sooo much faster.
def test_logdet_spd(A, order):
    """Assert that the log-determinant estimation matches the true log-determinant."""
    n, _ = np.shape(A)

    def matvec(x):
        return {"fx": A @ x["fx"]}

    key = prng.prng_key(1)
    args_like = {"fx": np.ones((n,), dtype=float)}
    sampler = hutchinson.sampler_normal(args_like, num=10)
    integrand = slq.integrand_logdet_spd(order, matvec)
    estimate = hutchinson.hutchinson(integrand, sampler)
    received = estimate(key)

    expected = linalg.slogdet(A)[1]
    print_if_assert_fails = ("error", np.abs(received - expected), "target:", expected)
    assert np.allclose(received, expected, atol=1e-2, rtol=1e-2), print_if_assert_fails
