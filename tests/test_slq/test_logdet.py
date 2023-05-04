"""Tests for Lanczos functionality."""

from matfree import montecarlo, slq, test_util
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
def test_logdet(A, order):
    """Assert that the log-determinant estimation matches the true log-determinant."""
    n, _ = np.shape(A)
    key = prng.prng_key(1)
    fun = montecarlo.normal(shape=(n,))
    received = slq.logdet(
        lambda v: A @ v,
        order,
        key=key,
        num_samples_per_batch=10,
        num_batches=1,
        sample_fun=fun,
    )
    expected = linalg.slogdet(A)[1]
    print_if_assert_fails = ("error", np.abs(received - expected), "target:", expected)
    assert np.allclose(received, expected, atol=1e-2, rtol=1e-2), print_if_assert_fails
