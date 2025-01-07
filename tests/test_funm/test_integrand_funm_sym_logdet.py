"""Tests for Lanczos functionality."""

from matfree import decomp, funm, stochtrace, test_util
from matfree.backend import linalg, np, prng, testing


def make_A(n, num_significant_eigvals):
    """Make a positive definite matrix with certain spectrum."""
    # 'Invent' a spectrum. Use the number of pre-defined eigenvalues.
    d = np.arange(n) / n + 1.0
    d = d.at[num_significant_eigvals:].set(0.001)

    return test_util.symmetric_matrix_from_eigenvalues(d)


@testing.parametrize("n", [200])
@testing.parametrize("num_significant_eigvals", [30])
@testing.parametrize("num_matvecs", [10])
# usually: ~1.5 * num_significant_eigvals.
# But logdet seems to converge sooo much faster.
def test_logdet_spd(n, num_significant_eigvals, num_matvecs):
    """Assert that the log-determinant estimation matches the true log-determinant."""
    A = make_A(n, num_significant_eigvals)

    def matvec(x):
        return {"fx": A @ x["fx"]}

    key = prng.prng_key(1)
    args_like = {"fx": np.ones((n,), dtype=float)}
    sampler = stochtrace.sampler_normal(args_like, num=10)
    tridiag_sym = decomp.tridiag_sym(num_matvecs, materialize=True)
    integrand = funm.integrand_funm_sym_logdet(tridiag_sym)
    estimate = stochtrace.estimator(integrand, sampler)
    received = estimate(matvec, key)

    expected = linalg.slogdet(A)[1]
    print_if_assert_fails = ("error", np.abs(received - expected), "target:", expected)
    assert np.allclose(received, expected, atol=1e-2, rtol=1e-2), print_if_assert_fails


@testing.parametrize("n", [50])
# usually: ~1.5 * num_significant_eigvals.
# But logdet seems to converge sooo much faster.
def test_logdet_spd_exact_for_full_num_matvecs_lanczos(n):
    r"""Computing v^\top f(A) v with max-order Lanczos should be exact for _any_ v."""
    # Construct a (numerically nice) matrix
    eigvals = np.arange(1.0, 1.0 + n, step=1.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    # Set up max-num_matvecs Lanczos approximation inside SLQ for the matrix-logarithm
    num_matvecs = n - 1
    tridiag_sym = decomp.tridiag_sym(num_matvecs, materialize=True)
    integrand = funm.integrand_funm_sym_logdet(tridiag_sym)

    # Construct a vector without that does not have expected 2-norm equal to "dim"
    x = prng.normal(prng.prng_key(seed=1), shape=(n,)) + 10

    # Compute v^\top @ log(A) @ v via Lanczos
    received = integrand(lambda v: A @ v, x)

    # Compute the "true" value of v^\top @ log(A) @ v via eigenvalues
    eigvals, eigvecs = linalg.eigh(A)
    logA = eigvecs @ linalg.diagonal_matrix(np.log(eigvals)) @ eigvecs.T
    expected = x.T @ logA @ x

    # They should be identical
    assert np.allclose(received, expected)
