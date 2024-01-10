"""Test slq.logdet_prod()."""

from matfree import hutchinson, slq, test_util
from matfree.backend import linalg, np, prng, testing


@testing.fixture()
def A(nrows, ncols, num_significant_singular_vals):
    """Make a positive definite matrix with certain spectrum."""
    # 'Invent' a spectrum. Use the number of pre-defined eigenvalues.
    n = min(nrows, ncols)
    d = np.arange(n) + 1.0
    d = d.at[num_significant_singular_vals:].set(0.001)
    return test_util.asymmetric_matrix_from_singular_values(d, nrows=nrows, ncols=ncols)


@testing.parametrize("nrows", [50])
@testing.parametrize("ncols", [30])
@testing.parametrize("num_significant_singular_vals", [30])
@testing.parametrize("order", [20])
def test_logdet_product(A, order):
    """Assert that logdet_product yields an accurate estimate."""
    _, ncols = np.shape(A)
    key = prng.prng_key(3)

    def matvec(x):
        return {"fx": A @ x["fx"]}

    def vecmat(x):
        return {"fx": x["fx"] @ A}

    x_like = {"fx": np.ones((ncols,), dtype=float)}
    fun = hutchinson.sampler_normal(x_like, num=400)
    problem = slq.integrand_logdet_product(order, matvec, vecmat)
    estimate = hutchinson.hutchinson(problem, fun)
    received = estimate(key)

    expected = linalg.slogdet(A.T @ A)[1]
    print_if_assert_fails = ("error", np.abs(received - expected), "target:", expected)
    assert np.allclose(received, expected, atol=1e-2, rtol=1e-2), print_if_assert_fails


@testing.parametrize("n", [50])
# usually: ~1.5 * num_significant_eigvals.
# But logdet seems to converge sooo much faster.
def test_logdet_product_exact_for_full_order_lanczos(n):
    r"""Computing v^\top f(A^\top @ A) v with max-order Lanczos is exact for _any_ v."""
    # Construct a (numerically nice) matrix
    singular_values = np.sqrt(np.arange(1.0, 1.0 + n, step=1.0))
    A = test_util.asymmetric_matrix_from_singular_values(
        singular_values, nrows=n, ncols=n
    )

    # Set up max-order Lanczos approximation inside SLQ for the matrix-logarithm
    order = n - 1
    integrand = slq.integrand_logdet_product(order, lambda v: A @ v, lambda v: v @ A)

    # Construct a vector without that does not have expected 2-norm equal to "dim"
    x = prng.normal(prng.prng_key(seed=1), shape=(n,)) + 1

    # Compute v^\top @ log(A) @ v via Lanczos
    received = integrand(x)

    # Compute the "true" value of v^\top @ log(A) @ v via eigenvalues
    eigvals, eigvecs = linalg.eigh(A.T @ A)
    logA = eigvecs @ linalg.diagonal_matrix(np.log(eigvals)) @ eigvecs.T
    expected = x.T @ logA @ x

    # They should be identical
    assert np.allclose(received, expected)
