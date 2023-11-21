"""Test Schatten norm implementations."""

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


@testing.parametrize("nrows", [30])
@testing.parametrize("ncols", [30])
@testing.parametrize("num_significant_singular_vals", [30])
@testing.parametrize("order", [20])
@testing.parametrize("power", [1, 2, 5])
def test_schatten_norm(A, order, power):
    """Assert that schatten_norm yields an accurate estimate."""
    _, s, _ = linalg.svd(A, full_matrices=False)
    expected = np.sum(s**power) ** (1 / power)

    _, ncols = np.shape(A)
    key = prng.prng_key(1)
    fun = hutchinson.sampler_normal(shape=(ncols,))
    received = slq.schatten_norm(
        order,
        lambda v: A @ v,
        lambda v: A.T @ v,
        power=power,
        matrix_shape=np.shape(A),
        key=key,
        num_samples_per_batch=100,
        num_batches=5,
        sample_fun=fun,
    )
    print_if_assert_fails = ("error", np.abs(received - expected), "target:", expected)
    assert np.allclose(received, expected, atol=1e-2, rtol=1e-2), print_if_assert_fails
