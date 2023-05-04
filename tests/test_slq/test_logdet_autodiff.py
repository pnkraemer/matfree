"""Tests for (selected) autodiff functionality."""


from matfree import montecarlo, slq, test_util
from matfree.backend import np, prng, testing


@testing.fixture()
def A(n, num_significant_eigvals):
    """Make a positive definite matrix with certain spectrum."""
    # 'Invent' a spectrum. Use the number of pre-defined eigenvalues.
    d = np.arange(n) + 10.0
    d = d.at[num_significant_eigvals:].set(0.001)

    return test_util.symmetric_matrix_from_eigenvalues(d)


@testing.parametrize("n", [200])
@testing.parametrize("num_significant_eigvals", [30])
@testing.parametrize("order", [10])
# usually: ~1.5 * num_significant_eigvals.
# But logdet seems to converge sooo much faster.
def test_check_grads(A, order):
    """Assert that log-determinant computation admits valid VJPs and JVPs."""
    key = prng.prng_key(1)

    def fun(s):
        return _logdet(s, order, key)

    testing.check_grads(fun, (A,), order=1, atol=1e-1, rtol=1e-1)


def _logdet(A, order, key):
    n, _ = np.shape(A)
    fun = montecarlo.normal(shape=(n,))
    return slq.logdet(
        lambda v: A @ v,
        order,
        key=key,
        num_samples_per_batch=10,
        num_batches=1,
        sample_fun=fun,
    )
