"""Test the estimation with multiple statistics."""

from matfree import stochtrace
from matfree.backend import func, linalg, np, prng, testing, tree


@testing.parametrize("seed", [1, 2, 3])
@testing.parametrize("dtype", [float, complex])
def test_yields_correct_tree_structure(seed, dtype):
    """Assert that mean and standard-deviation are estimated correctly."""

    def fun(x):
        """Create a nonlinear, to-be-differentiated function."""
        fx = np.sin(np.flip(np.cos(x["params"])) + 1.0) * np.sin(x["params"])
        return {"params": fx}

    key = prng.prng_key(seed)

    # Linearise function
    key, subkey = prng.split(key, num=2)
    x0 = prng.normal(subkey, shape=(4,), dtype=dtype)
    args_like = {"params": x0}
    _, jvp = func.linearize(fun, args_like)

    # Estimate the matrix function
    integrand = stochtrace.monte_carlo_diagonal()
    integrand = stochtrace.monte_carlo_wrap_moments(
        integrand, moments={"moment_1st": 1, "moment_2nd": 2}
    )
    sampler = stochtrace.sampler_normal(args_like, num=100_000)
    estimate = stochtrace.estimator_monte_carlo(integrand, sampler=sampler)
    key, subkey = prng.split(key, num=2)
    received = estimate(jvp, subkey)

    irrelevant_value = 1.1
    expected = {
        "params": {"moment_1st": irrelevant_value, "moment_2nd": irrelevant_value}
    }
    assert tree.tree_structure(received) == tree.tree_structure(expected)


@testing.fixture(name="key")
@testing.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def fixture_key(seed):
    """Fix a pseudo-random number generator."""
    return prng.prng_key(seed)


@testing.fixture(name="J_and_jvp")
@testing.parametrize("dtype", [float])  # no complex because Rademacher not well defined
def fixture_J_and_jvp(key, dtype):
    """Create a nonlinear, to-be-differentiated function."""

    def fun(x):
        return np.sin(np.flip(np.cos(x)) + 1.0) * np.sin(x) + 1.0

    # Linearise function
    x0 = prng.normal(key, shape=(2,), dtype=dtype)
    _, jvp = func.linearize(fun, x0)
    J = func.jacfwd(fun, holomorphic=dtype is complex)(x0)

    return J, jvp, x0


def test_yields_correct_variance_normal(J_and_jvp, key):
    """Assert that the estimated trace approximates the true trace accurately."""
    # Estimate the trace
    J, jvp, args_like = J_and_jvp
    integrand = stochtrace.monte_carlo_trace()
    integrand = stochtrace.monte_carlo_wrap_moments(integrand, moments=[1, 2])
    sampler = stochtrace.sampler_normal(args_like, num=1_000_000)
    estimate = stochtrace.estimator_monte_carlo(integrand, sampler=sampler)
    first, second = estimate(jvp, key)

    # Assert the trace is correct
    truth = linalg.trace(J)
    assert np.allclose(first, truth, rtol=0.3)

    # Assert the variance is correct:
    norm = linalg.matrix_norm(J, which="fro") ** 2
    assert np.allclose(second - first**2, norm * 2, rtol=0.3)


def test_yields_correct_variance_signs(J_and_jvp, key):
    """Assert that the estimated trace approximates the true trace accurately."""
    # Estimate the trace
    J, jvp, args_like = J_and_jvp

    integrand = stochtrace.monte_carlo_trace()
    integrand = stochtrace.monte_carlo_wrap_moments(integrand, moments=[1, 2])
    sampler = stochtrace.sampler_signs(args_like, num=5000)
    estimate = stochtrace.estimator_monte_carlo(integrand, sampler=sampler)
    first, second = estimate(jvp, key)

    # Assert the trace is correct
    truth = linalg.trace(J)
    assert np.allclose(first, truth, rtol=0.2)

    # Assert the variance is correct:
    norm = linalg.matrix_norm(J, which="fro") ** 2
    truth = norm - linalg.trace(J**2)
    assert np.allclose(second - first**2, 2 * truth, atol=0.3, rtol=0.3)
