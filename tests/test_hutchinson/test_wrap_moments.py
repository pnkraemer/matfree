"""Test the estimation with multiple statistics."""

from matfree import hutchinson
from matfree.backend import func, linalg, np, prng, testing, tree_util


def test_yields_correct_tree_structure():
    """Assert that mean and standard-deviation are estimated correctly."""

    def fun(x):
        """Create a nonlinear, to-be-differentiated function."""
        fx = np.sin(np.flip(np.cos(x["params"])) + 1.0) * np.sin(x["params"])
        return {"params": fx}

    key = prng.prng_key(seed=2)

    # Linearise function
    x0 = prng.uniform(key, shape=(4,))  # random lin. point
    args_like = {"params": x0}
    _, jvp = func.linearize(fun, args_like)

    # Estimate the matrix function
    problem = hutchinson.integrand_diagonal(jvp)
    problem = hutchinson.integrand_wrap_moments(
        problem, moments={"moment_1st": 1, "moment_2nd": 2}
    )
    sampler = hutchinson.sampler_normal(args_like, num=100_000)
    estimate = hutchinson.hutchinson(problem, sample_fun=sampler)
    received = estimate(key)

    irrelevant_value = 1.1
    expected = {
        "params": {"moment_1st": irrelevant_value, "moment_2nd": irrelevant_value}
    }
    assert tree_util.tree_structure(received) == tree_util.tree_structure(expected)


@testing.fixture(name="key")
def fixture_key():
    """Fix a pseudo-random number generator."""
    return prng.prng_key(seed=1)


@testing.fixture(name="J_and_jvp")
def fixture_J_and_jvp(key):
    """Create a nonlinear, to-be-differentiated function."""

    def fun(x):
        return np.sin(np.flip(np.cos(x)) + 1.0) * np.sin(x) + 1.0

    # Linearise function
    x0 = prng.uniform(key, shape=(2,))  # random lin. point
    _, jvp = func.linearize(fun, x0)
    J = func.jacfwd(fun)(x0)

    return J, jvp, x0


def test_yields_correct_variance_normal(J_and_jvp, key):
    """Assert that the estimated trace approximates the true trace accurately."""
    # Estimate the trace
    J, jvp, args_like = J_and_jvp
    problem = hutchinson.integrand_trace(jvp)
    problem = hutchinson.integrand_wrap_moments(problem, moments=[1, 2])
    sampler = hutchinson.sampler_normal(args_like, num=1_000_000)
    estimate = hutchinson.hutchinson(problem, sample_fun=sampler)
    first, second = estimate(key)

    # Assert the trace is correct
    truth = linalg.trace(J)
    assert np.allclose(first, truth, rtol=1e-2)

    # Assert the variance is correct:
    norm = linalg.matrix_norm(J, which="fro") ** 2
    assert np.allclose(second - first**2, norm * 2, rtol=1e-2)


def test_yields_correct_variance_rademacher(J_and_jvp, key):
    """Assert that the estimated trace approximates the true trace accurately."""
    # Estimate the trace
    J, jvp, args_like = J_and_jvp

    problem = hutchinson.integrand_trace(jvp)
    problem = hutchinson.integrand_wrap_moments(problem, moments=[1, 2])
    sampler = hutchinson.sampler_rademacher(args_like, num=500)
    estimate = hutchinson.hutchinson(problem, sample_fun=sampler)
    first, second = estimate(key)

    # Assert the trace is correct
    truth = linalg.trace(J)
    assert np.allclose(first, truth, rtol=1e-2)

    # Assert the variance is correct:
    norm = linalg.matrix_norm(J, which="fro") ** 2
    truth = norm - linalg.trace(J**2)
    assert np.allclose(second - first**2, truth, atol=1e-2, rtol=1e-2)
