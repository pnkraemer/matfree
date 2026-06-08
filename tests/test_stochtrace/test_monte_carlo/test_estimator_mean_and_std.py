"""Test estimator_monte_carlo_mean_and_std."""

from matfree import stochtrace
from matfree.backend import func, linalg, np, prng, testing, tree


@testing.parametrize("seed", [1, 2, 3])
@testing.parametrize("dtype", [float, complex])
def test_mean_matches_estimator_monte_carlo(seed, dtype):
    """Assert that mean equals the result of estimator_monte_carlo."""

    def fun(x):
        return np.sin(np.flip(np.cos(x)) + 1.0) * np.sin(x)

    key = prng.prng_key(seed)
    key, subkey = prng.split(key, num=2)
    x0 = prng.normal(subkey, shape=(4,), dtype=dtype)
    _, jvp = func.linearize(fun, x0)

    integrand = stochtrace.monte_carlo_trace()
    sampler = stochtrace.sampler_signs(x0, num=10_000)

    key, subkey = prng.split(key, num=2)
    mean_only = stochtrace.estimator_monte_carlo(integrand, sampler=sampler)(
        jvp, subkey
    )
    mean, _std = stochtrace.estimator_monte_carlo_mean_and_std(
        integrand, sampler=sampler
    )(jvp, subkey)
    assert np.allclose(mean, mean_only)


@testing.parametrize("seed", [1, 2, 3])
@testing.parametrize("dtype", [float, complex])
def test_pytree_structure(seed, dtype):
    """Assert that mean and std_error share the integrand's pytree structure."""

    def fun(x):
        fx = np.sin(np.flip(np.cos(x["params"])) + 1.0) * np.sin(x["params"])
        return {"params": fx}

    key = prng.prng_key(seed)
    key, subkey = prng.split(key, num=2)
    x0 = prng.normal(subkey, shape=(4,), dtype=dtype)
    args_like = {"params": x0}
    _, jvp = func.linearize(fun, args_like)

    integrand = stochtrace.monte_carlo_diagonal()
    sampler = stochtrace.sampler_normal(args_like, num=100_000)
    estimate = stochtrace.estimator_monte_carlo_mean_and_std(integrand, sampler=sampler)
    key, subkey = prng.split(key, num=2)
    mean, std_error = estimate(jvp, subkey)

    expected = {"params": 1.1}
    assert tree.tree_structure(mean) == tree.tree_structure(expected)
    assert tree.tree_structure(std_error) == tree.tree_structure(expected)


@testing.fixture(name="key")
@testing.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def fixture_key(seed):
    return prng.prng_key(seed)


@testing.fixture(name="J_and_jvp")
@testing.parametrize("dtype", [float])
def fixture_J_and_jvp(key, dtype):
    def fun(x):
        return np.sin(np.flip(np.cos(x)) + 1.0) * np.sin(x) + 1.0

    x0 = prng.normal(key, shape=(2,), dtype=dtype)
    _, jvp = func.linearize(fun, x0)
    J = func.jacfwd(fun, holomorphic=dtype is complex)(x0)
    return J, jvp, x0


def test_std_error_magnitude_signs(J_and_jvp, key):
    """Assert the std_error is correct for Rademacher samples.

    For Rademacher samples, Var[X] = 2*(||J||_F^2 - tr(J^2)), so
    std_error = sqrt(Var[X] / n).
    """
    J, jvp, args_like = J_and_jvp
    integrand = stochtrace.monte_carlo_trace()
    num = 5_000
    sampler = stochtrace.sampler_signs(args_like, num=num)
    estimate = stochtrace.estimator_monte_carlo_mean_and_std(integrand, sampler=sampler)
    mean, std_error = estimate(jvp, key)

    truth = linalg.trace(J)
    assert np.allclose(mean, truth, rtol=0.2)

    variance_truth = linalg.matrix_norm(J, which="fro") ** 2 - linalg.trace(J**2)
    std_error_truth = np.sqrt(2 * variance_truth / num)
    assert np.allclose(std_error, std_error_truth, atol=0.3, rtol=0.3)
