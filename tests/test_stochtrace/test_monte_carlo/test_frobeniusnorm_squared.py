"""Test the estimation of squared Frobenius-norms."""

from matfree import stochtrace
from matfree.backend import func, linalg, np, prng, testing, tree


@testing.parametrize("seed", [1, 2, 3])
@testing.parametrize("dtype", [float, complex])
def test_frobeniusnorm_squared(seed, dtype):
    """Assert that the estimated squared Frobenius norm approximates accurately."""

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
    J_matrix = func.jacfwd(fun, holomorphic=dtype is complex)(args_like)
    [J] = tree.tree_leaves(J_matrix)
    expected = linalg.trace(J.T.conj() @ J)

    # Estimate the matrix function
    problem = stochtrace.monte_carlo_frobeniusnorm_squared()
    sampler = stochtrace.sampler_signs(args_like, num=100_000)
    estimate = stochtrace.estimator_monte_carlo(problem, sampler=sampler)
    received = estimate(jvp, key)

    assert np.allclose(expected, received, rtol=0.01)
