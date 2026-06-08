"""Test the diagonal estimation."""

from matfree import stochtrace
from matfree.backend import func, linalg, np, prng, testing, tree


@testing.parametrize("seed", [1, 2, 3])
@testing.parametrize("dtype", [float, complex])
def test_diagonal(seed, dtype):
    """Assert that the estimated diagonal approximates the true diagonal accurately."""

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
    J = func.jacfwd(fun, holomorphic=dtype is complex)(args_like)["params"]

    expected = tree.tree_map(linalg.diagonal, J)

    # Estimate the matrix function
    problem = stochtrace.monte_carlo_diagonal()
    sampler = stochtrace.sampler_normal(args_like, num=100_000)
    estimate = stochtrace.estimator_monte_carlo(problem, sampler=sampler)
    key, subkey = prng.split(key, num=2)
    received = estimate(jvp, subkey)

    def compare(a, b):
        return np.allclose(a, b, rtol=0.05, atol=0.05)

    assert tree.tree_all(tree.tree_map(compare, received, expected))
