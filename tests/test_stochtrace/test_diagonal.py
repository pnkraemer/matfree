"""Test the diagonal estimation."""

from matfree import stochtrace
from matfree.backend import func, linalg, np, prng, tree


def test_diagonal():
    """Assert that the estimated diagonal approximates the true diagonal accurately."""

    def fun(x):
        """Create a nonlinear, to-be-differentiated function."""
        fx = np.sin(np.flip(np.cos(x["params"])) + 1.0) * np.sin(x["params"])
        return {"params": fx}

    key = prng.prng_key(seed=2)

    # Linearise function
    x0 = prng.uniform(key, shape=(4,))  # random lin. point
    args_like = {"params": x0}
    _, jvp = func.linearize(fun, args_like)
    J = func.jacfwd(fun)(args_like)["params"]

    expected = tree.tree_map(linalg.diagonal, J)

    # Estimate the matrix function
    problem = stochtrace.integrand_diagonal()
    sampler = stochtrace.sampler_normal(args_like, num=100_000)
    estimate = stochtrace.estimator(problem, sampler=sampler)
    received = estimate(jvp, key)

    def compare(a, b):
        return np.allclose(a, b, rtol=1e-2)

    assert tree.tree_all(tree.tree_map(compare, received, expected))
