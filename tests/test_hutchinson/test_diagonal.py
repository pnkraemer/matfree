"""Tests for basic trace estimators."""

from matfree import hutchinson
from matfree.backend import func, linalg, np, prng, testing, tree_util


@testing.parametrize("sample_fun", [prng.normal, prng.rademacher])
def test_diagonal(sample_fun):
    """Assert that the estimated diagonal approximates the true diagonal accurately."""

    @testing.case()
    def case_diagonal():
        return hutchinson.integrand_diagonal, linalg.diagonal

    def fun(x):
        """Create a nonlinear, to-be-differentiated function."""
        fx = np.sin(np.flip(np.cos(x["params"])) + 1.0) * np.sin(x["params"])
        return {"params": fx}

    key = prng.prng_key(seed=2)

    # Linearise function
    x0 = prng.uniform(key, shape=(4,))  # random lin. point
    args_like = {"params": x0}
    _, jvp = func.linearize(fun, args_like)
    J = func.jacfwd(fun)(args_like)

    expected = tree_util.tree_map(linalg.diagonal, J["params"])

    # Estimate the trace
    problem = hutchinson.integrand_diagonal(jvp)
    sampler = hutchinson.sampler_from_prng(prng.normal, args_like, num=100_000)
    estimate = hutchinson.montecarlo(problem, sample_fun=sampler, stats_fun=np.mean)
    received = estimate(key)

    def compare(a, b):
        return np.allclose(a, b, rtol=1e-2)

    print(received)
    print(expected)
    assert tree_util.tree_all(tree_util.tree_map(compare, received, expected))
