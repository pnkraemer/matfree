"""Test the estimation of squared Frobenius-norms."""
from matfree import hutchinson
from matfree.backend import func, linalg, np, prng, tree_util


def test_frobeniusnorm_squared():
    """Assert that the estimated squared Frobenius norm approximates accurately."""

    def fun(x):
        """Create a nonlinear, to-be-differentiated function."""
        fx = np.sin(np.flip(np.cos(x["params"])) + 1.0) * np.sin(x["params"])
        return {"params": fx}

    key = prng.prng_key(seed=2)

    # Linearise function
    x0 = prng.uniform(key, shape=(4,))  # random lin. point
    args_like = {"params": x0}
    _, jvp = func.linearize(fun, args_like)
    [J] = tree_util.tree_leaves(func.jacfwd(fun)(args_like))
    expected = linalg.trace(J.T @ J)

    # Estimate the matrix function
    problem = hutchinson.integrand_frobeniusnorm_squared(jvp)
    sampler = hutchinson.sampler_rademacher(args_like, num=100_000)
    estimate = hutchinson.hutchinson(problem, sample_fun=sampler, stats_fun=np.mean)
    received = estimate(key)

    assert np.allclose(expected, received, rtol=1e-2)
