from matfree import hutchinson
from matfree.backend import func, np, prng, testing, tree_util


@testing.parametrize("sample_fun", [prng.normal, prng.rademacher])
def test_estimate_multiple_stats(sample_fun):
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
    sampler = hutchinson.sampler_from_prng(prng.normal, args_like, num=100_000)
    stats = hutchinson.stats_mean_and_std()
    estimate = hutchinson.hutchinson(problem, sample_fun=sampler, stats_fun=stats)
    received = estimate(key)

    irrelevant_value = 1.1
    expected_structure = {"params": {"mean": irrelevant_value, "std": irrelevant_value}}
    assert tree_util.tree_structure(received) == tree_util.tree_structure(
        expected_structure
    )
