"""Test joint trace and diagonal estimation."""
from matfree import hutchinson
from matfree.backend import func, np, prng, tree_util


def test_trace_and_diagonal():
    """Assert that traces and diagonals are estimated correctly."""

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
    problem = hutchinson.integrand_trace_and_diagonal(jvp)
    sampler = hutchinson.sampler_rademacher(args_like, num=100_000)
    estimate = hutchinson.hutchinson(problem, sample_fun=sampler, stats_fun=np.mean)
    received = estimate(key)

    # Estimate the trace
    problem = hutchinson.integrand_trace(jvp)
    sampler = hutchinson.sampler_rademacher(args_like, num=100_000)
    estimate = hutchinson.hutchinson(problem, sample_fun=sampler, stats_fun=np.mean)
    expected_trace = estimate(key)

    # Estimate the diagonal
    problem = hutchinson.integrand_diagonal(jvp)
    sampler = hutchinson.sampler_rademacher(args_like, num=100_000)
    estimate = hutchinson.hutchinson(problem, sample_fun=sampler, stats_fun=np.mean)
    expected_diagonal = estimate(key)

    expected = {"trace": expected_trace, "diagonal": expected_diagonal}

    assert tree_util.tree_all(tree_util.tree_map(np.allclose, received, expected))
