"""Test joint trace and diagonal estimation."""

from matfree import stochtrace
from matfree.backend import func, np, prng, tree


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
    problem = stochtrace.integrand_trace_and_diagonal()
    sampler = stochtrace.sampler_rademacher(args_like, num=100_000)
    estimate = stochtrace.estimator(problem, sampler=sampler)
    received = estimate(jvp, key)

    # Estimate the trace
    problem = stochtrace.integrand_trace()
    sampler = stochtrace.sampler_rademacher(args_like, num=100_000)
    estimate = stochtrace.estimator(problem, sampler=sampler)
    expected_trace = estimate(jvp, key)

    # Estimate the diagonal
    problem = stochtrace.integrand_diagonal()
    sampler = stochtrace.sampler_rademacher(args_like, num=100_000)
    estimate = stochtrace.estimator(problem, sampler=sampler)
    expected_diagonal = estimate(jvp, key)

    expected = {"trace": expected_trace, "diagonal": expected_diagonal}

    assert tree.tree_all(tree.tree_map(np.allclose, received, expected))
