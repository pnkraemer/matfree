"""Test joint trace and diagonal estimation."""

from matfree import stochtrace
from matfree.backend import func, np, prng, testing, tree


@testing.parametrize("seed", [1, 2, 3])
@testing.parametrize("dtype", [float, complex])
def test_trace_and_diagonal(seed, dtype):
    """Assert that traces and diagonals are estimated correctly."""

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

    # Estimate the trace and diagonal jointly
    problem = stochtrace.integrand_trace_and_diagonal()
    sampler = stochtrace.sampler_normal(args_like, num=100_000)
    estimate = stochtrace.estimator(problem, sampler=sampler)
    key, subkey = prng.split(key, num=2)
    received = estimate(jvp, subkey)

    # Estimate the trace
    problem = stochtrace.integrand_trace()
    sampler = stochtrace.sampler_normal(args_like, num=100_000)
    estimate = stochtrace.estimator(problem, sampler=sampler)
    expected_trace = estimate(jvp, subkey)

    # Estimate the diagonal
    problem = stochtrace.integrand_diagonal()
    sampler = stochtrace.sampler_normal(args_like, num=100_000)
    estimate = stochtrace.estimator(problem, sampler=sampler)
    expected_diagonal = estimate(jvp, subkey)

    expected = {"trace": expected_trace, "diagonal": expected_diagonal}
    assert tree.tree_all(tree.tree_map(np.allclose, received, expected))
