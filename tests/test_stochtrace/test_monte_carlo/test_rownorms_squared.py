"""Test the estimation of squared row norms."""

from matfree import stochtrace, test_util
from matfree.backend import func, linalg, np, prng, testing, tree


@testing.parametrize("seed", [1, 2, 3])
@testing.parametrize("dtype", [float, complex])
def test_rownorms_squared(seed, dtype):
    """Assert that the estimated squared row norms approximate the true values accurately."""

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

    expected = tree.tree_map(lambda j: np.sum(linalg.abs2(j), axis=1), J)

    # Estimate the matrix function
    integrand = stochtrace.monte_carlo_rownorms_squared()
    sampler = stochtrace.sampler_normal(args_like, num=100_000)
    estimate = stochtrace.estimator_monte_carlo(integrand, sampler=sampler)
    key, subkey = prng.split(key, num=2)
    received = estimate(jvp, subkey)

    def compare(a, b):
        return np.allclose(a, b, rtol=0.05, atol=0.05)

    assert tree.tree_all(tree.tree_map(compare, received, expected))


@testing.parametrize("dtype", [float, complex])
def test_rownorms_squared_sums_to_frobeniusnorm_squared(dtype):
    """Assert that summing the squared row norms over all entries gives the squared Frobenius norm.

    This identity, sum_i rownorms_squared(v)[i] == frobeniusnorm_squared(v), holds exactly
    for every sample vector v (not just in expectation), since sum_i |x_i|^2 = vdot(x, x).
    Also exercises heterogeneous pytree support via two differently-sized blocks.
    """
    n1, n2 = 3, 5
    key_mat1, key_mat2, key_v = prng.split(prng.prng_key(1), 3)
    A1 = prng.normal(key_mat1, shape=(n1, n1), dtype=dtype)
    A2 = prng.normal(key_mat2, shape=(n2, n2), dtype=dtype)

    def matvec(v, A1, A2):
        return {"fx": A1 @ v["fx"], "fy": A2 @ v["fy"]}

    x_like = {"fx": np.ones(n1, dtype=dtype), "fy": np.ones(n2, dtype=dtype)}
    v = test_util.tree_random_like(key_v, x_like)

    rownorms_squared = stochtrace.monte_carlo_rownorms_squared()
    frobeniusnorm_squared = stochtrace.monte_carlo_frobeniusnorm_squared()

    rownorms_out = rownorms_squared(matvec, v, A1, A2)
    frobenius_out = frobeniusnorm_squared(matvec, v, A1, A2)

    assert set(rownorms_out.keys()) == {"fx", "fy"}
    total = sum(np.sum(leaf) for leaf in tree.tree_leaves(rownorms_out))
    test_util.assert_allclose(total, frobenius_out)


@testing.parametrize("dtype", [float, complex])
def test_rownorms_squared_rectangular(dtype):
    """Assert correct squared row norms for a rectangular (non-square) matvec."""
    m, n = 6, 10
    key_mat, key_est = prng.split(prng.prng_key(2), 2)
    A = prng.normal(key_mat, shape=(m, n), dtype=dtype)
    expected = np.sum(linalg.abs2(A), axis=1)

    def matvec(v, A):
        return A @ v

    integrand = stochtrace.monte_carlo_rownorms_squared()
    sampler = stochtrace.sampler_normal(np.ones(n, dtype=dtype), num=200_000)
    estimate = stochtrace.estimator_monte_carlo(integrand, sampler=sampler)
    received = estimate(matvec, key_est, A)
    assert np.allclose(received, expected, rtol=0.05, atol=0.05)


@testing.parametrize("dtype", [float, complex])
def test_rownorms_squared_adjoint_gives_column_norms(dtype):
    """Assert that passing the adjoint matvec estimates squared column norms instead."""
    m, n = 6, 10
    key_mat, key_est = prng.split(prng.prng_key(3), 2)
    A = prng.normal(key_mat, shape=(m, n), dtype=dtype)
    expected = np.sum(linalg.abs2(A), axis=0)

    def matvec_adjoint(v, A):
        return A.T.conj() @ v

    integrand = stochtrace.monte_carlo_rownorms_squared()
    sampler = stochtrace.sampler_normal(np.ones(m, dtype=dtype), num=200_000)
    estimate = stochtrace.estimator_monte_carlo(integrand, sampler=sampler)
    received = estimate(matvec_adjoint, key_est, A)
    assert np.allclose(received, expected, rtol=0.05, atol=0.05)
