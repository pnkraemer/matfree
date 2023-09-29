"""Tests for pseudo-inverse functionality."""
from matfree import pinv
from matfree.backend import func, linalg, np, prng, testing


def fun_tall(x):
    """Evaluate a nonlinear function with a tall Jacobian."""
    P = prng.uniform(prng.prng_key(seed=2), shape=(len(x) - 2, len(x)))
    return np.cos(P @ np.sin(x))


def fun_wide(x):
    """Evaluate a nonlinear function with a wide Jacobian."""
    P = prng.uniform(prng.prng_key(seed=2), shape=(len(x) + 2, len(x)))
    return np.cos(P @ np.sin(x))


def densify(fun, input_dim):
    """Densify a linear operator."""
    return func.vmap(fun, in_axes=-1, out_axes=-1)(np.eye(input_dim))


def solve_dense(Av, b):
    """Solve a linear system by densifying + LU decomposition."""
    return linalg.solve(func.jacfwd(Av)(b), b)


def solve_cg(Av, b):
    """Solve a linear system with conjugate gradients."""
    return linalg.cg(Av, b)[0]


@testing.parametrize(
    "fun, pinv_fun", ([fun_tall, pinv.pinv_tall], [fun_wide, pinv.pinv_wide])
)
@testing.parametrize("solve_fun", [solve_dense, solve_cg])
def test_inverse_correct(fun, pinv_fun, solve_fun):
    """Compare the values of the computed pseudo-inverses to numpy.pinv()."""
    key = prng.prng_key(seed=1)
    x = prng.uniform(key, shape=(7,))

    J = func.jacfwd(fun)(x)

    # Invert
    inverse = pinv_fun(lambda v: J @ v, lambda v: v @ J, solve=solve_fun)

    # Dense invert
    inverse_matrix = densify(inverse, len(fun(x)))
    inverse_expected = linalg.pinv(func.jacfwd(fun)(x))
    assert np.allclose(inverse_matrix, inverse_expected, rtol=1e-3, atol=1e-3)
