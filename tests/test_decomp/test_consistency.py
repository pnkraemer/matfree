"""Ensure that bidiag, tridiag, etc. have consistent signatures."""

from matfree import decomp
from matfree.backend import np, prng, testing, tree


def case_method_bidiag():
    return decomp.bidiag


def case_method_tridiag_sym():
    return lambda n: decomp.tridiag_sym(n, reortho="none")


def case_method_tridiag_sym_reortho():
    return lambda n: decomp.tridiag_sym(n, reortho="full")


def case_method_hessenberg():
    return lambda n: decomp.hessenberg(n, reortho="none")


def case_method_hessenberg_reortho():
    return lambda n: decomp.hessenberg(n, reortho="full")


@testing.parametrize("nrows", [13])
@testing.parametrize("num_matvecs", [6, 13, 0])
@testing.parametrize_with_cases("method", cases=".", prefix="case_method_")
def test_output_shape_as_expected(nrows, num_matvecs, method):
    """Test that all factorisation methods yield consistent output shapes."""
    key = prng.prng_key(1)
    key1, key2 = prng.split(key, num=2)
    A = prng.normal(key1, shape=(nrows, nrows))
    v0 = prng.normal(key2, shape=(nrows,))

    algorithm = method(num_matvecs)
    Us, B, res, ln = algorithm(lambda v: A @ v, v0)

    # Normalise the Us to always have a list
    Us = tree.tree_leaves(Us)

    for U in Us:
        assert np.shape(U) == (nrows, num_matvecs)

    assert np.shape(B) == (num_matvecs, num_matvecs)
    assert np.shape(res) == (nrows,)
    assert np.shape(ln) == ()


@testing.parametrize("nrows", [13])
@testing.parametrize("num_matvecs", [-1, 14])  # 0 and 13 must work (see above)
@testing.parametrize_with_cases("method", cases=".", prefix="case_method_")
def test_value_error_for_unusual_num_matvecs(nrows, num_matvecs, method):
    """Assert a ValueError is raised when the num_matvecs exceeds the matrix size."""
    key = prng.prng_key(1)
    key1, key2 = prng.split(key, num=2)
    A = prng.normal(key1, shape=(nrows, nrows))
    v0 = prng.normal(key2, shape=(nrows,))

    with testing.raises(ValueError, match="exceeds"):
        alg = method(num_matvecs)
        _ = alg(lambda v, p: p @ v, v0, A)
