"""Tests for Hessenberg factorisations (-> Arnoldi)."""

from matfree import decomp, test_util
from matfree.backend import linalg, np, prng, testing


@testing.parametrize("nrows", [10])
@testing.parametrize("num_matvecs", [0, 5, 9])
@testing.parametrize("reortho", ["none", "full"])
@testing.parametrize("dtype", [float])
def test_decomposition_is_satisfied(nrows, num_matvecs, reortho, dtype):
    # Create a well-conditioned test-matrix
    A = prng.normal(prng.prng_key(1), shape=(nrows, nrows), dtype=dtype)
    v = prng.normal(prng.prng_key(2), shape=(nrows,), dtype=dtype)

    # Decompose
    algorithm = decomp.hessenberg(num_matvecs, reortho=reortho)
    Q, H, r, c = algorithm(lambda s, p: p @ s, v, A)

    # Assert shapes
    assert Q.shape == (nrows, num_matvecs)
    assert H.shape == (num_matvecs, num_matvecs)
    assert r.shape == (nrows,)
    assert c.shape == ()

    # Test the decompositions
    e0, ek = np.eye(num_matvecs)[[0, -1], :]
    test_util.assert_allclose(A @ Q - Q @ H - linalg.outer(r, ek), 0.0)
    test_util.assert_allclose(Q.T.conj() @ Q - np.eye(num_matvecs), 0.0)

    if num_matvecs > 0:
        test_util.assert_allclose(Q @ e0, c * v)


@testing.parametrize("nrows", [10])
@testing.parametrize("num_matvecs", [5, 9])
@testing.parametrize("reortho", ["full"])
def test_reorthogonalisation_improves_the_estimate(nrows, num_matvecs, reortho):
    # Create an ill-conditioned test-matrix (that requires reortho=True)
    A = linalg.hilbert(nrows)
    v = prng.normal(prng.prng_key(2), shape=(nrows,))

    # Decompose
    algorithm = decomp.hessenberg(num_matvecs, reortho=reortho)
    Q, H, r, c = algorithm(lambda s, p: p @ s, v, A)

    # Assert shapes
    assert Q.shape == (nrows, num_matvecs)
    assert H.shape == (num_matvecs, num_matvecs)
    assert r.shape == (nrows,)
    assert c.shape == ()

    # Test the decompositions
    e0, ek = np.eye(num_matvecs)[[0, -1], :]
    test_util.assert_allclose(A @ Q - Q @ H - linalg.outer(r, ek), 0.0)
    test_util.assert_allclose(Q.T @ Q - np.eye(num_matvecs), 0.0)
    test_util.assert_allclose(Q @ e0, c * v)


@testing.parametrize("reortho_wrong", [True, "full_with_sparsity", "None"])
def test_raises_error_for_wrong_reorthogonalisation_flag(reortho_wrong):
    with testing.raises(TypeError, match="Unexpected input"):
        _ = decomp.hessenberg(1, reortho=reortho_wrong)
