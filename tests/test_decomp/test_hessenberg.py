"""Tests for Hessenberg factorisations (-> Arnoldi)."""

from matfree import decomp
from matfree.backend import linalg, np, prng, testing


@testing.parametrize("nrows", [10])
@testing.parametrize("krylov_depth", [1, 5, 10])
@testing.parametrize("reortho", ["none", "full"])
@testing.parametrize("dtype", [float])
def test_decomposition_is_satisfied(nrows, krylov_depth, reortho, dtype):
    # Create a well-conditioned test-matrix
    A = prng.normal(prng.prng_key(1), shape=(nrows, nrows), dtype=dtype)
    v = prng.normal(prng.prng_key(2), shape=(nrows,), dtype=dtype)

    # Decompose
    algorithm = decomp.hessenberg(krylov_depth, reortho=reortho)
    Q, H, r, c = algorithm(lambda s, p: p @ s, v, A)

    # Assert shapes
    assert Q.shape == (nrows, krylov_depth)
    assert H.shape == (krylov_depth, krylov_depth)
    assert r.shape == (nrows,)
    assert c.shape == ()

    # Tie the test-strictness to the floating point accuracy
    small_value = np.sqrt(np.finfo_eps(np.dtype(H)))
    tols = {"atol": small_value, "rtol": small_value}

    # Test the decompositions
    e0, ek = np.eye(krylov_depth)[[0, -1], :]
    assert np.allclose(A @ Q - Q @ H - linalg.outer(r, ek), 0.0, **tols)
    assert np.allclose(Q.T.conj() @ Q - np.eye(krylov_depth), 0.0, **tols)
    assert np.allclose(Q @ e0, c * v, **tols)


@testing.parametrize("nrows", [10])
@testing.parametrize("krylov_depth", [1, 5, 10])
@testing.parametrize("reortho", ["full"])
def test_reorthogonalisation_improves_the_estimate(nrows, krylov_depth, reortho):
    # Create an ill-conditioned test-matrix (that requires reortho=True)
    A = linalg.hilbert(nrows)
    v = prng.normal(prng.prng_key(2), shape=(nrows,))

    # Decompose
    algorithm = decomp.hessenberg(krylov_depth, reortho=reortho)
    Q, H, r, c = algorithm(lambda s, p: p @ s, v, A)

    # Assert shapes
    assert Q.shape == (nrows, krylov_depth)
    assert H.shape == (krylov_depth, krylov_depth)
    assert r.shape == (nrows,)
    assert c.shape == ()

    # Tie the test-strictness to the floating point accuracy
    small_value = np.sqrt(np.finfo_eps(np.dtype(H)))
    tols = {"atol": small_value, "rtol": small_value}

    # Test the decompositions
    e0, ek = np.eye(krylov_depth)[[0, -1], :]
    assert np.allclose(A @ Q - Q @ H - linalg.outer(r, ek), 0.0, **tols)
    assert np.allclose(Q.T @ Q - np.eye(krylov_depth), 0.0, **tols)
    assert np.allclose(Q @ e0, c * v, **tols)


def test_raises_error_for_wrong_depth_too_small():
    algorithm = decomp.hessenberg(0, reortho="none")
    with testing.raises(ValueError, match="depth"):
        _ = algorithm(lambda s: s, np.ones((2,)))


def test_raises_error_for_wrong_depth_too_high():
    algorithm = decomp.hessenberg(3, reortho="none")
    with testing.raises(ValueError, match="depth"):
        _ = algorithm(lambda s: s, np.ones((2,)))


@testing.parametrize("reortho_wrong", [True, "full_with_sparsity", "None"])
def test_raises_error_for_wrong_reorthogonalisation_flag(reortho_wrong):
    with testing.raises(TypeError, match="Unexpected input"):
        _ = decomp.hessenberg(1, reortho=reortho_wrong)
