"""Test the tri-diagonalisation."""

from matfree import decomp, test_util
from matfree.backend import linalg, np, testing


@testing.parametrize("reortho", ["full", "none"])
@testing.parametrize("ndim", [12])
def test_full_rank_reconstruction_is_exact(reortho, ndim):
    # Set up a test-matrix and an initial vector
    eigvals = np.arange(1.0, 2.0, step=1 / ndim)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    vector = np.flip(np.arange(1.0, 1.0 + len(eigvals)))

    # Run Lanczos approximation
    algorithm = decomp.tridiag_sym(ndim, reortho=reortho, materialize=True)
    (lanczos_vectors, dense_matrix), _ = algorithm(lambda s, p: p @ s, vector, matrix)

    # Reconstruct the original matrix from the full-order approximation
    matrix_reconstructed = lanczos_vectors.T @ dense_matrix @ lanczos_vectors

    if reortho == "full":
        tols = {"atol": 1e-5, "rtol": 1e-5}
    else:
        tols = {"atol": 1e-1, "rtol": 1e-1}

    # Assert the reconstruction was "exact"
    assert np.allclose(matrix_reconstructed, matrix, **tols)

    # Assert all vectors are orthogonal
    eye = np.eye(len(lanczos_vectors))
    assert np.allclose(lanczos_vectors @ lanczos_vectors.T, eye, **tols)
    assert np.allclose(lanczos_vectors.T @ lanczos_vectors, eye, **tols)


# anything 0 <= k < n works; k=n is full reconstruction
# and the (q, b) values become meaningless
@testing.parametrize("krylov_depth", [1, 5, 11])
@testing.parametrize("ndim", [12])
@testing.parametrize("reortho", ["full", "none"])
def test_mid_rank_reconstruction_satisfies_decomposition(ndim, krylov_depth, reortho):
    # Set up a test-matrix and an initial vector
    eigvals = np.arange(1.0, 2.0, step=1 / ndim)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    vector = np.flip(np.arange(1.0, 1.0 + len(eigvals)))

    # Run Lanczos approximation
    algorithm = decomp.tridiag_sym(krylov_depth, reortho=reortho, materialize=True)
    (Q, T), (q, b) = algorithm(lambda s, p: p @ s, vector, matrix)

    # Verify the decomposition
    tols = {"atol": 1e-5, "rtol": 1e-5}
    e_K = np.eye(krylov_depth)[-1]
    assert np.allclose(matrix @ Q.T, Q.T @ T + linalg.outer(e_K, q * b).T, **tols)
