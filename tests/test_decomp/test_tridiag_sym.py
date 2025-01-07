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
    Q, T, *_ = algorithm(lambda s, p: p @ s, vector, matrix)

    # Reconstruct the original matrix from the full-num_matvecs approximation
    matrix_reconstructed = Q @ T @ Q.T

    if reortho == "full":
        tols = {"atol": 1e-5, "rtol": 1e-5}
    else:
        tols = {"atol": 1e-1, "rtol": 1e-1}

    # Assert the reconstruction was "exact"
    assert np.allclose(matrix_reconstructed, matrix, **tols)

    # Assert all vectors are orthogonal
    test_util.assert_columns_orthonormal(Q)
    test_util.assert_columns_orthonormal(Q.T)


# anything 0 <= k < n works; k=n is full reconstruction
# and the (q, b) values become meaningless
@testing.parametrize("num_matvecs", [1, 5, 11])
@testing.parametrize("ndim", [12])
@testing.parametrize("reortho", ["full", "none"])
def test_mid_rank_reconstruction_satisfies_decomposition(ndim, num_matvecs, reortho):
    # Set up a test-matrix and an initial vector
    eigvals = np.arange(1.0, 2.0, step=1 / ndim)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    vector = np.flip(np.arange(1.0, 1.0 + len(eigvals)))

    # Run Lanczos approximation
    algorithm = decomp.tridiag_sym(num_matvecs, reortho=reortho, materialize=True)
    Q, T, q, _n = algorithm(lambda s, p: p @ s, vector, matrix)

    # Verify the decomposition
    e_K = np.eye(num_matvecs)[-1]
    test_util.assert_allclose(matrix @ Q, Q @ T + linalg.outer(q, e_K))
