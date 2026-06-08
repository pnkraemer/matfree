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

    def matvec(s, p):
        [(x,)] = s
        return [(p @ x,)]

    # Run Lanczos approximation
    algorithm = decomp.tridiag_sym(ndim, reortho=reortho, materialize=True)
    Q_pytree, T, *_ = algorithm(matvec, [(vector,)], matrix)
    [(Q,)] = Q_pytree

    # Reconstruct the original matrix from the full-num_matvecs approximation
    # Q shape is (k, n) -- rows are Krylov vectors
    matrix_reconstructed = Q.T @ T @ Q

    if reortho == "full":
        tols = {"atol": 1e-5, "rtol": 1e-5}
    else:
        tols = {"atol": 1e-1, "rtol": 1e-1}

    # Assert the reconstruction was "exact"
    assert np.allclose(matrix_reconstructed, matrix, **tols)

    # Assert all vectors are orthogonal (Q is square for full-rank)
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

    def matvec(s, p):
        [(x,)] = s
        return [(p @ x,)]

    # Run Lanczos approximation
    algorithm = decomp.tridiag_sym(num_matvecs, reortho=reortho, materialize=True)
    Q_pytree, T, q_pytree, _n = algorithm(matvec, [(vector,)], matrix)
    [(Q,)] = Q_pytree
    [(q,)] = q_pytree

    # Verify the decomposition: A Q.T = Q.T T + q e_K^T
    # Q shape (k, n), Q.T shape (n, k)
    e_K = np.eye(num_matvecs)[-1]
    test_util.assert_allclose(matrix @ Q.T - Q.T @ T - linalg.outer(q, e_K), 0.0)
