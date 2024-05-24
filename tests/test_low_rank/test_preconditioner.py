"""Test preconditioning with partial Cholesky decompositions."""

from matfree import low_rank, test_util
from matfree.backend import linalg, np


def test_preconditioner_solves_correctly(n=10):
    # Create a relatively ill-conditioned matrix
    cov_eig = 1.5 ** np.arange(-n // 2, n // 2, step=1.0)
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    def element(i, j):
        return cov[i, j]

    # Assert that the Cholesky decomposition is full-rank.
    # This is important to ensure that the test below makes sense.
    cholesky = low_rank.cholesky_partial(element, nrows=n, rank=n)
    matrix, _info = cholesky()
    assert np.allclose(matrix @ matrix.T, cov)

    # Set up the test-problem
    small_value = 1e-1
    b = np.arange(1.0, 1 + len(cov))
    b /= linalg.vector_norm(b)

    # Solve the linear system
    cov_added = cov + small_value * np.eye(len(cov))
    expected = linalg.solve(cov_added, b)

    # Derive the preconditioner
    precondition = low_rank.preconditioner(cholesky)
    received, info = precondition(b, small_value)

    # Test that the preconditioner solves correctly
    assert np.allclose(received, expected, rtol=1e-2)
