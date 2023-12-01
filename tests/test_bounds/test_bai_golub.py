"""Tests for Bai and Golub's log-determinant bounds."""

from matfree import bounds, test_util
from matfree.backend import linalg, np


def test_logdet():
    """Test that Bai and Golub's log-determinant bound is correct."""
    # Set up a test-problem.
    eigvals = np.asarray([1.0, 2.0, 3.0, 4.0])
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    # Compute the bound
    trace = linalg.trace(matrix)
    fronorm = linalg.matrix_norm(matrix, which="fro") ** 2
    eigvals_bounds = (0.9, 4.1)
    lower = bounds.baigolub96_logdet_spd(eigvals_bounds[0], 4, trace, fronorm)
    upper = bounds.baigolub96_logdet_spd(eigvals_bounds[1], 4, trace, fronorm)

    # Reference
    _sign, logdet = linalg.slogdet(matrix)

    # Assert that the bounds are satisfied,
    # but also that they are somewhat tight
    # (because we use tight eigenvalue bounds)
    assert 0.9 * logdet < lower < logdet < upper < 1.1 * logdet
