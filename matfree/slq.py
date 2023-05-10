"""Stochastic Lanczos quadrature."""

from matfree import decomp, montecarlo
from matfree.backend import func, linalg, np


def logdet(*args, **kwargs):
    """Estimate the log-determinant of a symmetric, positive definite matrix."""
    return trace_of_matfun_symmetric(np.log, *args, **kwargs)


# todo: nuclear norm, schatten-p norms.
#  But for this we should use bi-diagonalisation


def trace_of_matfun_symmetric(matfun, Av, order, /, **kwargs):
    """Compute the trace of the function of a symmetric matrix."""
    quadratic_form = quadratic_form_slq_symmetric(matfun, Av, order)
    return montecarlo.estimate(quadratic_form, **kwargs)


def quadratic_form_slq_symmetric(matfun, Av, order, /):
    """Approximate quadratic form for stochastic Lanczos quadrature."""

    def quadform(v0, /):
        algorithm = decomp.lanczos_full_reortho(order)
        _, tridiag = decomp.decompose_fori_loop(v0, Av, algorithm=algorithm)
        (diag, off_diag) = tridiag

        # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
        #  use it here. Until then: an eigen-decomposition of size (order + 1)
        #  does not hurt too much...
        diag = linalg.diagonal_matrix(diag)
        offdiag1 = linalg.diagonal_matrix(off_diag, -1)
        offdiag2 = linalg.diagonal_matrix(off_diag, 1)
        dense_matrix = diag + offdiag1 + offdiag2
        eigvals, eigvecs = linalg.eigh(dense_matrix)

        # Since Q orthogonal (orthonormal) to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        (dim,) = v0.shape

        fx_eigvals = func.vmap(matfun)(eigvals)
        return dim * linalg.vecdot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform
