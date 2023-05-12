"""Stochastic Lanczos quadrature."""

from matfree import decomp, montecarlo
from matfree.backend import func, linalg, np


def logdet_spd(*args, **kwargs):
    """Estimate the log-determinant of a symmetric, positive definite matrix."""
    return trace_of_matfun_spd(np.log, *args, **kwargs)


def trace_of_matfun_spd(matfun, Av, order, /, **kwargs):
    """Compute the trace of the function of a symmetric matrix."""
    quadratic_form = quadratic_form_slq_spd(matfun, Av, order)
    return montecarlo.estimate(quadratic_form, **kwargs)


def quadratic_form_slq_spd(matfun, Av, order, /):
    """Quadratic form for stochastic Lanczos quadrature.

    Assumes a symmetric, positive definite matrix.
    """

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


def logdet_product(*args, **kwargs):
    r"""Compute the log-determinant of a product of matrices.

    Here, "product" refers to $X = A^\top A$.
    """
    return trace_of_matfun_product(np.log, *args, **kwargs)


def trace_of_matfun_product(matfun, order, *matvec_funs, matrix_shape, **kwargs):
    r"""Compute the trace of a function of a product of matrices.

    Here, "product" refers to $X = A^\top A$.
    """
    quadratic_form = quadratic_form_slq_product(
        matfun, order, *matvec_funs, matrix_shape=matrix_shape
    )
    return montecarlo.estimate(quadratic_form, **kwargs)


def quadratic_form_slq_product(matfun, depth, *matvec_funs, matrix_shape):
    r"""Quadratic form for stochastic Lanczos quadrature.

    "Product" equivalent of
    [matfree.slq.quadratic_form_slq_spd(...)][matfree.slq.quadratic_form_slq_spd].

    Instead of the trace of a function of a matrix,
    compute the trace of a function of the product of matrices.
    Here, "product" refers to $X = A^\top A$.

    """

    def quadform(v0, /):
        # Decompose into orthogonal-bidiag-orthogonal
        algorithm = decomp.gkl_full_reortho(depth, matrix_shape=matrix_shape)
        output = decomp.decompose_fori_loop(v0, *matvec_funs, algorithm=algorithm)
        u, (d, e), vt, _ = output

        # Compute SVD of factorisation
        B = _bidiagonal_dense(d, e)
        _, S, Vt = linalg.svd(B, full_matrices=False)

        # Since Q orthogonal (orthonormal) to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        _, ncols = matrix_shape
        eigvals, eigvecs = S**2, Vt.T
        fx_eigvals = func.vmap(matfun)(eigvals)
        return ncols * linalg.vecdot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform


def _bidiagonal_dense(d, e):
    diag = linalg.diagonal_matrix(d)
    offdiag = linalg.diagonal_matrix(e, 1)
    return diag + offdiag
