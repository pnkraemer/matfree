"""Stochastic Lanczos quadrature."""

from matfree import decomp, hutch
from matfree.backend import func, linalg, np


def trace_of_matfun(matfun, Av, order, /, **kwargs):
    """Compute the trace of the function of a matrix.

    For example, logdet(M) = trace(log(M)) ~ trace(U log(D) Ut) = E[v U log(D) Ut vt].
    """
    quadratic_form = quadratic_form_slq(matfun, Av, order)
    return hutch.stochastic_estimate(quadratic_form, **kwargs)


def quadratic_form_slq(matfun, Av, order, /):
    """Approximate quadratic form for stochastic Lanczos quadrature."""

    def quadform(v0, /):
        algorithm = decomp.lanczos(order)
        _, tridiag = decomp.decompose_fori_loop(0, order + 1, Av, v0, alg=algorithm)
        (diag, off_diag) = tridiag

        # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
        #  use it here. Until then: an eigen-decomposition of size (order + 1)
        #  does not hurt too much...
        dense_matrix = (
            np.diagonal(diag) + np.diagonal(off_diag, -1) + np.diagonal(off_diag, 1)
        )
        eigvals, eigvecs = linalg.eigh(dense_matrix)

        # Since Q orthogonal (orthonormal) to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        (dim,) = v0.shape

        fx_eigvals = func.vmap(matfun)(eigvals)
        return dim * np.vecdot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform
