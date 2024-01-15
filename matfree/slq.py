"""Stochastic Lanczos quadrature.

The [slq][matfree.slq] module extends the integrands
in [hutchinson][matfree.hutchinson] to those that implement
stochastic Lanczos quadrature.
"""

from matfree import lanczos
from matfree.backend import func, linalg, np, tree_util


def integrand_logdet_spd(order, matvec, /):
    """Construct the integrand for the log-determinant.

    This function assumes a symmetric, positive definite matrix.
    """
    return integrand_slq_spd(np.log, order, matvec)


def integrand_slq_spd(matfun, order, matvec, /):
    """Quadratic form for stochastic Lanczos quadrature.

    This function assumes a symmetric, positive definite matrix.
    """

    def quadform(v0, *parameters):
        v0_flat, v_unflatten = tree_util.ravel_pytree(v0)
        length = linalg.vector_norm(v0_flat)
        v0_flat /= length

        def matvec_flat(v_flat):
            v = v_unflatten(v_flat)
            Av = matvec(v, *parameters)
            flat, unflatten = tree_util.ravel_pytree(Av)
            return flat

        algorithm = decomp.lanczos_tridiag_full_reortho(order)
        _, tridiag = decomp.decompose_fori_loop(
            v0_flat, matvec_flat, algorithm=algorithm
        )
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
        fx_eigvals = func.vmap(matfun)(eigvals)
        return length**2 * linalg.vecdot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform
