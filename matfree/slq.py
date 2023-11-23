"""Stochastic Lanczos quadrature.

The [slq][matfree.slq] module extends the integrands
in [hutchinson][matfree.hutchinson] to those that implement
stochastic Lanczos quadrature.
"""

from matfree import decomp
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

    def quadform(v0, /):
        v0_flat, v_unflatten = tree_util.ravel_pytree(v0)

        def matvec_flat(v_flat):
            v = v_unflatten(v_flat)
            Av = matvec(v)
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
        (dim,) = v0_flat.shape

        fx_eigvals = func.vmap(matfun)(eigvals)
        return dim * linalg.vecdot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform


def integrand_logdet_product(depth, matvec, vecmat, /):
    r"""Construct the integrand for the log-determinant of a matrix-product.

    Here, "product" refers to $X = A^\top A$.
    """
    return integrand_slq_product(np.log, depth, matvec, vecmat)


def integrand_schatten_norm(power, depth, matvec, vecmat, /):
    r"""Construct the integrand for the p-th power of the Schatten-p norm."""

    def matfun(x):
        """Matrix-function for Schatten-p norms."""
        return x ** (power / 2)

    return integrand_slq_product(matfun, depth, matvec, vecmat)


def integrand_slq_product(matfun, depth, matvec, vecmat, /):
    r"""Construct the integrand for the trace of a function of a matrix-product.

    Instead of the trace of a function of a matrix,
    compute the trace of a function of the product of matrices.
    Here, "product" refers to $X = A^\top A$.
    """

    def quadform(v0, /):
        v0_flat, v_unflatten = tree_util.ravel_pytree(v0)

        def matvec_flat(v_flat):
            v = v_unflatten(v_flat)
            Av = matvec(v)
            flat, unflatten = tree_util.ravel_pytree(Av)
            return flat, tree_util.partial_pytree(unflatten)

        w0_flat, w_unflatten = func.eval_shape(matvec_flat, v0_flat)
        matrix_shape = (*np.shape(w0_flat), *np.shape(v0_flat))

        def vecmat_flat(w_flat):
            w = w_unflatten(w_flat)
            wA = vecmat(w)
            return tree_util.ravel_pytree(wA)[0]

        # Decompose into orthogonal-bidiag-orthogonal
        algorithm = decomp.lanczos_bidiag_full_reortho(depth, matrix_shape=matrix_shape)
        output = decomp.decompose_fori_loop(
            v0_flat, lambda v: matvec_flat(v)[0], vecmat_flat, algorithm=algorithm
        )
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
