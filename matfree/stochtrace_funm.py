"""Stochastic estimation of traces of **functions of matrices**.

This module extends [matfree.stochtrace][matfree.stochtrace].

"""

from matfree import lanczos
from matfree.backend import func, linalg, np, tree_util

# todo: currently, all dense matrix-functions are computed
#  via eigh(). But for e.g. log and exp, we might want to do
#  something else.


def integrand_sym_logdet(order, matvec, /):
    """Construct the integrand for the log-determinant.

    This function assumes a symmetric, positive definite matrix.
    """
    return integrand_sym(np.log, order, matvec)


def integrand_sym(matfun, order, matvec, /):
    """Construct the integrand for matrix-function-trace estimation.

    This function assumes a symmetric matrix.
    """

    def quadform(v0, *parameters):
        v0_flat, v_unflatten = tree_util.ravel_pytree(v0)
        length = linalg.vector_norm(v0_flat)
        v0_flat /= length

        def matvec_flat(v_flat, *p):
            v = v_unflatten(v_flat)
            Av = matvec(v, *p)
            flat, unflatten = tree_util.ravel_pytree(Av)
            return flat

        algorithm = lanczos.alg_tridiag_full_reortho(matvec_flat, order)
        _, (diag, off_diag) = algorithm(v0_flat, *parameters)
        eigvals, eigvecs = _eigh_tridiag(diag, off_diag)

        # Since Q orthogonal (orthonormal) to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        fx_eigvals = func.vmap(matfun)(eigvals)
        return length**2 * linalg.vecdot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform


def integrand_product_logdet(depth, matvec, vecmat, /):
    r"""Construct the integrand for the log-determinant of a matrix-product.

    Here, "product" refers to $X = A^\top A$.
    """
    return integrand_product(np.log, depth, matvec, vecmat)


def integrand_product_schatten_norm(power, depth, matvec, vecmat, /):
    r"""Construct the integrand for the p-th power of the Schatten-p norm."""

    def matfun(x):
        """Matrix-function for Schatten-p norms."""
        return x ** (power / 2)

    return integrand_product(matfun, depth, matvec, vecmat)


def integrand_product(matfun, depth, matvec, vecmat, /):
    """Construct the integrand for matrix-function-trace estimation.

    Instead of the trace of a function of a matrix,
    compute the trace of a function of the product of matrices.
    Here, "product" refers to $X = A^\top A$.
    """

    def quadform(v0, *parameters):
        v0_flat, v_unflatten = tree_util.ravel_pytree(v0)
        length = linalg.vector_norm(v0_flat)
        v0_flat /= length

        def matvec_flat(v_flat, *p):
            v = v_unflatten(v_flat)
            Av = matvec(v, *p)
            flat, unflatten = tree_util.ravel_pytree(Av)
            return flat, tree_util.partial_pytree(unflatten)

        w0_flat, w_unflatten = func.eval_shape(matvec_flat, v0_flat)
        matrix_shape = (*np.shape(w0_flat), *np.shape(v0_flat))

        def vecmat_flat(w_flat):
            w = w_unflatten(w_flat)
            wA = vecmat(w, *parameters)
            return tree_util.ravel_pytree(wA)[0]

        # Decompose into orthogonal-bidiag-orthogonal
        algorithm = lanczos.alg_bidiag_full_reortho(
            lambda v: matvec_flat(v)[0], vecmat_flat, depth, matrix_shape=matrix_shape
        )
        output = algorithm(v0_flat, *parameters)
        u, (d, e), vt, _ = output

        # Compute SVD of factorisation
        B = _bidiagonal_dense(d, e)
        _, S, Vt = linalg.svd(B, full_matrices=False)

        # Since Q orthogonal (orthonormal) to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        eigvals, eigvecs = S**2, Vt.T
        fx_eigvals = func.vmap(matfun)(eigvals)
        return length**2 * linalg.vecdot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform


def _bidiagonal_dense(d, e):
    diag = linalg.diagonal_matrix(d)
    offdiag = linalg.diagonal_matrix(e, 1)
    return diag + offdiag


def _eigh_tridiag(diag, off_diag):
    # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
    #  use it here. Until then: an eigen-decomposition of size (order + 1)
    #  does not hurt too much...
    diag = linalg.diagonal_matrix(diag)
    offdiag1 = linalg.diagonal_matrix(off_diag, -1)
    offdiag2 = linalg.diagonal_matrix(off_diag, 1)
    dense_matrix = diag + offdiag1 + offdiag2
    eigvals, eigvecs = linalg.eigh(dense_matrix)
    return eigvals, eigvecs
