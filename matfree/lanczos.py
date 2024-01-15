"""All things Lanczos' algorithm."""

from matfree import decomp
from matfree.backend import func, linalg


def funm_vector_product_spd(matfun, order, matvec, /):
    """Implement a matrix-function-vector product via Lanczos' algorithm.

    This algorithm uses Lanczos' tridiagonalisation with full re-orthogonalisation
    and therefore applies only to symmetric, positive definite matrices.
    """
    # Lanczos' algorithm
    algorithm = decomp.lanczos_tridiag_full_reortho(order)

    def estimate(vec, *parameters):
        def matvec_p(v):
            return matvec(v, *parameters)

        length = linalg.vector_norm(vec)
        vec /= length
        basis, tridiag = decomp.decompose_fori_loop(vec, matvec_p, algorithm=algorithm)
        (diag, off_diag) = tridiag

        # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
        #  use it here. Until then: an eigen-decomposition of size (order + 1)
        #  does not hurt too much...
        diag = linalg.diagonal_matrix(diag)
        offdiag1 = linalg.diagonal_matrix(off_diag, -1)
        offdiag2 = linalg.diagonal_matrix(off_diag, 1)
        dense_matrix = diag + offdiag1 + offdiag2
        eigvals, eigvecs = linalg.eigh(dense_matrix)

        fx_eigvals = func.vmap(matfun)(eigvals)
        return length * (basis.T @ (eigvecs @ (fx_eigvals * eigvecs[0, :])))

    return estimate
