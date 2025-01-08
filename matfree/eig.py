"""Matrix-free eigenvalue and singular-value analysis."""

from matfree.backend import func, linalg, tree
from matfree.backend.typing import Array, Callable


def svd_partial(bidiag: Callable) -> Callable:
    """Partial singular value decomposition.

    Combines bidiagonalisation with a full SVD of the (small) bidiagonal matrix.

    Parameters
    ----------
    bidiag:
        An implementation of bidiagonalisation.
        For example, the output of
        [decomp.bidiag][matfree.decomp.bidiag].
        Note how this function assumes that the bidiagonalisation
        materialises the bidiagonal matrix.

    """

    def svd(Av: Callable, v0: Array, *parameters):
        # Flatten in- and outputs
        Av_flat, flattened = _partial_and_flatten_matvec(Av, v0, *parameters)
        (u0_flat, u_unravel), (v0_flat, v_unravel) = flattened

        # Call the flattened SVD
        ut, s, vt = svd_flat(Av_flat, v0_flat)

        # Unravel the singular vectors
        ut_tree = func.vmap(u_unravel)(ut)
        vt_tree = func.vmap(v_unravel)(vt)
        return ut_tree, s, vt_tree

    def svd_flat(Av: Callable, v0: Array):
        # Factorise the matrix
        (u, v), B, *_ = bidiag(Av, v0)

        # Compute SVD of factorisation
        U, S, Vt = linalg.svd(B, full_matrices=False)

        # Combine orthogonal transformations
        return (u @ U).T, S, Vt @ v.T

    return svd


def eigh_partial(tridiag_sym: Callable) -> Callable:
    """Partial symmetric/Hermitian eigenvalue decomposition.

    Combines tridiagonalization with a decomposition
    of the (small) tridiagonal matrix.

    Parameters
    ----------
    tridiag_sym:
        An implementation of tridiagonalization.
        For example, the output of
        [decomp.tridiag_sym][matfree.decomp.tridiag_sym].

    """

    def eigh(Av: Callable, v0: Array, *parameters):
        # Flatten in- and outputs
        Av_flat, flattened = _partial_and_flatten_matvec(Av, v0, *parameters)
        _, (v0_flat, v_unravel) = flattened

        # Call the flattened eigh
        vals, vecs = eigh_flat(Av_flat, v0_flat)

        # Unravel the eigenvectors
        vecs = func.vmap(v_unravel)(vecs)
        return vals, vecs

    def eigh_flat(Av: Callable, v0: Array):
        # Factorise the matrix
        Q, H, *_ = tridiag_sym(Av, v0)

        # Compute eigh of factorisation
        vals, vecs = linalg.eigh(H)
        vecs = Q @ vecs
        return vals, vecs.T

    return eigh


def eig_partial(hessenberg: Callable) -> Callable:
    """Partial eigenvalue decomposition.

    Combines Hessenberg factorisation with a decomposition
    of the (small) Hessenberg matrix.

    Parameters
    ----------
    hessenberg:
        An implementation of Hessenberg factorisation.
        For example, the output of
        [decomp.hessenberg][matfree.decomp.hessenberg].

    """

    def eig(Av: Callable, v0: Array, *parameters):
        # Flatten in- and outputs
        Av_flat, flattened = _partial_and_flatten_matvec(Av, v0, *parameters)
        _, (v0_flat, v_unravel) = flattened

        # Call the flattened eig
        vals, vecs = eig_flat(Av_flat, v0_flat)

        # Unravel the eigenvectors
        vecs = func.vmap(v_unravel)(vecs)
        return vals, vecs

    def eig_flat(Av: Callable, v0: Array):
        # Factorise the matrix
        Q, H, *_ = hessenberg(Av, v0)

        # Compute eig of factorisation
        vals, vecs = linalg.eig(H)
        vecs = Q @ vecs
        return vals, vecs.T

    return eig


def _partial_and_flatten_matvec(Av, v0, *parameters):
    # Evaluate the output shape of Av and flatten
    u0 = func.eval_shape(lambda z: Av(z, *parameters), v0)

    # Flatten in- and outputs
    u0_flat, u_unravel = func.eval_shape(tree.ravel_pytree, u0)
    v0_flat, v_unravel = tree.ravel_pytree(v0)

    def Av_flat(v_flat):
        """Evaluate a flattened matvec."""
        result = Av(v_unravel(v_flat), *parameters)
        result_flat, _ = tree.ravel_pytree(result)
        return result_flat

    return Av_flat, ((u0_flat, u_unravel), (v0_flat, v_unravel))
