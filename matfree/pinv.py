"""Pseudo-inverses of linear operators."""

from matfree.backend.typing import Callable

# todo: what happens if we want to reuse e.g. LU decompositions of AA^*?
# todo: should we not just return pinv-matvec but also pinv-vecmat?


def pinv_tall(Av: Callable, vA: Callable, *, solve: Callable) -> Callable:
    r"""Construct the (Moore-Penrose) pseudo-inverse of a full-rank, ''tall'' matrix.

    Implemented as a left-inverse, $A^\dagger := A^* (A A^*)^{-1}$,
    which is applicable if $A$ has full **column**-rank.

    Parameters
    ----------
    Av:
        Matrix-vector product function.
    vA:
        Vector-matrix product function.
    solve:
        Solution of a linear system. Maps a matrix-vector production function and a
        right-hand-side vector to the solution of the linear system.

    Returns
    -------
    Callable
        Matrix-vector product function that implements
        the pseudo-inverse of the matrix-vector product function `Av`.
    """

    def pinv(s):
        return vA(solve(lambda v: Av(vA(v)), s))

    return pinv


def pinv_wide(Av: Callable, vA: Callable, *, solve: Callable) -> Callable:
    r"""Construct the (Moore-Penrose) pseudo-inverse of a full-rank, ''wide'' matrix.

    Implemented as a right-inverse, $A^\dagger := (A^* A)^{-1} A^*$,
    which is applicable if $A$ has full **row**-rank.

    Parameters
    ----------
    Av:
        Matrix-vector product function.
    vA:
        Vector-matrix product function.
    solve:
        Solution of a linear system. Maps a matrix-vector production function and a
        right-hand-side vector to the solution of the linear system.

    Returns
    -------
    Callable
        Matrix-vector product function that implements
        the pseudo-inverse of the matrix-vector product function `Av`.
    """

    def pinv(s):
        return solve(lambda v: vA(Av(v)), vA(s))

    return pinv
