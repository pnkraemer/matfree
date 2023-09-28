"""Pseudo-inverses."""

from matfree.backend.typing import Callable

# todo: what happens if we want to reuse e.g. LU decompositions of AA^*?
# todo: should we not just return pinv-matvec but also pinv-vecmat?


def pinv_tall(matvec: Callable, vecmat: Callable, *, solve: Callable) -> Callable:
    r"""Construct the (Moore-Penrose) pseudo-inverse of a full-rank, ''tall'' matrix.

    Implemented as a left-inverse, $A^\dagger := A^* (A A^*)^{-1}$,
    which is applicable if $A$ has full **column**-rank.
    """

    def pinv(s):
        return vecmat(solve(lambda v: matvec(vecmat(v)), s))

    return pinv


def pinv_wide(matvec: Callable, vecmat: Callable, *, solve: Callable) -> Callable:
    r"""Construct the (Moore-Penrose) pseudo-inverse of a full-rank, ''wide'' matrix.

    Implemented as a right-inverse, $A^\dagger := (A^* A)^{-1} A^*$,
    which is applicable if $A$ has full **row**-rank.
    """

    def pinv(s):
        return solve(lambda v: vecmat(matvec(v)), vecmat(s))

    return pinv
