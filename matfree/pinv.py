"""Pseudo-inverses of linear operators."""

import warnings

from matfree.backend.typing import Callable


def _warn_deprecated():
    msg = "The module matfree.pinv has been deprecated and will be removed soon. "
    msg += "The removal will happen either in v0.0.17 or in v0.1.0, "
    msg += "or on the 15th of June 2024, "
    msg += "depending on which of the three comes first. "
    msg += "If your code relies on matfree.pinv, create an issue *now*."

    warnings.warn(msg, DeprecationWarning, stacklevel=1)


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
    _warn_deprecated()

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
    _warn_deprecated()

    def pinv(s):
        return solve(lambda v: vA(Av(v)), vA(s))

    return pinv
