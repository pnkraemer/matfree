"""Pseudo-inverses."""


def pinv_tall(matvec, vecmat, *, solve):
    """Moore-Penrose pseudo-inverse.

    Implemented as a left-inverse: M = A^* (A A*)^{-1}
    which is applicable if A has full column-rank.
    """

    def pinv(s):
        return vecmat(solve(lambda v: matvec(vecmat(v)), s))

    return pinv


def pinv_wide(matvec, vecmat, *, solve):
    """Moore-Penrose pseudo-inverse.

    Implemented as a right-inverse: M = (A* A)^{-1} A^*
    which is applicable if A has full row-rank.
    """

    def pinv(s):
        return solve(lambda v: vecmat(matvec(v)), vecmat(s))

    return pinv
