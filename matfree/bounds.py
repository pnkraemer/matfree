"""Bounds on functions of matrices."""

from matfree.backend import linalg, np


def baigolub96_logdet_spd(bound_spectrum, /, nrows, trace, norm_frobenius_squared):
    """Bound the log-determinant of a symmatric, positive definite matrix.

    This function implements Theorem 2 in the paper by Bai and Golub (1996).

    ``bound_spectrum`` is either an upper or a lower bound
    on the spectrum of the matrix.
    If it is an upper bound,
    the function returns an upper bound of the log-determinant.
    If it is a lower bound,
    the function returns a lower bound of the log-determinant.



    ??? note "BibTex for Bai and Golub (1996)"
        ```tex
        @article{bai1996bounds,
            title={Bounds for the trace of the inverse and the
            determinant of symmetric positive definite matrices},
            author={Bai, Zhaojun and Golub, Gene H},
            journal={Annals of Numerical Mathematics},
            volume={4},
            pages={29--38},
            year={1996},
            publisher={Citeseer}
        }
        ```

    """
    mu1, mu2 = trace, norm_frobenius_squared
    beta = bound_spectrum
    tbar = (beta * mu1 - mu2) / (beta * nrows - mu1)
    v = np.asarray([np.log(beta), np.log(tbar)])
    w = np.asarray([mu1, mu2])
    A = np.asarray([[beta, tbar], [beta**2, tbar**2]])
    return v @ linalg.solve(A, w)
