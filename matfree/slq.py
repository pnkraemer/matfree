"""Stochastic Lanczos quadrature."""

from matfree import decomp, montecarlo
from matfree.backend import func, linalg, np, prng


def trace_of_matfun(
    matfun,
    Av,
    order,
    /,
    *,
    key,
    num_samples_per_batch,
    num_batches,
    tangents_shape,
    tangents_dtype,
    sample_fun=prng.normal,
):
    """Compute the trace of the function of a matrix.

    For example, logdet(M) = trace(log(M)) ~ trace(U log(D) Ut) = E[v U log(D) Ut vt].
    """

    def sample(k, /):
        return sample_fun(k, shape=tangents_shape, dtype=tangents_dtype)

    quadform = quadratic_form_slq(matfun, Av, order)
    quadform_mc = montecarlo.montecarlo(quadform, sample_fun=sample)

    quadform_batch = montecarlo.mean_vmap(quadform_mc, num_samples_per_batch)
    quadform_batch = montecarlo.mean_map(quadform_batch, num_batches)
    return quadform_batch(key)


def quadratic_form_slq(matfun, Av, order, /):
    """Approximate quadratic form for stochastic Lanczos quadrature."""

    def quadform(v0, /):
        algorithm = decomp.lanczos(order)
        _, tridiag = decomp.decompose_fori_loop(0, order + 1, Av, v0, alg=algorithm)
        (diag, off_diag) = tridiag

        # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
        #  use it here. Until then: an eigen-decomposition of size (order + 1)
        #  does not hurt too much...
        dense_matrix = (
            np.diagonal(diag) + np.diagonal(off_diag, -1) + np.diagonal(off_diag, 1)
        )
        eigvals, eigvecs = linalg.eigh(dense_matrix)

        # Since Q orthogonal (orthonormal) to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        (dim,) = v0.shape

        fx_eigvals = func.vmap(matfun)(eigvals)
        return dim * np.vecdot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform
