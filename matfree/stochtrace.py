"""Stochastic estimation of traces, diagonals, and more."""

from matfree.backend import control_flow, func, linalg, np, prng, tree
from matfree.backend.typing import Callable


def estimator(integrand: Callable, /, sampler: Callable) -> Callable:
    """Construct a stochastic trace-/diagonal-estimator.

    Parameters
    ----------
    integrand
        The integrand function. For example, the return-value of
        [integrand_trace][matfree.stochtrace.integrand_trace].
        But any other integrand works, too.
    sampler
        The sample function. Usually, either
        [sampler_normal][matfree.stochtrace.sampler_normal] or
        [sampler_rademacher][matfree.stochtrace.sampler_rademacher].

    Returns
    -------
    estimate
        A function that maps a random key to an estimate.
        This function can be compiled, vectorised, differentiated,
        or looped over as the user desires.

    """

    def estimate(matvecs, key, *parameters):
        samples = sampler(key)
        Qs = func.vmap(lambda vec: integrand(matvecs, vec, *parameters))(samples)
        return tree.tree_map(lambda s: np.mean(s, axis=0), Qs)

    return estimate


def estimator_leave_one_out(integrand: Callable, /, sampler: Callable) -> Callable:
    """Construct a leave-one-out stochastic estimator.

    Unlike :func:`estimator`, which vmaps an integrand over individual sample
    vectors, this variant passes the **full** test matrix to the integrand at
    once.  This is required for the XTrace family of algorithms, which need a
    batch QR or Cholesky decomposition before the second matvec phase.

    Parameters
    ----------
    integrand
        An integrand that accepts ``(matvec, Omega, *parameters)`` where
        ``Omega`` has shape ``(num, n)``.  The integrand is responsible for
        averaging over the sample axis internally.
    sampler
        The sample function, e.g. the return-value of
        :func:`sampler_normal`.

    Returns
    -------
    estimate
        A function ``estimate(matvec, key, *parameters) -> result``.

    """

    def estimate(matvec, key, *parameters):
        samples = sampler(key)
        return np.mean(integrand(matvec, samples, *parameters), axis=0)

    return estimate


def leave_one_out_xtrace(*, resphere: bool = True) -> Callable:
    """Construct an integrand for estimating the trace using the XTrace algorithm (Epperly et al. 2024).

    Parameters
    ----------
    resphere
        If ``True`` (default), apply resphering (see Epperly (2025)),
        which projects test vectors onto the range of the residual matrix, reducing
        the variance of the trace estimate. Requires test vectors drawn from a
        rotationally invariant distribution (e.g. Gaussian or sphere).

    Returns
    -------
    integrand
        An integrand function compatible with `estimator_leave_one_out` whose input
        has the signature ``(matvec, samples, *params)`` and whose output is a vector
        of trace estimates with shape ``(samples.shape[0],)``.

    Notes
    -----
    The number of samples must be less than or equal to the dimension of the operator.
    Additionally, the algorithm assumes that the samples are unique. For low-dimensional
    operators, samples generated from `sampler_rademacher` may violate this assumption, and
    it is recommended to use a different sampler instead.

    References
    ----------
    - Epperly EN, Tropp JA, Webber RJ (2024). Xtrace: Making the most of every sample in stochastic trace estimation.
        SIAM J Matrix Anal A. 45.1: 1-23.
        doi: [10.1137/23M1548323](https://doi.org/10.1137/23M1548323)
        arXiv: [2301.07825](https://arxiv.org/abs/2301.07825)
    - Epperly EN (2025). Make the most of what you have: Resource-efficient randomized algorithms for matrix computations. PhD Thesis.
        arXiv: [2512.15929](https://arxiv.org/abs/2512.15929)
    """

    def integrand(matvec, samples, *params):
        _, unflatten = tree.ravel_pytree(samples[0, :])
        Omega = func.vmap(lambda v: tree.ravel_pytree(v)[0])(samples).T
        n, num_samples = Omega.shape

        if num_samples > n:
            raise ValueError(_error_num_samples(num_samples, maxval=n, minval=1))

        def matvec_flat(v):
            return tree.ravel_pytree(matvec(unflatten(v), *params))[0]

        if num_samples == n:
            # It's faster and more accurate to compute the trace exactly and deterministically
            # when num_samples == n
            Omega = np.eye(n, dtype=Omega.dtype)
            return func.vmap(lambda v, i: matvec_flat(v)[i], in_axes=-1)(
                Omega, np.arange(n)
            )

        matmat = func.vmap(matvec_flat, in_axes=-1, out_axes=-1)

        Y = matmat(Omega)
        Q, R = linalg.qr_reduced(Y)
        Z = matmat(Q)

        def _trace_exact():
            tr_B = linalg.vdot(Q, Z)
            return np.ones((num_samples,), dtype=R.dtype) * tr_B

        def _trace_estimate():
            S = linalg.solve_triangular(R, np.eye(R.shape[0], dtype=R.dtype), trans=2)
            S = S / func.vmap(linalg.vector_norm, in_axes=-1)(S)

            Q_H = Q.T.conj()
            H = (
                Q_H @ Z
            )  # tr(H) == tr(B_hat), where B_hat = Q @ Q.H @ B is a low-rank approximation to the operator B
            W = Q_H @ Omega
            T = Z.T.conj() @ Omega
            W_vd_S = func.vmap(linalg.vdot, in_axes=1)(W, S)
            X = (
                W - S * W_vd_S.conj()
            )  # samples.T projected onto the subspace spanned by Q_i, i.e. Q formed leaving out samples[i, :]

            if resphere:
                # residual is B - B_hat_{-i}, where B_hat_{-i} approximates B leaving out samples[i, :]
                rank_residual = n - num_samples + 1
                # squared norm of each sample after projection to the subspace spanned by the residual
                sqnorm_samples_projected = np.sum(linalg.abs2(Omega), axis=0) - np.sum(
                    linalg.abs2(X), axis=0
                )
                sqnorm_samples_projected = np.where(
                    sqnorm_samples_projected == 0.0, 1.0, sqnorm_samples_projected
                )
                residual_scale = rank_residual / sqnorm_samples_projected
            else:
                residual_scale = 1.0

            tr_B_hat = linalg.trace(H)
            tr_B_hat_loo = tr_B_hat - func.vmap(linalg.vdot, in_axes=1)(
                S, H @ S
            )  # tr(B_hat) leaving out one sample
            tr_residual_loo = (  # Hutchinson estimate of tr(B - B_hat_{-i}) using as probe samples[i, :]
                -func.vmap(linalg.vdot, in_axes=1)(T, X)
                + func.vmap(linalg.vdot, in_axes=1)(X, H @ X)
                + W_vd_S * func.vmap(linalg.vdot, in_axes=1)(S, R)
            )
            return tr_B_hat_loo + residual_scale * tr_residual_loo

        Y_rank = np.sum(np.abs(linalg.diagonal(R)) > np.finfo_eps(R.dtype))

        # NOTE: If Y_rank < num_samples, then Y is rank-deficient because either:
        # 1. rank(B) < num_samples, and/or
        # 2. the samples are not unique.
        # This check assumes samples are unique, which can be violated for low n and Rademacher samples.
        # If rank(B) < num_samples, then B_hat=B_hat_{-i}=B, so the residual is zero and tr(B_hat)=tr(B).
        return control_flow.cond(Y_rank < num_samples, _trace_exact, _trace_estimate)

    return integrand


def integrand_diagonal():
    """Construct the integrand for estimating the diagonal.

    When plugged into the Monte-Carlo estimator,
    the result will be an Array or PyTree of Arrays with the
    same tree-structure as
    ``
    matvec(*args_like)
    ``
    where ``*args_like`` is an argument of the sampler.
    """

    def integrand(matvec, v, *parameters):
        Qv = matvec(v, *parameters)
        v_flat, unflatten = tree.ravel_pytree(v)
        Qv_flat, _unflatten = tree.ravel_pytree(Qv)
        return unflatten(v_flat * Qv_flat)

    return integrand


def integrand_trace():
    """Construct the integrand for estimating the trace."""

    def integrand(matvec, v, *parameters):
        Qv = matvec(v, *parameters)
        v_flat, _unflatten = tree.ravel_pytree(v)
        Qv_flat, _unflatten = tree.ravel_pytree(Qv)
        return linalg.inner(v_flat, Qv_flat)

    return integrand


def integrand_trace_and_diagonal():
    """Construct the integrand for estimating the trace and diagonal jointly."""

    def integrand(matvec, v, *parameters):
        Qv = matvec(v, *parameters)
        v_flat, unflatten = tree.ravel_pytree(v)
        Qv_flat, _unflatten = tree.ravel_pytree(Qv)
        trace_form = linalg.inner(v_flat, Qv_flat)
        diagonal_form = unflatten(v_flat * Qv_flat)
        return {"trace": trace_form, "diagonal": diagonal_form}

    return integrand


def integrand_frobeniusnorm_squared():
    """Construct the integrand for estimating the squared Frobenius norm."""

    def integrand(matvec, vec, *parameters):
        x = matvec(vec, *parameters)
        v_flat, _unflatten = tree.ravel_pytree(x)
        return linalg.inner(v_flat, v_flat)

    return integrand


def integrand_wrap_moments(integrand, /, moments):
    """Wrap an integrand into another integrand that computes moments.

    Parameters
    ----------
    integrand
        Any integrand function compatible with Hutchinson-style estimation.
    moments
        Any Pytree (tuples, lists, dictionaries) whose leafs that are
        valid inputs to ``lambda m: x**m`` for an array ``x``,
        usually, with data-type ``float``
        (but that depends on the wrapped integrand).
        For example, ``moments=4``, ``moments=(1,2)``,
        or ``moments={"a": 1, "b": 2}``.

    Returns
    -------
    integrand
        An integrand function compatible with Hutchinson-style estimation whose
        output has a PyTree-structure that mirrors the structure of the ``moments``
        argument.

    """

    def integrand_wrapped(vec, *parameters):
        Qs = integrand(vec, *parameters)
        return tree.tree_map(moment_fun, Qs)

    def moment_fun(x, /):
        return tree.tree_map(lambda m: x**m, moments)

    return integrand_wrapped


def sampler_normal(*args_like, num):
    """Construct a function that samples from a standard-normal distribution."""
    return _sampler_from_jax_random(prng.normal, *args_like, num=num)


def sampler_rademacher(*args_like, num):
    """Construct a function that samples from a Rademacher distribution."""
    return _sampler_from_jax_random(prng.rademacher, *args_like, num=num)


def _sampler_from_jax_random(sampler, *args_like, num):
    x_flat, unflatten = tree.ravel_pytree(*args_like)

    def sample(key):
        samples = sampler(key, shape=(num, *x_flat.shape), dtype=x_flat.dtype)
        return func.vmap(unflatten)(samples)

    return sample


def _error_num_samples(num, maxval, minval):
    msg1 = f"Number of samples num={num} exceeds the acceptable range. "
    msg2 = f"Expected: {minval} <= num <= {maxval}."
    return msg1 + msg2
