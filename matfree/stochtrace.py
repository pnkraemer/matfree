"""Stochastic estimation of traces, diagonals, and more."""

from matfree.backend import control_flow, func, linalg, np, prng, tree
from matfree.backend.typing import Array, Callable


def estimator_monte_carlo(integrand: Callable, /, sampler: Callable) -> Callable:
    """Construct a stochastic trace-/diagonal-estimator.

    Parameters
    ----------
    integrand
        An integrand function with signature ``integrand(matvec, vec, *parameters)``,
        where ``vec`` is a single sample vector (contrast with
        [estimator_leave_one_out][matfree.stochtrace.estimator_leave_one_out],
        which passes the full sample batch).
        Use any of the ``monte_carlo_*`` constructors, e.g.
        [monte_carlo_trace][matfree.stochtrace.monte_carlo_trace],
        [monte_carlo_diagonal][matfree.stochtrace.monte_carlo_diagonal],
        [monte_carlo_frobeniusnorm_squared][matfree.stochtrace.monte_carlo_frobeniusnorm_squared],
        [monte_carlo_rownorms_squared][matfree.stochtrace.monte_carlo_rownorms_squared],
        or any of the ``monte_carlo_funm_*`` functions from [matfree.funm][matfree.funm].
    sampler
        The sample function. See below for recommendations.

    Returns
    -------
    estimate
        A function that maps a random key to an estimate.
        This function can be compiled, vectorised, differentiated,
        or looped over as the user desires.

    Notes
    -----
    The statistical efficiency of the estimator for a given sampler depends on properties
    of the operator, but we can provide some general advice. For an `n`-dimensional operator (see references):
    - `n > O(100)`, use [sampler_signs][matfree.stochtrace.sampler_signs].
    - `n < O(100)`, use [sampler_signs][matfree.stochtrace.sampler_signs] if the operator is known to be diagonal-dominant or [sampler_sphere][matfree.stochtrace.sampler_sphere] otherwise.
    - If the operator is complex-valued, pass a complex dtype to the sampler to approximately double the efficiency.

    References
    ----------
    - Epperly, E. (2023). [Stochastic trace estimation](https://www.ethanepperly.com/index.php/2023/01/26/stochastic-trace-estimation/).
    - Epperly, E. (2024). [Don't use Gaussians in stochastic trace estimation](https://www.ethanepperly.com/index.php/2024/01/28/dont-use-gaussians-in-stochastic-trace-estimation/).
    """

    def estimate(matvecs, key, *parameters):
        samples = sampler(key)
        Qs = func.vmap(lambda vec: integrand(matvecs, vec, *parameters))(samples)
        return tree.tree_map(lambda s: np.mean(s, axis=0), Qs)

    return estimate


def estimator_monte_carlo_mean_and_sem(
    integrand: Callable, /, sampler: Callable
) -> Callable:
    """Construct a stochastic estimator that returns mean and standard error.

    Like [estimator_monte_carlo][matfree.stochtrace.estimator_monte_carlo],
    but returns ``(mean, sem)`` where ``sem = std(samples) / sqrt(num_samples)``
    is the standard error of the mean -- the direct uncertainty on the estimate.
    The number of samples is already encoded in the sampler,
    so the caller does not need to track it separately.

    Parameters
    ----------
    integrand
        Any integrand compatible with
        [estimator_monte_carlo][matfree.stochtrace.estimator_monte_carlo].
    sampler
        The sample function. See [estimator_monte_carlo][matfree.stochtrace.estimator_monte_carlo]
        for recommendations.

    Returns
    -------
    estimate
        A function that returns ``(mean, sem)``, both with the same
        PyTree structure as the integrand output.
    """

    def estimate(matvecs, key, *parameters):
        samples = sampler(key)
        Qs = func.vmap(lambda vec: integrand(matvecs, vec, *parameters))(samples)
        mean = tree.tree_map(lambda s: np.mean(s, axis=0), Qs)
        sem = tree.tree_map(lambda s: np.std(s, axis=0) / np.sqrt(s.shape[0]), Qs)
        return mean, sem

    return estimate


def estimator_leave_one_out(integrand: Callable, /, sampler: Callable) -> Callable:
    """Construct a leave-one-out stochastic estimator.

    Parameters
    ----------
    integrand
        An integrand that accepts ``(matvec, samples, *parameters)`` where
        ``samples`` has shape ``(num, n)``. For example, the return-value of
        [leave_one_out_xtrace][matfree.stochtrace.leave_one_out_xtrace].
    sampler
        The sample function, e.g. the return-value of
        [sampler_normal][matfree.stochtrace.sampler_normal] or
        [sampler_signs][matfree.stochtrace.sampler_signs].

    Returns
    -------
    estimate
        A function ``estimate(matvec, key, *parameters) -> result``.
        This function can be compiled, vectorised, differentiated,
        or looped over as the user desires.

    """

    def estimate(matvec, key, *parameters):
        samples = sampler(key)
        Qs = integrand(matvec, samples, *parameters)
        return tree.tree_map(lambda s: np.mean(s, axis=0), Qs)

    return estimate


def estimator_leave_one_out_mean_and_sem(
    integrand: Callable, /, sampler: Callable
) -> Callable:
    """Construct a LOO estimator that returns mean and standard error.

    Like [estimator_leave_one_out][matfree.stochtrace.estimator_leave_one_out],
    but returns ``(mean, sem)`` where ``sem = std(loo_estimates) / sqrt(num_samples)``.
    The LOO integrand produces one estimate per leave-one-out, so their
    standard deviation is a natural uncertainty measure.

    Parameters
    ----------
    integrand
        Any integrand compatible with
        [estimator_leave_one_out][matfree.stochtrace.estimator_leave_one_out].
    sampler
        The sample function.

    Returns
    -------
    estimate
        A function that returns ``(mean, sem)``, both scalars (or arrays
        with the same shape as a single LOO estimate).
    """

    def estimate(matvec, key, *parameters):
        samples = sampler(key)
        Qs = integrand(matvec, samples, *parameters)
        n_samples = tree.tree_leaves(Qs)[0].shape[0]
        mean = tree.tree_map(lambda s: np.mean(s, axis=0), Qs)
        sem = tree.tree_map(lambda s: np.std(s, axis=0) / np.sqrt(n_samples), Qs)
        return mean, sem

    return estimate


def leave_one_out_xtrace(*, apply_resphering: bool = True) -> Callable:
    """Construct an integrand for estimating the trace using the XTrace algorithm (Epperly et al. 2024).

    Parameters
    ----------
    apply_resphering
        If ``True`` (default), project test vectors onto the range of the
        residual matrix, reducing the variance of the trace estimate.
        Requires test vectors drawn from a rotationally invariant distribution
        (e.g. normal or sphere). See Epperly, 2025 for more details.

    Returns
    -------
    integrand
        An integrand function compatible with `estimator_leave_one_out` whose input
        has the signature ``(matvec, samples, *params)`` and whose output is a vector
        of trace estimates with one estimate per sample.

    Notes
    -----
    The number of samples must be less than or equal to the dimension of the operator.
    Additionally, the algorithm assumes that the samples are unique. For low-dimensional
    operators, samples generated from `sampler_signs` may violate this assumption, and
    it is recommended to use a different sampler instead.

    References
    ----------
    - Epperly EN, Tropp JA, Webber RJ (2024). XTrace: Making the most of every sample in stochastic trace estimation.
        SIAM J Matrix Anal A. 45.1: 1-23.
        doi: [10.1137/23M1548323](https://doi.org/10.1137/23M1548323)
        arXiv: [2301.07825](https://arxiv.org/abs/2301.07825)
    - Epperly EN (2025). Make the most of what you have: Resource-efficient randomized algorithms for matrix computations. PhD Thesis.
        arXiv: [2512.15929](https://arxiv.org/abs/2512.15929)
    """

    def integrand(matvec, samples, *params):
        sample0 = tree.tree_map(lambda s: s[0], samples)
        _, unflatten = tree.ravel_pytree(sample0)

        Omega = func.vmap(lambda s: tree.ravel_pytree(s)[0])(samples).T
        n, num_samples = Omega.shape

        if num_samples > n:
            raise ValueError(_error_num_samples(num_samples, maxval=n, minval=1))

        def matvec_flat(v):
            return tree.ravel_pytree(matvec(unflatten(v), *params))[0]

        if 2 * num_samples >= n:
            # It's faster, more accurate, and allocates less memory to compute the trace exactly
            # and deterministically on the materialized operator when num_samples >= n/2
            B_mat = _materialize_operator(matvec_flat, Omega[:, 0])
            return np.ones((num_samples,), dtype=B_mat.dtype) * linalg.trace(B_mat)

        matmat = func.vmap(matvec_flat, in_axes=-1, out_axes=-1)

        Y = matmat(Omega)
        Q, R = linalg.qr_reduced(Y)
        Z = matmat(Q)

        def _trace_exact():
            tr_B = linalg.vdot(Q, Z)
            return np.ones((num_samples,), dtype=R.dtype) * tr_B

        def _trace_estimate():
            S = _qr_leave_one_out_factor(R)

            Q_H = Q.T.conj()
            # tr(H) == tr(B_hat), where B_hat = Q @ Q.H @ B is a low-rank approximation to the operator B
            H = Q_H @ Z
            W = Q_H @ Omega
            T = Z.T.conj() @ Omega
            W_vd_S = func.vmap(linalg.vdot, in_axes=1)(W, S)
            # Omega projected onto the subspace spanned by Q_i, i.e. Q formed leaving out Omega[:, i]
            X = W - S * W_vd_S.conj()
            T_vd_X = func.vmap(linalg.vdot, in_axes=1)(T, X)
            X_vd_HX = func.vmap(linalg.vdot, in_axes=1)(X, H @ X)
            S_vd_R = func.vmap(linalg.vdot, in_axes=1)(S, R)

            if apply_resphering:
                # residual is B - B_hat_{-i}, where B_hat_{-i} approximates B leaving out Omega[:, i]
                rank_residual = n - num_samples + 1
                # squared norm of each sample after projection to the subspace spanned by the residual
                sqnorm_Omega = np.sum(linalg.abs2(Omega), axis=0)
                sqnorm_X = np.sum(linalg.abs2(X), axis=0)
                sqnorm_samples_projected = sqnorm_Omega - sqnorm_X
                has_zero_norm = sqnorm_samples_projected == 0.0
                sqnorm_samples_projected = np.where(
                    has_zero_norm, 1.0, sqnorm_samples_projected
                )
                residual_scale = rank_residual / sqnorm_samples_projected
            else:
                residual_scale = 1.0

            tr_B_hat = linalg.trace(H)
            # tr(B_hat) leaving out one sample
            tr_B_hat_loo = tr_B_hat - func.vmap(linalg.vdot, in_axes=1)(S, H @ S)
            # Hutchinson estimate of tr(B - B_hat_{-i}) using samples[i, :] as probes.
            tr_residual_loo = -T_vd_X + X_vd_HX + S_vd_R * W_vd_S
            return tr_B_hat_loo + residual_scale * tr_residual_loo

        Y_rank = np.sum(np.abs(linalg.diagonal(R)) > np.finfo_eps(R.dtype))

        # NOTE: If Y_rank < num_samples, then Y is rank-deficient because either:
        # 1. rank(B) < num_samples, and/or
        # 2. the samples are not unique.
        # This check assumes samples are unique, which can be violated for low n and Rademacher samples.
        # If rank(B) < num_samples, then B_hat=B_hat_{-i}=B, so the residual is zero and tr(B_hat)=tr(B).
        return control_flow.cond(Y_rank < num_samples, _trace_exact, _trace_estimate)

    return integrand


def leave_one_out_xdiag() -> Callable:
    """Construct an integrand for estimating the diagonal using the XDiag algorithm (Epperly et al. 2024).

    Returns
    -------
    integrand
        An integrand function compatible with
        [estimator_leave_one_out][matfree.stochtrace.estimator_leave_one_out]
        whose input has the signature ``(matvec, samples, *params)`` and whose output is a
        pytree with each leaf having shape ``(num_samples, n_k)``, giving one diagonal
        estimate per leave-one-out sample.

    Notes
    -----
    The number of samples must be less than or equal to the dimension of the operator.

    The sum of the diagonal estimate over all entries is an unbiased estimate of the trace but
    generally has higher variance than the estimate produced by
    [leave_one_out_xnystrace][matfree.stochtrace.leave_one_out_xnystrace].

    References
    ----------
    - Epperly EN, Tropp JA, Webber RJ (2024). XTrace: Making the most of every sample in stochastic trace estimation.
        SIAM J Matrix Anal A. 45.1: 1-23.
        doi: [10.1137/23M1548323](https://doi.org/10.1137/23M1548323)
        arXiv: [2301.07825](https://arxiv.org/abs/2301.07825)
    - Epperly EN (2025). Make the most of what you have: Resource-efficient randomized algorithms for matrix computations. PhD Thesis.
        arXiv: [2512.15929](https://arxiv.org/abs/2512.15929)
    """

    def integrand(matvec, samples, *params):
        sample0 = tree.tree_map(lambda s: s[0], samples)
        _, unflatten = tree.ravel_pytree(sample0)

        Omega = func.vmap(lambda s: tree.ravel_pytree(s)[0])(samples).T
        n, num_samples = Omega.shape

        if num_samples > n:
            raise ValueError(_error_num_samples(num_samples, maxval=n, minval=1))

        def matvec_flat(v):
            return tree.ravel_pytree(matvec(unflatten(v), *params))[0]

        if 2 * num_samples >= n:
            B_mat = _materialize_operator(matvec_flat, Omega[:, 0])
            diag_B = linalg.diagonal(B_mat)
            return func.vmap(unflatten)(
                np.ones((num_samples, 1), dtype=diag_B.dtype) * diag_B
            )

        matvec_flat_adjoint = func.linear_adjoint(matvec_flat, Omega[:, 0])

        Y = func.vmap(matvec_flat, in_axes=-1, out_axes=-1)(Omega)
        Q, R = linalg.qr_reduced(Y)
        (Z,) = func.vmap(matvec_flat_adjoint, in_axes=-1, out_axes=-1)(Q)

        def _diag_exact():
            diag_B = func.vmap(linalg.vdot, in_axes=0)(Z, Q)
            return np.ones(num_samples, dtype=diag_B.dtype) * diag_B[:, None]

        def _diag_estimate():
            S = _qr_leave_one_out_factor(R)
            QS = Q @ S
            S_vd_R = func.vmap(linalg.vdot, in_axes=1)(S, R)

            diag_B_hat = func.vmap(linalg.vdot, in_axes=0)(Z, Q)
            diag_B_hat_loo = diag_B_hat[:, None] - QS * (Z @ S).conj()
            diag_residual_loo = QS * S_vd_R * Omega.conj()
            return diag_B_hat_loo + diag_residual_loo

        Y_rank = np.sum(np.abs(linalg.diagonal(R)) > np.finfo_eps(R.dtype))

        diag_loo = control_flow.cond(Y_rank < num_samples, _diag_exact, _diag_estimate)
        return func.vmap(unflatten)(diag_loo.T)

    return integrand


def leave_one_out_xnystrace(
    *,
    nystrom: Callable[[Callable, Array], tuple[Array, Array, Array]] | None = None,
    apply_resphering: bool = True,
    qr_r: Callable[[Array], Array] | None = None,
) -> Callable:
    """Construct an integrand for estimating the trace of a positive semi-definite operator using the XNysTrace algorithm (Epperly et al. 2024).

    Parameters
    ----------
    nystrom
        A callable with signature ``(matvec_flat, Omega) -> (nystrom_left, downdate, shift)``,
        where ``Omega`` has shape ``(n, num_samples)``, ``nystrom_left`` and ``downdate``
        have shape ``(n, num_samples)``, and ``shift`` is a scalar.
        ``nystrom_left @ nystrom_left.T.conj()`` approximates the operator (shifted by ``shift * I``),
        and subtracting ``outer(downdate[:, i], downdate[:, i].conj())``
        approximates it leaving out the ``i``-th column of ``Omega``.
        Usually the return value of [`nystrom_shifted_cholesky`][matfree.stochtrace.nystrom_shifted_cholesky]
        or [`nystrom_eigh`][matfree.stochtrace.nystrom_eigh] (default: `nystrom_eigh`).
    apply_resphering
        If ``True`` (default), project test vectors onto the range of the
        residual matrix, reducing the variance of the trace estimate.
        Requires test vectors drawn from a rotationally invariant distribution
        (e.g. normal or sphere). See Epperly, 2025 for more details.
    qr_r
        A callable that computes the R factor of a QR decomposition, used if `apply_resphering` is `True`.
        If not provided, `linalg.qr_r` is used.

    Returns
    -------
    integrand
        An integrand function compatible with `estimator_leave_one_out` whose input
        has the signature ``(matvec, samples, *params)`` and whose output is a vector
        of trace estimates with shape ``(samples.shape[0],)``.
        The `matvec` must be a positive semi-definite operator. That is,
        `vdot(v, matvec(v))` is real and non-negative for all vectors `v`,
        and `vdot(x, matvec(y)) = vdot(matvec(x), y)` for all vectors `x` and `y`.

    References
    ----------
    - Epperly EN, Tropp JA, Webber RJ (2024). XTrace: Making the most of every sample in stochastic trace estimation.
        SIAM J Matrix Anal A. 45.1: 1-23.
        doi: [10.1137/23M1548323](https://doi.org/10.1137/23M1548323)
        arXiv: [2301.07825](https://arxiv.org/abs/2301.07825)
    - Epperly EN (2025). Make the most of what you have: Resource-efficient randomized algorithms for matrix computations. PhD Thesis.
        arXiv: [2512.15929](https://arxiv.org/abs/2512.15929)
    """
    # NOTE: The paper and thesis use the shifted Nystrom approximation with a
    # Cholesky decomposition, but empirically, this is brittle and fails for
    # many low-rank operators. The eigh-based Nystrom approximation seems to be
    # more robust.
    if nystrom is None:
        nystrom = nystrom_eigh()

    # NOTE: The paper computes R via the QR decomposition, while for efficiency, the
    # thesis uses the upper Cholesky factor of the Gram matrix. We use the QR approach
    # because it may be less brittle and prone to NaNs than Cholesky.
    if qr_r is None:
        qr_r = linalg.qr_r

    def integrand(matvec, samples, *params):
        sample0 = tree.tree_map(lambda s: s[0], samples)
        _, unflatten = tree.ravel_pytree(sample0)

        Omega = func.vmap(lambda s: tree.ravel_pytree(s)[0])(samples).T
        n, num_samples = Omega.shape

        if num_samples > n:
            raise ValueError(_error_num_samples(num_samples, maxval=n, minval=1))

        def matvec_flat(v):
            return tree.ravel_pytree(matvec(unflatten(v), *params))[0]

        if num_samples == n:
            # It's faster and more accurate to compute the trace exactly and deterministically
            # when num_samples == n
            B_mat = _materialize_operator(matvec_flat, Omega[:, 0])
            trace_samples = np.ones((num_samples,), dtype=B_mat.dtype) * linalg.trace(
                B_mat
            )
            return trace_samples.real

        F, Z, shift = nystrom(matvec_flat, Omega)

        if apply_resphering:
            # Ensure T (the R factor) is square
            T = qr_r(Omega)[:num_samples, :num_samples]
            S = _qr_leave_one_out_factor(T)
            # Omega projected onto the subspace spanned by Q_i, i.e. Q from qr(Omega_{-i}) leaving out Omega[:, i]
            X = T - S * func.vmap(linalg.vdot, in_axes=1)(S, T)
            # residual is B - B_hat_{-i}, where B_hat_{-i} approximates B leaving out Omega[:, i]
            rank_residual = n - num_samples + 1
            # squared norm of each sample after projection to the subspace spanned by the residual
            sqnorm_Omega = np.sum(linalg.abs2(Omega), axis=0)
            sqnorm_X = np.sum(linalg.abs2(X), axis=0)
            sqnorm_samples_projected = sqnorm_Omega - sqnorm_X
            sqnorm_samples_projected = np.where(
                sqnorm_samples_projected == 0.0, 1.0, sqnorm_samples_projected
            )
            residual_scale = rank_residual / sqnorm_samples_projected
        else:
            residual_scale = 1.0

        # Compute the trace estimate, correcting for shift in _nystrom_shifted
        tr_B_hat = np.sum(linalg.abs2(F)) - shift * n
        tr_B_hat_loo = tr_B_hat - np.sum(linalg.abs2(Z), axis=0)
        tr_residual_loo = linalg.abs2(func.vmap(linalg.vdot, in_axes=1)(Z, Omega))
        return tr_B_hat_loo + residual_scale * tr_residual_loo

    return integrand


def leave_one_out_xnysdiag(
    *, nystrom: Callable[[Callable, Array], tuple[Array, Array, Array]] | None = None
) -> Callable:
    """Construct an integrand for estimating the diagonal of a positive semi-definite operator using the XNysDiag algorithm (Epperly et al. 2025).

    Parameters
    ----------
    nystrom
        A callable with signature ``(matvec_flat, Omega) -> (nystrom_left, downdate, shift)``.
        Usually the return value of
        [`nystrom_shifted_cholesky`][matfree.stochtrace.nystrom_shifted_cholesky]
        or [`nystrom_eigh`][matfree.stochtrace.nystrom_eigh] (default: `nystrom_eigh`).

    Returns
    -------
    integrand
        An integrand function compatible with
        [estimator_leave_one_out][matfree.stochtrace.estimator_leave_one_out]
        whose input has the signature ``(matvec, samples, *params)`` and whose output is a
        pytree with each leaf having shape ``(num_samples, n_k)``, giving one diagonal
        estimate per leave-one-out sample.
        The `matvec` must be a positive semi-definite operator.

    Notes
    -----
    The number of samples must be less than or equal to the dimension of the operator.
    The output diagonal is real-valued (PSD operators have real diagonal).

    The sum of the diagonal estimate over all entries equals the corresponding
    [leave_one_out_xnystrace][matfree.stochtrace.leave_one_out_xnystrace]
    trace estimate exactly for the same operator and samples (when ``apply_resphering=False``).

    References
    ----------
    - Epperly EN (2025). Make the most of what you have: Resource-efficient randomized algorithms for matrix computations. PhD Thesis.
        arXiv: [2512.15929](https://arxiv.org/abs/2512.15929)
    """
    if nystrom is None:
        nystrom = nystrom_eigh()

    def integrand(matvec, samples, *params):
        sample0 = tree.tree_map(lambda s: s[0], samples)
        _, unflatten = tree.ravel_pytree(sample0)

        Omega = func.vmap(lambda s: tree.ravel_pytree(s)[0])(samples).T
        n, num_samples = Omega.shape

        if num_samples > n:
            raise ValueError(_error_num_samples(num_samples, maxval=n, minval=1))

        def matvec_flat(v):
            return tree.ravel_pytree(matvec(unflatten(v), *params))[0]

        if num_samples == n:
            B_mat = _materialize_operator(matvec_flat, Omega[:, 0])
            diag_B = linalg.diagonal(B_mat)
            diag_all = np.ones((num_samples, 1), dtype=diag_B.dtype) * diag_B
            return tree.tree_map(lambda x: x.real, func.vmap(unflatten)(diag_all))

        F, Z, shift = nystrom(matvec_flat, Omega)
        Z_vd_Omega = func.vmap(linalg.vdot, in_axes=1)(Z, Omega)

        diag_B_hat = np.sum(linalg.abs2(F), axis=1) - shift
        diag_B_hat_loo = diag_B_hat[:, None] - linalg.abs2(Z)
        diag_res_loo = Z * Z_vd_Omega * Omega.conj()
        diag_loo = (diag_B_hat_loo + diag_res_loo).T
        return tree.tree_map(lambda x: x.real, func.vmap(unflatten)(diag_loo))

    return integrand


def leave_one_out_xrownorms_squared() -> Callable:
    """Construct an integrand for estimating squared row norms using the XRowNorm algorithm (Epperly, 2025).

    Returns
    -------
    integrand
        An integrand function compatible with
        [estimator_leave_one_out][matfree.stochtrace.estimator_leave_one_out]
        whose input has the signature ``(matvec, samples, *params)`` and whose output is a
        pytree with each leaf having shape ``(num_samples, n_k)``, giving one squared-row-norm
        estimate per leave-one-out sample.

    Notes
    -----
    To estimate squared column norms instead, pass the adjoint (i.e. conjugate-transpose-conjugate) of the matvec.

    For normal operators (those that commute with their adjoint),
    [leave_one_out_xsymrownorms_squared][matfree.stochtrace.leave_one_out_xsymrownorms_squared],
    requires fewer matvecs per sample but is typically less accurate per sample.

    References
    ----------
    - Epperly EN (2025). Make the most of what you have: Resource-efficient randomized algorithms for matrix computations. PhD Thesis.
        arXiv: [2512.15929](https://arxiv.org/abs/2512.15929)
    """
    return _leave_one_out_rownorms_squared(iterate_subspace=True)


def leave_one_out_xsymrownorms_squared() -> Callable:
    """Construct an integrand for estimating squared row norms using the XSymRowNorm algorithm (Epperly, 2025).

    Returns
    -------
    integrand
        An integrand function compatible with
        [estimator_leave_one_out][matfree.stochtrace.estimator_leave_one_out]
        whose input has the signature ``(matvec, samples, *params)`` and whose output is a
        pytree with each leaf having shape ``(num_samples, n_k)``, giving one squared-row-norm
        estimate per leave-one-out sample.
        The ``matvec`` must be a normal operator, i.e. it must commute with its adjoint
        (conjugate-transpose-conjugate) up to numerical precision.

    Notes
    -----
    To estimate squared column norms instead, pass the adjoint (i.e. conjugate-transpose-conjugate) of the matvec.

    For general (non-normal) operators, use
    [leave_one_out_xrownorms_squared][matfree.stochtrace.leave_one_out_xrownorms_squared].

    References
    ----------
    - Epperly EN (2025). Make the most of what you have: Resource-efficient randomized algorithms for matrix computations. PhD Thesis.
        arXiv: [2512.15929](https://arxiv.org/abs/2512.15929)
    """
    return _leave_one_out_rownorms_squared(iterate_subspace=False)


def _leave_one_out_rownorms_squared(*, iterate_subspace: bool):
    def integrand(matvec, samples, *params):
        sample0 = tree.tree_map(lambda s: s[0], samples)
        _, unflatten = tree.ravel_pytree(sample0)

        out_shape = func.eval_shape(matvec, sample0, *params)
        out0 = tree.tree_map(lambda s: np.zeros(s.shape, dtype=s.dtype), out_shape)
        _, unflatten_out = tree.ravel_pytree(out0)

        Omega = func.vmap(lambda s: tree.ravel_pytree(s)[0])(samples).T
        n, num_samples = Omega.shape

        if num_samples > n:
            raise ValueError(_error_num_samples(num_samples, maxval=n, minval=1))

        def matvec_flat(v):
            return tree.ravel_pytree(matvec(unflatten(v), *params))[0]

        matmat = func.vmap(matvec_flat, in_axes=-1, out_axes=-1)
        matvec_flat_adjoint = func.linear_adjoint(matvec_flat, Omega[:, 0])

        num_matvecs_per_sample = 2 + int(iterate_subspace)
        if num_matvecs_per_sample * num_samples >= n:
            # It's faster, more accurate, and allocates less memory to compute the row
            # norms exactly and deterministically on the materialized operator when
            # the number of matvecs exceeds the input dimension of the operator
            B_mat = _materialize_operator(matvec_flat, Omega[:, 0])
            rownorms_squared = _rownorms_squared(B_mat)
            srn_all = (
                np.ones(num_samples, dtype=rownorms_squared.dtype)
                * rownorms_squared[:, None]
            )
            return func.vmap(unflatten_out)(srn_all.T)

        G = matmat(Omega)
        if iterate_subspace:
            (Y,) = func.vmap(matvec_flat_adjoint, in_axes=-1, out_axes=-1)(G)
        else:
            Y = G
        Q, R = linalg.qr_reduced(Y)

        Z = matmat(Q)
        srn_B_hat = _rownorms_squared(Z)

        def _rownorms_squared_exact():
            return np.ones(num_samples, dtype=srn_B_hat.dtype) * srn_B_hat[:, None]

        def _rownorms_squared_estimate():
            S = _qr_leave_one_out_factor(R)

            W = Q.T.conj() @ Omega
            W_vd_S = func.vmap(linalg.vdot, in_axes=1)(W, S)
            # Omega projected onto the subspace spanned by Q_i, i.e. Q formed leaving out Omega[:, i]
            X = W - S * W_vd_S.conj()

            # srn(B_hat) leaving out one sample
            srn_B_hat_loo = srn_B_hat[:, None] - linalg.abs2(Z @ S)
            # Johnson-Lindenstrauss estimate of srn(B - B_hat_{-i}) using samples[i, :] as probes.
            srn_residual_loo = linalg.abs2(G - Z @ X)
            return srn_B_hat_loo + srn_residual_loo

        is_Y_rank_deficient = (
            np.sum(np.abs(linalg.diagonal(R)) > np.finfo_eps(R.dtype)) < num_samples
        )

        srn_loo = control_flow.cond(
            is_Y_rank_deficient, _rownorms_squared_exact, _rownorms_squared_estimate
        )

        return func.vmap(unflatten_out)(srn_loo.T)

    return integrand


def _qr_leave_one_out_factor(R):
    r"""Compute the downdate factor for a QR decomposition leaving out a single column.

    Given a QR decomposition \(Q R = Y\) and the QR decomposition \(Q_i R_i = Y_{-i}\),
    where \(Y_{-i}\) is \(Y\) leaving out column \(i\),
    the downdate factor is a matrix \(S\) such that \(Q_i Q_i^H = Q (I - s_i s_i^H) Q^H\).

    Parameters
    ----------
    R
        The R factor of a QR decomposition.

    Returns
    -------
    downdate
        The downdate factor.
    """
    downdate = linalg.solve_triangular(R, np.eye(R.shape[0], dtype=R.dtype), trans=2)
    col_norms = func.vmap(linalg.vector_norm, in_axes=1)(downdate)
    return downdate / col_norms


def nystrom_shifted_cholesky(
    shift: float | None = None, rtol: float | None = None, symmetrize_input: bool = True
):
    """Construct a Nystrom approximation of a shifted operator using a Cholesky decomposition.

    Parameters
    ----------
    shift
        A small positive shift to add to the operator to ensure the resulting operator
        is positive definite for Cholesky decomposition.
        If not provided, the `rtol` is used to compute the shift.
    rtol
        A relative tolerance used in computing the shift.
    symmetrize_input
        If ``True`` (default), internally symmetrizes before computing the Cholesky factor.

    Returns
    -------
    nystrom
        A function that computes the Nystrom approximation of a shifted operator using a Cholesky decomposition.
        The function has the signature `(matvec_flat, Omega) -> (nystrom_left, downdate, shift)`,
        where `nystrom_left` is a left factor of the Nystrom approximation matrix of shape ``(n, num_samples)``,
        such that `nystrom_left @ nystrom_left.T.conj()` approximates the operator,
        `downdate` is a matrix of shape ``(n, num_samples)`` whose columns are downdate vectors for the Nystrom approximation,
        and `shift` is the shift used.
    """

    def nystrom(matvec_flat, Omega):
        n = Omega.shape[0]
        Y = func.vmap(matvec_flat, in_axes=-1, out_axes=-1)(Omega)
        Y_norm = linalg.vector_norm(Y)
        if shift is None:
            shift_rtol = np.finfo_eps(Y_norm.dtype) if rtol is None else rtol
            mu = shift_rtol * Y_norm / n**0.5
        else:
            mu = shift
        Y_shifted = Y + mu * Omega
        H = Omega.T.conj() @ Y_shifted

        # Compute left-square-root of inv(H)
        if symmetrize_input:
            H = _symmetrize(H)
        H_cholu = linalg.cholesky(H).T.conj()
        Id = np.eye(H_cholu.shape[0], dtype=H_cholu.dtype)
        H_inv_left_sqrt = linalg.solve_triangular(H_cholu, Id)

        # Compute left-square-root of Nystrom approximation
        nystrom_right = linalg.solve_triangular(H_cholu, Y_shifted.T.conj(), trans=2)
        nystrom_left = nystrom_right.T.conj()

        norms = func.vmap(linalg.vector_norm, in_axes=0)(H_inv_left_sqrt)
        downdate = linalg.solve_triangular(H_cholu, nystrom_right).T.conj()
        downdate = downdate / norms[None, :]

        return nystrom_left, downdate, mu

    return nystrom


def nystrom_eigh(
    eigenvalues_rtol: float | None = None,
    leverage_rtol: float | None = None,
    symmetrize_input: bool = True,
):
    """Construct a Nystrom approximation of an operator using a Hermitian eigendecomposition.

    Parameters
    ----------
    eigenvalues_rtol
        A relative tolerance used to determine which eigenvalues are close enough to 0.
    leverage_rtol
        A relative tolerance used in computing the leverage scores to determine which
        test vectors are essential.
    symmetrize_input
        If ``True`` (default), internally symmetrizes before computing the eigendecomposition.

    Returns
    -------
    nystrom
        A function that computes the Nystrom approximation of an operator using a Hermitian eigendecomposition.
        The function has the signature `(matvec_flat, Omega) -> (nystrom_left, downdate, shift)`,
        where `nystrom_left` is a left factor of the Nystrom approximation matrix of shape ``(n, num_samples)``,
        such that `nystrom_left @ nystrom_left.T.conj()` approximates the operator,
        `downdate` is a matrix of shape ``(n, num_samples)`` whose columns are downdate vectors for the Nystrom approximation,
        and `shift=0` is the shift used (for common API).
    """

    def nystrom(matvec_flat, Omega):
        num_samples = Omega.shape[1]
        Y = func.vmap(matvec_flat, in_axes=-1, out_axes=-1)(Omega)
        # select rtol using same heuristic as jax.numpy.linalg.lstsq
        if eigenvalues_rtol is None:
            vals_rtol = np.finfo_eps(Y.dtype) * num_samples
        else:
            vals_rtol = eigenvalues_rtol
        H = Omega.T.conj() @ Y

        # Compute left-square-root of pinv(H)
        if symmetrize_input:
            H = _symmetrize(H)
        H_eigh = linalg.eigh(H)
        vals = H_eigh.eigenvalues
        vecs = H_eigh.eigenvectors
        mask = vals >= vals_rtol * np.abs(vals[-1])
        inv_sqrt_vals = np.where(mask, vals ** (-0.5), 0.0)
        vecs = np.where(mask, vecs, 0.0)
        H_pinv_left_sqrt = vecs * inv_sqrt_vals

        # Compute left-square-root of Nystrom approximation
        nystrom_left = Y @ H_pinv_left_sqrt

        # Compute the leverage scores of each column
        leverage = np.sum(np.abs(vecs) ** 2, axis=1)
        if leverage_rtol is None:
            _leverage_rtol = np.sqrt(np.finfo_eps(leverage.dtype))
        else:
            _leverage_rtol = leverage_rtol
        is_essential = leverage + _leverage_rtol > 1.0

        # Compute downdate Z s.t. B_hat_{-i} = B_hat - outer(Z[:, i], Z[:, i].conj()).
        # Since pinv(P H P') = P pinv(H) P', WLOG take i=k with Hk = H without row/col k.
        # Non-essential k, rank(Hk) = rank(H): B_hat_{-k} = B_hat, so Z[:, k] = 0.
        # Essential k, rank(Hk) = rank(H) - 1: by Albert (1969) Thm. 3,
        #   pinv(H) = [pinv(Hk) 0; 0 0] + a v v'  (v = pinv(H)[:, k], a = 1/pinv(H)[k, k]).
        # So B_hat = Y_{-k} pinv(Hk) Y_{-k}' + a (Yv)(Yv)' = B_hat_{-k} + a (Yv)(Yv)',
        # giving Z[:, k] = sqrt(a) Yv = F L[k, :]' / norm(L[k, :])
        # (F, L are left-sqrt factors of B_hat and pinv(H)).
        # Albert (1969). SIAM J. Appl. Math. 17(2), 434-440. doi:10.1137/0117041
        norms = func.vmap(linalg.vector_norm, in_axes=0)(H_pinv_left_sqrt)
        downdate = (nystrom_left @ H_pinv_left_sqrt.T.conj()) / norms
        downdate = np.where(is_essential, downdate, 0.0)

        return nystrom_left, downdate, np.asarray(0.0).astype(vals.dtype)

    return nystrom


def _symmetrize(x):
    r"""Symmetrize a matrix by computing \((x + x^H) / 2\)."""
    return (x + x.T.conj()) / 2


def _rownorms_squared(X):
    """Compute the squared row norms of a matrix."""
    return np.sum(linalg.abs2(X), axis=1)


def monte_carlo_diagonal():
    """Construct the integrand for estimating the diagonal.

    Use with [estimator_monte_carlo][matfree.stochtrace.estimator_monte_carlo].
    The result will be an Array or PyTree of Arrays with the same tree-structure as
    ``matvec(*args_like)`` where ``*args_like`` is an argument of the sampler.
    """

    def integrand(matvec, v, *parameters):
        Qv = matvec(v, *parameters)
        v_flat, unflatten = tree.ravel_pytree(v)
        Qv_flat, _unflatten = tree.ravel_pytree(Qv)
        return unflatten(v_flat.conj() * Qv_flat)

    return integrand


def monte_carlo_trace():
    """Construct the integrand for estimating the trace.

    Use with [estimator_monte_carlo][matfree.stochtrace.estimator_monte_carlo].
    """

    def integrand(matvec, v, *parameters):
        Qv = matvec(v, *parameters)
        v_flat, _unflatten = tree.ravel_pytree(v)
        Qv_flat, _unflatten = tree.ravel_pytree(Qv)
        return linalg.inner(v_flat.conj(), Qv_flat)

    return integrand


def monte_carlo_trace_and_diagonal():
    """Construct the integrand for estimating the trace and diagonal jointly.

    Use with [estimator_monte_carlo][matfree.stochtrace.estimator_monte_carlo].
    """

    def integrand(matvec, v, *parameters):
        Qv = matvec(v, *parameters)
        v_flat, unflatten = tree.ravel_pytree(v)
        Qv_flat, _unflatten = tree.ravel_pytree(Qv)
        trace_form = linalg.inner(v_flat.conj(), Qv_flat)
        diagonal_form = unflatten(v_flat.conj() * Qv_flat)
        return {"trace": trace_form, "diagonal": diagonal_form}

    return integrand


def monte_carlo_rownorms_squared():
    """Construct the integrand for estimating the squared row norms.

    Use with [estimator_monte_carlo][matfree.stochtrace.estimator_monte_carlo].

    Notes
    -----
    To estimate squared column norms instead, pass the adjoint (i.e. conjugate-transpose-conjugate) of the matvec.
    """

    def integrand(matvec, v, *parameters):
        Qv = matvec(v, *parameters)
        Qv_flat, unflatten = tree.ravel_pytree(Qv)
        return unflatten(linalg.abs2(Qv_flat))

    return integrand


def monte_carlo_frobeniusnorm_squared():
    """Construct the integrand for estimating the squared Frobenius norm.

    Use with [estimator_monte_carlo][matfree.stochtrace.estimator_monte_carlo].
    """

    def integrand(matvec, vec, *parameters):
        x = matvec(vec, *parameters)
        v_flat, _unflatten = tree.ravel_pytree(x)
        return linalg.inner(v_flat.conj(), v_flat)

    return integrand


def _materialize_operator(matvec_flat, x):
    """Materialize the operator defined by an already-flattened matvec and a vector."""
    # if the operator is complex, holomorphic=True is needed, which requires complex input --
    # cast x to the operator's output dtype first
    Bx_like = func.eval_shape(matvec_flat, x)
    x = x.astype(Bx_like.dtype)
    is_complex = x.dtype.kind == "c"
    return func.jacfwd(matvec_flat, holomorphic=is_complex)(x)


def sampler_normal(*args_like, num):
    """Construct a function that samples from a standard-normal distribution."""
    return _sampler_from_jax_random(prng.normal, *args_like, num=num)


def sampler_signs(*args_like, num):
    """Construct a function that samples signs uniformly.

    For real dtypes, this samples from a Rademacher distribution (uniformly over `{-1, 1}`). For complex dtypes, this samples from a Steinhaus distribution on the complex unit circle.
    """
    return _sampler_from_jax_random(_uniform_signs, *args_like, num=num)


def sampler_sphere(*args_like, num):
    """Construct a function that samples from a unit sphere scaled to have identity covariance."""
    x_flat, unflatten = tree.ravel_pytree(*args_like)
    dtype = x_flat.dtype
    rdtype = dtype.type(0).real.dtype
    n = x_flat.shape[0]
    sqrtn = np.sqrt(n).astype(rdtype)

    def sample(key):
        samples = prng.normal(key, shape=(num, n), dtype=dtype)
        return func.vmap(lambda x: unflatten(x * (sqrtn / linalg.vector_norm(x))))(
            samples
        )

    return sample


def _sampler_from_jax_random(sampler, *args_like, num):
    x_flat, unflatten = tree.ravel_pytree(*args_like)

    def sample(key):
        samples = sampler(key, shape=(num, *x_flat.shape), dtype=x_flat.dtype)
        return func.vmap(unflatten)(samples)

    return sample


def _steinhaus(key, /, shape, dtype):
    """Sample from a Steinhaus distribution on the complex unit circle."""
    rdtype = np.dtype(dtype).type(0).real.dtype
    angle = prng.uniform(key, shape=shape, dtype=rdtype) * (2 * np.pi())
    return np.cos(angle) + 1j * np.sin(angle)


def _uniform_signs(key, /, shape, dtype):
    if np.dtype(dtype).kind == "c":
        return _steinhaus(key, shape=shape, dtype=dtype)
    return prng.rademacher(key, shape=shape, dtype=dtype)


def _error_num_samples(num, maxval, minval):
    msg1 = f"Number of samples num={num} exceeds the acceptable range. "
    msg2 = f"Expected: {minval} <= num <= {maxval}."
    return msg1 + msg2
