"""Stochastic estimation of traces, diagonals, and more."""

from matfree.backend import control_flow, func, linalg, np, prng, tree
from matfree.backend.typing import Array, Callable


def estimator(integrand: Callable, /, sampler: Callable) -> Callable:
    """Construct a stochastic trace-/diagonal-estimator.

    Parameters
    ----------
    integrand
        The integrand function. For example, the return-value of
        [integrand_trace][matfree.stochtrace.integrand_trace].
        But any other integrand works, too.
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
    - If the operator is real-valued and `n > O(100)`, use [sampler_rademacher][matfree.stochtrace.sampler_rademacher].
    - If the operator is real-valued and `n < O(100)`, use [sampler_rademacher][matfree.stochtrace.sampler_rademacher] if the operator is known to be diagonal-dominant or [sampler_sphere][matfree.stochtrace.sampler_sphere] otherwise.
    - If the operator is complex-valued, use [sampler_sphere][matfree.stochtrace.sampler_sphere] with a complex dtype.

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
        [sampler_rademacher][matfree.stochtrace.sampler_rademacher].

    Returns
    -------
    estimate
        A function ``estimate(matvec, key, *parameters) -> result``.
        This function can be compiled, vectorised, differentiated,
        or looped over as the user desires.

    """

    def estimate(matvec, key, *parameters):
        samples = sampler(key)
        return np.mean(integrand(matvec, samples, *parameters), axis=0)

    return estimate


def leave_one_out_xtrace(*, apply_resphering: bool = True) -> Callable:
    """Construct an integrand for estimating the trace using the XTrace algorithm (Epperly et al. 2024).

    Parameters
    ----------
    apply_resphering
        If ``True`` (default), project test vectors onto the range of the
        residual matrix, reducing the variance of the trace estimate.
        Requires test vectors drawn from a rotationally invariant distribution
        (e.g. Gaussian or sphere). See Epperly, 2025 for more details.

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

            if apply_resphering:
                # residual is B - B_hat_{-i}, where B_hat_{-i} approximates B leaving out Omega[:, i]
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
            # tr(B_hat) leaving out one sample
            tr_B_hat_loo = tr_B_hat - func.vmap(linalg.vdot, in_axes=1)(S, H @ S)
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


def leave_one_out_xnystrace(
    *,
    nystrom: Callable[[Callable, Array], tuple[Array, Array, Array]] | None = None,
    apply_resphering: bool = True,
) -> Callable:
    """Construct an integrand for estimating the trace of a positive semi-definite operator using the XNysTrace algorithm (Epperly et al. 2024).

    Parameters
    ----------
    nystrom
        An implementation of the Nystrom approximation with arguments:
        - `matvec_flat`: A flattened version of the matvec.
        - `Omega`: A matrix of shape ``(n, num_samples)`` containing the test vectors.
        and returns:
        - `nystrom_left`: A left factor of the Nystrom approximation matrix of shape ``(n, num_samples)``,
            such that `nystrom_left @ nystrom_left.T.conj()` approximates the operator.
        - `downdate`: A matrix of shape ``(n, num_samples)`` whose columns are downdate vectors for the Nystrom approximation.
            Specifically, `nystrom_left @ nystrom_left.T.conj() - np.outer(downdate[:, i], downdate[:, i].conj())` approximates the
            operator leaving out the `i`-th column of `Omega`.
        - `correction`: A correction factor to add to the trace estimate.
        Usually `nystrom` is the return value of `nystrom_shifted_cholesky` or `nystrom_eigh`.
        If not provided, `nystrom_eigh` is used.
    apply_resphering
        If ``True`` (default), project test vectors onto the range of the
        residual matrix, reducing the variance of the trace estimate.
        Requires test vectors drawn from a rotationally invariant distribution
        (e.g. Gaussian or sphere). See Epperly, 2025 for more details.

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
    - Epperly EN, Tropp JA, Webber RJ (2024). XNystrace: Making the most of every sample in stochastic trace estimation.
        SIAM J Matrix Anal A. 45.1: 1-23.
        doi: [10.1137/23M1548323](https://doi.org/10.1137/23M1548323)
        arXiv: [2301.07825](https://arxiv.org/abs/2301.07825)
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
            # It's faster and more accurate to compute the trace exactly and deterministically
            # when num_samples == n
            B_mat = _materialize_operator(matvec_flat, Omega[:, 0])
            return (
                np.ones((num_samples,), dtype=B_mat.dtype) * linalg.trace(B_mat)
            ).real

        F, Z, correction = nystrom(matvec_flat, Omega)

        if apply_resphering:
            # Efficiently form T: the R factor of a QR decomposition of Omega
            # Assumes Omega is well-conditioned (typically true for Gaussian/sphere samples if num_samples <= n/2)
            T = linalg.cholesky(Omega.T.conj() @ Omega).T.conj()
            S = _qr_leave_one_out_factor(T)
            # Omega projected onto the subspace spanned by Q_i, i.e. Q from qr(Omega_{-i}) leaving out Omega[:, i]
            X = T - S * func.vmap(linalg.vdot, in_axes=1)(S, T)
            # residual is B - B_hat_{-i}, where B_hat_{-i} approximates B leaving out Omega[:, i]
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

        # Compute the trace estimate, correcting for shift in _nystrom_shifted
        tr_B_hat = np.sum(linalg.abs2(F)) + correction
        tr_B_hat_loo = tr_B_hat - np.sum(linalg.abs2(Z), axis=0)
        tr_residual_loo = linalg.abs2(func.vmap(linalg.vdot, in_axes=1)(Z, Omega))
        return tr_B_hat_loo + residual_scale * tr_residual_loo

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
    S
        The downdate factor.
    """
    S = linalg.solve_triangular(R, np.eye(R.shape[0], dtype=R.dtype), trans=2)
    return S / func.vmap(linalg.vector_norm, in_axes=-1)(S)


def nystrom_shifted_cholesky(
    symmetrize_input: bool = True, shift: float | None = None, rtol: float | None = None
):
    """Compute shifted Nystrom approximation in an outer product form and compute the downdate matrix.

    Parameters
    ----------
    matvec_flat
        A positive semi-definite operator that maps a length `n` vector to a length `n` vector.
    Omega
        A matrix of shape ``(n, num_samples)``.
    symmetrize_input
        If ``True`` (default), internally symmetrizes before computing a Cholesky factor.
    rtol
        A relative tolerance used in computing the shift.

    Returns
    -------
    nystrom_factor
        A matrix `F` such that `F @ F.T.conj()` approximates the operator.
    downdate
        A matrix `Z` whose columns are downdate vectors for the Nystrom approximation.
    correction
        A correction factor to add to the trace estimate to account for the shift.
    """

    def nystrom(matvec_flat, Omega):
        n = Omega.shape[0]
        Y = func.vmap(matvec_flat, in_axes=-1, out_axes=-1)(Omega)
        Y_norm = linalg.vector_norm(Y)
        if shift is None:
            shift_rtol = np.finfo_eps(Y_norm.dtype) / n**0.5 if rtol is None else rtol
            mu = shift_rtol * Y_norm
        else:
            mu = shift
        Y_shifted = Y + mu * Omega
        H = Omega.T.conj() @ Y_shifted
        if symmetrize_input:
            H = (H + H.T.conj()) / 2
        chol_upper = linalg.cholesky(H).T.conj()
        nystrom_left = linalg.solve_triangular(
            chol_upper, Y_shifted.T.conj(), trans=2
        ).T.conj()

        Y_Hinv = linalg.solve_triangular(chol_upper, nystrom_left.T.conj()).T.conj()
        Id = np.eye(chol_upper.shape[0], dtype=chol_upper.dtype)
        chol_upper_inv = linalg.solve_triangular(chol_upper, Id)
        downdate = (
            Y_Hinv / func.vmap(linalg.vector_norm, in_axes=0)(chol_upper_inv)[None, :]
        )

        return nystrom_left, downdate, -mu * n

    return nystrom


def nystrom_eigh(
    eigenvalues_rtol: float | None = None, leverage_rtol: float | None = None
):
    """Compute Nystrom approximation using eigh and compute the downdate matrix."""

    def nystrom(matvec_flat, Omega):
        k = Omega.shape[1]
        Y = func.vmap(matvec_flat, in_axes=-1, out_axes=-1)(Omega)
        # select rtol using same heuristic as jax.numpy.linalg.lstsq
        vals_rtol = (
            np.finfo_eps(Y.dtype) * k if eigenvalues_rtol is None else eigenvalues_rtol
        )
        H = Omega.T.conj() @ Y
        eigh = linalg.eigh(H)
        vals = eigh.eigenvalues
        vecs = eigh.eigenvectors
        mask = vals >= vals_rtol * np.abs(vals[-1])
        s = np.where(mask, vals ** (-0.5), 0.0)
        vecs = np.where(mask, vecs, 0.0)
        H_left_sqrt = vecs * s
        nystrom_left = Y @ H_left_sqrt

        # Compute the leverage scores of each column
        leverage = np.sum(np.abs(vecs) ** 2, axis=1)
        _leverage_rtol = (
            np.sqrt(np.finfo_eps(leverage.dtype))
            if leverage_rtol is None
            else leverage_rtol
        )
        is_essential = leverage + _leverage_rtol > 1.0

        # Compute the downdate matrix Z s.t. B_hat_{-i} = B_hat - np.outer(Z[:, i], Z[:, i].conj())
        downdate = (nystrom_left @ H_left_sqrt.T.conj()) / func.vmap(
            linalg.vector_norm, in_axes=0
        )(H_left_sqrt)
        downdate = np.where(is_essential, downdate, 0.0)

        return nystrom_left, downdate, np.asarray(0.0).astype(vals.dtype)

    return nystrom


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
        return unflatten(v_flat.conj() * Qv_flat)

    return integrand


def integrand_trace():
    """Construct the integrand for estimating the trace."""

    def integrand(matvec, v, *parameters):
        Qv = matvec(v, *parameters)
        v_flat, _unflatten = tree.ravel_pytree(v)
        Qv_flat, _unflatten = tree.ravel_pytree(Qv)
        return linalg.inner(v_flat.conj(), Qv_flat)

    return integrand


def integrand_trace_and_diagonal():
    """Construct the integrand for estimating the trace and diagonal jointly."""

    def integrand(matvec, v, *parameters):
        Qv = matvec(v, *parameters)
        v_flat, unflatten = tree.ravel_pytree(v)
        Qv_flat, _unflatten = tree.ravel_pytree(Qv)
        trace_form = linalg.inner(v_flat.conj(), Qv_flat)
        diagonal_form = unflatten(v_flat.conj() * Qv_flat)
        return {"trace": trace_form, "diagonal": diagonal_form}

    return integrand


def integrand_frobeniusnorm_squared():
    """Construct the integrand for estimating the squared Frobenius norm."""

    def integrand(matvec, vec, *parameters):
        x = matvec(vec, *parameters)
        v_flat, _unflatten = tree.ravel_pytree(x)
        return linalg.inner(v_flat.conj(), v_flat)

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


def _materialize_operator(matvec_flat, x):
    """Materialize the operator defined by an already-flattened matvec and a vector."""
    # if x is real but the operator is complex, then jacfwd will error, so we infer the
    # dtype to find out if the operator might be complex
    Bx_like = func.eval_shape(matvec_flat, x)
    x = x.astype(Bx_like.dtype)
    is_complex = x.dtype.kind == "c"
    return func.jacfwd(matvec_flat, holomorphic=is_complex)(x)


def sampler_normal(*args_like, num):
    """Construct a function that samples from a standard-normal distribution."""
    return _sampler_from_jax_random(prng.normal, *args_like, num=num)


def sampler_rademacher(*args_like, num):
    """Construct a function that samples from a Rademacher distribution."""
    return _sampler_from_jax_random(prng.rademacher, *args_like, num=num)


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


def _error_num_samples(num, maxval, minval):
    msg1 = f"Number of samples num={num} exceeds the acceptable range. "
    msg2 = f"Expected: {minval} <= num <= {maxval}."
    return msg1 + msg2
