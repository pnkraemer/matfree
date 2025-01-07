"""Matrix-free matrix decompositions.

This module includes various Lanczos-decompositions of matrices
(tri-diagonal, bi-diagonal, etc.).

For stochastic Lanczos quadrature and
matrix-function-vector products, see
[matfree.funm][matfree.funm].
"""

from matfree.backend import containers, control_flow, func, linalg, np, tree
from matfree.backend.typing import Array, Callable, Union


class _DecompResult(containers.NamedTuple):
    # If an algorithm returns a single Q, place it here.
    # If it returns multiple Qs, stack them
    # into a tuple and place them here.
    Q_tall: Union[Array, tuple[Array, ...]]

    # If an algorithm returns a materialized matrix,
    # place it here. If it returns a sparse representation
    # (e.g. two vectors representing diagonals), place it here
    J_small: Union[Array, tuple[Array, ...]]

    residual: Array
    init_length_inv: Array


def tridiag_sym(
    num_matvecs: int,
    /,
    *,
    materialize: bool = True,
    reortho: str = "full",
    custom_vjp: bool = True,
):
    r"""Construct an implementation of **tridiagonalisation**.

    Uses pre-allocation, and full reorthogonalisation if `reortho` is set to `"full"`.
    It tends to be a good idea to use full reorthogonalisation.

    This algorithm assumes a **symmetric matrix**.

    Decompose a matrix into a product of orthogonal-**tridiagonal**-orthogonal matrices.
    Use this algorithm for approximate **eigenvalue** decompositions.

    Setting `custom_vjp` to `True` implies using efficient, numerically stable
    gradients of the Lanczos iteration according to what has been proposed by
    Krämer et al. (2024).
    These gradients are exact, so there is little reason not to use them.
    If you use this configuration, please consider
    citing Krämer et al. (2024; bibtex below).

    ??? note "BibTex for Krämer et al. (2024)"
        ```bibtex
        @article{kraemer2024gradients,
            title={Gradients of functions of large matrices},
            author={Kr\"amer, Nicholas and Moreno-Mu\~noz, Pablo and
            Roy, Hrittik and Hauberg, S{\o}ren},
            journal={arXiv preprint arXiv:2405.17277},
            year={2024}
        }
        ```

    Parameters
    ----------
    num_matvecs
        The number of matrix-vector products aka the depth of the Krylov space.
        The deeper the Krylov space, the more accurate the factorisation tends to be.
        However, the computational complexity increases linearly
        with the number of matrix-vector products.
    materialize
        The value of this flag indicates whether the tridiagonal matrix
        should be returned in a sparse format (which means, as a tuple of diagonas)
        or as a dense matrix.
        The dense matrix is helpful if different decompositions should be used
        interchangeably. The sparse representation requires less memory.
    reortho
        The value of this parameter indicates whether to reorthogonalise the
        basis vectors during the forward pass.
        Reorthogonalisation makes the forward pass more expensive, but helps
        (significantly) with numerical stability.
    custom_vjp
        The value of this flag indicates whether to use a custom vector-Jacobian
        product as proposed by Krämer et al. (2024; bibtex above).
        Generally, using a custom VJP tends to be a good idea.
        However, due to JAX's mechanics, a custom VJP precludes the use of forward-mode
        differentiation
        ([see here](https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_vjp.html)),
        so don't use a custom VJP if you need forward-mode differentiation.

    Returns
    -------
    decompose
        A decomposition function that maps
        ``(matvec, vector, *params)`` to the decomposition.
        The decomposition is a tuple of (nested) arrays.
        The first element is the Krylov basis,
        the second element represents the tridiagonal matrix
        (how it is represented depends on the value of ``materialize''),
        the third element is
        the residual, and the fourth element is
        the (inverse of the) length of the initial vector.
    """
    if reortho == "full":
        return _tridiag_reortho_full(
            num_matvecs, custom_vjp=custom_vjp, materialize=materialize
        )
    if reortho == "none":
        return _tridiag_reortho_none(
            num_matvecs, custom_vjp=custom_vjp, materialize=materialize
        )

    msg = f"reortho={reortho} unsupported. Choose eiter {'full', 'none'}."
    raise ValueError(msg)


def _tridiag_reortho_full(num_matvecs: int, /, *, custom_vjp: bool, materialize: bool):
    # Implement via Arnoldi to use the reorthogonalised adjoints.
    # The complexity difference is minimal with full reortho.
    alg = hessenberg(num_matvecs, custom_vjp=custom_vjp, reortho="full")

    def estimate(matvec, vec, *params):
        Q, H, v, norm = alg(matvec, vec, *params)

        T = 0.5 * (H + H.T)
        diags = linalg.diagonal(T, offset=0)
        offdiags = linalg.diagonal(T, offset=1)

        matrix = (diags, offdiags)
        if materialize:
            matrix = _todense_tridiag_sym(diags, offdiags)

        return _DecompResult(
            Q_tall=Q, J_small=matrix, residual=v, init_length_inv=1.0 / norm
        )

    return estimate


def _todense_tridiag_sym(diag, off_diag):
    diag = linalg.diagonal_matrix(diag)
    offdiag1 = linalg.diagonal_matrix(off_diag, -1)
    offdiag2 = linalg.diagonal_matrix(off_diag, 1)
    return diag + offdiag1 + offdiag2


def _tridiag_reortho_none(num_matvecs: int, /, *, custom_vjp: bool, materialize: bool):
    def estimate(matvec, vec, *params):
        if num_matvecs < 0 or num_matvecs > len(vec):
            msg = _error_num_matvecs(num_matvecs, maxval=len(vec), minval=0)
            raise ValueError(msg)

        (Q, H), (q, b) = _estimate(matvec, vec, *params)
        v = b * q

        if materialize:
            H = _todense_tridiag_sym(*H)

        length = linalg.vector_norm(vec)
        return _DecompResult(
            Q_tall=Q.T, J_small=H, residual=v, init_length_inv=1.0 / length
        )

    def _estimate(matvec, vec, *params):
        *values, _ = _tridiag_forward(matvec, num_matvecs, vec, *params)
        return values

    def estimate_fwd(matvec, vec, *params):
        value = _estimate(matvec, vec, *params)
        return value, (value, (linalg.vector_norm(vec), *params))

    def estimate_bwd(matvec, cache, vjp_incoming):
        # Read incoming gradients and stack related quantities
        (dxs, (dalphas, dbetas)), (dx_last, dbeta_last) = vjp_incoming
        dxs = np.concatenate((dxs, dx_last[None]))
        dbetas = np.concatenate((dbetas, dbeta_last[None]))

        # Read the cache and stack related quantities
        ((xs, (alphas, betas)), (x_last, beta_last)), (vector_norm, *params) = cache
        xs = np.concatenate((xs, x_last[None]))
        betas = np.concatenate((betas, beta_last[None]))

        # Compute the adjoints, discard the adjoint states, and return the gradients
        grads, _lambdas_and_mus_and_nus = _tridiag_adjoint(
            matvec=matvec,
            params=params,
            initvec_norm=vector_norm,
            alphas=alphas,
            betas=betas,
            xs=xs,
            dalphas=dalphas,
            dbetas=dbetas,
            dxs=dxs,
        )
        return grads

    if custom_vjp:
        _estimate = func.custom_vjp(_estimate, nondiff_argnums=[0])
        _estimate.defvjp(estimate_fwd, estimate_bwd)  # type: ignore

    return estimate


def _tridiag_forward(matvec, num_matvecs, vec, *params):
    # Pre-allocate
    vectors = np.zeros((num_matvecs + 1, len(vec)))
    offdiags = np.zeros((num_matvecs,))
    diags = np.zeros((num_matvecs,))

    # Normalize (not all Lanczos implementations do that)
    v0 = vec / linalg.vector_norm(vec)
    vectors = vectors.at[0].set(v0)

    # Lanczos initialisation
    ((v1, offdiag), diag) = _tridiag_fwd_init(matvec, v0, *params)

    if num_matvecs == 0:
        decomposition = vectors[:-1], (diags, offdiags[:-1])
        remainder = v1, offdiag
        return decomposition, remainder, 1 / linalg.vector_norm(vec)

    # Store results
    k = 0
    vectors = vectors.at[k + 1].set(v1)
    offdiags = offdiags.at[k].set(offdiag)
    diags = diags.at[k].set(diag)

    # Run Lanczos-loop
    init = (v1, offdiag, v0), (vectors, diags, offdiags)
    step_fun = func.partial(_tridiag_fwd_step, matvec, params)
    (v1, offdiag, _), (vectors, diags, offdiags) = control_flow.fori_loop(
        lower=1, upper=num_matvecs, body_fun=step_fun, init_val=init
    )

    # Reorganise the outputs
    decomposition = vectors[:-1], (diags, offdiags[:-1])
    remainder = v1, offdiag
    return decomposition, remainder, 1 / linalg.vector_norm(vec)


def _tridiag_fwd_init(matvec, vec, *params):
    """Initialize Lanczos' algorithm.

    Solve A x_{k} = a_k x_k + b_k x_{k+1}
    for x_{k+1}, a_k, and b_k, using
    orthogonality of the x_k.
    """
    a = vec @ (matvec(vec, *params))
    r = (matvec(vec, *params)) - a * vec
    b = linalg.vector_norm(r)
    x = r / b
    return (x, b), a


def _tridiag_fwd_step(matvec, params, i, val):
    (v1, offdiag, v0), (vectors, diags, offdiags) = val
    ((v1, offdiag), diag), v0 = (
        _tridiag_fwd_step_apply(matvec, v1, offdiag, v0, *params),
        v1,
    )

    # Store results
    vectors = vectors.at[i + 1].set(v1)
    offdiags = offdiags.at[i].set(offdiag)
    diags = diags.at[i].set(diag)

    return (v1, offdiag, v0), (vectors, diags, offdiags)


def _tridiag_fwd_step_apply(matvec, vec, b, vec_previous, *params):
    """Apply Lanczos' recurrence."""
    a = vec @ (matvec(vec, *params))
    r = matvec(vec, *params) - a * vec - b * vec_previous
    b = linalg.vector_norm(r)
    x = r / b
    return (x, b), a


def _tridiag_adjoint(
    *, matvec, params, initvec_norm, alphas, betas, xs, dalphas, dbetas, dxs
):
    def adjoint_step(xi_and_lambda, inputs):
        return _tridiag_adjoint_step(
            *xi_and_lambda, matvec=matvec, params=params, **inputs
        )

    # Scan over all input gradients and output values
    xs0 = xs
    xs0 = xs0.at[-1, :].set(np.zeros_like(xs[-1, :]))

    loop_over = {
        "dx": dxs[:-1],
        "da": dalphas,
        "db": dbetas,
        "xs": (xs[1:], xs[:-1]),
        "a": alphas,
        "b": betas,
    }
    init_val = (xs0, -dxs[-1], np.zeros_like(dxs[-1]))
    (_, lambda_1, _lambda_2), (grad_summands, *other) = control_flow.scan(
        adjoint_step, init_val, xs=loop_over, reverse=True
    )

    # Compute the gradients
    grad_matvec = tree.tree_map(lambda s: np.sum(s, axis=0), grad_summands)
    grad_initvec = ((lambda_1.T @ xs[0]) * xs[0] - lambda_1) / initvec_norm

    # Return values
    return (grad_initvec, grad_matvec), (lambda_1, *other)


def _tridiag_adjoint_step(
    xs_all, xi, lambda_plus, /, *, matvec, params, dx, da, db, xs, a, b
):
    # Read inputs
    (xplus, x) = xs

    # Apply formula
    xi /= b
    mu = db - lambda_plus.T @ x + xplus.T @ xi
    nu = da + x.T @ xi
    lambda_ = -xi + mu * xplus + nu * x

    # Value-and-grad of matrix-vector product
    matvec_lambda, vjp = func.vjp(lambda *p: matvec(lambda_, *p), *params)
    (gradient_increment,) = vjp(x)

    # Prepare next step
    xi = -dx - matvec_lambda + a * lambda_ + b * lambda_plus - b * nu * xplus

    # Return values
    return (xs_all, xi, lambda_), (gradient_increment, lambda_, mu, nu, xi)


def hessenberg(
    num_matvecs, /, *, reortho: str, custom_vjp: bool = True, reortho_vjp: str = "match"
):
    r"""Construct a **Hessenberg-factorisation** via the Arnoldi iteration.

    Uses pre-allocation, and full reorthogonalisation if `reortho` is set to `"full"`.
    It tends to be a good idea to use full reorthogonalisation.

    This algorithm works for **arbitrary matrices**.

    Setting `custom_vjp` to `True` implies using efficient, numerically stable
    gradients of the Arnoldi iteration according to what has been proposed by
    Krämer et al. (2024).
    These gradients are exact, so there is little reason not to use them.
    If you use this configuration,
    please consider citing Krämer et al. (2024; bibtex below).

    ??? note "BibTex for Krämer et al. (2024)"
        ```bibtex
        @article{kraemer2024gradients,
            title={Gradients of functions of large matrices},
            author={Kr\"amer, Nicholas and Moreno-Mu\~noz, Pablo and
            Roy, Hrittik and Hauberg, S{\o}ren},
            journal={arXiv preprint arXiv:2405.17277},
            year={2024}
        }
        ```
    """
    reortho_expected = ["none", "full"]
    if reortho not in reortho_expected:
        msg = f"Unexpected input for {reortho}: either of {reortho_expected} expected."
        raise TypeError(msg)

    def estimate(matvec, v, *params):
        matvec_convert, aux_args = func.closure_convert(matvec, v, *params)
        return _estimate(matvec_convert, v, *params, *aux_args)

    def _estimate(matvec_convert: Callable, v, *params):
        reortho_ = reortho_vjp if reortho_vjp != "match" else reortho_vjp
        return _hessenberg_forward(
            matvec_convert, num_matvecs, v, *params, reortho=reortho_
        )

    def estimate_fwd(matvec_convert: Callable, v, *params):
        outputs = _estimate(matvec_convert, v, *params)
        return outputs, (outputs, params)

    def estimate_bwd(matvec_convert: Callable, cache, vjp_incoming):
        (Q, H, r, c), params = cache
        dQ, dH, dr, dc = vjp_incoming

        return _hessenberg_adjoint(
            matvec_convert,
            *params,
            Q=Q,
            H=H,
            r=r,
            c=c,
            dQ=dQ,
            dH=dH,
            dr=dr,
            dc=dc,
            reortho=reortho,
        )

    if custom_vjp:
        _estimate = func.custom_vjp(_estimate, nondiff_argnums=(0,))
        _estimate.defvjp(estimate_fwd, estimate_bwd)  # type: ignore
    return estimate


def _hessenberg_forward(matvec, num_matvecs, v, *params, reortho: str):
    if num_matvecs < 0 or num_matvecs > len(v):
        msg = _error_num_matvecs(num_matvecs, maxval=len(v), minval=0)
        raise ValueError(msg)

    # Initialise the variables
    (n,), k = np.shape(v), num_matvecs
    Q = np.zeros((n, k), dtype=v.dtype)
    H = np.zeros((k, k), dtype=v.dtype)
    initlength = np.sqrt(linalg.inner(v, v))
    init = (Q, H, v, initlength)

    if num_matvecs == 0:
        return _DecompResult(
            Q_tall=Q, J_small=H, residual=v, init_length_inv=1 / initlength
        )

    # Fix the step function
    def forward_step(i, val):
        return _hessenberg_forward_step(*val, matvec, *params, idx=i, reortho=reortho)

    # Loop and return
    Q, H, v, _length = control_flow.fori_loop(0, k, forward_step, init)
    return _DecompResult(
        Q_tall=Q, J_small=H, residual=v, init_length_inv=1 / initlength
    )


def _hessenberg_forward_step(Q, H, v, length, matvec, *params, idx, reortho: str):
    # Save
    v /= length
    Q = Q.at[:, idx].set(v)

    # Evaluate
    v = matvec(v, *params)

    # Orthonormalise
    h = Q.T @ v
    v = v - Q @ h

    # Re-orthonormalise
    if reortho != "none":
        v = v - Q @ (Q.T @ v)

    # Read the length
    length = np.sqrt(linalg.inner(v, v))

    # Save
    h = h.at[idx + 1].set(length)
    H = H.at[:, idx].set(h)

    return Q, H, v, length


def _hessenberg_adjoint(matvec, *params, Q, H, r, c, dQ, dH, dr, dc, reortho: str):
    # Extract the matrix shapes from Q
    _, num_matvecs = np.shape(Q)

    # Prepare a bunch of auxiliary matrices

    def lower(m):
        m_tril = np.tril(m)
        return m_tril - 0.5 * _extract_diag(m_tril)

    e_1, e_K = np.eye(num_matvecs)[[0, -1], :]
    lower_mask = lower(np.ones((num_matvecs, num_matvecs)))

    # Initialise
    eta = dH @ e_K - Q.T @ dr
    lambda_k = dr + Q @ eta
    Lambda = np.zeros_like(Q)
    Gamma = np.zeros_like(dQ.T @ Q)
    dp = tree.tree_map(np.zeros_like, params)

    # Prepare more  auxiliary matrices
    Pi_xi = dQ.T + linalg.outer(eta, r)
    Pi_gamma = -dc * c * linalg.outer(e_1, e_1) + H @ dH.T - (dQ.T @ Q)

    # Prepare reorthogonalisation:
    P = Q.T
    ps = dH.T
    ps_mask = np.tril(np.ones((num_matvecs, num_matvecs)), 1)

    # Loop over those values
    indices = np.arange(0, len(H), step=1)
    beta_minuses = np.concatenate([np.ones((1,)), linalg.diagonal(H, -1)])
    alphas = linalg.diagonal(H)
    beta_pluses = H - _extract_diag(H) - _extract_diag(H, -1)
    scan_over = {
        "beta_minus": beta_minuses,
        "alpha": alphas,
        "beta_plus": beta_pluses,
        "idx": indices,
        "lower_mask": lower_mask,
        "Pi_gamma": Pi_gamma,
        "Pi_xi": Pi_xi,
        "p": ps,
        "p_mask": ps_mask,
        "q": Q.T,
    }

    # Fix the step function
    def adjoint_step(x, y):
        output = _hessenberg_adjoint_step(
            *x, **y, matvec=matvec, params=params, Q=Q, reortho=reortho
        )
        return output, ()

    # Scan
    init = (lambda_k, Lambda, Gamma, P, dp)
    result, _ = control_flow.scan(adjoint_step, init, xs=scan_over, reverse=True)
    (lambda_k, Lambda, Gamma, _P, dp) = result

    # Solve for the input gradient
    dv = lambda_k * c

    return dv, *dp


def _hessenberg_adjoint_step(
    # Running variables
    lambda_k,
    Lambda,
    Gamma,
    P,
    dp,
    *,
    # Matrix-vector product
    matvec,
    params,
    # Loop over: index
    idx,
    # Loop over: submatrices of H
    beta_minus,
    alpha,
    beta_plus,
    # Loop over: auxiliary variables for Gamma
    lower_mask,
    Pi_gamma,
    Pi_xi,
    q,
    # Loop over: reorthogonalisation
    p,
    p_mask,
    # Other parameters
    Q,
    reortho: str,
):
    # Reorthogonalise
    if reortho == "full":
        P = p_mask[:, None] * P
        p = p_mask * p
        lambda_k = lambda_k - P.T @ (P @ lambda_k) + P.T @ p

    # Transposed matvec and parameter-gradient in a single matvec
    _, vjp = func.vjp(lambda u, v: matvec(u, *v), q, params)
    vecmat_lambda, dp_increment = vjp(lambda_k)
    dp = tree.tree_map(lambda g, h: g + h, dp, dp_increment)

    # Solve for (Gamma + Gamma.T) e_K
    tmp = lower_mask * (Pi_gamma - vecmat_lambda @ Q)
    Gamma = Gamma.at[idx, :].set(tmp)

    # Solve for the next lambda (backward substitution step)
    Lambda = Lambda.at[:, idx].set(lambda_k)
    xi = Pi_xi + (Gamma + Gamma.T)[idx, :] @ Q.T
    lambda_k = xi - (alpha * lambda_k - vecmat_lambda) - beta_plus @ Lambda.T
    lambda_k /= beta_minus
    return lambda_k, Lambda, Gamma, P, dp


def _extract_diag(x, offset=0):
    diag = linalg.diagonal(x, offset=offset)
    return linalg.diagonal_matrix(diag, offset=offset)


def bidiag(num_matvecs: int, /, materialize: bool = True):
    """Construct an implementation of **bidiagonalisation**.

    Uses pre-allocation and full reorthogonalisation.

    Works for **arbitrary matrices**. No symmetry required.

    Decompose a matrix into a product of orthogonal-**bidiagonal**-orthogonal matrices.
    Use this algorithm for approximate **singular value** decompositions.

    Internally, Matfree uses JAX to turn matrix-vector- into vector-matrix-products.

    ??? note "A note about differentiability"
        Unlike [tridiag_sym][matfree.decomp.tridiag_sym] or
        [hessenberg][matfree.decomp.hessenberg], this function's reverse-mode
        derivatives are very efficient. Custom gradients for bidiagonalisation
        are a work in progress, and if you need to differentiate the decompositions,
        consider using [tridiag_sym][matfree.decomp.tridiag_sym] for the time being.

    """

    def estimate(Av: Callable, v0, *parameters):
        # Infer the size of A from v0
        (ncols,) = np.shape(v0)
        w0_like = func.eval_shape(Av, v0, *parameters)
        (nrows,) = np.shape(w0_like)

        # Complain if the shapes don't match
        max_num_matvecs = min(nrows, ncols)
        if num_matvecs > max_num_matvecs or num_matvecs < 0:
            msg = _error_num_matvecs(num_matvecs, maxval=min(nrows, ncols), minval=0)
            raise ValueError(msg)

        v0_norm, length = _normalise(v0)
        init_val = init(v0_norm, nrows=nrows, ncols=ncols)

        if num_matvecs == 0:
            uk_all_T, J, vk_all, (beta, vk) = extract(init_val)
            return _DecompResult(
                Q_tall=(uk_all_T, vk_all.T),
                J_small=J,
                residual=beta * vk,
                init_length_inv=1 / length,
            )

        def body_fun(_, s):
            return step(Av, s, *parameters)

        result = control_flow.fori_loop(
            0, num_matvecs, body_fun=body_fun, init_val=init_val
        )
        uk_all_T, J, vk_all, (beta, vk) = extract(result)
        return _DecompResult(
            Q_tall=(uk_all_T, vk_all.T),
            J_small=J,
            residual=beta * vk,
            init_length_inv=1 / length,
        )

    class State(containers.NamedTuple):
        i: int
        Us: Array
        Vs: Array
        alphas: Array
        betas: Array
        beta: Array
        vk: Array

    def init(init_vec: Array, *, nrows, ncols) -> State:
        alphas = np.zeros((num_matvecs,))
        betas = np.zeros((num_matvecs,))
        Us = np.zeros((num_matvecs, nrows))
        Vs = np.zeros((num_matvecs, ncols))
        v0, _ = _normalise(init_vec)
        return State(0, Us, Vs, alphas, betas, np.zeros(()), v0)

    def step(Av, state: State, *parameters) -> State:
        i, Us, Vs, alphas, betas, beta, vk = state
        Vs = Vs.at[i].set(vk)
        betas = betas.at[i].set(beta)

        # Use jax.vjp to evaluate the vector-matrix product
        Av_eval, vA = func.vjp(lambda v: Av(v, *parameters), vk)
        uk = Av_eval - beta * Us[i - 1]
        uk, alpha = _normalise(uk)
        uk, *_ = _gram_schmidt_classical(uk, Us)  # full reorthogonalisation
        Us = Us.at[i].set(uk)
        alphas = alphas.at[i].set(alpha)

        (vA_eval,) = vA(uk)
        vk = vA_eval - alpha * vk
        vk, beta = _normalise(vk)
        vk, *_ = _gram_schmidt_classical(vk, Vs)  # full reorthogonalisation

        return State(i + 1, Us, Vs, alphas, betas, beta, vk)

    def extract(state: State, /):
        _, uk_all, vk_all, alphas, betas, beta, vk = state

        if materialize:
            B = _todense_bidiag(alphas, betas[1:])
            return uk_all.T, B, vk_all, (beta, vk)

        return uk_all.T, (alphas, betas[1:]), vk_all, (beta, vk)

    def _gram_schmidt_classical(vec, vectors):  # Gram-Schmidt
        vec, coeffs = control_flow.scan(_gram_schmidt_classical_step, vec, xs=vectors)
        vec, length = _normalise(vec)
        return vec, length, coeffs

    def _gram_schmidt_classical_step(vec1, vec2):
        coeff = linalg.inner(vec1, vec2)
        vec_ortho = vec1 - coeff * vec2
        return vec_ortho, coeff

    def _normalise(vec):
        length = linalg.vector_norm(vec)
        return vec / length, length

    def _todense_bidiag(d, e):
        diag = linalg.diagonal_matrix(d)
        offdiag = linalg.diagonal_matrix(e, 1)
        return diag + offdiag

    return estimate


def _error_num_matvecs(num, maxval, minval):
    msg1 = f"Parameter 'num_matvecs'={num} exceeds the acceptable range. "
    msg2 = f"Expected: {minval} <= num_matvecs <= {maxval}."
    return msg1 + msg2
