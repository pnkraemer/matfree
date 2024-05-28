"""Matrix-free matrix decompositions.

This module includes various Lanczos-decompositions of matrices
(tri-diagonal, bi-diagonal, etc.).

For stochastic Lanczos quadrature, see
[matfree.stochtrace_funm][matfree.stochtrace_funm].
For matrix-function-vector products, see
[matfree.funm][matfree.funm].
"""

from matfree.backend import containers, control_flow, func, linalg, np, tree_util
from matfree.backend.typing import Array, Callable, Tuple


class _LanczosAlg(containers.NamedTuple):
    """Lanczos decomposition algorithm."""

    init: Callable
    """Initialise the state of the algorithm. Usually, this involves pre-allocation."""

    step: Callable
    """Compute the next iteration."""

    extract: Callable
    """Extract the solution from the state of the algorithm."""

    lower_upper: Tuple[int, int]
    """Range of the for-loop used to decompose a matrix."""


def tridiag_sym(
    matvec, krylov_depth, /, *, reortho: str = "full", custom_vjp: bool = True
):
    """Construct an implementation of **tridiagonalisation**.

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
            author={Kr\"amer, Nicholas and Moreno-Mu\\~noz, Pablo and
            Roy, Hrittik and Hauberg S\\o{}ren},
            journal={arXiv preprint arXiv:2405.17277},
            year={2024}
        }
        ```

    """

    if reortho == "full":
        return _tridiag_reortho_full(matvec, krylov_depth, custom_vjp=custom_vjp)
    if reortho == "none":
        return _tridiag_reortho_none(matvec, krylov_depth, custom_vjp=custom_vjp)

    msg = f"reortho={reortho} unsupported. Choose eiter {'full', 'none'}."
    raise ValueError(msg)


def _tridiag_reortho_full(matvec, krylov_depth, /, *, custom_vjp):
    # Implement via Arnoldi to use the reorthogonalised adjoints.
    # The complexity difference is minimal with full reortho.
    alg = hessenberg(matvec, krylov_depth, custom_vjp=custom_vjp, reortho="full")

    def estimate(vec, *params):
        Q, H, v, _norm = alg(vec, *params)

        T = 0.5 * (H + H.T)
        diags = linalg.diagonal(T, offset=0)
        offdiags = linalg.diagonal(T, offset=1)
        decomposition = (Q.T, (diags, offdiags))
        remainder = (v / linalg.vector_norm(v), linalg.vector_norm(v))
        return decomposition, remainder

    return estimate


def _tridiag_reortho_none(matvec, krylov_depth, /, *, custom_vjp):
    def estimate(vec, *params):
        *values, _ = _tridiag_forward(matvec, krylov_depth, vec, *params)
        return values

    def estimate_fwd(vec, *params):
        value = estimate(vec, *params)
        return value, (value, (linalg.vector_norm(vec), *params))

    def estimate_bwd(cache, vjp_incoming):
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
        estimate = func.custom_vjp(estimate)
        estimate.defvjp(estimate_fwd, estimate_bwd)  # type: ignore

    return estimate


def _tridiag_forward(matvec, krylov_depth, vec, *params):
    # Pre-allocate
    vectors = np.zeros((krylov_depth + 1, len(vec)))
    offdiags = np.zeros((krylov_depth,))
    diags = np.zeros((krylov_depth,))

    # Normalize (not all Lanczos implementations do that)
    v0 = vec / linalg.vector_norm(vec)
    vectors = vectors.at[0].set(v0)

    # Lanczos initialisation
    ((v1, offdiag), diag) = _tridiag_fwd_init(matvec, v0, *params)

    # Store results
    k = 0
    vectors = vectors.at[k + 1].set(v1)
    offdiags = offdiags.at[k].set(offdiag)
    diags = diags.at[k].set(diag)

    # Run Lanczos-loop
    init = (v1, offdiag, v0), (vectors, diags, offdiags)
    step_fun = func.partial(_tridiag_fwd_step, matvec, params)
    _, (vectors, diags, offdiags) = control_flow.fori_loop(
        lower=1, upper=krylov_depth, body_fun=step_fun, init_val=init
    )

    # Reorganise the outputs
    decomposition = vectors[:-1], (diags, offdiags[:-1])
    remainder = vectors[-1], offdiags[-1]
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
    grad_matvec = tree_util.tree_map(lambda s: np.sum(s, axis=0), grad_summands)
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
    matvec,
    krylov_depth,
    /,
    *,
    reortho: str,
    custom_vjp: bool = True,
    reortho_vjp: str = "match",
):
    """Construct a **Hessenberg-factorisation** via the Arnoldi iteration.

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
            author={Kr\"amer, Nicholas and Moreno-Mu\\~noz, Pablo and
            Roy, Hrittik and Hauberg S\\o{}ren},
            journal={arXiv preprint arXiv:2405.17277},
            year={2024}
        }
        ```

    """
    reortho_expected = ["none", "full"]
    if reortho not in reortho_expected:
        msg = f"Unexpected input for {reortho}: either of {reortho_expected} expected."
        raise TypeError(msg)

    def estimate_public(v, *params):
        matvec_convert, aux_args = func.closure_convert(matvec, v, *params)
        return estimate_backend(matvec_convert, v, *params, *aux_args)

    def estimate_backend(matvec_convert: Callable, v, *params):
        reortho_ = reortho_vjp if reortho_vjp != "match" else reortho_vjp
        return _hessenberg_forward(
            matvec_convert, krylov_depth, v, *params, reortho=reortho_
        )

    def estimate_fwd(matvec_convert: Callable, v, *params):
        outputs = estimate_backend(matvec_convert, v, *params)
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
        estimate_backend = func.custom_vjp(estimate_backend, nondiff_argnums=(0,))
        estimate_backend.defvjp(estimate_fwd, estimate_bwd)  # type: ignore
    return estimate_public


def _hessenberg_forward(matvec, krylov_depth, v, *params, reortho: str):
    if krylov_depth < 1 or krylov_depth > len(v):
        msg = f"Parameter depth {krylov_depth} is outside the expected range"
        raise ValueError(msg)

    # Initialise the variables
    (n,), k = np.shape(v), krylov_depth
    Q = np.zeros((n, k), dtype=v.dtype)
    H = np.zeros((k, k), dtype=v.dtype)
    initlength = np.sqrt(linalg.inner(v, v))
    init = (Q, H, v, initlength)

    # Fix the step function
    def forward_step(i, val):
        return _hessenberg_forward_step(*val, matvec, *params, idx=i, reortho=reortho)

    # Loop and return
    Q, H, v, _length = control_flow.fori_loop(0, k, forward_step, init)
    return Q, H, v, 1 / initlength


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
    _, krylov_depth = np.shape(Q)

    # Prepare a bunch of auxiliary matrices

    def lower(m):
        m_tril = np.tril(m)
        return m_tril - 0.5 * _extract_diag(m_tril)

    e_1, e_K = np.eye(krylov_depth)[[0, -1], :]
    lower_mask = lower(np.ones((krylov_depth, krylov_depth)))

    # Initialise
    eta = dH @ e_K - Q.T @ dr
    lambda_k = dr + Q @ eta
    Lambda = np.zeros_like(Q)
    Gamma = np.zeros_like(dQ.T @ Q)
    dp = tree_util.tree_map(np.zeros_like, params)

    # Prepare more  auxiliary matrices
    Pi_xi = dQ.T + linalg.outer(eta, r)
    Pi_gamma = -dc * c * linalg.outer(e_1, e_1) + H @ dH.T - (dQ.T @ Q)

    # Prepare reorthogonalisation:
    P = Q.T
    ps = dH.T
    ps_mask = np.tril(np.ones((krylov_depth, krylov_depth)), 1)

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
    dp = tree_util.tree_map(lambda g, h: g + h, dp, dp_increment)

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


def bidiag(
    Av: Callable, vA: Callable, depth, /, matrix_shape, validate_unit_2_norm=False
):
    """Construct an implementation of **bidiagonalisation**.

    Uses pre-allocation and full reorthogonalisation.

    Works for **arbitrary matrices**. No symmetry required.

    Decompose a matrix into a product of orthogonal-**bidiagonal**-orthogonal matrices.
    Use this algorithm for approximate **singular value** decompositions.

    ??? note "A note about differentiability"
        Unlike [tridiag_sym][matfree.decomp.tridiag_sym] or
        [hessenberg][matfree.decomp.hessenberg], this function's reverse-mode
        derivatives are very efficient. Custom gradients for bidiagonalisation
        are a work in progress, and if you need to differentiate the decompositions,
        consider using [tridiag_sym][matfree.decomp.tridiag_sym] for the time being.

    """
    nrows, ncols = matrix_shape
    max_depth = min(nrows, ncols) - 1
    if depth > max_depth or depth < 0:
        msg1 = f"Depth {depth} exceeds the matrix' dimensions. "
        msg2 = f"Expected: 0 <= depth <= min(nrows, ncols) - 1 = {max_depth} "
        msg3 = f"for a matrix with shape {matrix_shape}."
        raise ValueError(msg1 + msg2 + msg3)

    class State(containers.NamedTuple):
        i: int
        Us: Array
        Vs: Array
        alphas: Array
        betas: Array
        beta: Array
        vk: Array

    def init(init_vec: Array) -> State:
        if validate_unit_2_norm:
            init_vec = _validate_unit_2_norm(init_vec)

        alphas = np.zeros((depth + 1,))
        betas = np.zeros((depth + 1,))
        Us = np.zeros((depth + 1, nrows))
        Vs = np.zeros((depth + 1, ncols))
        v0, _ = _normalise(init_vec)
        return State(0, Us, Vs, alphas, betas, np.zeros(()), v0)

    def apply(state: State, *parameters) -> State:
        i, Us, Vs, alphas, betas, beta, vk = state
        Vs = Vs.at[i].set(vk)
        betas = betas.at[i].set(beta)

        uk = Av(vk, *parameters) - beta * Us[i - 1]
        uk, alpha = _normalise(uk)
        uk, *_ = _gram_schmidt_classical(uk, Us)  # full reorthogonalisation
        Us = Us.at[i].set(uk)
        alphas = alphas.at[i].set(alpha)

        vk = vA(uk, *parameters) - alpha * vk
        vk, beta = _normalise(vk)
        vk, *_ = _gram_schmidt_classical(vk, Vs)  # full reorthogonalisation

        return State(i + 1, Us, Vs, alphas, betas, beta, vk)

    def extract(state: State, /):
        _, uk_all, vk_all, alphas, betas, beta, vk = state
        return uk_all.T, (alphas, betas[1:]), vk_all, (beta, vk)

    alg = _LanczosAlg(
        init=init, step=apply, extract=extract, lower_upper=(0, depth + 1)
    )
    return func.partial(_decompose_fori_loop, algorithm=alg)


def _validate_unit_2_norm(v, /):
    # todo: replace this functionality with normalising internally.
    #
    # Lanczos assumes a unit-2-norm vector as an input
    # We cannot raise an error based on values of the init_vec,
    # but we can make it obvious that the result is unusable.
    is_not_normalized = np.abs(linalg.vector_norm(v) - 1.0) > 10 * np.finfo_eps(v.dtype)
    return control_flow.cond(
        is_not_normalized, lambda s: np.nan() * np.ones_like(s), lambda s: s, v
    )


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


def _decompose_fori_loop(v0, *parameters, algorithm: _LanczosAlg):
    r"""Decompose a matrix purely based on matvec-products with A.

    The behaviour of this function is equivalent to

    ```python
    def decompose(v0, *matvec_funs, algorithm):
        init, step, extract, (lower, upper) = algorithm
        state = init(v0)
        for _ in range(lower, upper):
            state = step(state, *matvec_funs)
        return extract(state)
    ```

    but the implementation uses JAX' fori_loop.
    """
    init, step, extract, (lower, upper) = algorithm
    init_val = init(v0)

    def body_fun(_, s):
        return step(s, *parameters)

    result = control_flow.fori_loop(lower, upper, body_fun=body_fun, init_val=init_val)
    return extract(result)
