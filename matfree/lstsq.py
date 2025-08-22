"""Matrix-free algorithms for least-squares-type problems."""

from matfree.backend import containers, control_flow, func, linalg, np, tree
from matfree.backend.typing import Array, Callable

# todo: make functions compatible with pytree-valued vecmats


_LARGE_VALUE = 1e10
"""A placeholder for np.inf.

In stopping criteria, if we divide something by something that's almost zero,
we don't want the result to be infinite
but only 'large enough to terminate the iteration'.
"""


def lsmr(
    *,
    atol: float = 1e-6,
    btol: float = 1e-6,
    ctol: float = 1e-8,
    maxiter: int = 1_000_000,
    while_loop: Callable = control_flow.while_loop,
    custom_vjp: bool = True,
):
    """Construct an experimental implementation of LSMR.

    Follows the [implementation in SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html),
    but uses JAX.
    """
    # todo: stop iteration when NaN or Inf are detected.

    @tree.register_dataclass
    @containers.dataclass
    class State:
        """LSMR state."""

        # Iteration count:
        itn: int
        # Bidiagonalisation variables:
        alpha: float
        u: Array
        v: Array
        # LSMR-specific variables:
        alphabar: float
        rhobar: float
        rho: float
        zeta: float
        sbar: float
        cbar: float
        zetabar: float
        hbar: Array
        h: Array
        x: Array
        # Variables for estimation of ||r||:
        betadd: float
        thetatilde: float
        rhodold: float
        betad: float
        tautildeold: float
        d: float
        # Variables for estimation of ||A|| and cond(A)
        normA2: float
        maxrbar: float
        minrbar: float
        normA: float
        condA: float
        normx: float

        # Variables for use in stopping rules
        normar: float
        normr: float
        # Reason for stopping
        istop: int

    # more often than not, the matvec is defined after the LSMR
    # solver has been constructed. So it's part of the run()
    # function, not the LSMR constructor.
    def run(vecmat, b, *vecmat_args, x0, damp=0.0):
        x_like = func.eval_shape(vecmat, b, *vecmat_args)
        (ncols,) = x_like.shape
        x = x0 if x0 is not None else np.zeros(ncols, dtype=b.dtype)

        # Combine the lstsq_fun wiht a closure convert, because
        # typically, vecmat is a lambda function and if we want to
        # have explicit parameter-VJPs, all parameters need to be explicit.
        # This means that in this function here, we always use lstsq_public
        # (and return lstsq_public!), but provide lstsq_fun with the custom VJP.
        # Thereby, the function that gets the custom VJP is, from now on, only
        # called after a previous call to closure convert which 'fixes' all namespaces.
        vecmat_closure, args = func.closure_convert(
            lambda s: vecmat(s, *vecmat_args), b
        )
        return _run(vecmat_closure, b, args, x, damp)

    def _run(vecmat, b, vecmat_args, x0, damp):
        def vecmat_noargs(v):
            return vecmat(v, *vecmat_args)

        state, normb, matvec_noargs = init(vecmat_noargs, b, x0)
        step_fun = make_step(matvec_noargs, normb=normb, damp=damp)
        cond_fun = make_cond_fun()
        state = while_loop(cond_fun, step_fun, state)
        stats_ = stats(state)
        return state.x - x0, stats_

    def init(vecmat, b, x):
        normb = linalg.vector_norm(b)
        beta = normb

        u = b
        u = u / np.where(beta > 0, beta, 1.0)

        v, matvec = func.vjp(vecmat, u)
        alpha = linalg.vector_norm(v)
        v = v / np.where(alpha > 0, alpha, 1)
        v = np.where(beta == 0, np.zeros_like(v), v)
        alpha = np.where(beta == 0, np.zeros_like(alpha), alpha)

        # Initialize variables for 1st iteration.

        zetabar = alpha * beta
        alphabar = alpha
        rho = 1.0
        rhobar = 1.0
        cbar = 1.0
        sbar = 0.0

        h = v
        hbar = np.zeros_like(x)

        # Initialize variables for estimation of ||r||.

        betadd = beta
        betad = 0.0
        rhodold = 1.0
        tautildeold = 0.0
        thetatilde = 0.0
        zeta = 0.0
        d = 0.0

        # Initialize variables for estimation of ||A|| and cond(A)

        normA2 = alpha * alpha
        maxrbar = 0.0
        minrbar = 1e10
        normA = np.sqrt(normA2)
        condA = 1.0
        normx = 0.0

        # Items for use in stopping rules, normb set earlier
        normr = beta

        # Reverse the order here from the original matlab code because
        # there was an error on return when arnorm==0
        normar = alpha * beta

        # Main iteration loop.
        state = State(  # type: ignore
            itn=0,
            alpha=alpha,
            u=u,
            v=v,
            alphabar=alphabar,
            rho=rho,
            rhobar=rhobar,
            zeta=zeta,
            sbar=sbar,
            cbar=cbar,
            zetabar=zetabar,
            hbar=hbar,
            h=h,
            x=x,
            betadd=betadd,
            thetatilde=thetatilde,
            rhodold=rhodold,
            betad=betad,
            tautildeold=tautildeold,
            d=d,
            normA2=normA2,
            maxrbar=maxrbar,
            minrbar=minrbar,
            normar=normar,
            normr=normr,
            normA=normA,
            condA=condA,
            normx=normx,
            istop=0,
        )
        state = tree.tree_map(np.asarray, state)
        return state, normb, lambda *a: matvec(*a)[0]

    def make_step(matvec, normb: float, damp: float) -> Callable:
        def step(state: State) -> State:
            # Perform the next step of the bidiagonalization

            Av, A_t = func.vjp(matvec, state.v)
            u = Av - state.alpha * state.u
            beta = linalg.vector_norm(u)

            u = u / np.where(beta > 0, beta, 1.0)
            v = A_t(u)[0] - beta * state.v
            alpha = linalg.vector_norm(v)
            v = v / np.where(alpha > 0, alpha, 1)

            # Construct rotation Qhat_{k,2k+1}.

            chat, shat, alphahat = _sym_ortho(state.alphabar, damp)

            # Use a plane rotation (Q_i) to turn B_i to R_i

            rhoold = state.rho
            c, s, rho = _sym_ortho(alphahat, beta)
            thetanew = s * alpha
            alphabar = c * alpha

            # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

            rhobarold = state.rhobar
            zetaold = state.zeta
            thetabar = state.sbar * rho
            rhotemp = state.cbar * rho
            cbar, sbar, rhobar = _sym_ortho(rhotemp, thetanew)
            zeta = cbar * state.zetabar
            zetabar = -sbar * state.zetabar

            # Update h, h_hat, x.

            hbar = state.h - state.hbar * (thetabar * rho / (rhoold * rhobarold))
            x = state.x + (zeta / (rho * rhobar)) * hbar
            h = v - state.h * (thetanew / rho)

            # Estimate of ||r||.

            # Apply rotation Qhat_{k,2k+1}.
            betaacute = chat * state.betadd
            betacheck = -shat * state.betadd

            # Apply rotation Q_{k,k+1}.
            betahat = c * betaacute
            betadd = -s * betaacute

            # Apply rotation Qtilde_{k-1}.

            thetatildeold = state.thetatilde
            ctildeold, stildeold, rhotildeold = _sym_ortho(state.rhodold, thetabar)
            thetatilde = stildeold * rhobar
            rhodold = ctildeold * rhobar
            betad = -stildeold * state.betad + ctildeold * betahat

            tautildeold = (zetaold - thetatildeold * state.tautildeold) / rhotildeold
            taud = (zeta - thetatilde * tautildeold) / rhodold
            d = state.d + betacheck * betacheck
            normr = np.sqrt(d + (betad - taud) ** 2 + betadd * betadd)

            # Estimate ||A||.
            normA2 = state.normA2 + beta * beta
            normA = np.sqrt(normA2)
            normA2 = normA2 + alpha * alpha

            # Estimate cond(A).
            maxrbar = np.elementwise_max(state.maxrbar, rhobarold)
            minrbar = np.where(
                state.itn > 1,
                np.elementwise_min(state.minrbar, rhobarold),
                state.minrbar,
            )
            condA = np.elementwise_max(maxrbar, rhotemp) / np.elementwise_min(
                minrbar, rhotemp
            )

            # Compute norms for convergence testing.
            normar = np.abs(zetabar)
            normx = linalg.vector_norm(x)

            # Check whether we should stop
            itn = state.itn + 1
            test1 = normr / normb
            z = normA * normr
            z_safe = np.where(z != 0, z, 1.0)
            test2 = np.where(z != 0, normar / z_safe, _LARGE_VALUE)
            test3 = 1 / condA
            t1 = test1 / (1 + normA * normx / normb)
            rtol = btol + atol * normA * normx / normb

            # Early exits
            istop = 0
            istop = np.where(normar == 0, 9, istop)
            istop = np.where(normb == 0, 8, istop)
            istop = np.where(itn >= maxiter, 7, istop)
            istop = np.where(1 + test3 <= 1, 6, istop)
            istop = np.where(1 + test2 <= 1, 5, istop)
            istop = np.where(1 + t1 <= 1, 4, istop)
            istop = np.where(test3 <= ctol, 3, istop)
            istop = np.where(test2 <= atol, 2, istop)
            istop = np.where(test1 <= rtol, 1, istop)

            return State(  # type: ignore
                itn=itn,
                alpha=alpha,
                u=u,
                v=v,
                alphabar=alphabar,
                rho=rho,
                rhobar=rhobar,
                zeta=zeta,
                sbar=sbar,
                cbar=cbar,
                zetabar=zetabar,
                hbar=hbar,
                h=h,
                x=x,
                betadd=betadd,
                thetatilde=thetatilde,
                rhodold=rhodold,
                betad=betad,
                tautildeold=tautildeold,
                d=d,
                normA2=normA2,
                maxrbar=maxrbar,
                minrbar=minrbar,
                normar=normar,
                normr=normr,
                normA=normA,
                condA=condA,
                normx=normx,
                istop=istop,
            )

        return step

    def make_cond_fun() -> Callable:
        def cond(state):
            state_flat, _ = tree.ravel_pytree(state)
            no_nans = np.logical_not(np.any(np.isnan(state_flat)))
            proceed = np.where(state.istop == 0, True, False)
            return np.logical_and(proceed, no_nans)

        return cond

    def stats(state: State) -> dict:
        return {
            "iteration_count": state.itn,
            "norm_residual": state.normr,
            "norm_At_residual": state.normar,
            "norm_A": state.normA,
            "cond_A": state.condA,
            "norm_x": state.normx,
            "istop": state.istop,
        }

    if custom_vjp:
        _run = _lstsq_custom_vjp(_run)
    return run


def _sym_ortho(a, b):
    """Stable implementation of Givens rotation. Like in Scipy."""
    idx = 3  # The "else" branch.
    idx = np.where(np.abs(b) > np.abs(a), 2, idx)
    idx = np.where(a == 0, 1, idx)
    idx = np.where(b == 0, 0, idx)

    branches = [_sym_ortho_0, _sym_ortho_1, _sym_ortho_2, _sym_ortho_3]
    return control_flow.switch(idx, branches, a, b)


def _sym_ortho_0(a, _b):
    zero = np.zeros((), dtype=a.dtype)
    return np.sign(a), zero, np.abs(a)


def _sym_ortho_1(_a, b):
    zero = np.zeros((), dtype=b.dtype)
    return zero, np.sign(b), np.abs(b)


def _sym_ortho_2(a, b):
    tau = a / b
    s = np.sign(b) / np.sqrt(1 + tau * tau)
    c = s * tau
    r = b / s
    return c, s, r


def _sym_ortho_3(a, b):
    tau = b / a
    c = np.sign(a) / np.sqrt(1 + tau * tau)
    s = c * tau
    r = a / c
    return c, s, r


def _lstsq_custom_vjp(lstsq_fun: Callable) -> Callable:
    def lstsq_fwd(vecmat, rhs, vecmat_args, x0, damp):
        x, stats = lstsq_fun(vecmat, rhs, vecmat_args, x0, damp)
        cache = {"x": x, "rhs": rhs, "vecmat_args": vecmat_args, "x0": x0, "damp": damp}
        return (x, stats), cache

    def lstsq_rev(vecmat, x0, damp, cache, dmu_dx):
        dmu_dx, _ = dmu_dx
        x_like = func.eval_shape(vecmat, cache["rhs"], *cache["vecmat_args"])
        if cache["rhs"].size <= x_like.size:
            return lstsq_rev_wide(vecmat, x0, damp, cache, dmu_dx)
        return lstsq_rev_tall(vecmat, x0, damp, cache, dmu_dx)

    def lstsq_rev_tall(vecmat, x0, damp, cache, dmu_dx):
        x = cache["x"]
        rhs = cache["rhs"]
        x0 = cache["x0"]
        damp = cache["damp"]
        vecmat_args = cache["vecmat_args"]

        def vecmat_noargs(z):
            return vecmat(z, *vecmat_args)

        def matvec_noargs(z):
            return func.vjp(vecmat_noargs, rhs)[1](z)[0]

        x0_rev = np.zeros_like(rhs)
        dmu_db = lstsq_fun(matvec_noargs, dmu_dx, (), x0_rev, damp)[0]
        p = lstsq_fun(vecmat_noargs, -dmu_db, (), x0, damp)[0]

        Ap = matvec_noargs(p)
        Ax_minus_b = matvec_noargs(x) - rhs

        @func.grad
        def grad_theta(theta):
            rA = vecmat(Ax_minus_b, *theta)
            pAA = vecmat(Ap, *theta)
            return linalg.inner(rA, p) + linalg.inner(pAA, x)

        dmu_dparams = grad_theta(vecmat_args)
        return dmu_db, dmu_dparams

    def lstsq_rev_wide(vecmat, x0, damp, cache, dmu_dx):
        x = cache["x"]
        rhs = cache["rhs"]
        vecmat_args = cache["vecmat_args"]

        def vecmat_noargs(z):
            return vecmat(z, *vecmat_args)

        def matvec_noargs(z):
            return func.linear_transpose(vecmat_noargs, rhs)(z)[0]

        # Compute the Lagrange multiplier from the forward pass
        x0_rev = np.zeros_like(rhs)
        y = lstsq_fun(matvec_noargs, x, (), x0_rev, damp)[0]

        # Compute the two solutions of the backward pass
        p = dmu_dx - lstsq_fun(vecmat_noargs, matvec_noargs(dmu_dx), (), x0, damp)[0]
        q = lstsq_fun(matvec_noargs, p - dmu_dx, (), x0_rev, damp)[0]

        @func.grad
        def grad_theta(theta):
            yA = vecmat(y, *theta)
            qA = vecmat(q, *theta)
            return linalg.inner(yA, p) + linalg.inner(qA, x)

        grad_vecmat_args = grad_theta(vecmat_args)
        grad_rhs = -q
        return grad_rhs, grad_vecmat_args

    lstsq_fun = func.custom_vjp(lstsq_fun, nondiff_argnums=(0, 3, 4))
    lstsq_fun.defvjp(lstsq_fwd, lstsq_rev)  # type: ignore
    return lstsq_fun
