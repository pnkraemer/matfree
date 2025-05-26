"""Matrix-free algorithms for least-squares problems."""

import dataclasses
from typing import Callable

import jax
import jax.flatten_util
import jax.numpy as jnp

# todo: make functions compatible with pytree-valued vecmats


LARGE_VALUE = 1e10


def lsmr(
    *,
    atol: float = 1e-6,
    btol: float = 1e-6,
    ctol: float = 1e-8,
    maxiter: int = 100_000,
    damp=0.0,
    while_loop=jax.lax.while_loop,
):
    """Construct an experimental implementation of LSMR.

    Follows the implementation in SciPy, but uses JAX.

    Link to Scipy's version:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html
    """
    # todo: move 'damp' to run() and derive custom VJPs?
    # todo: stop iteration when NaN or Inf are detected.

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass
    class State:
        """LSMR state."""

        # Iteration count:
        itn: int
        # Bidiagonalisation variables:
        alpha: float
        u: jax.Array
        v: jax.Array
        # LSMR-specific variables:
        alphabar: float
        rhobar: float
        rho: float
        zeta: float
        sbar: float
        cbar: float
        zetabar: float
        hbar: jax.Array
        h: jax.Array
        x: jax.Array
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
    def run(vecmat, b, vecmat_args=()):
        def vecmat_noargs(v):
            return vecmat(v, *vecmat_args)

        (ncols,) = jax.eval_shape(vecmat, b, *vecmat_args).shape

        state, normb, matvec_noargs = init(vecmat_noargs, b, ncols=ncols)
        step_fun = make_step(matvec_noargs, normb=normb)
        cond_fun = make_cond_fun()
        state = while_loop(cond_fun, step_fun, state)
        stats_ = stats(state)
        return state.x, stats_

    def init(vecmat, b, ncols: int):
        normb = jnp.linalg.norm(b)
        x = jnp.zeros(ncols, b.dtype)
        beta = normb

        u = b
        u = u / jnp.where(beta > 0, beta, 1.0)

        v, matvec = jax.vjp(vecmat, u)
        alpha = jnp.linalg.norm(v)
        v = v / jnp.where(alpha > 0, alpha, 1)
        v = jnp.where(beta == 0, jnp.zeros_like(v), v)
        alpha = jnp.where(beta == 0, jnp.zeros_like(alpha), alpha)

        # Initialize variables for 1st iteration.

        zetabar = alpha * beta
        alphabar = alpha
        rho = 1.0
        rhobar = 1.0
        cbar = 1.0
        sbar = 0.0

        h = v
        hbar = jnp.zeros(ncols, b.dtype)

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
        normA = jnp.sqrt(normA2)
        condA = 1.0
        normx = 0.0

        # Items for use in stopping rules, normb set earlier
        normr = beta

        # Reverse the order here from the original matlab code because
        # there was an error on return when arnorm==0
        normar = alpha * beta

        # Main iteration loop.
        state = State(
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
        state = jax.tree.map(lambda z: jnp.asarray(z), state)
        return state, normb, lambda *a: matvec(*a)[0]

    def make_step(matvec, normb: float) -> Callable:
        def step(state: State) -> State:
            # Perform the next step of the bidiagonalization

            Av, A_t = jax.vjp(matvec, state.v)
            u = Av - state.alpha * state.u
            beta = jnp.linalg.norm(u)

            u = u / jnp.where(beta > 0, beta, 1.0)
            v = A_t(u)[0] - beta * state.v
            alpha = jnp.linalg.norm(v)
            v = v / jnp.where(alpha > 0, alpha, 1)

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
            normr = jnp.sqrt(d + (betad - taud) ** 2 + betadd * betadd)

            # Estimate ||A||.
            normA2 = state.normA2 + beta * beta
            normA = jnp.sqrt(normA2)
            normA2 = normA2 + alpha * alpha

            # Estimate cond(A).
            maxrbar = jnp.maximum(state.maxrbar, rhobarold)
            minrbar = jnp.where(
                state.itn > 1, jnp.minimum(state.minrbar, rhobarold), state.minrbar
            )
            condA = jnp.maximum(maxrbar, rhotemp) / jnp.minimum(minrbar, rhotemp)

            # Compute norms for convergence testing.
            normar = jnp.abs(zetabar)
            normx = jnp.linalg.norm(x)

            # Check whether we should stop
            itn = state.itn + 1
            test1 = normr / normb
            z = normA * normr
            z_safe = jnp.where(z != 0, z, 1.0)
            test2 = jnp.where(z != 0, normar / z_safe, LARGE_VALUE)
            test3 = 1 / condA
            t1 = test1 / (1 + normA * normx / normb)
            rtol = btol + atol * normA * normx / normb

            # Early exits
            istop = 0
            istop = jnp.where(normar == 0, 9, istop)
            istop = jnp.where(normb == 0, 8, istop)
            istop = jnp.where(itn >= maxiter, 7, istop)
            istop = jnp.where(1 + test3 <= 1, 6, istop)
            istop = jnp.where(1 + test2 <= 1, 5, istop)
            istop = jnp.where(1 + t1 <= 1, 4, istop)
            istop = jnp.where(test3 <= ctol, 3, istop)
            istop = jnp.where(test2 <= atol, 2, istop)
            istop = jnp.where(test1 <= rtol, 1, istop)

            return State(
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
            state_flat, _ = jax.flatten_util.ravel_pytree(state)
            no_nans = jnp.logical_not(jnp.any(jnp.isnan(state_flat)))
            proceed = jnp.where(state.istop == 0, True, False)
            return jnp.logical_and(proceed, no_nans)

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

    return run


def _sym_ortho(a, b):
    """Stable implementation of Givens rotation. Like in Scipy."""
    idx = 3  # The "else" branch.
    idx = jnp.where(jnp.abs(b) > jnp.abs(a), 2, idx)
    idx = jnp.where(a == 0, 1, idx)
    idx = jnp.where(b == 0, 0, idx)

    branches = [_sym_ortho_0, _sym_ortho_1, _sym_ortho_2, _sym_ortho_3]
    return jax.lax.switch(idx, branches, a, b)


def _sym_ortho_0(a, _b):
    zero = jnp.zeros((), dtype=a.dtype)
    return jnp.sign(a), zero, jnp.abs(a)


def _sym_ortho_1(_a, b):
    zero = jnp.zeros((), dtype=b.dtype)
    return zero, jnp.sign(b), jnp.abs(b)


def _sym_ortho_2(a, b):
    tau = a / b
    s = jnp.sign(b) / jnp.sqrt(1 + tau * tau)
    c = s * tau
    r = b / s
    return c, s, r


def _sym_ortho_3(a, b):
    tau = b / a
    c = jnp.sign(a) / jnp.sqrt(1 + tau * tau)
    s = c * tau
    r = a / c
    return c, s, r
