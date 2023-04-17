"""Lanczos-style functionality."""
from hutch import montecarlo
from hutch.backend import containers, flow, linalg, np, prng, transform


# todo: rethink name of function.
# todo: move to hutch.py?
def trace_of_matfn(
    matfn,
    matvec_fn,
    order,
    /,
    *,
    key,
    num_samples_per_batch,
    num_batches,
    tangents_shape,
    tangents_dtype,
    generate_samples_fn=prng.normal,
):
    """Compute the trace of the function of a matrix.

    For example, logdet(M) = trace(log(M))  ~ trace(U log(D) Ut) = E[v U log(D) Ut vt].
    """

    def sample_fn(k):
        return generate_samples_fn(k, shape=tangents_shape, dtype=tangents_dtype)

    Q = quadratic_form(matfn, matvec_fn, order)
    Q_mc = montecarlo.montecarlo(Q, sample_fn=sample_fn)

    Q_batch = montecarlo.mean_vmap(Q_mc, num_samples_per_batch)
    Q_batch = montecarlo.mean_map(Q_batch, num_batches)
    return Q_batch(key)


def quadratic_form(matfn, matvec_fn, order, /):
    """Approximate quadratic form."""

    def Q(v0):
        _, (d, e) = tridiagonal(matvec_fn, order, v0)

        # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
        #  use it here. Until then: an eigen-decomposition of size (order + 1)
        #  does not hurt too much...
        T = np.diag(d) + np.diag(e, -1) + np.diag(e, 1)
        s, Q = linalg.eigh(T)

        # Since Q orthogonal (orthonormal) to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        (d,) = v0.shape
        return d * np.dot(Q[0, :], transform.vmap(matfn)(s) * Q[0, :])

    return Q


# all arguments are positional-only because we will rename arguments a lot
def tridiagonal(matvec_fn, order, init_vec, /):
    r"""Decompose A = V T V^t purely based on matvec-products with A.

    Orthogonally project the original matrix onto the (n+1)-th order Krylov subspace

    \{ v, Av, A^2v, ..., A^n v \}

    using the Gram-Schmidt procedure.
    The result is a tri-diagonal matrix whose spectrum
    approximates the spectrum of the original matrix.
    (More specifically, the spectrum tends to the 'most extreme' eigenvalues.)
    """
    # this algorithm is massively unstable.
    # but despite this instability, quadrature might be stable?
    # https://www.sciencedirect.com/science/article/abs/pii/S0920563200918164

    (ncols,) = np.shape(init_vec)
    if order >= ncols or order < 1:
        raise ValueError

    diag = np.zeros((order + 1,))
    offdiag = np.zeros((order,))
    Q = np.zeros((order + 1, ncols))

    init_val = _lanczos_init(Q, (diag, offdiag), init_vec)
    body_fun = transform.partial(_lanczos_apply, matvec_fn=matvec_fn)
    output_val = flow.fori_loop(0, order + 1, body_fun=body_fun, init_val=init_val)
    return _lanczos_extract(output_val)


_LanczosResult = containers.namedtuple("_LanczosState", ["Q", "diag_and_offdiag"])
_LanczosState = containers.namedtuple("_LanczosState", ["result", "q"])


def _lanczos_init(Q, diag_and_offdiag, v):
    result = _LanczosResult(Q, diag_and_offdiag)
    return _LanczosState(result, v)


def _lanczos_apply(i, val: _LanczosState, *, matvec_fn) -> _LanczosState:
    (Q, (diag, offdiag)), q = val

    # This one is a hack:
    #
    # Re-orthogonalise agains ALL basis elements (see note) before storing
    # This has a big impact on numerical stability
    # But is a little bit of a hack.
    # Note: we reorthogonalise against ALL columns of Q, not just
    # the ones we have already computed. This increases the complexity
    # of the whole iteration from n(n+1)/2 to n^2, but has the advantage
    # that the whole computation has static bounds (thus we can JIT it all).
    # Since 'Q' is padded with zeros, the numerical values are identical
    # between both modes of computing.
    #
    # Todo: only reorthogonalise if |q| = 0?
    q, bj = _normalise(q)
    q, _ = _gram_schmidt_orthogonalise_set(q, Q)

    # I don't know why, but this re-normalisation is soooo crucial
    q, _ = _normalise(q)
    Q = Q.at[i, :].set(q)

    # When i==0, Q[i-1] is Q[-1] and again, we benefit from the fact
    #  that Q is initialised with zeros.
    q = matvec_fn(q)
    q, (aj, _) = _gram_schmidt_orthogonalise_set(q, [Q[i], Q[i - 1]])
    diag = diag.at[i].set(aj)
    offdiag = offdiag.at[i - 1].set(bj)

    result = _LanczosResult(Q, (diag, offdiag))
    return _LanczosState(result, q)


def _lanczos_extract(val):
    (Q, (diag, offdiag)), _ = val
    return Q, (diag, offdiag)


def _normalise(v):
    b = linalg.norm(v)
    return v / b, b


def _gram_schmidt_orthogonalise_set(w, Q):  # Gram-Schmidt
    w, coeffs = flow.scan(_gram_schmidt_orthogonalise, init=w, xs=np.asarray(Q))
    return w, coeffs


def _gram_schmidt_orthogonalise(v, w):
    c = np.dot(v, w)
    v_new = v - c * w
    return v_new, c
