"""Lanczos-style functionality."""
from hutch.backend import containers, flow, linalg, np, prng, transform


# todo: rethink name of function.
# todo: move to hutch.py?
def trace_of_matfn(
    matfn,
    matvec_fn,
    order,
    /,
    *,
    keys,
    tangents_shape,
    tangents_dtype,
    generate_samples_fn=prng.normal,
):
    @transform.vmap
    def key_to_trace(k):
        v0 = generate_samples_fn(k, shape=tangents_shape, dtype=tangents_dtype)
        _, (d, e) = tridiagonal(matvec_fn, order, v0)

        # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
        #  use it here. Until then: an eigen-decomposition of size (order + 1)
        #  does not hurt too much...
        T = np.diag(d) + np.diag(e, -1) + np.diag(e, 1)
        s, Q = linalg.eigh(T)

        (d,) = tangents_shape
        return np.dot(Q[0, :] ** 2, transform.vmap(matfn)(s)) * d

    # todo: return number (and indices) of NaNs filtered out?
    # todo: make lower-memory by combining map and vmap.
    # todo: can we use the full power of hutch.py here?
    #  (e.g. control variates, batching, etc.)
    traces = key_to_trace(keys)
    is_nan_index = np.isnan(traces)
    is_nan_where = np.where(is_nan_index)

    is_not_nan_index = np.logical_not(is_nan_index)
    return np.mean(traces[is_not_nan_index]), *is_nan_where


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

    # j = 1:
    q, _ = _normalise(init_vec)
    Q = Q.at[0, :].set(q)

    q = matvec_fn(q)
    q, aj = _gram_schmidt_orthogonalise(q, Q[0])
    diag = diag.at[0].set(aj)

    init_val = _lanczos_init(Q, (diag, offdiag), q)
    body_fun = transform.partial(_lanczos_iterate, matvec_fn=matvec_fn)
    output_val = flow.fori_loop(
        lower=1, upper=order + 1, body_fun=body_fun, init_val=init_val
    )
    return _lanczos_extract(output_val)


_LanczosResult = containers.namedtuple("_LanczosState", ["Q", "diag", "offdiag"])
_LanczosState = containers.namedtuple("_LanczosState", ["result", "q"])


def _lanczos_init(Q, diag_and_offdiag, q):
    result = _LanczosResult(Q, *diag_and_offdiag)
    return _LanczosState(result, q)


def _lanczos_iterate(i, val: _LanczosState, *, matvec_fn) -> _LanczosState:
    (Q, diag, offdiag), q = val

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
    # if bj == 0:
    q, _ = _gram_schmidt_orthogonalise_set(q, Q)
    q, _ = _normalise(q)  # I don't know why, but this line is soooo crucial

    Q = Q.at[i, :].set(q)
    q = matvec_fn(q)
    q, (aj, _) = _gram_schmidt_orthogonalise_set(q, [Q[i], Q[i - 1]])
    diag = diag.at[i].set(aj)
    offdiag = offdiag.at[i - 1].set(bj)

    result = _LanczosResult(Q, diag, offdiag)
    return _LanczosState(result, q)


def _lanczos_extract(val):
    (Q, *diag_and_offdiag), _ = val
    return Q, diag_and_offdiag


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
