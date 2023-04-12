"""Lanczos-style functionality."""
from hutch.backend import flow, linalg, np, prng, transform


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


# all arguments positional-only because we will rename arguments a lot
@transform.partial(transform.jit, static_argnums=(0, 1))
def tridiagonal(matvec_fn, order, init_vec, /):
    """Decompose A = V T V^t purely based on matvec-products with A."""
    # this algorithm is massively unstable.
    # but despite this instability, quadrature might be stable?
    # https://www.sciencedirect.com/science/article/abs/pii/S0920563200918164

    (ncols,) = np.shape(init_vec)
    if order >= ncols or order < 1:
        raise ValueError

    diag = np.empty((order + 1,))
    offdiag = np.empty((order,))
    Q = np.empty((order + 1, ncols))

    # j = 1:
    vj = init_vec / linalg.norm(init_vec)
    wj_dash = matvec_fn(vj)
    aj = np.dot(wj_dash, vj)
    wj = wj_dash - aj * vj
    vj_prev = vj
    Q = Q.at[0, :].set(vj)
    diag = diag.at[0].set(aj)

    # todo: use scan() (maybe padd Q and alpha/beta in zeros).
    for idx_diag, idx_offdiag in zip(range(1, order + 1), range(order)):
        bj = linalg.norm(wj)
        vj = wj / bj
        vj = _reorthogonalise(vj, Q)
        Q = Q.at[idx_diag, :].set(vj)

        wj_dash = matvec_fn(vj)
        aj = np.dot(wj_dash, vj)
        wj = wj_dash - aj * vj - bj * vj_prev
        vj_prev = vj
        diag = diag.at[idx_diag].set(aj)
        offdiag = offdiag.at[idx_offdiag].set(bj)

    return Q, (diag, offdiag)


def _reorthogonalise(w, Q):  # Gram-Schmidt
    Q = np.stack(Q)

    def body_fn(carry, x):
        w = carry
        tau = x
        coeff = np.dot(w, tau)
        w = w - coeff * tau
        w = w / linalg.norm(w)
        return w, None

    w, _ = flow.scan(body_fn, init=w, xs=Q)
    return w
