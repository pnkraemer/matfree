"""Lanczos-style functionality."""
from hutch.backend import flow, linalg, np, prng, transform


# todo: rethink name of function.
# todo: move to hutch.py?
def trace_of_matfn(
    matfn,
    matrix_vector_product,
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
        _, (d, e) = tridiagonal(matrix_vector_product, order, v0)

        # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
        #  use it here. Until then: an eigen-decomposition of size (order + 1)
        #  does not hurt too much...
        T = np.diag(d) + np.diag(e, -1) + np.diag(e, 1)
        s, Q = linalg.eigh(T)

        (d,) = tangents_shape
        return np.dot(Q[0, :] ** 2, transform.vmap(matfn)(s)) * d

    # todo: return number (and indices) of NaNs filtered out?
    # todo: make lower-memory by combining map and vmap.
    # todo: assert that
    #  -- while tridiagonal() is not reliable for large matrices --
    #  trace_of_matfn is.
    # todo: can we use the full power of hutch.py here?
    #  (e.g. control variates, batching, etc.)
    traces = key_to_trace(keys)
    is_not_nan_index = np.logical_not(np.isnan(traces))
    return np.mean(traces[is_not_nan_index])


# all arguments positional-only because we will rename arguments a lot
@transform.partial(transform.jit, static_argnums=(0, 1))
def tridiagonal(matrix_vector_product, order, init_vec, /):
    """Decompose A = V T V^t purely based on matvec-products with A."""
    # this algorithm is massively unstable.
    # but despite this instability, quadrature might be stable?
    # https://www.sciencedirect.com/science/article/abs/pii/S0920563200918164

    (d,) = np.shape(init_vec)
    if order >= d or order < 1:
        raise ValueError

    ds, es, Ws = [], [], []

    # j = 1:
    vj = init_vec / linalg.norm(init_vec)
    wj_dash = matrix_vector_product(vj)
    aj = np.dot(wj_dash, vj)
    wj = wj_dash - aj * vj
    vj_prev = vj
    Ws.append(vj)
    ds.append(aj)

    # todo: use scan() (maybe padd Ws and alpha/beta in zeros).
    for _ in range(order):
        bj = linalg.norm(wj)
        vj = wj / bj
        vj = _reorthogonalise(vj, Ws)
        Ws.append(vj)

        wj_dash = matrix_vector_product(vj)
        aj = np.dot(wj_dash, vj)
        wj = wj_dash - aj * vj - bj * vj_prev
        vj_prev = vj
        ds.append(aj)
        es.append(bj)

    return np.asarray(Ws), (np.asarray(ds), np.asarray(es))


def _reorthogonalise(w, Ws):  # Gram-Schmidt
    Ws = np.stack(Ws)

    def body_fn(carry, x):
        w = carry
        tau = x
        coeff = np.dot(w, tau)
        w = w - coeff * tau
        w = w / linalg.norm(w)
        return w, None

    w, _ = flow.scan(body_fn, init=w, xs=Ws)
    return w
