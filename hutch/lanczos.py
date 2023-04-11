"""Lanczos-style functionality."""
from hutch.backend import flow, linalg, np, prng


def tridiagonal(matrix_vector_product, order, /, *, key, shape):
    """Decompose A = V T V^t purely based on matvec-products with A."""

    init_vec = prng.normal(key, shape=shape)
    (d,) = shape
    if order >= d:
        raise ValueError

    # Don't trust the result if orthogonality gets close to floating-point
    # arithmetic limits
    ds, es, Ws = [], [], []
    threshold = 100 * np.finfo(np.dtype(init_vec)).eps

    # j = 1:
    vj = init_vec / linalg.norm(init_vec)
    wj_dash = matrix_vector_product(vj)
    aj = np.dot(wj_dash, vj)
    wj = wj_dash - aj * vj
    vj_prev = vj
    Ws.append(vj)
    ds.append(aj)
    print(aj)
    # todo: use scan() (maybe padd Ws and alpha/beta in zeros).
    for _ in range(order):
        bj = linalg.norm(wj)

        def true_fn(_a, _b, B, k):
            _, k = prng.split(k)
            a = prng.normal(k, shape=shape)
            a = _reorthogonalise(a, B)
            return a, _b, B, k

        def false_fn(a, b, B, k):
            a = a / b
            return a, b, B, k

        reortho = bj < threshold
        vj, _, _, key = flow.cond(reortho, true_fn, false_fn, wj, bj, Ws, key)

        Ws.append(vj)

        wj_dash = matrix_vector_product(vj)
        aj = np.dot(wj_dash, vj)
        wj = wj_dash - aj * vj - bj * vj_prev
        vj_prev = vj
        ds.append(aj)
        print(aj)
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
