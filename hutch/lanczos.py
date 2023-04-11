"""Lanczos-style functionality."""
from hutch.backend import linalg, np, prng


def tridiagonal_sym(matrix_vector_product, order, /, *, key, shape):
    ds, es, Ws = [], [], []
    init_vec = prng.normal(key, shape=shape)

    # Don't trust the result if orthogonality gets close to floating-point
    # arithmetic limits
    threshold = 100 * np.finfo(np.dtype(init_vec)).eps

    # j = 1:
    vj = init_vec / linalg.norm(init_vec)
    wj_dash = matrix_vector_product(vj)
    aj = np.dot(wj_dash, vj)
    wj = wj_dash - aj * vj

    vj_prev = vj
    Ws.append(vj)
    ds.append(aj)
    for _ in range(2, order + 1):
        bj = linalg.norm(wj)

        print(linalg.norm(bj))
        if bj < threshold:
            _, key = prng.split(key)
            vj = prng.normal(key, shape=shape)
            vj = _reorthogonalise(vj, Ws)
        else:
            vj = wj / bj
        Ws.append(vj)

        wj_dash = matrix_vector_product(vj)
        aj = np.dot(wj_dash, vj)
        wj = wj_dash - aj * vj - bj * vj_prev
        vj_prev = vj
        ds.append(aj)
        es.append(bj)

    return True, (np.asarray(ds), np.asarray(es)), np.asarray(Ws)


def _reorthogonalise(w, Ws):
    for tau in Ws:
        coeff = np.dot(w, tau)
        w = w - coeff * tau
        w = w / linalg.norm(w)
    return w
