"""Lanczos-style functionality."""
from hutch.backend import flow, linalg, np, prng, transform


def trace_fun(fn, matrix_vector_product, order, /, *, keys, shape):
    gamma = 0.0
    for key in keys:
        u = prng.rademacher(key, shape=shape)
        v = u / linalg.norm(u)
        diags, offdiags = tridiagonal_sym(matrix_vector_product, order, init_vec=v)
        print(diags, offdiags)
        vals, vecs = linalg.eigh_tridiagonal(diags, offdiags)
        tau = vecs[0, :]
        print(vals)
        gamma = gamma + np.dot(transform.vmap(fn)(np.abs(vals)), tau**2)
        print()
    (d,) = shape
    return gamma * d / len(keys)


def tridiagonal_sym(matrix_vector_product, order, /, init_vec):
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
    # todo: use scan() (maybe padd Ws and alpha/beta in zeros).
    for _ in range(2, order + 1):
        bj = linalg.norm(wj)

        def true_fn(a, _b, B):
            a = _reorthogonalise(a, B)
            return a, _b, B

        def false_fn(a, b, B):
            a = a / b
            return a, b, B

        reortho = bj < threshold
        if reortho:
            print("Reorthogonalising...")
        vj, bj, Ws = flow.cond(reortho, true_fn, false_fn, vj, bj, Ws)
        Ws.append(vj)

        wj_dash = matrix_vector_product(vj)
        aj = np.dot(wj_dash, vj)
        wj = wj_dash - aj * vj - bj * vj_prev
        vj_prev = vj
        ds.append(aj)
        es.append(bj)

    return np.asarray(ds), np.asarray(es)


def tridiagonal_sym_reorthogonalise(matrix_vector_product, order, /, *, key, shape):
    init_vec = prng.normal(key, shape=shape)

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
    # todo: use scan() (maybe padd Ws and alpha/beta in zeros).
    for _ in range(2, order + 1):
        bj = linalg.norm(wj)

        def true_fn(_a, _b, B, k):
            _, k = prng.split(k)
            a = prng.normal(k, shape=shape)
            a = _reorthogonalise(a, B)
            return a, _b, B, k

        def false_fn(a, b, B, k):
            a = a / b
            return a, b, B, k

        print(bj)
        reortho = bj < threshold
        print(reortho)
        vj, _, _, key = flow.cond(reortho, true_fn, false_fn, wj, bj, Ws, key)

        Ws.append(vj)

        wj_dash = matrix_vector_product(vj)
        aj = np.dot(wj_dash, vj)
        wj = wj_dash - aj * vj - bj * vj_prev
        vj_prev = vj
        ds.append(aj)
        es.append(bj)

    return (np.asarray(ds), np.asarray(es)), np.asarray(Ws)


def _reorthogonalise(w, Ws):  # Gram-Schmidt
    Ws = np.stack(Ws)
    w = w / linalg.norm(w)

    def body_fn(carry, x):
        w = carry
        tau = x
        coeff = np.dot(w, tau)
        w = w - coeff * tau
        w = w / linalg.norm(w)
        return w, None

    w, _ = flow.scan(body_fn, init=w, xs=Ws)
    return w
