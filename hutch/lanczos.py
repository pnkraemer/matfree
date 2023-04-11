"""Lanczos-style functionality."""
from hutch.backend import linalg, np


def tridiagonal_sym(matrix_vector_product, order, /, init_vec, threshold=1e-5):
    ds, es, Ws = [], [], []

    v, v_old = init_vec / linalg.norm(init_vec), 0.0

    beta = 0
    success_so_far = True

    for _ in range(order):
        w = matrix_vector_product(v)
        w = w - beta * v_old

        alpha = np.dot(w, v)
        w = w - alpha * v
        w = _reorthogonalise(w, Ws)

        beta = linalg.norm(w)
        beta_small = beta < threshold
        success_so_far = success_so_far and not beta_small

        ds.append(alpha)
        es.append(beta)
        Ws.append(v)
        v, v_old = w / beta, v
    return success_so_far, (np.asarray(ds), np.asarray(es[:-1])), np.stack(Ws, axis=1)


def _reorthogonalise(w, Ws):
    for tau in Ws:
        coeff = np.dot(w, tau)
        w = w - coeff * tau
    return w
