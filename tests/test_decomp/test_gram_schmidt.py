from matfree.backend import control_flow, linalg, np, prng, testing


def qr_via_gs_c(M):
    Q0 = np.zeros_like(M)
    R0 = np.zeros_like(M)
    for i in range(len(M)):
        v, n, c = gram_schmidt_classical_set(i, M, Q0)
        Q0 = Q0.at[:, i].set(v)
        R0 = R0.at[:, i].set(c)
        R0 = R0.at[i, i].set(n)
    return Q0, R0


def gram_schmidt_classical_set(i, matrix, vectors):  # Gram-Schmidt
    vec, coeffs = control_flow.scan(gram_schmidt_classical, matrix[:, i], xs=vectors.T)
    return vec / linalg.vector_norm(vec), linalg.vector_norm(vec), coeffs


def gram_schmidt_classical(vec1, vec2):
    coeff = linalg.vecdot(vec1, vec2)
    vec_ortho = vec1 - coeff * vec2
    return vec_ortho, coeff


def qr_via_gs_m(M):
    Q0 = np.zeros_like(M)
    R0 = np.zeros_like(M)
    for i in range(len(M)):
        M, (v, c) = gram_schmidt_modified_set(i, M)
        Q0 = Q0.at[:, i].set(v)
        R0 = R0.at[i, :].set(c)
    return Q0, R0


def gram_schmidt_modified_set(i, matrix):  # Gram-Schmidt
    n = linalg.vector_norm(matrix[:, i])
    q = matrix[:, i] / n

    coeffs = q.T @ matrix
    matrix = matrix - (coeffs[:, None] * q[None, :]).T
    matrix = matrix.at[:, i].set(0.0)
    return matrix, (q, coeffs)


def positive_diag(q, r):
    signs = np.sign(linalg.diagonal(r))
    return q * signs[None, :], signs[:, None] * r


@testing.parametrize("func", [qr_via_gs_c, qr_via_gs_m])
def test_manual_qr_matches_numpy_qr(func):
    n = 3
    key = prng.prng_key(2)
    key1, key2 = prng.split(key, 2)

    M = prng.normal(key2, shape=(n, n))
    Q, R = positive_diag(*linalg.qr_reduced(M))
    Q_, R_ = positive_diag(*func(M))
    assert np.allclose(R, R_)
    assert np.allclose(Q, Q_)
