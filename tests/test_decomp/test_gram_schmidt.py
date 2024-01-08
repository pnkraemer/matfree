from matfree.backend import control_flow, linalg, np, prng


def qr_via_gs_c(M):
    Q0 = np.zeros_like(M)
    R0 = np.zeros_like(M)
    for i in range(len(M)):
        v, n, c = gram_schmidt_orthogonalise_set(M[:, i], Q0.T)
        Q0 = Q0.at[:, i].set(v / linalg.vector_norm(v))
        R0 = R0.at[:, i].set(c)
        R0 = R0.at[i, i].set(n)
    return Q0, R0


def positive_diag(q, r):
    signs = np.sign(linalg.diagonal(r))
    return q * signs[None, :], signs[:, None] * r


def gram_schmidt_orthogonalise_set(vec, vectors):  # Gram-Schmidt
    (vec, _x), coeffs = control_flow.scan(
        gram_schmidt_orthogonalise, (vec, vec), xs=vectors
    )
    return vec / linalg.vector_norm(vec), linalg.vector_norm(vec), coeffs


def gram_schmidt_orthogonalise(x, vec2):
    vec1, x1 = x
    coeff = linalg.vecdot(x1, vec2)
    vec_ortho = vec1 - coeff * vec2
    return (vec_ortho, x1), coeff


def test_sth():
    n = 3
    key = prng.prng_key(2)
    key1, key2 = prng.split(key, 2)

    M = prng.normal(key2, shape=(n, n))
    Q, R = positive_diag(*linalg.qr_reduced(M))
    Q_, R_ = positive_diag(*qr_via_gs_c(M))
    print(Q, Q_)
    print(R, R_)
    assert np.allclose(R, R_)
    assert np.allclose(Q, Q_)
