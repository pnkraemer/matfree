"""Lanczos-style functionality."""
from matfree import montecarlo
from matfree.backend import containers, flow, linalg, np, prng, transform


# todo: rethink name of function.
# todo: move to matfree.py?
def trace_of_matfun(
    matfun,
    matvec_fun,
    order,
    /,
    *,
    key,
    num_samples_per_batch,
    num_batches,
    tangents_shape,
    tangents_dtype,
    sample_fun=prng.normal,
):
    """Compute the trace of the function of a matrix.

    For example, logdet(M) = trace(log(M)) ~ trace(U log(D) Ut) = E[v U log(D) Ut vt].
    """

    def sample(k, /):
        return sample_fun(k, shape=tangents_shape, dtype=tangents_dtype)

    quadform = quadratic_form_slq(matfun, matvec_fun, order)
    quadform_mc = montecarlo.montecarlo(quadform, sample_fun=sample)

    quadform_batch = montecarlo.mean_vmap(quadform_mc, num_samples_per_batch)
    quadform_batch = montecarlo.mean_map(quadform_batch, num_batches)
    return quadform_batch(key)


def quadratic_form_slq(matfun, matvec_fun, order, /):
    """Approximate quadratic form for stochastic Lanczos quadrature."""

    def quadform(init_vec, /):
        _, (diag, off_diag) = tridiagonal(matvec_fun, order, init_vec)

        # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
        #  use it here. Until then: an eigen-decomposition of size (order + 1)
        #  does not hurt too much...
        dense_matrix = np.diag(diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
        eigvals, eigvecs = linalg.eigh(dense_matrix)

        # Since Q orthogonal (orthonormal) to init_vec, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        (dim,) = init_vec.shape

        fx_eigvals = transform.vmap(matfun)(eigvals)
        return dim * np.dot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform


# all arguments are positional-only because we will rename arguments a lot
def tridiagonal(matvec_fun, order, init_vec, /):
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
    basis = np.zeros((order + 1, ncols))

    init_val = _lanczos_init(basis, (diag, offdiag), init_vec)
    body_fun = transform.partial(_lanczos_apply, matvec_fun=matvec_fun)
    output_val = flow.fori_loop(0, order + 1, body_fun=body_fun, init_val=init_val)
    return _lanczos_extract(output_val)


_LanczosState = containers.namedtuple("_LanczosState", ["basis", "tridiag", "q"])


def _lanczos_init(basis, tridiag, init_vec):
    return _LanczosState(basis, tridiag, init_vec)


def _lanczos_apply(i, state: _LanczosState, *, matvec_fun) -> _LanczosState:
    basis, (diag, offdiag), vec = state

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
    vec, length = _normalise(vec)
    vec, _ = _gram_schmidt_orthogonalise_set(vec, basis)

    # I don't know why, but this re-normalisation is soooo crucial
    vec, _ = _normalise(vec)
    basis = basis.at[i, :].set(vec)

    # When i==0, Q[i-1] is Q[-1] and again, we benefit from the fact
    #  that Q is initialised with zeros.
    vec = matvec_fun(vec)
    basis_vectors_previous = np.asarray([basis[i], basis[i - 1]])
    vec, (coeff, _) = _gram_schmidt_orthogonalise_set(vec, basis_vectors_previous)
    diag = diag.at[i].set(coeff)
    offdiag = offdiag.at[i - 1].set(length)

    return _LanczosState(basis, (diag, offdiag), vec)


def _lanczos_extract(state: _LanczosState, /):
    basis, (diag, offdiag), _ = state
    return basis, (diag, offdiag)


def _normalise(vec):
    length = linalg.norm(vec)
    return vec / length, length


def _gram_schmidt_orthogonalise_set(vec, vectors):  # Gram-Schmidt
    vec, coeffs = flow.scan(_gram_schmidt_orthogonalise, init=vec, xs=vectors)
    return vec, coeffs


def _gram_schmidt_orthogonalise(vec1, vec2):
    coeff = np.dot(vec1, vec2)
    vec_ortho = vec1 - coeff * vec2
    return vec_ortho, coeff
