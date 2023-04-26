"""Lanczos-style functionality."""

from matfree.backend import containers, control_flow, func, linalg, np

Decomp = containers.namedtuple("Decomp", ["allocate", "init", "step", "extract"])


# all arguments are positional-only because we will rename arguments a lot
def tridiagonal(matvec_fun, depth, init_vec, /, method: Decomp):
    r"""Decompose A = V T V^t purely based on matvec-products with A.

    Orthogonally project the original matrix onto the (n+1)-deep Krylov subspace

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
    if depth >= ncols or depth < 1:
        raise ValueError

    empty_solution = method.allocate(depth, init_vec)
    init_val = method.init(empty_solution, init_vec)
    body_fun = func.partial(method.step, matvec_fun=matvec_fun)
    # todo: why from 0 to depth+1?
    result = control_flow.fori_loop(0, depth + 1, body_fun=body_fun, init_val=init_val)
    return method.extract(result)


def lanczos():
    """Lanczos tridiagonalisation."""
    return Decomp(
        allocate=_lanczos_allocate,
        init=_lanczos_init,
        step=_lanczos_apply,
        extract=_lanczos_extract,
    )


# todo: this below is a decomposition algorithm (init, step, extract),
#  and the function above is more of a decompose() method.

_LanczosState = containers.namedtuple("_LanczosState", ["basis", "tridiag", "q"])


def _lanczos_allocate(depth, init_vec, /):
    (ncols,) = init_vec.shape
    diag = np.zeros((depth + 1,))
    offdiag = np.zeros((depth,))
    basis = np.zeros((depth + 1, ncols))
    return basis, (diag, offdiag)


def _lanczos_init(empty_solution, init_vec):
    basis, tridiag = empty_solution
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
    vec, coeffs = control_flow.scan(_gram_schmidt_orthogonalise, init=vec, xs=vectors)
    return vec, coeffs


def _gram_schmidt_orthogonalise(vec1, vec2):
    coeff = np.dot(vec1, vec2)
    vec_ortho = vec1 - coeff * vec2
    return vec_ortho, coeff
