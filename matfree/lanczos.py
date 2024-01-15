"""All things Lanczos' algorithm.

This includes
stochastic Lanczos quadrature (extending the integrands
in [hutchinson][matfree.hutchinson] to those that implement
stochastic Lanczos quadrature),
Lanczos-implementations of matrix-function-vector products,
and various Lanczos-decompositions of matrices.

Examples
--------
>>> import jax.random
>>> import jax.numpy as jnp
>>>
>>> jnp.set_printoptions(1)
>>>
>>> M = jax.random.normal(jax.random.PRNGKey(1), shape=(10, 10))
>>> A = M.T @ M
>>> v = jax.random.normal(jax.random.PRNGKey(2), shape=(10,))
>>>
>>> # Compute a matrix-logarithm with Lanczos' algorithm
>>> matfun_vec = funm_vector_product_spd(jnp.log, 4, lambda s: A @ s)
>>> matfun_vec(v)
Array([-4. , -2.1, -2.7, -1.9, -1.3, -3.5, -0.5, -0.1,  0.3,  1.5],      dtype=float32)
"""

from matfree.backend import containers, control_flow, func, linalg, np, tree_util
from matfree.backend.typing import Array, Callable, Tuple


def integrand_spd_logdet(order, matvec, /):
    """Construct the integrand for the log-determinant.

    This function assumes a symmetric, positive definite matrix.
    """
    return integrand_spd(np.log, order, matvec)


def integrand_spd(matfun, order, matvec, /):
    """Quadratic form for stochastic Lanczos quadrature.

    This function assumes a symmetric, positive definite matrix.
    """

    def quadform(v0, *parameters):
        v0_flat, v_unflatten = tree_util.ravel_pytree(v0)
        length = linalg.vector_norm(v0_flat)
        v0_flat /= length

        def matvec_flat(v_flat, *p):
            v = v_unflatten(v_flat)
            Av = matvec(v, *p)
            flat, unflatten = tree_util.ravel_pytree(Av)
            return flat

        algorithm = alg_tridiag_full_reortho(matvec_flat, order)
        _, (diag, off_diag) = algorithm(v0_flat, *parameters)
        eigvals, eigvecs = _eigh_tridiag(diag, off_diag)

        # Since Q orthogonal (orthonormal) to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        fx_eigvals = func.vmap(matfun)(eigvals)
        return length**2 * linalg.vecdot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform


def integrand_product_logdet(depth, matvec, vecmat, /):
    r"""Construct the integrand for the log-determinant of a matrix-product.

    Here, "product" refers to $X = A^\top A$.
    """
    return integrand_product(np.log, depth, matvec, vecmat)


def integrand_product_schatten_norm(power, depth, matvec, vecmat, /):
    r"""Construct the integrand for the p-th power of the Schatten-p norm."""

    def matfun(x):
        """Matrix-function for Schatten-p norms."""
        return x ** (power / 2)

    return integrand_product(matfun, depth, matvec, vecmat)


def integrand_product(matfun, depth, matvec, vecmat, /):
    r"""Construct the integrand for the trace of a function of a matrix-product.

    Instead of the trace of a function of a matrix,
    compute the trace of a function of the product of matrices.
    Here, "product" refers to $X = A^\top A$.
    """

    def quadform(v0, *parameters):
        v0_flat, v_unflatten = tree_util.ravel_pytree(v0)
        length = linalg.vector_norm(v0_flat)
        v0_flat /= length

        def matvec_flat(v_flat, *p):
            v = v_unflatten(v_flat)
            Av = matvec(v, *p)
            flat, unflatten = tree_util.ravel_pytree(Av)
            return flat, tree_util.partial_pytree(unflatten)

        w0_flat, w_unflatten = func.eval_shape(matvec_flat, v0_flat)
        matrix_shape = (*np.shape(w0_flat), *np.shape(v0_flat))

        def vecmat_flat(w_flat):
            w = w_unflatten(w_flat)
            wA = vecmat(w, *parameters)
            return tree_util.ravel_pytree(wA)[0]

        # Decompose into orthogonal-bidiag-orthogonal
        algorithm = alg_bidiag_full_reortho(
            lambda v: matvec_flat(v)[0], vecmat_flat, depth, matrix_shape=matrix_shape
        )
        output = algorithm(v0_flat, *parameters)
        u, (d, e), vt, _ = output

        # Compute SVD of factorisation
        B = _bidiagonal_dense(d, e)
        _, S, Vt = linalg.svd(B, full_matrices=False)

        # Since Q orthogonal (orthonormal) to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        eigvals, eigvecs = S**2, Vt.T
        fx_eigvals = func.vmap(matfun)(eigvals)
        return length**2 * linalg.vecdot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform


def _bidiagonal_dense(d, e):
    diag = linalg.diagonal_matrix(d)
    offdiag = linalg.diagonal_matrix(e, 1)
    return diag + offdiag


def funm_vector_product_spd(matfun, order, matvec, /):
    """Implement a matrix-function-vector product via Lanczos' algorithm.

    This algorithm uses Lanczos' tridiagonalisation with full re-orthogonalisation
    and therefore applies only to symmetric, positive definite matrices.
    """
    algorithm = alg_tridiag_full_reortho(matvec, order)

    def estimate(vec, *parameters):
        length = linalg.vector_norm(vec)
        vec /= length
        basis, (diag, off_diag) = algorithm(vec, *parameters)
        eigvals, eigvecs = _eigh_tridiag(diag, off_diag)

        fx_eigvals = func.vmap(matfun)(eigvals)
        return length * (basis.T @ (eigvecs @ (fx_eigvals * eigvecs[0, :])))

    return estimate


def _eigh_tridiag(diag, off_diag):
    # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
    #  use it here. Until then: an eigen-decomposition of size (order + 1)
    #  does not hurt too much...
    diag = linalg.diagonal_matrix(diag)
    offdiag1 = linalg.diagonal_matrix(off_diag, -1)
    offdiag2 = linalg.diagonal_matrix(off_diag, 1)
    dense_matrix = diag + offdiag1 + offdiag2
    eigvals, eigvecs = linalg.eigh(dense_matrix)
    return eigvals, eigvecs


def svd_approx(
    v0: Array, depth: int, Av: Callable, vA: Callable, matrix_shape: Tuple[int, ...]
):
    """Approximate singular value decomposition.

    Uses GKL with full reorthogonalisation to bi-diagonalise the target matrix
    and computes the full SVD of the (small) bidiagonal matrix.

    Parameters
    ----------
    v0:
        Initial vector for Golub-Kahan-Lanczos bidiagonalisation.
    depth:
        Depth of the Krylov space constructed by Golub-Kahan-Lanczos bidiagonalisation.
        Choosing `depth = min(nrows, ncols) - 1` would yield behaviour similar to
        e.g. `np.linalg.svd`.
    Av:
        Matrix-vector product function.
    vA:
        Vector-matrix product function.
    matrix_shape:
        Shape of the matrix involved in matrix-vector and vector-matrix products.
    """
    # Factorise the matrix
    algorithm = alg_bidiag_full_reortho(Av, vA, depth, matrix_shape=matrix_shape)
    u, (d, e), vt, _ = algorithm(v0)

    # Compute SVD of factorisation
    B = _bidiagonal_dense(d, e)
    U, S, Vt = linalg.svd(B, full_matrices=False)

    # Combine orthogonal transformations
    return u @ U, S, Vt @ vt


class _LanczosAlg(containers.NamedTuple):
    """Lanczos decomposition algorithm."""

    init: Callable
    """Initialise the state of the algorithm. Usually, this involves pre-allocation."""

    step: Callable
    """Compute the next iteration."""

    extract: Callable
    """Extract the solution from the state of the algorithm."""

    lower_upper: Tuple[int, int]
    """Range of the for-loop used to decompose a matrix."""


def alg_tridiag_full_reortho(
    Av: Callable, depth, /, validate_unit_2_norm=False
) -> Callable:
    """Construct an implementation of **tridiagonalisation**.

    Uses pre-allocation. Fully reorthogonalise vectors at every step.

    This algorithm assumes a **symmetric matrix**.

    Decompose a matrix into a product of orthogonal-**tridiagonal**-orthogonal matrices.
    Use this algorithm for approximate **eigenvalue** decompositions.

    """

    class State(containers.NamedTuple):
        i: int
        basis: Array
        tridiag: Tuple[Array, Array]
        q: Array
        length: Array

    def init(init_vec: Array) -> State:
        (ncols,) = np.shape(init_vec)
        if depth >= ncols or depth < 1:
            raise ValueError

        if validate_unit_2_norm:
            init_vec = _validate_unit_2_norm(init_vec)

        diag = np.zeros((depth + 1,))
        offdiag = np.zeros((depth,))
        basis = np.zeros((depth + 1, ncols))

        return State(0, basis, (diag, offdiag), init_vec, 1.0)

    def apply(state: State, *parameters) -> State:
        i, basis, (diag, offdiag), vec, length = state

        # Compute the next off-diagonal entry
        offdiag = offdiag.at[i - 1].set(length)

        # Re-orthogonalise against ALL basis elements before storing.
        # Note: we re-orthogonalise against ALL columns of Q, not just
        # the ones we have already computed. This increases the complexity
        # of the whole iteration from n(n+1)/2 to n^2, but has the advantage
        # that the whole computation has static bounds (thus we can JIT it all).
        # Since 'Q' is padded with zeros, the numerical values are identical
        # between both modes of computing.
        vec, *_ = _gram_schmidt_classical(vec, basis)

        # Store new basis element
        basis = basis.at[i, :].set(vec)

        # When i==0, Q[i-1] is Q[-1] and again, we benefit from the fact
        #  that Q is initialised with zeros.
        vec = Av(vec, *parameters)
        basis_vectors_previous = np.asarray([basis[i], basis[i - 1]])
        vec, length, (coeff, _) = _gram_schmidt_classical(vec, basis_vectors_previous)
        diag = diag.at[i].set(coeff)

        return State(i + 1, basis, (diag, offdiag), vec, length)

    def extract(state: State, /):
        # todo: return final output "_ignored"
        _, basis, (diag, offdiag), *_ignored = state
        return basis, (diag, offdiag)

    alg = _LanczosAlg(
        init=init, step=apply, extract=extract, lower_upper=(0, depth + 1)
    )
    return func.partial(_decompose_fori_loop, algorithm=alg)


def alg_bidiag_full_reortho(
    Av: Callable, vA: Callable, depth, /, matrix_shape, validate_unit_2_norm=False
):
    """Construct an implementation of **bidiagonalisation**.

    Uses pre-allocation. Fully reorthogonalise vectors at every step.

    Works for **arbitrary matrices**. No symmetry required.

    Decompose a matrix into a product of orthogonal-**bidiagonal**-orthogonal matrices.
    Use this algorithm for approximate **singular value** decompositions.
    """
    nrows, ncols = matrix_shape
    max_depth = min(nrows, ncols) - 1
    if depth > max_depth or depth < 0:
        msg1 = f"Depth {depth} exceeds the matrix' dimensions. "
        msg2 = f"Expected: 0 <= depth <= min(nrows, ncols) - 1 = {max_depth} "
        msg3 = f"for a matrix with shape {matrix_shape}."
        raise ValueError(msg1 + msg2 + msg3)

    class State(containers.NamedTuple):
        i: int
        Us: Array
        Vs: Array
        alphas: Array
        betas: Array
        beta: Array
        vk: Array

    def init(init_vec: Array) -> State:
        if validate_unit_2_norm:
            init_vec = _validate_unit_2_norm(init_vec)

        alphas = np.zeros((depth + 1,))
        betas = np.zeros((depth + 1,))
        Us = np.zeros((depth + 1, nrows))
        Vs = np.zeros((depth + 1, ncols))
        v0, _ = _normalise(init_vec)
        return State(0, Us, Vs, alphas, betas, np.zeros(()), v0)

    def apply(state: State, *parameters) -> State:
        i, Us, Vs, alphas, betas, beta, vk = state
        Vs = Vs.at[i].set(vk)
        betas = betas.at[i].set(beta)

        uk = Av(vk, *parameters) - beta * Us[i - 1]
        uk, alpha = _normalise(uk)
        uk, *_ = _gram_schmidt_classical(uk, Us)  # full reorthogonalisation
        Us = Us.at[i].set(uk)
        alphas = alphas.at[i].set(alpha)

        vk = vA(uk, *parameters) - alpha * vk
        vk, beta = _normalise(vk)
        vk, *_ = _gram_schmidt_classical(vk, Vs)  # full reorthogonalisation

        return State(i + 1, Us, Vs, alphas, betas, beta, vk)

    def extract(state: State, /):
        _, uk_all, vk_all, alphas, betas, beta, vk = state
        return uk_all.T, (alphas, betas[1:]), vk_all, (beta, vk)

    alg = _LanczosAlg(
        init=init, step=apply, extract=extract, lower_upper=(0, depth + 1)
    )
    return func.partial(_decompose_fori_loop, algorithm=alg)


def _validate_unit_2_norm(v, /):
    # Lanczos assumes a unit-2-norm vector as an input
    # We cannot raise an error based on values of the init_vec,
    # but we can make it obvious that the result is unusable.
    is_not_normalized = np.abs(linalg.vector_norm(v) - 1.0) > 10 * np.finfo_eps(v.dtype)
    return control_flow.cond(
        is_not_normalized,
        lambda s: np.nan() * np.ones_like(s),
        lambda s: s,
        v,
    )


def _gram_schmidt_classical(vec, vectors):  # Gram-Schmidt
    vec, coeffs = control_flow.scan(_gram_schmidt_classical_step, vec, xs=vectors)
    vec, length = _normalise(vec)
    return vec, length, coeffs


def _gram_schmidt_classical_step(vec1, vec2):
    coeff = linalg.vecdot(vec1, vec2)
    vec_ortho = vec1 - coeff * vec2
    return vec_ortho, coeff


def _normalise(vec):
    length = linalg.vector_norm(vec)
    return vec / length, length


def _decompose_fori_loop(v0, *parameters, algorithm: _LanczosAlg):
    r"""Decompose a matrix purely based on matvec-products with A.

    The behaviour of this function is equivalent to

    ```python
    def decompose(v0, *matvec_funs, algorithm):
        init, step, extract, (lower, upper) = algorithm
        state = init(v0)
        for _ in range(lower, upper):
            state = step(state, *matvec_funs)
        return extract(state)
    ```

    but the implementation uses JAX' fori_loop.
    """
    init, step, extract, (lower, upper) = algorithm
    init_val = init(v0)

    def body_fun(_, s):
        return step(s, *parameters)

    result = control_flow.fori_loop(lower, upper, body_fun=body_fun, init_val=init_val)
    return extract(result)
