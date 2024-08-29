"""Matrix-free implementations of functions of matrices.

This includes matrix-function-vector products

$$
(f, A, v, p) \\mapsto f(A(p))v
$$

as well as matrix-function extensions for stochastic trace estimation,
which provide

$$
(f, A, v, p) \\mapsto v^\\top f(A(p))v.
$$

Plug these integrands into
[matfree.stochtrace.estimator][matfree.stochtrace.estimator].


Examples
--------
>>> import jax.random
>>> import jax.numpy as jnp
>>> from matfree import decomp
>>>
>>> M = jax.random.normal(jax.random.PRNGKey(1), shape=(10, 10))
>>> A = M.T @ M
>>> v = jax.random.normal(jax.random.PRNGKey(2), shape=(10,))
>>>
>>> # Compute a matrix-logarithm with Lanczos' algorithm
>>> matfun = dense_funm_sym_eigh(jnp.log)
>>> tridiag = decomp.tridiag_sym(4)
>>> matfun_vec = funm_lanczos_sym(matfun, tridiag)
>>> fAx = matfun_vec(lambda s: A @ s, v)
>>> print(fAx.shape)
(10,)

"""

from matfree.backend import containers, control_flow, func, linalg, np, tree_util
from matfree.backend.typing import Array, Callable


def funm_chebyshev(matfun: Callable, order: int, matvec: Callable, /) -> Callable:
    """Compute a matrix-function-vector product via Chebyshev's algorithm.

    This function assumes that the **spectrum of the matrix-vector product
    is contained in the interval (-1, 1)**, and that the **matrix-function
    is analytic on this interval**. If this is not the case,
    transform the matrix-vector product and the matrix-function accordingly.
    """
    # Construct nodes
    nodes = _chebyshev_nodes(order)
    fx_nodes = matfun(nodes)

    class _ChebyshevState(containers.NamedTuple):
        interpolation: Array
        poly_coefficients: tuple[Array, Array]
        poly_values: tuple[Array, Array]

    def init_func(vec, *parameters):
        # Initialize the scalar recursion
        # (needed to compute the interpolation weights)
        t2_n, t1_n = nodes, np.ones_like(nodes)
        c1 = np.mean(fx_nodes * t1_n)
        c2 = 2 * np.mean(fx_nodes * t2_n)

        # Initialize the vector-valued recursion
        # (this is where the matvec happens)
        t2_x, t1_x = matvec(vec, *parameters), vec
        value = c1 * t1_x + c2 * t2_x
        return _ChebyshevState(value, (t2_n, t1_n), (t2_x, t1_x))

    def recursion_func(val: _ChebyshevState, *parameters) -> _ChebyshevState:
        value, (t2_n, t1_n), (t2_x, t1_x) = val

        # Apply the next scalar recursion and
        # compute the next coefficient
        t2_n, t1_n = 2 * nodes * t2_n - t1_n, t2_n
        c2 = 2 * np.mean(fx_nodes * t2_n)

        # Apply the next matrix-vector product recursion and
        # compute the next interpolation-value
        t2_x, t1_x = 2 * matvec(t2_x, *parameters) - t1_x, t2_x
        value += c2 * t2_x
        return _ChebyshevState(value, (t2_n, t1_n), (t2_x, t1_x))

    def extract_func(val: _ChebyshevState):
        return val.interpolation

    alg = (0, order - 1), init_func, recursion_func, extract_func
    return _funm_polyexpand(alg)


def _chebyshev_nodes(n, /):
    k = np.arange(n, step=1.0) + 1
    return np.cos((2 * k - 1) / (2 * n) * np.pi())


def _funm_polyexpand(matrix_poly_alg, /):
    """Compute a matrix-function-vector product via a polynomial expansion."""
    (lower, upper), init_func, step_func, extract_func = matrix_poly_alg

    def matrix_function_vector_product(vec, *parameters):
        final_state = control_flow.fori_loop(
            lower=lower,
            upper=upper,
            body_fun=lambda _i, v: step_func(v, *parameters),
            init_val=init_func(vec, *parameters),
        )
        return extract_func(final_state)

    return matrix_function_vector_product


def funm_lanczos_sym(dense_funm: Callable, tridiag_sym: Callable, /) -> Callable:
    """Implement a matrix-function-vector product via Lanczos' tridiagonalisation.

    This algorithm uses Lanczos' tridiagonalisation
    and therefore applies only to symmetric matrices.

    Parameters
    ----------
    dense_funm
        An implementation of a function of a dense matrix.
        For example, the output of
        [funm.dense_funm_sym_eigh][matfree.funm.dense_funm_sym_eigh]
        [funm.dense_funm_schur][matfree.funm.dense_funm_schur]
    tridiag_sym
        An implementation of tridiagonalisation.
        E.g., the output of
        [decomp.tridiag_sym][matfree.decomp.tridiag_sym].
    """

    def estimate(matvec: Callable, vec, *parameters):
        length = linalg.vector_norm(vec)
        vec /= length
        Q, matrix, *_ = tridiag_sym(matvec, vec, *parameters)

        funm = dense_funm(matrix)
        e1 = np.eye(len(matrix))[0, :]
        return length * (Q @ funm @ e1)

    return estimate


def funm_arnoldi(dense_funm: Callable, hessenberg: Callable, /) -> Callable:
    """Implement a matrix-function-vector product via the Arnoldi iteration.

    This algorithm uses the Arnoldi iteration
    and therefore applies only to all square matrices.

    Parameters
    ----------
    dense_funm
        An implementation of a function of a dense matrix.
        For example, the output of
        [funm.dense_funm_sym_eigh][matfree.funm.dense_funm_sym_eigh]
        [funm.dense_funm_schur][matfree.funm.dense_funm_schur]
    hessenberg
        An implementation of Hessenberg-factorisation.
        E.g., the output of
        [decomp.hessenberg][matfree.decomp.hessenberg].
    """

    def estimate(matvec: Callable, vec, *parameters):
        length = linalg.vector_norm(vec)
        vec /= length
        basis, matrix, *_ = hessenberg(matvec, vec, *parameters)

        funm = dense_funm(matrix)
        e1 = np.eye(len(matrix))[0, :]
        return length * (basis @ funm @ e1)

    return estimate


def integrand_funm_sym_logdet(tridiag_sym: Callable, /):
    """Construct the integrand for the log-determinant.

    This function assumes a symmetric, positive definite matrix.

    Parameters
    ----------
    tridiag_sym
        An implementation of tridiagonalisation.
        E.g., the output of
        [decomp.tridiag_sym][matfree.decomp.tridiag_sym].

    """
    dense_funm = dense_funm_sym_eigh(np.log)
    return integrand_funm_sym(dense_funm, tridiag_sym)


def integrand_funm_sym(dense_funm, tridiag_sym, /):
    """Construct the integrand for matrix-function-trace estimation.

    This function assumes a symmetric matrix.

    Parameters
    ----------
    dense_funm
        An implementation of a function of a dense matrix.
        For example, the output of
        [funm.dense_funm_sym_eigh][matfree.funm.dense_funm_sym_eigh]
        [funm.dense_funm_schur][matfree.funm.dense_funm_schur]
    tridiag_sym
        An implementation of tridiagonalisation.
        E.g., the output of
        [decomp.tridiag_sym][matfree.decomp.tridiag_sym].

    """

    def quadform(matvec, v0, *parameters):
        v0_flat, v_unflatten = tree_util.ravel_pytree(v0)
        length = linalg.vector_norm(v0_flat)
        v0_flat /= length

        def matvec_flat(v_flat, *p):
            v = v_unflatten(v_flat)
            Av = matvec(v, *p)
            flat, unflatten = tree_util.ravel_pytree(Av)
            return flat

        _, dense, *_ = tridiag_sym(matvec_flat, v0_flat, *parameters)

        fA = dense_funm(dense)
        e1 = np.eye(len(fA))[0, :]
        return length**2 * linalg.inner(e1, fA @ e1)

    return quadform


def integrand_funm_product_logdet(bidiag: Callable, /):
    r"""Construct the integrand for the log-determinant of a matrix-product.

    Here, "product" refers to $X = A^\top A$.
    """
    return integrand_funm_product(np.log, bidiag)


# todo: expect bidiag() to be passed here
def integrand_funm_product_schatten_norm(power, bidiag: Callable, /):
    r"""Construct the integrand for the $p$-th power of the Schatten-p norm."""

    def matfun(x):
        """Matrix-function for Schatten-p norms."""
        return x ** (power / 2)

    return integrand_funm_product(matfun, bidiag)


# todo: expect bidiag() to be passed here
# todo: expect dense_funm_svd() to be passed here
def integrand_funm_product(matfun, algorithm, /):
    r"""Construct the integrand for matrix-function-trace estimation.

    Instead of the trace of a function of a matrix,
    compute the trace of a function of the product of matrices.
    Here, "product" refers to $X = A^\top A$.
    """

    dense_funm = dense_funm_product_svd(matfun)

    def quadform(matvecs, v0, *parameters):
        matvec, vecmat = matvecs
        v0_flat, v_unflatten = tree_util.ravel_pytree(v0)
        length = linalg.vector_norm(v0_flat)
        v0_flat /= length

        def matvec_flat(v_flat, *p):
            v = v_unflatten(v_flat)
            Av = matvec(v, *p)
            flat, unflatten = tree_util.ravel_pytree(Av)
            return flat, tree_util.partial_pytree(unflatten)

        w0_flat, w_unflatten = func.eval_shape(matvec_flat, v0_flat)

        def vecmat_flat(w_flat):
            w = w_unflatten(w_flat)
            wA = vecmat(w, *parameters)
            return tree_util.ravel_pytree(wA)[0]

        # Decompose into orthogonal-bidiag-orthogonal
        matvec_flat_p = lambda v: matvec_flat(v)[0]  # noqa: E731
        output = algorithm(matvec_flat_p, vecmat_flat, v0_flat, *parameters)
        _, B, *_ = output

        fA = dense_funm(B)
        e1 = np.eye(len(fA))[0, :]
        return length**2 * linalg.inner(e1, fA @ e1)

    return quadform


def dense_funm_product_svd(matfun):
    def dense_funm(matrix, /):
        # Compute SVD of factorisation
        _, S, Vt = linalg.svd(matrix, full_matrices=False)

        # Since Q orthogonal (orthonormal) to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        eigvals, eigvecs = S**2, Vt.T
        fx_eigvals = func.vmap(matfun)(eigvals)
        return eigvecs @ (fx_eigvals[:, None] * eigvecs.T)

    return dense_funm


# todo: rename to *_eigh_sym
def dense_funm_sym_eigh(matfun):
    """Implement dense matrix-functions via symmetric eigendecompositions.

    Use it to construct one of the matrix-free matrix-function implementations,
    e.g. [matfree.funm.funm_lanczos_sym][matfree.funm.funm_lanczos_sym].
    """

    def fun(dense_matrix):
        eigvals, eigvecs = linalg.eigh(dense_matrix)
        fx_eigvals = func.vmap(matfun)(eigvals)
        return eigvecs @ linalg.diagonal(fx_eigvals) @ eigvecs.T

    return fun


def dense_funm_schur(matfun):
    """Implement dense matrix-functions via symmetric Schur decompositions.

    Use it to construct one of the matrix-free matrix-function implementations,
    e.g. [matfree.funm.funm_lanczos_sym][matfree.funm.funm_lanczos_sym].
    """

    def fun(dense_matrix):
        return linalg.funm_schur(dense_matrix, matfun)

    return fun


def dense_funm_pade_exp():
    """Implement dense matrix-exponentials using a Pade approximation.

    Use it to construct one of the matrix-free matrix-function implementations,
    e.g. [matfree.funm.funm_arnoldi][matfree.funm.funm_arnoldi].
    """

    def fun(dense_matrix):
        return linalg.funm_pade_exp(dense_matrix)

    return fun
