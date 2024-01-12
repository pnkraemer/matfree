"""Implement approximations of matrix-function-vector products.

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
>>> matfun_vec = lanczos_spd(jnp.log, 4, lambda s: A @ s)
>>> matfun_vec(v)
Array([-4. , -2.1, -2.7, -1.9, -1.3, -3.5, -0.5, -0.1,  0.3,  1.5],      dtype=float32)
"""

from matfree import decomp
from matfree.backend import containers, control_flow, func, linalg, np
from matfree.backend.typing import Array


def lanczos_spd(matfun, order, matvec, /):
    """Implement a matrix-function-vector product via Lanczos' algorithm.

    This algorithm uses Lanczos' tridiagonalisation with full re-orthogonalisation.
    """
    # Lanczos' algorithm
    algorithm = decomp.lanczos_tridiag_full_reortho(order)

    def estimate(vec, *parameters):
        def matvec_p(v):
            return matvec(v, *parameters)

        length = linalg.vector_norm(vec)
        vec /= length
        basis, tridiag = decomp.decompose_fori_loop(vec, matvec_p, algorithm=algorithm)
        (diag, off_diag) = tridiag

        # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
        #  use it here. Until then: an eigen-decomposition of size (order + 1)
        #  does not hurt too much...
        diag = linalg.diagonal_matrix(diag)
        offdiag1 = linalg.diagonal_matrix(off_diag, -1)
        offdiag2 = linalg.diagonal_matrix(off_diag, 1)
        dense_matrix = diag + offdiag1 + offdiag2
        eigvals, eigvecs = linalg.eigh(dense_matrix)

        fx_eigvals = func.vmap(matfun)(eigvals)
        return length * (basis.T @ (eigvecs @ (fx_eigvals * eigvecs[0, :])))

    return estimate


def matrix_poly_vector_product(matrix_poly_alg, /):
    """Implement a matrix-function-vector product via a polynomial expansion.

    Parameters
    ----------
    matrix_poly_alg
        Which algorithm to use.
        For example, the output of
        [matrix_poly_chebyshev][matfree.matfun.matrix_poly_chebyshev].
    """
    lower, upper, init_func, step_func, extract_func = matrix_poly_alg

    def matvec(vec, *parameters):
        final_state = control_flow.fori_loop(
            lower=lower,
            upper=upper,
            body_fun=lambda _i, v: step_func(v, *parameters),
            init_val=init_func(vec, *parameters),
        )
        return extract_func(final_state)

    return matvec


def _chebyshev_nodes(n, /):
    k = np.arange(n, step=1.0) + 1
    return np.cos((2 * k - 1) / (2 * n) * np.pi())


def matrix_poly_chebyshev(matfun, order, matvec, /):
    """Construct an implementation of matrix-Chebyshev-polynomial interpolation."""
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

    return 0, order - 1, init_func, recursion_func, extract_func
