"""Low-rank approximations (like partial Cholesky decompositions) of matrices."""

from matfree.backend import control_flow, func, linalg, np
from matfree.backend.typing import Array, Callable


def preconditioner(cholesky: Callable, /) -> Callable:
    r"""Turn a low-rank approximation into a preconditioner.

    Parameters
    ----------
    cholesky
        (Partial) Cholesky decomposition.
        Usually, the result of either
        [cholesky_partial][matfree.low_rank.cholesky_partial]
        or
        [cholesky_partial_pivot][matfree.low_rank.cholesky_partial_pivot].


    Returns
    -------
    solve
        A function that computes

        $$
        (v, s, *p) \mapsto (sI + L(*p) L(*p)^\top)^{-1} v,
        $$

        where $K = [k(i,j,*p)]_{ij} \approx L(*p) L(*p)^\top$
        and $L$ comes from the low-rank approximation.
    """

    def solve(v: Array, s: float, *cholesky_params):
        chol, info = cholesky(*cholesky_params)

        # Assert that the low-rank matrix is tall,
        # not wide (every sign has a story...)
        N, n = np.shape(chol)
        assert n <= N, (N, n)

        # Scale
        U = chol / np.sqrt(s)
        V = chol.T / np.sqrt(s)
        v /= s

        # Cholesky decompose the capacitance matrix
        # and solve the system
        eye_n = np.eye(n)
        chol_cap = linalg.cho_factor(eye_n + V @ U)
        sol = linalg.cho_solve(chol_cap, V @ v)
        return v - U @ sol, info

    return solve


def cholesky_partial(mat_el: Callable, /, *, nrows: int, rank: int) -> Callable:
    """Compute a partial Cholesky factorisation."""

    def cholesky(*params):
        if rank > nrows:
            msg = f"Rank exceeds n: {rank} >= {nrows}."
            raise ValueError(msg)
        if rank < 1:
            msg = f"Rank must be positive, but {rank} < {1}."
            raise ValueError(msg)

        step = _cholesky_partial_body(mat_el, nrows, *params)
        chol = np.zeros((nrows, rank))
        return control_flow.fori_loop(0, rank, step, chol), {}

    return cholesky


def _cholesky_partial_body(fn: Callable, n: int, *args):
    idx = np.arange(n)

    def matrix_element(i, j):
        return fn(i, j, *args)

    def matrix_column(i):
        fun = func.vmap(matrix_element, in_axes=(0, None))
        return fun(idx, i)

    def body(i, L):
        element = matrix_element(i, i)
        l_ii = np.sqrt(element - linalg.inner(L[i], L[i]))

        column = matrix_column(i)
        l_ji = column - L @ L[i, :]
        l_ji /= l_ii

        return L.at[:, i].set(l_ji)

    return body


def cholesky_partial_pivot(mat_el: Callable, /, *, nrows: int, rank: int) -> Callable:
    """Compute a partial Cholesky factorisation with pivoting."""

    def cholesky(*params):
        if rank > nrows:
            msg = f"Rank exceeds nrows: {rank} >= {nrows}."
            raise ValueError(msg)
        if rank < 1:
            msg = f"Rank must be positive, but {rank} < {1}."
            raise ValueError(msg)

        body = _cholesky_partial_pivot_body(mat_el, nrows, *params)

        L = np.zeros((nrows, rank))
        P = np.arange(nrows)

        init = (L, P, P, True)
        (L, P, _matrix, success) = control_flow.fori_loop(0, rank, body, init)
        return _pivot_invert(L, P), {"success": success}

    return cholesky


def _cholesky_partial_pivot_body(fn: Callable, n: int, *args):
    idx = np.arange(n)

    def matrix_element(i, j):
        return fn(i, j, *args)

    def matrix_element_p(i, j, *, permute):
        return matrix_element(permute[i], permute[j])

    def matrix_column_p(i, *, permute):
        fun = func.vmap(matrix_element, in_axes=(0, None))
        return fun(permute[idx], permute[i])

    def matrix_diagonal_p(*, permute):
        fun = func.vmap(matrix_element)
        return fun(permute[idx], permute[idx])

    def body(i, carry):
        L, P, P_matrix, success = carry

        # Access the matrix
        diagonal = matrix_diagonal_p(permute=P_matrix)

        # Find the largest entry for the residuals
        residual_diag = diagonal - func.vmap(linalg.inner)(L, L)
        res = np.abs(residual_diag)
        k = np.argmax(res)

        # Pivot [pivot!!! pivot!!! pivot!!! :)]
        P_matrix = _swap_cols(P_matrix, i, k)
        L = _swap_rows(L, i, k)
        P = _swap_rows(P, i, k)

        # Access the matrix
        element = matrix_element_p(i, i, permute=P_matrix)
        column = matrix_column_p(i, permute=P_matrix)

        # Perform a Cholesky step
        # (The first line could also be accessed via
        #  residual_diag[k], but it might
        #  be more readable to do it again)
        l_ii_squared = element - linalg.inner(L[i], L[i])
        l_ii = np.sqrt(l_ii_squared)
        l_ji = column - L @ L[i, :]
        l_ji /= l_ii
        success = np.logical_and(success, l_ii_squared > 0.0)

        # Update the estimate
        L = L.at[:, i].set(l_ji)
        return L, P, P_matrix, success

    return body


def _swap_cols(arr, i, j):
    return _swap_rows(arr.T, i, j).T


def _swap_rows(arr, i, j):
    ai, aj = arr[i], arr[j]
    arr = arr.at[i].set(aj)
    return arr.at[j].set(ai)


def _pivot_invert(arr, pivot, /):
    """Invert and apply a pivoting array to a matrix."""
    return arr[np.argsort(pivot)]
