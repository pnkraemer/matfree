"""Low-rank approximations (like partial Cholesky decompositions) of matrices."""

from matfree.backend import control_flow, func, linalg, np, prng
from matfree.backend.typing import Array, Callable


def preconditioner(cholesky: Callable, /) -> Callable:
    r"""Turn a low-rank approximation into a preconditioner.

    Parameters
    ----------
    cholesky
        (Partial) Cholesky decomposition.
        Usually, the result of
        [cholesky_partial][matfree.low_rank.cholesky_partial],
        [cholesky_partial_pivot][matfree.low_rank.cholesky_partial_pivot],
        or
        [cholesky_partial_pivot_random][matfree.low_rank.cholesky_partial_pivot_random].


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
    """Compute a partial Cholesky factorisation.

    The function `mat_el(i, j, *params)` must return elements of a
    **positive definite** matrix. No pivoting is performed, so the algorithm
    may fail silently on rank-deficient matrices; use
    [cholesky_partial_pivot][matfree.low_rank.cholesky_partial_pivot] instead.
    """

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
    """Compute a partial Cholesky factorisation with pivoting.

    The function `mat_el(i, j, *params)` must return elements of a
    **positive semi-definite** matrix. Pivoting improves numerical stability
    for low-rank or near-degenerate matrices; the `"success"` flag in the
    returned info dict indicates whether all diagonal factors were positive.
    """

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


def cholesky_partial_pivot_random(
    mat_el: Callable,
    /,
    *,
    nrows: int,
    rank: int,
    block_size: int = 1,
    atol: float = 0.0,
    rtol: float = 0.0,
    regularization: float | None = None,
) -> Callable:
    r"""Compute a partial Cholesky factorisation with random pivoting (RPCholesky).

    Pivots are sampled with probabilities proportional to the diagonal of the
    residual $K - L L^*$, so only the diagonal and `rank` rows of $K$ are
    evaluated (Chen, Epperly, Tropp, Webber, "Randomly pivoted Cholesky:
    Practical approximation of a kernel matrix with few entry evaluations",
    Comm. Pure Appl. Math., 2025). With `block_size > 1`, each round samples
    `block_size` distinct pivots and factors the corresponding block
    (block RPCholesky); `rank` must be a multiple of `block_size`.

    The factorisation stops after `rank` columns, or earlier once the trace of
    the residual drops below `atol` or below `rtol` times the trace of $K$.
    Columns beyond the computed rank are zero, so the result can be passed to
    [preconditioner][matfree.low_rank.preconditioner].

    Each pivot block $C$ is factored as $C + r \, \mathrm{tr}(C) I$, which keeps
    the factorisation stable when the block is (numerically) singular; by
    default, $r$ is the machine precision for `block_size > 1` and zero for
    `block_size = 1` (a single pivot is positive by construction).

    The function `mat_el(i, j, *params)` must return elements of a
    **positive semi-definite** (Hermitian) matrix.

    Returns
    -------
    cholesky
        A function `cholesky(key, *params)` that returns `L` of shape
        `(nrows, rank)` with $K \approx L L^*$, and a dict with
        `"success"` (whether every pivot block was positive definite),
        `"rank"` (the number of computed columns),
        `"pivots"` (the pivot indices; `-1` beyond `"rank"`),
        `"trace_residual"` and `"trace"`.
    """
    if rank > nrows:
        msg = f"Rank exceeds nrows: {rank} >= {nrows}."
        raise ValueError(msg)
    if rank < 1:
        msg = f"Rank must be positive, but {rank} < {1}."
        raise ValueError(msg)
    if rank % block_size != 0:
        msg = f"Rank must be a multiple of block_size, but {rank} % {block_size} != 0."
        raise ValueError(msg)

    def cholesky(key, *params):
        idx = np.arange(nrows)

        def element(i, j):
            return mat_el(i, j, *params)

        rows = func.vmap(func.vmap(element, in_axes=(None, 0)), in_axes=(0, None))
        diagonal = func.vmap(element)(idx, idx).real
        trace = np.sum(diagonal)
        dtype = np.dtype(element(0, 0))
        if regularization is None:
            shift = np.finfo_eps(diagonal.dtype) if block_size > 1 else 0.0
        else:
            shift = regularization

        def cond(carry):
            counter, _R, _pivots, residual, success, _key = carry
            trace_residual = np.sum(residual)
            tolerance = np.elementwise_max(atol, rtol * trace)
            return np.logical_and(
                np.logical_and(success, trace_residual > tolerance), counter < rank
            )

        def body(carry):
            counter, R, pivots, residual, success, key = carry

            # Sample distinct pivots proportionally to the residual diagonal
            key, subkey = prng.split(key)
            probs = residual / np.sum(residual)
            S = prng.choice(subkey, nrows, shape=(block_size,), p=probs, replace=False)

            # Residual rows at the pivots and the (regularised) pivot block.
            # R stores the rows of L^*, so K ~ R^* R.
            G = rows(S, idx) - R[:, S].T.conj() @ R
            C = G[:, S]
            C = C + shift * linalg.trace(C).real * np.eye(block_size, dtype=dtype)
            chol_C = linalg.cholesky(C)
            ok = np.logical_not(np.any(np.isnan(chol_C)))

            # New rows of L^*, and update of the residual diagonal
            G_solved = linalg.solve_triangular(chol_C, G, lower=True)
            update = counter + np.arange(block_size)
            R = np.where(ok, R.at[update].set(G_solved), R)
            residual_new = residual - np.sum(linalg.abs2(G_solved), axis=0)
            residual_new = np.elementwise_max(residual_new, 0.0)
            residual = np.where(ok, residual_new, residual)
            pivots = np.where(ok, pivots.at[update].set(S), pivots)
            counter = np.where(ok, counter + block_size, counter)
            return counter, R, pivots, residual, np.logical_and(success, ok), key

        R = np.zeros((rank, nrows), dtype=dtype)
        pivots = -np.ones((rank,), dtype=idx.dtype)
        init = (np.asarray(0), R, pivots, diagonal, np.asarray(True), key)
        counter, R, pivots, residual, success, _ = control_flow.while_loop(
            cond, body, init
        )
        info = {
            "success": success,
            "rank": counter,
            "pivots": pivots,
            "trace_residual": np.sum(residual),
            "trace": trace,
        }
        return R.T.conj(), info

    return cholesky


def _swap_cols(arr, i, j):
    return _swap_rows(arr.T, i, j).T


def _swap_rows(arr, i, j):
    ai, aj = arr[i], arr[j]
    arr = arr.at[i].set(aj)
    return arr.at[j].set(ai)


def _pivot_invert(arr, pivot, /):
    """Invert and apply a pivoting array to a matrix."""
    return arr[np.argsort(pivot)]
