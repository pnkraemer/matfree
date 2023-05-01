"""Benchmark utilities."""

from matfree.backend import func, linalg, np, time


def rmse_relative(received, *, expected):
    """Compute the relative root-mean-square error."""
    return linalg.vector_norm((received - expected) / expected) / np.sqrt(expected.size)


def error_and_time(fun, error_fun):
    """Compute error and runtime of a function with a single outputs."""

    @func.wraps(fun)
    def fun_wrapped(*args, **kwargs):
        # Execute once for compilation
        _ = fun(*args, **kwargs)

        # Execute and time
        t0 = time.perf_counter()
        result = fun(*args, **kwargs)
        result.block_until_ready()
        t1 = time.perf_counter()
        return error_fun(result), (t1 - t0)

    return fun_wrapped
