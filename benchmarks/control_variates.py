"""Control_variates benchmark.

Runtime: ~10 seconds.
"""

from matfree import benchmark_util, hutchinson, montecarlo
from matfree.backend import func, linalg, np, plt, prng, progressbar


def problem(n):
    """Create an example problem."""

    # This function has a Jacobian with x-shaped sparsity pattern
    # We expect control variates to do pretty well
    # (But I don't know why)
    def f(x):
        return np.sin(np.roll(np.sin(np.flip(np.cos(x)) + 1) ** 2, 1)) * np.sin(x**2)

    key = prng.prng_key(seed=2)
    x0 = prng.uniform(key, shape=(n,))

    _, jvp = func.linearize(f, x0)
    J = func.jacfwd(f)(x0)
    trace = linalg.trace(J)
    sample_fun = montecarlo.normal(shape=(n,), dtype=float)

    return (jvp, trace, J), (key, sample_fun)


if __name__ == "__main__":
    dim = 100
    num_samples = 2 ** np.arange(4, 10)
    num_restarts = 5

    (Av, trace, J), (k, sample_fun) = problem(dim)

    error_fun = func.partial(benchmark_util.rmse_relative, expected=trace)

    @func.partial(benchmark_util.error_and_time, error_fun=error_fun)
    @func.partial(func.jit, static_argnums=0)
    def fun1(num, key):
        """Estimate the trace conventionally."""
        return hutchinson.trace(
            Av, key=key, sample_fun=sample_fun, num_batches=num, num_samples_per_batch=1
        )

    @func.partial(benchmark_util.error_and_time, error_fun=error_fun)
    @func.partial(func.jit, static_argnums=0)
    def fun2(num, key):
        """Estimate trace and diagonal jointly and discard the diagonal."""
        trace2, _ = hutchinson.trace_and_diagonal(
            Av, key=key, num_levels=num, sample_fun=sample_fun
        )
        return trace2

    errors1, stds1, times1 = [], [], []
    errors2, stds2, times2 = [], [], []

    for n in progressbar.progressbar(num_samples):
        test_keys = prng.split(k, num=num_restarts)
        e1, t1 = zip(*[fun1(int(n), ke) for ke in test_keys])
        e2, t2 = zip(*[fun2(int(n), ke) for ke in test_keys])

        errors1.append(np.mean(np.asarray(e1)))
        stds1.append(np.std(np.asarray(e1)))
        times1.append(min(t1))

        errors2.append(np.mean(np.asarray(e2)))
        stds2.append(np.std(np.asarray(e2)))
        times2.append(min(t2))

    errors1 = np.asarray(errors1)
    stds1 = np.asarray(stds1)
    times1 = np.asarray(times1)

    errors2 = np.asarray(errors2)
    stds2 = np.asarray(stds2)
    times2 = np.asarray(times2)

    fig, axes = plt.subplots(ncols=2, tight_layout=True, figsize=(10, 5), dpi=100)

    ax_wp, ax_sparsity = axes
    ax_wp.set_title("Trace estimation")
    ax_wp.loglog(
        times1,
        errors1,
        "o-",
        markeredgecolor="k",
        label="Conventional trace estimation",
    )

    ax_wp.fill_between(
        times1,
        errors1 - stds1,  # type: ignore
        errors1 + stds1,  # type: ignore
        alpha=0.25,
    )
    ax_wp.loglog(
        times2,
        errors2,
        "^-",
        markeredgecolor="k",
        label="Joint trace and diagonal estimation",
    )
    ax_wp.fill_between(
        times2,
        errors2 - stds2,  # type: ignore
        errors2 + stds2,  # type: ignore
        alpha=0.25,
    )
    ax_wp.set_ylabel(f"Relative error (mean & std, {num_restarts} restarts)")
    ax_wp.set_xlabel(f"Wall clock time (minimum, {num_restarts} restarts)")
    ax_wp.legend()

    ax_sparsity.set_title("Sparsity pattern of Jacobian")
    ax_sparsity.spy(J)
    ax_sparsity.set_xticks(())
    ax_sparsity.set_yticks(())
    plt.show()
