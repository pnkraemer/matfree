"""What is the fastest way of computing trace(A^5)."""
from matfree import benchmark_util, hutchinson, montecarlo, slq
from matfree.backend import func, linalg, np, plt, prng
from matfree.backend.progressbar import progressbar


def problem(n):
    """Create an example problem."""

    # This function has a Jacobian with x-shaped sparsity pattern
    # We expect control variates to do pretty well
    # (But I don't know why)
    def f(x):
        return np.sin(np.roll(np.sin(np.flip(np.cos(x)) + 1) ** 2, 2)) * np.sin(x**2)

    key = prng.prng_key(seed=2)
    x0 = prng.uniform(key, shape=(n,))

    _, jvp = func.linearize(f, x0)
    J = func.jacfwd(f)(x0)
    A = J @ J @ J @ J
    trace = linalg.trace(A)
    sample_fun = montecarlo.normal(shape=(n,), dtype=float)

    def Av(v):
        return jvp(jvp(jvp(jvp(v))))

    return (lambda x: x**4, jvp, Av, trace, A), (key, sample_fun)


def evaluate_all(fun, outer_loop, inner_loop):
    """Evaluate all metrics of a function."""
    errors, stds, times = [], [], []
    for n in outer_loop:
        fun_bind = func.partial(fun, int(n))
        err, tim = evaluate(fun_bind, inner_loop)

        errors.append(np.mean(err))
        stds.append(np.std(err))
        times.append(np.array_min(tim))

    return np.asarray(errors), np.asarray(stds), np.asarray(times)


def evaluate(fun, keys):
    """Evaluate all metrics of a function."""
    errors, times = zip(*[fun(key) for key in keys])
    return np.asarray(errors), np.asarray(times)


if __name__ == "__main__":
    dim = 10
    num_samples = 2 ** np.arange(4, 12)
    num_restarts = 10

    (matfun, jvp, Av, trace, JJ), (k, sample_fun) = problem(dim)

    x = hutchinson.trace(Av, key=k, sample_fun=sample_fun)
    y = slq.trace_of_matfun_symmetric(matfun, jvp, 5, key=k, sample_fun=sample_fun)
    assert np.allclose(x, trace, atol=1e-1, rtol=1e-1), (x, trace)
    assert np.allclose(y, trace, atol=1e-1, rtol=1e-1), (y, trace)

    error_fun = func.partial(benchmark_util.rmse_relative, expected=trace)

    @func.partial(benchmark_util.error_and_time, error_fun=error_fun)
    @func.partial(func.jit, static_argnums=0)
    def matvec(num, key):
        """Matrix-vector mult."""
        return hutchinson.trace(
            Av, key=key, sample_fun=sample_fun, num_samples_per_batch=num
        )

    @func.partial(benchmark_util.error_and_time, error_fun=error_fun)
    @func.partial(func.jit, static_argnums=0)
    def slq_low(num, key):
        """SLQ(1)"""  # noqa: D400,D415
        return slq.trace_of_matfun_symmetric(
            matfun,
            jvp,
            1,
            key=key,
            sample_fun=sample_fun,
            num_samples_per_batch=num,
        )

    @func.partial(benchmark_util.error_and_time, error_fun=error_fun)
    @func.partial(func.jit, static_argnums=0)
    def slq_high(num, key):
        """SLQ(5)"""  # noqa: D400,D415
        return slq.trace_of_matfun_symmetric(
            matfun,
            jvp,
            5,
            key=key,
            sample_fun=sample_fun,
            num_samples_per_batch=num,
        )

    test_keys = prng.split(k, num=num_restarts)
    fig, axes = plt.subplots(ncols=2, tight_layout=True, figsize=(10, 5), dpi=100)
    ax_wp, ax_sparsity = axes

    ax_wp.set_title("Estimating trace(A^2)")
    for fun in [matvec, slq_low, slq_high]:
        errors, stds, times = evaluate_all(fun, progressbar(num_samples), test_keys)
        ax_wp.loglog(
            times,
            errors,
            "o-",
            markeredgecolor="k",
            label=fun.__doc__,
        )
        ax_wp.fill_between(times, errors - stds, errors + stds, alpha=0.25)

    ax_wp.set_ylabel(f"Relative error (mean & std, {num_restarts} restarts)")
    ax_wp.set_xlabel(f"Wall clock time (minimum, {num_restarts} restarts)")
    ax_wp.legend()

    ax_sparsity.set_title("Sparsity pattern of A^2")
    ax_sparsity.spy(JJ)
    ax_sparsity.set_xticks(())
    ax_sparsity.set_yticks(())
    plt.show()
