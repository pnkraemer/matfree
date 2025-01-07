from matfree import decomp, test_util
from matfree.backend import config, func, linalg, np, prng, testing


@testing.parametrize("nrows", [3])
@testing.parametrize("num_matvecs", [2])
@testing.parametrize("reortho", ["none", "full"])
@testing.parametrize("dtype", [float])
def test_adjoint_matches_jax_dot_vjp(nrows, num_matvecs, reortho, dtype):
    # Create a matrix and a direction as a test-case
    A = prng.normal(prng.prng_key(1), shape=(nrows, nrows), dtype=dtype)
    v = prng.normal(prng.prng_key(2), shape=(nrows,), dtype=dtype)

    # Set up the algorithms
    algorithm_autodiff = decomp.hessenberg(
        num_matvecs, reortho=reortho, custom_vjp=False
    )
    algorithm_adjoint = decomp.hessenberg(num_matvecs, reortho=reortho, custom_vjp=True)

    # Forward pass
    algorithm_autodiff = func.partial(algorithm_autodiff, lambda s, p: p @ s)
    algorithm_adjoint = func.partial(algorithm_adjoint, lambda s, p: p @ s)
    fwd_output, vjp_autodiff = func.vjp(algorithm_autodiff, v, A)
    _fwd_output, vjp_adjoint = func.vjp(algorithm_adjoint, v, A)

    # Random input gradients (no sparsity at all)
    vjp_input = test_util.tree_random_like(prng.prng_key(3), fwd_output)

    # Call the auto-diff VJP
    dv_autodiff, dp_autodiff = vjp_autodiff(vjp_input)
    dv_adjoint, dp_adjoint = vjp_adjoint(vjp_input)

    # Assert gradients match
    test_util.assert_allclose(dv_adjoint, dv_autodiff)
    test_util.assert_allclose(dp_adjoint, dp_autodiff)

    # Assert that the values are only similar, not identical
    assert not np.all(dv_adjoint == dv_autodiff)
    assert not np.all(dp_adjoint == dp_autodiff)


@testing.parametrize("nrows", [15])
@testing.parametrize("num_matvecs", [10])
@testing.parametrize("reortho", ["full"])
def test_adjoint_matches_jax_dot_vjp_hilbert_matrix_and_full_reortho(
    nrows, num_matvecs, reortho
):
    config.update("jax_enable_x64", True)

    # Create a matrix and a direction as a test-case
    A = _lower(linalg.hilbert(nrows))
    v = prng.normal(prng.prng_key(2), shape=(nrows,), dtype=A.dtype)

    def matvec(s, p):
        return (p + p.T) @ s

    # Set up the algorithms
    algorithm_autodiff = decomp.hessenberg(
        num_matvecs, reortho=reortho, custom_vjp=False
    )
    algorithm_adjoint = decomp.hessenberg(num_matvecs, reortho=reortho, custom_vjp=True)

    # Forward pass
    algorithm_autodiff = func.partial(algorithm_autodiff, matvec)
    algorithm_adjoint = func.partial(algorithm_adjoint, matvec)
    fwd_output, vjp_autodiff = func.vjp(algorithm_autodiff, v, A)
    _fwd_output, vjp_adjoint = func.vjp(algorithm_adjoint, v, A)

    # Random input gradients (no sparsity at all)
    vjp_input = test_util.tree_random_like(prng.prng_key(3), fwd_output)

    # Call the auto-diff VJP
    dv_autodiff, dp_autodiff = vjp_autodiff(vjp_input)
    dv_adjoint, dp_adjoint = vjp_adjoint(vjp_input)

    # Assert gradients match
    print(dv_adjoint - dv_autodiff)
    test_util.assert_allclose(dv_adjoint, dv_autodiff)
    test_util.assert_allclose(dp_adjoint, dp_autodiff)

    # Assert that the values are only similar, not identical
    assert not np.all(dv_adjoint == dv_autodiff)
    assert not np.all(dp_adjoint == dp_autodiff)

    config.update("jax_enable_x64", False)


def _lower(m):
    m_tril = np.tril(m)
    return m_tril - 0.5 * linalg.diagonal_matrix(linalg.diagonal(m_tril))


@testing.parametrize("reortho_wrong", [True, "full_with_sparsity", "None"])
def test_raises_type_error_for_wrong_reorthogonalisation_flag(reortho_wrong):
    # Create a matrix and a direction as a test-case

    # Set up the algorithms
    with testing.raises(TypeError, match="Unexpected input"):
        _ = decomp.hessenberg(1, reortho=reortho_wrong)
