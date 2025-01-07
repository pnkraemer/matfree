"""Test the adjoint of tri-diagonalisation."""

from matfree import decomp, test_util
from matfree.backend import func, linalg, np, prng, testing, tree


@testing.parametrize("reortho", ["full", "none"])
def test_adjoint_vjp_matches_jax_vjp(reortho, n=10, krylov_num_matvecs=4):
    """Test that the custom VJP yields the same output as autodiff."""
    # Set up a test-matrix
    eigvals = prng.uniform(prng.prng_key(2), shape=(n,)) + 1.0
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    params = _sym(matrix)

    def matvec(s, p):
        return (p + p.T) @ s

    # Set up an initial vector
    vector = prng.normal(prng.prng_key(1), shape=(n,))

    # Flatten the inputs
    flat, unflatten = tree.ravel_pytree((vector, params))

    # Construct a vector-to-vector decomposition function
    def decompose(f, *, custom_vjp):
        kwargs = {"reortho": reortho, "custom_vjp": custom_vjp, "materialize": False}
        algorithm = decomp.tridiag_sym(krylov_num_matvecs, **kwargs)
        output = algorithm(matvec, *unflatten(f))
        return tree.ravel_pytree(output)[0]

    # Construct the two implementations
    reference = func.jit(func.partial(decompose, custom_vjp=False))
    implementation = func.jit(func.partial(decompose, custom_vjp=True))

    # Compute both VJPs
    fx_ref, vjp_ref = func.vjp(reference, flat)
    fx_imp, vjp_imp = func.vjp(implementation, flat)
    # Assert that the forward-passes are identical
    assert np.allclose(fx_ref, fx_imp)

    # Assert that the VJPs into a bunch of random directions are identical
    for seed in [4, 5, 6]:
        key = prng.prng_key(seed)
        dnu = prng.normal(key, shape=np.shape(reference(flat)))
        assert np.allclose(*vjp_ref(dnu), *vjp_imp(dnu), atol=1e-4, rtol=1e-4)


def _sym(m):
    return np.triu(m) - linalg.diagonal_matrix(0.5 * linalg.diagonal(m))
