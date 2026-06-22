"""Unit tests for nystrom methods."""

from matfree import stochtrace, test_util
from matfree.backend import linalg, np, prng, testing
from matfree.backend.typing import Array


def _drop_column(x: Array, i: int):
    """Return x with column i dropped."""
    idx = np.arange(x.shape[1])
    return x[:, idx != i]


def _append_columns(x: Array, idx_copy: Array):
    """Return x with the columns at idx_copy appended to the end."""
    x_copy = x[:, idx_copy]
    return np.concatenate([x.T, x_copy.T]).T


def sample_psd_matrix(key, n: int, rank: int, dtype):
    A = prng.normal(key, shape=(n, rank), dtype=dtype)
    return (A @ A.T.conj()) / n


@testing.fixture(name="nystrom")
@testing.parametrize(
    "factory", [stochtrace.nystrom_eigh, stochtrace.nystrom_shifted_cholesky]
)
def fixture_nystrom(factory):
    """Create a nystrom with consistent settings for LOO downdate testing."""
    if factory is stochtrace.nystrom_shifted_cholesky:
        # Choose a large shift since we want to test the shift-based approximation.
        return factory(shift=1.0)
    return factory()


@testing.parametrize("n", [10, 20])
@testing.parametrize("dtype", [float, complex])
def test_nystrom_shifted_cholesky_kwargs_customizable(n, dtype):
    """Assert nystrom_shifted_cholesky kwargs can be customized."""

    num_samples = 4
    key = prng.prng_key(1)
    key_mat, key_omega = prng.split(key)
    A = sample_psd_matrix(key_mat, n, n, dtype)
    Omega = prng.normal(key_omega, shape=(n, num_samples), dtype=dtype)
    Y = A @ Omega
    Y_norm = linalg.vector_norm(Y)

    def matvec(v):
        return A @ v

    nystrom = stochtrace.nystrom_shifted_cholesky()
    _, _, shift = nystrom(matvec, Omega)
    assert shift == np.finfo_eps(Y_norm.dtype) * Y_norm / n**0.5

    nystrom = stochtrace.nystrom_shifted_cholesky(shift=1.0)
    _, _, shift = nystrom(matvec, Omega)
    assert shift == 1.0

    nystrom = stochtrace.nystrom_shifted_cholesky(shift=None, rtol=1e-3)
    _, _, shift = nystrom(matvec, Omega)
    assert shift == 1e-3 * Y_norm / n**0.5


@testing.parametrize("n, rank, num_samples", [(10, 2, 4), (20, 3, 8)])
@testing.parametrize("dtype", [float, complex])
def test_nystrom_eigh_kwargs_customizable(n, rank, num_samples, dtype):
    """Assert nystrom_eigh kwargs can be customized."""

    num_duplicated_cols = 2
    key = prng.prng_key(1)
    key_mat, key_omega = prng.split(key)
    A = sample_psd_matrix(key_mat, n, rank, dtype)
    num_unique_cols = num_samples - num_duplicated_cols
    Omega = prng.normal(key_omega, shape=(n, num_unique_cols), dtype=dtype)
    Omega = np.concatenate([Omega.T, Omega[:, :num_duplicated_cols].T]).T

    def matvec(v):
        return A @ v

    nystrom = stochtrace.nystrom_eigh()
    F, Z, shift = nystrom(matvec, Omega)
    assert shift == 0.0

    # set a too-loose eigenvalues rtol
    nystrom = stochtrace.nystrom_eigh(eigenvalues_rtol=1.0)
    F2, _, _ = nystrom(matvec, Omega)
    assert not np.allclose(F2, F)

    # set a too-loose leverage rtol
    nystrom = stochtrace.nystrom_eigh(leverage_rtol=1.0)
    _, Z3, _ = nystrom(matvec, Omega)
    assert not np.allclose(Z3, Z)


@testing.parametrize("dtype", [float, complex])
@testing.parametrize("n, rank, num_samples", [(10, 10, 4), (20, 5, 8)])
def test_nystrom_approximation_is_psd(nystrom, dtype, n, rank, num_samples):
    """Assert that F @ F.T.conj() is PSD."""
    key = prng.prng_key(1)
    key_mat, key_omega = prng.split(key)
    A = sample_psd_matrix(key_mat, n, rank, dtype)
    Omega = prng.normal(key_omega, shape=(n, num_samples), dtype=dtype)

    def matvec(v):
        return A @ v

    F, _, _ = nystrom(matvec, Omega)
    eigenvalues = linalg.eigh(F @ F.T.conj()).eigenvalues
    # Allow small negative values relative to the trace (sum of eigenvalues)
    assert np.all(eigenvalues >= -1e-6 * np.sum(np.abs(eigenvalues)))


@testing.parametrize("dtype", [float, complex])
@testing.parametrize("n, rank, num_samples", [(10, 10, 4), (20, 5, 8)])
def test_nystrom_interpolatory_property(nystrom, dtype, n, rank, num_samples):
    """Assert A_hat @ Omega = A @ Omega."""
    key = prng.prng_key(2)
    key_mat, key_omega = prng.split(key)
    A = sample_psd_matrix(key_mat, n, rank, dtype)
    Omega = prng.normal(key_omega, shape=(n, num_samples), dtype=dtype)

    def matvec(v):
        return A @ v

    F, _, shift = nystrom(matvec, Omega)
    received = F @ (F.T.conj() @ Omega) - shift * Omega
    expected = A @ Omega
    test_util.assert_allclose(received, expected)


@testing.parametrize("dtype", [float, complex])
@testing.parametrize("n, rank, num_samples", [(10, 10, 4), (20, 5, 8)])
def test_nystrom_right_mul_invariance(nystrom, dtype, n, rank, num_samples):
    """Assert Nystrom approximation is invariant under right multiplication."""
    key = prng.prng_key(2)
    key_mat, key_omega, key_V = prng.split(key, 3)
    A = sample_psd_matrix(key_mat, n, rank, dtype)
    Omega = prng.normal(key_omega, shape=(n, num_samples), dtype=dtype)
    V = prng.normal(key_V, shape=(num_samples, num_samples), dtype=dtype)
    Omega_V = Omega @ V

    def matvec(v):
        return A @ v

    F, _, _ = nystrom(matvec, Omega)
    F_V, _, _ = nystrom(matvec, Omega_V)
    test_util.assert_allclose(F_V @ F_V.T.conj(), F @ F.T.conj())


@testing.parametrize("dtype", [float, complex])
@testing.parametrize("n, rank", [(10, 5), (20, 8)])
def test_nystrom_eigh_rank_min_of_num_samples_and_operator_rank(dtype, n, rank):
    """Assert Nystrom rank is min(operator rank, num test vectors)."""
    key = prng.prng_key(4)
    key_mat, key_omega = prng.split(key)
    A = sample_psd_matrix(key_mat, n, rank, dtype)
    Omega = prng.normal(key_omega, shape=(n, n), dtype=dtype)
    nystrom = stochtrace.nystrom_eigh()

    def matvec(v):
        return A @ v

    num_samples = np.arange(1, n + 1)
    nnz_cols = []
    for k in num_samples:
        F, _, _ = nystrom(matvec, Omega[:, :k])
        nnz_cols.append(np.sum(np.sum(linalg.abs2(F), axis=0) > 0))

    rank_expected = np.elementwise_min(num_samples, rank)
    nnz_cols = np.asarray(nnz_cols)
    assert np.allclose(nnz_cols, rank_expected)


@testing.parametrize("dtype", [float, complex])
@testing.parametrize("n, rank, num_samples", [(8, 8, 4), (12, 5, 8)])
def test_nystrom_loo_downdate_matches_direct_nystrom(
    nystrom, dtype, n, rank, num_samples
):
    """Assert downdate matrix Z is correct.

    Explicitly computes the Nystrom approximation with each column removed and
    compares it to the downdate formula using the returned Z matrix.
    """
    key = prng.prng_key(3)
    key_mat, key_omega = prng.split(key)
    A = sample_psd_matrix(key_mat, n, rank, dtype)
    Omega = prng.normal(key_omega, shape=(n, num_samples), dtype=dtype)

    def matvec(v):
        return A @ v

    F, Z, _shift = nystrom(matvec, Omega)
    B_hat = F @ F.T.conj()

    for i in range(num_samples):
        Omega_i = _drop_column(Omega, i)
        F_i, _Z_i, _shift_i = nystrom(matvec, Omega_i)
        B_hat_i_direct = F_i @ F_i.T.conj()

        z_i = Z[:, i]
        B_hat_i_downdate = B_hat - linalg.outer(z_i, z_i.conj())

        test_util.assert_allclose(B_hat_i_downdate, B_hat_i_direct)


@testing.parametrize("dtype", [float, complex])
@testing.parametrize("n, rank, num_samples", [(10, 3, 4), (20, 10, 11)])
def test_nystrom_eigh_low_rank_operator_recovers_exact_approximation(
    dtype, n, rank, num_samples
):
    """Assert nystrom_eigh recovers the exact operator when num_samples > rank(A).

    With Gaussian test vectors, range(A @ Omega) = range(A) with probability 1,
    so the Nystrom approximation A_hat = A exactly.  This also verifies that
    the pseudoinverse correctly handles rank(H) < num_samples.
    """
    key = prng.prng_key(4)
    key_mat, key_omega = prng.split(key)
    A = sample_psd_matrix(key_mat, n, rank, dtype=dtype)
    Omega = prng.normal(key_omega, shape=(n, num_samples), dtype=dtype)
    nystrom = stochtrace.nystrom_eigh()

    def matvec(v):
        return A @ v

    F, _, _ = nystrom(matvec, Omega)
    test_util.assert_allclose(F @ F.T.conj(), A)


@testing.parametrize("dtype", [float, complex])
@testing.parametrize("n, rank, num_samples", [(10, 3, 6), (20, 4, 10)])
def test_nystrom_eigh_low_rank_operator_loo_downdate(dtype, n, rank, num_samples):
    """Assert the LOO downdate formula is correct when rank(A) < num_samples."""
    rdtype = np.abs(dtype(0)).dtype
    key = prng.prng_key(7)
    key_U, key_omega = prng.split(key)
    U = linalg.qr_reduced(prng.normal(key_U, shape=(n, rank), dtype=dtype))[0]
    d = np.ones(rank, dtype=rdtype)
    A = (U * d) @ U.T.conj()

    Omega = prng.normal(key_omega, shape=(n, num_samples), dtype=dtype)
    nystrom = stochtrace.nystrom_eigh()

    def matvec(v):
        return A @ v

    F, Z, _shift = nystrom(matvec, Omega)
    A_hat = F @ F.T.conj()

    for i in range(num_samples):
        Omega_i = _drop_column(Omega, i)
        F_i, _Z_i, _shift_i = nystrom(matvec, Omega_i)
        A_hat_i_direct = F_i @ F_i.T.conj()

        z_i = Z[:, i]
        A_hat_i_downdate = A_hat - linalg.outer(z_i, z_i.conj())

        test_util.assert_allclose(A_hat_i_downdate, A_hat_i_direct)


@testing.parametrize("dtype", [float, complex])
@testing.parametrize("n, num_unique_samples, num_copies", [(10, 3, 2), (12, 4, 3)])
def test_nystrom_eigh_non_essential_test_vectors_ignored(
    dtype, n, num_unique_samples, num_copies
):
    """Assert nystrom_eigh is invariant to non-essential test vectors."""
    key = prng.prng_key(5)
    key_mat, key_omega = prng.split(key)
    A = sample_psd_matrix(key_mat, n, n, dtype)
    Omega_unique = prng.normal(key_omega, shape=(n, num_unique_samples), dtype=dtype)
    Omega = _append_columns(Omega_unique, np.arange(num_copies))

    def matvec(v):
        return A @ v

    F_unique, Z_unique, shift_unique = stochtrace.nystrom_eigh()(matvec, Omega_unique)
    F, Z, shift = stochtrace.nystrom_eigh()(matvec, Omega)

    assert np.allclose(F @ F.T.conj(), F_unique @ F_unique.T.conj(), atol=1e-5)
    received = Z[:, num_copies:-num_copies]
    expected = Z_unique[:, num_copies:]
    assert np.allclose(received, expected, atol=1e-5)
    assert np.allclose(Z[:, :num_copies], 0.0)
    assert np.allclose(Z[:, -num_copies:], 0.0)
    assert np.allclose(shift, shift_unique)


@testing.parametrize("dtype", [float, complex])
@testing.parametrize("n, num_unique_samples, num_copies", [(10, 3, 2), (12, 4, 1)])
def test_nystrom_eigh_non_essential_test_vectors_loo_downdate(
    dtype, n, num_unique_samples, num_copies
):
    """Assert the LOO downdate formula is correct for non-essential test vectors."""
    key = prng.prng_key(6)
    key_mat, key_omega = prng.split(key)
    A = sample_psd_matrix(key_mat, n, n, dtype)
    Omega_unique = prng.normal(key_omega, shape=(n, num_unique_samples), dtype=dtype)
    Omega = _append_columns(Omega_unique, np.arange(num_copies))
    num_samples = num_unique_samples + num_copies

    nystrom = stochtrace.nystrom_eigh()

    def matvec(v):
        return A @ v

    F, Z, _shift = nystrom(matvec, Omega)
    A_hat = F @ F.T.conj()

    for i in range(num_samples):
        Omega_i = _drop_column(Omega, i)
        F_i, _Z_i, _shift_i = nystrom(matvec, Omega_i)
        A_hat_i_direct = F_i @ F_i.T.conj()

        z_i = Z[:, i]
        A_hat_i_downdate = A_hat - linalg.outer(z_i, z_i.conj())

        test_util.assert_allclose(A_hat_i_downdate, A_hat_i_direct)
