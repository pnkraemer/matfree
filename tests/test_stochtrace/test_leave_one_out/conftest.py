"""Helpers for the leave-one-out experiments."""

from matfree import stochtrace
from matfree.backend import testing


@testing.fixture(name="nystrom")
@testing.parametrize(
    "factory", [stochtrace.nystrom_eigh, stochtrace.nystrom_shifted_cholesky]
)
def fixture_nystrom(factory):
    """Create a nystrom with default settings."""
    return factory()
