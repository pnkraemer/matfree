"""Helpers for the leave-one-out experiments."""

import numpy as np


def exp_eigvals(n, /):
    """Eigenvalues that decay rapidly."""
    return 0.7 ** np.arange(n)


def step_eigvals(n, /, *, num_flat=50, drop_value=1e-3):
    """Eigenvalues that are flat with a sudden drop."""
    eigvals_flat = np.ones(num_flat)
    eigvals_drop = np.ones(n - num_flat) * drop_value
    return np.concatenate([eigvals_flat, eigvals_drop])
