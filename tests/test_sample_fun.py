"""Tests for some sample functions."""
from hutch import sample
from hutch.backend import np


def test_van_der_corput_next():
    expected = np.asarray([0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625])
    received = np.asarray([sample.van_der_corput(i) for i in range(9)])
    assert np.allclose(received, expected)

    expected = np.asarray([0.0, 1 / 3, 2 / 3, 1 / 9, 4 / 9, 7 / 9, 2 / 9, 5 / 9, 8 / 9])
    received = np.asarray([sample.van_der_corput(i, base=3) for i in range(9)])
    assert np.allclose(received, expected)
