"""Test-utilities."""

import jax.test_util
import pytest
import pytest_cases


def fixture(name=None):
    return pytest_cases.fixture(name=name)


def parametrize(argnames, argvalues, /):
    return pytest.mark.parametrize(argnames, argvalues)


def check_grads(fun, /, args, *, order, atol, rtol):
    return jax.test_util.check_grads(fun, args, order=order, atol=atol, rtol=rtol)


def raises(err, /):
    return pytest.raises(err)
