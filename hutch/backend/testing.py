"""Test-utilities."""

import jax.test_util
import pytest
import pytest_cases

fixture = pytest_cases.fixture
parametrize = pytest.mark.parametrize
raises = pytest.raises
check_grads = jax.test_util.check_grads
