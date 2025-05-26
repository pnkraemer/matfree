"""Container types."""

import dataclasses
from typing import NamedTuple  # noqa: F401


def dataclass(dcls, /):
    return dataclasses.dataclass(dcls)
