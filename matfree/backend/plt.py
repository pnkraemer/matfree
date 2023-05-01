"""Plotting functionality."""

# Not part of dependencies because not used by default, only for debugging.
import matplotlib.pyplot  # noqa: ICN001


def subplots(nrows=1, ncols=1, *, figsize, tight_layout=True, dpi=100):
    return matplotlib.pyplot.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, tight_layout=tight_layout, dpi=dpi
    )


def show():
    return matplotlib.pyplot.show()
