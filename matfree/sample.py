"""Sampling algorithms."""

from matfree.backend import containers, control_flow, func, prng


def normal(*, shape, dtype):
    """Standard normal distributions."""

    def fun(key):
        return prng.normal(key, shape=shape, dtype=dtype)

    return fun


def rademacher(*, shape, dtype):
    """Normalised Rademacher distributions."""

    def fun(key):
        return prng.rademacher(key, shape=shape, dtype=dtype)

    return fun


_VDCState = containers.namedtuple("VDCState", ["n", "vdc", "denom"])


def van_der_corput(n, /, base=2):
    """Compute the 'n'th element of the Van-der-Corput sequence."""
    state = _VDCState(n, vdc=0, denom=1)

    vdc_modify = func.partial(_van_der_corput_modify, base=base)
    state = control_flow.while_loop(_van_der_corput_cond, vdc_modify, state)
    return state.vdc


def _van_der_corput_cond(state: _VDCState):
    return state.n > 0


def _van_der_corput_modify(state: _VDCState, *, base):
    denom = state.denom * base
    num, remainder = divmod(state.n, base)
    vdc = state.vdc + remainder / denom
    return _VDCState(num, vdc, denom)
