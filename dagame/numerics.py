"""One shared definition of "equal", "greater" and "tie" for the whole package.

Every strict comparison in the solver, the tree engine and the checker goes through these helpers,
so a tie means the same thing everywhere. The tolerance is relative to the magnitude of the numbers
being compared (but never smaller than an absolute 1e-9).
"""
from __future__ import annotations

import numpy as np

TOL = 1e-9


def _scale(a, b):
    return np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))


def gt(a, b):
    """a is strictly greater than b, beyond numerical noise. Works on floats and arrays."""
    return (np.asarray(a) - np.asarray(b)) > TOL * _scale(a, b)


def lt(a, b):
    return gt(b, a)


def close(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)) <= TOL * _scale(a, b)
