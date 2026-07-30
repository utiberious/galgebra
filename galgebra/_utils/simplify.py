"""Compatibility helpers for simplification across SymPy releases."""

import re

import sympy
from sympy import preorder_traversal, simplify, trigsimp
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction


def _major_minor(version):
    """Return the leading major and minor numbers from a version string."""
    match = re.match(r'^(\d+)\.(\d+)', version)
    if match is None:
        return (0, 0)
    return tuple(map(int, match.groups()))


_SYMPY_MAJOR_MINOR = _major_minor(sympy.__version__)

# SymPy 1.13's gh-26390 added a nested replace traversal to the FU
# simplifier. Its cost is proportional to the expression tree size times the
# number of trig and hyperbolic nodes.
_FU_TRAVERSAL_COST_LIMIT = 4096
_TRIG_FUNCTIONS = (TrigonometricFunction, HyperbolicFunction)


def _has_expensive_fu_traversal(expr):
    """Whether ``simplify`` is likely to hit SymPy's slow FU traversal."""
    if _SYMPY_MAJOR_MINOR < (1, 13):
        return False

    trig_nodes = 0
    for node_count, node in enumerate(preorder_traversal(expr), 1):
        if isinstance(node, _TRIG_FUNCTIONS):
            trig_nodes += 1
        if node_count * trig_nodes >= _FU_TRAVERSAL_COST_LIMIT:
            return True
    return False


def simplify_for_display(expr):
    """Simplify display output while avoiding a SymPy 1.13+ regression.

    This helper is only for rendering. Algebraic operations retain ordinary
    ``simplify``. Remove the fallback after SymPy replaces the nested
    traversal introduced by gh-26390 and galgebra's minimum supported SymPy
    includes that fix.
    """
    if _has_expensive_fu_traversal(expr):
        return trigsimp(expr, method='old')
    return simplify(expr)
