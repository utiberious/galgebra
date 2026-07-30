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
# simplifier. The observed slow expression combines trigonometric and
# hyperbolic functions under non-integral powers; its cost is proportional to
# the expression tree size times the number of those function nodes.
_FU_TRAVERSAL_COST_LIMIT = 4096


def _has_expensive_fu_traversal(expr):
    """Whether ``simplify`` is likely to hit SymPy's slow FU traversal."""
    if _SYMPY_MAJOR_MINOR < (1, 13):
        return False

    nodes = list(preorder_traversal(expr))
    trig_nodes = sum(
        isinstance(node, TrigonometricFunction) for node in nodes
    )
    hyperbolic_nodes = sum(
        isinstance(node, HyperbolicFunction) for node in nodes
    )
    traversal_cost = len(nodes) * (trig_nodes + hyperbolic_nodes)
    if (
        trig_nodes == 0
        or hyperbolic_nodes == 0
        or traversal_cost < _FU_TRAVERSAL_COST_LIMIT
    ):
        return False

    return any(
        (
            node.is_Pow
            and node.exp.is_integer is False
            and node.base.is_Add
            and node.base.has(TrigonometricFunction)
            and node.base.has(HyperbolicFunction)
        )
        for node in nodes
    )


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
