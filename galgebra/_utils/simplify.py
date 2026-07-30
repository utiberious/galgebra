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
# simplifier. The observed slow expression contains a sufficiently complex
# additive base of trigonometric and hyperbolic functions under a
# non-integral power. Score each such base independently so unrelated terms
# elsewhere in the expression cannot make a small radical look expensive.
_FU_TRAVERSAL_COST_LIMIT = 18


def _fu_candidate_cost(base):
    """Estimate nested traversal work within one candidate power base."""
    nodes = list(preorder_traversal(base))
    function_nodes = sum(
        isinstance(node, (TrigonometricFunction, HyperbolicFunction))
        for node in nodes
    )
    return len(nodes) * function_nodes


def _has_expensive_fu_traversal(expr):
    """Whether ``simplify`` is likely to hit SymPy's slow FU traversal."""
    if _SYMPY_MAJOR_MINOR < (1, 13):
        return False

    return any(
        (
            node.is_Pow
            and node.exp.is_integer is False
            and node.base.is_Add
            and node.base.has(TrigonometricFunction)
            and node.base.has(HyperbolicFunction)
            and _fu_candidate_cost(node.base) >= _FU_TRAVERSAL_COST_LIMIT
        )
        for node in preorder_traversal(expr)
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
