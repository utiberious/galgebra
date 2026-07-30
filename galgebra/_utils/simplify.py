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
# simplifier. Match only the observed two-term prolate radical shape. A
# numerical tree-cost heuristic admitted benign expressions whose unrelated
# terms happened to produce the same score.


def _is_squared_function(term, function_type):
    return (
        term.is_Pow
        and term.exp == 2
        and isinstance(term.base, function_type)
    )


def _is_mixed_squared_base(base):
    """Whether ``base`` is one trig square plus one hyperbolic square."""
    if not base.is_Add or len(base.args) != 2:
        return False
    return (
        any(
            _is_squared_function(term, TrigonometricFunction)
            for term in base.args
        )
        and any(
            _is_squared_function(term, HyperbolicFunction)
            for term in base.args
        )
    )


def _has_expensive_fu_traversal(expr):
    """Whether ``simplify`` is likely to hit SymPy's slow FU traversal."""
    if _SYMPY_MAJOR_MINOR < (1, 13):
        return False

    return any(
        (
            node.is_Pow
            and abs(node.exp) == sympy.S.Half
            and _is_mixed_squared_base(node.base)
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
