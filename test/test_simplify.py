from unittest import mock

import pytest
from sympy import Add, Rational, cos, cosh, simplify, sin, sinh, sqrt, symbols

from galgebra._utils import simplify as simplify_module
from galgebra.ga import Ga
from galgebra.metric import Simp


x, y, u, v = symbols('x y u v')
z = symbols('z:4')


def _paired_trig_expression(count, extra=0):
    terms = [sin(x + i) + cos(x + i) for i in range(count)]
    if extra != 0:
        terms.append(extra)
    return Add(*terms, evaluate=False)


def _mixed_nested_expression():
    return 1/sqrt(sin(x)**2 + sinh(y)**2)


def test_major_minor():
    assert simplify_module._major_minor('1.13.3') == (1, 13)
    assert simplify_module._major_minor('1.15.dev') == (1, 15)
    assert simplify_module._major_minor('unknown') == (0, 0)


def test_routes_only_observed_mixed_squared_shape():
    plain = sqrt(sin(x) + sinh(y))
    observed = _mixed_nested_expression()
    other_power = (sin(x)**2 + sinh(y)**2)**Rational(1, 3)

    with mock.patch.object(
        simplify_module, '_SYMPY_MAJOR_MINOR', (1, 13)
    ):
        assert not simplify_module._has_expensive_fu_traversal(plain)
        assert not simplify_module._has_expensive_fu_traversal(other_power)
        assert simplify_module._has_expensive_fu_traversal(observed)


def test_shallow_trig_sum_uses_real_general_simplifier():
    rational = (y**2 - 1)/(y - 1)
    expr = _paired_trig_expression(16, rational)

    result = simplify_module.simplify_for_display(expr)

    assert not result.has(rational)
    assert simplify(result - expr) == 0


def test_small_mixed_radical_cannot_borrow_unrelated_expression_cost():
    rational = (y**2 - 1)/(y - 1)
    expr = Add(
        *[sin(x + i) + cos(x + i) for i in range(16)],
        rational,
        sqrt(sin(u) + sinh(v)),
        evaluate=False,
    )

    with mock.patch.object(
        simplify_module, '_SYMPY_MAJOR_MINOR', (1, 13)
    ):
        assert not simplify_module._has_expensive_fu_traversal(expr)

    result = simplify_module.simplify_for_display(expr)

    assert not result.has(rational)
    assert simplify(result - expr) == 0


def test_large_benign_mixed_radical_uses_general_simplifier():
    rational = (y**2 - 1)/(y - 1)
    expr = Add(
        *[sin(x + i) + cos(x + i) for i in range(16)],
        rational,
        sqrt(sin(u) + sinh(v) + sum(z)),
        evaluate=False,
    )

    with mock.patch.object(
        simplify_module, '_SYMPY_MAJOR_MINOR', (1, 13)
    ):
        assert not simplify_module._has_expensive_fu_traversal(expr)

    result = simplify_module.simplify_for_display(expr)

    assert not result.has(rational)
    assert simplify(result - expr) == 0


def test_shallow_mixed_sum_does_not_match_failure_shape():
    terms = [
        sin(x + i) + cos(x + i) + sinh(y + i) + cosh(y + i)
        for i in range(8)
    ]
    expr = Add(*terms, evaluate=False)

    with mock.patch.object(
        simplify_module, '_SYMPY_MAJOR_MINOR', (1, 13)
    ):
        assert not simplify_module._has_expensive_fu_traversal(expr)


def test_algebra_keeps_general_simplification():
    rational = (y**2 - 1)/(y - 1)

    with mock.patch(
        'galgebra.metric.simplify_for_display'
    ) as display:
        result = Simp.apply(rational)

    assert not result.has(rational)
    assert simplify(result - rational) == 0
    display.assert_not_called()


def test_display_route_can_preserve_unrelated_algebraic_form():
    rational = (y**2 - 1)/(y - 1)
    expr = _mixed_nested_expression() + rational

    with (
        mock.patch.object(
            simplify_module, '_SYMPY_MAJOR_MINOR', (1, 13)
        ),
        mock.patch.object(simplify_module, 'simplify') as general,
        mock.patch.object(
            simplify_module, 'trigsimp', return_value=expr
        ) as old,
    ):
        assert Simp.apply_display(expr) == expr

    general.assert_not_called()
    old.assert_called_once()
    routed = old.call_args.args[0]
    assert old.call_args.kwargs == {'method': 'old'}
    assert simplify(routed - expr) == 0


def test_small_display_expression_uses_general_simplification():
    expr = sin(x)**2 + cos(x)**2

    with (
        mock.patch.object(
            simplify_module, 'simplify', return_value=1
        ) as general,
        mock.patch.object(simplify_module, 'trigsimp') as old,
    ):
        assert Simp.apply_display(expr) == 1

    general.assert_called_once()
    assert simplify(general.call_args.args[0] - expr) == 0
    old.assert_not_called()


def test_sympy_before_1_13_uses_general_display_simplification():
    expr = _paired_trig_expression(16)

    with (
        mock.patch.object(simplify_module, '_SYMPY_MAJOR_MINOR', (1, 12)),
        mock.patch.object(
            simplify_module, 'simplify', return_value=1
        ) as general,
        mock.patch.object(simplify_module, 'trigsimp') as old,
    ):
        assert Simp.apply_display(expr) == 1

    general.assert_called_once()
    assert simplify(general.call_args.args[0] - expr) == 0
    old.assert_not_called()


def test_custom_profile_overrides_display_fallback():
    original_modes = Simp.modes
    custom = mock.Mock(return_value=x)
    Simp.profile([custom])
    try:
        assert Simp.apply_display(_paired_trig_expression(16)) == x
    finally:
        Simp.modes = original_modes

    custom.assert_called_once()


def test_default_profile_object_restores_display_fallback():
    original_modes = Simp.modes
    try:
        Simp.profile([mock.Mock(return_value=x)])
        Simp.profile(original_modes)
        with mock.patch(
            'galgebra.metric.simplify_for_display', return_value=1
        ) as display:
            assert Simp.apply_display(x) == 1
    finally:
        Simp.modes = original_modes

    display.assert_called_once_with(x)


def test_explicit_simplify_profile_overrides_display_fallback():
    original_modes = Simp.modes
    try:
        Simp.profile([simplify])
        with mock.patch(
            'galgebra.metric.simplify_for_display'
        ) as display:
            assert Simp.apply_display(sin(x)**2 + cos(x)**2) == 1
    finally:
        Simp.modes = original_modes

    display.assert_not_called()


def test_in_place_profile_change_overrides_display_fallback():
    custom = mock.Mock(return_value=x)
    Simp.modes.append(custom)
    try:
        with mock.patch(
            'galgebra.metric.simplify_for_display'
        ) as display:
            assert Simp.apply_display(x) == x
    finally:
        Simp.modes.remove(custom)

    custom.assert_called_once_with(x)
    display.assert_not_called()


def test_prolate_spheroidal_divergence_renders():
    if simplify_module._SYMPY_MAJOR_MINOR < (1, 13):
        pytest.skip('display fallback targets SymPy 1.13 and newer')

    a = symbols('a', real=True)
    coords = xi, eta, phi = symbols('xi eta phi', real=True)
    ps3d, *_ = Ga.build(
        'e_xi e_eta e_phi',
        X=[
            a*sinh(xi)*sin(eta)*cos(phi),
            a*sinh(xi)*sin(eta)*sin(phi),
            a*cosh(xi)*cos(eta),
        ],
        coords=coords,
        norm=True,
    )
    vector = ps3d.mv('A', 'vector', f=True)

    rendered = str(ps3d.grad | vector)

    assert 'D{eta}A__eta' in rendered
    assert 'D{phi}A__phi' in rendered
    assert 'D{xi}A__xi' in rendered
