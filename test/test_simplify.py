from unittest import mock

from sympy import Add, cos, cosh, simplify, sin, sinh, symbols

from galgebra._utils import simplify as simplify_module
from galgebra.ga import Ga
from galgebra.metric import Simp


x, y = symbols('x y')


def _paired_trig_expression(count, extra=0):
    terms = [sin(x + i) + cos(x + i) for i in range(count)]
    if extra != 0:
        terms.append(extra)
    return Add(*terms, evaluate=False)


def test_major_minor():
    assert simplify_module._major_minor('1.13.3') == (1, 13)
    assert simplify_module._major_minor('1.15.dev') == (1, 15)
    assert simplify_module._major_minor('unknown') == (0, 0)


def test_boundary_routes_only_at_or_above_limit():
    below = _paired_trig_expression(15)
    above = _paired_trig_expression(16)

    assert not simplify_module._has_expensive_fu_traversal(below)
    assert simplify_module._has_expensive_fu_traversal(above)


def test_algebra_keeps_general_simplification_above_display_boundary():
    rational = (y**2 - 1)/(y - 1)
    expr = _paired_trig_expression(16, rational)

    result = Simp.apply(expr)

    assert not result.has(rational)
    assert simplify(result - expr) == 0


def test_display_route_can_preserve_unrelated_algebraic_form():
    rational = (y**2 - 1)/(y - 1)
    expr = _paired_trig_expression(16, rational)

    with (
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


def test_copied_default_profile_restores_display_fallback():
    original_modes = Simp.modes
    restored_modes = Simp.modes[:]
    try:
        Simp.profile([mock.Mock(return_value=x)])
        Simp.profile(restored_modes)
        with mock.patch(
            'galgebra.metric.simplify_for_display', return_value=1
        ) as display:
            assert Simp.apply_display(x) == 1
    finally:
        Simp.modes = original_modes

    display.assert_called_once_with(x)


def test_prolate_spheroidal_divergence_renders():
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
