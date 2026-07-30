from unittest import mock

from sympy import Add, cos, cosh, sin, sinh, symbols

from galgebra._utils import simplify as simplify_module
from galgebra.ga import Ga
from galgebra.metric import Simp


x = symbols('x')


def test_major_minor():
    assert simplify_module._major_minor('1.13.3') == (1, 13)
    assert simplify_module._major_minor('1.15.dev') == (1, 15)
    assert simplify_module._major_minor('unknown') == (0, 0)


def test_small_expression_uses_standard_simplify():
    expr = sin(x)**2 + cos(x)**2

    with (
        mock.patch.object(simplify_module, 'simplify', return_value=1) as new,
        mock.patch.object(simplify_module, 'trigsimp') as old,
    ):
        assert simplify_module.simplify_compat(expr) == 1

    new.assert_called_once_with(expr)
    old.assert_not_called()


def test_large_mixed_expression_avoids_standard_simplify():
    terms = [sin(x + i) + sinh(x + i) for i in range(40)]
    expr = Add(*terms)

    with (
        mock.patch.object(simplify_module, 'simplify') as new,
        mock.patch.object(simplify_module, 'trigsimp', return_value=expr) as old,
    ):
        assert simplify_module.simplify_compat(expr) == expr

    new.assert_not_called()
    old.assert_called_once_with(expr, method='old')


def test_sympy_before_1_13_uses_standard_simplify():
    terms = [sin(x + i) + sinh(x + i) for i in range(40)]
    expr = Add(*terms)

    with (
        mock.patch.object(simplify_module, '_SYMPY_MAJOR_MINOR', (1, 12)),
        mock.patch.object(simplify_module, 'simplify', return_value=1) as new,
        mock.patch.object(simplify_module, 'trigsimp') as old,
    ):
        assert simplify_module.simplify_compat(expr) == 1

    new.assert_called_once_with(expr)
    old.assert_not_called()


def test_custom_simp_profile_overrides_compatibility_default():
    original_modes = Simp.modes[:]
    custom = mock.Mock(return_value=x)
    Simp.profile([custom])
    try:
        assert Simp.apply(sin(x)) == x
    finally:
        Simp.profile(original_modes)

    custom.assert_called_once_with(sin(x))


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
