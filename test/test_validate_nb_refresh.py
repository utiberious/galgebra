from sympy import Abs, cos, simplify, sin, symbols, tan, trigsimp

from scripts.validate_nb_refresh import _norm_spherical_curl


def test_spherical_curl_forms_are_symbolically_equal():
    theta = symbols('theta', real=True)
    r = symbols('r', nonzero=True, real=True)
    A_phi, dtheta_A_phi, dphi_A_theta = symbols(
        'A_phi dtheta_A_phi dphi_A_theta', real=True
    )
    dr_A_phi, dphi_A_r = symbols('dr_A_phi dphi_A_r', real=True)
    sin_theta = sin(theta)

    old_radial = (
        A_phi*sin_theta**2
        + A_phi*sin_theta*cos(theta)*tan(theta)
        + sin_theta**2*tan(theta)*dtheta_A_phi
        - tan(theta)*dphi_A_theta
    ) / (tan(theta)*Abs(sin_theta))
    factored_radial = (
        2*A_phi/tan(theta)
        + dtheta_A_phi
        - dphi_A_theta/sin_theta**2
    ) * Abs(sin_theta)
    polar_numerator = (
        r**2*sin_theta**2*dr_A_phi
        + 2*r*A_phi*sin_theta**2
        - dphi_A_r
    )
    old_polar = -polar_numerator / (r**2*Abs(sin_theta))
    sign_distributed_polar = (
        -r**2*sin_theta**2*dr_A_phi
        - 2*r*A_phi*sin_theta**2
        + dphi_A_r
    ) / (r**2*Abs(sin_theta))

    assert simplify(trigsimp(old_radial - factored_radial)) == 0
    assert simplify(old_polar - sign_distributed_polar) == 0


def test_normalize_spherical_curl_radial_coefficient():
    old = (
        r'\frac{A^{\phi }  {\sin{\left (\theta  \right )}}^{2} + '
        r'A^{\phi }  \sin{\left (\theta  \right )} '
        r'\cos{\left (\theta  \right )} \tan{\left (\theta  \right )} + '
        r'{\sin{\left (\theta  \right )}}^{2} '
        r'\tan{\left (\theta  \right )} \partial_{\theta } A^{\phi }  - '
        r'\tan{\left (\theta  \right )} \partial_{\phi } A^{\theta } }'
        r'{\tan{\left (\theta  \right )} '
        r'\left|{\sin{\left (\theta  \right )}}\right|} '
        r'\boldsymbol{e}_{r}'
    )
    factored = (
        r'\left(\frac{2 A^{\phi } }{\tan{\left (\theta  \right )}} + '
        r'\partial_{\theta } A^{\phi }  - '
        r'\frac{\partial_{\phi } A^{\theta } }'
        r'{{\sin{\left (\theta  \right )}}^{2}}\right) '
        r'\left|{\sin{\left (\theta  \right )}}\right| '
        r'\boldsymbol{e}_{r}'
    )

    assert _norm_spherical_curl(old) == factored
    assert _norm_spherical_curl(factored) == factored


def test_normalize_spherical_curl_polar_coefficient_sign():
    old = (
        r'- \frac{r^{2} {\sin{\left (\theta  \right )}}^{2} '
        r'\partial_{r} A^{\phi }  + 2 r A^{\phi }  '
        r'{\sin{\left (\theta  \right )}}^{2} - '
        r'\partial_{\phi } A^{r} }'
        r'{r^{2} \left|{\sin{\left (\theta  \right )}}\right|} '
        r'\boldsymbol{e}_{\theta }'
    )
    sign_distributed = (
        r'+ \frac{- r^{2} {\sin{\left (\theta  \right )}}^{2} '
        r'\partial_{r} A^{\phi }  - 2 r A^{\phi }  '
        r'{\sin{\left (\theta  \right )}}^{2} + '
        r'\partial_{\phi } A^{r} }'
        r'{r^{2} \left|{\sin{\left (\theta  \right )}}\right|} '
        r'\boldsymbol{e}_{\theta }'
    )

    assert _norm_spherical_curl(old) == sign_distributed
    assert _norm_spherical_curl(sign_distributed) == sign_distributed


def test_spherical_curl_normalizer_rejects_nearby_change():
    changed = (
        r'\frac{A^{\phi }  {\sin{\left (\theta  \right )}}^{3}}'
        r'{\tan{\left (\theta  \right )} '
        r'\left|{\sin{\left (\theta  \right )}}\right|} '
        r'\boldsymbol{e}_{r}'
    )

    assert _norm_spherical_curl(changed) == changed
