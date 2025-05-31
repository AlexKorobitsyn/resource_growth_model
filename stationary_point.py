# stationary_point.py
import numpy as np
from model_params import params


def stationary_point(params):

    alpha = params['alpha']
    beta = params['beta']
    mu = params['mu']
    rho = params['rho']
    p0 = params['p0']
    a = params['a']
    M0 = params['M0']
    b = params.get('b', 0.04)  # b — это коэффициент из статьи, можно подбирать

    # Формулы для A*, B* (по статье)
    A_star = ((rho - b/alpha) * ((1 - alpha) * rho**2 + (alpha * beta - b) * (mu + rho))) / ((1 - alpha) * rho**2)
    B_star = (alpha * (1 - alpha) * rho**2 * (beta - rho) + b * (alpha * beta - b) * (rho + mu)) / (
        alpha * rho * (1 - alpha) * (rho + alpha * mu))

    # Теперь по картинке:
    z1_star = A_star * B_star / (rho - b / alpha)
    z2_star = 1 / rho * (1 - b / alpha * z1_star)
    z3_star = mu / rho * z2_star - (1 - alpha) / (alpha * rho) * (1 + A_star * B_star)
    x3_star = 1 / z2_star + mu
    x1_star = p0 / (a ** (1 / alpha) * A_star) * x3_star ** ((1 - alpha) / alpha)
    x2_star = M0 * b / alpha * x1_star
    print(x1_star, x2_star, x3_star, A_star, B_star)
    z_star = np.array([z1_star, z2_star, z3_star])
    x_star = np.array([x1_star, x2_star, x3_star])
    return x_star, z_star
