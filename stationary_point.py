# stationary_point.py
import numpy as np
from model_params import params

def stationary_point_exists(params, verbose=True):
    alpha = params['alpha']
    beta = params['beta']
    b = params.get('b', 0.04)
    rho = params['rho']

    lower = b / alpha
    upper = beta * (1 - b / (alpha * beta))

    exists = (lower < rho < upper)
    if verbose:
        print(f"Проверка условия существования стационарной точки:")
        print(f"  b/alpha = {lower:.4f}")
        print(f"  upper = {upper:.4f}")
        print(f"  rho = {rho:.4f}")
        if exists:
            print("Условие выполнено: стационарная точка существует.")
        else:
            print("Условие НЕ выполнено: стационарная точка НЕ существует для этих параметров!")
    return exists


def stationary_point(params):
    stationary_point_exists(params, verbose=True)
    alpha = params['alpha']
    beta = params['beta']
    mu = params['mu']
    rho = params['rho']
    p0 = params['p0']
    a = params['a']
    M0 = params['M0']
    b = params.get('b', 0.04)

    A_star = ((rho - b/alpha) * ((1 - alpha) * rho**2 + (alpha * beta - b) * (mu + rho))) / ((1 - alpha) * rho**2)
    B_star = (alpha * (1 - alpha) * rho**2 * (beta - rho) + b * (alpha * beta - b) * (rho + mu)) / (
        alpha * rho * (1 - alpha) * (rho + alpha * mu))

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
