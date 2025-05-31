# hamiltonian_system.py
import numpy as np

def hamiltonian_rhs(t, Y, params):
    alpha = params['alpha']
    beta = params['beta']
    mu = params['mu']
    M0 = params['M0']
    p0 = params['p0']
    a = params['a']
    b = params.get('b', 0.04)
    rho = params['rho']


    x1, x2, x3, z1, z2, z3 = Y

    # Вспомогательные функции:
    def A(x1, x3):
        if x3 <= 0:
            # Можешь сразу аварийно завершить/записать nan/выбросить ошибку
            return np.nan  # или 0, или выбрасывай Exception
        return beta / (1 - alpha) * (p0 / (a ** (1 / alpha) * x1)) * x3 ** ((1 - alpha) / alpha)
    def B(z2, z3):
        return -z2 - alpha*z3

    # Уравнения:
    x1_dot = b/alpha * x1 - 1/M0 * x2
    x2_dot = x2 * (A(x1, x3) + 1/B(z2, z3) - 1/z2 + x3 - (alpha*beta - b)/(alpha*(1-alpha)) - mu)
    x3_dot = alpha * x3 * (A(x1, x3) + 1/B(z2, z3) - (alpha*beta-b)/(alpha*(1-alpha)))
    z1_dot = rho*z1 - 1/M0*x2/x1*z1 - A(x1, x3)*B(z2, z3)
    z2_dot = rho*z2 + 1/M0*x2/x1*z1 - 1
    z3_dot = rho*z3 - x3*z2 + (1-alpha)/alpha*A(x1, x3)*B(z2, z3) + 1/alpha
    return np.array([x1_dot, x2_dot, x3_dot, z1_dot, z2_dot, z3_dot])
