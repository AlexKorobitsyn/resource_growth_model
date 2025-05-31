# stabilized_system.py
import numpy as np

def numerical_jacobian(rhs, Y_star, params, eps=1e-6):
    """Вычисляет якобиан функции rhs(Y) по Y в точке Y_star численно."""
    n = Y_star.size
    J = np.zeros((n, n))
    f0 = rhs(0, Y_star, params)
    for i in range(n):
        dY = np.zeros(n)
        dY[i] = eps
        f1 = rhs(0, Y_star + dY, params)
        J[:, i] = (f1 - f0) / eps
    return J

def get_stable_subspace(J):
    """Возвращает собственные векторы, соответствующие отрицательным Re(λ)."""
    eigvals, eigvecs = np.linalg.eig(J)
    idx = np.argsort(np.real(eigvals))  # От меньших к большим
    stable_idx = idx[np.real(eigvals[idx]) < 0]
    unstable_idx = idx[np.real(eigvals[idx]) > 0]
    return eigvals, eigvecs, stable_idx, unstable_idx

def build_H11_H12(eigvecs, stable_idx):
    """H11 (3x3), H12 (3x3) — по фазовым/сопряжённым переменным"""
    H1 = np.real(eigvecs[:, stable_idx[:3]])  # Всего 6x3, берём первые три по отрицательным λ
    H11 = H1[:3, :]  # фазовые
    H12 = H1[3:, :]  # сопряжённые
    return H11, H12


def stabilized_rhs(t, X, x_star, z_star, H11, H12, params):
    x_delta = X - x_star
    z_delta = H12 @ np.linalg.inv(H11) @ x_delta
    z = z_star + z_delta
    Y = np.concatenate([X, z])
    from hamiltonian_system import hamiltonian_rhs
    return hamiltonian_rhs(t, Y, params)[:3]  # Только dx/dt
