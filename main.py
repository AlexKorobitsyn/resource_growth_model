# main.py
from model_params import params
from stationary_point import stationary_point
from hamiltonian_system import hamiltonian_rhs
from visualization import plot_3d_trajectory
import numpy as np
from scipy.integrate import solve_ivp

def valid_initial_point(Y):
    x1, x2, x3, z1, z2, z3 = Y
    return (x1 > 0) and (x2 > 0) and (x3 > 0)

def sample_points_around(x_star, z_star, r, n_points=10):
    # Генерируем n_points случайных точек в шаре радиуса r вокруг x_star, z_star
    points = []
    for _ in range(n_points):
        v_x = np.random.randn(3)
        v_x /= np.linalg.norm(v_x)
        x0 = x_star + r * v_x

        v_z = np.random.randn(3)
        v_z /= np.linalg.norm(v_z)
        z0 = z_star + r * v_z

        y0 = np.concatenate([x0, z0])
        print(y0)
        points.append(y0)
    return points

if __name__ == "__main__":
    # 1. Стационарная точка
    x_star, z_star = stationary_point(params)
    print("Стационарная точка x:", x_star)
    print("Стационарная точка z:", z_star)

    # 2. Настройки шара
    radius = 0.35 * np.linalg.norm(x_star)  # 5% от x_star
    n_traj = 8  # Сколько траекторий запускать

    # 3. Подбираем стартовые точки
    start_points = sample_points_around(x_star, z_star, radius, n_points=n_traj)

    # 4. Интегрируем каждую в обратном времени
    T_max = 30
    all_xs, all_ys, all_zs = [], [], []
    for i, Y0 in enumerate(start_points):
        if not valid_initial_point(Y0):
            print(f"Skipping initial point {i}: x1,x2,x3 must be > 0, got x1={Y0[0]}, x2={Y0[1]}, x3={Y0[2]}")
            continue
        sol = solve_ivp(
            lambda t, y: hamiltonian_rhs(t, y, params),
            [0, -T_max], Y0,
            method="RK45",
            max_step=0.1
        )
        xs, ys, zs = sol.y[0], sol.y[1], sol.y[2]
        all_xs.append(xs)
        all_ys.append(ys)
        all_zs.append(zs)

    # 5. Визуализация всех траекторий в одном 3D-портрете
    import plotly.graph_objects as go

    fig = go.Figure()
    for xs, ys, zs in zip(all_xs, all_ys, all_zs):
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines'))
    # Добавим стационарную точку
    fig.add_trace(go.Scatter3d(x=[x_star[0]], y=[x_star[1]], z=[x_star[2]], mode='markers', marker=dict(size=8, color='red'), name='Stationary Point'))
    fig.update_layout(scene=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='x3'), width=700, margin=dict(r=20, b=10, l=10, t=10))
    fig.show()
