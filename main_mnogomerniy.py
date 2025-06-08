from model_params import params
from stationary_point import stationary_point
from hamiltonian_system import hamiltonian_rhs
from stabilized_system import numerical_jacobian, get_stable_subspace, build_H11_H12
from visualization import plot_3d_trajectory
import numpy as np
from scipy.integrate import solve_ivp

def fibonacci_sphere_points(n, radius=1.0, center=None):
    # Генерация равномерных точек на сфере, вокруг center (обычно x_star)
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius_xy = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius_xy
        z = np.sin(theta) * radius_xy
        pt = np.array([x, y, z]) * radius
        if center is not None:
            pt = center + pt
        points.append(pt)
    return np.array(points)



def valid_initial_point(X0):
    return np.all(X0 > 0)

if __name__ == "__main__":
    # 1. Стационарная точка и полный вектор Y*
    x_star, z_star = stationary_point(params)
    Y_star = np.concatenate([x_star, z_star])
    print("Стационарная точка x*:", x_star)
    print("Стационарная точка z*:", z_star)

    # 2. Якобиан и собственные векторы в стационарной точке
    J = numerical_jacobian(hamiltonian_rhs, Y_star, params)
    eigvals, eigvecs, stable_idx, _ = get_stable_subspace(J)
    H11, H12 = build_H11_H12(eigvecs, stable_idx)
    print("H11:\n", H11)
    print("H12:\n", H12)

    # 3. Генерируем N точек на сфере вокруг x*
    n_traj = 16
    sphere_radius = 0.14 * np.linalg.norm(x_star)  # можешь уменьшить или увеличить для плотности
    points_x = fibonacci_sphere_points(n_traj, radius=sphere_radius, center=x_star)

    # 4. Для каждой точки на сфере вычисляем индивидуальный z₀ и стартуем свою траекторию
    start_points = []
    for pt in points_x:
        if not valid_initial_point(pt):
            continue  # фильтруем некорректные точки, если вышли в x < 0
        # z0 = z_star + H12 @ inv(H11) @ (pt - x_star)
        H11_reg = H11 + np.eye(3) * 1e-10  # регуляризация на случай вырождения
        z0 = z_star + H12 @ np.linalg.pinv(H11_reg) @ (pt - x_star)
        Y0 = np.concatenate([pt, z0])
        start_points.append(Y0)

    print(f"Будет построено {len(start_points)} траекторий")

    # 5. Интегрируем систему для каждой точки (вперёд по времени)
    T_max = 30


    def limit_event(t, y):
        x = y[:3]
        return min(3 - x.max(), x.min())


    limit_event.terminal = True
    limit_event.direction = 0

    all_xs, all_ys, all_zs = [], [], []
    starts, ends = [], []

    for Y0 in start_points:
        x1, x2, x3 = Y0[:3]
        if not (x1 > 0 and x2 > 0 and x3 > 0):
            print(f"Пропуск: стартовая x0 не вся > 0: {Y0[:3]}")
            continue
        sol = solve_ivp(
            lambda t, y: hamiltonian_rhs(t, y, params),
            [0, T_max], Y0,
            max_step=0.1,
            rtol=1e-6, atol=1e-8,
            events=limit_event
        )
        xs, ys, zs = sol.y[0], sol.y[1], sol.y[2]
        all_xs.append(xs)
        all_ys.append(ys)
        all_zs.append(zs)
        starts.append([xs[0], ys[0], zs[0]])
        ends.append([xs[-1], ys[-1], zs[-1]])

    # 6. Визуализация с траекториями и стрелочками
    import plotly.graph_objects as go

    # Ограничиваем конечные точки, чтобы не выходили за пределы ±15 по любой оси
    ends = np.array(ends)
    ends_clipped = np.clip(ends, -15, 15)
    directions = ends_clipped - starts  # стрелки тоже пересчитываем!

    fig = go.Figure()
    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
    X = x_star[0] + sphere_radius * np.cos(u) * np.sin(v)
    Y = x_star[1] + sphere_radius * np.sin(u) * np.sin(v)
    Z = x_star[2] + sphere_radius * np.cos(v)

    # Сама сфера (прозрачная, только граница)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, opacity=0.15, showscale=False, colorscale=[[0, "lightblue"], [1, "lightblue"]],
        name="Initial sphere"
    ))
    # Фазовые траектории
    for xs, ys, zs in zip(all_xs, all_ys, all_zs):
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines'))

    # Стационарная точка (красная)
    fig.add_trace(go.Scatter3d(
        x=[x_star[0]], y=[x_star[1]], z=[x_star[2]],
        mode='markers', marker=dict(size=10, color='red'), name='Stationary Point'
    ))

    # Начальные точки (синие)
    starts = np.array(starts)
    fig.add_trace(go.Scatter3d(
        x=starts[:, 0], y=starts[:, 1], z=starts[:, 2],
        mode='markers', marker=dict(size=6, color='blue', symbol='circle'), name='Start'
    ))

    # Конечные точки (зелёные)
    ends = np.array(ends)
    # Конечные точки (зелёные)
    fig.add_trace(go.Scatter3d(
        x=ends_clipped[:, 0], y=ends_clipped[:, 1], z=ends_clipped[:, 2],
        mode='markers', marker=dict(size=6, color='green', symbol='diamond'), name='End'
    ))

    # Стрелки направления (cones)
    fig.add_trace(go.Cone(
        x=starts[:, 0], y=starts[:, 1], z=starts[:, 2],
        u=directions[:, 0], v=directions[:, 1], w=directions[:, 2],
        colorscale='Viridis', sizemode="absolute", sizeref=0.3,
        showscale=False, anchor='tail', name='Direction'
    ))

    # Рисуем прозрачную сферу (поверхность, на которой лежат начальные точки)
    # Центр = x_star, радиус = sphere_radius

    fig.update_layout(
        scene=dict(
            xaxis_title='x1', yaxis_title='x2', zaxis_title='x3'
        ),
        width=900, margin=dict(r=20, b=10, l=10, t=10)
    )
    fig.show()
