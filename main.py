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



def fibonacci_sphere_points(center, r, n_points=10):
    points = []
    offset = 2.0 / n_points
    increment = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n_points):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y * y)
        phi = i * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        point = center + radius * np.array([x, y, z])
        points.append(point)
    points.append(x_star)
    return points

if __name__ == "__main__":
    x_star, z_star = stationary_point(params)
    print("Стационарная точка x:", x_star)
    print("Стационарная точка z:", z_star)

    radius = 0.01 * np.linalg.norm(x_star)  # 1% от x_star
    n_traj = 8  # Сколько траекторий запускать

    min_xstar = min(x_star) * 0.2


    def trim_trajectory(xs, ys, zs):
        for k in range(len(xs)):
            if xs[k] < min_xstar or ys[k] < min_xstar or zs[k] < min_xstar:
                return xs[:k], ys[:k], zs[:k]
        return xs, ys, zs
    points_x = fibonacci_sphere_points(x_star, radius, n_traj)
    print(points_x)
    start_points = [np.concatenate([pt, z_star]) for pt in points_x]
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
        xs, ys, zs = trim_trajectory(xs, ys, zs)
        all_xs.append(xs)
        all_ys.append(ys)
        all_zs.append(zs)


    import plotly.graph_objects as go

    fig = go.Figure()
    from stabilized_system import numerical_jacobian

    Y_star = np.concatenate([x_star, z_star])
    J = numerical_jacobian(hamiltonian_rhs, Y_star, params)
    eigvals, eigvecs = np.linalg.eig(J)
    print("eigvecs:", eigvecs)
    idx_stable = np.argsort(np.real(eigvals))[:3]
    h_vectors = [eigvecs[:3, i].real for i in idx_stable]

    arrow_scale = radius * 2.5
    for i, h in enumerate(h_vectors):
        h = h / np.linalg.norm(h)
        x_end = x_star + arrow_scale * h
        fig.add_trace(go.Scatter3d(
            x=[x_star[0], x_end[0]],
            y=[x_star[1], x_end[1]],
            z=[x_star[2], x_end[2]],
            mode='lines+text',
            line=dict(width=4, color=['green', 'blue', 'purple'][i], dash='dash'),
            text=[None, f"h{i + 1}"],
            textposition="top right",
            name=f"h{i + 1}",
            legendgroup=f"h{i + 1}",
            showlegend=True
        ))
    # for xs, ys, zs in zip(all_xs, all_ys, all_zs):
    #     fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines'))
    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
    X = x_star[0] + radius * np.cos(u) * np.sin(v)
    Y = x_star[1] + radius * np.sin(u) * np.sin(v)
    Z = x_star[2] + radius * np.cos(v)


    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, opacity=0.15, showscale=False, colorscale=[[0, "lightblue"], [1, "lightblue"]],
        name="Initial sphere"
    ))
    for idx, (xs, ys, zs) in enumerate(zip(all_xs, all_ys, all_zs)):
        name = f'Траектория #{idx + 1}'
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            name=name,
            legendgroup=name,
            showlegend=True
        ))
        fig.add_trace(go.Scatter3d(
            x=[xs[0]], y=[ys[0]], z=[zs[0]],
            mode='markers',
            marker=dict(size=2, color='orange', symbol='diamond'),
            name=name,
            legendgroup=name,
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[xs[-1]], y=[ys[-1]], z=[zs[-1]],
            mode='markers',
            marker=dict(size=1, color='black', symbol='circle'),
            name=name,
            legendgroup=name,
            showlegend=False
        ))
    fig.add_trace(go.Scatter3d(x=[x_star[0]], y=[x_star[1]], z=[x_star[2]], mode='markers', marker=dict(size=8, color='red'), name='Stationary Point'))
    fig.update_layout(scene=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='x3'), width=700, margin=dict(r=20, b=10, l=10, t=10))
    fig.show()
