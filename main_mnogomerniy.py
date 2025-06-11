from model_params import params
from stationary_point import stationary_point
from hamiltonian_system import hamiltonian_rhs
from stabilized_system import numerical_jacobian, get_stable_subspace, build_H11_H12
from visualization import plot_3d_trajectory
import numpy as np
from scipy.integrate import solve_ivp

def fibonacci_sphere_points(n, radius=1.0, center=None):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2
        radius_xy = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius_xy
        z = np.sin(theta) * radius_xy
        pt = np.array([x, y, z]) * radius
        if center is not None:
            pt = center + pt
        points.append(pt)
    points.append(x_star)
    return np.array(points)



def valid_initial_point(X0):
    return np.all(X0 > 0)

if __name__ == "__main__":
    x_star, z_star = stationary_point(params)
    Y_star = np.concatenate([x_star, z_star])
    print("Стационарная точка x*:", x_star)
    print("Стационарная точка z*:", z_star)

    J = numerical_jacobian(hamiltonian_rhs, Y_star, params)
    eigvals, eigvecs, stable_idx, _ = get_stable_subspace(J)
    H11, H12 = build_H11_H12(eigvecs, stable_idx)
    print("H11:\n", H11)
    print("H12:\n", H12)

    n_traj = 30
    sphere_radius = 0.75 * np.linalg.norm(x_star)  # можно уменьшить или увеличить
    points_x = fibonacci_sphere_points(n_traj, radius=sphere_radius, center=x_star)

    start_points = []
    for pt in points_x:
        if not valid_initial_point(pt):
            continue
        # z0 = z_star + H12 @ inv(H11) @ (pt - x_star)
        H11_reg = H11 + np.eye(3) * 1e-10
        z0 = z_star + H12 @ np.linalg.pinv(H11_reg) @ (pt - x_star)
        Y0 = np.concatenate([pt, z0])
        start_points.append(Y0)

    print(f"Будет построено {len(start_points)} траекторий")

    T_max = 30


    def limit_event(t, y):
        x = y[:3]
        return min(3 - x.max(), x.min())


    limit_event.terminal = True
    limit_event.direction = 0

    all_xs, all_ys, all_zs = [], [], []
    starts, ends, directions = [], [], []

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
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        dz = zs[1] - zs[0]
        directions.append([dx, dy, dz])
        starts.append([xs[0], ys[0], zs[0]])
        ends.append([xs[-1], ys[-1], zs[-1]])

    import plotly.graph_objects as go
    maxDx = max(directions[:][0])
    maxDy = max(directions[:][1])
    maxDz = max(directions[:][2])
    directions1 = []
    normMax = np.linalg.norm([maxDx, maxDy, maxDz])
    for dx, dy, dz in directions:
        norm1 = np.linalg.norm([dx, dy, dz])
        norm = normMax/norm1
        if norm == 0:
            directions1.append([0, 0, 0])
        else:
            directions1.append([dx*norm, dy* norm, dz* norm])
    ends = np.array(ends)
    ends_clipped = np.clip(ends, -15, 15)

    fig = go.Figure()
    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
    X = x_star[0] + sphere_radius * np.cos(u) * np.sin(v)
    Y = x_star[1] + sphere_radius * np.sin(u) * np.sin(v)
    Z = x_star[2] + sphere_radius * np.cos(v)
    i=0
    print("stable_idx", stable_idx)
    for vec in eigvecs.T:
        i+=1
        vec = np.real(vec)
        if i-1 in stable_idx:
            fig.add_trace(go.Scatter3d(
                x=[x_star[0] , x_star[0] + vec[0]],
                y=[x_star[1] , x_star[1] + vec[1]],
                z=[x_star[2] , x_star[2] + vec[2]],
                mode='lines',

                line=dict(dash='dot', color='red'),
                name=f'Отрицательный собственный вектор №{i}'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[x_star[0] , x_star[0] + vec[0]],
                y=[x_star[1] , x_star[1] + vec[1]],
                z=[x_star[2] , x_star[2] + vec[2]],
                mode='lines',

                line=dict(dash='dot', color='blue'),
                name=f'Положительный собственный вектор №{i}'
            ))

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, opacity=0.15, showscale=False, colorscale=[[0, "lightblue"], [1, "lightblue"]],
        name="Initial sphere"
    ))
    i = 0
    custom_colorscale = [[0, 'blue'], [0.5, 'green'], [1, 'red']]
    for xs, ys, zs in zip(all_xs, all_ys, all_zs):
        i+=1
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', name=f'Траектория №{i}', legendgroup=f'traj{i}', marker=dict(color=[0, 0.5, 1], colorscale=custom_colorscale)))

    fig.add_trace(go.Scatter3d(
        x=[x_star[0]], y=[x_star[1]], z=[x_star[2]],
        mode='markers', marker=dict(size=10, color='red'), name='Stationary Point'
    ))
    Hs = np.vstack([H11, H12])
    Hs = Hs / np.linalg.norm(Hs, axis=0)


    def project_to_stable_subspace(delta):
        return Hs @ (Hs.T @ delta)




    starts = np.array(starts)
    fig.add_trace(go.Scatter3d(
        x=starts[:, 0], y=starts[:, 1], z=starts[:, 2],
        mode='markers', marker=dict(size=6, color='blue', symbol='circle'), name='Start'
    ))


    ends = np.array(ends)
    fig.add_trace(go.Scatter3d(
        x=ends_clipped[:, 0], y=ends_clipped[:, 1], z=ends_clipped[:, 2],
        mode='markers', marker=dict(size=6, color='green', symbol='diamond'), name='End'
    ))
    norms = [np.linalg.norm(vec) for vec in directions]
    max_norm = max(norms)
    sizerefs = [0.1 * (max_norm / n) if n > 1e-8 else 0.1 for n in norms]
    directions1 = np.array(directions)
    # fig.add_trace(go.Scatter3d(
    #     x=[starts[:, 0], -directions1[:, 0]],
    #     y=[starts[:, 1], -directions1[:, 1]],
    #     z=[starts[:, 2], -directions1[:, 2]],
    #     mode='lines+markers',
    #     marker=dict(size=2, color='black'),
    #     line=dict(width=5, color='orange'),
    #     name='Vector' if i == 0 else None,  # Чтобы в легенде было только один раз
    #     showlegend=(i == 0)
    # ))
    for i in range(len(starts)):
        fig.add_trace(go.Cone(
            x=[starts[i][0]], y=[starts[i][1]], z=[starts[i][2]],
            u=[-directions[i][0]], v=[-directions[i][1]], w=[-directions[i][2]],
            sizemode="scaled", sizeref=sizerefs[i],
            anchor='tail', showscale=False, colorscale='Viridis',
            name='Direction' if i == 0 else None,
            legendgroup=f'traj{i+1}',
            showlegend=(i == 0)
        ))


    fig.update_layout(
        scene=dict(
            xaxis_title='x1', yaxis_title='x2', zaxis_title='x3'
        ),
        width=900, margin=dict(r=20, b=10, l=10, t=10)
    )
    fig.show()
