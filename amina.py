import numpy as np
from scipy.integrate import solve_ivp

from model_params import params
from stationary_point import stationary_point
from stabilized_system import numerical_jacobian, get_stable_subspace, build_H11_H12, stabilized_rhs
from hamiltonian_system import hamiltonian_rhs
from бэки.main_forward import x_star

alpha = params['alpha']
beta = params['beta']
b = params['b']
mu = params['mu']
rho = params['rho']

u_c = b / (alpha * beta)
v_c = ((1 - alpha) * mu * rho * (alpha * (beta - rho) - b)) / (
    (mu + rho) * (alpha * (1 - alpha) * rho * (beta - rho) + b * (alpha * beta - b))
    - (1 - alpha) * b * mu * rho
)

u_const = params.get('u_const', u_c)
v_const = params.get('v_const', v_c)

def hamiltonian_control1(t, Y, p):
    p1 = p.copy()
    p1['b'], p1['v'] = u_c, v_c
    return hamiltonian_rhs(t, Y, p1)

def hamiltonian_control2(t, Y, p):
    p2 = p.copy()
    p2['b'], p2['v'] = u_const, v_const
    return hamiltonian_rhs(t, Y, p2)


def compute_stationary_and_manifold(params):
    x_star, z_star = stationary_point(params)
    Y_star = np.concatenate([x_star, z_star])
    J = numerical_jacobian(hamiltonian_rhs, Y_star, params)
    eigvals, eigvecs, stable_idx, _ = get_stable_subspace(J)
    H11, H12 = build_H11_H12(eigvecs, stable_idx)
    return x_star, z_star, eigvals, H11, H12


def integrate_stabilized(x_star, z_star, H11, H12, params, t_final=10.0, eps=1e-4):
    x0 = x_star + eps * H11[:, 2]
    z0 = z_star + H12 @ np.linalg.inv(H11) @ (x0 - x_star)
    sol = solve_ivp(
        lambda t, X: stabilized_rhs(t, X, x_star, z_star, H11, H12, params),
        [0, t_final], x0, max_step=0.01
    )
    return sol.y[:, -1]


def sample_points_in_sphere(center, radius, n_points):
    pts = []
    for _ in range(n_points):
        u = np.random.normal(size=3)
        u /= np.linalg.norm(u)
        r = np.random.rand() ** (1/3) * radius
        pts.append(center + u * r)
    return pts


def compute_T0(eigvals, params, radius, x_star3, gamma=None):
    nu = max(np.real(eigvals))
    delta = params['b'] / (1 - params['alpha'])
    eps2 = radius ** 2
    gamma_val = gamma or params.get('gamma', 1.0)
    L3 = params['v'] * x_star3
    theta = L3 / (gamma_val * delta)
    return (1.0 / (nu - delta)) * np.log(theta / eps2)


def boundary_event_factory(boundary_func):
    def event(t, Y): return boundary_func(Y)
    event.terminal = True; event.direction = 0
    return event


def target_event_factory(x0, eps):
    def event(t, Y): return np.linalg.norm(Y[:3] - x0) - eps
    event.terminal = True; event.direction = 0
    return event


def integrate_piecewise_hamiltonian(Y0, params, T0, regime_switches, eps_target=1e-4):
    t_start, t_end = 0.0, -T0
    solutions, metrics = [], []
    reached_target = False
    x0_target = Y0[:3]

    for i, (boundary_func, rhs_func) in enumerate(regime_switches):
        events = [target_event_factory(x0_target, eps_target), boundary_event_factory(boundary_func)]
        sol = solve_ivp(lambda t, y: rhs_func(t, y, params), [t_start, t_end], Y0,
                        method='RK45', max_step=0.1, events=events)
        solutions.append(sol)
        if sol.t_events[0].size:
            etype, etime = 'target', sol.t_events[0][0]; reached_target = True
        elif sol.t_events[1].size:
            etype, etime = 'boundary', sol.t_events[1][0][0]
        else:
            etype, etime = 'none', sol.t[-1]
        metrics.append({'regime': i, 't_start': t_start, 't_end': sol.t[-1],
                        'event_type': etype, 'event_time': etime})
        if etype == 'boundary':
            Y0, t_start = sol.y_events[1][0], sol.t[-1]
            continue
        break
    return solutions, reached_target, metrics


def build_optimal_trajectories(params, radius=0.01, n_traj=8,
                                regime_switches=None, gamma=None, eps_target=1e-4):
    x_star, z_star, eigvals, H11, H12 = compute_stationary_and_manifold(params)
    h1, h2, h3 = H11[:,0], H11[:,1], H11[:,2]
    P = integrate_stabilized(x_star, z_star, H11, H12, params)
    pts = sample_points_in_sphere(P, radius, n_traj)
    T0 = compute_T0(eigvals, params, radius, x_star[2], gamma)

    results = []
    for x0 in pts:
        z0 = z_star + H12 @ np.linalg.inv(H11) @ (x0 - x_star)
        Y0 = np.concatenate([x0, z0])
        sols, reached, metrics = integrate_piecewise_hamiltonian(
            Y0, params, T0,
            regime_switches or [
                (lambda Y: np.linalg.norm(Y[:3]-x_star)-params.get('r_boundary',0.05), hamiltonian_control1),
                (lambda Y: np.linalg.norm(Y[:3]-x_star)-params.get('r_boundary2',0.02), hamiltonian_control2)
            ], eps_target)
        if reached:
            results.append({'Y0': Y0, 'solutions': sols, 'metrics': metrics,
                            'x_star': x_star, 'h1': h1, 'h2': h2, 'h3': h3})
    return results

if __name__ == '__main__':
    regime_switches = [
        (lambda Y: np.linalg.norm(Y[:3]-x_star)-params.get('r_boundary',0.05), hamiltonian_control1),
        (lambda Y: np.linalg.norm(Y[:3]-x_star)-params.get('r_boundary2',0.02), hamiltonian_control2)
    ]
    good_trajs = build_optimal_trajectories(
        params, radius=0.05*np.linalg.norm(x_star), n_traj=12,
        regime_switches=regime_switches, gamma=params.get('gamma',1.0), eps_target=1e-3)

    import matplotlib.pyplot as plt
    import mpl_toolkits

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for res in good_trajs:
        for sol in res['solutions']:
            Y = sol.y
            ax.plot(Y[0], Y[1], Y[2])
    if good_trajs:
        x_star = good_trajs[0]['x_star']
        for name, vec in [('h1', good_trajs[0]['h1']), ('h2', good_trajs[0]['h2']), ('h3', good_trajs[0]['h3'])]:
            ax.quiver(x_star[0], x_star[1], x_star[2],
                      vec[0], vec[1], vec[2], length=0.05, normalize=True)
            ax.text(x_star[0] + vec[0]*0.06,
                    x_star[1] + vec[1]*0.06,
                    x_star[2] + vec[2]*0.06,
                    name)
    ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('x3')
    plt.title('Optimal Hamiltonian Trajectories with h-vectors')
    plt.show()
