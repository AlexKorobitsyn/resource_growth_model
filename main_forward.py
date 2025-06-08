import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go

# Параметры модели (пример)
alpha, beta, gamma = 0.5, 0.4, 0.3

# Примерная система — абстрактная динамика капитала, труда и технологии
def growth_model(t, y):
    x, y_, z = y
    dxdt = alpha * (y_ - x)  # замедляющееся притяжение к y_
    dydt = beta * (z - y_)   # замедляющееся притяжение к z
    dzdt = -gamma * (z - 5)  # z стягивается к 5
    return [dxdt, dydt, dzdt]

# Находим стационарную точку (простая аналитика)
# При dx/dt=0, dy/dt=0, dz/dt=0:
# x = y = z = 5
x_star = y_star = z_star = 5.0
stationary = np.array([x_star, y_star, z_star])

# Создаём множество начальных условий
n_traj = 30
np.random.seed(42)
initial_points = np.random.uniform(0.5, 10, size=(n_traj, 3))

# Время интегрирования
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 150)

# Решения для каждой траектории
trajectories = []
for y0 in initial_points:
    sol = solve_ivp(growth_model, t_span, y0, t_eval=t_eval)
    trajectories.append(sol.y)

# Визуализация в plotly
fig = go.Figure()

# Добавим траектории
for traj in trajectories:
    fig.add_trace(go.Scatter3d(
        x=traj[0], y=traj[1], z=traj[2],
        mode='lines',
        line=dict(width=2),
        opacity=0.7
    ))
    # Можно добавить стрелку направления
    fig.add_trace(go.Scatter3d(
        x=[traj[0][-1]],
        y= [traj[1][-1]],
        z=[ traj[2][-1]],
        mode='markers',
        marker=dict(size=3, color='red', symbol='diamond')
    ))

# Отметим стационарную точку
fig.add_trace(go.Scatter3d(
    x=[x_star], y=[y_star], z=[z_star],
    mode='markers+text',
    marker=dict(size=8, color='green'),
    text=["Стационарная точка"],
    textposition="top center"
))

fig.update_layout(
    title="Фазовая диаграмма 3D: Модель экономического роста",
    scene=dict(
        xaxis_title="Капитал (x)",
        yaxis_title="Труд (y)",
        zaxis_title="Технология (z)"
    ),
    width=900, height=700
)
fig.show()
