# visualization.py
import plotly.graph_objects as go
import numpy as np

def plot_3d_trajectory(xs, ys, zs, x_star=None, sphere_radius=0.05):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', name='Trajectory'))
    if x_star is not None:
        fig.add_trace(go.Scatter3d(x=[x_star[0]], y=[x_star[1]], z=[x_star[2]], mode='markers',
                                   marker=dict(size=8, color='red'), name='Stationary Point'))
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        X = x_star[0] + sphere_radius * np.cos(u) * np.sin(v)
        Y = x_star[1] + sphere_radius * np.sin(u) * np.sin(v)
        Z = x_star[2] + sphere_radius * np.cos(v)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.2, showscale=False))
    fig.update_layout(scene=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='x3'),
                      width=700, margin=dict(r=20, b=10, l=10, t=10))
    fig.show()
