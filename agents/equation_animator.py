"""
agents/equation_animator.py
------------------------------
Renders short animated GIFs of equations for social posts or teaching use.
Pure matplotlib -- no internet or GPU required, so this runs anywhere.

Two modes:
  - "directional_cosine": a pre-built animation grounded in the actual
    motion descriptor math in models/motion_engine.py -- a deformation
    vector phi rotating against a fixed reference r, with the resulting
    alpha = cos(theta) traced live alongside.
  - "generic": animate any sympy-parseable f(x, t) as t sweeps a range.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sympy as sp


def animate_generic_equation(expr_str: str, var: str, t_var: str, t_range, x_range,
                              output_path: str, title: str = "", fps: int = 15) -> str:
    """
    expr_str: sympy expression string in terms of `var` and `t_var`, e.g. "sin(x - t)"
    t_range: (start, end, num_frames)
    x_range: (start, end, num_points)
    """
    x_sym, t_sym = sp.symbols(f"{var} {t_var}")
    expr = sp.sympify(expr_str)
    f = sp.lambdify((x_sym, t_sym), expr, "numpy")

    xs = np.linspace(*x_range)
    t_start, t_end, n_frames = t_range
    ts = np.linspace(t_start, t_end, n_frames)

    fig, ax = plt.subplots(figsize=(6, 4))
    line, = ax.plot([], [], lw=2, color="crimson")
    ax.set_xlim(xs.min(), xs.max())

    y_all = np.array([np.broadcast_to(f(xs, t), xs.shape) for t in ts], dtype=float)
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
    pad = 0.1 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_title(title or f"${sp.latex(expr)}$")
    ax.set_xlabel(var)
    ax.grid(alpha=0.3)

    def update(frame):
        y = np.broadcast_to(f(xs, ts[frame]), xs.shape)
        line.set_data(xs, y)
        ax.set_ylabel(f"{t_var} = {ts[frame]:.2f}")
        return (line,)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    return output_path


def animate_directional_cosine(output_path: str, fps: int = 15, n_frames: int = 60) -> str:
    """
    Grounded in models/motion_engine.py: shows a deformation vector phi
    rotating relative to a fixed radial reference vector r, with the
    directional cosine similarity alpha = cos(theta) plotted live alongside.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ref = np.array([1.0, 0.0])
    thetas = np.linspace(0, 2 * np.pi, n_frames)
    alphas = np.cos(thetas)

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect("equal")
    ax1.set_title(r"Deformation vector $\phi$ vs. reference $r$")
    ax1.arrow(0, 0, ref[0], ref[1], color="gray", width=0.02, length_includes_head=True)
    ax1.text(1.05, 0.05, "r", color="gray")

    state = {"arrow": None}

    ax2.set_xlim(0, 2 * np.pi)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_title(r"$\alpha = \cos\theta$")
    ax2.set_xlabel(r"$\theta$")
    ax2.grid(alpha=0.3)
    alpha_line, = ax2.plot([], [], color="crimson", lw=2)

    def update(frame):
        theta = thetas[frame]
        phi = np.array([np.cos(theta), np.sin(theta)])
        if state["arrow"] is not None:
            state["arrow"].remove()
        state["arrow"] = ax1.arrow(
            0, 0, phi[0], phi[1], color="crimson", width=0.02, length_includes_head=True
        )
        alpha_line.set_data(thetas[: frame + 1], alphas[: frame + 1])
        return (state["arrow"], alpha_line)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    return output_path