import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from memory_profiler import profile


# general plotting styles-related settings
plt.style.use("seaborn-ticks")
rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "font.size": 18,
        "axes.linewidth": 2.0,
        "lines.dashed_pattern": (5, 2.5),
    }
)


# only for Rosenbrock
x_range = [-2, 2]
y_range = [-1, 3]


# PSO parameters
n_iterations = 75
n_particles = 100
learning_rate = 0.002
inertia_coeff = 1
local_coeff = 1
global_coeff = 1


def objective_function(x, y, name="rosenbrock"):
    """Return value of the objective function at coordinates x1, x2."""
    if name == "rosenbrock":
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    else:
        raise NotImplementedError


def global_best_position(x, y):
    """Find the overall-swarm best position."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    g_xyt = objective_function(x_flat, y_flat)
    global_best = np.argmin(g_xyt)
    return x_flat[global_best], y_flat[global_best]


def particle_best_position(x, y):
    """Find the historical best position for each particle."""
    f_xy = objective_function(x, y)
    particle_best = np.argmin(f_xy, axis=1)
    return (
        x[range(len(particle_best)), particle_best],
        y[range(len(particle_best)), particle_best],
    )


def plot_rosenbrock_contour(ax=None):
    """Plot contours of the Rosenbrock function."""
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
    X = np.linspace(x_range[0], x_range[1], num=30)
    Y = np.linspace(y_range[0], y_range[1], num=30)
    X, Y = np.meshgrid(X, Y)
    Z = objective_function(X, Y)
    ax.contour(X, Y, Z, levels=200, cmap="coolwarm")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("PSO on Rosenbrock")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    return ax


def plot_particles(x, y, ax=None):
    """Plot each particle in the swarm."""
    ax.plot(x, y, "o", color="xkcd:tomato", ms=6, mec="w", mew=1.0)


@profile
def run_pso():
    """Run a PSO simulation."""
    rng = np.random.default_rng()

    # Initialize particle positions as a function of time
    x = np.zeros((n_particles, n_iterations + 1))
    y = np.zeros((n_particles, n_iterations + 1))
    x[:, 0] = rng.random(size=n_particles)
    x[:, 0] = x[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    y[:, 0] = rng.random(size=n_particles)
    y[:, 0] = y[:, 0] * (y_range[1] - y_range[0]) + y_range[0]

    # Initialize particle velocities as a function of time
    v_x = np.zeros((n_particles, n_iterations + 1))
    v_y = np.zeros((n_particles, n_iterations + 1))

    particle_best_history = []
    global_best_history = []

    for it in range(n_iterations):
        global_best = global_best_position(x, y)
        particle_best = particle_best_position(x, y)
        global_best_history.append(objective_function(*global_best))
        particle_best_history.append(np.mean(objective_function(*particle_best)))
        v_x[:, it + 1] = (
            inertia_coeff * v_x[:, it]
            + local_coeff * rng.random() * (particle_best[0] - x[:, it])
            + global_coeff * rng.random() * (global_best[0] - x[:, it])
        )
        v_y[:, it + 1] = (
            inertia_coeff * v_y[:, it]
            + local_coeff * rng.random() * (particle_best[1] - y[:, it])
            + global_coeff * rng.random() * (global_best[1] - y[:, it])
        )
        x[:, it + 1] = x[:, it] + learning_rate * v_x[:, it + 1]
        y[:, it + 1] = y[:, it] + learning_rate * v_y[:, it + 1]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        plot_rosenbrock_contour(ax=ax)
        plot_particles(x[:, it + 1], y[:, it + 1], ax=ax)
        figname = f"frame-{it:02d}.png"
        plt.savefig(figname, bbox_inches="tight", dpi=100)
        plt.close()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(
        range(len(global_best_history)),
        global_best_history,
        "o-",
        color="xkcd:tomato",
        mec="w",
        mew=1.0,
    )
    ax.plot(
        range(len(particle_best_history)),
        particle_best_history,
        "o-",
        color="xkcd:turquoise",
        mec="w",
        mew=1.0,
    )
    ax.text(35, 0.05, "Global-best value", color="xkcd:tomato")
    ax.text(25, 0.35, "Mean particle-best value", color="xkcd:turquoise")
    ax.set_xlabel("PSO step")
    ax.set_ylabel("Best value of function found")
    figname = "best_vs_iteration.png"
    plt.savefig(figname, bbox_inches="tight", dpi=100)

    print(global_best_position(x, y))


if __name__ == "__main__":
    run_pso()
