# from memory_profiler import profile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from pymatgen import Composition
from matminer.featurizers.conversions import StrToComposition

from discworld.datasets import DATASETS


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


# PSO parameters
# number of particles in the swarm
N_PARTICLES = 30
# learning rate
LEARNING_RATE = 0.5
# inertia coefficient
INERTIA_COEFFICIENT = 1
# cognitive coefficient
COGNITIVE_COEFFICIENT = 1
# social coefficient
SOCIAL_COEFFICIENT = 1
# number of iterations
N_ITERATIONS = 50


class DiscworldError(Exception):
    """Base error class."""

    pass


def create_design_space(name="thermoelectrics_starrydata", interpolate=False):
    """Load dataset. Optionally interpolate between data to create a search space of materials."""
    if not name:
        raise DiscworldError("No dataset specified")
    print(f"Loading dataset {name}...")
    df = DATASETS[name]
    print(f"Dataset size: {len(df)}")
    if not interpolate:
        return df
    else:
        raise NotImplementedError


def get_dim_n_bounds(design_space_df, features="magpie"):
    """Get the dimensions of the design space and the bounds for each dimension.

    dimensions = elements for one-hot, = all Magpie features for magpie.
    bounds = [0, 1] for one-hot (elemental fractions), [0, max(dataset)] for each magpie
    feature/dimension.
    """
    if features == "one-hot":
        df = StrToComposition(target_col_id="pmg_composition").featurize_dataframe(
            design_space_df, "composition"
        )
        elements = set()
        for composition in df["pmg_composition"]:
            [elements.add(e.symbol) for e in composition]
        bounds = [0, 1]
        return df, sorted(list(elements)), bounds
    else:
        raise NotImplementedError


def get_onehot_features(df, elements=None):
    """Helper function to get onehot (element-fraction) encoded features from composition."""
    if elements is None:
        elements = set()
        for composition in df["pmg_composition"]:
            for e in composition:
                elements.add(e.symbol)
    elements = sorted(list(elements))

    onehot = np.zeros((len(df["pmg_composition"]), len(elements)))
    for i, composition in enumerate(df["pmg_composition"]):
        for j, element in enumerate(elements):
            onehot[i][j] = composition.get_atomic_fraction(element)
    print("Sample one-hot:")
    print(f"Composition: {composition}")
    print(f"One-hot features: {onehot[i]}")
    return onehot


def get_merits_n_snap(P, features, df):
    """Calculates the figure of merit at each particle coordinate.

    The figure of merit for a given particle coordinate is that of the nearest grid point in the
    search space. The particle positions are "snapped" to the nearest grid point as well.
    """
    print("Sample merit and snap:")
    merits = []
    distances = []
    print(f"Particle -1 coordinates: {P[-1]}")
    for i, particle in enumerate(P):
        _distances = [np.linalg.norm(P[i] - f) for f in features]
        distances.append(_distances)
        min_distance = np.min(_distances)
        min_idx = np.argmin(_distances)
        P[i] = features[min_idx]
        merits.append(df["figure_of_merit"][min_idx])
    print(f"Snapped -1 coordinates: {P[-1]}")
    print(f"Distance of particle -1 to search space grid point 0: {_distances[0]}")
    print(f"Minimum distance: {min_distance}")
    print(f"Minimum features index: {min_idx}")
    print(f"Figure of merit: {merits[-1]}")
    return P, np.array(merits)


def get_current_global_best(P, merits):
    """Find the current swarm-overall best position."""
    min_idx = np.argmin(merits)
    return P[min_idx]


def _normalize_compositions(P, elements):
    print("Sample normalization:")
    print(f"Input composition: {P[0]}")
    for i, particle in enumerate(P):
        comp = Composition({e: amt for e, amt in zip(elements, P[i])})
        for j, element in enumerate(elements):
            P[i][j] = comp.get_atomic_fraction(element)
    print(f"Normalized composition: {P[0]}")
    return P


def run_pso(
    n_particles=N_PARTICLES,
    learning_rate=LEARNING_RATE,
    inertia=INERTIA_COEFFICIENT,
    cognitive=COGNITIVE_COEFFICIENT,
    social=SOCIAL_COEFFICIENT,
    n_iterations=N_ITERATIONS,
):
    """Run a PSO simulation."""
    # get the dataset and features
    space_df = create_design_space()
    df, elements, bounds = get_dim_n_bounds(space_df, features="magpie")
    features = get_onehot_features(df, elements=elements)

    rng = np.random.default_rng()

    # Initialize particle positions
    n_dim = len(elements)
    P = np.zeros((n_particles, n_dim))
    for i, particle in enumerate(P):
        for j in np.random.choice(n_dim, 3):
            P[i][j] = 1
    P = _normalize_compositions(P, elements)

    # Initialize particle velocities to zero (in all dimensions)
    V = rng.random((n_particles, n_dim))

    # Get merit for each particle, and snap position to nearest grid point in search space
    P, current_pmerits = get_merits_n_snap(P, features, df)
    print(f"Initial particle merits: {current_pmerits}")

    # Get current best particle (and best global positions)
    current_pbest = P.copy()
    current_gbest = get_current_global_best(P, current_pmerits)
    print(f"Current global best: {current_gbest}")

    # Lists to store history of particle best and global best positions
    particle_best_history = []
    global_best_history = []

    # Lists to store best merits found, and mean particle merit
    pbest_merits = current_pmerits
    gbest_merits = [max(current_pmerits)]
    mean_particle_merits = [np.mean(current_pmerits)]

    print(f"Initial mean particle merit: {mean_particle_merits[0]}")
    print(f"Initial best swarm merit: {gbest_merits[0]}")
    print()

    for it in range(n_iterations):
        print(f"ITERATION #{it:02d}")
        global_best_history.append(current_gbest)
        particle_best_history.append(current_pbest)

        # Update each particle's velocity according to standard PSO
        V = (
            inertia * V
            + cognitive * rng.random() * (current_pbest - P)
            + social * rng.random() * (current_gbest - P)
        )

        # Update position of each particle using velocity per standard PSO
        P = P + learning_rate * V

        # Snap particle coordinates and get particle merits
        P, _current_pmerits = get_merits_n_snap(P, features, df)
        print(f"Current particle merits: {_current_pmerits}")

        # Update current particle and global best positions
        for i, merit in enumerate(_current_pmerits):
            if _current_pmerits[i] > pbest_merits[i]:
                pbest_merits[i] = _current_pmerits[i]
                current_pbest[i] = P[i]

        # Store best swarm and mean particle merits
        mean_particle_merits.append(np.mean(_current_pmerits))
        gbest_merits.append(max(pbest_merits))

        current_pmerits = _current_pmerits.copy()
    print(gbest_merits)
    return gbest_merits


if __name__ == "__main__":
    import sys

    run_pso()
    sys.exit(0)

    gbest_merits = []
    for i in range(5):
        gbest_merits.append(run_pso())

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    for gm in gbest_merits:
        ax.plot(range(len(gm)), gm, "-", color="xkcd:tomato")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Figure of merit")
    plt.savefig("merit_vs_iteration.png", bbox_inches="tight", dpi=300)
