import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.interpolate import SmoothBivariateSpline

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

from discworld.datasets import DATASETS


# PSO parameters
# number of particles in the swarm
N_PARTICLES = 30
# learning rate
LEARNING_RATE = 0.01
# inertia coefficient
INERTIA_COEFFICIENT = 0.7
# cognitive coefficient
COGNITIVE_COEFFICIENT = 2
# social coefficient
SOCIAL_COEFFICIENT = 2
# number of iterations
N_ITERATIONS = 100


# Feature space parameters
# number of dimensions to reduce features to
N_DIM = 2
# number of grid points per feature direction
N_FEATURE_GRID = 5000


class DiscworldError(Exception):
    """Base error class."""

    pass


def _predict_outliers(df, method="mod-zscore"):
    X = df["merit"].values
    if method == "zscore":
        Z = (X - np.mean(X)) / np.std(X)
        return Z
    elif method == "mod-zscore":
        Z = 0.6745 * (X - np.median(X)) / np.median(np.abs(X - np.median(X)))
        return Z


def remove_y_outliers(df, method="mod-zscore"):
    """Remove outliers in merit values based on standard/modified z-scores."""
    scores = _predict_outliers(df, method=method)
    if method == "zscore":
        return df[abs(scores) < 3.5]
    # outlier threshold for mod-zscore from here:
    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    elif method == "mod-zscore":
        return df[abs(scores) < 4.5]


def get_magpie_features(df):
    """Remove non-Magpie-feature columns from the input dataframe, return as numpy array."""
    print("Getting only magpie features out of dataframe...")
    df = df.drop(columns=["composition", "pmg_composition", "year", "merit"])
    print(df, len(df))
    return df.to_numpy(dtype=np.float32)


def add_magpie_features(df):
    """Add Magpie features to the input dataframe."""
    print("Adding Magpie features to dataframe...")
    df = StrToComposition(target_col_id="pmg_composition").featurize_dataframe(df, "composition")
    featurizer = MultipleFeaturizer([ElementProperty.from_preset("magpie")])
    df = featurizer.featurize_dataframe(df, col_id="pmg_composition")
    print(df, len(df), len(df.columns))
    return df


def get_outlier_scores(df, features):
    """Calculate outlier scores for features + figure of merit using isolation forests."""
    print("Calculating outlier scores using isolation forest...")
    X = np.column_stack([df["merit"], features])
    scores = IsolationForest().fit_predict(X)
    print(scores, len(scores), len([s for s in scores if s < 0]))
    return scores


def remove_outliers(df, features):
    """Remove outliers in the dataframe using isolation forests."""
    scores = get_outlier_scores(df, features)
    return df[scores > 0], features[scores > 0]


def get_reduced_features(features, n_dim=N_DIM, method="random_projections"):
    """Reduce input features to n_dim dimensions using random projections."""
    print(f"Reducing input features to {n_dim} dimensions using {method}...")
    if method == "random":
        transformer = GaussianRandomProjection(n_components=n_dim)
    elif method == "pca":
        transformer = PCA(n_components=n_dim)
    X_reduced = transformer.fit_transform(features)
    print(X_reduced, X_reduced.shape)
    return X_reduced


def normalize_features(features, scaler="minmax"):
    """Min/max scale the input features."""
    print("Scaling all features using their min/max values...")
    if scaler == "minmax":
        scaler = MinMaxScaler()
    elif scaler == "robust":
        scaler = RobustScaler()
    elif scaler == "standard":
        scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(features_scaled, features_scaled.shape)
    print(min(features_scaled[:, 0]), max(features_scaled[:, 1]))
    return features_scaled


def get_interpolated_data(df, features, n_dim=N_DIM, n_feature_grid=N_FEATURE_GRID):
    """Interpolate features vs merit data on a (n_feature_grid, n_dim) uniform grid."""
    print(f"Interpolating data on a uniform ({n_dim} x {n_feature_grid}) grid...")
    actual_m = df["merit"].to_numpy()
    vectors = [np.linspace(0, 1, num=n_feature_grid) for _ in range(n_dim)]
    grid = np.array(np.meshgrid(*vectors), dtype=np.float32)
    grid = np.dstack([g.flatten() for g in grid])[0]
    print(features.shape, actual_m.shape)
    interp_m = griddata(features, actual_m, grid, method="linear")
    print(grid, interp_m)
    print(grid.shape, interp_m.shape)
    interp_n = griddata(features, actual_m, grid, method="nearest")
    sbs = SmoothBivariateSpline(features[:, 0], features[:, 1], actual_m)
    interp_c = sbs.ev(grid[:, 0], grid[:, 1])
    print(max(actual_m[~np.isnan(actual_m)]), min(actual_m[~np.isnan(actual_m)]))
    print(max(interp_m[~np.isnan(interp_m)]), min(interp_m[~np.isnan(interp_m)]))
    print(max(interp_n[~np.isnan(interp_n)]), min(interp_n[~np.isnan(interp_n)]))
    print(max(interp_c[~np.isnan(interp_c)]), min(interp_c[~np.isnan(interp_c)]))
    return grid, interp_m, interp_n, interp_c


def curate_dataset(
    dataset="thermoelectrics_starrydata",
    y_filter_method="mod-zscore",
    n_dim=N_DIM,
    n_feature_grid=N_FEATURE_GRID,
):
    """Curate data: remove outliers, generate features, reduce dimensions, interpolate."""
    df = DATASETS[dataset]
    print(f"Size of the sampled dataset: {len(df)}")
    df = remove_y_outliers(df, method=y_filter_method)
    print(f"Size of the dataset post z-score based filtering: {len(df)}")
    df[["composition", "year", "merit"]].to_csv("post_zscore.csv", index=False)
    # generate magpie features using matminer
    df = add_magpie_features(df)
    # get only magpie features as numpy array
    features = get_magpie_features(df)
    # remove outliers using scores from isolation forest
    df, features = remove_outliers(df, features)
    print(f"Size of the dataset post isolation forests based filtering: {len(df)}")
    df[["composition", "year", "merit"]].to_csv("post_forest.csv", index=False)
    # reduce dimensionality of features to n_dim
    features = normalize_features(features, scaler="robust")
    features = get_reduced_features(features, n_dim=n_dim, method="random")
    # min/max scale all features
    features = normalize_features(features, scaler="minmax")
    np.save("features.npy", features, allow_pickle=False)
    np.save("merits.npy", df["merit"].to_numpy(), allow_pickle=False)
    # generate interpolated data
    grid, merits_l, merits_n, merits_c = get_interpolated_data(
        df, features, n_dim=n_dim, n_feature_grid=n_feature_grid
    )
    np.save("interpolated_features.npy", grid, allow_pickle=False)
    np.save("interpolated_merits_linear.npy", merits_l, allow_pickle=False)
    np.save("interpolated_merits_nearest.npy", merits_n, allow_pickle=False)
    np.save("interpolated_merits_cubic.npy", merits_c, allow_pickle=False)
    merits_n[~np.isnan(merits_l)] = 0
    merits_l[np.isnan(merits_l)] = 0
    merits_ln = merits_l + merits_n
    np.save("interpolated_merits_linearest.npy", merits_ln, allow_pickle=False)


def get_merits_n_snap(P, grid, merits, kdtree=None):
    """Calculates the figure of merit at each particle coordinate.

    The figure of merit for a given particle coordinate is that of the nearest grid point in the
    search space. The particle positions are "snapped" to the nearest grid point as well.
    """
    # The following block does a "manual" calculation of distance between a particle P and all the
    # grid points G. Is >100 **slower** than using a k-dimensional tree (above). Only for testing
    # purposes. Unless you have a good reason for doing otherwise, use the scipy.KDTree block!
    if kdtree is None:
        print("Sample merit and snap:")
        _merits = []
        distances = []
        _min_distances = []
        _P = P.copy()
        print(f"Particle -1 coordinates: {P[-1]}")
        for i, particle in enumerate(P):
            _distances = [np.linalg.norm(P[i] - g) for g in grid]
            distances.append(_distances)
            min_distance = np.min(_distances)
            _min_distances.append(min_distance)
            min_idx = np.argmin(_distances)
            _P[i] = grid[min_idx]
            _merits.append(merits[min_idx])
        print(f"Snapped -1 coordinates: {_P[-1]}")
        print(f"Distance of particle -1 to search space grid point 0: {_distances[0]}")
        print(f"Minimum distance: {min_distance}")
        print(f"Minimum features index: {min_idx}")
        print(f"Figure of merit: {merits[-1]}")
        return _P, np.array(_merits)
    # KDTree is a binary tree that provides an index into a set of k-dimensional points to rapidly
    # look up the nearst neighbors of any point.
    # Each node represents an axis-aligned hyperrectangle, and splits the set of points basaed on
    # whether their coordinate along that axis if greater than or less than a particular value
    distances, idx = kdtree.query(P, k=1)
    return grid[idx], merits[idx]


def get_current_global_best(P, merits):
    """Find the current swarm-overall best position."""
    max_idx = np.argmax(merits)
    return P[max_idx]


def run_pso(
    dataset="thermoelectrics_starrydata",
    n_dim=N_DIM,
    n_feature_grid=N_FEATURE_GRID,
    n_particles=N_PARTICLES,
    learning_rate=LEARNING_RATE,
    inertia=INERTIA_COEFFICIENT,
    cognitive=COGNITIVE_COEFFICIENT,
    social=SOCIAL_COEFFICIENT,
    n_iterations=N_ITERATIONS,
):
    """Run a PSO simulation."""
    rng = np.random.default_rng()
    grid = np.load("interpolated_features.npy")
    merits = np.load("interpolated_merits_linearest.npy")

    # build kd-tree for quick nearest neighbor lookup
    kdtree = cKDTree(grid)

    # Initialize particle positions
    P = rng.random(size=(n_particles, n_dim), dtype=np.float32)

    # Initialize particle velocities to zero (in all dimensions)
    V = np.zeros((n_particles, n_dim))

    # Get merit for each particle, and snap position to nearest grid point in search space
    P, init_pmerits = get_merits_n_snap(P, grid, merits, kdtree=kdtree)
    print(f"Initial particle merits: {init_pmerits}")

    # Get current best particle (and best global positions)
    current_pbest = P.copy()
    current_gbest = get_current_global_best(P, init_pmerits)
    print(f"Current global best: {current_gbest}")

    # List to store history of particle best positions (n_particles, n_iterations)
    pbest_history = [current_pbest]
    # List to store history of global best positions (n_iterations, )
    gbest_history = [current_gbest]
    # List to store particle positions
    P_history = [P]

    # Lists to store all particle merits and the best merits found
    pmerit_history = [init_pmerits]
    pbest_merits = init_pmerits
    gbest_merits = [max(init_pmerits)]
    print(f"Initial best swarm merit: {gbest_merits[0]}")

    for it in range(n_iterations):
        print(f"ITERATION #{it:02d}")

        # Update each particle's velocity according to standard PSO
        V = (
            inertia * V
            + cognitive * rng.random() * (current_pbest - P)
            + social * rng.random() * (current_gbest - P)
        )

        # Update position of each particle using velocity per standard PSO
        P = P + learning_rate * V
        np.clip(P, 0, 1)

        # Snap particle coordinates and get particle merits
        P, current_pmerits = get_merits_n_snap(P, grid, merits, kdtree=kdtree)
        print(f"Current particle merits: {current_pmerits}")
        P_history.append(P)

        # Update current particle and global best positions
        for i, merit in enumerate(current_pmerits):
            if current_pmerits[i] > pbest_merits[i]:
                pbest_merits[i] = current_pmerits[i]
                current_pbest[i] = P[i]

        # Store particle best and global best histories
        gbest_history.append(current_gbest)
        pbest_history.append(current_pbest)

        # Store particle merits and best swarm merit
        pmerit_history.append(current_pmerits)
        gbest_merits.append(max(pbest_merits))

    np.save("pbest_history.npy", np.array(pbest_history), allow_pickle=False)
    np.save("gbest_history.npy", np.array(gbest_history), allow_pickle=False)
    np.save("particle_history.npy", np.array(P_history), allow_pickle=False)
    np.save("pmerit_history.npy", np.array(pmerit_history), allow_pickle=False)

    print(gbest_merits)


if __name__ == "__main__":
    import sys

    curate_dataset()
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
