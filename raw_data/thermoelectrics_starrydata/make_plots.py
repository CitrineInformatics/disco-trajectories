import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams


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


def plot_data(sampled_csv="sampled_data.csv", curated_csv="curated_data.csv"):
    """Plot year of discovery vs figure of merit."""
    sampled = pd.read_csv(sampled_csv)
    curated = pd.read_csv(curated_csv)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    year_vs_merit = {}
    for y, m in zip(curated["year"], curated["merit"]):
        if y not in year_vs_merit:
            year_vs_merit[y] = []
        else:
            year_vs_merit[y].append(m)
    years = sorted(list(set(curated["year"])))

    # plot the sample data first to show outliers in contrast to curated data
    ax.plot(sampled["year"], sampled["merit"], "o", c="xkcd:tomato", ms=8, mew=1, mec="w")
    # plot the curated data on top of it -> only outliers from sample data will be visible
    ax.plot(curated["year"], curated["merit"], "o", c="xkcd:turquoise", ms=8, mew=1, mec="w")

    # draw a line connecting the maximum merit per year
    max_merits = [max(year_vs_merit[y]) for y in years]
    ax.plot(years, max_merits, "-", color="xkcd:grey", lw=2.0)
    # draw a line connecting the median merit per year
    median_merits = [np.median(year_vs_merit[y]) for y in years]
    ax.plot(years, median_merits, "-", c="xkcd:blue", lw=2.0)

    ax.set_xlabel("Year reported")
    ax.set_ylabel("Figure of merit (ZT)")
    ax.set_xlim([1990, 2021])
    ax.set_ylim([0, 5])

    plt.savefig("plot.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    plot_data()
