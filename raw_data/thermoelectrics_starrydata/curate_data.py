import os
import pkg_resources

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


# location of the post-processed data, filenames
datafile_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
pkg_datafile_path = pkg_resources.resource_filename("discworld.datasets", f"{datafile_name}.csv")


def _predict_outliers(df, method="mod-zscore"):
    X = df["merit"].values
    if method == "isolation-forest":
        X = np.reshape(X, (len(X), 1))
        model = IsolationForest(random_state=0).fit(X)
        return model.predict(X)
    elif method == "zscore":
        Z = (X - np.mean(X)) / np.std(X)
        return Z
    elif method == "mod-zscore":
        Z = 0.6745 * (X - np.median(X)) / np.median(np.abs(X - np.median(X)))
        return Z


def remove_outliers(df, method="mod-zscore"):
    scores = _predict_outliers(df, method=method)
    if method == "isolation-forest":
        return df[scores == 1]
    elif method == "zscore":
        return df[scores < 2.5]
    # outlier threshold for mod-zscore from here:
    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    elif method == "mod-zscore":
        return df[scores < 3.5]


def curate_data(csv_path="sampled_data.csv", write_to_pkg_data=True, method="mod-zscore"):
    df = pd.read_csv(csv_path)
    print(f"Size of the sampled dataset: {len(df)}")
    curated_df = remove_outliers(df, method=method)
    print(f"Size of the curated dataset: {len(curated_df)}")
    curated_df.to_csv("curated_data.csv")
    curated_df.to_csv(pkg_datafile_path)


if __name__ == "__main__":
    curate_data()
