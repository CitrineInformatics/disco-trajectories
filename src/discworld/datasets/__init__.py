import os
import pkg_resources

import pandas as pd


__all__ = ["DATASETS"]

DATASETS = {}


dataset_listdir = pkg_resources.resource_listdir("discworld", "datasets")
for filename in dataset_listdir:
    root, ext = os.path.splitext(filename)
    if not ext == ".csv":
        continue
    resource = pkg_resources.resource_filename("discworld.datasets", filename)
    DATASETS[root] = pd.read_csv(resource)
