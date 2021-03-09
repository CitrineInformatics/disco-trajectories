import math
import json

import numpy as np
import pandas as pd

from pymatgen import Composition
from pymatgen import DummySpecies


# raw data and related information
RAW_DATA = pd.read_csv("rawdata_interpolated.csv")
PROPERTIES = pd.read_csv("properties.csv")


def parse_data(raw_data=RAW_DATA, properties=PROPERTIES, T=1000):
    """Parse figure-of-merit vs year of discovery from raw data."""
    ddict = {}
    for row in raw_data.iterrows():
        sample_id = row[1]["sampleid"]
        composition = row[1]["composition"]
        year = row[1]["year"]
        zt = row[1]["8"]
        T = row[1]["1"]
        if math.isnan(zt):
            continue
        if year < 1990:
            continue
        try:
            pmg_composition = Composition(composition)
        except ValueError:
            continue
        else:
            if any([isinstance(e, DummySpecies) for e in pmg_composition.elements]):
                continue
        if sample_id not in ddict:
            ddict[sample_id] = {
                "composition": composition,
                "year": year,
                "zt": [],
                "T": [],
            }
        ddict[sample_id]["zt"].append(zt)
        ddict[sample_id]["T"].append(T)
    with open("parsed_data.json", "w") as fw:
        json.dump(ddict, fw, indent=2)

    max_zt_T1000 = []
    for sample_id, sample_data in ddict.items():
        for zt, _T in zip(sample_data["zt"], sample_data["T"]):
            if not np.isclose(_T, T):
                continue
            else:
                max_zt_T1000.append(
                    {
                        "composition": sample_data["composition"],
                        "year": sample_data["year"],
                        "merit": zt,
                    }
                )
    print(f"Number of data points: {len(max_zt_T1000)}")
    max_zt_T1000_df = pd.DataFrame(max_zt_T1000)
    max_zt_T1000_df.to_csv("sampled_data.csv", index=False)


if __name__ == "__main__":
    parse_data()
