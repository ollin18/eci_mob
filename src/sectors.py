import numpy as np
import pandas as pd

SECTORS = ["Agriculture", "Art/Culture", "Business Support", "Construction",
           "Corporate", "Education", "Energy", "Financial", "Health", "Hospitality",
           "Information", "Manufacturing", "Professional Service",
           "Public Administration", "Real Estate", "Retail", "Transportation",
           "Wholesale"]


def sector_rank_matrix(sector_eci: pd.DataFrame, country: str) -> pd.DataFrame:
    """City by sector rank of mean complexity for one country (1 = most complex)."""
    d = sector_eci[sector_eci["country"] == country]
    wide = d.pivot_table(index="city", columns="naics", values="mean_eci_zscore")
    return wide.rank(axis=0, ascending=False)


def worker_weighted_sector_eci(workers: pd.DataFrame, eci: pd.DataFrame) -> pd.DataFrame:
    """Mean complexity of each sector, weighted by the workers it employs."""
    d = workers.merge(eci[["geomid", "eci"]], on="geomid").dropna(subset=["eci"])
    d = d[d["naics"].isin(SECTORS)]
    rows = []
    for s, g in d.groupby("naics"):
        w, x = g["workers"].values, g["eci"].values
        m = np.average(x, weights=w)
        sd = np.sqrt(np.average((x - m) ** 2, weights=w))
        rows.append(dict(sector=s, mean_eci=m, sem=sd / np.sqrt(len(g))))
    return pd.DataFrame(rows).sort_values("mean_eci", ascending=False).reset_index(drop=True)


def dominant_sector(workers: pd.DataFrame) -> pd.DataFrame:
    """The sector employing the most workers in each location."""
    d = workers[workers["naics"].isin(SECTORS)]
    return d.loc[d.groupby("geomid")["workers"].idxmax(), ["geomid", "naics"]]
