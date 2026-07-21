import numpy as np
import pandas as pd


def mean_sector_eci(sector_eci: pd.DataFrame) -> pd.DataFrame:
    """Average complexity of each sector across cities, sorted high to low."""
    g = (sector_eci.groupby("naics")["mean_eci_zscore"]
         .agg(["mean", "sem"]).reset_index()
         .rename(columns={"naics": "sector"}))
    return g.sort_values("mean", ascending=False).reset_index(drop=True)


def sector_rank_matrix(sector_eci: pd.DataFrame, country: str) -> pd.DataFrame:
    """City by sector rank of mean complexity for one country (1 = most complex)."""
    d = sector_eci[sector_eci["country"] == country]
    wide = d.pivot_table(index="city", columns="naics", values="mean_eci_zscore")
    return wide.rank(axis=0, ascending=False)


def city_profile(rank_zscore: pd.DataFrame, city: str) -> pd.Series:
    """A city's within-country sector z-score profile."""
    d = rank_zscore[rank_zscore["city"] == city]
    return d.set_index("sector")["rank_zscore"]


def specialization(rank_zscore: pd.DataFrame) -> pd.DataFrame:
    """For each sector, the cities that stand out most above their country mean."""
    d = rank_zscore.copy()
    return d.sort_values(["sector", "rank_zscore"], ascending=[True, False])
