import pandas as pd


def sector_rank_matrix(sector_eci: pd.DataFrame, country: str) -> pd.DataFrame:
    """City by sector rank of mean complexity for one country (1 = most complex)."""
    d = sector_eci[sector_eci["country"] == country]
    wide = d.pivot_table(index="city", columns="naics", values="mean_eci_zscore")
    return wide.rank(axis=0, ascending=False)
