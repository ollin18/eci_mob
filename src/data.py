import pandas as pd

CITY_LABEL = {
    "mexico_city": "Mexico City", "guadalajara": "Guadalajara", "monterrey": "Monterrey",
    "puebla": "Puebla", "toluca": "Toluca", "tijuana": "Tijuana",
    "sao_paulo": "Sao Paulo", "rio_de_janeiro": "Rio de Janeiro", "salvador": "Salvador",
    "brasilia": "Brasilia", "fortaleza": "Fortaleza", "belo_horizonte": "Belo Horizonte",
    "buenos_aires": "Buenos Aires", "cordoba": "Cordoba", "la_plata": "La Plata",
    "mendoza": "Mendoza", "rosario": "Rosario", "san_miguel_de_tucuman": "San Miguel de Tucuman",
    "bay": "Bay Area", "nyc": "New York", "la": "Los Angeles", "chicago": "Chicago",
    "houston": "Houston", "phoenix": "Phoenix", "boston": "Boston", "miami": "Miami",
    "dc_baltimore": "DC/Baltimore",
}


def load_city_income(path: str = "data/validation/city_income.csv") -> pd.DataFrame:
    """Per-city income and residential education, one row per metro.

    Income is in the units each national source reports natively: household for
    the United States and Mexico, per capita for Argentina and Brazil.
    """
    return pd.read_csv(path)


def load_origin_wealth(path: str = "data/validation/origin_wealth.csv") -> pd.DataFrame:
    """Origin wealth at destination, one row per work location.

    For each work cell, the commuting-flow-weighted average of a small-area
    residential wealth proxy over the home zones that supply its workers.
    """
    return pd.read_csv(path, dtype={"geomid": str})


def load_rank_zscore(path: str = "data/sectors/rank_zscore.csv") -> pd.DataFrame:
    """City by sector complexity profile, z-scored within country."""
    return pd.read_csv(path)


def load_sector_eci(path: str = "data/sectors/sector_eci.csv") -> pd.DataFrame:
    """Mean within-city complexity of each sector, z-scored within country."""
    return pd.read_csv(path)


def wide_profiles(rank_zscore: pd.DataFrame) -> pd.DataFrame:
    """Pivot the rank-z-score table to one row per city, one column per sector."""
    w = rank_zscore.pivot_table(index=["country", "city"], columns="sector",
                                values="rank_zscore")
    return w.reset_index()
