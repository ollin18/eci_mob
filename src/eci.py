import numpy as np
import pandas as pd

COUNTRIES = ["US", "MX", "BR", "AR"]
COUNTRY_NAME = {"US": "United States", "MX": "Mexico", "BR": "Brazil", "AR": "Argentina"}

# ======================
# Loading
# ======================

def load_panel(country: str, path: str = "data/od") -> pd.DataFrame:
    """Aggregated origin-destination panel for a country.

    One row per work location and education quantile, with `advanced` the share
    of that location's workforce coming from origins in the quantile. Work
    locations are H3 resolution-8 cells; origins are summarised by the
    percentile bin of their residential education rate.
    """
    p = pd.read_csv(f"{path}/{country}.csv", dtype={"geomid": str})
    p["advanced"] = p.groupby("city")["advanced"].transform(lambda s: s / s.sum())
    return p


# ======================
# Economic complexity
# ======================

def _rca_matrix(panel: pd.DataFrame, rca_thresh: float):
    m = panel.groupby(["geomid", "quantile"])["advanced"].sum().unstack().fillna(0.0)
    geo = m.index.astype(str).values
    v = m.values
    row, col, tot = v.sum(1, keepdims=True), v.sum(0, keepdims=True), v.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        rca = np.nan_to_num((v / row) / (col / tot))
    M = (rca >= rca_thresh).astype(float)
    while True:
        rm, cm = M.sum(1) > 0, M.sum(0) > 0
        if rm.all() and cm.all():
            break
        M, geo = M[rm][:, cm], geo[rm]
    return M, geo


def location_eci(panel: pd.DataFrame, rca_thresh: float = 1.0) -> pd.DataFrame:
    """Hidalgo-Hausmann ECI over the pooled panel, one value per work location.

    The bipartite matrix is work locations by education quantiles; a location is
    complex when it draws selectively from the higher-education tail rather than
    the whole distribution. The complexity eigenvector is taken on the quantile
    side of the matrix, which is small, and mapped back to locations, so the same
    result scales to the tens of thousands of cells in the largest country.
    """
    M, geo = _rca_matrix(panel, rca_thresh)
    kc, kp = M.sum(1), M.sum(0)
    mn = M / np.sqrt(kc)[:, None]
    s = mn.T @ mn / np.sqrt(kp)[:, None] / np.sqrt(kp)[None, :]
    _, vecs = np.linalg.eigh((s + s.T) / 2)
    val = M @ (vecs[:, -2] / np.sqrt(kp))
    val = (val - val.mean()) / val.std()
    if np.corrcoef(kc, val)[0, 1] < 0:
        val = -val
    eci = pd.DataFrame({"geomid": geo, "eci": val})
    place = panel[["geomid", "city"]].drop_duplicates()
    eci = eci.merge(place, on="geomid", how="left")
    eci["eci_z"] = eci.groupby("city")["eci"].transform(lambda s: (s - s.mean()) / s.std())
    return eci


def city_eci(eci: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Worker-weighted mean ECI per city, with a standard error."""
    w = panel.groupby("geomid")["advanced"].sum().rename("weight")
    d = eci.merge(w, on="geomid", how="left").dropna(subset=["eci"])
    rows = []
    for city, g in d.groupby("city"):
        wt = g["weight"].fillna(0).values
        x = g["eci"].values
        if wt.sum() == 0:
            wt = np.ones_like(x)
        mean = np.average(x, weights=wt)
        neff = wt.sum() ** 2 / np.sum(wt ** 2)
        se = np.sqrt(np.average((x - mean) ** 2, weights=wt) / max(neff, 1))
        rows.append(dict(city=city, mean_eci=mean, se_mean_eci=se, n=len(g)))
    return pd.DataFrame(rows).sort_values("mean_eci").reset_index(drop=True)


def all_locations(path: str = "data/od") -> pd.DataFrame:
    """ECI for every work location across all countries, city and z-score attached."""
    out = []
    for country in COUNTRIES:
        panel = load_panel(country, path)
        e = location_eci(panel)
        e["country"] = country
        out.append(e)
    return pd.concat(out, ignore_index=True)
