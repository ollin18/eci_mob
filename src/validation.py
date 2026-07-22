import numpy as np
import pandas as pd
from scipy import stats

COUNTRY_NAME = {"US": "United States", "MX": "Mexico", "BR": "Brazil", "AR": "Argentina"}


def sig_stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "n.s."


def _wls_r2(x, y, w):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    r = cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))
    return r ** 2, r


# ======================
# Metropolitan scale: city income vs mean ECI and vs education
# ======================


def city_fits(city: pd.DataFrame) -> pd.DataFrame:
    """R2 and p for income against ECI and against education, by country."""
    rows = []
    for c, g in city.groupby("country"):
        for pred, name in [("eci", "eci"), ("education", "education")]:
            r, p = stats.pearsonr(g[pred], g["income"])
            rows.append(dict(country=c, predictor=name, r2=round(r ** 2, 3),
                             p=round(p, 3), sig=sig_stars(p), n=len(g)))
    return pd.DataFrame(rows)


# ======================
# Sub-city scale: origin wealth at destination vs ECI
# ======================

def subcity_fits(wealth: pd.DataFrame) -> pd.DataFrame:
    """Population-weighted R2 of origin wealth on ECI, within each country."""
    d = wealth.dropna(subset=["eci", "mean_origin_wealth", "h3_pop"])
    rows = []
    for c, g in d.groupby("country"):
        zx = (g["eci"] - g["eci"].mean()) / g["eci"].std()
        zy = ((g["mean_origin_wealth"] - g["mean_origin_wealth"].mean())
              / g["mean_origin_wealth"].std())
        r2, r = _wls_r2(zx.values, zy.values, g["h3_pop"].values)
        _, p = stats.pearsonr(zx, zy)
        rows.append(dict(country=c, r2=round(r2, 3), p=round(p, 4),
                         sig=sig_stars(p), n=len(g)))
    return pd.DataFrame(rows)


def subcity_points(wealth: pd.DataFrame) -> pd.DataFrame:
    """ECI and country-standardised origin wealth for every work cell, for plotting."""
    d = wealth.dropna(subset=["eci", "mean_origin_wealth"]).copy()
    d["wealth_z"] = d.groupby("country")["mean_origin_wealth"].transform(
        lambda s: (s - s.mean()) / s.std())
    return d
