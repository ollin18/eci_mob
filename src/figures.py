import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from scipy import stats

FIG_DIR = "figs"
LINE = "#E74C3C"
COUNTRY_COLOR = {"US": "#2563eb", "MX": "#059669", "BR": "#d97706", "AR": "#7c3aed"}
COUNTRY_NAME = {"US": "United States", "MX": "Mexico", "BR": "Brazil", "AR": "Argentina"}
ECI_MOB = r"$\mathbf{ECI}^{\mathbf{mob}}$"
ECI_SECT = r"$\mathbf{ECI}^{\mathbf{sect}}$"


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#7f8c8d", "figure.dpi": 110,
    })


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "n.s."


# ======================
# Distributions and maps (fig 2)
# ======================

def ridge(eci, name):
    """ECI distribution per city, magma gradient fill, one country per panel."""
    from data import CITY_LABEL
    order = ["US", "MX", "BR", "AR"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 8))
    for ax, c in zip(axes, order):
        d = eci[eci["country"] == c]
        cities = d.groupby("city")["eci"].mean().sort_values().index
        vmin, vmax = d["eci"].quantile(.01), min(d["eci"].quantile(.99), 5)
        norm = Normalize(vmin, vmax)
        for i, city in enumerate(cities):
            v = d[d["city"] == city]["eci"].dropna()
            xs = np.linspace(v.min(), min(v.max(), 5), 300)
            dens = stats.gaussian_kde(v)(xs)
            dens = dens / dens.max() * .85
            for j in range(len(xs) - 1):
                ax.fill_between(xs[j:j + 2], i, i + dens[j:j + 2], lw=0,
                                color=plt.cm.magma(norm(xs[j])))
            ax.plot(xs, i + dens, color="black", lw=1)
            ax.text(v.min(), i - .07, CITY_LABEL.get(city, city), fontsize=11, va="top")
        ax.set_title(COUNTRY_NAME[c], fontweight="bold", fontsize=16)
        ax.set_yticks([])
        ax.set_xlabel(ECI_MOB, fontsize=14)
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, name)


def _hex_patches(geomids):
    import h3
    boundary = getattr(h3, "cell_to_boundary", None) or h3.h3_to_geo_boundary
    polys = []
    for g in geomids:
        try:
            b = boundary(g)
        except Exception:
            b = boundary(g, True)
        polys.append(Polygon([(lng, lat) for lat, lng in b]))
    return polys


def city_maps(eci, cities, name):
    """ECI over each city's work cells, magma, geometry from the H3 index."""
    from data import CITY_LABEL
    fig, axes = plt.subplots(1, len(cities), figsize=(5.2 * len(cities), 5.4))
    for ax, city in zip(np.atleast_1d(axes), cities):
        d = eci[eci["city"] == city].dropna(subset=["eci"])
        pc = PatchCollection(_hex_patches(d["geomid"]), cmap="magma", edgecolor="none")
        pc.set_array(d["eci"].clip(*d["eci"].quantile([.01, .99])).values)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(CITY_LABEL.get(city, city), fontweight="bold", fontsize=14)
        fig.colorbar(pc, ax=ax, shrink=.55, label=ECI_MOB)
    fig.tight_layout()
    save(fig, name)


# ======================
# Validation scatters (fig 3)
# ======================

def _scatter_fit(ax, x, y, color, r2_only=False):
    if len(x) < 2:
        return None, None
    b, a = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, b * xs + a, color=LINE, lw=2.5, zorder=1)
    return stats.pearsonr(x, y)


def city_income_panels(city, name, x="mean_eci", xlabel=ECI_MOB):
    """City income against a city predictor, one country per panel with labels."""
    from data import CITY_LABEL
    order = ["US", "MX", "BR", "AR"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for ax, c in zip(axes, order):
        g = city[city["country"] == c]
        if g.empty:
            ax.axis("off")
            continue
        sc = ax.scatter(g[x], g["income"], s=150, c=g[x], cmap="magma",
                        edgecolor="#2c3e50", lw=1.5, zorder=3)
        r, p = _scatter_fit(ax, g[x].values, g["income"].values, None)
        ax.set_title(COUNTRY_NAME[c], fontweight="bold", fontsize=15)
        ax.text(.05, .93, rf"$R^2$ = {r**2:.3f} {_stars(p)}", transform=ax.transAxes,
                fontsize=14, va="top")
        ax.set_xlabel(xlabel, fontsize=14)
        for _, row in g.iterrows():
            ax.annotate(CITY_LABEL.get(row["city"], row["city"]),
                        (row[x], row["income"]), fontsize=8, color="#2c3e50",
                        xytext=(4, 4), textcoords="offset points")
    axes[0].set_ylabel("Monthly income", fontsize=13)
    fig.tight_layout()
    save(fig, name)


def subcity_panels(points, fits, name):
    """Origin wealth against ECI over all work cells, coloured by city, per country."""
    order = ["US", "MX", "BR", "AR"]
    fmap = fits.set_index("country")
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), sharey=True)
    for ax, c in zip(axes, order):
        g = points[points["country"] == c]
        if g.empty or c not in fmap.index:
            ax.axis("off")
            continue
        for city, gc in g.groupby("city"):
            ax.scatter(gc["eci"], gc["wealth_z"], s=7, alpha=.3, edgecolor="none")
        _scatter_fit(ax, g["eci"].values, g["wealth_z"].values, None)
        row = fmap.loc[c]
        ax.set_title(COUNTRY_NAME[c], fontweight="bold", fontsize=15)
        ax.text(.05, .93, rf"$R^2$ = {row.r2:.3f} {row.sig}", transform=ax.transAxes,
                fontsize=14, va="top")
        ax.set_xlabel(ECI_MOB, fontsize=14)
    axes[0].set_ylabel("Origin wealth (z-score)", fontsize=13)
    fig.tight_layout()
    save(fig, name)


# ======================
# Pipeline, one city (fig 1)
# ======================

def sector_bar(ss, name):
    """Mean sectoral complexity, worker-weighted, highest to lowest."""
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(len(ss))[::-1]
    colors = plt.cm.magma(np.linspace(.15, .85, len(ss)))
    ax.barh(y, ss["mean_eci"], xerr=ss["sem"], color=colors, edgecolor="#2c3e50",
            error_kw=dict(elinewidth=1.2, capsize=3))
    ax.set_yticks(y)
    ax.set_yticklabels(ss["sector"])
    ax.axvline(0, color="#555", lw=1, ls="--")
    ax.set_xlabel(ECI_SECT)
    ax.set_title("Mean sectoral complexity", fontweight="bold")
    fig.tight_layout()
    save(fig, name)


# ======================
# Sectors (fig 4)
# ======================

def sector_heatmap(rank_mat, name, country_name):
    """City by sector rank of mean complexity, 1 = most complex."""
    from data import CITY_LABEL
    n = rank_mat.shape[0]
    fig, ax = plt.subplots(figsize=(13, .55 * n + 2))
    im = ax.imshow(rank_mat.values, cmap="RdBu", aspect="auto", vmin=1, vmax=rank_mat.shape[0])
    ax.set_xticks(range(rank_mat.shape[1]))
    ax.set_xticklabels(rank_mat.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels([CITY_LABEL.get(c, c) for c in rank_mat.index], fontsize=12)
    ax.set_xticks(np.arange(-.5, rank_mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", size=0)
    for s in ax.spines.values():
        s.set_color("#bdc3c7")
    fig.colorbar(im, ax=ax, shrink=.7, label="Rank")
    ax.set_title(f"Ranking of mean {ECI_SECT} - {country_name}", fontweight="bold")
    fig.tight_layout()
    save(fig, name)


def specialization_dotplot(rank_zscore, name):
    """Each sector's cities on the rank z-score axis, coloured by country."""
    sectors = sorted(rank_zscore["sector"].unique())
    fig, ax = plt.subplots(figsize=(9, 10))
    for i, sec in enumerate(sectors):
        d = rank_zscore[rank_zscore["sector"] == sec]
        ax.axhline(i, color="#eee", lw=6, zorder=0)
        ax.scatter(d["rank_zscore"], [i] * len(d),
                   c=[COUNTRY_COLOR[c] for c in d["country"]], s=45,
                   edgecolor="white", lw=.6, zorder=3)
    ax.axvline(0, color="#c0392b", ls="--", lw=1.2)
    ax.set_yticks(range(len(sectors)))
    ax.set_yticklabels(sectors, fontsize=10)
    ax.set_xlabel("Rank z-score")
    ax.set_title("City specialization", fontweight="bold")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COUNTRY_COLOR[c],
                          label=COUNTRY_NAME[c]) for c in ["US", "MX", "BR", "AR"]]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    fig.tight_layout()
    save(fig, name)


# ======================
# City typology (fig 5)
# ======================

def cosine_heatmap(sim, cl, name):
    """City by city cosine similarity, ordered by cluster."""
    from data import CITY_LABEL
    order = cl.sort_values().index
    s = sim.loc[order, order]
    lab = [CITY_LABEL.get(c, c) for c in order]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(s.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(lab)))
    ax.set_xticklabels(lab, rotation=90, fontsize=7)
    ax.set_yticks(range(len(lab)))
    ax.set_yticklabels(lab, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=.7, label="Cosine similarity")
    fig.tight_layout()
    save(fig, name)

