import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy import stats
from scipy.cluster.hierarchy import dendrogram

FIG_DIR = "figs"
LINE = "#E74C3C"
COUNTRY_COLOR = {"US": "#6a51a3", "MX": "#b30000", "BR": "#238b45", "AR": "#2171b5"}
CLUSTER_COLOR = ["#c9186b", "#178a7a", "#7a4a2a", "#e23b2e", "#e79a1f", "#7d3fbf"]


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


ECI_MOB = r"$\mathbf{ECI}^{\mathbf{mob}}$"
ECI_SECT = r"$\mathbf{ECI}^{\mathbf{sect}}$"


# ======================
# Validation scatters
# ======================

def _fit(ax, x, y):
    if len(x) < 2:
        return (float("nan"), float("nan"))
    b, a = np.polyfit(x, y, 1)
    xs = np.array(ax.get_xlim())
    ax.plot(xs, b * xs + a, color=LINE, lw=2.5, zorder=1)
    return stats.pearsonr(x, y)


def city_income_panels(city, name, x="mean_eci", xlabel=ECI_MOB):
    """One panel per country: city income against a city predictor."""
    order = ["US", "MX", "BR", "AR"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    for ax, c in zip(axes, order):
        g = city[city["country"] == c]
        ax.scatter(g[x], g["income"], s=90, color=COUNTRY_COLOR[c],
                   edgecolor="white", lw=1.5, zorder=3)
        r, p = _fit(ax, g[x].values, g["income"].values)
        star = "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "n.s."
        ax.set_title({"US": "United States", "MX": "Mexico", "BR": "Brazil",
                      "AR": "Argentina"}[c], fontweight="bold")
        ax.text(.05, .92, rf"$R^2$ = {r**2:.3f} {star}", transform=ax.transAxes,
                fontsize=13, va="top")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Monthly income")
    fig.tight_layout()
    save(fig, name)


def subcity_panels(points, fits, name):
    """One panel per country: origin wealth against ECI over all work cells."""
    order = ["US", "MX", "BR", "AR"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), sharey=True)
    fmap = fits.set_index("country")
    for ax, c in zip(axes, order):
        g = points[points["country"] == c]
        if g.empty or c not in fmap.index:
            ax.axis("off")
            continue
        ax.scatter(g["eci_z"], g["wealth_z"], s=6, alpha=.25,
                   color=COUNTRY_COLOR[c], edgecolor="none")
        _fit(ax, g["eci_z"].values, g["wealth_z"].values)
        row = fmap.loc[c]
        ax.set_title({"US": "United States", "MX": "Mexico", "BR": "Brazil",
                      "AR": "Argentina"}[c], fontweight="bold")
        ax.text(.05, .93, rf"$R^2$ = {row.r2:.3f} {row.sig}", transform=ax.transAxes,
                fontsize=13, va="top")
        ax.set_xlabel(ECI_MOB)
    axes[0].set_ylabel("Origin wealth (z-score)")
    fig.tight_layout()
    save(fig, name)


# ======================
# Distributions and maps
# ======================

def ridge(eci, name):
    """ECI distribution per city, one country per panel."""
    order = ["US", "MX", "BR", "AR"]
    from data import CITY_LABEL
    fig, axes = plt.subplots(1, 4, figsize=(18, 7))
    for ax, c in zip(axes, order):
        d = eci[eci["country"] == c]
        cities = d.groupby("city")["eci"].mean().sort_values().index
        for i, city in enumerate(cities):
            v = d[d["city"] == city]["eci"].dropna()
            xs = np.linspace(v.min(), v.max(), 200)
            dens = stats.gaussian_kde(v)(xs)
            dens = dens / dens.max() * .8
            ax.fill_between(xs, i, i + dens, color=COUNTRY_COLOR[c], alpha=.55, lw=0)
            ax.plot(xs, i + dens, color="black", lw=.8)
            ax.text(v.min(), i - .05, CITY_LABEL.get(city, city), fontsize=9, va="top")
        ax.set_title({"US": "United States", "MX": "Mexico", "BR": "Brazil",
                      "AR": "Argentina"}[c], fontweight="bold")
        ax.set_yticks([])
        ax.set_xlabel(ECI_MOB)
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


def city_map(eci, city, name, cmap="viridis"):
    """ECI over a city's work cells, hexagon geometry from the H3 index."""
    d = eci[eci["city"] == city].dropna(subset=["eci"])
    patches = _hex_patches(d["geomid"])
    fig, ax = plt.subplots(figsize=(6, 6))
    pc = PatchCollection(patches, cmap=cmap, edgecolor="none")
    pc.set_array(d["eci"].values)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(pc, ax=ax, shrink=.6, label=ECI_MOB)
    save(fig, name)


# ======================
# Sectors
# ======================

def sector_bar(ms, name):
    """Mean sectoral complexity, highest to lowest."""
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(len(ms))[::-1]
    colors = plt.cm.magma(np.linspace(.15, .85, len(ms)))
    ax.barh(y, ms["mean"], xerr=ms["sem"], color=colors, edgecolor="#2c3e50")
    ax.set_yticks(y)
    ax.set_yticklabels(ms["sector"])
    ax.axvline(0, color="#555", lw=1, ls="--")
    ax.set_xlabel(ECI_SECT)
    ax.set_title("Mean sectoral complexity", fontweight="bold")
    fig.tight_layout()
    save(fig, name)


def sector_heatmap(rank_mat, name, country_name):
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(rank_mat.values, cmap="RdBu", aspect="auto")
    ax.set_xticks(range(rank_mat.shape[1]))
    ax.set_xticklabels(rank_mat.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(rank_mat.shape[0]))
    ax.set_yticklabels(rank_mat.index)
    fig.colorbar(im, ax=ax, shrink=.7, label="Rank")
    ax.set_title(f"Ranking of mean {ECI_SECT[:-1]}$ - {country_name}", fontweight="bold")
    fig.tight_layout()
    save(fig, name)


def _radar(ax, sectors, values, color, title):
    ang = np.linspace(0, 2 * np.pi, len(sectors), endpoint=False)
    ang = np.r_[ang, ang[:1]]
    v = np.r_[values, values[:1]]
    ax.plot(ang, v, color=color, lw=2)
    ax.fill(ang, v, color=color, alpha=.25)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(sectors, fontsize=7)
    ax.set_title(title, fontweight="bold", pad=15)


def city_radars(rank_zscore, cities, name):
    from data import CITY_LABEL
    fig, axes = plt.subplots(1, len(cities), figsize=(5 * len(cities), 5),
                             subplot_kw=dict(polar=True))
    for ax, city in zip(np.atleast_1d(axes), cities):
        d = rank_zscore[rank_zscore["city"] == city].sort_values("sector")
        c = COUNTRY_COLOR.get(d["country"].iloc[0], "#333")
        _radar(ax, d["sector"].tolist(), d["rank_zscore"].values, c,
               CITY_LABEL.get(city, city))
    fig.tight_layout()
    save(fig, name)


def cluster_radars(profiles, name):
    clusters = sorted(profiles["cluster"].unique())
    fig, axes = plt.subplots(1, len(clusters), figsize=(4 * len(clusters), 4.2),
                             subplot_kw=dict(polar=True))
    for ax, cl in zip(np.atleast_1d(axes), clusters):
        d = profiles[profiles["cluster"] == cl].sort_values("sector")
        _radar(ax, d["sector"].tolist(), d["mean_z"].values,
               CLUSTER_COLOR[(cl - 1) % len(CLUSTER_COLOR)], f"Cluster {cl}")
    fig.tight_layout()
    save(fig, name)


# ======================
# Clustering figures
# ======================

def dendro(lmat, labels, cl, name):
    from data import CITY_LABEL
    fig, ax = plt.subplots(figsize=(7, 9))
    lab = [CITY_LABEL.get(c, c) for c in labels]
    dendrogram(lmat, labels=lab, orientation="right", ax=ax,
               color_threshold=.7 * lmat[:, 2].max())
    ax.set_xlabel("Distance (1 - cosine similarity)")
    ax.set_title("Hierarchical clustering", fontweight="bold")
    fig.tight_layout()
    save(fig, name)


def city_network(g, cl, name):
    import networkx as nx
    from data import CITY_LABEL
    pos = nx.spring_layout(g, weight="weight", seed=42)
    colors = [CLUSTER_COLOR[(cl[n] - 1) % len(CLUSTER_COLOR)] for n in g.nodes]
    fig, ax = plt.subplots(figsize=(9, 9))
    w = [g[u][v]["weight"] for u, v in g.edges]
    nx.draw_networkx_edges(g, pos, ax=ax, width=w, edge_color="#b0b0b0", alpha=.6)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=520,
                           edgecolors="#2c3e50")
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=9,
                            labels={n: CITY_LABEL.get(n, n) for n in g.nodes})
    ax.axis("off")
    save(fig, name)
