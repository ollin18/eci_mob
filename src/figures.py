import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy import stats
from scipy.cluster.hierarchy import dendrogram

FIG_DIR = "figs"
LINE = "#E74C3C"
COUNTRY_COLOR = {"US": "#2563eb", "MX": "#059669", "BR": "#d97706", "AR": "#7c3aed"}
COUNTRY_NAME = {"US": "United States", "MX": "Mexico", "BR": "Brazil", "AR": "Argentina"}
CLUSTER_COLOR = {1: "#be185d", 2: "#0f766e", 3: "#7c2d12",
                 4: "#dc2626", 5: "#f59e0b", 6: "#6b21a8"}
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


COUNTRY_LIGHT = {"US": "#9db8f0", "MX": "#8fd6bf", "BR": "#f2c98a", "AR": "#c9aef0"}
SHORT = {"Agriculture": "AGR", "Art/Culture": "A/C", "Business Support": "BUS",
         "Construction": "CONST", "Corporate": "COR", "Education": "EDU", "Energy": "EN",
         "Financial": "FIN", "Health": "HC", "Hospitality": "HOSP", "Information": "INF",
         "Manufacturing": "MFG", "Professional Service": "PROF",
         "Public Administration": "PA", "Real Estate": "RE", "Retail": "RET",
         "Transportation": "TRAN", "Wholesale": "WHOLE"}


def _radar(ax, sectors, values, sd, color, light, title):
    """One paper-style radar: RCA wedges, plus/minus one SD band, country mean, markers."""
    n = len(sectors)
    ang = [i / n * 2 * np.pi for i in range(n)] + [0.0]
    vals = np.asarray(values, float)
    limit = max(2.0, np.ceil(np.nanmax(np.abs(vals)) * 10) / 10)
    loop = np.r_[vals, vals[:1]]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    wedge = 2 * np.pi / n
    for i in range(n):
        if vals[i] > 0:
            ax.bar(ang[i], 2 * limit, bottom=-limit, width=wedge, color=light,
                   alpha=.25, edgecolor="none", zorder=0)
    ax.grid(color="#cccccc", lw=.8, alpha=.6)
    ax.set_ylim(-limit - .15, limit + .15)
    ax.set_yticks(np.linspace(-limit, limit, 5))
    ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(-limit, limit, 5)],
                       color="#666", size=8)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([SHORT.get(s, s) for s in sectors], fontsize=8)
    up = np.r_[sd, sd[:1]]
    ax.fill(np.r_[ang, ang[::-1]], np.r_[up, (-up)[::-1]], color="#888", alpha=.15, zorder=1)
    ax.plot(ang, [0] * (n + 1), color="#555", ls="--", lw=1.5, dashes=(5, 3), zorder=2)
    ax.plot(ang, loop, color=color, lw=2.5, zorder=3)
    ax.fill(ang, loop, color=color, alpha=.2, zorder=2)
    mark = [i for i in range(n) if vals[i] > sd[i]]
    if mark:
        ax.scatter([ang[i] for i in mark], [vals[i] for i in mark], s=90,
                   facecolor="white", edgecolor=color, lw=2.5, zorder=4)
    ax.set_title(title, size=14, y=1.14, fontweight="bold")


def city_radars(rank_zscore, cities, name):
    """City sector profile against the country mean and its plus/minus one SD band."""
    from data import CITY_LABEL
    fig, axes = plt.subplots(1, len(cities), figsize=(5.2 * len(cities), 5.4),
                             subplot_kw=dict(polar=True))
    for ax, city in zip(np.atleast_1d(axes), cities):
        country = rank_zscore.loc[rank_zscore["city"] == city, "country"].iloc[0]
        cd = rank_zscore[rank_zscore["country"] == country]
        sectors = sorted(cd["sector"].unique())
        sd = cd.groupby("sector")["rank_zscore"].std(ddof=1).reindex(sectors).values
        v = (rank_zscore[rank_zscore["city"] == city].set_index("sector")["rank_zscore"]
             .reindex(sectors).values)
        _radar(ax, sectors, v, sd, COUNTRY_COLOR[country], COUNTRY_LIGHT[country],
               CITY_LABEL.get(city, city))
    fig.subplots_adjust(wspace=.5)
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


def dendro(lmat, labels, cl, country, name):
    """Circular dendrogram with cluster-coloured arcs and country-coloured leaves."""
    from data import CITY_LABEL
    d = dendrogram(lmat, labels=list(labels), no_plot=True,
                   color_threshold=.7 * lmat[:, 2].max())
    leaves = d["ivl"]
    n = len(leaves)
    ang = {leaf: 2 * np.pi * i / n for i, leaf in enumerate(leaves)}
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    icoord = np.array(d["icoord"])
    dcoord = np.array(d["dcoord"])
    dmax = dcoord.max()
    xmax = icoord.max()
    for xs, ys, col in zip(icoord, dcoord, d["color_list"]):
        theta = [x / xmax * 2 * np.pi for x in xs]
        r = [dmax - y for y in ys]
        ax.plot([theta[0], theta[1]], [r[0], r[1]], color=col, lw=2)
        seg = np.linspace(theta[1], theta[2], 30)
        ax.plot(seg, [r[1]] * 30, color=col, lw=2)
        ax.plot([theta[2], theta[3]], [r[2], r[3]], color=col, lw=2)
    for i, leaf in enumerate(leaves):
        th = (2 * i + 1) / (2 * n) * 2 * np.pi
        c = country.get(leaf, "US")
        ax.scatter(th, dmax, s=90, color=COUNTRY_COLOR[c], edgecolor="#2c3e50", zorder=5)
        ax.text(th, dmax * 1.12, CITY_LABEL.get(leaf, leaf),
                rotation=np.degrees(th) - 90, ha="center", va="center", fontsize=8)
    ax.set_axis_off()
    ax.set_title("Hierarchical clustering", fontweight="bold", pad=20)
    save(fig, name)


def _forced_layout(g, cl):
    node2c = {n: int(cl[n]) for n in g.nodes()}
    clusters = sorted(set(node2c.values()))
    centers = {c: (4 * np.cos(2 * np.pi * i / len(clusters) - np.pi / 2),
                   4 * np.sin(2 * np.pi * i / len(clusters) - np.pi / 2))
               for i, c in enumerate(clusters)}
    by_c = {}
    for n, c in node2c.items():
        by_c.setdefault(c, []).append(n)
    pos = {}
    for c, nodes in by_c.items():
        cx, cy = centers[c]
        if len(nodes) == 1:
            pos[nodes[0]] = (cx, cy)
            continue
        local = nx.spring_layout(g.subgraph(nodes), k=1.5, iterations=50, seed=42)
        for n, (x, y) in local.items():
            pos[n] = (cx + x * 1.2, cy + y * 1.2)
    return pos, node2c


def city_network(g, cl, name):
    """Similarity network, nodes forced into cluster regions and coloured by cluster."""
    from data import CITY_LABEL
    pos, node2c = _forced_layout(g, cl)
    fig, ax = plt.subplots(figsize=(11, 11), facecolor="white")
    w = [g[u][v]["weight"] for u, v in g.edges]
    nx.draw_networkx_edges(g, pos, ax=ax, width=[x * 2.5 for x in w],
                           edge_color="#b0b0b0", alpha=.55)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=620,
                           node_color=[CLUSTER_COLOR[node2c[n]] for n in g.nodes],
                           edgecolors="#1a252f", linewidths=2)
    texts = [ax.text(x, y, CITY_LABEL.get(n, n), ha="center", va="center", fontsize=10)
             for n, (x, y) in pos.items()]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, expand=(1.3, 1.6),
                    arrowprops=dict(arrowstyle="-", color="#bbb", lw=.5))
    except Exception:
        pass
    ax.axis("off")
    ax.margins(.1)
    save(fig, name)


def cluster_radars(profiles, name):
    """Average sector profile per cluster against the cross-cluster mean and SD band."""
    sectors = sorted(profiles["sector"].unique())
    clusters = sorted(profiles["cluster"].unique())
    sd = profiles.groupby("sector")["mean_z"].std(ddof=1).reindex(sectors).values
    fig, axes = plt.subplots(1, len(clusters), figsize=(5 * len(clusters), 5.2),
                             subplot_kw=dict(polar=True))
    for ax, cl in zip(np.atleast_1d(axes), clusters):
        v = profiles[profiles["cluster"] == cl].set_index("sector")["mean_z"].reindex(sectors).values
        col = CLUSTER_COLOR[(cl - 1) % 6 + 1]
        _radar(ax, sectors, v, sd, col, col, f"Cluster {cl}")
    fig.subplots_adjust(wspace=.5)
    save(fig, name)
