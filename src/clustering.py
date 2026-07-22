import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity


def similarity(profiles: pd.DataFrame):
    """Cosine similarity between city complexity profiles.

    Returns the labels and a city-by-city similarity matrix.
    """
    sect = [c for c in profiles.columns if c not in ("country", "city")]
    labels = profiles["city"].tolist()
    sim = cosine_similarity(profiles[sect].fillna(0).values)
    return labels, pd.DataFrame(sim, index=labels, columns=labels)


def clusters(sim: pd.DataFrame, k: int = 6) -> pd.Series:
    """Complete-linkage clusters on cosine distance."""
    d = 1 - sim.values
    np.fill_diagonal(d, 0)
    d = (d + d.T) / 2
    z = linkage(squareform(d, checks=False), method="complete")
    return pd.Series(fcluster(z, k, criterion="maxclust"), index=sim.index, name="cluster")




def cluster_profiles(rank_zscore: pd.DataFrame, cl: pd.Series) -> pd.DataFrame:
    """Average sector profile per cluster, in long form."""
    d = rank_zscore.merge(cl.rename("cluster"), left_on="city", right_index=True)
    return (d.groupby(["cluster", "sector"])["rank_zscore"].mean()
            .reset_index(name="mean_z"))
