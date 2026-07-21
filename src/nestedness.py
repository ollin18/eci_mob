import numpy as np
import pandas as pd

# ======================
# Binary complexity matrix
# ======================

def binary_matrix(panel: pd.DataFrame, rca_thresh: float = 1.0) -> np.ndarray:
    """Work location by education quantile presence/absence matrix.

    A cell is 1 when the location's share of workers from that quantile exceeds
    what its overall size would predict (revealed comparative advantage >= 1).
    """
    m = (panel.groupby(["geomid", "quantile"])["advanced"].sum()
         .unstack().fillna(0.0).values)
    row = m.sum(1, keepdims=True)
    col = m.sum(0, keepdims=True)
    tot = m.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        rca = (m / row) / (col / tot)
    b = (np.nan_to_num(rca) >= rca_thresh).astype(np.int32)
    return b[b.sum(1) > 0][:, b.sum(0) > 0]


# ======================
# NODF
# ======================

def nodf(M: np.ndarray, chunk: int = 256) -> float:
    """Nestedness metric based on overlap and decreasing fill (0-100).

    Rows are compared in blocks so the pairwise overlap never has to be held in
    memory all at once, which keeps it feasible on the tens of thousands of work
    locations in the larger countries.
    """
    M = np.asarray(M, dtype=np.int32)
    n_rows, n_cols = M.shape
    rs, cs = M.sum(1), M.sum(0)
    score, pairs = 0.0, 0
    for start in range(0, n_rows, chunk):
        end = min(start + chunk, n_rows)
        overlap = M[start:end] @ M.T
        for k in range(end - start):
            i = start + k
            if i + 1 >= n_rows:
                continue
            lo = np.minimum(rs[i], rs[i + 1:])
            valid = (rs[i] != rs[i + 1:]) & (lo > 0)
            if valid.any():
                score += (overlap[k, i + 1:][valid] / lo[valid]).sum()
                pairs += int(valid.sum())
    n_row = score / pairs if pairs else 0.0
    oc = (M.T @ M)
    pu, qu = np.triu_indices(n_cols, 1)
    lo = np.minimum(cs[pu], cs[qu])
    valid = (cs[pu] != cs[qu]) & (lo > 0)
    n_col = (oc[pu, qu][valid] / lo[valid]).sum() / valid.sum() if valid.any() else 0.0
    return 100.0 * (n_row + n_col) / 2.0


def nodf_significance(M: np.ndarray, n_null: int = 49, seed: int = 42) -> dict:
    """Observed NODF against a fill-matched random null; returns z and p."""
    obs = nodf(M)
    fill = M.mean()
    rng = np.random.default_rng(seed)
    null = np.array([nodf((rng.random(M.shape) < fill).astype(np.int32))
                     for _ in range(n_null)])
    z = (obs - null.mean()) / null.std()
    p = (null >= obs).mean()
    return dict(nodf=round(obs, 1), null_mean=round(float(null.mean()), 1),
                z=round(float(z), 1), p=round(float(p), 3),
                fill=round(float(fill), 3), n_zones=M.shape[0], n_bins=M.shape[1])
