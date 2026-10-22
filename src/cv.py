"""Blocked spatio-temporal cross-validation: fold definitions, metrics and kernel comparison.

Three validation types, each a leave-one-block-out test (never observation-level LOO):

    spatial   sliding lat x lon windows (stride = window, so the blocks partition the globe). Training
              excludes the held-out block and a buffer: every cell whose centre lies within
              spatial_buffer_km (great circle) of a cell centre of the block, in every month.
    temporal  sliding windows of temporal_window_months (stride = window, partitioning the record).
              Training excludes the window and temporal_buffer_months on either side, everywhere.
              Training uses months before and after the window: the task is retrospective reconstruction
              and ERA5 night-time OLR exists for every month, so this is not a forecast test.
    joint     a spatial block over a time window. Training excludes the block during the window and the
              space-time halo: cells within joint_buffer_km of the block (or in it) during the window
              +- joint_buffer_months. Block-window pairs sweep space and time deterministically.

Every observation is test once per spatial and temporal type. The fold table is written once
(results/model/folds.csv) and every kernel run checks its SHA-256, so all kernels see identical folds.
"""

import hashlib

import numpy as np
import pandas as pd
from scipy import stats

import read_in

EARTH_RADIUS_KM = 6371.0
Z95 = stats.norm.ppf(0.975)
MIN_TEST = 30  # minimum held-out observations of a temporal or joint block


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def unit_vectors(lat, lon):
    phi, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.column_stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])


def great_circle_km(u1, u2):
    """(n1, n2) great-circle distances between unit vectors."""
    return EARTH_RADIUS_KM * np.arccos(np.clip(u1 @ u2.T, -1, 1))


def _grid_cells(res):
    lat_c, lon_c = read_in.grid_centres(res)
    lat, lon = np.meshgrid(lat_c, lon_c, indexing="ij")
    return lat.ravel().round(4), lon.ravel().round(4)


def spatial_blocks(block_lat, block_lon):
    """[(block_id, lat0, lat1, lon0, lon1)] sliding windows with stride = window."""
    out = []
    for lat0 in np.arange(-90, 90, block_lat):
        for lon0 in np.arange(-180, 180, block_lon):
            lat1, lon1 = min(lat0 + block_lat, 90), min(lon0 + block_lon, 180)
            out.append((f"S{lat0:+04.0f}{lon0:+05.0f}", lat0, lat1, lon0, lon1))
    return out


def in_block(lat, lon, b):
    _, lat0, lat1, lon0, lon1 = b
    return (lat >= lat0) & (lat < lat1 if lat1 < 90 else lat <= 90) & (lon >= lon0) & (lon < lon1)


def block_distance(lat, lon, b, res):
    """Great-circle distance (km) from (lat, lon) points to the nearest grid-cell centre inside block b.

    For a point outside the block the nearest block cell lies on the block's edge, so only the edge cells
    are needed (points inside the block get distance 0). Distances are computed once per distinct cell."""
    glat, glon = _grid_cells(res)
    inside = in_block(glat, glon, b)
    ilat, ilon = glat[inside], glon[inside]
    edge = (ilat == ilat.min()) | (ilat == ilat.max()) | (ilon == ilon.min()) | (ilon == ilon.max())
    pts = pd.DataFrame({"lat": np.round(lat, 4), "lon": np.round(lon, 4)})
    uniq = pts.drop_duplicates().reset_index(drop=True)
    d = np.empty(len(uniq))
    ue, uq = unit_vectors(ilat[edge], ilon[edge]), unit_vectors(uniq.lat, uniq.lon)
    for k in range(0, len(uniq), 20000):
        d[k:k + 20000] = great_circle_km(uq[k:k + 20000], ue).min(1)
    d[in_block(uniq.lat.to_numpy(), uniq.lon.to_numpy(), b)] = 0.0
    return pts.merge(uniq.assign(d=d), on=["lat", "lon"], how="left").d.to_numpy()


def month_windows(first, last, window):
    """[(block_id, start Period, end Period inclusive)] consecutive windows of `window` months."""
    months = pd.period_range(first, last, freq="M")
    return [(f"T{months[k]}", months[k], months[min(k + window, len(months)) - 1])
            for k in range(0, len(months), window)]


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def make_folds(obs, cfg):
    """Fold table with columns cv_type, fold, block_id, obs_id, role (test / buffer); every other
    observation of the fold is training."""
    c, res = cfg["cv"], cfg["grid"]["res_deg"]
    lat, lon = obs.lat.to_numpy(), obs.lon.to_numpy()
    per = obs.time.dt.to_period("M")
    mi = (per.dt.year * 12 + per.dt.month).to_numpy()
    rows = []

    def add(cv_type, fold, block_id, test, buffer):
        buffer = buffer & ~test
        for role, mask in (("test", test), ("buffer", buffer)):
            ids = obs.obs_id.to_numpy()[mask]
            rows.append(pd.DataFrame({"cv_type": cv_type, "fold": fold, "block_id": block_id, "obs_id": ids,
                                      "role": role}))

    # spatial
    blocks = spatial_blocks(c["spatial_block_lat"], c["spatial_block_lon"])
    dists = {}
    fold = 0
    for b in blocks:
        test = in_block(lat, lon, b)
        if not test.any():
            continue
        dists[b[0]] = block_distance(lat, lon, b, res)
        add("spatial", fold, b[0], test, dists[b[0]] < c["spatial_buffer_km"])
        fold += 1

    # temporal: a window with fewer than MIN_TEST observations (mostly inside the CLARA gap) is merged
    # into the following window (the last one into the previous), so no block score rests on a handful of cells
    wins = month_windows(cfg["time"]["first_month"], cfg["time"]["last_month"], c["temporal_window_months"])
    count = lambda w0, w1: int(((mi >= w0.year * 12 + w0.month) & (mi <= w1.year * 12 + w1.month)).sum())
    merged = []
    for wid, w0, w1 in wins:
        if merged and count(merged[-1][1], merged[-1][2]) < MIN_TEST:
            merged[-1] = (merged[-1][0], merged[-1][1], w1)
        else:
            merged.append((wid, w0, w1))
    if len(merged) > 1 and count(merged[-1][1], merged[-1][2]) < MIN_TEST:
        last = merged.pop()
        merged[-1] = (merged[-1][0], merged[-1][1], last[2])
    fold, bt = 0, c["temporal_buffer_months"]
    for wid, w0, w1 in merged:
        a, z = w0.year * 12 + w0.month, w1.year * 12 + w1.month
        test = (mi >= a) & (mi <= z)
        if not test.any():
            continue
        longer = z - a + 1 > c["temporal_window_months"]  # a merged window is labelled with its end
        add("temporal", fold, f"{wid}_{w1}" if longer else wid, test,
            (mi >= a - bt) & (mi <= z + bt))
        fold += 1

    # joint: deterministic sweep over (block, window) pairs with enough test observations
    jblocks = spatial_blocks(c["joint_block_lat"], c["joint_block_lon"])
    jwins = month_windows(cfg["time"]["first_month"], cfg["time"]["last_month"], c["joint_window_months"])
    bt, bkm = c["joint_buffer_months"], c["joint_buffer_km"]
    used_b, used_w, fold, k = set(), set(), 0, 0
    nb, nw = len(jblocks), len(jwins)
    step = next(s for s in (7, 5, 11, 13, 3, 1) if np.gcd(s, nw) == 1)
    while fold < c["joint_n_folds"] and k < nb * nw:
        b, (wid, w0, w1) = jblocks[k % nb], jwins[(k * step) % nw]
        k += 1
        a, z = w0.year * 12 + w0.month, w1.year * 12 + w1.month
        sp = in_block(lat, lon, b)
        test = sp & (mi >= a) & (mi <= z)
        if test.sum() < MIN_TEST or (b[0], wid) in used_b:
            continue
        used_b.add((b[0], wid))
        near = sp | (block_distance(lat, lon, b, res) < bkm)
        add("joint", fold, f"{b[0]}_{wid}", test, near & (mi >= a - bt) & (mi <= z + bt))
        fold += 1
    return pd.concat(rows, ignore_index=True)


def folds_hash(folds):
    text = folds.sort_values(["cv_type", "fold", "role", "obs_id"]).to_csv(index=False)
    return hashlib.sha256(text.encode()).hexdigest()


def fold_summary(folds, n_obs):
    s = folds.groupby(["cv_type", "fold", "block_id", "role"]).size().unstack("role", fill_value=0)
    s["train"] = n_obs - s.get("test", 0) - s.get("buffer", 0)
    return s.reset_index()


def split(folds, cv_type, fold, obs_ids):
    """Boolean masks (train, test) over obs_ids for one fold."""
    f = folds[(folds.cv_type == cv_type) & (folds.fold == fold)]
    test = np.isin(obs_ids, f.obs_id[f.role == "test"])
    excluded = np.isin(obs_ids, f.obs_id)
    return ~excluded, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def crps_gaussian(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def metrics(y, mean, obs_var, area, mean_fn=None):
    """Metrics of predictions (mean, obs_var = predictive variance of a new cell-month mean) against y.
    area: cell areas for the area-weighted RMSE. bias = mean(prediction - observation)."""
    y, mean, obs_var, area = (np.asarray(v, float) for v in (y, mean, obs_var, area))
    sd = np.sqrt(obs_var)
    err = mean - y
    w = area / area.sum()
    out = {"n": len(y), "rmse": np.sqrt(np.mean(err ** 2)), "rmse_area": np.sqrt(w @ err ** 2),
           "mae": np.mean(np.abs(err)), "bias": np.mean(err),
           "nlpd": np.mean(0.5 * np.log(2 * np.pi * obs_var) + 0.5 * err ** 2 / obs_var),
           "crps": np.mean(crps_gaussian(y, mean, sd)), "width95": np.mean(2 * Z95 * sd),
           "r2": 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2) if len(y) > 1 else np.nan}
    for level in (0.5, 0.8, 0.95):
        z = stats.norm.ppf(0.5 + level / 2)
        out[f"cover{int(level * 100)}"] = np.mean(np.abs(err) <= z * sd)
    if mean_fn is not None:
        e2 = np.asarray(mean_fn, float) - y
        out["rmse_mean_fn"] = np.sqrt(np.mean(e2 ** 2))
        out["r2_mean_fn"] = 1 - np.sum(e2 ** 2) / np.sum((y - y.mean()) ** 2) if len(y) > 1 else np.nan
    return out


LAT_BANDS = [(-90, -60, "90S-60S"), (-60, -30, "60S-30S"), (-30, 0, "30S-EQ"), (0, 30, "EQ-30N"),
             (30, 60, "30N-60N"), (60, 90.01, "60N-90N")]
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
           9: "SON", 10: "SON", 11: "SON"}


def lat_band(lat):
    lat = np.asarray(lat)
    out = np.empty(len(lat), dtype=object)
    for lo, hi, name in LAT_BANDS:
        out[(lat >= lo) & (lat < hi)] = name
    return out


def breakdown(pred, by):
    """Metrics of a prediction table grouped by the columns `by`."""
    rows = []
    for key, g in pred.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(by, key)) | metrics(g.clara_obs, g.pred_mean, g.pred_var_obs, g.cell_area_km2,
                                                  g.mean_fn))
    return pd.DataFrame(rows)


def paired_bootstrap(fold_rmse, a, b, n_boot, seed):
    """Mean over folds of rmse[a] - rmse[b] with a block-bootstrap 95% CI (folds resampled with replacement).
    fold_rmse: DataFrame indexed by fold with one column per kernel."""
    d = (fold_rmse[a] - fold_rmse[b]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def primary_score(fold_metrics):
    """Winner criterion fixed in advance: per kernel, the mean over the three validation types of the
    mean per-fold RMSE (each type weighted equally)."""
    per_type = fold_metrics.groupby(["kernel", "cv_type"]).rmse.mean().unstack("cv_type")
    return per_type.mean(1).sort_values(), per_type
