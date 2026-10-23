"""Phase 3: the harmonised CLARA - ERA5 dataset on the model grid (grid.res_deg cells x months).

Steps (each documented in results/preprocessing/decisions.md):
  1. Units and sign. CLARA OLR is used as stored (W m-2, positive upward). ERA5 ttr (J m-2, negative
     upward) becomes OLR = -ttr / 3600 per hourly accumulation (era5_night.py). Physical units only;
     statistical standardisation happens inside each CV fold on training rows (gp.GP.fit).
  2. Time. CLARA Julian dates (UTC) are binned by calendar month; ERA5 monthly products are stamped at
     the first of the month. The model time of a cell-month is the middle of the month (the estimand is
     the monthly mean), never the CLARA sample times.
  3. Space. Longitudes in [-180, 180), latitudes ascending. ERA5 0.25 deg points -> 1 deg cell area means
     (read_in.era5_to_ceres_grid) -> night-time mean -> optional spherical-cap smoothing (ERA5 only) ->
     res deg area means (read_in.coarsen, NaN aware). CLARA footprints are binned into the res deg cells by
     position; no CLARA value is ever interpolated or spread to another cell.
  4. QC. read_in's earth_view & in_range, plus night side (SZA > qc.night_sza_min) and VZA <= qc.vza_max
     (samples.py).
  5. Pairing. Each observed cell-month (>= 1 QC'd sample) gets the ERA5 night-time OLR of the same cell
     and month. Cells and months without CLARA stay missing in the observation table; the prediction grid
     holds every cell-month with its clara_observed flag, so modelled and observed values never mix.
  6. Regime. post_gap = 1 from time.gap_start on, 0 before (post_gap_indicator): the predictor of the
     ~40 W m-2 calibration step across the CLARA gap. The gap months, never observed, count as post-gap.

Output:
    data/processed/harmonised_<res>deg_monthly.parquet     observations (one row per observed cell-month)
    data/processed/prediction_grid_<res>deg_monthly.parquet every cell-month with ERA5 (and CLARA if any)
    data/processed/eda_reference_<res>deg_monthly.parquet  ERA5 24 h and CERES per cell-month (EDA only)
    results/preprocessing/decisions.md, qc_counts.csv, validation_checks.csv, observations_summary.csv

Usage:
    python src/harmonise.py
"""

import numpy as np
import pandas as pd
import xarray as xr

import config
import era5_night
import read_in
import samples

EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def cell_index(lat, lon, res):
    """Row and column of the res deg cells (read_in.grid_edges) holding each (lat, lon); lat = 90 and
    lon = 180 fall in the last row / column."""
    lat_c, lon_c = read_in.grid_centres(res)
    i = np.clip(np.floor((np.asarray(lat) + 90) / res).astype(int), 0, len(lat_c) - 1)
    j = np.clip(np.floor((np.asarray(lon) + 180) / res).astype(int), 0, len(lon_c) - 1)
    return i, j


def cell_area_km2(lat_c, res):
    """Area (km2) of a res x res deg cell centred at lat_c (exact on the sphere)."""
    lat_edges, _ = read_in.grid_edges(res)
    i = np.clip(np.floor((np.asarray(lat_c) + 90) / res).astype(int), 0, len(lat_edges) - 2)
    s, n = np.radians(lat_edges[i]), np.radians(lat_edges[i + 1])
    return EARTH_RADIUS_KM ** 2 * (np.sin(n) - np.sin(s)) * np.radians(res)


def post_gap_indicator(time, cfg):
    """1.0 for months from time.gap_start on (the gap months included), 0.0 before."""
    per = pd.Series(pd.to_datetime(time)).dt.to_period("M")
    return (per >= pd.Period(cfg["time"]["gap_start"], "M")).astype(float).to_numpy()


def mid_month(month_start):
    """Middle of the month of each month-start timestamp (Series or array)."""
    t = pd.Series(pd.to_datetime(month_start))
    return (t + pd.to_timedelta(t.dt.days_in_month / 2, unit="D")).to_numpy()


# ---------------------------------------------------------------------------
# CLARA cell-months
# ---------------------------------------------------------------------------

def bin_cell_months(q, res):
    """Cell-month table of QC'd CLARA samples q: lat, lon (cell centre), time (month start), clara (mean),
    clara_sd (within-cell sample SD), n_samples, n_overpasses, mean_sample_time."""
    lat_c, lon_c = read_in.grid_centres(res)
    i, j = cell_index(q.clara_lat.to_numpy(), q.clara_lon.to_numpy(), res)
    df = pd.DataFrame({"time": q.time.dt.to_period("M").dt.to_timestamp().to_numpy(), "i": i, "j": j,
                       "olr": q.olr.to_numpy(), "overpass": q.overpass.to_numpy(),
                       "t": q.time.to_numpy()})
    g = df.groupby(["time", "i", "j"], sort=True)
    out = g.agg(clara=("olr", "mean"), clara_sd=("olr", "std"), n_samples=("olr", "size"),
                n_overpasses=("overpass", "nunique"), mean_sample_time=("t", "mean")).reset_index()
    out.insert(1, "lat", lat_c[out.i])
    out.insert(2, "lon", lon_c[out.j])
    return out


# ---------------------------------------------------------------------------
# ERA5 / CERES on the model grid
# ---------------------------------------------------------------------------

def era5_on_grid(night, res, radii_km):
    """ERA5 night-time OLR (smoothed at each radius) and 24 h OLR as res deg cell-month area means."""
    sm = era5_night.smoothed(night, "olr_night", radii_km)
    sm["olr_24h"] = night.olr_24h
    sm["night_frac"] = night.night_frac
    sm["night_fallback"] = night.night_fallback.astype(float)  # > 0 where (part of) the cell is in polar day
    return read_in.coarsen(sm, res)


def grid_to_table(ds, names):
    """(time, lat, lon) Dataset -> long table with the given variables, coordinates rounded for joins."""
    df = ds[list(names)].to_dataframe().reset_index()
    df["lat"], df["lon"] = df.lat.round(4), df.lon.round(4)
    return df


def build(cfg):
    res = cfg["grid"]["res_deg"]
    radii = cfg["era5"]["smoothing_radii_km"]
    s = samples.load_samples(cfg)
    q = s[s.qc]
    cells = bin_cell_months(q, res)
    cells["lat"], cells["lon"] = cells.lat.round(4), cells.lon.round(4)

    night = era5_night.load()
    era = era5_on_grid(night, res, radii)
    names = [f"olr_night_r{int(r)}" for r in radii] + ["olr_24h", "night_frac", "night_fallback"]
    era_t = grid_to_table(era, names).rename(columns={f"olr_night_r{int(r)}": f"era5_night_r{int(r)}" for r in radii}
                                             | {"olr_24h": "era5_24h"})

    obs = cells.merge(era_t, on=["time", "lat", "lon"], how="left")
    obs["night_fallback"] = obs.night_fallback > 0
    obs["model_time"] = mid_month(obs.time)
    obs["cell_area_km2"] = cell_area_km2(obs.lat, res)
    gs, ge = pd.Period(cfg["time"]["gap_start"], "M"), pd.Period(cfg["time"]["gap_end"], "M")
    per = obs.time.dt.to_period("M")
    obs["period"] = np.where(per < gs, "pre-gap", np.where(per > ge, "post-gap", "gap"))
    obs["post_gap"] = post_gap_indicator(obs.time, cfg)
    obs = obs.sort_values(["time", "lat", "lon"]).reset_index(drop=True)
    obs.insert(0, "obs_id", np.arange(len(obs)))
    obs.insert(1, "year", obs.time.dt.year)
    obs.insert(2, "month", obs.time.dt.month)
    obs = obs.drop(columns=["i", "j"])

    grid = era_t.copy()
    grid = grid[(grid.time >= pd.Timestamp(cfg["time"]["first_month"]))
                & (grid.time <= pd.Timestamp(cfg["time"]["last_month"]))]
    grid = grid.merge(obs[["time", "lat", "lon", "obs_id", "clara", "n_samples", "n_overpasses"]],
                      on=["time", "lat", "lon"], how="left")
    grid["clara_observed"] = grid.obs_id.notna()
    grid["obs_id"] = grid.obs_id.fillna(-1).astype(int)
    grid["n_overpasses"] = grid.n_overpasses.fillna(0).astype(int)
    grid["n_samples"] = grid.n_samples.fillna(0).astype(int)
    grid["night_fallback"] = grid.night_fallback > 0
    grid["model_time"] = mid_month(grid.time)
    grid["post_gap"] = post_gap_indicator(grid.time, cfg)
    grid["cell_area_km2"] = cell_area_km2(grid.lat, res)
    grid.insert(0, "year", grid.time.dt.year)
    grid.insert(1, "month", grid.time.dt.month)

    ceres = read_in.coarsen(read_in.read_ceres(), res)
    ref = grid_to_table(ceres.rename(olr="ceres"), ["ceres"])
    ref = obs[["obs_id", "time", "lat", "lon", "era5_24h"]].merge(ref, on=["time", "lat", "lon"], how="left")
    return s, obs, grid, ref


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def qc_counts(s, cfg):
    stages = [("all samples", np.ones(len(s), bool)),
              ("geolocated", s.sat_lat.notna().to_numpy()),
              ("Earth viewing", s.earth_view.to_numpy()),
              ("Earth viewing & in range (read_in usable)", s.usable.to_numpy()),
              (f"+ night side (SZA > {cfg['qc']['night_sza_min']:g} deg)", (s.usable & s.night).to_numpy()),
              (f"+ VZA <= {cfg['qc']['vza_max']:g} deg (model observations)", s.qc.to_numpy())]
    rows = [{"stage": k, "samples": int(v.sum()), "fraction_of_usable": v.sum() / s.usable.sum()} for k, v in stages]
    return pd.DataFrame(rows)


EXPECTED_OBS = {1: "~188 k (1 deg feasibility test)", 5: "~38 k (inspection)"}


def validation_checks(obs, grid, cfg):
    night_val = pd.read_csv(config.PREPROC_DIR / "era5_night_validation.csv")
    w = np.cos(np.radians(grid.lat))
    res = cfg["grid"]["res_deg"]
    n_cells = int(180 / res) * int(360 / res)
    checks = [
        ("ERA5 hour-of-day 24 h mean equals monthly product (max |diff|, W m-2)",
         night_val.max_abs_diff_24h_vs_monthly.max(), "< 0.1"),
        ("ERA5 global mean OLR, 24 h (W m-2)", np.average(grid.era5_24h, weights=w), "~240"),
        ("ERA5 global mean OLR, night-time (fallback in polar day) (W m-2)",
         np.average(grid.era5_night_r0, weights=w), "close to the 24 h mean"),
        ("observed cell-months", len(obs), EXPECTED_OBS.get(res, "")),
        ("observed cell-months without an ERA5 night value", int(obs.era5_night_r0.isna().sum()), "0"),
        ("observed cell-months in polar-day fallback cells", int(obs.night_fallback.sum()), "~0"),
        ("cell-months in prediction grid", len(grid), f"{grid.time.nunique()} months x {n_cells:,} cells = "
                                                      f"{grid.time.nunique() * n_cells:,}"),
        ("prediction-grid cell-months without a predictor value (gaps)",
         int(grid[[c for c in grid.columns if c.startswith("era5_night_r")]].isna().any(axis=1).sum()), "0"),
        ("prediction-grid cell-months in polar day (darkest-hours fallback)", int(grid.night_fallback.sum()),
         "flagged, extrapolation"),
        ("duplicate (time, lat, lon) in observations", int(obs.duplicated(["time", "lat", "lon"]).sum()), "0"),
        ("observations whose post_gap disagrees with the period label",
         int((obs.post_gap != (obs.period != "pre-gap")).sum()), "0"),
        ("prediction-grid months with post_gap = 1", grid.time[grid.post_gap == 1].nunique(),
         f"{len(pd.period_range(cfg['time']['gap_start'], cfg['time']['last_month'], freq='M'))} "
         f"(every month from {cfg['time']['gap_start']})"),
        ("CLARA cell-month means outside [olr_min, 400] W m-2", int(((obs.clara < 50) | (obs.clara > 400)).sum()), "0"),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "expected"])


DECISIONS = """# Preprocessing decisions (generated by src/harmonise.py)

Configuration: `{cfg_path}`, grid {res} deg, QC night SZA > {sza:g} deg and VZA <= {vza:g} deg.

| # | Decision | Why | Effect on resolution / information |
|---|---|---|---|
| 1 | CLARA OLR used as stored, W m-2 positive upward; ERA5 OLR = -ttr/3600 per hourly accumulation (monthly by hour of day) | ERA5 ttr is a J m-2 accumulation, negative upward; validated: the 24 h mean of the hourly product equals the monthly product to {maxdiff:.4f} W m-2 | none |
| 2 | Physical-unit harmonisation only; y and ERA5 are standardised inside each CV fold with training rows | prevents leakage of test-fold statistics | none |
| 3 | QC: read_in earth_view & in_range [{olr_min:g} (raised {olr_from}), {olr_max:g}] W m-2, footprint SZA > {sza:g} deg, VZA <= {vza:g} deg | CLARA measures OLR on the night side only; daylit footprints read tens of W m-2 higher; VZA > 60 deg shows strong limb darkening (results/eda/physics_*.png) | keeps {n_qc:,} of {n_usable:,} usable samples ({frac_qc:.0%}) |
| 4 | Overpasses: consecutive QC'd samples separated by <= {gap:g} min belong to one overpass; n_overpasses per cell-month enters the noise model | samples along one track are strongly correlated (lag-1 r ~ 0.9), so the sampling error scales with overpasses, not samples | none |
| 5 | Monthly resolution, {res} deg cells (grid.res_deg; user and supervisor choice), model time = middle of the month | ERA5 night-time product is monthly; a cell-month mean is the CLARA estimand | CLARA values are only binned by footprint centre, never interpolated or copied to another cell. {res_note} "day" is not used as a predictor |
| 6 | ERA5 night-time mean: hourly monthly means weighted by the fraction of the month's days on which each hour is night (SZA > {sza:g} deg at the 1 deg cell centre); polar-day cells (no night hour in the month) use the mean of the {n_dark} darkest hours instead and are flagged night_fallback | matches what CLARA can observe; a 24 h mean mixes in daytime OLR; the fallback gives the prediction grid every lat-lon cell (user requirement) | the fallback value is a different quantity (darkest hours of a sunlit month) and CLARA never observes those cells: predictions there are extrapolation and are hatched on maps |
| 7 | ERA5 0.25 deg -> 1 deg trapezoid area means{coarsen_note} (cos-lat weights) | conservative area averaging, not point interpolation | ERA5 effective resolution degraded to the model grid; no new information |
| 8 | ERA5 spherical-cap smoothing at R in {radii} km kept as candidate predictors; R is chosen per CV fold on training rows | CLARA sees a wide footprint (unknown FOV); matching effective resolutions without using test data | smoothing uses ERA5 only; it never fills NaN cells |
| 9 | Observation table holds only cell-months with >= 1 QC'd CLARA sample; the prediction grid holds every lat-lon cell of every month and flags clara_observed and night_fallback | genuine missingness is kept explicit; the grid has no gaps | none |
| 10 | post_gap (1 from {gap_start} on, 0 before) enters every model's mean function as beta2 (gp.gap_step); the period labels pre-gap / post-gap are kept for reporting | CLARA - ERA5 changes by ~40 W m-2 across the {gap_start} -> {gap_end} data gap (calibration step, EDA); the user chose a step predictor. The gap months are assigned to the post-gap regime by convention | CLARA never observes the gap months, so predictions there are extrapolation and the step is placed at the start of the gap |
| 11 | CERES EBAF is joined only into the EDA reference table, never into the model inputs | user decision: CERES is an EDA reference | none |

## Attrition

{qc_table}

## Validation checks

{val_table}
"""


def md_table(df, floatfmt="{:,.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else
                 (f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    cfg = config.load()
    res = cfg["grid"]["res_deg"]
    s, obs, grid, ref = build(cfg)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    obs.to_parquet(config.PROCESSED_DIR / f"harmonised_{res}deg_monthly.parquet", index=False)
    grid.to_parquet(config.PROCESSED_DIR / f"prediction_grid_{res}deg_monthly.parquet", index=False)
    ref.to_parquet(config.PROCESSED_DIR / f"eda_reference_{res}deg_monthly.parquet", index=False)

    out = config.PREPROC_DIR
    out.mkdir(parents=True, exist_ok=True)
    qc = qc_counts(s, cfg)
    val = validation_checks(obs, grid, cfg)
    qc.to_csv(out / "qc_counts.csv", index=False)
    val.to_csv(out / "validation_checks.csv", index=False)
    summary = obs.groupby("period").agg(cell_months=("clara", "size"), months=("time", "nunique"),
                                        median_samples=("n_samples", "median"),
                                        median_overpasses=("n_overpasses", "median"),
                                        clara_mean=("clara", "mean"), era5_night_mean=("era5_night_r0", "mean"))
    summary.to_csv(out / "observations_summary.csv")
    nv = pd.read_csv(out / "era5_night_validation.csv")
    (out / "decisions.md").write_text(DECISIONS.format(
        cfg_path=cfg["_path"], res=res, sza=cfg["qc"]["night_sza_min"], vza=cfg["qc"]["vza_max"],
        maxdiff=nv.max_abs_diff_24h_vs_monthly.max(), olr_min=read_in.OLR_MIN, olr_from=read_in.OLR_MIN_FROM,
        olr_max=read_in.OLR_MAX, n_qc=int(s.qc.sum()), n_usable=int(s.usable.sum()),
        frac_qc=s.qc.sum() / s.usable.sum(), gap=cfg["qc"]["overpass_gap_min"],
        radii=", ".join(f"{r:g}" for r in cfg["era5"]["smoothing_radii_km"]),
        n_dark=era5_night.N_DARKEST, gap_start=cfg["time"]["gap_start"], gap_end=cfg["time"]["gap_end"],
        coarsen_note="" if res == 1 else f" -> {res} deg area means",
        res_note=(f"At 1 deg, {(obs.n_samples == 1).mean():.0%} of cell-months hold a single 30 s sample: the grid is much finer than CLARA's "
                  "effective resolution (~10^3 km, EDA), so 1 deg detail in the maps comes from ERA5 through beta1, "
                  "while the GP's latent field stays smooth on its fitted length scales." if res == 1 else
                  f"{res} deg is coarser than ERA5 and no finer than CLARA's effective footprint."),
        qc_table=md_table(qc), val_table=md_table(val.astype({"value": float}))))
    config.save(cfg, out / "config_used.json")
    print(qc.to_string(index=False))
    print(val.to_string(index=False))
    print(summary.round(1).to_string())
    print(f"wrote {len(obs):,} observations and {len(grid):,} prediction-grid cell-months; records in {out}")


if __name__ == "__main__":
    main()
