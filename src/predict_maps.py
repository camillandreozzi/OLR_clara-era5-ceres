"""Phase 6: monthly maps of predicted CLARA OLR and its uncertainty from the winning kernel's full-data fit.

The model is fitted at monthly resolution, so each prediction is directly the monthly mean CLARA
night-side OLR of a cell; no daily-to-monthly aggregation of variances is involved. Two uncertainties:

    latent_sd     SD of the GP posterior of the monthly-mean CLARA OLR (what the uncertainty maps show)
    pred_sd_obs1  SD for a new noisy CLARA cell-month mean from ONE overpass = sqrt(latent_var + sigma0^2 +
                  sigma1^2); a cell-month with n overpasses would have noise sigma0^2 + sigma1^2 / n

Residuals are observed - predicted: in-sample (full-data fit, the observation was used to fit) and
out-of-sample (spatial block CV, the observation's block and buffer were held out).

The prediction grid covers every lat-lon cell of every month. Polar-day cells (no night at all that month)
use the darkest-hours ERA5 fallback (era5_night.py) and are never observed by CLARA: their predictions are
extrapolation and are hatched on the maps (night_fallback in the NetCDF/CSV). The months of the 2021-22 CLARA
gap are never observed either; with the gap step they take the post-gap level (post_gap = 1).

Outputs, results/model/<kernel>/:
    grid_predictions.csv          every cell of every month in one table (time, lat, lon sorted)
    <yyyy>/<yyyy>_<mm>_predicted_olr.png, _uncertainty.png        every month
    <yyyy>/<yyyy>_<mm>_observed_olr.png, _residuals.png           months with CLARA observations
    <yyyy>/<yyyy>_predictions.nc, <yyyy>/<yyyy>_<mm>_predictions.csv (one CSV per month)
    global_mean_monthly.csv / .png (area-weighted global mean with the variance from the full posterior
    covariance, compared with the two naive alternatives)

Usage:
    python src/predict_maps.py               # winner from results/model/kernel_comparison.csv
    python src/predict_maps.py --kernel matern32
"""

import argparse

import numpy as np
import pandas as pd
import torch
import xarray as xr

import config
import gp
import read_in
import run_cv
from style import (CLARA_UNIT, DIVERGING, INK, KERNEL_LABELS, MUTED, OLR_CMAP, SERIES, UNCERTAINTY_CMAP, ccrs,
                   colorbar, draw_field, map_axes, plt, save, style_axes, title)


def load_model(kernel):
    st = torch.load(config.MODEL_DIR / kernel / "full_model.pt", weights_only=False)
    return gp.GP.from_state(st), st["era5_radius_km"]


def predict_grid(cfg, model, radius):
    grid = pd.read_parquet(config.PROCESSED_DIR / f"prediction_grid_{cfg['grid']['res_deg']}deg_monthly.parquet")
    e = grid[f"era5_night_r{int(radius)}"].to_numpy()
    ok = np.isfinite(e)
    if not ok.all():
        raise RuntimeError(f"{(~ok).sum()} prediction-grid cell-months have no ERA5 predictor; the grid must be gap-free")
    x = gp.embed(grid.lat, grid.lon, grid.model_time, cfg["time"]["origin"])
    p = model.predict(x[ok], e[ok], np.ones(ok.sum()), s=grid.post_gap.to_numpy()[ok])
    for col, v in (("pred_mean", p["mean"]), ("mean_fn", p["mean_fn"]), ("latent_sd", np.sqrt(p["latent_var"])),
                   ("pred_sd_obs1", np.sqrt(p["obs_var"]))):
        grid[col] = np.nan
        grid.loc[ok, col] = v
    grid["era5_night"] = e
    grid["residual_insample"] = np.where(grid.clara_observed, grid.clara - grid.pred_mean, np.nan)
    files =sorted((config.MODEL_DIR / model.family / "folds").glob("spatial_[0-9][0-9].parquet"))
    if files:
        oos = pd.concat([pd.read_parquet(f, columns=["obs_id", "residual"]) for f in files])
        grid = grid.merge(oos.rename(columns={"residual": "residual_oos_spatial"}), on="obs_id", how="left")
        grid.loc[~grid.clara_observed, "residual_oos_spatial"] = np.nan
    else:
        grid["residual_oos_spatial"] = np.nan
    return grid, x, ok


def to_dataset(g, res):
    lat_c, lon_c = read_in.grid_centres(res)
    months = sorted(g.time.unique())
    shape = (len(months), len(lat_c), len(lon_c))
    ti = pd.DatetimeIndex(months).get_indexer(g.time)
    i = np.round((g.lat.to_numpy() + 90 - res / 2) / res).astype(int)
    j = np.round((g.lon.to_numpy() + 180 - res / 2) / res).astype(int)
    ds = xr.Dataset(coords={"time": months, "lat": lat_c, "lon": lon_c})
    spec = {"pred_mean": ("W m-2 sr-1", "GP posterior mean of monthly-mean CLARA night-side OLR"),
            "mean_fn": ("W m-2 sr-1", "mean function beta0 + beta1 ERA5 + beta2 post_gap of the prediction"),
            "latent_sd": ("W m-2 sr-1", "posterior SD of the monthly-mean CLARA OLR (latent GP)"),
            "pred_sd_obs1": ("W m-2 sr-1", "predictive SD of a new CLARA cell-month mean from one overpass"),
            "clara": ("W m-2 sr-1", "observed CLARA cell-month mean OLR, a radiance (QC'd samples)"),
            "n_overpasses": ("1", "CLARA overpasses behind the observation"),
            "residual_insample": ("W m-2 sr-1", "observed - predicted, full-data fit (in-sample)"),
            "residual_oos_spatial": ("W m-2 sr-1", "observed - predicted, spatial block CV (out-of-sample)"),
            "era5_night": ("W m-2", "ERA5 night-time OLR used as predictor (darkest hours where night_fallback)"),
            "post_gap": ("1", "1 from the start of the 2021-22 CLARA gap on (post-gap calibration regime), 0 before"),
            "clara_observed": ("1", "1 where CLARA observed the cell-month"),
            "night_fallback": ("1", "1 where the cell has no night that month (polar day): extrapolation")}
    for v, (units, name) in spec.items():
        a = np.full(shape, np.nan)
        a[ti, i, j] = g[v].astype(float).to_numpy()
        ds[v] = (("time", "lat", "lon"), a)
        ds[v].attrs = {"units": units, "long_name": name}
    return ds


def hatch(ax, mask, res):
    """Hatch the cells where mask is True (polar-day fallback)."""
    if mask is None or not np.nanmax(mask) > 0:
        return
    lat_c, lon_c = read_in.grid_centres(res)
    extra = {"transform": ccrs.PlateCarree()} if ccrs is not None else {}
    ax.contourf(lon_c, lat_c, np.nan_to_num(mask), levels=[0.5, 1.5], colors="none", hatches=["////"], **extra)


def map_one(values, res, cmap, vmin, vmax, label, head, path, extend="both", fallback=None):
    fig = plt.figure(figsize=(9, 5), layout="constrained")
    ax = map_axes(fig, 1, 1, 1)
    mesh = draw_field(ax, values, res, cmap, vmin, vmax)
    hatch(ax, fallback, res)
    colorbar(fig, mesh, ax, label, extend=extend, shrink=0.7)
    title(ax, head, 10)
    save(fig, path, dpi=110)


def residual_maps(ins, oos, res, lim, head, path):
    fig = plt.figure(figsize=(14, 4.6), layout="constrained")
    for k, (v, lab) in enumerate(((ins, "in-sample (full-data fit)"), (oos, "out-of-sample (spatial block CV)"))):
        ax = map_axes(fig, 1, 2, k + 1)
        mesh = draw_field(ax, v, res, DIVERGING, -lim, lim)
        ok = np.isfinite(v)
        title(ax, f"{head}\n{lab}: RMSE {np.sqrt(np.nanmean(v ** 2)):.1f} {CLARA_UNIT}, n = {int(ok.sum())}"
              if ok.any() else f"{head}\n{lab}: no data", 9)
    colorbar(fig, mesh, fig.axes, f"residual = observed − predicted ({CLARA_UNIT}); grey = no CLARA observation",
             shrink=0.5)
    save(fig, path, dpi=110)


def global_mean(cfg, model, grid, x, ok, radius):
    """Area-weighted global mean per month with the posterior variance from the full covariance."""
    rows = []
    e = grid[f"era5_night_r{int(radius)}"].to_numpy()
    s = grid.post_gap.to_numpy()
    groups = grid[ok].groupby("time").groups
    # within one month every cell has the same time coordinates, so w^T K w is the same for every month
    first = np.asarray(next(iter(groups.values())))
    prior_quad = model.prior_quadratic(x[first], grid.cell_area_km2.to_numpy()[first])
    for t, idx in groups.items():
        idx = np.asarray(idx)
        w = grid.cell_area_km2.to_numpy()[idx]
        same_cells = len(idx) == len(first)
        m, v = model.aggregate(x[idx], e[idx], w, prior_quad if same_cells else None, s=s[idx])
        wn = w / w.sum()
        lv = grid.latent_sd.to_numpy()[idx] ** 2
        fb = grid.night_fallback.to_numpy()[idx]
        rows.append({"time": t, "global_mean": m, "sd_full_covariance": np.sqrt(v),
                     "sd_if_independent": np.sqrt(wn ** 2 @ lv), "sd_mean_of_variances": np.sqrt(wn @ lv),
                     "polar_day_area_fraction": wn @ fb})
    return pd.DataFrame(rows)


GRID_COLUMNS = ["year", "month", "time", "lat", "lon", "pred_mean", "latent_sd", "pred_sd_obs1", "mean_fn", "era5_night",
                "post_gap", "clara_observed", "clara", "n_overpasses", "night_fallback", "residual_insample",
                "residual_oos_spatial", "cell_area_km2"]


def write_grid_table(cfg, grid, path):
    """Every cell of every month in one CSV, checked to be the full lat x lon x month product."""
    res = cfg["grid"]["res_deg"]
    months = len(pd.period_range(cfg["time"]["first_month"], cfg["time"]["last_month"], freq="M"))
    expected = months * int(180 / res) * int(360 / res)
    table = grid[GRID_COLUMNS].sort_values(["time", "lat", "lon"]).astype({"clara_observed": int, "night_fallback": int})
    if len(table) != expected or table.duplicated(["time", "lat", "lon"]).any() or table.pred_mean.isna().any():
        raise RuntimeError(f"prediction table has {len(table):,} rows, expected {expected:,} unique cell-months with a "
                           "prediction each")
    table.to_csv(path, index=False, float_format="%.3f")
    return len(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel", choices=gp.KERNELS)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--config", help="configuration file (default config/pipeline.toml)")
    args = parser.parse_args()
    cfg = config.load(args.config)
    config.MODEL_DIR = config.RESULTS_DIR / cfg.get("output", {}).get("model_dir", "model")
    torch.set_num_threads(cfg["gp"]["threads"])
    res = cfg["grid"]["res_deg"]
    kernel = args.kernel or pd.read_csv(config.MODEL_DIR / "kernel_comparison.csv").kernel.iloc[0]
    model, radius = load_model(kernel)
    grid, x, ok = predict_grid(cfg, model, radius)
    out = config.MODEL_DIR / kernel
    n_rows = write_grid_table(cfg, grid, out / "grid_predictions.csv")
    print(f"wrote {n_rows:,} cell-months (every cell of every month) to {out / 'grid_predictions.csv'}")

    vals = np.concatenate([grid.pred_mean.dropna(), grid.clara.dropna()])
    vmin, vmax = np.percentile(vals, [1, 99]).round(-1)
    umax = np.ceil(np.nanpercentile(grid.latent_sd, 99))
    rlim = np.ceil(np.nanpercentile(np.abs(np.r_[grid.residual_insample.dropna(), grid.residual_oos_spatial.dropna()]),
                                    98) / 5) * 5
    label = KERNEL_LABELS[kernel]
    for year, g in grid.groupby("year"):
        ydir = out / str(year)
        ydir.mkdir(parents=True, exist_ok=True)
        ds = to_dataset(g, res)
        ds.attrs = {"kernel": kernel, "era5_radius_km": radius, "estimand": "monthly-mean CLARA night-side OLR",
                    "source": "src/predict_maps.py"}
        ds.to_netcdf(ydir / f"{year}_predictions.nc")
        for old in ydir.glob(f"{year}_predictions*.csv"):
            old.unlink()
        for t, gm in g.groupby("time"):
            gm.drop(columns=["obs_id"]).to_csv(ydir / f"{pd.Timestamp(t):%Y_%m}_predictions.csv", index=False,
                                               float_format="%.3f")
        if args.no_figures:
            continue
        for k, t in enumerate(pd.DatetimeIndex(ds.time.values)):
            stem = ydir / f"{t:%Y_%m}"
            n_obs = int(np.nansum(ds.clara_observed.values[k]))
            head = f"{t:%Y-%m} · {label} GP" + ("" if n_obs else " · no CLARA data this month (extrapolation)")
            fb = ds.night_fallback.values[k]
            map_one(ds.pred_mean.values[k], res, OLR_CMAP, vmin, vmax,
                    f"predicted CLARA OLR ({CLARA_UNIT}): posterior mean of the monthly-mean night-side OLR; "
                    "hatched = polar day (extrapolation)", f"{head} · predicted CLARA OLR",
                    stem.with_name(stem.name + "_predicted_olr.png"), fallback=fb)
            map_one(ds.latent_sd.values[k], res, UNCERTAINTY_CMAP, 0, umax,
                    f"posterior SD of the monthly-mean CLARA OLR (latent GP, {CLARA_UNIT}); excludes observation noise; "
                    "hatched = polar day", f"{head} · predictive uncertainty (latent SD)",
                    stem.with_name(stem.name + "_uncertainty.png"), extend="max", fallback=fb)
            if n_obs:
                map_one(ds.clara.values[k], res, OLR_CMAP, vmin, vmax,
                        f"observed CLARA OLR ({CLARA_UNIT}): mean of QC'd night-side samples; grey = not observed",
                        f"{head} · observed CLARA OLR ({n_obs} cells)", stem.with_name(stem.name + "_observed_olr.png"))
                residual_maps(ds.residual_insample.values[k], ds.residual_oos_spatial.values[k], res, rlim,
                              f"{head} · residuals", stem.with_name(stem.name + "_residuals.png"))
        print(f"{year}: maps written to {ydir}")

    gm = global_mean(cfg, model, grid, x, ok, radius)
    gm.to_csv(out / "global_mean_monthly.csv", index=False, float_format="%.4f")
    fig, ax = plt.subplots(figsize=(11, 4.4), layout="constrained")
    ax.fill_between(gm.time, gm.global_mean - 2 * gm.sd_full_covariance, gm.global_mean + 2 * gm.sd_full_covariance,
                    color=SERIES[0], alpha=0.2, lw=0, label="±2 SD, full posterior covariance")
    ax.plot(gm.time, gm.global_mean, color=SERIES[0], lw=2, label="predicted global mean (all cells)")
    ax.plot(gm.time, gm.global_mean + 2 * gm.sd_if_independent, color=MUTED, lw=1, ls=":",
            label="+2 SD if cells were independent (too narrow)")
    ax.set_ylabel(f"area-weighted global mean ({CLARA_UNIT})", color=INK)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    title(ax, f"{label} GP: monthly global mean of predicted CLARA OLR (polar-day cells: fallback, extrapolated)")
    save(fig, out / "global_mean_monthly.png")
    print(f"wrote predictions for {grid.year.nunique()} years" + ("" if args.no_figures else ", maps for every month"))


if __name__ == "__main__":
    main()
