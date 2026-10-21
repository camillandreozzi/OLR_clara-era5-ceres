"""Correlation between the CLARA, CERES EBAF and ERA5 monthly OLR fields, in three steps.

All three are put on a common res x res deg monthly grid exactly as in plot_raw_means.py (CLARA
Earth-viewing samples within [OLR_MIN, OLR_MAX] binned by footprint; CERES and ERA5 area-averaged).
CLARA samples are further limited to a viewing zenith angle of at most --max-vza (default 75 deg), as
near-limb views see a long slant path through the colder upper atmosphere. A CLARA cell-month is used when it holds at least --min-samples samples. Three pairs are compared,
each as y against x (bias = y - x):
    CLARA vs CERES EBAF,  CLARA vs ERA5,  ERA5 vs CERES EBAF

1. overall   All cell-months where both members of a pair have data, pooled: Pearson r and
             regression (area weighted, cos(lat)), Spearman r (unweighted), bias, RMSE, std ratio.
2. spatial   Per grid cell, the Pearson r of the two monthly time series: over all months, and over
             the months of each year. A cell needs MIN_MONTHS_ALL / MIN_MONTHS_YEAR months with
             both datasets, otherwise it is left blank. p-values are two-sided t-tests (n - 2 dof).
3. temporal  (a) per month, the pattern correlation across all cells where both have data
             (area weighted), and (b) per year, the spatial mean of that year's per-cell
             correlations from step 2 (area weighted, averaged in Fisher z = arctanh r); left out
             when fewer than MIN_CELLS cells have a correlation that year.

The correlations are of the raw fields, so they include the seasonal cycle (step 2) and the
equator-to-pole gradient (steps 1 and 3a).

Output (one set per grid spacing, <res> in the file names):
    results/correlation/overall/stats_<res>deg.csv, scatter_<res>deg.png
    results/correlation/spatial/corr_map_all_<res>deg.png, corr_map_<yyyy>_<res>deg.png,
                                corr_maps_<res>deg.nc (r, p and n for every map)
    results/correlation/temporal/pattern_corr_monthly_<res>deg.csv, local_corr_yearly_<res>deg.csv,
                                 corr_timeseries_<res>deg.png

Usage:
    python src/correlation.py               # 5 x 5 deg (CLARA is too sparse for per-cell r at 1 deg)
    python src/correlation.py --res 5 10    # any whole-degree spacing(s)
    python src/correlation.py --max-vza 90  # keep all Earth-viewing CLARA samples
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from plot_raw_means import CMAP as COUNT_CMAP, INK, MUTED, RESULTS_DIR, ccrs, load, to_grid

OUT_DIR = RESULTS_DIR / "correlation"

# (x, y) pairs; y is compared against x.
PAIRS = [("CERES EBAF", "CLARA"), ("ERA5", "CLARA"), ("CERES EBAF", "ERA5")]
PAIR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed categorical order: blue, orange, aqua

MIN_MONTHS_ALL = 12  # months with both datasets needed for a cell's all-years correlation
MIN_MONTHS_YEAR = 6  # ... and for a cell's correlation within one year
MIN_CELLS = 20  # cells needed for a month's pattern correlation and for a correlation map's mean
P_SIG = 0.05
MAX_VZA = 75.0  # deg, default CLARA viewing zenith angle limit
LIMS = (40, 400)  # W m-2, scatter axes; CLARA is kept within [50, 400]

# Diverging blue (positive r) <-> red (negative r) around a neutral grey midpoint.
CORR_CMAP = LinearSegmentedColormap.from_list("corr", [
    "#8f1f1f", "#e34948", "#f4b3b1", "#f0efec", "#9ec5f4", "#3987e5", "#0d366b"])
CORR_CMAP.set_bad("#bdbcb6")  # darker grey = too few months
GRID = "#e1e0d9"


def pair_label(x, y):
    return f"{y} vs {x}"


def slug(x, y):
    return f"{y}_vs_{x}".replace(" EBAF", "").replace(" ", "_")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def monthly_fields(monthly, min_samples):
    """Monthly OLR DataArrays (time, lat, lon) of all three products on one time axis and grid."""
    clara = monthly["CLARA"]
    ref = clara.olr
    fields = {"CLARA": ref.where(clara["count"] >= min_samples)}
    for name in ("CERES EBAF", "ERA5"):
        da = monthly[name].olr
        # coarsen() and grid_clara() build the cell centres differently; snap them to one set.
        np.testing.assert_allclose(da.lat, ref.lat, atol=1e-6)
        np.testing.assert_allclose(da.lon, ref.lon, atol=1e-6)
        fields[name] = da.assign_coords(lat=ref.lat, lon=ref.lon)
    aligned = xr.align(*fields.values(), join="outer")
    t = aligned[0].indexes["time"]
    months = pd.date_range(t.min(), t.max(), freq="MS")  # full axis, so gaps show as gaps
    return {k: v.reindex(time=months).transpose("time", "lat", "lon") for k, v in zip(fields, aligned)}


def area_weights(da):
    return (np.cos(np.radians(da.lat)) * xr.ones_like(da)).transpose(*da.dims)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def agreement(x, y, w):
    """Agreement of y with x (finite 1-D arrays) with weights w; r and the regression are weighted."""
    keys = ["n", "r", "spearman_r", "mean_x", "mean_y", "bias", "rmse", "std_ratio", "slope", "intercept"]
    if len(x) < 3:
        return dict.fromkeys(keys, np.nan) | {"n": len(x)}
    w = w / w.sum()
    mx, my = w @ x, w @ y
    dx, dy = x - mx, y - my
    vx, vy, cxy = w @ dx ** 2, w @ dy ** 2, w @ (dx * dy)
    slope = cxy / vx
    return {"n": len(x), "r": cxy / np.sqrt(vx * vy), "spearman_r": stats.spearmanr(x, y).statistic,
            "mean_x": mx, "mean_y": my, "bias": my - mx, "rmse": np.sqrt(w @ (y - x) ** 2),
            "std_ratio": np.sqrt(vy / vx), "slope": slope, "intercept": my - slope * mx}


def pooled(x, y):
    """Finite (x, y, cos-lat weight) values of all cell-months where both have data."""
    ok = (x.notnull() & y.notnull()).values
    return x.values[ok], y.values[ok], area_weights(x).values[ok]


def local_corr(x, y, min_n):
    """Per-cell Pearson r of two (time, lat, lon) series over time, with p-value and n."""
    ok = x.notnull() & y.notnull()
    x, y = x.where(ok), y.where(ok)
    n = ok.sum("time")
    dx, dy = x - x.mean("time"), y - y.mean("time")
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (dx * dy).sum("time") / np.sqrt((dx ** 2).sum("time") * (dy ** 2).sum("time"))
        r = r.where((n >= min_n) & np.isfinite(r)).clip(-1, 1)
        t = r * np.sqrt((n - 2) / (1 - r ** 2))
    p = xr.apply_ufunc(lambda t, n: 2 * stats.t.sf(np.abs(t), n - 2), t, n).where(r.notnull())
    return xr.Dataset({"r": r, "p": p, "n": n})


def fisher_mean(r):
    """Area-weighted mean of a correlation map, averaged in Fisher z."""
    if not r.notnull().any():
        return np.nan
    z = np.arctanh(r.clip(-0.9999, 0.9999))
    w = np.cos(np.radians(r.lat)) * xr.ones_like(r.lon)
    return float(np.tanh(z.weighted(w.where(r.notnull(), 0)).mean()))


def area_fraction(mask, valid=None):
    """Area-weighted fraction of cells where mask is True (out of the valid cells, default all)."""
    w = np.cos(np.radians(mask.lat)) * xr.ones_like(mask.lon)
    if valid is not None:
        w = w.where(valid, 0)
    return float((w * mask).sum() / w.sum()) if float(w.sum()) > 0 else np.nan


# ---------------------------------------------------------------------------
# 1. Overall
# ---------------------------------------------------------------------------

def overall(fields, res, grid):
    out_dir = OUT_DIR / "overall"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(16, 5.4), layout="constrained")
    for ax, (xn, yn) in zip(axes, PAIRS):
        x, y, w = pooled(fields[xn], fields[yn])
        s = agreement(x, y, w)
        rows[pair_label(xn, yn)] = s
        hb = ax.hexbin(x, y, gridsize=70, bins="log", mincnt=1, cmap=COUNT_CMAP, extent=LIMS + LIMS)
        ax.plot(LIMS, LIMS, color=MUTED, lw=1, ls="--", label="1:1")
        ax.plot(LIMS, s["intercept"] + s["slope"] * np.array(LIMS), color=INK, lw=1.5, label="weighted fit")
        ax.set(xlim=LIMS, ylim=LIMS, aspect="equal",
               xlabel=f"{xn} OLR (W m$^{{-2}}$)", ylabel=f"{yn} OLR (W m$^{{-2}}$)")
        ax.set_title(pair_label(xn, yn), color=INK, fontsize=11, loc="left")
        ax.text(0.03, 0.97,
                f"r = {s['r']:.3f}   Spearman = {s['spearman_r']:.3f}\n"
                f"bias = {s['bias']:+.1f}   RMSE = {s['rmse']:.1f} W m$^{{-2}}$\n"
                f"y = {s['slope']:.2f} x {s['intercept']:+.1f}   n = {s['n']:,}",
                transform=ax.transAxes, va="top", fontsize=9, color=INK,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        ax.legend(loc="lower right", fontsize=8, labelcolor=INK, framealpha=0.85, edgecolor="none")
        _style(ax)
    cbar = fig.colorbar(hb, ax=axes, shrink=0.8, aspect=30)
    cbar.set_label("cell-months per bin", color=INK, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    fig.suptitle(f"Monthly OLR, all cell-months with both datasets ({grid})",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.savefig(out_dir / f"scatter_{res}deg.png", dpi=130, facecolor="white")
    plt.close(fig)

    table = pd.DataFrame(rows).T.rename_axis("pair")
    table.to_csv(out_dir / f"stats_{res}deg.csv", float_format="%.4f")
    print(f"\noverall ({res} deg):\n{table[['n', 'r', 'spearman_r', 'bias', 'rmse', 'slope']].round(3).to_string()}")
    print(f"wrote {out_dir}")


# ---------------------------------------------------------------------------
# 2. Spatial
# ---------------------------------------------------------------------------

def correlation_maps(fields):
    """{period: {(x, y): Dataset(r, p, n)}} for period 'all' and each year."""
    years = np.unique(fields["CERES EBAF"].time.dt.year)
    maps = {"all": {(x, y): local_corr(fields[x], fields[y], MIN_MONTHS_ALL) for x, y in PAIRS}}
    for year in years:
        sel = {k: v.sel(time=v.time.dt.year == year) for k, v in fields.items()}
        maps[int(year)] = {(x, y): local_corr(sel[x], sel[y], MIN_MONTHS_YEAR) for x, y in PAIRS}
    return maps


def plot_maps(pair_maps, title, min_n, path):
    kw = {"subplot_kw": {"projection": ccrs.PlateCarree()}} if ccrs else {}
    fig, axes = plt.subplots(1, len(pair_maps), figsize=(18, 4.6), layout="constrained", **kw)
    for ax, ((xn, yn), ds) in zip(axes, pair_maps.items()):
        res = float(ds.lat[1] - ds.lat[0])
        lon_edges = np.arange(-180, 180 + res / 2, res)
        lat_edges = np.arange(-90, 90 + res / 2, res)
        extra = {"transform": ccrs.PlateCarree()} if ccrs else {}
        mesh = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(ds.r.values),
                             cmap=CORR_CMAP, vmin=-1, vmax=1, **extra)
        if ccrs:
            ax.coastlines(linewidth=0.5, color=INK)
            ax.set_global()
        else:
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_aspect("equal")
            ax.set_xticks(range(-180, 181, 60))
            ax.set_yticks(range(-90, 91, 30))
            _style(ax, grid=False)
        valid = ds.r.notnull()
        if valid.sum() >= MIN_CELLS:
            sub = (f"mean r {fisher_mean(ds.r):.2f}  |  {area_fraction(valid):.0%} of area, "
                   f"{area_fraction(ds.p < P_SIG, valid):.0%} of it p < {P_SIG:g}")
        else:
            sub = f"only {int(valid.sum())} cells with ≥ {min_n} months, too few for a mean"
        ax.set_title(f"{pair_label(xn, yn)}\n{sub}", fontsize=10, color=INK, loc="left")
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", shrink=0.4, aspect=40, pad=0.04)
    cbar.set_label(f"Pearson r of the monthly series, grey = fewer than {min_n} months with both",
                   color=INK, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def spatial(maps, res, grid):
    out_dir = OUT_DIR / "spatial"
    out_dir.mkdir(parents=True, exist_ok=True)
    for period, pair_maps in maps.items():
        if period == "all":
            title, min_n = f"Per-cell correlation over all months ({grid})", MIN_MONTHS_ALL
        else:
            title, min_n = f"Per-cell correlation over the months of {period} ({grid})", MIN_MONTHS_YEAR
        plot_maps(pair_maps, title, min_n, out_dir / f"corr_map_{period}_{res}deg.png")

    ds = xr.Dataset()
    for period, pair_maps in maps.items():
        for (xn, yn), m in pair_maps.items():
            for v in m.data_vars:
                ds[f"{v}_{slug(xn, yn)}_{period}"] = m[v]
    ds.attrs = {"description": "Per-cell Pearson r (and two-sided p, n months) of monthly OLR; "
                               "<var>_<y>_vs_<x>_<all|yyyy>", "min_months_all": MIN_MONTHS_ALL,
                "min_months_year": MIN_MONTHS_YEAR}
    ds.to_netcdf(out_dir / f"corr_maps_{res}deg.nc")
    print(f"wrote {len(maps)} maps and their data to {out_dir}")


# ---------------------------------------------------------------------------
# 3. Temporal
# ---------------------------------------------------------------------------

def pattern_corr(fields):
    """Per month and pair, the area-weighted agreement across all cells where both have data."""
    rows = []
    for t in fields["CERES EBAF"].time.values:
        for xn, yn in PAIRS:
            x, y, w = pooled(fields[xn].sel(time=t), fields[yn].sel(time=t))
            s = agreement(x, y, w) if len(x) >= MIN_CELLS else agreement(x[:0], y[:0], w[:0]) | {"n": len(x)}
            rows.append({"time": pd.Timestamp(t), "pair": pair_label(xn, yn)} | s)
    return pd.DataFrame(rows).rename(columns={"n": "n_cells"})


def local_corr_by_year(maps):
    rows = []
    for period, pair_maps in maps.items():
        for (xn, yn), m in pair_maps.items():
            valid = m.r.notnull()
            mean_r = fisher_mean(m.r) if valid.sum() >= MIN_CELLS else np.nan
            rows.append({"year": period, "pair": pair_label(xn, yn), "mean_r": mean_r,
                         "n_cells": int(valid.sum()), "area_fraction": area_fraction(valid),
                         "sig_fraction": area_fraction(m.p < P_SIG, valid) if valid.any() else np.nan})
    return pd.DataFrame(rows)


def temporal(fields, maps, res, grid):
    out_dir = OUT_DIR / "temporal"
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly = pattern_corr(fields)
    yearly = local_corr_by_year(maps)
    monthly.to_csv(out_dir / f"pattern_corr_monthly_{res}deg.csv", index=False, float_format="%.4f")
    yearly.to_csv(out_dir / f"local_corr_yearly_{res}deg.csv", index=False, float_format="%.4f")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, layout="constrained")
    by_year = yearly[yearly.year != "all"]
    for (xn, yn), color in zip(PAIRS, PAIR_COLORS):
        label = pair_label(xn, yn)
        m = monthly[monthly.pair == label]
        ax1.plot(m.time, m.r, color=color, lw=2, marker="o", ms=3.5, label=label)
        y = by_year[by_year.pair == label]
        ax2.plot([pd.Timestamp(int(v), 7, 1) for v in y.year], y.mean_r, color=color, lw=2, marker="o",
                 ms=7, mec="white", mew=1.5, label=label)
    ax1.set_title("(a) Pattern correlation per month: cells with both datasets, area weighted",
                  color=INK, fontsize=10, loc="left")
    ax1.set_ylabel("Pearson r across cells", color=INK)
    ax2.set_title(f"(b) Mean of the per-cell correlations within each year (step 2; cells with "
                  f"≥ {MIN_MONTHS_YEAR} months, Fisher z, area weighted)", color=INK, fontsize=10, loc="left")
    ax2.set_ylabel("mean per-cell r", color=INK)
    for ax in (ax1, ax2):
        _style(ax)
    ax1.legend(loc="lower left", frameon=False, fontsize=9, ncol=len(PAIRS), labelcolor=INK)
    fig.suptitle(f"Correlation through time ({grid})", color=INK, fontsize=13, x=0.01, ha="left")
    fig.savefig(out_dir / f"corr_timeseries_{res}deg.png", dpi=130, facecolor="white")
    plt.close(fig)

    summary = monthly.groupby("pair").r.describe()[["count", "mean", "min", "max"]]
    print(f"\nmonthly pattern correlation ({res} deg):\n{summary.round(3).to_string()}")
    print(f"\nyearly mean per-cell correlation ({res} deg):\n"
          f"{yearly.pivot(index='year', columns='pair', values='mean_r').round(3).to_string()}")
    print(f"wrote {out_dir}")


def _style(ax, grid=True):
    ax.tick_params(colors=MUTED, labelsize=8)
    for side, spine in ax.spines.items():
        spine.set_color("#c3c2b7")
        spine.set_visible(side in ("left", "bottom") or not grid)
    if grid:
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--res", type=int, nargs="+", default=[5],
                        help="common grid spacing(s) in whole degrees (default: 5)")
    parser.add_argument("--min-samples", type=int, default=1,
                        help="CLARA samples needed for a cell-month to be used (default: 1)")
    parser.add_argument("--max-vza", type=float, default=MAX_VZA,
                        help=f"largest CLARA viewing zenith angle kept, deg (default: {MAX_VZA:g})")
    args = parser.parse_args()

    clara, ceres, era5 = load()
    n = len(clara)
    clara = clara[clara.vza <= args.max_vza]
    print(f"CLARA: kept {len(clara):,} of {n:,} samples with vza <= {args.max_vza:g} deg")
    for res in args.res:
        monthly, _ = to_grid(clara, ceres, era5, res)
        fields = monthly_fields(monthly, args.min_samples)
        grid = f"{res}° × {res}° grid, CLARA vza ≤ {args.max_vza:g}°"
        overall(fields, res, grid)
        maps = correlation_maps(fields)
        spatial(maps, res, grid)
        temporal(fields, maps, res, grid)
    if ccrs is None:
        print("note: install cartopy to draw coastlines")


if __name__ == "__main__":
    main()
