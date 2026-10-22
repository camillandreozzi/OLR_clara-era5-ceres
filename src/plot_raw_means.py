"""Side-by-side maps of the raw mean OLR from CLARA, CERES EBAF and ERA5, per month and per year.

All three are put on a common res x res deg grid in W m-2, positive upward:
    CLARA  Earth-viewing samples within [OLR_MIN, OLR_MAX] (OLR_MIN raised per OLR_MIN_FROM) (read_in.read_clara), binned by footprint
           location; each cell is the mean of all samples that fell in it during the period.
    CERES  toa_lw_all_mon is already a monthly mean flux in W m-2; only area-averaged to the grid.
    ERA5   ttr is an accumulation in J m-2, negative upward; monthly means of daily accumulations are
           converted with OLR = -ttr / 86400 (read_in.read_era5), then area-averaged from the 0.25 deg
           points onto the CERES 1 deg cells (read_in.era5_to_ceres_grid) and on to the common grid.

Yearly CERES and ERA5 maps are day-weighted means of the available months; the yearly CLARA map
bins all of that year's samples. No sampling matching is applied - these are raw means.

Panel titles give the area-weighted mean over all cells with data, and for CERES/ERA5 also over
only the cells CLARA sampled in that period, which is the fairer comparison for sparse months.

Output, one folder per grid spacing:
    results/rawmeans_<res>/monthly/rawmeans_<yyyy>-<mm>.png
    results/rawmeans_<res>/yearly/rawmeans_<yyyy>.png

Usage:
    python src/plot_raw_means.py            # 1 deg (CERES native) and 5 x 5 deg averages
    python src/plot_raw_means.py --res 2    # any whole-degree spacing(s)
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

import read_in

try:
    import cartopy.crs as ccrs
except ImportError:  # coastlines are drawn only when cartopy is installed
    ccrs = None

RESULTS_DIR = read_in.ROOT / "results"

VMIN, VMAX = 100, 320  # W m-2, fixed so every figure shares one colour scale
# Single-hue sequential ramp (light = low OLR, dark = high OLR).
CMAP = LinearSegmentedColormap.from_list("olr_blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
CMAP.set_bad("#d6d5d0")  # neutral grey = no data
INK, MUTED = "#1f1f1e", "#6b6a66"


def area_mean(da, mask=None):
    """Area-weighted mean over cells with data (optionally only where mask is True)."""
    if mask is not None:
        da = da.where(mask)
    if not da.notnull().any():
        return np.nan
    w = np.cos(np.radians(da.lat)) * xr.ones_like(da.lon)
    return float(da.weighted(w.where(da.notnull(), 0)).mean())


def yearly_mean(ds):
    """Day-weighted yearly mean of a monthly Dataset; time is set to 1 Jan, with 'n_months'."""
    days = ds.time.dt.days_in_month
    grouped = (ds.olr * days).groupby("time.year").sum() / days.groupby("time.year").sum()
    n = ds.time.groupby("time.year").count()
    out = xr.Dataset({"olr": grouped, "n_months": n}).rename(year="time")
    return out.assign_coords(time=[pd.Timestamp(int(y), 1, 1) for y in out.time.values])


def load():
    """Return usable CLARA samples and CERES / ERA5 monthly OLR on the 1 deg CERES grid."""
    clara = read_in.read_clara()
    clara = clara[clara.earth_view & clara.in_range]
    return clara, read_in.read_ceres(), read_in.era5_to_ceres_grid(read_in.read_era5("monthly"))


def to_grid(clara, ceres, era5, res):
    """Put all three on the res x res deg grid; return (monthly, yearly) dicts of Datasets."""
    ceres = read_in.coarsen(ceres, res)
    era5 = read_in.coarsen(era5, res)
    monthly = {"CLARA": read_in.grid_clara(clara, res, "M"), "CERES EBAF": ceres, "ERA5": era5}
    yearly = {"CLARA": read_in.grid_clara(clara, res, "Y"), "CERES EBAF": yearly_mean(ceres),
              "ERA5": yearly_mean(era5)}
    return monthly, yearly


def _select(ds, t):
    return ds.sel(time=t) if t in ds.indexes["time"] else None


def _panel_title(name, sel, clara_mask, yearly):
    if sel is None:
        return f"{name}\nno data"
    if name == "CLARA":
        coverage = float((sel["count"] > 0).mean())
        return (f"{name}  ({int(sel['count'].sum()):,} samples, {coverage:.0%} of cells)\n"
                f"mean {area_mean(sel.olr):.1f} W m$^{{-2}}$")
    line = f"mean {area_mean(sel.olr):.1f}"
    if clara_mask is not None and bool(clara_mask.any()) and bool((sel.olr.notnull() & ~clara_mask).any()):
        line += f"  |  on CLARA cells {area_mean(sel.olr, clara_mask):.1f}"
    months = f"  ({int(sel.n_months)} months)" if yearly else ""
    return f"{name}{months}\n{line} W m$^{{-2}}$"


def plot_period(datasets, t, label, path, yearly, title="Raw mean OLR"):
    sels = {name: _select(ds, t) for name, ds in datasets.items()}
    if all(s is None for s in sels.values()):
        return False
    clara_mask = sels["CLARA"]["count"] > 0 if sels["CLARA"] is not None else None

    kw = {"subplot_kw": {"projection": ccrs.PlateCarree()}} if ccrs else {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), layout="constrained", **kw)
    ref = next(ds for ds in datasets.values())
    res = float(ref.lat[1] - ref.lat[0])
    lat_edges, lon_edges = read_in.grid_edges(round(res))

    mesh = None
    for ax, (name, sel) in zip(axes, sels.items()):
        values = sel.olr.values if sel is not None else np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan)
        extra = {"transform": ccrs.PlateCarree()} if ccrs else {}
        mesh = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(values),
                             cmap=CMAP, vmin=VMIN, vmax=VMAX, **extra)
        if ccrs:
            ax.coastlines(linewidth=0.5, color=INK)
            ax.set_global()
        else:
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_aspect("equal")
            ax.set_xticks(range(-180, 181, 60))
            ax.set_yticks(range(-90, 91, 30))
            ax.tick_params(colors=MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(MUTED)
        ax.set_title(_panel_title(name, sel, clara_mask, yearly), fontsize=10, color=INK, loc="left")

    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", shrink=0.4, aspect=40, extend="both", pad=0.04)
    cbar.set_label("OLR (W m$^{-2}$), grey = no data", color=INK, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    fig.suptitle(f"{title}, {label}", color=INK, fontsize=13, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--res", type=int, nargs="+", default=[1, 5],
                        help="common grid spacing(s) in whole degrees (default: 1 5)")
    args = parser.parse_args()

    clara, ceres, era5 = load()
    all_times = lambda d: sorted(set().union(*(ds.indexes["time"] for ds in d.values())))

    for res in args.res:
        monthly, yearly = to_grid(clara, ceres, era5, res)
        out_dir = RESULTS_DIR / f"rawmeans_{res}"
        grid = f"{res}° × {res}° grid"
        n = sum(plot_period(monthly, t, f"{t:%Y-%m} ({grid})", out_dir / "monthly" / f"rawmeans_{t:%Y-%m}.png", False)
                for t in all_times(monthly))
        print(f"wrote {n} monthly figures to {out_dir / 'monthly'}")
        n = sum(plot_period(yearly, t, f"{t:%Y} ({grid})", out_dir / "yearly" / f"rawmeans_{t:%Y}.png", True)
                for t in all_times(yearly))
        print(f"wrote {n} yearly figures to {out_dir / 'yearly'}")
    if ccrs is None:
        print("note: install cartopy to draw coastlines")


if __name__ == "__main__":
    main()
