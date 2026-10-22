"""Compare the matched monthly means of CLARA, CERES EBAF and ERA5, on the cells CLARA sampled only.

The fields are those of build_matched.py: per res x res deg cell and month, the mean of the CLARA
Earth-viewing samples within [OLR_MIN, OLR_MAX] (OLR_MIN raised per OLR_MIN_FROM), and the CERES / ERA5 monthly means of that cell.
CERES and ERA5 are blanked wherever CLARA has no sample that month, so all three panels and
histograms cover the same cell-months.

Maps        side by side CLARA | CERES EBAF | ERA5, one figure per month
Histograms  as in plot_olr_hist.py (era5_ceres_clara): density step histograms of the cell values,
            overall and per year. All three are weighted by cos(lat), as they are now cell means.
            Drawn in W m-2 and standardised (z = (OLR - mean) / std per dataset and figure, so shapes
            compare independently of units, offset and scale).

Output, <res> in the file names:
    results/month_smooth_comparison/<yyyy>/comparison_<yyyy>-<mm>_<res>deg.png
    results/month_smooth_comparison/histograms/olr_hist_all_<res>deg.png
    results/month_smooth_comparison/histograms/olr_hist_<yyyy>_<res>deg.png
    results/month_smooth_comparison/histograms_normalised/olr_hist_all_<res>deg.png, olr_hist_<yyyy>_<res>deg.png

Usage:
    python src/plot_month_smooth.py            # 1 deg (CERES native) and 5 x 5 deg
    python src/plot_month_smooth.py --res 2    # any whole-degree spacing(s)
"""

import argparse

import numpy as np

from plot_olr_hist import grid_values, plot_compare
from plot_raw_means import RESULTS_DIR, load, plot_period, to_grid

OUT_DIR = RESULTS_DIR / "month_smooth_comparison"


def matched(clara, ceres, era5, res):
    """Monthly {name: Dataset} on the res deg grid, CERES and ERA5 kept only on CLARA cell-months."""
    monthly, _ = to_grid(clara, ceres, era5, res)
    grid = monthly["CLARA"]
    out = {"CLARA": grid}
    for name in ("CERES EBAF", "ERA5"):
        ds = monthly[name].sel(time=monthly[name].time.isin(grid.time))
        mask = grid["count"].sel(time=ds.time) > 0
        out[name] = ds.where(mask.assign_coords(lat=ds.lat, lon=ds.lon))
    return out


def maps(datasets, res):
    n = 0
    for t in datasets["CLARA"].indexes["time"]:
        path = OUT_DIR / f"{t:%Y}" / f"comparison_{t:%Y-%m}_{res}deg.png"
        n += plot_period(datasets, t, f"{t:%Y-%m} ({res}° × {res}° grid, CLARA cells only)", path, False,
                         title="Monthly mean OLR")
    print(f"wrote {n} monthly maps to {OUT_DIR}/<yyyy>")


def histograms(datasets, res):
    points = {name: grid_values(ds) for name, ds in datasets.items()}
    grid = f"{res}° cell monthly means on CLARA cells"
    years = np.unique(points["CLARA"][2])
    for folder, normalised, prefix in (("histograms", False, "OLR"),
                                       ("histograms_normalised", True, "Standardised OLR")):
        out_dir = OUT_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_compare({k: (v, w) for k, (v, w, _) in points.items()},
                     f"{prefix}, all years ({grid})", out_dir / f"olr_hist_all_{res}deg.png", normalised)
        for year in years:
            series = {k: (v[y == year], w[y == year]) for k, (v, w, y) in points.items() if (y == year).any()}
            plot_compare(series, f"{prefix}, {year} ({grid})", out_dir / f"olr_hist_{year}_{res}deg.png",
                         normalised)
        print(f"wrote {1 + len(years)} histograms to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--res", type=int, nargs="+", default=[1, 5],
                        help="common grid spacing(s) in whole degrees (default: 1 5)")
    args = parser.parse_args()

    clara, ceres, era5 = load()
    for res in args.res:
        datasets = matched(clara, ceres, era5, res)
        maps(datasets, res)
        histograms(datasets, res)


if __name__ == "__main__":
    main()
