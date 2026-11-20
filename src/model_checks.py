"""Checks of the winning GP that need no refit: interval coverage by group, and the polar-day cells in G_t.

1. Coverage of the central 95 % predictive interval (cv.metrics, the same definition as metrics_summary.csv) of
   the held-out cell-months of each CV type, split by calibration period (pre-gap / post-gap) and by the number
   of CLARA overpasses behind the observation (1, 2-3, >= 4), from oos_predictions_<cv>.csv.
2. Polar-day cells: their area share per month (global_mean_monthly.csv, polar_day_area_fraction), and the
   monthly mean of the posterior mean map over the cells that are not polar day, compared with G_t over all
   cells. Both are area-weighted means of grid_predictions.csv's pred_mean, so no model is needed.

Outputs, results/model/<winner>/:
    coverage_by_group.csv          cv_type, grouping (period / overpasses), group, n, cover95
    global_mean_no_polar_day.csv   per month: G_t, the mean without polar-day cells, their difference, polar share
    polar_day_summary.csv          polar-day share (min / median / max) and the difference (max |.|, mean, mean |.|)

Usage:
    python src/model_checks.py
"""

import numpy as np
import pandas as pd

import config
import cv
import kernel_global_means

CV_TYPES = ("spatial", "temporal", "joint")
OVERPASS_GROUPS = {"1": (1, 1), "2-3": (2, 3), ">=4": (4, np.inf)}


def coverage_by_group(d):
    rows = []
    for t in CV_TYPES:
        p = pd.concat([pd.read_csv(f) for f in sorted(d.glob(f"oos_predictions_{t}*.csv"))], ignore_index=True)
        groups = [("period", g, p.period == g) for g in ("pre-gap", "post-gap")] + \
                 [("overpasses", g, p.n_overpasses.between(lo, hi)) for g, (lo, hi) in OVERPASS_GROUPS.items()]
        for grouping, g, sel in groups:
            q = p[sel]
            m = cv.metrics(q.clara_obs, q.pred_mean, q.pred_var_obs, q.cell_area_km2, q.mean_fn)
            rows.append({"cv_type": t, "grouping": grouping, "group": g, "n": len(q), "cover95": m["cover95"]})
    return pd.DataFrame(rows)


def polar_day(d, cfg):
    grid = pd.read_csv(d / "grid_predictions.csv", usecols=["time", "pred_mean", "cell_area_km2", "night_fallback"],
                       parse_dates=["time"])
    gm = pd.read_csv(d / "global_mean_monthly.csv", parse_dates=["time"]).set_index("time")
    wsum = lambda g: np.average(g.pred_mean, weights=g.cell_area_km2)
    out = pd.DataFrame({"global_mean": gm.global_mean, "polar_day_area_fraction": gm.polar_day_area_fraction,
                        "mean_all_cells_from_grid": grid.groupby("time").apply(wsum, include_groups=False),
                        "mean_without_polar_day": grid[grid.night_fallback == 0].groupby("time").apply(
                            wsum, include_groups=False)})
    # the grid table is rounded to 3 decimals; G_t from it must equal the stored G_t to that precision
    check = (out.mean_all_cells_from_grid - out.global_mean).abs().max()
    if check > 1e-2:
        raise RuntimeError(f"G_t from grid_predictions.csv differs from global_mean_monthly.csv by {check:.4f}")
    out["diff"] = out.mean_without_polar_day - out.global_mean
    out["period"] = kernel_global_means.period(out.index, cfg)
    s = out.polar_day_area_fraction
    summary = pd.DataFrame([{
        "months": len(out), "months_with_polar_day": int((s > 0).sum()), "polar_share_min": s.min(),
        "polar_share_median": s.median(), "polar_share_max": s.max(), "month_of_max_share": f"{s.idxmax():%Y-%m}",
        "max_abs_diff": out["diff"].abs().max(), "month_of_max_diff": f"{out['diff'].abs().idxmax():%Y-%m}",
        "mean_diff": out["diff"].mean(), "mean_abs_diff": out["diff"].abs().mean(),
        "grid_vs_stored_max_abs": check}])
    return out, summary


def main():
    cfg = config.load()
    winner = pd.read_csv(config.MODEL_DIR / "kernel_comparison.csv").kernel.iloc[0]
    d = config.MODEL_DIR / winner
    cov = coverage_by_group(d)
    cov.to_csv(d / "coverage_by_group.csv", index=False, float_format="%.5f")
    gm, summ = polar_day(d, cfg)
    gm.rename_axis("time").to_csv(d / "global_mean_no_polar_day.csv", float_format="%.4f")
    summ.to_csv(d / "polar_day_summary.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 200)
    print(cov.round(4).to_string(index=False), "\n")
    print(summ.round(4).T.to_string())


if __name__ == "__main__":
    main()
