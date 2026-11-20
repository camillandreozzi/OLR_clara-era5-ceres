"""Every figure of the paper (docs/paper/main.tex), drawn at its print size from the pipeline's tables.

One routine for all 17 figures (src/style.py):
  - width 8.5 cm (one column) or 17.5 cm (two columns); main.tex includes each file at its natural size, never
    scaled, so what is drawn here is what is printed
  - every text >= 8 pt at that size; save_paper() refuses a figure that breaks this or the width rule
  - one unit and one colour scale per quantity in every figure: CLARA is a radiance (W m-2 sr-1), ERA5 and CERES
    are fluxes (W m-2), ratios are sr-1; CLARA OLR maps share the blue scale of predict_maps.map_limits(),
    uncertainty is orange, residuals blue <-> red, counts of observations aqua; kernels and datasets keep their
    categorical colours, the pre- and post-gap periods are a light and a dark blue, and the 2021-22 CLARA gap is
    the same grey band on every time axis
  - bold panel letters (a), (b), ... with short panel titles; the message is in the caption (main.tex)

Nothing is refitted: the figures read the tables of eda.py (results/eda/), run_cv.py --summarise
(results/model/<model>/metrics_by_fold.csv, oos_predictions_<cv>.csv, model_comparison.csv), predict_maps.py
(grid_predictions.csv, global_mean_monthly.csv), kernel_global_means.py (kernel_global_means.csv), the
harmonised observation tables (data/processed/) and report.py (numbers.tex, for the pipeline diagram).

Outputs, docs/paper/figures/:
    <name>.pdf       vector, included by main.tex
    png/<name>.png   300 dpi preview

Usage:
    python src/paper_figures.py                   # all figures
    python src/paper_figures.py maps physics      # some of them
"""

import argparse
import functools
import re

import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy import stats

import config
import eda
import predict_maps
import report
from style import (AXIS, CLARA_UNIT, CM, COLUMN_W, COUNT_CMAP, DATASET_COLORS, DIVERGING, FLUX_UNIT, GAP_FILL, GRID,
                   INK, KERNEL_COLORS, MUTED, OLR_CMAP, PAGE_W, PERIOD_COLORS, RATIO_UNIT, UNCERTAINTY_CMAP,
                   draw_field, map_axes, model_color, panel, paper_figure, plt, save_paper, style_axes,
                   use_paper_style)

OUT = config.ROOT / "docs" / "paper" / "figures"
R = config.RESULTS_DIR
CFG = config.load()
RES = CFG["grid"]["res_deg"]
T0, T1 = pd.Timestamp(CFG["time"]["first_month"]), pd.Timestamp(CFG["time"]["last_month"])
GAP = (pd.Timestamp(CFG["time"]["gap_start"]), pd.Timestamp(CFG["time"]["gap_end"]) + pd.offsets.MonthEnd(0))
XLIM = (T0 - pd.DateOffset(days=20), T1 + pd.DateOffset(days=20))  # every time axis
CV_TYPES = ("spatial", "temporal", "joint")
PERIODS = ("pre-gap", "post-gap")
LETTERS = "abcdefghijklmnop"
KERNEL_NAMES = {"exponential": "Exponential", "gaussian": "Gaussian", "matern32": "Matérn 3/2"}
BAND_NAMES = {"90S-60S": "90°S–60°S", "60S-30S": "60°S–30°S", "30S-EQ": "30°S–0°", "EQ-30N": "0°–30°N",
              "30N-60N": "30°N–60°N", "60N-90N": "60°N–90°N"}
PIT_BINS = 20             # bins of the PIT histograms
FOLD_JITTER = 0.18        # vertical spread of the fold dots in model_comparison (rows are 1 apart)


def model_name(model):
    """Short display name of a model id (the 2 x 2 ablation of src/baselines.py)."""
    if model in KERNEL_NAMES:
        return f"{KERNEL_NAMES[model]} GP + ERA5"
    if model.removesuffix("_noera5") in KERNEL_NAMES:
        return f"{KERNEL_NAMES[model.removesuffix('_noera5')]} GP, no ERA5"
    return {"linear": "Regression on ERA5", "linear_noera5": "Per-period constant"}[model]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def winner():
    return pd.read_csv(R / "model/kernel_comparison.csv").kernel.iloc[0]


def kernels():
    return [k for k in CFG["gp"]["kernels"] if (R / f"model/{k}/metrics_by_fold.csv").exists()]


@functools.cache
def observations():
    return pd.read_parquet(config.PROCESSED_DIR / f"harmonised_{RES}deg_monthly.parquet")


@functools.cache
def reference():
    return pd.read_parquet(config.PROCESSED_DIR / f"eda_reference_{RES}deg_monthly.parquet")


@functools.cache
def oos(model, cv_type):
    return pd.read_csv(R / f"model/{model}/oos_predictions_{cv_type}.csv",
                       usecols=["residual", "pred_sd_obs", "pit"])


@functools.cache
def grid_predictions():
    cols = ["time", "lat", "lon", "pred_mean", "latent_sd", "clara", "night_fallback", "residual_insample",
            "residual_oos_spatial"]
    return pd.read_csv(R / f"model/{winner()}/grid_predictions.csv", usecols=cols, parse_dates=["time"])


def summary_numbers():
    return pd.read_csv(R / "eda/summary_numbers.csv").set_index("name").value


def tex_numbers():
    """The macros of results/report/numbers.tex, the numbers the paper's text quotes, as display strings."""
    text = (config.REPORT_DIR / "numbers.tex").read_text()
    return {k: v.replace("\\,", ",") for k, v in re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}  %", text)}


def month_field(col, month):
    g = grid_predictions()
    return eda.to_field(g[g.time == month], col)


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

def style(ax, grid=True):
    style_axes(ax, grid=grid)
    if grid:
        ax.grid(color=GRID, lw=0.4)


def time_axis(ax, gap_label=False, label_at="top"):
    """Calendar x axis of the record with the CLARA gap shaded (the same band in every figure)."""
    ax.set_xlim(*XLIM)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.axvspan(*GAP, color=GAP_FILL, lw=0, zorder=0)
    if gap_label:
        top = label_at == "top"
        ax.text(GAP[0] + (GAP[1] - GAP[0]) / 2, 0.97 if top else 0.03, "gap", transform=ax.get_xaxis_transform(),
                ha="center", va="top" if top else "bottom", color=MUTED)


def field(ax, values, res, cmap, vmin, vmax):
    """A gridded field on a map, rasterised in the PDF: thousands of adjacent vector cells show hairline seams in
    PDF viewers. Text, coastlines and hatching stay vector."""
    mesh = draw_field(ax, values, res, cmap, vmin, vmax)
    mesh.set_rasterized(True)
    return mesh


def paper_map(fig, spec):
    ax = map_axes(fig, spec)
    if "geo" in ax.spines:
        ax.spines["geo"].set(linewidth=0.5, edgecolor=AXIS)
    return ax


def hbar(fig, mappable, ax, label, extend="neither", shrink=0.7, aspect=32, orientation="horizontal", ticks=None):
    cb = fig.colorbar(mappable, ax=ax, orientation=orientation, extend=extend, shrink=shrink, aspect=aspect,
                      pad=0.02, ticks=ticks)
    cb.set_label(label, color=INK, labelpad=2)
    cb.ax.tick_params(colors=MUTED, width=0.6, length=2.5, pad=1.5)
    cb.outline.set_visible(False)
    return cb


def threshold(ax, x, kept_side):
    """The QC threshold as a dashed line, with 'kept' and 'removed' on either side."""
    ax.axvline(x, color=MUTED, ls="--", lw=0.8, zorder=1)
    for side, word in ((kept_side, "kept"), ("left" if kept_side == "right" else "right", "removed")):
        right = side == "right"
        ax.annotate(word, (x, 1), xycoords=("data", "axes fraction"), xytext=(3 if right else -3, -2),
                    textcoords="offset points", ha="left" if right else "right", va="top", color=MUTED)


def lat_ticks(ax):
    ticks = np.arange(-90, 91, 30)
    ax.set_xticks(ticks, [f"{abs(v)}°{'S' if v < 0 else 'N' if v > 0 else ''}" for v in ticks])
    ax.set_xlim(-90, 90)


# ---------------------------------------------------------------------------
# Preprocessing and quality control
# ---------------------------------------------------------------------------

def physics():
    """Binned CLARA radiance and CLARA / ERA5 against solar and viewing zenith angle, by period."""
    b = pd.read_csv(R / "eda/physics_bins.csv")
    fig = paper_figure(PAGE_W, 9.6)
    axes = fig.subplots(2, 2)
    rows = [("sza", CFG["qc"]["night_sza_min"], "right", "Footprint solar zenith angle (°)", "clara_over_era5_hour",
             ("Radiance against solar zenith angle", "Ratio to ERA5 at the same hour")),
            ("vza (night)", CFG["qc"]["vza_max"], "left", "Viewing zenith angle (°)", "clara_over_era5_night",
             ("Radiance against viewing zenith angle, night side", "Ratio to night-time ERA5"))]
    for r, (x, thr, kept, xlabel, ratio, titles) in enumerate(rows):
        for c, (y, ylabel) in enumerate((("clara", f"CLARA radiance\n({CLARA_UNIT})"),
                                         (ratio, f"CLARA / ERA5\n({RATIO_UNIT})"))):
            ax = axes[r, c]
            for p in PERIODS:
                g = b[(b.x == x) & (b.y == y) & (b.group == p)]
                ax.plot(g.centre, g["mean"], color=PERIOD_COLORS[p], marker="o", label=p)
            threshold(ax, thr, kept)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            style(ax)
            panel(ax, LETTERS[2 * r + c], titles[c])
    axes[0, 0].legend(loc="lower left")
    return fig


def data_coverage():
    """Observed months per cell (map) and observed cell-months per latitude band and month."""
    obs = observations()
    count, hov = eda.coverage(obs)
    fig = paper_figure(COLUMN_W, 10.0)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1])
    ax = paper_map(fig, gs[0])
    mesh = field(ax, eda.to_field(count, "months"), RES, COUNT_CMAP, 0, count.months.max())
    hbar(fig, mesh, ax, "Observed months per cell")
    panel(ax, "a", "Months with CLARA data, per cell")
    ax = fig.add_subplot(gs[1])
    im = ax.imshow(np.ma.masked_equal(hov.to_numpy(), 0), aspect="auto", cmap=COUNT_CMAP, origin="lower",
                   interpolation="nearest", extent=[0, hov.shape[1], -0.5, len(hov) - 0.5], vmin=0)
    ax.set_yticks(range(len(hov)), [BAND_NAMES[b] for b in hov.index])
    ticks = [k for k, t in enumerate(hov.columns) if t.month == 1]
    ax.set_xticks(ticks, [hov.columns[k].year for k in ticks])
    style(ax, grid=False)
    hbar(fig, im, ax, "Observed cell-months per band and month")
    panel(ax, "b", "Observed cell-months by latitude band")
    return fig


# ---------------------------------------------------------------------------
# Comparison with ERA5 and CERES
# ---------------------------------------------------------------------------

def global_eda():
    """Sampling-matched monthly means: ERA5 and CERES (flux), CLARA (radiance), and their ratio."""
    g = pd.read_csv(R / "eda/temporal_global.csv", index_col=0, parse_dates=True)
    sn = summary_numbers()
    fig = paper_figure(COLUMN_W, 11.8)
    axes = fig.subplots(3, 1, sharex=True)
    ax = axes[0]
    for col, lab, color, ls in (("era5_night_r0", "ERA5 night", DATASET_COLORS["ERA5"], "-"),
                                ("era5_24h", "ERA5 24 h", DATASET_COLORS["ERA5"], "--"),
                                ("ceres", "CERES 24 h", DATASET_COLORS["CERES EBAF"], ":")):
        ax.plot(g.index, g[col], color=color, ls=ls, lw=1.1, label=lab)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.35 * (hi - lo), hi)  # room for the legend below the curves
    ax.legend(loc="lower center", ncol=3, handlelength=1.5, handletextpad=0.4, columnspacing=0.8)
    ax.set_ylabel(f"OLR flux\n({FLUX_UNIT})")
    panel(ax, "a", "ERA5 and CERES on the cells CLARA observed")
    ax = axes[1]
    ax.plot(g.index, g.clara, color=DATASET_COLORS["CLARA"], marker="o", ms=2.5, lw=1.1)
    ax.set_ylabel(f"CLARA radiance\n({CLARA_UNIT})")
    panel(ax, "b", "CLARA on the same cells")
    ax = axes[2]
    ax.plot(g.index, g.clara / g.era5_night_r0, color=DATASET_COLORS["CLARA"], marker="o", ms=2.5, lw=1.1,
            zorder=2)
    for p, (a, b), v in (("pre-gap", (XLIM[0], GAP[0]), sn["ratio_pre"]), ("post-gap", (GAP[1], XLIM[1]), sn["ratio_post"])):
        ax.hlines(v, a, b, color=PERIOD_COLORS[p], lw=1.6, ls=(0, (3, 1.5)), zorder=3,
                  label=f"{p} ratio of means, {v:.2f}")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.3 * (hi - lo))  # room for the legend above the pre-gap months
    ax.legend(loc="upper left", handlelength=1.8)
    ax.set_ylabel(f"CLARA / ERA5 night\n({RATIO_UNIT})")
    panel(ax, "c", "Ratio CLARA / ERA5 night")
    for ax in axes:
        time_axis(ax, gap_label=ax is axes[1])
        style(ax)
    return fig


def zonal():
    """Zonal means over the observed cell-months, each relative to its own area-weighted mean."""
    _, z = eda.zonal_relative(eda.with_ceres(observations(), reference()))
    fig = paper_figure(COLUMN_W, 6.4)
    ax = fig.subplots()
    ax.axhline(1, color=AXIS, lw=0.8)
    for col, lab, color, ls in (("clara", "CLARA", DATASET_COLORS["CLARA"], "-"),
                                ("era5_night_r0", "ERA5 night", DATASET_COLORS["ERA5"], "-"),
                                ("era5_24h", "ERA5 24 h", DATASET_COLORS["ERA5"], "--"),
                                ("ceres", "CERES 24 h", DATASET_COLORS["CERES EBAF"], ":")):
        ax.plot(z.index, z[col], color=color, ls=ls, lw=1.3 if col == "clara" else 1.1, label=lab)
    lat_ticks(ax)
    ax.set(xlabel="Latitude", ylabel="Zonal mean / own global mean")
    ax.legend(loc="lower center", ncol=2)
    style(ax)
    return fig


def correlogram():
    """Residual spatial correlogram (CLARA minus a linear ERA5 fit, month means removed) with exponential fits."""
    cg = pd.read_csv(R / "eda/spatial_correlogram.csv")
    fit = pd.read_csv(R / "eda/spatial_correlogram_fit.csv").set_index("period")
    fig = paper_figure(COLUMN_W, 6.4)
    ax = fig.subplots()
    ax.axhline(0, color=AXIS, lw=0.8)
    handles = []
    for p in ("all", *PERIODS):
        g, color = cg[cg.period == p], PERIOD_COLORS[p]
        c, length = fit.loc[p, ["c", "range_km"]]
        ax.plot(g.distance_km, g.correlation, ls="none", marker="o", ms=3 if p == "all" else 2.5, color=color,
                zorder=3 if p == "all" else 2)
        ax.plot(g.distance_km, c * np.exp(-g.distance_km / length), color=color, lw=1.1)
        name = "all data" if p == "all" else p
        handles.append(Line2D([], [], color=color, marker="o", ms=3, lw=1.1,
                              label=f"{name}: {c:.2f} exp(−d / {length:,.0f} km)"))
    ax.set(xlabel="Great-circle distance d (km)", ylabel="Residual correlation, same month")
    ax.legend(handles=handles, loc="upper right")
    style(ax)
    return fig


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def model_comparison():
    """Per-fold RMSE of the three kernels and the three reduced models on identical folds."""
    order = list(pd.read_csv(R / "model/model_comparison.csv", index_col=0).index)
    fm = pd.concat([pd.read_csv(R / f"model/{m}/metrics_by_fold.csv") for m in order], ignore_index=True)
    fig = paper_figure(PAGE_W, 6.6)
    axes = fig.subplots(1, 3, sharey=True)
    rng = np.random.default_rng(0)
    for k, (ax, t) in enumerate(zip(axes, CV_TYPES)):
        for i, m in enumerate(order):
            v = fm[(fm.cv_type == t) & (fm.kernel == m)].rmse.to_numpy()
            color = model_color(m)
            ax.plot(v, i + rng.uniform(-FOLD_JITTER, FOLD_JITTER, len(v)), "o", ms=3, color=color, alpha=0.7,
                    mfc="white" if m.endswith("_noera5") else color, mew=0.8)
            ax.vlines(v.mean(), i - 0.34, i + 0.34, color=color, lw=2.2, zorder=3)
        ax.set_xlabel(f"Fold RMSE ({CLARA_UNIT})")
        style(ax)
        ax.grid(axis="y", visible=False)
        panel(ax, LETTERS[k], f"{t.capitalize()} CV")
    axes[0].set_yticks(range(len(order)), [model_name(m) for m in order])
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.5, color=MUTED, label="one fold, with ERA5"),
                        Line2D([], [], ls="none", marker="o", ms=3.5, color=MUTED, mfc="white", mew=0.8,
                               label="one fold, without ERA5"),
                        Line2D([], [], ls="none", marker="|", ms=8, mew=2.2, color=MUTED, label="mean over folds")],
               loc="outside lower center", ncol=3)
    return fig


def coverage():
    """Empirical minus nominal coverage of central predictive intervals, by CV type and kernel."""
    levels = np.array([0.5, 0.8, 0.95])
    fig = paper_figure(COLUMN_W, 6.6)
    axes = fig.subplots(1, 3, sharey=True)
    for k, (ax, t) in enumerate(zip(axes, CV_TYPES)):
        ax.axhline(0, color=MUTED, lw=0.8, ls="--")
        for kern in kernels():
            p = oos(kern, t)
            z = np.abs(p.residual / p.pred_sd_obs)
            emp = np.array([np.mean(z <= stats.norm.ppf(0.5 + lv / 2)) for lv in levels])
            ax.plot(range(len(levels)), 100 * (emp - levels), color=KERNEL_COLORS[kern], marker="o",
                    label=KERNEL_NAMES[kern])
        ax.set_xticks(range(len(levels)), [f"{100 * lv:.0f}" for lv in levels])
        ax.set_xlim(-0.4, len(levels) - 0.6)
        style(ax)
        ax.grid(axis="x", visible=False)
        panel(ax, LETTERS[k], t.capitalize())
    axes[0].set_ylabel("Empirical − nominal coverage\n(percentage points)")
    lo, hi = axes[0].get_ylim()
    lim = max(abs(lo), abs(hi)) * 1.15
    axes[0].set_ylim(-lim, lim)
    axes[0].yaxis.set_major_locator(plt.MultipleLocator(5))
    for y, word, va in ((0.98, "too wide", "top"), (0.02, "too narrow", "bottom")):
        axes[0].text(0.04, y, word, transform=axes[0].transAxes, ha="left", va=va, color=MUTED)
    fig.supxlabel("Nominal coverage of the central interval (%)", fontsize=8, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3)
    return fig


def kernel_gap():
    """Monthly global mean of every kernel's full-data fit with +-2 SD, and the spread between kernels."""
    km = pd.read_csv(R / "model/kernel_global_means.csv", parse_dates=["time"])
    w = winner()
    order = [w] + [k for k in KERNEL_NAMES if k != w and k in set(km.kernel)]
    piv = km.pivot(index="time", columns="kernel", values="global_mean")
    sd = km.pivot(index="time", columns="kernel", values="sd_full_covariance")
    spread = piv.max(axis=1) - piv.min(axis=1)
    fig = paper_figure(PAGE_W, 9.0)
    axes = fig.subplots(2, 1, sharex=True, height_ratios=[1.7, 1])
    ax = axes[0]
    styles = {"exponential": "-", "gaussian": "--", "matern32": "-."}
    for k in order:
        c = KERNEL_COLORS[k]
        ax.fill_between(piv.index, piv[k] - 2 * sd[k], piv[k] + 2 * sd[k], color=c, alpha=0.14 if k == w else 0.08,
                        lw=0)
        ax.plot(piv.index, piv[k], color=c, lw=1.5 if k == w else 1.1, ls=styles[k],
                label=f"{KERNEL_NAMES[k]}{' (CV winner)' if k == w else ''}, ± 2 SD")
    ax.set_ylabel(f"Global mean\n({CLARA_UNIT})")
    ax.legend(loc="upper left", ncol=3)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.18 * (hi - lo))  # room for the legend above the curves
    panel(ax, "a", "Global mean of predicted CLARA radiance, by kernel")
    ax = axes[1]
    ax.plot(spread.index, spread, color=INK, lw=1.3, label="spread between kernels (max − min)")
    ax.plot(sd.index, 2 * sd[w], color=KERNEL_COLORS[w], lw=1.1, ls=styles[w],
            label=f"2 SD of the {KERNEL_NAMES[w]} global mean")
    t_max = spread.idxmax()
    ax.annotate(f"{spread.max():.1f} in {t_max:%Y-%m}", (t_max, spread.max()), xytext=(-5, -1),
                textcoords="offset points", color=INK, va="top", ha="right")
    ax.set_ylim(0, 1.32 * spread.max())  # room for the one-row legend
    ax.set_ylabel(f"Spread or 2 SD\n({CLARA_UNIT})")
    ax.legend(loc="upper right", ncol=2)
    panel(ax, "b", "Spread between the kernels against the stated uncertainty")
    for ax in axes:
        time_axis(ax, gap_label=ax is axes[0], label_at="bottom")
        style(ax)
    return fig


def maps():
    """Observed, predicted and posterior SD in the example month; predicted in the first gap month."""
    g = grid_predictions()
    (vmin, vmax), umax, _ = predict_maps.map_limits(g)
    ex, gap = pd.Timestamp(report.EXAMPLE_MONTH), GAP[0]
    fig = paper_figure(PAGE_W, 12.4)
    gs = fig.add_gridspec(2, 2)
    olr_label = f"CLARA radiance ({CLARA_UNIT})"
    olr_ticks = np.arange(np.ceil(vmin / 20) * 20, vmax + 1, 20)  # the same ticks on every OLR colour bar
    ax_a, ax_b, ax_c, ax_d = (paper_map(fig, gs[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    n_obs = int(np.isfinite(month_field("clara", ex)).sum())
    mesh = field(ax_a, month_field("clara", ex), RES, OLR_CMAP, vmin, vmax)
    panel(ax_a, "a", f"Observed, {ex:%B %Y} ({n_obs} cells)")
    field(ax_b, month_field("pred_mean", ex), RES, OLR_CMAP, vmin, vmax)
    predict_maps.hatch(ax_b, month_field("night_fallback", ex), RES)
    panel(ax_b, "b", f"Predicted, {ex:%B %Y}")
    hbar(fig, mesh, [ax_a, ax_b], olr_label, extend="both", shrink=0.5, aspect=45, ticks=olr_ticks)
    m_sd = field(ax_c, month_field("latent_sd", ex), RES, UNCERTAINTY_CMAP, 0, umax)
    predict_maps.hatch(ax_c, month_field("night_fallback", ex), RES)
    panel(ax_c, "c", f"Posterior SD, {ex:%B %Y}")
    hbar(fig, m_sd, ax_c, f"Posterior SD of the monthly mean ({CLARA_UNIT})", extend="max")
    m_gap = field(ax_d, month_field("pred_mean", gap), RES, OLR_CMAP, vmin, vmax)
    predict_maps.hatch(ax_d, month_field("night_fallback", gap), RES)
    panel(ax_d, "d", f"Predicted, {gap:%B %Y} (no CLARA data)")
    hbar(fig, m_gap, ax_d, olr_label, extend="both", ticks=olr_ticks)
    return fig


def global_mean():
    """Global mean of the winning kernel with its full-covariance SD, and that SD against independent cells."""
    w = winner()
    gm = pd.read_csv(R / f"model/{w}/global_mean_monthly.csv", parse_dates=["time"])
    color = KERNEL_COLORS[w]
    fig = paper_figure(COLUMN_W, 9.0)
    axes = fig.subplots(2, 1, sharex=True, height_ratios=[1.5, 1])
    ax = axes[0]
    ax.fill_between(gm.time, gm.global_mean - 2 * gm.sd_full_covariance, gm.global_mean + 2 * gm.sd_full_covariance,
                    color=color, alpha=0.2, lw=0, label="± 2 SD")
    ax.plot(gm.time, gm.global_mean, color=color, lw=1.3, label=f"{KERNEL_NAMES[w]} GP")
    ax.set_ylabel(f"Global mean\n({CLARA_UNIT})")
    ax.legend(loc="lower right", ncol=2)
    panel(ax, "a", "Monthly global mean of predicted CLARA radiance")
    ax = axes[1]
    ax.semilogy(gm.time, gm.sd_full_covariance, color=color, lw=1.3, label="full posterior covariance")
    ax.semilogy(gm.time, gm.sd_if_independent, color=MUTED, lw=1.1, ls="--", label="cells assumed independent")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel(f"SD of the global mean\n({CLARA_UNIT})")
    ax.legend(loc="center right")
    panel(ax, "b", "Its uncertainty")
    for ax in axes:
        time_axis(ax, gap_label=ax is axes[0])
        style(ax)
    return fig


# ---------------------------------------------------------------------------
# Appendix
# ---------------------------------------------------------------------------

def pipeline():
    """The research pipeline in six stages (a print-size version of docs/pipeline.png)."""
    n = tex_numbers()
    width, height = PAGE_W / CM, 7.9  # cm; the axes run in cm
    fig = paper_figure(PAGE_W, height, layout=None)
    ax = fig.add_axes([0, 0, 1, 1])
    renderer = fig.canvas.get_renderer()
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    gap_x, bh = 0.5, 3.45
    bw = (width - 2 * gap_x - 0.1) / 3
    tops = (height - 0.05, height - 0.05 - bh - 0.9)
    beta = r"$\beta_0 + \beta_1\,$ERA5$\, + \beta_2\,$post-gap$\, + f(\mathbf{x})$"
    stages = [
        ("Read and quality-control", [f"CLARA radiance, {CLARA_UNIT}", f"ERA5 and CERES flux, {FLUX_UNIT}",
                                      f"night side: SZA > {n['szaMin']}°", f"viewing angle: VZA ≤ {n['vzaMax']}°",
                                      f"{n['nQC']} samples, {n['nOverpasses']} overpasses"]),
        (f"Harmonise to {n['gridRes']}° × month", ["ERA5 night-time monthly mean", "CLARA cell-month means paired",
                                                   "with ERA5 night of the same cell",
                                                   f"{n['nObs']} observed cell-months",
                                                   f"grid: {n['nMonthsGrid']} months × {n['nCells']} cells"]),
        ("Explore: CLARA, ERA5, CERES", ["QC thresholds from the physics", "effective footprint ~10³ km",
                                         f"residual range {n['corrRangeKm']} km", "→ CV blocks and buffers",
                                         "calibration step across the gap"]),
        ("Select the kernel: blocked CV", [f"{n['nFolds']} folds: spatial, temporal, joint",
                                           "sparse GP fitted per fold", beta, None,
                                           "winner: lowest mean fold RMSE"]),
        ("Ablation on the same folds", ["GP + ERA5 (the winner)", "GP without ERA5", "regression on ERA5",
                                        "per-period constant", f"paired fold bootstrap, {n['ciLevel']} % CI"]),
        ("Map the winning kernel", ["full-data fit", "every cell-month: mean and SD",
                                    "global mean, full covariance", "all kernels through the gap",
                                    "tables, maps and report"]),
    ]
    boxes = []
    for i, (name, lines) in enumerate(stages):
        x, top = 0.05 + (i % 3) * (bw + gap_x), tops[i // 3]
        ax.add_patch(FancyBboxPatch((x, top - bh), bw, bh, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc="#f4f3ef", ec=AXIS, lw=0.8))
        ax.add_patch(Circle((x + 0.33, top - 0.33), 0.2, fc=INK, ec="none"))
        ax.text(x + 0.33, top - 0.335, str(i + 1), color="white", fontweight="bold", ha="center", va="center")
        ax.text(x + 0.65, top - 0.33, name, color=INK, fontweight="bold", ha="left", va="center")
        for j, line in enumerate(lines):
            y = top - 0.85 - 0.5 * j
            if line is None:  # the three kernels, in their colours, each name after the previous one's width
                kx = x + 0.3
                for k in KERNEL_NAMES:
                    ax.add_patch(Rectangle((kx, y - 0.09), 0.18, 0.18, fc=KERNEL_COLORS[k], ec="none"))
                    t = ax.text(kx + 0.24, y, KERNEL_NAMES[k], color=INK, ha="left", va="center")
                    kx = t.get_window_extent(renderer).transformed(ax.transData.inverted()).x1 + 0.14
                if kx - 0.14 > x + bw - 0.05:
                    raise ValueError("pipeline: the kernel names overrun stage box 4")
            else:
                ax.text(x + 0.3, y, line, color=INK if j < 4 or i in (1, 3) else MUTED, ha="left", va="center")
        boxes.append((x, top))
    arrow = dict(arrowstyle="-|>", mutation_scale=8, color=INK, lw=0.9, shrinkA=0, shrinkB=0)
    for i in (0, 1, 3, 4):
        x, top = boxes[i]
        y = top - bh / 2
        ax.add_patch(FancyArrowPatch((x + bw + 0.08, y), (x + bw + gap_x - 0.08, y), **arrow))
    (x3, top3), (x4, top4) = boxes[2], boxes[3]
    mid = (top3 - bh + top4) / 2
    ax.add_patch(FancyArrowPatch(path=_polyline([(x3 + bw / 2, top3 - bh - 0.06), (x3 + bw / 2, mid),
                                                 (x4 + bw / 2, mid), (x4 + bw / 2, top4 + 0.06)]), **arrow))
    ax.text(width / 2, mid, " the exploration sets QC, grid and CV blocks (config/pipeline.toml) ", color=MUTED,
            ha="center", va="center", style="italic", bbox=dict(fc="white", ec="none", pad=0.5))
    return fig


def _polyline(points):
    from matplotlib.path import Path as MplPath
    return MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))


def geometry():
    """Satellite altitude, viewing angle, daylit share and local solar time by month."""
    g = pd.read_csv(R / "eda/physics_geometry.csv", parse_dates=["month"], index_col="month")
    g = g.reindex(pd.date_range(T0, T1, freq="MS"))  # empty months break the lines
    fig = paper_figure(PAGE_W, 8.6)
    axes = fig.subplots(2, 2, sharex=True)
    for k, (ax, (col, scale, ylabel, name)) in enumerate(zip(axes.ravel(), [
            ("altitude_km", 1, "Altitude (km)", "Mean satellite altitude"),
            ("vza_median_night", 1, "Median VZA (°)", "Viewing zenith angle, night-side samples"),
            ("daylit_fraction", 100, "Daylit share (%)", "Daylit share of the usable samples"),
            ("lst_median_qc", 1, "Median local solar time (h)", "Local solar time, quality-checked samples")])):
        ax.plot(g.index, scale * g[col], color=DATASET_COLORS["CLARA"], marker="o", ms=2.3, lw=1.1)
        ax.set_ylabel(ylabel)
        time_axis(ax, gap_label=k == 0)
        style(ax)
        panel(ax, LETTERS[k], name)
    return fig


def resolution():
    """Along-track autocorrelation; CLARA-ERA5 agreement against ERA5 smoothing radius and grid spacing."""
    acf = pd.read_csv(R / "eda/resolution_along_track.csv")
    smooth = pd.read_csv(R / "eda/resolution_smoothing.csv")
    grid = pd.read_csv(R / "eda/resolution_grid.csv")
    fig = paper_figure(PAGE_W, 7.0)
    axes = fig.subplots(1, 3)
    ax = axes[0]
    for name, lab, color, ls in (("CLARA", "CLARA", DATASET_COLORS["CLARA"], "-"),
                                 ("CLARA / ERA5 hour-matched", "CLARA / ERA5, same hour", INK, "--")):
        a = acf[acf.series == name]
        ax.plot(a.distance_km, a.r, color=color, ls=ls, marker="o", ms=2.3, lw=1.1, label=lab)
    ax.axhline(np.exp(-1), color=MUTED, ls=":", lw=0.8)
    ax.annotate("1/e", (1, np.exp(-1)), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                textcoords="offset points", ha="right", va="bottom", color=MUTED)
    ax.set(xlabel="Along-track separation (km)", ylabel="Correlation")
    ax.legend(loc="upper right")
    panel(ax, "a", "Along-track autocorrelation")
    ax = axes[1]
    for p in ("all", *PERIODS):
        s = smooth[smooth.period == p]
        ax.plot(s.radius_km, s.r, color=PERIOD_COLORS[p], marker="o", lw=1.1)
        ax.plot(s.radius_km, s.r_anomaly_within_band_month, color=PERIOD_COLORS[p], marker="s", ms=3, mfc="white",
                lw=1.1, ls="--")
    ax.legend(handles=[Line2D([], [], color=PERIOD_COLORS[p], lw=1.1, label="all data" if p == "all" else p)
                       for p in ("all", *PERIODS)]
              + [Line2D([], [], color=MUTED, marker="o", lw=1.1, label="pooled"),
                 Line2D([], [], color=MUTED, marker="s", ms=3, mfc="white", ls="--", lw=1.1, label="anomaly")],
              loc="lower center", ncol=2, columnspacing=1.0)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.55 * (hi - lo), hi)  # room for the legend below the curves
    ax.set(xlabel="ERA5 smoothing radius (km)", ylabel="r, CLARA vs ERA5 night")
    panel(ax, "b", "Agreement against ERA5 smoothing")
    ax = axes[2]
    ax.semilogx(grid.res_deg, grid.r, color=DATASET_COLORS["CLARA"], marker="o", lw=1.1)
    for _, r in grid.iterrows():
        ax.annotate(f"{r.median_overpasses:.0f}", (r.res_deg, r.r), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", color=INK)
    ax.text(0.03, 0.97, "labels: median overpasses\nper cell-month", transform=ax.transAxes, ha="left", va="top",
            color=MUTED)
    ax.set_xticks(grid.res_deg, [f"{v:g}°" for v in grid.res_deg])
    ax.minorticks_off()
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.45 * (hi - lo))
    ax.set(xlabel="Grid spacing", ylabel="r, CLARA vs ERA5 night")
    panel(ax, "c", "Agreement against grid size")
    for ax in axes:
        style(ax)
    return fig


def joint_scatter():
    """CLARA against ERA5 night-time OLR per cell-month, before and after the gap."""
    obs = observations()
    st = pd.read_csv(R / "eda/joint_stats.csv")
    fig = paper_figure(PAGE_W, 7.6)
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    extent = (100, 330, 40, 400)
    hbs = []
    for k, (ax, p) in enumerate(zip(axes, PERIODS)):
        g = obs[obs.period == p]
        hbs.append(ax.hexbin(g.era5_night_r0, g.clara, gridsize=55, mincnt=1, cmap=COUNT_CMAP, extent=extent,
                             norm=LogNorm(), linewidths=0.1, edgecolors="face", rasterized=True))
        s = st[(st.group == "period") & (st.label == p) & (st.pair == "CLARA vs ERA5 night")].iloc[0]
        ax.plot(extent[:2], s.intercept + s.slope * np.array(extent[:2]), color=INK, lw=1.2,
                path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
        ax.text(0.03, 0.97, f"r = {s.r:.2f}\nratio of means {s.ratio:.2f} {RATIO_UNIT}\n"
                            f"slope {s.slope:.2f} {RATIO_UNIT}\nn = {s.n:,.0f}", transform=ax.transAxes,
                ha="left", va="top", color=INK)
        ax.set_xlabel(f"ERA5 night-time OLR ({FLUX_UNIT})")
        style(ax)
        panel(ax, LETTERS[k], p.capitalize())
    axes[0].set_ylabel(f"CLARA radiance ({CLARA_UNIT})")
    top = max(h.get_array().max() for h in hbs)
    for h in hbs:
        h.set_norm(LogNorm(1, top))
    cb = hbar(fig, hbs[0], axes, "Cell-months per bin", orientation="vertical", shrink=0.8, aspect=25)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    return fig


def pit():
    """PIT histograms by kernel (rows) and CV type (columns)."""
    ks = kernels()
    fig = paper_figure(PAGE_W, 10.4)
    axes = fig.subplots(len(ks), 3, sharex=True, sharey=True)
    for r, k in enumerate(ks):
        for c, t in enumerate(CV_TYPES):
            ax = axes[r, c]
            ax.hist(oos(k, t).pit, bins=PIT_BINS, range=(0, 1), density=True, color=KERNEL_COLORS[k],
                    edgecolor="white", lw=0.4)
            ax.axhline(1, color=INK, ls="--", lw=0.8)
            style(ax)
            ax.grid(axis="x", visible=False)
            panel(ax, LETTERS[3 * r + c], f"{KERNEL_NAMES[k]}, {t} CV")
        axes[r, 0].set_ylabel("Density")
    for ax in axes[-1]:
        ax.set_xlabel("PIT")
    return fig


def rmse_by_fold():
    """Per-fold RMSE of the three kernels, by CV type."""
    ks = kernels()
    fm = pd.concat([pd.read_csv(R / f"model/{k}/metrics_by_fold.csv") for k in ks], ignore_index=True)
    n_folds = [fm[fm.cv_type == t].fold.nunique() for t in CV_TYPES]
    fig = paper_figure(PAGE_W, 6.8)
    axes = fig.subplots(1, 3, sharey=True, width_ratios=n_folds)
    for c, (ax, t) in enumerate(zip(axes, CV_TYPES)):
        g = fm[fm.cv_type == t]
        for i, k in enumerate(ks):
            v = g[g.kernel == k].sort_values("fold")
            ax.plot(v.fold + (i - (len(ks) - 1) / 2) * 0.22, v.rmse, "o", ms=3.2, color=KERNEL_COLORS[k],
                    mec="white", mew=0.4, label=KERNEL_NAMES[k])
            ax.hlines(v.rmse.mean(), v.fold.min() - 0.5, v.fold.max() + 0.5, color=KERNEL_COLORS[k], lw=0.9)
        ax.set_xticks(range(0, g.fold.max() + 1, 2))
        ax.set_xlim(-0.6, g.fold.max() + 0.6)
        ax.set_xlabel("Fold (held-out block)")
        style(ax)
        ax.grid(axis="x", visible=False)
        panel(ax, LETTERS[c], f"{t.capitalize()} CV")
    axes[0].set_ylabel(f"RMSE ({CLARA_UNIT})")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, lw=0.9, label="mean over folds"))
    fig.legend(handles, labels + ["mean over folds"], loc="outside lower center", ncol=4)
    return fig


def residuals():
    """In-sample and out-of-sample (spatial CV) residual maps of the example month."""
    g = grid_predictions()
    _, _, rlim = predict_maps.map_limits(g)
    ex = pd.Timestamp(report.EXAMPLE_MONTH)
    fig = paper_figure(PAGE_W, 6.0)
    gs = fig.add_gridspec(1, 2)
    axes = []
    for k, (col, name) in enumerate((("residual_insample", "In-sample (full-data fit)"),
                                     ("residual_oos_spatial", "Out-of-sample (spatial CV)"))):
        ax = paper_map(fig, gs[0, k])
        v = month_field(col, ex)
        mesh = field(ax, v, RES, DIVERGING, -rlim, rlim)
        ok = np.isfinite(v)
        panel(ax, LETTERS[k], f"{name}: RMSE {np.sqrt(np.mean(v[ok] ** 2)):.1f}, n = {ok.sum()}")
        axes.append(ax)
    hbar(fig, mesh, axes, f"Observed − predicted, {ex:%B %Y} ({CLARA_UNIT})", extend="both", shrink=0.5, aspect=45)
    return fig


# in the order of the paper
FIGURES = {
    "physics": physics, "data_coverage": data_coverage,                                       # QC
    "global_eda": global_eda, "zonal": zonal, "correlogram": correlogram,                     # ERA5 and CERES
    "model_comparison": model_comparison, "coverage": coverage, "kernel_gap": kernel_gap,     # results
    "maps": maps, "global_mean": global_mean,
    "pipeline": pipeline, "geometry": geometry, "resolution": resolution,                     # appendix
    "joint_scatter": joint_scatter, "pit": pit, "rmse_by_fold": rmse_by_fold, "residuals": residuals,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", metavar="NAME", help=f"figures to draw (default: all): {', '.join(FIGURES)}")
    args = parser.parse_args()
    unknown = sorted(set(args.names) - set(FIGURES))
    if unknown:
        parser.error(f"unknown figure(s) {', '.join(unknown)}; choose from {', '.join(FIGURES)}")
    use_paper_style()
    for name in args.names or FIGURES:
        w, h = save_paper(FIGURES[name](), OUT / name)
        print(f"{name:18s} {w:5.1f} x {h:4.1f} cm  -> {(OUT / name).with_suffix('.pdf').relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
