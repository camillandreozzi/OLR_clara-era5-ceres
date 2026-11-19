"""Phase 6b: monthly global mean of every kernel's full-data fit, through the 2021-22 CLARA gap.

The three kernels cannot be told apart in cross-validation (kernel_pairwise.csv), and CV only scores cell-months that
CLARA observed. This script asks where they still differ: it applies each saved full_model.pt to the whole prediction
grid with predict_maps.predict_grid() and reduces every month to its area-weighted global mean and full-covariance SD
with predict_maps.global_mean(), the same code as the winner's global_mean_monthly.csv. Nothing is refitted.

For month t, with A_i the area of cell i and w_i = A_i / sum_j A_j over all cells of the grid (polar-day cells
included):

    G_t      = sum_i w_i mu_i                              mu_i = h_i' beta + E[f(x_i) | y], the posterior mean map
    Var[G_t] = w' (K** - Q** + K*u Sigma Ku*) w            full posterior covariance of the latent values (gp.aggregate)

Outputs, results/model/:
    kernel_global_means.csv           one row per kernel x month: global mean, SDs, observed area share, period
    kernel_gap_summary.csv            per period (pre-gap / gap / post-gap): mean SD per kernel and the kernel spread
    figures/kernel_global_means.png   (a) the three global means with +-2 SD; (b) kernel spread against 2 SD

Usage:
    python src/kernel_global_means.py
"""

import argparse

import numpy as np
import pandas as pd
import torch

import config
import gp
import predict_maps
from style import (CLARA_UNIT, GRID, INK, KERNEL_COLORS, KERNEL_LABELS, MUTED, NO_DATA, plt, save, style_axes,
                   title)

LINESTYLES = {"exponential": "-", "gaussian": "--", "matern32": "-."}


def period(time, cfg):
    per = pd.DatetimeIndex(time).to_period("M")
    gs, ge = pd.Period(cfg["time"]["gap_start"], "M"), pd.Period(cfg["time"]["gap_end"], "M")
    return np.where(per < gs, "pre-gap", np.where(per > ge, "post-gap", "gap"))


def global_means(cfg, kernels):
    rows = []
    for k in kernels:
        model, radius = predict_maps.load_model(k)
        grid, x, ok = predict_maps.predict_grid(cfg, model, radius)
        gm = predict_maps.global_mean(cfg, model, grid, x, ok, radius)
        obs = grid.assign(a=grid.cell_area_km2 * grid.clara_observed).groupby("time")
        gm["observed_area_fraction"] = gm.time.map(obs.a.sum() / obs.cell_area_km2.sum())
        rows.append(gm.assign(kernel=k))
        print(f"{k}: {len(gm)} months", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out["period"] = period(out.time, cfg)
    return out[["kernel", "time", "period", "global_mean", "sd_full_covariance", "sd_if_independent",
                "observed_area_fraction", "polar_day_area_fraction"]]


def gap_summary(km, winner):
    piv = km.pivot(index="time", columns="kernel", values="global_mean")
    spread = (piv.max(1) - piv.min(1)).rename("kernel_spread")
    per = km.drop_duplicates("time").set_index("time").period
    sd = km.pivot(index="time", columns="kernel", values="sd_full_covariance")
    t = pd.concat([spread, sd.add_prefix("sd_"), piv.add_prefix("mean_"), per], axis=1)
    rows = []
    for p in ("pre-gap", "gap", "post-gap"):
        g = t[t.period == p]
        r = {"period": p, "months": len(g), "mean_kernel_spread": g.kernel_spread.mean(),
             "max_kernel_spread": g.kernel_spread.max(), "month_of_max_spread": f"{g.kernel_spread.idxmax():%Y-%m}",
             f"months_spread_above_2sd_{winner}": int((g.kernel_spread > 2 * g[f"sd_{winner}"]).sum())}
        r |= {f"mean_sd_{k}": g[f"sd_{k}"].mean() for k in sd.columns}
        r |= {f"mean_global_mean_{k}": g[f"mean_{k}"].mean() for k in sd.columns}
        rows.append(r)
    return pd.DataFrame(rows)


def figure(km, cfg, winner, path):
    gs, ge = pd.Timestamp(cfg["time"]["gap_start"]), pd.Timestamp(cfg["time"]["gap_end"]) + pd.offsets.MonthEnd(1)
    order = [winner] + [k for k in KERNEL_LABELS if k != winner and k in set(km.kernel)]
    piv = km.pivot(index="time", columns="kernel", values="global_mean")
    sd = km.pivot(index="time", columns="kernel", values="sd_full_covariance")
    spread = piv.max(1) - piv.min(1)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), layout="constrained", sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    ax = axes[0]
    for a in axes:
        a.axvspan(gs, ge, color=NO_DATA, alpha=0.6, lw=0, zorder=0)
    for k in order:
        c = KERNEL_COLORS[k]
        ax.fill_between(piv.index, piv[k] - 2 * sd[k], piv[k] + 2 * sd[k], color=c, alpha=0.13 if k == winner else 0.08,
                        lw=0)
        ax.plot(piv.index, piv[k], color=c, lw=2.2 if k == winner else 1.6, ls=LINESTYLES[k],
                label=f"{KERNEL_LABELS[k]}{' (CV winner)' if k == winner else ''} ± 2 SD")
    ends = piv.iloc[-1].sort_values()
    ypos = ends.to_numpy().copy()
    gap = 0.06 * np.ptp(ax.get_ylim())  # keep end labels apart by 6 % of the axis height
    for i in range(1, len(ypos)):
        ypos[i] = max(ypos[i], ypos[i - 1] + gap)
    for (k, _), y in zip(ends.items(), ypos):
        ax.annotate(KERNEL_LABELS[k], (piv.index[-1], y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=7, color=INK)
    ax.text(gs + (ge - gs) / 2, 0.03, "no CLARA data", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=8, color=MUTED)
    ax.set_ylabel(f"global mean ({CLARA_UNIT})", color=INK, fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    style_axes(ax)
    title(ax, "(a) Global mean of predicted CLARA OLR per kernel, ± 2 SD (full posterior covariance)", 9)

    ax = axes[1]
    ax.plot(spread.index, spread, color=INK, lw=1.8, label="spread between kernels (max − min)")
    ax.plot(sd.index, 2 * sd[winner], color=KERNEL_COLORS[winner], lw=1.4, ls=LINESTYLES[winner],
            label=f"2 SD of the {KERNEL_LABELS[winner]} global mean")
    t_max = spread.idxmax()
    ax.annotate(f"{spread.max():.1f} in {t_max:%Y-%m}", (t_max, spread.max()), xytext=(-6, 0),
                textcoords="offset points", fontsize=8, color=INK, va="top", ha="right")
    ax.set_ylim(0, None)
    ax.set_ylabel(CLARA_UNIT, color=INK, fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    style_axes(ax)
    ax.grid(axis="x", visible=False)
    title(ax, "(b) Spread between the kernels against the stated uncertainty", 9)
    axes[0].grid(axis="x", visible=False)
    axes[0].tick_params(axis="y", colors=MUTED)
    for a in axes:
        a.spines["bottom"].set_color(GRID)
    save(fig, path, dpi=200)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="configuration file (default config/pipeline.toml)")
    args = parser.parse_args()
    cfg = config.load(args.config)
    config.MODEL_DIR = config.RESULTS_DIR / cfg.get("output", {}).get("model_dir", "model")
    torch.set_num_threads(cfg["gp"]["threads"])
    kernels = [k for k in gp.KERNELS if (config.MODEL_DIR / k / "full_model.pt").exists()]
    winner = pd.read_csv(config.MODEL_DIR / "kernel_comparison.csv").kernel.iloc[0]
    km = global_means(cfg, kernels)
    km.to_csv(config.MODEL_DIR / "kernel_global_means.csv", index=False, float_format="%.4f")
    summ = gap_summary(km, winner)
    summ.to_csv(config.MODEL_DIR / "kernel_gap_summary.csv", index=False, float_format="%.3f")
    figure(km, cfg, winner, config.MODEL_DIR / "figures" / "kernel_global_means.png")
    pd.set_option("display.width", 220)
    print(summ.round(2).to_string(index=False))
    print(f"wrote kernel_global_means.csv, kernel_gap_summary.csv and figures/kernel_global_means.png in {config.MODEL_DIR}")


if __name__ == "__main__":
    main()
