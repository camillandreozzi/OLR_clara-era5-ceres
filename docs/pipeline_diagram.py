"""Flowchart of the pipeline for the README: docs/pipeline.png.

Six numbered stages, read left to right and top to bottom; folders are files on disk, the page icon is
the configuration every script reads. Stage 5 is the 2 x 2 ablation of src/baselines.py (kernel x ERA5).
Colours follow src/style.py (categorical slots and ink).

Usage:
    python docs/pipeline_diagram.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as MplPath

OUT = Path(__file__).resolve().parent / "pipeline.png"

W, H = 14.0, 12.75
INK, MUTED, LINE = "#1f1f1e", "#6b6a66", "#3a3a38"
SANS = ["Avenir Next", "Helvetica Neue", "DejaVu Sans"]
CONDENSED = ["Avenir Next Condensed", "Arial Narrow", "DejaVu Sans"]
MONO = ["Menlo", "DejaVu Sans Mono"]

# per stage: (accent, container fill, step-box fill)
STAGE = {
    1: ("#2a78d6", "#e9f1fc", "#cfe1f7"),
    2: ("#7b5cd6", "#f0ebfb", "#ddd2f6"),
    3: ("#1baf7a", "#e5f6ee", "#c4ebdb"),
    4: ("#eb6834", "#fdede5", "#f9d3c1"),
    5: ("#4f6b8a", "#edf1f6", "#d6dfea"),
    6: ("#c8407a", "#fbe9f1", "#f4cadc"),
}
KERNELS = [("Exponential", "#2a78d6"), ("Gaussian", "#eb6834"), ("Matérn 3/2", "#1baf7a")]
FOLDER_FILL, FOLDER_EDGE, FOLDER_TAB = "#f6ebc4", "#a88f3c", "#e9d48a"
PILL = "#26324a"


def text(ax, x, y, s, size=10, color=INK, family=SANS, weight="normal", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontfamily=family, fontweight=weight, ha=ha, va=va, **kw)


def box(ax, x, y, w, h, fill, edge, lw=1.6, r=0.16, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fill, ec=edge,
                                lw=lw, zorder=z))


def badge(ax, x, y, label, color, r=0.24, size=15):
    ax.add_patch(Circle((x, y), r, fc=color, ec="white", lw=2, zorder=5))
    text(ax, x, y - 0.01, label, size=size, color="white", family=CONDENSED, weight="bold", zorder=6)


def stage(ax, n, x, y, w, h, title, script=None):
    """Container of one stage: numbered badge, title and (optionally) the script that runs it."""
    accent, fill, _ = STAGE[n]
    box(ax, x, y, w, h, fill, accent, lw=2.2, r=0.22)
    badge(ax, x + 0.36, y + h - 0.34, str(n), accent, r=0.25, size=16)
    text(ax, x + 0.72, y + h - 0.34, title, size=16, family=CONDENSED, weight="bold", ha="left")
    if script:
        text(ax, x + w - 0.2, y + h - 0.34, script, size=10, color=MUTED, family=MONO, ha="right")


def step(ax, n, x, y, w, h, letter, title, lines=(), mono=None):
    """A step inside a stage: lettered tab, bold title, short lines, optional command in monospace."""
    accent, _, fill = STAGE[n]
    box(ax, x, y, w, h, fill, accent, lw=1.4, r=0.14, z=2)
    ax.add_patch(Circle((x + 0.02, y + h - 0.02), 0.15, fc="white", ec=accent, lw=1.4, zorder=4))
    text(ax, x + 0.02, y + h - 0.025, letter, size=10, color=accent, weight="bold", zorder=5)
    top = y + h - 0.3
    text(ax, x + w / 2, top, title, size=12.5, weight="bold", zorder=3)
    for i, line in enumerate(lines):
        text(ax, x + w / 2, top - 0.3 - 0.25 * i, line, size=10, zorder=3)
    if mono:
        text(ax, x + w / 2, y + 0.2, mono, size=9, color=MUTED, family=MONO, zorder=3)


def folder(ax, x, y, w, h, path, name, sub=None):
    """A folder icon: its path on the tab, the content in bold, a short note below."""
    tab_w, tab_h = 0.071 * len(path) + 0.3, 0.22
    ax.add_patch(Polygon([(x, y + h), (x, y + h + tab_h), (x + tab_w - 0.12, y + h + tab_h), (x + tab_w, y + h)],
                         closed=True, fc=FOLDER_TAB, ec=FOLDER_EDGE, lw=1.2, zorder=2))
    text(ax, x + 0.1, y + h + tab_h / 2, path, size=8.5, color=INK, family=MONO, ha="left", zorder=4)
    box(ax, x, y, w, h, FOLDER_FILL, FOLDER_EDGE, lw=1.2, r=0.05, z=3)
    text(ax, x + w / 2, y + (0.63 if sub else 0.5) * h, name, size=11, weight="bold", zorder=4)
    if sub:
        text(ax, x + w / 2, y + 0.27 * h, sub, size=9, color=MUTED, zorder=4)


def page(ax, x, y, w, h, name, lines):
    """The configuration file: a page with a folded corner."""
    c = 0.28
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - c), (x + w - c, y + h), (x, y + h)], closed=True,
                         fc="white", ec=LINE, lw=1.5, zorder=3))
    ax.add_patch(Polygon([(x + w - c, y + h), (x + w - c, y + h - c), (x + w, y + h - c)], closed=True,
                         fc="#e4e3de", ec=LINE, lw=1.2, zorder=4))
    text(ax, x + w / 2 - 0.08, y + h - 0.42, name, size=11, weight="bold", family=MONO, zorder=5)
    for i, line in enumerate(lines):
        text(ax, x + 0.22, y + h - 0.8 - 0.25 * i, line, size=9.5, family=MONO, ha="left", zorder=5)


def pill(ax, x, y, w, h, label):
    box(ax, x, y, w, h, PILL, PILL, lw=0, r=h / 2, z=3)
    text(ax, x + w / 2, y + h / 2, label, size=13, color="white", family=CONDENSED, weight="bold", zorder=4)


def arrow(ax, *points, color=LINE, lw=2.0, dashed=False, head=16, z=6):
    """Orthogonal polyline arrow through the given points, head at the last one."""
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
    ax.add_patch(FancyArrowPatch(path=MplPath(points, codes), arrowstyle="-|>", mutation_scale=head, color=color,
                                 lw=lw, linestyle=(0, (4, 3)) if dashed else "-", joinstyle="miter",
                                 capstyle="butt", zorder=z))


def label(ax, x, y, s, size=9.5, color=MUTED, **kw):
    text(ax, x, y, s, size=size, color=color, style="italic", zorder=7,
         bbox=dict(fc="white", ec="none", pad=1.5), **kw)


def draw():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---- row 1: data in ------------------------------------------------------------------------------
    y1, h1 = 9.55, 2.45
    pill(ax, 0.25, y1 + 2.65, 1.25, 0.4, "START")
    for i, (path, name, sub) in enumerate([("data/CLARA/", "CLARA", "NorSat-1 OLR + geoloc."),
                                           ("data/ERA5/", "ERA5", "retrieve_era5.py, CDS"),
                                           ("data/CERES/", "CERES EBAF", "EDA reference only")]):
        folder(ax, 0.25, y1 + 1.57 - 0.9 * i, 1.9, 0.58, path, name, sub)
    # the three inputs join into one arrow
    ax.plot([2.15, 2.33, 2.33, 2.15], [y1 + 1.86, y1 + 1.86, y1 + 0.06, y1 + 0.06], color=LINE, lw=2,
            solid_joinstyle="miter", zorder=6)
    ax.plot([2.15, 2.33], [y1 + 0.96, y1 + 0.96], color=LINE, lw=2, zorder=6)
    arrow(ax, (0.88, y1 + 2.65), (0.88, y1 + 2.37))
    arrow(ax, (2.32, y1 + 0.96), (2.7, y1 + 0.96))

    stage(ax, 1, 2.5, y1, 4.75, h1, "Read & quality-control")
    step(ax, 1, 2.7, y1 + 0.2, 1.4, 1.55, "a", "Read", ["unpack, geolocate", "CLARA W m⁻² sr⁻¹", "ERA5 W m⁻²"],
         mono="read_in.py")
    step(ax, 1, 4.25, y1 + 0.2, 1.4, 1.55, "b", "QC", ["SZA > 100°", "VZA ≤ 60°", "overpass ids"],
         mono="samples.py")
    step(ax, 1, 5.8, y1 + 0.2, 1.3, 1.55, "c", "Inventory", ["files and", "coverage"], mono="inventory.py")

    stage(ax, 2, 7.55, y1, 3.45, h1, "Harmonise: 5° × month")
    step(ax, 2, 7.75, y1 + 0.2, 1.45, 1.55, "a", "ERA5 night", ["hour-of-day", "means, night", "hours only"],
         mono="era5_night.py")
    step(ax, 2, 9.35, y1 + 0.2, 1.45, 1.55, "b", "Pair", ["CLARA", "cell-months +", "ERA5 night"],
         mono="harmonise.py")

    folder(ax, 11.35, y1 + 1.5, 2.4, 0.7, "data/processed/", "observations", "37.9 k cell-months")
    folder(ax, 11.35, y1 + 0.4, 2.4, 0.7, "data/processed/", "prediction grid", "210 k cell-months, no gaps")

    for xa, xb in ((4.1, 4.25), (5.65, 5.8)):
        arrow(ax, (xa, y1 + 0.97), (xb, y1 + 0.97), lw=1.5, head=11)
    arrow(ax, (7.25, y1 + 1.2), (7.75, y1 + 1.2))
    arrow(ax, (9.2, y1 + 0.97), (9.35, y1 + 0.97), lw=1.5, head=11)
    arrow(ax, (11.0, y1 + 1.2), (11.18, y1 + 1.2), (11.18, y1 + 0.75), (11.35, y1 + 0.75))
    arrow(ax, (11.0, y1 + 1.2), (11.18, y1 + 1.2), (11.18, y1 + 1.85), (11.35, y1 + 1.85))

    # ---- row 2: explore, then select the kernel ------------------------------------------------------
    y2, h2 = 5.95, 2.95
    stage(ax, 3, 0.25, y2, 2.85, h2, "Explore", "eda.py")
    for i, (name, note) in enumerate([("physics", "SZA, VZA"), ("resolution", "~10³ km"),
                                      ("spatial", "range 1,060 km"), ("temporal", "ACF, regime step")]):
        yy = y2 + 1.95 - 0.47 * i
        box(ax, 0.45, yy, 2.45, 0.38, STAGE[3][2], STAGE[3][0], lw=1.2, r=0.08, z=2)
        text(ax, 0.6, yy + 0.19, name, size=10.5, weight="bold", ha="left", zorder=3)
        text(ax, 2.78, yy + 0.19, note, size=9.5, color=MUTED, ha="right", zorder=3)
    text(ax, 1.68, y2 + 0.22, "→ results/eda/summary.md", size=9, color=MUTED, family=MONO)

    page(ax, 3.45, y2 + 0.35, 1.75, 2.25, "pipeline.toml", ["[qc]", "[grid]", "[era5]", "[gp]", "[cv]",
                                                              "[baselines]"])
    label(ax, 4.33, y2 + 2.8, "values set by the EDA", size=9)
    text(ax, 4.33, y2 + 0.13, "config/  read by every script", size=9, color=MUTED)

    stage(ax, 4, 5.55, y2, 8.2, h2, "Select the kernel: blocked CV", "run_cv.py")
    sb_y, sb_h = y2 + 0.72, 1.6
    step(ax, 4, 5.75, sb_y, 1.6, sb_h, "a", "Folds", ["spatial, temporal,", "joint blocks", "+ buffers"],
         mono="--make-folds")
    step(ax, 4, 7.55, sb_y, 1.7, sb_h, "b", "Per fold", ["train rows only:", "ERA5 smoothing R,", "inducing inputs"])
    step(ax, 4, 9.45, sb_y, 2.5, sb_h, "c", "Sparse GP (VFE)", ["β₀ + β₁·ERA5 + β₂·gap", "+ f(space, season, t)"])
    for i, (name, color) in enumerate(KERNELS):
        kx = 9.51 + 0.81 * i
        box(ax, kx, sb_y + 0.12, 0.76, 0.32, color, color, lw=0, r=0.07, z=3)
        text(ax, kx + 0.38, sb_y + 0.28, name, size=8.5, color="white", weight="bold", zorder=4)
    step(ax, 4, 12.15, sb_y, 1.4, sb_h, "d", "Compare", ["RMSE, CRPS,", "coverage, PIT"], mono="--summarise")
    for xa, xb in ((7.35, 7.55), (9.25, 9.45), (11.95, 12.15)):
        arrow(ax, (xa, sb_y + 0.8), (xb, sb_y + 0.8), lw=1.5, head=11)
    # loop over folds and kernels, under steps b and c
    arrow(ax, (10.7, sb_y), (10.7, sb_y - 0.32), (8.4, sb_y - 0.32), (8.4, sb_y), lw=1.5, head=11)
    label(ax, 9.55, sb_y - 0.32, "49 folds × 3 kernels, resumable", size=8.5)
    text(ax, 5.75, y2 + 0.2, "benchmark_gp.py: sparse vs exact GP   ·   euler/: the 150 fits as one Slurm array",
         size=9, color=MUTED, ha="left")

    # data/processed -> explore (sweep back to the start of row 2)
    arrow(ax, (12.55, y1 + 0.4), (12.55, y1 - 0.32), (2.05, y1 - 0.32), (2.05, y2 + h2))
    arrow(ax, (3.1, y2 + 1.5), (3.45, y2 + 1.5))
    arrow(ax, (5.2, y2 + 1.5), (5.75, y2 + 1.5))

    # ---- row 3: 2 x 2 ablation on the same folds -----------------------------------------------------
    y3, h3 = 2.75, 2.8
    accent = STAGE[5][0]
    stage(ax, 5, 0.25, y3, 13.5, h3, "Ablation: what the kernel and ERA5 each add", "run_cv.py --model")
    cw, ch = 2.5, 0.62
    col_x = {False: 2.1, True: 5.45}           # left edge of the cells without / with ERA5
    row_y = {True: y3 + 1.2, False: y3 + 0.2}  # bottom edge of the cells with / without a GP
    for era5, x in col_x.items():
        text(ax, x + cw / 2, y3 + 2.05, "with ERA5" if era5 else "without ERA5", size=10.5, weight="bold")
    for is_gp, y in row_y.items():
        text(ax, 1.2, y + ch / 2 + 0.11, "GP" if is_gp else "Regression", size=11.5, weight="bold")
        text(ax, 1.2, y + ch / 2 - 0.13, "mean + kernel" if is_gp else "no kernel", size=9, color=MUTED)
    for is_gp, era5, model, formula in [(True, True, "<winner>", "β₀ + β₁·ERA5 + β₂·gap + f(x)"),
                                        (True, False, "<winner>_noera5", "β₀ + β₂·gap + f(x)"),
                                        (False, True, "linear", "β₀ + β₁·ERA5 + β₂·gap"),
                                        (False, False, "linear_noera5", "β₀ + β₂·gap  (per period)")]:
        x, y = col_x[era5], row_y[is_gp]
        winner = model == "<winner>"  # the full model, already fitted in stage 4
        box(ax, x, y, cw, ch, STAGE[4][2] if winner else "white", STAGE[4][0] if winner else accent,
            lw=1.6, r=0.1, z=2)
        text(ax, x + cw / 2, y + 0.43, model, size=8.5, color=MUTED, family=MONO, zorder=3)
        text(ax, x + cw / 2, y + 0.2, formula, size=9.5, weight="bold", zorder=3)
    # each arrow is one of the four contrasts
    for era5, x in col_x.items():
        cx = x + cw / 2
        arrow(ax, (cx, row_y[False] + ch), (cx, row_y[True]), color=accent, lw=1.6, head=12)
        text(ax, cx + 0.12, (row_y[False] + ch + row_y[True]) / 2, "+ kernel", size=9.5, color=accent,
             style="italic", weight="bold", ha="left")
    for is_gp, y in row_y.items():
        arrow(ax, (col_x[False] + cw, y + ch / 2), (col_x[True], y + ch / 2), color=accent, lw=1.6, head=12)
        text(ax, (col_x[False] + cw + col_x[True]) / 2, y + ch / 2 + 0.16, "+ ERA5", size=9.5, color=accent,
             style="italic", weight="bold")

    step(ax, 5, 8.45, y3 + 0.2, 2.4, 2.0, "a", "Same folds", ["the 49 CV folds,", "same noise σ₀² + σ₁²/n,",
                                                              "same metrics;", "regression by GLS"],
         mono="--model <id>")
    step(ax, 5, 11.1, y3 + 0.2, 2.45, 2.0, "b", "Contrasts", ["the four arrows,", "paired fold bootstrap,",
                                                             "95 % CI per CV type"], mono="--summarise")
    arrow(ax, (10.85, y3 + 1.2), (11.1, y3 + 1.2), lw=1.5, head=11)

    # ---- row 4: map the winning kernel ---------------------------------------------------------------
    y4, h4 = 0.25, 2.25
    stage(ax, 6, 0.25, y4, 8.6, h4, "Map the winning kernel")
    step(ax, 6, 0.45, y4 + 0.2, 2.45, 1.4, "a", "Full-data fit", ["winner: lowest", "mean fold RMSE"],
         mono="--kernel <k> --full")
    step(ax, 6, 3.15, y4 + 0.2, 3.0, 1.4, "b", "Monthly maps", ["every month: mean,", "uncertainty, residuals"],
         mono="predict_maps.py")
    step(ax, 6, 6.4, y4 + 0.2, 2.25, 1.4, "c", "Report", ["CV, ablation, maps:", "numbers from tables"],
         mono="report.py")
    for xa, xb in ((2.9, 3.15), (6.15, 6.4)):
        arrow(ax, (xa, y4 + 0.9), (xb, y4 + 0.9), lw=1.5, head=11)

    folder(ax, 9.25, 1.35, 2.6, 0.7, "results/model/<kernel>/", "maps, NetCDF, CSV", "every cell, every month")
    folder(ax, 9.25, 0.3, 2.6, 0.7, "results/report/", "report.md")
    pill(ax, 12.35, 0.45, 1.25, 0.4, "END")

    # the winner goes to the ablation (its no-ERA5 twin) and to the full-data fit (row 4, from the left)
    sweep = y3 + h3 + 0.2
    arrow(ax, (12.85, sb_y), (12.85, sweep), (0.1, sweep), (0.1, y4 + 0.9), (0.45, y4 + 0.9))
    arrow(ax, (8.15, sweep), (8.15, row_y[True] + ch / 2), (col_x[True] + cw, row_y[True] + ch / 2))
    label(ax, 10.5, sweep, "winning kernel", size=9)
    # ablation results -> report
    arrow(ax, (12.3, y3 + 0.2), (12.3, y4 + h4 + 0.13), (7.52, y4 + h4 + 0.13), (7.52, y4 + 1.6))
    arrow(ax, (8.85, 1.35), (9.05, 1.35), (9.05, 1.7), (9.25, 1.7))
    arrow(ax, (8.85, 1.35), (9.05, 1.35), (9.05, 0.65), (9.25, 0.65))
    arrow(ax, (11.85, 0.65), (12.35, 0.65))

    fig.savefig(OUT, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    draw()
