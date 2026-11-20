"""Shared figure style for the EDA, model and map figures (reuses plot_raw_means' palette).

Colour by job: sequential blue for OLR magnitude, sequential orange for uncertainty, diverging
blue <-> red around a neutral grey for signed differences (bias, residuals), and the first three
categorical slots (blue, orange, aqua; the slots that stay distinguishable for every pair) for the
three datasets and for the three kernels, always in this fixed order. The pre- and post-gap periods are a light
and a dark step of CLARA's blue; counts of observations take a third sequential hue (aqua).

The paper figures (src/paper_figures.py) add print rules: one-column (8.5 cm) or full (17.5 cm) width, every text
>= 8 pt at that size, bold panel letters; save_paper() refuses a figure that breaks them.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text

import read_in
from plot_raw_means import CMAP as OLR_CMAP, INK, MUTED  # noqa: F401  (re-exported)

try:
    import cartopy.crs as ccrs
except ImportError:  # maps fall back to plain lat-lon axes without coastlines
    ccrs = None

GRID, AXIS = "#e1e0d9", "#c3c2b7"
NO_DATA = "#d6d5d0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3: blue, orange, aqua
DATASET_COLORS = {"CLARA": SERIES[0], "ERA5": SERIES[1], "CERES EBAF": SERIES[2]}
KERNEL_COLORS = {"exponential": SERIES[0], "gaussian": SERIES[1], "matern32": SERIES[2]}
KERNEL_LABELS = {"exponential": "Exponential", "gaussian": "Gaussian (SE)", "matern32": "Matérn 3/2"}
# baselines (src/baselines.py): a GP without ERA5 keeps its kernel's colour (drawn hollow); regression-only
# models are neutral
BASELINE_COLORS = {"linear": INK, "linear_noera5": MUTED}
BASELINE_LABELS = {"linear": "Regression on ERA5 (no GP)", "linear_noera5": "Pre/post-gap constant (no GP, no ERA5)"}
# Units: CLARA's OLR is a radiance, ERA5 and CERES are fluxes. Never subtract one from the other; compare them
# through correlations, ratios (sr-1) or values relative to each series' own mean.
CLARA_UNIT, FLUX_UNIT, RATIO_UNIT = "W m$^{-2}$ sr$^{-1}$", "W m$^{-2}$", "sr$^{-1}$"


def model_label(model):
    """Display name of a model id: a GP kernel, <kernel>_noera5, linear or linear_noera5, or a _gapslope
    sensitivity model."""
    if model.endswith("_gapslope"):
        return model_label(model.removesuffix("_gapslope")).replace(" (no GP)", "") + ", period-specific slope"
    if model in KERNEL_LABELS:
        return f"{KERNEL_LABELS[model]} GP + ERA5"
    if model.removesuffix("_noera5") in KERNEL_LABELS:
        return f"{KERNEL_LABELS[model.removesuffix('_noera5')]} GP, no ERA5"
    return BASELINE_LABELS[model]


def model_color(model):
    model = model.removesuffix("_gapslope")
    return BASELINE_COLORS.get(model) or KERNEL_COLORS[model.removesuffix("_noera5")]

DIVERGING = LinearSegmentedColormap.from_list("div", [
    "#0d366b", "#3987e5", "#9ec5f4", "#f0efec", "#f4b3b1", "#e34948", "#8f1f1f"])
DIVERGING.set_bad(NO_DATA)
UNCERTAINTY_CMAP = LinearSegmentedColormap.from_list("unc", [
    "#fbe3d7", "#f6b496", "#f08a5f", "#eb6834", "#c24d1f", "#8f3412", "#5c1f09"])
UNCERTAINTY_CMAP.set_bad(NO_DATA)
OLR_CMAP.set_bad(NO_DATA)


def style_axes(ax, grid=True):
    ax.tick_params(colors=MUTED, labelsize=8)
    for side, spine in ax.spines.items():
        spine.set_color(AXIS)
        spine.set_visible(side in ("left", "bottom"))
    if grid:
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def title(ax, text, size=10):
    ax.set_title(text, color=INK, fontsize=size, loc="left")


def map_axes(fig, *args, projection="robinson"):
    """A global map axes (Robinson with coastlines when cartopy is available)."""
    if ccrs is None:
        ax = fig.add_subplot(*args)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        style_axes(ax, grid=False)
        return ax
    proj = ccrs.Robinson() if projection == "robinson" else ccrs.PlateCarree()
    ax = fig.add_subplot(*args, projection=proj)
    ax.set_global()
    ax.coastlines(linewidth=0.4, color=INK)
    return ax


def draw_field(ax, values, res, cmap, vmin, vmax):
    """pcolormesh of a (lat, lon) field on the res deg grid of read_in.grid_edges."""
    lat_edges, lon_edges = read_in.grid_edges(round(res))
    extra = {"transform": ccrs.PlateCarree()} if ccrs is not None else {}
    return ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="flat", **extra)


def colorbar(fig, mesh, ax, label, extend="both", orientation="horizontal", **kw):
    cb = fig.colorbar(mesh, ax=ax, orientation=orientation, extend=extend, shrink=kw.pop("shrink", 0.6),
                      aspect=kw.pop("aspect", 35), pad=kw.pop("pad", 0.04), **kw)
    cb.set_label(label, color=INK, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    return cb


def save(fig, path, dpi=130):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Paper figures (src/paper_figures.py): drawn at their print size, never scaled by LaTeX
# ---------------------------------------------------------------------------

CM = 1 / 2.54                              # inches per cm
COLUMN_W, PAGE_W = 8.5 * CM, 17.5 * CM     # one column and the full text width of docs/paper/main.tex
MIN_FONT_PT = 8                            # every text of a paper figure, at print size

# The two calibration periods are two steps of CLARA's blue (ordinal: light = earlier); "all" is ink. They never
# share a figure with the kernel or dataset colours, so blue/orange/aqua keep one meaning each.
PERIOD_COLORS = {"pre-gap": "#6da7ec", "post-gap": "#104281", "all": INK}
GAP_FILL = "#ebeae6"                       # the 2021-22 CLARA data gap on every time axis
# Counts of observations (coverage maps, scatter density): the third sequential hue, after OLR (blue) and
# uncertainty (orange), so a map of counts is never read as a map of OLR.
COUNT_CMAP = LinearSegmentedColormap.from_list("count", [
    "#d9f2e8", "#a6e0c9", "#6cc9a5", "#1baf7a", "#138a5f", "#0e6646", "#0a442f"])
COUNT_CMAP.set_bad(NO_DATA)

PAPER_RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": MIN_FONT_PT, "axes.titlesize": MIN_FONT_PT, "axes.labelsize": MIN_FONT_PT,
    "xtick.labelsize": MIN_FONT_PT, "ytick.labelsize": MIN_FONT_PT, "legend.fontsize": MIN_FONT_PT,
    "legend.title_fontsize": MIN_FONT_PT, "figure.titlesize": MIN_FONT_PT,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "mathtext.cal": "Arial:italic",
    "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titlepad": 3, "axes.labelpad": 2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2, "ytick.major.pad": 2, "xtick.color": MUTED, "ytick.color": MUTED,
    "lines.linewidth": 1.2, "lines.markersize": 3.5, "hatch.linewidth": 0.4, "hatch.color": MUTED,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.3,
    "legend.labelspacing": 0.3, "legend.columnspacing": 1.2, "legend.borderpad": 0.2,
    "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
    "figure.constrained_layout.hspace": 0.03, "figure.constrained_layout.wspace": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "axes.unicode_minus": True,
}


def use_paper_style():
    plt.rcParams.update(PAPER_RC)


def paper_figure(width, height_cm, layout="constrained", **kw):
    """A figure exactly `width` (COLUMN_W or PAGE_W) wide and height_cm tall, constrained layout by default."""
    if width not in (COLUMN_W, PAGE_W):
        raise ValueError("a paper figure is one column (COLUMN_W) or the full width (PAGE_W)")
    return plt.figure(figsize=(width, height_cm * CM), layout=layout, **kw)


def panel(ax, letter, text=""):
    """Panel letter in bold, then an optional short panel title, on the left above the axes."""
    ax.set_title(rf"$\mathbf{{({letter})}}$" + (f" {text}" if text else ""), loc="left", color=INK,
                 fontsize=MIN_FONT_PT)


def print_problems(fig):
    """What breaks the print rules: a width other than one column or the full width, or text below MIN_FONT_PT."""
    w = fig.get_figwidth()
    out = [] if any(abs(w - target) < 1e-6 for target in (COLUMN_W, PAGE_W)) else [f"width {w / CM:.2f} cm"]
    fig.canvas.draw()
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_FONT_PT - 1e-6:
            out.append(f"{t.get_fontsize():g} pt text {t.get_text()[:40]!r}")
    return out


def save_paper(fig, path):
    """Check the print rules, then write <path>.pdf (vector, for LaTeX) and png/<name>.png (300 dpi preview)."""
    problems = print_problems(fig)
    if problems:
        plt.close(fig)
        raise ValueError(f"{path.name}: " + "; ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "png").mkdir(exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), facecolor="white", dpi=600,  # dpi of the rasterised fields only
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path.parent / "png" / (path.stem + ".png"), facecolor="white", dpi=300)
    size = fig.get_size_inches() / CM
    plt.close(fig)
    return size
