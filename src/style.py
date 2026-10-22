"""Shared figure style for the EDA, model and map figures (reuses plot_raw_means' palette).

Colour by job: sequential blue for OLR magnitude, sequential orange for uncertainty, diverging
blue <-> red around a neutral grey for signed differences (bias, residuals), and the first three
categorical slots (blue, orange, aqua; the slots that stay distinguishable for every pair) for the
three datasets and for the three kernels, always in this fixed order.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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
BASELINE_LABELS = {"linear": "Regression on ERA5 (no GP)", "linear_noera5": "Constant (no GP, no ERA5)"}


def model_label(model):
    """Display name of a model id: a GP kernel, <kernel>_noera5, linear or linear_noera5."""
    if model in KERNEL_LABELS:
        return f"{KERNEL_LABELS[model]} GP + ERA5"
    if model.removesuffix("_noera5") in KERNEL_LABELS:
        return f"{KERNEL_LABELS[model.removesuffix('_noera5')]} GP, no ERA5"
    return BASELINE_LABELS[model]


def model_color(model):
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
