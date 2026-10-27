"""Histograms of the Earth-viewing CLARA OLR samples within [0, 500] W m-2 sr-1, overall and per year, and
the same compared with ERA5 and CERES.

Only Earth-viewing samples (read_in.read_clara, earth_view) are kept, and of those only negative values
and values above 500 W m-2 sr-1 are removed; the [OLR_MIN, OLR_MAX] in_range filter is not applied.
In the comparison, ERA5 and CERES are the monthly mean values of all 1 deg cells (ERA5 area-averaged
onto the CERES grid), weighted by cos(lat); all three are normalised to a probability density.
The raw OLR is quantised in steps of OLR_STEP, so each bin spans a whole number of steps, centred on
them; otherwise neighbouring bins would alternately hold more and fewer possible values.

The comparison is also drawn standardised: each dataset as z = (OLR - mean) / std, with its own
(weighted) mean and std over the values in that figure, so the shapes can be compared independently of
units, offset and scale: CLARA is a radiance (W m-2 sr-1; the .save files carry no unit), ERA5 and CERES are
fluxes (W m-2). The legend gives each mean +- std in its own unit.

Output:
    results/olr_hist/olr_hist_all.png
    results/olr_hist/olr_hist_<yyyy>.png
    results/olr_hist/era5_ceres_clara/olr_hist_all.png
    results/olr_hist/era5_ceres_clara/olr_hist_<yyyy>.png
    results/olr_hist/era5_ceres_clara_normalised/olr_hist_all.png
    results/olr_hist/era5_ceres_clara_normalised/olr_hist_<yyyy>.png

Usage:
    python src/plot_olr_hist.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import read_in
from plot_raw_means import RESULTS_DIR

OLR_LO, OLR_HI = 0.0, 500.0  # W m-2 sr-1, samples kept
OLR_STEP = 0.65336  # W m-2 sr-1, quantisation of the raw OLR (values are whole multiples of it)
STEPS_PER_BIN = 8  # bin width = 8 steps ~ 5.2 W m-2 sr-1
BINS = (np.arange(0, OLR_HI / OLR_STEP + STEPS_PER_BIN, STEPS_PER_BIN) - 0.5) * OLR_STEP
Z_BINS = np.linspace(-5, 5, 101)  # standardised OLR, 0.1 std per bin


def plot(olr, title, path):
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.hist(olr, bins=BINS)
    ax.set_title(title)
    ax.set_xlabel("CLARA OLR radiance (W m$^{-2}$ sr$^{-1}$)")
    ax.set_ylabel("samples")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def standardise(values, weights=None):
    """Return ((values - mean) / std, mean, std) with the (weighted) mean and std of values."""
    mean = np.average(values, weights=weights)
    std = np.sqrt(np.average((values - mean) ** 2, weights=weights))
    return (values - mean) / std, mean, std


def plot_compare(series, title, path, normalised=False):
    """series: {label: (values, weights or None)}, drawn as density step histograms on one axis.

    normalised=True standardises each series (see standardise) and plots z instead of physical units. CLARA
    (labels starting with "CLARA") is a radiance in W m-2 sr-1, the others are fluxes in W m-2.
    """
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for label, (values, weights) in series.items():
        if normalised:
            values, mean, std = standardise(values, weights)
            unit = "W m$^{-2}$ sr$^{-1}$" if label.startswith("CLARA") else "W m$^{-2}$"
            label = f"{label}: {mean:.1f} ± {std:.1f} {unit}"
        ax.hist(values, bins=Z_BINS if normalised else BINS, weights=weights, density=True,
                histtype="step", lw=1.5, label=label)
    ax.set_title(title)
    if normalised:
        ax.set_xlabel("standardised OLR, (OLR − mean) / std")
        ax.set_ylabel("probability density (per std)")
    else:
        ax.set_xlabel("OLR (CLARA radiance in W m$^{-2}$ sr$^{-1}$; ERA5 and CERES flux in W m$^{-2}$)")
        ax.set_ylabel("probability density (per unit)")
    long = max(len(label) for label in ax.get_legend_handles_labels()[1]) > 40  # stack long labels
    fig.legend(loc="outside lower center", ncol=1 if long else len(series), frameon=False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def grid_values(ds):
    """Cell values of a monthly (time, lat, lon) Dataset within [OLR_LO, OLR_HI], with cos(lat) weights and years."""
    olr = ds.olr.transpose("time", "lat", "lon")
    w = np.broadcast_to(np.cos(np.radians(olr.lat.values))[None, :, None], olr.shape)
    year = np.broadcast_to(olr.time.dt.year.values[:, None, None], olr.shape)
    v = olr.values
    keep = (v >= OLR_LO) & (v <= OLR_HI)
    return v[keep], w[keep], year[keep]


def compare(clara):
    """Density histograms of CLARA vs ERA5 and CERES, overall and per year."""
    gridded = {"ERA5": grid_values(read_in.era5_to_ceres_grid(read_in.read_era5("monthly"))),
               "CERES EBAF": grid_values(read_in.read_ceres())}
    points = {"CLARA Earth viewing": (clara.olr.to_numpy(), None, clara.time.dt.year.to_numpy())} | gridded
    span = lambda y: f"{y.min()}-{y.max()}" if y.min() != y.max() else f"{y.min()}"
    years = sorted(set().union(*(np.unique(y) for _, _, y in points.values())))
    for folder, normalised, prefix in (("era5_ceres_clara", False, "OLR"),
                                       ("era5_ceres_clara_normalised", True, "Standardised OLR")):
        out_dir = RESULTS_DIR / "olr_hist" / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_compare({f"{k} ({span(y)})": (v, w) for k, (v, w, y) in points.items()},
                     f"{prefix}, all years", out_dir / "olr_hist_all.png", normalised)
        for year in years:
            series = {k: (v[y == year], None if w is None else w[y == year])
                      for k, (v, w, y) in points.items() if (y == year).any()}
            plot_compare(series, f"{prefix}, {year}", out_dir / f"olr_hist_{year}.png", normalised)
        print(f"wrote {1 + len(years)} figures to {out_dir}")


def main():
    clara = read_in.read_clara()
    clara = clara[clara.earth_view & clara.olr.between(OLR_LO, OLR_HI)]
    out_dir = RESULTS_DIR / "olr_hist"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot(clara.olr, f"CLARA Earth-viewing OLR, all years ({len(clara):,} samples)", out_dir / "olr_hist_all.png")
    for year, df in clara.groupby(clara.time.dt.year):
        plot(df.olr, f"CLARA Earth-viewing OLR, {year} ({len(df):,} samples)", out_dir / f"olr_hist_{year}.png")
    print(f"wrote {1 + clara.time.dt.year.nunique()} figures to {out_dir}")
    compare(clara)


if __name__ == "__main__":
    main()
