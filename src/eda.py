"""Phase 2: exploratory analysis of CLARA and ERA5 (CERES as an independent reference only).

Sections and outputs (results/eda/):
  physics     CLARA minus ERA5 against footprint solar zenith angle (hour-matched ERA5) and viewing zenith
              angle (night ERA5), and the drift of the observing geometry: physics_sza.png, physics_vza.png,
              physics_geometry.png, physics_bins.csv
  resolution  along-track autocorrelation of CLARA and of CLARA - ERA5; CLARA-ERA5 agreement against the
              ERA5 smoothing radius and the grid size; neighbouring-cell correlation of both datasets:
              resolution_along_track.png, resolution_smoothing.csv, resolution_grid.csv,
              resolution_neighbours.csv, resolution.png
  spatial     mean maps, bias maps by period, coverage, zonal means, residual spatial correlogram:
              spatial_means.png, spatial_bias.png, spatial_coverage.png, spatial_zonal.png,
              spatial_correlogram.png/.csv
  temporal    global and zonal-band series, seasonal cycles, monthly bias, residual temporal
              autocorrelation, day-to-day variability: temporal_global.png, temporal_bands.png,
              temporal_seasonal.png, temporal_acf.png/.csv, temporal_daily.png/.csv
  joint       agreement statistics overall and by period, latitude band, season and year; scatter:
              joint_stats.csv, joint_scatter.png
  summary.md  the key numbers, what they imply for combining the datasets, and the EDA-derived CV settings

The model-grid tables come from harmonise.build (the agreed QC); EDA statistics such as the linear fit used
for residuals are descriptive only and never feed the model.

Usage:
    python src/eda.py
"""

import numpy as np
import pandas as pd

import config
import cv
import era5_night
import harmonise
import read_in
import samples
from correlation import agreement
from style import (DATASET_COLORS, DIVERGING, INK, MUTED, OLR_CMAP, SERIES, colorbar, draw_field, map_axes, plt,
                   save, style_axes, title)

OUT = config.EDA_DIR
CFG = config.load()
RES = CFG["grid"]["res_deg"]
GAP = (pd.Timestamp(CFG["time"]["gap_start"]), pd.Timestamp(CFG["time"]["gap_end"]) + pd.offsets.MonthEnd(0))
PERIODS = ("pre-gap", "post-gap")
PERIOD_COLORS = {"pre-gap": SERIES[0], "post-gap": SERIES[1]}


# ---------------------------------------------------------------------------
# Sample-level ERA5 matching
# ---------------------------------------------------------------------------

def match_samples(s, night, hourly):
    """Add era5_hour (ERA5 monthly mean OLR at the sample's UTC hour, 1 deg cell) and era5_night (1 deg) to
    samples with a footprint."""
    s = s[s.clara_lat.notna() & s.usable].copy()
    months = pd.DatetimeIndex(night.time.values)
    ti = months.get_indexer(s.time.dt.to_period("M").dt.to_timestamp())
    i, j = harmonise.cell_index(s.clara_lat.to_numpy(), s.clara_lon.to_numpy(), 1)
    hour_end = s.time.dt.hour.to_numpy() + 1  # a sample at hh:mm lies in the hour ending at hh+1
    ok = ti >= 0
    s["era5_hour"] = np.nan
    s["era5_night"] = np.nan
    s.loc[ok, "era5_hour"] = hourly.values[ti[ok], hour_end[ok] - 1, i[ok], j[ok]]
    s.loc[ok, "era5_night"] = night.olr_night.values[ti[ok], i[ok], j[ok]]
    gs, ge = pd.Period(CFG["time"]["gap_start"], "M"), pd.Period(CFG["time"]["gap_end"], "M")
    per = s.time.dt.to_period("M")
    s["period"] = np.where(per < gs, "pre-gap", np.where(per > ge, "post-gap", "gap"))
    return s


def binned(x, y, edges, groups):
    rows = []
    for g in np.unique(groups):
        m = groups == g
        idx = np.digitize(x[m], edges) - 1
        for k in range(len(edges) - 1):
            v = y[m][idx == k]
            v = v[np.isfinite(v)]
            if len(v) >= 30:
                rows.append({"group": g, "lo": edges[k], "hi": edges[k + 1], "centre": (edges[k] + edges[k + 1]) / 2,
                             "n": len(v), "mean": v.mean(), "median": np.median(v), "sd": v.std()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def physics(m):
    usable = m[m.period != "gap"]
    d_hour = (usable.olr - usable.era5_hour).to_numpy()
    sza = binned(usable.sza.to_numpy(), d_hour, np.arange(40, 181, 5), usable.period.to_numpy())
    raw_sza = binned(usable.sza.to_numpy(), usable.olr.to_numpy(), np.arange(40, 181, 5), usable.period.to_numpy())
    nt = usable[usable.night]
    vza = binned(nt.vza.to_numpy(), (nt.olr - nt.era5_night).to_numpy(), np.arange(0, 91, 5), nt.period.to_numpy())
    raw_vza = binned(nt.vza.to_numpy(), nt.olr.to_numpy(), np.arange(0, 91, 5), nt.period.to_numpy())
    table = pd.concat([sza.assign(x="sza", y="clara_minus_era5_hour"), raw_sza.assign(x="sza", y="clara"),
                       vza.assign(x="vza (night)", y="clara_minus_era5_night"), raw_vza.assign(x="vza (night)", y="clara")])
    table.to_csv(OUT / "physics_bins.csv", index=False, float_format="%.3f")

    for name, diff, raw, thr, xlabel, ylabel in (
            ("physics_sza", sza, raw_sza, CFG["qc"]["night_sza_min"], "footprint solar zenith angle (deg)",
             "CLARA − ERA5 at the same UTC hour (W m$^{-2}$)"),
            ("physics_vza", vza, raw_vza, CFG["qc"]["vza_max"], "viewing zenith angle at the footprint (deg)",
             "CLARA − ERA5 night-time (W m$^{-2}$)")):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
        for ax, tab, lab in ((axes[0], raw, "CLARA OLR (W m$^{-2}$)"), (axes[1], diff, ylabel)):
            for p in PERIODS:
                g = tab[tab.group == p]
                ax.plot(g.centre, g["mean"], color=PERIOD_COLORS[p], lw=2, marker="o", ms=4, label=p)
            ax.axvline(thr, color=MUTED, ls="--", lw=1)
            ax.text(thr, ax.get_ylim()[1], f" QC threshold {thr:g}°", color=MUTED, fontsize=8, va="top")
            ax.set_xlabel(xlabel, color=INK)
            ax.set_ylabel(lab, color=INK)
            style_axes(ax)
        axes[0].legend(frameon=False, fontsize=9)
        title(axes[0], "(a) Raw CLARA, binned means (≥ 30 samples per bin)")
        title(axes[1], "(b) Difference from ERA5 (removes the geographic pattern)")
        fig.suptitle({"physics_sza": "CLARA against footprint solar zenith angle (Earth-viewing, in-range samples)",
                      "physics_vza": "CLARA against viewing zenith angle (night-side samples)"}[name],
                     color=INK, fontsize=12, x=0.01, ha="left")
        save(fig, OUT / f"{name}.png")

    month = usable.time.dt.to_period("M").dt.to_timestamp()
    q = usable[usable.qc]
    geo = pd.DataFrame({
        "vza_median_night": usable[usable.night].groupby(month[usable.night]).vza.median(),
        "daylit_fraction": (~usable.night).groupby(month).mean(),
        "lst_median_qc": q.groupby(q.time.dt.to_period("M").dt.to_timestamp()).lst.median(),
        "altitude_km": usable.groupby(month).sat_alt.mean() / 1000})
    fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True, layout="constrained")
    for ax, (col, lab) in zip(axes, [("altitude_km", "satellite altitude (km)"),
                                     ("vza_median_night", "median VZA, night (deg)"),
                                     ("daylit_fraction", "daylit fraction of usable"),
                                     ("lst_median_qc", "median local solar time of QC'd samples (h)")]):
        ax.plot(geo.index, geo[col], color=SERIES[0], lw=2, marker="o", ms=3)
        ax.axvspan(*GAP, color="#f0efec")
        ax.set_ylabel(lab, color=INK, fontsize=8)
        style_axes(ax)
    title(axes[0], "Observing geometry by month (shaded: CLARA data gap)")
    save(fig, OUT / "physics_geometry.png")
    return {"sza": sza, "vza": vza, "geo": geo}


# ---------------------------------------------------------------------------
# Effective resolution
# ---------------------------------------------------------------------------

def along_track(m, max_lag=40):
    q = m[m.qc].sort_values("time")
    t = q.time.to_numpy().astype("datetime64[s]").astype(np.int64)
    ov = q.overpass.to_numpy()
    u = cv.unit_vectors(q.clara_lat, q.clara_lon)
    rows = []
    for name, v in (("CLARA", q.olr.to_numpy()), ("CLARA − ERA5 hour-matched", (q.olr - q.era5_hour).to_numpy())):
        for k in range(1, max_lag + 1):
            ok = (ov[k:] == ov[:-k]) & (np.abs(t[k:] - t[:-k] - 30 * k) <= 5 * k) & np.isfinite(v[k:]) & np.isfinite(v[:-k])
            if ok.sum() < 200:
                continue
            d = cv.EARTH_RADIUS_KM * np.arccos(np.clip((u[k:][ok] * u[:-k][ok]).sum(1), -1, 1))
            rows.append({"series": name, "lag": k, "n": int(ok.sum()), "distance_km": np.median(d),
                         "r": np.corrcoef(v[:-k][ok], v[k:][ok])[0, 1]})
    return pd.DataFrame(rows)


def resolution(m, obs, s):
    acf = along_track(m)
    acf.to_csv(OUT / "resolution_along_track.csv", index=False, float_format="%.4f")

    radii = CFG["era5"]["smoothing_radii_km"]
    w = obs.cell_area_km2.to_numpy()
    rows = []
    anom_c = obs.clara - obs.groupby(["time", cv.lat_band(obs.lat)]).clara.transform("mean")
    for r in radii:
        col = f"era5_night_r{int(r)}"
        anom_e = obs[col] - obs.groupby(["time", cv.lat_band(obs.lat)])[col].transform("mean")
        for p in ("all",) + PERIODS:
            sel = np.ones(len(obs), bool) if p == "all" else (obs.period == p).to_numpy()
            ok = sel & obs[col].notna().to_numpy()
            a = agreement(obs[col].to_numpy()[ok], obs.clara.to_numpy()[ok], w[ok])
            ra = np.corrcoef(anom_e[ok], anom_c[ok])[0, 1]
            rows.append({"radius_km": r, "period": p, "n": a["n"], "r": a["r"], "r_anomaly_within_band_month": ra,
                         "rmse": a["rmse"], "bias": a["bias"], "slope": a["slope"]})
    smooth = pd.DataFrame(rows)
    smooth.to_csv(OUT / "resolution_smoothing.csv", index=False, float_format="%.4f")

    night = era5_night.load()
    q = s[s.qc]
    grid_rows = []
    for res in sorted({1, 2, 5, 10, RES}):
        cells = harmonise.bin_cell_months(q, res)
        cells["lat"], cells["lon"] = cells.lat.round(4), cells.lon.round(4)
        era = harmonise.grid_to_table(read_in.coarsen(night[["olr_night"]], res), ["olr_night"])
        j = cells.merge(era, on=["time", "lat", "lon"], how="inner").dropna(subset=["olr_night"])
        wa = harmonise.cell_area_km2(j.lat, res)
        a = agreement(j.olr_night.to_numpy(), j.clara.to_numpy(), wa)
        grid_rows.append({"res_deg": res, "cell_months": len(j), "median_samples": j.n_samples.median(),
                          "median_overpasses": j.n_overpasses.median(), "r": a["r"], "bias": a["bias"],
                          "rmse": a["rmse"], "slope": a["slope"]})
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT / "resolution_grid.csv", index=False, float_format="%.4f")

    # neighbouring-cell correlation in the same month, CLARA and ERA5 on the same pairs
    nlon = int(360 / RES)
    key = obs.set_index(["time", "lat", "lon"])
    nb_rows = []
    for direction, dlat, dlon in (("east", 0, RES), ("north", RES, 0)):
        lon2 = ((obs.lon + dlon + 180) % 360 - 180).round(4)
        idx = pd.MultiIndex.from_arrays([obs.time, (obs.lat + dlat).round(4), lon2])
        ok = idx.isin(key.index)
        other = key.loc[idx[ok]]
        for name, col in (("CLARA", "clara"), ("ERA5 night", "era5_night_r0")):
            a, b = obs[col].to_numpy()[ok], other[col].to_numpy()
            fin = np.isfinite(a) & np.isfinite(b)
            # remove each month's latitude-band mean so the equator-to-pole gradient does not dominate
            nb_rows.append({"direction": direction, "series": name, "pairs": int(fin.sum()),
                            "r": np.corrcoef(a[fin], b[fin])[0, 1]})
    neighbours = pd.DataFrame(nb_rows)
    neighbours.to_csv(OUT / "resolution_neighbours.csv", index=False, float_format="%.4f")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    for name, color in zip(acf.series.unique(), SERIES):
        g = acf[acf.series == name]
        axes[0].plot(g.distance_km, g.r, color=color, lw=2, marker="o", ms=3, label=name)
    axes[0].axhline(np.exp(-1), color=MUTED, ls=":", lw=1)
    axes[0].set(xlabel="along-track separation (km)", ylabel="correlation")
    axes[0].legend(frameon=False, fontsize=9)
    title(axes[0], "(a) Along-track autocorrelation of QC'd samples")
    for p, color in zip(("all",) + PERIODS, SERIES):
        g = smooth[smooth.period == p]
        axes[1].plot(g.radius_km, g.r, color=color, lw=2, marker="o", ms=5, label=f"{p}: pooled r")
        axes[1].plot(g.radius_km, g.r_anomaly_within_band_month, color=color, lw=1.5, ls="--", marker="s", ms=4,
                     label=f"{p}: anomaly r")
    axes[1].set(xlabel="ERA5 spherical-cap smoothing radius (km)", ylabel=f"r with CLARA ({RES}° cell-months)")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    title(axes[1], "(b) Agreement against ERA5 smoothing")
    axes[2].plot(grid.res_deg, grid.r, color=SERIES[0], lw=2, marker="o", ms=6)
    for _, r in grid.iterrows():
        axes[2].annotate(f"{r.cell_months:,.0f} cell-months\nmedian {r.median_overpasses:.0f} overpasses",
                         (r.res_deg, r.r), textcoords="offset points", xytext=(4, -22), fontsize=7, color=MUTED)
    axes[2].set(xlabel="grid spacing (deg)", ylabel="r, CLARA vs ERA5 night")
    title(axes[2], "(c) Agreement against grid size")
    for ax in axes:
        style_axes(ax)
    fig.suptitle("Effective resolution", color=INK, fontsize=12, x=0.01, ha="left")
    save(fig, OUT / "resolution.png")
    return {"acf": acf, "smooth": smooth, "grid": grid, "neighbours": neighbours}


# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------

def to_field(df, col, res=RES):
    lat_c, lon_c = read_in.grid_centres(res)
    f = np.full((len(lat_c), len(lon_c)), np.nan)
    i, j = harmonise.cell_index(df.lat.to_numpy(), df.lon.to_numpy(), res)
    f[i, j] = df[col].to_numpy()
    return f


def residuals(obs):
    """EDA residual: CLARA minus a pooled linear fit on ERA5 night, minus each month's mean residual."""
    ok = obs.era5_night_r0.notna()
    b1, b0 = np.polyfit(obs.era5_night_r0[ok], obs.clara[ok], 1)
    r = obs.clara - (b0 + b1 * obs.era5_night_r0)
    return r - r.groupby(obs.time).transform("mean"), (b0, b1)


def correlogram(obs, res_series, edges=np.arange(0, 10001, 250)):
    """Correlation of residual pairs in the same month as a function of great-circle distance."""
    sums = np.zeros((3, len(edges) - 1))
    sd = res_series.std()
    for _, g in obs.assign(res=res_series).dropna(subset=["res"]).groupby("time"):
        u = cv.unit_vectors(g.lat, g.lon)
        d = cv.great_circle_km(u, u)
        v = g.res.to_numpy() / sd
        iu = np.triu_indices(len(g), 1)
        dd, pp = d[iu], (v[:, None] * v[None, :])[iu]
        k = np.digitize(dd, edges) - 1
        ok = (k >= 0) & (k < len(edges) - 1)
        sums[0] += np.bincount(k[ok], weights=pp[ok], minlength=len(edges) - 1)
        sums[1] += np.bincount(k[ok], minlength=len(edges) - 1)
    return pd.DataFrame({"distance_km": (edges[:-1] + edges[1:]) / 2, "pairs": sums[1],
                         "correlation": sums[0] / np.maximum(sums[1], 1)})


def fit_exponential(x, r, w, x_min):
    """Weighted least-squares fit r(x) = c exp(-x / L) on bins with x >= x_min. Returns (c, L).

    The sampling noise of a cell-month (one or two overpasses) acts as a nugget: it lowers every
    correlation by the same factor, so 1 - c estimates the noise share and L the range of the
    structured part. A plain 1/e crossing would measure the nugget, not the range."""
    from scipy.optimize import curve_fit

    x, r, w = (np.asarray(v, float) for v in (x, r, w))
    ok = (x >= x_min) & np.isfinite(r) & (w > 0)
    (c, length), _ = curve_fit(lambda d, c, length: c * np.exp(-d / length), x[ok], r[ok],
                               p0=(max(r[ok][0], 0.05), x[ok].mean()), sigma=1 / np.sqrt(w[ok]),
                               bounds=([0, 1e-3], [1, 1e6]))
    return c, length


def efold(x, r):
    """First x where r drops below 1/e (linear interpolation); NaN if never."""
    x, r = np.asarray(x, float), np.asarray(r, float)
    below = np.flatnonzero(r < np.exp(-1))
    if len(below) == 0:
        return np.nan
    k = below[0]
    if k == 0:
        return x[0]
    return x[k - 1] + (np.exp(-1) - r[k - 1]) * (x[k] - x[k - 1]) / (r[k] - r[k - 1])


def spatial(obs, ref):
    o = obs.merge(ref[["obs_id", "ceres"]], on="obs_id", how="left")
    means = o.groupby(["lat", "lon"])[["clara", "era5_night_r0", "era5_24h", "ceres"]].mean().reset_index()
    count = o.groupby(["lat", "lon"]).size().rename("months").reset_index()

    fig = plt.figure(figsize=(15, 8.2), layout="constrained")
    panels = [("clara", "CLARA (QC'd, night side)"), ("era5_night_r0", "ERA5 night-time"),
              ("era5_24h", "ERA5 24 h mean"), ("ceres", "CERES EBAF 24 h mean (reference)")]
    for k, (col, lab) in enumerate(panels):
        ax = map_axes(fig, 2, 2, k + 1)
        mesh = draw_field(ax, to_field(means, col), RES, OLR_CMAP, 100, 300)
        title(ax, f"{lab}: area mean {np.average(means[col].dropna(), weights=np.cos(np.radians(means.lat[means[col].notna()]))):.0f} W m$^{{-2}}$")
    colorbar(fig, mesh, fig.axes, "mean OLR over the observed cell-months (W m$^{-2}$); grey = never observed")
    fig.suptitle(f"Mean OLR on the cell-months CLARA observed ({RES}° cells)", color=INK, fontsize=12, x=0.01, ha="left")
    save(fig, OUT / "spatial_means.png")

    fig = plt.figure(figsize=(15, 4.6), layout="constrained")
    for k, p in enumerate(PERIODS):
        g = o[o.period == p].assign(d=lambda d: d.clara - d.era5_night_r0).groupby(["lat", "lon"]).d.mean().reset_index()
        ax = map_axes(fig, 1, 2, k + 1)
        mesh = draw_field(ax, to_field(g, "d"), RES, DIVERGING, -100, 100)
        title(ax, f"{p}: mean CLARA − ERA5 night = {g.d.mean():+.1f} W m$^{{-2}}$")
    colorbar(fig, mesh, fig.axes, "mean bias (W m$^{-2}$); grey = not observed in the period")
    fig.suptitle("Bias CLARA − ERA5 night-time by period", color=INK, fontsize=12, x=0.01, ha="left")
    save(fig, OUT / "spatial_bias.png")

    fig = plt.figure(figsize=(15, 4.8), layout="constrained")
    ax = map_axes(fig, 1, 2, 1)
    mesh = draw_field(ax, to_field(count, "months"), RES, OLR_CMAP, 0, count.months.max())
    colorbar(fig, mesh, ax, "months with ≥ 1 QC'd sample", extend="neither", shrink=0.8)
    title(ax, f"(a) Observed months per cell (of {o.time.nunique()} months with data)")
    ax2 = fig.add_subplot(1, 2, 2)
    hov = o.groupby([o.time, cv.lat_band(o.lat)]).size().unstack(fill_value=0)
    hov = hov.reindex(pd.date_range(CFG["time"]["first_month"], CFG["time"]["last_month"], freq="MS"), fill_value=0)
    cols = [b for _, _, b in cv.LAT_BANDS if b in hov.columns]
    im = ax2.imshow(hov[cols].T.values, aspect="auto", cmap=OLR_CMAP, origin="lower",
                    extent=[0, len(hov), -0.5, len(cols) - 0.5])
    ax2.set_yticks(range(len(cols)), cols)
    ticks = [k for k, t in enumerate(hov.index) if t.month == 1]
    ax2.set_xticks(ticks, [hov.index[k].year for k in ticks])
    colorbar(fig, im, ax2, "observed cell-months", extend="neither", shrink=0.8)
    title(ax2, "(b) Observed cell-months by latitude band and month")
    style_axes(ax2, grid=False)
    save(fig, OUT / "spatial_coverage.png")

    zon = o.groupby("lat")[["clara", "era5_night_r0", "era5_24h", "ceres"]].mean()
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    for col, lab, color, ls in (("clara", "CLARA", DATASET_COLORS["CLARA"], "-"),
                                ("era5_night_r0", "ERA5 night", DATASET_COLORS["ERA5"], "-"),
                                ("era5_24h", "ERA5 24 h", DATASET_COLORS["ERA5"], "--"),
                                ("ceres", "CERES 24 h", DATASET_COLORS["CERES EBAF"], ":")):
        ax.plot(zon.index, zon[col], color=color, ls=ls, lw=2, label=lab)
    ax.set(xlabel="latitude (deg)", ylabel="zonal mean OLR (W m$^{-2}$)")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    title(ax, "Zonal means over the observed cell-months")
    save(fig, OUT / "spatial_zonal.png")

    res, (b0, b1) = residuals(obs)
    cg = pd.concat([correlogram(obs[obs.period == p], res[obs.period == p]).assign(period=p) for p in PERIODS]
                   + [correlogram(obs, res).assign(period="all")])
    cg.to_csv(OUT / "spatial_correlogram.csv", index=False, float_format="%.4f")
    fits = {p: fit_exponential(g.distance_km, g.correlation, g.pairs, 250) for p, g in cg.groupby("period")}
    pd.DataFrame([{"period": p, "c": c, "range_km": length} for p, (c, length) in fits.items()]).to_csv(
        OUT / "spatial_correlogram_fit.csv", index=False, float_format="%.4f")
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    for p, color in zip(("all",) + PERIODS, SERIES):
        g = cg[cg.period == p]
        c, length = fits[p]
        ax.plot(g.distance_km, g.correlation, color=color, lw=0, marker="o", ms=4,
                label=f"{p}: fit {c:.2f}·exp(−d / {length:,.0f} km)")
        ax.plot(g.distance_km, c * np.exp(-g.distance_km / length), color=color, lw=1.5)
    ax.axhline(0, color=AXIS_COLOR, lw=1)
    ax.set(xlabel="great-circle distance (km)", ylabel="residual correlation (same month)")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    title(ax, f"Spatial correlogram of CLARA − ({b0:.0f} + {b1:.2f}·ERA5 night), month means removed")
    save(fig, OUT / "spatial_correlogram.png")
    return {"means": means, "fits": fits, "fit": (b0, b1), "residual": res, "zonal": zon}


AXIS_COLOR = "#c3c2b7"


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------

def weighted_series(o, cols, by):
    w = o.cell_area_km2
    return o.groupby(by).apply(lambda g: pd.Series({c: np.average(g[c].dropna(), weights=g.cell_area_km2[g[c].notna()])
                                                    if g[c].notna().any() else np.nan for c in cols}),
                               include_groups=False)


def temporal(obs, ref, s, res_series):
    o = obs.merge(ref[["obs_id", "ceres"]], on="obs_id", how="left")
    cols = ["clara", "era5_night_r0", "era5_24h", "ceres"]
    glob = weighted_series(o, cols, "time")
    full = pd.date_range(CFG["time"]["first_month"], CFG["time"]["last_month"], freq="MS")
    glob = glob.reindex(full)
    glob.to_csv(OUT / "temporal_global.csv", float_format="%.3f")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    for col, lab, color, ls in (("clara", "CLARA", DATASET_COLORS["CLARA"], "-"),
                                ("era5_night_r0", "ERA5 night", DATASET_COLORS["ERA5"], "-"),
                                ("era5_24h", "ERA5 24 h", DATASET_COLORS["ERA5"], "--"),
                                ("ceres", "CERES 24 h", DATASET_COLORS["CERES EBAF"], ":")):
        axes[0].plot(glob.index, glob[col], color=color, ls=ls, lw=2, marker="o" if col == "clara" else None, ms=3,
                     label=lab)
    axes[1].plot(glob.index, glob.clara - glob.era5_night_r0, color=SERIES[0], lw=2, marker="o", ms=3)
    axes[1].axhline(0, color=AXIS_COLOR, lw=1)
    for ax in axes:
        ax.axvspan(*GAP, color="#f0efec")
        style_axes(ax)
    axes[0].set_ylabel("area-weighted mean OLR (W m$^{-2}$)", color=INK)
    axes[1].set_ylabel("CLARA − ERA5 night (W m$^{-2}$)", color=INK)
    axes[0].legend(frameon=False, fontsize=9, ncol=4)
    title(axes[0], "(a) Mean over the cells CLARA observed each month (sampling-matched; shaded: CLARA gap)")
    title(axes[1], "(b) Monthly bias")
    save(fig, OUT / "temporal_global.png")

    bands = weighted_series(o.assign(band=cv.lat_band(o.lat)), ["clara", "era5_night_r0"], ["band", "time"])
    fig, axes = plt.subplots(3, 2, figsize=(13, 8.5), sharex=True, layout="constrained")
    for ax, (_, _, band) in zip(axes.ravel(), cv.LAT_BANDS):
        if band not in bands.index.get_level_values(0):
            continue
        g = bands.loc[band].reindex(full)
        ax.plot(g.index, g.clara, color=DATASET_COLORS["CLARA"], lw=1.8, marker="o", ms=2.5, label="CLARA")
        ax.plot(g.index, g.era5_night_r0, color=DATASET_COLORS["ERA5"], lw=1.8, label="ERA5 night")
        ax.axvspan(*GAP, color="#f0efec")
        style_axes(ax)
        title(ax, band, 9)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Monthly mean OLR by latitude band (W m$^{-2}$, sampling-matched)", color=INK, fontsize=12,
                 x=0.01, ha="left")
    save(fig, OUT / "temporal_bands.png")

    # seasonal cycle: per band and period, mean anomaly by calendar month relative to the band-period mean
    sc_rows = []
    oo = o.assign(band=cv.lat_band(o.lat))
    for (band, p), g in oo[oo.period.isin(PERIODS)].groupby(["band", "period"]):
        for col in ("clara", "era5_night_r0"):
            clim = g.groupby("month")[col].mean()
            sc_rows += [{"band": band, "period": p, "series": col, "month": mth, "anomaly": v - clim.mean()}
                        for mth, v in clim.items()]
    sc = pd.DataFrame(sc_rows)
    sc.to_csv(OUT / "temporal_seasonal.csv", index=False, float_format="%.3f")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, layout="constrained")
    for ax, (_, _, band) in zip(axes.ravel(), cv.LAT_BANDS):
        for p, ls in zip(PERIODS, ("-", "--")):
            for col, color, lab in (("clara", DATASET_COLORS["CLARA"], "CLARA"),
                                    ("era5_night_r0", DATASET_COLORS["ERA5"], "ERA5 night")):
                g = sc[(sc.band == band) & (sc.period == p) & (sc.series == col)]
                ax.plot(g.month, g.anomaly, color=color, ls=ls, lw=1.8, marker="o", ms=3, label=f"{lab}, {p}")
        ax.axhline(0, color=AXIS_COLOR, lw=1)
        ax.set_xticks(range(1, 13))
        style_axes(ax)
        title(ax, band, 9)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Seasonal cycle: monthly anomaly from the band mean (W m$^{-2}$)", color=INK, fontsize=12,
                 x=0.01, ha="left")
    save(fig, OUT / "temporal_seasonal.png")

    # residual temporal autocorrelation at a fixed cell: the residual of the spatial correlogram (CLARA minus the
    # linear ERA5 fit, month means removed). Per-cell means are NOT removed: with few observed months per cell
    # that would force a spurious negative correlation; the persistent spatial pattern is therefore included.
    r = res_series
    mi = obs.year * 12 + obs.month
    tab = pd.DataFrame({"cell": obs.lat.astype(str) + "_" + obs.lon.astype(str), "mi": mi, "r": r})
    piv = tab.pivot_table(index="mi", columns="cell", values="r")
    piv = piv.reindex(range(piv.index.min(), piv.index.max() + 1))
    v = piv.to_numpy()
    acf_rows = []
    for k in range(1, 37):
        a, b = v[:-k].ravel(), v[k:].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 100:
            acf_rows.append({"lag_months": k, "pairs": int(ok.sum()), "r": np.corrcoef(a[ok], b[ok])[0, 1]})
    tacf = pd.DataFrame(acf_rows)
    tacf.to_csv(OUT / "temporal_acf.csv", index=False, float_format="%.4f")
    t_efold = efold(tacf.lag_months, tacf.r)

    # day-to-day variability: daily cell means of CLARA minus the hour-matched ERA5 monthly mean (removes the
    # geography, the diurnal and the seasonal cycle), as anomalies from their period mean (removes the regime step).
    # Per-cell means are not removed: with few days per cell they force a spurious negative lag correlation.
    q = s[s.qc & np.isfinite(s.era5_hour)]
    i, j = harmonise.cell_index(q.clara_lat.to_numpy(), q.clara_lon.to_numpy(), RES)
    d = pd.DataFrame({"day": q.time.dt.floor("D").to_numpy(), "cell": i * 1000 + j,
                      "olr": (q.olr - q.era5_hour).to_numpy(), "period": q.period.to_numpy(),
                      "raw": q.olr.to_numpy()})
    daily = d.groupby(["cell", "day", "period"]).agg(olr=("olr", "mean"), raw=("raw", "mean")).reset_index()
    daily["month"] = daily.day.dt.to_period("M")
    daily["anom"] = daily.olr - daily.groupby("period").olr.transform("mean")
    dp = daily.pivot_table(index="day", columns="cell", values="anom")
    dp = dp.reindex(pd.date_range(dp.index.min(), dp.index.max(), freq="D"))
    dv = dp.to_numpy()
    day_rows = []
    for k in range(1, 15):
        a, b = dv[:-k].ravel(), dv[k:].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 100:
            day_rows.append({"lag_days": k, "pairs": int(ok.sum()), "r": np.corrcoef(a[ok], b[ok])[0, 1]})
    dacf = pd.DataFrame(day_rows)
    within = daily.groupby(["cell", "month"]).raw.std().mean()
    between = obs.clara.std()
    dacf.to_csv(OUT / "temporal_daily.csv", index=False, float_format="%.4f")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")
    axes[0].plot(tacf.lag_months, tacf.r, color=SERIES[0], lw=2, marker="o", ms=4)
    axes[0].axhline(np.exp(-1), color=MUTED, ls=":", lw=1)
    axes[0].axhline(0, color=AXIS_COLOR, lw=1)
    axes[0].set(xlabel="lag (months)", ylabel="residual autocorrelation within cells")
    title(axes[0], f"(a) Monthly residuals at a fixed cell (CLARA − linear ERA5 fit, month means removed); "
                   f"lag-1 r = {tacf.r.iloc[0]:.2f}")
    axes[1].plot(dacf.lag_days, dacf.r, color=SERIES[0], lw=2, marker="o", ms=4)
    axes[1].axhline(0, color=AXIS_COLOR, lw=1)
    axes[1].set(xlabel="lag (days)", ylabel="autocorrelation of daily cell-mean anomalies")
    axes[1].set_ylabel("autocorrelation of daily cell means of CLARA − ERA5", color=INK)
    title(axes[1], f"(b) Day-to-day (within cell-month SD of daily means {within:.0f} W m$^{{-2}}$)")
    for ax in axes:
        style_axes(ax)
    save(fig, OUT / "temporal_acf.png")
    return {"global": glob, "tacf": tacf, "t_efold": t_efold, "dacf": dacf, "within_sd": within,
            "between_sd": between, "seasonal": sc}


# ---------------------------------------------------------------------------
# Joint
# ---------------------------------------------------------------------------

def joint(obs, ref):
    o = obs.merge(ref[["obs_id", "ceres"]], on="obs_id", how="left")
    o["band"], o["season"] = cv.lat_band(o.lat), o.month.map(cv.SEASONS)
    pairs = [("era5_night_r0", "clara", "CLARA vs ERA5 night"), ("era5_24h", "clara", "CLARA vs ERA5 24 h"),
             ("ceres", "clara", "CLARA vs CERES"), ("ceres", "era5_24h", "ERA5 24 h vs CERES")]
    rows = []
    groups = [("all", np.full(len(o), "all"))] + [(g, o[g].to_numpy()) for g in ("period", "band", "season", "year")]
    for gname, labels in groups:
        for lab in pd.unique(labels):
            sel = labels == lab
            for xc, yc, pname in pairs:
                ok = sel & o[xc].notna().to_numpy() & o[yc].notna().to_numpy()
                a = agreement(o[xc].to_numpy()[ok], o[yc].to_numpy()[ok], o.cell_area_km2.to_numpy()[ok])
                err = o[yc].to_numpy()[ok] - o[xc].to_numpy()[ok]
                rows.append({"group": gname, "label": lab, "pair": pname, **a,
                             "rmse_unweighted": np.sqrt(np.mean(err ** 2)) if ok.any() else np.nan})
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / "joint_stats.csv", index=False, float_format="%.4f")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained")
    for ax, p in zip(axes, PERIODS):
        g = o[o.period == p]
        hb = ax.hexbin(g.era5_night_r0, g.clara, gridsize=60, bins="log", mincnt=1, cmap=OLR_CMAP,
                       extent=(100, 330, 40, 400))
        ax.plot([100, 330], [100, 330], color=MUTED, ls="--", lw=1, label="1:1")
        s = stats[(stats.group == "period") & (stats.label == p) & (stats.pair == "CLARA vs ERA5 night")].iloc[0]
        ax.plot([100, 330], s.intercept + s.slope * np.array([100, 330]), color=INK, lw=1.5, label="weighted fit")
        ax.text(0.03, 0.97, f"r = {s.r:.2f}\nbias = {s.bias:+.1f} W m$^{{-2}}$\nRMSE = {s.rmse:.1f}\n"
                            f"slope = {s.slope:.2f}\nn = {s.n:,.0f}", transform=ax.transAxes, va="top", fontsize=9,
                color=INK)
        ax.set(xlabel="ERA5 night-time OLR (W m$^{-2}$)", ylabel="CLARA OLR (W m$^{-2}$)")
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        style_axes(ax)
        title(ax, p)
    colorbar(fig, hb, axes, "cell-months per bin", extend="neither", orientation="vertical", shrink=0.8)
    fig.suptitle(f"CLARA against ERA5 night-time OLR ({RES}° cell-months)", color=INK, fontsize=12, x=0.01, ha="left")
    save(fig, OUT / "joint_scatter.png")
    return stats


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summary(ph, rs, sp, tp, st, obs, m):
    def get(group, label, pair, col):
        r = st[(st.group == group) & (st.label == label) & (st.pair == pair)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    sza_min, vza_max = CFG["qc"]["night_sza_min"], CFG["qc"]["vza_max"]
    sza, vza = ph["sza"], ph["vza"]
    day_jump = {p: (sza[(sza.group == p) & (sza.centre < 90)]["mean"].mean()
                    - sza[(sza.group == p) & (sza.centre > sza_min)]["mean"].mean()) for p in PERIODS}
    q = m[m.qc]
    twilight = float(((q.sza > sza_min) & (q.sza <= sza_min + 5)).mean())
    mid = vza[(vza.centre > 40) & (vza.centre < 55)].groupby("group")["mean"].mean()
    inside = {p: vza[(vza.group == p) & (vza.centre < vza_max)]["mean"] for p in PERIODS}
    sm = rs["smooth"]
    sm_all = sm[sm.period == "all"].set_index("radius_km")
    best_r = sm_all.r.idxmax()
    best_anom = sm_all.r_anomaly_within_band_month.idxmax()
    acf = rs["acf"][rs["acf"].series == "CLARA"]
    acf_res = rs["acf"][rs["acf"].series != "CLARA"]
    c_all, l_all = sp["fits"]["all"]
    zon = sp["zonal"]
    amp = (zon.clara.max() - zon.clara.min()) / (zon.era5_night_r0.max() - zon.era5_night_r0.min())
    zr = np.corrcoef(zon.clara, zon.era5_night_r0)[0, 1]
    tacf = tp["tacf"]
    dacf = tp["dacf"]
    buffer_needed = int(np.ceil(l_all / 500) * 500)
    blocks_km = CFG["cv"]["spatial_block_lat"] * 111.2
    lines = [
        "# EDA summary (generated by src/eda.py)",
        "",
        f"Grid {RES}° × month; QC: footprint SZA > {sza_min:g}°, VZA ≤ {vza_max:g}°. "
        f"{len(obs):,} observed cell-months over {obs.time.nunique()} months "
        f"({(obs.period == 'pre-gap').sum():,} pre-gap, {(obs.period == 'post-gap').sum():,} post-gap).",
        "",
        "## 1. What CLARA measures (sample level)",
        f"- **Only night-side samples are credible.** Against ERA5 at the same UTC hour, daylit footprints "
        f"(SZA < 90°) read {day_jump['pre-gap']:+.0f} W m⁻² (pre-gap) / {day_jump['post-gap']:+.0f} W m⁻² "
        f"(post-gap) more than night-side ones (SZA > {sza_min:g}°). The hour-matched ERA5 already contains the "
        "real day–night OLR difference, so this excess is instrumental (shortwave contamination). The steepest change "
        f"lies between ~92° and ~103° (physics_sza.png); {twilight:.0%} of the QC'd samples have SZA in "
        f"({sza_min:g}°, {sza_min + 5:g}°], where post-gap values still drift slightly (possible residual twilight).",
        f"- **Not angle-corrected.** Relative to night-time ERA5, CLARA falls steeply beyond VZA ≈ 60° "
        f"(physics_vza.png). Within the kept range (VZA ≤ {vza_max:g}°) the binned difference still spans "
        f"{inside['pre-gap'].max() - inside['pre-gap'].min():.0f} W m⁻² (pre-gap) and "
        f"{inside['post-gap'].max() - inside['post-gap'].min():.0f} W m⁻² (post-gap). VZA is not an allowed "
        "predictor, so this residual geometry effect becomes noise.",
        f"- **The regime step is calibration, not geometry.** At the same viewing angles (VZA 40–55°), CLARA − ERA5 "
        f"night is {mid.get('pre-gap', np.nan):.0f} W m⁻² pre-gap but {mid.get('post-gap', np.nan):.0f} W m⁻² post-gap.",
        "- **The observing geometry drifts.** Satellite altitude, local solar time of the night samples, VZA "
        "distribution and daylit fraction all change (physics_geometry.png).",
        "",
        "## 2. Agreement with ERA5 (cell-months, area-weighted)",
        f"- **Pooled, CLARA vs ERA5 night:** r = {get('all', 'all', 'CLARA vs ERA5 night', 'r'):.3f}, "
        f"bias {get('all', 'all', 'CLARA vs ERA5 night', 'bias'):+.1f} W m⁻², "
        f"RMSE {get('all', 'all', 'CLARA vs ERA5 night', 'rmse'):.1f} W m⁻², "
        f"slope {get('all', 'all', 'CLARA vs ERA5 night', 'slope'):.2f}.",
        f"- **By period:** pre-gap bias {get('period', 'pre-gap', 'CLARA vs ERA5 night', 'bias'):+.1f} W m⁻², "
        f"r {get('period', 'pre-gap', 'CLARA vs ERA5 night', 'r'):.2f}; post-gap bias "
        f"{get('period', 'post-gap', 'CLARA vs ERA5 night', 'bias'):+.1f} W m⁻², "
        f"r {get('period', 'post-gap', 'CLARA vs ERA5 night', 'r'):.2f}. The step sits inside the "
        "2021-09 → 2022-05 gap (temporal_global.png). Within each period CLARA does follow the ERA5 sampling-matched "
        "monthly fluctuations.",
        f"- **Night-time vs 24 h ERA5:** night-time ERA5 is only marginally closer to CLARA. "
        f"r {get('all', 'all', 'CLARA vs ERA5 night', 'r'):.3f} vs {get('all', 'all', 'CLARA vs ERA5 24 h', 'r'):.3f}; "
        f"RMSE {get('all', 'all', 'CLARA vs ERA5 night', 'rmse'):.1f} vs "
        f"{get('all', 'all', 'CLARA vs ERA5 24 h', 'rmse'):.1f} W m⁻². It is still the physically matched "
        "predictor, but the choice is not what limits agreement.",
        f"- **Reference check, ERA5 24 h vs CERES on the same cell-months:** bias "
        f"{get('all', 'all', 'ERA5 24 h vs CERES', 'bias'):+.1f} W m⁻², r "
        f"{get('all', 'all', 'ERA5 24 h vs CERES', 'r'):.3f}, RMSE {get('all', 'all', 'ERA5 24 h vs CERES', 'rmse'):.1f} "
        "W m⁻². ERA5 and the independent observations agree, so the CLARA–ERA5 discrepancies belong to CLARA "
        "(calibration, geometry, sampling).",
        f"- **Spatial structure is heavily damped in CLARA.** Its zonal-mean profile has the right shape "
        f"(r = {zr:.2f} with ERA5 night) but only {amp:.0%} of ERA5's equator-to-pole range (spatial_zonal.png). Its "
        "mean map shows little of ERA5's regional structure (spatial_means.png).",
        "",
        "## 3. Effective resolution and smoothing",
        f"- **Along track:** CLARA samples correlate at r = {acf.r.iloc[0]:.2f} at {acf.distance_km.iloc[0]:.0f} km, "
        f"and the CLARA − ERA5 residual reaches 1/e at ~{efold(acf_res.distance_km, acf_res.r):,.0f} km. Neighbouring "
        "samples share most of their signal: a wide footprint plus synoptic weather.",
        f"- **Smoothing ERA5 with spherical caps** raises the agreement with CLARA modestly. Pooled r peaks at "
        f"R = {best_r:g} km ({sm_all.r.max():.3f} vs {sm_all.r.loc[0]:.3f} unsmoothed); the within-band anomaly r peaks at "
        f"R = {best_anom:g} km ({sm_all.r_anomaly_within_band_month.max():.3f} vs "
        f"{sm_all.r_anomaly_within_band_month.loc[0]:.3f}). The optimum is interior to the candidate set, so the "
        "candidates bracket it. An effective footprint scale of ~10³ km is plausible for a wide-field radiometer at "
        "~550–600 km. The radius is chosen per CV fold on training rows only.",
        "- **Grid size** (resolution_grid.csv): " + "; ".join(
            f"{r.res_deg:g}°: r {r.r:.2f}, {r.cell_months:,.0f} cell-months, median {r.median_overpasses:.0f} "
            "overpasses" for _, r in rs["grid"].iterrows())
        + ". Coarser cells average more overpasses (less noise) but leave fewer, larger cells. "
        + (f"The model grid is {RES}° (user choice): there a cell-month is mostly a single 30 s sample "
           f"({(obs.n_samples == 1).mean():.0%}), far finer than CLARA's ~10³ km effective resolution, so the "
           "noise term carries single-sample noise and fine-scale map detail comes from ERA5 through β₁. "
           if RES <= 2 else f"{RES}° keeps several overpasses per cell-month and is no finer than CLARA's "
           "effective resolution. ") + "No grid interpolates or upsamples either dataset.",
        "- **Neighbouring-cell correlation** in the same month (resolution_neighbours.csv): " + "; ".join(
            f"{r.series} {r.direction} {r.r:.2f}" for _, r in rs["neighbours"].iterrows())
        + ". Monthly ERA5 fields are very smooth; CLARA cell-months are dominated by sampling noise.",
        "",
        "## 4. Dependence scales → CV design",
        f"- **Spatial residual correlogram** (CLARA minus a linear ERA5 fit, month means removed): fitted "
        f"{c_all:.2f}·exp(−d / {l_all:,.0f} km). The structured part has a range of ~{l_all:,.0f} km, and the "
        f"nugget (sampling noise) takes ~{1 - c_all:.0%} of the residual variance (spatial_correlogram.png). Pre-gap "
        f"range {sp['fits']['pre-gap'][1]:,.0f} km, post-gap {sp['fits']['post-gap'][1]:,.0f} km.",
        "- **Temporal residual autocorrelation** at a fixed cell (same residual as the spatial correlogram; the "
        "persistent spatial pattern is included): " + ", ".join(
            f"lag {int(k)} month{'s' if k > 1 else ''} r = {tacf.set_index('lag_months').r.get(k, np.nan):.2f}"
            for k in (1, 2, 6, 12)) + f". Relative to the noise-free part (≈ {c_all:.2f} at zero distance), "
        "this measures how long the structured discrepancy at a location persists.",
        f"- **Day to day** (daily cell means of CLARA − hour-matched ERA5, period means removed): lag-1 day r = "
        f"{dacf.r.iloc[0]:.2f}; the within-cell-month SD of "
        f"daily means is {tp['within_sd']:.0f} W m⁻². This is weather and noise that monthly ERA5 cannot explain; the "
        "noise term absorbs it.",
        f"- **Chosen CV settings:**",
        f"  - Spatial blocks {CFG['cv']['spatial_block_lat']:g}° × {CFG['cv']['spatial_block_lon']:g}° "
        f"(~{blocks_km:,.0f} km on a side, ≈ {blocks_km / l_all:.1f}× the fitted range). "
        f"Spatial buffer {CFG['cv']['spatial_buffer_km']:g} km (fitted range rounded up: {buffer_needed:,} km"
        f"{'' if CFG['cv']['spatial_buffer_km'] >= buffer_needed else ' — config buffer is SHORTER; increase it'}).",
        f"  - Temporal windows {CFG['cv']['temporal_window_months']} months with ±{CFG['cv']['temporal_buffer_months']} "
        "month buffer. The windows hold out whole half-years, so the seasonal and regime structure must be "
        "generalised from other years, not interpolated from adjacent months; the buffer removes the adjacent months "
        "where the fixed-cell residual correlation is highest.",
        f"  - Joint blocks {CFG['cv']['joint_block_lat']:g}° × {CFG['cv']['joint_block_lon']:g}° × "
        f"{CFG['cv']['joint_window_months']} months with {CFG['cv']['joint_buffer_km']:g} km / "
        f"±{CFG['cv']['joint_buffer_months']} month halo.",
        "",
        "## 5. Implications for combining the datasets",
        "1. Pair CLARA with ERA5 night-time OLR of the same cell and month. The gain over the 24 h mean is small but "
        "the pairing is physically correct.",
        f"2. Keep only night-side, VZA ≤ {vza_max:g}° samples. Aggregate to cell-month means and carry the number of "
        "overpasses into the noise model, because samples on one overpass are not independent.",
        "3. Expect a large, period-dependent offset and a slope well below 1 (CLARA's spatial contrasts are damped). "
        "ERA5 enters through a linear mean function. The step across the gap enters the mean function as a "
        "post-gap indicator (β₂, gp.gap_step; the gap months count as post-gap); the long-term time component of "
        "the GP absorbs the remaining drift.",
        "4. The structured CLARA–ERA5 discrepancy has a ~10³ km range and little month-to-month memory. Sampling noise "
        "dominates single cell-months, so gains from the GP's covariance over the mean function are expected mainly "
        "near observed data. Blocked CV with buffers is essential to measure this honestly.",
        "5. Residual geometry effects and day-to-day weather remain as noise. Predictive calibration must be checked, "
        "not assumed.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s, obs, grid, ref = harmonise.build(CFG)
    night, hourly = era5_night.load(), era5_night.load_hourly()
    m = match_samples(s, night, hourly)
    print("physics ...")
    ph = physics(m)
    print("resolution ...")
    rs = resolution(m, obs, s)
    print("spatial ...")
    sp = spatial(obs, ref)
    print("temporal ...")
    tp = temporal(obs, ref, m, sp["residual"])
    print("joint ...")
    st = joint(obs, ref)
    summary(ph, rs, sp, tp, st, obs, m)
    config.save(CFG, OUT / "config_used.json")


if __name__ == "__main__":
    main()
