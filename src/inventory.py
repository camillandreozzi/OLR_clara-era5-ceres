"""Phase 1: inventory of every CLARA, ERA5 and CERES input file, read directly from the files.

Nothing is modified: the CLARA archives are read through their existing extraction in
data/CLARA/extracted/ (read_in._extract), the NetCDF files are opened read-only.

Output (results/eda/inventory/):
    clara_olr_files.csv          one row per monthly .save file: samples, time span, cadence, value ranges
    clara_geolocation_files.csv  one row per daily geolocation CSV: rows, cadence, viewing-state counts
    gridded_products.csv         one row per NetCDF file: dims, grid, conventions, time stamps, units, ranges
    coverage_by_month.csv        per month: CLARA samples at each QC stage, days with data, ERA5 / CERES presence
    dataset_comparison.csv/.md   side-by-side comparison of CLARA and ERA5 (CERES as reference), with the
                                 numbers above filled in

Usage:
    python src/inventory.py
"""

import re

import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import readsav

import config
import read_in
import samples

OUT_DIR = config.EDA_DIR / "inventory"
OLR_STEP = 0.65336  # W m-2 sr-1, quantisation of the raw CLARA OLR (plot_olr_hist.OLR_STEP)
SUN_VIEW = 1000.0  # raw values above this are sun-viewing periods


# ---------------------------------------------------------------------------
# CLARA raw files
# ---------------------------------------------------------------------------

def clara_olr_files():
    rows = []
    winners = {}
    for archive in sorted(read_in.CLARA_DIR.glob("CLARA_OLR*.tar.gz")):
        for path in sorted(read_in._extract(archive).glob("*.save")):
            m = read_in.OLR_FILE_RE.search(path.name)
            if not m:
                continue
            key = (int(m.group(1)), int(m.group(2)))
            winners[key] = archive.name  # a later (yearly) archive wins, as in read_in.read_clara_olr
            rec = readsav(str(path))["olr_all"][0]
            jday = np.asarray(rec["JDAY"], dtype=np.float64)
            olr = np.asarray(rec["OLR"], dtype=np.float64)
            dt = np.diff(jday) * 86400
            earthish = olr[(olr > 0) & (olr < SUN_VIEW)]
            steps = np.round(earthish / OLR_STEP, 3)
            rows.append({
                "archive": archive.name, "file": path.name, "year": key[0], "month": key[1],
                "fields": ",".join(rec.dtype.names), "n_samples": len(olr),
                "t_first_utc": pd.to_datetime(jday.min() - read_in.JULIAN_UNIX_EPOCH, unit="D").round("s"),
                "t_last_utc": pd.to_datetime(jday.max() - read_in.JULIAN_UNIX_EPOCH, unit="D").round("s"),
                "cadence_median_s": np.median(dt) if len(dt) else np.nan,
                "cadence_p99_s": np.percentile(dt, 99) if len(dt) else np.nan,
                "max_gap_h": dt.max() / 3600 if len(dt) else np.nan,
                "n_nan": int(np.isnan(olr).sum()), "n_negative": int((olr < 0).sum()),
                "n_sun_view": int((olr > SUN_VIEW).sum()),
                "olr_min": np.nanmin(olr), "olr_max": np.nanmax(olr),
                "olr_earth_median": np.median(earthish) if len(earthish) else np.nan,
                "frac_on_quantisation_grid": float(np.mean(np.abs(steps - np.round(steps)) < 1e-2))
                if len(steps) else np.nan,
            })
    df = pd.DataFrame(rows)
    df["used"] = [winners[(y, m)] == a for y, m, a in zip(df.year, df.month, df.archive)]
    return df.sort_values(["year", "month", "archive"]).reset_index(drop=True)


TUPLE = re.compile(read_in.TUPLE_RE)


def clara_geolocation_files():
    rows = []
    for archive in sorted(read_in.CLARA_DIR.glob("CLARA_lat_lon_*.tar.gz")):
        for path in sorted(read_in._extract(archive).glob("*.csv")):
            m = read_in.GEO_FILE_RE.search(path.name)
            if not m:
                continue
            df = pd.read_csv(path)
            df.columns = [c.lstrip("# ").strip() for c in df.columns]
            fov = df["clara_lat_long"].fillna("")
            zero = fov.str.match(r"^\(\s*0(\.0+)?\s*,\s*0(\.0+)?\s*,\s*0(\.0+)?\s*\)$")
            empty = fov.str.strip() == ""
            t = pd.to_timedelta(df["datetime"]).dt.total_seconds().to_numpy()
            dt = np.diff(t)
            rows.append({
                "archive": archive.name, "file": path.name,
                "date": pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3))).date(),
                "test_version": "_test" in path.name, "columns": "|".join(df.columns), "n_rows": len(df),
                "cadence_median_s": np.median(dt) if len(dt) else np.nan,
                "max_gap_min": dt.max() / 60 if len(dt) else np.nan,
                "n_earth_view": int((~zero & ~empty).sum()), "n_looking_away": int(zero.sum()),
                "n_above_limb": int(empty.sum()),
                "z_angle_min": df.z_vector_satellite_position_angles.min(),
                "z_angle_max": df.z_vector_satellite_position_angles.max(),
            })
    df = pd.DataFrame(rows)
    # read_in.read_clara_geolocation keeps the final version when a day also has a _test file
    df["used"] = ~(df.test_version & df.duplicated("date", keep=False))
    return df.sort_values(["date", "test_version"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gridded products
# ---------------------------------------------------------------------------

def _describe_nc(path, product, var):
    with xr.open_dataset(path) as ds:
        da = ds[var]
        lat = next(c for c in ("latitude", "lat") if c in ds.coords)
        lon = next(c for c in ("longitude", "lon") if c in ds.coords)
        tname = next(c for c in ("valid_time", "time") if c in ds.coords)
        t = pd.to_datetime(ds[tname].values)
        v = da.values
        lats, lons = ds[lat].values, ds[lon].values
        return {
            "product": product, "file": path.name, "variable": var,
            "long_name": da.attrs.get("long_name", ""), "units": da.attrs.get("units", ""),
            "grib_step_type": da.attrs.get("GRIB_stepType", ""),
            "dims": str(dict(da.sizes)), "dtype": str(da.dtype),
            "lat_first": lats[0], "lat_last": lats[-1], "lat_step": float(np.diff(lats).mean()),
            "lon_first": lons[0], "lon_last": lons[-1], "lon_step": float(np.diff(lons).mean()),
            "grid_type": "points on cell edges (incl. poles)" if abs(abs(lats[0]) - 90) < 1e-6 else "cell centres",
            "n_times": len(t), "time_first": t.min(), "time_last": t.max(),
            "time_stamp_hours": ",".join(sorted({f"{h:02d}" for h in t.hour})),
            "time_stamp_days": ",".join(sorted({f"{d:02d}" for d in t.day})),
            "n_nan": int(np.isnan(v).sum()), "value_min": float(np.nanmin(v)), "value_max": float(np.nanmax(v)),
            "missing_value_attr": da.attrs.get("GRIB_missingValue", da.attrs.get("_FillValue", "")),
            "valid_range_attr": f"{da.attrs.get('valid_min', '')}..{da.attrs.get('valid_max', '')}".strip(),
        }


def gridded_products():
    rows = []
    for path in sorted((read_in.ERA5_DIR / "monthly").glob("*.nc")):
        rows.append(_describe_nc(path, "ERA5 monthly means of daily means", "ttr"))
    for path in sorted((read_in.ERA5_DIR / "monthly_by_hour").glob("*.nc")):
        rows.append(_describe_nc(path, "ERA5 monthly means by hour of day", "ttr"))
    for path in sorted(read_in.CERES_DIR.glob("CERES_EBAF*.nc")):
        rows.append(_describe_nc(path, "CERES EBAF Ed4.2.1", "toa_lw_all_mon"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage_by_month(s):
    month = s.time.dt.to_period("M")
    stages = {"raw": np.ones(len(s), bool), "geolocated": s.sat_lat.notna().to_numpy(),
              "earth_view": s.earth_view.to_numpy(), "usable": s.usable.to_numpy(),
              "usable_night": (s.usable & s.night).to_numpy(), "qc": s.qc.to_numpy()}
    cov = pd.DataFrame({f"n_{k}": pd.Series(v).groupby(month.to_numpy()).sum() for k, v in stages.items()})
    q = s[s.qc]
    qm = q.time.dt.to_period("M")
    cov["days_with_qc"] = q.time.dt.date.groupby(qm).nunique()
    cov["overpasses_qc"] = q.overpass.groupby(qm).nunique()
    cov["daylit_frac_of_usable"] = (s.usable & ~s.night & s.sza.notna()).groupby(month).sum() / cov.n_usable
    full = pd.period_range(cfg_first_month(), cfg_last_month(), freq="M")
    cov = cov.reindex(full).fillna(0)
    era5 = {pd.Period(t, "M") for t in read_in.read_era5("monthly").indexes["time"]}
    ceres = {pd.Period(t, "M") for t in read_in.read_ceres().indexes["time"]}
    cov["era5_monthly"] = [p in era5 for p in cov.index]
    cov["ceres"] = [p in ceres for p in cov.index]
    cov.index.name = "month"
    return cov.reset_index().astype({c: int for c in cov.columns if c.startswith(("n_", "days", "overpasses"))})


_CFG = config.load()
cfg_first_month = lambda: _CFG["time"]["first_month"]
cfg_last_month = lambda: _CFG["time"]["last_month"]


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------

def comparison(olr_files, geo_files, grids, cov, s):
    used = olr_files[olr_files.used]
    era = grids[grids["product"].str.startswith("ERA5 monthly means of")]
    eh = grids[grids["product"].str.startswith("ERA5 monthly means by hour")]
    ce = grids[grids["product"].str.startswith("CERES")]
    q = s[s.qc]
    missing = cov.loc[cov.n_usable == 0, "month"].astype(str).tolist()
    few_days = cov.loc[(cov.days_with_qc > 0) & (cov.days_with_qc < 10), "month"].astype(str).tolist()
    u = s[s.usable]
    rows = [
        ("File format",
         f"IDL .save per month ({len(used)} months, struct OLR_ALL{{JDAY, OLR}}) + daily geolocation CSV "
         f"({int(geo_files.used.sum())} days), in tar.gz archives",
         f"NetCDF4 (cfgrib), one file per year: {len(era)} monthly files, {len(eh)} hour-of-day files",
         f"NetCDF4, {len(ce)} file"),
        ("Variable",
         "OLR, a radiance (no units attribute in the file; W m-2 sr-1 per the instrument team); position: "
         "satellite (lat, lon, alt) and boresight footprint",
         f"ttr '{era.long_name.iloc[0]}' [{era.units.iloc[0]}], GRIB stepType {era.grib_step_type.iloc[0]} "
         f"(monthly) / {eh.grib_step_type.iloc[0] if len(eh) else 'n/a'} (by hour)",
         f"toa_lw_all_mon [{ce.units.iloc[0]}]"),
        ("Sampling",
         f"instantaneous samples, median cadence {used.cadence_median_s.median():.1f} s; geolocation every "
         f"{geo_files.cadence_median_s.median():.0f} s, linearly interpolated onto the samples (on the sphere)",
         "complete grid, every point every month", "complete grid"),
        ("Spatial structure",
         "points along the orbit track (footprint = near intersection of the line of sight with a 6371 km "
         "sphere); field of view not documented in the data",
         f"regular {era.lat_step.abs().iloc[0]:g} deg point grid {era.dims.iloc[0]}, lat {era.lat_first.iloc[0]:g}"
         f"->{era.lat_last.iloc[0]:g}, lon {era.lon_first.iloc[0]:g}->{era.lon_last.iloc[0]:g} (0-360)",
         f"regular {ce.lat_step.iloc[0]:g} deg cell-centre grid, lat {ce.lat_first.iloc[0]:g}->{ce.lat_last.iloc[0]:g}, "
         f"lon {ce.lon_first.iloc[0]:g}->{ce.lon_last.iloc[0]:g} (0-360)"),
        ("CRS", "WGS84 geodetic for positions (EPSG:4979); footprint on a 6371 km sphere",
         "regular lat-lon on the ECMWF sphere", "regular lat-lon"),
        ("Time coverage",
         f"{s.time.min():%Y-%m-%d} -> {s.time.max():%Y-%m-%d} (UTC, Julian date); months without usable "
         f"samples: {', '.join(missing)}; months with < 10 days of QC data: {', '.join(few_days)}",
         f"{pd.Timestamp(era.time_first.min()):%Y-%m} -> {pd.Timestamp(era.time_last.max()):%Y-%m}, stamped day "
         f"{era.time_stamp_days.iloc[0]} {era.time_stamp_hours.iloc[0]}:00 UTC (monthly); hour-of-day stamps "
         f"01..24 UTC, each the accumulation over the preceding hour",
         f"{pd.Timestamp(ce.time_first.iloc[0]):%Y-%m} -> {pd.Timestamp(ce.time_last.iloc[0]):%Y-%m}, stamped day "
         f"{ce.time_stamp_days.iloc[0]}"),
        ("Temporal meaning",
         "instantaneous (30 s integration); local solar time drifts with the orbit",
         "monthly mean of daily means (24 h, day + night); by-hour product: monthly mean at each UTC hour",
         "monthly mean (24 h)"),
        ("Units / sign",
         f"W m-2 sr-1 (radiance), positive upward; quantised in steps of {OLR_STEP} W m-2 sr-1 "
         f"({used.frac_on_quantisation_grid.mean():.1%} of Earth-range values on the step grid)",
         "J m-2 accumulated per day (monthly) or per hour (by hour), negative upward: "
         "OLR = -ttr/86400 or -ttr/3600",
         "W m-2, positive upward"),
        ("Value range",
         f"raw {used.olr_min.min():.0f} .. {used.olr_max.max():.0f}; Earth-viewing usable "
         f"{u.olr.min():.0f} .. {u.olr.max():.0f} (median {u.olr.median():.0f}); QC'd {q.olr.min():.0f} .. "
         f"{q.olr.max():.0f} (median {q.olr.median():.0f}) W m-2 sr-1",
         f"OLR {-era.value_max.max() / 86400:.0f} .. {-era.value_min.min() / 86400:.0f} W m-2",
         f"{ce.value_min.iloc[0]:.0f} .. {ce.value_max.iloc[0]:.0f} W m-2"),
        ("Missing values / flags",
         f"no NaN ({int(used.n_nan.sum())}); sun-viewing values > {SUN_VIEW:g} ({int(used.n_sun_view.sum()):,}), "
         f"negatives ({int(used.n_negative.sum()):,}); geolocation flags looking-away '(0,0,0)' "
         f"({int(geo_files.n_looking_away.sum()):,} rows) and above-limb '' ({int(geo_files.n_above_limb.sum()):,}); "
         f"{int((~s.sat_lat.notna()).sum()):,} samples without geolocation; read_in in_range "
         f"[{read_in.OLR_MIN:g} (raised {read_in.OLR_MIN_FROM}), {read_in.OLR_MAX:g}]",
         f"NaN: {int(era.n_nan.sum())}; GRIB missing value {era.missing_value_attr.iloc[0]}",
         f"NaN: {int(ce.n_nan.sum())}; valid range {ce.valid_range_attr.iloc[0]}"),
        ("Samples / size",
         f"{len(s):,} raw, {int(s.sat_lat.notna().sum()):,} geolocated, {int(s.earth_view.sum()):,} Earth-viewing, "
         f"{int(s.usable.sum()):,} usable, {int((s.usable & s.night).sum()):,} night-side, {int(s.qc.sum()):,} QC'd "
         f"(night & VZA <= {_CFG['qc']['vza_max']:g}) in {int(q.overpass.nunique()):,} overpasses",
         f"{era.dims.iloc[0]} per year", f"{ce.dims.iloc[0]}"),
        ("Existing aggregation / smoothing",
         "none in the samples; spatial averaging over the instrument field of view (size unknown) and along "
         "track during the 30 s integration (~225 km)",
         "model fields (effective resolution several grid lengths); monthly and daily averaging",
         "1 deg monthly, gap-filled ('Energy Balanced and Filled')"),
        ("Physical meaning",
         "night-side TOA OLR radiance according to the instrument team, not angle-corrected (strong viewing-angle "
         "dependence, see EDA); daylit footprints exist from 2023",
         "model TOA all-sky OLR flux (CF toa_outgoing_longwave_flux)", "observed TOA all-sky LW flux"),
    ]
    return pd.DataFrame(rows, columns=["aspect", "CLARA (NorSat-1)", "ERA5", "CERES EBAF (reference only)"])


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = samples.load_samples(_CFG)
    olr_files = clara_olr_files()
    geo_files = clara_geolocation_files()
    grids = gridded_products()
    cov = coverage_by_month(s)
    olr_files.to_csv(OUT_DIR / "clara_olr_files.csv", index=False)
    geo_files.to_csv(OUT_DIR / "clara_geolocation_files.csv", index=False)
    grids.to_csv(OUT_DIR / "gridded_products.csv", index=False)
    cov.to_csv(OUT_DIR / "coverage_by_month.csv", index=False)
    comp = comparison(olr_files, geo_files, grids, cov, s)
    comp.to_csv(OUT_DIR / "dataset_comparison.csv", index=False)
    (OUT_DIR / "dataset_comparison.md").write_text(
        "# Dataset comparison (generated by src/inventory.py)\n\n" + to_markdown(comp) + "\n")
    config.save(_CFG, OUT_DIR / "config_used.json")
    print(to_markdown(comp))
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
