"""Read CLARA, CERES and ERA5 OLR data into common, analysis-ready structures.

Data layout (relative to the repo root):

    data/CLARA/CLARA_OLR*.tar.gz           IDL .save files, one per month (CLARA_OLR_<yyyy>_<m>.save):
                                           CLARA_OLR.tar.gz holds 2020-2024, CLARA_OLR_<yyyy>.tar.gz one year
                                           each (2025 onwards; a month in a yearly archive takes precedence).
                                           Each holds a struct OLR_ALL with JDAY (Julian date, UTC) and
                                           OLR at ~30 s cadence. The OLR channel also sees the Sun:
                                           values of ~1e4-9e4 are sun-viewing periods, not Earth OLR.
    data/CLARA/CLARA_lat_lon_<yyyy>.tar.gz CSV files, one per day (CLARA_lon_lat_alt-<yyyy>-<m>-<d>_V50_1_4[_test].csv)
                                           at 1 min cadence with the satellite position and the CLARA
                                           footprint as "(lat, lon, alt_m)" tuples (WGS84 geodetic). The
                                           footprint is "(0, 0, 0)" when CLARA looks away from the Earth
                                           (z angle > 90 deg) and empty when its line of sight passes
                                           above the limb. The footprint is the near intersection of the
                                           line of sight with a 6371 km sphere, the point CLARA sees.
                                           Releases before 2026-10-21 stored the FAR intersection instead;
                                           read_clara_geolocation() recomputes the near one either way.
    data/CERES/CERES_EBAF*.nc              CERES EBAF Ed4.2.1 monthly TOA all-sky LW flux (toa_lw_all_mon), 1 deg.
    data/ERA5/*.nc                         ERA5 top net thermal radiation (ttr), downloaded by retrieve_era5.py.

All gridded products are returned with OLR as a positive upward flux in W m-2, dims (time, lat, lon),
lat ascending and lon in [-180, 180). Monthly products use the first of the month as their time stamp.

Usage:
    python src/read_in.py             # read everything and print a summary
    python src/read_in.py --save      # also write data/processed/clara_olr_geolocated.parquet
"""

import argparse
import os
import re
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.io import readsav

ROOT = Path(__file__).resolve().parent.parent  # repo root; this file is in src/
DATA_DIR = ROOT / "data"
CLARA_DIR = DATA_DIR / "CLARA"
CERES_DIR = DATA_DIR / "CERES"
ERA5_DIR = DATA_DIR / "ERA5"
PROCESSED_DIR = DATA_DIR / "processed"
EXTRACT_DIR = CLARA_DIR / "extracted"

JULIAN_UNIX_EPOCH = 2440587.5  # Julian date of 1970-01-01T00:00Z

# Plausible range for Earth-viewing OLR in W m-2; samples outside it get in_range=False.
OLR_MIN = 50.0
OLR_MAX = 400.0

# Maximum spacing between two geolocation rows that may be interpolated between (rows are 1 min apart).
GEO_MAX_GAP = pd.Timedelta("65s")

# The file footprints lie exactly on this sphere (|ECEF| = 6371.000 km for every row).
FOOTPRINT_SPHERE_R = 6_371_000.0  # m
_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)  # WGS84 (lon, lat, h) -> ECEF
_FROM_ECEF = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)

OLR_FILE_RE = re.compile(r"^CLARA_OLR_(\d{4})_(\d{1,2})\.save$")
GEO_FILE_RE = re.compile(r"^CLARA_lon_lat_alt-(\d{4})-(\d{1,2})-(\d{1,2})_V50_1_4(?:_test)?\.csv$")
TUPLE_RE = r"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)"


# ---------------------------------------------------------------------------
# CLARA
# ---------------------------------------------------------------------------

def _extract(archive):
    """Extract a tar.gz archive into data/CLARA/extracted/<archive stem>/ and return that folder.

    The extraction is reused until the archive is replaced by a newer one.
    """
    out = EXTRACT_DIR / archive.name.removesuffix(".tar.gz")
    if out.exists() and archive.stat().st_mtime > out.stat().st_mtime:
        print(f"{archive.name} is newer than its extraction; extracting again")
        shutil.rmtree(out)
    if not out.exists():
        tmp = out.with_name(out.name + ".part")
        tmp.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        tmp.rename(out)
        os.utime(out)  # stamp the extraction time, compared with the archive's on the next call
    return out


def _in_years(year, years):
    return years is None or year in years


def read_clara_olr(years=None):
    """Return the CLARA OLR time series as a DataFrame with columns [time, olr].

    The raw signal is kept as is (including sun-viewing samples); use the
    geolocation merge in read_clara() to select Earth-viewing samples.
    """
    paths = {}
    # "CLARA_OLR.tar.gz" sorts before "CLARA_OLR_<yyyy>.tar.gz", so a month in a yearly archive wins.
    for archive in sorted(CLARA_DIR.glob("CLARA_OLR*.tar.gz")):
        for path in _extract(archive).glob("*.save"):
            m = OLR_FILE_RE.search(path.name)
            if m and _in_years(int(m.group(1)), years):
                paths[int(m.group(1)), int(m.group(2))] = path
    frames = []
    for key in sorted(paths):
        rec = readsav(str(paths[key]))["olr_all"][0]
        # IDL save files are big-endian; convert to native floats for pandas.
        jday = np.asarray(rec["JDAY"], dtype=np.float64)
        olr = np.asarray(rec["OLR"], dtype=np.float64)
        frames.append(pd.DataFrame({
            "time": pd.to_datetime(jday - JULIAN_UNIX_EPOCH, unit="D").round("ms"),
            "olr": olr,
        }))
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def _ecef(lat, lon, alt):
    return np.stack(_TO_ECEF.transform(lon, lat, alt), axis=-1)


def near_footprint(sat_lat, sat_lon, sat_alt, fp_lat, fp_lon, fp_alt):
    """Return (lat, lon, vza) of the near intersection of the satellite -> file-footprint line with the
    6371 km sphere, i.e. the point CLARA looks at. For current files this is the file footprint itself;
    older releases stored the far intersection of the same line, which this maps back to the near one.

    vza is the viewing zenith angle at the footprint (deg; 0 = satellite overhead, 90 = grazing the limb),
    measured from the WGS84 surface normal.
    """
    s = _ecef(sat_lat, sat_lon, sat_alt)
    d = _ecef(fp_lat, fp_lon, fp_alt) - s
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    b = np.sum(s * d, axis=-1)
    disc = b * b - (np.sum(s * s, axis=-1) - FOOTPRINT_SPHERE_R ** 2)
    f = s + (-b - np.sqrt(np.maximum(disc, 0)))[:, None] * d
    lon, lat, _ = _FROM_ECEF.transform(f[:, 0], f[:, 1], f[:, 2])
    up = np.stack(_latlon_to_xyz(lat, lon), axis=-1)
    vza = np.degrees(np.arccos(np.clip(np.sum(up * -d, axis=-1), -1, 1)))
    return lat, lon, vza


def _read_geo_csv(path, date, raw_footprint=False):
    df = pd.read_csv(path)
    df.columns = [c.lstrip("# ").strip() for c in df.columns]
    sat = df["satellite_lat_long"].str.extract(TUPLE_RE).astype(float).to_numpy()
    fov = df["clara_lat_long"].str.extract(TUPLE_RE).astype(float).to_numpy()
    # "(0, 0, 0)" = looking away from the Earth, empty (NaN) = line of sight above the limb.
    earth_view = np.isfinite(fov).all(axis=1) & ~(fov == 0).all(axis=1)
    lat, lon, vza = np.full((3, len(df)), np.nan)
    if raw_footprint:
        lat[earth_view], lon[earth_view] = fov[earth_view, 0], fov[earth_view, 1]
    elif earth_view.any():
        lat[earth_view], lon[earth_view], vza[earth_view] = near_footprint(*sat[earth_view].T, *fov[earth_view].T)
    return pd.DataFrame({
        "time": date + pd.to_timedelta(df["datetime"]),
        "z_angle": df["z_vector_satellite_position_angles"].astype(float),
        "sat_lat": sat[:, 0], "sat_lon": sat[:, 1], "sat_alt": sat[:, 2],
        "clara_lat": lat, "clara_lon": lon, "vza": vza,
        "earth_view": earth_view,
    })


def _read_geo_csv_empty():
    """An empty geolocation table with the columns of _read_geo_csv()."""
    df = pd.DataFrame({c: pd.Series(dtype=float) for c in
                       ("z_angle", "sat_lat", "sat_lon", "sat_alt", "clara_lat", "clara_lon", "vza")})
    df.insert(0, "time", pd.Series(dtype="datetime64[ns]"))
    df["earth_view"] = pd.Series(dtype=bool)
    return df


def read_clara_geolocation(years=None, raw_footprint=False):
    """Return the 1-minute CLARA geolocation as a DataFrame.

    Columns: time, z_angle (deg, line of sight from nadir), sat_lat, sat_lon, sat_alt (m),
    clara_lat, clara_lon (footprint, NaN when not Earth viewing), vza (deg), earth_view (bool).

    The footprint is the near-side intersection of the line of sight (see near_footprint);
    raw_footprint=True returns the point as stored in the files instead (no vza).
    When a day exists both as a "_test" and a final file, the final one is used.
    """
    paths = {}
    for archive in sorted(CLARA_DIR.glob("CLARA_lat_lon_*.tar.gz")):
        year = int(re.search(r"(\d{4})", archive.name).group(1))
        if not _in_years(year, years):
            continue
        # "_test" files sort after the final ones, so setdefault keeps the final version.
        for path in sorted(_extract(archive).glob("*.csv"), key=lambda p: "_test" in p.name):
            m = GEO_FILE_RE.search(path.name)
            if m:
                paths.setdefault(pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3))), path)
    frames = [_read_geo_csv(paths[date], date, raw_footprint) for date in sorted(paths)]
    if not frames:  # e.g. OLR years without a CLARA_lat_lon_<yyyy> archive yet
        return _read_geo_csv_empty()
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("time", kind="stable").drop_duplicates("time").reset_index(drop=True)


def _latlon_to_xyz(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)


def _xyz_to_latlon(x, y, z):
    return np.degrees(np.arctan2(z, np.hypot(x, y))), np.degrees(np.arctan2(y, x))


def geolocate(olr, geo):
    """Interpolate the 1-minute geolocation onto the OLR sample times.

    Positions are interpolated on the unit sphere so the dateline and poles are
    handled correctly. A sample is only geolocated when both neighbouring
    geolocation rows are at most GEO_MAX_GAP apart; the footprint is only
    filled when both neighbours are Earth viewing.
    """
    if len(geo) < 2:  # nothing to interpolate between, so no sample is geolocated
        out = olr.copy()
        for col in ("z_angle", "sat_lat", "sat_lon", "sat_alt", "clara_lat", "clara_lon", "vza"):
            out[col] = np.nan
        out["earth_view"] = False
        out["in_range"] = out["olr"].between(OLR_MIN, OLR_MAX)
        return out
    tg = geo["time"].to_numpy("datetime64[ns]").astype(np.int64)
    to = olr["time"].to_numpy("datetime64[ns]").astype(np.int64)
    right = np.searchsorted(tg, to).clip(1, len(tg) - 1)
    left = right - 1
    dt = tg[right] - tg[left]
    ok = (to >= tg[0]) & (to <= tg[-1]) & (dt <= GEO_MAX_GAP.value)
    w = np.where(dt > 0, (to - tg[left]) / np.maximum(dt, 1), 0.0)

    def lerp(col):
        a, b = geo[col].to_numpy()[left], geo[col].to_numpy()[right]
        return np.where(ok, a + w * (b - a), np.nan)

    def lerp_latlon(prefix):
        xyz_l = _latlon_to_xyz(geo[f"{prefix}_lat"].to_numpy()[left], geo[f"{prefix}_lon"].to_numpy()[left])
        xyz_r = _latlon_to_xyz(geo[f"{prefix}_lat"].to_numpy()[right], geo[f"{prefix}_lon"].to_numpy()[right])
        lat, lon = _xyz_to_latlon(*(a + w * (b - a) for a, b in zip(xyz_l, xyz_r)))
        return np.where(ok, lat, np.nan), np.where(ok, lon, np.nan)

    ev = geo["earth_view"].to_numpy()
    out = olr.copy()
    out["z_angle"] = lerp("z_angle")
    out["sat_lat"], out["sat_lon"] = lerp_latlon("sat")
    out["sat_alt"] = lerp("sat_alt")
    out["clara_lat"], out["clara_lon"] = lerp_latlon("clara")
    out["vza"] = lerp("vza")
    out["earth_view"] = ok & ev[left] & ev[right]
    out["in_range"] = out["olr"].between(OLR_MIN, OLR_MAX)
    return out


def read_clara(years=None, raw_footprint=False):
    """Return geolocated CLARA OLR samples (see geolocate() for the columns).

    Select usable Earth-viewing OLR with: df[df.earth_view & df.in_range]
    raw_footprint=True keeps the footprint as stored in the files, for comparison only.
    """
    return geolocate(read_clara_olr(years), read_clara_geolocation(years, raw_footprint))


# ---------------------------------------------------------------------------
# Gridded products
# ---------------------------------------------------------------------------

def _standardise_grid(ds):
    """Rename coords to (time, lat, lon), lat ascending, lon in [-180, 180)."""
    ds = ds.rename({k: v for k, v in {"valid_time": "time", "latitude": "lat", "longitude": "lon"}.items()
                    if k in ds.dims or k in ds.coords})
    ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
    return ds.sortby(["lat", "lon"])


def _month_start(ds):
    return ds.assign_coords(time=ds.indexes["time"].to_period("M").to_timestamp())


def read_ceres(path=None):
    """Return CERES EBAF monthly all-sky TOA LW flux as a Dataset with variable 'olr' (W m-2)."""
    if path is None:
        files = sorted(CERES_DIR.glob("CERES_EBAF*.nc"))
        if len(files) != 1:
            raise FileNotFoundError(f"expected one CERES_EBAF*.nc file in {CERES_DIR}, found {[f.name for f in files]}")
        path = files[0]
    ds = xr.open_dataset(path).rename({"toa_lw_all_mon": "olr"})
    return _month_start(_standardise_grid(ds))


def read_era5(product="monthly", folder=None):
    """Return ERA5 OLR as a Dataset with variable 'olr' (W m-2), and 'olr_clr' if clear-sky was downloaded.

    ERA5 'ttr' (top net thermal radiation) is an accumulation in J m-2, negative upward.
    Monthly means of daily means are per day (divide by 86400 s); hourly data are
    accumulated over the hour ending at the time stamp (divide by 3600 s).
    """
    seconds = {"monthly": 86400.0, "hourly": 3600.0}[product]
    folder = Path(folder) if folder else ERA5_DIR / product
    files = sorted(folder.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5 files in {folder}; run src/retrieve_era5.py --product {product} first.")
    ds = xr.concat([xr.open_dataset(f) for f in files], dim="valid_time")
    ds = _standardise_grid(ds.drop_vars(["number", "expver"], errors="ignore"))
    out = xr.Dataset({"olr": -ds["ttr"] / seconds})
    if "ttrc" in ds:
        out["olr_clr"] = -ds["ttrc"] / seconds
    for v in out.data_vars:
        out[v].attrs = {"units": "W m-2", "long_name": "Outgoing longwave radiation at TOA"}
    out = out.sortby("time")
    return _month_start(out) if product == "monthly" else out


# ---------------------------------------------------------------------------
# Regridding
# ---------------------------------------------------------------------------

def era5_to_ceres_grid(ds):
    """Area-average ERA5 from its 0.25 deg point grid onto the CERES 1 deg cells.

    ERA5 points sit on cell edges (lat -90..90, lon -180..179.75), so each 1 deg
    box spans 5 points per axis; the edge points get half weight (trapezoid rule)
    and latitudes are weighted by cos(lat). Longitudes wrap at the dateline.
    """
    lat, lon = ds.lat.values, ds.lon.values
    if len(lat) != 721 or len(lon) != 1440 or lat[0] != -90 or lon[0] != -180:
        raise ValueError("expected the native ERA5 0.25 deg grid (721 x 1440)")
    trap = np.array([0.5, 1, 1, 1, 0.5])
    w_lat = np.zeros((180, 721))
    w_lon = np.zeros((360, 1440))
    for i in range(180):
        w = trap * np.cos(np.radians(lat[4 * i:4 * i + 5]))
        w_lat[i, 4 * i:4 * i + 5] = w / w.sum()
    for k in range(360):
        w_lon[k, np.arange(4 * k, 4 * k + 5) % 1440] += trap / trap.sum()

    out = xr.Dataset(coords={"time": ds.time, "lat": np.arange(-89.5, 90), "lon": np.arange(-179.5, 180)})
    for v in ds.data_vars:
        out[v] = (("time", "lat", "lon"), (w_lat @ ds[v].values.astype(np.float64)) @ w_lon.T)
        out[v].attrs = ds[v].attrs
    return out


def grid_edges(res):
    """Cell edges (lat, lon) of the res x res deg grid starting at (-90, -180).

    When res does not divide 180 / 360 (e.g. 7 deg), the last row / column is narrower and ends at 90 / 180.
    """
    res = int(res)
    return (np.minimum(np.arange(-90, 90 + res, res), 90).astype(float),
            np.minimum(np.arange(-180, 180 + res, res), 180).astype(float))


def grid_centres(res):
    """Cell centres (lat, lon) of grid_edges(res)."""
    return tuple((e[:-1] + e[1:]) / 2 for e in grid_edges(res))


def coarsen(ds, res):
    """Area-weighted average of a 1 deg (time, lat, lon) Dataset onto the res x res deg cells of grid_edges(res)."""
    if res == 1:
        return ds
    w = np.cos(np.radians(ds.lat)) * xr.ones_like(ds.lon)
    lat, lon = grid_centres(res)
    out = xr.Dataset()
    for v in ds.data_vars:
        wv = w.where(ds[v].notnull())
        num = (ds[v] * wv).coarsen(lat=res, lon=res, boundary="pad").sum()
        den = wv.coarsen(lat=res, lon=res, boundary="pad").sum()
        out[v] = (num / den).assign_coords(lat=lat, lon=lon)
        out[v].attrs = ds[v].attrs
    return out


def grid_clara(df, res, freq="M"):
    """Bin CLARA footprint samples onto a res deg grid per period ("M" monthly, "Y" yearly).

    Returns a Dataset with 'olr' (mean of all samples in the cell, W m-2) and 'count'.
    """
    lat_c, lon_c = grid_centres(res)
    nlat, nlon = len(lat_c), len(lon_c)
    i = np.clip(np.floor((df.clara_lat.to_numpy() + 90) / res).astype(int), 0, nlat - 1)
    j = np.clip(np.floor((df.clara_lon.to_numpy() + 180) / res).astype(int), 0, nlon - 1)
    periods = df.time.dt.to_period(freq)
    labels = np.sort(periods.unique())
    p = np.searchsorted(labels, periods)
    key = (p * nlat + i) * nlon + j
    size = len(labels) * nlat * nlon
    count = np.bincount(key, minlength=size).reshape(len(labels), nlat, nlon)
    total = np.bincount(key, weights=df.olr.to_numpy(), minlength=size).reshape(count.shape)
    with np.errstate(invalid="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    coords = {"time": [l.to_timestamp() for l in labels],
              "lat": lat_c, "lon": lon_c}
    return xr.Dataset({"olr": (("time", "lat", "lon"), mean), "count": (("time", "lat", "lon"), count)},
                      coords=coords)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summarise_clara(df):
    good = df[df.earth_view & df.in_range]
    print(f"CLARA: {len(df):,} samples {df.time.min():%Y-%m-%d} -> {df.time.max():%Y-%m-%d}; "
          f"{df.earth_view.sum():,} Earth-viewing, {len(good):,} Earth-viewing within "
          f"[{OLR_MIN:g}, {OLR_MAX:g}] W m-2 (mean {good.olr.mean():.1f})")
    per_month = good.groupby(good.time.dt.to_period("M")).olr.agg(["size", "mean"])
    all_months = pd.period_range(df.time.min(), df.time.max(), freq="M")
    missing = [str(p) for p in all_months if p not in per_month.index]
    print(f"  months with usable OLR: {len(per_month)}; without: {', '.join(missing) or 'none'}")
    located = df.sat_lat.notna().groupby(df.time.dt.to_period("M")).any()
    unlocated = [str(p) for p, ok in located.items() if not ok]
    print(f"  months with OLR but no geolocation: {', '.join(unlocated) or 'none'}")


def _summarise_grid(name, ds):
    t = ds.indexes["time"]
    print(f"{name}: {dict(ds.sizes)} {t.min():%Y-%m} -> {t.max():%Y-%m}; "
          f"global mean OLR (area weighted) {ds.olr.weighted(np.cos(np.radians(ds.lat))).mean().item():.1f} W m-2")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="*", help="CLARA years to read (default: all)")
    parser.add_argument("--save", action="store_true", help="write geolocated CLARA OLR to data/processed/")
    args = parser.parse_args()

    clara = read_clara(set(args.years) if args.years else None)
    _summarise_clara(clara)
    _summarise_grid("CERES", read_ceres())
    for product in ("monthly", "hourly"):
        try:
            _summarise_grid(f"ERA5 {product}", read_era5(product))
        except FileNotFoundError as e:
            print(f"ERA5 {product}: {e}")

    if args.save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = PROCESSED_DIR / "clara_olr_geolocated.parquet"
        clara.to_parquet(out, index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
