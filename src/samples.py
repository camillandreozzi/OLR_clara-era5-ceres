"""CLARA footprint samples with solar geometry, QC flags and overpass ids.

load_samples() returns every geolocated CLARA sample of read_in.read_clara() with these added columns:
    sza        solar zenith angle at the footprint (deg), solar.solar_zenith
    lst        local solar time at the footprint (h)
    usable     earth_view & in_range, read_in's definition of an Earth-viewing OLR sample
    night      sza > qc.night_sza_min (CLARA measures OLR on the night side only)
    vza_ok     vza <= qc.vza_max
    qc         usable & night & vza_ok: the observations the model uses
    overpass   id of the overpass a qc sample belongs to (-1 otherwise); a new overpass starts after a
               gap of more than qc.overpass_gap_min minutes between consecutive qc samples

The geolocated samples are cached in data/processed/clara_olr_geolocated.parquet (written by
read_in.py --save); the cache is rebuilt from the raw archives when any archive or read_in.py is newer.
"""

import numpy as np
import pandas as pd

import read_in
import solar

CACHE = read_in.PROCESSED_DIR / "clara_olr_geolocated.parquet"


def _cache_is_fresh():
    if not CACHE.exists():
        return False
    sources = list(read_in.CLARA_DIR.glob("*.tar.gz")) + [read_in.ROOT / "src" / "read_in.py"]
    return CACHE.stat().st_mtime > max(p.stat().st_mtime for p in sources)


def geolocated(rebuild=False):
    """All geolocated CLARA samples (read_in.read_clara), from the cache when it is up to date."""
    if not rebuild and _cache_is_fresh():
        return pd.read_parquet(CACHE)
    df = read_in.read_clara()
    read_in.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def overpass_ids(time, gap_min):
    """Overpass id per (sorted) sample time: a new overpass starts after a gap longer than gap_min."""
    t = pd.to_datetime(time).to_numpy("datetime64[ns]").astype(np.int64)
    new = np.r_[True, np.diff(t) > gap_min * 60e9]
    return np.cumsum(new) - 1


def add_flags(df, qc):
    """Add sza, lst and the QC flags of the module docstring to geolocated samples (a copy)."""
    df = df.copy()
    has_fp = df.clara_lat.notna().to_numpy()
    df["sza"] = np.nan
    df["lst"] = np.nan
    df.loc[has_fp, "sza"] = solar.solar_zenith(df.time[has_fp], df.clara_lat[has_fp], df.clara_lon[has_fp])
    df.loc[has_fp, "lst"] = solar.local_solar_time(df.time[has_fp], df.clara_lon[has_fp])
    df["usable"] = df.earth_view & df.in_range
    df["night"] = df.sza > qc["night_sza_min"]
    df["vza_ok"] = df.vza <= qc["vza_max"]
    df["qc"] = df.usable & df.night & df.vza_ok
    df["overpass"] = -1
    sel = df.qc.to_numpy()
    df.loc[sel, "overpass"] = overpass_ids(df.time[sel], qc["overpass_gap_min"])
    return df


def load_samples(cfg, rebuild=False):
    """Geolocated CLARA samples with geometry and QC flags (see module docstring)."""
    return add_flags(geolocated(rebuild), cfg["qc"])
