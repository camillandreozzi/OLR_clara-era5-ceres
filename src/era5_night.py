"""ERA5 monthly night-time OLR, built from the ERA5 monthly means by hour of day.

CLARA measures OLR on the night side only, while the ERA5 monthly product is a 24 h mean. The
"monthly averaged reanalysis by hour of day" product (retrieve_era5.py --product monthly_by_hour)
holds, for each month, the mean accumulation of ttr over each UTC hour (stamps 01..24 UTC, each covering
the preceding hour). From it, per 1 deg cell and month:

    OLR_h       = -ttr_h / 3600                                   W m-2, positive upward
    w_h         = fraction of the month's days on which the solar zenith angle at the cell centre,
                  at the middle of hour h, exceeds era5.night_sza_min (the CLARA night definition)
    olr_night   = sum_h w_h OLR_h / sum_h w_h
                  where no hour is ever night (polar day), the night-time value does not exist; so that the
                  prediction grid has no gaps, these cells use the mean of the N_DARKEST hours with the
                  largest monthly-mean solar zenith angle (around local solar midnight) and are flagged
                  night_fallback = 1. CLARA never observes them, so predictions there are extrapolation.
    olr_24h     = mean_h OLR_h                                    equals the monthly product (checked)
    night_frac  = sum_h w_h / 24                                  fraction of the month that is night (0 = polar day)

The 0.25 deg points are first area-averaged onto 1 deg cells per hour with read_in.era5_to_ceres_grid
(trapezoid rule, cos-lat weights), so the night weights are evaluated at 1 deg resolution.

cap_smooth() averages a gridded field over spherical caps (area weighted, NaN aware); it uses ERA5 only
and degrades its effective resolution towards CLARA's wide footprint (see harmonise.py).

Output: data/processed/era5_night_1deg.nc, data/processed/era5_hourly_1deg.nc (the 1 deg hourly monthly
means, float32, for the EDA's hour-matched comparisons) and results/preprocessing/era5_night_validation.csv

Usage:
    python src/era5_night.py
"""

import numpy as np
import pandas as pd
import xarray as xr

import config
import read_in
import solar

OUT = config.PROCESSED_DIR / "era5_night_1deg.nc"
OUT_HOURLY = config.PROCESSED_DIR / "era5_hourly_1deg.nc"
FOLDER = read_in.ERA5_DIR / "monthly_by_hour"
EARTH_RADIUS_KM = 6371.0
N_DARKEST = 3  # hours averaged in polar-day cells (fallback)


def night_weights(month, lat, lon, sza_min):
    """(24, nlat, nlon) fraction of the days of month on which hour h (ending at h+1 UTC) is night."""
    return sun_geometry(month, lat, lon, sza_min)[0]


def sun_geometry(month, lat, lon, sza_min):
    """Night weights (see night_weights) and the monthly-mean cos(SZA) of each hour, both (24, nlat, nlon)."""
    start = month.to_timestamp()
    days = pd.date_range(start, periods=month.days_in_month, freq="D")
    phi = np.radians(np.asarray(lat))[:, None]
    lon = np.asarray(lon)[None, :]
    cos_min = np.cos(np.radians(sza_min))
    w = np.zeros((24, phi.shape[0], lon.shape[1]))
    mean_cos = np.zeros_like(w)
    for day in days:
        mid = day + pd.to_timedelta(np.arange(24) + 0.5, unit="h")  # middle of the hours ending 01..24 UTC
        decl, eot = solar.declination_eot(mid)
        utc_min = (np.arange(24) + 0.5) * 60
        for h in range(24):
            ha = np.radians(((utc_min[h] + eot[h] + 4 * lon) % 1440) / 4 - 180)
            d = np.radians(decl[h])
            cos_sza = np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.cos(ha)
            w[h] += cos_sza < cos_min  # sza > sza_min
            mean_cos[h] += cos_sza
    return w / len(days), mean_cos / len(days)


def fallback_weights(w, mean_cos, k=N_DARKEST):
    """Weights with polar-day cells (no night hour) replaced by 1/k on their k darkest hours.
    Returns (weights, fallback mask)."""
    fb = w.sum(0) == 0
    if not fb.any():
        return w, fb
    darkest = np.argsort(mean_cos, axis=0)[:k]  # smallest cos(SZA) = sun lowest
    wf = np.zeros_like(w)
    np.put_along_axis(wf, darkest, 1.0 / k, axis=0)
    return np.where(fb[None], wf, w), fb


def _month_blocks(path):
    """Yield (month Period, standardised hourly OLR Dataset with 24 times) from one yearly file."""
    with xr.open_dataset(path) as ds:
        stamps = pd.to_datetime(ds.valid_time.values)
        months = (stamps - pd.Timedelta("1h")).to_period("M")  # the 24:00 stamp belongs to its month
        for month in months.unique():
            idx = np.flatnonzero(months == month)
            if len(idx) != 24:
                raise ValueError(f"{path.name} {month}: {len(idx)} hourly fields, expected 24")
            sub = ds.isel(valid_time=idx).drop_vars(["number", "expver"], errors="ignore").load()
            sub = read_in._standardise_grid(sub)
            yield month, xr.Dataset({"olr": -sub["ttr"].astype(np.float64) / 3600.0})


def build(cfg):
    sza_min = cfg["era5"]["night_sza_min"]
    files = sorted(FOLDER.glob("era5_ttr_monthly_by_hour_*.nc"))
    if not files:
        raise FileNotFoundError(f"No files in {FOLDER}; run src/retrieve_era5.py --product monthly_by_hour")
    monthly = read_in.read_era5("monthly")  # 24 h monthly means at 0.25 deg, for the consistency check
    out, checks, hourly_1deg = [], [], []
    for path in files:
        for month, hourly in _month_blocks(path):
            one = read_in.era5_to_ceres_grid(hourly)  # (24, 180, 360) area means on 1 deg cells
            w, mean_cos = sun_geometry(month, one.lat.values, one.lon.values, sza_min)
            olr = one.olr.values
            night_frac = w.sum(0) / 24
            w, fallback = fallback_weights(w, mean_cos)
            night = (w * olr).sum(0) / w.sum(0)
            ds = xr.Dataset({"olr_night": (("lat", "lon"), night),
                             "olr_24h": (("lat", "lon"), olr.mean(0)),
                             "night_frac": (("lat", "lon"), night_frac),
                             "night_fallback": (("lat", "lon"), fallback.astype(np.int8))},
                            coords={"lat": one.lat, "lon": one.lon}).expand_dims(time=[month.to_timestamp()])
            out.append(ds)
            hourly_1deg.append(xr.DataArray(olr.astype(np.float32)[None], dims=("time", "hour_end", "lat", "lon"),
                                            coords={"time": [month.to_timestamp()], "hour_end": np.arange(1, 25),
                                                    "lat": one.lat, "lon": one.lon}))
            # 24 h mean of the hourly product vs the monthly-means product, on the native 0.25 deg points
            ref = monthly.olr.sel(time=month.to_timestamp()).values
            diff = hourly.olr.values.mean(0) - ref
            checks.append({"month": str(month), "mean_diff_24h_vs_monthly": diff.mean(),
                           "max_abs_diff_24h_vs_monthly": np.abs(diff).max(),
                           "global_mean_olr_24h": _area_mean(ds.olr_24h),
                           "global_mean_olr_night": _area_mean(ds.olr_night),
                           "polar_day_cells_fallback": int(fallback.sum())})
            print(f"{month}: night-time mean built ({checks[-1]['polar_day_cells_fallback']} polar-day cells "
                  f"use the {N_DARKEST} darkest hours)")
    ds = xr.concat(out, dim="time").sortby("time")
    ds.olr_night.attrs = {"units": "W m-2", "long_name": f"ERA5 monthly mean night-time OLR (SZA > {sza_min:g} deg); "
                          f"polar-day cells: mean of the {N_DARKEST} darkest hours (night_fallback = 1)"}
    ds.night_fallback.attrs = {"units": "1", "long_name": "1 where the cell has no night that month (polar day)"}
    ds.olr_24h.attrs = {"units": "W m-2", "long_name": "ERA5 monthly mean OLR (24 h)"}
    ds.night_frac.attrs = {"units": "1", "long_name": "fraction of the month that is night at the cell centre"}
    ds.attrs = {"source": "ERA5 monthly averaged reanalysis by hour of day, ttr; built by src/era5_night.py",
                "night_sza_min_deg": sza_min}
    hourly = xr.concat(hourly_1deg, dim="time").sortby("time").rename("olr")
    hourly.attrs = {"units": "W m-2", "long_name": "ERA5 monthly mean OLR over the hour ending at hour_end UTC",
                    "note": "1 deg trapezoid area means of the 0.25 deg points"}
    return ds, pd.DataFrame(checks), hourly


def _area_mean(da):
    w = np.cos(np.radians(da.lat)) * xr.ones_like(da)
    return float(da.weighted(w.where(da.notnull(), 0).fillna(0)).mean())


# ---------------------------------------------------------------------------
# Spherical-cap smoothing
# ---------------------------------------------------------------------------

def cap_kernel(lat, lon, radius_km):
    """rfft along longitude of the cap weights K[i, j, dlon]: area weight of source row j inside the cap
    of radius radius_km around (lat_i, 0). Rows are exact: the cap depends only on (lat_i, lat_j, dlon)."""
    phi = np.radians(np.asarray(lat, float))
    dl = np.radians(np.asarray(lon, float) - lon[0])
    cosd = (np.sin(phi)[:, None, None] * np.sin(phi)[None, :, None]
            + np.cos(phi)[:, None, None] * np.cos(phi)[None, :, None] * np.cos(dl)[None, None, :])
    inside = cosd >= np.cos(radius_km / EARTH_RADIUS_KM)
    k = inside * np.cos(phi)[None, :, None]  # cell area is proportional to cos(lat) on a regular grid
    return np.fft.rfft(k, axis=-1)


def cap_smooth(values, lat, lon, radius_km):
    """Area-weighted, NaN-aware mean of values (..., nlat, nlon) over spherical caps of radius_km around
    each cell centre of a regular global grid. radius_km <= 0 returns the input unchanged."""
    values = np.asarray(values, float)
    if radius_km <= 0:
        return values.copy()
    kh = cap_kernel(lat, lon, radius_km)  # (i, j, f)
    n = values.shape[-1]
    valid = np.isfinite(values)
    lead = values.shape[:-2]
    f = np.fft.rfft(np.where(valid, values, 0.0).reshape(-1, *values.shape[-2:]), axis=-1)  # (b, j, f)
    v = np.fft.rfft(valid.reshape(-1, *values.shape[-2:]).astype(float), axis=-1)
    kt = np.transpose(kh, (2, 1, 0))  # (f, j, i)
    num = np.fft.irfft(np.transpose(np.matmul(np.transpose(f, (2, 0, 1)), kt), (1, 2, 0)), n=n, axis=-1)
    den = np.fft.irfft(np.transpose(np.matmul(np.transpose(v, (2, 0, 1)), kt), (1, 2, 0)), n=n, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-9 * den.max(), num / den, np.nan)
    return out.reshape(*lead, *values.shape[-2:])


def smoothed(ds, var, radii_km):
    """Dataset with var smoothed at each radius, as f'{var}_r{int(R)}' (r0 = unsmoothed).

    Cells that are NaN in var stay NaN; smoothing never fills them (the product has no NaN since polar-day
    cells use the darkest-hours fallback, but the guard keeps any future gap explicit).
    """
    out = xr.Dataset(coords=ds.coords)
    for r in radii_km:
        sm = cap_smooth(ds[var].values, ds.lat.values, ds.lon.values, r)
        out[f"{var}_r{int(r)}"] = (ds[var].dims, np.where(np.isfinite(ds[var].values), sm, np.nan))
        out[f"{var}_r{int(r)}"].attrs = {**ds[var].attrs, "cap_radius_km": r}
    return out


def load():
    """The night-time ERA5 product of build(), from data/processed."""
    if not OUT.exists():
        raise FileNotFoundError(f"{OUT} missing; run src/era5_night.py")
    return xr.open_dataset(OUT).load()


def load_hourly():
    """The 1 deg hourly monthly means (time, hour_end, lat, lon) written by build()."""
    if not OUT_HOURLY.exists():
        raise FileNotFoundError(f"{OUT_HOURLY} missing; run src/era5_night.py")
    return xr.open_dataarray(OUT_HOURLY).load()


def main():
    cfg = config.load()
    ds, checks, hourly = build(cfg)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT)
    hourly.to_netcdf(OUT_HOURLY)
    config.PREPROC_DIR.mkdir(parents=True, exist_ok=True)
    checks.to_csv(config.PREPROC_DIR / "era5_night_validation.csv", index=False, float_format="%.5f")
    worst = checks.max_abs_diff_24h_vs_monthly.max()
    print(f"wrote {OUT}: {ds.sizes['time']} months; max |24 h mean of hourly product - monthly product| "
          f"= {worst:.4f} W m-2")
    if worst > 0.1:
        raise RuntimeError("hour-of-day product inconsistent with the monthly product (> 0.1 W m-2)")


if __name__ == "__main__":
    main()
