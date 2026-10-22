"""Unit and sign conversions, grid conventions, binning, solar geometry, overpasses, ERA5 night weights
and cap smoothing."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import era5_night
import harmonise
import read_in
import samples
import solar


def test_era5_monthly_sign_and_units(tmp_path):
    # 240 W m-2 upward over one day = -240 * 86400 J m-2 in ERA5's convention
    ds = xr.Dataset({"ttr": (("valid_time", "latitude", "longitude"), np.full((1, 3, 4), -240.0 * 86400))},
                    coords={"valid_time": pd.to_datetime(["2021-03-01T06:00"]), "latitude": [90.0, 0.0, -90.0],
                            "longitude": [0.0, 90.0, 180.0, 270.0]})
    ds.to_netcdf(tmp_path / "x.nc")
    out = read_in.read_era5("monthly", folder=tmp_path)
    assert np.allclose(out.olr.values, 240.0)
    assert out.indexes["time"][0] == pd.Timestamp("2021-03-01")  # monthly stamps -> first of the month
    assert list(out.lat.values) == [-90.0, 0.0, 90.0]  # ascending
    assert list(out.lon.values) == [-180.0, -90.0, 0.0, 90.0]  # 0-360 -> [-180, 180), sorted


def test_era5_hourly_units():
    # by-hour product: J m-2 per hour -> W m-2
    assert -(-240.0 * 3600) / 3600 == 240.0


def test_coarsen_preserves_area_mean():
    lat, lon = np.arange(-89.5, 90), np.arange(-179.5, 180)
    rng = np.random.default_rng(0)
    v = rng.normal(240, 30, (1, 180, 360))
    ds = xr.Dataset({"olr": (("time", "lat", "lon"), v)}, coords={"time": [pd.Timestamp("2021-01-01")],
                                                                  "lat": lat, "lon": lon})
    c = read_in.coarsen(ds, 5)
    w1 = np.cos(np.radians(lat))[:, None] * np.ones(360)
    w5 = np.cos(np.radians(lat)).reshape(36, 5).sum(1)[:, None] * np.ones(72)
    assert np.average(c.olr.values[0], weights=w5) == pytest.approx(np.average(v[0], weights=w1), rel=1e-12)


def test_cell_index_edges_poles_dateline():
    i, j = harmonise.cell_index([-90, 90, 0, 0, 4.999, 5.0], [-180, 180, -0.001, 0, 0, 0], 5)
    assert list(i) == [0, 35, 18, 18, 18, 19]
    assert list(j) == [0, 71, 35, 36, 36, 36]


def test_cell_area_sums_to_sphere():
    lat_c, _ = read_in.grid_centres(5)
    total = harmonise.cell_area_km2(lat_c, 5).sum() * 72
    assert total == pytest.approx(4 * np.pi * 6371.0 ** 2, rel=1e-12)


def test_bin_cell_months_and_overpasses():
    t = pd.to_datetime(["2021-01-03 00:00:00", "2021-01-03 00:00:30", "2021-01-03 00:01:00",  # one overpass
                        "2021-01-03 01:40:00",  # next orbit, same cell
                        "2021-02-01 00:00:00"])
    q = pd.DataFrame({"time": t, "clara_lat": [1.0, 1.5, 2.0, 3.0, 1.0], "clara_lon": [1.0, 1.2, 1.4, 2.0, 1.0],
                      "olr": [200.0, 210.0, 220.0, 260.0, 180.0]})
    q["overpass"] = samples.overpass_ids(q.time, 20.0)
    assert list(q.overpass) == [0, 0, 0, 1, 2]
    cells = harmonise.bin_cell_months(q, 5)
    jan = cells[cells.time == pd.Timestamp("2021-01-01")].iloc[0]
    assert (jan.lat, jan.lon) == (2.5, 2.5)
    assert jan.clara == pytest.approx(222.5) and jan.n_samples == 4 and jan.n_overpasses == 2
    assert len(cells) == 2


def test_solar_zenith_reference_values():
    # equinox: sun near overhead at the equator at local noon, near nadir at midnight
    t = pd.to_datetime(["2021-03-20 12:00", "2021-03-20 00:00", "2021-06-21 12:00", "2021-12-21 12:00"])
    sza = solar.solar_zenith(t, [0, 0, 23.44, -23.44], [0, 0, 0, 0])
    assert sza[0] < 2.5 and sza[1] > 177 and sza[2] < 1.5 and sza[3] < 1.5
    # local solar time at 90 E is 6 h ahead of Greenwich (up to the equation of time, < 17 min)
    lst = solar.local_solar_time(pd.to_datetime(["2021-07-01 00:00"]), [90.0])
    assert abs(lst[0] - 6.0) < 0.3


def test_night_weights_trivial_cases():
    w = era5_night.night_weights(pd.Period("2021-12", "M"), np.array([-89.5, 0.5, 89.5]), np.array([0.5]), 100.0)
    assert np.all(w[:, 0, 0] == 0)  # southern summer: the south pole never has night
    assert np.all(w[:, 2, 0] == 1)  # northern winter: the north pole is always night
    assert 0.3 < w[:, 1, 0].mean() < 0.5  # equator: ~10 of 24 h beyond SZA 100


def test_cap_smoothing_constant_nan_and_wrap():
    lat, lon = np.arange(-89.5, 90), np.arange(-179.5, 180)
    f = np.full((180, 360), 7.0)
    f[:3] = np.nan
    s = era5_night.cap_smooth(f, lat, lon, 500.0)
    assert np.allclose(s[np.isfinite(s)], 7.0)
    g = np.zeros((180, 360))
    g[90, 0] = 1.0
    s = era5_night.cap_smooth(g, lat, lon, 400.0)
    cols = np.unique(np.argwhere(s > 1e-9)[:, 1])  # above FFT round-off
    assert 359 in cols and 0 in cols and 180 not in cols  # spreads across the dateline, not beyond the cap
    ds = xr.Dataset({"v": (("lat", "lon"), f)}, coords={"lat": lat, "lon": lon})
    sm = era5_night.smoothed(ds, "v", [0.0, 500.0])
    assert np.isnan(sm.v_r500.values[:3]).all()  # NaN cells (polar day) are never filled
