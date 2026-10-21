"""Build one flat table of monthly mean OLR per grid cell from CLARA, CERES EBAF and ERA5.

One row per (year, month, cell) where CLARA has at least one usable sample:
    year, month, lat, lon   cell centre (deg) on the res x res deg grid
    clara                   mean of the CLARA Earth-viewing samples in the cell that month (W m-2)
    clara_count             number of CLARA samples behind that mean
    ceres                   CERES EBAF monthly mean all-sky OLR (W m-2)
    era5                    ERA5 monthly mean OLR, -ttr/86400, area-averaged to the cell (W m-2)

CERES and ERA5 are means over every hour of the month; CLARA is the mean of the few samples it took,
so differences include CLARA's sampling (local time, viewing angle), not only calibration.

Output: data/processed/matched_monthly_<res>deg.parquet and .csv

Usage:
    python src/build_matched.py            # 1 deg (CERES native)
    python src/build_matched.py --res 5    # any whole-degree spacing
"""

import argparse

import numpy as np

import read_in


def build(res):
    clara = read_in.read_clara()
    clara = clara[clara.earth_view & clara.in_range]
    ceres = read_in.coarsen(read_in.read_ceres(), res)
    era5 = read_in.coarsen(read_in.era5_to_ceres_grid(read_in.read_era5("monthly")), res)
    grid = read_in.grid_clara(clara, res, "M")

    df = grid.rename(olr="clara", count="clara_count").to_dataframe().reset_index()
    df = df[df.clara_count > 0]
    # Grid cell centres agree to float precision; round so the joins match exactly.
    for name, ds in (("ceres", ceres), ("era5", era5)):
        other = ds.olr.rename(name).to_dataframe().reset_index()
        for d in (df, other):
            d["lat"], d["lon"] = d.lat.round(3), d.lon.round(3)
        df = df.merge(other, on=["time", "lat", "lon"], how="left")
    df.insert(0, "year", df.time.dt.year)
    df.insert(1, "month", df.time.dt.month)
    df["clara_count"] = df.clara_count.astype(np.int64)
    return df.drop(columns="time")[["year", "month", "lat", "lon", "clara", "clara_count", "ceres", "era5"]] \
        .sort_values(["year", "month", "lat", "lon"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--res", type=int, default=1, help="grid spacing in whole degrees (default: 1)")
    args = parser.parse_args()

    df = build(args.res)
    read_in.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = read_in.PROCESSED_DIR / f"matched_monthly_{args.res}deg"
    df.to_parquet(out.with_suffix(".parquet"), index=False)
    df.to_csv(out.with_suffix(".csv"), index=False, float_format="%.2f")
    print(f"wrote {len(df):,} rows to {out}.parquet/.csv")
    print(df.describe().T[["count", "mean", "min", "max"]].round(1))


if __name__ == "__main__":
    main()
