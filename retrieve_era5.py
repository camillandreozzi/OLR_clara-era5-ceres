"""Download ERA5 top net thermal radiation (ttr = -OLR) from the Copernicus Climate Data Store.

Requires a CDS account with the ERA5 licence accepted, and credentials in ~/.cdsapirc:

    url: https://cds.climate.copernicus.eu/api
    key: <your-personal-access-token>

Products:
    monthly  reanalysis-era5-single-levels-monthly-means (monthly means of daily means), one file per year
             -> data/ERA5/monthly/era5_ttr_monthly_<yyyy>.nc. Matches the CERES EBAF monthly product.
    hourly   reanalysis-era5-single-levels, one file per month
             -> data/ERA5/hourly/era5_ttr_hourly_<yyyy>_<mm>.nc. Needed to collocate with CLARA
             samples in time. Defaults to a 1 deg grid to keep the volume manageable (~200 MB/month).

Existing files are skipped, so the script can be re-run to resume an interrupted download.
Convert to OLR in W m-2 with read_in.read_era5(), which applies -ttr/86400 (monthly) or -ttr/3600 (hourly).

Examples:
    python retrieve_era5.py                                  # monthly, 2020-2025, native 0.25 deg
    python retrieve_era5.py --product hourly --years 2021    # hourly 2021 on a 1 deg grid
    python retrieve_era5.py --clear-sky --dry-run            # print the requests only
"""

import argparse
import sys
from pathlib import Path

import cdsapi

ERA5_DIR = Path(__file__).resolve().parent / "data" / "ERA5"
CDSAPIRC = Path.home() / ".cdsapirc"

DATASETS = {
    "monthly": "reanalysis-era5-single-levels-monthly-means",
    "hourly": "reanalysis-era5-single-levels",
}
DEFAULT_GRID = {"monthly": None, "hourly": 1.0}  # None = native 0.25 deg
DEFAULT_YEARS = list(range(2020, 2026))  # CLARA 2020-2025, CERES 2021-2025

MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
HOURS = [f"{h:02d}:00" for h in range(24)]


def build_requests(product, years, months, variables, grid):
    """Yield (target path, request dict) pairs, one per output file."""
    base = {
        "product_type": ["monthly_averaged_reanalysis" if product == "monthly" else "reanalysis"],
        "variable": variables,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if grid:
        base["grid"] = [grid, grid]

    out_dir = ERA5_DIR / product
    for year in years:
        if product == "monthly":
            yield (out_dir / f"era5_ttr_monthly_{year}.nc",
                   {**base, "year": [str(year)], "month": months, "time": ["00:00"]})
        else:
            for month in months:
                yield (out_dir / f"era5_ttr_hourly_{year}_{month}.nc",
                       {**base, "year": [str(year)], "month": [month], "day": DAYS, "time": HOURS})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--product", choices=DATASETS, default="monthly")
    parser.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    parser.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--grid", type=float, help="output grid spacing in degrees "
                        "(default: native 0.25 for monthly, 1.0 for hourly)")
    parser.add_argument("--clear-sky", action="store_true", help="also download clear-sky ttrc")
    parser.add_argument("--dry-run", action="store_true", help="print the requests without downloading")
    args = parser.parse_args()

    variables = ["top_net_thermal_radiation"]
    if args.clear_sky:
        variables.append("top_net_thermal_radiation_clear_sky")
    grid = args.grid if args.grid is not None else DEFAULT_GRID[args.product]
    months = [f"{m:02d}" for m in args.months]
    dataset = DATASETS[args.product]
    requests = list(build_requests(args.product, args.years, months, variables, grid))

    if args.dry_run:
        for target, request in requests:
            print(f"{dataset} -> {target}\n  {request}")
        return

    if not CDSAPIRC.exists():
        sys.exit(f"Missing {CDSAPIRC}. Create it with your CDS url and key "
                 "(see https://cds.climate.copernicus.eu/how-to-api).")

    client = cdsapi.Client()
    for target, request in requests:
        if target.exists():
            print(f"skip {target.name} (exists)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(".nc.part")
        print(f"requesting {target.name} ...")
        client.retrieve(dataset, request, str(part))
        part.rename(target)
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
