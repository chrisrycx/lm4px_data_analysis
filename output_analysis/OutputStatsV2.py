'''
Compare snow season length ("snow days") and peak SWE date/magnitude between SNOTEL
observations and LM4 model output, for one or many sites.

Reuses the gap-screened snow days / peak SWE date calculations from
data_processing/snotel_phenology.py so both datasets share identical thresholds and window
definitions. Model output has no gaps, so those functions simply see missing_frac == 0 for every
water year -- no separate no-gap logic is needed.

Since SNOTEL sites can have individual water years skipped due to data gaps, but the model never
does, each site's model average is computed twice: once over all of the model's own valid water
years (avg_*_model_all), and once restricted to the water years also valid in the SNOTEL record
for that site (avg_*_model / n_matched_*_years) -- the latter is the fair apples-to-apples number
to compare against avg_*_snotel. Matching is a cheap index intersection done inline; no per-year
detail files are written, to keep output manageable across ~750 sites.

Set SNOTEL_PATH and MODEL_OUTPUT_PATH in a .env file at the repo root (see load_env_to_conda.py /
allpaths.env for the paths used on each machine).

Usage:
    python OutputStatsV2.py SITES.txt s01011010_ghelaps_v0008 [--year-range START END] [--output OUT.csv] [--spinup]
'''

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data_processing'))

import argparse

import pandas as pd
from model_datasets.output import LM4ModelOutput
from model_datasets.snotel import PNNLSnotel
from snotel_phenology import compute_peak_swe_date, compute_snow_days, filter_water_years


def parse_folder_code(folder_code: str) -> tuple[str, str, str]:
    '''
    Parse a model output folder code (e.g. 's01011010_ghelaps_v0008', or with a multi-part
    namelist like 's01011010_ghelaps_veg_v0009') into (forcing_string, namelist, version).
    '''
    if not folder_code.startswith('s'):
        raise ValueError(f"Folder code '{folder_code}' must start with 's'.")

    code_parts = folder_code.split('_')
    forcing_string = code_parts[0][1:]
    if len(code_parts) > 3:
        namelist = code_parts[1] + '_' + code_parts[2]
        version = code_parts[3][1:]
    else:
        namelist = code_parts[1]
        version = code_parts[2][1:]

    return forcing_string, namelist, version


def full_water_year_range(index: pd.Index) -> tuple[int, int]:
    '''
    Return (start_wy, end_wy), the range of water years fully covered by index -- i.e. excluding a
    partial water year at either end of the record (e.g. from spinup trimming leaving a start date
    that isn't Oct 1, or a run that doesn't end on Sep 30).
    '''
    if len(index) == 0:
        raise ValueError('Cannot determine water year range: index is empty.')

    start, end = index.min(), index.max()

    first_wy = start.year + 1 if start.month >= 10 else start.year
    if not (start.month == 10 and start.day == 1):
        first_wy += 1  # start date isn't Oct 1, so this water year is partial

    last_wy = end.year + 1 if end.month >= 10 else end.year
    if not (end.month == 9 and end.day == 30):
        last_wy -= 1  # end date isn't Sep 30, so this water year is partial

    return first_wy, last_wy


def load_model_swe(
    site: str,
    forcing_string: str,
    namelist: str,
    version: str,
    spinup: bool = False,
) -> pd.DataFrame:
    '''
    Load daily model SWE for one site, renamed to match the SNOTEL 'swe_mm' column, with the
    first year of the run dropped to avoid spinup effects.
    '''
    model_site_name = site.replace(' ', '').replace('#', '').lower()
    model_results = LM4ModelOutput(
        site=model_site_name, forcing_string=forcing_string, namelist=namelist, version=version, spinup=spinup
    )
    try:
        model_swe = model_results.get_data(['snow'], 'land_daily')
    finally:
        model_results.close()

    model_swe = model_swe.rename(columns={'snow': 'swe_mm'})[['swe_mm']]

    start_date = model_swe.index[0] + pd.DateOffset(years=1)
    return model_swe[model_swe.index >= start_date]


def process_site(
    site: str,
    snotel_path: str,
    forcing_string: str,
    namelist: str,
    version: str,
    spinup: bool = False,
    year_range: tuple[int, int] | None = None,
) -> dict:
    '''
    Compute matched SNOTEL vs. model phenology stats for one site.

    Returns a summary dict with one row's worth of fields for the batch CSV.
    '''
    snotel = PNNLSnotel(site, snotel_path)
    snotel.load_data()
    snotel_data = snotel.data
    if year_range is not None:
        snotel_data = filter_water_years(snotel_data, year_range)

    snotel_snow = compute_snow_days(snotel_data)
    snotel_peak = compute_peak_swe_date(snotel_data)

    model_swe = load_model_swe(site, forcing_string, namelist, version, spinup)
    model_range = full_water_year_range(model_swe.index)
    if year_range is not None:
        model_range = (max(model_range[0], year_range[0]), min(model_range[1], year_range[1]))
        if model_range[0] > model_range[1]:
            raise ValueError(f'No complete model water year overlaps --year-range for site {site}.')
    model_data = filter_water_years(model_swe, model_range)

    model_snow = compute_snow_days(model_data)
    model_peak = compute_peak_swe_date(model_data)

    matched_snow_years = snotel_snow.index[snotel_snow['valid']].intersection(model_snow.index[model_snow['valid']])
    matched_peak_years = snotel_peak.index[snotel_peak['valid']].intersection(model_peak.index[model_peak['valid']])

    avg_snow_days_snotel = snotel_snow.loc[snotel_snow['valid'], 'snow_days'].mean()
    avg_snow_days_model = model_snow.loc[matched_snow_years, 'snow_days'].mean()
    avg_snow_days_model_all = model_snow.loc[model_snow['valid'], 'snow_days'].mean()

    avg_peak_doy_snotel = snotel_peak.loc[snotel_peak['valid'], 'peak_doy'].mean()
    avg_peak_doy_model = model_peak.loc[matched_peak_years, 'peak_doy'].mean()
    avg_peak_doy_model_all = model_peak.loc[model_peak['valid'], 'peak_doy'].mean()

    avg_peak_swe_snotel = snotel_peak.loc[snotel_peak['valid'], 'peak_swe'].mean()
    avg_peak_swe_model = model_peak.loc[matched_peak_years, 'peak_swe'].mean()
    avg_peak_swe_model_all = model_peak.loc[model_peak['valid'], 'peak_swe'].mean()

    def r(x):
        return round(x, 1) if pd.notna(x) else None

    return {
        'site': site,
        'latitude': snotel.latitude,
        'longitude': snotel.longitude,
        'avg_snow_days_snotel': r(avg_snow_days_snotel),
        'avg_snow_days_model': r(avg_snow_days_model),
        'avg_snow_days_model_all': r(avg_snow_days_model_all),
        'n_matched_snow_years': int(len(matched_snow_years)),
        'n_snow_years_model': int(model_snow['valid'].sum()),
        'avg_peak_doy_snotel': r(avg_peak_doy_snotel),
        'avg_peak_doy_model': r(avg_peak_doy_model),
        'avg_peak_doy_model_all': r(avg_peak_doy_model_all),
        'avg_peak_swe_snotel': r(avg_peak_swe_snotel),
        'avg_peak_swe_model': r(avg_peak_swe_model),
        'avg_peak_swe_model_all': r(avg_peak_swe_model_all),
        'n_matched_peak_years': int(len(matched_peak_years)),
        'n_peak_years_model': int(model_peak['valid'].sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare snow season length and peak SWE date between SNOTEL observations and LM4 model output.'
    )
    parser.add_argument('sites_file', help='Path to a text file of site names, one per line.')
    parser.add_argument('folder_code', help="Model output folder code, e.g. 's01011010_ghelaps_v0008'.")
    parser.add_argument('--output', default=None, help='CSV path for output (default: phenology_<sites_file>_<folder_code>.csv)')
    parser.add_argument(
        '--year-range', nargs=2, type=int, metavar=('START', 'END'), default=None,
        help='Restrict to water years START-END, inclusive (default: use all available data)',
    )
    parser.add_argument('--spinup', action='store_true', help='Load model output from the spinup run directory.')
    return parser.parse_args()


def main():
    args = parse_args()

    snotel_path = os.getenv('SNOTEL_PATH')
    if not snotel_path:
        raise SystemExit('SNOTEL_PATH environment variable is not set (check .env).')

    try:
        forcing_string, namelist, version = parse_folder_code(args.folder_code)
    except ValueError as e:
        raise SystemExit(str(e))

    year_range = None
    if args.year_range:
        start_wy, end_wy = args.year_range
        if start_wy > end_wy:
            raise SystemExit(f'--year-range start ({start_wy}) must be <= end ({end_wy}).')
        year_range = (start_wy, end_wy)

    with open(args.sites_file) as f:
        sites = [line.strip() for line in f if line.strip()]

    results = []
    for site in sites:
        print(f'Processing site {site}')
        try:
            result = process_site(site, snotel_path, forcing_string, namelist, version, args.spinup, year_range)
        except (ValueError, FileNotFoundError) as e:
            print(f'  Skipping {site}: {e}')
            continue
        results.append(result)

    output_path = args.output
    if output_path is None:
        base_sites_file = os.path.splitext(os.path.basename(args.sites_file))[0]
        output_path = f'phenology_{base_sites_file}_{args.folder_code}.csv'

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f'\nWrote {len(results_df)} sites to {output_path}')
    return results_df


if __name__ == '__main__':
    main()
