'''
Compare snow season length ("snow days"), onset/meltout timing, and peak SWE date/magnitude
between SNOTEL observations and LM4 model output, for one or many sites.

Gap-screened snow days / onset / meltout / peak SWE date calculations were originally developed
in the exploratory data_processing/snotel_phenology.py script; that script has been retired and
the calculations now live here, self-contained. Model output has no gaps, so these functions
simply see missing_frac == 0 for every water year -- no separate no-gap logic is needed.

All *_wydoy fields (onset/meltout/peak timing) are day of water year, Oct 1 = day 1, rather than
calendar day-of-year. The peak/onset/meltout search window (Oct-Jul) straddles the Jan 1
boundary, so a year with e.g. a December onset (calendar DOY ~350) and a year with a January
onset (calendar DOY ~15) would average to a meaningless calendar-DOY midpoint; anchoring every
date to the water year it belongs to puts all years on one continuous, averaging-safe scale.

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

import argparse

import numpy as np
import pandas as pd
from model_datasets.output import LM4ModelOutput
from model_datasets.snotel import PNNLSnotel

# Snow days parameters
SWE_THRESHOLD_MM = 5.0      # SWE above this counts as a "snow day"
SNOW_WINDOW_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7]  # Oct-Jul; Aug-Sep excluded (reliably snow-free)
MAX_GAP_DAYS = 10           # longest single missing run that is safe to interpolate unconditionally
MAX_GAP_DAYS_FLAT = 30      # longest gap allowed when it's bounded by below-threshold SWE on both sides
MAX_GAP_FRACTION = 0.10     # total missing days allowed within the window, as a fraction of window length
MIN_RUN_DAYS = 5            # min consecutive above-threshold days to count as an onset/meltout run

# Peak SWE date parameters
PEAK_GAP_MONTHS = [2, 3, 4, 5]  # Feb-May: where peak SWE is typically found
PEAK_MAX_GAP_DAYS = 4            # max gap allowed if it overlaps PEAK_GAP_MONTHS, no flat-bounded exception


def water_year_start(water_year: int, tz: str | None = None) -> pd.Timestamp:
    '''Oct 1 immediately preceding the given water year (water year WY runs Oct 1 (WY-1) - Sep 30 WY).'''
    ts = pd.Timestamp(year=water_year - 1, month=10, day=1)
    return ts.tz_localize(tz) if tz is not None else ts


def water_year_doy(date: pd.Timestamp, water_year: int) -> float:
    '''Day of water year (Oct 1 = day 1) for a date, anchored to its own water_year label.'''
    if pd.isna(date):
        return np.nan
    return (date - water_year_start(water_year, date.tz)).days + 1


def filter_water_years(data: pd.DataFrame, year_range: tuple[int, int]) -> pd.DataFrame:
    '''Restrict data to water years start_wy-end_wy, inclusive.'''
    start_wy, end_wy = year_range
    water_year = np.where(data.index.month >= 10, data.index.year + 1, data.index.year)
    return data.loc[(water_year >= start_wy) & (water_year <= end_wy)]


def compute_snow_days(
    data: pd.DataFrame,
    swe_col: str = 'swe_mm',
    threshold_mm: float = SWE_THRESHOLD_MM,
    window_months: list[int] = SNOW_WINDOW_MONTHS,
    max_gap_days: int = MAX_GAP_DAYS,
    max_gap_days_flat: int = MAX_GAP_DAYS_FLAT,
    max_gap_fraction: float = MAX_GAP_FRACTION,
    min_run_days: int = MIN_RUN_DAYS,
) -> pd.DataFrame:
    '''
    Compute snow days, onset, and meltout per water year (Oct 1 - Sep 30) from a daily SWE series.

    A gap is safe to interpolate if it is no longer than max_gap_days, or -- if longer -- it is
    no longer than max_gap_days_flat AND the SWE values immediately bounding the gap are both
    below threshold_mm (i.e. the gap sits in a snow-free stretch rather than spanning an
    accumulation/melt transition). A water year is skipped (valid=False) if any gap fails both
    checks, or if total missing days in the window exceed max_gap_fraction of the window length.
    Otherwise gaps are linearly interpolated (time-based) before counting days with
    swe_col > threshold_mm.

    onset_wydoy is the day of water year of the first day of the first continuous run of
    swe_col > threshold_mm lasting at least min_run_days days. meltout_wydoy is the day of water
    year of the last day of the last such run.

    Returns a DataFrame indexed by water_year with columns: valid, snow_days, onset_wydoy,
    meltout_wydoy, max_gap_days, missing_frac, n_window_days.
    '''
    full_swe = data[swe_col]
    df = data[[swe_col]].copy()
    df['water_year'] = np.where(df.index.month >= 10, df.index.year + 1, df.index.year)
    in_window = df.index.month.isin(window_months)

    records = []
    for wy, group in df[in_window].groupby('water_year'):
        n_window = len(group)
        swe = group[swe_col]
        is_missing = swe.isna()
        total_missing = int(is_missing.sum())
        missing_frac = total_missing / n_window

        longest_gap = 0
        gaps_ok = True
        if is_missing.any():
            run_id = (is_missing != is_missing.shift()).cumsum()
            for _, run_index in is_missing[is_missing].groupby(run_id[is_missing]).groups.items():
                gap_len = len(run_index)
                longest_gap = max(longest_gap, gap_len)

                if gap_len <= max_gap_days:
                    continue

                # Longer gap: only acceptable if bounded by below-threshold SWE on both sides
                start_pos = full_swe.index.get_loc(run_index.min())
                end_pos = full_swe.index.get_loc(run_index.max())
                before_val = full_swe.iloc[start_pos - 1] if start_pos > 0 else np.nan
                after_val = full_swe.iloc[end_pos + 1] if end_pos + 1 < len(full_swe) else np.nan
                flat_bounded = (
                    pd.notna(before_val) and pd.notna(after_val)
                    and before_val <= threshold_mm and after_val <= threshold_mm
                )

                if gap_len > max_gap_days_flat or not flat_bounded:
                    gaps_ok = False
                    break

        valid = gaps_ok and (missing_frac <= max_gap_fraction)

        snow_days = np.nan
        onset_wydoy = np.nan
        meltout_wydoy = np.nan
        if valid:
            interpolated = swe.interpolate(method='time', limit=max_gap_days_flat, limit_area='inside')
            snow_days = int((interpolated > threshold_mm).sum())

            above = interpolated > threshold_mm
            run_id_above = (above != above.shift()).cumsum()
            for _, run_group in above[above].groupby(run_id_above[above]):
                if len(run_group) >= min_run_days:
                    onset_wydoy = water_year_doy(run_group.index[0], wy)
                    break

            # Switch direction for melt out: scan backward from the end of the window so the
            # first qualifying run found is the last one in real time.
            above_rev = above[::-1]
            run_id_above_rev = (above_rev != above_rev.shift()).cumsum()
            for _, run_group in above_rev[above_rev].groupby(run_id_above_rev[above_rev]):
                if len(run_group) >= min_run_days:
                    meltout_wydoy = water_year_doy(run_group.index[0], wy)
                    break

        records.append({
            'water_year': wy,
            'valid': valid,
            'snow_days': snow_days,
            'onset_wydoy': onset_wydoy,
            'meltout_wydoy': meltout_wydoy,
            'max_gap_days': longest_gap,
            'missing_frac': missing_frac,
            'n_window_days': n_window,
        })

    return pd.DataFrame(records).set_index('water_year')


def compute_peak_swe_date(
    data: pd.DataFrame,
    swe_col: str = 'swe_mm',
    threshold_mm: float = SWE_THRESHOLD_MM,
    window_months: list[int] = SNOW_WINDOW_MONTHS,
    max_gap_days: int = MAX_GAP_DAYS,
    max_gap_days_flat: int = MAX_GAP_DAYS_FLAT,
    max_gap_fraction: float = MAX_GAP_FRACTION,
    peak_gap_months: list[int] = PEAK_GAP_MONTHS,
    peak_max_gap_days: int = PEAK_MAX_GAP_DAYS,
) -> pd.DataFrame:
    '''
    Compute the date of peak SWE per water year (Oct 1 - Sep 30) from a daily SWE series.

    Uses the same gap screening as compute_snow_days, plus a stricter check: any missing run
    that overlaps peak_gap_months (Feb-May, where peak SWE is typically found) invalidates the
    water year if it's longer than peak_max_gap_days -- the flat-bounded exception does not apply
    here, since a gap during accumulation/peak season is unlikely to sit in a flat, snow-free
    stretch. This guards against a gap swallowing the true peak or shifting the apparent peak
    date. Gaps outside peak_gap_months still fall back to the standard max_gap_days /
    max_gap_days_flat / max_gap_fraction rules.

    Returns a DataFrame indexed by water_year with columns: valid, peak_date, peak_wydoy, peak_swe.
    '''
    full_swe = data[swe_col]
    df = data[[swe_col]].copy()
    df['water_year'] = np.where(df.index.month >= 10, df.index.year + 1, df.index.year)
    in_window = df.index.month.isin(window_months)

    records = []
    for wy, group in df[in_window].groupby('water_year'):
        n_window = len(group)
        swe = group[swe_col]
        is_missing = swe.isna()
        missing_frac = int(is_missing.sum()) / n_window

        gaps_ok = True
        if is_missing.any():
            run_id = (is_missing != is_missing.shift()).cumsum()
            for _, run_index in is_missing[is_missing].groupby(run_id[is_missing]).groups.items():
                gap_len = len(run_index)

                if run_index.month.isin(peak_gap_months).any():
                    if gap_len > peak_max_gap_days:
                        gaps_ok = False
                        break
                    continue

                if gap_len <= max_gap_days:
                    continue

                # Longer gap: only acceptable if bounded by below-threshold SWE on both sides
                start_pos = full_swe.index.get_loc(run_index.min())
                end_pos = full_swe.index.get_loc(run_index.max())
                before_val = full_swe.iloc[start_pos - 1] if start_pos > 0 else np.nan
                after_val = full_swe.iloc[end_pos + 1] if end_pos + 1 < len(full_swe) else np.nan
                flat_bounded = (
                    pd.notna(before_val) and pd.notna(after_val)
                    and before_val <= threshold_mm and after_val <= threshold_mm
                )

                if gap_len > max_gap_days_flat or not flat_bounded:
                    gaps_ok = False
                    break

        valid = gaps_ok and (missing_frac <= max_gap_fraction)

        peak_date = pd.NaT
        peak_swe = np.nan
        peak_wydoy = np.nan
        if valid:
            interpolated = swe.interpolate(method='time', limit=max_gap_days_flat, limit_area='inside')
            peak_date = interpolated.idxmax()
            peak_swe = interpolated.loc[peak_date]
            peak_wydoy = water_year_doy(peak_date, wy)

        records.append({
            'water_year': wy,
            'valid': valid,
            'peak_date': peak_date,
            'peak_wydoy': peak_wydoy,
            'peak_swe': peak_swe,
        })

    return pd.DataFrame(records).set_index('water_year')


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

    avg_onset_wydoy_model_all = model_snow.loc[model_snow['valid'], 'onset_wydoy'].mean()
    avg_meltout_wydoy_model_all = model_snow.loc[model_snow['valid'], 'meltout_wydoy'].mean()

    avg_peak_wydoy_snotel = snotel_peak.loc[snotel_peak['valid'], 'peak_wydoy'].mean()
    avg_peak_wydoy_model = model_peak.loc[matched_peak_years, 'peak_wydoy'].mean()
    avg_peak_wydoy_model_all = model_peak.loc[model_peak['valid'], 'peak_wydoy'].mean()

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
        'avg_onset_wydoy_model_all': r(avg_onset_wydoy_model_all),
        'avg_meltout_wydoy_model_all': r(avg_meltout_wydoy_model_all),
        'n_matched_snow_years': int(len(matched_snow_years)),
        'n_snow_years_model': int(model_snow['valid'].sum()),
        'avg_peak_wydoy_snotel': r(avg_peak_wydoy_snotel),
        'avg_peak_wydoy_model': r(avg_peak_wydoy_model),
        'avg_peak_wydoy_model_all': r(avg_peak_wydoy_model_all),
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
