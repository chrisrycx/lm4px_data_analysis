'''
Calculate snow days per water year for a SNOTEL site.

Season length is calculated as "snow days": the count of days with `swe_mm` above a threshold
within the snow-relevant window of each water year (Oct 1 - Sep 30, water year labeled by the
year containing Jan-Sep). This avoids needing to detect a single contiguous onset/melt-out span,
so it's more robust at sites with intermittent snowpack.

Missing data is screened per water year, restricted to the plausible snow window (Oct-Jul,
excluding Aug-Sep when SWE is reliably zero). A water year is skipped entirely if either:
- the longest single run of missing days in the window exceeds MAX_GAP_DAYS (unless it's bounded
  by below-threshold SWE on both sides, in which case MAX_GAP_DAYS_FLAT applies instead), or
- the total missing days in the window exceed MAX_GAP_FRACTION of the window length.

Otherwise, short gaps are linearly interpolated (time-based) before counting snow days.

Peak SWE date is the date of maximum interpolated SWE within each water year. It uses the same
gap screening as snow days, plus a stricter check: gaps overlapping Feb-May cannot exceed
PEAK_MAX_GAP_DAYS, since a gap there risks hiding or shifting the true peak. Valid/skipped water
years are tracked separately from the snow days calculation, since the two use different gap
criteria.

Set the SNOTEL_PATH environment variable before running.

Usage:
    python snotel_phenology.py [--site SITE] [--year-range START END]
'''

import argparse
import os

import numpy as np
import pandas as pd
from model_datasets.snotel import PNNLSnotel

SITE = 'Garita Peak'

# Snow days parameters
SWE_THRESHOLD_MM = 5.0      # SWE above this counts as a "snow day"
SNOW_WINDOW_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7]  # Oct-Jul; Aug-Sep excluded (reliably snow-free)
MAX_GAP_DAYS = 10           # longest single missing run that is safe to interpolate unconditionally
MAX_GAP_DAYS_FLAT = 30      # longest gap allowed when it's bounded by below-threshold SWE on both sides
MAX_GAP_FRACTION = 0.10     # total missing days allowed within the window, as a fraction of window length

# Peak SWE date parameters
PEAK_GAP_MONTHS = [2, 3, 4, 5]  # Feb-May: where peak SWE is typically found
PEAK_MAX_GAP_DAYS = 4            # max gap allowed if it overlaps PEAK_GAP_MONTHS, no flat-bounded exception


def compute_snow_days(
    data: pd.DataFrame,
    swe_col: str = 'swe_mm',
    threshold_mm: float = SWE_THRESHOLD_MM,
    window_months: list[int] = SNOW_WINDOW_MONTHS,
    max_gap_days: int = MAX_GAP_DAYS,
    max_gap_days_flat: int = MAX_GAP_DAYS_FLAT,
    max_gap_fraction: float = MAX_GAP_FRACTION,
) -> pd.DataFrame:
    '''
    Compute snow days per water year (Oct 1 - Sep 30) from a daily SWE series.

    A gap is safe to interpolate if it is no longer than max_gap_days, or -- if longer -- it is
    no longer than max_gap_days_flat AND the SWE values immediately bounding the gap are both
    below threshold_mm (i.e. the gap sits in a snow-free stretch rather than spanning an
    accumulation/melt transition). A water year is skipped (valid=False, snow_days=NaN) if any
    gap fails both checks, or if total missing days in the window exceed max_gap_fraction of the
    window length. Otherwise gaps are linearly interpolated (time-based) before counting days
    with swe_col > threshold_mm.

    Returns a DataFrame indexed by water_year with columns: valid, snow_days, max_gap_days,
    missing_frac, n_window_days.
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
        if valid:
            interpolated = swe.interpolate(method='time', limit=max_gap_days_flat, limit_area='inside')
            snow_days = int((interpolated > threshold_mm).sum())

        records.append({
            'water_year': wy,
            'valid': valid,
            'snow_days': snow_days,
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

    Returns a DataFrame indexed by water_year with columns: valid, peak_date, peak_doy, peak_swe.
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
        if valid:
            interpolated = swe.interpolate(method='time', limit=max_gap_days_flat, limit_area='inside')
            peak_date = interpolated.idxmax()
            peak_swe = interpolated.loc[peak_date]

        records.append({
            'water_year': wy,
            'valid': valid,
            'peak_date': peak_date,
            'peak_doy': peak_date.dayofyear if pd.notna(peak_date) else np.nan,
            'peak_swe': peak_swe,
        })

    return pd.DataFrame(records).set_index('water_year')


def filter_water_years(data: pd.DataFrame, year_range: tuple[int, int]) -> pd.DataFrame:
    '''Restrict data to water years start_wy-end_wy, inclusive.'''
    start_wy, end_wy = year_range
    water_year = np.where(data.index.month >= 10, data.index.year + 1, data.index.year)
    return data.loc[(water_year >= start_wy) & (water_year <= end_wy)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Calculate snow days and peak SWE date per water year for a SNOTEL site.')
    parser.add_argument('--site', default=SITE, help=f"SNOTEL site name (default: '{SITE}')")
    parser.add_argument('--year-range', nargs=2, type=int, metavar=('START', 'END'), default=None,
                         help='Restrict to water years START-END, inclusive (default: use all available data)')
    return parser.parse_args()


def main():
    args = parse_args()
    site = args.site

    snotel_path = os.getenv('SNOTEL_PATH')
    if not snotel_path:
        raise SystemExit("SNOTEL_PATH environment variable is not set.")

    snotel = PNNLSnotel(site, snotel_path)
    snotel.load_data()

    data = snotel.data
    if args.year_range:
        start_wy, end_wy = args.year_range
        if start_wy > end_wy:
            raise SystemExit(f"--year-range start ({start_wy}) must be <= end ({end_wy}).")
        data = filter_water_years(data, args.year_range)

    snow_day_results = compute_snow_days(data)

    n_valid = snow_day_results['valid'].sum()
    n_skipped = (~snow_day_results['valid']).sum()
    mean_snow_days = snow_day_results.loc[snow_day_results['valid'], 'snow_days'].mean()

    print(f"{site}: {n_valid} valid water years, {n_skipped} skipped (snow days)")
    print(f"Mean snow days per season: {mean_snow_days:.1f}")
    print(snow_day_results)

    peak_results = compute_peak_swe_date(data)

    n_peak_valid = peak_results['valid'].sum()
    n_peak_skipped = (~peak_results['valid']).sum()
    mean_peak_doy = peak_results.loc[peak_results['valid'], 'peak_doy'].mean()

    print(f"\n{site}: {n_peak_valid} valid water years, {n_peak_skipped} skipped (peak SWE date)")
    if pd.notna(mean_peak_doy):
        mean_peak_date = pd.Timestamp('2001-01-01') + pd.Timedelta(days=mean_peak_doy - 1)
        print(f"Mean peak SWE DOY: {mean_peak_doy:.1f} (~{mean_peak_date.strftime('%b %d')})")
    print(peak_results)

    return {'snow_days': snow_day_results, 'peak_swe': peak_results}


if __name__ == '__main__':
    main()
