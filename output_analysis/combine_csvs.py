import pandas as pd
import glob
import os

# Helper function to extract the middle part of the filename for use as a suffix
def extract_filename_suffix(filepath):
    """
    Extract the middle part of the filename to use as a suffix.
    E.g., 'snotel_sites_v2_s00000000_ghec_v0003.csv' -> 's00000000_ghec'
    """
    filename = os.path.basename(filepath)
    # Remove prefix 'snotel_sites_v2'
    if filename.startswith('snotel_sites_v2'):
        middle = filename[len('snotel_sites_v2'):]
    else:
        middle = filename

    # Remove last 10 characters (e.g., '_v0003.csv')
    middle = middle[:-10]

    # Remove leading underscore if present
    middle = middle.lstrip('_')

    return middle

# Find all CSV files matching the pattern 'snotel_sites_v2*'
file_paths = sorted(glob.glob('snotel_sites_v2*.csv'))

if len(file_paths) < 1:
    print("Error: No files matching 'snotel_sites_v2*.csv' found in current directory.")
    exit(1)

print(f"Found {len(file_paths)} file(s):")
file_suffixes = []
for fp in file_paths:
    suffix = extract_filename_suffix(fp)
    file_suffixes.append(suffix)
    print(f"  - {fp} -> suffix: {suffix}")

# Read all files and validate column consistency
dfs = [pd.read_csv(fp) for fp in file_paths]

# Extract columns that are NOT bias or rmse for the first file
first_file_cols = dfs[0].columns.tolist()
common_cols = [col for col in first_file_cols if 'bias' not in col.lower() and 'rmse' not in col.lower()]
bias_cols = [col for col in first_file_cols if 'bias' in col.lower()]
rmse_cols = [col for col in first_file_cols if 'rmse' in col.lower()]

print(f"\nCommon columns (should be identical across files): {common_cols}")
print(f"Bias columns found: {bias_cols}")
print(f"RMSE columns found: {rmse_cols}")

# Validate that all files have the same common columns
print("\nValidating column consistency across files...")
all_valid = True
for i, df in enumerate(dfs, start=1):
    file_cols = [col for col in df.columns.tolist() if 'bias' not in col.lower() and 'rmse' not in col.lower()]
    if file_cols != common_cols:
        print(f"ERROR in file {i} ({file_paths[i-1]}):")
        print(f"  Expected common columns: {common_cols}")
        print(f"  Got common columns: {file_cols}")
        all_valid = False
    else:
        print(f"✓ File {i} ({file_paths[i-1]}): Valid")

if not all_valid:
    print("\nERROR: Not all files have the same common columns. Please check your files.")
    exit(1)

# Validate that the 'site' column is identical across all files
print("\nValidating 'site' column consistency...")
if 'site' not in common_cols:
    print("WARNING: 'site' column not found in common columns.")
else:
    first_sites = dfs[0]['site'].tolist()
    all_sites_match = True
    for i, df in enumerate(dfs[1:], start=2):
        current_sites = df['site'].tolist()
        if current_sites != first_sites:
            print(f"ERROR in file {i} ({file_paths[i-1]}):")
            print(f"  'site' column values do not match file 1.")
            all_sites_match = False
        else:
            print(f"✓ File {i} ({file_paths[i-1]}): 'site' column matches file 1")

    if not all_sites_match:
        print("\nERROR: Not all files have the same 'site' column values. Please check your files.")
        exit(1)

# Start with the first dataframe
df_combined = dfs[0].copy()

# Rename bias and rmse columns with suffix from filename
suffix = file_suffixes[0]
rename_dict = {col: f"{col}_{suffix}" for col in bias_cols + rmse_cols}
df_combined.rename(columns=rename_dict, inplace=True)

# Process remaining files
for i, df in enumerate(dfs[1:], start=1):
    # Get bias and rmse columns from this file
    file_bias_cols = [col for col in df.columns.tolist() if 'bias' in col.lower()]
    file_rmse_cols = [col for col in df.columns.tolist() if 'rmse' in col.lower()]

    # Extract bias and rmse columns and rename with suffix from filename
    df_subset = df[file_bias_cols + file_rmse_cols].copy()
    suffix = file_suffixes[i]
    rename_dict = {col: f"{col}_{suffix}" for col in file_bias_cols + file_rmse_cols}
    df_subset.rename(columns=rename_dict, inplace=True)

    # Concatenate horizontally (by index)
    df_combined = pd.concat([df_combined, df_subset], axis=1)

# Calculate differences from baseline (s00000000_ghec)
print("\nCalculating differences from baseline (s00000000_ghec)...")
baseline_rmse_col = 'rmse_s00000000_ghec'
baseline_bias_col = 'bias_s00000000_ghec'

# Check if baseline columns exist
if baseline_rmse_col not in df_combined.columns:
    print(f"WARNING: Baseline column '{baseline_rmse_col}' not found. Skipping difference calculations.")
elif baseline_bias_col not in df_combined.columns:
    print(f"WARNING: Baseline column '{baseline_bias_col}' not found. Skipping difference calculations.")
else:
    # Find all rmse and bias columns
    all_rmse_cols = [col for col in df_combined.columns if col.startswith('rmse_')]
    all_bias_cols = [col for col in df_combined.columns if col.startswith('bias_')]

    # Calculate differences for RMSE columns
    for rmse_col in all_rmse_cols:
        if rmse_col != baseline_rmse_col:
            # Extract the suffix after 'rmse_'
            suffix = rmse_col[5:]  # Remove 'rmse_' prefix
            diff_col_name = f'drmse_{suffix}'
            df_combined[diff_col_name] = df_combined[rmse_col] - df_combined[baseline_rmse_col]
            print(f"  Created {diff_col_name} = {rmse_col} - {baseline_rmse_col}")

    # Calculate differences for bias columns
    for bias_col in all_bias_cols:
        if bias_col != baseline_bias_col:
            # Extract the suffix after 'bias_'
            suffix = bias_col[5:]  # Remove 'bias_' prefix
            diff_col_name = f'dbias_{suffix}'
            df_combined[diff_col_name] = df_combined[bias_col] - df_combined[baseline_bias_col]
            print(f"  Created {diff_col_name} = {bias_col} - {baseline_bias_col}")

# Save the combined dataframe
output_path = 'combined_results.csv'
df_combined.to_csv(output_path, index=False)
print(f"\n✓ Combined CSV saved to: {output_path}")
print(f"Shape: {df_combined.shape}")
print(f"\nColumns in combined file:")
print(df_combined.columns.tolist())
