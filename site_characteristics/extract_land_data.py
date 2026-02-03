'''
Extract data from ESM1k land datasets for SNOTEL station locations.
'''

import xarray as xr
import pandas as pd
import os
from tqdm import tqdm
import time

# Load the SNOTEL summary CSV file
SNOTEL_PATH = os.getenv('SNOTEL_PATH')
ESM1K_PATH = os.getenv('ESM1K_PATH')
if not SNOTEL_PATH or not ESM1K_PATH:
	raise ValueError("Environment variable 'SNOTEL_PATH' or 'ESM1K_PATH' is not set.")
snotel_file = os.path.join(SNOTEL_PATH, 'bcqc_data_v2', 'SNOTEL_summary.csv')
snotel_data = pd.read_csv(snotel_file,header=0)

# Simplify loaded Snotel data by selecting relevant columns
snotel_data = snotel_data[['SNOTEL Name', 'SNOTEL Lat ()', 'SNOTEL Lon ()']] 

# Extract unique latitudes and longitudes
unique_coords = snotel_data[['SNOTEL Lat ()', 'SNOTEL Lon ()']].drop_duplicates()

print(f"Processing {len(unique_coords)} unique coordinate pairs...")

# Extract data from ESM1k land datasets
datasets = ['ELEVATION', 'SLOPE', 'ASPECT', 'SKY_VIEW_FACTOR', 'STDEV_ELEV']
dataset_decimals = [0,1,1,4,4,4]  # Decimal places for each dataset

print(f"\nExtracting data from {len(datasets)} datasets...")
for dataset, decimals in tqdm(zip(datasets, dataset_decimals), total=len(datasets), desc="Overall Progress"):
    print(f"\n  Processing {dataset}...")
    start_time = time.time()

    netcdf_file = os.path.join(ESM1K_PATH, f"{dataset}_1k_c230606.nc")

    # Open dataset with chunking for better performance
    ds = xr.open_dataset(netcdf_file, chunks='auto')

    # Select the nearest data points for all unique coordinates
    print(f"  Selecting {len(unique_coords)} coordinate pairs...")
    subset_ds = ds.sel(
        lat=xr.DataArray(unique_coords['SNOTEL Lat ()'], dims="points"),
        lon=xr.DataArray(unique_coords['SNOTEL Lon ()'], dims="points"),
        method="nearest"
    )

    # Load data into memory and convert to DataFrame
    print(f"  Loading data into memory...")
    subset_df = subset_ds.load().to_dataframe().reset_index()

    ds.close()

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f} seconds")

    # Round latitude and longitude values to 2 decimal places for consistency
    snotel_data['SNOTEL Lat ()'] = snotel_data['SNOTEL Lat ()'].round(2)
    snotel_data['SNOTEL Lon ()'] = snotel_data['SNOTEL Lon ()'].round(2)

    # Just use relevant columns
    subset_df = subset_df[['lat','lon',dataset]]

    subset_df['lat'] = subset_df['lat'].round(2)
    subset_df['lon'] = subset_df['lon'].round(2)

    # Merge the dataset back to the original DataFrame
    snotel_data = snotel_data.merge(
        subset_df,
        left_on=['SNOTEL Lat ()', 'SNOTEL Lon ()'],
        right_on=['lat', 'lon'],
        how='left'
    )

    # Drop the 'lat' and 'lon' columns from the merged DataFrame
    snotel_data.drop(columns=['lat', 'lon'], inplace=True)

    # Round the dataset values to the specified number of decimals
    snotel_data[dataset] = snotel_data[dataset].round(decimals)


# Check if the merge was successful
print("\n" + "="*60)
print("Merged SNOTEL Data Sample:")
print(snotel_data.head())

# Save the data to a new CSV file
output_file = os.path.join(SNOTEL_PATH, 'snotel_land_data.csv')

print(f"\nSaving results to {output_file}...")
snotel_data.to_csv(output_file, index=False)

print(f"\nSUCCESS! SNOTEL land data saved to {output_file}")
print(f"Processed {len(snotel_data)} stations with {len(datasets)} datasets")