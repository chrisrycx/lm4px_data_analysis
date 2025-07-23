'''
Extract data from ESM1k land datasets for SNOTEL station locations.
'''

import xarray as xr
import pandas as pd
import os

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

# Extract data from ESM1k land datasets
datasets = ['ELEVATION', 'SLOPE', 'ASPECT', 'SKY_VIEW_FACTOR', 'TERRAIN_CONFIG']
dataset_decimals = [0,0,0,2,2]  # Decimal places for each dataset
for dataset, decimals in zip(datasets, dataset_decimals):
    netcdf_file = os.path.join(ESM1K_PATH, f"{dataset}_1k_c230606.nc")
    ds = xr.open_dataset(netcdf_file)

    # Select the nearest data points for all unique coordinates
    subset_ds = ds.sel(
        lat=xr.DataArray(unique_coords['SNOTEL Lat ()'], dims="points"),
        lon=xr.DataArray(unique_coords['SNOTEL Lon ()'], dims="points"),
        method="nearest"
    )

    # Convert the subset dataset to a DataFrame for easier mapping
    subset_df = subset_ds.to_dataframe().reset_index()

    # Round latitude and longitude values to 4 decimal places for consistency
    snotel_data['SNOTEL Lat ()'] = snotel_data['SNOTEL Lat ()'].round(2)
    snotel_data['SNOTEL Lon ()'] = snotel_data['SNOTEL Lon ()'].round(2)

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

    ds.close()  # Close the dataset to free resources


# Check if the merge was successful
print("Merged SNOTEL Data Sample:")
print(snotel_data.head())

# Save the data to a new CSV file
output_file = os.path.join(SNOTEL_PATH, 'snotel_land_data.csv')
# Round specific columns to the desired number of decimals before saving

# Save the data to a new CSV file
snotel_data.to_csv(output_file, index=False)

print(f"SNOTEL land data saved to {output_file}")