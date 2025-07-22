'''
Extract data from ESM1k land datasets for SNOTEL station locations.
'''

import xarray as xr
import pandas as pd
import os

# Load the SNOTEL summary CSV file
SNOTEL_PATH = os.getenv('SNOTEL_PATH')
if not SNOTEL_PATH:
	raise ValueError("Environment variable 'SNOTEL_PATH' is not set.")
snotel_file = os.path.join(SNOTEL_PATH, 'bcqc_data_v2', 'SNOTEL_summary.csv')
snotel_data = pd.read_csv(snotel_file,header=0)

# Print the first few rows of the SNOTEL data for verification
print("SNOTEL Data Sample:")
print(snotel_data.head())

# Simplify loaded Snotel data by selecting relevant columns
snotel_data = snotel_data[['SNOTEL Name', 'SNOTEL Elev (ft)', 'SNOTEL Lat ()', 'SNOTEL Lon ()']]

# Convert elevation from feet to meters
snotel_data['SNOTEL Elev (m)'] = snotel_data['SNOTEL Elev (ft)'] / 3.28084  

# Extract unique latitudes and longitudes
unique_coords = snotel_data[['SNOTEL Lat ()', 'SNOTEL Lon ()']].drop_duplicates()

# Open the NetCDF file
ESM1K_PATH = "c:/Users/clmbn/NMT_PhD/data/ESM1k_land"
netcdf_file = os.path.join(ESM1K_PATH,"ELEVATION_1k_c230606.nc")
ds = xr.open_dataset(netcdf_file)

# Use xarray to select the nearest data points for all unique coordinates
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

# Merge the elevation data back to the original DataFrame
snotel_data = snotel_data.merge(
    subset_df,
    left_on=['SNOTEL Lat ()', 'SNOTEL Lon ()'],
    right_on=['lat', 'lon'],
    how='left'
)

# Calculate difference in elevation
snotel_data['Elevation Difference (m)'] = subset_df['ELEVATION'] - snotel_data['SNOTEL Elev (m)']

# Check if the merge was successful
print("Merged SNOTEL Data Sample:")
print(snotel_data.head())

# Save the subset dataset to a new NetCDF file
#output_file = "c:/Users/clmbn/NMT_PhD/data/ESM1k_land/ELEVATION_subset.nc"
#subset_ds.to_netcdf(output_file)

# Close the dataset
ds.close()
subset_ds.close()

#print(f"Subset dataset saved to {output_file}")