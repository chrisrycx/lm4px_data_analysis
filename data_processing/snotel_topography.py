import xarray as xr

# Load the NetCDF file
file_path = "path_to_your_file.nc"  # Replace with your file path
dataset = xr.open_dataset(file_path)

# Define the list of latitude and longitude coordinates
latitudes = [30.5, 31.0, 32.5]  # Replace with your desired latitude values
longitudes = [-120.5, -119.0, -118.5]  # Replace with your desired longitude values

# Extract the subset of data for the specified coordinates
subset = dataset.sel(lat=xr.DataArray(latitudes, dims="points"),
                     lon=xr.DataArray(longitudes, dims="points"))

# Save the subset to a new NetCDF file
output_path = "subset_elevation_data.nc"  # Replace with your desired output file name
subset.to_netcdf(output_path)

print(f"Subset saved to {output_path}")