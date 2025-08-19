'''
Calculate the Root Mean Square Error (RMSE) and bias for model outputs against snotel data.
-- Input path to list of sites and output folder code such as 's00000000_gbc_v0.2.1'
-- Output is a CSV file with RMSE and bias values for each site.

Set environment variables: OUTPUT_PATH and SNOTEL_PATH before running
'''

import pandas as pd
from model_datasets.output import LM4ModelOutput 
from buildforcing.datasets import PNNLSnotel

# Load SNOTEL_PATH from environment variables
import os
SNOTEL_PATH = os.getenv('SNOTEL_PATH')
if not SNOTEL_PATH:
    print("SNOTEL_PATH environment variable is not set.")
    sys.exit(1)

# Load the command line arguments
import sys
if len(sys.argv) != 3:
    print("Usage: python calcRMSEbias.py <sites file> <folder code>")
    sys.exit(1)

sites_file = sys.argv[1]
folder_code = sys.argv[2]

# Parse the folder code
if not folder_code.startswith('s'):
    print("Folder code must start with 's'.")
    sys.exit(1)

code_parts = folder_code.split('_')
forcing_string = code_parts[0][1:]  # Remove the 's' prefix
namelist = code_parts[1]
version = code_parts[2][1:]

# Read the sites from the provided file, there should be one site per line
with open(sites_file, 'r') as f:
    sites = [line.strip() for line in f if line.strip()]

# Initialize a list to store results
results_list = []

# Loop through each site and calculate RMSE and bias
for site in sites:
    print(f'Processing site {site}')
    model_site_name = site.replace(' ', '').replace('#', '').lower()
    model_results = LM4ModelOutput(site=model_site_name, forcing_string=forcing_string, namelist=namelist, version=version, spinup=False)
    snotel = PNNLSnotel(site_name=site, storage_path=SNOTEL_PATH)
    snotel.load_data()
    
    # Get just the SWE data
    model_swe = model_results.get_data(['snow'], 'land_daily')
    snotel_swe = snotel.data['swe_mm']
    snotel_swe.index = snotel_swe.index.tz_localize(None)
    snotel_swe.index = snotel_swe.index + pd.Timedelta(hours=12)  # Adjust SNOTEL data to match model time

    # Merge model and SNOTEL data on index
    snow_data = pd.merge(
        model_swe,
        snotel_swe,
        how='left',
        left_index=True, right_index=True
    ).dropna()

    # Calculate RMSE and bias
    if not snow_data.empty:
        rmse = ((snow_data['snow'] - snow_data['swe_mm']) ** 2).mean() ** 0.5
        bias = (snow_data['snow'] - snow_data['swe_mm']).mean()
    else:
        rmse = None
        bias = None

    # Append results to the list
    results_list.append({
        'site': site,
        'latitude': snotel.latitude,
        'longitude': snotel.longitude,
        'bias': round(bias, 1) if bias is not None else None,
        'rmse': round(rmse, 1) if rmse is not None else None
    })

    # Close the objects to free resources
    model_results.close()

# Convert results list to DataFrame
results = pd.DataFrame(results_list)
print(results.head())

# Save results to CSV in the current directory, removing any path from sites_file
base_sites_file = os.path.splitext(os.path.basename(sites_file))[0]
results.to_csv(f'{base_sites_file}_{folder_code}.csv', index=False)
