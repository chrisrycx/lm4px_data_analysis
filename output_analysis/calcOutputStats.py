'''
Calculate the Root Mean Square Error (RMSE), bias, and other statistics for model outputs against snotel data.
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
    try:
        snotel = PNNLSnotel(site_name=site, storage_path=SNOTEL_PATH)
    except ValueError as ve:
        print(f'Site not found:{site}')
        continue
        
    snotel.load_data()
    
    # Get just the SWE data from model output
    try:
        model_results = LM4ModelOutput(site=model_site_name, forcing_string=forcing_string, namelist=namelist, version=version, spinup=False)
        model_swe = model_results.get_data(['snow'], 'land_daily')
    except FileNotFoundError as e:
        print(f'Error loading model results for site {site}: {e}')
        continue
    
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

    # Remove first year of data to avoid spinup effects
    if not snow_data.empty:
        start_date = snow_data.index[0] + pd.DateOffset(years=1)
        snow_data = snow_data[snow_data.index >= start_date]

    # Save number of years of data, mean, and variance of SNOTEL SWE
    num_years = (snow_data.index[-1] - snow_data.index[0]).days / 365.25
    mean_snotel_swe = snow_data['swe_mm'].mean() if not snow_data.empty else None
    var_snotel_swe = float(snow_data['swe_mm'].var()) if not snow_data.empty else None

    # Calculate RMSE and bias
    if not snow_data.empty:
        rmse = ((snow_data['snow'] - snow_data['swe_mm']) ** 2).mean() ** 0.5
        bias = (snow_data['snow'] - snow_data['swe_mm']).mean()
    else:
        rmse = None
        bias = None

    # Calculate the number of "model snow days" and "snotel snow days" where SWE > 5mm
    model_snow_days = (snow_data['snow'] > 5).count() if not snow_data.empty else 0
    snotel_snow_days = (snow_data['swe_mm'] > 5).count() if not snow_data.empty else 0

    # Append results to the list
    results_list.append({
        'site': site,
        'latitude': snotel.latitude,
        'longitude': snotel.longitude,
        'output_years': round(num_years, 1),
        'bias': round(bias, 1) if bias is not None else None,
        'rmse': round(rmse, 1) if rmse is not None else None,
        'model_snow_days': model_snow_days,
        'snotel_snow_days': snotel_snow_days,
        'mean_snotel_swe': round(mean_snotel_swe, 1) if mean_snotel_swe is not None else None,
        'var_snotel_swe': round(var_snotel_swe, 1) if var_snotel_swe is not None else None
    })

    # Close the objects to free resources
    model_results.close()

# Convert results list to DataFrame
results = pd.DataFrame(results_list)
print(results.head())

# Save results to CSV in the current directory, removing any path from sites_file
base_sites_file = os.path.splitext(os.path.basename(sites_file))[0]
results.to_csv(f'{base_sites_file}_{folder_code}.csv', index=False)
