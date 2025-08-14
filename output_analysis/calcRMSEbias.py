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

# Initialize an empty DataFrame to store results
results = pd.DataFrame(columns=['site', 'bias', 'rmse'])

# Loop through each site and calculate RMSE and bias
for site in sites:
    model_site_name = site.replace(' ', '').replace('#', '').lower()
    model_results = LM4ModelOutput(site=model_site_name, forcing_string=forcing_string, namelist=namelist, version=version, spinup=False)
    snotel_data = PNNLSnotel(site_name=site, storage_path=SNOTEL_PATH)
    


