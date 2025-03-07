'''
Objects to get and store data from NLDAS or the PNNL snotel data (on Google Drive).
'''

import requests
import pandas as pd
from gdriver.gdriver import get_file, get_credentials
from datetime import datetime

class PNNLSnotel:
    '''
    Class to store and manipulate the PNNL Snotel data.
    '''
    
    def __init__(self, site_name):
        self.site_name = site_name

        # Load gdriver credentials
        self.credentials = get_credentials()

        # Load the metadata and associated data
        self.exists = self.load_metadata()
        if self.exists:
            self.load_data()

    def load_metadata(self):
        '''
        Load the metadata for the PNNL Snotel data.
        '''
        # Download the metadata file from Google Drive
        summary_file = get_file('bcqc_data_v2/SNOTEL_summary.csv', creds=self.credentials)

        # For each line in the metadata file, split the line by commas and search for site name in column 3
        site_match = False
        with open(summary_file, 'r') as f:
            for line in f:
                column_values = line.split(',')
                if column_values[3] == self.site_name:
                    site_match = True
                    self.elevation = int(column_values[4])
                    self.latitude = float(column_values[5])
                    self.longitude = float(column_values[6])
                    self.start_date = datetime.strptime(column_values[7], '%m/%d/%Y')
                    self.end_date = datetime.strptime(column_values[8], '%m/%d/%Y')
        
        return site_match
    
    def load_data(self):

        # Build the file name for the site. The file name uses the site latitude and longitude to 5 decimal places: "bcqc_latitude_longitude.txt"
        file_name = f'bcqc_{self.latitude:.5f}_{self.longitude:.5f}.txt'

        # Download the txt file from Google Drive
        rawfile = get_file(file_name, creds=self.credentials)

        # Read the txt file into a pandas dataframe. No header, separator is spaces
        snotel_data = pd.read_csv(rawfile, delimiter=r'\s+', header=None, names=['Year', 'Month', 'Day', 'Precipitation', 'Max Temp', 'Min Temp', 'Avg Temp', 'Snow Water Equivalent'])

        # convert Month, Day, Year to a datetime object and set as index
        snotel_data['Date'] = pd.to_datetime(snotel_data[['Year', 'Month', 'Day']])
        snotel_data.set_index('Date', inplace=True)
        snotel_data.drop(columns=['Year', 'Month', 'Day'], inplace=True)

        # Rename columns
        snotel_data.rename(columns={'Precipitation': 'precip_in', 'Max Temp': 'T_max_F', 'Min Temp': 'T_min_F', 'Avg Temp': 'T_avg_F', 'Snow Water Equivalent': 'swe_in'}, inplace=True)

        self.data = snotel_data



