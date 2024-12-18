'''
Initial gdriver test script
'''

from gdriver.gdriver import get_file_id, get_credentials, getCSVtoPandas
from googleapiclient.discovery import build

#filename = "TonyGrove/LR_TG_C_WindSp_Max_SourceID_1_QC_1.csv"
filename = "GAMUT/testdata.csv"

creds = get_credentials()
service = build("drive", "v3", credentials=creds)

file_id = get_file_id(service, filename)

if file_id:
    print(f"File ID: {file_id}")

df = getCSVtoPandas(filename, creds, skiprows=3)

print(df.head())