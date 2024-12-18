'''
Various Google Drive API functions for getting and uploading files.
Credential files should be in the parent directory of this file.
'''
import os.path
import pandas as pd
import io

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# See https://developers.google.com/drive/api/guides/api-specific-auth
# If modifying these scopes, delete the file token.json.
# The drive scope should allow downloads and uploads.
SCOPES = ["https://www.googleapis.com/auth/drive"]

def get_credentials() -> Credentials:
    """Gets the credentials for the Google Drive API.
    If no valid credentials are found, the user is prompted to log in.
    Returns:
        The obtained credentials.
    """
    creds = None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    token_path = os.path.join(parent_dir, "token.json")
    credentials_path = os.path.join(parent_dir, "credentials.json")
  
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES
            )
            creds = flow.run_local_server(port=0)

    # Save the credentials for the next run
    with open(token_path, "w") as token:
        token.write(creds.to_json())

    return creds

def get_folder_id(service, folder_name: str, parent_folder_id: str = None) -> str:
    """
    Gets the folder ID of a folder in Google Drive. In Google Drive, folders are
    treated as files with a specific MIME type. To find a folder ID, the parent folder
    ID is needed to narrow down the search.
    Args:
        service: The Google Drive API service instance.
        folder_name: The name of the folder.
        parent_folder_id: The ID of the parent folder to search within (optional).
    Returns:
        The folder ID of the folder.
    """
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    if parent_folder_id:
        query += f" and '{parent_folder_id}' in parents"
    
    results = service.files().list(
        q=query,
        spaces='drive',
        fields="files(id, name)").execute()
    items = results.get('files', [])
    if not items:
        print('No folders found.')
        return None
    else:
        # Assuming the first match is the desired folder
        return items[0]['id']

def get_file_id(service, file_path: str) -> str:
    """
    Gets the file ID of a file in Google Drive.
    Current algorithm assumes folder/file is unique.
    
    Args:
        file_path: The path with filename to the file in Google Drive.
    Returns:
        The file ID of the file.
    """
    # Break down the file path into folder names and the file name
    folders = file_path.split("/")
    file_name = folders.pop()

    # Get parent folder ID, if any
    if folders:
        parent_folder_id = get_folder_id(service, folders[-1])
        query = f"name='{file_name}' and '{parent_folder_id}' in parents"
    else:
        query = f"name='{file_name}'"
    

    results = service.files().list(
        q=query,
        spaces='drive',
        fields="files(id, name)").execute()
    items = results.get('files', [])
    if not items:
        print('No files found.')
        return None
    else:
        # Assuming the first match is the desired file
        if len(items) > 1:
            print("Warning: Multiple files found. Returning the first match.")
        return items[0]['id']

def getCSVtoPandas(file_path: str, creds: Credentials, skiprows=None) -> pd.DataFrame:
    '''
        Reads a CSV file from Google Drive and returns it as a Pandas DataFrame.
            Args:
                file_path: The path with filename to the CSV file in Google Drive.
    '''
    try:
        # Set up the API service
        service = build("drive", "v3", credentials=creds)

        # Get the file id of the file specified in the file_path
        file_id = get_file_id(service, file_path)
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}.")

    except HttpError as error:
        print(f"An error occurred: {error}")

    # Read the CSV file into a Pandas DataFrame
    fh.seek(0)
    if skiprows is None:
        df = pd.read_csv(fh)
    else:
        df = pd.read_csv(fh, skiprows=skiprows)

    return df

