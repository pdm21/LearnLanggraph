from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Define the required Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

def auth():        
    try:
        # Initialize the OAuth flow
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)

        # Create a service object for Google Drive API
        drive_service = build('drive', 'v3', credentials=creds)

        # Test: List the files in your Google Drive
        results = drive_service.files().list(pageSize=10, fields="files(id, name)").execute()
        items = results.get('files', [])
        print("################################################################################")
        print("Success!")
    except Exception as e:
        print("Error authorizing user. More info: ", e)