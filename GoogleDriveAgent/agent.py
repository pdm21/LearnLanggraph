from langchain_googledrive.document_loaders import GoogleDriveLoader
import os

# Set GOOGLE_APPLICATION_CREDENTIALS to point to the credentials.json file in .credentials within the current directory
current_dir = os.path.abspath(os.path.dirname(__file__))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(current_dir, ".credentials", "credentials.json")

# Ensure the credentials file exists
if not os.path.exists(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]):
    raise FileNotFoundError(f"Credentials file not found: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")

# Set the folder ID to "root" or a specific folder you want to query
folder_id = "root"

# Initialize the Google Drive Loader
loader = GoogleDriveLoader(
    gdrive_api_file=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
    folder_id=folder_id,
    recursive=False,  # Set to True to load files from subfolders
    template="gdrive-query",  # Default template to use
    query="pandelis",  # The query to search for
    num_results=2,  # Maximum number of files to load
    supportsAllDrives=False,  # Set to True if using shared drives
)

for doc in loader.load():
        print("---")
        print(doc.page_content.strip()[:60] + "...")