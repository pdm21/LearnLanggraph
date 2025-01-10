from langchain_google_community import GoogleDriveLoader
from langchain_googledrive.document_loaders import GoogleDriveLoader

loader = GoogleDriveLoader(
    folder_id="13LE8SEQNHfmfHxTaNEA9hzqzPeODcsKB",
    credentials_path="credentials.json",
    token_path="token.json",
    load_auth=True,
    # Optional: configure whether to load authorized identities for each Document.
)

docs = loader.load()
print(len(docs))

# docs:
# https://python.langchain.com/docs/integrations/document_loaders/google_drive/