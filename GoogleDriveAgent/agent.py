from langchain_google_community import GoogleDriveLoader

loader = GoogleDriveLoader(
    folder_id="13LE8SEQNHfmfHxTaNEA9hzqzPeODcsKB",
    credentials_path="./.credentials/credentials.json",
    token_path="./.credentials/token.json",
    file_types=["document", "sheet"],
    recursive=True,
)

docs = loader.load()
print("len docs: ", len(docs))


# docs:
# https://python.langchain.com/docs/integrations/document_loaders/google_drive/