# Function 1: 
# Load docs from GDrive folder
# ------------------------------------------------------------------------------
from langchain_google_community import GoogleDriveLoader
from langchain_googledrive.document_loaders import GoogleDriveLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
def load_gdrive():
    loader = GoogleDriveLoader(
        folder_id="13LE8SEQNHfmfHxTaNEA9hzqzPeODcsKB",
        credentials_path="credentials.json",
        token_path="token.json",
        load_auth=True,
    )
    docs = loader.load()
    filtered_docs = filter_complex_metadata(docs) 
    return filtered_docs

# Function 2: 
# Get OpenAI embeddings
# ------------------------------------------------------------------------------
from langchain_openai import OpenAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
def get_embeddings():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return embeddings

# Function 3: 
# Vector Store
# ------------------------------------------------------------------------------
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
def create_vector_store():
    """Create an in-memory vector store and index the document splits."""
    embeddings = get_embeddings()
    vector_store = InMemoryVectorStore(embeddings)
    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    filtered_docs = load_gdrive()
    all_splits = text_splitter.split_documents(filtered_docs)
    vector_store.add_documents(documents=all_splits)
    return vector_store

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Run all in one func()

def conc_vector_store():
    print("Loading documents from Google Drive...")
    loader = GoogleDriveLoader(
        folder_id="13LE8SEQNHfmfHxTaNEA9hzqzPeODcsKB",
        credentials_path="credentials.json",
        token_path="token.json",
        load_auth=True,
    )

    # Load and filter documents
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    filtered_docs = filter_complex_metadata(docs)
    print(f"Filtered {len(filtered_docs)} documents.")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(filtered_docs)
    print(f"Split documents into {len(splits)} chunks.")

    # Index chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(splits)
    print("Vector store created successfully!")

    return vector_store
