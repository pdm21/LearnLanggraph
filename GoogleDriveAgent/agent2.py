from langchain.chat_models.openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_google_community.drive import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_googledrive.document_loaders import GoogleDriveLoader
import os

# Set GOOGLE_APPLICATION_CREDENTIALS to point to the credentials.json file in .credentials within the current directory
current_dir = os.path.abspath(os.path.dirname(__file__))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(current_dir, ".credentials", "credentials.json")

# Ensure the credentials file exists
if not os.path.exists(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]):
    raise FileNotFoundError(f"Credentials file not found: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")

# Set the folder ID to "root" or a specific folder you want to query
# folder_id = "root"
folder_id = "13LE8SEQNHfmfHxTaNEA9hzqzPeODcsKB"

loader = GoogleDriveLoader(
    folder_id=folder_id,
    recursive=False
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )

texts = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever()

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

while True:
    query = input("> ")
    answer = qa.run(query)
    print(answer)