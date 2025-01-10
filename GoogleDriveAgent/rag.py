from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_community import GoogleDriveLoader
from langchain_googledrive.document_loaders import GoogleDriveLoader

from dotenv import load_dotenv
import os
dotenv_path = os.getenv(".env")
load_dotenv(dotenv_path)

loader = GoogleDriveLoader(
    folder_id="13LE8SEQNHfmfHxTaNEA9hzqzPeODcsKB",
    credentials_path="credentials.json",
    token_path="token.json",
    load_auth=True,
    # Optional: configure whether to load authorized identities for each Document.
)

docs = loader.load()
print(len(docs))

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