from langchain_openai import ChatOpenAI
from langchain_google_community import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata  
from dotenv import load_dotenv
import os

from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

dotenv_path = os.getenv(".env")
load_dotenv(dotenv_path)

# Load documents from Google Drive
loader = GoogleDriveLoader(
    folder_id="13LE8SEQNHfmfHxTaNEA9hzqzPeODcsKB",
    credentials_path="credentials.json",
    token_path="token.json",
    load_auth=True,
)

docs = loader.load()

# Filter out complex metadata from the documents
filtered_docs = filter_complex_metadata(docs) 
print(len(filtered_docs))

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)


# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=0
)

all_splits = text_splitter.split_documents(filtered_docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


response = graph.invoke({"question": "How have ayurvedic practicioners preserved ayurvedic learning?"})
print(response["answer"])