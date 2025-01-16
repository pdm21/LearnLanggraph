import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
os.environ.get('OPENAI_API_KEY')
print("API Key Loaded", os.environ.get('OPENAI_API_KEY') is not None)

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
