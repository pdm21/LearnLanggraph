from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
dotenv_path = os.getenv("../../.env")
load_dotenv(dotenv_path)

tools = [TavilySearchResults(max_results=1)]
