# Import requests for Perplexity API call
import requests
# Import custom tools
from langchain_core.tools import tool
# Load API Keys from .env
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
os.environ.get('OPENAI_API_KEY')
os.environ.get('PERPLEXITY_API_KEY')
print("OpenAI API Key Loaded", os.environ.get('OPENAI_API_KEY') is not None)
print("Perplexity API Key Loaded", os.environ.get('PERPLEXITY_API_KEY') is not None)
perplexity_api_key = os.environ.get('PERPLEXITY_API_KEY')
openai_api_key = os.environ.get('OPENAI_API_KEY')
################################################################################
from langgraph.types import Command
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from utils.state import AgentState
from langchain_openai import ChatOpenAI

@tool
def perplexity_research(user_message: str) -> dict:
    """Uses the athlete's name from the initial user query as its input.
    Calls the Perplexity API, with the following query "Current news about {athlete}", to research the athlete and gather the most recent news about them. 
    """
    # This information will be used to generate a tweet
    api_key = perplexity_api_key
    model = "llama-3.1-sonar-small-128k-online"

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"Current news about {user_message}"}],
        "temperature": 0.2,
        "max_tokens": 300
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    response = response.json()
    return response['choices'][0]['message']['content']

from datetime import date
@tool
def date_and_time():
    """Returns today's date"""
    today = date.today()
    return today

tools = [perplexity_research, date_and_time]


