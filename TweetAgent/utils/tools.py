# Import requests for Perplexity API call
import requests
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
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from utils.state import AgentState
from langchain_core.messages import BaseMessage
from langchain_core.tools import InjectedToolCallId
from typing import TypedDict, Annotated, Sequence

@tool
def perplexity_research(user_message: str, tool_call_id: Annotated[str, InjectedToolCallId]):
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
        "messages": 
        [
            {"role": "system", "content": f"Be price and accurate. Only consider results posted today, which is {date.today()}"},
            {"role": "user", "content": f"Current news about {user_message}"}
        ],

        "temperature": 0.2,
        "max_tokens": 200,
        "return_images": False,
        "search_recency_filter": "month"
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    response = response.json()
    return Command(
        update={
            # update the research citations
            "research_citations": response['citations'],
            # update the research content
            "research_content": response['choices'][0]['message']['content'],
            "messages": [
                ToolMessage("Successfully conducted research", tool_call_id=tool_call_id)  
            ]
        }
    )

@tool
def get_research_content():
    """Get the research content from state"""
    research_content = state['research_content']
    return research_content
    # return state["research_content"]


from datetime import date
@tool
def get_date():
    """Returns today's date"""
    today = date.today()
    return today

tools = [perplexity_research, get_date]
