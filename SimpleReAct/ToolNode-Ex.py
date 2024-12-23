# Load .env variables (API Keys)
from dotenv import load_dotenv
_ = load_dotenv()

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from typing import Literal
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState

@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."


@tool
def get_coolest_cities():
    """Get a list of coolest cities"""
    return "nyc, sf"

tools = [get_weather, get_coolest_cities]
tool_node = ToolNode(tools)

message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"location": "sf"},
            "id": "tool_call_id",
            "type": "tool_call",
        }
    ],
)

tool_node.invoke({"messages": [message_with_single_tool_call]})

model_with_tools = ChatOpenAI(
    model="gpt-3.5-turbo-1106", temperature=0
).bind_tools(tools)

model_with_tools.invoke("what's the weather in sf?").tool_calls
print(tool_node.invoke({"messages": [model_with_tools.invoke("what's the weather in sf?")]}))
