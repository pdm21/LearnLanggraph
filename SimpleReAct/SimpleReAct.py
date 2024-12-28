from dotenv import load_dotenv
import os
dotenv_path = os.getenv("../.env")
load_dotenv(dotenv_path)

from typing import Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai.chat_models import ChatOpenAI

# ------------------------------------------------------------
# TOOLS:

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
    return "nyc, sf (ca)"

tools = [get_weather, get_coolest_cities]
tool_node = ToolNode(tools)
# ------------------------------------------------------------
# NODES
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}
# ------------------------------------------------------------
# GRAPH:

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

app = workflow.compile()

model_with_tools = ChatOpenAI(
    model="gpt-3.5-turbo-1106", temperature=0
).bind_tools(tools)

for chunk in app.stream(
    {"messages": [("human", "what's the weather in the coolest cities? Use the get_coolest_cities tool and stick to those cities.")]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()