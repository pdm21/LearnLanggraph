from dotenv import load_dotenv
import os
dotenv_path = os.getenv("../.env")
load_dotenv(dotenv_path)

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

tools = [TavilySearchResults(max_results=1)]
tool_node = ToolNode(tools)
model = ChatOpenAI(temperature=0, streaming=True)

# takes in langchain functions and converts to the format that openai functions expect
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
import json
from langgraph.graph import StateGraph, START, END

functions = [convert_to_openai_function(t) for t in tools]
model = model.bind_tools(functions)
# print(tool_node.invoke({"messages": [model.invoke("what's the weather in sf?")]}))

# adding onto the state every time (adding messages, not overwriting)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Defining the nodes:

def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else: 
        return "continue"

# AI processes messages, overwrites with a response (AI Message)
def call_model(state):
    messages = state['messages']
    # sent to AI model for processing
    response = model.invoke(messages)
    return {'messages': [response]}


# define the graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)

# always want to come back to agent after an action
workflow.add_edge("action", "agent")
app = workflow.compile()

# ADDED MEMORY CHECKPOINT 
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string(":memory:") as memory:
  messages = [HumanMessage(content="I attend Hamilton College")]
  thread = {"configurable": {"thread_id": "1"}}
  
  for event in app.stream({"messages": messages}, thread):
      for v in event.values():
          print(v['messages'])

  messages = [HumanMessage(content="What school do i attend?")]
  thread = {"configurable": {"thread_id": "1"}}
  
  for event in app.stream({"messages": messages}, thread):
      for v in event.values():
          print(v['messages'])
