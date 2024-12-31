from dotenv import load_dotenv
import os
import asyncio
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

# Load environment variables
dotenv_path = os.getenv("../.env")
load_dotenv(dotenv_path)

# Configure tools and model
tools = [TavilySearchResults(max_results=1)]
tool_node = ToolNode(tools)
model = ChatOpenAI(temperature=0, streaming=True)

# Convert tools to OpenAI functions
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
import json
from langgraph.graph import StateGraph, START, END

functions = [convert_to_openai_function(t) for t in tools]
model = model.bind_tools(functions)

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Function to decide the next action
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else: 
        return "continue"

# Function to call the AI model
def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)
    return {'messages': [response]}

# Define the workflow
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

workflow.add_edge("action", "agent")

# Database connection settings
conninfo = (
    f"postgresql+psycopg2://{os.getenv('PSQL_USERNAME')}:{os.getenv('PSQL_PASSWORD')}"
    f"@{os.getenv('PSQL_HOST')}:{os.getenv('PSQL_PORT')}/{os.getenv('PSQL_DATABASE')}"
    f"?sslmode={os.getenv('PSQL_SSLMODE')}"
)

async_pool = AsyncConnectionPool(
    conninfo=conninfo,
    max_size=20,
    kwargs={"autocommit": True}
)

# Function to save messages to memory
async def save_message_to_memory(memory, messages):
    for message in messages:
        if isinstance(message, HumanMessage):
            await memory.save_context({"input": message.content}, {"output": ""})
        elif isinstance(message, AIMessage):
            await memory.save_context({"input": "", "output": message.content})

# Main async function
async def main():
    conninfo = (
        f"postgresql+psycopg2://{os.getenv('PSQL_USERNAME')}:{os.getenv('PSQL_PASSWORD')}"
        f"@{os.getenv('PSQL_HOST')}:{os.getenv('PSQL_PORT')}/{os.getenv('PSQL_DATABASE')}?sslmode={os.getenv('PSQL_SSLMODE')}"
    )

    async with AsyncConnectionPool(conninfo=conninfo, max_size=20) as async_pool:
        async with async_pool.connection() as conn:
            memory = AsyncPostgresSaver(conn)
            await memory.setup()

            app_with_memory = workflow.compile(checkpointer=memory)

            # User query
            inputs = {"messages": [HumanMessage(content="What is the weather in the capital of Greece?")]}
            results = await app_with_memory.invoke(inputs)

            # Save conversation to memory
            await save_message_to_memory(memory, results["messages"])

            # Pretty print
            for msg in results["messages"]:
                if isinstance(msg, HumanMessage):
                    print(f"[Human]: {msg.content}")
                elif isinstance(msg, AIMessage):
                    print(f"[AI]: {msg.content}")

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())

# Pretty print function
# def org_output(messages):
#     for msg in messages:
#         if isinstance(msg, HumanMessage):
#             print("\n[Human]:", msg.content)
#         elif msg.additional_kwargs.get("tool_calls"):
#             print("\n[AI - Tool Call]:")
#             print(json.dumps(msg.additional_kwargs, indent=2)) 
#         elif isinstance(msg, ToolMessage):
#             print("\n[Tool Response]:")
#             print(json.dumps(msg.artifact, indent=2))  
#         elif isinstance(msg, AIMessage):
#             print("\n[AI]:", msg.content)

