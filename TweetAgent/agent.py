from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from utils.nodes import call_model, tool_node
from utils.state import AgentState
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_edge("tools", "agent")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

def main():
    # Get input from the user
    user_input = input("You: ").strip()
    
    # Run the agent with the user's input
    inputs = {"messages": [HumanMessage(content=user_input)]} 
    config = {"configurable": {"thread_id": "1"}}
    results = graph.invoke(inputs, config=config)

    # Check if `results` contains the expected data
    if isinstance(results, dict) and 'messages' in results:
        messages = results['messages'] 
        for message in messages:
            if hasattr(message, 'content') and isinstance(message, ToolMessage):
                print(f"ToolCall: {message.content}")
            if hasattr(message, 'content') and isinstance(message, AIMessage):
                print(f"Assistant: {message.content}")  

if __name__ == '__main__':
    main()


# from langchain_openai import ChatOpenAI
# from utils.tools import tools
# from langgraph.prebuilt import create_react_agent

# def print_stream(stream):
#     for s in stream:
#         message = s["messages"][-1]
#         if isinstance(message, tuple):
#             print(message)
#         else:
#             message.pretty_print()

# model = ChatOpenAI(temperature=0, model_name="gpt-4o")
# graph2 = create_react_agent(model, tools=tools)
# inputs = {"messages": [("user", "Print the research content from AgentState")]}
# print("################################################################################")
# print_stream(graph2.stream(inputs, stream_mode="values"))
