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
