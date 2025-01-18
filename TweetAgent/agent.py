from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from utils.nodes import call_model, should_continue, tool_node
from utils.state import AgentState
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

class GraphConfig(TypedDict):
    model_name: Literal["openai"]

workflow = StateGraph(AgentState, config_schema=GraphConfig)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue, 
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")

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
        messages = results['messages']  # Extract tweets
        for message in messages:
            if hasattr(message, 'content') and isinstance(message, AIMessage):
                print(f"Assistant: {message.content}")  
    print(graph.get_state(config).values["messages"]) 

if __name__ == '__main__':
    main()