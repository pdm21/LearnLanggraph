from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from agent.utils.nodes import call_model, should_continue, tool_node
from agent.utils.state import AgentState
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

# Define the config
class GraphConfig(TypedDict):
    # must be openai (have not added support for other models)
    model_name: Literal["openai"]

# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Define the two nodes we will cycle between
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
    try:
        print("Welcome to the Agent conversation!")
        print("Type 'exit', 'quit', or 'q' to end the conversation.")
        thread_id = 0
        while True:
            # Get input from the user
            user_input = input("You: ").strip()

            # Exit condition
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            # Run the agent with the user's input
            inputs = {"messages": [HumanMessage(content=user_input)]}  # HumanMessage object
            results = graph.invoke(inputs, config={"configurable": {"thread_id": "1"}})
            # results = graph.invoke(inputs, config={"configurable": {"thread_id": f"{thread_id}"}})

            # Check if `results` contains the expected data
            if isinstance(results, dict) and 'messages' in results:
                messages = results['messages']  # Extract messages
                for message in messages:
                    if hasattr(message, 'content') and isinstance(message, AIMessage):
                        print(f"Assistant: {message.content}")
            else:
                print("Assistant: I couldn't process the response.")
            thread_id += 1
    
    except Exception as e:
        print("There was an error in the process. More info:", e)

if __name__ == '__main__':
    main()