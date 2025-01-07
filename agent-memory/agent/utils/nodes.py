from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from agent.utils.tools import tools
from langgraph.prebuilt import ToolNode

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    try:
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    except:
        raise ValueError(f"Issue importing model: {model_name}, please check API Key")

    model = model.bind_tools(tools)
    return model

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"
    
system_prompt = """You are a helpful assistant. You answer whatever questions I may have to the best of your ability. Carefully answer the query, come up with a response, and then revisit the question to assess how well you answered the query. Answer every query in this way:
1) If you can answer the query and do not require other information, then answer it. 2) If you need external information to answer the query, then use the set of tools you have been provdied with to do so. 3) If you are asked to reference earlier parts of the conversation, then look through the previous messages to get your answer."""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
tool_node = ToolNode(tools)