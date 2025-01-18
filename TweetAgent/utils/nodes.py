from functools import lru_cache
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langgraph.graph import END

from utils.tools import tools
from utils.prompts import system_prompt, regeneration_prompt

@lru_cache(maxsize=4)
def _get_model():
   model = ChatOpenAI(temperature=0, model_name="gpt-4o")
   model = model.bind_tools(tools)
   return model

system_prompt = system_prompt

# Define the function that calls the model
def call_model(state):
   messages = state["messages"]
   messages = [{"role": "system", "content": system_prompt}] + messages
   model = _get_model()
   response = model.invoke(messages)
   
   if not response.tool_calls:
      next_node = END
   # Otherwise if there is, we continue
   else:
      next_node = "tools"
   return Command(
       goto=next_node,
       update={"messages": [response]}
   )     
   

# Define the function to execute tools
tool_node = ToolNode(tools)