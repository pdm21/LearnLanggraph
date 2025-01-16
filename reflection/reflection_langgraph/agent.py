from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from typing import List, Sequence
from langgraph.graph import START, END, StateGraph

model = ChatOpenAI(temperature=0, streaming=True)
# define the nodes
"""
    - get model
    - call model
    - should_continue
    - reflect
"""
# define the tools
# define the LLM
# bind the tools
# define the graph (workflow) and compile
