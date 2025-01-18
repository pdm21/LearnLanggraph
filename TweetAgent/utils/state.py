from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    research: Annotated[Sequence[BaseMessage], add_messages]
    tweets: Annotated[Sequence[BaseMessage], add_messages]
    status: Annotated[Sequence[int], add_messages]