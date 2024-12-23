# Load .env variables (API Keys)
from dotenv import load_dotenv
_ = load_dotenv()
# -----------------------------------
# Create the agent:

from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]

prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)
agent_runnable = create_openai_functions_agent(llm, tools, prompt)
# -----------------------------------
# Define the graph state:

from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # inputted string
    input: str
    # list of previous messages in conversation
    chat_history: list[BaseMessage]
    # outcome of a given agent call
    # needs "None" as a valid type, since this is what it starts as
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # list of actions and corresponding observations, added on to previous
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

# -----------------------------------
# Define the nodes:

from langgraph.prebuilt.tool_executor import ToolExecutor

# helper class (imported): takes in agent action, calls that tool, awaits result
tool_executor = ToolExecutor(tools)

# define the agent:
    # run the agent, store outcome
    # override the agent_outcome state
def run_agent(data):
    agent_outcome = agent_runnable.invoke(data)
    return {"agent_outcome": agent_outcome}

# define the function to execute tools
    # get most recent agent action from the agent_outcome state var
def execute_tools(data):
    agent_action = data['agent_outcome']
    output = tool_executor.invoke(agent_action)
    return {"intermediate_steps": [(agent_action, str(output))]}

# logic determines which conditional edge to follow (what next?)
def should_continue(data):
    # if agent outcome = AgentFinish, return 'exit' string
    if isinstance(data['agent_outcome'], AgentFinish):
        return "end"
    # otherwise, AgentAction is returned
    else:
        return "continue"
    
# -----------------------------------
# Define the graph:

from langgraph.graph import END, StateGraph

workflow = StateGraph(AgentState)

workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    # start node
    "agent",
    # function that determines what node is called next
    should_continue,
    # mapping of strings (think case statements)
    {
        "continue": "action",
        "end": END
    }
)

workflow.add_edge("action", "agent")
app = workflow.compile()

# run it
inputs = {"input": "what is the weather in sf", "chat_history": []}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("-----")