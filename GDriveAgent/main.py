# -------- LangGraph Imports -------- 
from langchain_openai import ChatOpenAI
from langchain import hub
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
# --------- Utility Imports ---------- 
from utils.rag_funcs import create_vector_store
from utils.state import State
# -------- Load env variables --------
from dotenv import load_dotenv
import os
dotenv_path = os.getenv(".env")
load_dotenv(dotenv_path)


vector_store = create_vector_store()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# LangGraph RAG Agent Workflow
workflow = StateGraph(State)

workflow.add_sequence([retrieve, generate])

workflow.add_edge(START, "retrieve")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

def main():
    try:
        print("Welcome to RAG with Google Drive!")
        print("Type 'exit', 'quit', or 'q' to end the conversation.")
        while True:
            user_input = input("You: ").strip()

            # Exit condition
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            # Run the agent with the user's input  
            response = graph.invoke({"question": user_input}, config={"configurable": {"thread_id": "2"}})

            # Print answer
            print("Agent: ", response["answer"])
    except Exception as e:
        print("Error: ", e)  
    
if __name__ == '__main__':
    main()

# Run
# response = graph.invoke({"question": "How have ayurvedic practicioners preserved ayurvedic learning?"})
# print(response["answer"])