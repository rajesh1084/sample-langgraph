# LangGraph 101: Basic example with Agents.
import os, getpass
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import Graph, START, END


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
      
# Need to set GOOGLE_API_KEY as an environment variable
_set_env("GOOGLE_API_KEY")

# define a simple Agent/LLM Node
def llm_node(input_str: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemma-3n-e4b-it", temperature=0.2)
    response = llm.invoke(input_str)
    return response.content


# Create a new Graph
workflow = Graph()
# Add the nodes
workflow.add_node("llm_node", llm_node)
# Add the Edges
workflow.add_edge(START, "llm_node")
workflow.add_edge("llm_node", END)

# Compile the workflow
app = workflow.compile()

# Run the workflow
print(app.invoke("Hello, LangGraph! This is a test of the LLM node."))
