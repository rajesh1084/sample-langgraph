# LangGraph 101 - basic example with nodes and edges

from langgraph.graph import Graph

# node1 takes an input string and concatenates with another string and returns
def node1(str):
    return str + "\nI've reached Node1.\n"

# node2 takes an input string and concatenates with another string and returns
def node2(str):
    return str + "And now at Node2."

# Create a new Graph
workflow = Graph()

# Add the nodes
workflow.add_node("node_1", node1)
workflow.add_node("node_2", node2)

# Add the Edges
workflow.add_edge("node_1", "node_2")
workflow.set_entry_point("node_1")
workflow.set_finish_point("node_2")

# Run the workflow
app = workflow.compile()

print(app.invoke("Hello"))
