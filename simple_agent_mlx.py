# LangGraph 101: Basic example with local MLX inference.
# Local MLX inference.

# Prerequisites:
# Install MLX and MLX-LM for language models
# pip install mlx mlx-lm

# Optional: Install additional MLX packages
# pip install mlx-transformers mlx-whisper

from mlx_lm import load, generate
from langgraph.graph import Graph, START, END

# Global variables to store the model and tokenizer
model = None
tokenizer = None


def load_model():
    """Load the MLX model and tokenizer once"""
    global model, tokenizer
    if model is None:
        print("Loading model...")
        model, tokenizer = load("mlx-community/QwQ-32B-4bit") # Change this as per your need.
        print("Model loaded successfully!")


# define a simple Agent/LLM Node using MLX
def llm_node(input_str: str) -> str:
    load_model()  # Ensure model is loaded

    # Generate response using MLX
    # response = generate(model, tokenizer, prompt=input_str, max_tokens=512, temp=0.2)
    response = generate(model, tokenizer, prompt=input_str)

    return response


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
print(app.invoke("Hello, LangGraph! This is a test of the local MLX node."))
