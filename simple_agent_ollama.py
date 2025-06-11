# LangGraph 101: Basic example with local Ollama inference.

# Prerequisites:
# Install Ollama: brew install ollama
# Install ollama Python package: pip install ollama
# Start Ollama service: ollama serve
# Pull a model: ollama pull llama3.2:latest

import ollama
from langgraph.graph import Graph, START, END


# Configuration
OLLAMA_MODEL = "llama3.2:latest"  # Change this to your preferred model


def check_ollama_model():
    """Check if the specified model is available locally"""
    try:
        # Try a simple test generation instead of listing models
        test_response = ollama.generate(
            model=OLLAMA_MODEL, prompt="test", options={"num_predict": 1}
        )
        return True
    except Exception as e:
        print(f"Model {OLLAMA_MODEL} not available: {e}")
        return False


# define a simple Agent/LLM Node using Ollama
def llm_node(input_str: str) -> str:
    if not check_ollama_model():
        return "Error: Ollama model not available"

    try:
        # Generate response using Ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": input_str,
                }
            ],
            options={
                "temperature": 0.2,
                "num_predict": 1024,  # max tokens
            },
        )

        return response["message"]["content"]

    except Exception as e:
        return f"Error generating response: {e}"


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
print(app.invoke("Hello, LangGraph! This is a test of the local Ollama node."))
