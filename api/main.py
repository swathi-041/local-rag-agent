
# PROJECT_ID = "upbeat-repeater-477110-q6"
# LOCATION = "us-east1" 

from agent.agent import answer
from agent.llm import LLM
from ingest.embed import EmbeddingModel
from vectorstore.store import VectorStore
PROJECT_ID = "upbeat-repeater-477110-q6"
LOCATION = "us-east1" 

llm = LLM(PROJECT_ID, LOCATION)
embedder = EmbeddingModel()
store = VectorStore(dim=384)

print("🧠 Vertex-AI-like RAG Agent (Gemini)")
print("Type 'exit' to quit\n")

while True:
    query = input("User: ")
    if query.lower() == "exit":
        break

    response = answer(
        query=query,
        embed_fn=embedder.embed_query,
        llm=llm,
        store=store,
    )
    print("\nAgent:", response, "\n")
