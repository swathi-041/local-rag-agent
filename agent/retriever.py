from ingest.embed import EmbeddingModel
from vectorstore.store import VectorStore

class Retriever:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.store = VectorStore(dim=384)

    def retrieve(self, query, k=5):
        query_embedding = self.embedder.embed_query(query)
        return self.store.search(query_embedding, k)
