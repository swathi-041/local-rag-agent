from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_texts(self, texts):
        # For documents (list[str])
        return self.model.encode(texts, show_progress_bar=False)

    def embed_query(self, query):
        # For single query (str)
        return self.model.encode([query])[0]
