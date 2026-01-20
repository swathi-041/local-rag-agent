import faiss
import numpy as np
import pickle
import os

class VectorStore:
    def __init__(self, dim, path="vectorstore/index"):
        self.dim = dim
        self.path = path
        self.texts = []

        if os.path.exists(f"{path}.faiss"):
            self.index = faiss.read_index(f"{path}.faiss")
            with open(f"{path}.pkl", "rb") as f:
                self.texts = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings).astype("float32"))
        self.texts.extend(texts)
        self._save()

    def search(self, query_embedding, k=5):
        # Safety guard (Vertex AI never crashes)
        if self.index.ntotal == 0 or len(self.texts) == 0:
            return []

        D, I = self.index.search(
            np.array([query_embedding]).astype("float32"), k
        )

        results = []
        for idx in I[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])

        return results

    def _save(self):
        faiss.write_index(self.index, f"{self.path}.faiss")
        with open(f"{self.path}.pkl", "wb") as f:
            pickle.dump(self.texts, f)
