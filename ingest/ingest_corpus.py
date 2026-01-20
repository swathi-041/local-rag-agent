import os
from ingest.loader import load_pdf, load_json
from ingest.chunker import chunk_text
from ingest.embed import EmbeddingModel
from vectorstore.store import VectorStore

DATA_DIR = "data/uploads"

def ingest():
    embedder = EmbeddingModel()
    store = VectorStore(dim=384)

    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)

        if file.endswith(".pdf"):
            text = load_pdf(path)
        elif file.endswith(".json"):
            text = load_json(path)
        else:
            continue

        chunks = chunk_text(text)
        embeddings = embedder.embed_texts(chunks)
        store.add(embeddings, chunks)

        print(f"✅ Ingested {file} ({len(chunks)} chunks)")

if __name__ == "__main__":
    ingest()
