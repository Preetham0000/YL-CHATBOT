from yantra_rag.config import load_settings
from yantra_rag.vector_store import VectorStore, EmbeddingClient

settings = load_settings()
embedder = EmbeddingClient(settings.embedding_model)
store = VectorStore(settings.faiss_index_path, settings.chunk_store_path, embedder)
store.load()

question = "What is the category of Sany SY215"
hits = store.search(question, top_k=5)

print(f"Found {len(hits)} hits for: {question}")
for i, hit in enumerate(hits):
    print(f"\n--- Hit {i+1} (Score: {hit.score:.4f}) ---")
    print(f"Source: {hit.chunk.source_file}")
    print(f"Content snippet: {hit.chunk.content[:200]}...")
