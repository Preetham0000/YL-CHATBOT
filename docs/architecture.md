# Yantra Live RAG Chatbot Architecture

## Goals
- Provide a guardrailed customer-support copilot that answers exclusively from Yantra Live brochures/text files.
- Extract both textual content and machine imagery from PDFs so the agent can attach visuals alongside textual responses.
- Refuse to answer ("I don't have information about that in my knowledge base") whenever retrieval fails to surface grounded evidence.

## High-Level Flow
1. **Ingestion Pipeline**
   - Walks through `data/raw` for PDF/TXT files.
   - Uses PyMuPDF to extract page-wise text and embedded images from PDFs, persisting the images under `data/processed/images/<doc>/<page>_<img>.png`.
   - Normalizes plain-text files directly.
   - Chunks content with a sentence-aware splitter and tags each chunk with provenance metadata (doc name, page, related images, etc.).
   - Embeds every chunk via `sentence-transformers` and saves:
     - FAISS vector index (`data/processed/index.faiss`).
     - Chunk metadata store (`data/processed/chunks.jsonl`).
     - Image manifest with captions / OCR fallbacks (`data/processed/images_manifest.json`).

2. **Retrieval Layer**
   - Loads the FAISS index plus metadata table.
   - For each query, generates embedding, performs similarity search, and returns the top-k chunks + any referenced images.
   - Applies confidence guardrails (minimum similarity + minimum supporting tokens). If retrieval is empty or too weak, the agent responds with the fallback sentence.

3. **Response Generation**
   - Packs retrieved snippets into a templated prompt that instructs the LLM to strictly quote the knowledge base and to refuse hallucinations.
   - When chunks reference stored images, the agent surfaces the image paths (or pre-signed URLs if hosted) alongside the textual answer payload.
   - All responses include a `citations` field identifying the originating brochure file/page for auditability.

4. **Interfaces**
   - CLI chatbot (`scripts/chat.py`) for local testing.
   - Modular `YantraRAGAgent` class so the same backend can power a web UI, Teams bot, or other channel later.
   - FastAPI web surface (`scripts/server.py`) serving both `/api/chat` and a lightweight static frontend under `/static`.

## Tech Stack
- **Python 3.10+**
- **PyMuPDF + Pillow** for PDF/text/image extraction.
- **SentenceTransformers + FAISS** for embeddings and nearest-neighbor search.
- **OpenAI Chat Completions** (configurable) for generation with defensive prompt template.
- **Pydantic** for strongly-typed metadata and configs.

## Key Guardrails
- Retrieval confidence thresholds and empty-result fallback message.
- Prompt instructions explicitly forbidding unsupported answers.
- Pluggable LLM layer (OpenAI Chat Completions or Gemini) selected via config.
- Optional `--dry-run` ingestion mode for metadata inspection without re-embedding.

## Directory Layout
```
├─ data/
│  ├─ raw/                # user-provided PDFs / text notes
│  └─ processed/          # generated assets: embeddings, metadata, images
├─ docs/
│  └─ architecture.md
├─ scripts/
│  ├─ ingest.py           # CLI for building the knowledge base
│  └─ chat.py             # CLI chatbot harness
├─ yantra_rag/
│  ├─ config.py
│  ├─ data_models.py
│  ├─ pdf_utils.py
│  ├─ text_splitter.py
│  ├─ vector_store.py
│  └─ rag_agent.py
├─ requirements.txt
└─ README.md
```

## Next Steps
1. Scaffold the package + scripts per layout.
2. Implement ingestion utilities with thorough logging.
3. Add CLI harnesses and documentation (README) describing setup + run commands.
