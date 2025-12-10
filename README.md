# YL RAG Chatbot

A retrieval-augmented chatbot tailored for Yantra Live brochures. The agent answers **only** from the curated knowledge base and returns brochure images whenever they support a reply. If the answer is missing, it responds with `I don't have information about that in my knowledge base`.

## Features
- PDF + text ingestion with PyMuPDF-based image extraction.
- Sentence-transformer embeddings and FAISS vector store persisted on disk.
- Guardrails that filter low-confidence hits before calling the LLM.
- Works with either OpenAI (Chat Completions) or Google Gemini models.
- CLI chatbot that surfaces referenced images (local file paths) alongside answers.
- Configurable through `.env` (OpenAI key, model, chunk sizes, etc.).

## Setup

1. **Install dependencies**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. **Add data**
   - Place PDFs and text notes under `data/raw/`.
   - Example file `data/raw/sample_catalog.txt` is provided as a template.

3. **Configure secrets**
   - Create a `.env` file to point at your preferred LLM provider:

```
# Pick openai or gemini
LLM_PROVIDER=openai

# OpenAI settings
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini

# Gemini settings (used when LLM_PROVIDER=gemini)
# Keys issued via Google AI Studio expect the "-latest" suffix
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-1.5-flash-latest
```
   - Only the variables for the selected provider are required; others can stay unset.

## Build the knowledge base

```powershell
python scripts/ingest.py
```

This command processes all brochures, extracts text/images, and writes:
- `data/processed/index.faiss` – FAISS vector index.
- `data/processed/chunks.jsonl` – chunk metadata with citations.
- `data/processed/images_manifest.json` – catalog of extracted images.

## Chat with the agent

```powershell
python scripts/chat.py
```

Use `--question "..."` for one-off answers. The CLI prints any referenced images so you can preview them or forward them through your UI layer.

## Web UI

1. Start the FastAPI server (serves both the API and static frontend):

```powershell
python scripts/server.py
```

2. Open `http://localhost:8000` in your browser. The left panel is a chat timeline; the right panel shows the images attached to the most recent answer. All responses come from your ingested brochures, and image thumbnails are loaded from `data/processed/images/**`.

## Extending / Integrating
- Wrap `YantraRAGAgent.answer()` inside your preferred channel (web chat, Teams, etc.).
- Host `data/processed/images/**` on a CDN or blob store if the client requires remote URLs; the agent already returns the file paths, so simply map them to URLs downstream.
- Adjust guardrails via `Settings.min_similarity` and `Settings.top_k` to tune precision vs. recall.

## Troubleshooting
- **No chunks generated**: confirm PDFs contain extractable text (not pure scans) or run OCR before ingestion.
- **Chatbot says knowledge missing**: verify ingestion ran after adding the latest brochures and that the FAISS index exists.
- **Model downloads**: the first ingestion run downloads the `all-MiniLM-L6-v2` embedding model; ensure outbound internet access during setup.
