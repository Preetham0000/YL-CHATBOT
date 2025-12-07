"""Vector store and embedding helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .data_models import DocumentChunk, RetrievalResult, ImageRecord


class EmbeddingClient:
    """Thin wrapper around SentenceTransformer with lazy loading."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


class VectorStore:
    """Persisted FAISS index + JSON metadata."""

    def __init__(self, index_path: Path, metadata_path: Path, embedder: EmbeddingClient) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedder = embedder
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: List[DocumentChunk] = []

    @staticmethod
    def _chunk_to_dict(chunk: DocumentChunk) -> dict:
        return {
            "id": chunk.id,
            "content": chunk.content,
            "source_file": chunk.source_file,
            "page_number": chunk.page_number,
            "section": chunk.section,
            "images": [
                {
                    "id": image.id,
                    "source_file": image.source_file,
                    "page_number": image.page_number,
                    "path": str(image.path),
                    "caption": image.caption,
                }
                for image in chunk.images
            ],
        }

    @staticmethod
    def _dict_to_chunk(payload: dict) -> DocumentChunk:
        images = [
            ImageRecord(
                id=image["id"],
                source_file=image["source_file"],
                page_number=image.get("page_number"),
                path=Path(image["path"]),
                caption=image.get("caption"),
            )
            for image in payload.get("images", [])
        ]
        return DocumentChunk(
            id=payload["id"],
            content=payload["content"],
            source_file=payload["source_file"],
            page_number=payload.get("page_number"),
            images=images,
            section=payload.get("section"),
        )

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("No chunks provided for indexing.")

        vectors = self.embedder.encode([chunk.content for chunk in chunks])
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.metadata = chunks

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("Index is empty; build it before saving.")
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            "\n".join(json.dumps(self._chunk_to_dict(chunk)) for chunk in self.metadata),
            encoding="utf-8",
        )

    def load(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Vector store files are missing. Run ingestion first.")
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = [
            self._dict_to_chunk(json.loads(line))
            for line in self.metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def search(self, query: str, top_k: int) -> List[RetrievalResult]:
        if self.index is None:
            raise RuntimeError("Index is not loaded.")
        query_vector = self.embedder.encode([query])
        scores, indices = self.index.search(query_vector, top_k)
        hits: List[RetrievalResult] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            hits.append(RetrievalResult(chunk=self.metadata[idx], score=float(score)))
        return hits
