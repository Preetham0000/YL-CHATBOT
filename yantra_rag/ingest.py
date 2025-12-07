"""Data ingestion pipeline."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Tuple

from .config import Settings
from .data_models import DocumentChunk, ImageRecord
from .pdf_utils import extract_pdf_content, ensure_dir
from .text_splitter import split_text
from .vector_store import EmbeddingClient, VectorStore

LOGGER = logging.getLogger(__name__)

SECTION_HEADERS = {
    "PERFORMANCE DIMENSIONS",
    "EXCAVATOR PERFORMANCE",
    "LOADER PERFORMANCE",
    "STATIC DIMENSIONS",
    "ENGINE",
    "TRANSMISSION",
    "BRAKES",
    "STEERING",
    "HYDRAULICS",
    "OPTIONS",
    "SERVICE",
}

ROW_PATTERN = re.compile(r"^([A-Z][A-Z0-9\/]*)\s+(.*)")


def _detect_section_heading(line: str) -> str | None:
    normalized = re.sub(r"\s+", " ", line.strip().upper())
    if normalized in SECTION_HEADERS:
        return normalized
    return None


def _extract_section_blocks(text: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    current_name: str | None = None
    buffer: List[str] = []
    for raw_line in text.splitlines():
        heading = _detect_section_heading(raw_line)
        if heading:
            if current_name and buffer:
                sections.append((current_name, "\n".join(buffer).strip()))
            current_name = heading
            buffer = []
        elif current_name:
            buffer.append(raw_line)
    if current_name and buffer:
        sections.append((current_name, "\n".join(buffer).strip()))
    return [section for section in sections if section[1]]


def _format_section_block(name: str, block_text: str) -> str:
    lines = [line.strip() for line in block_text.splitlines() if line.strip()]
    rows: List[Tuple[str, str]] = []
    notes: List[str] = []
    for line in lines:
        match = ROW_PATTERN.match(line)
        if match and len(match.group(1)) <= 4:
            rows.append((match.group(1), match.group(2).strip()))
        else:
            notes.append(line)

    section_lines = [name]
    if rows:
        section_lines.append("| Code | Details |")
        section_lines.append("|------|---------|")
        section_lines.extend([f"| {code} | {details} |" for code, details in rows])
    if notes:
        section_lines.append("Notes:")
        section_lines.extend([f"- {note}" for note in notes])
    if not rows and not notes:
        section_lines.extend(lines)
    return "\n".join(section_lines).strip()


def _ingest_text_file(file_path: Path, settings: Settings) -> List[DocumentChunk]:
    content = file_path.read_text(encoding="utf-8")
    chunks = split_text(content, settings.chunk_size, settings.chunk_overlap)
    return [
        DocumentChunk(content=chunk, source_file=file_path.name)
        for chunk in chunks
    ]


def _ingest_pdf(file_path: Path, settings: Settings) -> List[DocumentChunk]:
    image_dir = settings.processed_path / "images" / file_path.stem
    pages = extract_pdf_content(
        file_path,
        image_dir,
        settings.ocr_engine,
        settings.ocr_dpi,
        settings.ocr_min_text_chars,
        settings.ocr_always_on,
    )
    chunks: List[DocumentChunk] = []
    for text, page_number, images in pages:
        if not text:
            continue
        for section_name, section_block in _extract_section_blocks(text):
            formatted = _format_section_block(section_name, section_block)
            if formatted:
                chunks.append(
                    DocumentChunk(
                        content=formatted,
                        source_file=file_path.name,
                        page_number=page_number,
                        images=images,
                        section=section_name,
                    )
                )
        for chunk in split_text(text, settings.chunk_size, settings.chunk_overlap):
            chunks.append(
                DocumentChunk(
                    content=chunk,
                    source_file=file_path.name,
                    page_number=page_number,
                    images=images,
                )
            )
    return chunks


def collect_chunks(settings: Settings) -> List[DocumentChunk]:
    ensure_dir(settings.processed_path)
    raw_files = sorted(settings.raw_path.glob("**/*"))
    if not raw_files:
        raise FileNotFoundError("No source files found under data/raw. Add brochures first.")

    all_chunks: List[DocumentChunk] = []
    for file_path in raw_files:
        if file_path.is_dir():
            continue
        suffix = file_path.suffix.lower()
        LOGGER.info("Processing %s", file_path.name)
        if suffix in {".txt", ".md"}:
            all_chunks.extend(_ingest_text_file(file_path, settings))
        elif suffix == ".pdf":
            all_chunks.extend(_ingest_pdf(file_path, settings))
        else:
            LOGGER.warning("Skipping unsupported file: %s", file_path)
    if not all_chunks:
        raise RuntimeError("No chunks were produced. Ensure PDFs/text files contain content.")
    return all_chunks


def persist_image_manifest(chunks: List[DocumentChunk], manifest_path: Path) -> None:
    seen: dict[str, ImageRecord] = {}
    for chunk in chunks:
        for image in chunk.images:
            seen[image.id] = image
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "id": record.id,
                    "source_file": record.source_file,
                    "page_number": record.page_number,
                    "path": str(record.path),
                    "caption": record.caption,
                }
                for record in seen.values()
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def run(settings: Settings) -> None:
    LOGGER.info("Starting ingestion with data directory %s", settings.raw_path)
    chunks = collect_chunks(settings)
    embedder = EmbeddingClient(settings.embedding_model)
    store = VectorStore(settings.faiss_index_path, settings.chunk_store_path, embedder)
    store.index_chunks(chunks)
    store.save()
    persist_image_manifest(chunks, settings.image_manifest_path)
    LOGGER.info("Stored %s chunks", len(chunks))
