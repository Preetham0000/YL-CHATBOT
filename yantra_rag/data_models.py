"""Shared data models for Yantra Live RAG."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ImageRecord(BaseModel):
    """Metadata about an extracted image."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_file: str
    page_number: Optional[int] = None
    path: Path
    caption: Optional[str] = None


class DocumentChunk(BaseModel):
    """A text chunk tied to a brochure source and optional images."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    source_file: str
    page_number: Optional[int] = None
    images: List[ImageRecord] = Field(default_factory=list)
    section: Optional[str] = None


class RetrievalResult(BaseModel):
    """Nearest-neighbor search result."""

    chunk: DocumentChunk
    score: float


class AgentAnswer(BaseModel):
    """Structured response returned by the chatbot."""

    answer: str
    citations: List[str] = Field(default_factory=list)
    images: List[Path] = Field(default_factory=list)
    grounded: bool = True
    suggested_questions: List[str] = Field(default_factory=list)
