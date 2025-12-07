"""Configuration helpers for the Yantra Live RAG stack."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os


class Settings(BaseModel):
    """Runtime settings that can be overridden via environment variables."""

    data_root: Path = Field(default=Path("data"))
    raw_path: Path = Field(default=Path("data/raw"))
    processed_path: Path = Field(default=Path("data/processed"))
    chunk_store_path: Path = Field(default=Path("data/processed/chunks.jsonl"))
    image_manifest_path: Path = Field(default=Path("data/processed/images_manifest.json"))
    faiss_index_path: Path = Field(default=Path("data/processed/index.faiss"))

    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = Field(default=400)
    chunk_overlap: int = Field(default=100)

    top_k: int = Field(default=8)
    min_similarity: float = Field(default=0.25)

    ocr_engine: str = Field(default="tesseract", description="OCR backend name or 'none'")
    ocr_dpi: int = Field(default=300)
    ocr_min_text_chars: int = Field(default=40, description="Minimum characters before falling back to OCR")
    ocr_always_on: bool = Field(default=True)

    enable_image_context: bool = Field(default=True)
    image_context_limit: int = Field(default=2)

    llm_provider: str = Field(default="openai", description="openai, gemini, or groq")
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    gemini_api_key: Optional[str] = Field(default=None)
    gemini_model: str = Field(default="gemini-3-pro-preview")
    groq_api_key: Optional[str] = Field(default=None)
    groq_model: str = Field(default="llama3-70b-8192")
    
    gemini_store_id: Optional[str] = Field(default=None, description="ID of the Google File Search Store")

    class Config:
        arbitrary_types_allowed = True


def _first_env(*keys: str) -> Optional[str]:
    for key in keys:
        if not key:
            continue
        value = os.getenv(key)
        if value:
            return value
    return None


def load_settings(env_path: Optional[Path] = None) -> Settings:
    """Load `.env` overrides (if present) and return populated settings."""

    if env_path is None:
        env_path = Path(".env")

    if env_path.exists():
        load_dotenv(env_path)

    defaults = Settings()
    gemini_model = _first_env("GEMINI_MODEL", "GOOGLE_MODEL", "MODEL") or defaults.gemini_model
    groq_model = _first_env("GROQ_MODEL", "MODEL") or defaults.groq_model
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", defaults.llm_provider),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", defaults.openai_model),
        gemini_api_key=_first_env("GEMINI_API_KEY", "GOOGLE_API_KEY", "Gemini_API_KEY"),
        gemini_model=gemini_model,
        gemini_store_id=os.getenv("GEMINI_STORE_ID"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=groq_model,
        ocr_engine=os.getenv("OCR_ENGINE", defaults.ocr_engine),
        ocr_dpi=int(os.getenv("OCR_DPI", defaults.ocr_dpi)),
        ocr_min_text_chars=int(os.getenv("OCR_MIN_TEXT_CHARS", defaults.ocr_min_text_chars)),
        ocr_always_on=os.getenv("OCR_ALWAYS_ON", str(defaults.ocr_always_on)).lower()
        in ("1", "true", "yes"),
        enable_image_context=os.getenv("ENABLE_IMAGE_CONTEXT", str(defaults.enable_image_context)).lower()
        in ("1", "true", "yes"),
        image_context_limit=int(os.getenv("IMAGE_CONTEXT_LIMIT", defaults.image_context_limit)),
    )
