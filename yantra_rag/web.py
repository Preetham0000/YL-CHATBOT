"""FastAPI application exposing the Yantra Live RAG chatbot."""
from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import load_settings, Settings
from .rag_agent import YantraRAGAgent


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    grounded: bool
    citations: List[str]
    images: List[str]


def _image_url(path: Path, settings: Settings) -> str:
    images_root = settings.processed_path / "images"
    try:
        rel = path.relative_to(images_root)
        return f"/images/{rel.as_posix()}"
    except ValueError:
        return path.as_posix()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    agent = YantraRAGAgent(settings)

    app = FastAPI(title="Yantra Live RAG API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_frontend = Path("frontend")
    if static_frontend.exists():
        app.mount("/static", StaticFiles(directory=static_frontend), name="frontend-static")

        @app.get("/")
        def serve_index() -> FileResponse:
            return FileResponse(static_frontend / "index.html")

    images_root = settings.processed_path / "images"
    app.mount("/images", StaticFiles(directory=images_root, check_dir=False), name="images")

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        answer = agent.answer(request.question)
        image_urls = [_image_url(path, settings) for path in answer.images]
        return ChatResponse(
            answer=answer.answer,
            grounded=answer.grounded,
            citations=answer.citations,
            images=image_urls,
        )

    return app


settings = load_settings()
app = create_app(settings)
