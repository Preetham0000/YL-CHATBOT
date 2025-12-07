"""Launch the FastAPI server for the Yantra Live chatbot."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    reload_flag = os.getenv("UVICORN_RELOAD", "true").lower() not in {"0", "false", "no"}
    uvicorn.run(
        "yantra_rag.web:app",
        host="0.0.0.0",
        port=8000,
        reload=reload_flag,
    )


if __name__ == "__main__":
    main()
