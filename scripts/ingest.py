"""CLI entrypoint for building the Yantra Live knowledge base."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yantra_rag.config import Settings, load_settings
from yantra_rag import ingest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yantra Live RAG ingestion")
    parser.add_argument(
        "--data-root",
        dest="data_root",
        default=None,
        help="Override data directory (defaults to ./data)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Override chunk size",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        help="Override chunk overlap",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    settings = load_settings()
    if args.data_root:
        settings.raw_path = Path(args.data_root)
    if args.chunk_size:
        settings.chunk_size = args.chunk_size
    if args.chunk_overlap:
        settings.chunk_overlap = args.chunk_overlap
    ingest.run(settings)


if __name__ == "__main__":
    main()
