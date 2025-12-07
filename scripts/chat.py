"""Command-line chat client for the Yantra Live RAG agent."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yantra_rag.config import load_settings
from yantra_rag.rag_agent import YantraRAGAgent

EXIT_WORDS = {"exit", "quit", "q"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yantra Live chatbot")
    parser.add_argument(
        "--question",
        help="Optional single-shot mode question (skips interactive chat)",
    )
    return parser.parse_args()


def interactive_loop(agent: YantraRAGAgent) -> None:
    print("Yantra Live RAG agent ready. Type 'exit' to quit.")
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in EXIT_WORDS:
            print("Goodbye!")
            break
        answer = agent.answer(query)
        print(f"Agent: {answer.answer}")
        if answer.images:
            print("Images:")
            for path in answer.images:
                print(f" - {path}")


def single_shot(agent: YantraRAGAgent, question: str) -> None:
    answer = agent.answer(question)
    print(answer.answer)
    if answer.images:
        print("Images:")
        for path in answer.images:
            print(f" - {path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    settings = load_settings()
    agent = YantraRAGAgent(settings)
    if args.question:
        single_shot(agent, args.question)
    else:
        interactive_loop(agent)


if __name__ == "__main__":
    main()
