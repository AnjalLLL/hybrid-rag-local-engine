"""CLI entry point for the offline RAG system."""

from __future__ import annotations

import argparse
import sys

from src.utils import configure_offline_runtime


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""

    parser = argparse.ArgumentParser(description="Offline RAG system for R exam preparation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build the hybrid index from files in ./data/.")
    ingest_parser.add_argument("--rebuild", action="store_true", help="Force a fresh index build.")

    query_parser = subparsers.add_parser("query", help="Ask a question against the local index.")
    query_parser.add_argument("question", nargs="?", help="Question to ask. Omit for interactive mode.")
    query_parser.add_argument(
        "--model-path",
        help="Local GGUF model path for llama-cpp-python fallback if Ollama is unavailable.",
    )
    query_parser.add_argument(
        "--ollama-model",
        default="qwen2.5-coder:7b",
        help="Ollama model name to use (default: qwen2.5-coder:7b).",
    )

    return parser


def run_query(question: str | None, model_path: str | None, ollama_model: str) -> int:
    """Run a single query or start an interactive REPL."""

    from src.query import RagQueryEngine

    try:
        engine = RagQueryEngine(model_path=model_path, ollama_model=ollama_model)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    except OSError as exc:
        print(f"Failed to load the local embedding model: {exc}")
        return 1

    if question:
        return engine.answer(question)

    print("Interactive mode. Press Ctrl-C to exit.")
    try:
        while True:
            user_question = input("\nQuestion> ").strip()
            if not user_question:
                continue
            engine.answer(user_question)
            print()
    except KeyboardInterrupt:
        print("\nExiting.")
        return 0


def main() -> int:
    """Parse CLI arguments and dispatch commands."""

    configure_offline_runtime()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        from src.ingest import build_index

        return build_index(rebuild=args.rebuild)
    if args.command == "query":
        return run_query(args.question, args.model_path, args.ollama_model)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
