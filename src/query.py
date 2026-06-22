"""Retrieve relevant chunks and answer questions with a local LLM."""

from __future__ import annotations

import json
import pickle
import re
import site
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.append(user_site)

import numpy as np
import requests
from rank_bm25 import BM25Okapi

from src.utils import FAISS_DIR, MODELS_DIR, configure_offline_runtime, load_embedding_model


ChunkPayload = Dict[str, Dict[str, int | str] | str]
IndexAssets = Tuple[object, List[ChunkPayload], List[List[str]], Optional[np.ndarray]]
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"
TOP_K = 6
RETRIEVAL_CANDIDATES = 100
RRF_K = 60
MAX_CHUNK_CHARS = 1200
RETRY_CHUNK_CHARS = 350
OLLAMA_NUM_CTX = 4096
RETRY_OLLAMA_NUM_CTX = 1024
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RagQueryEngine:
    """Query engine that keeps the retrieval assets loaded in memory."""

    def __init__(self, model_path: Optional[str] = None, ollama_model: str = DEFAULT_OLLAMA_MODEL) -> None:
        configure_offline_runtime()
        self.model_path = model_path
        self.ollama_model = ollama_model
        self.index, self.chunks, self.tokenized_corpus, self.dense_embeddings = load_index_assets()
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        self.embedder = None
        self.reranker = load_reranker()
        self.retrieval_mode = "bm25"

        try:
            self.embedder = load_embedding_model(MODELS_DIR)
            self.retrieval_mode = "hybrid"
        except (FileNotFoundError, OSError) as exc:
            print(f"Embedding retriever unavailable ({exc}). Falling back to offline keyword retrieval.")

        if self.reranker is not None:
            self.retrieval_mode += "+rerank"

    def answer(self, question: str, top_k: int = TOP_K) -> int:
        """Retrieve context and stream an answer for a question."""

        print(f"Retrieving top {top_k} chunks using {self.retrieval_mode} retrieval...\n")
        results = retrieve_chunks(
            question,
            self.embedder,
            self.index,
            self.chunks,
            self.tokenized_corpus,
            self.dense_embeddings,
            self.bm25,
            self.reranker,
            top_k=top_k,
        )
        prompt = build_prompt(question, results, chunk_char_limit=MAX_CHUNK_CHARS)

        print("Answer:")
        try:
            stream_ollama_response(prompt, self.ollama_model, num_ctx=OLLAMA_NUM_CTX)
        except requests.HTTPError as exc:
            status_code, error_text = extract_http_error(exc)
            if status_code == 500 and "runner process has terminated" in error_text.lower():
                print("\nRetrying with a smaller offline prompt for low-memory mode...")
                compact_prompt = build_prompt(question, results, chunk_char_limit=RETRY_CHUNK_CHARS)
                try:
                    stream_ollama_response(
                        compact_prompt,
                        self.ollama_model,
                        num_ctx=RETRY_OLLAMA_NUM_CTX,
                    )
                except requests.HTTPError as retry_exc:
                    return self._handle_http_error(retry_exc, prompt)
            else:
                return self._handle_http_error(exc, prompt)
        except requests.RequestException:
            print()
            if not self.model_path:
                print(
                    "Ollama is unreachable at http://localhost:11434. "
                    "Start Ollama or rerun with --model-path to use llama-cpp-python."
                )
                return 1
            print(
                "Ollama is unreachable at http://localhost:11434. "
                f"Falling back to llama-cpp-python with {self.model_path}."
            )
            stream_llama_cpp_response(prompt, self.model_path)

        print("\n\nSources:")
        for source, page in format_sources(results):
            print(f"  - {source}, page {page}")

        return 0

    def _handle_http_error(self, exc: requests.HTTPError, prompt: str) -> int:
        """Print HTTP error details and optionally use llama-cpp fallback."""

        print()
        status_code, error_text = extract_http_error(exc)
        if status_code == 404:
            print(
                f"Ollama is running, but model '{self.ollama_model}' is not available locally. "
                f"Pull it with `ollama pull {self.ollama_model}` or rerun with --ollama-model using an installed model."
            )
        else:
            message = f"Ollama returned HTTP {status_code}."
            if error_text:
                message += f" Details: {error_text}"
            print(message)

        if not self.model_path:
            print("You can also supply --model-path to use llama-cpp-python as a fallback.")
            return 1
        print(f"Falling back to llama-cpp-python with {self.model_path}.")
        stream_llama_cpp_response(prompt, self.model_path)
        return 0


def load_index_assets(faiss_dir: Path = FAISS_DIR) -> IndexAssets:
    """Load the persisted FAISS index and chunk payload."""

    chunks_path = faiss_dir / "chunks.pkl"
    embeddings_path = faiss_dir / "embeddings.npy"

    if not chunks_path.exists():
        raise FileNotFoundError(
            "Chunk index not found. Run `python main.py ingest` before querying."
        )

    with chunks_path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict):
        chunks = payload.get("chunks", [])
        tokenized_corpus = payload.get("tokenized_corpus", [])
    else:
        chunks = payload
        tokenized_corpus = [tokenize_text(str(chunk["text"])) for chunk in chunks]

    dense_embeddings = np.load(embeddings_path) if embeddings_path.exists() else None

    return None, chunks, tokenized_corpus, dense_embeddings


def retrieve_chunks(
    question: str,
    embedder,
    index: object,
    chunks: Sequence[ChunkPayload],
    tokenized_corpus: Sequence[Sequence[str]],
    dense_embeddings: Optional[np.ndarray],
    bm25: Optional[BM25Okapi],
    reranker,
    top_k: int = TOP_K,
) -> List[ChunkPayload]:
    """Retrieve relevant chunks with hybrid search and optional reranking."""

    candidate_limit = min(RETRIEVAL_CANDIDATES, len(chunks))
    if candidate_limit <= 0:
        return []

    bm25_indices = retrieve_bm25_indices(question, bm25, candidate_limit)
    faiss_indices = retrieve_faiss_indices(question, embedder, index, dense_embeddings, candidate_limit)

    if faiss_indices and bm25_indices:
        ranked_indices = reciprocal_rank_fusion(faiss_indices, bm25_indices)
    else:
        ranked_indices = faiss_indices or bm25_indices

    if not ranked_indices and tokenized_corpus:
        ranked_indices = list(range(min(top_k, len(chunks))))

    rerank_candidates = ranked_indices[:candidate_limit]
    if reranker is not None and rerank_candidates:
        rerank_candidates = rerank_indices(question, chunks, rerank_candidates, reranker)

    return [chunks[idx] for idx in rerank_candidates[:top_k]]


def retrieve_bm25_indices(
    question: str,
    bm25: Optional[BM25Okapi],
    candidate_limit: int,
) -> List[int]:
    """Return top BM25 chunk indices for exact keyword/code matches."""

    if bm25 is None:
        return []

    query_tokens = tokenize_text(question)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    return [idx for idx in ranked[:candidate_limit] if scores[idx] > 0]


def retrieve_faiss_indices(
    question: str,
    embedder,
    index: object,
    dense_embeddings: Optional[np.ndarray],
    candidate_limit: int,
) -> List[int]:
    """Return top dense chunk indices for semantic matches."""

    if embedder is None:
        return []

    query_vector = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    if dense_embeddings is not None:
        scores = dense_embeddings @ query_vector[0]
        ranked = np.argsort(-scores)[:candidate_limit]
        return [int(idx) for idx in ranked]

    return []


def reciprocal_rank_fusion(*ranked_lists: Sequence[int], k: int = RRF_K) -> List[int]:
    """Merge ranked retrieval lists using Reciprocal Rank Fusion."""

    scores: Dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, idx in enumerate(ranked_list):
            scores[idx] = scores.get(idx, 0.0) + (1.0 / (k + rank + 1))

    return sorted(scores, key=lambda idx: scores[idx], reverse=True)


def rerank_indices(
    question: str,
    chunks: Sequence[ChunkPayload],
    candidate_indices: Sequence[int],
    reranker,
) -> List[int]:
    """Rerank candidate chunks with a cross-encoder."""

    pairs = [(question, str(chunks[idx]["text"])) for idx in candidate_indices]
    scores = reranker.predict(pairs)
    scored = sorted(zip(candidate_indices, scores), key=lambda item: float(item[1]), reverse=True)
    return [idx for idx, _ in scored]


def load_reranker(model_name: str = DEFAULT_RERANKER_MODEL):
    """Load an optional local cross-encoder reranker."""

    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(
            model_name,
            max_length=512,
            device="cpu",
            model_kwargs={"local_files_only": True},
            processor_kwargs={"local_files_only": True},
        )
    except Exception as exc:
        print(f"Cross-encoder reranker unavailable ({exc}). Continuing with hybrid retrieval only.")
        return None


def tokenize_text(text: str) -> List[str]:
    """Tokenize text for BM25 while preserving code-friendly terms."""

    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]*|[0-9]+", text.lower())


def truncate_context(text: str, limit: int) -> str:
    """Trim long retrieved text to a local-model-friendly size."""

    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def build_prompt(
    question: str,
    retrieved_chunks: Sequence[ChunkPayload],
    chunk_char_limit: int = MAX_CHUNK_CHARS,
) -> str:
    """Construct the RAG prompt from retrieved context and the user question."""

    context_blocks = []
    for chunk in retrieved_chunks:
        metadata = chunk["metadata"]
        context_blocks.append(
            f"[Source: {metadata['source']}, Page {metadata['page']}]\n"
            f"{truncate_context(str(chunk['text']), chunk_char_limit)}"
        )

    context = "\n\n".join(context_blocks)
    return (
        "Act as an R Programming Examiner. Solve the question EXACTLY as asked.\n\n"
        "STRICT RULES:\n"
        "1. Do NOT assume anything, including variable names or values. If file type is not explicitly given, assume it is an RData file and use load().\n"
        "2. Use ONLY variables defined in the question or context.\n"
        "3. Maintain the SAME variable names throughout the solution.\n"
        "4. If multiple parts exist, solve step-by-step using previous results.\n"
        "5. All methods must produce CONSISTENT results.\n"
        "6. Do NOT introduce undefined variables.\n"
        "7. Use correct R syntax only.\n"
        "8. Use base R unless explicitly asked otherwise.\n"
        "9. Do NOT add extra steps not asked in the question.\n"
        "10. Interpretation must be short, correct, and exam-style.\n"
        "11. Use ONLY the context below. If the context is insufficient, say so clearly.\n\n"
        "OUTPUT FORMAT:\n"
        "- If the question has parts, answer in the same part labels such as (a), (b), (c).\n"
        "- For code-only parts, output only the required R code for that part.\n"
        "- For interpretation parts, keep the interpretation brief and exam-style.\n"
        "- Do not include explanations before the answer.\n\n"
        "=== CONTEXT ===\n"
        f"{context}\n\n"
        "=== QUESTION ===\n"
        f"{question}\n\n"
        "=== ANSWER ===\n"
    )


def stream_ollama_response(prompt: str, model: str, num_ctx: int) -> None:
    """Stream a response from the local Ollama server."""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_ctx": num_ctx},
        },
        stream=True,
        timeout=(10, 600),
    )
    response.raise_for_status()

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        payload = json.loads(line)
        text = payload.get("response", "")
        if text:
            print(text, end="", flush=True)


def extract_http_error(exc: requests.HTTPError) -> Tuple[object, str]:
    """Return a status code and text payload from an HTTP error."""

    status_code = exc.response.status_code if exc.response is not None else "unknown"
    error_text = ""
    if exc.response is not None:
        try:
            payload = exc.response.json()
            error_text = payload.get("error", "") if isinstance(payload, dict) else ""
        except ValueError:
            error_text = exc.response.text.strip()
    return status_code, error_text


def stream_llama_cpp_response(prompt: str, model_path: str) -> None:
    """Stream a response from llama-cpp-python using a local GGUF model."""

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Install it and provide --model-path for fallback use."
        ) from exc

    llm = Llama(model_path=model_path, verbose=False)
    for chunk in llm.create_completion(prompt=prompt, stream=True):
        text = chunk["choices"][0]["text"]
        if text:
            print(text, end="", flush=True)


def format_sources(retrieved_chunks: Iterable[ChunkPayload]) -> List[Tuple[str, int]]:
    """Return sorted unique source/page pairs from retrieved chunks."""

    pairs = {
        (str(chunk["metadata"]["source"]), int(chunk["metadata"]["page"]))
        for chunk in retrieved_chunks
    }
    return sorted(pairs, key=lambda item: (item[0].lower(), item[1]))


if __name__ == "__main__":
    configure_offline_runtime()
    prompt = build_prompt(
        "What is a vector in R?",
        [
            {
                "text": "Vectors store elements of the same type.",
                "metadata": {"source": "smoke.pdf", "page": 1, "chunk_id": 0},
            }
        ],
    )
    print(prompt)
