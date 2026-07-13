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

from src.utils import (
    FAISS_DIR,
    MODELS_DIR,
    configure_offline_runtime,
    load_embedding_model,
    load_reranker,
)


ChunkPayload = Dict[str, Dict[str, int | str] | str]
IndexAssets = Tuple[List[ChunkPayload], List[List[str]], Optional[np.ndarray]]

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
PREFERRED_MODELS = ("qwen2.5-coder", "codellama", "deepseek-coder", "llama3", "mistral", "qwen")

TOP_K = 6
RETRIEVAL_CANDIDATES = 60
RERANK_CANDIDATES = 30
RRF_K = 60
DENSE_WEIGHT = 1.0
BM25_WEIGHT = 0.7
MAX_CHUNK_CHARS = 1500
RETRY_CHUNK_CHARS = 350
OLLAMA_NUM_CTX = 8192
RETRY_OLLAMA_NUM_CTX = 2048
OLLAMA_KEEP_ALIVE = "10m"
# Deterministic decoding: the default temperature of 0.8 made the same exam
# question produce a different (and often wrong) answer on every run.
GENERATION_OPTIONS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "num_predict": 1024,
}


class RagQueryEngine:
    """Query engine that keeps the retrieval assets loaded in memory."""

    def __init__(self, model_path: Optional[str] = None, ollama_model: Optional[str] = None) -> None:
        configure_offline_runtime()
        self.model_path = model_path
        self.chunks, self.tokenized_corpus, self.dense_embeddings = load_index_assets()
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        self.embedder = None
        self.reranker = load_reranker(MODELS_DIR)
        self.retrieval_mode = "bm25"

        try:
            self.embedder = load_embedding_model(MODELS_DIR)
            self.retrieval_mode = "hybrid"
        except (FileNotFoundError, OSError) as exc:
            print(f"Embedding retriever unavailable ({exc}). Falling back to offline keyword retrieval.")

        if self.reranker is not None:
            self.retrieval_mode += "+rerank"
        else:
            print(
                "Reranker not cached. Run `python main.py fetch-models` once (online) "
                "for noticeably better answers."
            )

        self.ollama_model = resolve_ollama_model(ollama_model)

    def answer(self, question: str, top_k: int = TOP_K) -> int:
        """Retrieve context and stream an answer for a question."""

        print(f"Retrieving top {top_k} chunks using {self.retrieval_mode} retrieval...\n")
        results = self.retrieve(question, top_k=top_k)

        if not results:
            print("No relevant context found in the index for this question.")
            return 1

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
                    f"Ollama is unreachable at {OLLAMA_HOST}. "
                    "Start Ollama or rerun with --model-path to use llama-cpp-python."
                )
                return 1
            print(
                f"Ollama is unreachable at {OLLAMA_HOST}. "
                f"Falling back to llama-cpp-python with {self.model_path}."
            )
            stream_llama_cpp_response(prompt, self.model_path)

        print("\n\nSources:")
        for source, page in format_sources(results):
            print(f"  - {source}, page {page}")

        return 0

    def retrieve(self, question: str, top_k: int = TOP_K) -> List[ChunkPayload]:
        """Retrieve the top chunks for a question."""

        return retrieve_chunks(
            question,
            self.embedder,
            self.chunks,
            self.dense_embeddings,
            self.bm25,
            self.reranker,
            top_k=top_k,
        )

    def _handle_http_error(self, exc: requests.HTTPError, prompt: str) -> int:
        """Print HTTP error details and optionally use llama-cpp fallback."""

        print()
        status_code, error_text = extract_http_error(exc)
        if status_code == 404:
            print(
                f"Ollama is running, but model '{self.ollama_model}' is not available locally. "
                f"Pull it with `ollama pull {self.ollama_model}`, or rerun with "
                f"--ollama-model using one of: {', '.join(list_ollama_models()) or 'none installed'}."
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


def list_ollama_models() -> List[str]:
    """Return the model names installed in the local Ollama server."""

    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    return [
        str(model["name"])
        for model in payload.get("models", [])
        if "embedding" not in model.get("capabilities", [])
    ]


def resolve_ollama_model(requested: Optional[str]) -> str:
    """Pick a generation model that is actually installed in Ollama.

    The previous hardcoded default (qwen2.5-coder:7b) 404'd on any machine that
    did not happen to have that exact tag pulled.
    """

    installed = list_ollama_models()

    if requested:
        if not installed or requested in installed:
            return requested
        stem = requested.split(":")[0]
        for name in installed:
            if name.split(":")[0] == stem:
                print(f"Model '{requested}' not installed; using '{name}'.")
                return name
        print(
            f"Model '{requested}' is not installed in Ollama. "
            f"Installed: {', '.join(installed)}. Trying it anyway."
        )
        return requested

    for preferred in PREFERRED_MODELS:
        for name in installed:
            if name.startswith(preferred):
                return name

    if installed:
        return installed[0]

    return "llama3:latest"


def load_index_assets(faiss_dir: Path = FAISS_DIR) -> IndexAssets:
    """Load the persisted FAISS index and chunk payload."""

    chunks_path = faiss_dir / "chunks.pkl"
    embeddings_path = faiss_dir / "embeddings.npy"
    index_path = faiss_dir / "index.faiss"

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

    if dense_embeddings is None and index_path.exists():
        print(
            "embeddings.npy is missing, so dense search is disabled. "
            "Run `python main.py ingest --rebuild`."
        )
    elif dense_embeddings is not None and len(dense_embeddings) != len(chunks):
        print(
            f"Embedding matrix holds {len(dense_embeddings)} vectors but the chunk payload has "
            f"{len(chunks)}. Run `python main.py ingest --rebuild`."
        )
        dense_embeddings = None

    # Dense search runs on the NumPy matrix rather than faiss.read_index(...).search(...).
    # faiss-cpu and torch both link libomp, and calling into faiss after torch is loaded
    # segfaults this process on macOS/ARM. The stored index is IndexFlatIP, i.e. exact
    # inner product, so the matrix multiply in retrieve_dense_indices is identical.
    return chunks, tokenized_corpus, dense_embeddings


def retrieve_chunks(
    question: str,
    embedder,
    chunks: Sequence[ChunkPayload],
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
    dense_indices = retrieve_dense_indices(question, embedder, dense_embeddings, candidate_limit)

    if dense_indices and bm25_indices:
        ranked_indices = reciprocal_rank_fusion(
            (dense_indices, DENSE_WEIGHT), (bm25_indices, BM25_WEIGHT)
        )
    else:
        ranked_indices = list(dense_indices or bm25_indices)

    if not ranked_indices:
        return []

    if reranker is not None:
        candidates = ranked_indices[:RERANK_CANDIDATES]
        ranked_indices = rerank_indices(question, chunks, candidates, reranker)

    selected = dedupe_indices(ranked_indices, chunks)[:top_k]
    return [chunks[idx] for idx in selected]


def dedupe_indices(indices: Sequence[int], chunks: Sequence[ChunkPayload]) -> List[int]:
    """Drop chunks whose text duplicates an already-selected chunk.

    Overlapping windows and repeated slide boilerplate otherwise burn several of
    the few context slots on the same sentences.
    """

    seen: set[str] = set()
    kept: List[int] = []

    for idx in indices:
        fingerprint = " ".join(str(chunks[idx]["text"]).split())[:200].lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept.append(idx)

    return kept


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

    scores = np.asarray(bm25.get_scores(query_tokens))
    if not scores.size or float(scores.max()) <= 0.0:
        return []

    top = np.argpartition(-scores, min(candidate_limit, scores.size - 1))[:candidate_limit]
    ranked = top[np.argsort(-scores[top])]
    return [int(idx) for idx in ranked if scores[idx] > 0]


def retrieve_dense_indices(
    question: str,
    embedder,
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

    if dense_embeddings is None:
        return []

    scores = dense_embeddings @ query_vector[0]
    top = np.argpartition(-scores, min(candidate_limit, scores.size - 1))[:candidate_limit]
    ranked = top[np.argsort(-scores[top])]
    return [int(idx) for idx in ranked]


def reciprocal_rank_fusion(
    *weighted_lists: Tuple[Sequence[int], float],
    k: int = RRF_K,
) -> List[int]:
    """Merge weighted ranked retrieval lists using Reciprocal Rank Fusion."""

    scores: Dict[int, float] = {}
    for ranked_list, weight in weighted_lists:
        for rank, idx in enumerate(ranked_list):
            scores[idx] = scores.get(idx, 0.0) + weight / (k + rank + 1)

    return sorted(scores, key=lambda idx: scores[idx], reverse=True)


def rerank_indices(
    question: str,
    chunks: Sequence[ChunkPayload],
    candidate_indices: Sequence[int],
    reranker,
) -> List[int]:
    """Rerank candidate chunks with a cross-encoder."""

    if not candidate_indices:
        return []

    pairs = [(question, str(chunks[idx]["text"])) for idx in candidate_indices]
    scores = reranker.predict(pairs, show_progress_bar=False)
    scored = sorted(zip(candidate_indices, scores), key=lambda item: float(item[1]), reverse=True)
    return [int(idx) for idx, _ in scored]


def tokenize_text(text: str) -> List[str]:
    """Tokenize text for BM25 while preserving code-friendly terms."""

    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]*|[0-9]+", text.lower())


def truncate_context(text: str, limit: int) -> str:
    """Trim long retrieved text to a local-model-friendly size, keeping line breaks."""

    normalized = re.sub(r"[ \t]+", " ", text).strip()
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
    for position, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]
        context_blocks.append(
            f"[{position}] Source: {metadata['source']}, page {metadata['page']}\n"
            f"{truncate_context(str(chunk['text']), chunk_char_limit)}"
        )

    context = "\n\n".join(context_blocks)
    return (
        "You are an R programming examiner and tutor. Answer the question exactly as asked, "
        "using the reference material below as your primary source.\n\n"
        "RULES:\n"
        "1. Prefer the context. If the context covers the question, follow it and cite the "
        "source number like [2].\n"
        "2. The context is course material, not the question's data. If the context does not "
        "cover the question, answer from standard R knowledge and say which part was not in "
        "the notes. Never refuse just because the context is thin.\n"
        "3. Use only the variables, data, and file names given in the question. Do not invent "
        "data, column names, or file formats, and do not add steps that were not asked for.\n"
        "4. Keep variable names consistent across every part of the answer, and reuse results "
        "from earlier parts in later parts.\n"
        "5. Write valid, runnable base R unless the question asks for a specific package.\n"
        "6. If the question has parts, answer under the same labels, for example (a), (b), (c). "
        "If it has no parts, do not invent part labels.\n"
        "7. Give code first, then a short exam-style interpretation only where one is asked for.\n"
        "8. Output only the answer. Start directly with the code or the answer text. Do not "
        "restate these rules and do not write a preamble such as 'Here is the answer'.\n\n"
        "=== CONTEXT ===\n"
        f"{context}\n\n"
        "=== QUESTION ===\n"
        f"{question}\n\n"
        "=== ANSWER ===\n"
    )


def stream_ollama_response(prompt: str, model: str, num_ctx: int) -> None:
    """Stream a response from the local Ollama server."""

    options = dict(GENERATION_OPTIONS)
    options["num_ctx"] = num_ctx

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": options,
        },
        stream=True,
        timeout=(10, 600),
    )
    response.raise_for_status()

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
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

    llm = Llama(model_path=model_path, n_ctx=OLLAMA_NUM_CTX, verbose=False)
    for chunk in llm.create_completion(
        prompt=prompt,
        stream=True,
        temperature=GENERATION_OPTIONS["temperature"],
        top_p=GENERATION_OPTIONS["top_p"],
        max_tokens=GENERATION_OPTIONS["num_predict"],
    ):
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
