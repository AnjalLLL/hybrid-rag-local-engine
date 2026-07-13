"""Shared helpers for the offline RAG pipeline."""

from __future__ import annotations

import os
import re
import shutil
import site
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.append(user_site)

try:
    from nltk.tokenize import sent_tokenize
except Exception:  # pragma: no cover - handled via fallback
    sent_tokenize = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FAISS_DIR = PROJECT_ROOT / "faiss_index"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"
TEMP_TEXT_DIR = PROJECT_ROOT / "tmp_extracted_text"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_CHUNK_TOKENS = 450
OVERLAP_TOKENS = 75
MIN_CHUNK_TOKENS = 30
_ENCODING: Optional[Any] = None


def configure_offline_runtime() -> None:
    """Configure libraries to stay on local caches and local runtimes."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def snapshot_root(repo_id: str, model_dir: Path = MODELS_DIR) -> Path:
    """Return the HF cache snapshot directory for a repo id."""

    return model_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots"


def resolve_cached_model_path(repo_id: str = MODEL_NAME, model_dir: Path = MODELS_DIR) -> Path | None:
    """Return the newest cached snapshot path for a repo, when available."""

    root = snapshot_root(repo_id, model_dir)
    if not root.exists():
        return None

    snapshots = sorted(
        (path for path in root.iterdir() if path.is_dir() and any(path.iterdir())),
        key=lambda path: path.stat().st_mtime,
    )
    return snapshots[-1] if snapshots else None


def get_chunk_encoding():
    """Load the cl100k_base encoding lazily."""

    global _ENCODING
    if _ENCODING is None:
        import tiktoken

        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    """Return the token count for a text snippet."""

    return len(get_chunk_encoding().encode(text))


def looks_like_code(text: str) -> bool:
    """Heuristically detect R/code-heavy text that must keep its line breaks."""

    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    code_markers = re.compile(r"(<-|%>%|\|>|^\s*#|\w+\s*\(|\$\w|~|==|\[\s*\d)")
    hits = sum(1 for line in lines if code_markers.search(line))
    return hits >= max(3, int(0.4 * len(lines)))


def split_sentences(text: str) -> List[str]:
    """Split prose into sentences, preferring NLTK with a regex fallback."""

    cleaned = re.sub(r"[ \t]+", " ", text)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()
    if not cleaned:
        return []

    if sent_tokenize is not None:
        try:
            return [sentence.strip() for sentence in sent_tokenize(cleaned) if sentence.strip()]
        except LookupError:
            pass

    fallback = re.split(r"(?<=[.!?])\s+|\n", cleaned)
    return [sentence.strip() for sentence in fallback if sentence.strip()]


def chunk_text(text: str) -> List[str]:
    """Chunk text into overlapping token-bounded windows.

    Code-like input is chunked by lines so that indentation and newlines survive;
    prose is chunked by sentences.
    """

    if not text.strip():
        return []

    if looks_like_code(text):
        return _chunk_by_lines(text)

    return _chunk_by_sentences(text)


def _chunk_by_sentences(text: str) -> List[str]:
    """Chunk prose into overlapping sentence windows."""

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []

    for sentence in sentences:
        if not current and count_tokens(sentence) > MAX_CHUNK_TOKENS:
            chunks.extend(_split_oversized_sentence(sentence))
            continue

        candidate = current + [sentence]
        if current and count_tokens(" ".join(candidate)) > MAX_CHUNK_TOKENS:
            chunks.append(" ".join(current).strip())
            current = _overlap_tail(current)
            if count_tokens(" ".join(current + [sentence])) > MAX_CHUNK_TOKENS:
                chunks.extend(_split_oversized_sentence(sentence))
                current = []
            else:
                current.append(sentence)
        else:
            current = candidate

    _flush_tail(chunks, " ".join(current).strip())
    return chunks


def _chunk_by_lines(text: str) -> List[str]:
    """Chunk code into overlapping line windows, preserving newlines."""

    lines = text.splitlines()
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = count_tokens(line) + 1
        if current and current_tokens + line_tokens > MAX_CHUNK_TOKENS:
            chunks.append("\n".join(current).strip())
            current = _overlap_tail_lines(current)
            current_tokens = sum(count_tokens(item) + 1 for item in current)
        current.append(line)
        current_tokens += line_tokens

    _flush_tail(chunks, "\n".join(current).strip())
    return chunks


def _flush_tail(chunks: List[str], tail: str) -> None:
    """Append the trailing chunk, merging it back when it is too small to stand alone.

    Short pages and slides used to be discarded outright, which silently dropped
    content from the index.
    """

    if not tail:
        return

    if count_tokens(tail) >= MIN_CHUNK_TOKENS or not chunks:
        chunks.append(tail)
        return

    separator = "\n" if "\n" in chunks[-1] else " "
    chunks[-1] = f"{chunks[-1]}{separator}{tail}"


def load_embedding_model(model_dir: Path = MODELS_DIR):
    """Load the embedding model strictly from local cache."""

    from sentence_transformers import SentenceTransformer

    configure_offline_runtime()
    model_dir.mkdir(parents=True, exist_ok=True)
    cached_snapshot = resolve_cached_model_path(MODEL_NAME, model_dir)
    if cached_snapshot is None:
        raise FileNotFoundError(
            "Embedding model cache is missing under ./models/. "
            "Run `python main.py fetch-models` once while online."
        )

    return SentenceTransformer(
        str(cached_snapshot),
        cache_folder=str(model_dir),
        local_files_only=True,
        device="cpu",
        model_kwargs={"low_cpu_mem_usage": True},
    )


def load_reranker(model_dir: Path = MODELS_DIR):
    """Load the cross-encoder reranker from local cache, or return None."""

    cached_snapshot = resolve_cached_model_path(RERANKER_NAME, model_dir)
    if cached_snapshot is None:
        return None

    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(
            str(cached_snapshot),
            max_length=512,
            device="cpu",
            local_files_only=True,
        )
    except Exception as exc:  # pragma: no cover - defensive, keeps retrieval usable
        print(f"Cross-encoder reranker failed to load ({exc}). Continuing without reranking.")
        return None


def fetch_models(model_dir: Path = MODELS_DIR) -> int:
    """Download the embedding and reranker models into ./models/ for offline use."""

    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    model_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    for repo_id in (MODEL_NAME, RERANKER_NAME):
        if resolve_cached_model_path(repo_id, model_dir) is not None:
            print(f"Already cached: {repo_id}")
            continue
        print(f"Downloading {repo_id} ...")
        try:
            snapshot_download(
                repo_id=repo_id,
                cache_dir=str(model_dir),
                ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*openvino*", "*.onnx"],
            )
        except Exception as exc:
            print(f"Failed to download {repo_id}: {exc}")
            return 1

    print(f"Models cached under {model_dir}. You can go offline again.")
    return 0


def cleanup() -> None:
    """Remove temporary artifacts outside persistent project directories."""

    removed = 0

    if TEMP_TEXT_DIR.exists():
        for txt_file in TEMP_TEXT_DIR.rglob("*.txt"):
            txt_file.unlink(missing_ok=True)
            removed += 1
        shutil.rmtree(TEMP_TEXT_DIR, ignore_errors=True)
        removed += 1

    for pycache_dir in SRC_DIR.rglob("__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir, ignore_errors=True)
            removed += 1

    for pyc_file in SRC_DIR.rglob("*.pyc"):
        pyc_file.unlink(missing_ok=True)
        removed += 1

    print(f"Cleanup complete. Removed {removed} temporary files.")


def _split_oversized_sentence(sentence: str) -> List[str]:
    """Split a very long sentence by token windows when needed."""

    encoding = get_chunk_encoding()
    token_ids = encoding.encode(sentence)
    chunks: List[str] = []
    start = 0
    step = max(1, MAX_CHUNK_TOKENS - OVERLAP_TOKENS)

    while start < len(token_ids):
        end = min(len(token_ids), start + MAX_CHUNK_TOKENS)
        text = encoding.decode(token_ids[start:end]).strip()
        if text:
            _flush_tail(chunks, text)
        if end == len(token_ids):
            break
        start += step

    return chunks


def _overlap_tail(sentences: Iterable[str]) -> List[str]:
    """Return the trailing overlap sentences for the next chunk."""

    kept: List[str] = []
    total = 0

    for sentence in reversed(list(sentences)):
        sentence_tokens = count_tokens(sentence)
        if kept and total + sentence_tokens > OVERLAP_TOKENS:
            break
        kept.append(sentence)
        total += sentence_tokens
        if total >= OVERLAP_TOKENS:
            break

    return list(reversed(kept))


def _overlap_tail_lines(lines: Iterable[str]) -> List[str]:
    """Return the trailing overlap lines for the next code chunk."""

    kept: List[str] = []
    total = 0

    for line in reversed(list(lines)):
        line_tokens = count_tokens(line) + 1
        if kept and total + line_tokens > OVERLAP_TOKENS:
            break
        kept.append(line)
        total += line_tokens

    return list(reversed(kept))


if __name__ == "__main__":
    configure_offline_runtime()
    sample = "R has loops. It also has vectorization. Use apply functions when possible."
    print(chunk_text(sample))
