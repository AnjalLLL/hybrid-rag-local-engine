"""Shared helpers for the offline RAG pipeline."""

from __future__ import annotations

import os
import re
import shutil
import site
import subprocess
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
SNAPSHOT_ROOT = MODELS_DIR / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
MAX_CHUNK_TOKENS = 450
OVERLAP_TOKENS = 75
MIN_CHUNK_TOKENS = 30
_ENCODING: Optional[Any] = None


def configure_offline_runtime() -> None:
    """Configure libraries to stay on local caches and local runtimes."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, preferring NLTK with a regex fallback."""

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    if sent_tokenize is not None:
        try:
            return [sentence.strip() for sentence in sent_tokenize(cleaned) if sentence.strip()]
        except LookupError:
            pass

    fallback = re.split(r"(?<=[.!?])\s+", cleaned)
    return [sentence.strip() for sentence in fallback if sentence.strip()]


def chunk_text(text: str) -> List[str]:
    """Chunk text into overlapping token-bounded windows."""

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []

    for sentence in sentences:
        if not current_sentences and count_tokens(sentence) > MAX_CHUNK_TOKENS:
            chunks.extend(_split_oversized_sentence(sentence))
            continue

        candidate_sentences = current_sentences + [sentence]
        candidate_text = " ".join(candidate_sentences).strip()
        if current_sentences and count_tokens(candidate_text) > MAX_CHUNK_TOKENS:
            chunk_text_value = " ".join(current_sentences).strip()
            if count_tokens(chunk_text_value) >= MIN_CHUNK_TOKENS:
                chunks.append(chunk_text_value)
            current_sentences = _overlap_tail(current_sentences)
            if count_tokens(" ".join(current_sentences + [sentence]).strip()) > MAX_CHUNK_TOKENS:
                if count_tokens(sentence) >= MIN_CHUNK_TOKENS:
                    chunks.extend(_split_oversized_sentence(sentence))
                current_sentences = []
            else:
                current_sentences.append(sentence)
        else:
            current_sentences = candidate_sentences

    final_chunk = " ".join(current_sentences).strip()
    if final_chunk and count_tokens(final_chunk) >= MIN_CHUNK_TOKENS:
        chunks.append(final_chunk)

    return chunks


def resolve_cached_model_path() -> Path | None:
    """Return the newest cached embedding snapshot path when available."""

    if not SNAPSHOT_ROOT.exists():
        return None

    snapshots = sorted(path for path in SNAPSHOT_ROOT.iterdir() if path.is_dir())
    return snapshots[-1] if snapshots else None


def load_embedding_model(model_dir: Path = MODELS_DIR):
    """Load the embedding model strictly from local cache."""

    from sentence_transformers import SentenceTransformer

    configure_offline_runtime()
    model_dir.mkdir(parents=True, exist_ok=True)
    cached_snapshot = resolve_cached_model_path()
    if cached_snapshot is None:
        raise FileNotFoundError(
            "Embedding model cache is missing under ./models/. Run `python main.py ingest --rebuild` once with the model already available locally."
        )

    return SentenceTransformer(
        str(cached_snapshot),
        cache_folder=str(model_dir),
        local_files_only=True,
        device="cpu",
        model_kwargs={"low_cpu_mem_usage": True},
    )


def cleanup() -> None:
    """Remove temporary artifacts outside persistent project directories."""

    removed = 0

    if TEMP_TEXT_DIR.exists():
        for txt_file in TEMP_TEXT_DIR.rglob("*.txt"):
            txt_file.unlink(missing_ok=True)
            removed += 1
        shutil.rmtree(TEMP_TEXT_DIR, ignore_errors=True)
        removed += 1

    _purge_pip_cache()

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
        if count_tokens(text) >= MIN_CHUNK_TOKENS:
            chunks.append(text)
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


def _purge_pip_cache() -> None:
    """Silently purge the local pip cache."""

    commands = (
        ["python", "-m", "pip", "cache", "purge"],
        ["pip", "cache", "purge"],
    )

    for command in commands:
        try:
            subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=PROJECT_ROOT,
            )
            return
        except OSError:
            continue


if __name__ == "__main__":
    configure_offline_runtime()
    sample = "R has loops. It also has vectorization. Use apply functions when possible."
    print(chunk_text(sample))
