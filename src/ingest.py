"""Build the local vector and keyword indexes from study files."""

from __future__ import annotations

import pickle
import re
import site
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.append(user_site)

from src.utils import DATA_DIR, FAISS_DIR, chunk_text, cleanup, configure_offline_runtime, load_embedding_model


ChunkRecord = Tuple[str, Dict[str, int | str]]
BATCH_SIZE = 64
SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".r"}


def tokenize_text(text: str) -> List[str]:
    """Tokenize text for BM25 while preserving code-friendly terms."""

    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]*|[0-9]+", text.lower())


def discover_source_files(data_dir: Path = DATA_DIR) -> List[Path]:
    """Return supported study files found recursively under the data directory."""

    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def discover_pdfs(data_dir: Path = DATA_DIR) -> List[Path]:
    """Return all PDFs found recursively under the data directory."""

    return sorted(path for path in data_dir.rglob("*.pdf") if path.is_file())


def extract_chunks_from_files(file_paths: Sequence[Path]) -> List[ChunkRecord]:
    """Extract, chunk, and annotate text from supported study files."""

    chunks: List[ChunkRecord] = []
    chunk_id = 0

    for file_path in file_paths:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            records = extract_chunks_from_pdf(file_path, chunk_id)
        elif suffix == ".pptx":
            records = extract_chunks_from_pptx(file_path, chunk_id)
        elif suffix == ".r":
            records = extract_chunks_from_text_file(file_path, chunk_id)
        else:
            records = []

        chunks.extend(records)
        chunk_id += len(records)

    return chunks


def extract_chunks_from_pdfs(pdf_paths: Sequence[Path]) -> List[ChunkRecord]:
    """Extract, chunk, and annotate text from PDFs."""

    chunks: List[ChunkRecord] = []
    chunk_id = 0

    for pdf_path in pdf_paths:
        records = extract_chunks_from_pdf(pdf_path, chunk_id)
        chunks.extend(records)
        chunk_id += len(records)

    return chunks


def extract_chunks_from_pdf(pdf_path: Path, start_chunk_id: int = 0) -> List[ChunkRecord]:
    """Extract chunk records from one PDF."""

    import fitz

    chunks: List[ChunkRecord] = []
    chunk_id = start_chunk_id

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if not text:
                print(f"Warning: No extractable text found in {pdf_path.name} page {page_index}. Skipping.")
                continue

            for chunk in chunk_text(text):
                chunks.append(
                    (
                        chunk,
                        {
                            "source": pdf_path.name,
                            "page": page_index,
                            "kind": "pdf",
                            "chunk_id": chunk_id,
                        },
                    )
                )
                chunk_id += 1

    return chunks


def extract_chunks_from_pptx(pptx_path: Path, start_chunk_id: int = 0) -> List[ChunkRecord]:
    """Extract chunk records from one PowerPoint file."""

    from pptx import Presentation

    chunks: List[ChunkRecord] = []
    chunk_id = start_chunk_id
    presentation = Presentation(pptx_path)

    for slide_index, slide in enumerate(presentation.slides, start=1):
        text_parts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                text_parts.append(shape.text)
        text = "\n".join(text_parts).strip()
        if not text:
            print(f"Warning: No extractable text found in {pptx_path.name} slide {slide_index}. Skipping.")
            continue

        for chunk in chunk_text(text):
            chunks.append(
                (
                    chunk,
                    {
                        "source": pptx_path.name,
                        "page": slide_index,
                        "kind": "pptx",
                        "chunk_id": chunk_id,
                    },
                )
            )
            chunk_id += 1

    return chunks


def extract_chunks_from_text_file(text_path: Path, start_chunk_id: int = 0) -> List[ChunkRecord]:
    """Extract chunk records from one plain-text/code file."""

    text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        print(f"Warning: No extractable text found in {text_path.name}. Skipping.")
        return []

    chunks: List[ChunkRecord] = []
    chunk_id = start_chunk_id
    for chunk in chunk_text(text):
        chunks.append(
            (
                chunk,
                {
                    "source": text_path.name,
                    "page": 1,
                    "kind": text_path.suffix.lower().lstrip("."),
                    "chunk_id": chunk_id,
                },
            )
        )
        chunk_id += 1

    return chunks


def embed_texts(model, texts: Sequence[str], batch_size: int = BATCH_SIZE):
    """Embed texts in batches and return normalized float32 vectors."""

    import numpy as np

    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype="float32")


def persist_index(embeddings, chunks: Sequence[ChunkRecord], faiss_dir: Path = FAISS_DIR) -> None:
    """Write the FAISS index and chunk metadata to disk."""

    import faiss
    import numpy as np

    faiss_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(faiss_dir / "index.faiss"))
    np.save(faiss_dir / "embeddings.npy", embeddings)

    payload = [{"text": text, "metadata": metadata} for text, metadata in chunks]
    tokenized_corpus = [tokenize_text(chunk["text"]) for chunk in payload]
    with (faiss_dir / "chunks.pkl").open("wb") as handle:
        pickle.dump({"chunks": payload, "tokenized_corpus": tokenized_corpus}, handle)


def build_index(rebuild: bool = False) -> int:
    """Build the FAISS index from the PDFs under the data directory."""

    configure_offline_runtime()
    index_path = FAISS_DIR / "index.faiss"
    chunks_path = FAISS_DIR / "chunks.pkl"

    if not rebuild and index_path.exists() and chunks_path.exists():
        print(f"Index already exists at {FAISS_DIR}. Use --rebuild to force a fresh build.")
        return 0

    source_paths = discover_source_files()
    if not source_paths:
        print("No PDF, PPTX, or R files found under ./data/. Add files and run `python main.py ingest` again.")
        return 1

    print(f"Found {len(source_paths)} study files. Extracting text...")
    chunks = extract_chunks_from_files(source_paths)
    if not chunks:
        print("No usable text chunks were extracted from the PDFs.")
        return 1

    print(f"Building embeddings for {len(chunks)} chunks...")
    model = load_embedding_model()
    texts = [text for text, _ in chunks]
    embeddings = embed_texts(model, texts)

    print("Writing FAISS index...")
    persist_index(embeddings, chunks)
    cleanup()
    print(f"Index build complete. Saved artifacts to {FAISS_DIR}.")
    return 0


if __name__ == "__main__":
    configure_offline_runtime()
    exit_code = build_index(rebuild=False) if discover_source_files() else 0
    sys.exit(exit_code)
