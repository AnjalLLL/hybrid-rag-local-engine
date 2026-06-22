# Offline RAG for R Exam Preparation

This project builds a local Retrieval-Augmented Generation (RAG) workflow for R programming exam prep. It ingests PDFs, PowerPoint slides, and R scripts from `data/`, stores local dense embeddings plus BM25 keyword tokens, and answers questions with a local Ollama model.

## How retrieval works

The query engine now uses hybrid retrieval:

1. Dense embedding search finds semantically similar chunks.
2. BM25 sparse search finds exact R terms such as `lm`, `ggplot`, `filter`, and `na.rm`.
3. Reciprocal Rank Fusion merges both ranked lists.
4. If a local cross-encoder reranker is cached, it reranks the fused candidates before the final prompt is sent to Ollama.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running with `ollama serve`
- Pull the default model once: `ollama pull qwen2.5-coder:3b`
- The embedding model must already be cached under `models/` for strict offline use
- Optional reranker cache: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Usage

```bash
# 1. Drop PDFs, PPTX slides, and R files into ./data/
# 2. Build or rebuild the hybrid index
.venv/bin/python main.py ingest --rebuild

# 3. Ask a question
.venv/bin/python main.py query "How do I fit and interpret lm() in R?"

# Or start interactive mode
.venv/bin/python main.py query
```

## Folder descriptions

| Folder | Purpose |
|---|---|
| `data/` | Source PDFs, PowerPoint slides, and R scripts |
| `faiss_index/` | Auto-generated FAISS artifact, dense embedding matrix, and BM25-ready chunk payload |
| `models/` | Cached embedding model for offline use |

## Rebuilding the index

If you add or change files in `data/`, run:

```bash
.venv/bin/python main.py ingest --rebuild
```
