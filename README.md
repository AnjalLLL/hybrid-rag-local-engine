# Offline RAG for R Exam Preparation

This project builds a local Retrieval-Augmented Generation (RAG) workflow for R programming exam prep. It ingests PDFs, PowerPoint slides, and R scripts from `data/`, stores local dense embeddings plus BM25 keyword tokens, and answers questions with a local Ollama model.

## Features

- **Fully offline.** After a one-time model download, nothing leaves your machine. No API keys, no cloud calls.
- **Hybrid retrieval.** Dense semantic search and BM25 keyword search are fused, so both "explain overfitting" and an exact term like `na.rm` land on the right pages.
- **Cross-encoder reranking.** The top fused candidates are re-scored by a local cross-encoder before they reach the model, which is the single biggest accuracy win in the pipeline.
- **Code-aware chunking.** R code is chunked by lines with indentation and newlines preserved; prose is chunked by sentence. Code no longer gets flattened into an unreadable single line.
- **Deterministic answers.** Decoding runs at `temperature=0.1`, so the same exam question gives the same answer every run instead of drifting.
- **Grounded and cited.** Every answer prints the source files and page numbers it drew from, so you can check it against the actual notes.
- **Mixed sources.** Indexes PDFs, PowerPoint decks, and `.R` scripts, plus `.Rmd`, `.txt`, and `.md`.
- **Honest ingest.** Files it cannot read (typically scanned, image-only PDFs) are reported by name instead of silently vanishing from the index.
- **Auto model detection.** Picks a generation model that is actually installed in Ollama, preferring a code-tuned one.
- **Near-duplicate filtering.** Overlapping chunks are dropped so the limited context is not spent twice on the same text.
- **Exam-depth aware.** Auto-detects whether a question is a short, direct 3-mark question or a
  long, multi-part 6-mark question (from an explicit marks tag or the number of lettered
  sub-parts), and shapes the answer's length and format accordingly.
- **Sub-part continuity.** Multi-part questions get every lettered sub-part extracted into an
  explicit checklist so the model can't silently skip one, and are instructed to reuse the same
  variable/model names across sub-parts instead of redefining them.
- **Curated anti-hallucination reference.** A hand-verified library of correct package/function
  usage (`src/r_reference.py`) for topics prone to invented arguments or missing `library()`
  calls -- k-means/cluster plotting, `multinom`, `caret`, decision trees, random forest, `knn`,
  `lda`/`qda`, ROC curves, PCA, `igraph`, `apriori`, and more -- is injected into the prompt
  whenever a question matches, regardless of what retrieval finds.
- **Optional execution validation.** With `--verify-r`, generated R code is actually run with
  `Rscript`; a genuine code error triggers one bounded self-correction attempt, while
  missing-data/missing-package errors (expected for a hypothetical exam dataset) are left alone.

## Performance

Measured on this machine (Apple Silicon, CPU only) over a 3,369-chunk index built from 59 files:

| Stage | Time |
|---|---|
| Startup (load index, embedder, and reranker) | ~2.2s, once per session |
| Retrieval (hybrid search + rerank of 30 candidates) | ~0.5s per question |
| Full answer, including the LLM writing it out | ~10s per question |

Retrieval itself is the cheap part; nearly all of the wall clock is the local LLM generating tokens.
Interactive mode (`main.py query` with no question) pays the startup cost once and then answers
follow-up questions at retrieval speed, and the Ollama model is kept warm for 10 minutes between
questions so it does not reload. Answers stream token by token, so text starts appearing well
before the 10s mark.

## How retrieval works

The query engine uses hybrid retrieval:

1. Dense embedding search finds semantically similar chunks (exact inner-product search over the stored embedding matrix).
2. BM25 sparse search finds exact R terms such as `lm`, `ggplot`, `filter`, and `na.rm`.
3. Weighted Reciprocal Rank Fusion merges both ranked lists.
4. A local cross-encoder reranks the top fused candidates before the final prompt is sent to Ollama.
5. Near-duplicate chunks are dropped so the few context slots are not spent twice on the same text.

Chunking is content-aware: R code is chunked by lines so indentation and newlines survive, while prose is chunked by sentences.

## Exam-aware answering

Before retrieval runs, each question is analyzed (`analyze_question()` in `src/query.py`):

- **Depth detection.** An explicit `(6 marks)` / `[3]` tag is used if present; otherwise the
  question is classified as long/multi-part if it has 2+ lettered sub-parts (`a)`, `b)`, `c)`...),
  short otherwise. Long questions get more retrieved context (`top_k=10` vs `6`) since they need
  to be grounded across more sub-parts. Override the guess with `--marks 3` or `--marks 6`.
- **Sub-part checklist.** Lettered sub-parts are split out and injected into the prompt as an
  explicit list the model must fully address, anchoring the "reuse objects from earlier parts"
  rule.
- **Reference lookup.** Topic keywords from the question are matched against
  `src/r_reference.py`; a match injects a verified, minimal-args code skeleton for that
  function/package directly into the prompt, so the model has a correct example to follow even
  when the retrieved course material doesn't cover that exact topic.
- **Identifier boost.** Dataset and function names mentioned in the question (e.g. `USArrests`,
  `multinom(`) are force-included in the retrieval candidates if an exact-match chunk exists,
  even if fusion/reranking would have otherwise ranked it below the cutoff.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running with `ollama serve`
- At least one generation model pulled. A code-tuned model gives noticeably better R answers:
  `ollama pull qwen2.5-coder:7b`. If you do not pass `--ollama-model`, the CLI auto-detects an
  installed model and prefers a coder model when one is available.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Usage

```bash
# 1. Download the embedding + reranker models once (needs network)
.venv/bin/python main.py fetch-models

# 2. Drop PDFs, PPTX slides, and R files into ./data/
# 3. Build or rebuild the hybrid index
.venv/bin/python main.py ingest --rebuild

# 4. Ask a question
.venv/bin/python main.py query "How do I fit and interpret lm() in R?"

# Or start interactive mode
.venv/bin/python main.py query

# Force exam depth and verify the generated R code actually runs
.venv/bin/python main.py query "Fit a multinomial logistic regression..." --marks 6 --verify-r
```

Once `fetch-models` has run, everything works fully offline.

Useful flags: `--ollama-model` to pick a model, `--top-k` to change how many chunks are sent to
the model, `--marks {3,6}` to force short/long exam-answer formatting instead of auto-detecting
it, `--verify-r` to execute the generated R code with `Rscript` and request one self-correction
on a real error, and `--model-path` for a llama-cpp-python GGUF fallback if Ollama is down.

## Folder descriptions

| Folder | Purpose |
|---|---|
| `data/` | Source PDFs, PowerPoint slides, and R scripts |
| `faiss_index/` | Dense embedding matrix, FAISS artifact, and BM25-ready chunk payload |
| `models/` | Cached embedding and reranker models for offline use |

## Rebuilding the index

If you add or change files in `data/`, run:

```bash
.venv/bin/python main.py ingest --rebuild
```

Ingest prints a warning listing any file it could not extract text from. Those are almost always
scanned/image-only PDFs, which need OCR before they can be indexed.

## Note on FAISS

`ingest` writes `faiss_index/index.faiss`, but queries search the stored embedding matrix with
NumPy rather than calling into FAISS. `faiss-cpu` and `torch` both link `libomp`, and calling
FAISS after torch is loaded segfaults the process on macOS/ARM. The saved index is an
`IndexFlatIP` (exact inner product), so the NumPy matrix multiply returns identical results, and
at this corpus size it takes about a millisecond.
