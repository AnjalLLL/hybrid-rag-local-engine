"""Retrieve relevant chunks and answer questions with a local LLM."""

from __future__ import annotations

import json
import pickle
import re
import site
import sys
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.append(user_site)

import numpy as np
import requests
from rank_bm25 import BM25Okapi

from src.r_reference import match_reference_entries, format_reference_block
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
TOP_K_LONG = 10
RETRIEVAL_CANDIDATES = 60
RERANK_CANDIDATES = 30
RRF_K = 60
DENSE_WEIGHT = 1.0
BM25_WEIGHT = 0.7
MAX_CHUNK_CHARS = 1500
# Total retrieved-context budget stays roughly constant regardless of top_k, so a bigger
# top_k (more chunks for a long question) doesn't linearly balloon the volume of context
# sitting between the rules and the question -- a big context-to-question ratio is what
# lets the model lose track of a short, specific instruction inside the question.
CONTEXT_CHAR_BUDGET = 9000
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

MARKS_TAG_RE = re.compile(r"\(?\s*(\d+)\s*marks?\s*\)?|\[\s*(\d+)\s*\]", re.IGNORECASE)
# Matches "a)", "(a)", and "a." (the last one requires trailing whitespace so it doesn't
# fire on abbreviations like "e.g." or "i.e.").
SUB_PART_RE = re.compile(r"(?:^|\s|\()([a-h])(?:\)|\.\s)\s*", re.IGNORECASE)
IDENTIFIER_CALL_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_.]*)\s*\(")
PROPER_NOUN_RE = re.compile(r"\b([A-Z][A-Za-z0-9]{2,})\b")

# Topic-agnostic: these don't know what "single graph" or "four variables" mean, they just
# flag that a clause carries an explicit count or a restrictive word, so it can be isolated
# and echoed back verbatim instead of getting diluted inside a much larger block of retrieved
# context. This is deliberately generic rather than a per-topic rule, so it catches whatever
# minor constraint the next question happens to include, not just ones already anticipated.
NUMBER_TOKEN_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.IGNORECASE
)
CONSTRAINT_KEYWORD_RE = re.compile(
    r"\b(single|only|exactly|each|every|both|same|combined|together|"
    r"at least|at most|one graph|one plot|one line|in one)\b",
    re.IGNORECASE,
)


class QuestionInfo(NamedTuple):
    """Lightweight analysis of a question used to shape retrieval and the prompt."""

    depth: str  # "short" (~3 marks) or "long" (~6 marks)
    marks: Optional[int]
    sub_parts: List[str]
    topic_tokens: List[str]
    identifier_tokens: List[str]
    literal_requirements: List[str]


def analyze_question(question: str, marks_override: Optional[int] = None) -> QuestionInfo:
    """Classify a question's depth/marks, split its sub-parts, and pull retrieval hints.

    Detection order: an explicit CLI override, then an explicit "(N marks)"/"[N]" tag in
    the question text, then a fallback heuristic (>=2 lettered sub-parts implies a long,
    multi-part 6-mark question; otherwise it's treated as a short 3-mark question).
    """

    sub_parts = _split_sub_parts(question)

    marks = marks_override
    if marks is None:
        tag_match = MARKS_TAG_RE.search(question)
        if tag_match:
            marks = int(tag_match.group(1) or tag_match.group(2))

    if marks is not None:
        depth = "long" if marks >= 5 else "short"
    else:
        depth = "long" if len(sub_parts) >= 2 else "short"

    topic_tokens = tokenize_text(question)
    identifier_tokens = sorted(
        {match.group(1) for match in IDENTIFIER_CALL_RE.finditer(question)}
        | {match.group(1) for match in PROPER_NOUN_RE.finditer(question)}
    )
    literal_requirements = extract_literal_requirements(question)

    return QuestionInfo(
        depth=depth,
        marks=marks,
        sub_parts=sub_parts,
        topic_tokens=topic_tokens,
        identifier_tokens=identifier_tokens,
        literal_requirements=literal_requirements,
    )


def extract_literal_requirements(question: str) -> List[str]:
    """Pull out question clauses that carry an explicit count or a restrictive word.

    This has no idea what "single graph" or "four variables" mean -- it just notices a
    clause has a number or a word like "single"/"only"/"each" and keeps it as its own line,
    verbatim, so it survives being placed next to a much larger block of retrieved context
    instead of getting lost inside a long sentence. Works the same regardless of topic.
    """

    clauses = re.split(r"[;,]", question)
    requirements: List[str] = []
    seen: set = set()

    for clause in clauses:
        text = clause.strip()
        if not text:
            continue
        if NUMBER_TOKEN_RE.search(text) or CONSTRAINT_KEYWORD_RE.search(text):
            key = text.lower()
            if key not in seen:
                seen.add(key)
                requirements.append(text)

    return requirements


def _split_sub_parts(question: str) -> List[str]:
    """Split a question into its lettered sub-parts, e.g. "a) ...", "b) ...".

    Returns an empty list when the question has no lettered sub-parts.
    """

    markers = list(SUB_PART_RE.finditer(question))
    if len(markers) < 2:
        return []

    parts = []
    for index, marker in enumerate(markers):
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(question)
        label = marker.group(1).lower()
        text = question[start:end].strip().rstrip(";").strip()
        if text:
            parts.append(f"{label}) {text}")
    return parts


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
        except Exception as exc:
            print(f"Embedding retriever unavailable ({exc}). Falling back to offline keyword retrieval.")

        if self.reranker is not None:
            self.retrieval_mode += "+rerank"
        else:
            print(
                "Reranker not cached. Run `python main.py fetch-models` once (online) "
                "for noticeably better answers."
            )

        self.ollama_model = resolve_ollama_model(ollama_model)

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        marks_override: Optional[int] = None,
        verify_r: bool = False,
    ) -> int:
        """Retrieve context and stream an answer for a question."""

        question_info = analyze_question(question, marks_override=marks_override)
        resolved_top_k = top_k or (TOP_K_LONG if question_info.depth == "long" else TOP_K)

        print(
            f"Retrieving top {resolved_top_k} chunks using {self.retrieval_mode} retrieval "
            f"({question_info.depth} question, {len(question_info.sub_parts)} sub-part(s))...\n"
        )
        results = self.retrieve(question, top_k=resolved_top_k, question_info=question_info)

        if not results:
            print("No relevant context found in the index for this question.")
            return 1

        chunk_char_limit = max(600, min(MAX_CHUNK_CHARS, CONTEXT_CHAR_BUDGET // resolved_top_k))
        prompt = build_prompt(
            question, results, chunk_char_limit=chunk_char_limit, question_info=question_info
        )

        print("Answer:")
        answer_text = ""
        try:
            answer_text = stream_ollama_response(prompt, self.ollama_model, num_ctx=OLLAMA_NUM_CTX)
        except requests.HTTPError as exc:
            status_code, error_text = extract_http_error(exc)
            if status_code == 500 and "runner process has terminated" in error_text.lower():
                print("\nRetrying with a smaller offline prompt for low-memory mode...")
                compact_prompt = build_prompt(
                    question,
                    results,
                    chunk_char_limit=RETRY_CHUNK_CHARS,
                    question_info=question_info,
                )
                try:
                    answer_text = stream_ollama_response(
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

        if verify_r and answer_text:
            self._verify_r_code(answer_text, prompt)

        print("\n\nSources:")
        for source, page in format_sources(results):
            print(f"  - {source}, page {page}")

        return 0

    def _verify_r_code(self, answer_text: str, original_prompt: str) -> None:
        """Execute the answer's R code with Rscript and request one correction on failure."""

        blocks = extract_r_code_blocks(answer_text)
        if not blocks:
            return

        combined_code = "\n\n".join(blocks)
        ok, stderr = run_r_code(combined_code)
        if ok:
            print("\n\n[R check: code executed without errors.]")
            return

        if not classify_r_error(stderr):
            detail = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
            print(
                "\n\n[R check: execution failed for a likely non-code reason (missing data file "
                f"or uninstalled package), skipping auto-fix. Details: {detail}]"
            )
            return

        print(f"\n\n[R check: found a code error, requesting one correction...\n{stderr.strip()}]\n")
        not_found_match = re.search(r"object '([^']+)' not found", stderr)
        not_found_hint = (
            f"The undefined object '{not_found_match.group(1)}' is used but never actually "
            "assigned anywhere in the code -- this is usually a variable you described "
            "extracting (e.g. in a comment) but forgot to write the real assignment line for. "
            "Add the missing assignment itself, not just a reference to the name.\n\n"
            if not_found_match
            else ""
        )
        correction_prompt = (
            f"{original_prompt}\n\n"
            "=== PREVIOUS ANSWER FAILED TO RUN ===\n"
            f"R error:\n{stderr.strip()}\n\n"
            f"{not_found_hint}"
            "Fix only what is broken and give the complete corrected answer, following the "
            "same rules as before.\n"
            "=== CORRECTED ANSWER ===\n"
        )
        try:
            corrected_text = stream_ollama_response(
                correction_prompt, self.ollama_model, num_ctx=OLLAMA_NUM_CTX
            )
        except requests.RequestException as exc:
            print(f"\n[R check: correction attempt failed: {exc}]")
            return

        corrected_blocks = extract_r_code_blocks(corrected_text)
        if not corrected_blocks:
            return
        ok_after, _ = run_r_code("\n\n".join(corrected_blocks))
        status = "now executes cleanly" if ok_after else "still has an issue -- review manually"
        print(f"\n\n[R check: corrected answer {status}.]")

    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
        question_info: Optional[QuestionInfo] = None,
    ) -> List[ChunkPayload]:
        """Retrieve the top chunks for a question."""

        identifier_tokens: List[str] = []
        if question_info is not None:
            # Topic keywords go first: they're curated, high-signal markers of "this is the
            # taught technique" (e.g. "kmeans"), so they should claim a boost slot before
            # generic identifiers like a dataset name that might just appear in passing on
            # many unrelated pages. Otherwise a question about "USArrests" never pulls in
            # the course's own worked k-means example just because it demonstrates on "iris"
            # instead. This ties into whatever topics r_reference.py already knows about, so
            # it generalizes to any topic added there, not just k-means.
            for entry in match_reference_entries(question_info.topic_tokens):
                identifier_tokens.extend(entry.keywords)
            identifier_tokens.extend(question_info.identifier_tokens)

        return retrieve_chunks(
            question,
            self.embedder,
            self.chunks,
            self.dense_embeddings,
            self.bm25,
            self.reranker,
            top_k=top_k,
            identifier_tokens=identifier_tokens,
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
    identifier_tokens: Sequence[str] = (),
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

    if identifier_tokens:
        ranked_indices = _boost_identifier_matches(ranked_indices, chunks, bm25_indices, identifier_tokens)

    selected = dedupe_indices(ranked_indices, chunks)[:top_k]
    return [chunks[idx] for idx in selected]


def _boost_identifier_matches(
    ranked_indices: List[int],
    chunks: Sequence[ChunkPayload],
    candidate_indices: Sequence[int],
    identifier_tokens: Sequence[str],
    max_boosted: int = 4,
) -> List[int]:
    """Move chunks that exactly mention a question identifier/topic keyword to the front.

    RRF fusion and cross-encoder reranking can both push an exact-name or exact-topic match
    (e.g. "USArrests", "kmeans(") below the final top_k cutoff if the reranker judges other
    chunks more semantically similar to the literal question wording -- this happens even
    when the match is already present somewhere in ranked_indices, not just when it's
    missing entirely, so a match already ranked low still needs promoting, not just chunks
    that are absent.

    Slots are allocated per distinct token (one chunk per token, in token order) rather than
    as one shared budget -- a shared budget lets an early, common token (like a dataset name
    that happens to appear on many pages) consume every slot before a later, more specific
    token (like the actual method name) ever gets a turn.
    """

    lowered_tokens = [token.lower() for token in identifier_tokens if len(token) >= 3]
    if not lowered_tokens:
        return ranked_indices

    boosted: List[int] = []
    boosted_set: set = set()
    for token in lowered_tokens:
        if len(boosted) >= max_boosted:
            break
        # Prefer an actual runnable-code match (.R/.Rmd) over a PDF/slide match: a worked
        # code example is what a student should mirror, and PDF theory pages tend to rank
        # ahead of it in BM25 order just because there are more of them in the corpus.
        code_match = None
        any_match = None
        for idx in candidate_indices:
            if idx in boosted_set:
                continue
            text = str(chunks[idx]["text"]).lower()
            if token not in text:
                continue
            if any_match is None:
                any_match = idx
            if str(chunks[idx]["metadata"].get("kind")) in ("r", "rmd") and code_match is None:
                code_match = idx
                break
        chosen = code_match if code_match is not None else any_match
        if chosen is not None:
            boosted.append(chosen)
            boosted_set.add(chosen)

    if not boosted:
        return ranked_indices

    remainder = [idx for idx in ranked_indices if idx not in boosted_set]
    return boosted + remainder


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
    question_info: Optional[QuestionInfo] = None,
) -> str:
    """Construct the RAG prompt from retrieved context and the user question."""

    if question_info is None:
        question_info = analyze_question(question)

    context_blocks = []
    for position, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]
        context_blocks.append(
            f"[{position}] Source: {metadata['source']}, page {metadata['page']}\n"
            f"{truncate_context(str(chunk['text']), chunk_char_limit)}"
        )

    context = "\n\n".join(context_blocks)

    reference_entries = match_reference_entries(question_info.topic_tokens)
    reference_block = format_reference_block(reference_entries)
    reference_section = (
        f"=== VERIFIED LIBRARY REFERENCE (use these exact functions/arguments if relevant) ===\n"
        f"{reference_block}\n\n"
        if reference_block
        else ""
    )

    checklist_section = ""
    if question_info.sub_parts:
        checklist = "\n".join(f"- {part}" for part in question_info.sub_parts)
        checklist_section = (
            "=== SUB-PARTS TO ANSWER (every one of these must appear in the answer, "
            "under its own label, reusing objects created in earlier parts) ===\n"
            f"{checklist}\n\n"
        )

    requirements_section = ""
    if question_info.literal_requirements:
        requirements = "\n".join(f"- {req}" for req in question_info.literal_requirements)
        requirements_section = (
            "=== LITERAL REQUIREMENTS FROM THE QUESTION (each of these is a word-for-word "
            "constraint pulled out of the question below because it is easy to miss -- your "
            "answer must satisfy every one of them exactly, not just approximately) ===\n"
            f"{requirements}\n\n"
        )

    if question_info.depth == "long":
        depth_rule = (
            "12. This is a long-form, multi-part exam question. For each sub-part listed above: "
            "give its code first, then a short exam-style interpretation for that sub-part "
            "directly underneath, before moving to the next sub-part. Reuse the same variable "
            "and model/object names you defined in earlier sub-parts -- never redefine or "
            "rename them."
        )
    else:
        depth_rule = (
            "12. This is a short exam question worth few marks. Answer directly and concisely: "
            "the minimum correct code needed, plus a one-line example of the syntax if it "
            "clarifies usage, and no extended prose."
        )

    return (
        "You are an R programming examiner and tutor. Answer the question exactly as asked, "
        "using the reference material below as your primary source.\n\n"
        "The exact question you must answer (read it carefully -- every word can carry a "
        f"requirement, e.g. a specific count or 'only one of X'): {question}\n\n"
        "RULES:\n"
        "1. Prefer the context. If the context covers the question, follow it and cite the "
        "source number like [2].\n"
        "2. The context is course material, not the question's data. If the context does not "
        "cover the question, answer from standard R knowledge and say which part was not in "
        "the notes. Never refuse just because the context is thin.\n"
        "3. Use only the variables, data, and file names given in the question. Do not invent "
        "data, column names, or file formats, and do not add steps that were not asked for.\n"
        "4. Every non-base-R function must be preceded by an explicit, real library(pkg) call, "
        "and every library(pkg) call must be for a package you actually call a function from -- "
        "no unused imports. Never invent a function argument -- only use arguments that are "
        "genuinely documented for that function. If the VERIFIED LIBRARY REFERENCE section "
        "below covers the topic, use its exact function/argument names instead of guessing.\n"
        "5. Check a variable's type before applying a type-specific function (for example, "
        "levels() only works on a factor, not a numeric or character column). Apply "
        "na.omit()/complete.cases() before any test or model that breaks on missing values "
        "(for example shapiro.test, t.test, lm).\n"
        "6. Distance-based methods (kmeans, hclust, knn, pam) treat variables with larger "
        "numeric ranges as more important unless you scale() the numeric predictors first -- "
        "do this whenever the question's variables are on different measurement units, which "
        "is the normal case for real datasets.\n"
        "7. Never chain $ twice to pull a value out of a test/model result (e.g. "
        "by(...)$p.value or summary(aov_model)$Group$`Pr(>F)`) -- both are invalid or "
        "silently wrong. If the VERIFIED LIBRARY REFERENCE below covers the exact extraction, "
        "use it; otherwise just print() the result and read the value from the printed "
        "output instead of extracting it programmatically.\n"
        "8. If the question has lettered sub-parts, answer under the same labels, for example "
        "(a), (b), (c), and address every sub-part listed below -- do not skip any requested "
        "computation, plot, or interpretation. If it has no parts, do not invent part labels.\n"
        "9. If a sub-part's text says 'plot', 'graph', or 'visualize', that sub-part's code must "
        "contain an actual call that renders a plot -- do not stop at fitting the model. If the "
        "question asks for a 'single graph'/'single plot' over 3+ variables, note that "
        "plot(df, col = ...) on the WHOLE multi-column data frame is wrong because it draws a "
        "scatterplot matrix, not one graph -- follow whatever single-plot approach the CONTEXT "
        "and VERIFIED LIBRARY REFERENCE below actually demonstrate for this exact technique "
        "(e.g. plotting two chosen variables in base R, or a dedicated multi-variable plotting "
        "function) rather than guessing a library that isn't shown there.\n"
        "10. Any interpretation must state the concrete pattern implied by the question/data, "
        "naming the actual variables and a direction (higher/lower, increases/decreases). Bad: "
        "'the data is divided into two clusters.' Good: 'cluster 1 has higher Murder and "
        "Assault rates than cluster 2, so it groups higher-crime states separately from "
        "lower-crime states.' Never submit an interpretation that would be equally true for "
        "any dataset -- it must reference this question's specific variables.\n"
        "11. Before writing your final answer, re-read the QUESTION and the LITERAL "
        "REQUIREMENTS section below (when present) line by line, and check your draft answer "
        "against each one -- a count, a word like 'single'/'only'/'each', or a sub-part is "
        "easy to lose track of once you're deep in the code, so verify it explicitly rather "
        "than trusting your first pass.\n"
        f"{depth_rule}\n"
        "13. Output only the answer. Start directly with the code or the answer text. Do not "
        "restate these rules and do not write a preamble such as 'Here is the answer'.\n\n"
        "=== CONTEXT ===\n"
        f"{context}\n\n"
        f"{reference_section}"
        f"{checklist_section}"
        f"{requirements_section}"
        "=== QUESTION ===\n"
        f"{question}\n\n"
        "=== ANSWER ===\n"
    )


def stream_ollama_response(prompt: str, model: str, num_ctx: int) -> str:
    """Stream a response from the local Ollama server, returning the full text."""

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

    pieces: List[str] = []
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
            pieces.append(text)

    return "".join(pieces)


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


CODE_FENCE_RE = re.compile(r"```(?:[rR]|[Rr]script)?\s*\n(.*?)```", re.DOTALL)
NON_ACTIONABLE_ERROR_MARKERS = (
    "cannot open file",
    "cannot open connection",
    "there is no package called",
    "unable to find an inherited method",  # usually a missing-package symptom
)
ACTIONABLE_ERROR_MARKERS = (
    "unexpected symbol",
    "unexpected string constant",
    "unexpected '",
    "could not find function",
    "unused argument",
    "argument .* matches multiple formal arguments",
    "object '",
)


def extract_r_code_blocks(answer_text: str) -> List[str]:
    """Pull fenced R code blocks out of a generated answer."""

    return [block.strip() for block in CODE_FENCE_RE.findall(answer_text) if block.strip()]


def run_r_code(code: str, timeout: int = 20) -> Tuple[bool, str]:
    """Execute R code with Rscript in a throwaway temp file and return (ok, stderr)."""

    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False, encoding="utf-8") as handle:
        handle.write(code)
        script_path = handle.name

    try:
        result = subprocess.run(
            ["Rscript", "--vanilla", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stderr
    except FileNotFoundError:
        # Rscript isn't installed; nothing to verify against, so don't block the answer.
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Execution timed out (likely an infinite loop or blocking input() call)."
    finally:
        Path(script_path).unlink(missing_ok=True)


def classify_r_error(stderr: str) -> bool:
    """Return True if stderr looks like an actionable code bug worth a correction retry.

    Missing data files and uninstalled packages are legitimate for a hypothetical exam
    question's dataset and aren't something the model can fix by rewriting the code, so
    those are treated as non-actionable and skipped rather than retried.
    """

    lowered = stderr.lower()
    if any(marker in lowered for marker in NON_ACTIONABLE_ERROR_MARKERS):
        return False
    return any(re.search(marker, lowered) for marker in ACTIONABLE_ERROR_MARKERS)


def format_sources(retrieved_chunks: Iterable[ChunkPayload]) -> List[Tuple[str, int]]:
    """Return sorted unique source/page pairs from retrieved chunks."""

    pairs = {
        (str(chunk["metadata"]["source"]), int(chunk["metadata"]["page"]))
        for chunk in retrieved_chunks
    }
    return sorted(pairs, key=lambda item: (item[0].lower(), item[1]))
