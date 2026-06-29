# Design — Upload & Review Your Own Paper

**Date:** 2026-06-28
**Feature:** Let a user upload their own research paper (PDF), ingest it, auto-fetch
related work from arxiv, and generate a structured improvement review.

## Goal

Add a one-click "Analyze My Paper" flow: upload a PDF → it is indexed into the
knowledge base (tagged as an upload) → related papers are fetched from arxiv →
a single grounded LLM call produces a structured critique (gaps, missing related
work, methodology notes, concrete suggestions).

## Decisions (from brainstorming)

- **Trigger:** dedicated "Analyze My Paper" button (not chat-only).
- **Compare scope:** auto-fetch related papers from arxiv first, then compare.
- **Storage:** persist the upload in ChromaDB, tagged `source_type="upload"`.
- **Review generation:** Approach A — one structured LLM call (no agentic tool
  loop), avoiding the Groq `tool_use_failed` flakiness and keeping token cost low.

## Components

### 1. `config.py` (additions)
- `UPLOAD_RELATED_DEFAULT = 3` — related papers fetched per analysis.
- `REVIEW_PAPER_CHAR_BUDGET = 6000` — cap on uploaded-paper text sent to the LLM.
- `REVIEW_RELATED_TOP_K = 6` — related excerpts retrieved from the corpus.
- `ANALYZE_LIMIT_PER_SESSION = 2` — analyses/min per session (also consumes the
  global daily cap, since analyze = fetch + index + a large LLM call).

### 2. `core/pdf_parser.py` (change)
- `chunk_paper(..., source_type: str = "arxiv")` — adds `source_type` to each
  chunk's metadata. Default keeps existing arxiv behaviour unchanged.

### 3. `core/uploader.py` (new)
- `ingest_uploaded_pdf(vector_store, file_bytes, filename) -> dict`
  - Save to `data/papers/upload_<hash>.pdf` (hash of bytes → stable id, cache).
  - Extract text, derive title from filename, synthetic id `upload_<hash>`.
  - Chunk with `source_type="upload"`, add to ChromaDB.
  - Returns `{paper_id, title, num_chunks, text}`.
- `derive_search_query(title, text) -> str` — build an arxiv query from the
  title (and a few salient keywords) for the related-work fetch.

### 4. `core/reviewer.py` (new)
- `fetch_related_work(query, n, vector_store, exclude_id) -> list[dict]`
  - `paper_fetcher.fetch_and_download` → parse → chunk (`source_type="arxiv"`) →
    index. Returns the related papers indexed.
- `select_representative_text(text, char_budget) -> str`
  - Prefer abstract/intro/method/conclusion regions; truncate to budget.
- `generate_review(vector_store, paper_id, title, text, num_related) -> dict`
  - Derive query → fetch related work → retrieve related excerpts from the
    corpus (`vector_store.query` filtered with `arxiv_id != paper_id`).
  - Build a structured prompt and make ONE `ChatGroq` call.
  - Returns `{report (markdown), related_papers, sources}`.
  - Report sections: Summary · Strengths · Gaps & Weaknesses · Missing Related
    Work (with arxiv citations) · Methodology Notes · Concrete Suggestions.

### 5. `core/vector_store.py` (change)
- `get_all_papers` returns `source_type` (default `"arxiv"`) so the sidebar can
  label uploads. Querying excludes the upload via a `{"arxiv_id": {"$ne": id}}`
  filter (Chroma `$ne`).

### 6. `app.py` (change)
- New sidebar section "📤 Upload Your Paper": `st.file_uploader` (PDF), a
  "related papers to fetch" number input (default 3), and "🔬 Analyze My Paper".
- On click: enforce the analyze limiter → ingest → `generate_review` (spinner) →
  store report in `st.session_state["review"]`.
- Main area renders the persisted report in a container with a "Dismiss" button.
- Sidebar KB list labels uploads as "(your upload)" with a 🗑️ remove button and
  no arxiv link.

## Data flow

```
Upload PDF
  → ingest_uploaded_pdf (save, extract, chunk[source_type=upload], index)
  → derive_search_query
  → fetch_related_work (arxiv search + download + index)
  → vector_store.query (related excerpts, exclude own chunks)
  → generate_review (1 ChatGroq call)
  → markdown report rendered in main area + persisted in session_state
```

## Error handling
- Non-PDF / unreadable upload → friendly sidebar error, no crash.
- Empty text extraction → warn, abort analysis.
- arxiv fetch failure → proceed with corpus-only comparison, note it in the report.
- Missing GROQ key / LLM error → surfaced as a clear message (reuse existing pattern).
- Rate limit hit → "try again in Ns" message.

## Token-budget rationale (Groq free tier: 12K tokens/min, 100K/day)
- Paper text capped at ~6000 chars (~1.5K tokens) + ~6 related excerpts (~1.5K
  tokens) + instructions + ~800 output ≈ 4–5K tokens per analysis — one call,
  within the per-minute limit. The analyze limiter + daily cap bound total usage.

## Out of scope (YAGNI)
- Multi-call per-dimension review (Approach C).
- Inline PDF annotation / diffing.
- Non-PDF formats (docx, LaTeX source).
