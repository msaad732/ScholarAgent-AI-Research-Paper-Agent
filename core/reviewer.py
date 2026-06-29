"""Generate a structured improvement review of an uploaded paper.

Approach A: a single grounded LLM call over representative sections of the user's
paper plus related excerpts retrieved from the corpus (and freshly fetched from
arxiv). No agentic tool loop — predictable structure, low token cost.
"""
import re

from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    REVIEW_PAPER_CHAR_BUDGET,
    REVIEW_RELATED_TOP_K,
)
from core import paper_fetcher
from core.pdf_parser import chunk_paper, extract_text_from_pdf
from core.uploader import derive_search_query
from core.vector_store import VectorStoreManager
from utils.helpers import get_logger, truncate

logger = get_logger(__name__)

_REVIEWER_SYSTEM = (
    "You are an experienced academic peer reviewer. You are given excerpts from a "
    "user's paper and excerpts from related papers in a knowledge base. Write a "
    "constructive, specific review. Ground every claim about related/prior work in "
    "the provided related excerpts and cite them by arxiv ID. Do not invent papers "
    "or results. If the related excerpts are sparse, say so honestly."
)

_REPORT_FORMAT = """Write the review in Markdown with EXACTLY these sections:

## Summary
A 2-3 sentence summary of what the paper does.

## Strengths
Bullet points of what the paper does well.

## Gaps & Weaknesses
Bullet points of limitations, unclear points, or missing analysis.

## Missing Related Work
Related papers (from the provided excerpts) the author should cite or compare
against. Cite each by arxiv ID and title. If none are relevant, say so.

## Methodology Notes
Specific observations about the methods/experiments and how they could be stronger.

## Concrete Improvement Suggestions
A numbered list of actionable, specific changes the author can make."""


# A percentage like "92%" or "90-95 %".
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
# A real metric keyword (must co-occur with a digit to count).
_METRIC_KW_RE = re.compile(
    r"\b(accuracy|f1[\s-]?score|f1|bleu|rouge|precision|recall|perplexity|"
    r"auc|map\b|mae|rmse|mse|wer|cer|top-?\d|error rate|"
    r"state[- ]of[- ]the[- ]art|sota)\b",
    re.IGNORECASE,
)
_DIGIT_RE = re.compile(r"\d")
# Lines that look like references/citations, which we must NOT treat as metrics.
_REFERENCE_RE = re.compile(
    r"\b(doi|et\s+al|pp\.|vol\.|no\.|arxiv|proceedings|conference|"
    r"isbn|issn|ieee|acm|springer|journal|preprint)\b",
    re.IGNORECASE,
)


# Named metric categories: (display name, pattern). Order controls table rows.
_METRIC_CATEGORIES = [
    ("Accuracy", re.compile(r"\baccuracy\b", re.IGNORECASE)),
    ("F1", re.compile(r"\bf1[\s-]?(?:score)?\b", re.IGNORECASE)),
    ("Precision", re.compile(r"\bprecision\b", re.IGNORECASE)),
    ("Recall", re.compile(r"\brecall\b", re.IGNORECASE)),
    ("ROC-AUC", re.compile(r"\b(?:roc[\s-]?auc|auc)\b", re.IGNORECASE)),
    ("BLEU", re.compile(r"\bbleu\b", re.IGNORECASE)),
    ("ROUGE", re.compile(r"\brouge\b", re.IGNORECASE)),
    ("Perplexity", re.compile(r"\bperplexity\b", re.IGNORECASE)),
    ("Error (RMSE/MAE/MSE)", re.compile(r"\b(?:rmse|mae|mse)\b", re.IGNORECASE)),
    ("WER/CER", re.compile(r"\b(?:wer|cer)\b", re.IGNORECASE)),
    ("Hit ratio/rate", re.compile(r"\bhit\s+(?:ratio|rate)\b", re.IGNORECASE)),
    ("Miss ratio/rate", re.compile(r"\bmiss\s+(?:ratio|rate)\b", re.IGNORECASE)),
    ("Latency/response time", re.compile(r"\b(?:latency|response time)\b", re.IGNORECASE)),
]

# Public list of metric display names in table order.
METRIC_NAMES = [name for name, _ in _METRIC_CATEGORIES]


# A numeric value, optionally a range, optionally a percent sign.
_VALUE_RE = re.compile(r"\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\s*%|\d+\.\d+|\d+")


def _extract_value(sentence: str, keyword_match: re.Match) -> str | None:
    """Return the number (with unit) nearest the metric keyword, or None.

    Numbers that are part of the keyword itself (e.g. the ``1`` in ``F1``) are
    skipped, and percentages/decimals are preferred over bare integers.
    """
    kw_start, kw_end = keyword_match.start(), keyword_match.end()
    candidates = [
        m
        for m in _VALUE_RE.finditer(sentence)
        if not (m.start() < kw_end and m.end() > kw_start)  # exclude keyword's own digits
    ]
    if not candidates:
        return None

    def _rank(m: re.Match) -> tuple:
        token = m.group()
        kind = 0 if "%" in token else (1 if "." in token else 2)  # %, decimal, int
        return (abs(m.start() - kw_start), kind)

    best = min(candidates, key=_rank)
    value = re.sub(r"\s+", "", best.group())
    # Attach a percent sign if the value is immediately followed by % / 'percent'.
    tail = sentence[best.end() : best.end() + 9].lstrip().lower()
    if not value.endswith("%") and (tail.startswith("%") or tail.startswith("percent")):
        value += "%"
    return value


def extract_metric_map(text: str) -> dict[str, str]:
    """Map each detected metric category to a concise value or short explanation.

    For each metric found, the value is the number (with unit) nearest the
    metric keyword; if no number can be tied to it, a one-line explanation is
    used instead.

    Args:
        text: Paper text to scan.

    Returns:
        A dict of ``{metric_name: value-or-explanation}`` for metrics found.
    """
    sentences = [" ".join(s.split()) for s in re.split(r"(?<=[.!?])\s+", text)]
    candidates = [
        s
        for s in sentences
        if 20 <= len(s) <= 260 and _DIGIT_RE.search(s) and not _REFERENCE_RE.search(s)
    ]
    found: dict[str, str] = {}
    for name, pattern in _METRIC_CATEGORIES:
        for sentence in candidates:
            match = pattern.search(sentence)
            if match:
                value = _extract_value(sentence, match)
                found[name] = value if value else truncate(sentence, 120)
                break
    return found


def extract_metric_rows(text: str, source_label: str, limit: int = 12) -> list[dict]:
    """Pull sentences that genuinely report numeric metrics.

    A sentence qualifies only if it contains a percentage, or a recognised
    metric keyword alongside a digit. Reference/citation lines (DOIs, page
    ranges, venue names) are skipped to avoid false positives.

    Args:
        text: Paper text to scan.
        source_label: Label identifying where the finding came from.
        limit: Maximum number of rows to return.

    Returns:
        A list of de-duplicated ``{"source", "finding"}`` dicts.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = " ".join(sentence.split())  # normalise whitespace
        if not (20 <= len(sentence) <= 260):
            continue
        if _REFERENCE_RE.search(sentence):
            continue
        has_percent = bool(_PERCENT_RE.search(sentence))
        has_metric = bool(_METRIC_KW_RE.search(sentence)) and bool(_DIGIT_RE.search(sentence))
        if not (has_percent or has_metric):
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"source": source_label, "finding": sentence})
        if len(rows) >= limit:
            break
    return rows


def _get_llm():
    """Construct a ChatGroq client, raising a clear error if the key is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file or Streamlit secrets."
        )
    from langchain_groq import ChatGroq

    return ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=LLM_TEMPERATURE)


def select_representative_text(text: str, char_budget: int = REVIEW_PAPER_CHAR_BUDGET) -> str:
    """Pick representative regions of a paper (opening + conclusion) within a budget.

    Args:
        text: Full cleaned paper text.
        char_budget: Maximum number of characters to return.

    Returns:
        A trimmed excerpt favouring the abstract/intro and the conclusion.
    """
    if len(text) <= char_budget:
        return text

    head_budget = int(char_budget * 0.65)
    head = text[:head_budget]

    # Try to append the conclusion region for the remaining budget.
    lower = text.lower()
    concl_idx = lower.rfind("conclusion")
    if concl_idx > head_budget:
        tail_budget = char_budget - head_budget
        tail = text[concl_idx : concl_idx + tail_budget]
        return f"{head}\n\n[... omitted ...]\n\n{tail}"
    return text[:char_budget]


def fetch_related_work(
    query: str, n: int, vector_store: VectorStoreManager, exclude_id: str
) -> list[dict]:
    """Fetch related papers from arxiv and index them into the knowledge base.

    Args:
        query: Search query derived from the uploaded paper.
        n: Number of related papers to fetch.
        vector_store: Vector store manager to index into.
        exclude_id: Paper id to skip (the user's own upload).

    Returns:
        A list of ``{arxiv_id, title}`` dicts for the related papers now indexed.
    """
    try:
        papers = paper_fetcher.fetch_and_download(query, n)
    except Exception as exc:  # noqa: BLE001
        logger.error("Related-work fetch failed: %s", exc)
        return []

    indexed: list[dict] = []
    for paper in papers:
        if paper["arxiv_id"] == exclude_id:
            continue
        try:
            if vector_store.is_paper_indexed(paper["arxiv_id"]):
                indexed.append({"arxiv_id": paper["arxiv_id"], "title": paper["title"]})
                continue
            text = extract_text_from_pdf(paper["local_path"])
            if not text:
                continue
            chunks = chunk_paper(text, paper["arxiv_id"], paper["title"], paper["authors"])
            if vector_store.add_paper(chunks):
                indexed.append({"arxiv_id": paper["arxiv_id"], "title": paper["title"]})
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to index related paper %s: %s", paper.get("arxiv_id"), exc)

    logger.info("Related-work fetch indexed %d papers", len(indexed))
    return indexed


def _format_related_excerpts(chunks: list[dict]) -> str:
    """Format retrieved related-paper excerpts for the review prompt."""
    if not chunks:
        return "(No related excerpts were found in the knowledge base.)"
    blocks = []
    for c in chunks:
        meta = c["metadata"]
        blocks.append(
            f"--- {meta.get('title', 'Unknown')} (arxiv:{meta.get('arxiv_id', '?')}) | "
            f"Section: {meta.get('section', 'Unknown')} ---\n{truncate(c['text'], 600)}"
        )
    return "\n\n".join(blocks)


# Aspects compared side-by-side across papers.
COMPARISON_FIELDS = [
    "Objective / Problem",
    "Methodology / Approach",
    "Datasets / Experimental Setup",
    "Key Results & Metrics",
    "Strengths",
    "Limitations",
    "Novelty",
]


def compare_papers_table(papers: list[dict]) -> str:
    """Generate a side-by-side comparison of papers across fixed aspects.

    Args:
        papers: A list of ``{"title", "text"}`` dicts (2 or more).

    Returns:
        A Markdown comparison (table + short verdict), or an error message.
    """
    if len(papers) < 2:
        return "Select at least two papers to compare them."

    blocks = "\n\n".join(
        f"=== PAPER {i + 1}: {p['title']} ===\n"
        f"{select_representative_text(p.get('text', ''))}"
        for i, p in enumerate(papers)
    )
    fields = "\n".join(f"- {f}" for f in COMPARISON_FIELDS)
    prompt = (
        "You are an academic reviewer comparing research papers. Compare the "
        "papers below across these aspects:\n"
        f"{fields}\n\n"
        "Produce a Markdown table with one ROW per aspect and one COLUMN per "
        "paper (use short paper labels like P1, P2 and list the mapping above the "
        "table). Keep cells concise. After the table, add a 2-3 sentence verdict "
        "on relative strengths and which paper suits which need. Base everything "
        "ONLY on the provided text; if an aspect is unclear for a paper, write "
        "'not stated'.\n\n"
        f"{blocks}\n\n=== COMPARISON ==="
    )
    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001
        logger.error("Paper comparison failed: %s", exc)
        return f"An error occurred while comparing the papers: {exc}"


def generate_review(
    vector_store: VectorStoreManager,
    paper_id: str,
    title: str,
    text: str,
    num_related: int,
) -> dict:
    """Fetch related work and produce a structured review of the uploaded paper.

    Args:
        vector_store: Initialised vector store manager.
        paper_id: Synthetic id of the uploaded paper (``upload_<hash>``).
        title: Title of the uploaded paper.
        text: Full cleaned text of the uploaded paper.
        num_related: Number of related papers to fetch from arxiv.

    Returns:
        A dict with ``report`` (markdown), ``related_papers`` and ``sources`` keys.
    """
    query = derive_search_query(title, text)
    related_papers = fetch_related_work(query, num_related, vector_store, paper_id)

    # Retrieve related excerpts from the corpus, excluding the user's own upload.
    related_chunks = vector_store.query(
        query, top_k=REVIEW_RELATED_TOP_K, filter_dict={"arxiv_id": {"$ne": paper_id}}
    )

    paper_excerpt = select_representative_text(text)
    related_block = _format_related_excerpts(related_chunks)

    fetch_note = (
        ""
        if related_papers
        else "\n(Note: no related papers could be fetched from arxiv; "
        "the comparison relies only on the existing knowledge base.)"
    )

    prompt = (
        f"{_REVIEWER_SYSTEM}\n\n"
        f"=== USER'S PAPER: {title} ===\n{paper_excerpt}\n\n"
        f"=== RELATED PAPERS (excerpts from the knowledge base) ===\n{related_block}\n\n"
        f"{_REPORT_FORMAT}{fetch_note}"
    )

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        report = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001
        logger.error("Review generation failed: %s", exc)
        report = f"An error occurred while generating the review: {exc}"

    sources: dict[str, dict] = {}
    for c in related_chunks:
        meta = c["metadata"]
        aid = meta.get("arxiv_id", "?")
        sources.setdefault(aid, {"arxiv_id": aid, "title": meta.get("title", "Unknown")})

    # Surface reported numbers from the user's paper and the related excerpts.
    metrics = extract_metric_rows(text, "Your paper")
    for c in related_chunks:
        meta = c["metadata"]
        metrics.extend(
            extract_metric_rows(c["text"], f"arxiv:{meta.get('arxiv_id', '?')}", limit=3)
        )

    return {
        "report": report,
        "related_papers": related_papers,
        "sources": list(sources.values()),
        "metrics": metrics,
    }
