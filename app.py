"""ScholarAgent — Streamlit frontend and main entry point."""
import base64
import os
import uuid

import streamlit as st

from config import (
    ANALYZE_LIMIT_PER_SESSION,
    COLLECTION_NAME,
    DAILY_WINDOW,
    FETCH_LIMIT_PER_SESSION,
    GROQ_API_KEY,
    RATE_LIMIT_GLOBAL,
    RATE_LIMIT_GLOBAL_DAILY,
    RATE_LIMIT_PER_SESSION,
    RATE_LIMIT_WINDOW,
    STATIC_DIR,
    UPLOAD_RELATED_DEFAULT,
)
from core.agent import build_agent, run_agent
from core.exporter import markdown_to_pdf_bytes, review_to_markdown
from core.paper_fetcher import fetch_and_download
from core.pdf_parser import chunk_paper, extract_text_from_pdf
from core.rag_chain import query_and_respond, stream_answer
from core.reviewer import (
    METRIC_NAMES,
    compare_papers_table,
    extract_metric_map,
    generate_review,
)
from core.uploader import ingest_uploaded_pdf
from core.vector_store import VectorStoreManager
from utils.helpers import RateLimiter, get_logger, truncate

logger = get_logger("app")

EXAMPLE_TOPICS = [
    "Transformer architectures",
    "Reinforcement learning from human feedback",
    "Diffusion models for image generation",
]

st.set_page_config(
    page_title="ScholarAgent — AI Research Paper Agent",
    page_icon="📚",
    layout="wide",
)


def get_session_id() -> str:
    """Return a stable per-browser-session id (created once per session)."""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex[:12]
    return st.session_state["session_id"]


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    """Return a single shared ChromaDB client (one per process)."""
    import chromadb

    from config import CHROMA_PERSIST_DIR

    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


@st.cache_resource(show_spinner=False)
def get_embedding_fn():
    """Return a single shared embedding model (loaded once per process)."""
    from chromadb.utils import embedding_functions

    from config import EMBEDDING_MODEL

    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False, max_entries=100)
def get_vector_store(session_id: str) -> VectorStoreManager:
    """Return a vector store with a per-session collection (isolated per user).

    The heavy client + embedding model are shared across sessions; only the
    collection differs, so memory does not grow per user.
    """
    return VectorStoreManager(
        collection_name=f"{COLLECTION_NAME}_{session_id}",
        client=get_chroma_client(),
        embedding_fn=get_embedding_fn(),
    )


@st.cache_resource(show_spinner=False, max_entries=100)
def get_agent(session_id: str, _vector_store: VectorStoreManager):
    """Return a compiled agent bound to this session's vector store.

    ``session_id`` is the cache key; ``_vector_store`` is excluded from hashing
    (leading underscore) but used to build the agent.
    """
    return build_agent(_vector_store)


@st.cache_resource(show_spinner=False)
def get_global_limiters() -> dict[str, RateLimiter]:
    """Return process-wide rate limiters shared across ALL user sessions."""
    return {
        "chat": RateLimiter(RATE_LIMIT_GLOBAL, RATE_LIMIT_WINDOW),
        "chat_daily": RateLimiter(RATE_LIMIT_GLOBAL_DAILY, DAILY_WINDOW),
    }


def get_session_limiters() -> dict[str, RateLimiter]:
    """Return per-session rate limiters, ensuring every expected key exists."""
    limiters = st.session_state.setdefault("limiters", {})
    defaults = {
        "chat": (RATE_LIMIT_PER_SESSION, RATE_LIMIT_WINDOW),
        "fetch": (FETCH_LIMIT_PER_SESSION, RATE_LIMIT_WINDOW),
        "analyze": (ANALYZE_LIMIT_PER_SESSION, RATE_LIMIT_WINDOW),
    }
    for key, (max_calls, window) in defaults.items():
        if key not in limiters:
            limiters[key] = RateLimiter(max_calls, window)
    return limiters


def build_paper_context(vector_store: VectorStoreManager) -> str:
    """Build a note telling the agent which indexed papers are the user's uploads.

    Args:
        vector_store: The shared vector store manager.

    Returns:
        A system-note string, or an empty string if no uploads are indexed.
    """
    uploads = [
        p for p in vector_store.get_all_papers() if p.get("source_type") == "upload"
    ]
    if not uploads:
        return ""
    listing = "; ".join(f"'{p['title']}' (id: {p['arxiv_id']})" for p in uploads)
    return (
        "IMPORTANT CONTEXT: The user has uploaded their OWN paper(s) into the "
        f"knowledge base: {listing}. When the user says 'my paper' or 'this paper', "
        "they mean the uploaded paper(s). In tool results, excerpts from these are "
        "marked 'YOUR PAPER'. You can search them and compare them against the other "
        "indexed papers just like any other source."
    )


def analyze_uploaded_papers(vector_store: VectorStoreManager, files: list, num_related: int) -> None:
    """Ingest one or more uploaded PDFs, fetch related work, and review each.

    Stores reviews in ``st.session_state['reviews']`` keyed by paper id.

    Args:
        vector_store: The shared vector store manager.
        files: A list of Streamlit ``UploadedFile`` objects (PDFs).
        num_related: Number of related arxiv papers to fetch per paper.
    """
    # Throttle once per batch: analyze = fetch + index + a large LLM call.
    if not get_session_limiters()["analyze"].allow():
        wait = get_session_limiters()["analyze"].retry_after()
        st.sidebar.warning(f"Please wait {wait}s before analyzing more papers.")
        return

    reviews = st.session_state.setdefault("reviews", {})
    done = 0
    skipped = 0
    for file in files:
        file_bytes = file.getvalue()
        with st.spinner(f"Reading '{truncate(file.name, 30)}'…"):
            ingested = ingest_uploaded_pdf(vector_store, file_bytes, file.name)
        if not ingested:
            st.sidebar.error(
                f"Could not extract text from '{truncate(file.name, 30)}'. It may be "
                "empty, corrupted, password-protected, or a scanned image with no "
                "text layer. Skipping."
            )
            continue

        publish_to_static(ingested["paper_id"], file_bytes)
        st.session_state["current_review"] = ingested["paper_id"]

        # Already reviewed this session → reuse it; don't spend another LLM call.
        if ingested["paper_id"] in reviews:
            skipped += 1
            continue

        # New paper needs a review (related-work fetch + LLM call) — gate on the cap.
        if not get_global_limiters()["chat_daily"].allow():
            st.sidebar.warning("Daily free-tier cap reached — stopping here.")
            break
        with st.spinner(f"Fetching related work and reviewing '{truncate(ingested['title'], 30)}'…"):
            result = generate_review(
                vector_store,
                ingested["paper_id"],
                ingested["title"],
                ingested["text"],
                num_related,
            )
        reviews[ingested["paper_id"]] = {
            "paper_id": ingested["paper_id"],
            "title": ingested["title"],
            "local_path": ingested["local_path"],
            "report": result["report"],
            "related_papers": result["related_papers"],
            "sources": result["sources"],
            "metrics": result.get("metrics", []),
        }
        done += 1

    if done:
        st.sidebar.success(
            f"Reviewed {done} paper(s) — see the **📄 My Paper & Review** tab."
        )
    if skipped:
        st.sidebar.info(
            f"{skipped} paper(s) were already analyzed — showing the existing review. "
            "Remove and re-upload to force a fresh analysis."
        )


def publish_to_static(paper_id: str, file_bytes: bytes) -> str:
    """Copy a PDF into the Streamlit static dir so it has a browser URL.

    Args:
        paper_id: Synthetic upload id, used as the file name.
        file_bytes: Raw PDF bytes.

    Returns:
        The relative static URL (``app/static/<paper_id>.pdf``).
    """
    os.makedirs(STATIC_DIR, exist_ok=True)
    dest = os.path.join(STATIC_DIR, f"{paper_id}.pdf")
    try:
        if not (os.path.exists(dest) and os.path.getsize(dest) == len(file_bytes)):
            with open(dest, "wb") as fh:
                fh.write(file_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to publish PDF to static: %s", exc)
    return f"app/static/{paper_id}.pdf"


def ingest_topic(vector_store: VectorStoreManager, topic: str, num_papers: int) -> None:
    """Fetch, parse, chunk, embed, and store papers for a topic, with UI progress.

    Args:
        vector_store: The shared vector store manager.
        topic: Research topic to search arxiv for.
        num_papers: Number of papers to fetch.
    """
    topic = topic.strip()
    if not topic:
        st.sidebar.warning("Please enter a research topic first.")
        return

    fetch_limiter = get_session_limiters()["fetch"]
    if not fetch_limiter.allow():
        st.sidebar.warning(
            f"You're fetching too quickly. Please wait {fetch_limiter.retry_after()}s "
            "before fetching more papers."
        )
        return

    progress = st.sidebar.progress(0.0, text="Searching arxiv…")
    try:
        papers = fetch_and_download(topic, num_papers)
    except Exception as exc:  # noqa: BLE001
        logger.error("Fetch failed: %s", exc)
        progress.empty()
        st.sidebar.error(f"Failed to fetch papers: {exc}")
        return

    if not papers:
        progress.empty()
        st.sidebar.error("No papers found or downloads failed. Try a different topic.")
        return

    indexed = 0
    total = len(papers)
    for i, paper in enumerate(papers, start=1):
        frac = i / total
        progress.progress(frac, text=f"Indexing {i}/{total}: {truncate(paper['title'], 40)}")
        try:
            if vector_store.is_paper_indexed(paper["arxiv_id"]):
                continue
            text = extract_text_from_pdf(paper["local_path"])
            if not text:
                logger.warning("No text extracted from %s", paper["arxiv_id"])
                continue
            chunks = chunk_paper(text, paper["arxiv_id"], paper["title"], paper["authors"])
            if vector_store.add_paper(chunks):
                indexed += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to index %s: %s", paper.get("arxiv_id"), exc)

    progress.empty()
    if indexed:
        st.sidebar.success(f"Indexed {indexed} new paper(s) on '{topic}'.")
    else:
        st.sidebar.info("No new papers added (they may already be in the knowledge base).")


def render_sidebar(vector_store: VectorStoreManager) -> None:
    """Render the paper-management sidebar (search/ingest + knowledge base)."""
    st.sidebar.title("📚 Paper Management")

    # --- Search & Ingest ---
    st.sidebar.subheader("Search & Ingest")
    # An example chip queues a topic; apply it to the widget BEFORE instantiation
    # (Streamlit forbids modifying a widget's state after it is created).
    auto_fetch = False
    if "pending_topic" in st.session_state:
        st.session_state["topic_input"] = st.session_state.pop("pending_topic")
        auto_fetch = True

    topic = st.sidebar.text_input("Enter a research topic", key="topic_input")
    num_papers = st.sidebar.number_input(
        "Number of papers to fetch", min_value=1, max_value=10, value=5, step=1
    )
    if st.sidebar.button("🔎 Fetch Papers", use_container_width=True) or auto_fetch:
        ingest_topic(vector_store, topic, int(num_papers))
        st.rerun()

    st.sidebar.divider()

    # --- Upload & Review your own paper(s) ---
    st.sidebar.subheader("📤 Upload Your Paper(s)")
    uploaded = st.sidebar.file_uploader(
        "Upload one or more PDFs to review",
        type=["pdf"],
        key="paper_upload",
        accept_multiple_files=True,
    )
    num_related = st.sidebar.number_input(
        "Related papers to fetch (each)",
        min_value=0,
        max_value=6,
        value=UPLOAD_RELATED_DEFAULT,
        step=1,
        help="Fetched from arxiv to ground the 'missing related work' section.",
    )
    if st.sidebar.button("🔬 Analyze My Paper(s)", use_container_width=True):
        if not uploaded:
            st.sidebar.warning("Please choose at least one PDF first.")
        else:
            analyze_uploaded_papers(vector_store, uploaded, int(num_related))
            st.rerun()

    st.sidebar.divider()

    # --- Knowledge Base ---
    st.sidebar.subheader("Knowledge Base")
    stats = vector_store.get_collection_stats()
    st.sidebar.caption(
        f"**{stats['total_papers']}** papers · **{stats['total_chunks']}** chunks indexed"
    )

    papers = vector_store.get_all_papers()
    if not papers:
        st.sidebar.info("No papers indexed yet.")
        return

    for paper in papers:
        is_upload = paper.get("source_type") == "upload"
        col1, col2 = st.sidebar.columns([5, 1])
        with col1:
            if is_upload:
                st.markdown(
                    f"⬆️ **{truncate(paper['title'], 60)}**  \n_(your upload)_"
                )
            else:
                # Title links to the live arxiv abstract page (opens in a new tab).
                url = f"https://arxiv.org/abs/{paper['arxiv_id']}"
                st.markdown(
                    f"📄 **[{truncate(paper['title'], 60)}]({url})**  \n"
                    f"[Read on arxiv ↗]({url})",
                    unsafe_allow_html=False,
                )
        with col2:
            if st.button("🗑️", key=f"del_{paper['arxiv_id']}", help="Remove paper"):
                vector_store.delete_paper(paper["arxiv_id"])
                st.rerun()


def render_empty_state() -> None:
    """Render the welcome/empty state with clickable example topics."""
    st.markdown("### 👋 Welcome to ScholarAgent!")
    st.markdown(
        "Start by fetching some papers using the sidebar, or pick an example topic below."
    )
    cols = st.columns(len(EXAMPLE_TOPICS))
    for col, topic in zip(cols, EXAMPLE_TOPICS):
        with col:
            if st.button(topic, use_container_width=True, key=f"chip_{topic}"):
                st.session_state["pending_topic"] = topic
                st.rerun()


_SUGGESTIONS_HEADER = "## Concrete Improvement Suggestions"


def _split_report(report: str) -> tuple[str, str]:
    """Split a review report into (main body, improvement-suggestions) sections.

    Args:
        report: The full markdown report.

    Returns:
        A ``(body, suggestions)`` tuple. ``suggestions`` is empty if the
        expected header is absent.
    """
    idx = report.find(_SUGGESTIONS_HEADER)
    if idx == -1:
        return report, ""
    body = report[:idx].rstrip()
    suggestions = report[idx + len(_SUGGESTIONS_HEADER) :].strip()
    return body, suggestions


def render_pdf_viewer(review: dict) -> None:
    """Render the uploaded PDF inline (base64) plus a download button.

    Uses a base64 data URI rather than Streamlit static serving, which is
    unreliable behind hosting proxies (e.g. Hugging Face Spaces).
    """
    path = review.get("local_path")
    if not path or not os.path.exists(path):
        st.caption("PDF file is no longer available (the app may have restarted).")
        return

    with open(path, "rb") as fh:
        data = fh.read()

    st.download_button(
        "⬇️ Download PDF",
        data,
        file_name=f"{review.get('title', 'paper')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    # Inline preview via base64 data URI (works without static serving).
    b64 = base64.b64encode(data).decode()
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600" '
        f'style="border:1px solid #ddd;border-radius:8px;"></iframe>',
        unsafe_allow_html=True,
    )


def render_review_tab() -> None:
    """Render the 'My Paper & Review' tab: selector, PDF, review, metrics, export."""
    reviews = st.session_state.get("reviews", {})
    if not reviews:
        st.info(
            "📤 Upload one or more PDFs in the sidebar and click **Analyze My "
            "Paper(s)** to see your paper(s) and an improvement review here."
        )
        return

    # Paper selector (when more than one upload).
    ids = list(reviews.keys())
    if len(ids) > 1:
        labels = {pid: truncate(reviews[pid]["title"], 50) for pid in ids}
        current = st.session_state.get("current_review", ids[-1])
        if current not in ids:
            current = ids[-1]
        selected = st.selectbox(
            "Select an uploaded paper",
            ids,
            index=ids.index(current),
            format_func=lambda pid: labels[pid],
        )
        st.session_state["current_review"] = selected
    else:
        selected = ids[0]

    review = reviews[selected]

    header_col, dismiss_col = st.columns([6, 1])
    with header_col:
        st.subheader(f"🔬 {review['title']}")
    with dismiss_col:
        if st.button("🗑️ Remove", key="dismiss_review", use_container_width=True):
            reviews.pop(selected, None)
            st.rerun()

    related = review.get("related_papers", [])
    if related:
        links = ", ".join(
            f"[{truncate(p['title'], 40)}](https://arxiv.org/abs/{p['arxiv_id']})"
            for p in related
        )
        st.caption(f"Related work fetched from arxiv: {links}")

    # The review text is the important part — render it FIRST, full width, so
    # nothing below (export buttons, PDF viewer) can blank it out if it errors.
    st.markdown("#### 📝 Review")
    body, suggestions = _split_report(review.get("report", "") or "")
    st.markdown(body or "_No review text was produced — try re-analysing._")
    if suggestions:
        st.markdown("### 🛠️ Improvement Suggestions (action items)")
        st.warning(suggestions)

    # Export buttons — isolated so a PDF/export hiccup can't break the page.
    try:
        render_review_export(review)
    except Exception as exc:  # noqa: BLE001
        logger.error("Review export failed: %s", exc)
        st.caption("Export is temporarily unavailable.")

    # PDF viewer — isolated, in an expander, base64 (no static-serving dependency).
    if review.get("paper_id"):
        with st.expander("📄 View your uploaded PDF"):
            try:
                render_pdf_viewer(review)
            except Exception as exc:  # noqa: BLE001
                logger.error("PDF viewer failed: %s", exc)
                st.caption("Couldn't display the PDF inline — try downloading it.")

    st.divider()
    st.info(
        "💬 Ask about your paper in the **Chat** tab, or compare it with other "
        "papers in the **🔀 Compare** tab."
    )


def render_review_export(review: dict) -> None:
    """Render Markdown / PDF download buttons for a review."""
    md = review_to_markdown(review)
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in review["title"])[:50]
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        st.download_button(
            "⬇️ Markdown",
            md.encode("utf-8"),
            file_name=f"review_{safe_name}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with c2:
        pdf_bytes = markdown_to_pdf_bytes(md, f"Review: {review['title']}")
        if pdf_bytes:
            st.download_button(
                "⬇️ PDF",
                pdf_bytes,
                file_name=f"review_{safe_name}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.caption("PDF export needs `fpdf2`.")


def render_metrics_matrix(
    vector_store: VectorStoreManager, selected: list[str], title_by_id: dict
) -> None:
    """Render a metric-by-paper matrix: one row per metric, one column per paper.

    Args:
        vector_store: The shared vector store manager.
        selected: Selected paper ids.
        title_by_id: Map of paper id -> title.
    """
    st.markdown("#### 📊 Reported Numbers / Metrics")
    st.caption("Each row is a metric; each column is a paper. Cells show one detail line.")

    # Extract a metric -> detail map for each selected paper.
    maps = {pid: extract_metric_map(vector_store.get_paper_text(pid)) for pid in selected}
    present = [name for name in METRIC_NAMES if any(name in maps[pid] for pid in selected)]
    if not present:
        st.info("No clearly-reported metrics detected in the selected papers.")
        return

    col_labels = {pid: truncate(title_by_id.get(pid, pid), 28) for pid in selected}
    rows = []
    for name in present:
        row = {"Metric": name}
        for pid in selected:
            row[col_labels[pid]] = truncate(maps[pid].get(name, "—"), 160)
        rows.append(row)

    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_compare_tab(vector_store: VectorStoreManager) -> None:
    """Render the 'Compare' tab: pick papers, compare side-by-side, see metrics."""
    papers = vector_store.get_all_papers()
    if len(papers) < 2:
        st.info(
            "🔀 Add at least two papers (fetch from arxiv or upload your own) to "
            "compare them here."
        )
        return

    labels = {
        p["arxiv_id"]: ("📄 " if p.get("source_type") == "upload" else "")
        + truncate(p["title"], 55)
        for p in papers
    }
    selected = st.multiselect(
        "Choose papers to compare (2 or more)",
        options=list(labels.keys()),
        format_func=lambda pid: labels[pid],
    )

    if len(selected) >= 2:
        if st.button("🔀 Compare selected papers", use_container_width=False):
            if not get_global_limiters()["chat_daily"].allow():
                st.warning("Daily free-tier cap reached — try again later.")
            else:
                try:
                    title_by_id = {p["arxiv_id"]: p["title"] for p in papers}
                    docs = [
                        {"title": title_by_id.get(pid, pid), "text": vector_store.get_paper_text(pid)}
                        for pid in selected
                    ]
                    with st.spinner("Comparing papers across key aspects…"):
                        st.session_state["comparison_result"] = {
                            "ids": selected,
                            "markdown": compare_papers_table(docs),
                        }
                except Exception as exc:  # noqa: BLE001
                    logger.error("Comparison failed: %s", exc)
                    st.error(f"Comparison failed: {exc}")
    else:
        st.caption("Select two or more papers above to enable comparison.")

    result = st.session_state.get("comparison_result")
    if result and result.get("markdown"):
        st.markdown("### 🔀 Comparison")
        st.markdown(result["markdown"])

    # Metrics matrix for the selected papers (regex-based, no LLM call).
    if selected:
        st.divider()
        try:
            title_by_id = {p["arxiv_id"]: p["title"] for p in papers}
            render_metrics_matrix(vector_store, selected, title_by_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Metrics matrix failed: %s", exc)
            st.caption("Couldn't extract metrics for the selected papers.")


_AVATARS = {"user": "🧑", "assistant": "📚"}


_ALL_PAPERS = "All papers"


def render_chat(vector_store: VectorStoreManager) -> None:
    """Render a chat window: per-paper filter, scrollable history, input, chips."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    papers = vector_store.get_all_papers()

    # Per-paper scope filter.
    scope_id = None
    if papers:
        options = {_ALL_PAPERS: None}
        for p in papers:
            label = ("📄 " if p.get("source_type") == "upload" else "") + truncate(
                p["title"], 45
            )
            options[label] = p["arxiv_id"]
        choice = st.selectbox(
            "Scope", list(options.keys()), help="Limit answers to a single paper."
        )
        scope_id = options[choice]

    # Scrollable conversation window — results appear above the input box.
    chat_window = st.container(height=340, border=True)
    with chat_window:
        if not st.session_state["messages"]:
            st.caption("💬 Your conversation will appear here. Ask a question below.")
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"], avatar=_AVATARS.get(msg["role"])):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("meta"):
                    _render_meta(msg["meta"])

    # Suggested-question chips (only before the first message).
    if papers and not st.session_state["messages"]:
        st.caption("Try one of these:")
        chip_cols = st.columns(2)
        for i, q in enumerate(suggested_questions(papers)):
            with chip_cols[i % 2]:
                if st.button(q, key=f"suggest_{i}", use_container_width=True):
                    st.session_state["pending_prompt"] = q
                    st.rerun()

    typed = st.chat_input("Ask anything across your indexed papers…")
    prompt = typed or st.session_state.pop("pending_prompt", None)
    if not prompt:
        return

    if vector_store.get_collection_stats()["total_papers"] == 0:
        st.warning("Please fetch some papers (or upload one) first using the sidebar.")
        return

    blocked = _check_chat_limits()
    if blocked:
        st.warning(blocked)
        return

    # Render the new turn into the same scrollable window.
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with chat_window:
        with st.chat_message("user", avatar=_AVATARS["user"]):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=_AVATARS["assistant"]):
            answer, meta = _stream_or_answer(vector_store, prompt, scope_id)
            _render_meta(meta)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "meta": meta}
    )


def _stream_or_answer(
    vector_store: VectorStoreManager, prompt: str, scope_id: str | None
) -> tuple[str, dict]:
    """Answer a query, streaming direct (single-call) answers token-by-token.

    Design note: the multi-step agent runs NON-streamed — streaming an LLM
    tool-call makes llama on Groq emit a malformed function format
    (``tool_use_failed``). Single-call RAG (scoped queries and the agent's
    fallback) has no tools bound, so it streams reliably.

    Returns:
        A ``(answer_text, meta)`` tuple where ``meta`` holds tool calls + sources.
    """
    # Scoped to a single paper: single-call RAG → stream it.
    if scope_id:
        try:
            gen, sources = stream_answer(prompt, vector_store, {"arxiv_id": scope_id})
            answer = st.write_stream(gen)
            return answer, {
                "tool_calls": [
                    {"tool": "search (scoped to selected paper)", "input": {"query": prompt}}
                ],
                "sources": sources,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scoped streaming failed (%s); using non-streaming.", exc)
            result = _answer_query(vector_store, prompt, scope_id)
            st.markdown(result["answer"])
            return result["answer"], {
                "tool_calls": result.get("tool_calls", []),
                "sources": result.get("sources", []),
            }

    # All papers: run the agent non-streamed (reliable tool calls).
    agent = get_agent(get_session_id(), vector_store)
    history = st.session_state["messages"][:-1]
    paper_context = build_paper_context(vector_store)
    result = run_agent(agent, prompt, history, paper_context=paper_context)
    if not result.get("error"):
        st.markdown(result["answer"])
        return result["answer"], {
            "tool_calls": result.get("tool_calls", []),
            "sources": result.get("sources", []),
        }

    # Agent failed (e.g. tool_use_failed) → stream a plain RAG fallback answer.
    logger.warning("Agent failed; streaming RAG fallback.")
    try:
        gen, sources = stream_answer(prompt, vector_store)
        answer = st.write_stream(gen)
        return answer, {
            "tool_calls": [
                {"tool": "search_knowledge_base (fallback)", "input": {"query": prompt}}
            ],
            "sources": sources,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("RAG fallback streaming failed: %s", exc)
        rag = query_and_respond(prompt, vector_store)
        st.markdown(rag["answer"])
        return rag["answer"], {"tool_calls": [], "sources": rag["sources"]}


def suggested_questions(papers: list[dict]) -> list[str]:
    """Build a few cheap, guiding starter questions from the indexed papers."""
    suggestions: list[str] = []
    uploads = [p for p in papers if p.get("source_type") == "upload"]
    if uploads:
        suggestions.append("How does my paper compare to the other papers?")
        suggestions.append("What related work is my paper missing?")
    if len(papers) > 1:
        suggestions.append("Compare the approaches across the papers.")
    suggestions.append("What are the key contributions of each paper?")
    suggestions.append("What accuracy or metrics are reported?")
    return suggestions[:4]


def _check_chat_limits() -> str:
    """Apply the chat rate limiters; return a warning string if blocked, else ''."""
    if not get_session_limiters()["chat"].allow():
        wait = get_session_limiters()["chat"].retry_after()
        return f"⏳ You're sending messages too fast. Please wait {wait}s and try again."
    if not get_global_limiters()["chat"].allow():
        wait = get_global_limiters()["chat"].retry_after()
        return f"🚦 ScholarAgent is busy right now. Please retry in about {wait}s."
    if not get_global_limiters()["chat_daily"].allow():
        hours = max(1, get_global_limiters()["chat_daily"].retry_after() // 3600)
        return f"📊 Today's free-tier usage cap has been reached. Try again in ~{hours}h."
    return ""


def _answer_query(
    vector_store: VectorStoreManager, prompt: str, scope_id: str | None = None
) -> dict:
    """Answer a query, optionally scoped to a single paper.

    When ``scope_id`` is set, retrieval is restricted to that paper via plain RAG
    (the multi-step agent is bypassed since the scope is explicit). Otherwise the
    paper-aware agent runs, falling back to plain RAG on tool-call failure.
    """
    try:
        if scope_id:
            rag = query_and_respond(prompt, vector_store, filter_dict={"arxiv_id": scope_id})
            return {
                "answer": rag["answer"],
                "tool_calls": [
                    {"tool": "search (scoped to selected paper)", "input": {"query": prompt}}
                ],
                "sources": rag["sources"],
            }

        agent = get_agent(get_session_id(), vector_store)
        history = st.session_state["messages"][:-1]
        paper_context = build_paper_context(vector_store)
        result = run_agent(agent, prompt, history, paper_context=paper_context)
        # If the agent's tool-calling failed, fall back to plain RAG so the user
        # still gets a grounded answer.
        if result.get("error"):
            logger.warning("Agent failed; falling back to direct RAG.")
            rag = query_and_respond(prompt, vector_store)
            return {
                "answer": rag["answer"],
                "tool_calls": [
                    {"tool": "search_knowledge_base (fallback)", "input": {"query": prompt}}
                ],
                "sources": rag["sources"],
            }
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error("Agent error: %s", exc)
        return {"answer": f"Something went wrong: {exc}", "tool_calls": [], "sources": []}


def _render_meta(meta: dict) -> None:
    """Render the tools-used / sources expander beneath an assistant answer."""
    tool_calls = meta.get("tool_calls", [])
    sources = meta.get("sources", [])
    if not tool_calls and not sources:
        return
    with st.expander("🔍 Tools used & sources"):
        if tool_calls:
            st.markdown("**Tools Used:**")
            for call in tool_calls:
                st.markdown(f"- `{call['tool']}` — {call.get('input', {})}")
        if sources:
            st.markdown("**Sources:**")
            for src in sources:
                section = src.get("section", "Unknown")
                if src.get("is_own"):
                    st.markdown(f"- 📄 **Your paper** — _{section}_")
                elif src.get("arxiv_id"):
                    url = f"https://arxiv.org/abs/{src['arxiv_id']}"
                    st.markdown(
                        f"- **{src.get('title', 'Unknown')}** — _{section}_ "
                        f"([arxiv:{src['arxiv_id']} ↗]({url}))"
                    )
                else:
                    st.markdown(f"- **{src.get('title', 'Unknown')}** — _{section}_")
                snippet = src.get("snippet")
                if snippet:
                    st.markdown(
                        f"<blockquote style='margin:2px 0 8px 14px;color:#555;"
                        f"font-size:0.85em;'>{truncate(snippet, 280)}</blockquote>",
                        unsafe_allow_html=True,
                    )


def main() -> None:
    """Application entry point."""
    st.title("📚 ScholarAgent — AI Research Paper Agent")
    st.caption("Point it at a topic → it builds a knowledge base → ask anything across all papers.")

    if not GROQ_API_KEY:
        st.error(
            "**GROQ_API_KEY is not set.**\n\n"
            "Add your Groq API key to a `.env` file in the project root:\n\n"
            "```\nGROQ_API_KEY=gsk_your_actual_key_here\n```\n\n"
            "Or, on Streamlit Cloud, add it under **App Settings → Secrets**. "
            "Get a free key at https://console.groq.com."
        )

    vector_store = get_vector_store(get_session_id())
    render_sidebar(vector_store)

    # Header stats strip + About expander.
    stats = vector_store.get_collection_stats()
    uploads = sum(
        1 for p in vector_store.get_all_papers() if p.get("source_type") == "upload"
    )
    s1, s2, s3 = st.columns(3)
    s1.metric("Papers indexed", stats["total_papers"])
    s2.metric("Chunks", stats["total_chunks"])
    s3.metric("Your uploads", uploads)
    with st.expander("ℹ️ About ScholarAgent"):
        st.markdown(
            "ScholarAgent fetches papers from **arxiv**, embeds them locally into "
            "**ChromaDB**, and answers across them with a **LangGraph** agent "
            "(LLaMA-3 via Groq). Retrieved chunks are **reranked** with a "
            "cross-encoder for precision. Upload your own paper for a structured "
            "review, and use the **Compare** tab for side-by-side analysis."
        )

    review_count = len(st.session_state.get("reviews", {}))
    review_label = "📄 My Paper & Review" + (f" ({review_count})" if review_count else "")
    chat_tab, review_tab, compare_tab = st.tabs(
        ["💬 Chat", review_label, "🔀 Compare"]
    )

    with chat_tab:
        if st.session_state.get("messages"):
            col1, col2 = st.columns([6, 1])
            with col2:
                if st.button("🧹 Clear chat", use_container_width=True):
                    st.session_state["messages"] = []
                    st.rerun()

        stats = vector_store.get_collection_stats()
        if stats["total_papers"] == 0 and not st.session_state.get("messages"):
            render_empty_state()

        render_chat(vector_store)

    with review_tab:
        render_review_tab()

    with compare_tab:
        render_compare_tab(vector_store)


if __name__ == "__main__":
    main()
