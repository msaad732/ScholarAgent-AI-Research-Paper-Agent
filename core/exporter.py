"""Export a paper review to Markdown or a simple PDF document."""
import re

from utils.helpers import get_logger

logger = get_logger(__name__)

# Map common non-latin-1 characters to ASCII so the PDF core fonts can render them.
_ASCII_MAP = {
    "—": "-", "–": "-", "…": "...", "•": "-", "→": "->", "←": "<-",
    "“": '"', "”": '"', "‘": "'", "’": "'", "≈": "~", "×": "x", "±": "+/-",
}


def review_to_markdown(review: dict) -> str:
    """Render a review dict as a self-contained Markdown document.

    Args:
        review: A review dict with ``title``, ``report`` and ``related_papers``.

    Returns:
        A Markdown string.
    """
    lines = [f"# Review: {review.get('title', 'Uploaded Paper')}", ""]

    related = review.get("related_papers", [])
    if related:
        lines.append("## Related Work Fetched From arxiv")
        for p in related:
            lines.append(f"- {p['title']} (arxiv:{p['arxiv_id']})")
        lines.append("")

    metrics = review.get("metrics", [])
    if metrics:
        lines.append("## Reported Numbers / Metrics")
        lines.append("")
        lines.append("| Source | Finding |")
        lines.append("| --- | --- |")
        for m in metrics:
            finding = m["finding"].replace("|", "\\|")
            lines.append(f"| {m['source']} | {finding} |")
        lines.append("")

    lines.append(review.get("report", ""))
    return "\n".join(lines)


def _sanitize(text: str) -> str:
    """Replace non-latin-1 characters so the PDF core fonts can render the text."""
    for src, dst in _ASCII_MAP.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "ignore").decode("latin-1")


def markdown_to_pdf_bytes(markdown_text: str, title: str) -> bytes | None:
    """Render Markdown-ish text into a simple PDF byte string.

    Headings (``#``/``##``/``###``), bullet lists, numbered lists, and tables are
    given light formatting; everything else is rendered as paragraphs.

    Args:
        markdown_text: The Markdown content to render.
        title: Document title shown at the top.

    Returns:
        The PDF as bytes, or None if the ``fpdf`` package is unavailable.
    """
    try:
        from fpdf import FPDF
    except Exception as exc:  # noqa: BLE001
        logger.error("fpdf2 not available for PDF export: %s", exc)
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 9, _sanitize(title))
    pdf.ln(2)

    for raw in markdown_text.splitlines():
        line = _sanitize(raw.rstrip())
        stripped = line.strip()
        if not stripped:
            pdf.ln(3)
            continue
        if stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 15)
            pdf.ln(2)
            pdf.multi_cell(0, 8, stripped[2:])
        elif stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.ln(1)
            pdf.multi_cell(0, 7, stripped[4:])
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.ln(2)
            pdf.multi_cell(0, 7, stripped[3:])
        elif stripped.startswith(("- ", "* ")):
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, f"  - {stripped[2:]}")
        elif re.match(r"^\d+\.\s", stripped):
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, f"  {stripped}")
        elif stripped.startswith("|"):
            # Render table rows as plain " | "-separated text (skip separators).
            if set(stripped) <= set("|-: "):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, " | ".join(cells))
        else:
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, stripped)

    return bytes(pdf.output())
