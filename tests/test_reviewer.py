"""Tests for metric extraction in the reviewer."""
import re

from core.reviewer import _extract_value, extract_metric_map, extract_metric_rows


def test_metric_rows_keep_real_metric_and_skip_references():
    text = (
        "The model reached 92% accuracy on the benchmark. "
        "Smith et al., 2019, doi: 10.1109/ABC.2019.123456."
    )
    rows = extract_metric_rows(text, "P1")
    findings = " ".join(r["finding"] for r in rows)
    assert "92% accuracy" in findings
    assert "doi" not in findings.lower()


def test_metric_rows_ignore_plain_numbers():
    # A number with no metric keyword or percent must not be treated as a metric.
    text = "The dataset contains 50000 documents split into three folders."
    assert extract_metric_rows(text, "P1") == []


def test_metric_map_extracts_values():
    text = "Our system achieved 88.5% accuracy. The F1 score was 0.91 on the test set."
    metrics = extract_metric_map(text)
    assert metrics.get("Accuracy", "").endswith("%")
    assert "0.91" in metrics.get("F1", "")


def test_extract_value_appends_percent_word():
    sentence = "It reached 14.15 percent accuracy overall"
    match = re.search(r"accuracy", sentence)
    assert _extract_value(sentence, match) == "14.15%"
