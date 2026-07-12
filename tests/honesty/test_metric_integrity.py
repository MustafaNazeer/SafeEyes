"""Every reported headline number must match the committed metrics JSON it cites,
and the temporal accuracy must never appear without its false alarm rate beside it.
"""

import pytest

from ._docs import load_docs, load_metric, metric_value, render_pct, render_ratio

EYE_CARD = "docs/ml/eye-state-model-card.md"
TEMPORAL_METHODOLOGY = "docs/ml/temporal-classifier-methodology.md"

# (json file, key path, doc that reports it, render kind, decimals).
CLAIMS = [
    ("eye-state-metrics.json", ("overall_accuracy",), EYE_CARD, "pct", 2),
    ("eye-state-metrics.json", ("balanced_accuracy",), EYE_CARD, "pct", 2),
    ("eye-state-metrics.json", ("per_class_recall", "closed"), EYE_CARD, "pct", 2),
    ("eye-state-metrics.json", ("per_class_recall", "open"), EYE_CARD, "pct", 2),
    ("temporal-metrics.json", ("overall_accuracy",), TEMPORAL_METHODOLOGY, "pct", 1),
    ("temporal-metrics.json", ("macro_auroc",), TEMPORAL_METHODOLOGY, "ratio", 3),
    ("temporal-metrics.json", ("false_alarm_rate",), TEMPORAL_METHODOLOGY, "ratio", 3),
    ("temporal-metrics-gbt.json", ("overall_accuracy",), TEMPORAL_METHODOLOGY, "pct", 1),
    ("temporal-metrics-gbt.json", ("macro_auroc",), TEMPORAL_METHODOLOGY, "ratio", 3),
    ("temporal-metrics-gbt.json", ("false_alarm_rate",), TEMPORAL_METHODOLOGY, "ratio", 3),
]

TEMPORAL_ACCURACY = "54.5%"
TEMPORAL_FALSE_ALARM_RATE = "0.139"


def _render(value: float, kind: str, decimals: int) -> str:
    return render_pct(value, decimals) if kind == "pct" else render_ratio(value, decimals)


def _doc_text(doc_file: str) -> str:
    for name, text in load_docs():
        if name == doc_file:
            return text
    raise AssertionError(f"expected public doc not found: {doc_file}")


@pytest.mark.parametrize(("json_file", "json_path", "doc_file", "kind", "decimals"), CLAIMS)
def test_reported_number_matches_committed_metrics(
    json_file: str, json_path: tuple[str, ...], doc_file: str, kind: str, decimals: int
) -> None:
    value = metric_value(load_metric(json_file), json_path)
    rendered = _render(value, kind, decimals)
    assert rendered in _doc_text(doc_file), (
        f"{doc_file} should report {'.'.join(json_path)}={rendered} from {json_file}; "
        "regenerate the doc from the metrics JSON or the number has drifted"
    )


def test_temporal_accuracy_is_never_reported_without_false_alarm_rate() -> None:
    # Honesty rule: the false alarm rate is reported beside accuracy, never hidden.
    offenders = [
        name
        for name, text in load_docs()
        if TEMPORAL_ACCURACY in text and TEMPORAL_FALSE_ALARM_RATE not in text
    ]
    assert offenders == [], (
        f"these public docs state the temporal accuracy without its false alarm rate: {offenders}"
    )


def test_metric_check_catches_drift() -> None:
    # Positive control: a doc that misstates the number does not satisfy the check.
    value = metric_value(load_metric("eye-state-metrics.json"), ("overall_accuracy",))
    rendered = _render(value, "pct", 2)
    drifted_doc = "the eye state model reaches 99.99% accuracy on the held out split"
    assert rendered not in drifted_doc
