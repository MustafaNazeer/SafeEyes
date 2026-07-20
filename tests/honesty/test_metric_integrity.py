"""Every reported headline number must match the committed metrics JSON it cites,
and the temporal accuracy must never appear without its false alarm rate beside it.
"""

import pytest

from ._docs import (
    ALERT_VALIDATION,
    DISTRACTION_CARD,
    YAWN_CARD,
    load_docs,
    load_metric,
    metric_value,
    render_pct,
    render_ratio,
)

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
    (
        "distraction-mobilenet_v3_small-metrics.json",
        ("overall_accuracy",),
        DISTRACTION_CARD,
        "pct",
        2,
    ),
    (
        "distraction-mobilenet_v3_small-metrics.json",
        ("balanced_accuracy",),
        DISTRACTION_CARD,
        "pct",
        2,
    ),
    (
        "distraction-mobilenet_v3_small-metrics.json",
        ("per_class_recall", "safe_drive"),
        DISTRACTION_CARD,
        "pct",
        2,
    ),
    (
        "distraction-mobilenet_v3_small-metrics.json",
        ("per_class_recall", "reach_side"),
        DISTRACTION_CARD,
        "pct",
        2,
    ),
    ("distraction-majority-metrics.json", ("balanced_accuracy",), DISTRACTION_CARD, "pct", 2),
    ("temporal-cross-dmd-metrics.json", ("drowsy_recall",), TEMPORAL_METHODOLOGY, "pct", 1),
    ("temporal-cross-dmd-metrics.json", ("false_alarm_rate",), TEMPORAL_METHODOLOGY, "ratio", 3),
    ("temporal-cross-dmd-metrics.json", ("overall_accuracy",), TEMPORAL_METHODOLOGY, "pct", 1),
    (
        "alert-validation-metrics.json",
        ("thresholds", "AUDIBLE", "false_alarms_per_hour"),
        ALERT_VALIDATION,
        "ratio",
        2,
    ),
    (
        "alert-validation-metrics.json",
        ("thresholds", "AUDIBLE", "false_alarms_per_hour_alert_clips_only"),
        ALERT_VALIDATION,
        "ratio",
        2,
    ),
    (
        "alert-validation-metrics.json",
        ("thresholds", "AUDIBLE", "fraction_not_drowsy_clips_with_alarm"),
        ALERT_VALIDATION,
        "pct",
        1,
    ),
    (
        "alert-validation-metrics.json",
        ("thresholds", "AUDIBLE", "drowsy_detection_rate"),
        ALERT_VALIDATION,
        "pct",
        1,
    ),
    (
        "alert-validation-metrics.json",
        ("thresholds", "VISUAL", "false_alarms_per_hour"),
        ALERT_VALIDATION,
        "ratio",
        2,
    ),
    ("yawdd-yawn-metrics.json", ("threshold",), ALERT_VALIDATION, "ratio", 3),
    ("yawdd-yawn-metrics.json", ("mirror", "precision"), ALERT_VALIDATION, "pct", 1),
    ("yawdd-yawn-metrics.json", ("mirror", "recall"), ALERT_VALIDATION, "pct", 1),
    (
        "yawdd-yawn-metrics.json",
        ("mirror", "talking_false_positive_rate"),
        ALERT_VALIDATION,
        "pct",
        1,
    ),
    ("yawdd-yawn-metrics.json", ("dash_recall_only", "recall"), ALERT_VALIDATION, "pct", 1),
    ("yawn-model-metrics.json", ("threshold",), YAWN_CARD, "ratio", 6),
    ("yawn-model-metrics.json", ("tau",), YAWN_CARD, "ratio", 2),
    ("yawn-model-metrics.json", ("baseline_mar", "precision"), YAWN_CARD, "ratio", 4),
    ("yawn-model-metrics.json", ("baseline_mar", "recall"), YAWN_CARD, "ratio", 4),
    (
        "yawn-model-metrics.json",
        ("baseline_mar", "talking_false_positive_rate"),
        YAWN_CARD,
        "ratio",
        4,
    ),
    ("yawn-model-metrics.json", ("baseline_duration", "precision"), YAWN_CARD, "ratio", 4),
    ("yawn-model-metrics.json", ("baseline_duration", "recall"), YAWN_CARD, "ratio", 4),
    (
        "yawn-model-metrics.json",
        ("baseline_duration", "talking_false_positive_rate"),
        YAWN_CARD,
        "ratio",
        4,
    ),
    ("yawn-model-metrics.json", ("cnn", "precision"), YAWN_CARD, "ratio", 4),
    ("yawn-model-metrics.json", ("cnn", "recall"), YAWN_CARD, "ratio", 4),
    (
        "yawn-model-metrics.json",
        ("cnn", "talking_false_positive_rate"),
        YAWN_CARD,
        "ratio",
        4,
    ),
]

# Each yawn detector's precision and the recall that must never be separated
# from it. Rendered at the same precision the card publishes them at.
YAWN_PRECISION_RECALL = [
    (("baseline_mar", "precision"), ("baseline_mar", "recall")),
    (("baseline_duration", "precision"), ("baseline_duration", "recall")),
    (("cnn", "precision"), ("cnn", "recall")),
]

TEMPORAL_ACCURACY = "47.1%"
TEMPORAL_FALSE_ALARM_RATE = "0.100"

ALERT_DETECTION_RATE = "100.0%"
ALERT_FALSE_ALARMS_PER_HOUR = "15.92"


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


def test_alert_detection_is_never_reported_without_false_alarms_per_hour() -> None:
    # Honesty rule: the alert level detection rate never appears without the
    # alert level false alarm rate beside it in the same document.
    offenders = [
        name
        for name, text in load_docs()
        if ALERT_DETECTION_RATE in text and ALERT_FALSE_ALARMS_PER_HOUR not in text
    ]
    assert offenders == [], (
        f"docs stating the alert detection rate without its false alarm rate: {offenders}"
    )


def test_yawn_precision_is_never_reported_without_its_recall() -> None:
    # Honesty rule: a yawn detector's precision never stands alone. Any table row
    # that states one of these precisions states that detector's recall in the
    # same row, and any document that states one states its recall somewhere.
    data = load_metric("yawn-model-metrics.json")
    offenders: list[str] = []
    for precision_path, recall_path in YAWN_PRECISION_RECALL:
        detector = precision_path[0]
        precision = render_ratio(metric_value(data, precision_path), 4)
        recall = render_ratio(metric_value(data, recall_path), 4)
        for name, text in load_docs():
            if precision in text and recall not in text:
                offenders.append(f"{name} states {detector} precision with no recall in the doc")
            for number, line in enumerate(text.splitlines(), start=1):
                if not line.lstrip().startswith("|"):
                    continue
                if precision in line and recall not in line:
                    offenders.append(f"{name}:{number} table row states {detector} precision alone")
    assert offenders == [], f"yawn precision reported without its recall: {offenders}"


def test_yawn_precision_recall_guard_flags_a_bare_row() -> None:
    # Positive control: the row rule is what rejects a precision only table row.
    data = load_metric("yawn-model-metrics.json")
    precision = render_ratio(metric_value(data, ("cnn", "precision")), 4)
    recall = render_ratio(metric_value(data, ("cnn", "recall")), 4)
    bare_row = f"| Mouth crop classifier | {precision} |"
    assert bare_row.lstrip().startswith("|")
    assert precision in bare_row
    assert recall not in bare_row


def test_metric_check_catches_drift() -> None:
    # Positive control: a doc that misstates the number does not satisfy the check.
    value = metric_value(load_metric("eye-state-metrics.json"), ("overall_accuracy",))
    rendered = _render(value, "pct", 2)
    drifted_doc = "the eye state model reaches 99.99% accuracy on the held out split"
    assert rendered not in drifted_doc
