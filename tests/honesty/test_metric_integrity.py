"""Every reported headline number must match the committed metrics JSON it cites,
and the temporal accuracy must never appear without its false alarm rate beside it.
"""

import pytest

from ._docs import (
    ALERT_VALIDATION,
    DISTRACTION_CARD,
    SAFETY_REVIEW,
    YAWN_ADR,
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
GAZE_CARD = "docs/ml/gaze-model-card.md"
GAZE_METRICS = "gaze-model-metrics.json"

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
    (
        "yawdd-crop-coverage.json",
        ("per_category", "Normal", "retained_fraction"),
        YAWN_CARD,
        "pct",
        1,
    ),
    (
        "yawdd-crop-coverage.json",
        ("per_category", "Talking", "retained_fraction"),
        YAWN_CARD,
        "pct",
        1,
    ),
    (
        "yawdd-crop-coverage.json",
        ("per_category", "Yawning", "retained_fraction"),
        YAWN_CARD,
        "pct",
        1,
    ),
    (
        "yawdd-crop-coverage.json",
        ("per_category", "Talking&Yawning", "retained_fraction"),
        YAWN_CARD,
        "pct",
        1,
    ),
    ("yawdd-crop-coverage.json", ("all", "retained_fraction"), YAWN_CARD, "pct", 1),
    # The decision record and the safety review both republish the same three
    # detectors' rate figures, so they are pinned to the metrics file too.
    ("yawn-model-metrics.json", ("baseline_mar", "precision"), YAWN_ADR, "ratio", 4),
    ("yawn-model-metrics.json", ("baseline_mar", "recall"), YAWN_ADR, "ratio", 4),
    (
        "yawn-model-metrics.json",
        ("baseline_mar", "talking_false_positive_rate"),
        YAWN_ADR,
        "ratio",
        4,
    ),
    ("yawn-model-metrics.json", ("baseline_duration", "precision"), YAWN_ADR, "ratio", 4),
    ("yawn-model-metrics.json", ("baseline_duration", "recall"), YAWN_ADR, "ratio", 4),
    (
        "yawn-model-metrics.json",
        ("baseline_duration", "talking_false_positive_rate"),
        YAWN_ADR,
        "ratio",
        4,
    ),
    ("yawn-model-metrics.json", ("cnn", "precision"), YAWN_ADR, "ratio", 4),
    ("yawn-model-metrics.json", ("cnn", "recall"), YAWN_ADR, "ratio", 4),
    ("yawn-model-metrics.json", ("cnn", "talking_false_positive_rate"), YAWN_ADR, "ratio", 4),
    ("yawn-model-metrics.json", ("baseline_mar", "precision"), SAFETY_REVIEW, "ratio", 4),
    ("yawn-model-metrics.json", ("baseline_mar", "recall"), SAFETY_REVIEW, "ratio", 4),
    (
        "yawn-model-metrics.json",
        ("baseline_mar", "talking_false_positive_rate"),
        SAFETY_REVIEW,
        "ratio",
        4,
    ),
    ("yawn-model-metrics.json", ("baseline_duration", "precision"), SAFETY_REVIEW, "ratio", 4),
    ("yawn-model-metrics.json", ("baseline_duration", "recall"), SAFETY_REVIEW, "ratio", 4),
    (
        "yawn-model-metrics.json",
        ("baseline_duration", "talking_false_positive_rate"),
        SAFETY_REVIEW,
        "ratio",
        4,
    ),
    ("yawn-model-metrics.json", ("cnn", "precision"), SAFETY_REVIEW, "ratio", 4),
    ("yawn-model-metrics.json", ("cnn", "recall"), SAFETY_REVIEW, "ratio", 4),
    (
        "yawn-model-metrics.json",
        ("cnn", "talking_false_positive_rate"),
        SAFETY_REVIEW,
        "ratio",
        4,
    ),
    (
        "yawdd-crop-coverage.json",
        ("per_category", "Normal", "crop_rows"),
        YAWN_CARD,
        "count",
        0,
    ),
    (
        "yawdd-crop-coverage.json",
        ("per_category", "Normal", "feature_rows"),
        YAWN_CARD,
        "count",
        0,
    ),
    # The per class recalls in the primary vs baseline table (stored as
    # per_class_accuracy: 0 alert, 1 low vigilance, 2 drowsy for both models).
    ("temporal-metrics.json", ("per_class_accuracy", "0"), TEMPORAL_METHODOLOGY, "pct", 1),
    ("temporal-metrics.json", ("per_class_accuracy", "1"), TEMPORAL_METHODOLOGY, "pct", 1),
    ("temporal-metrics.json", ("per_class_accuracy", "2"), TEMPORAL_METHODOLOGY, "pct", 1),
    ("temporal-metrics-gbt.json", ("per_class_accuracy", "0"), TEMPORAL_METHODOLOGY, "pct", 1),
    ("temporal-metrics-gbt.json", ("per_class_accuracy", "1"), TEMPORAL_METHODOLOGY, "pct", 1),
    ("temporal-metrics-gbt.json", ("per_class_accuracy", "2"), TEMPORAL_METHODOLOGY, "pct", 1),
    (GAZE_METRICS, ("interval", "binary", "false_alarm_rate"), GAZE_CARD, "ratio", 4),
    (GAZE_METRICS, ("interval", "binary", "detection_rate"), GAZE_CARD, "ratio", 4),
    (GAZE_METRICS, ("interval", "zone", "accuracy"), GAZE_CARD, "ratio", 4),
    (GAZE_METRICS, ("frame", "binary", "false_alarm_rate"), GAZE_CARD, "ratio", 4),
    (GAZE_METRICS, ("frame", "binary", "detection_rate"), GAZE_CARD, "ratio", 4),
    (GAZE_METRICS, ("frame", "zone", "accuracy"), GAZE_CARD, "ratio", 4),
    (GAZE_METRICS, ("interval", "binary", "n_on_road"), GAZE_CARD, "count", 0),
    (GAZE_METRICS, ("interval", "binary", "n_off_road"), GAZE_CARD, "count", 0),
    (GAZE_METRICS, ("interval", "binary", "false_alarms"), GAZE_CARD, "count", 0),
    (GAZE_METRICS, ("interval", "binary", "detected"), GAZE_CARD, "count", 0),
    (GAZE_METRICS, ("frame", "zone", "n"), GAZE_CARD, "count", 0),
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
    if kind == "pct":
        return render_pct(value, decimals)
    if kind == "count":
        return f"{int(value):,}"
    return render_ratio(value, decimals)


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


def yawn_precision_recall_offenders(docs: list[tuple[str, str]], data: dict) -> list[str]:
    """Documents and table rows that state a yawn precision without its recall.

    A yawn detector's precision never stands alone. Any table row that states
    one of these precisions must state that detector's recall in the same row,
    and any document that states one must state its recall somewhere in it.
    """
    offenders: list[str] = []
    for precision_path, recall_path in YAWN_PRECISION_RECALL:
        detector = precision_path[0]
        precision = render_ratio(metric_value(data, precision_path), 4)
        recall = render_ratio(metric_value(data, recall_path), 4)
        for name, text in docs:
            if precision in text and recall not in text:
                offenders.append(f"{name} states {detector} precision with no recall in the doc")
            for number, line in enumerate(text.splitlines(), start=1):
                if not line.lstrip().startswith("|"):
                    continue
                if precision in line and recall not in line:
                    offenders.append(f"{name}:{number} table row states {detector} precision alone")
    return offenders


def test_yawn_precision_is_never_reported_without_its_recall() -> None:
    offenders = yawn_precision_recall_offenders(load_docs(), load_metric("yawn-model-metrics.json"))
    assert offenders == [], f"yawn precision reported without its recall: {offenders}"


def test_yawn_precision_recall_guard_flags_a_bare_row() -> None:
    # Positive control: the same scan the guard above runs, fed a synthetic
    # document whose table row states a detector's precision and omits its
    # recall, must return that row as an offender. This fails if the row rule
    # is ever dropped from the scan, which asserting on a locally built string
    # would not catch.
    data = load_metric("yawn-model-metrics.json")
    precision = render_ratio(metric_value(data, ("cnn", "precision")), 4)
    recall = render_ratio(metric_value(data, ("cnn", "recall")), 4)
    synthetic = (
        "# Synthetic card\n"
        "\n"
        f"The classifier reached {recall} recall on the held out split.\n"
        "\n"
        "| Detector | Precision |\n"
        "|----------|-----------|\n"
        f"| Mouth crop classifier | {precision} |\n"
    )
    offenders = yawn_precision_recall_offenders([("synthetic.md", synthetic)], data)
    assert offenders != [], "the scan failed to flag a table row stating precision without recall"
    assert any("table row" in offender for offender in offenders)


def gaze_per_zone_recall_offenders(card_text: str, data: dict) -> list[str]:
    """Per zone recall rows in the gaze card that disagree with the metrics.

    The card publishes a recall per zone that is not stored as a key: it is
    correct over glances. Pinning only the counts would miss a typo in the
    recall column, so each row is checked as a whole. The glance count, the
    correct count, and the recall computed from them must all appear together
    in one table row.
    """
    offenders: list[str] = []
    per_class = data["interval"]["zone"]["per_class"]
    rows = [line for line in card_text.splitlines() if line.lstrip().startswith("|")]
    for zone, counts in per_class.items():
        n = counts["n"]
        correct = counts["correct"]
        recall = render_ratio(correct / n, 3)
        row = next((line for line in rows if f"| {zone} " in line), None)
        if row is None:
            offenders.append(f"gaze card has no per zone row for {zone}")
            continue
        for label, value in (("glances", str(n)), ("correct", str(correct)), ("recall", recall)):
            if value not in row:
                offenders.append(f"gaze card {zone} row is missing {label} {value}: {row.strip()}")
    return offenders


def test_gaze_per_zone_recall_matches_the_metrics() -> None:
    offenders = gaze_per_zone_recall_offenders(_doc_text(GAZE_CARD), load_metric(GAZE_METRICS))
    assert offenders == [], f"gaze per zone recall drifted from the metrics: {offenders}"


def test_gaze_per_zone_recall_guard_flags_a_drifted_row() -> None:
    # Positive control: a row whose recall does not equal correct over glances
    # must be flagged, which asserting on a hand built string would not prove
    # about the real scan.
    data = load_metric(GAZE_METRICS)
    synthetic = (
        "| Zone | Glances | Correct | Recall |\n"
        "|------|---------|---------|--------|\n"
        "| front | 51 | 47 | 0.999 |\n"
    )
    offenders = gaze_per_zone_recall_offenders(synthetic, data)
    assert offenders != [], "the scan failed to flag a per zone recall that does not match its counts"


def distraction_per_class_offenders(card_text: str, data: dict) -> list[str]:
    """Per class recall rows in the distraction card that disagree with the metrics.

    The card publishes a recall and a test support for every testable class, of
    which only two were pinned individually. A class absent from the test split
    (support zero) is shown as untestable rather than as a recall, so those are
    checked for the support only. Every other class row must carry both the
    recall the JSON stores and its support count.
    """
    offenders: list[str] = []
    recalls = data["per_class_recall"]
    support = data["support"]
    rows = [line for line in card_text.splitlines() if line.lstrip().startswith("|")]
    for cls, n in support.items():
        row = next((line for line in rows if f"| {cls} " in line), None)
        if row is None:
            offenders.append(f"distraction card has no row for {cls}")
            continue
        if n == 0:
            if "untestable" not in row:
                offenders.append(f"distraction card {cls} has zero support but is not untestable")
            continue
        recall = render_pct(recalls[cls], 2)
        for label, value in (("recall", recall), ("support", str(n))):
            if value not in row:
                offenders.append(f"distraction card {cls} row is missing {label} {value}")
    return offenders


def test_distraction_per_class_recall_matches_the_metrics() -> None:
    offenders = distraction_per_class_offenders(
        _doc_text(DISTRACTION_CARD), load_metric("distraction-mobilenet_v3_small-metrics.json")
    )
    assert offenders == [], f"distraction per class recall drifted from the metrics: {offenders}"


def test_distraction_per_class_guard_flags_a_drifted_row() -> None:
    # Positive control: a recall that does not match the JSON must be flagged.
    data = load_metric("distraction-mobilenet_v3_small-metrics.json")
    synthetic = (
        "| Class | Recall | Test support |\n"
        "|-------|--------|--------------|\n"
        "| safe_drive | 99.99% | 192 |\n"
    )
    offenders = distraction_per_class_offenders(synthetic, data)
    assert offenders != [], "the scan failed to flag a per class recall that does not match the JSON"


def test_metric_check_catches_drift() -> None:
    # Positive control: a doc that misstates the number does not satisfy the check.
    value = metric_value(load_metric("eye-state-metrics.json"), ("overall_accuracy",))
    rendered = _render(value, "pct", 2)
    drifted_doc = "the eye state model reaches 99.99% accuracy on the held out split"
    assert rendered not in drifted_doc
