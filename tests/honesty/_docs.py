"""Shared helpers for the honesty-guard regression tests.

These guards protect the project's public surface: that every reported number
traces to a committed metrics file, that no affirmative overclaim slips into a
document, and that the assistive-prototype disclaimers stay present. The scan
operates only on the tracked, public Markdown so it mirrors what a reader sees.
"""

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Path, relative to the repo root, of the public distraction model card. Kept here
# so the metric-integrity pins and any future honesty guard reference one source.
DISTRACTION_CARD = "docs/ml/distraction-model-card.md"

# The alert level validation and YawDD results doc, pinned the same way.
ALERT_VALIDATION = "docs/ml/alert-validation.md"

# The yawn detector comparison card, which publishes the held out figures for
# both geometric baselines and the trained classifier that did not beat them.
YAWN_CARD = "docs/ml/yawn-model-card.md"

# The two other public documents that republish the yawn detector rate figures.
# They are pinned to the same metrics file as the card so that regenerating the
# artifact fails all three together, rather than failing the card while these
# two quietly keep stale numbers.
YAWN_ADR = "docs/adr/0006-geometric-duration-rule-over-mouth-crop-cnn.md"
SAFETY_REVIEW = "docs/ml/safety-review.md"


def public_docs() -> list[Path]:
    """Tracked Markdown that ships publicly: the README and the docs/ tree.

    Reads the git index, so local-only files such as docs/plan.md are excluded
    and the scan matches exactly the surface a public reader can see.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z", "README.md", "docs"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    names = [n for n in result.stdout.split("\0") if n.endswith(".md")]
    return [REPO_ROOT / n for n in sorted(names)]


def load_docs() -> list[tuple[str, str]]:
    """Each public doc as (relative path, text)."""
    return [
        (str(path.relative_to(REPO_ROOT)), path.read_text(encoding="utf-8"))
        for path in public_docs()
    ]


def load_metric(name: str) -> dict:
    """A committed metrics JSON from docs/ml, parsed."""
    return json.loads((REPO_ROOT / "docs" / "ml" / name).read_text(encoding="utf-8"))


def metric_value(data: dict, keys: tuple[str, ...]) -> float:
    """Follow a nested key path into a metrics dict and return the leaf as a float."""
    cursor: object = data
    for key in keys:
        cursor = cursor[key]  # type: ignore[index]
    return float(cursor)  # type: ignore[arg-type]


def render_pct(value: float, decimals: int) -> str:
    """Render a 0..1 ratio as a percentage string, for example 0.9436 -> 94.36%."""
    return f"{value * 100:.{decimals}f}%"


def render_ratio(value: float, decimals: int) -> str:
    """Render a ratio at fixed precision, for example 0.69303 -> 0.693."""
    return f"{value:.{decimals}f}"


def find_blocked_phrases(
    docs: Iterable[tuple[str, str]], phrases: Iterable[str]
) -> list[tuple[str, str]]:
    """Every (doc, phrase) where a blocked phrase appears, matched case-insensitively."""
    phrase_list = list(phrases)
    hits: list[tuple[str, str]] = []
    for name, text in docs:
        lowered = text.lower()
        hits.extend((name, phrase) for phrase in phrase_list if phrase.lower() in lowered)
    return hits


def missing_substrings(text: str, required: Iterable[str]) -> list[str]:
    """The required substrings absent from text, matched case-insensitively."""
    lowered = text.lower()
    return [needle for needle in required if needle.lower() not in lowered]
