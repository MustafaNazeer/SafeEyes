"""The assistive-prototype disclaimers must stay present, so a future edit cannot
silently strip the framing that keeps the project's claims honest.
"""

from ._docs import missing_substrings, public_docs

# Doc (relative path) -> substrings that must all be present, case-insensitively.
REQUIRED = {
    "README.md": [
        "not a medical device",
        "diagnostic or safety of life",
        "research prototype",
    ],
    "docs/ml/safety-review.md": [
        "assistive prototype",
        "not a medical device",
    ],
}


def _read(doc_file: str) -> str:
    for path in public_docs():
        if str(path).endswith(doc_file):
            return path.read_text(encoding="utf-8")
    raise AssertionError(f"expected public doc not found: {doc_file}")


def test_required_disclaimers_are_present() -> None:
    for doc_file, required in REQUIRED.items():
        missing = missing_substrings(_read(doc_file), required)
        assert missing == [], f"{doc_file} is missing required disclaimer text: {missing}"


def test_disclaimer_check_catches_removal() -> None:
    # Positive control: a doc stripped of the disclaimer is reported as missing it.
    stripped = "SafeEyes detects drowsiness from a cabin camera."
    missing = missing_substrings(stripped, REQUIRED["README.md"])
    assert "not a medical device" in missing
