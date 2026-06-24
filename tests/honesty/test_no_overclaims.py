"""No affirmative overclaim may appear in any public document.

The blocklist holds only phrases that are always overclaims. It deliberately
excludes bare words like "medical", "diagnostic", "certified", or "reliable",
because the honest disclaimers negate them ("not a medical device") and the
provenance notes use them in ordinary ways ("a reliable download mirror").
"""

from ._docs import find_blocked_phrases, load_docs

BLOCKED_PHRASES = [
    "clinically validated",
    "clinically proven",
    "fda approved",
    "fda-approved",
    "fda cleared",
    "fda-cleared",
    "medically accurate",
    "prevents accidents",
    "prevent accidents",
    "guarantees safety",
    "guaranteed safe",
    "100% accurate",
    "saves lives",
    "life saving",
    "life-saving",
    "reliable detector",
    "certified safety device",
]


def test_public_docs_contain_no_overclaim_phrases() -> None:
    hits = find_blocked_phrases(load_docs(), BLOCKED_PHRASES)
    assert hits == [], f"overclaim phrases found in public docs: {hits}"


def test_blocklist_scanner_flags_a_violation() -> None:
    # Positive control: the scanner catches planted overclaims.
    planted = [("planted.md", "SafeEyes is clinically validated and prevents accidents.")]
    hits = find_blocked_phrases(planted, BLOCKED_PHRASES)
    assert ("planted.md", "clinically validated") in hits
    assert ("planted.md", "prevents accidents") in hits
