# ADR 0004: Acquire UTA-RLDD from the Kaggle mirror

- Status: Accepted
- Date: 2026-06-18

## Context

UTA-RLDD is distributed from the project's official Google Drive folder as ten
files of roughly 11 GB each. Automated download of those files hit a Google
imposed per file quota wall ("too many users have viewed or downloaded this file
recently"), which an anonymous downloader cannot bypass and which can take up to
24 hours to clear. The browser, using an authenticated session, was likely to hit
the same wall for a heavily downloaded public file.

A community mirror exists on Kaggle (`rishab260/uta-reallife-drowsiness-dataset`),
released under CC0-1.0, served through the Kaggle API, which does not have the
Drive quota behavior.

## Decision

Acquire UTA-RLDD from the Kaggle mirror using the Kaggle CLI. The dataset
provenance and the obtain steps are recorded in
[docs/source/datasets.md](../source/datasets.md).

## Consequences

- A reliable, scriptable download, unaffected by the Drive quota.
- Requires a free Kaggle account and API token, a one time setup.
- A dependency on a community mirror rather than the primary source. Before
  relying on it, its contents were checked against the expected structure: per
  fold per subject folders with the class encoded in the file name, the layout
  the split tooling expects. The permissive CC0-1.0 license is suitable for this
  use.
- The mirror carries folds 1 to 4 only, 48 of the 60 subjects (141 clips), not the
  full dataset. This was verified after download and is accepted: 48 subjects with a
  subject independent split is sufficient for the project's bar, and chasing the
  missing fold means re-entering the Drive quota for marginal gain. The coverage is
  recorded in [docs/source/datasets.md](../source/datasets.md) and the temporal
  methodology so no reported number implies the full 60 subjects.
- If the mirror were removed or changed, the official Drive source remains the
  fallback once its quota clears, and the recorded provenance allows the contents
  to be re-verified.
