# Changelog

## 0.2.3

Packaging release. No code/API changes.

- **Renamed the PyPI distribution to `rag-quantguard`** -- `quantguard` was
  already registered on PyPI by an unrelated project. The import name is
  unchanged (`import quantguard`), and so is the CLI command (`quantguard
  verify ...`); only `pip install <name>` and the name in `pyproject.toml`
  changed. This is a normal, supported split (distribution name vs. import
  name don't have to match).
- Added the missing `LICENSE` file (MIT) -- `pyproject.toml` already
  declared `license = "MIT"` but the file itself didn't exist in the repo.
- Switched `license` from the deprecated `{ text = "MIT" }` table form to
  the modern SPDX string form (`license = "MIT"` + `license-files`);
  the old form is deprecated as of setuptools and will stop being
  supported for new releases.
- Filled in `authors`, `keywords`, `classifiers`, and `project.urls` in
  `pyproject.toml` (previously entirely absent) so the PyPI project page
  shows an author, is findable via PyPI search, and links back to the
  repo (`https://github.com/BalakrishnaSM/RAG-QuantGuard`),
  changelog, and issue tracker.
- Added an "Installation" section to the README with the actual
  `pip install rag-quantguard` command, since previously the README only
  documented the local editable-install workflow.
- Verified the full release pipeline end-to-end: `python -m build`,
  `twine check dist/*` (passes on both wheel and sdist), install from the
  built wheel into a clean environment, `import quantguard` and the
  `quantguard` CLI both still resolve correctly post-install.

## 0.2.2

Bugfix release. No public API changes.

### Fixed

- **The "2024 vs 2025" temporal trap wasn't actually guarded for `BOUND`
  and `RANGE` claims.** `compare_numeric` has always hard-rejected a
  match whose evidence carries a different `time_context` than the
  claim (this is the flagship scenario in the README:
  `guard.verify("Latency in 2024 was 12ms.", ["2024 value = 25ms.",
  "2025 value = 12ms."])` → `CONTRADICTED`). `compare_bound` and
  `compare_range` never had this check at all -- `time_context` is
  extracted and populated for every claim kind, but only the ranking
  score in `evidence.py` penalized a time mismatch; nothing *forced*
  a contradiction the way `compare_numeric` did. So a bound/range
  claim scoped to one period could be silently `VERIFIED` against
  evidence for a completely different period, as long as the number
  happened to satisfy the bound/fall in the range. Example that used
  to pass incorrectly:
  ```python
  guard.verify("In 2024, latency was under 50ms.", ["In 2025, latency was 12ms."])
  # was: VERIFIED (wrong -- no evidence for 2024 exists at all)
  # now: CONTRADICTED ("Claim refers to 2024 but the matched evidence is for 2025.")
  ```
  Factored the check out of `compare_numeric` into a shared
  `_time_context_mismatch` helper and applied it to `compare_bound`
  and `compare_range` as well, so the guard can't silently apply to
  only some claim kinds again. Added two new benchmark cases
  (`temporal_bound_mismatch`, `temporal_range_mismatch`) specifically
  because the existing benchmark, despite claiming to cover "every
  claim type," had zero temporal-trap coverage for these two kinds --
  which is exactly why the gap shipped unnoticed. Benchmark is now
  22/22 (was 20/20); naive baseline is 14/22 (63.6%, was 70.0%) since
  the two new cases are additional naive failure modes.

## 0.2.1

Bugfix release. No public API changes.

### Fixed

- **`compare_reference` no longer contradicts citations that omit the
  document number.** A claim like `"3GPP Section 5.3.5.4"` (no `TS
  38.331`) was being compared with `claim.document != evidence.document`
  (`None != "TS 38.331"`), which is the same code path used for a
  citation to a genuinely *different* document. That meant a citation
  that was simply less specific than the evidence got marked
  `CONTRADICTED` instead of `VERIFIED`/`APPROXIMATE`. Document identity
  is now only treated as a mismatch when **both** sides specify a
  document and they disagree; when only one side specifies one, the
  match is downgraded to `APPROXIMATE` rather than rejected outright.
  Citations to an actually-wrong document are still caught correctly
  (see `test_reference_wrong_document_is_still_contradicted`).

- **`try_arithmetic_sum` no longer drops a real component that happens
  to equal the claimed total.** The evidence pool was pre-filtered with
  `e.value != claim.value` to avoid degenerate matches, but this could
  silently remove a legitimate component (e.g. `"20 completed, 0
  pending"` against a claimed total of `20`, where `20` is itself one
  of the two real components), leaving too few components to verify an
  otherwise-correct total. The filter is unnecessary: subset search
  only considers subsets of size >= 2, so a lone matching value can
  never produce a trivial single-item "match" on its own. Removed.

### Notes

- `GuardResult.is_valid` docstring now explains why `Status.NOT_APPLICABLE`
  is intentionally excluded from the set of "bad" statuses (reserved
  for future claim kinds that aren't yet produced by `extract_claims`;
  behavior unchanged).
- Added regression tests for both fixes, including cases that confirm
  the *original* (correct) contradiction behavior -- wrong document,
  wrong organization -- still holds.

## 0.2.0

Initial tracked release (auto-correction, streaming verifier, LangChain /
LlamaIndex / OpenTelemetry integrations, pluggable NLI/LLM fallback).