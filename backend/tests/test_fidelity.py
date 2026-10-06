"""Fidelity counts (spec Section 10.13)."""

from __future__ import annotations

from app.models.record import Refinement
from app.pipeline.fidelity import compute_fidelity, preserved
from tests.conftest import make_record, seg


def test_numbers_and_negations():
    raw = [seg("S001", 0, "latency went from fifteen to 50 ms"), seg("S002", 3, "we can't and won't do it")]
    same = [seg("S001", 0, "latency went from fifteen to 50 ms"), seg("S002", 3, "we can't and won't do it")]
    changed = [seg("S001", 0, "latency went from fifty to 50 ms"), seg("S002", 3, "we can and won't do it")]
    assert preserved(raw, same, "numbers") == "2/2"
    assert preserved(raw, same, "negations") == "2/2"
    assert preserved(raw, changed, "numbers") == "0/2"
    assert preserved(raw, changed, "negations") == "0/2"


def test_coverage():
    rec = make_record("a" * 32)
    # Cited: S004, S005, S006, S008 -> 4 of 8 segments.
    f = compute_fidelity(rec, Refinement(), dropped_items=1, demoted=1, sentences_removed=2)
    assert f.transcript_coverage_pct == 50.0
    assert f.items_downgraded_by_verifier == 2
    assert f.sentences_removed_by_verifier == 2
