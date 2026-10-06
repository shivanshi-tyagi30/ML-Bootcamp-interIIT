"""Guard checks on LM1 edits (spec Section 10.10)."""

from __future__ import annotations

from app.config import Settings
from app.models.record import Edit
from app.pipeline.guard import check_edit, guard_edits
from tests.conftest import seg

S = Settings()
VOCAB = {"domain": "ML", "terms": [{"term": "CI"}, {"term": "Kubernetes"}, {"term": "CUDA"}]}
TERMS = {"ci", "kubernetes", "cuda"}


def check(text: str, original: str, replacement: str, category: str = "technical_term", conf: float = 0.9) -> str | None:
    e = Edit(segment_id="S001", original=original, replacement=replacement, category=category, confidence=conf)
    return check_edit(e, seg("S001", 0, text), TERMS, S)


def test_number_changed():
    assert check("latency went from fifteen to fifty", "fifteen", "fifty", "homophone") == "number_changed"


def test_negation_changed():
    assert check("we can't push it", "can't", "can", "homophone") == "negation_changed"


def test_modal_changed():
    assert check("we might ship it", "might", "will", "homophone") == "modal_changed"


def test_accepts_spelled_acronym():
    assert check("until the sea eye pipeline passes", "sea eye", "CI", "acronym") is None


def test_accepts_casing():
    assert check("push the cuda kernels", "cuda", "CUDA", "acronym") is None


def test_accepts_sound_alike_term():
    assert check("the cooper netties job is failing", "cooper netties", "Kubernetes") is None


def test_rejects_not_sound_alike():
    assert check("until the sea eye pipeline passes", "sea eye", "deployment") == "not_sound_alike"


def test_rejects_over_rewrite():
    assert check("the cuda job", "cuda", "the entire GPU programming framework") == "over_rewrite"


def test_rejects_not_found():
    assert check("push the kernels", "cuda", "CUDA", "acronym") == "not_found"


def test_rejects_low_confidence():
    assert check("push the cuda kernels", "cuda", "CUDA", "acronym", conf=0.5) == "low_confidence"


def test_rejects_name_change():
    assert check("then Ravi said so", "Ravi", "Robbie", "homophone") == "name_changed"


def test_disputed_frozen_word():
    s = seg("S001", 0, "we need fifteen nodes", disputed={"fifteen": "fifty"})
    e = Edit(segment_id="S001", original="fifteen nodes", replacement="fifteen Nodes", category="technical_term",
             confidence=0.9)
    assert check_edit(e, s, TERMS, S) == "touches_disputed_frozen"


def test_apply_keeps_timing_and_resolves_overlap():
    raw = [seg("S001", 0, "we can't push the cuda kernels until the sea eye pipeline passes")]
    edits = [
        Edit(segment_id="S001", original="sea eye", replacement="CI", category="acronym", confidence=0.9),
        Edit(segment_id="S001", original="cuda", replacement="CUDA", category="acronym", confidence=0.95),
        Edit(segment_id="S001", original="sea eye pipeline", replacement="CI pipeline", category="acronym", confidence=0.7),
    ]
    refined, accepted, rejected = guard_edits(raw, edits, VOCAB, S)
    assert refined[0].text == "we can't push the CUDA kernels until the CI pipeline passes"
    assert raw[0].text.startswith("we can't push the cuda")  # raw is never modified
    assert len(accepted) == 2 and [r.reject_reason for r in rejected] == ["not_found"]
    ci = next(w for w in refined[0].words if w.w == "CI")
    sea = next(w for w in raw[0].words if w.w == "sea")
    eye = next(w for w in raw[0].words if w.w == "eye")
    assert (ci.start, ci.end) == (sea.start, eye.end)
