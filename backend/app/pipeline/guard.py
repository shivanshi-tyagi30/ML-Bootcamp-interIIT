"""GUARDING: deterministic checks on every LM1 edit, then apply the safe ones (spec Section 10.10)."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jellyfish

from app.config import Settings
from app.core.stages import Stage
from app.core.text import MODAL, NEG, NUM, capitalized_non_initial, modal_set, negation_count, numbers
from app.models.record import Edit, Refinement, RejectedEdit, Segment, Word

if TYPE_CHECKING:
    from app.pipeline.runner import JobContext

LETTER = dict(zip(
    "abcdefghijklmnopqrstuvwxyz",
    "ay bee see dee ee ef jee aitch eye jay kay el em en oh pee cue ar ess tee you vee doubleyou ex why zee".split(),
))


def _metaphone(s: str) -> str:
    """Metaphone of a phrase, ignoring word boundaries ("sea eye" ~ "see eye")."""
    return "".join(jellyfish.metaphone(w) for w in re.findall(r"[a-z]+", s.lower()))


def sounds_alike(orig: str, repl: str, threshold: float) -> bool:
    """Whether `repl` plausibly sounds like `orig` (casing-only and spelled acronyms allowed)."""
    o = re.sub(r"\s+", "", orig.lower())
    r = re.sub(r"\s+", "", repl.lower())
    if o == r:  # casing/spacing only: cuda -> CUDA
        return True
    spoken = repl
    if re.fullmatch(r"[A-Z0-9/&.\-]{2,8}", repl):  # acronym -> compare with spoken letters
        spoken = " ".join(LETTER[c] for c in repl.lower() if c in LETTER)
    a, b = _metaphone(orig), _metaphone(spoken)
    if jellyfish.jaro_winkler_similarity(o, r) >= threshold:
        return True
    if not a or not b:
        return False
    return jellyfish.jaro_winkler_similarity(a, b) >= threshold


def _word_spans(seg: Segment) -> list[tuple[int, int]]:
    """Character span of each word in seg.text (words are joined with single spaces)."""
    spans, pos = [], 0
    for w in seg.words:
        i = seg.text.find(w.w, pos)
        if i < 0:
            return []
        spans.append((i, i + len(w.w)))
        pos = i + len(w.w)
    return spans


def _touches_disputed_frozen(edit: Edit, seg: Segment, start: int) -> bool:
    """Edit overlaps a disputed word that is a number, negation or modal."""
    end = start + len(edit.original)
    for (a, b), w in zip(_word_spans(seg), seg.words):
        if a < end and start < b and w.disputed and any(p.search(w.w) for p in (NUM, NEG, MODAL)):
            return True
    return False


KNOWN_TERM_FILES = ("places.txt", "tech_terms.txt")


@lru_cache
def known_terms() -> frozenset[str]:
    """Lower-cased standard spellings of places, institutions and technical terms (app/data/*.txt)."""
    out: set[str] = set()
    for name in KNOWN_TERM_FILES:
        path = Path(__file__).resolve().parent.parent / "data" / name
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        out |= {ln.strip().lower() for ln in lines if ln.strip() and not ln.startswith("#")}
    return frozenset(out)


known_places = known_terms  # older name


def _squash(s: str) -> str:
    """Lower case without spaces, hyphens or dots ("Fast API" == "FastAPI")."""
    return re.sub(r"[\s\-.]", "", s.lower())


STRONG_RENAME_CONFIDENCE = 0.85
RENAME_CATEGORIES = {"proper_noun", "technical_term", "product", "acronym"}
TITLE_NAME = re.compile(r"\b(?:mr|mrs|ms|miss|dr|prof|sir|madam)\.?\s+([A-Z][a-z]+)")
# Context that marks a capitalized word as a person: "with Rahul", "ask Priya", "Rahul said", "Priya will".
BEFORE_PERSON = re.compile(r"\b(?:with|ask|asked|tell|told|thanks|thank|ping|call|cc|meet|met)\s+$", re.I)
AFTER_PERSON = re.compile(r"^\s*(?:said|says|told|asked|mentioned|thinks|thought|wants|wanted|will|would|can|"
                          r"could|should|agreed|suggested|'ll|'s)\b", re.I)


def people_names(segments: list[Segment]) -> frozenset[str]:
    """Lower-cased names of people in this meeting: speaker names, self-introductions, greetings, Mr/Dr X.

    These are never respelled, even when LM1 is confident.
    """
    from app.pipeline.speaker_names import ADDRESS, SELF_INTRO, _find

    out: set[str] = set()
    for s in segments:
        if s.speaker and not s.speaker.lower().startswith("speaker"):
            out |= {w.lower() for w in s.speaker.split()}
        for name in _find(SELF_INTRO + ADDRESS, s.text):
            out |= {w.lower() for w in name.split()}
        out |= {m.group(1).lower() for m in TITLE_NAME.finditer(s.text)}
    return frozenset(out)


def _person_context(text: str, token: str) -> bool:
    """Whether `token` is used like a person's name in `text` ("with Rahul", "Priya said")."""
    for m in re.finditer(r"(?<!\w)" + re.escape(token) + r"(?!\w)", text):
        if BEFORE_PERSON.search(text[: m.start()]) or AFTER_PERSON.search(text[m.end():]):
            return True
    return False


def _confident_respelling(edit: Edit, changed: list[str], people: frozenset[str], seg_text: str = "") -> bool:
    """A capitalized word may be respelled without any list ("Guhati" -> "Guwahati", "Pie Torch" ->
    "PyTorch") when LM1 is very confident, calls it a place/product/technical term, the new spelling sounds
    almost the same, and the word is not a person's name heard in this meeting."""
    if edit.confidence < STRONG_RENAME_CONFIDENCE or edit.category not in RENAME_CATEGORIES:
        return False
    if any(c.lower() in people for c in changed):
        return False
    if any(_person_context(seg_text, c) for c in changed):
        return False
    a, b = _squash(edit.original), _squash(edit.replacement)
    close_spelling = jellyfish.jaro_winkler_similarity(a, b) >= 0.8
    close_sound = jellyfish.jaro_winkler_similarity(_metaphone(edit.original), _metaphone(edit.replacement)) >= 0.9
    return close_spelling or close_sound


def _name_changed(edit: Edit, seg: Segment, start: int, vocab_terms: set[str],
                  people: frozenset[str] = frozenset()) -> bool:
    """A non-initial capitalized token in `original` is changed without support.

    Supported: only spacing or capitalization changes ("Fast API" -> "FastAPI"), a meeting vocabulary
    term (which includes the user's glossary), or a known place/institution/technical spelling
    (e.g. "Guhati" -> "Guwahati", "Pie Torch" -> "PyTorch"). People's names are in none of these, so they
    stay as heard. Sound-alike is still checked afterwards.
    """
    if _squash(edit.original) == _squash(edit.replacement):
        return False
    toks = [t for t in re.findall(r"\b[A-Z][a-z]+\b", edit.original) if t not in edit.replacement]
    # People heard in this meeting are never respelled, wherever the word sits ("Rahul said...").
    if any(t.lower() in people for t in toks):
        return True
    r = edit.replacement.lower().strip()
    in_vocab = r in vocab_terms or any(t and t in r for t in vocab_terms)
    if in_vocab or r in known_terms():  # a known place/term (or the user's glossary) is never a person
        return False
    # Words used like a person's name ("with Rahul", "Priya said") are protected too.
    if any(_person_context(seg.text, t) for t in toks):
        return True
    caps = [c for c in capitalized_non_initial(seg.text[: start + len(edit.original)]) if c in edit.original]
    changed = [c for c in caps if c not in edit.replacement]
    if not changed:
        return False
    return not _confident_respelling(edit, changed, people, seg.text)


def check_edit(edit: Edit, seg: Segment | None, vocab_terms: set[str], settings: Settings,
               people: frozenset[str] = frozenset()) -> str | None:
    """First failing check's reject reason, or None if the edit is safe."""
    o, r = edit.original, edit.replacement
    if edit.confidence < settings.LM1_MIN_CONFIDENCE:
        return "low_confidence"
    if seg is None or not o or o not in seg.text or o.strip() == r.strip():
        return "not_found"
    if numbers(o) != numbers(r):
        return "number_changed"
    if negation_count(o) != negation_count(r):
        return "negation_changed"
    if modal_set(o) != modal_set(r):
        return "modal_changed"
    start = seg.text.index(o)
    if _name_changed(edit, seg, start, vocab_terms, people):
        return "name_changed"
    if len(r) > 3 * max(len(o), 4):
        return "over_rewrite"
    if not sounds_alike(o, r, settings.SOUND_ALIKE_THRESHOLD):
        return "not_sound_alike"
    if _touches_disputed_frozen(edit, seg, start):
        return "touches_disputed_frozen"
    return None


def apply_to_segment(seg: Segment, edits: list[Edit]) -> Segment:
    """Copy of `seg` with non-overlapping edits applied left to right.

    Words touched by an edit merge into one word that keeps the original
    start/end times, so playback still lines up.
    """
    text, words = seg.text, list(seg.words)
    spans = _word_spans(seg)
    located = sorted(((text.index(e.original), e) for e in edits), key=lambda x: x[0])
    new_text, new_words, pos, wi = "", [], 0, 0
    for start, e in located:
        end = start + len(e.original)
        new_text += text[pos:start] + e.replacement
        pos = end
        if spans:
            hit = [k for k, (a, b) in enumerate(spans) if a < end and start < b]
            while wi < len(words) and (not hit or wi < hit[0]):
                new_words.append(words[wi])
                wi += 1
            if hit:
                a, b = spans[hit[0]][0], spans[hit[-1]][1]
                merged_text = text[a:start] + e.replacement + text[end:b]
                ws = [words[k] for k in hit]
                new_words.append(Word(
                    w=merged_text, start=ws[0].start, end=ws[-1].end, conf=min(w.conf for w in ws),
                    disputed=any(w.disputed for w in ws), alt=next((w.alt for w in ws if w.alt), None),
                ))
                wi = hit[-1] + 1
    new_text += text[pos:]
    new_words.extend(words[wi:])
    return seg.model_copy(update={"text": new_text, "words": new_words if spans else words})


def guard_edits(
    raw: list[Segment], proposed: list[Edit], vocab: dict[str, Any], settings: Settings,
) -> tuple[list[Segment], list[Edit], list[RejectedEdit]]:
    """Check every edit, resolve overlaps (higher confidence wins) and build the refined transcript."""
    proposed = [e for e in proposed if e.original.strip() != e.replacement.strip()]
    by_id = {s.id: s for s in raw}
    vocab_terms = {t["term"].lower() for t in vocab.get("terms", []) if t.get("term")}
    people = people_names(raw)
    accepted: list[Edit] = []
    rejected: list[RejectedEdit] = []
    for e in proposed:
        reason = check_edit(e, by_id.get(e.segment_id), vocab_terms, settings, people)
        if reason:
            rejected.append(RejectedEdit(**e.model_dump(), reject_reason=reason))
        else:
            accepted.append(e)

    # Per segment: keep non-overlapping edits, preferring higher confidence.
    final: dict[str, list[Edit]] = {}
    for e in sorted(accepted, key=lambda x: -x.confidence):
        seg = by_id[e.segment_id]
        s = seg.text.index(e.original)
        taken = final.setdefault(e.segment_id, [])
        if any(s < seg.text.index(t.original) + len(t.original) and seg.text.index(t.original) < s + len(e.original)
               for t in taken):
            rejected.append(RejectedEdit(**e.model_dump(), reject_reason="not_found"))
        else:
            taken.append(e)
    kept = [e for e in accepted if e in final.get(e.segment_id, [])]
    refined = [apply_to_segment(s, final.get(s.id, [])) if final.get(s.id) else s.model_copy() for s in raw]
    return refined, kept, rejected


async def guard(ctx: "JobContext") -> None:
    """Stage function: write refined_transcript.json and 08_refinement.json."""
    raw = [Segment(**s) for s in ctx.read(ctx.output_name(Stage.RAW_SAVED))]
    vocab = ctx.read(ctx.output_name(Stage.VOCABULARY)) or {"domain": "unknown", "terms": []}
    proposed = [Edit(**e) for e in ctx.read(ctx.output_name(Stage.REFINING))["edits"]]
    refined, accepted, rejected = guard_edits(raw, proposed, vocab, ctx.settings)
    ctx.write("refined_transcript.json", [s.model_dump(mode="json") for s in refined])
    ctx.write(ctx.output_name(Stage.GUARDING), Refinement(vocabulary=vocab, accepted=accepted, rejected=rejected))
