"""Raw transcript lines: split number pieces, "Rs." and words between voice turns (real WhatsApp recording)."""

from __future__ import annotations

from app.pipeline.assemble import assemble_segments


def _whisper(*segments):
    """Whisper segments from lists of (word, start, end)."""
    return [{"words": [{"w": w, "start": s, "end": e, "conf": 0.9} for w, s, e in seg]} for seg in segments]


def test_amounts_stay_on_one_line_and_read_as_written():
    whisper = _whisper(
        [(" we", 60.0, 60.2), (" have", 60.2, 60.4), (" revised", 60.4, 60.8), (" it", 60.8, 61.0),
         (" to", 61.0, 61.2), (" Rs.", 61.2, 61.8)],
        [(" 2", 62.0, 62.3), (" ,75", 62.3, 62.7), (" ,000.", 62.7, 63.3)],
        [(" Of", 63.5, 63.7), (" which,", 63.7, 64.0), (" Rs.", 64.0, 64.5)],
        [(" 80", 64.6, 65.0)],
        [(" ,000", 65.0, 65.4), (" will", 65.4, 65.6), (" go", 65.6, 65.8), (" towards", 65.8, 66.2),
         (" the", 66.2, 66.3), (" open", 66.3, 66.6), (" -air", 66.6, 67.0), (" stage.", 67.0, 67.5)],
    )
    # The voice model left a gap around "80" (no turn covers it).
    turns = [{"start": 59.0, "end": 64.5, "speaker": "Shivanshi"}, {"start": 65.1, "end": 70.0, "speaker": "Shivanshi"}]
    segs = assemble_segments(whisper, {}, turns)
    texts = [s.text for s in segs]
    assert texts == ["we have revised it to Rs. 2,75,000.", "Of which, Rs. 80,000 will go towards the open-air stage."]
    assert all(s.speaker == "Shivanshi" for s in segs)
    # One shown word per word object, so highlighting and editing stay aligned with the text.
    assert all(len(s.text.split()) == len(s.words) for s in segs)
    amount = segs[0].words[-1]
    assert amount.w.strip() == "2,75,000." and amount.start == 62.0 and amount.end == 63.3


def test_real_sentence_ends_still_split_and_lone_words_far_from_speech_keep_no_speaker():
    whisper = _whisper([(" Done.", 0.0, 0.5)], [(" Next", 0.6, 0.9), (" topic.", 0.9, 1.2)], [(" hmm", 9.0, 9.2)])
    segs = assemble_segments(whisper, {}, [{"start": 0.0, "end": 1.2, "speaker": "Speaker 1"}])
    assert [s.text for s in segs] == ["Done.", "Next topic.", "hmm"]
    assert segs[-1].speaker is None
