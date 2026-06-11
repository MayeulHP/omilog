"""Unit tests for the eval harness metrics and case I/O.

All metric assertions are hand-computed examples — no jiwer / pyannote
cross-check at test time, so getting these numbers right matters: they ARE
the spec.
"""

import json

import pytest

from omilog.evals.cases import (
    fingerprint,
    load_cases,
    turns_from_segments,
)
from omilog.evals.metrics import (
    diarization_error_rate,
    normalize_tokens,
    user_attribution_accuracy,
    word_error_rate,
)

# ──────────────────────────────────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────────────────────────────────


def test_normalize_french_punctuation_case_elision():
    tokens = normalize_tokens("C'est l'été, peut-être !")
    assert tokens == ["c", "est", "l", "été", "peut", "être"]


def test_normalize_typographic_apostrophe_and_digits():
    assert normalize_tokens("L’autre à 14h30.") == ["l", "autre", "à", "14h30"]


def test_normalize_strips_collapse_marker_entirely():
    # The (×N) marker from collapse_repeated_segments is meta-text; the
    # digit must not survive as a token.
    assert normalize_tokens("c'est drôle  (×12)") == ["c", "est", "drôle"]


def test_normalize_keeps_accents_distinct():
    assert normalize_tokens("été") != normalize_tokens("ete")


# ──────────────────────────────────────────────────────────────────────────────
# WER
# ──────────────────────────────────────────────────────────────────────────────


def test_wer_identical_is_zero():
    r = word_error_rate("Bonjour, ça va ?", "bonjour ça va")
    assert r.wer == 0.0
    assert r.hits == 3
    assert r.errors == 0


def test_wer_substitution_and_insertion():
    # le chat est noir → le chien est très noir : 1 sub (chat→chien),
    # 1 ins (très), over 4 reference words = 0.5
    r = word_error_rate("le chat est noir", "le chien est très noir")
    assert r.substitutions == 1
    assert r.insertions == 1
    assert r.deletions == 0
    assert r.wer == pytest.approx(0.5)


def test_wer_deletion():
    r = word_error_rate("le chat est noir", "le chat noir")
    assert r.deletions == 1
    assert r.substitutions == 0
    assert r.wer == pytest.approx(0.25)


def test_wer_empty_hypothesis_is_all_deletions():
    r = word_error_rate("un deux trois", "")
    assert r.deletions == 3
    assert r.wer == pytest.approx(1.0)


def test_wer_empty_reference_rejected_unless_both_empty():
    assert word_error_rate("", "").wer == 0.0
    with pytest.raises(ValueError):
        word_error_rate("...", "bonjour")  # ref normalizes to nothing


def test_wer_can_exceed_one_on_heavy_insertion():
    r = word_error_rate("oui", "oui alors donc voilà")
    assert r.wer == pytest.approx(3.0)


# ──────────────────────────────────────────────────────────────────────────────
# DER — hand-computed interval examples (collar=0 unless noted)
# ──────────────────────────────────────────────────────────────────────────────


def _t(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def test_der_perfect_with_mismatched_labels():
    ref = [_t(0, 10, "USER"), _t(10, 20, "S1")]
    hyp = [_t(0, 10, "SPEAKER_07"), _t(10, 20, "SPEAKER_03")]
    r = diarization_error_rate(ref, hyp, collar=0.0)
    assert r.der == pytest.approx(0.0)
    assert r.mapping == {"USER": "SPEAKER_07", "S1": "SPEAKER_03"}


def test_der_two_speakers_collapsed_into_one_is_half_confusion():
    # ref: A[0,10] B[10,20]; hyp: one speaker for everything. The optimal
    # mapping matches one of them; the other 10s are speaker confusion.
    ref = [_t(0, 10, "A"), _t(10, 20, "B")]
    hyp = [_t(0, 20, "X")]
    r = diarization_error_rate(ref, hyp, collar=0.0)
    assert r.confusion_s == pytest.approx(10.0)
    assert r.miss_s == pytest.approx(0.0)
    assert r.false_alarm_s == pytest.approx(0.0)
    assert r.der == pytest.approx(0.5)


def test_der_missed_speech():
    ref = [_t(0, 10, "A")]
    hyp = [_t(0, 5, "X")]
    r = diarization_error_rate(ref, hyp, collar=0.0)
    assert r.miss_s == pytest.approx(5.0)
    assert r.der == pytest.approx(0.5)


def test_der_false_alarm():
    ref = [_t(0, 10, "A")]
    hyp = [_t(0, 15, "X")]
    r = diarization_error_rate(ref, hyp, collar=0.0)
    assert r.false_alarm_s == pytest.approx(5.0)
    assert r.der == pytest.approx(0.5)


def test_der_collar_forgives_boundary_jitter():
    ref = [_t(0, 10, "A")]
    hyp = [_t(0.1, 10.1, "X")]
    r = diarization_error_rate(ref, hyp, collar=0.25)
    assert r.der == pytest.approx(0.0)
    assert r.ref_speech_s == pytest.approx(9.5)


def test_der_overlapping_reference_speech_counts_as_miss():
    # B overlaps A in [5,10] but the hypothesis only ever has one active
    # speaker: 5s of the 15s of reference speech are missed.
    ref = [_t(0, 10, "A"), _t(5, 10, "B")]
    hyp = [_t(0, 10, "X")]
    r = diarization_error_rate(ref, hyp, collar=0.0)
    assert r.ref_speech_s == pytest.approx(15.0)
    assert r.miss_s == pytest.approx(5.0)
    assert r.confusion_s == pytest.approx(0.0)
    assert r.der == pytest.approx(5.0 / 15.0)


def test_der_empty_reference_rejected():
    with pytest.raises(ValueError):
        diarization_error_rate([], [_t(0, 5, "X")], collar=0.0)


def test_der_invalid_turns_dropped():
    ref = [_t(0, 10, "A"), _t(5, 5, "B"), {"start": 1, "end": 2}]  # zero-length / no label
    hyp = [_t(0, 10, "X")]
    r = diarization_error_rate(ref, hyp, collar=0.0)
    assert r.ref_speakers == 1
    assert r.der == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# USER attribution
# ──────────────────────────────────────────────────────────────────────────────


def test_user_attribution_partial_overlap():
    ref = [_t(0, 10, "USER"), _t(10, 20, "S1")]
    hyp = [_t(0, 8, "USER"), _t(8, 20, "S1")]
    ua = user_attribution_accuracy(ref, hyp)
    assert ua is not None
    assert ua.accuracy == pytest.approx(0.8)
    assert ua.ref_user_s == pytest.approx(10.0)


def test_user_attribution_none_without_user_in_reference():
    assert user_attribution_accuracy([_t(0, 5, "S1")], [_t(0, 5, "USER")]) is None


# ──────────────────────────────────────────────────────────────────────────────
# turns_from_segments
# ──────────────────────────────────────────────────────────────────────────────


def test_turns_from_segments_merges_small_gaps_only():
    segments = [
        {"start": 0.0, "end": 2.0, "speaker": "USER", "text": "a"},
        {"start": 2.3, "end": 4.0, "speaker": "USER", "text": "b"},  # 0.3s gap → merge
        {"start": 6.0, "end": 8.0, "speaker": "USER", "text": "c"},  # 2.0s gap → new turn
        {"start": 8.0, "end": 9.0, "speaker": "S1", "text": "d"},  # speaker change
        {"start": 9.0, "end": 9.5, "text": "unlabeled"},  # skipped
        {"start": 10.0, "end": 10.0, "speaker": "S1"},  # zero-length, skipped
    ]
    turns = turns_from_segments(segments, gap_tolerance_s=1.0)
    assert turns == [
        {"start": 0.0, "end": 4.0, "speaker": "USER"},
        {"start": 6.0, "end": 8.0, "speaker": "USER"},
        {"start": 8.0, "end": 9.0, "speaker": "S1"},
    ]


def test_turns_from_segments_sorts_input():
    segments = [
        {"start": 5.0, "end": 6.0, "speaker": "A"},
        {"start": 0.0, "end": 1.0, "speaker": "A"},
    ]
    turns = turns_from_segments(segments)
    assert [t["start"] for t in turns] == [0.0, 5.0]


# ──────────────────────────────────────────────────────────────────────────────
# Case loading
# ──────────────────────────────────────────────────────────────────────────────


def _make_case(root, name, *, text="bonjour", turns=None, meta=None, audio=True):
    d = root / name
    d.mkdir(parents=True)
    if text is not None:
        (d / "reference.txt").write_text(text, encoding="utf-8")
    if audio:
        (d / "audio.opus").write_bytes(b"\x00fake")
    if turns is not None:
        (d / "reference_turns.json").write_text(json.dumps(turns), encoding="utf-8")
    if meta is not None:
        (d / "case.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def test_load_cases_full_and_minimal(tmp_path):
    _make_case(
        tmp_path,
        "full",
        turns=[{"start": 0, "end": 1, "speaker": "USER"}],
        meta={"verified": True, "language": "fr"},
    )
    _make_case(tmp_path, "minimal")
    cases = load_cases(tmp_path)
    assert [c.name for c in cases] == ["full", "minimal"]
    full, minimal = cases
    assert full.verified is True
    assert full.reference_turns == [{"start": 0, "end": 1, "speaker": "USER"}]
    assert full.meta["language"] == "fr"
    assert minimal.verified is False
    assert minimal.reference_turns is None
    assert minimal.audio_path.name == "audio.opus"


def test_load_cases_skips_incomplete_dirs(tmp_path):
    _make_case(tmp_path, "no-audio", audio=False)
    _make_case(tmp_path, "no-text", text=None)
    _make_case(tmp_path, "empty-text", text="   \n")
    _make_case(tmp_path, "ok")
    assert [c.name for c in load_cases(tmp_path)] == ["ok"]


def test_load_cases_tolerates_bad_turns_json(tmp_path):
    d = _make_case(tmp_path, "bad-turns")
    (d / "reference_turns.json").write_text("{not json", encoding="utf-8")
    (case,) = load_cases(tmp_path)
    assert case.reference_turns is None  # DER skipped, WER still possible


def test_load_cases_missing_dir(tmp_path):
    assert load_cases(tmp_path / "nope") == []


# ──────────────────────────────────────────────────────────────────────────────
# Config fingerprint
# ──────────────────────────────────────────────────────────────────────────────


def test_fingerprint_stable_and_sensitive():
    a = {"language": "fr", "temperature": 0.0}
    assert fingerprint(a) == fingerprint({"temperature": 0.0, "language": "fr"})
    assert fingerprint(a) != fingerprint({"language": "fr", "temperature": 0.2})
