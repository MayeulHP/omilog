"""Pure-Python ASR / diarization metrics.

Word Error Rate (WER) via standard Levenshtein alignment with
substitution/insertion/deletion counts, and Diarization Error Rate (DER)
following the NIST formulation (missed speech + false alarm + speaker
confusion, over total reference speech time, with an optional no-score
collar around reference turn boundaries and an optimal one-to-one
speaker mapping).

Everything here is pure data — no audio access, no model deps — so it is
cheap to unit-test against hand-computed examples.
"""

from __future__ import annotations

import itertools
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence

# ──────────────────────────────────────────────────────────────────────────────
# Text normalization
# ──────────────────────────────────────────────────────────────────────────────

# "(×3)" markers added by stt.collapse_repeated_segments are meta-text, not
# speech — strip them before the alnum filter would otherwise leave a bare
# digit token behind.
_COLLAPSE_MARKER = re.compile(r"\(×\d+\)")


def normalize_tokens(text: str) -> list[str]:
    """Tokenize for WER: NFKC, lowercase, punctuation → separators.

    Choices (applied identically to reference and hypothesis, so they only
    forgive orthographic variance, never real word errors):
      * accents are KEPT ("été" ≠ "ete" stays distinct from "et")
      * apostrophes split elisions ("l'autre" → "l autre"), the common
        convention for French WER — partial credit on elision mismatches
      * hyphens split compounds ("est-ce" → "est ce") for the same reason
      * digits survive as-is ("14h30" is one token) — write references in
        the same digit style Whisper produces (see eval/README.md)
    """
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = _COLLAPSE_MARKER.sub(" ", t)
    return "".join(c if c.isalnum() else " " for c in t).split()


# ──────────────────────────────────────────────────────────────────────────────
# WER
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class WERResult:
    wer: float
    substitutions: int
    insertions: int
    deletions: int
    hits: int
    ref_words: int
    hyp_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions


def _edit_counts(ref: Sequence[Any], hyp: Sequence[Any]) -> tuple[int, int, int, int]:
    """Minimal-edit alignment counts: (substitutions, insertions, deletions,
    hits). Ties broken substitution-first (the standard convention)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return (0, m, 0, 0)
    if m == 0:
        return (0, 0, n, 0)

    # ops: 0=hit (diag), 1=sub (diag), 2=del (up), 3=ins (left)
    width = m + 1
    ops = bytearray((n + 1) * width)
    for j in range(1, width):
        ops[j] = 3
    prev = list(range(width))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        base = i * width
        ops[base] = 2
        ri = ref[i - 1]
        for j in range(1, width):
            if ri == hyp[j - 1]:
                cur[j] = prev[j - 1]
                ops[base + j] = 0
                continue
            sub = prev[j - 1]
            dele = prev[j]
            ins = cur[j - 1]
            if sub <= dele and sub <= ins:
                cur[j] = sub + 1
                ops[base + j] = 1
            elif dele <= ins:
                cur[j] = dele + 1
                ops[base + j] = 2
            else:
                cur[j] = ins + 1
                ops[base + j] = 3
        prev = cur

    subs = ins = dels = hits = 0
    i, j = n, m
    while i > 0 or j > 0:
        op = ops[i * width + j]
        if i > 0 and j > 0 and op == 0:
            hits += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and op == 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and op == 2:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return (subs, ins, dels, hits)


def word_error_rate(reference: str, hypothesis: str) -> WERResult:
    """WER between two raw texts (normalization applied to both sides).

    A WER of 0.0 with an empty reference and empty hypothesis; raises if the
    reference normalizes to nothing while the hypothesis doesn't (a labeling
    error, not a model error).
    """
    ref = normalize_tokens(reference)
    hyp = normalize_tokens(hypothesis)
    if not ref:
        if hyp:
            raise ValueError(
                "reference text normalized to zero tokens but hypothesis is "
                "non-empty — check the reference file"
            )
        return WERResult(0.0, 0, 0, 0, 0, 0, 0)
    subs, ins, dels, hits = _edit_counts(ref, hyp)
    return WERResult(
        wer=(subs + ins + dels) / len(ref),
        substitutions=subs,
        insertions=ins,
        deletions=dels,
        hits=hits,
        ref_words=len(ref),
        hyp_words=len(hyp),
    )


# ──────────────────────────────────────────────────────────────────────────────
# DER
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DERResult:
    der: float
    miss_s: float
    false_alarm_s: float
    confusion_s: float
    ref_speech_s: float  # scored reference speech time (collar excluded)
    mapping: dict[str, str]  # ref label → hyp label (only pairs with overlap)
    ref_speakers: int
    hyp_speakers: int


def _intervals_by_speaker(
    turns: list[dict[str, Any]],
) -> dict[str, list[tuple[float, float]]]:
    """Group turn dicts ({'start','end','speaker'}) into merged, sorted
    per-speaker interval lists. Invalid turns (end <= start, no speaker)
    are dropped."""
    raw: dict[str, list[tuple[float, float]]] = {}
    for t in turns:
        spk = t.get("speaker")
        if not spk:
            continue
        start = float(t.get("start", 0) or 0)
        end = float(t.get("end", start) or start)
        if end <= start:
            continue
        raw.setdefault(str(spk), []).append((start, end))
    return {spk: _merge_intervals(ivs) for spk, ivs in raw.items()}


def _merge_intervals(ivs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ivs = sorted(ivs)
    out: list[tuple[float, float]] = []
    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _optimal_mapping(
    ref_ids: list[str],
    hyp_ids: list[str],
    pair_overlap: dict[tuple[str, str], float],
) -> dict[str, str]:
    """One-to-one ref→hyp mapping maximizing total co-active time. Exact
    (brute force) for realistic speaker counts, greedy beyond that."""
    if not ref_ids or not hyp_ids:
        return {}
    k = min(len(ref_ids), len(hyp_ids))
    big = max(len(ref_ids), len(hyp_ids))
    if math.perm(big, k) <= 200_000:
        best: dict[str, str] = {}
        best_score = -1.0
        if len(ref_ids) <= len(hyp_ids):
            for perm in itertools.permutations(hyp_ids, len(ref_ids)):
                score = sum(pair_overlap.get((r, h), 0.0) for r, h in zip(ref_ids, perm))
                if score > best_score:
                    best_score = score
                    best = dict(zip(ref_ids, perm))
        else:
            for perm in itertools.permutations(ref_ids, len(hyp_ids)):
                score = sum(pair_overlap.get((r, h), 0.0) for r, h in zip(perm, hyp_ids))
                if score > best_score:
                    best_score = score
                    best = dict(zip(perm, hyp_ids))
        return {r: h for r, h in best.items() if pair_overlap.get((r, h), 0.0) > 0}
    # Greedy fallback: take pairs by descending overlap, skipping used labels.
    used_r: set[str] = set()
    used_h: set[str] = set()
    mapping: dict[str, str] = {}
    for (r, h), ov in sorted(pair_overlap.items(), key=lambda kv: -kv[1]):
        if ov <= 0 or r in used_r or h in used_h:
            continue
        mapping[r] = h
        used_r.add(r)
        used_h.add(h)
    return mapping


def diarization_error_rate(
    ref_turns: list[dict[str, Any]],
    hyp_turns: list[dict[str, Any]],
    *,
    collar: float = 0.25,
) -> DERResult:
    """NIST-style DER. ``collar`` seconds on each side of every reference
    turn boundary are excluded from scoring (forgives small boundary
    disagreement, the convention in diarization benchmarks). Labels are
    matched by optimal mapping, so reference and hypothesis label names
    don't need to agree."""
    ref = _intervals_by_speaker(ref_turns)
    hyp = _intervals_by_speaker(hyp_turns)
    if not ref:
        raise ValueError("reference has no valid speech turns")

    excluded: list[tuple[float, float]] = []
    if collar > 0:
        edges = sorted(
            {b for ivs in ref.values() for s, e in ivs for b in (s, e)}
        )
        excluded = _merge_intervals([(b - collar, b + collar) for b in edges])

    points = sorted(
        {b for ivs in ref.values() for iv in ivs for b in iv}
        | {b for ivs in hyp.values() for iv in ivs for b in iv}
        | {b for iv in excluded for b in iv}
    )

    # One sweep over elementary intervals, collecting (duration, active sets).
    cells: list[tuple[float, frozenset[str], frozenset[str]]] = []
    pair_overlap: dict[tuple[str, str], float] = {}
    for lo, hi in zip(points, points[1:]):
        d = hi - lo
        if d <= 1e-9:
            continue
        mid = (lo + hi) / 2
        if any(s <= mid < e for s, e in excluded):
            continue
        ref_active = frozenset(
            spk for spk, ivs in ref.items() if any(s <= mid < e for s, e in ivs)
        )
        hyp_active = frozenset(
            spk for spk, ivs in hyp.items() if any(s <= mid < e for s, e in ivs)
        )
        if not ref_active and not hyp_active:
            continue
        cells.append((d, ref_active, hyp_active))
        for r in ref_active:
            for h in hyp_active:
                pair_overlap[(r, h)] = pair_overlap.get((r, h), 0.0) + d

    mapping = _optimal_mapping(sorted(ref), sorted(hyp), pair_overlap)

    miss = fa = conf = total = 0.0
    for d, ref_active, hyp_active in cells:
        nref, nhyp = len(ref_active), len(hyp_active)
        total += d * nref
        ncorrect = sum(1 for r in ref_active if mapping.get(r) in hyp_active)
        miss += d * max(0, nref - nhyp)
        fa += d * max(0, nhyp - nref)
        conf += d * (min(nref, nhyp) - ncorrect)

    if total <= 0:
        raise ValueError(
            "no scoreable reference speech — collar excluded everything "
            "(reference turns may be shorter than 2×collar)"
        )
    return DERResult(
        der=(miss + fa + conf) / total,
        miss_s=miss,
        false_alarm_s=fa,
        confusion_s=conf,
        ref_speech_s=total,
        mapping=mapping,
        ref_speakers=len(ref),
        hyp_speakers=len(hyp),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Wearer attribution
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class UserAttribution:
    accuracy: float  # fraction of reference-USER time labeled USER in the hypothesis
    ref_user_s: float


def user_attribution_accuracy(
    ref_turns: list[dict[str, Any]],
    hyp_turns: list[dict[str, Any]],
    *,
    label: str = "USER",
) -> UserAttribution | None:
    """Literal-label metric (no mapping): how much of the reference wearer's
    speech the system labeled ``USER``. Measures the product behavior — the
    label rendered in the UI — i.e. the talk-time heuristic plus everything
    upstream of it. Returns None when the reference contains no USER turns."""
    ref_ivs = _intervals_by_speaker(ref_turns).get(label)
    if not ref_ivs:
        return None
    hyp_ivs = _intervals_by_speaker(hyp_turns).get(label, [])
    total = sum(e - s for s, e in ref_ivs)
    hit = 0.0
    for rs, re_ in ref_ivs:
        for hs, he in hyp_ivs:
            hit += max(0.0, min(re_, he) - max(rs, hs))
    return UserAttribution(accuracy=hit / total, ref_user_s=total)
