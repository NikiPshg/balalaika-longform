"""Content and duration metrics for RuLongTTS evaluation (PLAN.md §9.1, §9.2).

Everything here operates on *already normalized* text or normalizes with the
frozen spec in :mod:`src.eval.normalize`.  Nothing is capped: a hypothesis that
loops can have WER far above 1.0, which is the intended behaviour (PLAN.md §9.1).

A hard failure without a WAV enters the metrics as an empty hypothesis, i.e. as
100 % deletions -- never as a missing row (PLAN.md §9.1, §3.4).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

from .alignment import Alignment, align_words
from .normalize import normalize, words as normalized_words

__all__ = [
    "content_metrics",
    "duration_metrics",
    "char_metrics",
    "excess_repetition",
    "ngram_run_extremes",
    "last_aligned_run",
    "aggregate",
    "DEFAULT_NGRAM_RANGE",
    "DEFAULT_TAIL_FRACTION",
    "DEFAULT_ROBUST_MIN_RUN",
]

DEFAULT_NGRAM_RANGE = (3, 8)
DEFAULT_TAIL_FRACTION = 0.10
# EndCoverage-robust (Lead, 2026-08-28): how many *consecutive* aligned
# reference words are needed before we believe the model was really reading
# there, instead of one word coinciding by chance.
DEFAULT_ROBUST_MIN_RUN = 3


# ---------------------------------------------------------------------------
# character level
# ---------------------------------------------------------------------------


def char_metrics(ref_norm: str, hyp_norm: str) -> dict:
    """CER and character S/D/I on already-normalized strings (spaces kept)."""
    n_ref = len(ref_norm)
    n_hyp = len(hyp_norm)
    if n_ref == 0 and n_hyp == 0:
        return {"n_ref_chars": 0, "n_hyp_chars": 0, "cer": 0.0,
                "char_substitutions": 0, "char_deletions": 0, "char_insertions": 0, "char_hits": 0}
    if n_hyp == 0:
        return {"n_ref_chars": n_ref, "n_hyp_chars": 0, "cer": 1.0,
                "char_substitutions": 0, "char_deletions": n_ref, "char_insertions": 0, "char_hits": 0}
    if n_ref == 0:
        return {"n_ref_chars": 0, "n_hyp_chars": n_hyp, "cer": float("inf"),
                "char_substitutions": 0, "char_deletions": 0, "char_insertions": n_hyp, "char_hits": 0}

    import jiwer

    transform = jiwer.Compose([jiwer.ReduceToListOfListOfChars()])
    out = jiwer.process_characters(
        ref_norm, hyp_norm, reference_transform=transform, hypothesis_transform=transform
    )
    errors = int(out.substitutions) + int(out.deletions) + int(out.insertions)
    return {
        "n_ref_chars": n_ref,
        "n_hyp_chars": n_hyp,
        "cer": errors / n_ref,
        "char_substitutions": int(out.substitutions),
        "char_deletions": int(out.deletions),
        "char_insertions": int(out.insertions),
        "char_hits": int(out.hits),
    }


# ---------------------------------------------------------------------------
# repetition
# ---------------------------------------------------------------------------


def _ngrams(seq: Sequence[str], n: int) -> Counter:
    if len(seq) < n:
        return Counter()
    return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))


def _adjacent_runs(seq: Sequence[str], n: int) -> dict[tuple[str, ...], int]:
    """``{n-gram: longest back-to-back repetition}`` for every n-gram repeated >= 2x.

    A *run* is the loop signature: the same phrase spoken immediately again, e.g.
    ``... и так далее и так далее и так далее`` is a run of 3 for n=3.  Counting
    runs instead of total occurrences is what separates a stuck decoder from a
    text that simply mentions the same phrase twice in different places.
    """
    runs: dict[tuple[str, ...], int] = {}
    length = len(seq)
    i = 0
    while i + n <= length:
        gram = tuple(seq[i : i + n])
        k = 1
        j = i + n
        while j + n <= length and tuple(seq[j : j + n]) == gram:
            k += 1
            j += n
        if k >= 2:
            if k > runs.get(gram, 0):
                runs[gram] = k
            i = j
        else:
            i += 1
    return runs


def _max_adjacent_run(seq: Sequence[str], gram: tuple[str, ...]) -> int:
    """Longest back-to-back repetition of exactly ``gram`` inside ``seq`` (>= 0)."""
    n = len(gram)
    best = 0
    length = len(seq)
    i = 0
    while i + n <= length:
        if tuple(seq[i : i + n]) == gram:
            k = 1
            j = i + n
            while j + n <= length and tuple(seq[j : j + n]) == gram:
                k += 1
                j += n
            best = max(best, k)
            i = j
        else:
            i += 1
    return best


def ngram_run_extremes(
    ref_words: Sequence[str],
    hyp_words: Sequence[str],
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
) -> dict:
    """The two extreme repetition statistics used by the loop rule (configs/eval.yaml).

    ``max_ngram_run``  -- the longest back-to-back repetition of any 3..8-word
        phrase in the hypothesis, *discounted by what the source itself repeats
        back-to-back* (PLAN.md §7.3: source repetition is not hallucination)::

            effective_run(g) = run_hyp(g) - (run_ref(g) - 1),   run_ref(g) >= 1

        so a source that already says the phrase twice in a row pays for one of
        the hypothesis' repeats.  ``1`` means "no back-to-back repetition at all".

    ``max_excess_ngram_count`` -- the largest number of *extra* occurrences of a
        single phrase anywhere in the hypothesis, ``count_hyp(g) - max(1, count_ref(g))``.
        Diagnostic only: it does not gate anything, because a long text can
        legitimately return to a phrase (measured on human references: max 2).
    """
    lo, hi = ngram_range
    best_run, best_run_n, best_run_gram = 1, None, None
    best_cnt, best_cnt_n, best_cnt_gram = 0, None, None
    for n in range(lo, hi + 1):
        for gram, run_hyp in _adjacent_runs(hyp_words, n).items():
            run_ref = max(1, _max_adjacent_run(ref_words, gram))
            effective = run_hyp - (run_ref - 1)
            if effective > best_run:
                best_run, best_run_n, best_run_gram = effective, n, " ".join(gram)
        ref_c = _ngrams(ref_words, n)
        for gram, c in _ngrams(hyp_words, n).items():
            excess = c - max(1, ref_c.get(gram, 0))
            if excess > best_cnt:
                best_cnt, best_cnt_n, best_cnt_gram = excess, n, " ".join(gram)
    return {
        "max_ngram_run": best_run,
        "max_ngram_run_n": best_run_n,
        "max_ngram_run_gram": best_run_gram,
        "max_excess_ngram_count": best_cnt,
        "max_excess_ngram_count_n": best_cnt_n,
        "max_excess_ngram_count_gram": best_cnt_gram,
    }


def excess_repetition(
    ref_words: Sequence[str],
    hyp_words: Sequence[str],
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
) -> dict:
    """n-gram repetition in the hypothesis beyond what the source already repeats.

    For every n in ``ngram_range`` and every n-gram g::

        excess(g) = max(0, (count_hyp(g) - 1) - max(0, count_ref(g) - 1))

    i.e. a source that legitimately repeats a phrase twice "pays for" one repeat
    in the hypothesis (PLAN.md §7.3: source repetition is not hallucination).
    The per-n rate divides by the number of hypothesis n-gram positions, and the
    reported rate is the mean over n.
    """
    lo, hi = ngram_range
    per_n: dict[str, float] = {}
    counts_n: dict[str, int] = {}
    rates: list[float] = []
    for n in range(lo, hi + 1):
        ref_c = _ngrams(ref_words, n)
        hyp_c = _ngrams(hyp_words, n)
        total_hyp = max(0, len(hyp_words) - n + 1)
        excess = 0
        for g, c in hyp_c.items():
            excess += max(0, (c - 1) - max(0, ref_c.get(g, 0) - 1))
        counts_n[f"n{n}"] = excess
        rate = excess / total_hyp if total_hyp > 0 else 0.0
        per_n[f"n{n}"] = rate
        if total_hyp > 0:
            rates.append(rate)
    out = {
        "excess_repetition_rate": (sum(rates) / len(rates)) if rates else 0.0,
        "excess_repetition_per_n": per_n,
        "excess_repetition_counts": counts_n,
    }
    out.update(ngram_run_extremes(ref_words, hyp_words, ngram_range))
    return out


# ---------------------------------------------------------------------------
# robust end of reading
# ---------------------------------------------------------------------------


def last_aligned_run(
    aligned_mask: Sequence[bool],
    min_run: int = DEFAULT_ROBUST_MIN_RUN,
) -> tuple[int | None, int, int]:
    """Last run of ``>= min_run`` consecutive aligned reference words.

    ``aligned_mask[i]`` is True when reference word *i* was matched by a hit or
    a substitution (:meth:`Alignment.ref_aligned_mask`).  A *run* is a maximal
    block of consecutive True positions in the **reference**; insertions do not
    interrupt it, because they consume no reference word.

    Returns ``(end_index, run_length, n_runs)`` where ``end_index`` is the
    0-based index of the last reference word of the LAST qualifying run (``None``
    when no run reaches ``min_run``), ``run_length`` is that run's length (0 when
    there is none) and ``n_runs`` counts all qualifying runs.

    Why this exists (Lead, 2026-08-28): with a nearly empty hypothesis the global
    alignment still scatters a few isolated matches deep into the reference --
    on pilot bucket B3 the median EndCoverage was 0.55 while only 3 % of the
    source words were covered at all -- so "index of the last aligned word" reads
    as "the model got halfway" when it in fact stopped after a few seconds.
    Requiring three consecutive words makes a single coincidence unable to move
    the end point.
    """
    if min_run < 1:
        raise ValueError(f"min_run must be >= 1, got {min_run}")
    end_index: int | None = None
    run_length = 0
    n_runs = 0
    run = 0
    for i, flag in enumerate(aligned_mask):
        if flag:
            run += 1
            if run == min_run:
                n_runs += 1
            if run >= min_run:
                end_index = i
                run_length = run
        else:
            run = 0
    return end_index, run_length, n_runs


# ---------------------------------------------------------------------------
# content metrics
# ---------------------------------------------------------------------------


def content_metrics(
    reference_text: str,
    hypothesis_text: str | None,
    variant: str = "lenient",
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    already_normalized: bool = False,
    robust_min_run: int = DEFAULT_ROBUST_MIN_RUN,
) -> dict:
    """All PLAN.md §9.1 content metrics for one (reference, hypothesis) pair.

    ``hypothesis_text=None`` or ``""`` is a hard failure: the whole reference is
    deleted, WER = 1.0, coverage = 0.0, EndCoverage = 0.0.

    Two end-of-reading numbers are reported side by side (Lead, 2026-08-28) and
    neither replaces the other:

    ``end_coverage``         the pre-registered metric -- position of the last
                             reference word aligned to anything, unchanged;
    ``end_coverage_robust``  position of the end of the last run of
                             ``robust_min_run`` consecutive aligned reference
                             words, 0.0 when no such run exists.  Only
                             ``end_coverage`` gates the PLAN.md §3.4 status.
    """
    if already_normalized:
        ref_norm = reference_text or ""
        hyp_norm = hypothesis_text or ""
        ref_w = ref_norm.split()
        hyp_w = hyp_norm.split()
    else:
        ref_norm = normalize(reference_text or "", variant)
        hyp_norm = normalize(hypothesis_text or "", variant)
        ref_w = normalized_words(reference_text or "", variant)
        hyp_w = normalized_words(hypothesis_text or "", variant)

    al: Alignment = align_words(ref_w, hyp_w)
    n_ref = al.n_ref
    n_hyp = al.n_hyp
    errors = al.substitutions + al.deletions + al.insertions
    wer = (errors / n_ref) if n_ref > 0 else (float("inf") if n_hyp > 0 else 0.0)

    aligned = al.hits + al.substitutions
    source_coverage = (aligned / n_ref) if n_ref > 0 else 0.0

    last_idx = al.last_aligned_ref_index()
    end_coverage = ((last_idx + 1) / n_ref) if (n_ref > 0 and last_idx is not None) else 0.0

    robust_idx, robust_run_len, n_robust_runs = last_aligned_run(
        al.ref_aligned_mask(), robust_min_run
    )
    end_coverage_robust = (
        ((robust_idx + 1) / n_ref) if (n_ref > 0 and robust_idx is not None) else 0.0
    )

    del_mask = al.ref_deleted_mask()
    tail_len = max(1, int(math.ceil(tail_fraction * n_ref))) if n_ref > 0 else 0
    if tail_len > 0:
        tail_dels = sum(del_mask[n_ref - tail_len :])
        tail_deletion_rate = tail_dels / tail_len
    else:
        tail_dels = 0
        tail_deletion_rate = 0.0

    longest_del = al.longest_deletion_run()

    out = {
        "normalization_variant": variant,
        "n_ref_words": n_ref,
        "n_hyp_words": n_hyp,
        "hits": al.hits,
        "substitutions": al.substitutions,
        "deletions": al.deletions,
        "insertions": al.insertions,
        "word_errors": errors,
        "wer": wer,
        "source_coverage": source_coverage,
        "end_coverage": end_coverage,
        "end_coverage_robust": end_coverage_robust,
        "end_coverage_robust_min_run": robust_min_run,
        "end_coverage_robust_run_words": robust_run_len,
        "n_aligned_runs_ge_min_run": n_robust_runs,
        "longest_deletion_run": longest_del,
        "longest_deletion_run_frac": (longest_del / n_ref) if n_ref > 0 else 0.0,
        "tail_fraction": tail_fraction,
        "tail_words": tail_len,
        "tail_deletions": int(tail_dels),
        "tail_deletion_rate": tail_deletion_rate,
    }
    out.update(char_metrics(ref_norm, hyp_norm))
    out.update(excess_repetition(ref_w, hyp_w, ngram_range))
    return out


# ---------------------------------------------------------------------------
# duration metrics
# ---------------------------------------------------------------------------


def duration_metrics(
    raw_duration_sec: float | None,
    segments: Sequence[dict] | None = None,
    reference_duration_sec: float | None = None,
    hyp_word_count: int | None = None,
) -> dict:
    """PLAN.md §9.2 duration metrics.

    ``segments`` are VAD segments ``{"start": s, "end": e}`` in seconds, already
    sorted by time; they define voiced duration, silence ratio and the longest
    silence (head and tail silence included).
    """
    raw = float(raw_duration_sec) if raw_duration_sec else 0.0
    segs = [s for s in (segments or []) if s.get("end") is not None and s.get("start") is not None]
    segs = sorted(segs, key=lambda s: (s["start"], s["end"]))

    # merge overlapping segments so voiced duration cannot exceed raw duration
    merged: list[tuple[float, float]] = []
    for s in segs:
        a, b = float(s["start"]), float(s["end"])
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    voiced = sum(b - a for a, b in merged)
    if merged:
        gaps = [merged[0][0]]  # head silence
        gaps += [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
        gaps.append(max(0.0, raw - merged[-1][1]))  # tail silence
        longest_silence = max(gaps) if gaps else 0.0
    else:
        longest_silence = raw

    silence_ratio = (1.0 - voiced / raw) if raw > 0 else 0.0
    ratio = (raw / reference_duration_sec) if reference_duration_sec else None
    voiced_ratio = (voiced / reference_duration_sec) if reference_duration_sec else None

    rate_raw = (hyp_word_count / (raw / 60.0)) if (hyp_word_count is not None and raw > 0) else None
    rate_voiced = (
        hyp_word_count / (voiced / 60.0) if (hyp_word_count is not None and voiced > 0) else None
    )

    return {
        "raw_duration_sec": raw,
        "voiced_duration_sec": voiced,
        "n_voiced_segments": len(merged),
        "silence_ratio": silence_ratio,
        "longest_silence_sec": longest_silence,
        "reference_duration_sec": reference_duration_sec,
        "duration_ratio": ratio,
        "voiced_duration_ratio": voiced_ratio,
        "speaking_rate_wpm_raw": rate_raw,
        "speaking_rate_wpm_voiced": rate_voiced,
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _mean_stats(values: Iterable[float]) -> tuple[float | None, int, int, int]:
    """Mean over the *finite* values, plus the bookkeeping needed to be honest.

    Returns ``(mean, n_inf, n_finite, n_missing)``:

    * ``mean``       -- arithmetic mean of the finite values, ``inf`` when every
                        present value is infinite, ``None`` when nothing is present;
    * ``n_inf``      -- how many items carried ``+/-inf`` and were therefore NOT in
                        the mean.  An infinite WER is a real outcome (empty
                        reference with a non-empty hypothesis), so it must never
                        disappear without a trace (PLAN.md §9.1, §0 rule 4);
    * ``n_finite``   -- how many items entered the mean;
    * ``n_missing``  -- how many items had ``None``/NaN for this metric.
    """
    vals: list[float] = []
    n_missing = 0
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            n_missing += 1
            continue
        vals.append(v)
    infs = [v for v in vals if isinstance(v, float) and math.isinf(v)]
    finite = [v for v in vals if not (isinstance(v, float) and math.isinf(v))]
    if not vals:
        return None, 0, 0, n_missing
    if not finite:
        return float("inf"), len(infs), 0, n_missing
    return sum(finite) / len(finite), len(infs), len(finite), n_missing


def _mean(values: Iterable[float]) -> float | None:
    """Backwards-compatible thin wrapper around :func:`_mean_stats`."""
    return _mean_stats(values)[0]


# metrics whose macro mean is reported together with an explicit inf/missing count
_MACRO_FIELDS = (
    ("macro_wer", "wer"),
    ("macro_cer", "cer"),
    ("macro_source_coverage", "source_coverage"),
    ("macro_end_coverage", "end_coverage"),
    ("macro_end_coverage_robust", "end_coverage_robust"),
    ("macro_longest_deletion_run", "longest_deletion_run"),
    ("macro_tail_deletion_rate", "tail_deletion_rate"),
    ("macro_excess_repetition_rate", "excess_repetition_rate"),
    ("macro_max_ngram_run", "max_ngram_run"),
    ("macro_duration_ratio", "duration_ratio"),
    ("macro_silence_ratio", "silence_ratio"),
    ("macro_speaking_rate_wpm_raw", "speaking_rate_wpm_raw"),
)


def aggregate(
    items: Sequence[dict],
    valid_key: str = "valid",
    complete_key: str = "status_complete",
) -> dict:
    """Macro and micro aggregation over per-item metric dicts.

    ``*_all``  -- over every attempted item, failures included as empty hypotheses.
    ``*_valid`` -- only over items with ``item[valid_key] is True``.

    micro WER = sum(errors) / sum(reference words); macro WER = mean of item WERs.
    Neither is capped at 1.0.

    Two different rates are reported and they are NOT the same number:

    ``complete_rate``  share of items whose generation status is ``complete``
                       (PLAN.md §3.4) -- this is the "Complete %" of the §17 table;
    ``valid_rate``     share of items that additionally have usable audio and a
                       usable transcription, i.e. the denominator of WER-valid.

    Every macro mean is accompanied by ``<name>_n_inf`` / ``<name>_n_finite`` /
    ``<name>_n_missing`` so an infinite WER can never be silently dropped.
    """

    def block(subset: Sequence[dict], suffix: str) -> dict:
        n = len(subset)
        if n == 0:
            return {f"n{suffix}": 0}
        sum_err = sum(i.get("word_errors", 0) for i in subset)
        sum_ref = sum(i.get("n_ref_words", 0) for i in subset)
        sum_cerr = sum(
            i.get("char_substitutions", 0) + i.get("char_deletions", 0) + i.get("char_insertions", 0)
            for i in subset
        )
        sum_cref = sum(i.get("n_ref_chars", 0) for i in subset)
        out = {
            f"n{suffix}": n,
            f"micro_wer{suffix}": (sum_err / sum_ref) if sum_ref else None,
            f"micro_cer{suffix}": (sum_cerr / sum_cref) if sum_cref else None,
            f"substitutions{suffix}": sum(i.get("substitutions", 0) for i in subset),
            f"deletions{suffix}": sum(i.get("deletions", 0) for i in subset),
            f"insertions{suffix}": sum(i.get("insertions", 0) for i in subset),
            # items with an empty reference: their WER/CER is +inf by definition
            f"n_zero_ref_items{suffix}": sum(1 for i in subset if not i.get("n_ref_words", 0)),
            # loop flag of the frozen rule in configs/eval.yaml (set by the evaluator)
            f"n_loop_flag{suffix}": sum(1 for i in subset if i.get("loop_flag") is True),
        }
        for name, key in _MACRO_FIELDS:
            mean, n_inf, n_finite, n_missing = _mean_stats(i.get(key) for i in subset)
            out[f"{name}{suffix}"] = mean
            out[f"{name}{suffix}_n_inf"] = n_inf
            out[f"{name}{suffix}_n_finite"] = n_finite
            out[f"{name}{suffix}_n_missing"] = n_missing
        return out

    valid = [i for i in items if i.get(valid_key) is True]
    complete = [i for i in items if i.get(complete_key) is True]
    out = block(items, "_all")
    out.update(block(valid, "_valid"))
    out["n_attempted"] = len(items)
    out["n_valid"] = len(valid)
    out["n_status_complete"] = len(complete)
    # PLAN.md §3.4 / Lead 2026-08-28: "Complete %" is the share of status == complete.
    out["complete_rate"] = (len(complete) / len(items)) if items else None
    # denominator of WER-valid, published as its own column
    out["valid_rate"] = (len(valid) / len(items)) if items else None
    # total number of infinite item-level WERs anywhere in this group
    out["n_inf_wer_all"] = out.get("macro_wer_all_n_inf", 0)
    out["n_inf_wer_valid"] = out.get("macro_wer_valid_n_inf", 0)
    return out
