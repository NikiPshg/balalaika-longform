"""WER over the covered region only (reviewer response, A4-eval2, 2026-08-31).

Motivation.  The frozen unconditional WER (PLAN.md §9.1) charges a truncated
reading for every source word it never attempted: an output that reads the
first 5 % of a B4 text perfectly scores WER ~ 0.95.  That is the intended
behaviour for the main table, but the reviewer asked for the decomposition
"how much text was read" x "how well was the read part rendered".  The first
factor is the frozen ``source_coverage`` / ``end_coverage``; this module adds
the second:

    wer_covered = (S + D + I inside the covered region) / (ref words inside)

where the *covered region* is the closed interval ``[first, last]`` of
reference-word indices, ``first``/``last`` being the first/last reference word
aligned to the hypothesis by a hit or a substitution -- exactly the words that
``source_coverage`` counts (:meth:`eval.alignment.Alignment.ref_aligned_mask`)
and the same "last aligned word" that defines the frozen ``end_coverage``
(:meth:`Alignment.last_aligned_ref_index`).

Op accounting inside the region (ops from :func:`eval.alignment.align_words`,
monotonic, each ref/hyp word consumed exactly once):

* substitutions -- always inside by construction (they are aligned words);
* deletions     -- counted iff the deleted reference word index lies in
                   ``[first, last]``; deletions before ``first`` or after
                   ``last`` are the *un-read* text and belong to the coverage
                   factor, not to this metric;
* insertions    -- an insert op sits at reference position ``p == ref_start``
                   (between ref words ``p-1`` and ``p``); it is counted iff
                   ``first < p <= last``, i.e. the extra hypothesis words were
                   spoken strictly inside the covered span.  Leading babble
                   (before the first aligned word) and trailing babble (after
                   the last) are excluded -- they are outside the region on
                   either reading of the definition.

Edge cases (all pre-registered here, never silently defaulted):

* empty hypothesis (missing WAV, empty transcription)  -> no aligned word ->
  ``wer_covered = None`` (undefined, NEVER 0.0), ``covered_ref_words = 0``;
* empty reference -> no aligned word -> ``wer_covered = None``;
* a hypothesis with hits/substitutions somewhere always has a well-defined
  non-empty region (``covered_ref_words >= 1``), so the denominator is never 0
  when the metric is defined;
* the metric is NOT capped: a looped output can have ``wer_covered`` far
  above 1.0 through insertions, which is intended (same rule as WER-all).

This is a *side* metric: nothing here touches the frozen per_item.jsonl files,
the §3.4 status rule or any §17 table cell.
"""

from __future__ import annotations

from typing import Sequence

from .alignment import Alignment, align_words
from .normalize import words as normalized_words

__all__ = ["covered_region", "covered_metrics", "covered_metrics_from_texts"]


def covered_region(al: Alignment) -> tuple[int, int] | None:
    """Closed interval ``(first, last)`` of aligned reference-word indices.

    ``None`` when no reference word is aligned (empty hypothesis, empty
    reference, or -- impossible with a Levenshtein alignment of two non-empty
    sequences, but handled anyway -- an alignment with no hit/substitution).
    """
    mask = al.ref_aligned_mask()
    first = next((i for i, f in enumerate(mask) if f), None)
    if first is None:
        return None
    last = al.last_aligned_ref_index()
    if last is None:  # defensive: cannot happen when first is not None
        return None
    return first, last


def covered_metrics(al: Alignment) -> dict:
    """The covered-region error decomposition for one alignment.

    Returns a dict with ``wer_covered`` (``None`` when undefined),
    ``covered_ref_words``, ``errors_covered`` and the S/D/I split plus the
    region boundaries, so the number can always be re-derived by hand.
    """
    region = covered_region(al)
    if region is None:
        return {
            "wer_covered": None,
            "covered_ref_words": 0,
            "errors_covered": 0,
            "hits_covered": 0,
            "substitutions_covered": 0,
            "deletions_covered": 0,
            "insertions_covered": 0,
            "covered_first_ref_index": None,
            "covered_last_ref_index": None,
            "covered_region_frac": 0.0,
        }
    first, last = region
    n_cov = last - first + 1
    hits_cov = subs_cov = dels_cov = ins_cov = 0
    for op in al.ops:
        kind = op["op"]
        if kind == "insert":
            p = op["ref_start"]
            if first < p <= last:
                ins_cov += op["hyp_end"] - op["hyp_start"]
            continue
        # overlap of [ref_start, ref_end) with [first, last+1)
        lo = max(op["ref_start"], first)
        hi = min(op["ref_end"], last + 1)
        n = max(0, hi - lo)
        if kind == "equal":
            hits_cov += n
        elif kind == "substitute":
            subs_cov += n
        elif kind == "delete":
            dels_cov += n
    errors = subs_cov + dels_cov + ins_cov
    return {
        "wer_covered": errors / n_cov,
        "covered_ref_words": n_cov,
        "errors_covered": errors,
        "hits_covered": hits_cov,
        "substitutions_covered": subs_cov,
        "deletions_covered": dels_cov,
        "insertions_covered": ins_cov,
        "covered_first_ref_index": first,
        "covered_last_ref_index": last,
        "covered_region_frac": (n_cov / al.n_ref) if al.n_ref > 0 else 0.0,
    }


def covered_metrics_from_texts(
    reference_text: str,
    hypothesis_text: str | None,
    variant: str = "lenient",
) -> dict:
    """Normalize with the frozen spec and score -- same path as content_metrics.

    Uses the same :func:`eval.normalize.words` and
    :func:`eval.alignment.align_words` as the frozen pipeline, so the alignment
    this metric reads is bit-identical to the one behind the stored WER.
    """
    ref_w = normalized_words(reference_text or "", variant)
    hyp_w = normalized_words(hypothesis_text or "", variant)
    al = align_words(ref_w, hyp_w)
    out = covered_metrics(al)
    out["normalization_variant"] = variant
    return out


def realign(reference_text: str, hypothesis_text: str | None,
            variant: str = "lenient") -> Alignment:
    """The frozen normalize + align pipeline, exposed for verification."""
    ref_w = normalized_words(reference_text or "", variant)
    hyp_w = normalized_words(hypothesis_text or "", variant)
    return align_words(ref_w, hyp_w)


def verify_against_row(al: Alignment, row: dict, atol: float = 1e-9) -> list[str]:
    """Check that a re-derived alignment reproduces the frozen per_item row.

    Returns a list of human-readable mismatch strings (empty = consistent).
    Used by scripts/wer_covered_backfill.py to guarantee that wer_covered is
    computed on the SAME alignment as the stored WER, not a lookalike.
    """
    problems: list[str] = []
    for key, got in (
        ("n_ref_words", al.n_ref), ("n_hyp_words", al.n_hyp),
        ("hits", al.hits), ("substitutions", al.substitutions),
        ("deletions", al.deletions), ("insertions", al.insertions),
    ):
        want = row.get(key)
        if want is not None and int(want) != int(got):
            problems.append(f"{key}: stored {want} != recomputed {got}")
    stored_wer = row.get("wer")
    if stored_wer is not None and al.n_ref > 0:
        rec = (al.substitutions + al.deletions + al.insertions) / al.n_ref
        if abs(float(stored_wer) - rec) > atol:
            problems.append(f"wer: stored {stored_wer} != recomputed {rec}")
    return problems
