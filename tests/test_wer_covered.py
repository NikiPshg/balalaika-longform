"""Synthetic alignment cases for the covered-region WER (A4-eval2, 2026-08-31).

The metric under test: wer_covered = (S+D+I inside [first, last] aligned source
word) / (ref words inside), region per src/eval/wer_covered.py.  Every case is
small enough to check by hand.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.alignment import align_words  # noqa: E402
from eval.wer_covered import (  # noqa: E402
    covered_metrics,
    covered_metrics_from_texts,
    covered_region,
    realign,
    verify_against_row,
)


def cov(ref: str, hyp: str) -> dict:
    return covered_metrics(align_words(ref.split(), hyp.split()))


# ---------------------------------------------------------------------------
# perfect / trivial
# ---------------------------------------------------------------------------


def test_perfect_reading_is_zero():
    m = cov("a b c d", "a b c d")
    assert m["wer_covered"] == 0.0
    assert m["covered_ref_words"] == 4
    assert m["errors_covered"] == 0
    assert (m["covered_first_ref_index"], m["covered_last_ref_index"]) == (0, 3)
    assert m["covered_region_frac"] == 1.0


def test_empty_hypothesis_is_undefined_not_zero():
    """A missing WAV must never look like a perfect covered reading."""
    m = cov("a b c d", "")
    assert m["wer_covered"] is None
    assert m["covered_ref_words"] == 0
    assert m["errors_covered"] == 0


def test_empty_reference_is_undefined():
    m = cov("", "a b")
    assert m["wer_covered"] is None
    assert m["covered_ref_words"] == 0


def test_both_empty_is_undefined():
    m = cov("", "")
    assert m["wer_covered"] is None


# ---------------------------------------------------------------------------
# truncation: the reviewer's core case
# ---------------------------------------------------------------------------


def test_perfect_prefix_truncation_has_zero_covered_wer():
    """Reads the first half perfectly, then stops: WER-all ~ 0.5, covered 0."""
    ref = "w1 w2 w3 w4 w5 w6 w7 w8"
    m = cov(ref, "w1 w2 w3 w4")
    assert m["wer_covered"] == 0.0
    assert m["covered_ref_words"] == 4
    assert (m["covered_first_ref_index"], m["covered_last_ref_index"]) == (0, 3)
    # the un-read tail is the coverage factor, not an error of this metric
    assert m["deletions_covered"] == 0


def test_trailing_deletions_outside_region_do_not_count():
    m = cov("a b c d e f", "a b x")  # x substitutes c; d e f never attempted
    assert (m["covered_first_ref_index"], m["covered_last_ref_index"]) == (0, 2)
    assert m["substitutions_covered"] == 1
    assert m["deletions_covered"] == 0
    assert m["wer_covered"] == pytest.approx(1 / 3)


def test_leading_deletions_outside_region_do_not_count():
    m = cov("a b c d e f", "d e f")  # starts reading in the middle
    assert (m["covered_first_ref_index"], m["covered_last_ref_index"]) == (3, 5)
    assert m["covered_ref_words"] == 3
    assert m["wer_covered"] == 0.0


# ---------------------------------------------------------------------------
# errors inside the region
# ---------------------------------------------------------------------------


def test_substitution_inside_region():
    m = cov("a b c d", "a x c d")
    assert m["wer_covered"] == pytest.approx(1 / 4)
    assert m["substitutions_covered"] == 1


def test_deletion_inside_region_counts():
    m = cov("a b c d e", "a b d e")  # c skipped mid-reading
    assert (m["covered_first_ref_index"], m["covered_last_ref_index"]) == (0, 4)
    assert m["deletions_covered"] == 1
    assert m["wer_covered"] == pytest.approx(1 / 5)


def test_insertion_inside_region_counts():
    m = cov("a b c", "a x b c")  # extra word between a and b
    assert m["insertions_covered"] == 1
    assert m["covered_ref_words"] == 3
    assert m["wer_covered"] == pytest.approx(1 / 3)


def test_leading_insertion_outside_region_does_not_count():
    m = cov("a b c", "x a b c")
    assert m["insertions_covered"] == 0
    assert m["wer_covered"] == 0.0


def test_trailing_insertion_outside_region_does_not_count():
    """Babble after the last aligned word is not an error of the read part."""
    m = cov("a b c", "a b c x y")
    assert m["insertions_covered"] == 0
    assert m["wer_covered"] == 0.0


def test_loop_insertions_can_push_covered_wer_above_one():
    """Uncapped by design: a loop inside the read span is charged in full."""
    m = cov("a b c d", "a q q q q q q q q b c d")
    assert m["insertions_covered"] == 8
    assert m["wer_covered"] == pytest.approx(8 / 4)


# ---------------------------------------------------------------------------
# region boundaries
# ---------------------------------------------------------------------------


def test_single_aligned_word_region():
    # hyp shares exactly one token with ref, aligned as the only hit
    m = cov("a b c", "b")
    assert (m["covered_first_ref_index"], m["covered_last_ref_index"]) == (1, 1)
    assert m["covered_ref_words"] == 1
    assert m["wer_covered"] == 0.0


def test_disjoint_texts_still_have_a_region_via_substitutions():
    # Levenshtein prefers substitutions over delete+insert, so a non-empty
    # hyp against a non-empty ref always aligns something
    m = cov("a b c", "x y z")
    assert m["covered_ref_words"] == 3
    assert m["wer_covered"] == 1.0


def test_region_matches_source_coverage_and_end_coverage_definitions():
    """first/last come from the same masks the frozen metrics use."""
    ref = "a b c d e f g h".split()
    hyp = "c d x f".split()
    al = align_words(ref, hyp)
    region = covered_region(al)
    mask = al.ref_aligned_mask()
    assert region == (mask.index(True), al.last_aligned_ref_index())
    m = covered_metrics(al)
    # every aligned (hit+sub) ref word lies inside the region by construction
    assert sum(mask) == m["hits_covered"] + m["substitutions_covered"]


# ---------------------------------------------------------------------------
# identity S+D+I bookkeeping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ref,hyp", [
    ("a b c d e f g", "a b x d f g q"),
    ("a b c d e f g", "c d e"),
    ("a a a b b b c", "a b c"),
    ("a b c", "a b c"),
    ("один два три четыре пять", "один два четыре пять шесть"),
])
def test_covered_errors_never_exceed_total_errors(ref, hyp):
    al = align_words(ref.split(), hyp.split())
    m = covered_metrics(al)
    total = al.substitutions + al.deletions + al.insertions
    assert m["errors_covered"] <= total
    assert m["hits_covered"] + m["substitutions_covered"] + m["deletions_covered"] \
        == m["covered_ref_words"]


# ---------------------------------------------------------------------------
# text-level entry point uses the frozen normalizer
# ---------------------------------------------------------------------------


def test_from_texts_uses_frozen_normalizer():
    # ё→е + lowercase + punctuation stripping come from the frozen spec
    m = covered_metrics_from_texts("Ещё, раз!", "еще раз", variant="lenient")
    assert m["wer_covered"] == 0.0
    assert m["covered_ref_words"] == 2
    assert m["normalization_variant"] == "lenient"


def test_verify_against_row_detects_drift():
    al = realign("раз два три", "раз два три")
    assert verify_against_row(al, {"hits": 3, "wer": 0.0, "n_ref_words": 3}) == []
    bad = verify_against_row(al, {"hits": 2})
    assert bad and "hits" in bad[0]
