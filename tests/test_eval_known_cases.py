"""Known-case tests for the RuLongTTS evaluator (PLAN.md §14 A4 acceptance).

Covers pure deletion, pure insertion, substitution, loop (WER > 100 %),
early stop, empty hypothesis, and the frozen normalization rules.

Run:  .venv-eval/bin/python -m pytest tests/test_eval_known_cases.py -q
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.alignment import align_words  # noqa: E402
from eval.metrics import (  # noqa: E402
    DEFAULT_ROBUST_MIN_RUN,
    aggregate,
    content_metrics,
    duration_metrics,
    excess_repetition,
    last_aligned_run,
)
from eval.normalize import (  # noqa: E402
    load_spec,
    normalize_lenient,
    normalize_strict,
    number_to_russian_words,
    words,
)

REF10 = "один два три четыре пять шесть семь восемь девять десять"
SENT = "мы поехали в город рано утром"


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------


def test_alignment_pure_deletion():
    al = align_words(REF10.split(), REF10.split()[:6])
    assert al.hits == 6
    assert al.substitutions == 0
    assert al.insertions == 0
    assert al.deletions == 4
    assert al.longest_deletion_run() == 4
    assert al.last_aligned_ref_index() == 5


def test_alignment_pure_insertion():
    hyp = REF10.split() + ["лишний", "хвост"]
    al = align_words(REF10.split(), hyp)
    assert al.hits == 10
    assert al.deletions == 0
    assert al.substitutions == 0
    assert al.insertions == 2


def test_alignment_pure_substitution():
    hyp = REF10.split()
    hyp[3] = "ЧЕТЫРНАДЦАТЬ".lower()
    al = align_words(REF10.split(), hyp)
    assert al.substitutions == 1
    assert al.deletions == al.insertions == 0
    assert al.hits == 9


def test_alignment_covers_every_word_once():
    al = align_words(REF10.split(), "один два ЧТО-ТО пять".lower().split())
    ref_touch = [0] * al.n_ref
    hyp_touch = [0] * al.n_hyp
    for op in al.ops:
        for i in range(op["ref_start"], op["ref_end"]):
            ref_touch[i] += 1
        for j in range(op["hyp_start"], op["hyp_end"]):
            hyp_touch[j] += 1
    assert ref_touch == [1] * al.n_ref
    assert hyp_touch == [1] * al.n_hyp


# ---------------------------------------------------------------------------
# content metrics: the four canonical error types
# ---------------------------------------------------------------------------


def test_pure_deletion_metrics():
    m = content_metrics(REF10, " ".join(REF10.split()[:6]))
    assert m["deletions"] == 4
    assert m["substitutions"] == 0
    assert m["insertions"] == 0
    assert m["wer"] == pytest.approx(0.4)
    assert m["source_coverage"] == pytest.approx(0.6)
    assert m["end_coverage"] == pytest.approx(0.6)
    assert m["longest_deletion_run"] == 4
    assert m["tail_deletion_rate"] == pytest.approx(1.0)


def test_pure_insertion_metrics():
    m = content_metrics(REF10, REF10 + " лишний хвост")
    assert m["insertions"] == 2
    assert m["deletions"] == m["substitutions"] == 0
    assert m["wer"] == pytest.approx(0.2)
    assert m["source_coverage"] == pytest.approx(1.0)
    assert m["end_coverage"] == pytest.approx(1.0)
    assert m["tail_deletion_rate"] == pytest.approx(0.0)


def test_pure_substitution_metrics():
    hyp = REF10.split()
    hyp[0] = "адин"
    hyp[9] = "десятка"
    m = content_metrics(REF10, " ".join(hyp))
    assert m["substitutions"] == 2
    assert m["deletions"] == m["insertions"] == 0
    assert m["wer"] == pytest.approx(0.2)
    # a substitution still counts as "aligned" source
    assert m["source_coverage"] == pytest.approx(1.0)
    assert m["end_coverage"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# loop
# ---------------------------------------------------------------------------


def test_loop_wer_above_100_percent_and_excess_repetition():
    hyp = " ".join([SENT] * 5)
    m = content_metrics(SENT, hyp)
    assert m["wer"] > 1.0, "WER must not be capped at 100 %"
    assert m["wer"] == pytest.approx(4.0)  # 4 extra copies of 6 words / 6 words
    assert m["insertions"] == 24
    assert m["excess_repetition_rate"] > 0.0
    assert m["excess_repetition_counts"]["n3"] > 0
    assert m["excess_repetition_counts"]["n8"] > 0
    assert m["source_coverage"] == pytest.approx(1.0)


def test_source_repetition_is_not_penalized():
    """A source that already repeats a phrase must not be scored as a loop."""
    ref = " ".join([SENT] * 3)
    perfect = content_metrics(ref, ref)
    assert perfect["wer"] == pytest.approx(0.0)
    assert perfect["excess_repetition_rate"] == pytest.approx(0.0)
    # one extra copy beyond the source is excess
    looped = content_metrics(ref, " ".join([SENT] * 5))
    assert looped["excess_repetition_rate"] > 0.0


def test_excess_repetition_ngram_range():
    r = excess_repetition(SENT.split(), (SENT + " " + SENT).split())
    assert set(r["excess_repetition_per_n"]) == {"n3", "n4", "n5", "n6", "n7", "n8"}
    assert r["excess_repetition_rate"] > 0.0


# ---------------------------------------------------------------------------
# early stop / truncation
# ---------------------------------------------------------------------------


def test_early_stop_end_coverage_half_and_tail_deletion_one():
    ref = " ".join(f"слово{i}" for i in range(100))
    hyp = " ".join(f"слово{i}" for i in range(50))
    m = content_metrics(ref, hyp, variant="strict")
    assert m["end_coverage"] == pytest.approx(0.5)
    assert m["source_coverage"] == pytest.approx(0.5)
    assert m["tail_deletion_rate"] == pytest.approx(1.0)
    assert m["longest_deletion_run"] == 50
    assert m["wer"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# EndCoverage-robust (Lead, 2026-08-28)
# ---------------------------------------------------------------------------

REF100 = " ".join(f"слово{i}" for i in range(100))


def test_last_aligned_run_picks_the_last_qualifying_run():
    mask = [True, True, True, False, True, True, False, True, True, True, True]
    assert last_aligned_run(mask, 3) == (10, 4, 2)   # last run 7..10, two runs of >= 3
    assert last_aligned_run(mask, 5) == (None, 0, 0)  # nothing that long
    assert last_aligned_run([True, True], 3) == (None, 0, 0)
    assert last_aligned_run([], 3) == (None, 0, 0)
    assert last_aligned_run([False] * 5, 1) == (None, 0, 0)
    assert last_aligned_run([True, False, True], 1) == (2, 1, 2)


def test_last_aligned_run_rejects_a_nonsense_min_run():
    with pytest.raises(ValueError):
        last_aligned_run([True, True, True], 0)


def test_end_coverage_robust_default_min_run_is_three():
    assert DEFAULT_ROBUST_MIN_RUN == 3
    m = content_metrics(REF100, "слово0 слово1 слово2", variant="strict")
    assert m["end_coverage_robust_min_run"] == 3
    assert m["end_coverage_robust"] == pytest.approx(0.03)
    assert m["end_coverage_robust_run_words"] == 3


def test_end_coverage_robust_ignores_one_late_coincidence():
    """The pilot B3 pathology: the model read 5 words, one word matched at 55 %."""
    hyp = " ".join(f"слово{i}" for i in range(5)) + " слово55"
    m = content_metrics(REF100, hyp, variant="strict")
    assert m["end_coverage"] == pytest.approx(0.56), "the pre-registered metric is unchanged"
    assert m["end_coverage_robust"] == pytest.approx(0.05), "robust follows the real reading"
    assert m["source_coverage"] == pytest.approx(0.06)
    assert m["n_aligned_runs_ge_min_run"] == 1


def test_end_coverage_robust_is_zero_when_every_match_is_isolated():
    """Sparse spurious matches: EndCoverage high, robust exactly 0."""
    m = content_metrics(REF100, "слово10 слово50 слово90", variant="strict")
    assert m["end_coverage"] == pytest.approx(0.91)
    assert m["source_coverage"] == pytest.approx(0.03)
    assert m["end_coverage_robust"] == 0.0
    assert m["end_coverage_robust_run_words"] == 0
    assert m["n_aligned_runs_ge_min_run"] == 0


def test_end_coverage_robust_survives_a_two_word_tail_of_noise():
    """Ten words read, then two isolated late matches: robust stays at the read part."""
    hyp = " ".join(f"слово{i}" for i in range(10)) + " слово80 слово81"
    m = content_metrics(REF100, hyp, variant="strict")
    assert m["end_coverage"] == pytest.approx(0.82)
    assert m["end_coverage_robust"] == pytest.approx(0.10)


def test_end_coverage_robust_equals_end_coverage_on_a_clean_read():
    full = content_metrics(REF100, REF100, variant="strict")
    assert full["end_coverage"] == full["end_coverage_robust"] == pytest.approx(1.0)
    half = content_metrics(REF100, " ".join(f"слово{i}" for i in range(50)), variant="strict")
    assert half["end_coverage"] == half["end_coverage_robust"] == pytest.approx(0.5)


def test_end_coverage_robust_counts_substitutions_as_aligned():
    """Frozen definition: a run is hit OR substitution, not hits only.

    Also documents the price: a correct reading whose last two words survive a
    deletion ends at the last run of >= 3, so robust can sit below EndCoverage
    even when the model did reach the end.
    """
    m = content_metrics("а б в г д е ж з и к", "а б в г x y z и к", variant="strict")
    assert m["substitutions"] == 3 and m["deletions"] == 1
    assert m["end_coverage"] == pytest.approx(1.0)
    assert m["end_coverage_robust"] == pytest.approx(0.7)


@pytest.mark.parametrize("hyp", ["", None, "   "])
def test_end_coverage_robust_of_an_empty_hypothesis_is_zero(hyp):
    m = content_metrics(REF10, hyp)
    assert m["end_coverage_robust"] == 0.0
    assert m["n_aligned_runs_ge_min_run"] == 0


def test_end_coverage_robust_never_exceeds_end_coverage():
    """Structural: the robust end point is one of the aligned positions."""
    import random

    rng = random.Random(20260828)
    ref = REF100.split()
    for _ in range(200):
        hyp = [w for w in ref if rng.random() < 0.3]
        if rng.random() < 0.5:
            hyp += [f"чужое{rng.randrange(50)}" for _ in range(rng.randrange(5))]
        m = content_metrics(" ".join(ref), " ".join(hyp), variant="strict")
        assert m["end_coverage_robust"] <= m["end_coverage"] + 1e-12
        assert 0.0 <= m["end_coverage_robust"] <= 1.0


def test_end_coverage_robust_min_run_one_reproduces_end_coverage():
    hyp = " ".join(f"слово{i}" for i in range(5)) + " слово55"
    m = content_metrics(REF100, hyp, variant="strict", robust_min_run=1)
    assert m["end_coverage_robust"] == pytest.approx(m["end_coverage"])


def test_aggregate_reports_macro_end_coverage_robust():
    items = [
        {"end_coverage": 0.91, "end_coverage_robust": 0.0, "valid": True,
         "status_complete": False, "n_ref_words": 100, "word_errors": 97},
        {"end_coverage": 1.0, "end_coverage_robust": 1.0, "valid": True,
         "status_complete": True, "n_ref_words": 100, "word_errors": 5},
    ]
    agg = aggregate(items)
    assert agg["macro_end_coverage_all"] == pytest.approx(0.955)
    assert agg["macro_end_coverage_robust_all"] == pytest.approx(0.5)
    assert agg["macro_end_coverage_robust_all_n_finite"] == 2
    assert agg["macro_end_coverage_robust_all_n_missing"] == 0


def test_aggregate_counts_a_missing_end_coverage_robust_instead_of_inventing_it():
    agg = aggregate([{"end_coverage_robust": 0.4, "valid": True, "n_ref_words": 10},
                     {"valid": True, "n_ref_words": 10}])
    assert agg["macro_end_coverage_robust_all"] == pytest.approx(0.4)
    assert agg["macro_end_coverage_robust_all_n_missing"] == 1


# ---------------------------------------------------------------------------
# hard failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hyp", ["", None, "   "])
def test_empty_hypothesis_is_full_deletion(hyp):
    m = content_metrics(REF10, hyp)
    assert m["wer"] == pytest.approx(1.0)
    assert m["cer"] == pytest.approx(1.0)
    assert m["deletions"] == 10
    assert m["substitutions"] == m["insertions"] == 0
    assert m["source_coverage"] == pytest.approx(0.0)
    assert m["end_coverage"] == pytest.approx(0.0)
    assert m["longest_deletion_run"] == 10
    assert m["tail_deletion_rate"] == pytest.approx(1.0)
    assert m["excess_repetition_rate"] == pytest.approx(0.0)


def test_empty_reference_is_all_insertions():
    m = content_metrics("", "что-то лишнее")  # hyphen splits -> 3 words
    assert m["n_ref_words"] == 0
    assert m["n_hyp_words"] == 3
    assert m["insertions"] == 3
    assert math.isinf(m["wer"])


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------


def test_normalization_yo_and_case():
    assert normalize_strict("Ёжик, ёлка и ЁРШ") == "ежик елка и ерш"
    assert normalize_strict("ежик елка и ерш") == "ежик елка и ерш"


def test_normalization_punctuation_removed():
    assert normalize_strict("Привет, мир!  Как -- дела?") == "привет мир как дела"
    assert normalize_strict("«Цитата» — это, кстати, тест...") == "цитата это кстати тест"


def test_normalization_hyphen_splits_words():
    assert normalize_strict("какой-то") == "какой то"


def test_normalization_strict_keeps_digits():
    assert normalize_strict("в 1990 году") == "в 1990 году"


def test_normalization_lenient_expands_digits():
    assert normalize_lenient("в 1990 году") == "в одна тысяча девятьсот девяносто году"
    assert normalize_lenient("25 %") == "двадцать пять"
    assert normalize_lenient("дом 7") == "дом семь"


def test_normalization_lenient_leading_zeros_digit_by_digit():
    assert normalize_lenient("007") == "ноль ноль семь"


def test_normalization_variant_changes_wer():
    ref = "в одна тысяча девятьсот девяносто году"
    hyp = "в 1990 году"
    strict = content_metrics(ref, hyp, variant="strict")
    lenient = content_metrics(ref, hyp, variant="lenient")
    assert strict["wer"] > 0.0
    assert lenient["wer"] == pytest.approx(0.0)


def test_words_helper():
    assert words("Ёлка, 2 шт.", variant="lenient") == ["елка", "два", "шт"]


def test_number_to_russian_words_matches_num2words():
    """Covers exactly the range the reports claim: 0…2999 and 10^3…10^24.

    (Reviewer, non-blocking 11: the old test stopped at 1199 while
    reports/handoff_A4.md claimed 0…2999 and the powers of ten.)
    """
    num2words = pytest.importorskip("num2words").num2words
    cases = list(range(0, 3000)) + [10**k for k in range(3, 25)]
    cases += [1990, 2026, 100000, 123456789, 999999999999]
    for n in cases:
        assert number_to_russian_words(n) == num2words(n, lang="ru"), n


def test_normalization_spec_file_matches_implementation():
    spec = load_spec()
    assert spec["version"] == 1
    assert spec["primary_variant"] in ("strict", "lenient")


# ---------------------------------------------------------------------------
# duration metrics
# ---------------------------------------------------------------------------


def test_duration_metrics_basic():
    d = duration_metrics(
        100.0,
        [{"start": 5.0, "end": 45.0}, {"start": 60.0, "end": 95.0}],
        reference_duration_sec=80.0,
        hyp_word_count=150,
    )
    assert d["voiced_duration_sec"] == pytest.approx(75.0)
    assert d["silence_ratio"] == pytest.approx(0.25)
    assert d["longest_silence_sec"] == pytest.approx(15.0)  # 45 -> 60
    assert d["duration_ratio"] == pytest.approx(1.25)
    assert d["speaking_rate_wpm_raw"] == pytest.approx(90.0)


def test_duration_metrics_no_segments_all_silence():
    d = duration_metrics(30.0, [], reference_duration_sec=30.0, hyp_word_count=0)
    assert d["voiced_duration_sec"] == 0.0
    assert d["silence_ratio"] == pytest.approx(1.0)
    assert d["longest_silence_sec"] == pytest.approx(30.0)


def test_duration_metrics_overlapping_segments_merged():
    d = duration_metrics(20.0, [{"start": 0.0, "end": 10.0}, {"start": 5.0, "end": 12.0}])
    assert d["voiced_duration_sec"] == pytest.approx(12.0)
    assert d["n_voiced_segments"] == 1


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def test_aggregate_macro_and_micro_and_unconditional_failures():
    good = content_metrics(REF10, REF10)
    good["valid"] = True
    good["status_complete"] = True
    failed = content_metrics(REF10 + " " + REF10, "")  # 20 reference words, all deleted
    failed["valid"] = False
    failed["status_complete"] = False
    agg = aggregate([good, failed])
    assert agg["n_attempted"] == 2
    assert agg["n_valid"] == 1
    assert agg["complete_rate"] == pytest.approx(0.5)
    # macro = mean(0.0, 1.0)
    assert agg["macro_wer_all"] == pytest.approx(0.5)
    # micro = 20 errors / 30 reference words
    assert agg["micro_wer_all"] == pytest.approx(20 / 30)
    assert agg["macro_wer_valid"] == pytest.approx(0.0)
    assert agg["deletions_all"] == 20


def test_aggregate_not_capped():
    loop = content_metrics(SENT, " ".join([SENT] * 5))
    loop["valid"] = True
    loop["status_complete"] = True
    agg = aggregate([loop])
    assert agg["macro_wer_all"] > 1.0
    assert agg["micro_wer_all"] > 1.0


def test_aggregate_complete_rate_is_status_not_validity():
    """PLAN.md §3.4 / Lead 2026-08-28: 'Complete %' counts status == complete only.

    A run can be `complete` and still be unusable for WER-valid (its WAV was
    deleted, the ASR returned nothing).  The two rates must not collapse into one.
    """
    complete_and_usable = content_metrics(REF10, REF10)
    complete_and_usable.update(status_complete=True, valid=True)
    complete_but_no_audio = content_metrics(REF10, "")
    complete_but_no_audio.update(status_complete=True, valid=False)
    loop_cap_with_audio = content_metrics(REF10, REF10)
    loop_cap_with_audio.update(status_complete=False, valid=False)

    agg = aggregate([complete_and_usable, complete_but_no_audio, loop_cap_with_audio])
    assert agg["n_status_complete"] == 2
    assert agg["complete_rate"] == pytest.approx(2 / 3)
    assert agg["n_valid"] == 1
    assert agg["valid_rate"] == pytest.approx(1 / 3)
    assert agg["complete_rate"] != agg["valid_rate"]


def test_aggregate_counts_infinite_wer_instead_of_dropping_it():
    """An empty reference gives WER = inf; it must never vanish from the table."""
    finite = content_metrics(REF10, REF10)
    finite.update(status_complete=True, valid=True)
    infinite = content_metrics("", "текст которого в референсе нет")
    infinite.update(status_complete=True, valid=True)
    assert math.isinf(infinite["wer"])

    agg = aggregate([finite, infinite])
    assert agg["macro_wer_all_n_inf"] == 1
    assert agg["macro_wer_all_n_finite"] == 1
    assert agg["macro_cer_all_n_inf"] == 1
    assert agg["n_zero_ref_items_all"] == 1
    assert agg["n_inf_wer_all"] == 1
    # the reported mean is over the finite values only, and says so via the count
    assert agg["macro_wer_all"] == pytest.approx(0.0)


def test_aggregate_all_infinite_stays_infinite():
    only_inf = content_metrics("", "лишние слова")
    only_inf.update(status_complete=True, valid=True)
    agg = aggregate([only_inf])
    assert math.isinf(agg["macro_wer_all"])
    assert agg["macro_wer_all_n_inf"] == 1
    assert agg["macro_wer_all_n_finite"] == 0


# ---------------------------------------------------------------------------
# scripts/run_evaluation.py: canonical schema (PLAN.md §7.5) and the
# "transcribe every wav that exists" rule (Lead, 2026-08-28)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_evaluation as RE  # noqa: E402


def _bench_row(text_id="t1", **over):
    row = {
        "text_id": text_id,
        "root_id": "r1",
        "bucket": "B0",
        "genre": 1,
        "source": "dataset",
        "text_tts": "Один два три.",
        "text_ref": REF10,
        "human_audio_path": "/some/where.flac",
        "human_offset_start": 0.0,
        "human_offset_end": 30.0,
        "human_duration_sec": 30.0,
        "words": 10,
        "chars": 50,
        "sentences": 1,
    }
    row.update(over)
    return row


def _run_row(run_id="r1", text_id="t1", status="complete", output_path=None, **over):
    row = {
        "run_id": run_id,
        "experiment_id": "E1",
        "text_id": text_id,
        "voice_id": "v1",
        "seed": 0,
        "status": status,
        "stop_reason": "eos",
        "output_path": output_path,
        "raw_duration_sec": 30.0,
    }
    row.update(over)
    return row


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(path)


def test_benchmark_accepts_canonical_schema(tmp_path):
    p = _write_jsonl(tmp_path / "b.jsonl", [_bench_row()])
    by_id = RE.load_benchmark(p)
    assert list(by_id) == ["t1"]


def test_benchmark_rejects_missing_canonical_field(tmp_path):
    """No synonym fallback: `text` instead of `text_ref` is an error, not a guess."""
    bad = _bench_row()
    del bad["text_ref"]
    bad["text"] = REF10                      # A1-manifest spelling
    bad["sample_id"] = "t1"                  # A1-manifest spelling
    p = _write_jsonl(tmp_path / "b.jsonl", [bad])
    with pytest.raises(RE.BenchmarkContractError) as e:
        RE.load_benchmark(p)
    assert "text_ref" in str(e.value)


def test_benchmark_rejects_null_offsets_with_human_audio(tmp_path):
    p = _write_jsonl(tmp_path / "b.jsonl",
                     [_bench_row(human_offset_start=None, human_offset_end=None)])
    with pytest.raises(RE.BenchmarkContractError) as e:
        RE.load_benchmark(p)
    assert "offsets are mandatory" in str(e.value).replace("\n", " ")


def test_benchmark_rejects_slice_shorter_than_one_second(tmp_path):
    p = _write_jsonl(tmp_path / "b.jsonl",
                     [_bench_row(human_offset_start=10.0, human_offset_end=10.4)])
    with pytest.raises(RE.BenchmarkContractError) as e:
        RE.load_benchmark(p)
    assert "shorter than the 1 s minimum" in str(e.value)


def test_benchmark_allows_external_text_without_human_audio(tmp_path):
    p = _write_jsonl(tmp_path / "b.jsonl", [_bench_row(
        source="external", human_audio_path=None,
        human_offset_start=None, human_offset_end=None, human_duration_sec=33.2)])
    assert list(RE.load_benchmark(p)) == ["t1"]


def test_benchmark_rejects_duplicate_text_id(tmp_path):
    p = _write_jsonl(tmp_path / "b.jsonl", [_bench_row(), _bench_row()])
    with pytest.raises(RE.BenchmarkContractError) as e:
        RE.load_benchmark(p)
    assert "duplicate text_id" in str(e.value)


def test_runs_reject_text_id_absent_from_benchmark(tmp_path):
    bp = _write_jsonl(tmp_path / "b.jsonl", [_bench_row()])
    rp = _write_jsonl(tmp_path / "r.jsonl", [_run_row(text_id="t_NOT_IN_BENCHMARK")])
    by_id = RE.load_benchmark(bp)
    with pytest.raises(RE.BenchmarkContractError) as e:
        RE.load_runs([rp], by_id)
    assert "absent from the benchmark" in str(e.value).replace("\n", " ")


def test_runs_reject_missing_canonical_field(tmp_path):
    bp = _write_jsonl(tmp_path / "b.jsonl", [_bench_row()])
    bad = _run_row()
    del bad["text_id"]
    bad["sample_id"] = "t1"
    rp = _write_jsonl(tmp_path / "r.jsonl", [bad])
    with pytest.raises(RE.BenchmarkContractError) as e:
        RE.load_runs([rp], RE.load_benchmark(bp))
    assert "text_id" in str(e.value)


class _FakeTranscriber:
    """Duck-typed stand-in for GigaAMTranscriber (no model, no GPU)."""

    model_id = "gigaam-v3-rnnt"

    def __init__(self, texts):
        self.texts = texts
        self.calls = []

    def vad_fingerprint(self):
        return {"vad_payload": {"use_vad": True}, "vad_options_hash": "deadbeefdeadbeef"}

    def transcribe(self, path):
        from eval.asr_gigaam import TranscriptionResult

        self.calls.append(str(path))
        text = self.texts.get(str(path), "")
        return TranscriptionResult(
            audio_path=str(path), text=text,
            segments=[{"start": 0.0, "end": 30.0, "text": text}] if text else [],
            model_id=self.model_id, onnx_asr_version="0.12.0",
            audio_duration_sec=30.0, sample_rate=16000, elapsed_sec=0.3,
        )


def test_partial_output_is_transcribed_whatever_the_status(tmp_path):
    """loop_cap / degraded / timeout runs with real audio must be scored on that audio.

    Reviewer demonstration: three runs whose WAV exists but whose status is not
    `complete` used to be scored as empty hypotheses -> WER 100 % for all three.
    """
    wav = tmp_path / "partial.wav"
    wav.write_bytes(b"RIFF")  # existence is all evaluate_runs checks
    bench = {"t1": _bench_row()}
    runs = [
        _run_row("r_loop", status="loop_cap", output_path=str(wav)),
        _run_row("r_degr", status="degraded", output_path=str(wav)),
        _run_row("r_time", status="timeout", output_path=str(wav)),
        _run_row("r_nowav", status="early_eos", output_path=None),
    ]
    # the partial audio really contains the first six reference words
    tr = _FakeTranscriber({str(wav): " ".join(REF10.split()[:6])})
    rows = RE.evaluate_runs(runs, bench, tr, floor={}, variant="lenient")

    assert tr.calls == [str(wav)] * 3, "every existing WAV is transcribed"
    for r in rows[:3]:
        assert r["transcribed"] is True
        assert r["status_complete"] is False
        assert r["valid"] is False, "not complete -> not in WER-valid"
        assert r["wer"] == pytest.approx(0.4), "scored on the real partial audio"
        assert r["end_coverage"] == pytest.approx(0.6)
    # the run without a WAV is still attempted, as an empty hypothesis
    assert rows[3]["transcribed"] is False
    assert rows[3]["wer"] == pytest.approx(1.0)
    assert rows[3]["n_hyp_words"] == 0

    agg = aggregate(rows)
    assert agg["n_attempted"] == 4
    assert agg["complete_rate"] == pytest.approx(0.0)
    assert agg["valid_rate"] == pytest.approx(0.0)
    assert agg["macro_wer_all"] == pytest.approx((0.4 * 3 + 1.0) / 4)


def test_complete_run_with_missing_wav_becomes_empty_or_invalid_audio(tmp_path):
    """A provisional `complete` with no audio cannot stay complete: there is nothing to judge.

    The generator's label survives as `gen_status`, so "Complete %" drops to 0
    while WER-valid still reports the run as attempted-but-invalid.
    """
    bench = {"t1": _bench_row()}
    runs = [_run_row("r_gone", status="complete", output_path=str(tmp_path / "nope.wav"))]
    rows = RE.evaluate_runs(runs, bench, _FakeTranscriber({}), floor={}, variant="lenient")
    assert rows[0]["gen_status"] == "complete"
    assert rows[0]["status"] == "empty_or_invalid_audio"
    assert rows[0]["status_complete"] is False
    assert rows[0]["status_changed_by_eval"] is True
    assert rows[0]["output_exists"] is False
    assert rows[0]["valid"] is False
    assert rows[0]["invalid_reason"] == "output_missing"
    assert rows[0]["wer"] == pytest.approx(1.0)
    agg = aggregate(rows)
    assert agg["complete_rate"] == pytest.approx(0.0), "final status decides Complete %"
    assert agg["valid_rate"] == pytest.approx(0.0)


def test_floor_cache_key_covers_variant_model_and_vad():
    base = RE.floor_cache_key("t1", "gigaam-v3-rnnt", "lenient", "aaaa", "rrrr")
    assert base != RE.floor_cache_key("t1", "gigaam-v3-rnnt", "strict", "aaaa", "rrrr")
    assert base != RE.floor_cache_key("t1", "gigaam-v3-ctc", "lenient", "aaaa", "rrrr")
    assert base != RE.floor_cache_key("t1", "gigaam-v3-rnnt", "lenient", "bbbb", "rrrr")
    assert base != RE.floor_cache_key("t1", "gigaam-v3-rnnt", "lenient", "aaaa", "ssss")
    assert base == RE.floor_cache_key("t1", "gigaam-v3-rnnt", "lenient", "aaaa", "rrrr")


def test_floor_cache_key_tracks_the_benchmark_inputs():
    """A re-issued benchmark (edited text_ref or moved offsets) invalidates the floor."""
    item = _bench_row()
    h = RE.bench_ref_hash(item)
    assert h == RE.bench_ref_hash(dict(item, bucket="B4", words=999))  # not inputs of the floor
    assert h != RE.bench_ref_hash(dict(item, text_ref=item["text_ref"] + " ещё"))
    assert h != RE.bench_ref_hash(dict(item, human_offset_start=1.0))
    assert h != RE.bench_ref_hash(dict(item, human_offset_end=29.0))
    assert h != RE.bench_ref_hash(dict(item, human_audio_path="/other/file.flac"))


def test_vad_fingerprint_changes_with_options():
    from eval.asr_gigaam import FROZEN_VAD_OPTIONS

    payload_a = {"use_vad": True, "vad_model": "silero",
                 "vad_options": {k: float(v) for k, v in sorted(FROZEN_VAD_OPTIONS.items())},
                 "vad_batch_size": 8}
    moved = dict(FROZEN_VAD_OPTIONS, threshold=0.6)
    payload_b = {"use_vad": True, "vad_model": "silero",
                 "vad_options": {k: float(v) for k, v in sorted(moved.items())},
                 "vad_batch_size": 8}
    h = lambda p: hashlib.sha256(  # noqa: E731
        json.dumps(p, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    assert h(payload_a) != h(payload_b)


def test_hf_hub_cache_is_overridden_not_defaulted(monkeypatch, tmp_path):
    """PLAN.md §0.1: writes stay in the working directory, whatever the shell says."""
    from eval import asr_gigaam

    monkeypatch.setenv("HF_HUB_CACHE", "external/.cache/huggingface")
    path = asr_gigaam.ensure_hf_cache()
    assert os.environ["HF_HUB_CACHE"] == str(asr_gigaam.DEFAULT_HF_CACHE)
    assert str(path).startswith(str(REPO_ROOT))


# ---------------------------------------------------------------------------
# GPU guard (PLAN.md §0.1: index 1; owner granted 0 and 2 on 2026-08-30 — decisions.md)
# ---------------------------------------------------------------------------


def test_gpu_selection_respects_the_callers_environment(monkeypatch):
    from eval import asr_gigaam
    monkeypatch.setattr(asr_gigaam, '_nvidia_smi_gpus', lambda: {'3': 'GPU-test'})
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    assert asr_gigaam.assert_gpu_allowed('cuda')['cuda_visible_devices'] is None
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    rec = asr_gigaam.assert_gpu_allowed('cuda')
    assert rec['cuda_visible_devices'] == '3'
    assert rec['physical_gpu_uuid'] == 'GPU-test'


def test_gpu_guard_accepts_index_1_and_records_the_physical_gpu(monkeypatch):
    from eval import asr_gigaam

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    rec = asr_gigaam.assert_gpu_allowed("cuda")
    assert rec["guard"] == "cuda"
    assert rec["cuda_visible_devices"] == "1"
    assert rec["override"] is False
    # on a host without nvidia-smi the uuid is None; when it is there it must be
    # the uuid of physical index 1, never of another card
    if rec["nvidia_smi_gpus"]:
        assert rec["physical_gpu_uuid"] == rec["nvidia_smi_gpus"]["1"]
        assert rec["physical_gpu_uuid"] != rec["nvidia_smi_gpus"].get("0")


def test_gpu_guard_accepts_owner_granted_0_and_2(monkeypatch):
    """2026-08-30: the owner granted GPU 0 (E6 eval lane) and GPU 2 (E7) in addition to 1."""
    from eval import asr_gigaam

    monkeypatch.delenv(asr_gigaam.GPU_GUARD_OVERRIDE_ENV, raising=False)
    for ok in ("0", "2"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ok)
        rec = asr_gigaam.assert_gpu_allowed("cuda")
        assert rec["cuda_visible_devices"] == ok and rec["override"] is False
        if rec["nvidia_smi_gpus"]:
            assert rec["physical_gpu_uuid"] == rec["nvidia_smi_gpus"][ok]


def test_gpu_guard_is_inert_on_cpu(monkeypatch):
    from eval import asr_gigaam

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert asr_gigaam.assert_gpu_allowed("cpu")["guard"] == "cpu"


def test_gpu_guard_override_warns_instead_of_raising(monkeypatch):
    from eval import asr_gigaam

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv(asr_gigaam.GPU_GUARD_OVERRIDE_ENV, "1")
    rec = asr_gigaam.assert_gpu_allowed("cuda")
    assert rec["override"] is True


# ---------------------------------------------------------------------------
# secondary ASR: disagreement audit only (PLAN.md §9.4, §14 A4)
# ---------------------------------------------------------------------------


def test_secondary_asr_rejects_whisper_and_gigaam():
    from eval import asr_secondary as S

    for bad in ("whisper-base", "gigaam-v3-rnnt", "gigaam-v2-ctc", "nemo-parakeet-ctc-0.6b"):
        with pytest.raises(S.SecondaryAsrError):
            S.check_secondary_model(bad)
    for ok in S.SECONDARY_MODELS:
        assert S.check_secondary_model(ok) == ok


def test_secondary_disagreement_ranking_and_audit_subset():
    from eval import asr_secondary as S

    per_item = [
        {"run_id": "a", "text_id": "t1", "asr_text": REF10, "wer": 0.0,
         "output_exists": True, "output_path": "/x/a.wav"},
        {"run_id": "b", "text_id": "t2", "asr_text": REF10, "wer": 0.1,
         "output_exists": True, "output_path": "/x/b.wav"},
        {"run_id": "c", "text_id": "t3", "asr_text": REF10, "wer": 0.2,
         "output_exists": False, "output_path": None},
    ]
    secondary = {"a": REF10, "b": " ".join(REF10.split()[:3])}
    rows = S.disagreement_rows(per_item, secondary)
    assert [r["run_id"] for r in rows] == ["b", "a"], "most disagreement first"
    assert rows[0]["disagreement_wer"] == pytest.approx(0.7)
    assert rows[1]["disagreement_wer"] == pytest.approx(0.0)
    # a run with no secondary transcript is not invented into the worklist
    assert "c" not in {r["run_id"] for r in rows}
    sel = S.select_audit_subset(rows, top_frac=0.10)
    assert [r["run_id"] for r in sel] == ["b"]
    assert sel[0]["audit_selected"] is True


# ---------------------------------------------------------------------------
# final PLAN.md §3.4 status assigned by the evaluator
# (Lead decision 2026-08-28; thresholds frozen in configs/eval.yaml)
# ---------------------------------------------------------------------------

from eval import final_status as FS  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return FS.load_eval_config()


def _verdict(cfg, **over):
    """One final-status call on an otherwise perfect `complete` run."""
    kw = dict(gen_status="complete", stop_reason="eos", usable_audio=True,
              end_coverage=1.0, wer_minus_floor=0.0, floor_available=True,
              excess_repetition_rate=0.0, max_ngram_run=1, cfg=cfg)
    kw.update(over)
    return FS.assign_final_status(**kw)


def test_frozen_thresholds_are_the_preregistered_ones(cfg):
    """Pins configs/eval.yaml: an edit of a pre-registered threshold must fail the suite."""
    from eval.metrics import DEFAULT_NGRAM_RANGE

    fs = cfg["final_status"]
    assert cfg["frozen_at"] == "2026-08-28"
    assert fs["end_coverage_complete_min"] == 0.95
    assert fs["wer_minus_floor_degraded_min"] == 0.30
    assert fs["loop"]["excess_repetition_rate_min"] == 0.20
    assert fs["loop"]["ngram_run_min"] == 3
    assert tuple(fs["loop"]["ngram_range"]) == DEFAULT_NGRAM_RANGE
    assert fs["wer_criterion_when_floor_missing"] == "skip"
    assert cfg["wer_valid_denominator"] == "gen_status"


def test_missing_eval_config_is_an_error_not_a_default(tmp_path):
    with pytest.raises(FS.FinalStatusError) as e:
        FS.load_eval_config(tmp_path / "nope.yaml")
    assert "configs/eval.yaml" in str(e.value)


def test_eval_config_rejects_a_status_outside_the_taxonomy(tmp_path):
    import yaml

    cfg = yaml.safe_load(open(FS.EVAL_CONFIG_PATH, "rt", encoding="utf-8"))
    cfg["final_status"]["no_usable_audio_status"] = "silently_dropped"
    p = tmp_path / "eval.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(FS.FinalStatusError) as e:
        FS.load_eval_config(p)
    assert "3.4" in str(e.value)


@pytest.mark.parametrize("gen", ["loop_cap", "timeout", "context_limit", "oom",
                                 "empty_or_invalid_audio", "infrastructure_error",
                                 "hard_input_limit", "degraded"])
def test_generator_failure_statuses_pass_through_unchanged(cfg, gen):
    """PLAN.md §3.4 + Lead: only a provisional `complete` is refined."""
    v = _verdict(cfg, gen_status=gen, stop_reason="watchdog",
                 end_coverage=0.10, wer_minus_floor=5.0,
                 excess_repetition_rate=0.9, max_ngram_run=9)
    assert v["status"] == gen
    assert "passed through" in v["final_status_reason"]
    # the loop flag is still computed and published, it just does not relabel a
    # run the generator already classified (a looping watchdog stop is loop_cap)
    assert v["loop_flag"] is True


def test_complete_branch(cfg):
    v = _verdict(cfg, end_coverage=0.99, wer_minus_floor=0.05)
    assert v["status"] == "complete"
    assert v["loop_flag"] is False


def test_early_eos_branch(cfg):
    """The E1-native case: EOS with a fifth of the source never read."""
    v = _verdict(cfg, end_coverage=0.81, wer_minus_floor=0.14)
    assert v["status"] == "early_eos"
    assert "EndCoverage=0.8100<0.95" in v["final_status_reason"]


def test_degraded_by_wer_above_floor(cfg):
    v = _verdict(cfg, end_coverage=1.0, wer_minus_floor=0.31)
    assert v["status"] == "degraded"
    assert "wer_minus_floor" in v["final_status_reason"]


def test_degraded_by_loop_rate(cfg):
    v = _verdict(cfg, excess_repetition_rate=0.25)
    assert v["status"] == "degraded"
    assert v["loop_flag"] is True
    assert "excess_repetition_rate" in v["loop_reason"]


def test_degraded_by_ngram_run(cfg):
    """A short loop that the dispersed-repetition rate alone would miss."""
    v = _verdict(cfg, excess_repetition_rate=0.05, max_ngram_run=3)
    assert v["status"] == "degraded"
    assert v["loop_flag"] is True
    assert "max_ngram_run=3>=3" in v["loop_reason"]


def test_early_eos_wins_over_a_loop(cfg):
    """Statuses are mutually exclusive; the loop is still recorded in loop_flag."""
    v = _verdict(cfg, end_coverage=0.5, max_ngram_run=9)
    assert v["status"] == "early_eos"
    assert v["loop_flag"] is True


def test_thresholds_are_inclusive_at_the_boundary(cfg):
    assert _verdict(cfg, end_coverage=0.95)["status"] == "complete"
    assert _verdict(cfg, end_coverage=0.9499)["status"] == "early_eos"
    assert _verdict(cfg, wer_minus_floor=0.30)["status"] == "degraded"
    assert _verdict(cfg, wer_minus_floor=0.2999)["status"] == "complete"
    assert _verdict(cfg, max_ngram_run=2)["status"] == "complete"
    assert _verdict(cfg, excess_repetition_rate=0.1999)["status"] == "complete"


def test_external_text_without_floor_skips_the_wer_criterion(cfg):
    """No human recording -> no floor -> the WER branch cannot fire, and says so."""
    v = _verdict(cfg, floor_available=False, wer_minus_floor=None)
    assert v["status"] == "complete"
    assert v["final_status_note"] == "no_floor_wer_criterion_skipped"
    # a loop is still detectable without a floor
    v2 = _verdict(cfg, floor_available=False, wer_minus_floor=None, max_ngram_run=4)
    assert v2["status"] == "degraded"


def test_complete_without_usable_audio_is_empty_or_invalid_audio(cfg):
    v = _verdict(cfg, usable_audio=False)
    assert v["status"] == "empty_or_invalid_audio"


def test_provisional_complete_must_come_from_eos(cfg):
    with pytest.raises(FS.FinalStatusError) as e:
        _verdict(cfg, stop_reason="max_len")
    assert "stop_reason" in str(e.value)


def test_unknown_gen_status_is_an_error(cfg):
    with pytest.raises(FS.FinalStatusError):
        _verdict(cfg, gen_status="looks_fine")


def test_gen_status_of_prefers_the_explicit_field_and_rejects_a_contradiction():
    assert FS.gen_status_of({"status": "complete"}) == "complete"
    assert FS.gen_status_of({"gen_status": "loop_cap"}) == "loop_cap"
    assert FS.gen_status_of({"gen_status": "loop_cap", "status": "loop_cap"}) == "loop_cap"
    with pytest.raises(FS.FinalStatusError):
        FS.gen_status_of({"run_id": "r", "gen_status": "complete", "status": "timeout"})
    with pytest.raises(FS.FinalStatusError):
        FS.gen_status_of({"run_id": "r"})


# --- the run-manifest side of the same rule --------------------------------


def test_runs_reject_complete_without_eos(tmp_path):
    bp = _write_jsonl(tmp_path / "b.jsonl", [_bench_row()])
    rp = _write_jsonl(tmp_path / "r.jsonl",
                      [_run_row(status="complete", stop_reason="watchdog")])
    with pytest.raises(RE.BenchmarkContractError) as e:
        RE.load_runs([rp], RE.load_benchmark(bp))
    assert "stop_reason" in str(e.value)


def test_runs_reject_a_status_outside_the_taxonomy(tmp_path):
    bp = _write_jsonl(tmp_path / "b.jsonl", [_bench_row()])
    rp = _write_jsonl(tmp_path / "r.jsonl", [_run_row(status="ok")])
    with pytest.raises(RE.BenchmarkContractError) as e:
        RE.load_runs([rp], RE.load_benchmark(bp))
    assert "3.4" in str(e.value)


def test_runs_accept_a_manifest_that_spells_it_gen_status(tmp_path):
    bp = _write_jsonl(tmp_path / "b.jsonl", [_bench_row()])
    row = _run_row()
    row["gen_status"] = row.pop("status")
    rp = _write_jsonl(tmp_path / "r.jsonl", [row])
    assert len(RE.load_runs([rp], RE.load_benchmark(bp))) == 1


# --- end to end through evaluate_runs --------------------------------------


def test_evaluate_runs_relabels_early_eos_and_keeps_both_labels(tmp_path, cfg):
    """The integration finding: EOS at 60 % of the source is `early_eos`, not `complete`."""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    bench = {"t1": _bench_row()}
    runs = [_run_row("r1", status="complete", output_path=str(wav))]
    tr = _FakeTranscriber({str(wav): " ".join(REF10.split()[:6])})
    rows = RE.evaluate_runs(runs, bench, tr, floor={}, variant="lenient", eval_cfg=cfg)
    r = rows[0]
    assert r["gen_status"] == "complete" and r["status"] == "early_eos"
    assert r["status_complete"] is False and r["status_changed_by_eval"] is True
    assert r["end_coverage"] == pytest.approx(0.6)
    # WER-valid keeps its generation-side denominator
    assert r["valid"] is True
    agg = aggregate(rows)
    assert agg["complete_rate"] == pytest.approx(0.0)
    assert agg["valid_rate"] == pytest.approx(1.0)


def test_evaluate_runs_keeps_a_good_run_complete(tmp_path, cfg):
    wav = tmp_path / "b.wav"
    wav.write_bytes(b"RIFF")
    bench = {"t1": _bench_row()}
    runs = [_run_row("r1", status="complete", output_path=str(wav))]
    rows = RE.evaluate_runs(runs, bench, _FakeTranscriber({str(wav): REF10}), floor={},
                            variant="lenient", eval_cfg=cfg)
    assert rows[0]["status"] == "complete"
    assert rows[0]["status_changed_by_eval"] is False
    assert rows[0]["final_status_note"] == "no_floor_wer_criterion_skipped"


def test_evaluate_runs_relabels_a_loop_as_degraded(tmp_path, cfg):
    """A looped output that stopped by EOS is `degraded` -- §3.4 has no `loop` status."""
    wav = tmp_path / "c.wav"
    wav.write_bytes(b"RIFF")
    bench = {"t1": _bench_row()}
    runs = [_run_row("r1", status="complete", output_path=str(wav))]
    looped = REF10 + " " + " ".join(["восемь девять десять"] * 3)
    rows = RE.evaluate_runs(runs, bench, _FakeTranscriber({str(wav): looped}), floor={},
                            variant="lenient", eval_cfg=cfg)
    r = rows[0]
    assert r["end_coverage"] == pytest.approx(1.0), "the whole source was read"
    assert r["max_ngram_run"] >= 3
    assert r["loop_flag"] is True
    assert r["status"] == "degraded"
    assert r["gen_status"] == "complete"


def test_status_histogram_is_per_checkpoint_and_bucket():
    per_item = [
        {"checkpoint": "E1", "bucket": "B0", "status": "complete", "gen_status": "complete"},
        {"checkpoint": "E1", "bucket": "B0", "status": "early_eos", "gen_status": "complete"},
        {"checkpoint": "E1", "bucket": "B4", "status": "early_eos", "gen_status": "complete"},
        {"checkpoint": "E0", "bucket": "B0", "status": "complete", "gen_status": "complete"},
    ]
    md = "\n".join(RE.status_histogram_lines(per_item, "status"))
    assert "| E1 | B0 | 2 | 1 | 1 |" in md
    assert "| E1 | B4 | 1 | 0 | 1 |" in md
    assert "| E1 | ALL | 3 | 1 | 2 |" in md
    assert "| E0 | B0 | 1 | 1 | 0 |" in md
    # gen_status view of the same items collapses into one column
    gmd = "\n".join(RE.status_histogram_lines(per_item, "gen_status"))
    assert "| E1 | ALL | 3 | 3 |" in gmd


# --- the loop statistics themselves ----------------------------------------


def test_max_ngram_run_counts_back_to_back_repetition_only():
    from eval.metrics import ngram_run_extremes

    ref = REF10.split()
    assert ngram_run_extremes(ref, ref)["max_ngram_run"] == 1
    # the same phrase twice, far apart, is not a run
    far = ref + ["конец"] + ref[:3]
    assert ngram_run_extremes(ref, far)["max_ngram_run"] == 1
    looped = ref + ref[7:] * 2
    assert ngram_run_extremes(ref, looped)["max_ngram_run"] == 3


def test_source_repetition_is_paid_for_by_the_run_statistic():
    """PLAN.md §7.3: a source that repeats a phrase does not make the output a loop."""
    from eval.metrics import ngram_run_extremes

    phrase = ["и", "так", "далее"]
    ref = ["текст"] + phrase * 3 + ["конец"]
    assert ngram_run_extremes(ref, list(ref))["max_ngram_run"] == 1, "copying the source is not a loop"
    worse = ["текст"] + phrase * 6 + ["конец"]
    assert ngram_run_extremes(ref, worse)["max_ngram_run"] == 4


def test_natural_human_transcripts_are_far_below_the_loop_threshold():
    """The pre-registration evidence, recomputed: 20 human references of this benchmark.

    Reads the **v3.1** floor cache, which exists since the Lead's
    `bash scripts/run_pilot.sh v31_base` run of 2026-08-29: `results/v31_base/asr_floor.jsonl`
    holds 30 rows, the 20 dataset texts with `available: true` plus the 10 external texts
    that have no human audio.  So this test RUNS (A4, 2026-08-29) — the skip below is only
    the guard for a fresh checkout where the pilot has not been run yet.

    The v1 cache (`results/pilot_floor`) was archived to `results/_v1_2026-08-28/pilot_floor`
    on 2026-08-29 when the corpus was replaced: its `text_id`s are from the v1 benchmark and
    no longer exist in `data/benchmark/pilot.jsonl`, so reading it here would compare two
    different builds — hence the hard `pytest.fail` on an unknown `text_id` below.

    The same 20 references, plus 120 natural dev segments, 30 self-scored source texts and
    the 120 v3.1 pilot outputs, are tabulated in `reports/eval_thresholds_v31.md`.
    """
    from eval.metrics import excess_repetition
    from eval.normalize import words as nwords

    floor_path = REPO_ROOT / "results" / "v31_base" / "asr_floor.jsonl"
    bench_path = REPO_ROOT / "data" / "benchmark" / "pilot.jsonl"
    if not floor_path.exists():
        pytest.skip(
            "v3.1 ASR floor cache absent (results/v31_base/asr_floor.jsonl): it is written "
            "by `bash scripts/run_pilot.sh v31_base` on GPU 1. The v1 cache in "
            "results/_v1_2026-08-28/pilot_floor is NOT a substitute — its text_ids are from "
            "the v1 benchmark. COVERAGE GAP while this skip fires: the loop thresholds are "
            "then unverified on human speech.")
    if not bench_path.exists():
        pytest.skip("data/benchmark/pilot.jsonl absent — run `bash scripts/build_benchmark.sh`")
    bench = {}
    with open(bench_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            bench[row["text_id"]] = row
    variant = load_spec()["primary_variant"]
    n = 0
    with open(floor_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("available"):
                continue
            if row["text_id"] not in bench:
                pytest.fail(
                    f"floor cache {floor_path} contains text_id {row['text_id']!r}, which is "
                    f"not in {bench_path}: the cache was built for a different benchmark "
                    "(this is exactly what happened to the v1 cache on the v3.1 rebuild)")
            item = bench[row["text_id"]]
            rw = nwords(item["text_ref"], variant)
            hw = nwords(row["floor_text"], variant)
            stats = excess_repetition(rw, hw)
            assert stats["max_ngram_run"] < 3, row["text_id"]
            assert stats["excess_repetition_rate"] < 0.20, row["text_id"]
            n += 1
    assert n >= 20, "the frozen thresholds were justified on 20 human references"


# ---------------------------------------------------------------------------
# scripts/aggregate_results.py: the merged §17 main table
# ---------------------------------------------------------------------------

import aggregate_results as AG  # noqa: E402


def _item(run_id, checkpoint="E1", bucket="B0", status="complete", gen_status="complete",
          **over):
    row = {
        "run_id": run_id, "checkpoint": checkpoint, "experiment_id": checkpoint,
        "bucket": bucket, "status": status, "gen_status": gen_status,
        "status_complete": status == "complete",
        "gen_status_complete": gen_status == "complete",
        "valid": gen_status == "complete",
        "wer": 0.1, "cer": 0.05, "word_errors": 1, "n_ref_words": 10,
        "char_substitutions": 1, "char_deletions": 0, "char_insertions": 0, "n_ref_chars": 20,
        "end_coverage": 1.0, "end_coverage_robust": 1.0, "source_coverage": 1.0,
        "excess_repetition_rate": 0.0,
        "max_ngram_run": 1, "loop_flag": False, "duration_ratio": 1.0,
        "floor_wer": 0.08, "wer_minus_floor": 0.02,
    }
    row.update(over)
    return row


def test_aggregate_rows_are_experiment_by_bucket(tmp_path):
    a = _write_jsonl(tmp_path / "E1" / "per_item.jsonl",
                     [_item("r1", "E1", "B0"), _item("r2", "E1", "B1")])
    b = _write_jsonl(tmp_path / "E0" / "per_item.jsonl",
                     [_item("r1", "E0", "B0"), _item("r2", "E0", "B1")])
    rows, prov = AG.load_inputs([str(Path(a).parent), str(Path(b).parent)])
    table = AG.build_table(rows)
    assert [(r["experiment"], r["bucket"]) for r in table] == [
        ("E1", "B0"), ("E1", "B1"), ("E1", "ALL"),
        ("E0", "B0"), ("E0", "B1"), ("E0", "ALL"),
    ]
    assert [r["n_attempted"] for r in table] == [1, 1, 2, 1, 1, 2]
    assert len(prov) == 2


def test_aggregate_complete_pct_is_the_final_status(tmp_path):
    """The whole point of the rule: a generator-complete early_eos is not complete."""
    p = _write_jsonl(tmp_path / "E1" / "per_item.jsonl", [
        _item("r1", status="early_eos", gen_status="complete", end_coverage=0.81),
        _item("r2", status="complete", gen_status="complete"),
    ])
    table = AG.build_table(AG.load_inputs([p])[0])
    all_row = [r for r in table if r["bucket"] == "ALL"][0]
    assert all_row["complete_rate"] == pytest.approx(0.5)
    assert all_row["gen_complete_rate"] == pytest.approx(1.0)
    assert all_row["n_early_eos"] == 1 and all_row["n_complete"] == 1


def test_aggregate_rejects_a_duplicated_run(tmp_path):
    p1 = _write_jsonl(tmp_path / "a" / "per_item.jsonl", [_item("r1")])
    p2 = _write_jsonl(tmp_path / "b" / "per_item.jsonl", [_item("r1")])
    with pytest.raises(AG.AggregateError) as e:
        AG.load_inputs([p1, p2])
    assert "double-count" in str(e.value)


def test_aggregate_label_override_separates_two_runs_of_one_experiment(tmp_path):
    p1 = _write_jsonl(tmp_path / "a" / "per_item.jsonl", [_item("r1")])
    p2 = _write_jsonl(tmp_path / "b" / "per_item.jsonl", [_item("r1")])
    rows, _ = AG.load_inputs([f"seed0={p1}", f"seed1={p2}"])
    table = AG.build_table(rows)
    assert {r["experiment"] for r in table} == {"seed0", "seed1"}


def test_aggregate_leaves_unmeasured_columns_empty(tmp_path):
    p = _write_jsonl(tmp_path / "E1" / "per_item.jsonl", [_item("r1")])
    table = AG.build_table(AG.load_inputs([p])[0])
    assert table[0]["rmst_words"] is None and table[0]["voice_drift"] is None
    md = AG.render_markdown(table, [], AG.load_inputs([p])[0], "t")
    assert "n/a" in md and "never fills them with a guess" in md


def test_aggregate_cli_writes_md_and_csv(tmp_path):
    p = _write_jsonl(tmp_path / "E1" / "per_item.jsonl", [_item("r1"), _item("r2", bucket="B1")])
    out = tmp_path / "tables"
    assert AG.main([p, "--output-dir", str(out), "--name", "t"]) == 0
    md = (out / "t.md").read_text(encoding="utf-8")
    assert "Table 3" in md and "Table 5" in md
    with open(out / "t.csv", "rt", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["bucket"] for r in rows] == ["B0", "B1", "ALL"]
    assert rows[-1]["n_attempted"] == "2"


def test_aggregate_main_table_shows_coverage_and_endcov_robust(tmp_path):
    """Lead 2026-08-28: Coverage and EndCov-robust are printed next to EndCoverage."""
    p = _write_jsonl(tmp_path / "E1" / "per_item.jsonl", [
        _item("r1", end_coverage=0.552, end_coverage_robust=0.032, source_coverage=0.030),
        _item("r2", end_coverage=1.0, end_coverage_robust=1.0, source_coverage=0.99),
    ])
    rows, _ = AG.load_inputs([p])
    table = AG.build_table(rows)
    all_row = [r for r in table if r["bucket"] == "ALL"][0]
    assert all_row["macro_end_coverage_all"] == pytest.approx(0.776)
    assert all_row["macro_end_coverage_robust_all"] == pytest.approx(0.516)
    assert all_row["macro_source_coverage_all"] == pytest.approx(0.51)
    md = AG.render_markdown(table, [], rows, "t")
    header = [line for line in md.splitlines() if line.startswith("| Checkpoint | Length |")][0]
    cols = [c.strip() for c in header.strip("|").split("|")]
    assert cols.index("Coverage") == cols.index("EndCoverage") - 1
    assert cols.index("EndCov-robust") == cols.index("EndCoverage") + 1
    body = [line for line in md.splitlines() if line.startswith("| E1 | ALL |")][0]
    cells = [c.strip() for c in body.strip("|").split("|")]
    assert len(cells) == len(cols), "the row must have exactly as many cells as the header"
    assert cells[cols.index("Coverage")] == "0.510"
    assert cells[cols.index("EndCoverage")] == "0.776"
    assert cells[cols.index("EndCov-robust")] == "0.516"
    assert "macro_end_coverage_robust_all" in AG.CSV_COLUMNS


def test_aggregate_csv_carries_the_new_columns(tmp_path):
    p = _write_jsonl(tmp_path / "E1" / "per_item.jsonl",
                     [_item("r1", end_coverage_robust=0.25, source_coverage=0.4)])
    out = tmp_path / "tables"
    assert AG.main([p, "--output-dir", str(out), "--name", "t"]) == 0
    with open(out / "t.csv", "rt", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert float(rows[-1]["macro_end_coverage_robust_all"]) == pytest.approx(0.25)
    assert float(rows[-1]["macro_source_coverage_all"]) == pytest.approx(0.4)


def test_aggregate_flags_thresholds_that_do_not_match(tmp_path):
    prov = [{"path": "a", "labels": ["E1"], "n_rows": 1, "mtime": "-",
             "final_status_thresholds": {"end_coverage_complete_min": 0.95}},
            {"path": "b", "labels": ["E0"], "n_rows": 1, "mtime": "-",
             "final_status_thresholds": {"end_coverage_complete_min": 0.90}}]
    md = AG.render_markdown([], prov, [], "t")
    assert "different frozen" in md
