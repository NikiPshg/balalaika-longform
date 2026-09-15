#!/usr/bin/env python
"""Evaluate EN-probe runs (E9, EXPLORATORY) — English mirror of scripts/run_evaluation.py.

Why a separate runner (checked before writing it): scripts/run_evaluation.py has
no ASR-module or normalization override — `--model` is restricted to the frozen
GIGAAM_V3_VARIANTS and `--variant` to the frozen Russian strict/lenient spec, and
that file is FROZEN.  This script therefore REUSES, without modifying anything:

  * the metric functions of src/eval/metrics.py (`content_metrics` with
    `already_normalized=True`, `duration_metrics`, `aggregate` via build_summary),
  * the FROZEN final-status rule (src/eval/final_status.py + configs/eval.yaml
    thresholds: EndCoverage >= 0.95, WER-floor >= 0.30, loop 0.20 / run 3 —
    NOT re-tuned for English; exploratory caveat in the report),
  * the loading / validation / floor-cache / summary / markdown machinery of
    scripts/run_evaluation.py, imported as a module,

and swaps exactly two things: the transcriber (EnglishTranscriber, Parakeet via
the same onnx-asr + silero VAD) and the normalization (normalize_english,
variant "english_v1": lowercase, apostrophes deleted, diacritics folded,
punctuation stripped, whitespace collapsed — src/eval/normalize_english.py).

The benchmark's text_ref is already normalized with english_v1; normalize_english
is idempotent, so normalizing it again here is a no-op by construction.

Usage:
    export CUDA_VISIBLE_DEVICES=2
    .venv-eval/bin/python scripts/run_evaluation_en.py \
        --benchmark data/benchmark/english_probe.jsonl --floor-only \
        --output-dir results/v31_en --floor-cache results/v31_en/asr_floor.jsonl
    .venv-eval/bin/python scripts/run_evaluation_en.py \
        --run-manifest outputs/v31_en/E1/runs.jsonl \
        --benchmark data/benchmark/english_probe.jsonl \
        --output-dir results/v31_en/E1 --experiment-name E1 \
        --floor-cache results/v31_en/asr_floor.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_evaluation as RU  # noqa: E402  — frozen Russian runner, reused read-only

from eval.asr_english import (  # noqa: E402
    DEFAULT_MODEL as EN_DEFAULT_MODEL,
    ENGLISH_VARIANTS,
    EnglishTranscriber,
    GpuGuardError,
)
from eval.final_status import (  # noqa: E402
    EVAL_CONFIG_PATH, FinalStatusError, assign_final_status, load_eval_config,
)
from eval.metrics import content_metrics, duration_metrics  # noqa: E402
from eval.normalize_english import (  # noqa: E402
    NORMALIZATION_VERSION_EN, VARIANT_EN, normalize_english,
)

COMPLETE_STATUS = RU.COMPLETE_STATUS


def en_content_metrics(ref_text: str, hyp_text: str | None) -> dict:
    """src/eval/metrics.content_metrics with English normalization.

    Both sides are normalized HERE with english_v1 and passed through with
    ``already_normalized=True``, so the frozen Russian normalize() inside
    metrics.py is never invoked.  The returned ``normalization_variant`` is the
    English label.
    """
    return content_metrics(
        normalize_english(ref_text),
        normalize_english(hyp_text or ""),
        variant=VARIANT_EN,
        already_normalized=True,
    )


def compute_floor_en(bench, transcriber, cache_path: Path, refresh: bool = False):
    """RU.compute_floor with en_content_metrics (no Russian strict_* variant)."""
    fp = transcriber.vad_fingerprint()
    vad_hash = fp["vad_options_hash"]
    cache: dict[str, dict] = {}
    if cache_path.exists() and not refresh:
        for row in RU.read_jsonl(cache_path):
            cache[row["cache_key"]] = row
    errors: list[str] = []
    changed = False
    for item in bench:
        text_id = item["text_id"]
        key = RU.floor_cache_key(text_id, transcriber.model_id, VARIANT_EN, vad_hash,
                                 RU.bench_ref_hash(item))
        if key in cache:
            continue
        audio = item["human_audio_path"]
        if audio is None:
            cache[key] = {
                "cache_key": key, "text_id": text_id, "model_id": transcriber.model_id,
                "normalization_variant": VARIANT_EN, "vad_options_hash": vad_hash,
                "bench_ref_hash": RU.bench_ref_hash(item), "language": "en",
                "available": False, "reason": "no human reference audio",
                "floor_text": None, "floor_wer": None, "floor_cer": None,
                "human_duration_sec": item["human_duration_sec"],
            }
            changed = True
            continue
        if not Path(str(audio)).exists():
            errors.append(f"{text_id}: human_audio_path does not exist: {audio}")
            continue
        start = RU._as_float(item["human_offset_start"])
        end = RU._as_float(item["human_offset_end"])
        try:
            import soundfile as sf
            info = sf.info(str(audio))
            file_dur = float(info.frames) / float(info.samplerate)
        except Exception as exc:
            errors.append(f"{text_id}: cannot read {audio}: {type(exc).__name__}: {exc}")
            continue
        eff_start = max(0.0, min(start, file_dur))
        eff_end = max(eff_start, min(end, file_dur))
        if eff_end - eff_start < RU.MIN_HUMAN_SLICE_SEC:
            errors.append(f"{text_id}: slice [{start}, {end}] of {audio} is "
                          f"{eff_end - eff_start:.3f} s (< {RU.MIN_HUMAN_SLICE_SEC:g} s)")
            continue
        res = transcriber.transcribe_slice(str(audio), start, end)
        if res.audio_duration_sec < RU.MIN_HUMAN_SLICE_SEC:
            errors.append(f"{text_id}: slice read from {audio} is "
                          f"{res.audio_duration_sec:.3f} s")
            continue
        cm = en_content_metrics(item["text_ref"], res.text)
        cache[key] = {
            "cache_key": key, "text_id": text_id, "model_id": transcriber.model_id,
            "normalization_variant": VARIANT_EN, "vad_options_hash": vad_hash,
            "bench_ref_hash": RU.bench_ref_hash(item), "language": "en",
            "vad_payload": fp["vad_payload"],
            "onnx_asr_version": res.onnx_asr_version, "available": True,
            "audio": str(audio), "human_offset_start": start, "human_offset_end": end,
            "human_duration_sec": item["human_duration_sec"],
            "measured_duration_sec": res.audio_duration_sec,
            "floor_text": res.text, "n_segments": len(res.segments),
            "floor_wer": cm["wer"], "floor_cer": cm["cer"],
            "floor_source_coverage": cm["source_coverage"],
            "floor_end_coverage": cm["end_coverage"],
            "asr_error": res.error,
        }
        changed = True
        RU.write_jsonl(cache_path, cache.values())
        print(f"[floor-en] {text_id}: WER={cm['wer']:.4f} CER={cm['cer']:.4f} "
              f"({res.audio_duration_sec:.1f}s, x{(res.speed_x_realtime or 0):.1f})",
              file=sys.stderr)
    if errors:
        raise RU.BenchmarkContractError(
            "human reference audio is unusable:\n  - " + "\n  - ".join(errors))
    if changed:
        RU.write_jsonl(cache_path, cache.values())
    return cache


def evaluate_runs_en(runs, bench_by_id, transcriber, floor, eval_cfg):
    """RU.evaluate_runs with English normalization (no Russian strict_* block)."""
    vad_hash = transcriber.vad_fingerprint()["vad_options_hash"]
    out: list[dict] = []
    for run in runs:
        text_id = run["text_id"]
        item = bench_by_id[text_id]
        ref_text = item["text_ref"]
        gen_status = RU.gen_status_of(run)
        output_path = run["output_path"]
        exists = bool(output_path) and Path(str(output_path)).exists()
        gen_complete = gen_status == COMPLETE_STATUS

        row: dict[str, Any] = {
            "run_id": run["run_id"],
            "checkpoint": run["experiment_id"],
            "experiment_id": run["experiment_id"],
            "text_id": text_id,
            "voice_id": run.get("voice_id"),
            "seed": run.get("seed"),
            "bucket": item["bucket"],
            "language": "en",
            "gen_status": gen_status,
            "gen_status_complete": gen_complete,
            "stop_reason": run.get("stop_reason"),
            "output_path": output_path,
            "output_exists": exists,
            "n_ref_words_raw": len(str(ref_text).split()),
        }

        asr_ok = False
        if exists:
            res = transcriber.transcribe(str(output_path))
            asr_ok = (res.error is None) and bool(res.text.strip())
            row.update({
                "transcribed": True, "asr_text": res.text, "asr_segments": res.segments,
                "asr_model_id": res.model_id, "onnx_asr_version": res.onnx_asr_version,
                "asr_provider": res.provider, "asr_elapsed_sec": res.elapsed_sec,
                "asr_x_realtime": res.speed_x_realtime, "asr_error": res.error,
                "asr_ok": asr_ok,
            })
            hyp_text = res.text
            raw_duration = res.audio_duration_sec
            segments = res.segments
        else:
            row.update({"transcribed": False, "asr_text": "", "asr_segments": [],
                        "asr_error": None, "asr_ok": False})
            hyp_text = ""
            raw_duration = float(run.get("raw_duration_sec") or 0.0)
            segments = []

        usable_audio = bool(exists and asr_ok)
        row["valid"] = bool(gen_complete and usable_audio)
        row["invalid_reason"] = (None if row["valid"] else (
            "gen_status_not_complete" if not gen_complete
            else "output_missing" if not exists
            else (row.get("asr_error") or "empty_transcription")))

        cm = en_content_metrics(ref_text, hyp_text)
        row.update(cm)

        fl = floor.get(RU.floor_cache_key(text_id, transcriber.model_id, VARIANT_EN,
                                          vad_hash, RU.bench_ref_hash(item)), {})
        human_dur = fl.get("human_duration_sec")
        if human_dur is None:
            human_dur = item["human_duration_sec"]
        row.update(duration_metrics(
            raw_duration, segments,
            reference_duration_sec=float(human_dur) if human_dur else None,
            hyp_word_count=cm["n_hyp_words"]))
        row["floor_wer"] = fl.get("floor_wer")
        row["floor_cer"] = fl.get("floor_cer")
        row["floor_available"] = fl.get("available", False)
        row["wer_minus_floor"] = (
            row["wer"] - fl["floor_wer"]
            if (fl.get("floor_wer") is not None and row["wer"] is not None
                and row["wer"] != float("inf")) else None)

        verdict = assign_final_status(
            gen_status=gen_status, stop_reason=row["stop_reason"],
            usable_audio=usable_audio, end_coverage=row.get("end_coverage"),
            wer_minus_floor=row.get("wer_minus_floor"),
            floor_available=bool(row.get("floor_available")),
            excess_repetition_rate=row.get("excess_repetition_rate"),
            max_ngram_run=row.get("max_ngram_run"), cfg=eval_cfg)
        row.update(verdict)
        row["status"] = verdict["status"]
        row["status_complete"] = verdict["status"] == COMPLETE_STATUS
        row["status_changed_by_eval"] = verdict["status"] != gen_status
        out.append(row)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate EN-probe TTS runs (Parakeet + VAD)")
    ap.add_argument("--run-manifest", nargs="+", default=None)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--floor-only", action="store_true")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--experiment-name", default=None)
    ap.add_argument("--model", default=EN_DEFAULT_MODEL, choices=list(ENGLISH_VARIANTS))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--floor-cache", default=None)
    ap.add_argument("--refresh-floor", action="store_true")
    ap.add_argument("--skip-floor", action="store_true")
    ap.add_argument("--eval-config", default=str(EVAL_CONFIG_PATH))
    args = ap.parse_args(argv)

    eval_cfg = load_eval_config(args.eval_config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.floor_only and not args.run_manifest:
        ap.error("--run-manifest is required unless --floor-only is given")
    if args.floor_only and args.skip_floor:
        ap.error("--floor-only and --skip-floor contradict each other")

    bench_by_id = RU.load_benchmark(args.benchmark)
    bench = list(bench_by_id.values())
    runs = [] if args.floor_only else RU.load_runs(args.run_manifest, bench_by_id)
    print(f"[eval-en] {len(runs)} run rows, {len(bench)} benchmark texts", file=sys.stderr)

    bench_variants = {b.get("normalization_variant") for b in bench
                      if b.get("normalization_variant")}
    if bench_variants and bench_variants != {VARIANT_EN}:
        print(f"[eval-en] WARNING: benchmark text_ref normalized with "
              f"{sorted(bench_variants)}, scoring with {VARIANT_EN!r}", file=sys.stderr)

    t0 = time.perf_counter()
    tr = EnglishTranscriber(model_id=args.model, use_vad=True, device=args.device)
    print(json.dumps(tr.describe(), ensure_ascii=False), file=sys.stderr)

    floor_path = Path(args.floor_cache) if args.floor_cache else out_dir / "asr_floor.jsonl"
    floor = {} if args.skip_floor else compute_floor_en(bench, tr, floor_path,
                                                        args.refresh_floor)
    if args.floor_only:
        avail = [f for f in floor.values() if f.get("available")]
        wers = [f["floor_wer"] for f in avail]
        print(f"[eval-en] floor-only: {len(avail)}/{len(bench)} texts have human audio; "
              f"mean floor WER = {(sum(wers) / len(wers) * 100 if wers else float('nan')):.2f} % "
              f"-> {floor_path}", file=sys.stderr)
        return 0

    per_item = evaluate_runs_en(runs, bench_by_id, tr, floor, eval_cfg)
    RU.write_jsonl(out_dir / "per_item.jsonl", per_item)

    summary = RU.build_summary(per_item)
    fp = tr.vad_fingerprint()
    meta = {
        "experiment_name": args.experiment_name or out_dir.name,
        "exploratory": True,
        "language": "en",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "asr_model_id": tr.model_id,
        "onnx_asr_version": tr.onnx_asr_version,
        "vad_model": tr.vad_model_name,
        "vad_options": tr.vad_options,
        "vad_options_hash": fp["vad_options_hash"],
        "provider": tr.provider,
        "device": args.device,
        "normalization_variant": VARIANT_EN,
        "normalization_version": NORMALIZATION_VERSION_EN,
        "benchmark_normalization_variants": sorted(v for v in bench_variants if v),
        "run_manifests": list(args.run_manifest),
        "benchmark": args.benchmark,
        "floor_cache": str(floor_path),
        "n_attempted": len(per_item),
        "n_transcribed": sum(1 for r in per_item if r.get("transcribed")),
        "n_output_missing": sum(1 for r in per_item if not r.get("output_exists")),
        "n_status_complete": sum(1 for r in per_item if r.get("status_complete")),
        "n_gen_status_complete": sum(1 for r in per_item if r.get("gen_status_complete")),
        "n_status_changed_by_eval": sum(1 for r in per_item if r.get("status_changed_by_eval")),
        "n_loop_flag": sum(1 for r in per_item if r.get("loop_flag")),
        "n_valid": sum(1 for r in per_item if r.get("valid")),
        "eval_config": eval_cfg.get("__path__"),
        "eval_config_version": eval_cfg.get("version"),
        "eval_config_frozen_at": eval_cfg.get("frozen_at"),
        "final_status_thresholds": {
            k: v for k, v in eval_cfg["final_status"].items()
            if k in ("end_coverage_complete_min", "wer_minus_floor_degraded_min",
                     "wer_criterion_when_floor_missing", "no_usable_audio_status", "loop")},
        "wer_valid_denominator": eval_cfg.get("wer_valid_denominator"),
        "elapsed_sec": time.perf_counter() - t0,
    }
    with open(out_dir / "summary.json", "wt", encoding="utf-8") as f:
        json.dump({"meta": meta, **summary}, f, ensure_ascii=False, indent=1)
    md = RU.render_markdown(summary, meta, per_item)
    md = md.replace("# Evaluation summary:",
                    "# Evaluation summary (E9 EN-probe, EXPLORATORY):", 1)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(f"[eval-en] wrote {out_dir/'per_item.jsonl'}, {out_dir/'summary.md'} "
          f"in {meta['elapsed_sec']:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RU.BenchmarkContractError as exc:
        print(f"[eval-en] CONTRACT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except FinalStatusError as exc:
        print(f"[eval-en] FINAL-STATUS ERROR: {exc}", file=sys.stderr)
        raise SystemExit(4)
    except GpuGuardError as exc:
        print(f"[eval-en] GPU GUARD: {exc}", file=sys.stderr)
        raise SystemExit(3)
