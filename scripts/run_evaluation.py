#!/usr/bin/env python
"""Evaluate TTS run manifests against the benchmark (PLAN.md §9.1, §9.2, §17).

Input
    --run-manifest  one or more jsonl produced by scripts/run_generation.py
                    (schema PLAN.md §10)
    --benchmark     benchmark jsonl in the canonical schema of PLAN.md §7.5

Output (under --output-dir)
    per_item.jsonl   one row per attempted generation, failures included
    summary.json     aggregated numbers
    summary.md       PLAN.md §17 main table
    asr_floor.jsonl  ASR floor cache: one transcription per benchmark text

Contract (Lead, 2026-08-28)
    * **Canonical fields only.**  This script reads exactly ``text_id``,
      ``text_ref``, ``bucket``, ``human_audio_path``, ``human_offset_start``,
      ``human_offset_end``, ``human_duration_sec`` from the benchmark and
      ``run_id``, ``experiment_id``, ``text_id``, ``voice_id``, ``seed``,
      ``status``, ``stop_reason``, ``output_path``, ``raw_duration_sec`` from the
      run manifest (PLAN.md §7.5, §10).  There are no synonym lists: a missing
      canonical field is a startup error, never a silent fallback.
    * **Every WAV that exists is transcribed, whatever the status.**  A
      ``loop_cap`` / ``degraded`` / ``timeout`` run has real partial audio and
      real content errors; scoring it as an empty hypothesis would hide them
      (PLAN.md §3.3 "partial output saved", §0 rule 4).
    * **WER-all covers every attempted item.**  A run with no WAV enters as an
      empty hypothesis, i.e. 100 % deletions (PLAN.md §9.1) -- never a dropped row.
    * **The final PLAN.md §3.4 status is assigned here, not by the generator**
      (Lead, 2026-08-28).  The run manifest's label is provisional (``gen_status``,
      derived from the stop reason alone); this script refines a provisional
      ``complete`` into ``complete`` / ``early_eos`` / ``degraded`` /
      ``empty_or_invalid_audio`` from EndCoverage, WER-floor and the loop rule,
      with thresholds frozen in ``configs/eval.yaml``.  Every other §3.4 status
      passes through untouched.  ``per_item.jsonl`` keeps both labels.
    * **WER-valid covers only runs whose *generation* succeeded** (``gen_status ==
      complete``) and that produced usable audio.  Its denominator stays on the
      generation side on purpose: conditioning it on the final status would select
      on the very outcome being measured, because a final ``complete`` already
      requires a low WER-floor.  "Complete %" (final status) and "Valid for
      WER-valid %" (gen status) are printed side by side.
    * ASR is GigaAM-v3 through onnx-asr with VAD always on (PLAN.md §0.1).
    * The ASR floor (same ASR on the human reference) is cached under a key that
      includes the normalization variant, the ASR model id, a hash of the VAD
      configuration and a hash of the benchmark inputs (``text_ref`` + audio path
      + offsets), so changing any of them cannot reuse a stale floor.

Hard failures (the script exits non-zero instead of producing a plausible table)
    * a benchmark record missing a canonical field, or with a duplicate text_id;
    * a benchmark record with ``human_audio_path`` but null/invalid offsets;
    * a human reference slice that is empty or shorter than 1 s;
    * a run whose ``text_id`` is absent from the benchmark;
    * a run whose ``gen_status`` is not a PLAN.md §3.4 status, or that claims a
      provisional ``complete`` without ``stop_reason == eos``;
    * a missing or malformed ``configs/eval.yaml`` (the frozen thresholds are
      never defaulted in code).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.asr_gigaam import (  # noqa: E402
    DEFAULT_MODEL,
    GIGAAM_V3_VARIANTS,
    GigaAMTranscriber,
    GpuGuardError,
)
from eval.final_status import (  # noqa: E402
    EVAL_CONFIG_PATH,
    KNOWN_STATUSES,
    FinalStatusError,
    assign_final_status,
    gen_status_of,
    load_eval_config,
)
from eval.metrics import aggregate, content_metrics, duration_metrics  # noqa: E402
from eval.normalize import load_spec  # noqa: E402

# --- canonical schema, PLAN.md §7.5 / §10 -----------------------------------
# benchmark: keys that must be PRESENT on every record (value may be null only
# where the schema allows it)
BENCH_REQUIRED = (
    "text_id",
    "bucket",
    "text_ref",
    "human_audio_path",
    "human_offset_start",
    "human_offset_end",
    "human_duration_sec",
)
# run manifest: keys that must be PRESENT on every row.  The provisional status
# is checked separately, because a manifest may spell it `status` (current
# adapters, PLAN.md §10) or `gen_status` (Lead, 2026-08-28) -- see gen_status_of.
RUN_REQUIRED = ("run_id", "experiment_id", "text_id", "output_path")

COMPLETE_STATUS = "complete"
# PLAN.md §3.4 statuses (imported from eval.final_status, which owns the rule);
# the tuple order is also the column order of the failure table.
MIN_HUMAN_SLICE_SEC = 1.0


class BenchmarkContractError(RuntimeError):
    """A benchmark or run manifest violates the canonical schema of PLAN.md §7.5."""


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i}: bad json: {exc}") from exc
            obj["__source__"] = f"{path}:{i}"
            rows.append(obj)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "__source__"},
                               ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# canonical-schema validation (PLAN.md §7.5)
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_benchmark(path: str | Path) -> dict[str, dict]:
    """Read the benchmark jsonl and enforce the canonical schema of PLAN.md §7.5."""
    rows = read_jsonl(path)
    if not rows:
        raise BenchmarkContractError(f"{path}: benchmark is empty")

    errors: list[str] = []
    by_id: "OrderedDict[str, dict]" = OrderedDict()
    for row in rows:
        where = row.get("__source__", str(path))
        missing = [k for k in BENCH_REQUIRED if k not in row]
        if missing:
            errors.append(f"{where}: missing canonical field(s) {missing} (PLAN.md §7.5)")
            continue

        text_id = row["text_id"]
        if not isinstance(text_id, str) or not text_id.strip():
            errors.append(f"{where}: text_id must be a non-empty string, got {text_id!r}")
            continue
        if text_id in by_id:
            errors.append(f"{where}: duplicate text_id {text_id!r} "
                          f"(first seen at {by_id[text_id].get('__source__')})")
            continue

        text_ref = row["text_ref"]
        if not isinstance(text_ref, str) or not text_ref.strip():
            errors.append(f"{where}: text_ref must be a non-empty string for {text_id!r}")

        audio = row["human_audio_path"]
        start = _as_float(row["human_offset_start"])
        end = _as_float(row["human_offset_end"])
        if audio is not None:
            if not isinstance(audio, str) or not audio.strip():
                errors.append(f"{where}: human_audio_path must be a path or null for {text_id!r}")
            elif start is None or end is None:
                errors.append(
                    f"{where}: {text_id!r} has human_audio_path but "
                    f"human_offset_start={row['human_offset_start']!r} / "
                    f"human_offset_end={row['human_offset_end']!r}; both offsets are "
                    "mandatory when there is human audio (PLAN.md §7.5)"
                )
            elif end - start < MIN_HUMAN_SLICE_SEC:
                errors.append(
                    f"{where}: {text_id!r} human reference slice is "
                    f"{end - start:.3f} s ([{start}, {end}]), shorter than the "
                    f"{MIN_HUMAN_SLICE_SEC:g} s minimum"
                )
        by_id[text_id] = row

    if errors:
        raise BenchmarkContractError(
            f"benchmark {path} violates the canonical schema (PLAN.md §7.5):\n  - "
            + "\n  - ".join(errors)
        )
    return by_id


def load_runs(paths: Sequence[str], bench_by_id: dict[str, dict]) -> list[dict]:
    """Read run manifests and enforce PLAN.md §10 + the §7.5 join rule."""
    rows: list[dict] = []
    for p in paths:
        rows.extend(read_jsonl(p))
    if not rows:
        raise BenchmarkContractError(f"no run rows in {list(paths)}")

    errors: list[str] = []
    for row in rows:
        where = row.get("__source__", "?")
        missing = [k for k in RUN_REQUIRED if k not in row]
        if missing:
            errors.append(f"{where}: missing run-manifest field(s) {missing} (PLAN.md §10)")
            continue
        text_id = row["text_id"]
        if not isinstance(text_id, str) or not text_id.strip():
            errors.append(f"{where}: text_id must be a non-empty string, got {text_id!r}")
            continue
        if text_id not in bench_by_id:
            errors.append(
                f"{where}: run {row.get('run_id')!r} has text_id {text_id!r} which is "
                "absent from the benchmark (PLAN.md §7.5: an unknown text_id is an error)"
            )
        # The provisional generation status must be a §3.4 status, and a
        # provisional `complete` must really come from EOS -- the evaluator
        # refines exactly that case and may not guess what else it could mean.
        try:
            gen = gen_status_of(row)
        except FinalStatusError as exc:
            errors.append(f"{where}: {exc}")
            continue
        if gen not in KNOWN_STATUSES:
            errors.append(
                f"{where}: run {row.get('run_id')!r} has gen_status {gen!r}, which is not "
                f"a PLAN.md §3.4 status {list(KNOWN_STATUSES)}"
            )
        elif gen == COMPLETE_STATUS and row.get("stop_reason") != "eos":
            errors.append(
                f"{where}: run {row.get('run_id')!r} is provisionally {gen!r} but has "
                f"stop_reason={row.get('stop_reason')!r}; a provisional complete comes "
                "from stop_reason='eos' only (PLAN.md §10, §3.4)"
            )
    if errors:
        raise BenchmarkContractError(
            "run manifest(s) violate the contract:\n  - " + "\n  - ".join(errors)
        )
    return rows


# ---------------------------------------------------------------------------
# ASR floor
# ---------------------------------------------------------------------------


def bench_ref_hash(item: dict) -> str:
    """Identity of the *inputs* of one floor measurement inside the benchmark.

    The floor is ``WER(text_ref, ASR(human_audio[start:end]))``, so it goes stale
    when A2 re-issues the benchmark with an edited ``text_ref`` or moved offsets,
    even though ``text_id`` did not change.  Hashing them into the cache key makes
    that impossible to miss (same failure class as the variant bug the reviewer
    demonstrated, one level up).
    """
    payload = json.dumps(
        [item.get("text_ref"), item.get("human_audio_path"),
         item.get("human_offset_start"), item.get("human_offset_end")],
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def floor_cache_key(text_id: str, model_id: str, variant: str, vad_hash: str,
                    ref_hash: str = "") -> str:
    """Cache identity of one floor measurement.

    Changing the normalization variant, the ASR model or any VAD option changes
    the floor, so all three are part of the key (Lead, 2026-08-28); ``ref_hash``
    (see :func:`bench_ref_hash`) additionally pins the benchmark inputs.
    """
    return f"{text_id}|{model_id}|{variant}|{vad_hash}|{ref_hash}"


def compute_floor(
    bench: Sequence[dict],
    transcriber: GigaAMTranscriber,
    cache_path: Path,
    variant: str,
    refresh: bool = False,
) -> dict[str, dict]:
    """Transcribe the human reference of every benchmark text (cached).

    Hard-fails when a human slice turns out to be empty or shorter than
    ``MIN_HUMAN_SLICE_SEC`` once the audio is actually read.
    """
    fp = transcriber.vad_fingerprint()
    vad_hash = fp["vad_options_hash"]

    cache: dict[str, dict] = {}
    if cache_path.exists() and not refresh:
        for row in read_jsonl(cache_path):
            cache[row["cache_key"]] = row

    errors: list[str] = []
    changed = False
    for item in bench:
        text_id = item["text_id"]
        key = floor_cache_key(text_id, transcriber.model_id, variant, vad_hash,
                              bench_ref_hash(item))
        if key in cache:
            continue
        audio = item["human_audio_path"]
        ref_text = item["text_ref"]
        if audio is None:
            cache[key] = {
                "cache_key": key, "text_id": text_id, "model_id": transcriber.model_id,
                "normalization_variant": variant, "vad_options_hash": vad_hash,
                "bench_ref_hash": bench_ref_hash(item),
                "available": False, "reason": "external text: no human reference audio",
                "floor_text": None, "floor_wer": None, "floor_cer": None,
                "human_duration_sec": item["human_duration_sec"],
            }
            changed = True
            continue
        if not Path(str(audio)).exists():
            errors.append(f"{text_id}: human_audio_path does not exist: {audio}")
            continue

        start = _as_float(item["human_offset_start"])
        end = _as_float(item["human_offset_end"])

        # Validate the slice against the real file BEFORE the ASR sees it: an
        # empty array makes onnxruntime throw a page of cuDNN noise instead of a
        # readable message.
        try:
            import soundfile as sf

            info = sf.info(str(audio))
            file_dur = float(info.frames) / float(info.samplerate)
        except Exception as exc:  # unreadable file is itself a hard failure
            errors.append(f"{text_id}: cannot read {audio}: {type(exc).__name__}: {exc}")
            continue
        eff_start = max(0.0, min(start, file_dur))
        eff_end = max(eff_start, min(end, file_dur))
        if eff_end - eff_start < MIN_HUMAN_SLICE_SEC:
            errors.append(
                f"{text_id}: human reference slice [{start}, {end}] of {audio} "
                f"(file is {file_dur:.2f} s) yields {eff_end - eff_start:.3f} s "
                f"(< {MIN_HUMAN_SLICE_SEC:g} s); the offsets are outside the recording"
            )
            continue

        res = transcriber.transcribe_slice(str(audio), start, end)
        if res.audio_duration_sec < MIN_HUMAN_SLICE_SEC:
            errors.append(
                f"{text_id}: human reference slice read from {audio} "
                f"[{start}, {end}] is {res.audio_duration_sec:.3f} s "
                f"(< {MIN_HUMAN_SLICE_SEC:g} s); offsets are outside the recording "
                "or the file is truncated"
            )
            continue

        cm = content_metrics(ref_text, res.text, variant=variant)
        cm_strict = content_metrics(ref_text, res.text, variant="strict")
        cache[key] = {
            "cache_key": key, "text_id": text_id, "model_id": transcriber.model_id,
            "normalization_variant": variant, "vad_options_hash": vad_hash,
            "bench_ref_hash": bench_ref_hash(item),
            "vad_payload": fp["vad_payload"],
            "onnx_asr_version": res.onnx_asr_version, "available": True,
            "audio": str(audio), "human_offset_start": start, "human_offset_end": end,
            "human_duration_sec": item["human_duration_sec"],
            "measured_duration_sec": res.audio_duration_sec,
            "floor_text": res.text, "n_segments": len(res.segments),
            "floor_wer": cm["wer"], "floor_cer": cm["cer"],
            "floor_wer_strict": cm_strict["wer"], "floor_cer_strict": cm_strict["cer"],
            "floor_source_coverage": cm["source_coverage"],
            "floor_end_coverage": cm["end_coverage"],
            "asr_error": res.error,
        }
        changed = True
        # flush after every text: a long floor pass over multi-minute human
        # references must not lose its work if it is interrupted.
        write_jsonl(cache_path, cache.values())
        print(f"[floor] {text_id}: WER={cm['wer']:.4f} CER={cm['cer']:.4f} "
              f"({res.audio_duration_sec:.1f}s, x{(res.speed_x_realtime or 0):.1f})",
              file=sys.stderr)

    if errors:
        raise BenchmarkContractError(
            "human reference audio is unusable:\n  - " + "\n  - ".join(errors)
        )
    if changed:
        write_jsonl(cache_path, cache.values())
    return cache


# ---------------------------------------------------------------------------
# per-item evaluation
# ---------------------------------------------------------------------------


def evaluate_runs(
    runs: Sequence[dict],
    bench_by_id: dict[str, dict],
    transcriber: GigaAMTranscriber,
    floor: dict[str, dict],
    variant: str,
    also_strict: bool = True,
    eval_cfg: dict | None = None,
) -> list[dict]:
    """Score every run and assign its final PLAN.md §3.4 status.

    ``eval_cfg`` is the frozen ``configs/eval.yaml`` (see
    :func:`eval.final_status.load_eval_config`); it is loaded here only as a
    convenience for callers that do not need to show the thresholds.
    """
    cfg = eval_cfg if eval_cfg is not None else load_eval_config()
    vad_hash = transcriber.vad_fingerprint()["vad_options_hash"]
    out: list[dict] = []
    for run in runs:
        text_id = run["text_id"]
        item = bench_by_id[text_id]          # load_runs() guaranteed this exists
        ref_text = item["text_ref"]
        bucket = item["bucket"]
        checkpoint = run["experiment_id"]
        gen_status = gen_status_of(run)      # provisional label from the generator
        output_path = run["output_path"]

        exists = bool(output_path) and Path(str(output_path)).exists()
        gen_complete = gen_status == COMPLETE_STATUS

        row: dict[str, Any] = {
            "run_id": run["run_id"],
            "checkpoint": checkpoint,
            "experiment_id": checkpoint,
            "text_id": text_id,
            "voice_id": run.get("voice_id"),
            "seed": run.get("seed"),
            "bucket": bucket,
            # `status` is filled in below with the FINAL §3.4 status; the
            # generator's provisional label is kept as `gen_status`.
            "gen_status": gen_status,
            "gen_status_complete": gen_complete,
            "stop_reason": run.get("stop_reason"),
            "output_path": output_path,
            "output_exists": exists,
            "n_ref_words_raw": len(str(ref_text).split()),
        }

        # --- transcribe EVERY existing wav, whatever the status ------------
        # A loop_cap / timeout / degraded run has real partial audio with real
        # content errors; treating it as an empty hypothesis would hide them.
        asr_ok = False
        if exists:
            res = transcriber.transcribe(str(output_path))
            asr_ok = (res.error is None) and bool(res.text.strip())
            row.update({
                "transcribed": True,
                "asr_text": res.text,
                "asr_segments": res.segments,
                "asr_model_id": res.model_id,
                "onnx_asr_version": res.onnx_asr_version,
                "asr_provider": res.provider,
                "asr_elapsed_sec": res.elapsed_sec,
                "asr_x_realtime": res.speed_x_realtime,
                "asr_error": res.error,
                "asr_ok": asr_ok,
            })
            hyp_text = res.text
            raw_duration = res.audio_duration_sec
            segments = res.segments
        else:
            row.update({
                "transcribed": False,
                "asr_text": "",
                "asr_segments": [],
                "asr_error": None,
                "asr_ok": False,
            })
            hyp_text = ""
            raw_duration = float(run.get("raw_duration_sec") or 0.0)
            segments = []

        # WER-valid denominator: a run whose GENERATION succeeded (gen_status)
        # and that really produced usable audio.  It deliberately does not use
        # the final status: the final `complete` is partly defined by WER, so
        # using it here would select on the outcome (configs/eval.yaml,
        # `wer_valid_denominator: gen_status`).
        usable_audio = bool(exists and asr_ok)
        row["valid"] = bool(gen_complete and usable_audio)
        if not row["valid"]:
            row["invalid_reason"] = (
                "gen_status_not_complete" if not gen_complete
                else "output_missing" if not exists
                else (row.get("asr_error") or "empty_transcription")
            )
        else:
            row["invalid_reason"] = None

        cm = content_metrics(ref_text, hyp_text, variant=variant)
        row.update(cm)
        if also_strict:
            cs = content_metrics(ref_text, hyp_text, variant="strict")
            row.update({f"strict_{k}": v for k, v in cs.items()
                        if k in ("wer", "cer", "source_coverage", "end_coverage",
                                 "end_coverage_robust", "tail_deletion_rate",
                                 "excess_repetition_rate", "max_ngram_run")})

        fl = floor.get(
            floor_cache_key(text_id, transcriber.model_id, variant, vad_hash,
                            bench_ref_hash(item)), {})
        human_dur = fl.get("human_duration_sec")
        if human_dur is None:
            human_dur = item["human_duration_sec"]
        row.update(duration_metrics(raw_duration, segments,
                                    reference_duration_sec=float(human_dur) if human_dur else None,
                                    hyp_word_count=cm["n_hyp_words"]))
        row["floor_wer"] = fl.get("floor_wer")
        row["floor_cer"] = fl.get("floor_cer")
        row["floor_available"] = fl.get("available", False)
        row["wer_minus_floor"] = (
            row["wer"] - fl["floor_wer"]
            if (fl.get("floor_wer") is not None and row["wer"] is not None
                and row["wer"] != float("inf"))
            else None
        )

        # --- final PLAN.md §3.4 status (Lead, 2026-08-28) -------------------
        verdict = assign_final_status(
            gen_status=gen_status,
            stop_reason=row["stop_reason"],
            usable_audio=usable_audio,
            end_coverage=row.get("end_coverage"),
            wer_minus_floor=row.get("wer_minus_floor"),
            floor_available=bool(row.get("floor_available")),
            excess_repetition_rate=row.get("excess_repetition_rate"),
            max_ngram_run=row.get("max_ngram_run"),
            cfg=cfg,
        )
        row.update(verdict)
        row["status"] = verdict["status"]
        row["status_complete"] = verdict["status"] == COMPLETE_STATUS
        row["status_changed_by_eval"] = verdict["status"] != gen_status
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def _fmt(v: Any, pct: bool = False, digits: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != v:
            return "-"
        if v == float("inf"):
            return "inf"
        return f"{v * 100:.{digits}f}" if pct else f"{v:.{digits}f}"
    return str(v)


def _fmt_macro(row: dict, key: str, pct: bool = False, digits: int = 2) -> str:
    """Format a macro mean and make any dropped ``inf`` visible."""
    text = _fmt(row.get(key), pct=pct, digits=digits)
    n_inf = int(row.get(f"{key}_n_inf", 0) or 0)
    if n_inf:
        text = f"{text} (+{n_inf} inf)"
    return text


def build_summary(per_item: Sequence[dict]) -> dict:
    groups: "OrderedDict[tuple[str, str], list[dict]]" = OrderedDict()
    for row in per_item:
        groups.setdefault((str(row["checkpoint"]), str(row["bucket"])), []).append(row)
    for ck in sorted({str(r["checkpoint"]) for r in per_item}):
        groups[(ck, "ALL")] = [r for r in per_item if str(r["checkpoint"]) == ck]

    summary = []
    for (ck, bucket), rows in groups.items():
        agg = aggregate(rows)
        floors = [r["floor_wer"] for r in rows if r.get("floor_wer") is not None]
        agg.update({
            "checkpoint": ck,
            "bucket": bucket,
            "floor_wer_macro": (sum(floors) / len(floors)) if floors else None,
            "n_with_floor": len(floors),
            "n_transcribed": sum(1 for r in rows if r.get("transcribed")),
            "n_output_missing": sum(1 for r in rows if not r.get("output_exists")),
            # final §3.4 status histogram (PLAN.md §17 Table 5) for this bucket
            "statuses": dict(sorted(
                ((s, sum(1 for r in rows if r["status"] == s))
                 for s in {r["status"] for r in rows}), key=lambda kv: -kv[1])),
            # provisional generator label, for the reclassification audit
            "gen_statuses": dict(sorted(
                ((s, sum(1 for r in rows if r.get("gen_status") == s))
                 for s in {r.get("gen_status") for r in rows}), key=lambda kv: -kv[1])),
            "n_status_changed_by_eval": sum(1 for r in rows if r.get("status_changed_by_eval")),
            "n_gen_status_complete": sum(1 for r in rows if r.get("gen_status_complete")),
            "gen_complete_rate": (
                sum(1 for r in rows if r.get("gen_status_complete")) / len(rows)) if rows else None,
        })
        summary.append(agg)
    return {"rows": summary}


MAIN_HEADER = (
    "| Checkpoint | Length bucket | Attempted | Complete % | gen complete % | "
    "Valid for WER-valid % | WER-all | CER-all | Coverage | EndCoverage | EndCov-robust | "
    "Repeat % | max run | dur ratio | floor WER |"
)
MAIN_SEP = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


def render_markdown(summary: dict, meta: dict, per_item: Sequence[dict]) -> str:
    lines: list[str] = []
    lines.append(f"# Evaluation summary: {meta.get('experiment_name')}")
    lines.append("")
    lines.append(f"- generated: {meta.get('generated_at')}")
    lines.append(f"- ASR: `{meta.get('asr_model_id')}` (onnx-asr {meta.get('onnx_asr_version')}, "
                 f"VAD `{meta.get('vad_model')}` hash `{meta.get('vad_options_hash')}`, "
                 f"provider `{meta.get('provider')}`)")
    lines.append(f"- normalization: `{meta.get('normalization_variant')}` "
                 f"(spec v{meta.get('normalization_version')}, strict reported as "
                 "`strict_*` in per_item.jsonl)")
    lines.append(f"- run manifests: {', '.join(meta.get('run_manifests', []))}")
    lines.append(f"- benchmark: {meta.get('benchmark')}")
    lines.append(f"- transcribed {meta.get('n_transcribed')} of {meta.get('n_attempted')} "
                 f"attempted runs ({meta.get('n_output_missing')} had no WAV)")
    thr = meta.get("final_status_thresholds", {})
    loop = thr.get("loop", {})
    lines.append(f"- final status rule: `{meta.get('eval_config')}` v{meta.get('eval_config_version')} "
                 f"(frozen {meta.get('eval_config_frozen_at')}): EndCoverage >= "
                 f"{thr.get('end_coverage_complete_min')}, WER-floor < "
                 f"{thr.get('wer_minus_floor_degraded_min')}, loop if "
                 f"excess_repetition_rate >= {loop.get('excess_repetition_rate_min')} or "
                 f"max_ngram_run >= {loop.get('ngram_run_min')}")
    lines.append(f"- evaluator changed the status of {meta.get('n_status_changed_by_eval')} of "
                 f"{meta.get('n_attempted')} runs")
    lines.append("")
    lines.append("Definitions (PLAN.md §3.4, §9.1; Lead 2026-08-28):")
    lines.append("")
    lines.append("- **`gen_status`** is the generator's *provisional* label, derived from the stop "
                 "reason alone. **`status`** is the final PLAN.md §3.4 status assigned here from "
                 "the content metrics. `per_item.jsonl` keeps both, plus `final_status_reason`.")
    lines.append("- **Complete %** = share of runs whose **final** `status == complete`, i.e. stopped "
                 "by EOS, reached EndCoverage >= "
                 f"{thr.get('end_coverage_complete_min')}, did not loop and stayed below "
                 f"WER-floor {thr.get('wer_minus_floor_degraded_min')}.")
    lines.append("- **gen complete %** = share of runs the *generator* called complete (EOS + valid "
                 "audio). The gap between the two columns is the reclassification.")
    lines.append("- **Valid for WER-valid %** = share of runs with `gen_status == complete` **and** a "
                 "readable WAV with a non-empty transcription. This is the denominator of WER-valid. "
                 "It stays on the generation side on purpose: the final `complete` is itself defined "
                 "by WER, so using it here would select on the outcome.")
    lines.append("- **WER-all** covers every attempted run. Every WAV that exists is transcribed "
                 "whatever the status, so `loop_cap` / `degraded` / `timeout` partial audio is "
                 "scored on its real content. A run with no WAV enters as an empty hypothesis, "
                 "i.e. 100 % deletions.")
    lines.append("- **Coverage** = macro `source_coverage` (share of source words matched by a hit "
                 "or a substitution anywhere). **EndCoverage** = position of the LAST such word; "
                 "it is pre-registered and it alone gates the §3.4 status. **EndCov-robust** = "
                 "`end_coverage_robust`, the same position computed only over the last run of >= 3 "
                 "consecutive aligned source words (Lead 2026-08-28), so one word coinciding deep "
                 "in the text cannot claim the model read that far.")
    lines.append("- **max run** = `max_ngram_run`: longest back-to-back repetition of a 3-8-word "
                 "phrase, discounted by what the source repeats; 1 = no repetition.")
    lines.append("- WER is **not** capped at 100 %. An item with an empty reference has WER = inf; "
                 "such items are excluded from the macro mean and counted as `(+N inf)` next to it.")
    lines.append("")
    lines.append("## Main table (PLAN.md §17)")
    lines.append("")
    lines.append(MAIN_HEADER)
    lines.append(MAIN_SEP)
    for r in summary["rows"]:
        lines.append(
            "| {ck} | {bucket} | {att} | {comp} | {gcomp} | {val} | {wer} | {cer} | {cov} | "
            "{endcov} | {endcovr} | {rep} | {run} | {dur} | {floor} |".format(
                ck=r["checkpoint"], bucket=r["bucket"], att=r["n_attempted"],
                comp=_fmt(r.get("complete_rate"), pct=True, digits=1),
                gcomp=_fmt(r.get("gen_complete_rate"), pct=True, digits=1),
                val=_fmt(r.get("valid_rate"), pct=True, digits=1),
                wer=_fmt_macro(r, "macro_wer_all", pct=True),
                cer=_fmt_macro(r, "macro_cer_all", pct=True),
                cov=_fmt_macro(r, "macro_source_coverage_all", digits=3),
                endcov=_fmt_macro(r, "macro_end_coverage_all", digits=3),
                endcovr=_fmt_macro(r, "macro_end_coverage_robust_all", digits=3),
                rep=_fmt_macro(r, "macro_excess_repetition_rate_all", pct=True),
                run=_fmt_macro(r, "macro_max_ngram_run_all", digits=2),
                dur=_fmt_macro(r, "macro_duration_ratio_all", digits=3),
                floor=_fmt(r.get("floor_wer_macro"), pct=True),
            )
        )
    lines.append("")
    lines.append("## Secondary content metrics (macro over all attempted)")
    lines.append("")
    lines.append("| Checkpoint | Bucket | micro WER-all | WER-valid | n valid | coverage | "
                 "longest del run | tail del rate | silence ratio | wpm |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary["rows"]:
        lines.append(
            "| {ck} | {b} | {mi} | {wv} | {nv} | {cov} | {ldr} | {tdr} | {sil} | {wpm} |".format(
                ck=r["checkpoint"], b=r["bucket"],
                mi=_fmt(r.get("micro_wer_all"), pct=True),
                wv=_fmt_macro(r, "macro_wer_valid", pct=True),
                nv=r.get("n_valid", 0),
                cov=_fmt_macro(r, "macro_source_coverage_all", digits=3),
                ldr=_fmt_macro(r, "macro_longest_deletion_run_all", digits=1),
                tdr=_fmt_macro(r, "macro_tail_deletion_rate_all", pct=True),
                sil=_fmt_macro(r, "macro_silence_ratio_all", pct=True),
                wpm=_fmt_macro(r, "macro_speaking_rate_wpm_raw_all", digits=1),
            )
        )
    lines.append("")
    lines.append("## Failure taxonomy (PLAN.md §17 Table 5, final status per bucket)")
    lines.append("")
    lines.extend(status_histogram_lines(per_item, "status"))
    lines.append("")
    lines.append("### Provisional generator status (`gen_status`), same layout")
    lines.append("")
    lines.extend(status_histogram_lines(per_item, "gen_status"))
    lines.append("")
    lines.append("### Reclassification by the evaluator (`gen_status` -> final `status`)")
    lines.append("")
    moves: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in per_item:
        if row.get("status_changed_by_eval"):
            moves[(str(row["checkpoint"]), str(row.get("gen_status")),
                   str(row["status"]))].append(row)
    if not moves:
        lines.append("None: every final status equals the generator's provisional status.")
    else:
        lines.append("| Checkpoint | gen_status | final status | n | example reason |")
        lines.append("|---|---|---|---:|---|")
        for (ck, g, s), rows in sorted(moves.items()):
            lines.append(f"| {ck} | {g} | {s} | {len(rows)} | "
                         f"{rows[0].get('final_status_reason')} |")
    lines.append("")
    lines.append("## Runs with audio whose final status is not `complete` "
                 "(partial or degraded output, PLAN.md §3.3)")
    lines.append("")
    partial = [r for r in per_item if r.get("output_exists") and not r.get("status_complete")]
    if not partial:
        lines.append("None.")
    else:
        lines.append("| run_id | gen_status | status | stop_reason | WER-all | WER-floor | "
                     "coverage | EndCoverage | EndCov-robust | max run | raw dur s | hyp words |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in partial:
            lines.append(
                "| {rid} | {gs} | {st} | {sr} | {wer} | {wmf} | {cov} | {ec} | {ecr} | {run} | "
                "{dur} | {hw} |".format(
                    rid=r.get("run_id"), gs=r.get("gen_status"), st=r.get("status"),
                    sr=r.get("stop_reason"), wer=_fmt(r.get("wer"), pct=True),
                    wmf=_fmt(r.get("wer_minus_floor"), pct=True),
                    cov=_fmt(r.get("source_coverage"), digits=3),
                    ec=_fmt(r.get("end_coverage"), digits=3),
                    ecr=_fmt(r.get("end_coverage_robust"), digits=3),
                    run=r.get("max_ngram_run"),
                    dur=_fmt(r.get("raw_duration_sec"), digits=1),
                    hw=r.get("n_hyp_words"),
                )
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def status_histogram_lines(per_item: Sequence[dict], key: str) -> list[str]:
    """PLAN.md §17 Table 5: status histogram with one row per checkpoint x bucket."""
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in per_item:
        ck, bucket = str(row["checkpoint"]), str(row["bucket"])
        counts[(ck, bucket)][str(row.get(key))] += 1
        counts[(ck, "ALL")][str(row.get(key))] += 1
    seen = {s for c in counts.values() for s in c}
    statuses = [s for s in KNOWN_STATUSES if s in seen] + sorted(seen - set(KNOWN_STATUSES))
    out = ["| Checkpoint | Bucket | n | " + " | ".join(statuses) + " |",
           "|---|---|---:|" + "---:|" * len(statuses)]
    for (ck, bucket) in sorted(counts, key=lambda k: (k[0], k[1] == "ALL", k[1])):
        c = counts[(ck, bucket)]
        out.append(f"| {ck} | {bucket} | {sum(c.values())} | "
                   + " | ".join(str(c.get(s, 0)) for s in statuses) + " |")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate TTS runs (GigaAM-v3 + VAD)")
    ap.add_argument("--run-manifest", nargs="+", default=None)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--floor-only", action="store_true",
                    help="compute and cache the ASR floor of the benchmark, no runs needed "
                         "(usable before any generation exists)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--experiment-name", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(GIGAAM_V3_VARIANTS))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--variant", default=None, choices=["strict", "lenient"],
                    help="normalization variant (default: primary_variant from "
                         "configs/text_normalization.yaml)")
    ap.add_argument("--floor-cache", default=None,
                    help="path of the ASR floor cache (default: <output-dir>/asr_floor.jsonl)")
    ap.add_argument("--refresh-floor", action="store_true")
    ap.add_argument("--skip-floor", action="store_true", help="do not transcribe human references")
    ap.add_argument("--eval-config", default=str(EVAL_CONFIG_PATH),
                    help="frozen §3.4 final-status thresholds (default: configs/eval.yaml)")
    args = ap.parse_args(argv)

    spec = load_spec()
    variant = args.variant or spec["primary_variant"]
    # loaded before anything expensive: a malformed threshold file must not cost a GPU load
    eval_cfg = load_eval_config(args.eval_config)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # schema validation happens BEFORE the model is loaded, so a contract
    # violation costs a second instead of a GPU load.
    if not args.floor_only and not args.run_manifest:
        ap.error("--run-manifest is required unless --floor-only is given")
    if args.floor_only and args.skip_floor:
        ap.error("--floor-only and --skip-floor contradict each other")

    bench_by_id = load_benchmark(args.benchmark)
    bench = list(bench_by_id.values())
    runs = [] if args.floor_only else load_runs(args.run_manifest, bench_by_id)
    print(f"[eval] {len(runs)} run rows, {len(bench)} benchmark texts", file=sys.stderr)

    # The benchmark's text_ref is pre-normalized with A2's primary variant; scoring
    # with another variant is legal but must be visible.
    bench_variants = {b.get("normalization_variant") for b in bench if b.get("normalization_variant")}
    if bench_variants and bench_variants != {variant}:
        print(f"[eval] WARNING: benchmark text_ref was normalized with {sorted(bench_variants)} "
              f"but this run scores with {variant!r}", file=sys.stderr)

    t0 = time.perf_counter()
    tr = GigaAMTranscriber(model_id=args.model, use_vad=True, device=args.device)
    print(json.dumps(tr.describe(), ensure_ascii=False), file=sys.stderr)

    floor_path = Path(args.floor_cache) if args.floor_cache else out_dir / "asr_floor.jsonl"
    floor = {} if args.skip_floor else compute_floor(bench, tr, floor_path, variant,
                                                     args.refresh_floor)

    if args.floor_only:
        avail = [f for f in floor.values() if f.get("available")]
        wers = [f["floor_wer"] for f in avail]
        print(f"[eval] floor-only: {len(avail)}/{len(bench)} benchmark texts have human audio; "
              f"mean floor WER = {(sum(wers) / len(wers) * 100 if wers else float('nan')):.2f} % "
              f"-> {floor_path}", file=sys.stderr)
        return 0

    per_item = evaluate_runs(runs, bench_by_id, tr, floor, variant, eval_cfg=eval_cfg)
    write_jsonl(out_dir / "per_item.jsonl", per_item)

    summary = build_summary(per_item)
    fp = tr.vad_fingerprint()
    meta = {
        "experiment_name": args.experiment_name or out_dir.name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "asr_model_id": tr.model_id,
        "onnx_asr_version": tr.onnx_asr_version,
        "vad_model": tr.vad_model_name,
        "vad_options": tr.vad_options,
        "vad_options_hash": fp["vad_options_hash"],
        "provider": tr.provider,
        "device": args.device,
        "normalization_variant": variant,
        "normalization_version": spec.get("version"),
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
        # the frozen §3.4 rule this table was produced with (PLAN.md §3.5)
        "eval_config": eval_cfg.get("__path__"),
        "eval_config_version": eval_cfg.get("version"),
        "eval_config_frozen_at": eval_cfg.get("frozen_at"),
        "final_status_thresholds": {
            k: v for k, v in eval_cfg["final_status"].items()
            if k in ("end_coverage_complete_min", "wer_minus_floor_degraded_min",
                     "wer_criterion_when_floor_missing", "no_usable_audio_status", "loop")
        },
        "wer_valid_denominator": eval_cfg.get("wer_valid_denominator"),
        "elapsed_sec": time.perf_counter() - t0,
    }
    with open(out_dir / "summary.json", "wt", encoding="utf-8") as f:
        json.dump({"meta": meta, **summary}, f, ensure_ascii=False, indent=1)
    (out_dir / "summary.md").write_text(render_markdown(summary, meta, per_item), encoding="utf-8")

    print(f"[eval] wrote {out_dir/'per_item.jsonl'}, {out_dir/'summary.md'} "
          f"in {meta['elapsed_sec']:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkContractError as exc:
        print(f"[eval] CONTRACT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except FinalStatusError as exc:
        # frozen thresholds missing/malformed, or a run manifest that cannot be
        # classified: never fall back to an unregistered default (PLAN.md §3.5).
        print(f"[eval] FINAL-STATUS ERROR: {exc}", file=sys.stderr)
        raise SystemExit(4)
    except GpuGuardError as exc:
        # PLAN.md §0.1: wrong card -> stop before any session is created.
        print(f"[eval] GPU GUARD: {exc}", file=sys.stderr)
        raise SystemExit(3)
