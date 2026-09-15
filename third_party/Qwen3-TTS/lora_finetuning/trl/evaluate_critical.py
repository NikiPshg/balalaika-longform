#!/usr/bin/env python3
"""Resumable typed critical-span evaluation using the existing local evaluators."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
TRL_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, TRL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from critical_span_metrics import (  # noqa: E402
    aggregate_typed_metrics,
    infer_changed_token_span,
    score_typed_text_pair,
)
from grpo_dpo_finetuning.evaluate import (  # noqa: E402
    DistillMOSReDimNet2Backend,
    GigaAMRNNTBackend,
    QwenSynthesisBackend,
    run_batched_stage,
    synthesis_seed_for_input,
)


SCHEMA_VERSION = 1
DEFAULT_SEED = 20260812
OBJECTIVE_WEIGHTS = {
    "micro_critical_wer": 0.35,
    "micro_critical_cer": 0.35,
    "micro_utterance_wer": 0.15,
    "micro_utterance_cer": 0.15,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
    )


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def create_screen_manifest(
    targets_path: Path,
    references_path: Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    targets = _read_jsonl(targets_path)
    payload = json.loads(references_path.read_text(encoding="utf-8"))
    references = payload.get("references") if isinstance(payload, Mapping) else None
    if not isinstance(references, list) or len(references) < 2:
        raise ValueError("screen evaluation requires at least two fixed references")
    rows: list[dict[str, Any]] = []
    for target in targets:
        for reference in references:
            identity = {
                "seed": seed,
                "target": target["id"],
                "reference": reference["id"],
            }
            audio = _resolve_path(reference["wav"])
            input_id = _sha256_json(identity)
            rows.append(
                {
                    "input_id": input_id,
                    "utterance_id": target["id"],
                    "target_uid": target["id"],
                    "reference_key": reference["id"],
                    "audio_path": str(audio),
                    "reference_text": reference["ref_text"],
                    "stressed_text": target["stressed"],
                    "normalized_gold": target["source_text"],
                    "critical_spans": target["critical_spans"],
                    "groups": target.get("groups", {}),
                }
            )
    return _manifest(
        rows,
        seed=seed,
        sources={
            "targets": {"path": str(targets_path.resolve()), "sha256": _sha256_file(targets_path)},
            "references": {
                "path": str(references_path.resolve()),
                "sha256": _sha256_file(references_path),
            },
        },
    )


def _hard_numeric_span(row: Mapping[str, Any]) -> dict[str, Any]:
    start, end, words = infer_changed_token_span(str(row["text"]), str(row["normalized_gold"]))
    category = str(row["category"])
    span_type = {
        "date": "date",
        "time": "time",
        "ident": "identifier",
        "percent": "percent",
        "kopecks": "money",
        "ranges": "number_range",
    }.get(category, "number")
    return {
        "type": span_type,
        "surface": str(row.get("hard_number") or ""),
        "spoken_gold": list(words),
        "token_start": start,
        "token_end": end,
    }


def create_final_manifest(
    direct_path: Path,
    initials_path: Path,
    hard_numbers_path: Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    initials = {row["id"]: row for row in _read_jsonl(initials_path)}
    hard = {str(row["id"]): row for row in _read_jsonl(hard_numbers_path)}
    rows: list[dict[str, Any]] = []
    for source in _read_jsonl(direct_path):
        uid = str(source["target_uid"])
        if uid.startswith("initials-bench:test:"):
            item = initials[uid.removeprefix("initials-bench:test:")]
            target_text = str(item["text"])
            spoken = str(item["source_text"])
            from build_critical_curriculum import _find_unique_tokens

            start, end, gold = _find_unique_tokens(target_text, spoken)
            critical_spans = [
                {
                    "type": str(item["task"]),
                    "surface": spoken,
                    "spoken_gold": gold,
                    "token_start": start,
                    "token_end": end,
                }
            ]
        elif uid.startswith("hard-numbers:hard_number_eval:"):
            item = hard[uid.rsplit(":", 1)[-1]]
            critical_spans = [_hard_numeric_span(item)]
        else:
            raise ValueError(f"unsupported final target uid: {uid}")
        if str(source["reference"]).strip() == "":
            raise ValueError("final row has an empty normalized reference")
        input_id = _sha256_json(
            {"seed": seed, "row_id": source["id"], "target_uid": uid}
        )
        rows.append(
            {
                "input_id": input_id,
                "utterance_id": uid,
                "target_uid": uid,
                "reference_key": source["reference_id"],
                "audio_path": str(_resolve_path(source["reference_audio"])),
                "reference_text": source["reference_text"],
                "stressed_text": source["text"],
                "normalized_gold": source["reference"],
                "critical_spans": critical_spans,
                "groups": source.get("groups", {}),
            }
        )
    return _manifest(
        rows,
        seed=seed,
        sources={
            "direct": {"path": str(direct_path.resolve()), "sha256": _sha256_file(direct_path)},
            "initials": {"path": str(initials_path.resolve()), "sha256": _sha256_file(initials_path)},
            "hard_numbers": {
                "path": str(hard_numbers_path.resolve()),
                "sha256": _sha256_file(hard_numbers_path),
            },
        },
    )


def _manifest(rows: Sequence[Mapping[str, Any]], *, seed: int, sources: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("evaluation manifest must not be empty")
    input_ids = [str(row["input_id"]) for row in rows]
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("evaluation input ids must be unique")
    for row in rows:
        if not Path(str(row["audio_path"])).is_file():
            raise FileNotFoundError(f"reference audio does not exist: {row['audio_path']}")
        score_typed_text_pair(
            str(row["normalized_gold"]),
            str(row["normalized_gold"]),
            row["critical_spans"],
        )
    base = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "row_count": len(rows),
        "objective_weights": OBJECTIVE_WEIGHTS,
        "sources": dict(sources),
        "rows": [dict(row) for row in rows],
    }
    return {**base, "manifest_hash": _sha256_json(base)}


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    base = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("schema_version") != SCHEMA_VERSION or value.get("manifest_hash") != _sha256_json(base):
        raise ValueError("typed evaluation manifest hash mismatch")
    if value.get("row_count") != len(value.get("rows", [])):
        raise ValueError("typed evaluation manifest row count mismatch")
    return value


def subset_manifest(
    manifest: Mapping[str, Any], row_count: int, *, start: int = 0
) -> dict[str, Any]:
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ValueError("subset row_count must be a positive integer")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("subset start must be a non-negative integer")
    rows = list(manifest.get("rows", ()))
    if start + row_count > len(rows):
        raise ValueError("subset row_count exceeds parent manifest coverage")
    base = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    base["rows"] = rows[start : start + row_count]
    base["row_count"] = row_count
    base["subset_start"] = start
    base["parent_manifest_hash"] = manifest["manifest_hash"]
    return {**base, "manifest_hash": _sha256_json(base)}


def attach_model(
    manifest: Mapping[str, Any], *, label: str, revision: str, seed_offset: int = 0
) -> list[dict[str, Any]]:
    if isinstance(seed_offset, bool) or not isinstance(seed_offset, int) or seed_offset < 0:
        raise ValueError("seed_offset must be a non-negative integer")
    output = []
    for row in manifest["rows"]:
        cache_key = _sha256_json(
            {
                "manifest_hash": manifest["manifest_hash"],
                "model_revision": revision,
                "input_id": row["input_id"],
                "seed_offset": seed_offset,
            }
        )
        output.append(
            {
                **row,
                "model": label,
                "model_revision": revision,
                "manifest_hash": manifest["manifest_hash"],
                "cache_key": cache_key,
                "synthesis_seed": synthesis_seed_for_input(
                    manifest["seed"] + 1_000_003 * seed_offset, row["input_id"]
                ),
                "sampling_seed_offset": seed_offset,
                "status": "ok",
            }
        )
    return sorted(
        output,
        key=lambda row: (
            row["reference_key"],
            len(str(row["stressed_text"])),
            row["input_id"],
        ),
    )


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float)) and math.isfinite(float(row[field]))]
    return sum(values) / len(values) if values else None


def rerank_candidate_rows(
    candidate_sets: Sequence[Sequence[Mapping[str, Any]]],
    *,
    model_label: str,
    model_revision: str,
) -> list[dict[str, Any]]:
    """Select one ASR-verified critical-span candidate per immutable input."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate_set in candidate_sets:
        for raw in candidate_set:
            row = dict(raw)
            grouped.setdefault(str(row["input_id"]), []).append(row)
    if any(len(rows) != len(candidate_sets) for rows in grouped.values()):
        raise ValueError("every rerank input must have one row from every candidate set")

    output = []
    for input_id in sorted(grouped):
        ranked = []
        for row in grouped[input_id]:
            hypothesis = str(row.get("hypothesis") or "")
            if row.get("status") != "ok" or not hypothesis:
                ranked.append(((True, math.inf, math.inf, math.inf, math.inf), row, {}))
                continue
            metrics = score_typed_text_pair(
                str(row["normalized_gold"]), hypothesis, row["critical_spans"]
            )
            critical_error = (
                0.35 * metrics.critical_wer
                + 0.45 * metrics.critical_cer
                + 0.20 * (1.0 - metrics.critical_exact_rate)
            )
            utterance_error = 0.5 * metrics.utterance_wer + 0.5 * metrics.utterance_cer
            word_count = max(1, len(str(row["normalized_gold"]).split()))
            duration = float(row.get("generated_duration") or 0.0)
            duration_ratio = duration / (word_count / 2.5) if duration > 0 else math.inf
            pace_error = min(1.0, abs(math.log(max(duration_ratio, 1e-6))))
            outside_pace_gate = not 0.65 <= duration_ratio <= 1.30
            score = 0.55 * critical_error + 0.35 * utterance_error + 0.10 * pace_error
            ranked.append(
                (
                    (outside_pace_gate, score, critical_error, utterance_error, duration_ratio),
                    row,
                    {
                        "critical_error": critical_error,
                        "utterance_error": utterance_error,
                        "duration_ratio": duration_ratio,
                        "selection_score": score,
                    },
                )
            )
        rank, selected, diagnostics = min(ranked, key=lambda item: item[0])
        selected.update(
            {
                "model": model_label,
                "model_revision": model_revision,
                "rerank_candidate_count": len(ranked),
                "rerank_selection": diagnostics,
                "rerank_rank_key": list(rank),
            }
        )
        output.append(selected)
    return output


def score_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    results = []
    for row in rows:
        hypothesis = str(row.get("hypothesis") or "")
        result = score_typed_text_pair(
            str(row["normalized_gold"]), hypothesis, row["critical_spans"]
        )
        results.append(result)
        scored.append({**dict(row), "typed_metrics": result.as_dict()})
    typed = aggregate_typed_metrics(results)
    quality_ok = [row for row in scored if row.get("quality_status") == "ok"]
    failure_rate = sum(row.get("status") != "ok" or not row.get("hypothesis") for row in scored) / len(scored)
    overall = {
        "model": str(scored[0].get("model", "candidate")),
        "row_count": len(scored),
        "micro_critical_wer": typed["critical"]["micro_wer"],
        "micro_critical_cer": typed["critical"]["micro_cer"],
        "critical_exact_rate": typed["critical"]["exact_rate"],
        "micro_utterance_wer": typed["micro_utterance_wer"],
        "micro_utterance_cer": typed["micro_utterance_cer"],
        "mos": _mean(quality_ok, "mos"),
        "sim": _mean(quality_ok, "sim"),
        "duration_ratio_deviation": (
            _mean([{**row, "value": abs(float(row["duration_ratio"]) - 1.0)} for row in quality_ok if isinstance(row.get("duration_ratio"), (int, float))], "value")
        ),
        "repetition_score": _mean(quality_ok, "repetition_score"),
        "failure_rate": failure_rate,
        "quality_coverage": len(quality_ok) / len(scored),
        "manifest_hash": scored[0].get("manifest_hash"),
        "model_revision": scored[0].get("model_revision"),
        "quality_guard_passed": True,
    }
    overall["objective"] = sum(OBJECTIVE_WEIGHTS[name] * float(overall[name]) for name in OBJECTIVE_WEIGHTS)
    per_type = []
    for name, block in typed["per_type"].items():
        per_type.append({"type": name, **block, "objective": 0.5 * block["micro_wer"] + 0.5 * block["micro_cer"]})
    return scored, {"schema_version": 1, **overall, "per_type": per_type}


def compare_aggregates(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    regressions: list[str] = []
    limits = {
        "mos": -0.10,
        "sim": -0.02,
        "duration_ratio_deviation": 0.10,
        "repetition_score": 0.05,
        "failure_rate": 0.005,
        "quality_coverage": -0.005,
    }
    for field, tolerance in limits.items():
        base, cand = baseline.get(field), candidate.get(field)
        if not isinstance(base, (int, float)) or not isinstance(cand, (int, float)):
            regressions.append(f"missing {field}")
            continue
        delta = float(cand) - float(base)
        if (tolerance < 0 and delta < tolerance) or (tolerance >= 0 and delta > tolerance):
            regressions.append(f"{field} delta {delta:+.6f} exceeds {tolerance:+.6f}")
    baseline_types = {row["type"]: row for row in baseline.get("per_type", [])}
    for row in candidate.get("per_type", []):
        if row["type"] in baseline_types:
            delta = float(row["objective"]) - float(baseline_types[row["type"]]["objective"])
            if delta > 0.03:
                regressions.append(f"worst-type {row['type']} objective delta {delta:+.6f}")
    improved = float(candidate["objective"]) < float(baseline["objective"])
    passed = improved and not regressions
    return {
        **dict(candidate),
        "baseline_objective": baseline["objective"],
        "candidate_objective": candidate["objective"],
        "objective_improved": improved,
        "quality_guard_passed": passed,
        "selected_model": candidate.get("model") if passed else "baseline",
        "regressions": regressions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--targets", type=Path)
    prepare.add_argument("--references", type=Path)
    prepare.add_argument("--direct-final", type=Path)
    prepare.add_argument("--initials", type=Path)
    prepare.add_argument("--hard-numbers", type=Path)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    subset = sub.add_parser("subset")
    subset.add_argument("--manifest", type=Path, required=True)
    subset.add_argument("--output", type=Path, required=True)
    subset.add_argument("--rows", type=int, required=True)
    subset.add_argument("--start", type=int, default=0)
    for name in ("synth", "asr", "quality", "score"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--input-jsonl", type=Path)
        if name == "synth":
            command.add_argument("--model-path", required=True)
            command.add_argument("--model-revision", required=True)
            command.add_argument("--model-label", required=True)
            command.add_argument("--max-new-tokens", type=int, default=512)
            command.add_argument("--seed-offset", type=int, default=0)
    rerank = sub.add_parser("rerank")
    rerank.add_argument("--manifest", type=Path, required=True)
    rerank.add_argument("--output", type=Path, required=True)
    rerank.add_argument("--candidate-jsonl", type=Path, action="append", required=True)
    rerank.add_argument("--model-label", required=True)
    rerank.add_argument("--model-revision", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        if args.direct_final:
            if not args.initials or not args.hard_numbers:
                parser.error("direct final prepare requires --initials and --hard-numbers")
            manifest = create_final_manifest(args.direct_final, args.initials, args.hard_numbers, seed=args.seed)
        else:
            if not args.targets or not args.references:
                parser.error("screen prepare requires --targets and --references")
            manifest = create_screen_manifest(args.targets, args.references, seed=args.seed)
        if args.output.exists() and json.loads(args.output.read_text(encoding="utf-8")) != manifest:
            raise ValueError("immutable typed evaluation manifest differs from existing file")
        _atomic_json(args.output, manifest)
        print(f"prepared {manifest['row_count']} rows; hash={manifest['manifest_hash']}")
        return
    if args.command == "subset":
        manifest = subset_manifest(load_manifest(args.manifest), args.rows, start=args.start)
        if args.output.exists() and json.loads(args.output.read_text(encoding="utf-8")) != manifest:
            raise ValueError("immutable typed subset manifest differs from existing file")
        _atomic_json(args.output, manifest)
        print(f"prepared subset {manifest['row_count']} rows; hash={manifest['manifest_hash']}")
        return
    if args.command == "compare":
        result = compare_aggregates(
            json.loads(args.baseline.read_text(encoding="utf-8")),
            json.loads(args.candidate.read_text(encoding="utf-8")),
        )
        _atomic_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    manifest = load_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "synth":
        backend = QwenSynthesisBackend(
            args.model_path,
            args.model_revision,
            args.output / "wavs",
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        rows = attach_model(
            manifest,
            label=args.model_label,
            revision=backend.model_revision,
            seed_offset=args.seed_offset,
        )
        run_batched_stage(
            rows,
            stage="synth",
            backend=backend,
            output_path=args.output / "synth.jsonl",
            batch_size=args.batch_size,
            batch_group_key=lambda row: row["reference_key"],
            replay_full_batches_on_resume=True,
        )
    elif args.command == "rerank":
        rows = rerank_candidate_rows(
            [_read_jsonl(path) for path in args.candidate_jsonl],
            model_label=args.model_label,
            model_revision=args.model_revision,
        )
        if len(rows) != manifest["row_count"]:
            raise ValueError("rerank output does not match immutable manifest coverage")
        _write_jsonl(args.output / "reranked.jsonl", rows)
        print(f"reranked {len(rows)} rows from {len(args.candidate_jsonl)} candidates")
    elif args.command == "asr":
        run_batched_stage(
            _read_jsonl(args.input_jsonl or (args.output / "synth.jsonl")),
            stage="asr_gigaam_v3_rnnt",
            backend=GigaAMRNNTBackend(),
            output_path=args.output / "asr.jsonl",
            batch_size=args.batch_size,
        )
    elif args.command == "quality":
        run_batched_stage(
            _read_jsonl(args.input_jsonl or (args.output / "asr.jsonl")),
            stage="quality_distillmos_redimnet2",
            backend=DistillMOSReDimNet2Backend(device=args.device),
            output_path=args.output / "quality.jsonl",
            batch_size=args.batch_size,
        )
    elif args.command == "score":
        source = args.input_jsonl or (args.output / "quality.jsonl")
        if not source.exists():
            source = args.output / "asr.jsonl"
        scored, aggregate = score_rows(_read_jsonl(source))
        _write_jsonl(args.output / "scored.jsonl", scored)
        _atomic_json(args.output / "aggregate.json", aggregate)
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
