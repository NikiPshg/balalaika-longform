#!/usr/bin/env python3
"""Build immutable local RL/screen/final splits from pinned critical text sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
TRL_DIR = Path(__file__).resolve().parent
FINETUNING_ROOT = TRL_DIR.parent
for path in (REPO_ROOT, TRL_DIR, FINETUNING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from critical_span_metrics import infer_changed_token_span, normalize_text  # noqa: E402
from full_utterance_utils import apply_silero_stress  # noqa: E402


INITIALS_REVISION = "1c613d917c79036834b55bcead0cad928a9e5641"
NUMERIC_REVISION = "04d46ef44ff7c19b8ede9f9cc38069b14714e9cc"
SPLIT_SEED = 20260812


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_hash(value: str, namespace: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}\0{namespace}\0{value}".encode()).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _find_unique_tokens(reference: str, spoken_gold: str) -> tuple[int, int, list[str]]:
    reference_tokens = normalize_text(reference).split()
    gold = normalize_text(spoken_gold).split()
    matches = [
        start
        for start in range(len(reference_tokens) - len(gold) + 1)
        if reference_tokens[start : start + len(gold)] == gold
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one spoken span occurrence, found {len(matches)}")
    return matches[0], matches[0] + len(gold), gold


def initials_record(row: Mapping[str, Any]) -> dict[str, Any]:
    source_text = str(row["text"])
    spoken = str(row["source_text"])
    start, end, gold = _find_unique_tokens(source_text, spoken)
    task = str(row["task"])
    return {
        "id": f"initials-bench:test:{row['id']}",
        "source_dataset": "bitmanagerai/initials-bench",
        "source_revision": INITIALS_REVISION,
        "source_id": str(row["id"]),
        "prompt": str(row["text_stressed"]),
        "stressed": str(row["text_stressed"]),
        "source_text": source_text,
        "critical_spans": [
            {
                "type": task,
                "surface": spoken,
                "spoken_gold": gold,
                "token_start": start,
                "token_end": end,
            }
        ],
        "num_words": [],
        "groups": {
            "source": "initials-bench",
            "critical_type": task,
            "subset": row["subset"],
            "gender": row["gender"],
            "placement": row["placement_category"],
            "category": row["context_category"],
        },
    }


def _numeric_type(row: Mapping[str, Any]) -> str:
    spans = json.loads(str(row["number_spans_json"]))
    if len(spans) != 1:
        raise ValueError("repeated-digits v2 row must contain exactly one numeric span")
    numeric_format = str(spans[0].get("format") or "")
    category = str(row["category"])
    return {
        "phone": "phone",
        "short_identifier": "identifier",
        "money": "money",
        "measurement": "measurement",
        "percent": "percent",
        "cardinal": "number",
    }.get(numeric_format, "phone" if category == "hard" else "number")


def numeric_record(row: Mapping[str, Any], *, accentor: Any | None = None) -> dict[str, Any]:
    source_text = str(row["text_normalized"])
    start, end, gold = infer_changed_token_span(str(row["text_raw"]), source_text)
    stressed = apply_silero_stress(accentor, source_text) if accentor is not None else source_text
    if not stressed.strip():
        raise ValueError("Silero Stress returned an empty numeric prompt")
    span_payload = json.loads(str(row["number_spans_json"]))[0]
    span_type = _numeric_type(row)
    return {
        "id": f"rutts-rl-numeric:{row['id']}",
        "source_dataset": "bitmanagerai/rutts_rl_numeric_texts",
        "source_revision": NUMERIC_REVISION,
        "source_id": str(row["id"]),
        "prompt": stressed,
        "stressed": stressed,
        "source_text": source_text,
        "critical_spans": [
            {
                "type": span_type,
                "surface": span_payload["surface"],
                "spoken_gold": list(gold),
                "token_start": start,
                "token_end": end,
            }
        ],
        "num_words": list(gold),
        "groups": {
            "source": "rutts-rl-numeric",
            "critical_type": span_type,
            "category": row["category"],
            "numeric_format": span_payload["format"],
            "pronunciation_mode": span_payload["pronunciation_mode"],
            "run_length": len(str(span_payload["normalized_value"])),
            "digit": str(span_payload["normalized_value"])[0],
        },
    }


def _take_by_group(
    rows: Sequence[dict[str, Any]],
    *,
    group_fields: Sequence[str],
    count_for_group: Any,
    namespace: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row["groups"][field] for field in group_fields)
        grouped.setdefault(key, []).append(row)
    selected: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: repr(item[0])):
        ranked = sorted(values, key=lambda row: _identity_hash(row["id"], f"{namespace}:{key}"))
        count = int(count_for_group(key, len(ranked)))
        if not 0 <= count < len(ranked):
            raise ValueError(f"invalid split count {count} for group {key} with {len(ranked)} rows")
        selected.extend(ranked[:count])
        remaining.extend(ranked[count:])
    return selected, remaining


def _deterministic_order(rows: Iterable[dict[str, Any]], namespace: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _identity_hash(row["id"], namespace))


def _tag_split(rows: Iterable[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [{**row, "split": split} for row in rows]


def _validate_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    ids: dict[str, str] = {}
    texts: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            row_id = str(row["id"])
            text = normalize_text(row["source_text"])
            if row_id in ids:
                raise ValueError(f"id leakage between {ids[row_id]} and {split}: {row_id}")
            if text in texts:
                raise ValueError(f"normalized text leakage between {texts[text]} and {split}")
            ids[row_id] = split
            texts[text] = split


def build_curriculum(
    initials_rows: Sequence[Mapping[str, Any]],
    numeric_rows: Sequence[Mapping[str, Any]],
    final_initial_ids: set[str],
    *,
    accentor: Any | None = None,
) -> dict[str, list[dict[str, Any]]]:
    initials = [initials_record(row) for row in initials_rows]
    numeric = [numeric_record(row, accentor=accentor) for row in numeric_rows]
    initial_final = [row for row in initials if row["source_id"] in final_initial_ids]
    initial_pool = [row for row in initials if row["source_id"] not in final_initial_ids]
    if len(initial_final) != len(final_initial_ids):
        raise ValueError("not all immutable multivoice initials targets were found")
    initial_screen, initial_train = _take_by_group(
        initial_pool,
        group_fields=("critical_type", "subset"),
        count_for_group=lambda _key, _size: 50,
        namespace="initial-screen",
    )
    numeric_final, numeric_pool = _take_by_group(
        numeric,
        group_fields=("category",),
        count_for_group=lambda _key, size: round(size * 0.05),
        namespace="numeric-final",
    )
    numeric_screen, numeric_train = _take_by_group(
        numeric_pool,
        group_fields=("category",),
        count_for_group=lambda _key, size: round(size / 19.0),
        namespace="numeric-screen",
    )
    initial_screen_eval, _ = _take_by_group(
        initial_screen,
        group_fields=("critical_type", "subset"),
        count_for_group=lambda _key, _size: 12,
        namespace="initial-screen-eval",
    )
    numeric_screen_eval, _ = _take_by_group(
        numeric_screen,
        group_fields=("category",),
        count_for_group=lambda _key, size: max(1, round(size * 0.10)),
        namespace="numeric-screen-eval",
    )
    splits = {
        "train": _tag_split(
            _deterministic_order([*initial_train, *numeric_train], "train-order"), "train"
        ),
        "screen": _tag_split(
            _deterministic_order([*initial_screen, *numeric_screen], "screen-order"), "screen"
        ),
        "final": _tag_split(
            _deterministic_order([*initial_final, *numeric_final], "final-order"), "final"
        ),
        "screen_eval_targets": _tag_split(
            _deterministic_order(
                [*initial_screen_eval, *numeric_screen_eval], "screen-eval-order"
            ),
            "screen",
        ),
    }
    _validate_disjoint({key: splits[key] for key in ("train", "screen", "final")})
    eval_ids = {row["id"] for row in splits["screen_eval_targets"]}
    if not eval_ids <= {row["id"] for row in splits["screen"]}:
        raise ValueError("screen evaluator targets are not a subset of screen")
    return splits


def _load_numeric_rows(root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    files = sorted(root.glob("steps/step-*/train-00000-of-00000.csv"))
    if len(files) != 10:
        raise ValueError(f"expected ten repeated-digits step CSVs, found {len(files)}")
    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open(encoding="utf-8", newline="") as source:
            shard = list(csv.DictReader(source))
        if len(shard) != 1000:
            raise ValueError(f"expected 1,000 rows in {path}, found {len(shard)}")
        rows.extend(shard)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("numeric step shards contain duplicate ids")
    return rows, files


def _final_initial_ids(eval_manifest: Path) -> set[str]:
    values = set()
    for row in _read_jsonl(eval_manifest):
        uid = str(row.get("target_uid") or "")
        prefix = "initials-bench:test:"
        if uid.startswith(prefix):
            values.add(uid[len(prefix) :])
    if not values:
        raise ValueError("multivoice final manifest contains no initials targets")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = TRL_DIR / "assets" / "autoresearch" / "sources"
    parser.add_argument("--initials", type=Path, default=sources / "initials-bench" / "test.jsonl")
    parser.add_argument(
        "--numeric-root",
        type=Path,
        default=sources / "rutts_rl_numeric_texts" / "new-10k" / "repeated-digits-6-9-mixed-v2",
    )
    parser.add_argument(
        "--multivoice-final",
        type=Path,
        default=TRL_DIR / "assets" / "multivoice-benchmark" / "eval-2000.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-stress", action="store_true")
    args = parser.parse_args()
    initials = _read_jsonl(args.initials)
    numeric, numeric_files = _load_numeric_rows(args.numeric_root)
    accentor = None
    if not args.skip_stress:
        from silero_stress import load_accentor

        import torch

        grad_enabled = torch.is_grad_enabled()
        try:
            with torch.no_grad():
                accentor = load_accentor()
        finally:
            torch.set_grad_enabled(grad_enabled)
    splits = build_curriculum(
        initials,
        numeric,
        _final_initial_ids(args.multivoice_final),
        accentor=accentor,
    )
    outputs = {
        "train": args.output / "train.jsonl",
        "screen": args.output / "screen.jsonl",
        "final": args.output / "final.jsonl",
        "screen_eval_targets": args.output / "screen-eval-targets.jsonl",
    }
    for key, path in outputs.items():
        _write_jsonl(path, splits[key])
    contract = {
        "schema_version": 1,
        "split_seed": SPLIT_SEED,
        "stress_backend": None if args.skip_stress else "silero-stress==1.4",
        "sources": {
            "initials": {
                "repo": "bitmanagerai/initials-bench",
                "revision": INITIALS_REVISION,
                "path": str(args.initials.resolve()),
                "sha256": _sha256(args.initials),
                "rows": len(initials),
            },
            "numeric_steps": {
                "repo": "bitmanagerai/rutts_rl_numeric_texts",
                "revision": NUMERIC_REVISION,
                "paths": [str(path.resolve()) for path in numeric_files],
                "sha256": {str(path.resolve()): _sha256(path) for path in numeric_files},
                "rows": len(numeric),
            },
            "multivoice_final": {
                "path": str(args.multivoice_final.resolve()),
                "sha256": _sha256(args.multivoice_final),
            },
        },
        "outputs": {
            key: {"path": str(path.resolve()), "sha256": _sha256(path), "rows": len(splits[key])}
            for key, path in outputs.items()
        },
        "counts_by_source": {
            key: {
                source: sum(row["groups"]["source"] == source for row in rows)
                for source in ("initials-bench", "rutts-rl-numeric")
            }
            for key, rows in splits.items()
        },
    }
    _atomic_text(
        args.output / "split-contract.json",
        json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
