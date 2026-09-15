#!/usr/bin/env python3
"""Cross 20 pinned voice references with a deterministic 100-text eval set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--eval-repo", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--texts-per-reference", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_repo = args.eval_repo.expanduser().resolve()
    sys.path.insert(0, str(eval_repo / "src"))
    from speech_eval.config import RunConfig
    from speech_eval.datasets import stratified_limit
    from speech_eval.runner import validate_datasets

    reference_payload = json.loads(args.references.expanduser().resolve().read_text())
    references = reference_payload.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("references manifest has no references")
    config = RunConfig.load(args.target_config.expanduser().resolve())
    available, reports = validate_datasets(config)
    targets = stratified_limit(available, args.texts_per_reference, config.seed)
    if len(targets) != args.texts_per_reference:
        raise RuntimeError(
            f"Selected {len(targets)} targets, expected {args.texts_per_reference}"
        )
    target_ids = [sample.uid for sample in targets]
    target_sha = hashlib.sha256("\n".join(target_ids).encode()).hexdigest()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for reference in references:
        reference_id = str(reference["reference_id"])
        for sample in targets:
            source_group = sample.groups.get("category") or sample.groups.get("task") or ""
            rows.append(
                {
                    "id": f"{reference_id}::{sample.uid}",
                    "text": sample.input_text,
                    "reference": sample.reference_text,
                    "reference_id": reference_id,
                    "reference_audio": str(Path(reference["audio_path"]).resolve()),
                    "reference_text": str(reference["reference_text"]),
                    "reference_sha256": str(reference["sha256"]),
                    "reference_source_record_id": str(reference["source_record_id"]),
                    "target_uid": sample.uid,
                    "groups": {
                        "reference_id": reference_id,
                        "target_dataset": sample.dataset,
                        "target_group": source_group,
                    },
                }
            )
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "schema_version": 1,
        "references_manifest": str(args.references.expanduser().resolve()),
        "num_references": len(references),
        "texts_per_reference": len(targets),
        "rows": len(rows),
        "target_config": str(args.target_config.expanduser().resolve()),
        "target_seed": config.seed,
        "target_selection_sha256": target_sha,
        "target_dataset_counts": {
            name: sum(sample.dataset == name for sample in targets)
            for name in sorted({sample.dataset for sample in targets})
        },
        "target_ids": target_ids,
        "dataset_reports": reports,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: metadata[key] for key in (
        "num_references", "texts_per_reference", "rows",
        "target_selection_sha256", "target_dataset_counts"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
