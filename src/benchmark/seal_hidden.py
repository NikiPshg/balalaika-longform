#!/usr/bin/env python
"""A2 — seal the hidden benchmark (PLAN.md §3.5, §7.4; composition §7.6).

The hidden set is written once and then frozen.  Sealing means:

  1. sha256 of ``data/benchmark/hidden.jsonl`` as it lies on disk;
  2. sha256 of every human audio slice the manifest points at — both the file
     on disk (recomputed here and checked against the ``human_audio_sha256``
     that A1's manifest recorded) and the *slice reference* itself, i.e. the
     canonical string ``<path>|<offset_start>|<offset_end>``, so that a later
     change of an offset is detectable even when the audio file is untouched;
  3. per-text ``sha256_text_tts``, carried over from the manifest;
  4. a seal file ``data/benchmark/HIDDEN_SEAL.json`` with the four fields the
     Lead asked for (``sha256``, ``n_records``, ``built_utc``, ``rule_version``)
     plus the evidence above;
  5. mode 444 on the manifest and on the seal file.

The seal is verifiable without rebuilding anything:

    python src/benchmark/seal_hidden.py --verify

which recomputes every hash and exits non-zero on the first mismatch.  It reads
nothing but the manifest, the seal and the human audio; it never generates and
never evaluates.

CPU only.  No GPU, no model, no writes outside the work tree.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "data" / "benchmark" / "hidden.jsonl"
SEAL = ROOT / "data" / "benchmark" / "HIDDEN_SEAL.json"
CONFIG = ROOT / "configs" / "benchmark_hidden.yaml"
DECISIONS = ROOT / "reports" / "decisions.md"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slice_reference(rec: dict[str, Any]) -> str:
    """Canonical, hashable form of one human audio slice reference.

    The offsets are formatted with three decimals, exactly as the builder
    rounds them into the manifest, so the string is reproducible from the
    manifest alone.
    """
    return (
        f"{rec['human_audio_path']}"
        f"|{rec['human_offset_start']:.3f}"
        f"|{rec['human_offset_end']:.3f}"
    )


def read_records() -> list[dict[str, Any]]:
    with open(MANIFEST, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rule_version() -> str:
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return str(cfg["rule_version"])


def collect(records: list[dict[str, Any]], hash_audio: bool) -> dict[str, Any]:
    """Everything the seal records about the human audio and the texts."""
    slices: list[dict[str, Any]] = []
    files: dict[str, dict[str, Any]] = {}
    for rec in records:
        if rec["source"] != "dataset":
            continue
        path = rec["human_audio_path"]
        if path not in files:
            entry: dict[str, Any] = {
                "path": path,
                "exists": os.path.exists(path),
                "bytes": os.path.getsize(path) if os.path.exists(path) else None,
                "manifest_sha256": rec["human_audio_sha256"],
            }
            if hash_audio and entry["exists"]:
                entry["file_sha256"] = sha256_file(path)
                entry["matches_manifest"] = entry["file_sha256"] == rec["human_audio_sha256"]
            files[path] = entry
        ref = slice_reference(rec)
        slices.append(
            {
                "text_id": rec["text_id"],
                "human_audio_path": path,
                "human_offset_start": rec["human_offset_start"],
                "human_offset_end": rec["human_offset_end"],
                "human_duration_sec": rec["human_duration_sec"],
                "slice_reference": ref,
                "slice_reference_sha256": sha256_text(ref),
                "human_audio_sha256": rec["human_audio_sha256"],
            }
        )
    return {
        "audio_files": [files[k] for k in sorted(files)],
        "audio_slices": slices,
        "text_sha256": {r["text_id"]: r["sha256_text_tts"] for r in records},
    }


def build_seal(hash_audio: bool = True) -> dict[str, Any]:
    records = read_records()
    body = collect(records, hash_audio)
    return {
        "sha256": sha256_file(MANIFEST),
        "n_records": len(records),
        "built_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule_version": rule_version(),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "config": str(CONFIG.relative_to(ROOT)),
        "n_roots": len({r["root_id"] for r in records}),
        "n_dataset_roots": len({r["root_id"] for r in records if r["source"] == "dataset"}),
        "n_external_roots": len({r["root_id"] for r in records if r["source"] == "external"}),
        "external_documents": {
            str(Path(p).relative_to(ROOT)): sha256_file(p)
            for p in sorted(
                (ROOT / "data" / "benchmark" / "external" / "hidden").glob("*.txt")
            )
        },
        "seal_code_sha256": sha256_file(Path(__file__)),
        "builder_sha256": sha256_file(ROOT / "src" / "benchmark" / "build_pilot.py"),
        "config_sha256": sha256_file(CONFIG),
        **body,
    }


def chmod_444(path: Path) -> None:
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def do_seal(append_decision: bool) -> int:
    seal = build_seal(hash_audio=True)
    bad = [f["path"] for f in seal["audio_files"] if not f.get("matches_manifest", True)]
    if bad:
        print("REFUSING TO SEAL: human audio does not match the manifest sha256:", file=sys.stderr)
        for p in bad:
            print("   ", p, file=sys.stderr)
        return 2
    if SEAL.exists():
        os.chmod(SEAL, 0o644)
    SEAL.write_text(json.dumps(seal, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    chmod_444(MANIFEST)
    chmod_444(SEAL)
    print(f"SEALED {seal['manifest']}")
    print(f"  sha256       {seal['sha256']}")
    print(f"  n_records    {seal['n_records']} ({seal['n_roots']} roots)")
    print(f"  built_utc    {seal['built_utc']}")
    print(f"  rule_version {seal['rule_version']}")
    print(f"  audio files  {len(seal['audio_files'])} (all match the manifest sha256)")
    print(f"  audio slices {len(seal['audio_slices'])}")
    if append_decision:
        line = (
            f"- {seal['built_utc'][:10]} — A2: **SEALED** hidden benchmark "
            f"`{seal['manifest']}` — sha256 `{seal['sha256']}`, "
            f"{seal['n_records']} records ({seal['n_roots']} roots = "
            f"{seal['n_dataset_roots']} dataset + {seal['n_external_roots']} external, "
            f"5 buckets), built_utc {seal['built_utc']}, rule_version "
            f"`{seal['rule_version']}`; seal `data/benchmark/HIDDEN_SEAL.json`, both files "
            f"mode 444. Verify with `python src/benchmark/seal_hidden.py --verify`. — A2.\n"
        )
        with open(DECISIONS, "at", encoding="utf-8") as f:
            f.write(line)
        print(f"  appended SEALED line to {DECISIONS.relative_to(ROOT)}")
    return 0


def do_verify(skip_audio: bool) -> int:
    if not SEAL.exists():
        print(f"no seal at {SEAL}", file=sys.stderr)
        return 1
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    fails: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    now = sha256_file(MANIFEST)
    chk("hidden.jsonl sha256 matches the seal", now == seal["sha256"], now)
    records = read_records()
    chk("n_records matches the seal", len(records) == seal["n_records"], str(len(records)))
    chk("rule_version matches the config", rule_version() == seal["rule_version"])
    chk("manifest is read-only (mode 444)",
        stat.S_IMODE(os.stat(MANIFEST).st_mode) == 0o444,
        oct(stat.S_IMODE(os.stat(MANIFEST).st_mode)))
    chk("seal file is read-only (mode 444)",
        stat.S_IMODE(os.stat(SEAL).st_mode) == 0o444,
        oct(stat.S_IMODE(os.stat(SEAL).st_mode)))
    chk("per-text sha256_text_tts unchanged",
        all(seal["text_sha256"].get(r["text_id"]) == r["sha256_text_tts"] for r in records))
    live = collect(records, hash_audio=False)
    chk("audio slice references unchanged",
        [s["slice_reference_sha256"] for s in live["audio_slices"]]
        == [s["slice_reference_sha256"] for s in seal["audio_slices"]])
    chk("every human audio file still exists",
        all(os.path.exists(f["path"]) for f in seal["audio_files"]))
    if skip_audio:
        print("SKIP audio file sha256 (--skip-audio)")
    else:
        ok = True
        for f in seal["audio_files"]:
            if not os.path.exists(f["path"]):
                ok = False
                continue
            ok = ok and sha256_file(f["path"]) == f["file_sha256"]
        chk("human audio file sha256 unchanged", ok)
    chk("external documents unchanged",
        all(sha256_file(ROOT / p) == h for p, h in seal["external_documents"].items()))
    print("\nSEAL VERIFIED" if not fails else f"\nSEAL BROKEN: {fails}")
    return 1 if fails else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true", help="verify an existing seal, write nothing")
    ap.add_argument("--skip-audio", action="store_true",
                    help="--verify only: do not re-hash the 150 MB of human audio")
    ap.add_argument("--no-decision", action="store_true",
                    help="do not append the SEALED line to reports/decisions.md")
    a = ap.parse_args(argv)
    if a.verify:
        return do_verify(a.skip_audio)
    return do_seal(append_decision=not a.no_decision)


if __name__ == "__main__":
    raise SystemExit(main())
