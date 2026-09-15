"""Build the tasks jsonl for the A12-mos DistillMOS window scoring run.

Systems (outputs assigned by the owner, reviewer-response wave):
  pilot : E1  outputs/v31_base/E1_native
          E3  outputs/v31_sft/E3_epoch_3_step_3001
          E7  outputs/v31_punct/E7_punct_epoch_3_step_3001
          QE1/QE2P/QE3P outputs/v31_qwen/<arm>_pilot (60 items each)
          VCE1/VCE2P/VCE3P outputs/v31_voxcpm/<arm>_pilot (60 items each; A25)
  robust: E1/E3/E7 outputs/v31_robust/{E1,E3,E7}
          QE1/QE2P/QE3P outputs/v31_qwen/<arm>_robust (53 items each)
          VCE1/VCE2P/VCE3P outputs/v31_voxcpm/<arm>_robust (53 items each; A25)
  human : slices of the source recordings, from data/benchmark/pilot.jsonl and
          robust.jsonl (human_audio_path + human_offset_start/end, sliced on
          the fly).  Of the 6 pilot roots only the 4 dataset roots carry human
          audio; the 2 external roots (ex5_genre5_numbers, ex6_genre6_stress)
          have none and are SKIPPED (documented) -> 20 pilot + 53 robust
          human items.

Metadata comes from the per-run sidecar json (text_id, voice_id); bucket is
the trailing __B<k> of text_id.

Usage: python scripts/mos_build_tasks.py
Writes results/v31_mos/tasks.jsonl and prints a composition summary.
"""
import json
import sys
from pathlib import Path

ROOT = Path(".")

SYSTEMS = [
    ("pilot", "E1", "outputs/v31_base/E1_native"),
    ("pilot", "E3", "outputs/v31_sft/E3_epoch_3_step_3001"),
    ("pilot", "E7", "outputs/v31_punct/E7_punct_epoch_3_step_3001"),
    ("pilot", "QE1", "outputs/v31_qwen/QE1_pilot"),
    ("pilot", "QE2P", "outputs/v31_qwen/QE2P_pilot"),
    ("pilot", "QE3P", "outputs/v31_qwen/QE3P_pilot"),
    ("pilot", "VCE1", "outputs/v31_voxcpm/VCE1_pilot"),
    ("pilot", "VCE2P", "outputs/v31_voxcpm/VCE2P_pilot"),
    ("pilot", "VCE3P", "outputs/v31_voxcpm/VCE3P_pilot"),
    ("robust", "E1", "outputs/v31_robust/E1"),
    ("robust", "E3", "outputs/v31_robust/E3"),
    ("robust", "E7", "outputs/v31_robust/E7"),
    ("robust", "QE1", "outputs/v31_qwen/QE1_robust"),
    ("robust", "QE2P", "outputs/v31_qwen/QE2P_robust"),
    ("robust", "QE3P", "outputs/v31_qwen/QE3P_robust"),
    ("robust", "VCE1", "outputs/v31_voxcpm/VCE1_robust"),
    ("robust", "VCE2P", "outputs/v31_voxcpm/VCE2P_robust"),
    ("robust", "VCE3P", "outputs/v31_voxcpm/VCE3P_robust"),
]

EXPECTED = {"pilot": 60, "robust": 53}

# 2026-09-02+ the SSD ran short and the pre-VoxCPM output trees were evacuated
# to the HDD mirror (reports/decisions.md, disk policy).  Audio is byte-identical
# there; resolve each system's directory on the SSD first, then on the mirror.
HDD_MIRROR = Path("external/rulongtts_models_backup")


def resolve_output_dir(rel: str) -> Path:
    """SSD path if it holds wavs, else the read-only HDD mirror copy."""
    ssd = ROOT / rel
    if any(ssd.glob("*.wav")):
        return ssd
    hdd = HDD_MIRROR / rel
    if any(hdd.glob("*.wav")):
        return hdd
    sys.exit(f"FATAL {rel}: no wavs on SSD ({ssd}) nor HDD mirror ({hdd})")


def bucket_of(text_id: str) -> str:
    b = text_id.split("__")[-1]
    assert b.startswith("B"), text_id
    return b


def main() -> None:
    tasks = []
    skipped_human = []
    for set_name, system, rel in SYSTEMS:
        d = resolve_output_dir(rel)
        wavs = sorted(d.glob("*.wav"))
        if len(wavs) != EXPECTED[set_name]:
            sys.exit(f"FATAL {rel}: {len(wavs)} wavs, expected {EXPECTED[set_name]}")
        for wav in wavs:
            side = wav.with_suffix(".json")
            meta = json.loads(side.read_text())
            tasks.append({
                "set": set_name,
                "system": system,
                "text_id": meta["text_id"],
                "voice_id": meta["voice_id"],
                "bucket": bucket_of(meta["text_id"]),
                "wav": str(wav),
                "offset_start": None,
                "offset_end": None,
            })
    for set_name, bench in (("pilot", "pilot.jsonl"), ("robust", "robust.jsonl")):
        for line in (ROOT / "data/benchmark" / bench).open():
            row = json.loads(line)
            path = row.get("human_audio_path")
            if not path or not Path(path).exists():
                skipped_human.append((set_name, row["text_id"], row.get("source")))
                continue
            tasks.append({
                "set": set_name,
                "system": "human",
                "text_id": row["text_id"],
                "voice_id": "human",
                "bucket": row["bucket"],
                "wav": path,
                "offset_start": row["human_offset_start"],
                "offset_end": row["human_offset_end"],
            })
    out = ROOT / "results/v31_mos/tasks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter((t["set"], t["system"]) for t in tasks)
    for k in sorted(c):
        print(k, c[k])
    print(f"total tasks: {len(tasks)} -> {out}")
    print(f"skipped human (no human audio, external roots): {len(skipped_human)}")
    for s in skipped_human:
        print("  skip", s)


if __name__ == "__main__":
    main()
