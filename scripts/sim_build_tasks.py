"""Build results/v31_sim/tasks.jsonl for the A18 speaker-sim window scoring.

Same nine TTS arms as the A25-updated A12 MOS run (scripts/mos_build_tasks.py:
six CosyVoice3/Qwen3-TTS arms + the three VoxCPM2 arms added by A25)
plus the human anchor slices; every task carries the MASTER reference wav the
window similarities are computed against (speaker_drift.py convention: master,
not the stored 16 kHz copy — both sides of a cosine must go through the same
soxr resampler).

Reference resolution
--------------------
* pilot TTS items: voice_id in {ref_female_01, ref_male_01} ->
  data/references/references.jsonl `wav_path`.
* robust TTS items: per-item voices ref_r01..ref_r13 ->
  data/references/references_robust.jsonl `wav_path` (exactly what
  speaker_drift --references uses for the robust arms).
* robust human slices: the item's own voice (benchmark row `voice_id`) — the
  SAME speaker as the recording, so the human curve is a same-speaker ceiling.
* pilot human slices: the pilot benchmark has NO voice_id (items are generated
  for both pilot voices) and the source speaker is a different person from
  both reference voices.  Each of the 4 dataset roots (the 2 external roots
  have no human audio — skipped, as in A12) is scored ONCE PER PILOT VOICE
  (mirroring the item grid), giving a CROSS-SPEAKER FLOOR, not a ceiling —
  documented in the figure and whole_clip_sim.md.

Usage: .venv-eval/bin/python scripts/sim_build_tasks.py
Writes results/v31_sim/tasks.jsonl and prints a composition summary.
"""
import json
import sys
from collections import Counter
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
PILOT_VOICES = ["ref_female_01", "ref_male_01"]

# Same SSD -> HDD-mirror fallback as scripts/mos_build_tasks.py (the pre-VoxCPM
# output trees were evacuated to the mirror when the SSD ran short).
HDD_MIRROR = Path("external/rulongtts_models_backup")


def resolve_output_dir(rel: str) -> Path:
    ssd = ROOT / rel
    if any(ssd.glob("*.wav")):
        return ssd
    hdd = HDD_MIRROR / rel
    if any(hdd.glob("*.wav")):
        return hdd
    sys.exit(f"FATAL {rel}: no wavs on SSD ({ssd}) nor HDD mirror ({hdd})")


def load_refs(path: Path) -> dict:
    refs = {}
    for line in path.open():
        r = json.loads(line)
        refs[r["voice_id"]] = str(ROOT / r["wav_path"])
    return refs


def bucket_of(text_id: str) -> str:
    b = text_id.split("__")[-1]
    assert b.startswith("B"), text_id
    return b


def main() -> None:
    refs = load_refs(ROOT / "data/references/references.jsonl")
    refs.update(load_refs(ROOT / "data/references/references_robust.jsonl"))
    tasks, skipped_human = [], []
    for set_name, system, rel in SYSTEMS:
        wavs = sorted(resolve_output_dir(rel).glob("*.wav"))
        if len(wavs) != EXPECTED[set_name]:
            sys.exit(f"FATAL {rel}: {len(wavs)} wavs, expected {EXPECTED[set_name]}")
        for wav in wavs:
            meta = json.loads(wav.with_suffix(".json").read_text())
            tasks.append({
                "set": set_name,
                "system": system,
                "text_id": meta["text_id"],
                "voice_id": meta["voice_id"],
                "bucket": bucket_of(meta["text_id"]),
                "wav": str(wav),
                "offset_start": None,
                "offset_end": None,
                "ref_wav": refs[meta["voice_id"]],
            })
    for set_name, bench in (("pilot", "pilot.jsonl"), ("robust", "robust.jsonl")):
        for line in (ROOT / "data/benchmark" / bench).open():
            row = json.loads(line)
            path = row.get("human_audio_path")
            if not path or not Path(path).exists():
                skipped_human.append((set_name, row["text_id"], row.get("source")))
                continue
            voices = PILOT_VOICES if set_name == "pilot" else [row["voice_id"]]
            for vid in voices:
                tasks.append({
                    "set": set_name,
                    "system": "human",
                    "text_id": row["text_id"],
                    "voice_id": vid,
                    "bucket": row["bucket"],
                    "wav": path,
                    "offset_start": row["human_offset_start"],
                    "offset_end": row["human_offset_end"],
                    "ref_wav": refs[vid],
                })
    out = ROOT / "results/v31_sim/tasks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    c = Counter((t["set"], t["system"]) for t in tasks)
    for k in sorted(c):
        print(k, c[k])
    print(f"total tasks: {len(tasks)} -> {out}")
    print(f"skipped human (no human audio, external roots): {len(skipped_human)}")
    for s in skipped_human:
        print("  skip", s)


if __name__ == "__main__":
    main()
