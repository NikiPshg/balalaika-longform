#!/usr/bin/env python3
"""A1-ext / E6 — CosyVoice-style data dir + manifest for the `ext_short` corpus.

Input : data/train/ext_short/selection.jsonl (src/data/select_ext_short.py + ext_voice_gate.py)
Output: data/train/ext_short/{wav.scp,text,utt2spk,spk2utt,instruct,utt2offset,utt2dur,
                              windows_all,stats.json,manifest.jsonl}

Every clip is its own parent with a single window [0, dur], so utt2offset/windows_all are
degenerate by construction — they exist because src/training/make_parquet_arms.py and the
E3/E2 arms use exactly this layout, and make_parquet_ext.py reads them unchanged.

The kaldi files are written by src.training.prepare_arms.write_dir (imported, not copied), so
the schema of stats.json is identical to data/train/short/stats.json.

manifest.jsonl mirrors data/manifests/train_short.jsonl for the fields this corpus really has.
Fields that cannot be filled honestly (channel_id/channel_title/license — the incoming YouTube
dump carries no channel metadata) are OMITTED, never invented.

Usage:
    python src/training/prepare_ext_short.py
"""
import argparse
import hashlib
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from src.training.prepare_arms import write_dir                    # noqa: E402
from src.training.context_budget import INSTRUCT, check_manifest    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--des", default="data/train/ext_short")
    ap.add_argument("--selection", default=None)
    args = ap.parse_args()
    sel_path = args.selection or os.path.join(args.des, "selection.jsonl")
    sel = [json.loads(l) for l in open(sel_path, encoding="utf-8")]
    assert sel, sel_path
    utts = [r["utt"] for r in sel]
    assert len(set(utts)) == len(utts), "duplicate utt in selection.jsonl"
    for r in sel:
        assert " " not in r["utt"] and " " not in r["spk"], r["utt"]
        assert "\n" not in r["text"] and "\t" not in r["text"], r["utt"]

    manifest = []
    for r in sel:
        text = r["text"]
        manifest.append({
            "sample_id": r["utt"],
            "parent_sample_id": r["utt"],
            "window_index": 0,
            "video_id": r["video_id"],
            "local_speaker_id": r["speaker_id"],
            "speaker_key": r["spk"],
            "audio_path": r["path"],
            "offset_start": 0.0,
            "offset_end": float(r["duration"]),
            "duration": float(r["duration"]),
            "text": text,
            "words": len(text.split()),
            "chars": len(text),
            "asr_consistency": r["asr_consistency_percent"],
            "is_single_speaker": True,
            "split": "train_ext",
            "sha256_text": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "parent_duration_sec": float(r["duration"]),
            "clip_start_in_video_sec": r["start"],
            "clip_end_in_video_sec": r["end"],
            "DistillMOS": r["DistillMOS"],
            "music_prob": r["music_prob"],
            "source": "youtube_data_incoming/balalaika.parquet",
        })

    budgets = {b["sample_id"]: b for b in check_manifest(manifest, "duration")}
    bad = [b for b in budgets.values() if not b["fits"] or b["empty_text"]]
    assert not bad, f"{len(bad)} rows fail the context budget / have empty text: {bad[:3]}"

    rows = [{"utt": r["utt"], "wav": r["path"], "text": r["text"], "spk": r["spk"],
             "parent": r["utt"], "off0": 0.0, "off1": float(r["duration"]),
             "dur": float(r["duration"]),
             "n_speech_est": budgets[r["utt"]]["n_speech"],
             "n_text": budgets[r["utt"]]["n_text"]} for r in sel]

    os.makedirs(args.des, exist_ok=True)
    stats = write_dir(args.des, rows, "short")
    stats["excluded"] = 0
    stats["excluded_hours"] = 0.0
    stats["arm"] = "ext_short"
    stats["instruct"] = INSTRUCT
    with open(os.path.join(args.des, "windows_all"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write("{} {} {:.3f} {:.3f} 1\n".format(r["utt"], r["utt"], 0.0, r["dur"]))
    with open(os.path.join(args.des, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(os.path.join(args.des, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
