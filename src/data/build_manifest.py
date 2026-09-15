#!/usr/bin/env python
"""A1 / PLAN.md §6.1 — build data/manifests/all.jsonl from dataset v3.1.

Sources (all read-only; locations come from configs/dataset.yaml, never hard-coded):
  <root>/balalaika.parquet                          segment table (filepath, speaker_id, start,
                                                    end, total_duration, ..., source_root, video_id)
  <audio_dir>/<video_id>/<start>_<end>_audio_<video_id>.json   per-segment ASR sidecar
                                                    (asr{gigaam-v3-*}, asr_ts, rover, asr_consistency)
  metadata/selection_plan.parquet                   video_id -> catalog label (the only trustworthy
                                                    channel label; the parquet `channel` column is
                                                    scrambled on v3.1, see all_build_stats.json)
  metadata/catalog_<label>.json                     {id,title,published,lang,dur,cc} -> licence (cc)
  metadata/infojson/<video_id>.info.json            channel_id / channel / license / title (118 videos)

channel_id (unit of the split, PLAN §6.2) = YouTube channel:
  * a video listed in catalog_<label>.json  -> channel_id = <label>
    (every catalog label maps to exactly one YouTube channel_id where infojson exists);
  * a video with label `other` (a bag of unrelated channels) -> channel_id = "other:<youtube channel_id>"
    from its infojson; without infojson -> "other:unknown:<video_id>" (0 such videos on v3.1).

Usage:
  python src/data/build_manifest.py --out data/manifests/all.jsonl            # no audio hashing
  python src/data/build_manifest.py --out data/manifests/all.jsonl --sha256   # + sha256_audio (24 GB)
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.dataset_config import load_dataset_config  # noqa: E402

FNAME_RE = re.compile(r"^(?P<start>[0-9.]+)_(?P<end>[0-9.]+)_audio_(?P<vid>.+)\.flac$")


def norm_license(raw):
    """Map any licence string onto the YouTube Data API vocabulary.

    The API returns `creativeCommon` / `youtube`; yt-dlp's info.json returns the human
    string ("Creative Commons Attribution license (reuse allowed)") or omits the field;
    the v3 catalogs carry a boolean `cc`. All sources must land in the same `license`
    column, otherwise the licensing count silently splits into several vocabularies.
    """
    if raw is True:
        return "creativeCommon"
    if raw is False:
        return "youtube"
    s = (raw or "").strip()
    if not s:
        return "unknown"
    low = s.lower()
    if low == "creativecommon":
        return "creativeCommon"
    if low == "youtube":
        return "youtube"
    if "creative commons" in low:
        return "creativeCommon"
    if "standard youtube" in low:
        return "youtube"
    return f"other:{s}"


def load_catalogs(pattern):
    """video_id -> {label, cc, title}; also label -> n videos."""
    idx, per_label = {}, Counter()
    for p in sorted(glob.glob(pattern)):
        label = os.path.basename(p)[len("catalog_"):-len(".json")]
        with open(p, encoding="utf-8") as f:
            for e in json.load(f):
                vid = e.get("id")
                if not vid:
                    continue
                if vid in idx and idx[vid]["label"] != label:
                    raise RuntimeError(f"video {vid} listed in two catalogs: "
                                       f"{idx[vid]['label']} and {label}")
                idx[vid] = {"label": label, "cc": e.get("cc"), "title": e.get("title") or ""}
                per_label[label] += 1
    return idx, per_label


def load_infojson(dirpath):
    """video_id -> {channel_id, channel, license, title}."""
    out = {}
    for p in sorted(glob.glob(os.path.join(dirpath, "*.info.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        vid = d.get("id") or os.path.basename(p)[:-len(".info.json")]
        out[vid] = {
            "channel_id": d.get("channel_id") or d.get("uploader_id") or "",
            "channel": d.get("channel") or d.get("uploader") or "",
            "license": d.get("license"),
            "title": d.get("title") or "",
        }
    return out


def load_selection_plan(path):
    """video_id -> catalog label (must be unique per video)."""
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["video_id", "channel"]).to_pandas()
    lab = {}
    for vid, ch in zip(t["video_id"], t["channel"]):
        if vid in lab and lab[vid] != ch:
            raise RuntimeError(f"selection_plan gives two labels for {vid}: {lab[vid]} / {ch}")
        lab[vid] = ch
    return lab


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return path, h.hexdigest()


def _read_segment_json(task):
    path, rover_key, e2e_key = task
    if not os.path.exists(path):
        return path, {"__error__": "missing", "__missing__": True}
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:  # noqa: BLE001
        return path, {"__error__": f"{type(e).__name__}: {e}"}
    return path, {
        "rover": d.get(rover_key) or "",
        "text_e2e": (d.get("asr") or {}).get(e2e_key) or "",
        "asr_consistency": d.get("asr_consistency"),
        "speaker_id": d.get("speaker_id"),
        "is_single_speaker": d.get("is_single_speaker"),
        "asr_models": sorted((d.get("asr") or {}).keys()),
        "has_ts": e2e_key in (d.get("asr_ts") or {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-config", default=None, help="default configs/dataset.yaml")
    ap.add_argument("--parquet", default=None, help="override the parquet path from the config")
    ap.add_argument("--out", default="data/manifests/all.jsonl")
    ap.add_argument("--sha256", action="store_true", help="compute sha256_audio (reads the audio)")
    ap.add_argument("--sha-workers", type=int, default=4)
    ap.add_argument("--json-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    cfg = load_dataset_config(args.dataset_config)
    parquet = args.parquet or cfg["parquet"]
    meta_cfg = cfg["metadata"]
    ch_cfg = cfg.get("channel", {})
    other_label = ch_cfg.get("other_label", "other")
    other_prefix = ch_cfg.get("other_prefix", "other:")
    rover_key = cfg.get("text", {}).get("rover_key", "rover")
    e2e_key = cfg.get("text", {}).get("e2e_key", "gigaam-v3-e2e-ctc")

    t0 = time.time()
    df = pq.read_table(parquet).to_pandas()
    if args.limit:
        df = df.head(args.limit)
    print(f"[build_manifest] dataset {cfg['version']} root={cfg['root']} parquet rows={len(df)}",
          flush=True)

    plan = load_selection_plan(meta_cfg["selection_plan"])
    catalogs, cat_per_label = load_catalogs(meta_cfg["catalog_glob"])
    infojson = load_infojson(meta_cfg["infojson_dir"])
    print(f"[build_manifest] selection_plan videos={len(plan)} catalog videos={len(catalogs)} "
          f"({dict(cat_per_label)}) infojson={len(infojson)}", flush=True)

    # --- channel resolution per video ------------------------------------------------
    videos = sorted({os.path.basename(os.path.dirname(fp)) for fp in df["filepath"]})
    # youtube channel_id per catalog label (from the videos of that label that have infojson)
    label_ytid = defaultdict(Counter)
    for vid in videos:
        lab = plan.get(vid)
        if lab and lab != other_label and vid in infojson and infojson[vid]["channel_id"]:
            label_ytid[lab][(infojson[vid]["channel_id"], infojson[vid]["channel"])] += 1
    label_conflicts = {lab: dict(c) for lab, c in label_ytid.items() if len(c) > 1}
    if label_conflicts:
        raise RuntimeError(f"catalog labels mapping to several YouTube channels: {label_conflicts}")

    vmeta = {}
    stats = {
        "dataset_version": cfg["version"], "dataset_root": cfg["root"],
        "dataset_config": cfg["config_path"], "parquet": parquet,
        "n_rows": len(df), "null_meta_rows": 0,
        "json_missing": 0, "json_error": 0, "empty_rover": 0, "audio_missing": 0,
        "rows_without_ts": 0,
    }
    lic_conflicts = []
    for vid in videos:
        lab = plan.get(vid)
        cat = catalogs.get(vid)
        inf = infojson.get(vid)
        if lab is None and cat is not None:
            lab = cat["label"]
        if cat is not None and lab != cat["label"]:
            raise RuntimeError(f"{vid}: selection_plan label {lab} != catalog {cat['label']}")
        if lab is None:
            channel_id, channel_title, channel_source = f"{other_prefix}unknown:{vid}", "", "unknown"
            label = None
        elif lab == other_label:
            label = lab
            if inf and inf["channel_id"]:
                channel_id = f"{other_prefix}{inf['channel_id']}"
                channel_title, channel_source = inf["channel"], "infojson"
            else:
                channel_id, channel_title, channel_source = f"{other_prefix}unknown:{vid}", "", "unknown"
        else:
            label = lab
            channel_id = lab
            yt = next(iter(label_ytid[lab])) if label_ytid.get(lab) else None
            channel_title = yt[1] if yt else lab
            channel_source = "catalog" if cat is not None else "selection_plan"
        # licence
        lic_cat = norm_license(cat["cc"]) if cat is not None and cat.get("cc") is not None else None
        lic_inf = norm_license(inf["license"]) if inf and inf.get("license") else None
        if lic_cat and lic_inf:
            if lic_cat != lic_inf:
                lic_conflicts.append({"video_id": vid, "catalog": lic_cat, "infojson": lic_inf})
            lic, lic_src = lic_cat, "catalog+infojson"
            lic_raw = f"catalog cc={cat['cc']}; infojson {inf['license']}"
        elif lic_cat:
            lic, lic_src, lic_raw = lic_cat, "catalog", f"catalog cc={cat['cc']}"
        elif lic_inf:
            lic, lic_src, lic_raw = lic_inf, "infojson", inf["license"]
        else:
            lic, lic_src, lic_raw = "unknown", "none", ""
        yt_id = (inf["channel_id"] if inf and inf["channel_id"] else
                 (next(iter(label_ytid[lab]))[0] if lab and label_ytid.get(lab) else None))
        vmeta[vid] = {
            "channel_id": channel_id, "channel_title": channel_title,
            "channel_source": channel_source, "channel_label": label,
            "youtube_channel_id": yt_id,
            "video_title": (inf["title"] if inf and inf["title"] else (cat["title"] if cat else "")),
            "license": lic, "license_source": lic_src, "license_raw": lic_raw,
        }

    rows = []
    podcast_ids = set()
    rows_null_podcast_id = 0
    videos_null_podcast_id = set()
    parquet_channel_agree = 0
    for r in df.itertuples(index=False):
        fp = r.filepath
        base = os.path.basename(fp)
        m = FNAME_RE.match(base)
        if m is None:
            raise RuntimeError(f"unexpected filename: {fp}")
        vid_dir = os.path.basename(os.path.dirname(fp))
        vid = m.group("vid")
        assert vid == vid_dir, (vid, vid_dir)
        if isinstance(r.video_id, str) and r.video_id != vid:
            raise RuntimeError(f"parquet video_id {r.video_id} != path video_id {vid}")
        start_s, end_s = m.group("start"), m.group("end")

        pid = r.podcast_id
        if pid is None or (isinstance(pid, float)):
            stats["null_meta_rows"] += 1
            rows_null_podcast_id += 1
            videos_null_podcast_id.add(vid)
        else:
            podcast_ids.add(pid)
        start = float(r.start) if r.start == r.start else float(start_s)
        end = float(r.end) if r.end == r.end else float(end_s)
        spk = r.speaker_id
        local_spk = int(spk) if spk == spk and spk is not None else -1
        iss = r.is_single_speaker
        is_single = bool(iss) if iss is not None else None
        meta = vmeta[vid]
        pq_channel = r.channel if isinstance(r.channel, str) else None
        if pq_channel is not None and (pq_channel == meta["channel_label"]):
            parquet_channel_agree += 1

        def _f(x):
            return float(x) if x == x and x is not None else None

        rows.append({
            "sample_id": f"{vid}_{start_s}_{end_s}",
            "video_id": vid,
            "channel_id": meta["channel_id"],
            "channel_title": meta["channel_title"],
            "channel_source": meta["channel_source"],
            "channel_label": meta["channel_label"],
            "youtube_channel_id": meta["youtube_channel_id"],
            "parquet_channel": pq_channel,
            "video_title": meta["video_title"],
            "local_speaker_id": local_spk,
            "speaker_key": f"{meta['channel_id']}/{vid}/{local_spk}",
            "audio_path": fp,
            "json_path": fp[:-5] + ".json",
            "source_root": r.source_root if isinstance(r.source_root, str) else None,
            "start": start,
            "end": end,
            "duration_sec": float(r.total_duration),
            "silence_percent": _f(r.silence_percent),
            "max_silence_duration": _f(r.max_silence_duration),
            "is_single_speaker": is_single,
            "crest_factor": _f(r.crest_factor),
            "loudness_normalized": bool(r.loudness_normalized) if r.loudness_normalized is not None else None,
            "playlist_id": r.playlist_id if isinstance(r.playlist_id, str) else None,
            "text": None, "text_e2e": None, "asr_consistency": None,
            "words": 0, "chars": 0,
            "license": meta["license"],
            "license_source": meta["license_source"],
            "license_raw": meta["license_raw"],
            "split": "unassigned",
            "sha256_audio": None,
            "sha256_text": None,
        })

    # --- per-segment json (rover / e2e / consistency) --------------------------------
    t1 = time.time()
    tasks = [(x["json_path"], rover_key, e2e_key) for x in rows]
    jmap = {}
    with ThreadPoolExecutor(max_workers=args.json_workers) as ex:
        for i, (p, d) in enumerate(ex.map(_read_segment_json, tasks, chunksize=16)):
            jmap[p] = d
            if (i + 1) % 1000 == 0:
                print(f"[build_manifest] json {i+1}/{len(tasks)}", flush=True)
    t_json = time.time() - t1
    print(f"[build_manifest] json read: {t_json:.1f}s", flush=True)

    asr_models = Counter()
    for x in rows:
        d = jmap[x["json_path"]]
        if "__error__" in d:
            if d.get("__missing__"):
                stats["json_missing"] += 1
            else:
                stats["json_error"] += 1
            continue
        text = (d["rover"] or "").strip()
        x["text"] = text
        x["text_e2e"] = (d["text_e2e"] or "").strip()
        x["asr_consistency"] = d["asr_consistency"]
        x["words"] = len(text.split())
        x["chars"] = len(text)
        x["sha256_text"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        asr_models[",".join(d["asr_models"])] += 1
        if not d["has_ts"]:
            stats["rows_without_ts"] += 1
        if not text:
            stats["empty_rover"] += 1
        if x["local_speaker_id"] == -1 and d.get("speaker_id") is not None:
            x["local_speaker_id"] = int(d["speaker_id"])
            x["speaker_key"] = f"{x['channel_id']}/{x['video_id']}/{x['local_speaker_id']}"
        if x["is_single_speaker"] is None and d.get("is_single_speaker") is not None:
            x["is_single_speaker"] = bool(d["is_single_speaker"])

    for x in rows:
        if not os.path.exists(x["audio_path"]):
            stats["audio_missing"] += 1

    # --- sha256 of audio -------------------------------------------------------------
    t_sha = 0.0
    reused_sha = False
    if args.sha256:
        t2 = time.time()
        paths = [x["audio_path"] for x in rows]
        got = {}
        with ProcessPoolExecutor(max_workers=args.sha_workers) as ex:
            for i, (p, h) in enumerate(ex.map(sha256_file, paths, chunksize=4)):
                got[p] = h
                if (i + 1) % 500 == 0:
                    el = time.time() - t2
                    print(f"[build_manifest] sha256 {i+1}/{len(paths)} elapsed={el:.0f}s", flush=True)
        for x in rows:
            x["sha256_audio"] = got.get(x["audio_path"])
        t_sha = time.time() - t2
        print(f"[build_manifest] sha256 elapsed: {t_sha:.1f}s", flush=True)
    else:
        # keep previously computed hashes if the output already exists AND describes the
        # same audio files (a v1 manifest must never lend its hashes to v3.1 rows: the key
        # is (sample_id, audio_path))
        if os.path.exists(args.out):
            prev = {}
            with open(args.out, encoding="utf-8") as f:
                for line in f:
                    o = json.loads(line)
                    if o.get("sha256_audio"):
                        prev[(o["sample_id"], o.get("audio_path"))] = o["sha256_audio"]
            n = 0
            for x in rows:
                h = prev.get((x["sample_id"], x["audio_path"]))
                if h:
                    x["sha256_audio"] = h
                    n += 1
            n_new = len(rows) - n
            print(f"[build_manifest] reused {n} sha256_audio from existing {args.out}"
                  + (f"; {n_new} NEW rows have no hash — rerun with --sha256" if n_new else ""),
                  flush=True)
            reused_sha = True

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for x in sorted(rows, key=lambda z: z["sample_id"]):
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out)

    stats["video_id_definition"] = ("YouTube id in the audio path: the <video_id> directory, "
                                    "identical to the `_audio_<video_id>.flac` suffix and to the "
                                    "parquet `video_id`; never null. This is the ONLY video count "
                                    "used in reports.")
    stats["videos"] = len({x["video_id"] for x in rows})
    stats["videos_by_filepath"] = stats["videos"]
    stats["videos_by_podcast_id_nonnull"] = len(podcast_ids)
    stats["rows_with_null_podcast_id"] = rows_null_podcast_id
    stats["videos_with_null_podcast_id"] = sorted(videos_null_podcast_id)
    stats["channels"] = len({x["channel_id"] for x in rows})
    stats["channel_labels"] = dict(sorted(Counter(x["channel_label"] or "None" for x in rows).items()))
    stats["channels_per_label"] = {
        lab: len({x["channel_id"] for x in rows if (x["channel_label"] or "None") == lab})
        for lab in stats["channel_labels"]}
    stats["hours"] = round(sum(x["duration_sec"] for x in rows) / 3600.0, 3)
    stats["channel_id_definition"] = (
        "YouTube channel. Catalog videos: channel_id = catalog label (selection_plan.parquet == "
        "catalog_<label>.json; each label maps to exactly one YouTube channel_id where infojson "
        f"exists). Videos labelled '{other_label}': channel_id = '{other_prefix}<youtube "
        "channel_id>' from infojson. The parquet column `channel` is recorded as "
        "`parquet_channel` only.")
    stats["parquet_channel_agrees_with_label_rows"] = parquet_channel_agree
    stats["parquet_channel_agree_frac"] = round(parquet_channel_agree / max(1, len(rows)), 4)
    vid_pq = defaultdict(set)
    for x in rows:
        if x["parquet_channel"] is not None:
            vid_pq[x["video_id"]].add(x["parquet_channel"])
    stats["videos_with_several_parquet_channel_values"] = sum(1 for s in vid_pq.values() if len(s) > 1)
    vid_src = {x["video_id"]: x["channel_source"] for x in rows}
    stats["videos_channel_from_catalog"] = sum(1 for v in vid_src.values() if v == "catalog")
    stats["videos_channel_from_selection_plan"] = sum(1 for v in vid_src.values() if v == "selection_plan")
    stats["videos_channel_from_infojson"] = sum(1 for v in vid_src.values() if v == "infojson")
    stats["videos_channel_unknown"] = sum(1 for v in vid_src.values() if v == "unknown")
    stats["unknown_channel_videos"] = sorted({x["video_id"] for x in rows if x["channel_source"] == "unknown"})
    stats["videos_with_infojson"] = sum(1 for v in videos if v in infojson)
    stats["videos_in_catalogs"] = sum(1 for v in videos if v in catalogs)
    stats["label_to_youtube_channel"] = {lab: list(next(iter(c))) for lab, c in sorted(label_ytid.items())}
    stats["license_counts"] = dict(sorted(Counter(x["license"] for x in rows).items()))
    stats["license_source_counts"] = dict(sorted(Counter(x["license_source"] for x in rows).items()))
    stats["license_conflicts"] = lic_conflicts
    stats["asr_model_sets"] = dict(asr_models)
    stats["audio_bytes"] = sum(os.path.getsize(x["audio_path"]) for x in rows
                               if os.path.exists(x["audio_path"]))
    stats["json_read_sec"] = round(t_json, 1)
    side = os.path.join(os.path.dirname(os.path.abspath(args.out)), "all_build_stats.json")
    if args.sha256 and t_sha > 0:
        stats["sha256_sec"] = round(t_sha, 1)
        stats["sha256_MB_per_s"] = round(stats["audio_bytes"] / 1e6 / t_sha, 1)
        stats["sha256_workers"] = args.sha_workers
        stats["sha256_measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        stats["sha256_reused"] = False
    elif reused_sha and os.path.exists(side):
        try:
            with open(side, encoding="utf-8") as f:
                old = json.load(f)
            if old.get("dataset_root") == cfg["root"]:
                for k in ("sha256_sec", "sha256_MB_per_s", "sha256_workers", "sha256_measured_at"):
                    if k in old:
                        stats[k] = old[k]
        except Exception:  # noqa: BLE001
            pass
        stats["sha256_reused"] = True
    stats["n_rows_without_sha256"] = sum(1 for x in rows if not x["sha256_audio"])
    stats["total_sec"] = round(time.time() - t0, 1)
    with open(side, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("unknown_channel_videos", "license_conflicts")},
                     ensure_ascii=False, indent=2))
    if lic_conflicts:
        print(f"[build_manifest] WARNING: {len(lic_conflicts)} licence conflicts catalog vs infojson",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
