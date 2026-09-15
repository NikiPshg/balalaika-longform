#!/usr/bin/env python
"""A1 / PLAN.md §6.2 — cross-split leakage: text (exact + 8-word shingles) and voice
(campplus speaker embeddings).

Voice embeddings reproduce CosyVoiceFrontEnd._extract_spk_embedding exactly:
  torchaudio.load -> mean over channels -> resample to 16 kHz
  -> kaldi.fbank(num_mel_bins=80, dither=0, sample_frequency=16000)
  -> feat -= feat.mean(dim=0)  -> campplus.onnx (CPUExecutionProvider)

Stages:
  --embed   compute/refresh per-segment embeddings cache (data/manifests/spk_emb.npz).
            The cache is a CACHE, not an archive: entries whose sample_id is no longer
            selected from the current manifest are EVICTED, so a shrinking corpus can
            never leave phantom speakers behind (owner filters bad audio -> parquet
            changes -> some sample_ids disappear).
  --report  text + voice leakage against the current split -> reports/leakage_report.md

Frozen failure contract (Lead, 2026-08-28): `--report` exits non-zero when text leakage
exceeds the frozen thresholds
    exact sha256_text collisions > MAX_EXACT_COLLISIONS (= 0)
    max 8-word-shingle Jaccard   > MAX_JACCARD           (= 0.05)
on train<->dev/test OR dev<->test. The report is written first, then the process exits 2,
so `bash scripts/build_dataset.sh` aborts with evidence on disk. Voice leakage is NOT an
exit code here: it is handled by the remediation loop in scripts/build_dataset.sh, which
fails loudly itself if the loop does not converge.

Usage:
  python src/data/leakage_check.py --embed
  python src/data/leakage_check.py --report
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CAMPPLUS = "models/cosyvoice3/campplus.onnx"
EMB_CACHE = "data/manifests/spk_emb.npz"
SEGS_PER_SPEAKER = 3
WINDOW_SEC = 30.0
SHINGLE_N = 8

# --- frozen text-leakage thresholds (Lead, 2026-08-28) -------------------------------
MAX_EXACT_COLLISIONS = 0      # any identical document across splits is leakage
MAX_JACCARD = 0.05            # 8-word-shingle Jaccard between any cross-split pair
# Observed on the 2026-08-28 split: 0 exact collisions, max Jaccard 0.005587
# (train<->held) and 0.000250 (dev<->test) — an order of magnitude below the threshold.
POWER_SHIFTS = (0.0, 0.05, 0.10, 0.15, 0.20)
POWER_SEED = 20260827

_SESS = None


def _session():
    global _SESS
    if _SESS is None:
        import onnxruntime
        opt = onnxruntime.SessionOptions()
        opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        opt.intra_op_num_threads = 1
        _SESS = onnxruntime.InferenceSession(CAMPPLUS, sess_options=opt,
                                             providers=["CPUExecutionProvider"])
    return _SESS


def fail(msg):
    print(f"[leak] FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def embed_one(task):
    """task = (sample_id, audio_path, duration_sec)"""
    import soundfile as sf
    import torch
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
    sid, path, dur = task
    try:
        info = sf.info(path)
        sr = info.samplerate
        n_frames = info.frames
        want = int(WINDOW_SEC * sr)
        start = max(0, (n_frames - want) // 2)
        data, sr = sf.read(path, start=start, frames=min(want, n_frames), dtype="float32",
                           always_2d=True)
        w = torch.from_numpy(data.T).mean(dim=0, keepdim=True)
        if sr != 16000:
            w = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(w)
        feat = kaldi.fbank(w, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        s = _session()
        emb = s.run(None, {s.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0].flatten()
        return sid, emb.astype(np.float32), None
    except Exception as e:  # noqa: BLE001
        return sid, None, f"{type(e).__name__}: {e}"


def pick_segments(rows):
    """speaker_key -> up to SEGS_PER_SPEAKER (sample_id, path, dur), longest first."""
    by = defaultdict(list)
    for r in rows:
        by[r["speaker_key"]].append(r)
    out = {}
    for k, rs in by.items():
        rs = sorted(rs, key=lambda r: (-r["duration_sec"], r["sample_id"]))[:SEGS_PER_SPEAKER]
        out[k] = [(r["sample_id"], r["audio_path"], r["duration_sec"]) for r in rs]
    return out


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def stage_embed(args):
    rows = load_jsonl(args.manifest)
    picks = pick_segments(rows)
    tasks, owner = [], {}
    for k, segs in sorted(picks.items()):
        for sid, p, d in segs:
            tasks.append((sid, p, d))
            owner[sid] = k
    wanted = set(owner)

    have, evicted, refiled = {}, [], 0
    if os.path.exists(EMB_CACHE) and not args.force:
        cached = np.load(EMB_CACHE, allow_pickle=True)
        c_keys = cached["speaker_keys"] if "speaker_keys" in cached.files else None
        for i, s in enumerate(cached["sample_ids"]):
            s = str(s)
            if s not in wanted:
                # EVICTION: the sample_id is gone from the manifest (owner filtered the
                # audio, or the segment now falls outside the per-speaker pick). Keeping it
                # would file the embedding under a stale/empty speaker_key, i.e. invent a
                # phantom speaker on a phantom channel that silently inflates the null.
                evicted.append(s)
                continue
            have[s] = cached["emb"][i]
            if c_keys is not None and str(c_keys[i]) != owner[s]:
                refiled += 1
    todo = [t for t in tasks if t[0] not in have]
    print(f"[leak] speaker_keys={len(picks)} segments={len(tasks)} cached_reused={len(have)} "
          f"evicted_stale={len(evicted)} refiled_speaker_key={refiled} todo={len(todo)}",
          flush=True)
    if evicted:
        print(f"[leak] evicted examples: {sorted(evicted)[:5]}", flush=True)
    t0 = time.time()
    errors = []
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (sid, emb, err) in enumerate(ex.map(embed_one, todo, chunksize=4)):
                if err:
                    errors.append((sid, err))
                else:
                    have[sid] = emb
                if (i + 1) % 100 == 0:
                    print(f"[leak] embed {i+1}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    sids = sorted(have)
    keys = [owner[s] for s in sids]          # KeyError impossible: have ⊆ wanted
    if any(not k for k in keys):
        fail("empty speaker_key in the embedding cache — eviction logic is broken")
    np.savez(EMB_CACHE, sample_ids=np.array(sids),
             emb=np.stack([have[s] for s in sids]).astype(np.float32),
             speaker_keys=np.array(keys))
    print(f"[leak] cached {len(sids)} embeddings in {EMB_CACHE}, elapsed {time.time()-t0:.0f}s, "
          f"errors={len(errors)}")
    if errors:
        print(errors[:10])
        fail(f"{len(errors)} segments could not be embedded, e.g. {errors[:3]}")


# ------------------------------- text leakage ---------------------------------------
def _h64(s):
    """Stable 64-bit hash (python's hash() is salted per process)."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def shingles(text, n=SHINGLE_N):
    toks = text.split()
    if not toks:
        return set()
    if len(toks) < n:
        return {_h64(" ".join(toks))}
    return {_h64(" ".join(toks[i:i + n])) for i in range(len(toks) - n + 1)}


def text_leakage(train, held):
    """held: list of (split, row). Returns per-doc max Jaccard + exact-hash collisions.

    `train` is only a name: the same routine is used for train<->held and for dev<->test
    (with dev in the `train` position)."""
    held_sh = {r["sample_id"]: shingles(r["text"] or "") for _, r in held}
    inv = defaultdict(list)
    for sid, S in held_sh.items():
        for h in S:
            inv[h].append(sid)
    inter = defaultdict(Counter)   # held sid -> Counter(train sid -> shared shingles)
    train_size = {}
    for r in train:
        S = shingles(r["text"] or "")
        train_size[r["sample_id"]] = len(S)
        hit = Counter()
        for h in S:
            for sid in inv.get(h, ()):
                hit[sid] += 1
        for sid, c in hit.items():
            inter[sid][r["sample_id"]] = c
    res = []
    for sp, r in held:
        sid = r["sample_id"]
        A = len(held_sh[sid])
        top = []
        for tr, c in inter.get(sid, {}).items():
            B = train_size[tr]
            den = A + B - c
            top.append((c / den if den else 0.0, tr, c))
        top.sort(reverse=True)
        best = top[0][0] if top else 0.0
        best_tr = top[0][1] if top else None
        res.append({"split": sp, "sample_id": sid, "channel_id": r["channel_id"],
                    "words": r["words"], "n_shingles": A,
                    "max_jaccard": round(best, 6), "best_train_sample": best_tr,
                    "top": [(round(j, 6), tr, c) for j, tr, c in top[:3]]})
    th = defaultdict(list)
    for r in train:
        th[r["sha256_text"]].append(r["sample_id"])
    exact = []
    for sp, r in held:
        if r["sha256_text"] in th:
            exact.append({"split": sp, "sample_id": r["sample_id"],
                          "train_samples": th[r["sha256_text"]][:5]})
    return res, exact


# ------------------------------- voice leakage --------------------------------------
def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def positive_controls(by_key, rows, tau):
    """Empirical power of the tau-detector on positive controls built from real data.

    P1 (primary): the same `speaker_key` split into two disjoint halves of its segments,
        each half averaged exactly as a speaker is averaged in the decision. This is the
        *same-session* positive: it answers "if a held-out speaker really were one of the
        train speakers, would the statistic exceed tau?" — an UPPER bound on power,
        because both halves share room, mic and codec, which campplus also encodes.
    P1 shifted: the same statistic minus a hypothetical cross-session penalty delta,
        which brackets how fast power decays if the duplicate speaker were recorded in a
        different session.
    P2 (real cross-video): pairs of speaker_keys from DIFFERENT videos of the SAME
        channel, restricted to videos the diariser found single-speaker. On a one-host
        channel these are the same person recorded on different days; on a multi-host
        channel they are different people. The rate is therefore a MIXTURE, quoted as a
        realistic floor, not a clean power estimate.
    """
    rng = np.random.RandomState(POWER_SEED)
    p1, n1_keys = [], 0
    for k in sorted(by_key):
        v = by_key[k]
        if len(v) < 2:
            continue
        n1_keys += 1
        idx = np.arange(len(v))
        rng.shuffle(idx)
        h = max(1, len(idx) // 2)
        A = np.mean([v[i][1] for i in idx[:h]], axis=0)
        B = np.mean([v[i][1] for i in idx[h:]], axis=0)
        p1.append(cos(A, B))
    p1 = np.array(p1) if p1 else np.zeros(0)

    spk_by_video = defaultdict(set)
    single_ok = defaultdict(lambda: True)
    for r in rows:
        vk = (r["channel_id"], r["video_id"])
        spk_by_video[vk].add(r["local_speaker_id"])
        if not r.get("is_single_speaker"):
            single_ok[vk] = False
    avg = {k: np.mean([e for _, e in v], axis=0) for k, v in by_key.items()}
    by_chan = defaultdict(list)
    for k in by_key:
        parts = k.split("/")
        if len(parts) != 3:
            continue
        c, vid, _ls = parts
        if len(spk_by_video.get((c, vid), ())) == 1 and single_ok[(c, vid)]:
            by_chan[c].append((vid, k))
    p2 = []
    for c, lst in sorted(by_chan.items()):
        lst.sort()
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if lst[i][0] == lst[j][0]:
                    continue
                p2.append(cos(avg[lst[i][1]], avg[lst[j][1]]))
    p2 = np.array(p2) if p2 else np.zeros(0)

    def frac(a, t):
        return float((a > t).mean()) if a.size else float("nan")

    return {
        "tau": tau,
        "p1_n_speaker_keys": n1_keys,
        "p1_cos_p05": float(np.percentile(p1, 5)) if p1.size else float("nan"),
        "p1_cos_p50": float(np.percentile(p1, 50)) if p1.size else float("nan"),
        "p1_cos_p95": float(np.percentile(p1, 95)) if p1.size else float("nan"),
        "power_at_tau": frac(p1, tau),
        "power_vs_shift": {f"{d:.2f}": frac(p1 - d, tau) for d in POWER_SHIFTS},
        # a "positive" whose two halves are this far apart is almost certainly a
        # diarisation error (two people sharing one local_speaker_id), i.e. a contaminated
        # positive control, so power measured on P1 is itself pulled down
        "p1_frac_below_0.50": float((p1 < 0.50).mean()) if p1.size else float("nan"),
        "power_at_tau_clean_positives": (float((p1[p1 >= 0.50] > tau).mean())
                                         if p1.size and (p1 >= 0.50).any() else float("nan")),
        "p2_n_pairs": int(p2.size),
        "p2_n_channels": int(sum(1 for c, l in by_chan.items()
                                 if len({v for v, _ in l}) >= 2)),
        "p2_rate_above_tau": frac(p2, tau),
        "p2_cos_p50": float(np.percentile(p2, 50)) if p2.size else float("nan"),
        "p2_cos_p95": float(np.percentile(p2, 95)) if p2.size else float("nan"),
    }


def stage_report(args):
    rows = load_jsonl(args.manifest)
    train = load_jsonl("data/manifests/train_long.jsonl")
    dev = load_jsonl("data/manifests/dev.jsonl")
    test = load_jsonl("data/manifests/test.jsonl")
    held = [("dev", r) for r in dev] + [("test", r) for r in test]
    print(f"[leak] train={len(train)} dev={len(dev)} test={len(test)}", flush=True)

    t0 = time.time()
    text_res, exact = text_leakage(train, held)
    dt_res, dt_exact = text_leakage(dev, [("test", r) for r in test])
    print(f"[leak] text leakage done in {time.time()-t0:.0f}s", flush=True)

    if not os.path.exists(EMB_CACHE):
        fail(f"{EMB_CACHE} missing — run `--embed` first")
    z = np.load(EMB_CACHE, allow_pickle=True)
    emb = {str(s): e for s, e in zip(z["sample_ids"], z["emb"])}
    key_of = {str(s): str(k) for s, k in zip(z["sample_ids"], z["speaker_keys"])}

    # --- cache freshness: the cache must describe THIS manifest, nothing else ---------
    owner = {}
    for k, segs in pick_segments(rows).items():
        for sid, _p, _d in segs:
            owner[sid] = k
    stale = sorted(set(key_of) - set(owner))
    misfiled = sorted(s for s in key_of if s in owner and key_of[s] != owner[s])
    missing = sorted(set(owner) - set(key_of))
    if stale or misfiled:
        fail(f"stale embedding cache: {len(stale)} sample_ids no longer in {args.manifest} "
             f"(e.g. {stale[:3]}), {len(misfiled)} filed under an outdated speaker_key "
             f"(e.g. {misfiled[:3]}). Re-run `--embed` (it evicts) or `--embed --force`.")
    if missing:
        fail(f"{len(missing)} selected segments have no embedding (e.g. {missing[:3]}) — "
             "re-run `--embed`")
    if any(not k for k in key_of.values()):
        fail("embedding cache contains an empty speaker_key")

    by_key = defaultdict(list)
    for s, e in emb.items():
        by_key[key_of[s]].append((s, e))

    # channel -> split comes from split_channels.json so that speaker_keys whose rows were
    # removed by the dev/test consistency filter are still attributed to their channel.
    meta = json.load(open("data/manifests/split_channels.json", encoding="utf-8"))
    chan_split = {c: m["split"] for c, m in meta["channels"].items()}
    keys = sorted(by_key)
    key_channel = {k: k.split("/")[0] for k in keys}
    unknown_ch = sorted({key_channel[k] for k in keys if key_channel[k] not in chan_split})
    if unknown_ch:
        fail(f"{len(unknown_ch)} speaker_key channels are absent from split_channels.json "
             f"(e.g. {unknown_ch[:3]}) — the split and the embedding cache disagree")
    key_split = {k: chan_split[key_channel[k]] for k in keys}

    # The remediation loop is part of the split rule, so the report must state it even when
    # this run was started by hand without the --note lines scripts/build_dataset.sh passes.
    # split_channels.json is the durable record: make_split.py writes the channels it was
    # forced to keep in train.  (2026-08-29: reports/leakage_report.md §5 claimed "no channel
    # had to be moved" while its own §6 listed two — the verdict now reads this field.)
    forced_train = list(meta.get("forced_train_channels") or [])
    remediation_log = list(getattr(args, "note", None) or [])
    if forced_train and not remediation_log:
        remediation_log = [
            f"{len(forced_train)} channel(s) forced into train by the voice-leakage "
            f"remediation loop: " + " ".join(forced_train)
            + " (read from data/manifests/split_channels.json['forced_train_channels']; "
              "this run was not given the per-iteration --note lines that "
              "scripts/build_dataset.sh records)"]

    # every dev/test speaker_key present in the split files must be covered by an embedding
    heldout_keys_in_split = {r["speaker_key"] for r in dev + test}
    uncovered = sorted(heldout_keys_in_split - set(keys))
    if uncovered:
        fail(f"{len(uncovered)} dev/test speaker_keys have no embedding (e.g. "
             f"{uncovered[:3]}) — the voice check would silently skip them")

    avg = {k: np.mean([e for _, e in v], axis=0) for k, v in by_key.items()}
    M = np.stack([avg[k] for k in keys])
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    S = M @ M.T
    ch = np.array([key_channel[k] for k in keys])
    same_ch = ch[:, None] == ch[None, :]
    np.fill_diagonal(same_ch, True)
    X = np.where(same_ch, -2.0, S)          # cross-channel similarities only
    split_arr = np.array([key_split[k] for k in keys])
    tr_idx = np.where(split_arr == "train")[0]
    max_cross = X.max(axis=1)               # max over ALL other channels
    max_train = X[:, tr_idx].max(axis=1)    # max over TRAIN channels only

    pos = []
    for k, v in by_key.items():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                pos.append(cos(v[i][1], v[j][1]))
    pos = np.array(pos)
    iu = np.triu_indices(len(keys), 1)
    neg_pair = S[iu][(~same_ch)[iu]]
    # NULL of the decision statistic: for every TRAIN speaker_key, the max cosine to any
    # speaker_key of a DIFFERENT channel. The decision is a max over ~700 candidates, so a
    # per-pair percentile would be off by that multiple-comparison factor.
    null_max = max_cross[tr_idx]

    def pct(a, qs):
        return {q: float(np.percentile(a, q)) for q in qs}

    tau = float(np.percentile(null_max, args.tau_percentile))
    power = positive_controls(by_key, rows, tau)
    calib = {
        "n_speaker_keys": len(keys),
        "n_pos_pairs": int(pos.size),
        "n_neg_pairs": int(neg_pair.size),
        "pos": pct(pos, (1, 5, 25, 50, 75, 95, 99)),
        "pos_min": float(pos.min()),
        "neg_pair": pct(neg_pair, (50, 95, 99, 99.9, 99.99)),
        "neg_pair_max": float(neg_pair.max()),
        "null_max_train": pct(null_max, (50, 90, 95, 99, 99.9, 100)),
        "tau": tau,
        "tau_percentile": args.tau_percentile,
        "tau_rule": f"p{args.tau_percentile:g} of the train-internal null: max cosine of a "
                    "train speaker_key to any speaker_key of a different channel",
        "tau_pairwise_p999_for_reference": float(np.percentile(neg_pair, 99.9)),
        "watch_level_p90_null": float(np.percentile(null_max, 90)),
        "n_train_keys_in_null": int(null_max.size),
        "n_train_keys_above_tau": int((null_max > tau).sum()),
        "empirical_false_alarm_rate": float((null_max > tau).mean()),
        "power": power,
    }

    voice = []
    for i, k in enumerate(keys):
        if key_split[k] not in ("dev", "test"):
            continue
        order = np.argsort(-X[i, tr_idx])[:3]
        voice.append({"speaker_key": k, "split": key_split[k], "channel_id": key_channel[k],
                      "max_cos": float(max_train[i]),
                      "top": [(float(X[i, tr_idx[j]]), keys[tr_idx[j]]) for j in order]})
    voice.sort(key=lambda x: -x["max_cos"])
    flagged = [v for v in voice if v["max_cos"] > tau]
    watch = [v for v in voice if v["max_cos"] > calib["watch_level_p90_null"]]
    move = sorted({v["channel_id"] for v in flagged})
    calib["heldout_max_train_pct"] = pct(np.array([v["max_cos"] for v in voice]),
                                         (50, 90, 95, 99, 100)) if voice else {}

    # --- dev <-> test voice ----------------------------------------------------------
    d_idx = np.where(split_arr == "dev")[0]
    t_idx = np.where(split_arr == "test")[0]
    dev_test_voice = {"n_dev_keys": int(d_idx.size), "n_test_keys": int(t_idx.size),
                      "max_cos": float("nan"), "pair": None, "n_above_tau": 0}
    if d_idx.size and t_idx.size:
        sub = X[np.ix_(d_idx, t_idx)]
        i, j = np.unravel_index(int(np.argmax(sub)), sub.shape)
        dev_test_voice = {
            "n_dev_keys": int(d_idx.size), "n_test_keys": int(t_idx.size),
            "max_cos": float(sub.max()),
            "pair": [keys[d_idx[i]], keys[t_idx[j]]],
            "n_above_tau": int((sub.max(axis=1) > tau).sum()),
            "dev_max_pct": pct(sub.max(axis=1), (50, 90, 95, 100)),
        }

    # --- frozen text-leakage failure contract ---------------------------------------
    max_j_held = max((x["max_jaccard"] for x in text_res), default=0.0)
    max_j_dt = max((x["max_jaccard"] for x in dt_res), default=0.0)
    thresholds = {"max_exact_collisions": args.max_exact, "max_jaccard": args.max_jaccard,
                  "shingle_n": SHINGLE_N}
    fail_reasons = []
    if len(exact) > args.max_exact:
        fail_reasons.append(f"{len(exact)} exact sha256_text collisions train<->dev/test "
                            f"(threshold {args.max_exact})")
    if len(dt_exact) > args.max_exact:
        fail_reasons.append(f"{len(dt_exact)} exact sha256_text collisions dev<->test "
                            f"(threshold {args.max_exact})")
    if max_j_held > args.max_jaccard:
        fail_reasons.append(f"max {SHINGLE_N}-word Jaccard train<->dev/test = "
                            f"{max_j_held:.6f} > {args.max_jaccard}")
    if max_j_dt > args.max_jaccard:
        fail_reasons.append(f"max {SHINGLE_N}-word Jaccard dev<->test = {max_j_dt:.6f} > "
                            f"{args.max_jaccard}")

    out = {"calibration": calib, "voice": voice, "flagged": flagged, "watch": watch,
           "channels_to_move_to_train": move,
           "text_exact": exact,
           "text": sorted(text_res, key=lambda x: -x["max_jaccard"]),
           "dev_test_text": sorted(dt_res, key=lambda x: -x["max_jaccard"]),
           "dev_test_text_exact": dt_exact,
           "dev_test_voice": dev_test_voice,
           "text_thresholds": thresholds,
           "text_leakage_fail_reasons": fail_reasons,
           "n_train_keys": int(len(tr_idx)), "n_heldout_keys": len(voice),
           "n_train_docs": len(train), "n_dev_docs": len(dev), "n_test_docs": len(test),
           "forced_train_channels": forced_train,
           "remediation_log": remediation_log}
    os.makedirs("reports", exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    write_md(out, args.md_out, iteration_log=out["remediation_log"])
    print(json.dumps({"tau": tau, "n_flagged_keys": len(flagged),
                      "n_watch_keys": len(watch),
                      "channels_to_move": move,
                      "power_at_tau": power["power_at_tau"],
                      "max_jaccard_train_held": max_j_held,
                      "max_jaccard_dev_test": max_j_dt,
                      "n_exact_text_collisions": len(exact),
                      "n_exact_dev_test": len(dt_exact),
                      "text_leakage_fail_reasons": fail_reasons}, ensure_ascii=False,
                     indent=2))
    if fail_reasons:
        fail("text leakage above the frozen thresholds: " + "; ".join(fail_reasons)
             + f". Report written to {args.md_out}; move the offending channel with "
               "`make_split.py --force-train-channels <channel_id>` and rebuild.")


def write_md(out, path="reports/leakage_report.md", iteration_log=None):
    calib = out["calibration"]
    pw = calib.get("power", {})

    def g(d, k):
        return d.get(k, d.get(str(k), float("nan")))

    L = []
    A = L.append
    A("# Leakage report — RuLongTTS (PLAN.md §6.2)\n")
    A("Generated by `src/data/leakage_check.py --report`. Every number below comes from that "
      "run; raw values are in `reports/leakage_raw.json`.\n")
    A("## 0. Method\n")
    A("**Text.** Both sides use the ROVER transcript stored in `text`. "
      "(a) *exact*: collision of `sha256_text`. "
      f"(b) *shingles*: each document becomes the set of its {SHINGLE_N}-word shingles "
      "(blake2b-64 of the whitespace-joined tokens — stable across processes, unlike "
      "python's salted `hash`). For every held-out document the Jaccard index against "
      "**every** document of the other side is computed through an inverted index, and the "
      "maximum is reported. Two directions are checked: train↔dev/test (§3) and dev↔test "
      "(§4).\n")
    A("**Voice.** Speaker embeddings reproduce `CosyVoiceFrontEnd._extract_spk_embedding` "
      "(`~/CosyVoice/cosyvoice/cli/frontend.py:108-118`) exactly: mono mix → resample to "
      "16 kHz → `torchaudio.compliance.kaldi.fbank(num_mel_bins=80, dither=0, "
      "sample_frequency=16000)` → subtract the per-utterance mean over time → "
      "`campplus.onnx` from `pretrained_models/Fun-CosyVoice3-0.5B` (CPUExecutionProvider, "
      "`intra_op_num_threads=1`, `ORT_ENABLE_ALL`). Per `speaker_key` "
      f"(`channel_id/video_id/local_speaker_id`) up to {SEGS_PER_SPEAKER} segments are "
      f"embedded (longest first, centred {WINDOW_SEC:.0f} s window) and the embeddings are "
      "averaged. Similarity = cosine.\n")
    A("**Cache hygiene.** `--embed` treats `data/manifests/spk_emb.npz` as a cache, not an "
      "archive: every cached `sample_id` that the current manifest no longer selects is "
      "evicted before the file is rewritten, and `--report` refuses to run (exit 2) on a "
      "cache that still contains stale, misfiled or missing entries. Without eviction a "
      "shrinking corpus leaves embeddings filed under an empty `speaker_key`, i.e. a "
      "phantom speaker on a phantom channel, which enters the null and raises τ.\n")
    A("## 1. Voice threshold calibration\n")
    A(f"- speaker_keys embedded: {calib['n_speaker_keys']}")
    A(f"- positive pairs (same channel+video+local_speaker, **different** segments): "
      f"n = {calib['n_pos_pairs']}")
    A(f"- negative pairs (all cross-channel speaker_key pairs, averaged embeddings): "
      f"n = {calib['n_neg_pairs']}\n")
    ps, ng, nm = calib["pos"], calib["neg_pair"], calib["null_max_train"]
    A("| distribution | p50 | p90 | p95 | p99 | p99.9 | extreme |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    A(f"| same speaker_key, different segments (positives) | {g(ps,50):.4f} | - | - | "
      f"{g(ps,99):.4f} | - | p5 {g(ps,5):.4f}, min {calib['pos_min']:.4f} |")
    A(f"| cross-channel, **per pair** (negatives) | {g(ng,50):.4f} | - | {g(ng,95):.4f} | "
      f"{g(ng,99):.4f} | {g(ng,99.9):.4f} | max {calib['neg_pair_max']:.4f} |")
    A(f"| **null of the decision statistic**: max cosine of a *train* speaker_key to any "
      f"speaker_key of a different channel | {g(nm,50):.4f} | {g(nm,90):.4f} | "
      f"{g(nm,95):.4f} | {g(nm,99):.4f} | {g(nm,99.9):.4f} | max {g(nm,100):.4f} |")
    A("")
    A("**Why the null is a max, not a pair.** The decision for one held-out speaker asks "
      "*\"is this speaker close to **any** of the ~700 train speakers?\"* — a maximum over "
      "~700 comparisons. A per-pair percentile is therefore off by that multiple-comparison "
      f"factor: the per-pair p99.9 is {calib['tau_pairwise_p999_for_reference']:.4f}, but the "
      f"*median* train speaker already reaches {g(nm,50):.4f} as its own max over other "
      "channels, so that threshold would flag about half of the corpus. The null used here is "
      "the same statistic computed on train speakers only — channels that are different by "
      "construction (PLAN §0.2: there is no cross-video speaker identity in this corpus).\n")
    A(f"**Threshold τ = {calib['tau']:.4f}** — {calib['tau_rule']}. Empirical false-alarm "
      f"rate on that null: {calib['n_train_keys_above_tau']} of "
      f"{calib['n_train_keys_in_null']} "
      f"train speaker_keys = {100*calib['empirical_false_alarm_rate']:.1f} % (5 % by "
      f"construction). The stricter p99 of the same null is {g(nm,99):.4f}; a softer *watch* "
      f"level (p90) is {calib['watch_level_p90_null']:.4f}.\n")
    if pw:
        A("### 1.1 Empirical power of the detector at τ\n")
        A("A threshold without a power statement is unfalsifiable: τ = p95 of the null fixes "
          "the false-alarm rate at 5 % but says nothing about whether a real duplicate "
          "speaker would be caught. Power is measured on positive controls built from this "
          "corpus, not assumed.\n")
        A("**P1 — same `speaker_key`, disjoint halves of its segments** (each half averaged "
          "exactly as a speaker is averaged in the decision; seed "
          f"{POWER_SEED}). This is the same-recording-session positive and therefore an "
          "**upper bound**: both halves share room, microphone and codec, which campplus "
          "also encodes.\n")
        A(f"- speaker_keys with ≥2 embedded segments: {pw['p1_n_speaker_keys']}")
        A(f"- half-vs-half cosine: p05 {pw['p1_cos_p05']:.4f}, p50 {pw['p1_cos_p50']:.4f}, "
          f"p95 {pw['p1_cos_p95']:.4f}")
        A(f"- **power at τ = {pw['tau']:.4f}: {100*pw['power_at_tau']:.1f} %**")
        A(f"- of those positives, {100*pw['p1_frac_below_0.50']:.1f} % have half-vs-half "
          "cosine below 0.50 — two halves of one `speaker_key` that far apart are a "
          "*diarisation* error (one `local_speaker_id` covering two people), not a failure "
          "of the detector. Restricted to positives with cosine ≥ 0.50 the power is "
          f"**{100*pw['power_at_tau_clean_positives']:.1f} %**; the "
          f"{100*pw['power_at_tau']:.1f} %-level figure above is the conservative one and "
          "is the number quoted in the verdict.\n")
        A("Because a genuinely duplicated speaker across two channels would be a *different* "
          "session, the table below re-scores the same positives after subtracting a "
          "hypothetical cross-session penalty δ from every positive cosine — how fast power "
          "decays if the duplicate is recorded elsewhere:\n")
        A("| cross-session penalty δ | power at τ |")
        A("|---:|---:|")
        for d in POWER_SHIFTS:
            k = f"{d:.2f}"
            A(f"| {k} | {100*pw['power_vs_shift'][k]:.1f} % |")
        A("")
        A("**P2 — different videos of the same channel** (only videos the diariser found "
          "single-speaker), i.e. a real cross-session comparison. On a one-host channel "
          "these pairs are the same person on different days; on a multi-host channel they "
          "are different people, so the rate below is a **mixture**, quoted as a realistic "
          "floor rather than a clean power estimate.\n")
        A(f"- pairs: {pw['p2_n_pairs']} across {pw['p2_n_channels']} channels; cosine p50 "
          f"{pw['p2_cos_p50']:.4f}, p95 {pw['p2_cos_p95']:.4f}")
        A(f"- fraction above τ: {100*pw['p2_rate_above_tau']:.1f} %\n")
        A(f"**Reading.** Power is {100*pw['power_at_tau']:.1f} % against a same-session "
          f"duplicate and falls to {100*pw['power_vs_shift']['0.10']:.1f} % once a duplicate "
          "speaker loses 0.10 of cosine to a session change. The 'no voice leakage' verdict "
          "in §5 is therefore strong against re-uploaded/duplicated recordings and weak "
          "against the same person recorded twice under different acoustic conditions — "
          "which is exactly the residual risk PLAN §6.2 accepts when it uses `channel_id` as "
          "the speaker proxy.\n")
    A("## 2. Voice leakage — dev/test speaker_keys vs train\n")
    hm = calib.get("heldout_max_train_pct", {})
    if hm:
        A("Distribution of the decision statistic (max cosine to any **train** speaker_key):\n")
        A("| set | p50 | p90 | p95 | p99 | max |")
        A("|---|---:|---:|---:|---:|---:|")
        A(f"| train (null, vs other channels) | {g(nm,50):.4f} | {g(nm,90):.4f} | "
          f"{g(nm,95):.4f} | {g(nm,99):.4f} | {g(nm,100):.4f} |")
        A(f"| dev + test | {g(hm,50):.4f} | {g(hm,90):.4f} | {g(hm,95):.4f} | "
          f"{g(hm,99):.4f} | {g(hm,100):.4f} |")
        A("")
    A(f"- train speaker_keys: {out['n_train_keys']}; held-out speaker_keys: "
      f"{out['n_heldout_keys']}")
    A(f"- held-out speaker_keys above τ: **{len(out['flagged'])}**")
    A(f"- held-out speaker_keys above the watch level: {len(out.get('watch', []))}")
    forced = out.get("forced_train_channels") or []
    if forced:
        A(f"- **channels moved to train because of voice leakage: {len(forced)}** — "
          + " ".join(f"`{c}`" for c in forced)
          + ". They were forced into train by the remediation loop (see §6); the final "
            "split is `make_split.py --force-train-channels " + " ".join(forced) + "`.")
    else:
        A("- **channels moved to train because of voice leakage: none**")
    A(f"- channels this run would move (a further remediation iteration): "
      f"{out['channels_to_move_to_train'] if out['channels_to_move_to_train'] else 'none'}\n")
    A("Top-15 held-out speaker_keys by maximum cosine to any train speaker:\n")
    A("| split | speaker_key | max cos | nearest train speaker_key | vs τ | vs watch |")
    A("|---|---|---:|---|---|---|")
    for v in out["voice"][:15]:
        A(f"| {v['split']} | `{v['speaker_key']}` | {v['max_cos']:.4f} | "
          f"`{v['top'][0][1]}` | {'ABOVE' if v['max_cos'] > calib['tau'] else 'below'} | "
          f"{'above' if v['max_cos'] > calib['watch_level_p90_null'] else 'below'} |")
    A("")
    dtv = out.get("dev_test_voice") or {}
    if dtv.get("pair"):
        A("### 2.1 dev ↔ test voice\n")
        A("dev and test are disjoint channel sets, so a high similarity here would mean the "
          "*hidden* test shares a speaker with the set used for tuning.\n")
        A(f"- dev speaker_keys {dtv['n_dev_keys']}, test speaker_keys {dtv['n_test_keys']}")
        A(f"- max cosine dev↔test: **{dtv['max_cos']:.4f}** "
          f"(`{dtv['pair'][0]}` ↔ `{dtv['pair'][1]}`), τ = {calib['tau']:.4f}")
        dp = dtv.get("dev_max_pct", {})
        if dp:
            A(f"- per-dev-speaker max to test: p50 {g(dp,50):.4f}, p90 {g(dp,90):.4f}, "
              f"p95 {g(dp,95):.4f}, max {g(dp,100):.4f}")
        A(f"- dev speaker_keys above τ against test: **{dtv['n_above_tau']}**\n")
    A("## 3. Text leakage — train ↔ dev/test\n")
    th = out.get("text_thresholds", {})
    A(f"Frozen failure thresholds (Lead, 2026-08-28): exact collisions > "
      f"{th.get('max_exact_collisions')} **or** max Jaccard > {th.get('max_jaccard')} makes "
      "`leakage_check.py --report` exit 2 and `scripts/build_dataset.sh` abort.\n")
    A(f"- exact `sha256_text` collisions train↔dev/test: **{len(out['text_exact'])}**")
    tx = out["text"]
    A(f"- dev/test documents checked: {len(tx)} (against {out.get('n_train_docs', '?')} "
      "train documents)")
    if tx:
        import statistics
        vals = [t["max_jaccard"] for t in tx]
        A(f"- max Jaccard over all dev/test documents: **{tx[0]['max_jaccard']:.6f}** "
          f"(`{tx[0]['sample_id']}`, {tx[0]['top'][0][2] if tx[0]['top'] else 0} shared "
          f"{SHINGLE_N}-grams)")
        A(f"- median of the per-document maxima: {statistics.median(vals):.6f}; "
          f"documents with max Jaccard > 0.01: {sum(1 for v in vals if v > 0.01)}; "
          f"> 0.05: {sum(1 for v in vals if v > 0.05)}\n")
    A("Top-20 dev/test documents by max Jaccard against train:\n")
    A("| split | dev/test sample_id | words | max Jaccard | shared 8-grams | nearest train "
      "sample |")
    A("|---|---|---:|---:|---:|---|")
    for t in tx[:20]:
        top = t["top"][0] if t["top"] else (0.0, "-", 0)
        A(f"| {t['split']} | `{t['sample_id']}` | {t['words']} | {t['max_jaccard']:.6f} | "
          f"{top[2]} | `{t['best_train_sample']}` |")
    A("")
    if out["text_exact"]:
        A("### Exact text collisions\n")
        for e in out["text_exact"]:
            A(f"- {e['split']} `{e['sample_id']}` == train `{e['train_samples']}`")
        A("")
    A("## 4. Text leakage — dev ↔ test\n")
    A("dev stops being hidden the moment per-item errors are analysed (PLAN §6.2), so text "
      "shared between dev and test would contaminate the hidden set even though no training "
      "signal is involved. Same statistic, same frozen thresholds.\n")
    dt = out.get("dev_test_text", []) or []
    A(f"- exact `sha256_text` collisions dev↔test: **{len(out.get('dev_test_text_exact', []))}**")
    A(f"- test documents checked: {len(dt)} (against {out.get('n_dev_docs', '?')} dev "
      "documents)")
    if dt:
        import statistics
        v2 = [t["max_jaccard"] for t in dt]
        A(f"- max Jaccard dev↔test: **{dt[0]['max_jaccard']:.6f}** (`{dt[0]['sample_id']}` vs "
          f"`{dt[0]['best_train_sample']}`)")
        A(f"- median of the per-document maxima: {statistics.median(v2):.6f}; "
          f"> 0.01: {sum(1 for v in v2 if v > 0.01)}; > 0.05: {sum(1 for v in v2 if v > 0.05)}\n")
        A("Top-5 test documents by max Jaccard against dev:\n")
        A("| test sample_id | words | max Jaccard | shared 8-grams | nearest dev sample |")
        A("|---|---:|---:|---:|---|")
        for t in dt[:5]:
            top = t["top"][0] if t["top"] else (0.0, "-", 0)
            A(f"| `{t['sample_id']}` | {t['words']} | {t['max_jaccard']:.6f} | {top[2]} | "
              f"`{t['best_train_sample']}` |")
        A("")
    A("## 5. Verdict\n")
    mx1 = max([t["max_jaccard"] for t in tx] + [0.0])
    mx2 = max([t["max_jaccard"] for t in dt] + [0.0])
    if not out["flagged"] and not out["text_exact"] and not out.get("text_leakage_fail_reasons"):
        A("- **No cross-split text leakage.** Zero exact collisions in either direction; the "
          f"largest {SHINGLE_N}-word shingle overlap is {mx1:.6f} "
          f"(train↔dev/test) and {mx2:.6f} (dev↔test) — a handful of shared "
          f"8-grams, i.e. common Russian phrases, not shared source material — "
          f"{(th['max_jaccard']/mx1 if mx1 else float('inf')):.0f}× and "
          f"{(th['max_jaccard']/mx2 if mx2 else float('inf')):.0f}× below the frozen "
          f"threshold {th.get('max_jaccard')}.")
        A("- **No cross-split voice leakage detectable with campplus** *in the split as "
          "published*. No held-out speaker is "
          "closer to a train speaker than the 95th percentile of what train speakers already "
          "reach among themselves across channels; the held-out distribution sits *inside* the "
          "null."
          + (f" {len(forced)} channel(s) were forced into train by the remediation loop "
             "(see §6); the final split is `make_split.py --force-train-channels "
             + " ".join(forced) + "`. This verdict therefore describes the **remediated** "
             "split, not the unremediated optimum `make_split.py` returns with no arguments."
             if forced else
             " No channel had to be moved, so the split is exactly the one produced by "
             "`src/data/make_split.py` with no arguments (an exhaustive search over channel "
             "subsets — there is no seed)."))
        if pw:
            A(f"- **Strength of that verdict.** The detector's empirical power at τ is "
              f"{100*pw['power_at_tau']:.1f} % against a same-session duplicate speaker and "
              f"{100*pw['power_vs_shift']['0.10']:.1f} % if the duplicate loses 0.10 cosine to "
              "a session change (§1.1). Read the verdict as: gross speaker reuse is excluded; "
              "the same person recorded twice under clearly different conditions is not.")
        A("- **Caveat.** campplus similarity also encodes recording/channel characteristics, so "
          f"the null itself is wide (train p95 = {g(nm,95):.4f}, train max = {g(nm,100):.4f} — "
          "the highest train↔train cross-channel pairs are plausible genuine re-uploads inside "
          "the train pool and do not affect dev/test). A5 calibrates an independent "
          "encoder for the drift metric (PLAN §9.5) and should prefer references outside the "
          "watch list below.")
    else:
        A(f"- flagged speaker_keys: {[v['speaker_key'] for v in out['flagged']]}")
        A(f"- channels already forced into train by earlier iterations: "
          f"{out.get('forced_train_channels') or 'none'}")
        A(f"- channels this run would move to train: {out['channels_to_move_to_train']}")
        A(f"- exact text collisions: {len(out['text_exact'])}")
        for r in out.get("text_leakage_fail_reasons", []):
            A(f"- **TEXT LEAKAGE FAILURE:** {r}")
    A("")
    if out.get("watch"):
        A("### Watch list (above p90 of the null, kept in dev/test)\n")
        A("| split | speaker_key | max cos | nearest train speaker_key |")
        A("|---|---|---:|---|")
        for v in out["watch"]:
            A(f"| {v['split']} | `{v['speaker_key']}` | {v['max_cos']:.4f} | "
              f"`{v['top'][0][1]}` |")
        A("")
    if iteration_log:
        A("## 6. Remediation log\n")
        for line in iteration_log:
            A(f"- {line}")
        A("")
    os.makedirs("reports", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[leak] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/all.jsonl")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tau-percentile", type=float, default=95.0,
                    help="percentile of the train-internal null used as tau "
                         "(frozen at 95; lower values are for testing the "
                         "remediation loop only)")
    ap.add_argument("--max-exact", type=int, default=MAX_EXACT_COLLISIONS,
                    help="frozen at 0: any identical document across splits fails the build")
    ap.add_argument("--max-jaccard", type=float, default=MAX_JACCARD,
                    help="frozen at 0.05: larger 8-word-shingle overlap fails the build")
    ap.add_argument("--json-out", default="reports/leakage_raw.json")
    ap.add_argument("--md-out", default="reports/leakage_report.md")
    ap.add_argument("--note", action="append", default=[],
                    help="remediation-log line (repeatable); rendered as section 6 of the report")
    args = ap.parse_args()
    if args.embed:
        stage_embed(args)
    if args.report:
        stage_report(args)


if __name__ == "__main__":
    main()
