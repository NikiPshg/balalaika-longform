#!/usr/bin/env python
"""A1 / decisions.md 2026-08-29 item 3(b) — how many distinct voices does each channel hold?

Input : data/manifests/all.jsonl + data/manifests/spk_emb.npz (campplus embeddings written by
        `src/data/leakage_check.py --embed`: up to 3 segments per speaker_key, centred 30 s
        windows, CosyVoice's own front-end).
Output: data/manifests/voice_clusters.json  (cluster id per speaker_key, per-channel table)
        reports/voice_clusters.md

Method (split-independent, so it can run BEFORE the split and feed make_split.py):
  1. one embedding per speaker_key (`channel_id/video_id/local_speaker_id`) = mean of its
     segment embeddings, L2-normalised; similarity = cosine;
  2. tau = p95 of the null "max cosine of a speaker_key to any speaker_key of a DIFFERENT
     channel", computed over ALL speaker_keys (the v1 leakage rule computed the same
     statistic over train keys only; after the split `leakage_check.py --report` recomputes
     it on train and that value is the one that gates leakage);
  3. agglomerative clustering, average linkage on cosine distance, cut at distance
     1 - tau: two groups merge while their mean cosine exceeds tau. A cluster = one voice as
     far as campplus can tell (it also encodes room/mic/codec, so two sessions of one person
     may split, and two people on one channel with the same setup may merge — the numbers
     are a proxy, stated as such).
  4. per channel: number of speaker_keys, number of clusters, hours per cluster, share of the
     largest cluster; corpus-wide: clusters spanning several channels (candidate voice leakage
     between channels, used as a hard constraint by make_split.py).
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.leakage_check import EMB_CACHE, load_jsonl, pick_segments  # noqa: E402


def calibrate_tau(M, chan, percentile):
    S = M @ M.T
    same = chan[:, None] == chan[None, :]
    X = np.where(same, -2.0, S)
    null_max = X.max(axis=1)
    return float(np.percentile(null_max, percentile)), null_max, S


def cluster(M, tau):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    if len(M) == 1:
        return np.array([1])
    d = pdist(M, metric="cosine")
    Z = linkage(d, method="average")
    return fcluster(Z, t=1.0 - tau, criterion="distance")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/all.jsonl")
    ap.add_argument("--emb", default=EMB_CACHE)
    ap.add_argument("--tau-percentile", type=float, default=95.0)
    ap.add_argument("--json-out", default="data/manifests/voice_clusters.json")
    ap.add_argument("--md-out", default="reports/voice_clusters.md")
    args = ap.parse_args()

    rows = load_jsonl(args.manifest)
    z = np.load(args.emb, allow_pickle=True)
    key_of = {str(s): str(k) for s, k in zip(z["sample_ids"], z["speaker_keys"])}
    emb = {str(s): e for s, e in zip(z["sample_ids"], z["emb"])}
    owner = {}
    for k, segs in pick_segments(rows).items():
        for sid, _p, _d in segs:
            owner[sid] = k
    stale = sorted(set(key_of) - set(owner))
    missing = sorted(set(owner) - set(key_of))
    if stale or missing:
        raise SystemExit(f"[clusters] embedding cache does not match the manifest: "
                         f"{len(stale)} stale, {len(missing)} missing — run leakage_check.py --embed")
    by_key = defaultdict(list)
    for s, e in emb.items():
        by_key[key_of[s]].append(e)
    keys = sorted(by_key)
    M = np.stack([np.mean(by_key[k], axis=0) for k in keys])
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    chan = np.array([k.split("/")[0] for k in keys])
    tau, null_max, S = calibrate_tau(M, chan, args.tau_percentile)
    labels = cluster(M, tau)

    # positives for reference: same speaker_key, different segments
    pos = []
    for k in keys:
        v = by_key[k]
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                a, b = v[i], v[j]
                pos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    pos = np.array(pos) if pos else np.zeros(0)

    hours_key = defaultdict(float)
    n_rows_key = defaultdict(int)
    title = {}
    for r in rows:
        hours_key[r["speaker_key"]] += r["duration_sec"] / 3600.0
        n_rows_key[r["speaker_key"]] += 1
        title[r["channel_id"]] = r.get("channel_title") or ""
    cl_of = {k: int(l) for k, l in zip(keys, labels)}
    cl_keys = defaultdict(list)
    for k, l in cl_of.items():
        cl_keys[l].append(k)
    cl_channels = {l: sorted({k.split("/")[0] for k in ks}) for l, ks in cl_keys.items()}
    cross = {l: chs for l, chs in cl_channels.items() if len(chs) > 1}

    per_channel = {}
    for c in sorted(set(chan)):
        ks = [k for k in keys if k.split("/")[0] == c]
        cls = defaultdict(float)
        for k in ks:
            cls[cl_of[k]] += hours_key[k]
        hrs = sorted(cls.values(), reverse=True)
        tot = sum(hrs)
        per_channel[c] = {
            "channel_title": title.get(c, ""),
            "n_speaker_keys": len(ks),
            "n_videos": len({k.split("/")[1] for k in ks}),
            "hours": round(tot, 4),
            "n_clusters": len(cls),
            "hours_per_cluster": [round(h, 3) for h in hrs],
            "largest_cluster_share": round(hrs[0] / tot, 4) if tot else None,
            "clusters": sorted(cls),
            "cross_channel_clusters": sorted(l for l in cls if l in cross),
            "cross_channel_partners": sorted({cc for l in cls if l in cross for cc in cross[l] if cc != c}),
        }
    # speaker_keys that have rows but no embedding (should be none)
    keys_in_rows = {r["speaker_key"] for r in rows}
    out = {
        "method": {
            "embedding": "campplus (CosyVoice front-end), mean over <=3 segments per speaker_key, L2-normalised",
            "tau_rule": f"p{args.tau_percentile:g} of max cosine of a speaker_key to any speaker_key of a "
                        "different channel, over ALL speaker_keys (split-independent)",
            "clustering": "scipy average linkage on cosine distance, cut at 1 - tau",
        },
        "tau": tau,
        "tau_percentile": args.tau_percentile,
        "null_max_pct": {str(q): float(np.percentile(null_max, q)) for q in (50, 90, 95, 99, 100)},
        "pos_pct": ({str(q): float(np.percentile(pos, q)) for q in (1, 5, 50, 95)} if pos.size else {}),
        "n_pos_pairs": int(pos.size),
        "n_speaker_keys": len(keys),
        "n_speaker_keys_without_embedding": len(keys_in_rows - set(keys)),
        "n_channels": len(per_channel),
        "n_clusters": len(cl_keys),
        "n_cross_channel_clusters": len(cross),
        "cross_channel_clusters": {str(l): {"channels": chs, "speaker_keys": cl_keys[l]}
                                   for l, chs in sorted(cross.items())},
        "cluster_of_speaker_key": cl_of,
        "per_channel": per_channel,
    }
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    L = []
    A = L.append
    A("# Voices per channel — campplus clustering (decisions.md 2026-08-29 item 3b)\n")
    A("Generated by `src/data/voice_clusters.py`; raw values in `data/manifests/voice_clusters.json`.\n")
    A("## 1. Method\n")
    for k, v in out["method"].items():
        A(f"- **{k}**: {v}")
    A(f"- speaker_keys embedded: {len(keys)} over {len(per_channel)} channels; "
      f"same-speaker_key positive pairs: {int(pos.size)}")
    nm = out["null_max_pct"]
    A(f"- null (max cosine to another channel): p50 {nm['50']:.4f}, p90 {nm['90']:.4f}, "
      f"**p95 = τ = {tau:.4f}**, p99 {nm['99']:.4f}, max {nm['100']:.4f}")
    if pos.size:
        pp = out["pos_pct"]
        A(f"- positives (same speaker_key, different segments): p1 {pp['1']:.4f}, p5 {pp['5']:.4f}, "
          f"p50 {pp['50']:.4f}, p95 {pp['95']:.4f} — fraction above τ: {100*(pos > tau).mean():.1f} %")
    A(f"- clusters: **{len(cl_keys)}** in total, {len(cross)} spanning more than one channel\n")
    A("## 2. Per channel\n")
    A("`clusters` counts the voices campplus separates inside the channel; `hours/cluster` "
      "lists them largest first; `cross` = clusters shared with another channel (a merge across "
      "channels is a leakage candidate, so `make_split.py` refuses to hold out a channel whose "
      "cluster leaves the held-out set).\n")
    A("| channel_id | title | videos | speaker_keys | hours | clusters | largest share | hours/cluster (top 6) | cross-channel partners |")
    A("|---|---|---:|---:|---:|---:|---:|---|---|")
    for c, p in sorted(per_channel.items(), key=lambda kv: -kv[1]["hours"]):
        hp = ", ".join(f"{h:.2f}" for h in p["hours_per_cluster"][:6])
        A(f"| `{c}` | {p['channel_title'][:30].replace('|', '/')} | {p['n_videos']} | {p['n_speaker_keys']} | "
          f"{p['hours']:.2f} | {p['n_clusters']} | "
          f"{(100*p['largest_cluster_share']):.0f} % | {hp} | "
          f"{', '.join('`' + x + '`' for x in p['cross_channel_partners']) or '-'} |")
    A("")
    if cross:
        A("## 3. Clusters spanning several channels\n")
        A("| cluster | channels | speaker_keys |")
        A("|---:|---|---|")
        for l, chs in sorted(cross.items()):
            A(f"| {l} | {', '.join('`' + c + '`' for c in chs)} | {len(cl_keys[l])} |")
        A("")
    A("## 4. Reading\n")
    A("A channel with one dominant cluster is one narrator (an audiobook channel); a channel "
      "with many small clusters is a multi-host or multi-narrator source. `unseen channel` "
      "means `unseen speaker` only for clusters that do not cross channels; the split "
      "constraint above enforces exactly that for dev/test.")
    os.makedirs(os.path.dirname(args.md_out), exist_ok=True)
    with open(args.md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[clusters] tau={tau:.4f} keys={len(keys)} clusters={len(cl_keys)} cross={len(cross)}; "
          f"wrote {args.json_out}, {args.md_out}")


if __name__ == "__main__":
    main()
