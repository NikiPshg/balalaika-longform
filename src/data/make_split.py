#!/usr/bin/env python
"""A1 / PLAN.md §6.2 + decisions.md 2026-08-29 item 3 — channel-level train/dev/test split, v3 rule.

Unit of split = YouTube channel (`channel_id` of the manifest: catalog label, or
`other:<youtube channel_id>` for the videos of the `other` bag). The v1 rule ("8-15 channels
per held-out split") is impossible with the v3.1 corpus (41 channels, 10 of them carrying
88 % of the hours), so the rule is (reports/split_rule_v3.md):

  feasibility of ONE held-out set S (dev or test), evaluated on the rows of its channels:
    F1  |S| >= --min-channels (2) and <= --max-channels (8);
    F2  hours(S) >= --min-frac (0.04) of corpus hours and <= --max-frac (0.065);
    F3  eligible rows (asr_consistency >= 90, single-speaker, non-empty text) cover every
        benchmark bucket B0..B4 (PLAN §7.1) AND every duration bucket of PLAN §0.2;
    F4  >= --min-long-segments (8) eligible segments of >= 480 s from >= --min-long-videos (3)
        different videos (benchmark roots: pilot needs 4 dev roots, hidden set 8 test roots);
    F5  no voice cluster of S (data/manifests/voice_clusters.json, campplus, average linkage
        at tau) contains a speaker_key of a channel OUTSIDE S — "unseen channel" must mean
        "unseen voice" as far as campplus can tell; this also makes dev and test cluster-disjoint;
    F6  every channel of S is in the candidate pool: not `unknown`, >= --cand-min-eligible-hours
        (0.25) of eligible rows, <= --cand-max-total-hours (max-frac x corpus) total, not forced
        into train by the leakage remediation loop;
    F7  (test only) S holds >= 1 male AND >= 1 female speaker_key (F0 proxy of
        data/manifests/spk_f0.json, src/data/speaker_f0.py) with >= --gender-min-segments (3)
        eligible segments and >= --gender-min-sec (600) eligible seconds — exactly A5's
        enrollment gate in src/eval/make_references.py, so that a male and a female unseen
        reference can be cut from the test channels (PLAN §8).
  choice among all feasible disjoint pairs (S1, S2) — exhaustive enumeration, no seed:
    C1  minimise bin(S1) + bin(S2), bin = floor(|hours/H - 0.05| / 0.005)  (0.5-pp bins);
    C2  then maximise voice clusters in S1 + S2;
    C3  then minimise |hours(S1)/H - 0.05| + |hours(S2)/H - 0.05|;
    C4  then the lexicographically smallest (sorted channel ids of S1, of S2).
  test = the set of the pair that satisfies F7 (if both: more voice clusters, then more hours,
  then lexicographic); the other one is dev.

Row-level: dev/test keep only eligible rows (train keeps everything, PLAN §6.2); all.jsonl is
rewritten with `split` and `dev_test_eligible`.

Usage:
  python src/data/make_split.py
  python src/data/make_split.py --force-train-channels <channel_id> ...   # leakage remediation
"""
import argparse
import itertools
import json
import math
import os
from collections import Counter, defaultdict

DUR_BUCKETS = [(0, 40), (40, 90), (90, 180), (180, 360), (360, 720), (720, 10 ** 9)]
DUR_BUCKET_NAMES = ["(0,40]", "(40,90]", "(90,180]", "(180,360]", "(360,720]", "(720,900]"]
B_BUCKETS = {"B0": (20, 40), "B1": (60, 90), "B2": (120, 180), "B3": (240, 360), "B4": (480, 720)}
DEV_TEST_MIN_CONSISTENCY = 90.0
RULE_VERSION = "v3-2026-08-29"


def dur_bucket(d):
    for i, (a, b) in enumerate(DUR_BUCKETS):
        if a < d <= b:
            return i
    return 0


def b_buckets(d):
    return {k for k, (a, b) in B_BUCKETS.items() if a <= d <= b}


def is_eligible(r):
    return (
        r.get("asr_consistency") is not None
        and r["asr_consistency"] >= DEV_TEST_MIN_CONSISTENCY
        and bool(r.get("is_single_speaker"))
        and bool(r.get("text"))
    )


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def channel_stats(rows, cl_of, f0=None, gmin_seg=3, gmin_sec=600.0):
    st = defaultdict(lambda: dict(
        total_sec=0.0, n=0, videos=set(), title="", eligible_sec=0.0, n_eligible=0,
        dur_buckets=set(), b_buckets=set(), n_long=0, long_videos=set(), source="",
        unknown=False, clusters=set(), keys=set(), male_ok=set(), female_ok=set()))
    key_mat = defaultdict(lambda: [0, 0.0])
    for r in rows:
        if is_eligible(r):
            key_mat[r["speaker_key"]][0] += 1
            key_mat[r["speaker_key"]][1] += r["duration_sec"]
    for r in rows:
        c = st[r["channel_id"]]
        c["total_sec"] += r["duration_sec"]
        c["n"] += 1
        c["videos"].add(r["video_id"])
        c["title"] = r.get("channel_title") or ""
        c["source"] = r.get("channel_source") or ""
        c["unknown"] = r.get("channel_source") == "unknown" or ":unknown:" in str(r["channel_id"])
        c["keys"].add(r["speaker_key"])
        if f0 is not None and r["speaker_key"] in f0:
            g = f0[r["speaker_key"]].get("gender")
            n_seg, sec = key_mat[r["speaker_key"]]
            if g in ("male", "female") and n_seg >= gmin_seg and sec >= gmin_sec:
                c[g + "_ok"].add(r["speaker_key"])
        if r["speaker_key"] in cl_of:
            c["clusters"].add(cl_of[r["speaker_key"]])
        if is_eligible(r):
            c["eligible_sec"] += r["duration_sec"]
            c["n_eligible"] += 1
            c["dur_buckets"].add(dur_bucket(r["duration_sec"]))
            c["b_buckets"] |= b_buckets(r["duration_sec"])
            if r["duration_sec"] >= 480:
                c["n_long"] += 1
                c["long_videos"].add(r["video_id"])
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/all.jsonl")
    ap.add_argument("--clusters", default="data/manifests/voice_clusters.json")
    ap.add_argument("--outdir", default="data/manifests")
    ap.add_argument("--target-frac", type=float, default=0.05)
    ap.add_argument("--min-frac", type=float, default=0.04)
    ap.add_argument("--max-frac", type=float, default=0.065)
    ap.add_argument("--bin-pp", type=float, default=0.5, help="C1 bin width in percentage points")
    ap.add_argument("--min-channels", type=int, default=2)
    ap.add_argument("--max-channels", type=int, default=8)
    ap.add_argument("--min-long-segments", type=int, default=8)
    ap.add_argument("--min-long-videos", type=int, default=3)
    ap.add_argument("--cand-min-eligible-hours", type=float, default=0.25,
                    help="frozen 0.25 (v1 used 0.30): the only female speakers that pass A5's "
                         "600 s enrollment gate outside the large channels sit in 0.23-0.27 h "
                         "channels, so 0.30 would make F7 unsatisfiable")
    ap.add_argument("--f0", default="data/manifests/spk_f0.json")
    ap.add_argument("--gender-min-segments", type=int, default=3)
    ap.add_argument("--gender-min-sec", type=float, default=600.0)
    ap.add_argument("--no-gender-constraint", action="store_true",
                    help="drop F7 (diagnostics only; the frozen rule keeps it)")
    ap.add_argument("--no-cluster-constraint", action="store_true",
                    help="drop F5 (diagnostics only; the frozen rule keeps it)")
    ap.add_argument("--force-train-channels", nargs="*", default=[])
    ap.add_argument("--report-top", type=int, default=10)
    args = ap.parse_args()

    rows = load(args.manifest)
    clusters = json.load(open(args.clusters, encoding="utf-8"))
    cl_of = clusters["cluster_of_speaker_key"]
    f0 = None if args.no_gender_constraint else json.load(open(args.f0, encoding="utf-8"))
    st = channel_stats(rows, cl_of, f0, args.gender_min_segments, args.gender_min_sec)
    total_sec = sum(r["duration_sec"] for r in rows)
    H = total_sec / 3600.0
    forced = set(args.force_train_channels)
    cand_max_total_hours = args.max_frac * H
    chan_of_cluster = defaultdict(set)
    for c, s in st.items():
        for l in s["clusters"]:
            chan_of_cluster[l].add(c)

    def cand_reasons(c, s):
        out = []
        if s["unknown"]:
            out.append("unknown_pseudo_channel")
        if c in forced:
            out.append("forced_train")
        if s["eligible_sec"] < args.cand_min_eligible_hours * 3600:
            out.append("min_eligible_hours")
        if s["total_sec"] > cand_max_total_hours * 3600:
            out.append("max_total_hours")
        return out

    reasons = {c: cand_reasons(c, s) for c, s in st.items()}
    cand = sorted(c for c, rs in reasons.items() if not rs)
    excl_any, excl_only = Counter(), Counter()
    for c, rs in reasons.items():
        for r in rs:
            excl_any[r] += 1
        if len(rs) == 1:
            excl_only[rs[0]] += 1
    cand_hours = sum(st[c]["total_sec"] for c in cand) / 3600
    pool = {
        "n_channels": len(st), "n_candidates": len(cand),
        "candidate_hours": round(cand_hours, 4), "corpus_hours": round(H, 4),
        "candidate_share_of_hours": round(cand_hours / H, 6) if H else 0.0,
        "restrictions": {
            "cand_min_eligible_hours": args.cand_min_eligible_hours,
            "cand_max_total_hours": round(cand_max_total_hours, 4),
            "cand_max_total_hours_rule": f"max_frac ({args.max_frac}) x corpus hours",
            "cand_allow_unknown_channels": False,
            "dev_test_min_consistency": DEV_TEST_MIN_CONSISTENCY,
        },
        "excluded_by_any": dict(sorted(excl_any.items())),
        "excluded_as_sole_reason": dict(sorted(excl_only.items())),
        "largest_excluded_channels": [
            {"channel_id": c, "channel_title": st[c]["title"],
             "hours_total": round(st[c]["total_sec"] / 3600, 3), "reasons": reasons[c]}
            for c in sorted(reasons, key=lambda x: -st[x]["total_sec"]) if reasons[c]][:15],
    }
    print(f"[split] rule {RULE_VERSION}: channels={len(st)} candidates={len(cand)} "
          f"forced_train={len(forced)} corpus={H:.2f} h; 4 % = {args.min_frac*H:.2f} h, "
          f"5 % = {args.target_frac*H:.2f} h, cap {args.max_frac*H:.2f} h")
    print(f"[split] candidate pool: {cand_hours:.2f} h ({100*cand_hours/H:.1f} %); "
          f"excluded_by_any={dict(sorted(excl_any.items()))}")

    # ---- enumerate feasible held-out sets --------------------------------------------
    cand_set = set(cand)
    feas = []
    infeasible = Counter()
    lo, hi = args.min_frac * total_sec, args.max_frac * total_sec
    cand_sorted = sorted(cand, key=lambda c: -st[c]["total_sec"])
    n_enum = 0
    for k in range(args.min_channels, args.max_channels + 1):
        for combo in itertools.combinations(cand_sorted, k):
            n_enum += 1
            tot = sum(st[c]["total_sec"] for c in combo)
            if tot < lo:
                infeasible["F2_hours_below_min"] += 1
                continue
            if tot > hi:
                infeasible["F2_hours_above_max"] += 1
                continue
            bb, db = set(), set()
            nlong, lv, cls = 0, set(), set()
            for c in combo:
                bb |= st[c]["b_buckets"]
                db |= st[c]["dur_buckets"]
                nlong += st[c]["n_long"]
                lv |= st[c]["long_videos"]
                cls |= st[c]["clusters"]
            if len(bb) < len(B_BUCKETS) or len(db) < len(DUR_BUCKETS):
                infeasible["F3_bucket_coverage"] += 1
                continue
            if nlong < args.min_long_segments or len(lv) < args.min_long_videos:
                infeasible["F4_long_segments"] += 1
                continue
            S = set(combo)
            if not args.no_cluster_constraint:
                leak = [l for l in cls if not chan_of_cluster[l] <= S]
                if leak:
                    infeasible["F5_voice_cluster_leaves_set"] += 1
                    continue
            male_ok = set().union(*(st[c]["male_ok"] for c in combo))
            female_ok = set().union(*(st[c]["female_ok"] for c in combo))
            frac = tot / total_sec
            feas.append({
                "gender_ok": bool(male_ok) and bool(female_ok),
                "n_male_ok": len(male_ok), "n_female_ok": len(female_ok),
                "channels": tuple(sorted(combo)), "sec": tot, "frac": frac,
                "bin": int(math.floor(abs(frac - args.target_frac) / (args.bin_pp / 100.0))),
                "dev": abs(frac - args.target_frac),
                "clusters": cls, "n_clusters": len(cls), "n_long": nlong,
                "long_videos": len(lv), "elig_sec": sum(st[c]["eligible_sec"] for c in combo),
            })
    print(f"[split] enumerated {n_enum} subsets of {len(cand)} candidates "
          f"(sizes {args.min_channels}..{args.max_channels}); feasible single sets: {len(feas)}; "
          f"infeasible: {dict(infeasible)}")
    if not feas:
        raise SystemExit("[split] FALLBACK REQUIRED (decisions.md 2026-08-29 3c): no single held-out "
                         "set satisfies F1-F6")

    # ---- best disjoint, cluster-disjoint pair ----------------------------------------
    feas.sort(key=lambda f: (f["bin"], -f["n_clusters"], f["dev"], f["channels"]))
    best, best_key = None, None
    n_pairs = 0
    for i, a in enumerate(feas):
        # lower bound of the pair key with this `a` cannot beat `best` -> stop (feas is sorted by bin)
        if best_key is not None and (a["bin"] + feas[0]["bin"]) > best_key[0]:
            break
        for b in feas[i + 1:]:
            if best_key is not None and a["bin"] + b["bin"] > best_key[0]:
                break
            if set(a["channels"]) & set(b["channels"]):
                continue
            if a["clusters"] & b["clusters"]:
                continue
            if not args.no_gender_constraint and not (a["gender_ok"] or b["gender_ok"]):
                infeasible["F7_no_set_with_both_genders"] += 1
                continue
            n_pairs += 1
            key = (a["bin"] + b["bin"], -(a["n_clusters"] + b["n_clusters"]),
                   a["dev"] + b["dev"], tuple(sorted([a["channels"], b["channels"]])))
            if best_key is None or key < best_key:
                best_key, best = key, (a, b)
    if best is None:
        raise SystemExit("[split] FALLBACK REQUIRED (decisions.md 2026-08-29 3c): feasible single "
                         f"sets exist ({len(feas)}) but no disjoint, cluster-disjoint pair")
    a, b = best
    # test = the set satisfying F7, then more voice clusters (tie: more hours, then lexicographic)
    order = sorted([a, b], key=lambda f: (not f["gender_ok"], -f["n_clusters"], -f["sec"], f["channels"]))
    test_set, dev_set = order[0], order[1]
    print(f"[split] feasible pairs examined: {n_pairs}; best key={best_key[:3]}")
    for name, f in (("dev", dev_set), ("test", test_set)):
        print(f"[split] {name}: {len(f['channels'])} channels, {f['sec']/3600:.2f} h "
              f"({100*f['frac']:.2f} % of corpus), eligible {f['elig_sec']/3600:.2f} h, "
              f"voice clusters {f['n_clusters']}, n>=480s={f['n_long']} from {f['long_videos']} videos, "
              f"gender-capable speakers m={f['n_male_ok']} f={f['n_female_ok']}: {list(f['channels'])}")

    # runner-up table for the report (top-N pairs by the same key) — recomputed cheaply
    alternatives = []
    seen = set()
    for i, x in enumerate(feas):
        for y in feas[i + 1:]:
            if set(x["channels"]) & set(y["channels"]) or (x["clusters"] & y["clusters"]):
                continue
            if not args.no_gender_constraint and not (x["gender_ok"] or y["gender_ok"]):
                continue
            key = (x["bin"] + y["bin"], -(x["n_clusters"] + y["n_clusters"]), x["dev"] + y["dev"],
                   tuple(sorted([x["channels"], y["channels"]])))
            if key[3] in seen:
                continue
            seen.add(key[3])
            alternatives.append((key, x, y))
            if len(alternatives) > 20000:
                break
        if len(alternatives) > 20000:
            break
    alternatives.sort(key=lambda t: t[0])
    alt_out = [{
        "bin_sum": k[0], "clusters": -k[1], "dev_pp": round(100 * k[2], 3),
        "set1": list(x["channels"]), "set1_hours": round(x["sec"] / 3600, 3), "set1_clusters": x["n_clusters"],
        "set2": list(y["channels"]), "set2_hours": round(y["sec"] / 3600, 3), "set2_clusters": y["n_clusters"],
    } for k, x, y in alternatives[:args.report_top]]

    assign = {c: "train" for c in st}
    for c in dev_set["channels"]:
        assign[c] = "dev"
    for c in test_set["channels"]:
        assign[c] = "test"

    os.makedirs(args.outdir, exist_ok=True)
    out = {"train": [], "dev": [], "test": []}
    dropped = defaultdict(int)
    for r in rows:
        sp = assign[r["channel_id"]]
        rr = dict(r)
        rr["split"] = sp
        rr["dev_test_eligible"] = bool(is_eligible(r))
        if sp == "train":
            out["train"].append(rr)
        elif rr["dev_test_eligible"]:
            out[sp].append(rr)
        else:
            dropped[sp] += 1

    all_tmp = args.manifest + ".tmp"
    with open(all_tmp, "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda z: z["sample_id"]):
            r["split"] = assign[r["channel_id"]]
            r["dev_test_eligible"] = bool(is_eligible(r))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(all_tmp, args.manifest)
    print(f"[split] rewrote {args.manifest} with split / dev_test_eligible columns")

    paths = {"train": os.path.join(args.outdir, "train_long.jsonl"),
             "dev": os.path.join(args.outdir, "dev.jsonl"),
             "test": os.path.join(args.outdir, "test.jsonl")}
    for sp, p in paths.items():
        with open(p, "w", encoding="utf-8") as f:
            for r in sorted(out[sp], key=lambda z: z["sample_id"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        h = sum(r["duration_sec"] for r in out[sp]) / 3600
        print(f"[split] {p}: {len(out[sp])} rows, {h:.2f} h, "
              f"{len({r['channel_id'] for r in out[sp]})} channels, "
              f"{len({r['video_id'] for r in out[sp]})} videos, dropped_by_filter={dropped[sp]}")
    link = os.path.join(args.outdir, "test_hidden.jsonl")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink("test.jsonl", link)

    meta = {
        "rule_version": RULE_VERSION,
        "rule_doc": "reports/split_rule_v3.md",
        "target_frac": args.target_frac, "min_frac": args.min_frac, "max_frac": args.max_frac,
        "bin_pp": args.bin_pp,
        "min_channels": args.min_channels, "max_channels": args.max_channels,
        "min_long_segments": args.min_long_segments, "min_long_videos": args.min_long_videos,
        "cluster_constraint": not args.no_cluster_constraint,
        "gender_constraint": not args.no_gender_constraint,
        "gender_min_segments": args.gender_min_segments, "gender_min_sec": args.gender_min_sec,
        "f0_file": args.f0,
        "clusters_file": args.clusters, "clusters_tau": clusters.get("tau"),
        "cand_min_eligible_hours": args.cand_min_eligible_hours,
        "cand_max_total_hours": round(cand_max_total_hours, 4),
        "dev_test_min_consistency": DEV_TEST_MIN_CONSISTENCY,
        "forced_train_channels": sorted(forced),
        "n_candidates": len(cand), "total_hours": H,
        "search": {"n_subsets_enumerated": n_enum, "n_feasible_sets": len(feas),
                   "n_feasible_pairs_examined": n_pairs, "infeasible_by_reason": dict(infeasible),
                   "best_key": [best_key[0], -best_key[1], round(best_key[2], 6)],
                   "alternatives_top": alt_out},
        "candidate_pool": pool,
        "candidate_channels": cand,
        "dev_channels": list(dev_set["channels"]), "test_channels": list(test_set["channels"]),
        "held_out": {name: {"hours": round(f["sec"] / 3600, 4), "frac": round(f["frac"], 6),
                            "eligible_hours": round(f["elig_sec"] / 3600, 4),
                            "voice_clusters": f["n_clusters"], "segments_ge_480s": f["n_long"],
                            "videos_ge_480s": f["long_videos"], "gender_ok": f["gender_ok"],
                            "male_ok_speakers": f["n_male_ok"], "female_ok_speakers": f["n_female_ok"]}
                     for name, f in (("dev", dev_set), ("test", test_set))},
        "channels": {},
    }
    for c, s in sorted(st.items(), key=lambda kv: -kv[1]["total_sec"]):
        meta["channels"][c] = {
            "split": assign[c], "channel_title": s["title"],
            "hours_total": round(s["total_sec"] / 3600, 4),
            "hours_eligible": round(s["eligible_sec"] / 3600, 4),
            "segments": s["n"], "segments_eligible": s["n_eligible"],
            "videos": len(s["videos"]), "segments_ge_480s": s["n_long"],
            "videos_ge_480s": len(s["long_videos"]),
            "speaker_keys": len(s["keys"]), "voice_clusters": len(s["clusters"]),
            "male_ok_speakers": len(s["male_ok"]), "female_ok_speakers": len(s["female_ok"]),
            "dur_buckets": sorted(s["dur_buckets"]), "b_buckets": sorted(s["b_buckets"]),
            "channel_source": s["source"], "candidate": c in cand_set, "excluded_reasons": reasons[c],
        }
    with open(os.path.join(args.outdir, "split_channels.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[split] wrote split_channels.json")


if __name__ == "__main__":
    main()
