#!/usr/bin/env python
"""Build a benchmark set (PLAN.md §7): data/benchmark/{pilot,hidden}.jsonl.

The pilot is 6 root documents x 5 nested prefixes B0..B4 = 30 texts.
  * 4 roots come from dev.jsonl: one continuous single-speaker recording each,
    from four different channels and four different genres.  The prefix chain
    starts at the first sentence boundary whose next 60 s contain no
    microphone-check / greeting / stream-promo pattern (configs, prefix_start),
    so B0 is real content and not a roll-call; every prefix ends on a sentence
    boundary of the punctuated gigaam-v3-e2e-ctc transcript and its human
    duration is read off the word-group timestamps in the per-segment json, so
    B0..B4 are genuine nested prefixes of one human recording.
  * 2 roots are constructed documents for genres 5 (numbers/dates/units/
    abbreviations) and 6 (controlled stress).  They have no human audio, so
    their human duration is estimated from the dev words-per-minute median and
    human_audio_path / human_offset_* are null.

Nothing is hand-picked: roots are selected by the frozen criteria in
configs/benchmark_pilot.yaml, so the benchmark rebuilds after the parquet is
re-filtered.  Run:

    python src/benchmark/build_pilot.py [--out data/benchmark/pilot.jsonl]

The SAME script and the SAME frozen criteria build the sealed hidden set from
the test split (A2, 2026-08-30) — only the source manifest, the number of
dataset roots and the two constructed documents differ, all of them config
values (`configs/benchmark_hidden.yaml`, block `hidden:` instead of `pilot:`):

    python src/benchmark/build_pilot.py --config configs/benchmark_hidden.yaml \
        --out data/benchmark/hidden.jsonl \
        --stats data/benchmark/hidden_stats.json \
        --edits reports/benchmark_hidden_edits.json

The only code generalisation the hidden set needed is the genre assignment:
with more dataset roots than genres, each genre takes floor(n/4) or ceil(n/4)
roots (see ``assign_genres``); with n == 4 that is exactly the injective
assignment the pilot used, and the pilot artifacts are byte-identical.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import disfluency as dis  # noqa: E402
from src.eval import normalize as norm  # noqa: E402

SENT_END = ".!?…"
_SENT_SPLIT_RE = re.compile(r"[.!?…]+")

NARRATIVE_RE = re.compile(
    r"\b(был|была|были|было|стал|стала|стали|однажды|затем|потом|тогда|"
    r"впоследствии|некогда|жил|жили|отправил\w*|приехал\w*|родил\w*)\b"
)
EXPLANATORY_RE = re.compile(
    r"\b(то есть|потому что|например|таким образом|это значит|другими словами|"
    r"поэтому|иными словами|дело в том|проще говоря)\b"
)
NORMATIVE_RE = re.compile(
    r"\b(должн\w*|обязан\w*|в соответствии|согласно|стать[ияюе]\w*|пункт\w*|"
    r"порядк\w*|порядок|требован\w*|правил\w*|закон\w*|норматив\w*|инструкц\w*|"
    r"необходимо|следует|устанавлива\w*|регулиру\w*|кодекс\w*|договор\w*)\b"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, "rt", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def word_groups(json_path: str, key: str) -> list[tuple[float, float, str]]:
    """[(start, end, text)] word groups from the per-segment json ``asr_ts``."""
    with open(json_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    section, name = key.split(".", 1)
    raw = (payload.get(section) or {}).get(name) or ""
    out: list[tuple[float, float, str]] = []
    for line in raw.split("\n"):
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        out.append((float(parts[0]), float(parts[1]), parts[2].strip()))
    return out


def n_sentences(text: str) -> int:
    return len([s for s in _SENT_SPLIT_RE.split(text) if s.strip()])


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def transliterate_latin(text: str, table: dict[str, str]) -> tuple[str, int]:
    """Replace whole Latin tokens using ``table`` (case of first letter kept)."""
    changed = 0

    def repl(m: "re.Match[str]") -> str:
        nonlocal changed
        word = m.group(0)
        rus = table.get(word.lower())
        if rus is None:
            return word
        changed += 1
        return rus.capitalize() if word[0].isupper() else rus

    return re.sub(r"[A-Za-z]+", repl, text), changed


def latin_tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in re.finditer(r"[A-Za-z]+", text)}


_TOKENIZER: Any = None


def get_tokenizer(cfg: dict[str, Any]) -> Any:
    """The CosyVoice3 LLM tokenizer (CPU only, no weights are loaded)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(cfg["tokenizer"]["path"])
    return _TOKENIZER


def size_fields(text: str, human_duration_sec: float, cfg: dict[str, Any]) -> dict[str, Any]:
    """PLAN §7.1 model-side sizes: text tokens and the context-occupancy estimate."""
    tok = get_tokenizer(cfg)
    n = len(tok.encode(text, add_special_tokens=False))
    occ = cfg["context_occupancy"]
    total = (
        n
        + float(occ["speech_token_rate_hz"]) * float(human_duration_sec)
        + float(occ["service_tokens"])
    )
    return {
        "text_tts_tokens": n,
        "context_occupancy_est": round(total / float(cfg["tokenizer"]["context_window"]), 6),
    }


# ---------------------------------------------------------------------------
# dataset roots
# ---------------------------------------------------------------------------


def opening_regex(dcfg: dict[str, Any]) -> "re.Pattern[str]":
    pats = dcfg["prefix_start"]["opening_patterns"]
    return re.compile("|".join(f"(?:{p})" for p in pats))


def find_prefix_start(groups: list[tuple[float, float, str]],
                      dcfg: dict[str, Any]) -> tuple[int, float] | None:
    """First sentence boundary whose lookahead window is free of opening patterns.

    A recording opens with a microphone check, a greeting roll-call or a channel
    promo; B0 is only 20-40 s, so that block would BE the benchmark text.  The
    chain therefore starts at the first sentence boundary such that the next
    ``lookahead_sec`` seconds match none of ``opening_patterns``.  Returns
    ``(group_index, start_time)`` or ``None`` when no such boundary exists at or
    before ``max_start_sec`` (the root is then disqualified, not silently
    started at 0).
    """
    pcfg = dcfg["prefix_start"]
    look = float(pcfg["lookahead_sec"])
    max_start = float(pcfg["max_start_sec"])
    rx = opening_regex(dcfg)
    candidates = [0] + [
        i
        for i in range(1, len(groups))
        if groups[i - 1][2].strip() and groups[i - 1][2].strip()[-1] in SENT_END
    ]
    for i in candidates:
        t0 = groups[i][0]
        if t0 > max_start:
            break
        ahead = [g for g in groups if g[0] >= t0 and g[1] <= t0 + look]
        if not ahead:
            continue
        text = " ".join(g[2] for g in ahead).lower().replace("ё", "е")
        if not rx.search(text):
            return i, t0
    return None


def prepare_root(row: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Load one candidate root, edit its text and compute its selection stats.

    Returns ``(candidate, "")`` or ``(None, reject_reason)``.
    """
    dcfg = cfg["dataset_roots"]
    groups = word_groups(row["json_path"], dcfg["timestamp_source"])
    if not groups:
        return None, "no_timestamps"

    started = find_prefix_start(groups, dcfg)
    if started is None:
        return None, "no_clean_prefix_start"
    start_index, start_time = started

    window = float(dcfg["selection_window_sec"])
    win_groups = [g for g in groups if g[0] >= start_time and g[1] <= start_time + window]
    if not win_groups:
        return None, "empty_selection_window"
    raw_window_text = " ".join(g[2] for g in win_groups)

    # hard filters ------------------------------------------------------
    if dcfg["forbid_digits"] and re.search(r"\d", raw_window_text):
        return None, "digits"
    table = {k.lower(): v for k, v in (dcfg.get("latin_transliteration") or {}).items()}
    unknown_latin = latin_tokens(raw_window_text) - set(table)
    if unknown_latin:
        return None, "unknown_latin"

    # text pipeline: transliteration (per group) -> case repair (whole document,
    # length-preserving, so the group spans stay valid) -> disfluency filter
    pieces_source = [g[2] for g in win_groups]   # untouched ASR output
    pieces: list[str] = []
    n_translit = 0
    for _s, _e, txt in win_groups:
        txt, t = transliterate_latin(txt, table)
        n_translit += t
        pieces.append(txt)
    joined = " ".join(pieces)
    repaired, n_case = dis.repair_casing(joined)
    assert len(repaired) == len(joined), "repair_casing must preserve length"
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, piece in enumerate(pieces):
        if i:
            cursor += 1
        spans.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    pieces = [repaired[a:b] for a, b in spans]
    cleaned, edits = dis.clean_pieces(pieces)

    dens, n_fill, n_words = dis.density(raw_window_text)
    feats = genre_features(raw_window_text)
    return {
        "row": row,
        "start_index": start_index,
        "start_time": start_time,
        "groups": win_groups,
        "pieces_source": pieces_source,
        "pieces_raw": pieces,
        "pieces_clean": cleaned,
        "edits": edits,
        "density": dens,
        "n_filler_tokens": n_fill,
        "n_words_window": n_words,
        "n_case_repairs": n_case,
        "n_transliterations": n_translit,
        "features": feats,
    }, ""


def genre_features(text: str) -> dict[str, float]:
    toks = dis.tokenize(text)
    nw = max(1, len(toks))
    low = text.lower().replace("ё", "е")
    sents = max(1, n_sentences(text))
    proper = 0
    for t in toks:
        if t.text[:1].isupper():
            left = text[: t.start].rstrip()
            if left and left[-1] not in SENT_END:
                proper += 1
    return {
        "question_marks_per_100w": 100.0 * text.count("?") / nw,
        "quote_marks_per_100w": 100.0 * (text.count("«") + text.count('"')) / nw,
        "mean_sentence_length_words": nw / sents,
        "normative_markers_per_100w": 100.0 * len(NORMATIVE_RE.findall(low)) / nw,
        "explanatory_markers_per_100w": 100.0 * len(EXPLANATORY_RE.findall(low)) / nw,
        "narrative_markers_per_100w": 100.0 * len(NARRATIVE_RE.findall(low)) / nw,
        "proper_names_per_100w": 100.0 * proper / nw,
    }


def cut_points(cand: dict[str, Any], buckets: dict[str, list[float]]) -> dict[str, int] | None:
    """Group index per bucket: the sentence boundary nearest the bucket centre.

    Times are measured from ``cand["start_time"]``, i.e. from the prefix start,
    not from the beginning of the audio file.
    """
    t0 = cand["start_time"]
    ends = [g[1] - t0 for g in cand["groups"]]
    ok_cut = [
        i
        for i, piece in enumerate(cand["pieces_clean"])
        if piece.strip() and piece.strip()[-1] in SENT_END
    ]
    chosen: dict[str, int] = {}
    prev = -1
    for name in sorted(buckets):
        lo, hi = buckets[name]
        centre = (lo + hi) / 2.0
        pool = [i for i in ok_cut if i > prev and lo <= ends[i] <= hi]
        if not pool:
            return None
        best = min(pool, key=lambda i: (abs(ends[i] - centre), i))
        chosen[name] = best
        prev = best
    return chosen


def assign_genres(pool: list[dict[str, Any]],
                  cfg: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Assign every pool root to one of the dataset genres, by z-scored features.

    The pilot has exactly one root per genre, so the assignment is a bijection
    and the search is over the injective assignments (``permutations``).  The
    hidden set has more roots than genres (composition report §7.3 asks for an
    even genre coverage), so the search is over every assignment in which each
    genre receives ``floor(n/4)`` or ``ceil(n/4)`` roots — with ``n == 4`` that
    is the same set of assignments as before, evaluated in the same order, so
    the pilot output is unchanged.  Objective and tie-break are the frozen ones:
    maximise the summed genre-fit z-score (minus ``lambda`` x density z-score),
    then the lowest summed filler density, then the lexicographically smallest
    tuple of sample_ids read genre by genre.
    """
    gcfg = cfg["dataset_roots"]["genre_assignment"]
    lam = float(gcfg.get("filler_penalty_lambda", 0.0))
    featmap = {int(k): v for k, v in gcfg["features"].items()}
    keys = sorted({k for v in featmap.values() for k in v} | {"density"})
    z: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = [(c["density"] if key == "density" else c["features"][key]) for c in pool]
        mean = statistics.mean(vals)
        sd = statistics.pstdev(vals) or 1.0
        z[key] = {
            c["row"]["sample_id"]: (
                ((c["density"] if key == "density" else c["features"][key]) - mean) / sd
            )
            for c in pool
        }

    def score(genre: int, cand: dict[str, Any]) -> float:
        sid = cand["row"]["sample_id"]
        return sum(z[k][sid] for k in featmap[genre])

    genres = sorted(featmap)
    n_slots = len(pool)
    if n_slots < len(genres):
        raise RuntimeError(f"{n_slots} roots cannot cover {len(genres)} genres")
    base, extra = divmod(n_slots, len(genres))
    lo, hi = base, (base + 1 if extra else base)

    def in_genre_order(combo: tuple[int, ...]) -> list[tuple[int, dict[str, Any]]]:
        """(genre, cand) pairs, genres ascending, then density, then sample_id."""
        by_genre: dict[int, list[dict[str, Any]]] = {g: [] for g in genres}
        for cand, g in zip(pool, combo):
            by_genre[g].append(cand)
        out: list[tuple[int, dict[str, Any]]] = []
        for g in genres:
            for cand in sorted(by_genre[g],
                               key=lambda c: (c["density"], c["row"]["sample_id"])):
                out.append((g, cand))
        return out

    best: tuple[Any, ...] | None = None
    for combo in itertools.product(genres, repeat=n_slots):
        counts = collections.Counter(combo)
        if any(not (lo <= counts[g] <= hi) for g in genres):
            continue
        ordered = in_genre_order(combo)
        total = sum(
            score(g, c) - lam * z["density"][c["row"]["sample_id"]] for g, c in ordered
        )
        key = (
            -total,
            sum(c["density"] for _g, c in ordered),
            tuple(c["row"]["sample_id"] for _g, c in ordered),
        )
        if best is None or key < best[0]:
            best = (key, ordered, total)
    assert best is not None
    assignment: dict[int, list[dict[str, Any]]] = {g: [] for g in genres}
    for g, cand in best[1]:
        assignment[g].append(cand)
    return assignment


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


def make_dataset_records(genre: int, cand: dict[str, Any], cuts: dict[str, int],
                         cfg: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    row = cand["row"]
    root_id = f"ds{genre}_{row['sample_id']}"
    start = cand["start_time"]
    out = []
    for bucket in sorted(cuts):
        k = cuts[bucket]
        text_tts = dis.finalize(" ".join(cand["pieces_clean"][: k + 1]))
        text_un = dis.finalize(" ".join(cand["pieces_source"][: k + 1]))
        end = cand["groups"][k][1]
        duration = end - start
        prefix_len = len(" ".join(cand["pieces_raw"][: k + 1]))
        in_prefix = [e for e in cand["edits"] if e.start < prefix_len]
        n_edits = len(in_prefix)
        n_removed = sum(
            len(dis.tokenize(e.surface)) for e in in_prefix if e.kind != "stutter"
        )
        ref = norm.normalize(text_tts, variant)
        out.append(
            {
                "text_id": f"{root_id}__{bucket}",
                "root_id": root_id,
                "bucket": bucket,
                "genre": genre,
                "source": "dataset",
                "channel_id": row["channel_id"],
                "channel_title": row["channel_title"],
                "video_id": row["video_id"],
                "video_title": row["video_title"],
                "sample_id": row["sample_id"],
                "split": row["split"],
                "license": row["license"],
                "asr_consistency": row["asr_consistency"],
                "human_audio_path": row["audio_path"],
                "human_audio_sha256": row["sha256_audio"],
                "human_offset_start": round(start, 3),
                "human_offset_end": round(end, 3),
                "human_duration_sec": round(duration, 3),
                "human_duration_source": "asr_ts.gigaam-v3-e2e-ctc word-group times",
                "prefix_start_index": cand["start_index"],
                "prefix_group_index": k,
                "text_tts": text_tts,
                "text_tts_unedited": text_un,
                "text_ref": ref,
                "normalization_variant": variant,
                "words": len(dis.tokenize(text_tts)),
                "chars": len(text_tts),
                "sentences": n_sentences(text_tts),
                "ref_words": len(ref.split()),
                "ref_chars": len(ref),
                "n_disfluency_edits": n_edits,
                "n_words_removed": n_removed,
                # the human audio still contains the removed fillers, so the ASR
                # floor measured against text_tts carries at least this many
                # insertions per reference word (A4: report floor next to WER).
                "est_floor_insertion_rate": round(
                    n_removed / max(1, len(dis.tokenize(text_tts))), 5
                ),
                "sha256_text_tts": sha256_text(text_tts),
                **size_fields(text_tts, duration, cfg),
            }
        )
    return out


def make_external_records(spec: dict[str, Any], cfg: dict[str, Any], wpm: float,
                          variant: str) -> list[dict[str, Any]]:
    buckets = cfg["buckets"]
    path = ROOT / spec["path"]
    raw = path.read_text(encoding="utf-8")
    text = dis.apply_charset(re.sub(r"\s*\n\s*", " ", raw).strip())
    if re.search(r"\d", text) or re.search(r"[A-Za-z]", text):
        raise RuntimeError(f"{path} contains digits or Latin letters")
    edits = dis.scan(text)
    if edits:
        raise RuntimeError(
            f"{path} still matches the disfluency filter: "
            + "; ".join(f"{e.kind}:{e.surface!r}" for e in edits[:10])
        )
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]
    cum: list[tuple[int, int]] = []  # (sentence index, cumulative words)
    total = 0
    for i, s in enumerate(sentences):
        total += len(dis.tokenize(s))
        cum.append((i, total))

    root_id = f"ex{spec['genre']}_{path.stem}"
    out = []
    prev = -1
    for bucket in sorted(buckets):
        lo, hi = buckets[bucket]
        centre = (lo + hi) / 2.0
        pool = [(i, w) for i, w in cum if i > prev and lo <= w / wpm * 60.0 <= hi]
        if not pool:
            raise RuntimeError(f"{path}: no sentence boundary inside bucket {bucket}")
        i, w = min(pool, key=lambda t: (abs(t[1] / wpm * 60.0 - centre), t[0]))
        prev = i
        text_tts = " ".join(sentences[: i + 1])
        ref = norm.normalize(text_tts, variant)
        out.append(
            {
                "text_id": f"{root_id}__{bucket}",
                "root_id": root_id,
                "bucket": bucket,
                "genre": spec["genre"],
                "source": "external",
                "channel_id": None,
                "channel_title": None,
                "video_id": None,
                "video_title": spec.get("title"),
                "sample_id": None,
                "split": None,
                "license": "CC-BY-4.0 (written for this benchmark by A2)",
                "asr_consistency": None,
                "human_audio_path": None,
                "human_audio_sha256": None,
                "human_offset_start": None,
                "human_offset_end": None,
                "human_duration_sec": round(w / wpm * 60.0, 3),
                "human_duration_source": f"estimated: words / dev median wpm ({wpm:.3f})",
                "prefix_start_index": 0,
                "prefix_group_index": i,
                "text_tts": text_tts,
                "text_tts_unedited": text_tts,
                "text_ref": ref,
                "normalization_variant": variant,
                "words": len(dis.tokenize(text_tts)),
                "chars": len(text_tts),
                "sentences": n_sentences(text_tts),
                "ref_words": len(ref.split()),
                "ref_chars": len(ref),
                "n_disfluency_edits": 0,
                "n_words_removed": 0,
                "est_floor_insertion_rate": None,
                "sha256_text_tts": sha256_text(text_tts),
                **size_fields(text_tts, w / wpm * 60.0, cfg),
            }
        )
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "benchmark_pilot.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "benchmark" / "pilot.jsonl"))
    ap.add_argument("--stats", default=str(ROOT / "data" / "benchmark" / "pilot_stats.json"))
    ap.add_argument("--edits", default=str(ROOT / "reports" / "benchmark_edits.json"))
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(config_path)
    dis.load_spec()
    nspec = norm.load_spec()
    variant = nspec["primary_variant"]
    buckets = cfg["buckets"]

    dcfg = cfg["dataset_roots"]
    rows = read_jsonl(ROOT / dcfg["source_manifest"])
    # The constructed documents have no human audio, so their nominal human
    # duration is words / median wpm of the manifest named in
    # `external_roots.human_duration.source_manifest` (the pilot names dev.jsonl,
    # which is also its own root manifest; the hidden set keeps dev.jsonl so that
    # the estimate stays the SAME constant across the two benchmark sets).
    ext_wpm_manifest = (
        ((cfg.get("external_roots") or {}).get("human_duration") or {}).get("source_manifest")
        or dcfg["source_manifest"]
    )
    wpm_src = (
        rows if ext_wpm_manifest == dcfg["source_manifest"]
        else read_jsonl(ROOT / ext_wpm_manifest)
    )
    wpm_rows = [r for r in wpm_src if r.get("words") and r.get("duration_sec")]
    wpm = statistics.median(r["words"] / r["duration_sec"] * 60.0 for r in wpm_rows)

    eligible: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for row in rows:
        if row["duration_sec"] < dcfg["min_duration_sec"]:
            reject("duration")
            continue
        if row["asr_consistency"] < dcfg["min_asr_consistency"]:
            reject("consistency")
            continue
        if dcfg["require_single_speaker"] and not row["is_single_speaker"]:
            reject("multi_speaker")
            continue
        cand, why = prepare_root(row, cfg)
        if cand is None:
            reject(why)
            continue
        cuts = cut_points(cand, buckets)
        if cuts is None:
            reject("no_sentence_boundary_in_some_bucket")
            continue
        cand["cuts"] = cuts
        eligible.append(cand)

    if len(eligible) < dcfg_n(cfg):
        raise RuntimeError(
            f"only {len(eligible)} eligible dataset roots, need {dcfg_n(cfg)}; rejects={rejected}"
        )

    # One root per channel (lowest filler density).  Dataset v3.1 / split rule v3 holds out
    # only 2-8 channels per split, so when fewer channels than roots are eligible the pool is
    # topped up with the next-best roots from OTHER VIDEOS of the same channels (one root per
    # video, still by density) — configs `root_fallback: one_per_video`.  Never two roots
    # from one video.
    ranked = sorted(eligible, key=lambda c: (c["density"], c["row"]["sample_id"]))
    per_channel: dict[str, dict[str, Any]] = {}
    for cand in ranked:
        per_channel.setdefault(cand["row"]["channel_id"], cand)
    pool = sorted(per_channel.values(), key=lambda c: (c["density"], c["row"]["sample_id"]))
    n_channels_eligible = len(pool)
    fallback = dcfg.get("root_fallback", "one_per_video")
    if len(pool) < dcfg_n(cfg):
        if fallback != "one_per_video":
            raise RuntimeError(f"only {len(pool)} distinct channels eligible")
        used_videos = {c["row"]["video_id"] for c in pool}
        for cand in ranked:
            if len(pool) >= dcfg_n(cfg):
                break
            if cand["row"]["video_id"] in used_videos:
                continue
            pool.append(cand)
            used_videos.add(cand["row"]["video_id"])
        pool.sort(key=lambda c: (c["density"], c["row"]["sample_id"]))
        print(f"root_fallback=one_per_video: {n_channels_eligible} eligible channels < "
              f"{dcfg_n(cfg)} roots; pool topped up to {len(pool)} roots from distinct videos")
    if len(pool) < dcfg_n(cfg):
        raise RuntimeError(f"only {len(pool)} distinct videos eligible")

    assignment = assign_genres(pool, cfg)

    records: list[dict[str, Any]] = []
    edits_dump: dict[str, Any] = {}
    for genre in sorted(assignment):
        for cand in assignment[genre]:
            records += make_dataset_records(genre, cand, cand["cuts"], cfg, variant)
            edits_dump[f"ds{genre}_{cand['row']['sample_id']}"] = {
                "channel_title": cand["row"]["channel_title"],
                "video_title": cand["row"]["video_title"],
                "genre": genre,
                "filler_density_per_100w": round(cand["density"], 4),
                "n_filler_tokens": cand["n_filler_tokens"],
                "n_words_window": cand["n_words_window"],
                "n_case_repairs": cand["n_case_repairs"],
                "n_transliterations": cand["n_transliterations"],
                "prefix_start_sec": round(cand["start_time"], 3),
                "edits": [e.as_dict() for e in cand["edits"]],
            }

    documents = cfg["external_roots"]["documents"]
    n_external = set_block(cfg).get("n_external_roots")
    if n_external is not None and len(documents) != int(n_external):
        raise RuntimeError(
            f"config declares n_external_roots={n_external} but lists {len(documents)} documents"
        )
    for spec in documents:
        records += make_external_records(spec, cfg, wpm, variant)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tok_cfg = cfg["tokenizer"]
    tok_files = sorted(
        f for f in ("vocab.json", "merges.txt", "tokenizer_config.json")
        if (Path(tok_cfg["path"]) / f).exists()
    )
    stats = {
        # relative when the config lives inside the work tree, absolute otherwise
        "config": relpath_to_root(config_path),
        "tokenizer": {
            "path": tok_cfg["path"],
            "class": type(get_tokenizer(cfg)).__name__,
            "context_window": tok_cfg["context_window"],
            "files_sha256": {
                f: hashlib.sha256((Path(tok_cfg["path"]) / f).read_bytes()).hexdigest()[:16]
                for f in tok_files
            },
        },
        "normalization_variant": variant,
        "dev_median_wpm": wpm,
        "n_dev_rows": len(rows),
        "n_eligible_roots": len(eligible),
        "n_eligible_channels": n_channels_eligible,
        "n_pool_roots": len(pool),
        "root_fallback": fallback,
        "root_fallback_used": n_channels_eligible < dcfg_n(cfg),
        "rejects": rejected,
        "pool": [
            {
                "sample_id": c["row"]["sample_id"],
                "channel_title": c["row"]["channel_title"],
                "video_title": c["row"]["video_title"],
                "duration_sec": c["row"]["duration_sec"],
                "asr_consistency": c["row"]["asr_consistency"],
                "prefix_start_sec": round(c["start_time"], 3),
                "prefix_start_index": c["start_index"],
                "filler_density_per_100w": round(c["density"], 4),
                "features": {k: round(v, 4) for k, v in c["features"].items()},
                "assigned_genre": next(
                    (g for g, lst in assignment.items() if any(cc is c for cc in lst)), None
                ),
            }
            for c in pool
        ],
        "n_records": len(records),
    }
    Path(args.stats).write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(args.edits).write_text(json.dumps(edits_dump, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"wrote {len(records)} records to {out_path}")
    print(f"external-duration median wpm = {wpm:.3f} ({ext_wpm_manifest}); "
          f"{len(eligible)} eligible roots in {n_channels_eligible} channels "
          f"-> pool of {len(pool)} roots")
    for g in sorted(assignment):
        for c in assignment[g]:
            print(
                f"  genre {g}: {c['row']['channel_title'][:34]:36s} {c['row']['sample_id']:34s} "
                f"start={c['start_time']:7.2f}s density={c['density']:.2f}"
            )
    return 0


def relpath_to_root(path: Path) -> str:
    """``path`` relative to the work tree, or its absolute form when outside it."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def set_block(cfg: dict[str, Any]) -> dict[str, Any]:
    """The size block of the benchmark set this config builds.

    ``pilot:`` in configs/benchmark_pilot.yaml, ``hidden:`` in
    configs/benchmark_hidden.yaml.  Exactly one of them must be present, so a
    config can never silently build the wrong set.
    """
    blocks = [k for k in ("pilot", "hidden") if k in cfg]
    if len(blocks) != 1:
        raise RuntimeError(
            f"config must contain exactly one of 'pilot:' / 'hidden:', found {blocks}"
        )
    return cfg[blocks[0]]


def dcfg_n(cfg: dict[str, Any]) -> int:
    return int(set_block(cfg)["n_dataset_roots"])


if __name__ == "__main__":
    raise SystemExit(main())
