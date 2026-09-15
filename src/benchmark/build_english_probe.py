#!/usr/bin/env python
"""Build the E9 "EN-probe" benchmark (EXPLORATORY probe, not a paper benchmark).

Pre-registered in reports/decisions.md, Lead 2026-08-31 (E9 «EN-probe»): 4 LibriVox
narrators (public domain), one chapter each with its Project Gutenberg text; up to
1 item per bucket B0-B4 per narrator, bucket = actual HUMAN reading duration of a
CONTIGUOUS text fragment (nested prefixes, mirroring the pilot's §7.2 design and
the pilot's frozen bucket edges from configs/benchmark_pilot.yaml); a 5-10 s
reference clip per narrator from a DIFFERENT part of the chapter (outside every
item's span).

Stages (idempotent, all caching under data/external/english/):

    asr    Parakeet (src/eval/asr_english.py, GPU) with word timestamps over each
           chapter flac -> data/external/english/asr/<narrator>.words.json
    build  extract + clean the Gutenberg chapter text, align it to the ASR word
           stream, cut the bucket prefixes at sentence boundaries, pick the
           reference clips, write:
              data/benchmark/english_probe.jsonl
              data/references/references_english.jsonl + data/references/english/*.wav
              data/external/english/build_english_probe.json   (all numbers)
              reports/benchmark_english_composition.md         (generated, not typed)

Frozen Russian files (pilot benchmark, references, configs, src/eval/normalize.py,
asr_gigaam.py) are not read or written.

Usage:
    export CUDA_VISIBLE_DEVICES=2
    .venv-eval/bin/python src/benchmark/build_english_probe.py --stage asr
    .venv-eval/bin/python src/benchmark/build_english_probe.py --stage build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from eval.normalize_english import (  # noqa: E402
    NORMALIZATION_VERSION_EN, VARIANT_EN, normalize_english, words_english,
)

EXT = REPO / "data" / "external" / "english"
ASR_DIR = EXT / "asr"
BENCH_PATH = REPO / "data" / "benchmark" / "english_probe.jsonl"
REFS_PATH = REPO / "data" / "references" / "references_english.jsonl"
REF_WAV_DIR = REPO / "data" / "references" / "english"
BUILD_JSON = EXT / "build_english_probe.json"
REPORT_MD = REPO / "reports" / "benchmark_english_composition.md"

# Pilot bucket edges, mirrored VERBATIM from configs/benchmark_pilot.yaml
# (B0 [20,40] ... B4 [480,720]); the yaml itself is frozen and not imported to
# avoid any accidental write, but tests assert the two agree.
BUCKETS = {"B0": (20.0, 40.0), "B1": (60.0, 90.0), "B2": (120.0, 180.0),
           "B3": (240.0, 360.0), "B4": (480.0, 720.0)}
ORDER = ("B0", "B1", "B2", "B3", "B4")
MIN_MATCH_RUN = 3  # consecutive matched words required before a match may carry time

LICENSE_NOTE = ("audio: LibriVox recording, dedicated to the public domain "
                "(https://librivox.org/pages/public-domain/); text: public domain in the USA, "
                "distributed under the Project Gutenberg License")

NARRATORS = [
    dict(
        narrator_id="en_klett", voice_id="en_klett", reader="Elizabeth Klett",
        gender="female", book="Jane Eyre", author="Charlotte Brontë",
        chapter_label="Chapter XII",
        gutenberg_id=1260, gutenberg_file="gutenberg/pg1260.txt",
        gutenberg_url="https://www.gutenberg.org/ebooks/1260",
        start_heading=r"^CHAPTER XII$", start_occurrence=0,
        end_heading=r"^CHAPTER XIII$", end_occurrence=0,
        audio="audio/janeeyre_ch12_klett.flac", audio_mp3="audio/janeeyre_ch12_klett.mp3",
        audio_url="https://www.archive.org/download/jane_eyre_ver03_0809_librivox/janeeyre_12_bronte_64kb.mp3",
        librivox_id=2192, librivox_url="https://librivox.org/jane-eyre-version-3-by-charlotte-bronte/",
    ),
    dict(
        narrator_id="en_savage", voice_id="en_savage", reader="Karen Savage",
        gender="female", book="Persuasion", author="Jane Austen",
        chapter_label="Chapter X",
        gutenberg_id=105, gutenberg_file="gutenberg/pg105.txt",
        gutenberg_url="https://www.gutenberg.org/ebooks/105",
        start_heading=r"^CHAPTER X\.$", start_occurrence=0,
        end_heading=r"^CHAPTER XI\.$", end_occurrence=0,
        audio="audio/persuasion_ch10_savage.flac", audio_mp3="audio/persuasion_ch10_savage.mp3",
        audio_url="https://www.archive.org/download/persuasion_0905_librivox/persuasion_10_austen_64kb.mp3",
        librivox_id=2693, librivox_url="https://librivox.org/persuasion-by-jane-austen-4/",
    ),
    dict(
        narrator_id="en_smith", voice_id="en_smith", reader="Mark F. Smith",
        gender="male", book="White Fang", author="Jack London",
        chapter_label="Part III, Chapter I (The Makers of Fire)",
        gutenberg_id=910, gutenberg_file="gutenberg/pg910.txt",
        gutenberg_url="https://www.gutenberg.org/ebooks/910",
        start_heading=r"^CHAPTER I$", start_occurrence=2,   # 3rd body "CHAPTER I" = Part III
        end_heading=r"^CHAPTER II$", end_occurrence=2,
        audio="audio/whitefang_p3ch1_smith.flac", audio_mp3="audio/whitefang_p3ch1_smith.mp3",
        audio_url="https://www.archive.org/download/whitefang2_1010_librivox/whitefang2_09_london_64kb.mp3",
        librivox_id=4677, librivox_url="https://librivox.org/white-fang-by-jack-london-librivox-version-2/",
    ),
    dict(
        narrator_id="en_nelson", voice_id="en_nelson", reader="Mark Nelson",
        gender="male", book="A Princess of Mars", author="Edgar Rice Burroughs",
        chapter_label="Chapter XV (Sola Tells Me Her Story)",
        gutenberg_id=62, gutenberg_file="gutenberg/pg62.txt",
        gutenberg_url="https://www.gutenberg.org/ebooks/62",
        start_heading=r"^CHAPTER XV$", start_occurrence=0,
        end_heading=r"^CHAPTER XVI$", end_occurrence=0,
        audio="audio/princessmars_ch15-16_nelson.flac", audio_mp3="audio/princessmars_ch15-16_nelson.mp3",
        audio_url="https://www.archive.org/download/princess_mars_0810_librivox/aprincessofmars_15-16_burroughs_64kb.mp3",
        librivox_id=2481, librivox_url="https://librivox.org/a-princess-of-mars-by-edgar-rice-burroughs-2/",
    ),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# chapter extraction + cleaning
# ---------------------------------------------------------------------------


def extract_chapter(spec: dict) -> tuple[str, dict]:
    """Raw chapter body text between the spec'd headings (title lines dropped)."""
    lines = (EXT / spec["gutenberg_file"]).read_text(encoding="utf-8-sig").splitlines()
    def find(pattern: str, occurrence: int) -> int:
        rx = re.compile(pattern)
        hits = [i for i, ln in enumerate(lines) if rx.match(ln.strip("\r"))]
        if occurrence >= len(hits):
            raise RuntimeError(f"{spec['narrator_id']}: {pattern!r} occurrence "
                               f"{occurrence} not found ({len(hits)} hits)")
        return hits[occurrence]
    a = find(spec["start_heading"], spec["start_occurrence"])
    b = find(spec["end_heading"], spec["end_occurrence"])
    if b <= a:
        raise RuntimeError(f"{spec['narrator_id']}: end heading before start")
    body = lines[a + 1 : b]
    # drop leading blank lines and up to 2 short all-caps title lines
    dropped_titles = []
    while body and not body[0].strip():
        body.pop(0)
    while body and len(dropped_titles) < 2:
        first = body[0].strip()
        letters = [c for c in first if c.isalpha()]
        if (first and len(first) <= 60 and letters
                and sum(c.isupper() for c in letters) / len(letters) > 0.8
                and not first.endswith((".", "!", "?"))):
            dropped_titles.append(first)
            body.pop(0)
            while body and not body[0].strip():
                body.pop(0)
        else:
            break
    return "\n".join(body).strip(), {"start_line": a + 1, "end_line": b,
                                     "dropped_title_lines": dropped_titles}


QUOTES = "“”\"«»‟„❝❞‹›"
APO_FIX = re.compile(r"(?<=\w)’(?=\w)")
DASH_RUN = re.compile(r"--+")
WS = re.compile(r"\s+")


def clean_text(raw: str) -> tuple[str, dict]:
    """One text stream for the TTS input.

    Mirrors the pilot charset philosophy (configs/benchmark_pilot.yaml
    text_charset) transposed to English:
      * quotation marks are DELETED (the frozen-for-E9 normalizer strips them,
        so text_ref/WER are unaffected; only the TTS input changes);
      * apostrophes are KEPT (English contractions need them for natural
        reading; the normalizer deletes them on both sides for WER);
      * every dash variant becomes an em dash;
      * underscores (Gutenberg italics) are deleted;
      * whitespace/newlines collapse to single spaces.
    """
    stats = {}
    text = raw.replace("\r", "")
    stats["n_underscores"] = text.count("_")
    text = text.replace("_", "")
    text = APO_FIX.sub("'", text)               # ’ between letters -> ASCII apostrophe
    stats["n_quotes_deleted"] = sum(text.count(q) for q in QUOTES + "‘’")
    for q in QUOTES + "‘’":
        text = text.replace(q, "")
    stats["n_dash_runs"] = len(DASH_RUN.findall(text))
    text = DASH_RUN.sub("—", text)
    text = text.replace("–", "—").replace("―", "—")
    stats["n_ampersand"] = text.count("&")
    text = text.replace("&c.", "etc.").replace("&", " and ")
    text = WS.sub(" ", text).strip()
    return text, stats


# sentence boundary = token ending in .!?… whose last word is not an abbreviation
ABBREV_RX = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|St|Capt|Col|Lieut|Esq|Hon|Rev|Prof|Gen|Sergt)|(?<![A-Za-z])[A-Z])\.$")


def is_sentence_end(token: str) -> bool:
    t = token.rstrip("—")
    if not t or t[-1] not in ".!?…":
        return False
    return not ABBREV_RX.search(t)


# ---------------------------------------------------------------------------
# stage: asr
# ---------------------------------------------------------------------------


def stage_asr(device: str = "cuda") -> None:
    from eval.asr_english import EnglishTranscriber
    from eval.asr_gigaam import read_audio

    ASR_DIR.mkdir(parents=True, exist_ok=True)
    todo = [s for s in NARRATORS if not (ASR_DIR / f"{s['narrator_id']}.words.json").exists()]
    if not todo:
        print("[asr] all chapter word files cached; nothing to do")
        return
    tr = EnglishTranscriber(device=device)
    print(json.dumps(tr.describe(), ensure_ascii=False), file=sys.stderr)
    for spec in todo:
        out = ASR_DIR / f"{spec['narrator_id']}.words.json"
        audio = EXT / spec["audio"]
        wave, sr = read_audio(audio)
        t0 = time.time()
        words = tr.transcribe_words(wave, sr)
        elapsed = time.time() - t0
        payload = {
            "narrator_id": spec["narrator_id"], "audio": str(audio),
            "audio_sha256": sha256_file(audio),
            "sample_rate": sr, "duration_sec": len(wave) / sr,
            "model_id": tr.model_id, "describe": tr.describe(),
            "elapsed_sec": elapsed, "n_words": len(words), "words": words,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"[asr] {spec['narrator_id']}: {len(words)} words, "
              f"{payload['duration_sec']:.0f}s audio in {elapsed:.0f}s "
              f"(x{payload['duration_sec']/elapsed:.1f})")


# ---------------------------------------------------------------------------
# stage: build
# ---------------------------------------------------------------------------


def norm_with_map(tokens: list[str]) -> list[tuple[str, int]]:
    """[(normalized word, raw token index)] — a raw token may yield 0..n words."""
    out = []
    for i, tok in enumerate(tokens):
        for w in words_english(tok):
            out.append((w, i))
    return out


def align_times(ref_norm: list[tuple[str, int]], asr_words: list[dict]):
    """Times for every normalized reference word, via difflib matching blocks.

    Returns (t_start, t_end, matched) lists over ref_norm indices; unmatched
    positions are linearly interpolated between the nearest matched neighbours
    (extrapolated at 0.35 s/word at the edges).
    """
    asr_norm: list[tuple[str, float, float]] = []
    for w in asr_words:
        for piece in words_english(w["word"]):
            asr_norm.append((piece, float(w["t_start"]), float(w["t_end"])))
    sm = SequenceMatcher(None, [w for w, _ in ref_norm],
                         [w for w, _, _ in asr_norm], autojunk=False)
    n = len(ref_norm)
    t_start = [None] * n
    t_end = [None] * n
    matched = [False] * n
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            t_start[a + k] = asr_norm[b + k][1]
            t_end[a + k] = asr_norm[b + k][2]
            matched[a + k] = True
    # Kill ISOLATED matches (runs shorter than MIN_MATCH_RUN) before using any
    # time: a single stray "the" of the chapter opening matched into the
    # LibriVox intro ("...in the public domain...") and anchored en_smith 17 s
    # early, polluting every en_smith offset (caught by the B0 floor WER 0.65).
    # Only runs of >= MIN_MATCH_RUN consecutive matched words carry time; the
    # rest is interpolated between robust runs.
    i = 0
    while i < n:
        if matched[i]:
            j = i
            while j < n and matched[j]:
                j += 1
            if j - i < MIN_MATCH_RUN:
                for k in range(i, j):
                    matched[k] = False
                    t_start[k] = None
                    t_end[k] = None
            i = j
        else:
            i += 1
    idx = [i for i in range(n) if matched[i]]
    if not idx:
        raise RuntimeError("alignment failed: no matched words")
    STEP = 0.35
    for i in range(n):
        if matched[i]:
            continue
        prev = max((j for j in idx if j < i), default=None)
        nxt = min((j for j in idx if j > i), default=None)
        if prev is None:
            t_start[i] = t_start[nxt] - STEP * (nxt - i)
            t_end[i] = t_start[i] + STEP
        elif nxt is None:
            t_start[i] = t_end[prev] + STEP * (i - prev - 1)
            t_end[i] = t_start[i] + STEP
        else:
            frac = (i - prev) / (nxt - prev)
            t_start[i] = t_end[prev] + frac * (t_start[nxt] - t_end[prev])
            t_end[i] = t_start[i] + min(STEP, max(0.05, (t_start[nxt] - t_end[prev]) / (nxt - prev)))
    return t_start, t_end, matched


def build_narrator(spec: dict) -> dict:
    """Everything for one narrator: items + reference + self-check numbers."""
    import numpy as np
    import soundfile as sf

    asr_payload = json.loads((ASR_DIR / f"{spec['narrator_id']}.words.json").read_text())
    asr_words = asr_payload["words"]
    audio_path = EXT / spec["audio"]

    raw, extract_info = extract_chapter(spec)
    text, clean_stats = clean_text(raw)
    tokens = text.split()
    ref_norm = norm_with_map(tokens)
    t_start, t_end, matched = align_times(ref_norm, asr_words)
    match_rate_chapter = sum(matched) / len(matched)

    # raw token -> [first_norm_idx, last_norm_idx] (tokens with no norm words get None)
    first_norm = {}
    last_norm = {}
    for ni, (_, ri) in enumerate(ref_norm):
        first_norm.setdefault(ri, ni)
        last_norm[ri] = ni

    # chain start = first raw token that has a normalized word
    start_raw = min(first_norm)
    start_t = max(0.0, t_start[first_norm[start_raw]] - 0.2)

    sent_ends = [i for i, tok in enumerate(tokens) if is_sentence_end(tok) and i in last_norm]

    items = []
    prev_cut_raw = start_raw - 1
    prev_dur = 0.0
    for bucket in ORDER:
        lo, hi = BUCKETS[bucket]
        centre = (lo + hi) / 2.0
        cands = []
        for i in sent_ends:
            if i <= prev_cut_raw:
                continue
            dur = t_end[last_norm[i]] - start_t
            if lo <= dur <= hi and dur > prev_dur:
                cands.append((abs(dur - centre), i, dur))
        if not cands:
            print(f"[build] {spec['narrator_id']} {bucket}: no sentence boundary in "
                  f"[{lo},{hi}] s after previous cut — bucket skipped")
            continue
        _, cut_raw, dur = min(cands)
        text_tts = " ".join(tokens[start_raw : cut_raw + 1])
        if any(ch.isdigit() for ch in text_tts):
            raise RuntimeError(f"{spec['narrator_id']} {bucket}: digits in text_tts "
                               "(forbid_digits mirror); adjust the chapter choice")
        text_ref = normalize_english(text_tts)
        frag_norm_idx = range(first_norm[start_raw], last_norm[cut_raw] + 1)
        frag_matched = [matched[i] for i in frag_norm_idx]
        offset_end = t_end[last_norm[cut_raw]]
        items.append({
            "text_id": f"{spec['narrator_id']}__{bucket}",
            "root_id": f"{spec['narrator_id']}_pg{spec['gutenberg_id']}",
            "bucket": bucket,
            "genre": "english_probe",
            "source": "librivox",
            "exploratory": True,
            "language": "en",
            "voice_id": spec["voice_id"],
            "reader": spec["reader"],
            "book": spec["book"], "author": spec["author"],
            "chapter_label": spec["chapter_label"],
            "librivox_id": spec["librivox_id"], "librivox_url": spec["librivox_url"],
            "audio_url": spec["audio_url"],
            "gutenberg_id": spec["gutenberg_id"], "gutenberg_url": spec["gutenberg_url"],
            "license": LICENSE_NOTE,
            "human_audio_path": str(audio_path),
            "human_audio_sha256": asr_payload["audio_sha256"],
            "human_offset_start": round(start_t, 3),
            "human_offset_end": round(offset_end, 3),
            "human_duration_sec": round(offset_end - start_t, 3),
            "human_duration_source": f"{asr_payload['model_id']} word timestamps (silero VAD segments)",
            "text_tts": text_tts,
            "text_ref": text_ref,
            "normalization_variant": VARIANT_EN,
            "words": len(text_tts.split()),
            "chars": len(text_tts),
            "sentences": sum(1 for i in range(start_raw, cut_raw + 1) if is_sentence_end(tokens[i])),
            "ref_words": len(text_ref.split()),
            "ref_chars": len(text_ref),
            "sha256_text_tts": sha256_text(text_tts),
            "align_match_rate": round(sum(frag_matched) / len(frag_matched), 4),
            "cut_raw_index": cut_raw,
        })
        prev_cut_raw = cut_raw
        prev_dur = dur

    if not items:
        raise RuntimeError(f"{spec['narrator_id']}: no items at all")

    # ---- reference clip: after the LAST item's span, 5-10 s, sentence-aligned
    last_end_raw = items[-1]["cut_raw_index"]
    last_end_t = items[-1]["human_offset_end"]
    ref_span = None
    i = last_end_raw + 1
    sent_starts = [last_end_raw + 1] + [e + 1 for e in sent_ends if e > last_end_raw]
    for s0 in sent_starts:
        if s0 not in first_norm:
            continue
        if t_start[first_norm[s0]] < last_end_t + 1.0:
            continue
        # grow sentence by sentence until >= 5 s (cap 10 s)
        end_tok = None
        for e in sent_ends:
            if e < s0:
                continue
            dur = t_end[last_norm[e]] - t_start[first_norm[s0]]
            if dur > 10.0:
                break
            if dur >= 5.0 and matched[first_norm[s0]] and matched[last_norm[e]]:
                end_tok = e
                break
        if end_tok is not None:
            ref_span = (s0, end_tok)
            break
    if ref_span is None:
        raise RuntimeError(f"{spec['narrator_id']}: no 5-10 s sentence span after the last item")
    s0, e0 = ref_span
    ref_a = max(0.0, t_start[first_norm[s0]] - 0.2)
    ref_b = t_end[last_norm[e0]] + 0.1
    ref_text = " ".join(tokens[s0 : e0 + 1])

    wave, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    wave = wave[:, 0]
    seg = wave[int(ref_a * sr) : int(ref_b * sr)]
    REF_WAV_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = REF_WAV_DIR / f"{spec['voice_id']}.wav"
    sf.write(str(wav_path), seg, sr, subtype="PCM_16")
    import soxr
    seg16 = soxr.resample(seg, sr, 16000).astype(np.float32)
    wav16_path = REF_WAV_DIR / f"{spec['voice_id']}_16k.wav"
    sf.write(str(wav16_path), seg16, 16000, subtype="PCM_16")

    peak = float(np.abs(seg).max()) if len(seg) else 0.0
    rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    reference = {
        "voice_id": spec["voice_id"], "role": "primary",
        "reader": spec["reader"], "gender": spec["gender"],
        "language": "en", "exploratory": True,
        "book": spec["book"], "chapter_label": spec["chapter_label"],
        "librivox_id": spec["librivox_id"], "librivox_url": spec["librivox_url"],
        "audio_url": spec["audio_url"], "license": LICENSE_NOTE,
        "source_audio_path": str(audio_path),
        "offset_start_sec": round(ref_a, 3), "offset_end_sec": round(ref_b, 3),
        "duration_sec": round(ref_b - ref_a, 3),
        "outside_item_spans": bool(ref_a > last_end_t),
        "ref_text": ref_text,
        "ref_text_source": "Project Gutenberg chapter text aligned to the reading (verbatim, punctuated)",
        "n_words": len(ref_text.split()),
        "wav_path": str(wav_path.relative_to(REPO)),
        "wav_sample_rate": sr, "wav_subtype": "PCM_16", "wav_channels": 1,
        "wav_sha256": sha256_file(wav_path),
        "wav16k_path": str(wav16_path.relative_to(REPO)),
        "wav16k_sha256": sha256_file(wav16_path),
        "qc": {"peak": peak, "rms": rms, "clipped": peak >= 0.99},
    }

    return {
        "spec": {k: v for k, v in spec.items()},
        "extract": extract_info, "clean_stats": clean_stats,
        "chapter_words": len(tokens),
        "chapter_norm_words": len(ref_norm),
        "asr_words": len(asr_words),
        "align_match_rate_chapter": round(match_rate_chapter, 4),
        "audio_duration_sec": asr_payload["duration_sec"],
        "intro_skipped_sec": round(start_t, 3),
        "items": items, "reference": reference,
    }


def stage_build() -> None:
    results = [build_narrator(spec) for spec in NARRATORS]

    items = [it for r in results for it in r["items"]]
    BENCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCH_PATH, "wt", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(REFS_PATH, "wt", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r["reference"], ensure_ascii=False) + "\n")

    build = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exploratory": True,
        "normalization": {"variant": VARIANT_EN, "version": NORMALIZATION_VERSION_EN},
        "buckets": {k: list(v) for k, v in BUCKETS.items()},
        "n_items": len(items),
        "narrators": results,
    }
    BUILD_JSON.write_text(json.dumps(build, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(build)
    print(f"[build] {len(items)} items -> {BENCH_PATH}")
    print(f"[build] {len(results)} references -> {REFS_PATH}")
    print(f"[build] report -> {REPORT_MD}")


def write_report(build: dict) -> None:
    L = []
    L.append("# EN-probe benchmark composition (E9, EXPLORATORY)")
    L.append("")
    L.append("**This is an exploratory probe, not a paper benchmark** (pre-registration: "
             "reports/decisions.md, Lead 2026-08-31, «ПРЕРЕГИСТРАЦИЯ E9 EN-probe»). "
             "Generated by `src/benchmark/build_english_probe.py` from "
             "`data/external/english/build_english_probe.json`; every number below is "
             "computed, not typed.")
    L.append("")
    L.append(f"- generated: {build['generated_at']}")
    L.append(f"- artifact: `data/benchmark/english_probe.jsonl` — {build['n_items']} items")
    L.append(f"- references: `data/references/references_english.jsonl` + `data/references/english/*.wav`")
    L.append(f"- normalization: `{build['normalization']['variant']}` "
             f"(v{build['normalization']['version']}, `src/eval/normalize_english.py`): "
             "NFKC → lowercase → delete apostrophes → fold diacritics → non-[a-z0-9] → space → collapse. "
             "Digits kept literal; every text_tts is digit-free by construction.")
    L.append(f"- bucket edges (= pilot, configs/benchmark_pilot.yaml): "
             + ", ".join(f"{b} [{int(lo)},{int(hi)}] s" for b, (lo, hi) in build["buckets"].items()))
    L.append("- items are NESTED PREFIXES of one continuous chapter reading per narrator "
             "(B0 ⊂ B1 ⊂ … ⊂ B4, mirroring the pilot §7.2); bucket = actual human reading "
             "duration measured from Parakeet word timestamps on the chapter audio.")
    L.append("- the reference clip of every narrator lies AFTER the B4 span of the same "
             "chapter (never inside any item span).")
    L.append("")
    L.append("## Narrators and provenance (all public domain)")
    L.append("")
    L.append("| narrator | reader | gender | book (author) | chapter | LibriVox | audio file | Gutenberg |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in build["narrators"]:
        s = r["spec"]
        L.append(f"| `{s['narrator_id']}` | {s['reader']} | {s['gender']} | {s['book']} "
                 f"({s['author']}) | {s['chapter_label']} | [id {s['librivox_id']}]({s['librivox_url']}) | "
                 f"[mp3]({s['audio_url']}) | [#{s['gutenberg_id']}]({s['gutenberg_url']}) |")
    L.append("")
    L.append(f"- license: {LICENSE_NOTE}.")
    L.append("")
    L.append("## Audio ↔ text correspondence (self-check)")
    L.append("")
    L.append("`align_match_rate` = share of normalized Gutenberg chapter words exactly matched "
             "in the Parakeet word stream (difflib longest-block alignment). This is an "
             "alignment statistic, not a WER: the true ASR floor is measured separately by "
             "`scripts/run_evaluation_en.py --floor-only` into `results/v31_en/asr_floor.jsonl`.")
    L.append("")
    L.append("| narrator | audio dur s | chapter words | ASR words | chapter match rate | intro skipped s |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in build["narrators"]:
        L.append(f"| `{r['spec']['narrator_id']}` | {r['audio_duration_sec']:.0f} | "
                 f"{r['chapter_words']} | {r['asr_words']} | {r['align_match_rate_chapter']:.3f} | "
                 f"{r['intro_skipped_sec']:.1f} |")
    L.append("")
    L.append("- `intro skipped` = audio before the first aligned chapter word (LibriVox "
             "disclaimer + chapter announcement); items start after it.")
    L.append("")
    L.append("## Items")
    L.append("")
    L.append("| text_id | bucket | human dur s | offsets s | words | sentences | match rate | voice |")
    L.append("|---|---|---:|---|---:|---:|---:|---|")
    for r in build["narrators"]:
        for it in r["items"]:
            L.append(f"| `{it['text_id']}` | {it['bucket']} | {it['human_duration_sec']:.1f} | "
                     f"{it['human_offset_start']:.1f}–{it['human_offset_end']:.1f} | {it['words']} | "
                     f"{it['sentences']} | {it['align_match_rate']:.3f} | `{it['voice_id']}` |")
    L.append("")
    L.append("## References")
    L.append("")
    L.append("| voice_id | dur s | offsets s | words | peak | rms | outside items |")
    L.append("|---|---:|---|---:|---:|---:|---|")
    for r in build["narrators"]:
        ref = r["reference"]
        L.append(f"| `{ref['voice_id']}` | {ref['duration_sec']:.2f} | "
                 f"{ref['offset_start_sec']:.1f}–{ref['offset_end_sec']:.1f} | {ref['n_words']} | "
                 f"{ref['qc']['peak']:.3f} | {ref['qc']['rms']:.4f} | {ref['outside_item_spans']} |")
    L.append("")
    L.append("## Known asymmetry (must be quoted next to any E9 table)")
    L.append("")
    L.append("English text_tts carries full punctuation and capitalization, which matches the "
             "TRAINING distribution of E7 (Punct-SFT) better than that of E3 (ROVER text, "
             "~no punctuation). Any E7-vs-E3 difference on this probe is therefore "
             "confounded with the punctuation match and must not be read as a pure "
             "language-transfer effect.")
    L.append("")
    REPORT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["asr", "build", "all"], default="all")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()
    if args.stage in ("asr", "all"):
        stage_asr(args.device)
    if args.stage in ("build", "all"):
        stage_build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
