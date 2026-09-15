# coding=utf-8
"""Small, dependency-light helpers for full-utterance Qwen3-TTS LoRA."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


ASR_CONSENSUS_FIELDS = (
    "giga_ctc.txt",
    "giga_rnnt.txt",
    "vosk.txt",
)
GIGAAM_E2E_FIELD = "gigaam-v3-e2e-ctc.txt"
PUNCT_FIELD = "punct.txt"
ROVER_FIELD = "rover.txt"
WORD_RE = re.compile(r"[0-9a-zа-я]+", re.IGNORECASE)
STRESSED_WORD_RE = re.compile(r"\+?[A-Za-zА-Яа-яЁё]+(?:\+[A-Za-zА-Яа-яЁё]+)*")
FILLER_WORDS = {
    "а",
    "аа",
    "ааа",
    "м",
    "мм",
    "ммм",
    "угу",
    "хм",
    "э",
    "ээ",
    "эээ",
}


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def dotenv_value(path: str | Path, name: str) -> str | None:
    """Read one simple KEY=VALUE entry without sourcing executable shell text."""
    env_path = Path(path)
    if not env_path.exists():
        return None
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    return None


def normalize_asr_text(text: Any, *, drop_fillers: bool = True) -> str:
    normalized = str(text or "").lower().replace("ё", "е").replace("+", "")
    words = WORD_RE.findall(normalized)
    if drop_fillers:
        words = [word for word in words if word not in FILLER_WORDS]
    return " ".join(words)


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: Any, hypothesis: Any, *, drop_fillers: bool = True) -> float:
    ref_words = normalize_asr_text(reference, drop_fillers=drop_fillers).split()
    hyp_words = normalize_asr_text(hypothesis, drop_fillers=drop_fillers).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return float(_edit_distance(ref_words, hyp_words) / len(ref_words))


def asr_consensus(
    metadata: Mapping[str, Any],
    *,
    max_wer: float,
    fields: Sequence[str] = ASR_CONSENSUS_FIELDS,
) -> tuple[bool, dict[str, float]]:
    """Require the core ASR hypotheses to agree with ROVER.

    GigaAM E2E is scored too, but does not reject the row: its score selects
    between the better-punctuated E2E text and ``punct.txt`` downstream.
    Disfluency tokens are ignored only for this confidence gate.
    """
    rover = metadata.get(ROVER_FIELD)
    if not normalize_asr_text(rover):
        return False, {}

    scores: dict[str, float] = {}
    for field in fields:
        value = metadata.get(field)
        if not normalize_asr_text(value):
            return False, scores
        scores[field] = word_error_rate(rover, value)
    selected = metadata.get(GIGAAM_E2E_FIELD)
    if normalize_asr_text(selected):
        scores[GIGAAM_E2E_FIELD] = word_error_rate(rover, selected)
    accepted = all(scores[field] <= max_wer for field in fields)
    return accepted, scores


def select_training_transcript(
    metadata: Mapping[str, Any], *, max_wer: float
) -> tuple[str, str, float | None]:
    """Prefer punctuated GigaAM E2E only while it agrees with ROVER."""
    rover = metadata.get(ROVER_FIELD)
    e2e_text = str(metadata.get(GIGAAM_E2E_FIELD) or "").strip()
    e2e_wer = (
        word_error_rate(rover, e2e_text)
        if normalize_asr_text(rover) and normalize_asr_text(e2e_text)
        else None
    )
    if e2e_wer is not None and e2e_wer <= max_wer:
        return e2e_text, GIGAAM_E2E_FIELD, e2e_wer
    return str(metadata.get(PUNCT_FIELD) or "").strip(), PUNCT_FIELD, e2e_wer


def clean_short_word_stress(text: str, max_letters: int = 2) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        letters = token.replace("+", "")
        return letters if len(letters) <= max_letters else token

    return STRESSED_WORD_RE.sub(replace, text)


# --- Lead local edit (E10 P3, 2026-08-31): the accentor's RAM grows superlinearly with
# input length (measured in-venv: 686 chars -> +0.02 GB, 6184 chars -> +7.5 GB peak,
# 9455 chars fails even under a 20 GB address-space cap). qe3p rows carry full 15-min
# unit texts up to 15.6k chars, and several dataloader workers hit their longest rows at
# the same step (both try2 and try3 died at exactly optimizer step 161 with a worker
# SIGKILLed by the host OOM killer). Texts above _STRESS_CHUNK_LIMIT are therefore
# stressed sentence-chunk by sentence-chunk (separators preserved verbatim); shorter
# texts — every qe2p row — keep the original single-call path bit-identically.
_STRESS_CHUNK_LIMIT = 1000  # chars; whole-call path below this
_STRESS_CHUNK_TARGET = 800  # greedy sentence packing target per accentor call
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])(\s+)")


def _stress_chunks(text: str) -> list[str]:
    """Split into accentor-sized pieces at sentence boundaries, keeping separators.

    The concatenation of the returned pieces equals ``text`` exactly; pieces are
    packed greedily to ~_STRESS_CHUNK_TARGET chars.  A single oversized sentence is
    hard-split at whitespace as a last resort.
    """
    parts = _SENTENCE_SPLIT_RE.split(text)  # [sent, sep, sent, sep, ..., sent]
    units: list[str] = []
    for i in range(0, len(parts), 2):
        unit = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        while len(unit) > _STRESS_CHUNK_LIMIT:  # oversized sentence: hard-split
            cut = unit.rfind(" ", 1, _STRESS_CHUNK_TARGET)
            cut = cut + 1 if cut > 0 else _STRESS_CHUNK_TARGET
            units.append(unit[:cut])
            unit = unit[cut:]
        if unit:
            units.append(unit)
    chunks: list[str] = []
    for unit in units:
        if chunks and len(chunks[-1]) + len(unit) <= _STRESS_CHUNK_TARGET:
            chunks[-1] += unit
        else:
            chunks.append(unit)
    return chunks


def apply_silero_stress(accentor: Any, text: str) -> str:
    def _call(piece: str) -> str:
        return str(
            accentor(
                piece,
                put_stress=True,
                put_stress_homo=True,
                put_yo=False,
                put_yo_homo=False,
            )
        )

    text = str(text).strip()
    if len(text) <= _STRESS_CHUNK_LIMIT:
        stressed = _call(text)
    else:
        pieces = []
        for chunk in _stress_chunks(text):
            stripped = chunk.strip()
            if not stripped:
                pieces.append(chunk)
                continue
            lead = chunk[: len(chunk) - len(chunk.lstrip())]
            trail = chunk[len(chunk.rstrip()):]
            pieces.append(lead + _call(stripped) + trail)
        stressed = "".join(pieces)
    return clean_short_word_stress(stressed.strip())


def deterministic_prefix_frames(
    *,
    key: str,
    total_frames: int,
    seed: int,
    min_prefix_frames: int,
    max_prefix_frames: int,
    min_continuation_frames: int,
) -> int:
    """Choose a stable acoustic prefix without word timestamps or text split."""
    upper = min(int(max_prefix_frames), int(total_frames) - int(min_continuation_frames))
    lower = int(min_prefix_frames)
    if upper < lower:
        raise ValueError(
            f"Not enough codec frames: total={total_frames}, prefix>={lower}, "
            f"continuation>={min_continuation_frames}"
        )
    digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    return lower + (int.from_bytes(digest, "big") % (upper - lower + 1))


def build_assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def build_ref_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n"
