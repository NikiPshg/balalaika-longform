"""English text normalization for the E9 "EN-probe" (EXPLORATORY).

Pre-registered in reports/decisions.md, Lead 2026-08-31 ("нормализация текста
для WER — английская: lowercase, без пунктуации").  This module mirrors the
*shape* of the frozen Russian ``src/eval/normalize.py`` but is a separate file:
the Russian normalization and ``configs/text_normalization.yaml`` are FROZEN and
are not imported, touched or extended here.

One variant, ``english_v1``, applied identically to reference and hypothesis:

    NFKC -> lowercase -> DELETE apostrophes (word-internally: "don't" -> "dont",
    so a reference "don't" and an ASR "dont" agree on the word count -- the same
    reasoning the pilot charset used for Russian "О'кей" -> "Окей")
    -> FOLD diacritics (NFKD, combining marks dropped: "Adèle" -> "adele" -- the
    Gutenberg originals keep French accents, the English ASR does not)
    -> every character not in [a-z0-9] becomes a space -> collapse whitespace

Digits are kept literal (NO spell-out).  The probe texts are selected digit-free
at build time (mirroring ``forbid_digits`` of ``configs/benchmark_pilot.yaml``),
so on the benchmark itself the digit branch never fires; an ASR "7" against a
text "seven" WOULD count as an error, which is exactly why fragments containing
digits are excluded when the benchmark is built
(``src/benchmark/build_english_probe.py``).

Unit tests: tests/test_english_probe.py.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "NORMALIZATION_VERSION_EN",
    "VARIANT_EN",
    "normalize_english",
    "words_english",
]

NORMALIZATION_VERSION_EN = 1
VARIANT_EN = "english_v1"

# Same apostrophe family the pilot charset deletes (configs/benchmark_pilot.yaml
# ``apostrophes.delete``), plus the ASCII one.
_APOSTROPHES = "'’‘`´ʼ"
_KEEP = set("abcdefghijklmnopqrstuvwxyz0123456789")
_WS_RE = re.compile(r"\s+")


def normalize_english(text: str | None) -> str:
    """Frozen-for-E9 English normalization (variant ``english_v1``)."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    for ch in _APOSTROPHES:
        text = text.replace(ch, "")
    # diacritic fold: decompose and drop combining marks ("adèle" -> "adele")
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(ch))
    text = "".join(ch if ch in _KEEP else " " for ch in text)
    return _WS_RE.sub(" ", text).strip()


def words_english(text: str | None) -> list[str]:
    """Normalized whitespace-separated word list."""
    normalized = normalize_english(text)
    return normalized.split() if normalized else []
